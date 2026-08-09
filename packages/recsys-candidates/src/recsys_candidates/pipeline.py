"""Fan out to sources, merge, filter, diversify, truncate.

Two decisions shape everything here.

**A broken source must not blank the feed.** Retrieval is I/O, and in a fan-out
of six sources the probability that all six are healthy is lower than you would
like. A source that raises or times out is recorded and skipped; the rest of
the slate is still served. The cost is that a degraded feed looks like a
working one, so the result carries ``failures`` and ``short`` and you are
expected to read them.

**Order must not depend on completion time.** Sources run concurrently, but
their results are reassembled in the order the mapping declared them, never the
order they finished. Otherwise merge tie-breaks shift run to run, the same user
gets a different slate on identical data, and nothing in the metrics explains
why.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable, Mapping, Sequence, Set
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Generic

from recsys_candidates.merge import reciprocal_rank_fusion
from recsys_candidates.protocols import Source
from recsys_candidates.types import Budget, Candidate, ItemT, UserT

__all__ = ["Result", "generate", "max_per_key"]

logger = logging.getLogger(__name__)

Filter = Callable[["Candidate[ItemT]"], bool]
Diversity = Callable[[Sequence["Candidate[ItemT]"]], list["Candidate[ItemT]"]]


@dataclass(frozen=True)
class Result(Generic[ItemT]):
    """The slate, plus everything needed to tell a good one from a bad one.

    Attributes:
        candidates: The final, ordered slate.
        fetched: Source name -> how many it returned, before merge. A source
            that has quietly stopped contributing is invisible in
            ``candidates`` alone, because the others fill the gap.
        contributed: Source name -> how many of its items survived into
            ``candidates``. A source with a healthy ``fetched`` and a zero
            here is being out-competed or filtered away, which is a different
            problem from being broken.
        failures: Source name -> the exception text, for sources that raised
            or timed out. **Empty is the only good value.**
        short: True when fewer than ``budget.total`` candidates were produced.
            The quiet failure this exists to surface: a short slate renders
            like a working one and only shows up later as "the feed is empty
            for some users".
    """

    candidates: list[Candidate[ItemT]]
    fetched: Mapping[str, int] = field(default_factory=dict)
    contributed: Mapping[str, int] = field(default_factory=dict)
    failures: Mapping[str, str] = field(default_factory=dict)
    short: bool = False

    def __len__(self) -> int:
        return len(self.candidates)

    def __iter__(self) -> Any:
        return iter(self.candidates)

    def summary(self) -> dict[str, Any]:
        """A dict worth logging on every request during a rollout."""
        return {
            "returned": len(self.candidates),
            "fetched": dict(self.fetched),
            "contributed": dict(self.contributed),
            "failures": dict(self.failures),
            "short": self.short,
        }


def max_per_key(key: Callable[[Candidate[ItemT]], Any], limit: int) -> Diversity[ItemT]:
    """Cap how many candidates share a key — an author, a topic, a publisher.

    Greedy and order-preserving: it walks the merged slate once and drops
    anything that would exceed the cap. It never reorders, because reordering
    to satisfy diversity silently undoes the merge you just computed.

    Candidates whose key is ``None`` are exempt rather than pooled together.
    Pooling them would make "unknown author" the most over-represented author
    in the slate, which is the opposite of the intent.

    Example:
        >>> diversity = max_per_key(lambda c: c.meta.get("author"), 2)
    """
    if limit < 1:
        raise ValueError(f"limit must be at least 1, got {limit}.")

    def apply(candidates: Sequence[Candidate[ItemT]]) -> list[Candidate[ItemT]]:
        counts: dict[Any, int] = {}
        out: list[Candidate[ItemT]] = []
        for candidate in candidates:
            value = key(candidate)
            if value is None:
                out.append(candidate)
                continue
            seen = counts.get(value, 0)
            if seen >= limit:
                continue
            counts[value] = seen + 1
            out.append(candidate)
        return out

    return apply


def _normalise(name: str, raw: Iterable[Any]) -> list[Candidate[ItemT]]:
    """Stamp rank and source, whatever the source returned.

    Rank comes from position and source from the mapping key, always — so a
    source cannot mislabel itself, and cannot get its own ranks wrong.
    """
    out: list[Candidate[ItemT]] = []
    for position, entry in enumerate(raw):
        if isinstance(entry, Candidate):
            out.append(
                Candidate(
                    item=entry.item,
                    rank=position,
                    source=name,
                    score=entry.score,
                    meta=entry.meta,
                )
            )
        else:
            out.append(Candidate(item=entry, rank=position, source=name))
    return out


def generate(
    sources: Mapping[str, Source[UserT, ItemT]],
    user: UserT,
    *,
    budget: Budget,
    exclude: Set[ItemT] = frozenset(),
    merge: Callable[..., list[Candidate[ItemT]]] = reciprocal_rank_fusion,
    filters: Sequence[Filter[ItemT]] = (),
    diversity: Diversity[ItemT] | None = None,
    timeout: float | None = None,
    max_workers: int | None = None,
) -> Result[ItemT]:
    """Retrieve from every source concurrently and assemble one slate.

    Args:
        sources: Name -> source. The name is stamped onto every candidate and
            is what ``budget`` and merge ``weights`` key off, so it is part of
            your configuration, not a label.
        user: Who to retrieve for.
        budget: How many to return, and per-source caps. See
            :class:`~recsys_candidates.types.Budget`.
        exclude: Items no source should return. Converted to a ``frozenset``
            before being handed out, so one source cannot mutate it and shrink
            another's pool.
        merge: How to combine sources. Defaults to reciprocal rank fusion —
            see :mod:`recsys_candidates.merge` for why not score.
        filters: Predicates applied **after** merge; a candidate is kept when
            all return True. They run post-merge because a filter usually needs
            the deduplicated item, and because the budget's overfetch exists
            precisely to absorb what they remove.
        diversity: Applied after filtering, before truncation. See
            :func:`max_per_key`.
        timeout: Seconds to wait for the whole fan-out. On expiry, sources that
            have not finished are recorded in ``failures`` and the slate is
            built from the rest. ``None`` waits indefinitely, which is rarely
            what a request path wants.
        max_workers: Thread-pool size. Defaults to one per source.

    Returns:
        A :class:`Result`. Read ``failures`` and ``short`` — a degraded slate
        is indistinguishable from a healthy one by length alone.
    """
    if not sources:
        raise ValueError(
            "No sources given, so there is nothing to retrieve from. An empty "
            "slate here is a configuration error, not a cold-start result."
        )

    frozen: Set[ItemT] = frozenset(exclude)
    names = list(sources)
    per_source: dict[str, list[Candidate[ItemT]]] = {}
    failures: dict[str, str] = {}

    def call(name: str) -> list[Candidate[ItemT]]:
        raw = sources[name].fetch(user, budget.fetch_size(name), frozen)
        return _normalise(name, raw)

    with ThreadPoolExecutor(max_workers=max_workers or len(names)) as pool:
        futures = {name: pool.submit(call, name) for name in names}
        # Iterating `names`, not `as_completed`, is deliberate: the merge input
        # order decides tie-breaks, so completion order would make the same
        # user's slate depend on which source happened to be fast today.
        for name in names:
            try:
                per_source[name] = futures[name].result(timeout=timeout)
            # Deliberately broad: one bad source must not blank the feed.
            except Exception as exc:
                failures[name] = f"{type(exc).__name__}: {exc}"
                per_source[name] = []
                logger.warning(
                    "Candidate source %r failed; continuing without it: %s",
                    name,
                    exc,
                    exc_info=True,
                )

    merged = merge(per_source)

    for predicate in filters:
        merged = [c for c in merged if predicate(c)]
    if diversity is not None:
        merged = diversity(merged)

    slate = merged[: budget.total]
    contributed: dict[str, int] = {}
    for candidate in slate:
        contributed[candidate.source] = contributed.get(candidate.source, 0) + 1

    result = Result(
        candidates=slate,
        fetched={n: len(cs) for n, cs in per_source.items()},
        contributed=contributed,
        failures=failures,
        short=len(slate) < budget.total,
    )

    if failures:
        logger.warning(
            "Candidate generation degraded: %d/%d sources failed (%s)",
            len(failures),
            len(names),
            ", ".join(sorted(failures)),
        )
    if result.short:
        logger.warning(
            "Candidate generation returned %d of %d requested; raise "
            "budget.overfetch or check filters",
            len(slate),
            budget.total,
        )
    return result
