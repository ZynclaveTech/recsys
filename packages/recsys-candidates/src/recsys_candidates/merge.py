"""Combining candidates from sources that do not share a scale.

This is where retrieval pipelines usually go wrong, so it is worth stating the
problem plainly. Suppose three sources return the same item:

    ann         cosine similarity   0.82
    popularity  interaction count   4300
    recency     hours since posted  7

There is no weighting of those three numbers that means anything. Whichever
has the widest numeric range dominates the sum, and normalising to [0, 1]
per source only hides it — a min-max over a batch makes the top item of a
*bad* source look exactly as good as the top item of a good one.

The fix is to stop comparing scores and compare **ranks**. A source's own
ordering is the only signal it can be trusted to produce, and rank 1 means the
same thing coming out of every source. That is what Reciprocal Rank Fusion
does, and it is the default here.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence

from recsys_candidates.types import Candidate, ItemT

__all__ = [
    "RRF_K",
    "MergeStrategy",
    "by_score",
    "interleave",
    "reciprocal_rank_fusion",
    "sources_of",
]

MergeStrategy = Callable[
    [Mapping[str, Sequence["Candidate[ItemT]"]]], list["Candidate[ItemT]"]
]

RRF_K = 60
"""Rank-fusion damping constant.

Each source contributes ``1 / (K + rank)``. Larger K flattens the curve, so
deep results matter more relative to the top few; smaller K makes the head
dominate. 60 is the value from the original RRF paper and it is a reasonable
default, but it *is* a tuning knob — if your sources have very different
depths, it is worth revisiting.
"""


def _fused_order(
    per_source: Mapping[str, Sequence[Candidate[ItemT]]],
    weights: Mapping[str, float] | None,
    k: int,
) -> list[Candidate[ItemT]]:
    contribution: defaultdict[ItemT, float] = defaultdict(float)
    best: dict[ItemT, Candidate[ItemT]] = {}
    first_seen: dict[ItemT, int] = {}

    order = 0
    for name, candidates in per_source.items():
        weight = 1.0 if weights is None else weights.get(name, 1.0)
        for position, candidate in enumerate(candidates):
            item = candidate.item
            contribution[item] += weight / (k + position + 1)
            if item not in best or candidate.rank < best[item].rank:
                best[item] = candidate
            if item not in first_seen:
                first_seen[item] = order
                order += 1

    # `first_seen` breaks ties deterministically. Without it, items with equal
    # fused contribution come out in dict order, which is stable within a run
    # and changes the moment a source is added — noise that reads downstream as
    # a ranking change.
    ranked = sorted(
        contribution,
        key=lambda item: (-contribution[item], first_seen[item]),
    )
    return [best[item] for item in ranked]


def reciprocal_rank_fusion(
    per_source: Mapping[str, Sequence[Candidate[ItemT]]],
    *,
    weights: Mapping[str, float] | None = None,
    k: int = RRF_K,
) -> list[Candidate[ItemT]]:
    """Merge by rank. **The default, and the one to reach for first.**

    Each source contributes ``weight / (k + rank)`` for every item it returns.
    An item found by several sources accumulates from each, so agreement across
    sources is rewarded without any of them needing a comparable score.

    ``weights`` expresses *how much you trust a source*, not how its scores
    compare — which is a question you can actually answer. Absent sources
    default to 1.0.

    Returns:
        Candidates ordered by fused contribution, deduplicated. Each surviving
        candidate is the instance from whichever source ranked it best, so its
        ``source`` and ``meta`` reflect that source.
    """
    if k < 0:
        raise ValueError(f"k must be non-negative, got {k}.")
    if weights:
        negative = sorted(n for n, w in weights.items() if w < 0)
        if negative:
            raise ValueError(
                f"Negative weights for {', '.join(negative)}. A negative weight "
                "does not demote an item, it makes finding it in that source "
                "count against it — which is almost never what is meant. Drop "
                "the source, or filter it out explicitly."
            )
    return _fused_order(per_source, weights, k)


def interleave(
    per_source: Mapping[str, Sequence[Candidate[ItemT]]],
    *,
    order: Sequence[str] | None = None,
) -> list[Candidate[ItemT]]:
    """Round-robin: one from each source in turn, deduplicated.

    Guarantees every source appears near the top, which RRF does not — a source
    whose items no other source corroborates can be pushed down by fusion even
    when it is the only source of, say, fresh content.

    Use it when you want visible representation from each source. It ignores
    agreement between sources entirely, which is the trade.

    Args:
        order: Source names in the order to draw from. Sources omitted here
            still contribute, after the named ones. Defaults to mapping order.
    """
    names = list(order or per_source)
    names += [n for n in per_source if n not in names]

    seen: set[ItemT] = set()
    out: list[Candidate[ItemT]] = []
    depth = 0
    remaining = True
    while remaining:
        remaining = False
        for name in names:
            candidates = per_source.get(name, ())
            if depth >= len(candidates):
                continue
            remaining = True
            candidate = candidates[depth]
            if candidate.item not in seen:
                seen.add(candidate.item)
                out.append(candidate)
        depth += 1
    return out


def by_score(
    per_source: Mapping[str, Sequence[Candidate[ItemT]]],
    *,
    weights: Mapping[str, float] | None = None,
) -> list[Candidate[ItemT]]:
    """Merge on raw ``score``. **Only correct if your sources share a scale.**

    Provided because sometimes they genuinely do — several ANN indexes over the
    same embedding space, or sources that all emit a calibrated probability.
    In that case this is better than RRF, because it uses magnitude rather than
    discarding it.

    In every other case it is the wrong tool, and it fails silently: the source
    with the widest numeric range wins, no error is raised, and the slate just
    quietly stops reflecting the sources you thought you were blending. If you
    cannot say what unit the scores are in, use
    :func:`reciprocal_rank_fusion`.

    Raises:
        ValueError: If any candidate has ``score=None``. Treating a missing
            score as 0.0 would rank those items last while looking like it
            worked.
    """
    missing = sorted(
        {name for name, cs in per_source.items() if any(c.score is None for c in cs)}
    )
    if missing:
        raise ValueError(
            f"Source(s) {', '.join(missing)} returned candidates without a "
            "score, so they cannot be merged by score. Give them scores on the "
            "same scale as the others, or merge with reciprocal_rank_fusion, "
            "which does not need one."
        )

    best: dict[ItemT, Candidate[ItemT]] = {}
    weighted: dict[ItemT, float] = {}
    first_seen: dict[ItemT, int] = {}
    order = 0
    for name, candidates in per_source.items():
        weight = 1.0 if weights is None else weights.get(name, 1.0)
        for candidate in candidates:
            value = weight * float(candidate.score or 0.0)
            item = candidate.item
            if item not in weighted or value > weighted[item]:
                weighted[item] = value
                best[item] = candidate
            if item not in first_seen:
                first_seen[item] = order
                order += 1

    ranked = sorted(weighted, key=lambda i: (-weighted[i], first_seen[i]))
    return [best[item] for item in ranked]


def sources_of(candidates: Iterable[Candidate[ItemT]]) -> dict[str, int]:
    """Count candidates by source. For logging what a merge actually produced.

    Worth emitting on every request during a rollout: a source that has
    silently stopped contributing looks identical to one that is working, right
    up until someone notices the feed got worse.
    """
    counts: defaultdict[str, int] = defaultdict(int)
    for candidate in candidates:
        counts[candidate.source] += 1
    return dict(counts)
