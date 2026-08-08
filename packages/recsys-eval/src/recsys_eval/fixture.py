"""The evaluation fixture: everything both rankers are scored against.

A fixture is built **once** and reused for the candidate and the incumbent.
That is not an optimisation, it is the property the whole comparison rests on:
with the same holdout, the same users, the same labels and the same candidate
pool, the only variable left is the model.

Most of the ways a promotion gate goes wrong are ways that property quietly
stops holding. See ``docs/hazards.md``.
"""

from __future__ import annotations

import random
from collections.abc import Iterable, Mapping, Sequence, Set
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Generic, TypeVar, cast

from recsys_eval.types import Aggregate, Interaction, ItemT, RelevancePolicy, UserT

__all__ = ["Fixture"]

_T = TypeVar("_T")


def _stable_sorted(values: Iterable[_T], *, what: str) -> list[_T]:
    """Sort for reproducibility, with an error that says what to do.

    Deterministic sampling needs a total order, and ``sorted`` on unorderable
    ids raises a ``TypeError`` that names neither the argument nor the fix.
    """
    try:
        return sorted(cast("Any", values))
    except TypeError as exc:
        raise TypeError(
            f"{what} are not sortable, so sampling them could not be made "
            "reproducible. Two runs over the same data would disagree, and a "
            "promotion gate that disagrees with itself is not a gate. Convert "
            "the ids to str or int before building the fixture."
        ) from exc


def _check_comparable(sample: datetime, bound: datetime) -> None:
    """Fail early and clearly on mixed naive/aware datetimes."""
    if (sample.tzinfo is None) != (bound.tzinfo is None):
        got = "naive" if sample.tzinfo is None else "timezone-aware"
        want = "naive" if bound.tzinfo is None else "timezone-aware"
        raise ValueError(
            f"Interaction timestamps are {got} but the holdout bounds are "
            f"{want}. Python refuses to compare the two, and silently "
            "coercing would shift every event across the cutoff by the UTC "
            "offset. Make both consistent before building the fixture."
        )


@dataclass(frozen=True)
class Fixture(Generic[UserT, ItemT]):
    """Frozen labels and candidate pool for one evaluation run.

    Build with :meth:`from_interactions` rather than by hand.

    Attributes:
        holdout_start: Start of the label window, inclusive.
        holdout_end: End of the label window, exclusive.
        users: Users to score, in a reproducible order.
        ground_truth: user -> {item: graded gain}. Feeds NDCG.
        relevant: user -> {item}. Binary relevance; the keys of
            ``ground_truth``, precomputed. Feeds Recall, MAP, MRR, hit rate.
        seen: user -> {item} interacted with *before* ``holdout_start``. Pass
            these to a ranker as exclusions so it is not rewarded for
            re-surfacing something already consumed.
        candidates: The shared pool both rankers retrieve from.
        label_coverage: Fraction of labelled items that are actually present in
            ``candidates`` -- the ceiling on Recall. **Read this number.** At
            0.4, Recall cannot exceed 0.4 no matter how good the model is, and
            any absolute reading of the metric is meaningless.
    """

    holdout_start: datetime
    holdout_end: datetime
    users: tuple[UserT, ...]
    ground_truth: Mapping[UserT, Mapping[ItemT, float]]
    relevant: Mapping[UserT, Set[ItemT]]
    seen: Mapping[UserT, Set[ItemT]]
    candidates: tuple[ItemT, ...]
    label_coverage: float

    def __len__(self) -> int:
        return len(self.users)

    def summary(self) -> dict[str, Any]:
        """A small dict worth logging on every run, and storing beside results.

        ``label_coverage`` and ``candidates`` belong on the audit row next to
        the metrics: a score that moved because the candidate pool shrank looks
        identical to one that moved because the model changed, and six weeks
        later this is the only thing that tells them apart.
        """
        return {
            "holdout_start": self.holdout_start.isoformat(),
            "holdout_end": self.holdout_end.isoformat(),
            "users": len(self.users),
            "candidates": len(self.candidates),
            "labels": sum(len(items) for items in self.ground_truth.values()),
            "label_coverage": round(self.label_coverage, 4),
        }

    @classmethod
    def from_interactions(
        cls,
        events: Iterable[Interaction[UserT, ItemT]],
        *,
        candidates: Sequence[ItemT],
        holdout_start: datetime,
        holdout_end: datetime,
        policy: RelevancePolicy,
        max_users: int | None = None,
        max_candidates: int | None = None,
        require_history: bool = True,
        seed: int = 0,
    ) -> Fixture[UserT, ItemT]:
        """Split a stream of interactions into labels, exclusions, and a pool.

        Events are read **once**, in a single pass, so ``events`` may be a
        generator streaming from disk or a database cursor. Order does not
        matter: every aggregation here is order-independent by construction.

        Args:
            events: The interaction stream. Consumed once.
            candidates: The pool both rankers retrieve from -- normally your
                serving index, or the catalogue slice it covers. Required, and
                deliberately not derived from ``events``: deriving it would put
                every labelled item in the pool by construction, pin
                ``label_coverage`` at 1.0, and report a retrieval ceiling your
                production system does not have.

                Pass it in the order you want truncation to keep (usually
                newest first). If your own query orders by a non-unique column,
                add a unique tiebreaker there -- otherwise ties resolve
                arbitrarily and ``max_candidates`` cuts a different pool on
                every run, which reads downstream as model drift.
            holdout_start: Start of the label window, inclusive.
            holdout_end: End of the label window, exclusive.
            policy: Which actions count as relevance. See
                :class:`~recsys_eval.types.RelevancePolicy`.
            max_users: Sample at most this many eligible users. Deterministic
                given ``seed``.
            max_candidates: Truncate the pool to its first this-many entries.
            require_history: Drop users with no interactions before
                ``holdout_start``. On by default: a user the model has never
                seen is measuring cold-start behaviour, which is a real thing
                to measure but not the same thing, and mixing the two makes a
                gate move when your signup rate moves.
            seed: Sampling seed. Two runs over the same data and seed produce
                identical fixtures.

        Returns:
            A frozen :class:`Fixture`.

        Raises:
            ValueError: On an empty candidate pool, a non-positive window, a
                non-positive limit, or mixed naive/aware timestamps. These are
                construction errors and they raise loudly -- unlike the gate
                itself, which fails open. A misbuilt fixture silently scores
                every model 0.0 and reports "no regression" forever.
        """
        if holdout_end <= holdout_start:
            raise ValueError(
                f"holdout_end ({holdout_end.isoformat()}) must be after "
                f"holdout_start ({holdout_start.isoformat()})."
            )
        if not candidates:
            raise ValueError(
                "candidates is empty. Every ranking would be empty, every "
                "metric would be 0.0, and a gate comparing 0.0 against 0.0 "
                "reports 'no regression' forever with no visible symptom."
            )
        if max_users is not None and max_users <= 0:
            raise ValueError(f"max_users must be positive, got {max_users}.")
        if max_candidates is not None and max_candidates <= 0:
            raise ValueError(f"max_candidates must be positive, got {max_candidates}.")

        pool: tuple[ItemT, ...] = tuple(candidates[:max_candidates])

        graded: dict[UserT, dict[ItemT, float]] = {}
        seen: dict[UserT, set[ItemT]] = {}
        checked_tz = False

        for event in events:
            if not checked_tz:
                _check_comparable(event.at, holdout_start)
                checked_tz = True

            if event.at < holdout_start:
                # Everything before the cutoff is an exclusion, whatever the
                # action was: a user who blocked an item has consumed it just
                # as surely as one who liked it, and re-surfacing it is not a
                # success either way.
                seen.setdefault(event.user, set()).add(event.item)
                continue

            if event.at >= holdout_end:
                continue

            gain = policy.gain(event)
            if gain <= 0:
                continue

            items = graded.setdefault(event.user, {})
            prior = items.get(event.item)
            if prior is None:
                items[event.item] = gain
            elif policy.aggregate is Aggregate.SUM:
                items[event.item] = prior + gain
            else:
                items[event.item] = max(prior, gain)

        eligible = [user for user in graded if not require_history or user in seen]
        selected = _stable_sorted(eligible, what="User ids")

        if max_users is not None and len(selected) > max_users:
            selected = random.Random(seed).sample(selected, max_users)
            selected = _stable_sorted(selected, what="User ids")

        users = tuple(selected)

        ground_truth: dict[UserT, Mapping[ItemT, float]] = {
            user: dict(graded[user]) for user in users
        }
        relevant: dict[UserT, Set[ItemT]] = {
            user: frozenset(graded[user]) for user in users
        }
        # Restricted to the sampled users, and defaulted for users whose only
        # eligibility came from `require_history=False`.
        exclusions: dict[UserT, Set[ItemT]] = {
            user: frozenset(seen.get(user, ())) for user in users
        }

        labelled = {item for items in ground_truth.values() for item in items}
        coverage = len(labelled & set(pool)) / len(labelled) if labelled else 0.0

        return cls(
            holdout_start=holdout_start,
            holdout_end=holdout_end,
            users=users,
            ground_truth=ground_truth,
            relevant=relevant,
            seen=exclusions,
            candidates=pool,
            label_coverage=coverage,
        )
