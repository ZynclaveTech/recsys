"""Score one ranker over one fixture.

Macro-averaged across users: every user counts once, regardless of how many
labels they have. The alternative -- pooling all labels and micro-averaging --
lets a handful of hyperactive users decide the number, and those users are
exactly the ones whose behaviour a ranking change is least likely to alter.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence, Set
from dataclasses import dataclass, field
from typing import Any

from recsys_eval.fixture import Fixture
from recsys_eval.metrics import (
    average_precision_at_k,
    hit_rate_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank_at_k,
)
from recsys_eval.protocols import Ranker
from recsys_eval.types import ItemT, UserT

__all__ = [
    "DEFAULT_KS",
    "DEFAULT_METRICS",
    "METRIC_NAMES",
    "Scores",
    "score",
]

# Metrics taking graded gains: user -> {item: gain}.
_GRADED = {"ndcg": ndcg_at_k}

# Metrics taking binary relevance: user -> {item}.
_BINARY = {
    "recall": recall_at_k,
    "precision": precision_at_k,
    "ap": average_precision_at_k,
    "rr": reciprocal_rank_at_k,
    "hit": hit_rate_at_k,
}

METRIC_NAMES = frozenset(_GRADED) | frozenset(_BINARY)

DEFAULT_METRICS = ("ndcg", "recall")
"""One rank-sensitive metric and one set metric.

A ranker can trade these against each other -- reorder well while retrieving
less, or retrieve more and order it badly -- so a gate reading only one of them
can be walked straight past. Pairing them is the cheapest protection there is.
"""

DEFAULT_KS = (10, 50)
"""A shallow cutoff and a deep one: ordering at the top, coverage at depth."""


@dataclass(frozen=True)
class Scores:
    """One ranker's results over one fixture.

    Attributes:
        values: Metric name -> value, keyed ``"<metric>@<k>"``. Always includes
            ``"coverage@<k>"`` for every ``k``.
        scored_users: How many users contributed. A number that moves between
            runs means the fixture moved, not the model.
        label_coverage: Carried through from the fixture. It is the ceiling on
            Recall, and it belongs next to the metrics rather than somewhere
            else, because a score that fell because the pool shrank looks
            identical to one that fell because the model got worse.
    """

    values: Mapping[str, float] = field(default_factory=dict)
    scored_users: int = 0
    label_coverage: float = 0.0

    def __getitem__(self, key: str) -> float:
        return self.values[key]

    def __contains__(self, key: str) -> bool:
        return key in self.values

    def as_dict(self) -> dict[str, Any]:
        """Flat dict suitable for logging or an audit row."""
        return {
            **self.values,
            "scored_users": self.scored_users,
            "label_coverage": self.label_coverage,
        }


def _validate(metrics: Sequence[str], ks: Sequence[int]) -> None:
    unknown = sorted(set(metrics) - METRIC_NAMES)
    if unknown:
        raise ValueError(
            f"Unknown metric(s): {', '.join(unknown)}. "
            f"Available: {', '.join(sorted(METRIC_NAMES))}."
        )
    if not metrics:
        raise ValueError("metrics is empty; there would be nothing to report.")
    if not ks:
        raise ValueError("ks is empty; there would be nothing to report.")
    bad_ks = sorted(k for k in ks if k <= 0)
    if bad_ks:
        raise ValueError(f"ks must all be positive, got {bad_ks}.")


def score(
    ranker: Ranker[UserT, ItemT],
    fixture: Fixture[UserT, ItemT],
    *,
    metrics: Sequence[str] = DEFAULT_METRICS,
    ks: Sequence[int] = DEFAULT_KS,
) -> Scores:
    """Score ``ranker`` over ``fixture``.

    The ranker is prepared once with the fixture's pool, then asked for one
    ranking per user at the deepest ``k``; shallower cutoffs are taken from the
    same ranking rather than from repeated calls, so a ranker that is not
    stable across calls cannot produce inconsistent numbers here.

    ``coverage@k`` is always reported and cannot be switched off. A ranker can
    raise every per-user metric on this page by collapsing onto a small set of
    universally-popular items, and coverage is the only thing here that
    notices. Treat a large accuracy gain paired with a coverage collapse as a
    regression.

    Args:
        ranker: Anything satisfying :class:`~recsys_eval.protocols.Ranker`.
        fixture: Built by :meth:`~recsys_eval.fixture.Fixture.from_interactions`.
            The *same* fixture object must be used for every model in a
            comparison.
        metrics: Names from :data:`METRIC_NAMES`.
        ks: Cutoffs to report at.

    Returns:
        :class:`Scores`. Every value is in ``[0, 1]``.

    Raises:
        ValueError: On an unknown metric name or a non-positive cutoff. Raised
            eagerly, before the ranker is prepared, so a typo does not surface
            as an expensive run that scores nothing.
    """
    _validate(metrics, ks)

    if not fixture.users or not fixture.candidates:
        # Nothing to score. Report zeros rather than raising: an empty holdout
        # is a normal state for a young system, and the gate above will read
        # the user count and refuse to make a decision on it.
        empty = {f"{m}@{k}": 0.0 for m in metrics for k in ks}
        empty.update({f"coverage@{k}": 0.0 for k in ks})
        return Scores(
            values=empty,
            scored_users=0,
            label_coverage=fixture.label_coverage,
        )

    ranker.prepare(fixture.candidates)

    max_k = max(ks)
    totals = {f"{m}@{k}": 0.0 for m in metrics for k in ks}
    surfaced: dict[int, set[ItemT]] = {k: set() for k in ks}
    scored_users = 0

    for user in fixture.users:
        gains = fixture.ground_truth.get(user, {})
        relevant = fixture.relevant.get(user, frozenset())
        if not gains or not relevant:
            # Cannot happen for a fixture from `from_interactions`, which only
            # admits users with labels. Guarded because a hand-built fixture
            # can, and scoring a label-less user at 0.0 would make the metric
            # partly a headcount of unusable users.
            continue

        exclude: Set[ItemT] = fixture.seen.get(user, frozenset())
        ranked = list(ranker.rank(user, max_k, exclude))[:max_k]

        for k in ks:
            surfaced[k].update(ranked[:k])
            for name in metrics:
                if name in _GRADED:
                    value = _GRADED[name](ranked, gains, k)
                else:
                    value = _BINARY[name](ranked, relevant, k)
                totals[f"{name}@{k}"] += value

        scored_users += 1

    if scored_users == 0:
        values = dict.fromkeys(totals, 0.0)
        values.update({f"coverage@{k}": 0.0 for k in ks})
        return Scores(
            values=values,
            scored_users=0,
            label_coverage=fixture.label_coverage,
        )

    pool_size = len(fixture.candidates)
    values = {key: total / scored_users for key, total in totals.items()}
    values.update({f"coverage@{k}": min(len(surfaced[k]) / pool_size, 1.0) for k in ks})

    return Scores(
        values=values,
        scored_users=scored_users,
        label_coverage=fixture.label_coverage,
    )
