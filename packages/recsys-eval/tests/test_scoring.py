"""Tests for :mod:`recsys_eval.scoring`."""

from __future__ import annotations

import math
from collections.abc import Sequence, Set
from datetime import datetime, timedelta

import pytest

from recsys_eval.fixture import Fixture
from recsys_eval.protocols import Ranker
from recsys_eval.scoring import METRIC_NAMES, Scores, score
from recsys_eval.types import Interaction, RelevancePolicy

CUTOFF = datetime(2026, 1, 8)
WINDOW_END = datetime(2026, 1, 15)
BEFORE = CUTOFF - timedelta(days=1)
DURING = CUTOFF + timedelta(days=1)

POLICY = RelevancePolicy(positive_kinds=frozenset({"like"}))
POOL = ["i1", "i2", "i3", "i4"]


class PoolRanker:
    """Returns the pool in order, minus exclusions. Deterministic and dull."""

    def __init__(self, order: Sequence[str] | None = None) -> None:
        self.order = order
        self.prepared: list[Sequence[str]] = []
        self.rank_calls: list[tuple[str, int]] = []
        self._ordered: list[str] = []

    def prepare(self, candidates: Sequence[str]) -> None:
        self.prepared.append(candidates)
        self._ordered = list(self.order if self.order is not None else candidates)

    def rank(self, user: str, k: int, exclude: Set[str]) -> Sequence[str]:
        self.rank_calls.append((user, k))
        return [i for i in self._ordered if i not in exclude][:k]


def make_fixture(
    labels: dict[str, list[str]] | None = None,
) -> Fixture[str, str]:
    labels = labels or {"u1": ["i1"], "u2": ["i2"]}
    events: list[Interaction[str, str]] = []
    for user, items in labels.items():
        events.append(
            Interaction(user=user, item="i0", kind="view", weight=0.1, at=BEFORE)
        )
        for item in items:
            events.append(
                Interaction(user=user, item=item, kind="like", weight=1.0, at=DURING)
            )
    return Fixture.from_interactions(
        events,
        candidates=POOL,
        holdout_start=CUTOFF,
        holdout_end=WINDOW_END,
        policy=POLICY,
    )


# --------------------------------------------------------------------------
# Shape of the result
# --------------------------------------------------------------------------


def test_pool_ranker_satisfies_the_protocol() -> None:
    assert isinstance(PoolRanker(), Ranker)


def test_reports_every_requested_metric_at_every_k() -> None:
    scores = score(PoolRanker(), make_fixture(), metrics=["ndcg", "recall"], ks=[2, 4])
    assert set(scores.values) == {
        "ndcg@2",
        "ndcg@4",
        "recall@2",
        "recall@4",
        "coverage@2",
        "coverage@4",
    }


def test_coverage_is_always_reported_even_when_not_requested() -> None:
    """It cannot be switched off, because forgetting it is the failure mode."""
    scores = score(PoolRanker(), make_fixture(), metrics=["ndcg"], ks=[2])
    assert "coverage@2" in scores


def test_golden_macro_average() -> None:
    """Two users, one label each, both ranked [i1, i2] at k=2.

    u1's label is at rank 1: NDCG 1.0. u2's is at rank 2: 1/log2(3).
    Macro-average is their mean; Recall is 1.0 for both.
    """
    scores = score(PoolRanker(), make_fixture(), metrics=["ndcg", "recall"], ks=[2])
    expected_ndcg = (1.0 + 1.0 / math.log2(3)) / 2
    assert scores["ndcg@2"] == pytest.approx(expected_ndcg)
    assert scores["recall@2"] == pytest.approx(1.0)
    assert scores["coverage@2"] == pytest.approx(0.5)
    assert scores.scored_users == 2


def test_carries_label_coverage_through_from_the_fixture() -> None:
    """It belongs on the audit row beside the metrics, not somewhere else."""
    fixture = make_fixture({"u1": ["i1", "i9"]})
    scores = score(PoolRanker(), fixture, ks=[2])
    assert scores.label_coverage == pytest.approx(0.5)


def test_scores_accessors() -> None:
    scores = score(PoolRanker(), make_fixture(), metrics=["ndcg"], ks=[2])
    assert scores["ndcg@2"] == pytest.approx(scores.values["ndcg@2"])
    assert "ndcg@2" in scores
    assert "nope@2" not in scores
    flat = scores.as_dict()
    assert flat["scored_users"] == 2
    assert "label_coverage" in flat


# --------------------------------------------------------------------------
# How the ranker is driven
# --------------------------------------------------------------------------


def test_prepare_is_called_once_with_the_pool() -> None:
    ranker = PoolRanker()
    score(ranker, make_fixture(), ks=[2, 4])
    assert ranker.prepared == [("i1", "i2", "i3", "i4")]


def test_one_rank_call_per_user_at_the_deepest_k() -> None:
    """Shallower cutoffs are sliced from the same ranking.

    Re-calling per k would let a ranker that is unstable across calls report
    ``recall@10`` above ``recall@50``, which is arithmetically impossible and
    therefore very confusing to debug.
    """
    ranker = PoolRanker()
    score(ranker, make_fixture(), ks=[2, 4])
    assert ranker.rank_calls == [("u1", 4), ("u2", 4)]


def test_exclusions_are_immutable() -> None:
    """A ranker that adds its results to the caller's exclusion set would
    otherwise poison the next model scored with its own output."""
    captured: list[Set[str]] = []

    class Grabby:
        def prepare(self, candidates: Sequence[str]) -> None:
            pass

        def rank(self, user: str, k: int, exclude: Set[str]) -> Sequence[str]:
            captured.append(exclude)
            return list(POOL[:k])

    score(Grabby(), make_fixture(), ks=[2])
    assert captured
    for exclude in captured:
        assert isinstance(exclude, frozenset)
        with pytest.raises(AttributeError):
            exclude.add("x")  # type: ignore[attr-defined]


def test_rankings_longer_than_k_are_truncated() -> None:
    class Overeager:
        def prepare(self, candidates: Sequence[str]) -> None:
            pass

        def rank(self, user: str, k: int, exclude: Set[str]) -> Sequence[str]:
            return list(POOL)

    scores = score(Overeager(), make_fixture(), metrics=["recall"], ks=[1])
    # u1's label i1 is at rank 1; u2's i2 is at rank 2 and must not count.
    assert scores["recall@1"] == pytest.approx(0.5)


def test_under_delivery_is_not_excused() -> None:
    class Stingy:
        def prepare(self, candidates: Sequence[str]) -> None:
            pass

        def rank(self, user: str, k: int, exclude: Set[str]) -> Sequence[str]:
            return []

    scores = score(Stingy(), make_fixture(), metrics=["recall"], ks=[2])
    assert scores["recall@2"] == 0.0
    assert scores["coverage@2"] == 0.0


# --------------------------------------------------------------------------
# Coverage catches what per-user metrics cannot
# --------------------------------------------------------------------------


def test_coverage_detects_popularity_collapse() -> None:
    """Both rankers serve every user; the second serves everyone the same two.

    Recall is identical. Only coverage notices.
    """
    fixture = make_fixture({"u1": ["i1"], "u2": ["i2"]})
    diverse = score(PoolRanker(), fixture, metrics=["recall"], ks=[4])
    collapsed = score(
        PoolRanker(order=["i1", "i2"]), fixture, metrics=["recall"], ks=[4]
    )
    assert diverse["recall@4"] == pytest.approx(collapsed["recall@4"])
    assert collapsed["coverage@4"] < diverse["coverage@4"]


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------


def test_unknown_metric_is_rejected_before_the_ranker_is_prepared() -> None:
    """A typo must not surface as an expensive run that scores nothing."""
    ranker = PoolRanker()
    with pytest.raises(ValueError, match="Unknown metric"):
        score(ranker, make_fixture(), metrics=["ndgc"])
    assert ranker.prepared == []


def test_error_message_lists_the_valid_metric_names() -> None:
    with pytest.raises(ValueError) as excinfo:
        score(PoolRanker(), make_fixture(), metrics=["nope"])
    for name in METRIC_NAMES:
        assert name in str(excinfo.value)


@pytest.mark.parametrize("ks", [[0], [-1], [10, 0]])
def test_non_positive_k_is_rejected(ks: list[int]) -> None:
    with pytest.raises(ValueError, match="must all be positive"):
        score(PoolRanker(), make_fixture(), ks=ks)


def test_empty_metrics_or_ks_is_rejected() -> None:
    with pytest.raises(ValueError, match="nothing to report"):
        score(PoolRanker(), make_fixture(), metrics=[])
    with pytest.raises(ValueError, match="nothing to report"):
        score(PoolRanker(), make_fixture(), ks=[])


@pytest.mark.parametrize("name", sorted(METRIC_NAMES))
def test_every_advertised_metric_can_actually_be_scored(name: str) -> None:
    scores = score(PoolRanker(), make_fixture(), metrics=[name], ks=[2])
    assert 0.0 <= scores[f"{name}@2"] <= 1.0


# --------------------------------------------------------------------------
# Degenerate fixtures
# --------------------------------------------------------------------------


def test_empty_fixture_scores_zero_rather_than_raising() -> None:
    """An empty holdout is a normal state for a young system.

    The gate above reads ``scored_users`` and refuses to decide on it.
    """
    empty: Fixture[str, str] = Fixture.from_interactions(
        [Interaction(user="u1", item="i0", kind="view", weight=0.1, at=BEFORE)],
        candidates=POOL,
        holdout_start=CUTOFF,
        holdout_end=WINDOW_END,
        policy=POLICY,
    )
    scores = score(PoolRanker(), empty, metrics=["ndcg"], ks=[2])
    assert scores.scored_users == 0
    assert scores["ndcg@2"] == 0.0
    assert scores["coverage@2"] == 0.0


def test_hand_built_fixture_without_labels_is_survivable() -> None:
    """Scoring a label-less user at 0.0 would make the metric part headcount."""
    fixture: Fixture[str, str] = Fixture(
        holdout_start=CUTOFF,
        holdout_end=WINDOW_END,
        users=("u1",),
        ground_truth={"u1": {}},
        relevant={"u1": frozenset()},
        seen={"u1": frozenset()},
        candidates=tuple(POOL),
        label_coverage=0.0,
    )
    scores = score(PoolRanker(), fixture, metrics=["ndcg"], ks=[2])
    assert scores.scored_users == 0
    assert scores["ndcg@2"] == 0.0


def test_default_scores_are_empty() -> None:
    assert Scores().values == {}
    assert Scores().scored_users == 0


# --------------------------------------------------------------------------
# Per-user values
# --------------------------------------------------------------------------


def test_per_user_values_are_off_by_default() -> None:
    result = score(PoolRanker(), make_fixture(), metrics=["ndcg"], ks=[2])
    assert result.users == ()
    assert result.per_user == {}


def test_per_user_values_average_to_the_reported_metric() -> None:
    fixture = make_fixture({"u1": ["i1"], "u2": ["i4"], "u3": ["i2", "i3"]})
    result = score(
        PoolRanker(), fixture, metrics=["ndcg", "recall"], ks=[1, 4], per_user=True
    )
    assert result.users == fixture.users
    for key in ("ndcg@1", "ndcg@4", "recall@1", "recall@4"):
        values = result.per_user[key]
        assert len(values) == len(fixture.users)
        assert math.isclose(sum(values) / len(values), result.values[key])
    assert "coverage@1" not in result.per_user
    assert "per_user" not in result.as_dict()
