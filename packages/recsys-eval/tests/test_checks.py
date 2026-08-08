"""Tests for :mod:`recsys_eval.checks`.

Each check exists because of a specific production failure. The fakes here
reproduce those failures deliberately: a ranker whose features drift with the
clock, two rankers sharing a cache, a fixture builder with a non-deterministic
truncation.
"""

from __future__ import annotations

import dataclasses
import itertools
from collections.abc import Sequence, Set

import pytest
from test_scoring import POOL, PoolRanker, make_fixture

from recsys_eval.checks import (
    CheckFailed,
    assert_deterministic,
    assert_discriminating,
    assert_reproducible,
    assert_within_unit_interval,
    preflight,
)
from recsys_eval.fixture import Fixture
from recsys_eval.scoring import Scores


class DriftingRanker:
    """Feature drift, reproduced: the order depends on a mutating counter.

    Stands in for anything derived from ``now()`` -- item age, recency decay,
    a cache entry that expired between the two passes.
    """

    def __init__(self) -> None:
        self.clock = itertools.count()

    def prepare(self, candidates: Sequence[str]) -> None:
        tick = next(self.clock)
        self._ordered = list(candidates)[tick % 2 :] + list(candidates)[
            : tick % 2
        ]

    def rank(self, user: str, k: int, exclude: Set[str]) -> Sequence[str]:
        return [i for i in self._ordered if i not in exclude][:k]


# --------------------------------------------------------------------------
# assert_deterministic
# --------------------------------------------------------------------------


def test_deterministic_passes_for_a_pure_ranker() -> None:
    assert_deterministic(PoolRanker(), make_fixture(), ks=[2])


def test_deterministic_catches_clock_dependent_features() -> None:
    with pytest.raises(CheckFailed) as excinfo:
        assert_deterministic(DriftingRanker(), make_fixture(), ks=[2])
    message = str(excinfo.value)
    assert "different results" in message
    assert "docs/hazards.md" in message


def test_deterministic_error_names_the_metrics_that_moved() -> None:
    with pytest.raises(CheckFailed) as excinfo:
        assert_deterministic(
            DriftingRanker(), make_fixture(), metrics=["ndcg"], ks=[2]
        )
    assert "ndcg@2" in str(excinfo.value)


# --------------------------------------------------------------------------
# assert_discriminating
# --------------------------------------------------------------------------


def test_discriminating_passes_for_genuinely_different_rankers() -> None:
    assert_discriminating(
        PoolRanker(),
        PoolRanker(order=["i4", "i3", "i2", "i1"]),
        make_fixture(),
        ks=[2],
    )


def test_discriminating_catches_a_shared_cache() -> None:
    """Two "different" rankers reading the same cached order score identically.

    That reads as "no regression", which is a promotion.
    """
    shared = ["i1", "i2", "i3", "i4"]
    with pytest.raises(CheckFailed) as excinfo:
        assert_discriminating(
            PoolRanker(order=shared),
            PoolRanker(order=shared),
            make_fixture(),
            ks=[2],
        )
    message = str(excinfo.value)
    assert "identical scores" in message
    assert "cache" in message


# --------------------------------------------------------------------------
# assert_reproducible
# --------------------------------------------------------------------------


def test_reproducible_passes_for_a_stable_builder() -> None:
    assert_reproducible(make_fixture)


def test_reproducible_catches_non_deterministic_truncation() -> None:
    """A pool ordered by a non-unique column, truncated: ties cut differently."""
    orders = itertools.cycle([POOL, list(reversed(POOL))])

    def build() -> Fixture[str, str]:
        return dataclasses.replace(
            make_fixture(), candidates=tuple(next(orders))[:2]
        )

    with pytest.raises(CheckFailed) as excinfo:
        assert_reproducible(build)
    assert "candidates" in str(excinfo.value)


def test_reproducible_catches_a_moving_user_sample() -> None:
    users = itertools.cycle([{"u1": ["i1"]}, {"u2": ["i2"]}])

    def build() -> Fixture[str, str]:
        return make_fixture(next(users))

    with pytest.raises(CheckFailed) as excinfo:
        assert_reproducible(build)
    assert "users" in str(excinfo.value)


# --------------------------------------------------------------------------
# assert_within_unit_interval
# --------------------------------------------------------------------------


def test_unit_interval_passes_for_real_scores() -> None:
    assert_within_unit_interval(
        Scores(values={"ndcg@10": 0.4, "recall@10": 1.0, "hit@10": 0.0})
    )


@pytest.mark.parametrize(
    ("value", "hint"),
    [
        (1.6, "duplicate"),
        (-0.5, "negative gains"),
        (float("nan"), "NaN"),
    ],
)
def test_unit_interval_rejects_impossible_values(
    value: float, hint: str
) -> None:
    with pytest.raises(CheckFailed) as excinfo:
        assert_within_unit_interval(Scores(values={"ndcg@10": value}))
    message = str(excinfo.value)
    assert "ndcg@10" in message
    assert hint in message


# --------------------------------------------------------------------------
# preflight
# --------------------------------------------------------------------------


def test_preflight_returns_the_candidate_scores_when_everything_passes() -> None:
    scores = preflight(
        make_fixture,
        PoolRanker(),
        PoolRanker(order=["i4", "i3", "i2", "i1"]),
        metrics=["ndcg", "recall"],
        ks=[2],
    )
    assert scores.scored_users == 2
    assert "ndcg@2" in scores


def test_preflight_surfaces_a_failing_candidate() -> None:
    with pytest.raises(CheckFailed, match="different results"):
        preflight(
            make_fixture,
            DriftingRanker(),
            PoolRanker(),
            ks=[2],
        )


def test_preflight_surfaces_a_failing_incumbent() -> None:
    with pytest.raises(CheckFailed, match="different results"):
        preflight(
            make_fixture,
            PoolRanker(),
            DriftingRanker(),
            ks=[2],
        )
