"""Tests for :mod:`recsys_eval.stats`."""

from __future__ import annotations

import math
import random

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from recsys_eval.stats import PairedComparison, paired_bootstrap


def sparse(n: int, rate: float, seed: int) -> list[float]:
    """Mostly zeros, like NDCG@10 on a real feed."""
    rng = random.Random(seed)
    return [rng.random() if rng.random() < rate else 0.0 for _ in range(n)]


def test_identical_models_have_an_interval_around_zero() -> None:
    values = sparse(2000, 0.05, 1)
    result = paired_bootstrap(values, values)
    assert result is not None
    assert result.relative_change == 0.0
    assert result.low == 0.0 == result.high


def test_a_clearly_worse_candidate_has_an_interval_below_zero() -> None:
    incumbent = sparse(3000, 0.10, 2)
    candidate = [v * 0.5 for v in incumbent]
    result = paired_bootstrap(candidate, incumbent)
    assert result is not None
    assert math.isclose(result.relative_change, -0.5)
    assert result.high < 0.0


def test_two_independent_equal_quality_models_are_not_confidently_different() -> None:
    """The case a point comparison gets wrong: same quality, different hits."""
    incumbent = sparse(2000, 0.03, 3)
    candidate = sparse(2000, 0.03, 4)
    result = paired_bootstrap(candidate, incumbent)
    assert result is not None
    assert result.low < 0.0 < result.high


def test_interval_width_follows_confidence() -> None:
    incumbent = sparse(1000, 0.1, 5)
    candidate = sparse(1000, 0.1, 6)
    narrow = paired_bootstrap(candidate, incumbent, confidence=0.5)
    wide = paired_bootstrap(candidate, incumbent, confidence=0.99)
    assert narrow is not None and wide is not None
    assert wide.low <= narrow.low <= narrow.high <= wide.high


def test_same_seed_same_interval_different_seed_different_interval() -> None:
    incumbent = sparse(500, 0.2, 7)
    candidate = sparse(500, 0.2, 8)
    a = paired_bootstrap(candidate, incumbent, seed=1)
    b = paired_bootstrap(candidate, incumbent, seed=1)
    c = paired_bootstrap(candidate, incumbent, seed=2)
    assert a == b
    assert a != c


def test_zero_incumbent_mean_has_no_baseline() -> None:
    assert paired_bootstrap([0.5, 0.1], [0.0, 0.0]) is None


def test_describe_and_as_dict() -> None:
    result = paired_bootstrap([0.2] * 200, [0.4] * 200, metric="ndcg@10")
    assert isinstance(result, PairedComparison)
    assert result.describe().startswith("ndcg@10 0.4000 -> 0.2000 (-50.0%, 95% CI")
    assert result.as_dict()["metric"] == "ndcg@10"
    assert result.as_dict()["users"] == 200


def test_mismatched_lengths_raise() -> None:
    with pytest.raises(ValueError, match="same users in the same order"):
        paired_bootstrap([0.1, 0.2], [0.1])


def test_no_users_raises() -> None:
    with pytest.raises(ValueError, match="No users"):
        paired_bootstrap([], [])


def test_nan_raises_rather_than_disabling_the_gate() -> None:
    with pytest.raises(ValueError, match="not finite"):
        paired_bootstrap([math.nan, 0.1], [0.1, 0.1])


@pytest.mark.parametrize("samples", [0, 99])
def test_too_few_samples_raise(samples: int) -> None:
    with pytest.raises(ValueError, match="samples must be"):
        paired_bootstrap([0.1], [0.1], samples=samples)


@pytest.mark.parametrize("confidence", [0.0, 1.0, 1.5])
def test_confidence_must_be_open_unit_interval(confidence: float) -> None:
    with pytest.raises(ValueError, match="confidence must be"):
        paired_bootstrap([0.1], [0.1], confidence=confidence)


@settings(max_examples=40, deadline=None)
@given(
    st.lists(
        st.tuples(st.floats(0, 1), st.floats(0, 1)), min_size=1, max_size=60
    ).filter(lambda pairs: sum(i for _, i in pairs) > 0)
)
def test_point_estimate_lies_inside_its_interval(
    pairs: list[tuple[float, float]],
) -> None:
    candidate = [c for c, _ in pairs]
    incumbent = [i for _, i in pairs]
    result = paired_bootstrap(candidate, incumbent, samples=200)
    assert result is not None
    assert all(
        math.isfinite(v) for v in (result.low, result.high, result.relative_change)
    )
    assert result.low <= result.high
    slack = 1e-9
    assert result.low - slack <= result.relative_change <= result.high + slack
