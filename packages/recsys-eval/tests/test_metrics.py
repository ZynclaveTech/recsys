"""Tests for :mod:`recsys_eval.metrics`.

Two layers. The golden tests pin the exact arithmetic, so a refactor that
changes a definition fails loudly rather than shifting every dashboard by 2%.
The property tests pin the invariants that the promotion gate relies on --
chiefly that nothing ever escapes ``[0, 1]`` and nothing ever returns ``NaN``,
because a ``NaN`` propagating into a macro-average disables the gate silently.
"""

from __future__ import annotations

import math

import pytest
from hypothesis import given
from hypothesis import strategies as st

from recsys_eval.metrics import (
    average_precision_at_k,
    catalog_coverage,
    hit_rate_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank_at_k,
)

# --------------------------------------------------------------------------
# Degenerate input: the contract is 0.0, never NaN, never raising.
# --------------------------------------------------------------------------

BINARY_METRICS = [
    recall_at_k,
    precision_at_k,
    average_precision_at_k,
    reciprocal_rank_at_k,
    hit_rate_at_k,
]


@pytest.mark.parametrize("metric", BINARY_METRICS)
def test_binary_metrics_return_zero_on_empty_ranking(metric) -> None:  # type: ignore[no-untyped-def]
    assert metric([], {"a"}, 10) == 0.0


@pytest.mark.parametrize("metric", BINARY_METRICS)
def test_binary_metrics_return_zero_on_empty_labels(metric) -> None:  # type: ignore[no-untyped-def]
    assert metric(["a", "b"], set(), 10) == 0.0


@pytest.mark.parametrize("metric", BINARY_METRICS)
@pytest.mark.parametrize("k", [0, -1])
def test_binary_metrics_return_zero_on_nonpositive_k(metric, k: int) -> None:  # type: ignore[no-untyped-def]
    assert metric(["a", "b"], {"a"}, k) == 0.0


def test_ndcg_returns_zero_on_empty_inputs() -> None:
    assert ndcg_at_k([], {"a": 1.0}, 10) == 0.0
    assert ndcg_at_k(["a"], {}, 10) == 0.0
    assert ndcg_at_k(["a"], {"a": 1.0}, 0) == 0.0


def test_ndcg_returns_zero_when_all_gains_are_non_positive() -> None:
    # IDCG is 0, so the ratio is undefined. Must not raise ZeroDivisionError
    # and must not return NaN.
    result = ndcg_at_k(["a", "b"], {"a": 0.0, "b": -1.0}, 10)
    assert result == 0.0
    assert not math.isnan(result)


# --------------------------------------------------------------------------
# NDCG: graded relevance
# --------------------------------------------------------------------------


def test_ndcg_golden_value() -> None:
    # ranked = [a, b, c], gains = {a: 3.0, c: 1.0}, k = 3
    #   DCG  = 3/log2(2) + 0 + 1/log2(4) = 3.0 + 0.5           = 3.5
    #   IDCG = 3/log2(2) + 1/log2(3)     = 3.0 + 0.63092975... = 3.63092975...
    expected = 3.5 / (3.0 + 1.0 / math.log2(3))
    assert ndcg_at_k(["a", "b", "c"], {"a": 3.0, "c": 1.0}, 3) == pytest.approx(
        expected
    )
    assert expected == pytest.approx(0.9639404, abs=1e-6)


def test_ndcg_counts_duplicates_once() -> None:
    """A repeated item must not be paid for twice.

    Found by :func:`test_ndcg_stays_within_unit_interval`, not by hand. Without
    the guard, ``["a", "a"]`` against ``{"a": 1.0}`` accumulates gain twice
    while IDCG counts it once, scoring 1.63 -- an unbounded free pass for a
    candidate whose only talent is repeating itself.
    """
    assert ndcg_at_k(["a", "a"], {"a": 1.0}, 2) == pytest.approx(1.0)
    assert ndcg_at_k(["a", "a", "b"], {"a": 3.0, "b": 1.0}, 3) < 1.0


def test_ndcg_ignores_negative_gains_symmetrically() -> None:
    """Negative gains must be skipped in DCG exactly as in the ideal ordering.

    Also found by the property test. Counting a negative in DCG while the ideal
    ordering excludes it drives NDCG below zero -- and interaction-weight
    tables routinely carry negatives (``block: -5.0``), so this is one config
    change away from live rather than a theoretical concern.
    """
    assert ndcg_at_k(["a"], {"a": -1.0, "b": 1.0}, 1) == 0.0
    # A disliked item occupying rank 1 is invisible to NDCG, not negative.
    # Measure that harm with a separate penalty metric.
    assert ndcg_at_k(["bad", "good"], {"bad": -5.0, "good": 1.0}, 2) == pytest.approx(
        1.0 / math.log2(3)
    )


def test_ndcg_is_one_for_ideal_ordering() -> None:
    assert ndcg_at_k(["a", "c"], {"a": 3.0, "c": 1.0}, 10) == pytest.approx(1.0)


def test_ndcg_rewards_ordering_by_grade() -> None:
    """A share ranked above a like must beat a like ranked above a share.

    This is the whole reason for graded relevance. If this test can be made to
    pass with binary labels, the grading is not being used.
    """
    gains = {"share": 3.0, "like": 1.0}
    better = ndcg_at_k(["share", "like"], gains, 10)
    worse = ndcg_at_k(["like", "share"], gains, 10)
    assert better > worse


def test_ndcg_penalises_unretrieved_relevant_items() -> None:
    """IDCG spans all known gains, not only the retrieved ones.

    Retrieving one of two good items must score strictly below retrieving both,
    even though the retrieved item is in the ideal position.
    """
    gains = {"a": 3.0, "b": 3.0}
    partial = ndcg_at_k(["a"], gains, 10)
    complete = ndcg_at_k(["a", "b"], gains, 10)
    assert partial < complete == pytest.approx(1.0)


def test_ndcg_ignores_unlabelled_items() -> None:
    assert ndcg_at_k(["x", "y", "z"], {"a": 1.0}, 10) == 0.0


# --------------------------------------------------------------------------
# Recall: unreachable labels stay in the denominator
# --------------------------------------------------------------------------


def test_recall_golden_value() -> None:
    assert recall_at_k(["a", "b"], {"a", "c", "d"}, 2) == pytest.approx(1 / 3)


def test_recall_keeps_unretrievable_labels_in_the_denominator() -> None:
    """The load-bearing decision in this module.

    ``c`` is relevant but was never a candidate. Recall must be 1/2, not 1/1.
    Dropping it would hide a shrinking candidate index -- precisely the
    regression a promotion gate exists to catch.
    """
    assert recall_at_k(["a"], {"a", "c"}, 10) == pytest.approx(0.5)


def test_recall_is_capped_by_k() -> None:
    assert recall_at_k(["a", "b", "c"], {"a", "b", "c"}, 2) == pytest.approx(2 / 3)


# --------------------------------------------------------------------------
# Precision: denominator is k, not the delivered length
# --------------------------------------------------------------------------


def test_precision_golden_value() -> None:
    assert precision_at_k(["a", "b", "c", "d"], {"a", "c"}, 4) == pytest.approx(0.5)


def test_precision_divides_by_k_not_by_delivered_length() -> None:
    """Under-delivery is a regression, not a free pass.

    One relevant item returned when ten were requested is 0.1, not 1.0.
    """
    assert precision_at_k(["a"], {"a"}, 10) == pytest.approx(0.1)


# --------------------------------------------------------------------------
# Average Precision: rank-sensitive, denominator min(|relevant|, k)
# --------------------------------------------------------------------------


def test_average_precision_golden_value() -> None:
    # ranked = [a, b, c, d], relevant = {a, c}, k = 4
    #   hit at rank 1 -> precision 1/1 = 1.0
    #   hit at rank 3 -> precision 2/3 = 0.666...
    #   denominator   -> min(2, 4) = 2
    expected = (1.0 + 2 / 3) / 2
    assert average_precision_at_k(["a", "b", "c", "d"], {"a", "c"}, 4) == pytest.approx(
        expected
    )
    assert expected == pytest.approx(0.8333333, abs=1e-6)


def test_average_precision_is_one_for_ideal_ordering() -> None:
    assert average_precision_at_k(["a", "b", "c"], {"a", "b"}, 3) == pytest.approx(1.0)


def test_average_precision_rewards_earlier_hits() -> None:
    early = average_precision_at_k(["a", "x", "y"], {"a"}, 3)
    late = average_precision_at_k(["x", "y", "a"], {"a"}, 3)
    assert early > late


def test_average_precision_denominator_is_capped_at_k() -> None:
    """A perfect top-k must score 1.0 even when more labels exist below it.

    With five relevant items and k=2, dividing by 5 would cap the best possible
    score at 0.4 and make the metric unreadable.
    """
    relevant = {"a", "b", "c", "d", "e"}
    assert average_precision_at_k(["a", "b"], relevant, 2) == pytest.approx(1.0)


def test_average_precision_counts_duplicates_once() -> None:
    """A ranker that repeats an item must not be paid twice for it."""
    deduped = average_precision_at_k(["a", "b"], {"a", "b"}, 4)
    duplicated = average_precision_at_k(["a", "a", "b"], {"a", "b"}, 4)
    assert duplicated < deduped == pytest.approx(1.0)


# --------------------------------------------------------------------------
# Reciprocal rank and hit rate
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("ranked", "expected"),
    [
        (["a", "x", "y"], 1.0),
        (["x", "a", "y"], 0.5),
        (["x", "y", "a"], 1 / 3),
        (["x", "y", "z"], 0.0),
    ],
)
def test_reciprocal_rank(ranked: list[str], expected: float) -> None:
    assert reciprocal_rank_at_k(ranked, {"a"}, 3) == pytest.approx(expected)


def test_reciprocal_rank_respects_the_cutoff() -> None:
    assert reciprocal_rank_at_k(["x", "y", "a"], {"a"}, 2) == 0.0


def test_hit_rate_is_binary() -> None:
    assert hit_rate_at_k(["x", "a"], {"a"}, 2) == 1.0
    assert hit_rate_at_k(["x", "y"], {"a"}, 2) == 0.0
    assert hit_rate_at_k(["x", "y", "a"], {"a"}, 2) == 0.0


# --------------------------------------------------------------------------
# Catalog coverage
# --------------------------------------------------------------------------


def test_catalog_coverage_golden_value() -> None:
    lists = [["a", "b"], ["b", "c"]]
    assert catalog_coverage(lists, catalog_size=10, k=2) == pytest.approx(0.3)


def test_catalog_coverage_detects_popularity_collapse() -> None:
    """The failure mode this metric exists for.

    Both rankers serve every user. The second serves everyone the same two
    items -- a change every per-user accuracy metric here is blind to.
    """
    diverse = catalog_coverage([["a", "b"], ["c", "d"]], catalog_size=4, k=2)
    collapsed = catalog_coverage([["a", "b"], ["a", "b"]], catalog_size=4, k=2)
    assert diverse == pytest.approx(1.0)
    assert collapsed == pytest.approx(0.5)


def test_catalog_coverage_clamps_rather_than_exceeding_one() -> None:
    """A stale catalog_size must not produce a value above 1.0.

    Returning 1.5 here would silently break any downstream tolerance check that
    assumes the metric is a fraction.
    """
    assert catalog_coverage([["a", "b", "c"]], catalog_size=2, k=3) == 1.0


def test_catalog_coverage_handles_empty_input() -> None:
    assert catalog_coverage([], catalog_size=10, k=5) == 0.0
    assert catalog_coverage([["a"]], catalog_size=0, k=5) == 0.0


def test_catalog_coverage_accepts_a_generator() -> None:
    """``ranked_lists`` is consumed once; a generator must work."""
    lists = (["a", "b"] for _ in range(2))
    assert catalog_coverage(lists, catalog_size=4, k=2) == pytest.approx(0.5)


# --------------------------------------------------------------------------
# Properties. These are the invariants the promotion gate depends on.
# --------------------------------------------------------------------------

_items = st.sampled_from(["a", "b", "c", "d", "e", "f"])
_rankings = st.lists(_items, min_size=0, max_size=8)
_label_sets = st.sets(_items, min_size=0, max_size=6)
_gain_maps = st.dictionaries(
    _items, st.floats(min_value=-5, max_value=5, allow_nan=False), max_size=6
)
_ks = st.integers(min_value=0, max_value=12)


@pytest.mark.parametrize("metric", BINARY_METRICS)
@given(ranked=_rankings, relevant=_label_sets, k=_ks)
def test_binary_metrics_stay_within_unit_interval(  # type: ignore[no-untyped-def]
    metric, ranked: list[str], relevant: set[str], k: int
) -> None:
    result = metric(ranked, relevant, k)
    assert not math.isnan(result)
    assert 0.0 <= result <= 1.0


@given(ranked=_rankings, gains=_gain_maps, k=_ks)
def test_ndcg_stays_within_unit_interval(
    ranked: list[str], gains: dict[str, float], k: int
) -> None:
    result = ndcg_at_k(ranked, gains, k)
    assert not math.isnan(result)
    assert 0.0 <= result <= 1.0


@given(ranked=_rankings, relevant=_label_sets, k=st.integers(1, 12))
def test_hit_rate_agrees_with_reciprocal_rank(
    ranked: list[str], relevant: set[str], k: int
) -> None:
    """Both answer "was there a hit"; they must never disagree."""
    assert (hit_rate_at_k(ranked, relevant, k) > 0) == (
        reciprocal_rank_at_k(ranked, relevant, k) > 0
    )


@given(ranked=_rankings, relevant=_label_sets, k=st.integers(1, 12))
def test_recall_is_monotonic_in_k(
    ranked: list[str], relevant: set[str], k: int
) -> None:
    """Widening the cutoff can only ever find more."""
    assert recall_at_k(ranked, relevant, k) <= recall_at_k(ranked, relevant, k + 1)


@given(ranked=_rankings, relevant=_label_sets, k=st.integers(1, 12))
def test_recall_ignores_order_within_the_cutoff(
    ranked: list[str], relevant: set[str], k: int
) -> None:
    """Recall is a set metric. If order moves it, something is wrong.

    This is the property that makes Recall and NDCG complementary: pairing a
    set metric with a rank-sensitive one is what stops a gate from being fooled
    by a reshuffle.
    """
    reversed_head = list(reversed(ranked[:k])) + list(ranked[k:])
    assert recall_at_k(ranked, relevant, k) == pytest.approx(
        recall_at_k(reversed_head, relevant, k)
    )
