"""Tests for :mod:`recsys_candidates.merge` and the budget vocabulary.

The merge tests are mostly about *properties a merge must have* — agreement is
rewarded, ties are deterministic, no source silently disappears — rather than
exact fused values, which are an implementation detail of the RRF constant.
"""

from __future__ import annotations

import pytest

from recsys_candidates import (
    RRF_K,
    Budget,
    Candidate,
    by_score,
    interleave,
    reciprocal_rank_fusion,
    sources_of,
)


def cands(
    source: str, *names: str, scores: list[float] | None = None
) -> list[Candidate[str]]:
    return [
        Candidate(
            item=name,
            rank=n,
            source=source,
            score=None if scores is None else scores[n],
        )
        for n, name in enumerate(names)
    ]


def items(merged: list[Candidate[str]]) -> list[str]:
    return [c.item for c in merged]


# --------------------------------------------------------------------------
# Reciprocal rank fusion
# --------------------------------------------------------------------------


def test_agreement_between_sources_wins() -> None:
    """The whole reason to fuse: two sources agreeing beats one source's top.

    `b` is rank 1 in both sources; `a` is rank 0 in one and absent from the
    other. Fusion should surface `b`.
    """
    merged = reciprocal_rank_fusion(
        {"x": cands("x", "a", "b"), "y": cands("y", "c", "b")}
    )
    assert merged[0].item == "b"


def test_deduplicates_and_keeps_the_best_ranked_instance() -> None:
    """The surviving candidate should carry the source that ranked it best,
    because that is the one whose `meta` is most likely to be useful."""
    merged = reciprocal_rank_fusion({"x": cands("x", "z", "a"), "y": cands("y", "a")})
    a = next(c for c in merged if c.item == "a")
    assert a.source == "y"
    assert a.rank == 0
    assert len(items(merged)) == len(set(items(merged)))


def test_weights_express_trust_in_a_source() -> None:
    low = reciprocal_rank_fusion({"x": cands("x", "a"), "y": cands("y", "b")})
    high = reciprocal_rank_fusion(
        {"x": cands("x", "a"), "y": cands("y", "b")}, weights={"y": 10.0}
    )
    assert low[0].item == "a"
    assert high[0].item == "b"


def test_ties_break_deterministically_by_first_appearance() -> None:
    """Otherwise equal-contribution items come out in dict order, which shifts
    the moment a source is added — noise that reads as a ranking change."""
    a = reciprocal_rank_fusion({"x": cands("x", "p"), "y": cands("y", "q")})
    b = reciprocal_rank_fusion({"x": cands("x", "p"), "y": cands("y", "q")})
    assert items(a) == items(b) == ["p", "q"]


def test_no_source_is_silently_dropped() -> None:
    merged = reciprocal_rank_fusion(
        {"x": cands("x", "a"), "y": cands("y", "b"), "z": cands("z", "c")}
    )
    assert set(sources_of(merged)) == {"x", "y", "z"}


def test_empty_sources_are_survivable() -> None:
    assert reciprocal_rank_fusion({}) == []
    assert items(reciprocal_rank_fusion({"x": [], "y": cands("y", "a")})) == ["a"]


def test_negative_weights_are_rejected() -> None:
    """A negative weight makes being found count against an item, which is
    almost never what is meant."""
    with pytest.raises(ValueError, match="Negative weights"):
        reciprocal_rank_fusion({"x": cands("x", "a")}, weights={"x": -1.0})


def test_k_flattens_the_curve() -> None:
    """Large k reduces the advantage of a top rank, so deep agreement can win."""
    per_source = {"x": cands("x", "a", "b", "c"), "y": cands("y", "c", "b", "a")}
    assert reciprocal_rank_fusion(per_source, k=0)[0].item == "a"
    assert RRF_K > 0


def test_score_is_never_consulted_by_rrf() -> None:
    """The point of rank fusion. A wildly larger score must not change order."""
    plain = reciprocal_rank_fusion({"x": cands("x", "a", "b"), "y": cands("y", "c")})
    inflated = reciprocal_rank_fusion(
        {
            "x": cands("x", "a", "b", scores=[0.01, 0.01]),
            "y": cands("y", "c", scores=[9999.0]),
        }
    )
    assert items(plain) == items(inflated)


# --------------------------------------------------------------------------
# Interleave
# --------------------------------------------------------------------------


def test_interleave_gives_every_source_a_slot_near_the_top() -> None:
    merged = interleave({"x": cands("x", "a", "b"), "y": cands("y", "c", "d")})
    assert items(merged)[:2] == ["a", "c"]


def test_interleave_respects_an_explicit_order() -> None:
    merged = interleave({"x": cands("x", "a"), "y": cands("y", "c")}, order=["y", "x"])
    assert items(merged) == ["c", "a"]


def test_interleave_includes_sources_missing_from_order() -> None:
    """Omitting a source from `order` must not drop it — that would be a
    silent content loss with no error."""
    merged = interleave({"x": cands("x", "a"), "y": cands("y", "c")}, order=["y"])
    assert set(items(merged)) == {"a", "c"}


def test_interleave_drains_uneven_sources() -> None:
    merged = interleave({"x": cands("x", "a", "b", "c"), "y": cands("y", "d")})
    assert set(items(merged)) == {"a", "b", "c", "d"}


def test_interleave_deduplicates() -> None:
    merged = interleave({"x": cands("x", "a"), "y": cands("y", "a", "b")})
    assert items(merged) == ["a", "b"]


# --------------------------------------------------------------------------
# by_score
# --------------------------------------------------------------------------


def test_by_score_uses_magnitude() -> None:
    merged = by_score(
        {
            "x": cands("x", "a", "b", scores=[0.9, 0.1]),
            "y": cands("y", "c", scores=[0.5]),
        }
    )
    assert items(merged) == ["a", "c", "b"]


def test_by_score_refuses_missing_scores() -> None:
    """Treating None as 0.0 would rank those items last while looking like it
    worked — the exact class of silent failure this package exists to avoid."""
    with pytest.raises(ValueError, match="without a"):
        by_score({"x": cands("x", "a")})


def test_by_score_applies_weights() -> None:
    merged = by_score(
        {
            "x": cands("x", "a", scores=[0.9]),
            "y": cands("y", "b", scores=[0.5]),
        },
        weights={"y": 3.0},
    )
    assert items(merged) == ["b", "a"]


# --------------------------------------------------------------------------
# Budget
# --------------------------------------------------------------------------


def test_fetch_size_applies_overfetch() -> None:
    assert Budget(total=100).fetch_size("anything") == 300


def test_per_source_cap_lowers_the_ask() -> None:
    b = Budget(total=100, per_source={"cheap": 10})
    assert b.fetch_size("cheap") == 30
    assert b.fetch_size("other") == 300


def test_min_per_source_raises_a_capped_source() -> None:
    """A floor is pointless if the fetch is capped below it."""
    b = Budget(total=100, per_source={"small": 2}, min_per_source={"small": 5})
    assert b.fetch_size("small") == 15


@pytest.mark.parametrize("total", [0, -1])
def test_non_positive_total_is_rejected(total: int) -> None:
    with pytest.raises(ValueError, match="total must be positive"):
        Budget(total=total)


def test_overfetch_below_one_is_rejected() -> None:
    """It would ask for fewer than you intend to return, so filtering could
    only ever produce a short slate."""
    with pytest.raises(ValueError, match="overfetch must be at least 1"):
        Budget(total=10, overfetch=0)


def test_floors_exceeding_total_are_rejected() -> None:
    """Which source loses would otherwise depend on iteration order."""
    with pytest.raises(ValueError, match="cannot all be honoured"):
        Budget(total=5, min_per_source={"a": 3, "b": 3})


def test_negative_caps_are_rejected() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        Budget(total=10, per_source={"a": -1})


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------


def test_sources_of_counts_contributions() -> None:
    """Worth logging per request: a source that has silently stopped
    contributing looks identical to one that is working."""
    merged = reciprocal_rank_fusion({"x": cands("x", "a", "b"), "y": cands("y", "c")})
    assert sources_of(merged) == {"x": 2, "y": 1}
