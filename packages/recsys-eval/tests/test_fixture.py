"""Tests for :mod:`recsys_eval.fixture`.

The fixture is where an evaluation harness usually goes wrong: not by crashing,
but by quietly building labels that make every model look the same. Most of
these tests assert on that class of silent failure rather than on happy paths.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from recsys_eval.fixture import Fixture
from recsys_eval.types import Aggregate, Interaction, RelevancePolicy

CUTOFF = datetime(2026, 1, 8)
WINDOW_END = datetime(2026, 1, 15)
BEFORE = CUTOFF - timedelta(days=1)
DURING = CUTOFF + timedelta(days=1)
AFTER = WINDOW_END + timedelta(days=1)

POLICY = RelevancePolicy(positive_kinds=frozenset({"like", "share", "save"}))
POOL = ["i1", "i2", "i3", "i4"]


def ev(
    user: str,
    item: str,
    kind: str = "like",
    weight: float = 1.0,
    at: datetime = DURING,
) -> Interaction[str, str]:
    return Interaction(user=user, item=item, kind=kind, weight=weight, at=at)


def build(events: list[Interaction[str, str]], **kwargs: object) -> Fixture[str, str]:
    params: dict[str, object] = {
        "candidates": POOL,
        "holdout_start": CUTOFF,
        "holdout_end": WINDOW_END,
        "policy": POLICY,
    }
    params.update(kwargs)
    return Fixture.from_interactions(events, **params)  # type: ignore[arg-type]


# --------------------------------------------------------------------------
# The window
# --------------------------------------------------------------------------


def test_labels_come_only_from_inside_the_holdout_window() -> None:
    fixture = build(
        [
            ev("u1", "i1", at=BEFORE),
            ev("u1", "i2", at=DURING),
            ev("u1", "i3", at=AFTER),
        ]
    )
    assert fixture.relevant["u1"] == {"i2"}


def test_window_is_half_open() -> None:
    """``[start, end)``. An event exactly at ``end`` belongs to the next run.

    Inclusive-on-both-ends is the obvious implementation and it double-counts
    boundary events across consecutive windows.
    """
    at_start = build([ev("u1", "i1", at=BEFORE), ev("u1", "i2", at=CUTOFF)])
    assert at_start.relevant["u1"] == {"i2"}

    at_end = build([ev("u1", "i1", at=BEFORE), ev("u1", "i2", at=WINDOW_END)])
    assert at_end.users == ()


def test_rejects_a_non_positive_window() -> None:
    with pytest.raises(ValueError, match="must be after"):
        build([ev("u1", "i1")], holdout_end=CUTOFF)


# --------------------------------------------------------------------------
# Relevance policy
# --------------------------------------------------------------------------


def test_only_positive_kinds_become_labels() -> None:
    """The decision that decides whether the metric measures anything.

    Views outnumber likes by orders of magnitude. If they leak into the labels
    the metric mostly measures "did we show them something they scrolled past".
    """
    fixture = build(
        [
            ev("u1", "i0", at=BEFORE),
            ev("u1", "i1", kind="view", weight=0.1),
            ev("u1", "i2", kind="like", weight=1.0),
        ]
    )
    assert fixture.relevant["u1"] == {"i2"}


def test_non_positive_weights_are_discarded() -> None:
    """``block: -5.0`` must never arrive as a label.

    NDCG is undefined for negative relevance, so a negative weight reaching the
    ground truth is a corrupted metric rather than a strong negative signal.
    """
    policy = RelevancePolicy(positive_kinds=frozenset({"like", "block"}))
    fixture = build(
        [
            ev("u1", "i0", at=BEFORE),
            ev("u1", "i1", kind="block", weight=-5.0),
            ev("u1", "i2", kind="like", weight=1.0),
        ],
        policy=policy,
    )
    assert fixture.relevant["u1"] == {"i2"}
    assert all(g > 0 for g in fixture.ground_truth["u1"].values())


def test_empty_positive_kinds_is_rejected_at_construction() -> None:
    with pytest.raises(ValueError, match="positive_kinds is empty"):
        RelevancePolicy(positive_kinds=frozenset())


def test_max_aggregation_does_not_stack_signals() -> None:
    """Liking *and* sharing one item is one strong preference, not two.

    Summing would rank it above an item engaged with more decisively but only
    once, inverting what the grading was for.
    """
    fixture = build(
        [
            ev("u1", "i0", at=BEFORE),
            ev("u1", "i1", kind="like", weight=1.0),
            ev("u1", "i1", kind="share", weight=3.0),
        ]
    )
    assert fixture.ground_truth["u1"]["i1"] == 3.0


def test_sum_aggregation_is_available_when_repetition_is_the_signal() -> None:
    policy = RelevancePolicy(
        positive_kinds=frozenset({"play"}), aggregate=Aggregate.SUM
    )
    fixture = build(
        [
            ev("u1", "i0", at=BEFORE),
            ev("u1", "i1", kind="play", weight=1.0),
            ev("u1", "i1", kind="play", weight=1.0),
        ],
        policy=policy,
    )
    assert fixture.ground_truth["u1"]["i1"] == 2.0


@pytest.mark.parametrize("aggregate", list(Aggregate))
def test_aggregation_is_order_independent(aggregate: Aggregate) -> None:
    """Reordering the stream must not change the fixture.

    This is what lets ``events`` be an unsorted cursor, and it is why there is
    no ``LAST`` aggregate.
    """
    policy = RelevancePolicy(
        positive_kinds=frozenset({"like", "share"}), aggregate=aggregate
    )
    history = ev("u1", "i0", at=BEFORE)
    a = ev("u1", "i1", kind="like", weight=1.0)
    b = ev("u1", "i1", kind="share", weight=3.0)

    forward = build([history, a, b], policy=policy)
    backward = build([history, b, a], policy=policy)
    assert forward.ground_truth == backward.ground_truth


# --------------------------------------------------------------------------
# Seen sets and cold start
# --------------------------------------------------------------------------


def test_seen_holds_pre_cutoff_items_of_every_kind() -> None:
    """A blocked item has been consumed as surely as a liked one."""
    fixture = build(
        [
            ev("u1", "i1", kind="view", at=BEFORE),
            ev("u1", "i2", kind="block", weight=-5.0, at=BEFORE),
            ev("u1", "i3", kind="like", at=DURING),
        ]
    )
    assert fixture.seen["u1"] == {"i1", "i2"}


def test_seen_never_includes_holdout_items() -> None:
    """Leaking labels into the exclusion set would hide them from the ranker."""
    fixture = build([ev("u1", "i1", at=BEFORE), ev("u1", "i2", at=DURING)])
    assert "i2" not in fixture.seen["u1"]


def test_cold_start_users_are_excluded_by_default() -> None:
    """``u2`` has labels but no history: the model has never seen them.

    Scoring them measures cold-start behaviour, which is a real thing to
    measure and a different one. Mixed in, the gate moves when signups move.
    """
    fixture = build([ev("u1", "i1", at=BEFORE), ev("u1", "i2"), ev("u2", "i3")])
    assert fixture.users == ("u1",)


def test_cold_start_users_can_be_opted_back_in() -> None:
    fixture = build([ev("u2", "i3")], require_history=False)
    assert fixture.users == ("u2",)
    assert fixture.seen["u2"] == frozenset()


def test_users_without_labels_are_never_scored() -> None:
    """Scoring a label-less user at 0.0 makes the metric partly a headcount."""
    fixture = build([ev("u1", "i1", at=BEFORE), ev("u1", "i2", kind="view")])
    assert fixture.users == ()


# --------------------------------------------------------------------------
# Determinism
# --------------------------------------------------------------------------


def _many_users(n: int) -> list[Interaction[str, str]]:
    events: list[Interaction[str, str]] = []
    for i in range(n):
        events.append(ev(f"u{i:03d}", "i0", at=BEFORE))
        events.append(ev(f"u{i:03d}", "i1"))
    return events


def test_sampling_is_reproducible_for_a_given_seed() -> None:
    """Two runs that disagree make a promotion gate that disagrees with itself."""
    events = _many_users(50)
    a = build(events, max_users=10, seed=7)
    b = build(events, max_users=10, seed=7)
    assert a.users == b.users
    assert len(a.users) == 10


def test_different_seeds_select_different_users() -> None:
    events = _many_users(50)
    a = build(events, max_users=10, seed=1)
    b = build(events, max_users=10, seed=2)
    assert a.users != b.users


def test_sampling_is_independent_of_input_order() -> None:
    events = _many_users(50)
    forward = build(events, max_users=10, seed=7)
    backward = build(list(reversed(events)), max_users=10, seed=7)
    assert forward.users == backward.users


def test_users_are_returned_in_sorted_order() -> None:
    fixture = build(
        [
            ev("u2", "i0", at=BEFORE),
            ev("u2", "i1"),
            ev("u1", "i0", at=BEFORE),
            ev("u1", "i1"),
        ]
    )
    assert fixture.users == ("u1", "u2")


def test_unsortable_user_ids_raise_a_useful_error() -> None:
    marker = object()
    # Explicitly Interaction[object, str]: the ids are deliberately of mixed,
    # mutually unorderable types, which is the condition under test.
    events: list[Interaction[object, str]] = [
        Interaction(user=marker, item="i1", kind="like", weight=1.0, at=BEFORE),
        Interaction(user="u1", item="i1", kind="like", weight=1.0, at=BEFORE),
        Interaction(user=marker, item="i2", kind="like", weight=1.0, at=DURING),
        Interaction(user="u1", item="i2", kind="like", weight=1.0, at=DURING),
    ]
    with pytest.raises(TypeError, match="reproducible"):
        Fixture.from_interactions(
            events,
            candidates=POOL,
            holdout_start=CUTOFF,
            holdout_end=WINDOW_END,
            policy=POLICY,
        )


# --------------------------------------------------------------------------
# Candidate pool and coverage
# --------------------------------------------------------------------------


def test_candidate_pool_is_truncated_from_the_head() -> None:
    fixture = build([ev("u1", "i1", at=BEFORE), ev("u1", "i1")], max_candidates=2)
    assert fixture.candidates == ("i1", "i2")


def test_label_coverage_reports_the_recall_ceiling() -> None:
    """``i9`` is labelled but not in the pool: Recall can never exceed 0.5."""
    fixture = build([ev("u1", "i0", at=BEFORE), ev("u1", "i1"), ev("u1", "i9")])
    assert fixture.label_coverage == pytest.approx(0.5)


def test_label_coverage_is_zero_when_there_are_no_labels() -> None:
    assert build([ev("u1", "i1", kind="view")]).label_coverage == 0.0


def test_empty_candidate_pool_is_rejected() -> None:
    """Silently scoring every model 0.0 would report 'no regression' forever."""
    with pytest.raises(ValueError, match="candidates is empty"):
        build([ev("u1", "i1")], candidates=[])


@pytest.mark.parametrize("kwarg", ["max_users", "max_candidates"])
def test_non_positive_limits_are_rejected(kwarg: str) -> None:
    with pytest.raises(ValueError, match="must be positive"):
        build([ev("u1", "i1")], **{kwarg: 0})


# --------------------------------------------------------------------------
# Input handling
# --------------------------------------------------------------------------


def test_events_may_be_a_generator() -> None:
    """A single pass, so a cursor or a streamed file is a first-class input."""
    events = (e for e in [ev("u1", "i1", at=BEFORE), ev("u1", "i2")])
    fixture = Fixture.from_interactions(
        events,
        candidates=POOL,
        holdout_start=CUTOFF,
        holdout_end=WINDOW_END,
        policy=POLICY,
    )
    assert fixture.relevant["u1"] == {"i2"}


def test_mixed_naive_and_aware_timestamps_raise_a_useful_error() -> None:
    """Coercing would shift every event across the cutoff by the UTC offset."""
    aware = datetime(2026, 1, 9, tzinfo=timezone.utc)
    with pytest.raises(ValueError, match="timezone-aware"):
        build([ev("u1", "i1", at=aware)])


def test_aware_timestamps_work_when_bounds_match() -> None:
    fixture = Fixture.from_interactions(
        [
            ev("u1", "i1", at=datetime(2026, 1, 7, tzinfo=timezone.utc)),
            ev("u1", "i2", at=datetime(2026, 1, 9, tzinfo=timezone.utc)),
        ],
        candidates=POOL,
        holdout_start=datetime(2026, 1, 8, tzinfo=timezone.utc),
        holdout_end=datetime(2026, 1, 15, tzinfo=timezone.utc),
        policy=POLICY,
    )
    assert fixture.relevant["u1"] == {"i2"}


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------


def test_summary_carries_what_an_audit_row_needs() -> None:
    fixture = build([ev("u1", "i0", at=BEFORE), ev("u1", "i1"), ev("u1", "i9")])
    summary = fixture.summary()
    assert summary["users"] == 1
    assert summary["candidates"] == 4
    assert summary["labels"] == 2
    assert summary["label_coverage"] == pytest.approx(0.5)


def test_len_is_the_user_count() -> None:
    assert len(build([ev("u1", "i0", at=BEFORE), ev("u1", "i1")])) == 1
