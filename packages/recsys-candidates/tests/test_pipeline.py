"""Tests for :mod:`recsys_candidates.pipeline`.

Weighted toward degradation rather than happy paths: a fan-out's failure modes
are all quiet, and quiet is what makes them expensive.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Sequence, Set

import pytest

from recsys_candidates import Budget, Candidate, interleave
from recsys_candidates.pipeline import Result, generate, max_per_key
from recsys_candidates.protocols import Source


class Listed:
    """Returns a fixed list, honouring exclude and k."""

    def __init__(self, *items: str, delay: float = 0.0) -> None:
        self.items = list(items)
        self.delay = delay
        self.calls: list[tuple[str, int]] = []

    def fetch(self, user: str, k: int, exclude: Set[str]) -> Sequence[str]:
        self.calls.append((user, k))
        if self.delay:
            time.sleep(self.delay)
        return [i for i in self.items if i not in exclude][:k]


class Broken:
    def __init__(self, exc: Exception | None = None) -> None:
        self.exc = exc or RuntimeError("index unavailable")

    def fetch(self, user: str, k: int, exclude: Set[str]) -> Sequence[str]:
        raise self.exc


def ids(result: Result[str]) -> list[str]:
    return [c.item for c in result.candidates]


# --------------------------------------------------------------------------
# Happy path and the source contract
# --------------------------------------------------------------------------


def test_listed_satisfies_the_protocol() -> None:
    assert isinstance(Listed("a"), Source)


def test_combines_sources_and_truncates_to_budget() -> None:
    result = generate(
        {"x": Listed("a", "b", "c"), "y": Listed("d", "e")},
        "u1",
        budget=Budget(total=3),
    )
    assert len(result) == 3
    assert not result.short
    assert result.failures == {}


def test_rank_and_source_are_stamped_by_the_pipeline() -> None:
    """A source cannot mislabel itself or get its own ranks wrong."""
    result = generate({"x": Listed("a", "b")}, "u1", budget=Budget(total=2))
    assert [(c.source, c.rank) for c in result.candidates] == [("x", 0), ("x", 1)]


def test_source_supplied_rank_and_source_are_overwritten() -> None:
    class Liar:
        def fetch(
            self, user: str, k: int, exclude: Set[str]
        ) -> Sequence[Candidate[str]]:
            return [
                Candidate(item="a", rank=99, source="somewhere-else", score=0.5),
                Candidate(item="b", rank=99, source="somewhere-else"),
            ]

    result = generate({"real": Liar()}, "u1", budget=Budget(total=2))
    assert [(c.source, c.rank) for c in result.candidates] == [
        ("real", 0),
        ("real", 1),
    ]
    # score and meta survive; only the fields the pipeline owns are replaced.
    assert result.candidates[0].score == 0.5


def test_exclude_reaches_every_source_and_is_immutable() -> None:
    """The same set goes to all sources, so a mutating source would shrink the
    others' pools."""
    captured: list[Set[str]] = []

    class Grabby:
        def fetch(self, user: str, k: int, exclude: Set[str]) -> Sequence[str]:
            captured.append(exclude)
            return ["a"]

    generate(
        {"x": Grabby(), "y": Grabby()}, "u1", budget=Budget(total=5), exclude={"z"}
    )
    assert len(captured) == 2
    for e in captured:
        assert e == {"z"}
        assert isinstance(e, frozenset)
        with pytest.raises(AttributeError):
            e.add("boom")  # type: ignore[attr-defined]


def test_overfetch_is_applied_to_the_ask() -> None:
    src = Listed("a")
    generate({"x": src}, "u1", budget=Budget(total=10, overfetch=3))
    assert src.calls[0][1] == 30


# --------------------------------------------------------------------------
# Degradation — the reason this module exists
# --------------------------------------------------------------------------


def test_a_broken_source_does_not_blank_the_feed() -> None:
    # A heterogeneous source mapping widens to dict[str, object], so the
    # item type has to be stated rather than inferred.
    result: Result[str] = generate(
        {"good": Listed("a", "b"), "bad": Broken()}, "u1", budget=Budget(total=2)
    )
    assert ids(result) == ["a", "b"]
    assert "bad" in result.failures
    assert "index unavailable" in result.failures["bad"]


def test_every_source_failing_still_returns_a_result() -> None:
    """Callers should branch on `failures`, not on an exception."""
    result: Result[str] = generate(
        {"a": Broken(), "b": Broken()}, "u1", budget=Budget(total=5)
    )
    assert result.candidates == []
    assert set(result.failures) == {"a", "b"}
    assert result.short


def test_short_slate_is_flagged() -> None:
    """A short slate renders exactly like a full one."""
    result = generate({"x": Listed("a")}, "u1", budget=Budget(total=10))
    assert result.short
    assert len(result) == 1


def test_a_slow_source_is_bounded_by_timeout() -> None:
    result = generate(
        {"fast": Listed("a"), "slow": Listed("b", delay=2.0)},
        "u1",
        budget=Budget(total=5),
        timeout=0.2,
    )
    assert "slow" in result.failures
    assert ids(result) == ["a"]


def test_no_sources_is_a_configuration_error() -> None:
    """An empty slate here is not a cold-start result."""
    with pytest.raises(ValueError, match="No sources given"):
        generate({}, "u1", budget=Budget(total=5))


# --------------------------------------------------------------------------
# Determinism
# --------------------------------------------------------------------------


def test_order_does_not_depend_on_completion_time() -> None:
    """Sources run concurrently; the merge input order must still be the
    declaration order, or tie-breaks shift whenever a source is slow."""
    slow_first = generate(
        {"x": Listed("a", delay=0.15), "y": Listed("b")},
        "u1",
        budget=Budget(total=2),
        merge=interleave,
    )
    fast_first = generate(
        {"x": Listed("a"), "y": Listed("b", delay=0.15)},
        "u1",
        budget=Budget(total=2),
        merge=interleave,
    )
    assert ids(slow_first) == ids(fast_first) == ["a", "b"]


def test_sources_actually_run_concurrently() -> None:
    """Serial fan-out would make a six-source slate as slow as the sum."""
    active = 0
    peak = 0
    lock = threading.Lock()

    class Tracked:
        def fetch(self, user: str, k: int, exclude: Set[str]) -> Sequence[str]:
            nonlocal active, peak
            with lock:
                active += 1
                peak = max(peak, active)
            time.sleep(0.1)
            with lock:
                active -= 1
            return ["a"]

    generate({f"s{n}": Tracked() for n in range(4)}, "u1", budget=Budget(total=4))
    assert peak > 1


# --------------------------------------------------------------------------
# Filters and diversity
# --------------------------------------------------------------------------


def test_filters_apply_after_merge() -> None:
    result = generate(
        {"x": Listed("a", "bad", "b")},
        "u1",
        budget=Budget(total=5),
        filters=[lambda c: c.item != "bad"],
    )
    assert ids(result) == ["a", "b"]


def test_all_filters_must_pass() -> None:
    result = generate(
        {"x": Listed("a", "b", "c")},
        "u1",
        budget=Budget(total=5),
        filters=[lambda c: c.item != "a", lambda c: c.item != "b"],
    )
    assert ids(result) == ["c"]


def test_max_per_key_caps_an_author() -> None:
    class WithAuthors:
        def fetch(
            self, user: str, k: int, exclude: Set[str]
        ) -> Sequence[Candidate[str]]:
            return [
                Candidate(item="1", rank=0, meta={"author": "amy"}),
                Candidate(item="2", rank=1, meta={"author": "amy"}),
                Candidate(item="3", rank=2, meta={"author": "amy"}),
                Candidate(item="4", rank=3, meta={"author": "bob"}),
            ]

    result = generate(
        {"x": WithAuthors()},
        "u1",
        budget=Budget(total=10),
        diversity=max_per_key(lambda c: c.meta.get("author"), 2),
    )
    assert ids(result) == ["1", "2", "4"]


def test_max_per_key_preserves_order() -> None:
    """Reordering to satisfy diversity would silently undo the merge."""

    class WithAuthors:
        def fetch(
            self, user: str, k: int, exclude: Set[str]
        ) -> Sequence[Candidate[str]]:
            return [
                Candidate(item="1", rank=0, meta={"author": "amy"}),
                Candidate(item="2", rank=1, meta={"author": "bob"}),
                Candidate(item="3", rank=2, meta={"author": "amy"}),
            ]

    result = generate(
        {"x": WithAuthors()},
        "u1",
        budget=Budget(total=10),
        diversity=max_per_key(lambda c: c.meta.get("author"), 1),
    )
    assert ids(result) == ["1", "2"]


def test_missing_diversity_key_is_exempt_not_pooled() -> None:
    """Pooling them would make "unknown" the most over-represented author."""

    class Mixed:
        def fetch(
            self, user: str, k: int, exclude: Set[str]
        ) -> Sequence[Candidate[str]]:
            return [
                Candidate(item="1", rank=0),
                Candidate(item="2", rank=1),
                Candidate(item="3", rank=2),
            ]

    result = generate(
        {"x": Mixed()},
        "u1",
        budget=Budget(total=10),
        diversity=max_per_key(lambda c: c.meta.get("author"), 1),
    )
    assert ids(result) == ["1", "2", "3"]


def test_max_per_key_rejects_a_zero_limit() -> None:
    with pytest.raises(ValueError, match="at least 1"):
        max_per_key(lambda c: c.item, 0)


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------


def test_fetched_and_contributed_are_reported_separately() -> None:
    """A source with healthy `fetched` and zero `contributed` is being
    out-competed or filtered away — a different problem from being broken."""
    result = generate(
        {"x": Listed("a", "b"), "y": Listed("c")},
        "u1",
        budget=Budget(total=5),
        filters=[lambda c: c.source != "y"],
    )
    assert result.fetched == {"x": 2, "y": 1}
    assert result.contributed == {"x": 2}
    assert "y" not in result.contributed


def test_summary_carries_what_a_rollout_needs() -> None:
    result: Result[str] = generate(
        {"x": Listed("a"), "bad": Broken()}, "u1", budget=Budget(total=5)
    )
    s = result.summary()
    assert s["returned"] == 1
    assert s["short"] is True
    assert "bad" in s["failures"]


def test_result_is_iterable_and_sized() -> None:
    result = generate({"x": Listed("a", "b")}, "u1", budget=Budget(total=5))
    assert len(result) == 2
    assert [c.item for c in result] == ["a", "b"]
