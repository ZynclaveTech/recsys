"""Tests for :mod:`recsys_candidates.exploration`.

Statistical code is easy to write and hard to test, so these lean on
properties that must hold exactly — replayability from a seed, slate length,
the explored flag — plus one distributional check with enough trials that its
tolerance is not doing the work.
"""

from __future__ import annotations

import random

import pytest

from recsys_candidates import Candidate
from recsys_candidates.exploration import (
    EXPLORED_KEY,
    ArmStore,
    Beta,
    InMemoryArmStore,
    epsilon_greedy,
    thompson_order,
)


def cands(prefix: str, n: int, arm: str | None = None) -> list[Candidate[str]]:
    return [
        Candidate(
            item=f"{prefix}{i}",
            rank=i,
            source=arm or prefix,
            meta={"arm": arm or prefix},
        )
        for i in range(n)
    ]


def items(cs: list[Candidate[str]]) -> list[str]:
    return [c.item for c in cs]


# --------------------------------------------------------------------------
# Beta
# --------------------------------------------------------------------------


def test_updated_counts_rewards_and_misses() -> None:
    b = Beta()
    assert b.updated(True) == Beta(2.0, 1.0)
    assert b.updated(False) == Beta(1.0, 2.0)


def test_mean_reflects_evidence() -> None:
    assert Beta(1.0, 1.0).mean == pytest.approx(0.5)
    assert Beta(9.0, 1.0).mean == pytest.approx(0.9)


@pytest.mark.parametrize(("a", "b"), [(0.0, 1.0), (1.0, 0.0), (-1.0, 1.0)])
def test_non_positive_parameters_are_rejected(a: float, b: float) -> None:
    with pytest.raises(ValueError, match="must both be positive"):
        Beta(a, b)


def test_sample_is_replayable_from_a_seed() -> None:
    b = Beta(3.0, 7.0)
    assert b.sample(random.Random(1)) == b.sample(random.Random(1))


# --------------------------------------------------------------------------
# ArmStore
# --------------------------------------------------------------------------


def test_in_memory_store_satisfies_the_protocol() -> None:
    assert isinstance(InMemoryArmStore(), ArmStore)


def test_unseen_arm_returns_the_prior() -> None:
    store = InMemoryArmStore(prior=Beta(10.0, 90.0))
    assert store.get("never-seen") == Beta(10.0, 90.0)


def test_record_accumulates() -> None:
    store = InMemoryArmStore()
    store.record("a", True)
    store.record("a", True)
    store.record("a", False)
    assert store.get("a") == Beta(3.0, 2.0)


def test_arms_are_independent() -> None:
    store = InMemoryArmStore()
    store.record("a", True)
    assert store.get("b") == Beta()


# --------------------------------------------------------------------------
# epsilon-greedy
# --------------------------------------------------------------------------


def test_epsilon_zero_is_pure_exploitation() -> None:
    out = epsilon_greedy(
        cands("x", 3), cands("y", 3), 3, epsilon=0.0, rng=random.Random(0)
    )
    assert items(out) == ["x0", "x1", "x2"]
    assert not any(c.meta.get(EXPLORED_KEY) for c in out)


def test_epsilon_one_is_pure_exploration() -> None:
    out = epsilon_greedy(
        cands("x", 3), cands("y", 3), 3, epsilon=1.0, rng=random.Random(0)
    )
    assert items(out) == ["y0", "y1", "y2"]
    assert all(c.meta.get(EXPLORED_KEY) for c in out)


def test_explored_items_are_marked_and_exploited_ones_are_not() -> None:
    """Without the flag, offline eval scores an explored item as though the
    model chose it."""
    out = epsilon_greedy(
        cands("x", 10), cands("y", 10), 10, epsilon=0.5, rng=random.Random(7)
    )
    for c in out:
        assert c.meta.get(EXPLORED_KEY, False) is (c.item.startswith("y"))


def test_original_meta_survives_marking() -> None:
    out = epsilon_greedy(
        [], cands("y", 1, arm="topic-a"), 1, epsilon=1.0, rng=random.Random(0)
    )
    assert out[0].meta["arm"] == "topic-a"
    assert out[0].meta[EXPLORED_KEY] is True


def test_is_replayable_from_a_seed() -> None:
    """Exploration seeded from global state cannot be reproduced from logs."""
    a = epsilon_greedy(
        cands("x", 20), cands("y", 20), 10, epsilon=0.3, rng=random.Random(42)
    )
    b = epsilon_greedy(
        cands("x", 20), cands("y", 20), 10, epsilon=0.3, rng=random.Random(42)
    )
    assert items(a) == items(b)


def test_different_seeds_diverge() -> None:
    a = epsilon_greedy(
        cands("x", 20), cands("y", 20), 10, epsilon=0.5, rng=random.Random(1)
    )
    b = epsilon_greedy(
        cands("x", 20), cands("y", 20), 10, epsilon=0.5, rng=random.Random(2)
    )
    assert items(a) != items(b)


def test_slate_stays_full_when_the_explore_pool_runs_out() -> None:
    """Exploration must not trade a ranking loss for an invisible slate loss."""
    out = epsilon_greedy(
        cands("x", 10), cands("y", 1), 10, epsilon=0.9, rng=random.Random(3)
    )
    assert len(out) == 10


def test_slate_stays_full_when_the_exploit_pool_runs_out() -> None:
    out = epsilon_greedy(
        cands("x", 1), cands("y", 10), 10, epsilon=0.1, rng=random.Random(3)
    )
    assert len(out) == 10


def test_never_duplicates_across_pools() -> None:
    shared = cands("s", 5)
    out = epsilon_greedy(shared, shared, 5, epsilon=0.5, rng=random.Random(9))
    assert len(items(out)) == len(set(items(out)))


def test_epsilon_is_honoured_in_aggregate() -> None:
    """One distributional check, with enough slots that 0.05 tolerance is not
    doing the work: 4000 slots at eps=0.25."""
    rng = random.Random(11)
    explored = 0
    trials = 200
    for _ in range(trials):
        out = epsilon_greedy(cands("x", 40), cands("y", 40), 20, epsilon=0.25, rng=rng)
        explored += sum(1 for c in out if c.meta.get(EXPLORED_KEY))
    assert explored / (trials * 20) == pytest.approx(0.25, abs=0.05)


@pytest.mark.parametrize("epsilon", [-0.1, 1.1])
def test_epsilon_outside_zero_one_is_rejected(epsilon: float) -> None:
    """It is a probability per slot, not a count of slots."""
    with pytest.raises(ValueError, match=r"epsilon must be in \[0, 1\]"):
        epsilon_greedy([], [], 5, epsilon=epsilon, rng=random.Random(0))


def test_empty_pools_return_empty() -> None:
    assert epsilon_greedy([], [], 5, epsilon=0.5, rng=random.Random(0)) == []


# --------------------------------------------------------------------------
# Thompson sampling
# --------------------------------------------------------------------------


def test_samples_once_per_arm_not_per_candidate() -> None:
    """The bug this guards: the max of many draws is biased upward, so a
    chatty arm would beat an equally good quiet one on order statistics.

    Both arms share an identical posterior. Over many trials the arm with 30
    candidates must win the top slot about half the time, not almost always.
    """
    store = InMemoryArmStore()
    pool = cands("big", 30, arm="big") + cands("small", 3, arm="small")
    rng = random.Random(5)
    big_wins = 0
    trials = 400
    for _ in range(trials):
        top = thompson_order(
            pool, store=store, arm_of=lambda c: c.meta["arm"], rng=rng
        )[0]
        big_wins += top.meta["arm"] == "big"
    assert big_wins / trials == pytest.approx(0.5, abs=0.08)


def test_a_better_arm_wins_more_often() -> None:
    store = InMemoryArmStore()
    for _ in range(50):
        store.record("good", True)
        store.record("bad", False)
    pool = cands("g", 3, arm="good") + cands("b", 3, arm="bad")
    rng = random.Random(3)
    wins = sum(
        thompson_order(pool, store=store, arm_of=lambda c: c.meta["arm"], rng=rng)[
            0
        ].meta["arm"]
        == "good"
        for _ in range(100)
    )
    assert wins > 90


def test_order_within_an_arm_is_preserved() -> None:
    """The arm decides where the block goes; the merge decided the order
    inside it, and re-shuffling would discard that."""
    store = InMemoryArmStore()
    pool = cands("a", 4, arm="one")
    out = thompson_order(
        pool, store=store, arm_of=lambda c: c.meta["arm"], rng=random.Random(0)
    )
    assert items(out) == ["a0", "a1", "a2", "a3"]


def test_is_replayable_from_a_seed_too() -> None:
    store = InMemoryArmStore()
    pool = cands("a", 5, arm="one") + cands("b", 5, arm="two")
    a = thompson_order(
        pool, store=store, arm_of=lambda c: c.meta["arm"], rng=random.Random(8)
    )
    b = thompson_order(
        pool, store=store, arm_of=lambda c: c.meta["arm"], rng=random.Random(8)
    )
    assert items(a) == items(b)


def test_marks_by_default_and_can_be_turned_off() -> None:
    store = InMemoryArmStore()
    pool = cands("a", 2, arm="one")
    marked = thompson_order(
        pool, store=store, arm_of=lambda c: c.meta["arm"], rng=random.Random(0)
    )
    unmarked = thompson_order(
        pool,
        store=store,
        arm_of=lambda c: c.meta["arm"],
        rng=random.Random(0),
        mark=False,
    )
    assert all(c.meta[EXPLORED_KEY] for c in marked)
    assert not any(EXPLORED_KEY in c.meta for c in unmarked)


def test_empty_input_returns_empty() -> None:
    assert (
        thompson_order(
            [],
            store=InMemoryArmStore(),
            arm_of=lambda c: "x",
            rng=random.Random(0),
        )
        == []
    )


def test_every_candidate_survives_ordering() -> None:
    """Ordering must not drop anything — that would be a silent content loss."""
    store = InMemoryArmStore()
    pool = cands("a", 3, arm="one") + cands("b", 4, arm="two")
    out = thompson_order(
        pool, store=store, arm_of=lambda c: c.meta["arm"], rng=random.Random(0)
    )
    assert sorted(items(out)) == sorted(items(pool))
