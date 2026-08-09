"""Exploration: spending some of the slate on what you do not yet know.

The algorithms here are published and unremarkable — ε-greedy and Thompson
sampling. What is *not* shipped, deliberately, is any tuning of them:

- ``epsilon`` has **no default**. How much of your slate you are willing to
  spend on learning is a product decision with a real cost, and a library that
  guesses it for you has made that decision on your behalf.
- Arm priors and reward attribution live in **your** :class:`ArmStore`. What
  counts as a reward — a click, a 30-second dwell, a follow — is the part that
  actually encodes what you are optimising for, and it is not a library's
  business.

Two things this module does insist on.

**Explored items are marked.** Every candidate promoted by exploration comes
back with ``meta["explored"] = True``. Without that flag, offline evaluation
scores an explored item as though the model chose it, and you conclude the
model is worse than it is — or, if exploration happens to do well, better.
Exploration that is not logged as exploration quietly corrupts every
measurement downstream, including the promotion gate in ``recsys-eval``.

**Randomness is injected, never global.** Every function takes a
``random.Random``. Exploration seeded from global state cannot be replayed, so
a user who reports a bad slate cannot be reproduced, and an A/B arm cannot be
re-derived from logs.
"""

from __future__ import annotations

import random
import threading
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Protocol, runtime_checkable

from recsys_candidates.types import Candidate, ItemT

__all__ = [
    "ArmStore",
    "Beta",
    "InMemoryArmStore",
    "epsilon_greedy",
    "thompson_order",
]

EXPLORED_KEY = "explored"
"""The ``meta`` key set on every candidate promoted by exploration."""


@dataclass(frozen=True, slots=True)
class Beta:
    """Beta distribution parameters for one arm.

    ``alpha`` counts rewards, ``beta`` counts non-rewards, both starting from a
    prior. ``Beta(1.0, 1.0)`` is uniform — maximum uncertainty, the usual
    choice when an arm is new.

    A stronger prior such as ``Beta(10, 90)`` says "I already believe this arm
    converts around 10%, and it will take real evidence to move me". That is a
    genuine modelling choice and the reason this is not hidden behind a
    default.
    """

    alpha: float = 1.0
    beta: float = 1.0

    def __post_init__(self) -> None:
        if self.alpha <= 0 or self.beta <= 0:
            raise ValueError(
                f"alpha and beta must both be positive, got "
                f"alpha={self.alpha}, beta={self.beta}. Zero or negative is "
                "not an improper prior here, it is undefined."
            )

    @property
    def mean(self) -> float:
        """Posterior mean. For reporting — do **not** rank on it.

        Ranking on the mean is greedy: it ignores uncertainty entirely, so a
        new arm with two observations never outranks an established one and is
        never explored. Sampling is what makes Thompson work.
        """
        return self.alpha / (self.alpha + self.beta)

    def sample(self, rng: random.Random) -> float:
        return rng.betavariate(self.alpha, self.beta)

    def updated(self, reward: bool) -> Beta:
        """A new Beta with this observation folded in."""
        if reward:
            return Beta(self.alpha + 1.0, self.beta)
        return Beta(self.alpha, self.beta + 1.0)


@runtime_checkable
class ArmStore(Protocol):
    """Where arm statistics live. Yours, not the library's.

    An arm is whatever you are learning about — a source, a topic, a creator
    tier, a ranking variant. Implement this over Redis, Postgres, or whatever
    already survives a deploy.

    Persistence matters more than it looks: arm statistics held in process
    memory reset on every rollout, so the bandit restarts from its prior
    several times a week and never actually converges. That failure is
    invisible — the system keeps returning slates the whole time.
    """

    def get(self, arm: str) -> Beta:
        """Current parameters for ``arm``. Return your prior if it is unseen."""
        ...

    def record(self, arm: str, reward: bool) -> None:
        """Fold one observation into ``arm``."""
        ...


class InMemoryArmStore:
    """Reference :class:`ArmStore` for tests and local experiments.

    **Not for production.** It is per-process and per-deploy: with more than
    one replica each holds a different posterior, and every rollout resets them
    all. Both failures are silent.
    """

    def __init__(self, prior: Beta | None = None) -> None:
        self.prior = prior or Beta()
        self._arms: dict[str, Beta] = {}
        self._lock = threading.Lock()

    def get(self, arm: str) -> Beta:
        with self._lock:
            return self._arms.get(arm, self.prior)

    def record(self, arm: str, reward: bool) -> None:
        with self._lock:
            self._arms[arm] = self._arms.get(arm, self.prior).updated(reward)

    def snapshot(self) -> Mapping[str, Beta]:
        with self._lock:
            return dict(self._arms)


def _mark(candidate: Candidate[ItemT]) -> Candidate[ItemT]:
    return replace(candidate, meta={**candidate.meta, EXPLORED_KEY: True})


def epsilon_greedy(
    exploit: Sequence[Candidate[ItemT]],
    explore: Sequence[Candidate[ItemT]],
    k: int,
    *,
    epsilon: float,
    rng: random.Random,
) -> list[Candidate[ItemT]]:
    """Fill ``k`` slots, taking each from ``explore`` with probability ``epsilon``.

    The decision is made **per slot**, not per request. With ``epsilon=0.1`` and
    ``k=50`` that is roughly five explored items in every slate, not one slate
    in ten made entirely of exploration. Both are defensible designs and they
    behave completely differently; this is the former, and it is worth being
    sure it is the one you want.

    Slots are filled to ``k`` wherever possible: if one pool runs out, the
    other continues. Exploration that shortens the slate would trade a
    measurable ranking loss for an invisible coverage loss.

    Args:
        exploit: Your ranked slate, best first — what you would have served.
        explore: The pool to draw exploration from, best first. Usually
            candidates the ranker demoted, or ones with little history.
        k: Slots to fill.
        epsilon: Probability per slot of drawing from ``explore``. **Required**
            — see the module docstring.
        rng: Seeded ``random.Random``. Injected so a slate can be replayed.

    Returns:
        Up to ``k`` candidates, deduplicated. Explored ones carry
        ``meta["explored"] = True``.
    """
    if not 0.0 <= epsilon <= 1.0:
        raise ValueError(
            f"epsilon must be in [0, 1], got {epsilon}. It is a probability "
            "per slot, not a count of slots."
        )
    if k < 0:
        raise ValueError(f"k must be non-negative, got {k}.")

    exploit_q = list(exploit)
    explore_q = list(explore)
    taken: set[ItemT] = set()
    out: list[Candidate[ItemT]] = []

    def pop(queue: list[Candidate[ItemT]]) -> Candidate[ItemT] | None:
        while queue:
            candidate = queue.pop(0)
            if candidate.item not in taken:
                return candidate
        return None

    while len(out) < k and (exploit_q or explore_q):
        # Draw first, then fall back — so the RNG is consumed once per slot
        # regardless of which pool is empty, and a seeded run stays replayable
        # even as the pools drain at different rates.
        want_explore = rng.random() < epsilon
        first, second = (
            (explore_q, exploit_q) if want_explore else (exploit_q, explore_q)
        )
        chosen = pop(first)
        marked = want_explore
        if chosen is None:
            chosen = pop(second)
            marked = not want_explore
        if chosen is None:
            break
        taken.add(chosen.item)
        out.append(_mark(chosen) if marked else chosen)

    return out


def thompson_order(
    candidates: Sequence[Candidate[ItemT]],
    *,
    store: ArmStore,
    arm_of: Callable[[Candidate[ItemT]], str],
    rng: random.Random,
    mark: bool = True,
) -> list[Candidate[ItemT]]:
    """Order candidates by one posterior sample per **arm**.

    Each distinct arm is sampled once, and every candidate belonging to it
    inherits that sample. Order within an arm is preserved.

    Sampling once per *candidate* instead is a natural-looking mistake with a
    real consequence: the maximum of many draws is biased upward, so an arm
    contributing thirty candidates beats an equally good arm contributing three
    on order statistics alone. The bandit then looks like it has learned a
    preference when all it has learned is which arm is chatty.

    Args:
        candidates: Merged candidates to order.
        store: Where posteriors come from. See :class:`ArmStore`.
        arm_of: Maps a candidate to its arm name — its source, its topic,
            whatever you are actually learning about.
        rng: Seeded ``random.Random``.
        mark: Set ``meta["explored"]``. Leave it on unless you have a separate
            way to record that a slate was bandit-ordered; without it, offline
            evaluation cannot tell these apart from model-chosen items.

    Returns:
        Candidates ordered by their arm's sampled value, descending.
    """
    if not candidates:
        return []

    samples: dict[str, float] = {}
    for candidate in candidates:
        arm = arm_of(candidate)
        if arm not in samples:
            samples[arm] = store.get(arm).sample(rng)

    ordered = sorted(
        enumerate(candidates),
        key=lambda pair: (-samples[arm_of(pair[1])], pair[0]),
    )
    return [_mark(c) if mark else c for _, c in ordered]
