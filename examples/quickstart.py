"""End-to-end example: fixture, preflight, gate. Synthetic data, no downloads.

Run it::

    uv run python examples/quickstart.py

Deliberately self-contained. An example that needs a dataset download is an
example most people never actually run, and this one is meant to be executed
and then edited.

The scenario: users have a hidden topic preference, items have a topic. The
candidate ranker knows the topic; the incumbent only knows global popularity.
The candidate should win -- and the interesting part is watching the gate be
suspicious about *how much* it wins, which is exactly what it should do when
the two models are this far apart.
"""

from __future__ import annotations

import random
from collections.abc import Sequence, Set
from datetime import datetime, timedelta

from recsys_eval import (
    Fixture,
    Interaction,
    RelevancePolicy,
    preflight,
    score,
)
from recsys_eval.gate import Decision, GatePolicy, run_gate

SEED = 20260809
N_USERS = 400
N_ITEMS = 600
N_TOPICS = 8

EPOCH = datetime(2026, 1, 1)
HOLDOUT_START = EPOCH + timedelta(days=30)
HOLDOUT_END = HOLDOUT_START + timedelta(days=7)

# Weights are the strength of each signal, not a count. A share says more than
# a like; a view says almost nothing, which is why it is not in positive_kinds.
WEIGHTS = {"view": 0.1, "like": 1.0, "save": 2.5, "share": 3.0}
POLICY = RelevancePolicy(positive_kinds=frozenset({"like", "save", "share"}))


# ---------------------------------------------------------------------------
# Synthetic world
# ---------------------------------------------------------------------------

rng = random.Random(SEED)
item_topic = {f"i{n:04d}": rng.randrange(N_TOPICS) for n in range(N_ITEMS)}
user_topic = {f"u{n:04d}": rng.randrange(N_TOPICS) for n in range(N_USERS)}

# Popularity is topic-blind: a handful of items are shown to everyone. This is
# what the incumbent will learn, and why it is beatable.
popularity = {item: rng.random() ** 3 for item in item_topic}


def generate_events() -> list[Interaction[str, str]]:
    """Interactions before and during the holdout window.

    Uses its **own** freshly-seeded RNG rather than the module-level one. That
    is not fussiness: with a shared generator, calling this twice advances the
    stream and produces different events, so the fixture is different every
    time it is built. ``assert_reproducible`` caught exactly that while this
    example was being written -- which is the argument for the check.
    """
    rng = random.Random(SEED + 1)
    events: list[Interaction[str, str]] = []
    items = list(item_topic)

    for user, topic in user_topic.items():
        for day in range(37):
            at = EPOCH + timedelta(days=day, hours=rng.randrange(24))
            for item in rng.sample(items, 12):
                # Users engage mostly within their topic, sometimes outside it.
                on_topic = item_topic[item] == topic
                p_engage = 0.30 if on_topic else 0.02
                if rng.random() < p_engage:
                    kind = rng.choices(["like", "save", "share"], weights=[6, 3, 1])[0]
                else:
                    kind = "view"
                events.append(
                    Interaction(
                        user=user,
                        item=item,
                        kind=kind,
                        weight=WEIGHTS[kind],
                        at=at,
                    )
                )
    return events


# ---------------------------------------------------------------------------
# Two rankers
# ---------------------------------------------------------------------------


class TopicRanker:
    """Knows each user's topic. Stands in for a trained model."""

    def __init__(self) -> None:
        self.by_topic: dict[int, list[str]] = {}

    def prepare(self, candidates: Sequence[str]) -> None:
        self.by_topic = {}
        for item in candidates:
            self.by_topic.setdefault(item_topic[item], []).append(item)
        for items in self.by_topic.values():
            # Stable ordering: without a tiebreaker, ties resolve arbitrarily
            # and two runs disagree. assert_deterministic would catch it.
            items.sort(key=lambda i: (-popularity[i], i))

    def rank(self, user: str, k: int, exclude: Set[str]) -> Sequence[str]:
        preferred = self.by_topic.get(user_topic[user], [])
        return [i for i in preferred if i not in exclude][:k]


class PopularityRanker:
    """Topic-blind. The incumbent, and a baseline worth keeping around."""

    def __init__(self) -> None:
        self.ordered: list[str] = []

    def prepare(self, candidates: Sequence[str]) -> None:
        self.ordered = sorted(candidates, key=lambda i: (-popularity[i], i))

    def rank(self, user: str, k: int, exclude: Set[str]) -> Sequence[str]:
        return [i for i in self.ordered if i not in exclude][:k]


# ---------------------------------------------------------------------------
# Wiring it together
# ---------------------------------------------------------------------------


def build_fixture() -> Fixture[str, str]:
    """A zero-argument builder, so ``assert_reproducible`` can call it twice."""
    return Fixture.from_interactions(
        generate_events(),
        # The pool is a property of your serving system. Passing it explicitly
        # is what keeps label_coverage honest -- derive it from the events and
        # it pins at 1.0 and stops meaning anything.
        candidates=sorted(item_topic),
        holdout_start=HOLDOUT_START,
        holdout_end=HOLDOUT_END,
        policy=POLICY,
        max_users=200,
        # A serving index that does not cover the whole catalogue, as most do
        # not. Watch label_coverage in the output: it is the ceiling on Recall,
        # and Recall cannot exceed it however good the model is.
        max_candidates=500,
        seed=SEED,
    )


def main() -> None:
    fixture = build_fixture()
    print("fixture:", fixture.summary())
    print()

    candidate, incumbent = TopicRanker(), PopularityRanker()

    # Prove the harness works before trusting anything it says. In CI this
    # would run against a fixed snapshot on every change.
    preflight(build_fixture, candidate, incumbent, ks=[10, 50])
    print("preflight: passed\n")

    for name, ranker in [("topic", candidate), ("popularity", incumbent)]:
        result = score(ranker, fixture, ks=[10, 50])
        row = ", ".join(f"{k}={v:.4f}" for k, v in sorted(result.values.items()))
        print(f"{name:>10}: {row}")
    print()

    verdict = run_gate(
        candidate,
        incumbent,
        fixture,
        # Gate on one rank-sensitive metric and one set metric: a ranker can
        # trade them against each other, so either alone can be walked past.
        policy=GatePolicy(gated_metrics=["ndcg@10", "recall@50"]),
        ks=[10, 50],
        context="quickstart",
        recorder=lambda r: print("recorded:", r.decision.value),
    )

    print(f"\ndecision: {verdict.decision.value}")
    print(f"reason:   {verdict.reason}")

    if verdict.decision is Decision.SUSPICIOUS:
        print(
            "\nThe gate is right to balk. An improvement this large is "
            "normally a broken incumbent rather than a brilliant candidate, "
            "and a gate that waves it through cannot tell the two apart. "
            "Here it really is a better model, so this is the false positive "
            "you accept in exchange for catching the real thing."
        )


if __name__ == "__main__":
    main()
