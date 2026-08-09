"""End-to-end candidate generation. Synthetic data, no downloads.

Run it::

    uv run python examples/candidates_quickstart.py

The scenario is a small feed with three sources that have nothing in common:
an embedding-similarity source scoring in [0, 1], a popularity source counting
interactions in the thousands, and a freshness source measuring hours. It then
does the thing this package exists to warn about — merges them on raw score —
and prints the result next to rank fusion so the failure is visible rather than
described.

One source is deliberately broken, to show that a fan-out degrades instead of
failing.
"""

from __future__ import annotations

import logging
import random
from collections.abc import Sequence, Set

from recsys_candidates import (
    Budget,
    Candidate,
    Result,
    by_score,
    generate,
    max_per_key,
    reciprocal_rank_fusion,
)
from recsys_candidates.exploration import EXPLORED_KEY, epsilon_greedy

SEED = 20260809
N_ITEMS = 60

rng = random.Random(SEED)
ITEMS = [f"post{n:03d}" for n in range(N_ITEMS)]
AUTHOR = {i: f"author{rng.randrange(6)}" for i in ITEMS}

# Three genuinely incomparable units.
SIMILARITY = {i: round(rng.random(), 3) for i in ITEMS}  # 0.0 - 1.0
INTERACTIONS = {i: rng.randrange(0, 5000) for i in ITEMS}  # 0 - 5000
AGE_HOURS = {i: rng.randrange(1, 400) for i in ITEMS}  # 1 - 400


class Scored:
    """A source that ranks by a metric and reports it as `score`."""

    def __init__(self, table: dict[str, float], *, ascending: bool = False) -> None:
        self.table = table
        self.ascending = ascending

    def fetch(self, user: str, k: int, exclude: Set[str]) -> Sequence[Candidate[str]]:
        pool = [i for i in ITEMS if i not in exclude]
        pool.sort(key=lambda i: self.table[i], reverse=not self.ascending)
        return [
            # rank and source are stamped by the pipeline; only score and meta
            # are ours to set.
            Candidate(
                item=i, rank=0, score=float(self.table[i]), meta={"author": AUTHOR[i]}
            )
            for i in pool[:k]
        ]


class Unavailable:
    """Stands in for an index rebuild that has not finished."""

    def fetch(self, user: str, k: int, exclude: Set[str]) -> Sequence[str]:
        raise ConnectionError("ann index unavailable")


def show(title: str, result: Result[str]) -> None:
    print(f"\n{title}")
    print(f"  slate:        {[c.item for c in result.candidates[:8]]}")
    print(f"  contributed:  {dict(result.contributed)}")
    if result.failures:
        print(f"  failures:     {dict(result.failures)}")
    if result.short:
        print(f"  SHORT:        {len(result)} of requested")


def main() -> None:
    # The library logs a warning with a traceback for each failed source. That
    # is right in production and noise in a demo, so quieten it here — the
    # failure still surfaces through `Result.failures`, which is the point.
    logging.getLogger("recsys_candidates").setLevel(logging.CRITICAL)

    sources = {
        "similar": Scored(SIMILARITY),
        "popular": Scored(INTERACTIONS),
        "fresh": Scored(AGE_HOURS, ascending=True),
    }
    budget = Budget(total=10, per_source={"popular": 20}, min_per_source={"fresh": 3})

    # --- The failure this package exists to prevent -----------------------
    by_raw_score = generate(sources, "u1", budget=budget, merge=by_score)
    show(
        "merged on raw score  (popularity's 0-5000 range swamps the rest)", by_raw_score
    )

    fused = generate(sources, "u1", budget=budget, merge=reciprocal_rank_fusion)
    show("merged on rank       (every source is represented)", fused)

    print(
        "\n  ^ same three sources, same data. Scoring in the thousands is not\n"
        "    'more relevant' than scoring in [0, 1] — it is just a bigger number."
    )

    # --- Degradation ------------------------------------------------------
    degraded = generate(
        {**sources, "ann": Unavailable()}, "u1", budget=budget, timeout=0.5
    )
    show("with a broken source (feed still serves; read `failures`)", degraded)

    # --- Diversity --------------------------------------------------------
    diverse = generate(
        sources,
        "u1",
        budget=budget,
        diversity=max_per_key(lambda c: c.meta.get("author"), 2),
    )
    authors = [c.meta["author"] for c in diverse.candidates]
    print(f"\ncapped at 2 per author\n  authors:      {authors}")

    # --- Exploration ------------------------------------------------------
    cold = [
        Candidate(item=f"cold{n}", rank=n, source="cold", meta={"author": "new"})
        for n in range(10)
    ]
    slate = epsilon_greedy(
        fused.candidates, cold, k=10, epsilon=0.2, rng=random.Random(SEED)
    )
    explored = [c.item for c in slate if c.meta.get(EXPLORED_KEY)]
    print(
        f"\nepsilon-greedy at 0.2\n  slate:        {[c.item for c in slate]}\n"
        f"  explored:     {explored}\n"
        "  ^ these carry meta['explored']; log it, or offline eval will score\n"
        "    them as though the ranker chose them."
    )


if __name__ == "__main__":
    main()
