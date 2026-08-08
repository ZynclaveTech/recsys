# recsys

Open-source infrastructure for building recommender systems, extracted from
production and cleaned up for general use.

A monorepo. Each package stands alone on PyPI and can be adopted without the
others.

| Package | Status | What it does |
|---|---|---|
| [`recsys-eval`](packages/recsys-eval) | alpha | Offline evaluation and promotion gating |
| _candidate generation_ | planned | |
| _ANN / vector search wrappers_ | planned | |
| _reference two-tower implementation_ | planned | |

## Why this exists

Most published recommender code is either a research implementation that was
never operated, or a framework that assumes you will adopt all of it. The gap
in between — the unglamorous machinery that decides whether a retrained model
is allowed to ship — is where teams lose months, and almost nobody writes about
it.

These packages are that machinery. They are not a model, and they will not make
your recommendations better. They will tell you, honestly, whether the change
you are about to ship makes them worse.

## `recsys-eval`

```bash
pip install recsys-eval
```

Zero dependencies. Pure-stdlib metrics; everything else talks to your data
through protocols rather than through pandas, numpy, or an ORM.

```python
from recsys_eval import Fixture, Interaction, RelevancePolicy, preflight
from recsys_eval.gate import GatePolicy, run_gate

fixture = Fixture.from_interactions(
    events,  # your logs, as Interaction(...)
    candidates=serving_index,  # your pool, passed explicitly
    holdout_start=cutoff,
    holdout_end=cutoff + timedelta(days=7),
    policy=RelevancePolicy(
        positive_kinds=frozenset({"like", "share", "save"}),
    ),
)

preflight(build_fixture, candidate, incumbent)  # prove the harness works

verdict = run_gate(
    candidate,
    incumbent,
    fixture,
    policy=GatePolicy(gated_metrics=["ndcg@10", "recall@50"]),
    recorder=save_audit_row,
)
if verdict.promoted:
    deploy(candidate)
```

A runnable end-to-end version is in [`examples/quickstart.py`](examples/quickstart.py) —
synthetic data, no downloads, about two seconds.

### What makes it different

**It assumes your harness is broken until proven otherwise.** `preflight()`
scores the same model twice and demands identical results, scores two different
models and demands different ones, builds the fixture twice and demands
equality. Those four checks catch most of [`docs/hazards.md`](docs/hazards.md)
and take an afternoon to wire up.

**It gates on suspicious improvements, not just regressions.** An implausibly
large gain usually means the incumbent was never scored fairly — most often it
silently loaded as a randomly-initialised model after a shape mismatch. Gating
only on regressions leaves that entire class of bug undetectable, and it is the
class that promotes bad models.

**It refuses to let you skip the decisions that matter.** `positive_kinds` has
no default, because which signals count as relevance is the single most
consequential choice in offline evaluation and the one most often made by
accident. `candidates` is required and never derived from your events, because
deriving it pins coverage at 1.0 and reports a retrieval ceiling your
production system does not have. `coverage@k` is always reported and cannot be
switched off, because forgetting it is how a ranker collapses onto
universally-popular items while every other metric applauds.

**It is honest about what the numbers mean.** These metrics compare two models
under identical conditions. They are not a measure of absolute quality — a
truncated candidate pool caps Recall below 1.0 no matter how good the model is,
and `label_coverage` reports exactly where that ceiling sits.

### Documentation

- [Hazards](docs/hazards.md) — ten ways a promotion gate silently stops
  measuring anything real, and why nearly all of them bias toward promoting the
  candidate.
- [Package README](packages/recsys-eval/README.md) — API reference.

## Development

```bash
uv sync
uv run pytest
uv run ruff check .
uv run mypy
```

## License

Apache-2.0
