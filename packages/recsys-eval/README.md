# recsys-eval

Offline evaluation and promotion gating for recommender systems.

> **Status: alpha.** Feature-complete for v0.2 — metrics, fixture, ranker
> protocol, scoring, preflight checks, and a promotion gate with an optional
> paired-bootstrap decision. The API may still
> move before 1.0.

Built around one claim: the useful question is usually not *"how good is this
model"* but *"is this candidate worse than the one already in production"*.
Those need different machinery, and conflating them is how teams end up with a
dashboard full of numbers nobody can act on.

## The whole loop

```python
from recsys_eval import Fixture, Interaction, RelevancePolicy, preflight
from recsys_eval.gate import GatePolicy, run_gate

fixture = Fixture.from_interactions(
    events,  # your logs, as Interaction(...)
    candidates=serving_index,  # your pool, passed explicitly
    holdout_start=cutoff,
    holdout_end=cutoff + timedelta(days=7),
    policy=RelevancePolicy(positive_kinds=frozenset({"like", "share", "save"})),
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

### When the metric is sparse, decide on an interval

On a real feed only a few dozen of several thousand users have any hit in the
top ten, so NDCG@10 moves by a quarter of its own value from one user sample
to the next. A fixed 2% tolerance on two point estimates then rejects an
identical model more often than not. Set `bootstrap_samples` and the gate
compares the models user by user instead:

```python
policy = GatePolicy(
    gated_metrics=["ndcg@10", "recall@50"],
    bootstrap_samples=2000,  # paired bootstrap over per-user values
    confidence=0.95,
    tolerance=0.02,  # reject only if the WHOLE interval is below -2%
)
verdict = run_gate(candidate, incumbent, fixture, policy=policy)
verdict.comparisons["ndcg@10"].describe()
# 'ndcg@10 0.0038 -> 0.0016 (-57.0%, 95% CI -70.2% to -49.7%)'
```

Pairing is what makes this affordable: both models are scored on the same
users, so user-level noise shared by both cancels out. Score every eligible
user rather than a sample -- the interval narrows with users, and sampling
throws the width away. `paired_bootstrap` is exported for comparing any two
per-user series, e.g. training recipes across seeds.

Implementing a ranker is two methods:

```python
class MyRanker:
    def prepare(self, candidates): ...  # build an index, once
    def rank(self, user, k, exclude): ...  # return bare ids, best first
```

A runnable end-to-end version lives in `examples/quickstart.py`.

## Install

```bash
pip install recsys-eval
```

No dependencies. The metrics are pure stdlib; the fixture and gate layers talk
to your data through protocols rather than through pandas, numpy, or an ORM.

## Metrics

```python
from recsys_eval import ndcg_at_k, recall_at_k, catalog_coverage

# Graded relevance: say that a share is worth more than a like, and NDCG
# will reward ranking shares higher.
ndcg_at_k(["post_a", "post_b"], {"post_a": 3.0, "post_b": 1.0}, k=10)

# Relevant items your candidate index never even offered stay in the
# denominator, so a shrinking index shows up as a falling score.
recall_at_k(["post_a"], {"post_a", "post_c"}, k=10)  # 0.5, not 1.0

# Read this beside NDCG. A ranker can raise NDCG by collapsing onto
# universally-popular items, and every per-user metric is blind to that.
catalog_coverage(rankings, catalog_size=50_000, k=10)
```

| Function | Relevance | Sensitive to rank |
|---|---|---|
| `ndcg_at_k` | graded | yes |
| `recall_at_k` | binary | no |
| `precision_at_k` | binary | no |
| `average_precision_at_k` | binary | yes |
| `reciprocal_rank_at_k` | binary | yes |
| `hit_rate_at_k` | binary | no |
| `catalog_coverage` | — | no |

All are **per-user** quantities. MAP@k, MRR@k and NDCG@k are their
macro-averages — naming them honestly keeps the aggregation choice visible at
the call site, where it belongs.

Every function returns `0.0` — never `NaN`, never raising — on degenerate
input, so a user with no usable labels cannot poison a macro-average.

## What these numbers are not

They are not a measure of absolute recommendation quality. In any real system
the candidate pool is truncated, which structurally caps Recall below 1.0 no
matter how good the model is.

**Do not quote them as system quality. Do compare them run over run, candidate
against incumbent.**

## License

Apache-2.0
