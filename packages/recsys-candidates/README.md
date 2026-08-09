# recsys-candidates

Composable candidate generation for recommender systems.

> **Status: alpha.** Feature-complete for v0.1 — vocabulary, source protocol,
> merge strategies, pipeline, filters, diversity, exploration. The API may move
> before 1.0.

Zero dependencies. Sources reach your data through a protocol, so the library
never needs a database client, an ANN library, or a model.

## The position it takes

Three sources return the same item:

```
ann         cosine similarity   0.82
popularity  interaction count   4300
recency     hours since posted  7
```

No weighting of those numbers means anything. Whichever has the widest range
dominates the sum, and per-source min-max normalisation only hides it — it
makes the top item of a *bad* source look exactly as good as the top item of a
good one.

So the default merge compares **ranks**. A rank of 1 means the same thing
coming out of every source. Run `examples/candidates_quickstart.py` and watch
raw-score merging hand all ten slots to popularity while rank fusion returns
`{'fresh': 4, 'popular': 2, 'similar': 4}` — same sources, same data.

## The whole loop

```python
from recsys_candidates import Budget, generate, max_per_key

result = generate(
    {"ann": ann_source, "popular": popularity, "fresh": recency},
    user_id,
    budget=Budget(total=50, per_source={"popular": 10}, min_per_source={"fresh": 5}),
    exclude=already_seen,
    filters=[lambda c: c.item not in blocked],
    diversity=max_per_key(lambda c: c.meta.get("author"), 2),
    timeout=0.2,
)

if result.failures:
    metrics.incr("retrieval.degraded", tags=result.failures)
if result.short:
    metrics.incr("retrieval.short")
```

A source is one method:

```python
class RecentPosts:
    def fetch(self, user, k, exclude):
        return [i for i in self.by_recency if i not in exclude][:k]
```

Return bare ids and the pipeline stamps `rank` and `source` itself, so a source
cannot mislabel itself or get its own ranks wrong.

## Exploration

```python
from recsys_candidates.exploration import epsilon_greedy

slate = epsilon_greedy(
    ranked, cold_start_pool, k=50, epsilon=0.1, rng=random.Random(seed)
)
```

`epsilon` has no default — how much of your slate you spend on learning is a
product decision. Arm priors and reward attribution live in your `ArmStore`.

Explored items come back with `meta["explored"] = True`. Log it: without that
flag, offline evaluation scores an explored item as though the ranker chose it.

`thompson_order` samples once per **arm**, not per candidate — the maximum of
many draws is biased upward, so per-candidate sampling makes a chatty arm beat
an equally good quiet one on order statistics alone.

## What it will not do for you

- Merge scores across sources without you saying you know they share a scale
- Pick your `epsilon`
- Persist bandit state (`InMemoryArmStore` is for tests, and says so)
- Hide a short slate, a dead source, or a failed one — read `Result`

## Documentation

[Retrieval hazards](https://github.com/ZynclaveTech/recsys/blob/main/docs/retrieval-hazards.md)
— nine ways candidate generation degrades into something that still looks like
a product.

## License

Apache-2.0
