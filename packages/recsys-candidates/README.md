# recsys-candidates

Composable candidate generation for recommender systems.

> **Status: alpha.** Vocabulary and merge strategies land first; the pipeline,
> filters, diversity constraints and exploration scaffolding follow.

Zero dependencies. Sources reach your data through a protocol, so the library
never needs a database client, an ANN library, or a model.

## The problem it takes a position on

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

So the default merge compares **ranks**, not scores. A rank of 1 means the same
thing coming out of every source.

```python
from recsys_candidates import reciprocal_rank_fusion

merged = reciprocal_rank_fusion(
    {"ann": ann_hits, "popular": popular_hits, "fresh": fresh_hits},
    weights={"ann": 2.0},  # how much you trust a source — answerable
)
```

`by_score` exists for when your sources genuinely share a scale, and says so
in its docstring. It raises rather than treating a missing score as zero.

## License

Apache-2.0
