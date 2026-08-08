# Hazards in offline recommender evaluation

Every hazard on this page has the same shape: **the gate keeps reporting
success while the thing it measures quietly stops being real.** None of them
raise. None show up in a dashboard. Most were found in production, months after
they started.

There is a pattern worth internalising before the list. Almost every one of
these failures biases in the *same direction* — toward promoting the candidate.
That is not a coincidence. A bug that made the gate reject good models would be
found in a week, because someone would be blocked. A bug that makes it accept
bad ones has no complainant.

**Design your gate assuming its failures are silent and one-directional.**

---

## The one property

A promotion gate compares a candidate against an incumbent. For the comparison
to mean anything, the two must be scored under *identical* conditions: same
holdout window, same users, same labels, same candidate pool, same features.
Then the only variable left is the model.

Every hazard below is a way that property stops holding without anyone
noticing.

---

## 1. The ranker mutates the exclusion set you handed it

**Symptom:** none. Scores look plausible. The gate promotes everything.

Ranking APIs commonly take a set of already-seen items to filter out, and many
of them *add their own results to that set* — reasonable for paginated serving,
catastrophic here. Pass the same set object to both passes and the first
ranker's results become the second ranker's exclusions. The incumbent is now
forbidden from returning the items the candidate just returned, including
ground-truth items. The candidate wins every time.

**Do:** pass a fresh copy per call. `Fixture.seen` is a `frozenset` for this
reason — a mutating ranker fails loudly instead of corrupting the run.

---

## 2. Wall-clock features drift between the two passes

**Symptom:** none. The second model scored is systematically worse.

If any feature derives from "now" — item age, hour of day, recency decay — then
scoring the candidate at 02:00 and the incumbent at 02:40 feeds them different
inputs for the same item. An item that read one hour old to the first reads
1.7 hours old to the second. That difference is often larger than the
regression tolerance, and it moves in the promote-always direction, because the
candidate is usually scored first.

Caches make it worse, not better: a TTL that expires between passes recomputes
at the later wall clock, so the drift is silent and intermittent.

**Do:** snapshot every wall-clock-derived feature once, before either pass, and
serve both passes from the snapshot. Verify with a determinism check: score the
same model twice and assert the results are identical. If they are not, nothing
downstream means anything.

---

## 3. A shared cache makes every model score identically

**Symptom:** candidate and incumbent produce suspiciously similar metrics.
Often *exactly* identical, which reads as "no regression" — a promotion.

Embedding and feature caches are usually keyed by entity id, not by model. Load
a second model into the same process and it reads the first model's cached
embeddings. You are now scoring one model twice.

**Do:** key caches by model identity, or disable them entirely during
evaluation. Then verify: two genuinely different models must produce different
scores. If they never do, your isolation is broken — and *identical* scores are
the tell, because real models essentially never tie.

---

## 4. The incumbent silently loads as a random model

**Symptom:** the candidate looks enormously better than the incumbent. Every
run. The audit row looks plausible.

Model loaders often tolerate a shape mismatch by falling back to a freshly
initialised model rather than raising — a sane default in serving, where a
random model beats a crash. In evaluation it is a disaster: the incumbent
scores near chance, any candidate beats it, and the gate rubber-stamps the
promotion with a complete-looking record.

**Do:** check that the incumbent is *comparable* before comparing. If its
architecture no longer matches, that is not a comparison you can make — record
the run as ungated and say why. Do not compare against something you cannot
verify actually loaded.

A useful heuristic: an implausibly *large* improvement is a bug report, not a
win. Gate on suspicious improvements as well as regressions.

---

## 5. The candidate pool is derived from the labels

**Symptom:** coverage is 1.0. Recall looks great and never moves.

If the pool is built from the same event stream as the labels, every labelled
item is in the pool by construction. You have measured a retrieval ceiling your
production system does not have, and the metric can no longer detect a
shrinking index — one of the main things it exists to detect.

**Do:** the pool is a property of your *serving system*, not your logs. Pass it
explicitly. `Fixture.from_interactions` requires `candidates` for this reason,
and reports `label_coverage` so the real ceiling is visible.

---

## 6. Duplicate ids inflate NDCG above 1.0

**Symptom:** NDCG above 1.0, if anyone happens to look.

DCG accumulates gain per rank; the ideal ordering counts each item once. A
ranking containing the same item twice is therefore paid twice for it while the
denominator stays fixed. The ceiling is not 1.0 any more, and in a
relative-tolerance gate an inflated candidate score is an unbounded free pass.

Retrieval layers dedupe — right up until a merge between two candidate sources
stops doing so.

**Do:** score each id once, at its first occurrence. `recsys_eval` does this in
`ndcg_at_k` and `average_precision_at_k`. Assert your metrics are in `[0, 1]`;
property-based tests find this in seconds.

---

## 7. Negative gains drive NDCG below 0

**Symptom:** negative NDCG, or a metric that moves in the wrong direction.

The ideal ordering excludes non-positive gains — there is no point placing an
item you would rather not show. If DCG *includes* them, the two halves of the
ratio disagree and the result goes negative.

This is one config change from live in most systems, because interaction-weight
tables routinely carry entries like `block: -5.0` or `not_interested: -2.0`.
The moment someone widens the label filter to "use all the signals", the metric
breaks.

**Do:** treat non-positive gains identically in both halves — skip them.
NDCG cannot express dislike. If you need to penalise surfacing disliked items,
that is a **separate** metric reported alongside, not a negative number
smuggled into this one.

---

## 8. High-volume signals drown the labels

**Symptom:** metrics are stable and unmoved by changes that obviously matter.

Impressions and views outnumber deliberate actions by orders of magnitude.
Include them as relevance and the metric largely measures *"did we put
something in front of them that they scrolled past"* — which is nearly
guaranteed, nearly constant, and nearly uninformative. It will look reassuringly
flat while ranking quality degrades.

**Do:** label on actions the user had to *choose* to take. `RelevancePolicy`
has no default `positive_kinds` precisely so this decision gets made on purpose
rather than inherited.

---

## 9. Non-deterministic truncation reads as model drift

**Symptom:** metrics wobble a few percent between runs on unchanged data.

Truncating a candidate pool ordered by a non-unique column — a timestamp, a
score — cuts a different set each run, because ties resolve arbitrarily. The
resulting noise is easily larger than the regression tolerance, so the gate
starts flipping decisions at random.

**Do:** add a unique tiebreaker to the ordering. Seed every sample. Then prove
it: build the fixture twice and assert the two are equal. `recsys_eval` sorts
users and seeds sampling so this holds by default.

---

## 10. Fail open or fail closed?

Both, in different places, and the split matters.

**Construction fails loud.** An empty candidate pool, an inverted window,
mixed naive and aware timestamps — these produce a fixture that scores every
model 0.0 and reports "no regression" forever. There is no safe way to continue,
so `Fixture.from_interactions` raises.

**The gate itself fails open.** Once the fixture is valid, a *bug in evaluation*
should not stop the model refreshing. A stale model degrades every day it is not
retrained; a missed check costs one cycle. Failing closed can deadlock retraining
entirely, which is the worse outcome.

The condition that makes fail-open safe is that it is never silent. Every
ungated promotion must be recorded, with the reason, somewhere a human reviews.
A fail-open gate with no audit trail is not a gate — it is a promotion pipeline
with extra steps.

---

## A minimum viable checklist

Before trusting any offline number:

- [ ] Score the same model twice. Results identical?
- [ ] Score two different models. Results different?
- [ ] Are all metrics within `[0, 1]`?
- [ ] Build the fixture twice. Fixtures equal?
- [ ] Is `label_coverage` reported next to every result?
- [ ] Is the incumbent verified to have actually loaded its weights?
- [ ] Is every ungated promotion recorded with a reason someone reads?

The first four take an afternoon and catch most of this page.
