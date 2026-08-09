# Hazards in candidate generation

A companion to [`hazards.md`](hazards.md), which covers offline evaluation.
The failures there hid inside a gate that kept saying "no regression". The ones
here hide inside a feed that keeps rendering.

That is the pattern worth carrying: **retrieval degrades into something that
still looks like a product.** A slate that is short, stale, collapsed onto one
author, or missing a whole source renders exactly like a healthy one. There is
no stack trace, no error rate, no alert — just a number in a dashboard that
drifts a little, in a week when several other things also changed.

Assume every failure below is silent, because every one of them is.

---

## 1. Blending scores from sources that do not share a scale

**Symptom:** none. One source quietly decides the whole slate.

Three sources return the same item:

```
ann         cosine similarity   0.82
popularity  interaction count   4300
recency     hours since posted  7
```

There is no weighting of those numbers that means anything. Sum them and
popularity wins every time, not because it is better but because its numbers
are bigger. The weights look like they are doing something — you can tune them,
the slate changes — so the problem survives review.

Per-source min-max normalisation does not fix it. Rescaling each source to
`[0, 1]` makes the top item of your *worst* source exactly as valuable as the
top item of your best.

**Do:** merge on **rank**, not score. A rank of 1 means the same thing coming
out of every source, which is the only property you get for free.
`reciprocal_rank_fusion` is the default here for this reason, and its `weights`
express *how much you trust a source* — a question a team can actually answer.

Keep `by_score` for the case where sources genuinely share a scale: several ANN
indexes over one embedding space, or sources all emitting a calibrated
probability. If you cannot name the unit, you are not in that case.

---

## 2. A dead source that nobody notices

**Symptom:** none. The other sources fill the gap and the slate stays full.

This is the most common one. A source starts returning empty — an index rebuild
failed, a feature flag flipped, a query started timing out — and the merge
simply proceeds without it. Slate length is unchanged. Latency *improves*.
Engagement drifts down by an amount indistinguishable from a normal week.

**Do:** record per-source contribution on every request and alert on it, not on
errors. `Result.fetched` and `Result.contributed` are separate for exactly this
reason: a source with healthy `fetched` and zero `contributed` is being
out-competed or filtered away, which is a completely different bug from one
that returned nothing.

A source's contribution going to zero should page someone. It never does,
because nothing threw.

---

## 3. Overfetching too little, and discovering it through support tickets

**Symptom:** short slates, for some users only.

You ask each source for 50 to fill a slate of 50. Then dedup removes the
overlap, the seen-filter removes what the user already read, policy removes a
few more — and heavy users, who have seen the most, get the shortest feeds. The
users most invested in your product get the worst experience, and they are the
least likely to be in your test accounts.

**Do:** overfetch deliberately (the default here is 3×), and treat a short slate
as an error condition rather than a natural outcome. `Result.short` exists to be
alerted on. Watch it segmented by user tenure, because the average will look
fine.

---

## 4. Exploration that is not logged as exploration

**Symptom:** your offline metrics slowly stop matching online results.

Ten percent of slots are filled by a bandit. Those items were not chosen by the
ranker, but nothing in the logs says so. Every downstream consumer — offline
evaluation, the promotion gate, the training set — treats them as model
choices.

The model then gets credit for exploration's wins and blame for its losses. Your
retrain learns from labels the model did not generate. The
[promotion gate](hazards.md#10-fail-open-or-fail-closed) compares two models on
data that partly reflects neither.

**Do:** mark explored items at the point of selection and carry the flag all the
way into your event stream. `recsys_candidates.exploration` sets
`meta["explored"] = True` and does not offer a way to turn it off for
ε-greedy — the flag is the point.

---

## 5. Thompson sampling drawn per candidate instead of per arm

**Symptom:** the bandit converges confidently on the wrong arm.

Sampling a posterior once per candidate looks equivalent and is not. The maximum
of thirty draws from a distribution is systematically higher than the maximum of
three. An arm that returns more candidates therefore wins the top slot far more
often than its posterior justifies — and the more it wins, the more reward it
accrues, so it wins more.

The bandit looks like it learned a preference. It learned which arm is chatty.

**Do:** sample **once per arm** and let every candidate from that arm inherit
the value. `thompson_order` does this, and there is a test asserting a
30-candidate arm and a 3-candidate arm with identical posteriors split the top
slot evenly.

---

## 6. Bandit state that resets every deploy

**Symptom:** none. The bandit keeps returning slates; it just never learns.

Arm statistics in process memory die on every rollout. If you ship twice a week,
the posterior never accumulates more than a few days of evidence, so it stays
near its prior forever. With several replicas it is worse: each holds a
different posterior and they disagree, so the same user gets a different arm
depending on which pod answered.

**Do:** put arm state somewhere that survives a deploy and is shared across
replicas. `InMemoryArmStore` is included for tests and is labelled
not-for-production for precisely this reason.

---

## 7. Order that depends on which source was fastest

**Symptom:** the same user gets a different slate on identical data.

Fan out concurrently, collect with `as_completed`, and merge input order now
depends on network jitter. Any tie in the merge breaks differently run to run.
Nothing is wrong with the data, the model, or the config, and no experiment can
be reproduced.

**Do:** reassemble results in **declaration order**, never completion order.
`generate` iterates the source mapping, not the futures. There is a test that
runs the same two sources with the delay on each in turn and requires identical
output.

---

## 8. Diversity constraints that quietly reorder the slate

**Symptom:** your carefully tuned merge stops mattering.

A diversity pass that reorders to satisfy its constraint undoes the ranking it
was handed. You spent effort deciding item 3 belongs at position 3; a reordering
diversity filter moves it to 11 and the merge weights you tuned are now
decoration.

**Do:** make diversity **subtractive and order-preserving** — walk the slate
once and drop what exceeds the cap. `max_per_key` never reorders.

And be careful with missing keys: pooling every item whose author is unknown
under a single `None` bucket makes "unknown" the most over-represented author in
the slate, which is the exact opposite of the intent. Exempt them instead.

---

## 9. Mutating the exclusion set

**Symptom:** later sources return less than earlier ones, for no reason.

The same seen-set is handed to every source in a fan-out. A source that adds its
own results to it — a natural thing to do for within-call dedup — shrinks the
candidate pool of every source that runs after it. With concurrency, *which*
sources are affected varies per request.

**Do:** pass an immutable set. `generate` freezes `exclude` before handing it
out, so a mutating source raises instead of corrupting the run. This is the same
hazard that appears in offline evaluation between scoring passes, which is a
good sign it is inherent rather than incidental.

---

## A minimum viable checklist

- [ ] Is per-source contribution recorded on every request, and alerted on?
- [ ] Does a short slate raise something, or just render?
- [ ] Are explored items flagged all the way into the event stream?
- [ ] Does bandit state survive a deploy and get shared across replicas?
- [ ] Does the same input produce the same slate twice?
- [ ] Is anything merging raw scores across sources with different units?

The last one takes five minutes and is the most likely to be a yes.
