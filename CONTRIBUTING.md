# Contributing

Thanks for looking. Bug reports and small, well-argued patches are the most
useful things you can send.

## Getting set up

```bash
git clone https://github.com/ZynclaveTech/recsys
cd recsys
uv sync
```

Everything runs through [uv](https://docs.astral.sh/uv/). No other setup.

## The checks

These four must pass. CI runs exactly them, on Python 3.10 through 3.13.

```bash
uv run pytest
uv run ruff check .
uv run ruff format .
uv run mypy
```

Plus the example, which CI executes on every version because documentation
that nothing runs is documentation that rots:

```bash
uv run python examples/quickstart.py
```

`mypy` runs in strict mode and the public API is fully annotated. If a change
needs an `Any`, that is usually worth a sentence in the PR explaining why.

## What this project is trying to be

Worth reading before a larger change, because several of the design decisions
look like oversights until you know what they are for. Most of them are
recorded in [`docs/hazards.md`](docs/hazards.md), which is the closest thing
here to a design document.

**Zero runtime dependencies.** The metrics are pure stdlib; the fixture and
gate reach your data through protocols, not through pandas, numpy, or an ORM.
A PR that adds a dependency needs to argue that the alternative is genuinely
worse, not merely more code. An evaluation harness that knows how to query your
database is not a library, it is a second copy of your application, and it rots
the moment the two diverge.

**Nothing silently returns a wrong number.** Every function either returns a
value in `[0, 1]` or raises with an explanation. `NaN` is never an acceptable
output — every comparison against it is false, so a gate reading one stops
rejecting anything and reports success forever.

**Errors explain the fix.** Compare:

> `TypeError: '<' not supported between instances of 'object' and 'str'`

with what this library raises instead — that sorting failed, that determinism
is why it matters, and that converting ids to `str` or `int` fixes it. The
second one is the standard. It is more work to write and it is the difference
between a library people keep and one they wrap.

**Comments say why, not what.** The most valuable lines in this codebase
explain a decision that looks wrong — why Recall keeps unretrievable labels in
its denominator, why there is no `LAST` aggregate, why the gate fails open but
the fixture fails loud. If you change one of those, change the comment with it.

## Things that will probably be declined

Not because they are unreasonable, but because they have been considered.
Arguments against any of these are welcome — bring the argument, though, not
just the patch.

- **A default for `RelevancePolicy.positive_kinds`.** It is required on purpose.
  Which signals count as relevance is the most consequential decision in
  offline evaluation and the one most often made by accident. A default would
  let people skip it, and the resulting metric looks stable while measuring
  almost nothing.
- **Deriving `candidates` from the event stream.** It would be convenient and
  it would pin `label_coverage` at 1.0 by construction, reporting a retrieval
  ceiling that production does not have.
- **A tolerance in `assert_deterministic`.** Exact equality is the point. A
  tolerance would hide precisely the drift the check exists to find.
- **Making `coverage@k` opt-in.** Forgetting it is the failure mode. A ranker
  can raise every per-user metric by collapsing onto universally-popular items,
  and coverage is the only thing here that notices.
- **A `LAST` aggregate.** It would depend on iteration order, and a harness
  whose output changes when you reorder its input cannot be trusted in a gate.

## Tests

Every behavioural change needs a test. Two kinds are used here, and the
difference matters:

- **Golden tests** pin exact arithmetic, so a refactor that changes a
  definition fails loudly instead of shifting every dashboard by 2%. Show the
  working in a comment — the DCG and IDCG terms, the denominator choice.
- **Property tests** (via [Hypothesis](https://hypothesis.readthedocs.io/))
  pin invariants: values stay in `[0, 1]`, Recall is monotonic in `k`, hit rate
  agrees with reciprocal rank. Both of the bugs fixed before the first release
  were found this way and neither would have been found by hand.

Name tests for the behaviour, not the function
(`test_recall_keeps_unretrievable_labels_in_the_denominator`, not
`test_recall_2`). When a test exists because of a real failure, say so in its
docstring. That sentence is what stops someone deleting it in two years.

## Adding a metric

1. Write it in `metrics.py` as a pure function, with the same degenerate-input
   contract: return `0.0`, never `NaN`, never raise.
2. Register it in `scoring.py` under `_GRADED` or `_BINARY`.
3. Add golden tests and add it to the property-test parametrisation.
4. Document what it is *for* and, more usefully, when it is the wrong choice.
   Reciprocal rank is a good template: excellent for search, wrong for a feed.

The registry is checked by a test that scores every advertised metric, so a
half-registered one fails rather than being quietly unavailable.

## Pull requests

- One concern per PR.
- Explain the *why* in the description. What breaks without this change?
- If it fixes a bug, the test that would have caught it goes in the same PR.
- Formatting is `ruff format`'s job, so it is never a review topic.

## Reporting a bug

A ranking, the labels, the `k`, what you expected, and what you got. A failing
test is better than all of that and takes about the same effort.

If you think a metric is wrong, say which definition you are comparing against.
Several of these have more than one defensible form — `average_precision_at_k`
normalises by `min(len(relevant), k)` while `recall_at_k` deliberately does
not, and both docstrings explain why.

## Releasing (maintainers)

1. Bump `version` in `packages/<package>/pyproject.toml` and land it.
2. Publish a GitHub Release tagged `<package>-v<version>`, e.g.
   `recsys-eval-v0.1.0`. The prefix looks redundant with one package in the
   tree and stops looking redundant the moment there are two.
3. `publish.yml` verifies the tag matches the declared version, builds,
   runs `twine check`, installs the wheel into a clean venv and imports it,
   then uploads via PyPI trusted publishing.

The version check exists because the failure it prevents is expensive: PyPI
will not accept a re-upload of a version number, so a mismatch caught at upload
time burns the number entirely. Caught here it costs a re-tag.

To rehearse without touching PyPI, run the workflow manually with target
`testpypi`.

Publishing uses OIDC rather than an API token, so there is no secret to leak or
rotate. It needs a one-time trusted publisher configured at
<https://pypi.org/manage/account/publishing/> for owner `ZynclaveTech`, repo
`recsys`, workflow `publish.yml`, and GitHub environments named `pypi` and
`testpypi`.

## Security

For anything with security impact, email the maintainers rather than opening a
public issue.

## License

By contributing you agree that your contributions are licensed under
[Apache-2.0](LICENSE), the same terms as the project.
