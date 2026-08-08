"""Preflight checks: prove the harness works before trusting what it says.

Every hazard in ``docs/hazards.md`` shares a shape -- the gate keeps reporting
success while the thing it measures quietly stops being real -- and almost all
of them bias the same way, toward promoting the candidate. That direction is
not a coincidence. A bug that rejects good models is found in a week because
someone is blocked; a bug that accepts bad ones has no complainant.

So the checks here do not measure a model. They measure whether your
measurement means anything. Run them in CI, not once by hand.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence

from recsys_eval.fixture import Fixture
from recsys_eval.protocols import Ranker
from recsys_eval.scoring import DEFAULT_KS, DEFAULT_METRICS, Scores, score
from recsys_eval.types import ItemT, UserT

__all__ = [
    "CheckFailed",
    "assert_deterministic",
    "assert_discriminating",
    "assert_reproducible",
    "assert_within_unit_interval",
    "preflight",
]


class CheckFailed(AssertionError):  # noqa: N818
    """A preflight check failed. The harness is not trustworthy yet.

    Named for how it reads at a call site -- ``pytest.raises(CheckFailed)``,
    ``except CheckFailed`` -- rather than for the ``...Error`` convention. It
    subclasses ``AssertionError`` because that is what it is: these functions
    are assertions about the harness, and existing tooling already treats
    ``AssertionError`` as a failed check rather than a crash.
    """


def _differences(a: Scores, b: Scores) -> list[str]:
    keys = sorted(set(a.values) | set(b.values))
    out = []
    for key in keys:
        left = a.values.get(key)
        right = b.values.get(key)
        if left != right:
            out.append(f"  {key}: {left} != {right}")
    return out


def assert_deterministic(
    ranker: Ranker[UserT, ItemT],
    fixture: Fixture[UserT, ItemT],
    *,
    metrics: Sequence[str] = DEFAULT_METRICS,
    ks: Sequence[int] = DEFAULT_KS,
) -> None:
    """Score the same ranker twice and require identical results.

    The single most valuable check in this module, because it catches an entire
    family of bugs at once rather than one at a time: features derived from
    ``now()``, caches expiring between passes, unseeded sampling inside the
    ranker, iteration over an unordered set.

    All of those produce *small* differences, which is what makes them
    dangerous. A difference smaller than your regression tolerance is invisible
    to the gate and still large enough to decide a promotion. And since the
    candidate is conventionally scored first, the drift is directional.

    Exact equality is deliberate. Two runs of the same deterministic
    computation over the same inputs produce bit-identical floats; a tolerance
    here would only hide the thing being looked for.

    Raises:
        CheckFailed: If the two runs disagree anywhere.
    """
    first = score(ranker, fixture, metrics=metrics, ks=ks)
    second = score(ranker, fixture, metrics=metrics, ks=ks)

    if first.values != second.values:
        detail = "\n".join(_differences(first, second))
        raise CheckFailed(
            "Scoring the same ranker twice gave different results:\n"
            f"{detail}\n\n"
            "Something in the ranker is not a pure function of the fixture. "
            "The usual causes are a feature derived from the current time, a "
            "cache expiring between the two passes, or an unseeded sample. "
            "Until this passes, a comparison between two models is measuring "
            "the harness as much as the models. See docs/hazards.md."
        )


def assert_discriminating(
    candidate: Ranker[UserT, ItemT],
    incumbent: Ranker[UserT, ItemT],
    fixture: Fixture[UserT, ItemT],
    *,
    metrics: Sequence[str] = DEFAULT_METRICS,
    ks: Sequence[int] = DEFAULT_KS,
) -> None:
    """Require two genuinely different rankers to produce different scores.

    Identical scores from different models is the signature of shared state --
    an embedding or feature cache keyed by entity id rather than by model, so
    the second model loaded reads the first one's cached values. You are
    scoring one model twice.

    It reads as *"no regression"*, which is a promotion.

    Note the asymmetry with :func:`assert_deterministic`: exact equality is
    required there and forbidden here. Real models essentially never tie on a
    macro-average over a non-trivial fixture, so a tie is evidence of plumbing
    rather than of similarity.

    Raises:
        CheckFailed: If the two rankers score identically on every metric.
    """
    left = score(candidate, fixture, metrics=metrics, ks=ks)
    right = score(incumbent, fixture, metrics=metrics, ks=ks)

    if left.values == right.values:
        raise CheckFailed(
            "Two different rankers produced identical scores on every "
            f"metric ({left.values}).\n\n"
            "Real models essentially never tie, so this is almost certainly "
            "shared state rather than genuine similarity: an embedding or "
            "feature cache keyed by entity id instead of by model, so the "
            "second ranker is reading the first one's values. Key those "
            "caches by model identity, or disable them during evaluation. "
            "See docs/hazards.md."
        )


def assert_reproducible(
    build: Callable[[], Fixture[UserT, ItemT]],
) -> None:
    """Build the fixture twice and require the two to be equal.

    Catches non-deterministic truncation and unseeded sampling on the *data*
    side, which produce a few percent of run-to-run wobble -- easily more than
    a regression tolerance, so the gate starts flipping decisions at random on
    unchanged data.

    The usual cause is ordering a candidate pool by a non-unique column and
    truncating it: ties resolve arbitrarily, so a different pool is cut each
    time.

    Args:
        build: A zero-argument callable returning a fresh fixture. It must read
            the *same* source data each call for this check to mean anything --
            point it at a snapshot, not at a live table.

    Raises:
        CheckFailed: If the two fixtures differ.
    """
    first = build()
    second = build()

    mismatches = []
    if first.users != second.users:
        mismatches.append(
            f"  users: {len(first.users)} vs {len(second.users)} "
            f"(overlap {len(set(first.users) & set(second.users))})"
        )
    if first.candidates != second.candidates:
        overlap = len(set(first.candidates) & set(second.candidates))
        mismatches.append(
            f"  candidates: {len(first.candidates)} vs "
            f"{len(second.candidates)} (overlap {overlap})"
        )
    if first.ground_truth != second.ground_truth:
        mismatches.append("  ground_truth: labels differ")
    if first.seen != second.seen:
        mismatches.append("  seen: exclusion sets differ")

    if mismatches:
        detail = "\n".join(mismatches)
        raise CheckFailed(
            "Building the fixture twice gave different fixtures:\n"
            f"{detail}\n\n"
            "Two runs over the same data must agree, or the gate is comparing "
            "models across different holdouts and attributing the difference "
            "to the model. The usual cause is truncating a pool ordered by a "
            "non-unique column, where ties resolve arbitrarily; add a unique "
            "tiebreaker to the ordering. See docs/hazards.md."
        )


def assert_within_unit_interval(scores: Scores) -> None:
    """Require every reported value to be a real number in ``[0, 1]``.

    A metric above 1.0 is usually duplicate ids being paid for twice; below
    0.0, usually negative gains counted in the numerator but excluded from the
    ideal ordering. Both are unbounded free passes in a relative-tolerance
    gate. A ``NaN`` is worse: every comparison against it is false, so a gate
    reading it silently stops rejecting anything.

    Raises:
        CheckFailed: On any value outside ``[0, 1]``, or any ``NaN``.
    """
    bad = []
    for key, value in sorted(scores.values.items()):
        if math.isnan(value):
            bad.append(f"  {key}: NaN")
        elif not 0.0 <= value <= 1.0:
            bad.append(f"  {key}: {value}")

    if bad:
        detail = "\n".join(bad)
        raise CheckFailed(
            f"Metric values outside [0, 1]:\n{detail}\n\n"
            "Above 1.0 is usually duplicate ids scored more than once; below "
            "0.0 is usually negative gains counted in DCG but excluded from "
            "the ideal ordering; NaN makes every comparison false, so a gate "
            "reading it stops rejecting anything. See docs/hazards.md."
        )


def preflight(
    build: Callable[[], Fixture[UserT, ItemT]],
    candidate: Ranker[UserT, ItemT],
    incumbent: Ranker[UserT, ItemT],
    *,
    metrics: Sequence[str] = DEFAULT_METRICS,
    ks: Sequence[int] = DEFAULT_KS,
) -> Scores:
    """Run every check in this module, then return the candidate's scores.

    This is the afternoon's work that catches most of ``docs/hazards.md``. Put
    it in CI against a small fixed snapshot, so a change that breaks the
    harness fails a build rather than quietly promoting a worse model for a
    quarter.

    Raises:
        CheckFailed: On the first check that fails.
    """
    assert_reproducible(build)
    fixture = build()
    assert_deterministic(candidate, fixture, metrics=metrics, ks=ks)
    assert_deterministic(incumbent, fixture, metrics=metrics, ks=ks)
    assert_discriminating(candidate, incumbent, fixture, metrics=metrics, ks=ks)

    scores = score(candidate, fixture, metrics=metrics, ks=ks)
    assert_within_unit_interval(scores)
    return scores
