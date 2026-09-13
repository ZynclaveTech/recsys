"""Tests for :mod:`recsys_eval.gate`."""

from __future__ import annotations

import logging
from collections.abc import Sequence, Set

import pytest
from test_scoring import PoolRanker, make_fixture

from recsys_eval.gate import Decision, GatePolicy, GateResult, decide, run_gate
from recsys_eval.scoring import Scores

POLICY = GatePolicy(gated_metrics=["ndcg@2"], min_users=0)


def scores(
    ndcg: float, *, users: int = 100, extra: dict[str, float] | None = None
) -> Scores:
    return Scores(
        values={"ndcg@2": ndcg, **(extra or {})},
        scored_users=users,
        label_coverage=1.0,
    )


class Exploding:
    """A ranker that fails during preparation."""

    def __init__(self, exc: BaseException) -> None:
        self.exc = exc

    def prepare(self, candidates: Sequence[str]) -> None:
        raise self.exc

    def rank(self, user: str, k: int, exclude: Set[str]) -> Sequence[str]:
        return []


# --------------------------------------------------------------------------
# GatePolicy validation
# --------------------------------------------------------------------------


def test_empty_gated_metrics_is_rejected() -> None:
    with pytest.raises(ValueError, match="decorative"):
        GatePolicy(gated_metrics=[])


@pytest.mark.parametrize("tolerance", [-0.1, 1.0, 1.5])
def test_tolerance_must_be_a_relative_fraction(tolerance: float) -> None:
    with pytest.raises(ValueError, match="tolerance must be in"):
        GatePolicy(gated_metrics=["ndcg@2"], tolerance=tolerance)


def test_negative_min_users_is_rejected() -> None:
    with pytest.raises(ValueError, match="min_users"):
        GatePolicy(gated_metrics=["ndcg@2"], min_users=-1)


@pytest.mark.parametrize("value", [0.0, -1.0])
def test_non_positive_max_improvement_is_rejected(value: float) -> None:
    with pytest.raises(ValueError, match="max_improvement"):
        GatePolicy(gated_metrics=["ndcg@2"], max_improvement=value)


# --------------------------------------------------------------------------
# decide: the comparison
# --------------------------------------------------------------------------


def test_promotes_when_nothing_regressed() -> None:
    result = decide(scores(0.50), scores(0.50), policy=POLICY)
    assert result.decision is Decision.PROMOTED
    assert result.promoted


def test_promotes_a_drop_inside_the_tolerance() -> None:
    policy = GatePolicy(gated_metrics=["ndcg@2"], tolerance=0.02, min_users=0)
    result = decide(scores(0.495), scores(0.50), policy=policy)
    assert result.decision is Decision.PROMOTED


def test_rejects_a_drop_beyond_the_tolerance() -> None:
    policy = GatePolicy(gated_metrics=["ndcg@2"], tolerance=0.02, min_users=0)
    result = decide(scores(0.40), scores(0.50), policy=policy)
    assert result.decision is Decision.REJECTED
    assert not result.promoted
    assert "20.0% drop" in result.reason


def test_rejects_when_any_gated_metric_regresses() -> None:
    """A ranker can trade ordering against retrieval, so one bad metric is enough."""
    policy = GatePolicy(gated_metrics=["ndcg@2", "recall@2"], min_users=0)
    result = decide(
        scores(0.60, extra={"recall@2": 0.20}),
        scores(0.50, extra={"recall@2": 0.50}),
        policy=policy,
    )
    assert result.decision is Decision.REJECTED
    assert "recall@2" in result.reason
    assert "ndcg@2" not in result.reason


def test_flags_an_implausibly_large_improvement_as_suspicious() -> None:
    """The bug class that gating on regressions alone cannot see.

    A near-chance incumbent -- typically one that silently loaded as a randomly
    initialised model -- makes any candidate look enormous.
    """
    policy = GatePolicy(gated_metrics=["ndcg@2"], max_improvement=0.5, min_users=0)
    result = decide(scores(0.90), scores(0.10), policy=policy)
    assert result.decision is Decision.SUSPICIOUS
    assert not result.promoted
    assert "trained weights" in result.reason


def test_suspicion_check_can_be_disabled() -> None:
    policy = GatePolicy(gated_metrics=["ndcg@2"], max_improvement=None, min_users=0)
    result = decide(scores(0.90), scores(0.10), policy=policy)
    assert result.decision is Decision.PROMOTED


def test_a_regression_outranks_suspicion() -> None:
    policy = GatePolicy(
        gated_metrics=["ndcg@2", "recall@2"], max_improvement=0.5, min_users=0
    )
    result = decide(
        scores(0.90, extra={"recall@2": 0.10}),
        scores(0.10, extra={"recall@2": 0.50}),
        policy=policy,
    )
    assert result.decision is Decision.REJECTED


# --------------------------------------------------------------------------
# decide: refusing to decide
# --------------------------------------------------------------------------


def test_ungated_without_an_incumbent() -> None:
    result = decide(scores(0.50), None, policy=POLICY)
    assert result.decision is Decision.UNGATED
    assert result.promoted
    assert "no incumbent" in result.reason


def test_ungated_below_the_user_floor() -> None:
    policy = GatePolicy(gated_metrics=["ndcg@2"], min_users=50)
    result = decide(scores(0.40, users=10), scores(0.50), policy=policy)
    assert result.decision is Decision.UNGATED
    assert "10 scored users" in result.reason


def test_ungated_when_no_metric_is_comparable() -> None:
    """An incumbent scoring 0.0 everywhere has measured nothing.

    Treating that as "no regression" would promote unconditionally.
    """
    result = decide(scores(0.50), scores(0.0), policy=POLICY)
    assert result.decision is Decision.UNGATED
    assert "nothing to compare" in result.reason


def test_a_metric_with_a_zero_incumbent_is_skipped_not_counted() -> None:
    policy = GatePolicy(gated_metrics=["ndcg@2", "recall@2"], min_users=0)
    result = decide(
        scores(0.50, extra={"recall@2": 0.50}),
        scores(0.50, extra={"recall@2": 0.0}),
        policy=policy,
    )
    assert result.decision is Decision.PROMOTED
    assert "1 compared metric" in result.reason


def test_a_missing_gated_metric_raises_rather_than_passing() -> None:
    """Silently reading a missing metric as "no regression" disables the gate."""
    policy = GatePolicy(gated_metrics=["ndcg@10"], min_users=0)
    with pytest.raises(ValueError, match="not present in the scores"):
        decide(scores(0.50), scores(0.50), policy=policy)


# --------------------------------------------------------------------------
# GateResult
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("decision", "shippable"),
    [
        (Decision.PROMOTED, True),
        (Decision.UNGATED, True),
        (Decision.REJECTED, False),
        (Decision.SUSPICIOUS, False),
    ],
)
def test_promoted_property(decision: Decision, shippable: bool) -> None:
    assert GateResult(decision, "").promoted is shippable


def test_as_dict_carries_both_score_sets() -> None:
    result = decide(scores(0.50), scores(0.40), policy=POLICY)
    row = result.as_dict()
    assert row["decision"] == "promoted"
    assert row["candidate"]["ndcg@2"] == 0.50
    assert row["incumbent"]["ndcg@2"] == 0.40


def test_as_dict_handles_a_missing_incumbent() -> None:
    assert decide(scores(0.5), None, policy=POLICY).as_dict()["incumbent"] is None


# --------------------------------------------------------------------------
# run_gate
# --------------------------------------------------------------------------


def gate(**kwargs: object) -> GateResult:
    """Drive ``run_gate`` over a two-user fixture.

    ``max_improvement`` is disabled here on purpose: with two users the metric
    is coarse enough that any real difference clears the plausibility limit,
    and these tests are about plumbing. Threshold behaviour is covered against
    :func:`decide` directly, where the scores can be stated exactly.
    """
    params: dict[str, object] = {
        "policy": GatePolicy(
            gated_metrics=["ndcg@2"], min_users=0, max_improvement=None
        ),
        "metrics": ["ndcg"],
        "ks": [2],
    }
    params.update(kwargs)
    return run_gate(
        params.pop("candidate", PoolRanker()),  # type: ignore[arg-type]
        # Weaker than the candidate but not zero: an incumbent scoring 0.0
        # everywhere is correctly reported as "nothing to compare against",
        # which would make these tests pass for the wrong reason.
        params.pop("incumbent", PoolRanker(order=["i3", "i1", "i2", "i4"])),  # type: ignore[arg-type]
        make_fixture(),
        **params,  # type: ignore[arg-type]
    )


def test_run_gate_scores_both_and_decides() -> None:
    result = gate()
    assert result.decision is Decision.PROMOTED
    assert result.incumbent is not None
    assert result.candidate["ndcg@2"] > result.incumbent["ndcg@2"]


def test_run_gate_without_an_incumbent() -> None:
    result = gate(incumbent=None)
    assert result.decision is Decision.UNGATED
    assert result.incumbent is None
    assert result.candidate.scored_users == 2


def test_comparability_refusal_skips_the_comparison() -> None:
    """Refusing to compare beats comparing against a model that never loaded."""
    result = gate(comparability=lambda: "incumbent checkpoint dim mismatch")
    assert result.decision is Decision.UNGATED
    assert "dim mismatch" in result.reason
    assert result.incumbent is None


def test_context_is_prefixed_to_the_reason() -> None:
    assert gate(context="job-42").reason.startswith("job-42: ")


# --------------------------------------------------------------------------
# Failure policy
# --------------------------------------------------------------------------


def test_evaluation_failure_promotes_ungated_by_default(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A stale model degrades daily; a missed check costs one cycle."""
    with caplog.at_level(logging.WARNING):
        result = gate(candidate=Exploding(RuntimeError("index unavailable")))
    assert result.decision is Decision.UNGATED
    assert "index unavailable" in result.reason
    assert "Evaluation failed" in caplog.text


def test_fail_closed_propagates_instead() -> None:
    policy = GatePolicy(gated_metrics=["ndcg@2"], min_users=0, fail_open=False)
    with pytest.raises(RuntimeError, match="boom"):
        gate(candidate=Exploding(RuntimeError("boom")), policy=policy)


def test_reraise_types_escape_even_when_failing_open() -> None:
    """Celery's SoftTimeLimitExceeded is an ordinary Exception subclass.

    Swallowing it would publish a model after the worker was told to stop.
    """

    # Named exactly as Celery names it; the point is that a real-world
    # shutdown signal is an ordinary Exception subclass.
    class SoftTimeLimitExceeded(Exception):  # noqa: N818
        pass

    policy = GatePolicy(
        gated_metrics=["ndcg@2"],
        min_users=0,
        fail_open=True,
        reraise=(SoftTimeLimitExceeded,),
    )
    with pytest.raises(SoftTimeLimitExceeded):
        gate(candidate=Exploding(SoftTimeLimitExceeded()), policy=policy)


def test_keyboard_interrupt_is_never_swallowed() -> None:
    with pytest.raises(KeyboardInterrupt):
        gate(candidate=Exploding(KeyboardInterrupt()))


# --------------------------------------------------------------------------
# The recorder
# --------------------------------------------------------------------------


def test_recorder_receives_the_result() -> None:
    seen: list[GateResult] = []
    result = gate(recorder=seen.append)
    assert seen == [result]


def test_recorder_still_runs_when_evaluation_failed() -> None:
    """Fail-open is only safe because every ungated promotion is recorded."""
    seen: list[GateResult] = []
    gate(candidate=Exploding(RuntimeError("boom")), recorder=seen.append)
    assert len(seen) == 1
    assert seen[0].decision is Decision.UNGATED


def test_a_broken_recorder_does_not_take_down_the_gate(
    caplog: pytest.LogCaptureFixture,
) -> None:
    def explode(result: GateResult) -> None:
        raise OSError("audit table unreachable")

    with caplog.at_level(logging.WARNING):
        result = gate(recorder=explode)
    assert result.decision is Decision.PROMOTED
    assert "Could not record" in caplog.text


# --------------------------------------------------------------------------
# Paired bootstrap mode
# --------------------------------------------------------------------------

BOOTSTRAP = GatePolicy(
    gated_metrics=["ndcg@2"], min_users=0, max_improvement=None, bootstrap_samples=500
)


def paired(values: Sequence[float], users: Sequence[str] | None = None) -> Scores:
    users = users or [f"u{i}" for i in range(len(values))]
    return Scores(
        values={"ndcg@2": sum(values) / len(values)},
        scored_users=len(values),
        label_coverage=1.0,
        users=tuple(users),
        per_user={"ndcg@2": tuple(values)},
    )


def sparse_values(n: int, rate: float, seed: int) -> list[float]:
    import random

    rng = random.Random(seed)
    return [rng.random() if rng.random() < rate else 0.0 for _ in range(n)]


def test_bootstrap_promotes_an_equal_model_that_a_point_comparison_rejects() -> None:
    """Same quality, different lucky hits: the case this mode exists for."""
    incumbent = sparse_values(2000, 0.03, 11)
    candidate = sparse_values(2000, 0.03, 12)
    point = GatePolicy(gated_metrics=["ndcg@2"], min_users=0, max_improvement=None)
    cand, inc = paired(candidate), paired(incumbent)
    assert decide(cand, inc, policy=point).decision is Decision.REJECTED

    result = decide(cand, inc, policy=BOOTSTRAP)
    assert result.decision is Decision.PROMOTED
    assert "no metric confidently worse" in result.reason
    comparison = result.comparisons["ndcg@2"]
    assert comparison.low < 0.0 < comparison.high


def test_bootstrap_rejects_a_confidently_worse_model() -> None:
    incumbent = sparse_values(3000, 0.1, 13)
    result = decide(
        paired([v * 0.6 for v in incumbent]), paired(incumbent), policy=BOOTSTRAP
    )
    assert result.decision is Decision.REJECTED
    assert "the whole interval is below -2.0%" in result.reason
    assert result.comparisons["ndcg@2"].high < -0.02


def test_bootstrap_suspicion_needs_the_whole_interval_above_the_limit() -> None:
    policy = GatePolicy(
        gated_metrics=["ndcg@2"],
        min_users=0,
        max_improvement=0.5,
        bootstrap_samples=500,
    )
    incumbent = sparse_values(3000, 0.1, 14)
    result = decide(
        paired([v * 3 for v in incumbent]), paired(incumbent), policy=policy
    )
    assert result.decision is Decision.SUSPICIOUS
    assert "plausibility limit" in result.reason


def test_bootstrap_without_per_user_values_raises() -> None:
    with pytest.raises(ValueError, match="per-user values are missing"):
        decide(scores(0.5), scores(0.5), policy=BOOTSTRAP)


def test_bootstrap_on_different_users_raises() -> None:
    a = paired([0.1, 0.2], users=["u1", "u2"])
    b = paired([0.1, 0.2], users=["u1", "u3"])
    with pytest.raises(ValueError, match="different users"):
        decide(a, b, policy=BOOTSTRAP)


@pytest.mark.parametrize("samples", [0, 99])
def test_bootstrap_samples_floor(samples: int) -> None:
    with pytest.raises(ValueError, match="bootstrap_samples"):
        GatePolicy(gated_metrics=["ndcg@2"], bootstrap_samples=samples)


def test_confidence_is_validated() -> None:
    with pytest.raises(ValueError, match="confidence"):
        GatePolicy(gated_metrics=["ndcg@2"], confidence=1.0)


def test_run_gate_collects_per_user_scores_for_a_bootstrap_policy() -> None:
    fixture = make_fixture({"u1": ["i1"], "u2": ["i2"], "u3": ["i3"]})
    result = run_gate(
        PoolRanker(),
        PoolRanker(),
        fixture,
        policy=GatePolicy(gated_metrics=["ndcg@2"], min_users=0, bootstrap_samples=200),
        metrics=["ndcg"],
        ks=[2],
        context="job-1",
    )
    assert result.decision is Decision.PROMOTED
    assert result.reason.startswith("job-1: ")
    assert result.candidate.users == fixture.users
    assert "ndcg@2" in result.comparisons
    assert result.as_dict()["comparisons"]["ndcg@2"]["users"] == 3
