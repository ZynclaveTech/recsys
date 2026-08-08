"""The promotion gate: decide whether a candidate may replace the incumbent.

The question is deliberately narrow. Not *"is this model good"* -- offline
metrics cannot answer that -- but *"is this model worse than the one already
serving"*. That question is answerable, because both sides are scored on one
fixture and the only variable left is the model.

Two failure policies, applied in different places, and the split is the whole
design:

**Construction fails loud.** A malformed fixture or an unknown metric name
raises. There is no safe way to continue from a harness that scores every model
0.0 and reports "no regression" forever.

**Evaluation fails open.** Once inputs are valid, a *bug in evaluation* must not
stop the model refreshing. A stale model degrades every day it is not retrained;
a missed check costs one cycle. What makes that safe is that it is never
silent: every ungated promotion is recorded with its reason.
"""

from __future__ import annotations

import enum
import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from recsys_eval.fixture import Fixture
from recsys_eval.protocols import Ranker
from recsys_eval.scoring import DEFAULT_KS, DEFAULT_METRICS, Scores, score
from recsys_eval.types import ItemT, UserT

__all__ = ["Decision", "GatePolicy", "GateResult", "decide", "run_gate"]

logger = logging.getLogger(__name__)


class Decision(str, enum.Enum):
    """What the gate concluded. Each value implies a different next action."""

    PROMOTED = "promoted"
    """The candidate did not regress. Ship it."""

    REJECTED = "rejected"
    """The candidate is measurably worse. Do not ship it; investigate the model."""

    SUSPICIOUS = "suspicious"
    """The candidate looks *too* good. Do not ship it; investigate the harness.

    An implausibly large improvement is a bug report, not a win. In practice it
    means the incumbent was not scored under fair conditions -- most often it
    silently loaded as a randomly-initialised model after a shape mismatch, so
    it scored near chance and anything beats it.

    Gating only on regressions leaves this entire class of bug undetectable,
    and it is the class that promotes bad models.
    """

    UNGATED = "ungated"
    """No verdict was possible. Ship it, but the run needs a human eventually.

    Reached when there is no incumbent, too few users to be meaningful, nothing
    comparable to measure against, or evaluation itself failed. This is the
    fail-open path: safe only because it is recorded.
    """


@dataclass(frozen=True)
class GatePolicy:
    """Thresholds and failure behaviour for :func:`decide` and :func:`run_gate`.

    Attributes:
        gated_metrics: Keys from :class:`~recsys_eval.scoring.Scores` that can
            block a promotion, e.g. ``("ndcg@10", "recall@50")``. Gate on at
            least one rank-sensitive and one set metric: a ranker can trade
            them against each other, so a single-metric gate can be walked
            straight past.
        tolerance: Relative drop allowed before a metric counts as a
            regression. ``0.02`` means a 2% fall is noise. Set it from your own
            run-to-run variance, measured -- not guessed. If you have not
            measured it, run the same model through the gate on consecutive
            days and look at the spread.
        min_users: Below this many scored users, refuse to decide. A gate
            reading a handful of users is reading noise and will flip verdicts
            at random.
        max_improvement: Relative *gain* above which a result is called
            :attr:`Decision.SUSPICIOUS` instead of promoted. ``None`` disables
            the check, which is rarely what you want. See
            :attr:`Decision.SUSPICIOUS`.
        fail_open: When evaluation raises, promote as :attr:`Decision.UNGATED`
            rather than propagating. On by default. Turn it off only if a
            deadlocked retrain pipeline is genuinely less costly to you than a
            stale model, which is unusual.
        reraise: Exception types that must escape even when ``fail_open`` is
            set. The case this exists for: Celery raises
            ``SoftTimeLimitExceeded`` -- an ordinary ``Exception`` subclass --
            to tell a worker to wind down. Swallowing it would publish a model
            *after* the worker was told to stop, and risk being killed by the
            hard limit mid-upload. Pass ``(SoftTimeLimitExceeded,)`` and it
            propagates, so the task fails cleanly and nothing is promoted.
    """

    gated_metrics: Sequence[str]
    tolerance: float = 0.02
    min_users: int = 50
    max_improvement: float | None = 0.5
    fail_open: bool = True
    reraise: tuple[type[BaseException], ...] = ()

    def __post_init__(self) -> None:
        if not self.gated_metrics:
            raise ValueError(
                "gated_metrics is empty, so nothing could ever block a "
                "promotion and the gate would be decorative."
            )
        if not 0.0 <= self.tolerance < 1.0:
            raise ValueError(
                f"tolerance must be in [0, 1), got {self.tolerance}. It is a "
                "relative drop: 0.02 means 'a 2% fall is noise'."
            )
        if self.min_users < 0:
            raise ValueError(f"min_users must be >= 0, got {self.min_users}.")
        if self.max_improvement is not None and self.max_improvement <= 0:
            raise ValueError(
                f"max_improvement must be positive or None, got {self.max_improvement}."
            )


@dataclass(frozen=True)
class GateResult:
    """The verdict, the reasoning, and both sets of scores.

    Persist all of it. Six weeks later, a score that moved because the
    candidate pool shrank looks identical to one that moved because the model
    changed, and the fixture summary is the only thing that tells them apart.
    """

    decision: Decision
    reason: str
    candidate: Scores = field(default_factory=Scores)
    incumbent: Scores | None = None

    @property
    def promoted(self) -> bool:
        """Whether to ship. Only :attr:`Decision.PROMOTED` and
        :attr:`Decision.UNGATED` are ship-able; the latter needs review."""
        return self.decision in (Decision.PROMOTED, Decision.UNGATED)

    def as_dict(self) -> dict[str, Any]:
        """Flat dict for an audit row."""
        return {
            "decision": self.decision.value,
            "reason": self.reason,
            "candidate": self.candidate.as_dict(),
            "incumbent": self.incumbent.as_dict() if self.incumbent else None,
        }


def decide(
    candidate: Scores,
    incumbent: Scores | None,
    *,
    policy: GatePolicy,
) -> GateResult:
    """Compare two already-computed score sets. Pure; raises only on misuse.

    Args:
        candidate: Scores for the model being considered.
        incumbent: Scores for the model currently serving, or ``None``.
        policy: Thresholds. See :class:`GatePolicy`.

    Returns:
        A :class:`GateResult`.

    Raises:
        ValueError: If a gated metric is missing from either score set. That is
            a configuration error, not a model outcome, and silently treating
            it as "no regression" would disable the gate.
    """
    if incumbent is None:
        return GateResult(
            Decision.UNGATED,
            "no incumbent to compare against",
            candidate,
            None,
        )

    if candidate.scored_users < policy.min_users:
        return GateResult(
            Decision.UNGATED,
            f"only {candidate.scored_users} scored users, below the minimum "
            f"of {policy.min_users}; too few to distinguish a regression "
            "from noise",
            candidate,
            incumbent,
        )

    missing = sorted(
        m
        for m in policy.gated_metrics
        if m not in candidate.values or m not in incumbent.values
    )
    if missing:
        raise ValueError(
            f"Gated metric(s) not present in the scores: {', '.join(missing)}. "
            f"Available: {', '.join(sorted(candidate.values))}. Treating a "
            "missing metric as 'no regression' would silently disable the gate."
        )

    regressions: list[str] = []
    suspicious: list[str] = []
    comparable = 0

    for metric in policy.gated_metrics:
        base = incumbent[metric]
        new = candidate[metric]

        if base <= 0:
            # Nothing to regress from. Not counted as comparable: a gate whose
            # every metric reads 0.0 on the incumbent has measured nothing.
            continue

        comparable += 1
        change = (new - base) / base

        if -change > policy.tolerance:
            regressions.append(
                f"{metric} {base:.4f} -> {new:.4f} "
                f"({-change:.1%} drop, tolerance {policy.tolerance:.1%})"
            )
        elif policy.max_improvement is not None and change > policy.max_improvement:
            suspicious.append(
                f"{metric} {base:.4f} -> {new:.4f} "
                f"({change:.1%} gain, above the {policy.max_improvement:.1%} "
                "plausibility limit)"
            )

    if comparable == 0:
        return GateResult(
            Decision.UNGATED,
            "no gated metric had a positive incumbent value, so there was "
            "nothing to compare against",
            candidate,
            incumbent,
        )

    if regressions:
        return GateResult(
            Decision.REJECTED, "; ".join(regressions), candidate, incumbent
        )

    if suspicious:
        return GateResult(
            Decision.SUSPICIOUS,
            "; ".join(suspicious)
            + ". An improvement this large usually means the incumbent was "
            "not scored fairly -- check that it loaded its trained weights",
            candidate,
            incumbent,
        )

    return GateResult(
        Decision.PROMOTED,
        f"no regression beyond {policy.tolerance:.1%} on "
        f"{comparable} compared metric(s)",
        candidate,
        incumbent,
    )


def run_gate(
    candidate: Ranker[UserT, ItemT],
    incumbent: Ranker[UserT, ItemT] | None,
    fixture: Fixture[UserT, ItemT],
    *,
    policy: GatePolicy,
    metrics: Sequence[str] = DEFAULT_METRICS,
    ks: Sequence[int] = DEFAULT_KS,
    comparability: Callable[[], str | None] = lambda: None,
    recorder: Callable[[GateResult], None] | None = None,
    context: str = "",
) -> GateResult:
    """Score both rankers on one fixture and decide. Fails open by default.

    Args:
        candidate: The model being considered.
        incumbent: The model currently serving, or ``None`` on a first run.
        fixture: The **same** fixture object for both. This is what makes the
            comparison mean anything.
        policy: Thresholds and failure behaviour.
        metrics: Metric names to compute. Must cover ``policy.gated_metrics``.
        ks: Cutoffs to compute at.
        comparability: Called before scoring; return a reason string to refuse
            the comparison, or ``None`` to proceed. Use it to verify the
            incumbent is genuinely the trained model -- checkpoint loaders
            commonly fall back to a randomly-initialised model on a shape
            mismatch rather than raising, which scores near chance and makes
            any candidate look enormous. Refusing to compare beats comparing
            against noise.
        recorder: Called with the result before returning, including on
            failure. Write your audit row here. Exceptions from it are logged
            and swallowed -- a broken recorder must not take down the gate.
        context: Prefixed to the recorded reason. Useful for naming a run, e.g.
            a training job id.

    Returns:
        A :class:`GateResult`. Never raises when ``policy.fail_open`` is set,
        except for the types listed in ``policy.reraise``.
    """
    result: GateResult
    try:
        refusal = comparability()
        if refusal:
            result = GateResult(
                Decision.UNGATED,
                refusal,
                score(candidate, fixture, metrics=metrics, ks=ks),
                None,
            )
        else:
            candidate_scores = score(candidate, fixture, metrics=metrics, ks=ks)
            incumbent_scores = (
                score(incumbent, fixture, metrics=metrics, ks=ks)
                if incumbent is not None
                else None
            )
            result = decide(candidate_scores, incumbent_scores, policy=policy)
    except policy.reraise:
        # Listed explicitly by the caller as "must not be swallowed" -- a
        # worker shutdown signal, a cancellation. Let it propagate so the task
        # fails cleanly and nothing is promoted.
        raise
    except Exception as exc:  # fail open, deliberately
        # Deliberately broad and deliberately not bare: BaseException
        # (KeyboardInterrupt, SystemExit) still propagates. Everything else
        # means "evaluation is broken", which must not stop the model
        # refreshing. Never silent -- logged with a traceback, and recorded.
        if not policy.fail_open:
            raise
        logger.warning("Evaluation failed; promoting UNGATED: %s", exc, exc_info=True)
        result = GateResult(Decision.UNGATED, f"evaluation failed: {exc}")

    if context:
        result = GateResult(
            result.decision,
            f"{context}: {result.reason}",
            result.candidate,
            result.incumbent,
        )

    if result.decision is Decision.REJECTED:
        logger.error("Candidate REJECTED: %s", result.reason)
    elif result.decision is Decision.SUSPICIOUS:
        logger.error("Candidate SUSPICIOUS: %s", result.reason)
    elif result.decision is Decision.UNGATED:
        logger.warning("Candidate promoted UNGATED: %s", result.reason)
    else:
        logger.info("Candidate promoted: %s", result.reason)

    if recorder is not None:
        try:
            recorder(result)
        except Exception as exc:
            # The verdict is already decided; losing the audit row is bad but
            # losing the retrain is worse.
            logger.warning("Could not record the gate result: %s", exc)

    return result
