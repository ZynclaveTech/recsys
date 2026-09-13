"""Paired bootstrap: is the candidate *confidently* worse, or just unlucky?

A gate that compares two point estimates against a fixed tolerance is reading
noise whenever the metric is sparse. On a real feed, a few dozen of several
thousand users have any hit in the top ten, so NDCG@10 over a 2,000-user sample
moves by ~25% of its own value from sample to sample. A 2% tolerance on top of
that rejects an identical model more often than not.

The fix is to ask the question statistically. Both models are scored on the
same users, so the per-user *difference* is the thing to resample: user-level
noise that affects both models equally (a user who likes nothing) cancels out,
which makes the paired interval far tighter than two independent ones.

Pure stdlib, like the rest of the package. Deterministic given ``seed``.
"""

from __future__ import annotations

import math
import random
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

__all__ = ["PairedComparison", "paired_bootstrap"]


@dataclass(frozen=True)
class PairedComparison:
    """Candidate against incumbent on one metric, with a confidence interval.

    ``relative_change``, ``low`` and ``high`` are all relative to the
    incumbent's mean: ``-0.10`` is a 10% drop. The interval is the percentile
    bootstrap interval of the mean per-user difference, scaled by that same
    incumbent mean.
    """

    metric: str
    incumbent: float
    candidate: float
    relative_change: float
    low: float
    high: float
    confidence: float
    samples: int
    users: int

    def as_dict(self) -> dict[str, Any]:
        """Flat dict for an audit row."""
        return {
            "metric": self.metric,
            "incumbent": self.incumbent,
            "candidate": self.candidate,
            "relative_change": self.relative_change,
            "low": self.low,
            "high": self.high,
            "confidence": self.confidence,
            "samples": self.samples,
            "users": self.users,
        }

    def describe(self) -> str:
        """``ndcg@10 0.0038 -> 0.0016 (-57.0%, 95% CI -70.2% to -49.7%)``."""
        return (
            f"{self.metric} {self.incumbent:.4f} -> {self.candidate:.4f} "
            f"({self.relative_change:+.1%}, {self.confidence:.0%} CI "
            f"{self.low:+.1%} to {self.high:+.1%})"
        )


def paired_bootstrap(
    candidate: Sequence[float],
    incumbent: Sequence[float],
    *,
    metric: str = "",
    samples: int = 2000,
    confidence: float = 0.95,
    seed: int = 0,
) -> PairedComparison | None:
    """Bootstrap the relative change in a per-user metric between two models.

    Args:
        candidate: Per-user values for the candidate.
        incumbent: Per-user values for the incumbent, **for the same users in
            the same order**. Pairing is the whole point; misaligned inputs
            produce a confidently wrong interval, so a length mismatch raises.
        metric: Name carried into the result, for reporting.
        samples: Bootstrap resamples. 2,000 puts the 2.5th/97.5th percentiles
            within about a tenth of a standard error of their limit.
        confidence: Two-sided interval width, in ``(0, 1)``.
        seed: Resampling seed. Same inputs and seed, same interval.

    Returns:
        A :class:`PairedComparison`, or ``None`` when the incumbent's mean is
        zero: there is no baseline to be relative to, and inventing one would
        either divide by zero or quietly report a meaningless percentage.

    Raises:
        ValueError: On mismatched lengths, no users, a non-finite value, too
            few samples, or a confidence outside ``(0, 1)``.
    """
    if len(candidate) != len(incumbent):
        raise ValueError(
            f"candidate has {len(candidate)} users but incumbent has "
            f"{len(incumbent)}. A paired comparison needs the same users in the "
            "same order; score both models on one fixture."
        )
    n = len(candidate)
    if n == 0:
        raise ValueError("No users to compare.")
    if samples < 100:
        raise ValueError(
            f"samples must be >= 100, got {samples}. Fewer resamples leave the "
            "tail percentiles too noisy to decide anything on."
        )
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must be in (0, 1), got {confidence}.")

    diffs = [c - i for c, i in zip(candidate, incumbent, strict=True)]
    if not all(math.isfinite(d) for d in diffs):
        raise ValueError(
            "A per-user value is not finite. A NaN compares false against every "
            "threshold, so a gate reading one would never reject anything."
        )

    base = math.fsum(incumbent) / n
    if base <= 0.0:
        return None

    rng = random.Random(seed)
    means = sorted(math.fsum(rng.choices(diffs, k=n)) / n for _ in range(samples))
    alpha = (1.0 - confidence) / 2.0
    low = means[math.floor(alpha * (samples - 1))]
    high = means[math.ceil((1.0 - alpha) * (samples - 1))]

    return PairedComparison(
        metric=metric,
        incumbent=base,
        candidate=math.fsum(candidate) / n,
        relative_change=(math.fsum(diffs) / n) / base,
        low=low / base,
        high=high / base,
        confidence=confidence,
        samples=samples,
        users=n,
    )
