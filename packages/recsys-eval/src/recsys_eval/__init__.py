"""Offline evaluation and promotion gating for recommender systems.

The package is built around one claim: the useful question is not "how good is
this model" but "is this candidate worse than the one already in production".
Those need different machinery, and conflating them is how teams end up with a
dashboard full of numbers that nobody can act on.

See :mod:`recsys_eval.metrics` for the metric definitions.
"""

from __future__ import annotations

from recsys_eval.fixture import Fixture
from recsys_eval.metrics import (
    average_precision_at_k,
    catalog_coverage,
    hit_rate_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank_at_k,
)
from recsys_eval.types import Aggregate, Interaction, RelevancePolicy

__version__ = "0.1.0.dev0"

__all__ = [
    "Aggregate",
    "Fixture",
    "Interaction",
    "RelevancePolicy",
    "__version__",
    "average_precision_at_k",
    "catalog_coverage",
    "hit_rate_at_k",
    "ndcg_at_k",
    "precision_at_k",
    "recall_at_k",
    "reciprocal_rank_at_k",
]
