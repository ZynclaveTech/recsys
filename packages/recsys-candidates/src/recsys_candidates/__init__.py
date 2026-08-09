"""Composable candidate generation for recommender systems.

Built around one position: sources do not share a score scale, so merging on
score is wrong by default. Ranks are comparable; scores are not. See
:mod:`recsys_candidates.merge`.
"""

from __future__ import annotations

import importlib.metadata

from recsys_candidates.exploration import (
    ArmStore,
    Beta,
    InMemoryArmStore,
    epsilon_greedy,
    thompson_order,
)
from recsys_candidates.merge import (
    RRF_K,
    by_score,
    interleave,
    reciprocal_rank_fusion,
    sources_of,
)
from recsys_candidates.pipeline import Result, generate, max_per_key
from recsys_candidates.protocols import Source
from recsys_candidates.types import Budget, Candidate

__version__ = importlib.metadata.version("recsys-candidates")

__all__ = [
    "RRF_K",
    "ArmStore",
    "Beta",
    "Budget",
    "Candidate",
    "InMemoryArmStore",
    "Result",
    "Source",
    "__version__",
    "by_score",
    "epsilon_greedy",
    "generate",
    "interleave",
    "max_per_key",
    "reciprocal_rank_fusion",
    "sources_of",
    "thompson_order",
]
