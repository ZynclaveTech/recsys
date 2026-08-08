"""Core vocabulary: what an interaction is, and how it becomes a label.

This module is the boundary between your system and this library. Everything
downstream -- fixtures, scoring, the promotion gate -- is written against these
types and never against a database, a dataframe, or a file format.

That is deliberate. An evaluation harness that knows how to query your ORM is
not a library, it is a second copy of your application, and it rots the moment
the two diverge.
"""

from __future__ import annotations

import enum
from collections.abc import Hashable
from dataclasses import dataclass
from datetime import datetime
from typing import Generic, TypeVar

__all__ = [
    "Aggregate",
    "Interaction",
    "ItemT",
    "RelevancePolicy",
    "UserT",
]

UserT = TypeVar("UserT", bound=Hashable)
ItemT = TypeVar("ItemT", bound=Hashable)


@dataclass(frozen=True, slots=True)
class Interaction(Generic[UserT, ItemT]):
    """One thing a user did to an item, at a point in time.

    This is the only shape this library ingests. Getting your events into it is
    your job and should be a few lines -- a ``SELECT``, a ``read_parquet``, a
    loop over a log file. Keep that adapter in your own codebase: it is the
    part that legitimately knows about your schema.

    ``events`` is consumed exactly once wherever it is accepted, so a generator
    streaming from disk is a first-class input rather than something to
    materialise into a list first.

    Attributes:
        user: Who acted. Any hashable id.
        item: What they acted on. Any hashable id.
        kind: The action, as a plain string -- ``"like"``, ``"share"``,
            ``"impression"``. Compared against
            :attr:`RelevancePolicy.positive_kinds`, so the spelling has to match
            whatever you put there.
        weight: How much this action is worth as a relevance signal. Only read
            for actions in ``positive_kinds``; everything else is ignored, so
            it is fine to leave the weight of an impression at whatever your
            pipeline already produces.
        at: When it happened. Must be consistently timezone-aware or
            consistently naive across the whole stream, and must match the
            holdout bounds you pass alongside it -- Python raises on mixed
            comparisons, and this library does not paper over that.
    """

    user: UserT
    item: ItemT
    kind: str
    weight: float
    at: datetime


class Aggregate(str, enum.Enum):
    """How to combine repeated positive actions on the same item.

    Both options are order-independent, which is what lets ``events`` be an
    unsorted stream. There is deliberately no ``LAST``: it would silently
    depend on iteration order, and an evaluation harness whose output changes
    when you reorder its input is worse than useless in a promotion gate.
    """

    MAX = "max"
    """Take the strongest single signal. The default.

    A user who likes *and* shares one item is expressing one strong preference,
    not two stacked ones. Summing would rank that item above an item the user
    engaged with more decisively but only once, which inverts the thing you
    were trying to measure.
    """

    SUM = "sum"
    """Add the signals together.

    Defensible when repetition is itself the signal -- replays of a track,
    rewatches of a video -- and misleading when it is not.
    """


@dataclass(frozen=True, slots=True)
class RelevancePolicy:
    """Which actions count as relevance, and how repeats combine.

    ``positive_kinds`` has no default, on purpose. It is the single most
    consequential decision in offline evaluation and the one most often made by
    accident, so this library makes you write it down.

    The failure mode it guards against: high-volume, low-intent signals --
    impressions, views, autoplays -- outnumber deliberate ones by orders of
    magnitude. Include them and your metric largely measures *"did we put
    something in front of them that they scrolled past"*, which is weak
    evidence that anything improved, and it will look reassuringly stable while
    the ranking quietly degrades.

    Start with the actions a user had to choose to take.

    Example:
        >>> policy = RelevancePolicy(
        ...     positive_kinds=frozenset({"like", "share", "save"})
        ... )
        >>> policy.aggregate is Aggregate.MAX
        True

    Attributes:
        positive_kinds: Action names that count as relevant. Everything else in
            the stream is used only to build the pre-cutoff *seen* set.
        aggregate: How repeated positive actions on one item combine.
        min_weight: Positive actions at or below this weight are discarded.
            Left at ``0.0`` this only filters out zero and negative weights,
            which is what you want: NDCG is undefined for negative relevance,
            and interaction-weight tables routinely carry entries like
            ``block: -5.0`` that would otherwise arrive here as labels.
    """

    positive_kinds: frozenset[str]
    aggregate: Aggregate = Aggregate.MAX
    min_weight: float = 0.0

    def __post_init__(self) -> None:
        if not self.positive_kinds:
            raise ValueError(
                "positive_kinds is empty, so no interaction can ever be "
                "labelled relevant and every metric would be 0.0. Name the "
                "actions that count, e.g. frozenset({'like', 'share'})."
            )

    def gain(self, event: Interaction[UserT, ItemT]) -> float:
        """Relevance gain for one event, or ``0.0`` if it does not count."""
        if event.kind not in self.positive_kinds:
            return 0.0
        if event.weight <= self.min_weight:
            return 0.0
        return float(event.weight)
