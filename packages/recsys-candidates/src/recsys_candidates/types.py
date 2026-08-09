"""Core vocabulary: a candidate, and how many of them you want.

Nothing here knows what a "source" retrieves from. That is the point — the
value of a retrieval framework is in composing sources you already have, not in
owning them.
"""

from __future__ import annotations

from collections.abc import Hashable, Mapping
from dataclasses import dataclass, field
from typing import Any, Generic, TypeVar

__all__ = ["Budget", "Candidate", "ItemT", "UserT"]

UserT = TypeVar("UserT", bound=Hashable)
ItemT = TypeVar("ItemT", bound=Hashable)


@dataclass(frozen=True, slots=True)
class Candidate(Generic[ItemT]):
    """One item a source proposed, and why.

    Attributes:
        item: The item id.
        rank: Position in the source's own output, 0-based. **This is the
            field merging uses.** A source must return its candidates in its
            own preferred order; the rank is what makes two sources
            comparable, because a rank means the same thing everywhere and a
            score does not.
        source: Name of the source that produced it. Set by the pipeline from
            the mapping key, so a source cannot mislabel itself.
        score: The source's own score, if it has one. **Optional, and
            deliberately not used by the default merge.** A cosine similarity
            of 0.82, a popularity count of 4300, and a recency rank of 7 are
            not on a common scale, and combining them with weights silently
            lets whichever has the widest numeric range dominate. Keep it for
            logging, debugging, and downstream ranking; do not reach for it to
            merge unless you have actually calibrated the sources against each
            other.
        meta: Anything else the source wants to carry through — a reason
            string, a topic, an author id for diversity constraints.
    """

    item: ItemT
    rank: int
    source: str = ""
    score: float | None = None
    meta: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class Budget:
    """How many candidates to ask for, and from whom.

    Attributes:
        total: How many candidates the pipeline should return.
        per_source: Per-source caps, by source name. A source absent from this
            mapping is uncapped and bounded only by ``overfetch``. Use it to
            stop one cheap high-volume source from filling the whole slate.
        overfetch: Multiplier applied when asking each source, so there is
            slack for dedup and filtering. At the default of 3, a request for
            100 asks each source for 300.

            The default is deliberately generous. Under-fetching is the more
            common mistake and it fails quietly: filters remove more than
            expected, the slate comes back short, and the symptom shows up as
            "the feed is empty for some users" long after the cause.
        min_per_source: Floor reserved for each named source before the
            remaining slots are filled. This is how a small source survives
            contact with a large one — without it, a source contributing 2% of
            the pool is statistically absent from every slate, which is
            usually not what anyone intended when they added it.
    """

    total: int
    per_source: Mapping[str, int] = field(default_factory=dict)
    overfetch: int = 3
    min_per_source: Mapping[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.total <= 0:
            raise ValueError(f"total must be positive, got {self.total}.")
        if self.overfetch < 1:
            raise ValueError(
                f"overfetch must be at least 1, got {self.overfetch}. Below 1 "
                "you would ask sources for fewer candidates than you intend to "
                "return, so filtering could only ever produce a short slate."
            )
        for label, mapping in (
            ("per_source", self.per_source),
            ("min_per_source", self.min_per_source),
        ):
            for name, value in mapping.items():
                if value < 0:
                    raise ValueError(
                        f"{label}[{name!r}] must be non-negative, got {value}."
                    )
        reserved = sum(self.min_per_source.values())
        if reserved > self.total:
            raise ValueError(
                f"min_per_source reserves {reserved} slots but total is "
                f"{self.total}. The floors cannot all be honoured, and which "
                "source loses would depend on iteration order."
            )

    def fetch_size(self, source: str) -> int:
        """How many to ask ``source`` for."""
        cap = self.per_source.get(source, self.total)
        return max(cap, self.min_per_source.get(source, 0)) * self.overfetch
