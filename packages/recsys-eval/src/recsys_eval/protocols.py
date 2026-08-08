"""The ranker contract: two methods, and no opinion about what is behind them.

A ranker here can be a trained two-tower model, a heuristic, a SQL query, or a
stub that returns the most popular items. The scoring loop cannot tell, which is
the point -- a baseline you can implement in four lines is the most useful thing
to compare a new model against, and an interface that only fits neural networks
quietly makes that comparison impossible.
"""

from __future__ import annotations

from collections.abc import Hashable, Sequence, Set
from typing import Protocol, TypeVar, runtime_checkable

__all__ = ["Ranker"]

# Declared here rather than imported from `types`: a protocol's type variables
# must carry the variance implied by where they appear. `UserT` is only ever
# consumed, so it is contravariant; `ItemT` is both consumed (`prepare`,
# `exclude`) and produced (`rank`), so it stays invariant.
UserT_contra = TypeVar("UserT_contra", bound=Hashable, contravariant=True)
ItemT = TypeVar("ItemT", bound=Hashable)


@runtime_checkable
class Ranker(Protocol[UserT_contra, ItemT]):
    """Something that can order a candidate pool for a user.

    Implement this over whatever you already have. Parameter *names* are part
    of the contract -- static checkers match protocols structurally, including
    keyword names -- so keep them as written.

    Example:
        >>> class MostPopular:
        ...     def __init__(self, plays: dict[str, int]) -> None:
        ...         self.plays = plays
        ...         self.ordered: list[str] = []
        ...
        ...     def prepare(self, candidates):
        ...         self.ordered = sorted(
        ...             candidates, key=lambda i: -self.plays.get(i, 0)
        ...         )
        ...
        ...     def rank(self, user, k, exclude):
        ...         return [i for i in self.ordered if i not in exclude][:k]

        That is a legitimate baseline, and a new model that cannot beat it is
        telling you something worth hearing before it reaches a gate.
    """

    def prepare(self, candidates: Sequence[ItemT]) -> None:
        """Do any per-pool setup: build an index, cache embeddings, sort.

        Called **once** per scoring run, before any call to :meth:`rank`, with
        the pool every user will be ranked against.

        Anything derived from wall-clock time belongs here rather than in
        :meth:`rank`, and even here it should come from a snapshot rather than
        from ``now()``. Two models scored forty minutes apart otherwise see
        different item ages for the same items, and that difference is usually
        larger than a regression tolerance. See ``docs/hazards.md``.
        """
        ...

    def rank(
        self, user: UserT_contra, k: int, exclude: Set[ItemT]
    ) -> Sequence[ItemT]:
        """Return up to ``k`` item ids for ``user``, best first.

        Return **bare ids**, not rows or dicts. Scoring reads the sequence
        directly, so there is no key name to get wrong -- a mismatch becomes a
        type error rather than an empty ranking that scores every model 0.0 and
        reports "no regression" forever.

        Args:
            user: Who to rank for.
            k: Maximum number of ids to return. Returning fewer is allowed and
                is scored as under-delivery, not excused.
            exclude: Items to leave out, normally what the user already
                consumed before the holdout window. **Do not mutate it.** Some
                serving APIs add their returned ids to the caller's exclusion
                set; do that here and the first model scored poisons the
                second's exclusions with its own results, ground truth
                included. ``Fixture.seen`` hands you a ``frozenset`` so the
                attempt fails loudly.

        Returns:
            Item ids in rank order. Duplicates are scored once, at their first
            occurrence, but returning them wastes slots you were given.
        """
        ...
