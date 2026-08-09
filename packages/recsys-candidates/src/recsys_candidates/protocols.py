"""The source contract: one method, and no opinion about what is behind it.

A source can be an ANN index, a SQL query, a Redis sorted set, a remote
service, or a list of hand-picked ids. The pipeline cannot tell, which is what
makes a hand-picked list a legitimate A/B arm rather than something you have to
special-case.
"""

from __future__ import annotations

from collections.abc import Hashable, Sequence, Set
from typing import Protocol, TypeVar, runtime_checkable

from recsys_candidates.types import Candidate

__all__ = ["Source"]

# A protocol's type variables must carry the variance implied by where they
# appear. `user` is only consumed; items are both consumed (`exclude`) and
# produced (the return), so ItemT stays invariant.
UserT_contra = TypeVar("UserT_contra", bound=Hashable, contravariant=True)
ItemT = TypeVar("ItemT", bound=Hashable)


@runtime_checkable
class Source(Protocol[UserT_contra, ItemT]):
    """Something that can propose items for a user.

    Parameter *names* are part of the contract — static checkers match
    protocols structurally, including keyword names — so keep them as written.

    Example:
        >>> class RecentPosts:
        ...     def __init__(self, by_recency: list[str]) -> None:
        ...         self.by_recency = by_recency
        ...
        ...     def fetch(self, user, k, exclude):
        ...         return [i for i in self.by_recency if i not in exclude][:k]

        Returning bare ids is the common case and the safe one.
    """

    def fetch(
        self, user: UserT_contra, k: int, exclude: Set[ItemT]
    ) -> Sequence[ItemT] | Sequence[Candidate[ItemT]]:
        """Return up to ``k`` items for ``user``, **best first**.

        The ordering is the contract. It is what every merge strategy reads,
        and it is the only signal a source is trusted to produce — a score is
        optional and, across sources, usually meaningless.

        Return bare ids unless you have a reason not to. If you return
        :class:`~recsys_candidates.types.Candidate` objects instead, you may
        attach ``score`` and ``meta`` (an author id for diversity constraints,
        a reason string for debugging) — but ``rank`` and ``source`` are always
        overwritten by the pipeline from your position in the sequence and the
        mapping key you were registered under. A source therefore cannot
        mislabel itself or get its own ranks wrong.

        Args:
            user: Who to retrieve for.
            k: Maximum to return. Already includes the budget's overfetch, so
                it is normally larger than the final slate.
            exclude: Items to leave out — typically what the user has already
                seen. **Do not mutate it.** The same set is handed to every
                source, so mutating it lets one source silently shrink the
                others' candidate pools. The pipeline passes a ``frozenset``,
                so the attempt fails loudly.

        Returns:
            Items best-first. Returning fewer than ``k`` is normal and is not
            an error. Raising is also survivable — the pipeline records the
            failure and continues with the other sources — but a source that
            raises contributes nothing, so check the result's ``failures``.
        """
        ...
