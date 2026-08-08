"""Ranking metrics for offline recommender evaluation.

Every function here is pure: no I/O, no third-party dependencies, no global
state. Each takes a ranking plus some labels and returns a float in ``[0, 1]``.

What these numbers mean
-----------------------
These metrics exist to compare two rankers under **identical** conditions, so
that a retrain can be gated on not-a-regression. They are deliberately *not* a
measure of absolute recommendation quality: in any real system the candidate
pool is truncated, which structurally caps Recall below 1.0 no matter how good
the model is.

Do not quote these numbers as system quality. Do compare them run over run,
candidate against incumbent.

Degenerate input
----------------
Every function returns ``0.0`` -- never ``NaN``, never raising -- when there is
nothing to score. A user with no usable labels therefore cannot poison a
macro-average, which matters because the macro-average is the number the
promotion gate reads.

Per-user vs. aggregate
----------------------
These are all **per-user** quantities. The familiar aggregate names are their
macro-averages across users:

===========================  ==========================================
Aggregate                    Macro-average of
===========================  ==========================================
MAP@k                        :func:`average_precision_at_k`
MRR@k                        :func:`reciprocal_rank_at_k`
NDCG@k                       :func:`ndcg_at_k`
===========================  ==========================================

Naming them honestly at this layer keeps the aggregation visible at the call
site, where the choice between macro- and micro-averaging actually belongs.
"""

from __future__ import annotations

import math
from collections.abc import Hashable, Iterable, Mapping, Sequence, Set
from typing import TypeVar

__all__ = [
    "average_precision_at_k",
    "catalog_coverage",
    "hit_rate_at_k",
    "ndcg_at_k",
    "precision_at_k",
    "recall_at_k",
    "reciprocal_rank_at_k",
]

ItemT = TypeVar("ItemT", bound=Hashable)


def ndcg_at_k(
    ranked_ids: Sequence[ItemT],
    gains: Mapping[ItemT, float],
    k: int,
) -> float:
    """Normalised Discounted Cumulative Gain with **graded** relevance.

    ``gains`` maps item id -> relevance gain. Grading matters: if a share is
    worth more to you than a like, say so here and NDCG will reward putting
    shares higher. Items absent from ``gains`` score 0.

    **Gains must be non-negative.** Non-positive gains are ignored, in both the
    ranking and the ideal ordering. NDCG has no coherent reading below zero:
    the ideal ordering excludes negative gains, so counting them in ``DCG``
    while omitting them from ``IDCG`` drives the ratio negative and destroys
    the ``[0, 1]`` normalisation every downstream tolerance check assumes.

    This matters because interaction-weight tables routinely carry negatives --
    ``block: -5.0``, ``not_interested: -2.0`` -- and feeding one in directly is
    the obvious thing to do. If you need to penalise surfacing disliked items,
    that is a **separate** metric measured alongside NDCG, not a negative
    number smuggled into this one.

    ``DCG@k``  = sum over rank ``i`` (1-indexed) of ``gain_i / log2(i + 1)``

    ``IDCG@k`` = the same over the best possible ordering of the true gains.

    Note that ``IDCG`` is computed over **all** known gains, truncated to ``k``
    -- not only over the gains that appear in ``ranked_ids``. A relevant item
    the ranker failed to retrieve therefore still inflates the denominator and
    depresses the score, which is the intended behaviour: failing to surface a
    good item is a real loss, not something to normalise away.

    Duplicate ids in ``ranked_ids`` are scored once, at their first occurrence.
    Without that guard a ranker returning the same good item twice accumulates
    its gain twice while ``IDCG`` counts it once, and NDCG climbs **above 1.0**
    -- which, read by a promotion gate, is an unbounded free pass for a
    candidate whose only talent is repeating itself. Retrieval layers dedupe,
    right up until a merge between two candidate sources stops doing so.

    Args:
        ranked_ids: Items in rank order, best first. Only the first ``k`` are read.
        gains: Item id -> graded relevance gain. Non-positive gains are ignored
            when building the ideal ordering.
        k: Cutoff rank.

    Returns:
        A float in ``[0, 1]``, or ``0.0`` if there is nothing to score.
    """
    if k <= 0 or not ranked_ids or not gains:
        return 0.0

    dcg = 0.0
    scored: set[ItemT] = set()
    for rank, item_id in enumerate(ranked_ids[:k], start=1):
        if item_id in scored:
            continue
        scored.add(item_id)
        # `> 0`, not truthiness: a negative gain must be skipped here exactly
        # as it is skipped when building `ideal_gains` below. Counting it in
        # one and not the other is what makes NDCG go negative.
        gain = gains.get(item_id, 0.0)
        if gain > 0:
            dcg += gain / math.log2(rank + 1)

    ideal_gains = sorted((g for g in gains.values() if g > 0), reverse=True)[:k]
    idcg = 0.0
    for rank, gain in enumerate(ideal_gains, start=1):
        idcg += gain / math.log2(rank + 1)

    if idcg <= 0:
        return 0.0
    return dcg / idcg


def recall_at_k(
    ranked_ids: Sequence[ItemT],
    relevant_ids: Set[ItemT],
    k: int,
) -> float:
    """Fraction of relevant items retrieved in the top ``k``. Binary relevance.

    Relevant items that are **absent from the ranking entirely** -- typically
    because they fell outside a truncated candidate index -- remain in the
    denominator. Filtering them out would inflate the metric and hide a
    shrinking index, which is exactly the kind of regression a promotion gate
    exists to catch.

    This is why the value is structurally capped below 1.0 in most real
    deployments, and why it is only meaningful as a run-over-run comparison.
    Track the fraction of labels your index can even reach alongside it.

    Args:
        ranked_ids: Items in rank order, best first. Only the first ``k`` are read.
        relevant_ids: Items considered relevant for this user.
        k: Cutoff rank.

    Returns:
        A float in ``[0, 1]``, or ``0.0`` if there are no relevant items.
    """
    if k <= 0 or not relevant_ids:
        return 0.0
    hits = len(set(ranked_ids[:k]) & relevant_ids)
    return hits / len(relevant_ids)


def precision_at_k(
    ranked_ids: Sequence[ItemT],
    relevant_ids: Set[ItemT],
    k: int,
) -> float:
    """Fraction of the top ``k`` that is relevant. Binary relevance.

    The denominator is ``k``, not ``len(ranked_ids[:k])``. A ranker that
    returns three items when asked for ten has under-delivered, and dividing by
    the short length would score that as a full-credit list. Under-delivery is
    a regression the gate should see.

    Args:
        ranked_ids: Items in rank order, best first. Only the first ``k`` are read.
        relevant_ids: Items considered relevant for this user.
        k: Cutoff rank.

    Returns:
        A float in ``[0, 1]``, or ``0.0`` if there is nothing to score.
    """
    if k <= 0 or not relevant_ids:
        return 0.0
    hits = len(set(ranked_ids[:k]) & relevant_ids)
    return hits / k


def average_precision_at_k(
    ranked_ids: Sequence[ItemT],
    relevant_ids: Set[ItemT],
    k: int,
) -> float:
    """Average Precision at ``k``. Binary relevance, rank-sensitive.

    Precision is sampled at each rank that holds a relevant item, and averaged.
    Unlike Recall, this rewards putting relevant items *early* rather than
    merely retrieving them.

    The denominator is ``min(len(relevant_ids), k)``: with a cutoff of ``k`` you
    cannot retrieve more than ``k`` relevant items, so normalising by the full
    label count would make a perfect top-``k`` score below 1.0 and the metric
    unreadable.

    This is a deliberate departure from :func:`recall_at_k`, which *does* keep
    unreachable labels in its denominator. The two answer different questions:
    Recall asks "how much of what mattered did we find", and is allowed to be
    capped by a truncated index; AP asks "given what we returned, how well was
    it ordered", where that cap would be noise.

    Duplicate ids in ``ranked_ids`` are counted once, at their first occurrence.

    Args:
        ranked_ids: Items in rank order, best first. Only the first ``k`` are read.
        relevant_ids: Items considered relevant for this user.
        k: Cutoff rank.

    Returns:
        A float in ``[0, 1]``, or ``0.0`` if there is nothing to score.
    """
    if k <= 0 or not ranked_ids or not relevant_ids:
        return 0.0

    hits = 0
    precision_sum = 0.0
    seen: set[ItemT] = set()
    for rank, item_id in enumerate(ranked_ids[:k], start=1):
        if item_id in seen:
            continue
        seen.add(item_id)
        if item_id in relevant_ids:
            hits += 1
            precision_sum += hits / rank

    if hits == 0:
        return 0.0
    return precision_sum / min(len(relevant_ids), k)


def reciprocal_rank_at_k(
    ranked_ids: Sequence[ItemT],
    relevant_ids: Set[ItemT],
    k: int,
) -> float:
    """Reciprocal of the rank of the first relevant item, or ``0.0`` if none.

    Macro-averaged across users this is MRR@k. It is the right metric when the
    user is looking for one thing and everything below the first hit is wasted
    -- search, "resume where I left off", a single hero slot. It is the wrong
    metric for a feed, where the whole list gets consumed.

    Args:
        ranked_ids: Items in rank order, best first. Only the first ``k`` are read.
        relevant_ids: Items considered relevant for this user.
        k: Cutoff rank.

    Returns:
        ``1 / rank`` of the first relevant item within the top ``k``, else ``0.0``.
    """
    if k <= 0 or not relevant_ids:
        return 0.0
    for rank, item_id in enumerate(ranked_ids[:k], start=1):
        if item_id in relevant_ids:
            return 1.0 / rank
    return 0.0


def hit_rate_at_k(
    ranked_ids: Sequence[ItemT],
    relevant_ids: Set[ItemT],
    k: int,
) -> float:
    """``1.0`` if any relevant item appears in the top ``k``, else ``0.0``.

    The coarsest metric here, and the most robust to sparse labels. When a
    holdout window is too thin for NDCG to say anything stable, hit rate will
    still move. Macro-averaged, it reads as "what fraction of users saw at
    least one thing they went on to engage with".

    Args:
        ranked_ids: Items in rank order, best first. Only the first ``k`` are read.
        relevant_ids: Items considered relevant for this user.
        k: Cutoff rank.

    Returns:
        ``1.0`` or ``0.0``.
    """
    if k <= 0 or not relevant_ids:
        return 0.0
    return 1.0 if set(ranked_ids[:k]) & relevant_ids else 0.0


def catalog_coverage(
    ranked_lists: Iterable[Sequence[ItemT]],
    catalog_size: int,
    k: int,
) -> float:
    """Fraction of the catalogue that appears in *anybody's* top ``k``.

    Not an accuracy metric, and not optional. A ranker can improve NDCG by
    collapsing onto a small set of universally-popular items, and every
    per-user metric here will applaud that while the catalogue quietly dies.
    Coverage is the counterweight: read it beside NDCG, and treat a large
    accuracy gain paired with a coverage collapse as a regression rather than a
    win.

    Args:
        ranked_lists: One ranking per user. Consumed once; a generator is fine.
        catalog_size: Total number of distinct items that *could* be recommended.
        k: Cutoff rank applied to each list.

    Returns:
        A float in ``[0, 1]``, or ``0.0`` if the catalogue is empty.
    """
    if k <= 0 or catalog_size <= 0:
        return 0.0
    surfaced: set[ItemT] = set()
    for ranked in ranked_lists:
        surfaced.update(ranked[:k])
    if not surfaced:
        return 0.0
    # A caller can pass a catalog_size smaller than what was actually surfaced
    # (a stale count, or rankings drawn from a wider pool). Clamp rather than
    # return >1.0, which would silently break any downstream tolerance check.
    return min(len(surfaced) / catalog_size, 1.0)
