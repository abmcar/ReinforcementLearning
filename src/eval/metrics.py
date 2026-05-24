"""Evaluation metrics for crowdsourcing task recommendation.

Implements generic ranking metrics (HR, NDCG, MRR, Precision, Recall) and
domain-specific dual-objective metrics for worker and requester goals.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Set


# ---------------------------------------------------------------------------
# Generic ranking metrics (single query)
# ---------------------------------------------------------------------------


def hit_rate_at_k(ranked_list: List[int], ground_truth: int, k: int) -> float:
    """1.0 if ground_truth appears in ranked_list[:k], else 0.0."""
    return 1.0 if ground_truth in ranked_list[:k] else 0.0


def ndcg_at_k(ranked_list: List[int], ground_truth: int, k: int) -> float:
    """NDCG@K for a single relevant item.

    DCG  = 1 / log2(rank + 1)  if ground_truth in top-K, else 0
    IDCG = 1 / log2(2) = 1.0   (best case: relevant item at rank 1)
    """
    top_k = ranked_list[:k]
    if ground_truth not in top_k:
        return 0.0
    rank = top_k.index(ground_truth) + 1  # 1-based
    return 1.0 / math.log2(rank + 1)


def mrr(ranked_list: List[int], ground_truth: int) -> float:
    """Mean Reciprocal Rank (single query): 1/rank if found, else 0."""
    try:
        rank = ranked_list.index(ground_truth) + 1
        return 1.0 / rank
    except ValueError:
        return 0.0


def precision_at_k(
    ranked_list: List[int], ground_truths: Set[int], k: int
) -> float:
    """Precision@K = |relevant ∩ top-K| / K."""
    top_k = ranked_list[:k]
    hits = sum(1 for item in top_k if item in ground_truths)
    return hits / k if k > 0 else 0.0


def recall_at_k(
    ranked_list: List[int], ground_truths: Set[int], k: int
) -> float:
    """Recall@K = |relevant ∩ top-K| / |relevant|."""
    if not ground_truths:
        return 0.0
    top_k = ranked_list[:k]
    hits = sum(1 for item in top_k if item in ground_truths)
    return hits / len(ground_truths)


# ---------------------------------------------------------------------------
# Worker-objective metrics (single query)
# ---------------------------------------------------------------------------


def avg_award_value_at_k(
    ranked_list: List[int],
    project_award: Dict[int, float],
    k: int,
) -> float:
    """Average award_value of recommended top-K projects."""
    top_k = ranked_list[:k]
    if not top_k:
        return 0.0
    values = [project_award.get(pid, 0.0) for pid in top_k]
    return sum(values) / len(values)


def finalist_rate_at_k(
    ranked_list: List[int],
    project_finalist_set: Set[int],
    k: int,
) -> float:
    """Fraction of top-K projects where the worker became a finalist."""
    top_k = ranked_list[:k]
    if not top_k:
        return 0.0
    hits = sum(1 for pid in top_k if pid in project_finalist_set)
    return hits / len(top_k)


def winner_rate_at_k(
    ranked_list: List[int],
    project_winner_set: Set[int],
    k: int,
) -> float:
    """Fraction of top-K projects where the worker was the winner."""
    top_k = ranked_list[:k]
    if not top_k:
        return 0.0
    hits = sum(1 for pid in top_k if pid in project_winner_set)
    return hits / len(top_k)


def category_match_rate_at_k(
    ranked_list: List[int],
    project_category: Dict[int, int],
    worker_preferred_categories: Set[int],
    k: int,
) -> float:
    """Fraction of top-K projects whose category matches worker preference."""
    top_k = ranked_list[:k]
    if not top_k:
        return 0.0
    hits = sum(
        1
        for pid in top_k
        if project_category.get(pid, -1) in worker_preferred_categories
    )
    return hits / len(top_k)


# ---------------------------------------------------------------------------
# Requester-objective metrics (aggregated across all recommendations)
# ---------------------------------------------------------------------------


def avg_recommender_worker_quality(
    project_recommended_workers: Dict[int, List[int]],
    worker_quality: Dict[int, float],
    default_quality: float = 0.5,
) -> float:
    """Average quality of workers recommended to each project, macro-averaged.

    For each project, compute mean quality of all workers recommended to it,
    then average across projects.
    """
    if not project_recommended_workers:
        return 0.0
    project_means = []
    for _pid, workers in project_recommended_workers.items():
        if not workers:
            continue
        qualities = [worker_quality.get(w, default_quality) for w in workers]
        project_means.append(sum(qualities) / len(qualities))
    return sum(project_means) / len(project_means) if project_means else 0.0


def project_coverage(
    recommended_projects: Set[int],
    active_projects: Set[int],
) -> float:
    """Fraction of active projects that received at least one recommendation."""
    if not active_projects:
        return 0.0
    covered = recommended_projects & active_projects
    return len(covered) / len(active_projects)
