"""Baseline recommenders (JOB-05).

Provides simple non-RL, non-LLM baselines that implement the
``HasRecommend`` protocol.  All baselines rank *within* the candidate
set produced by JOB-06 (``get_candidates``).
"""

from src.baselines.random import RandomRecommender
from src.baselines.popularity import PopularityRecommender
from src.baselines.category_match import CategoryMatchRecommender
from src.baselines.quality_weighted import WorkerQualityWeightedRecommender

__all__ = [
    "RandomRecommender",
    "PopularityRecommender",
    "CategoryMatchRecommender",
    "WorkerQualityWeightedRecommender",
]
