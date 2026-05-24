"""Random baseline recommender.

Randomly shuffles the candidate set — serves as the lower bound
for all recommendation methods.
"""

from __future__ import annotations

import random as _random
from datetime import datetime
from typing import List


class RandomRecommender:
    """Recommend by randomly shuffling candidates (uniform random policy).

    This is the simplest possible baseline and establishes the lower
    bound that any meaningful model should beat.

    Parameters
    ----------
    seed : int
        Random seed for reproducibility.
    """

    def __init__(self, seed: int = 42):
        self._rng = _random.Random(seed)

    def recommend(
        self,
        worker_id: int,
        timestamp: datetime,
        candidates: List[int],
    ) -> List[int]:
        """Return candidates in random order."""
        shuffled = list(candidates)
        self._rng.shuffle(shuffled)
        return shuffled
