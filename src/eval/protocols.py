"""HasRecommend Protocol for evaluation framework.

Any model (baseline, DQN, LLM) that implements `recommend()` can be passed
to `evaluate()`.  Duck typing via `typing.Protocol`.
"""

from __future__ import annotations

from datetime import datetime
from typing import List, Protocol, runtime_checkable


@runtime_checkable
class HasRecommend(Protocol):
    """Protocol every recommender must satisfy.

    Parameters
    ----------
    worker_id : int
        The worker requesting recommendations.
    timestamp : datetime
        Current point in time (for time-aware filtering).
    candidates : list[int]
        Candidate project IDs to rank.

    Returns
    -------
    list[int]
        Project IDs sorted by predicted relevance (best first).
    """

    def recommend(
        self,
        worker_id: int,
        timestamp: datetime,
        candidates: List[int],
    ) -> List[int]: ...
