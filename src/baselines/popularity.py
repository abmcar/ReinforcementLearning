"""Popularity baseline recommender.

Ranks candidates by global popularity (recent entry count).
"""

from __future__ import annotations

import bisect
from datetime import datetime, timedelta
from typing import Dict, List, Optional


class PopularityRecommender:
    """Rank candidates by global popularity (entry count in a recent window).

    For each candidate project, score = number of entries submitted
    *strictly before* ``timestamp`` and within the last ``recency_days``
    days.  Ties are broken by project ID (deterministic).

    Parameters
    ----------
    recency_days : int
        Look-back window in days (default 60).
    shared_data : optional
        Pre-loaded SharedData instance.  If None, loads data itself.
    """

    def __init__(self, recency_days: int = 60, shared_data=None):
        self.recency_days = recency_days
        if shared_data is not None:
            self._entry_history = shared_data.entry_history
        else:
            self._entry_history = self._load_entry_history()
        # Pre-extract timestamps for binary search
        self._timestamps = [e["_parsed_ts"] for e in self._entry_history]

    @staticmethod
    def _load_entry_history() -> list[dict]:
        from src.baselines.data_loader import SharedData
        return SharedData.get().entry_history

    def _get_popularity_scores(
        self,
        timestamp: datetime,
    ) -> Dict[int, float]:
        """Count entries per project in [timestamp - window, timestamp)."""
        cutoff = timestamp - timedelta(days=self.recency_days)
        # Use binary search to find the range [cutoff, timestamp)
        lo = bisect.bisect_left(self._timestamps, cutoff)
        hi = bisect.bisect_left(self._timestamps, timestamp)
        counts: Dict[int, int] = {}
        for i in range(lo, hi):
            pid = self._entry_history[i]["project_id"]
            counts[pid] = counts.get(pid, 0) + 1
        return {pid: float(c) for pid, c in counts.items()}

    def recommend(
        self,
        worker_id: int,
        timestamp: datetime,
        candidates: List[int],
    ) -> List[int]:
        """Rank candidates by global popularity, descending."""
        scores = self._get_popularity_scores(timestamp)
        return sorted(
            candidates,
            key=lambda pid: (scores.get(pid, 0.0), -pid),
            reverse=True,
        )
