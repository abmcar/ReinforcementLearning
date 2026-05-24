"""Worker-quality-weighted popularity baseline.

Designed primarily for the *requester* objective: recommend projects
where high-quality workers are matched to high-value opportunities.

Scoring logic:
    For a given worker with quality q (normalised to [0, 1]):
    - High-quality workers (q >= median) see a blended score:
      score = (1 - alpha) * popularity + alpha * avg_award
      This steers them towards high-value projects.
    - Lower-quality workers see pure popularity ranking (fallback).

This ensures the ranking differs from plain PopularityRecommender
for workers with quality information.
"""

from __future__ import annotations

import bisect
from datetime import datetime, timedelta
from typing import Dict, List


class WorkerQualityWeightedRecommender:
    """Rank candidates by popularity weighted with worker quality and
    project award density.

    For requester objective: steer high-quality workers to high-value
    projects.

    Parameters
    ----------
    recency_days : int
        Look-back window for popularity counts (default 60).
    alpha : float
        Weight of award-density component for high-quality workers
        (default 0.5).
    shared_data : optional
        Pre-loaded SharedData instance.  If None, loads data itself.
    """

    def __init__(self, recency_days: int = 60, alpha: float = 0.5, shared_data=None):
        self.recency_days = recency_days
        self.alpha = alpha
        if shared_data is not None:
            self._entry_history = shared_data.entry_history
            self._project_meta = shared_data.project_meta
            self._worker_quality = shared_data.worker_quality
            self._wq_median = shared_data.wq_median
        else:
            from src.baselines.data_loader import SharedData
            data = SharedData.get()
            self._entry_history = data.entry_history
            self._project_meta = data.project_meta
            self._worker_quality = data.worker_quality
            self._wq_median = data.wq_median
        # Pre-extract timestamps for binary search
        self._timestamps = [e["_parsed_ts"] for e in self._entry_history]
        # Pre-compute average award snapshot for efficiency.
        # Computing per call is O(N); since we only need relative ranking
        # and avg award changes slowly, pre-compute once and reuse.
        self._precomputed_avg_award = self._precompute_avg_award()

    @staticmethod
    def _load_worker_quality():
        from src.baselines.data_loader import SharedData
        data = SharedData.get()
        return data.worker_quality, data.wq_median

    def _get_popularity_scores(
        self,
        timestamp: datetime,
    ) -> Dict[int, float]:
        """Count entries per project in [timestamp - window, timestamp)."""
        cutoff = timestamp - timedelta(days=self.recency_days)
        lo = bisect.bisect_left(self._timestamps, cutoff)
        hi = bisect.bisect_left(self._timestamps, timestamp)
        counts: Dict[int, int] = {}
        for i in range(lo, hi):
            pid = self._entry_history[i]["project_id"]
            counts[pid] = counts.get(pid, 0) + 1
        return {pid: float(c) for pid, c in counts.items()}

    def _precompute_avg_award(self) -> Dict[int, float]:
        """Pre-compute per-project average award from all history.

        Since average award changes slowly over time and is used only
        for relative ranking among candidates, a single pre-computed
        snapshot is sufficient and avoids O(N) per-call overhead.
        """
        project_award_sums: Dict[int, float] = {}
        project_counts: Dict[int, int] = {}
        for e in self._entry_history:
            pid = e["project_id"]
            project_counts[pid] = project_counts.get(pid, 0) + 1
            project_award_sums[pid] = (
                project_award_sums.get(pid, 0.0) + e.get("award_value", 0.0)
            )

        avg_award: Dict[int, float] = {}
        for pid, cnt in project_counts.items():
            total = project_award_sums.get(pid, 0.0)
            avg_award[pid] = total / cnt if cnt > 0 else 0.0
        return avg_award

    def recommend(
        self,
        worker_id: int,
        timestamp: datetime,
        candidates: List[int],
    ) -> List[int]:
        """Rank candidates by quality-weighted popularity."""
        pop_scores = self._get_popularity_scores(timestamp)
        wq = self._worker_quality.get(worker_id, self._wq_median)

        # Normalise popularity to [0, 1]
        max_pop = max(pop_scores.values(), default=1.0)
        if max_pop > 0:
            norm_pop = {pid: s / max_pop for pid, s in pop_scores.items()}
        else:
            norm_pop = pop_scores

        if wq >= self._wq_median:
            # High-quality worker: blend popularity with average award
            avg_award = self._precomputed_avg_award
            max_aa = max(avg_award.values(), default=1.0)
            if max_aa > 0:
                norm_aa = {pid: a / max_aa for pid, a in avg_award.items()}
            else:
                norm_aa = avg_award

            def score(pid: int) -> float:
                p = norm_pop.get(pid, 0.0)
                a = norm_aa.get(pid, 0.0)
                return (1 - self.alpha) * p + self.alpha * a
        else:
            # Lower-quality worker: pure popularity
            def score(pid: int) -> float:
                return norm_pop.get(pid, 0.0)

        return sorted(
            candidates,
            key=lambda pid: (score(pid), -pid),
            reverse=True,
        )
