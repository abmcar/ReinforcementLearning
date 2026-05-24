"""Worker-quality-weighted popularity baseline.

Designed primarily for the *requester* objective: recommend projects
where high-quality workers are matched to high-value opportunities.

Scoring logic:
    For a given worker with quality q (normalised to [0, 1]):
    - High-quality workers (q >= median) see a blended score:
      score = (1 - alpha) * popularity + alpha * award_density
      This steers them towards high-value projects (higher avg award).
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
        # Pre-compute cumulative award density snapshot for efficiency.
        # _get_award_density is expensive (O(N) per call).  Since we only
        # need density for ranking relative comparison and it changes slowly,
        # we pre-compute it once for the full history and reuse.
        self._precomputed_density = self._precompute_award_density()

    @staticmethod
    def _load_entry_history() -> list[dict]:
        from src.baselines.data_loader import SharedData
        return SharedData.get().entry_history

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

    def _precompute_award_density(self) -> Dict[int, float]:
        """Pre-compute per-project award density from all history.

        Since award density changes slowly over time and is used only
        for relative ranking among candidates, a single pre-computed
        snapshot is sufficient and avoids O(N) per-call overhead.
        """
        project_awards: Dict[int, float] = {}
        project_counts: Dict[int, int] = {}
        for e in self._entry_history:
            pid = e["project_id"]
            project_counts[pid] = project_counts.get(pid, 0) + 1
            prev = project_awards.get(pid, 0.0)
            project_awards[pid] = max(prev, e.get("award_value", 0.0))

        density: Dict[int, float] = {}
        for pid, cnt in project_counts.items():
            award = project_awards.get(pid, 0.0)
            density[pid] = award / cnt if cnt > 0 else 0.0
        return density

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
            # High-quality worker: blend popularity with award density
            density = self._precomputed_density
            max_den = max(density.values(), default=1.0)
            if max_den > 0:
                norm_den = {pid: d / max_den for pid, d in density.items()}
            else:
                norm_den = density

            def score(pid: int) -> float:
                p = norm_pop.get(pid, 0.0)
                d = norm_den.get(pid, 0.0)
                return (1 - self.alpha) * p + self.alpha * d
        else:
            # Lower-quality worker: pure popularity
            def score(pid: int) -> float:
                return norm_pop.get(pid, 0.0)

        return sorted(
            candidates,
            key=lambda pid: (score(pid), -pid),
            reverse=True,
        )
