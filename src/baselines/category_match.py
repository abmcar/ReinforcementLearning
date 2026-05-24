"""Category-match baseline recommender.

Ranks candidates by how well the project's category matches the
worker's historically most active categories.
"""

from __future__ import annotations

import bisect
from collections import defaultdict
from datetime import datetime
from typing import Dict, List


class CategoryMatchRecommender:
    """Rank candidates by category affinity to the worker's history.

    For each worker, build a histogram of categories they have
    historically participated in (strictly before ``timestamp``).
    Then score each candidate project by how many times the worker
    previously engaged in projects of the same category.

    Ties are broken by project ID (deterministic).

    Parameters
    ----------
    shared_data : optional
        Pre-loaded SharedData instance.  If None, loads data itself.
    """

    def __init__(self, shared_data=None):
        if shared_data is not None:
            self._entry_history = shared_data.entry_history
            self._project_meta = shared_data.project_meta
        else:
            self._entry_history = self._load_entry_history()
            from src.candidates.recall import load_project_metadata
            self._project_meta = load_project_metadata()
        # Pre-extract timestamps for binary search
        self._timestamps = [e["_parsed_ts"] for e in self._entry_history]
        # Pre-build worker -> sorted entry indices
        self._worker_entries: Dict[int, List[int]] = defaultdict(list)
        for i, e in enumerate(self._entry_history):
            self._worker_entries[e["worker_id"]].append(i)

    @staticmethod
    def _load_entry_history() -> list[dict]:
        from src.baselines.data_loader import SharedData
        return SharedData.get().entry_history

    def _get_worker_category_histogram(
        self,
        worker_id: int,
        timestamp: datetime,
    ) -> Dict[int, int]:
        """Build category participation counts for a worker before timestamp."""
        cat_counts: Dict[int, int] = {}
        for idx in self._worker_entries.get(worker_id, []):
            if self._timestamps[idx] >= timestamp:
                break  # indices are sorted chronologically
            pid = self._entry_history[idx]["project_id"]
            cat = self._project_meta.get(pid, {}).get("category", -1)
            if cat >= 0:
                cat_counts[cat] = cat_counts.get(cat, 0) + 1
        return cat_counts

    def recommend(
        self,
        worker_id: int,
        timestamp: datetime,
        candidates: List[int],
    ) -> List[int]:
        """Rank candidates by category affinity, descending."""
        cat_counts = self._get_worker_category_histogram(worker_id, timestamp)

        def score(pid: int) -> float:
            cat = self._project_meta.get(pid, {}).get("category", -1)
            return float(cat_counts.get(cat, 0))

        return sorted(
            candidates,
            key=lambda pid: (score(pid), -pid),
            reverse=True,
        )
