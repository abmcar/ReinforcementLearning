"""Candidate generator — main entry point (JOB-06).

Combines multiple recall strategies into a unified ``get_candidates()``
interface shared by both DQN and LLM pipelines.

Interface contract (see ``docs/roadmap.md`` section 6.5):
    get_candidates(worker_id: int, timestamp: datetime, K: int = 50) -> list[int]
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set

from dateutil.parser import parse as parse_dt

from src.candidates.recall import (
    category_recall,
    get_active_projects,
    load_entry_history,
    load_project_metadata,
    popularity_recall,
)

# ── configurable weights for multi-path merge ──────────────────────────
# Popularity weight vs category-match weight.
# These can be tuned; current defaults favour popularity (0.7) with a
# category affinity bonus (0.3).
DEFAULT_POPULARITY_WEIGHT = 0.7
DEFAULT_CATEGORY_WEIGHT = 0.3
DEFAULT_RECENCY_DAYS = 60


class CandidateGenerator:
    """Stateful candidate generator that pre-loads data once for efficiency.

    Usage:
        gen = CandidateGenerator()
        candidates = gen.get_candidates(worker_id=42, timestamp=dt, K=50)
    """

    def __init__(
        self,
        popularity_weight: float = DEFAULT_POPULARITY_WEIGHT,
        category_weight: float = DEFAULT_CATEGORY_WEIGHT,
        recency_days: int = DEFAULT_RECENCY_DAYS,
    ):
        self.popularity_weight = popularity_weight
        self.category_weight = category_weight
        self.recency_days = recency_days

        # Pre-load shared data
        self.project_meta = load_project_metadata()
        self.entry_history = load_entry_history()

        # Build a set of known worker IDs (from training history)
        self._known_workers: Set[int] = set()
        for entry in self.entry_history:
            self._known_workers.add(entry["worker_id"])

    def _is_cold_start_worker(self, worker_id: int, timestamp: datetime) -> bool:
        """Check if worker has zero history strictly before *timestamp*."""
        for entry in self.entry_history:
            t = entry.get("_parsed_ts") or parse_dt(entry["entry_created_at"])
            if t >= timestamp:
                break
            if entry["worker_id"] == worker_id:
                return False
        return True

    def get_candidates(
        self,
        worker_id: int,
        timestamp: datetime,
        K: int = 50,
    ) -> List[int]:
        """Return Top-K candidate project IDs for a worker at a given time.

        Parameters
        ----------
        worker_id : int
            The worker requesting recommendations.
        timestamp : datetime
            The point in time for the recommendation. Only data
            **at or before** this timestamp is used (anti-leakage).
        K : int, default 50
            Maximum number of candidates to return.

        Returns
        -------
        list[int]
            Project IDs sorted descending by recall score. May be shorter
            than K if fewer active projects exist, or empty if no projects
            are active at *timestamp*.
        """
        # 1. Filter active projects (start_date <= t <= deadline)
        active = get_active_projects(self.project_meta, timestamp)

        # Empty active set → return empty list (data_split.md § 2.3)
        if not active:
            return []

        # 2. Popularity recall (always computed)
        pop_scores = popularity_recall(
            self.entry_history, active, timestamp, self.recency_days
        )
        pop_dict: Dict[int, float] = {pid: score for pid, score in pop_scores}

        # Normalise popularity scores to [0, 1]
        max_pop = max((s for _, s in pop_scores), default=1.0)
        if max_pop > 0:
            pop_dict = {pid: s / max_pop for pid, s in pop_dict.items()}

        # 3. Category recall (skipped for cold-start workers)
        is_cold = self._is_cold_start_worker(worker_id, timestamp)

        if is_cold:
            # Cold-start fallback: pure global popularity
            merged: Dict[int, float] = pop_dict
        else:
            cat_scores = category_recall(
                self.entry_history,
                self.project_meta,
                active,
                worker_id,
                timestamp,
            )
            cat_dict: Dict[int, float] = {pid: score for pid, score in cat_scores}

            # Normalise category scores to [0, 1]
            max_cat = max((s for _, s in cat_scores), default=1.0)
            if max_cat > 0:
                cat_dict = {pid: s / max_cat for pid, s in cat_dict.items()}

            # Weighted merge
            merged = {}
            for pid in active:
                merged[pid] = (
                    self.popularity_weight * pop_dict.get(pid, 0.0)
                    + self.category_weight * cat_dict.get(pid, 0.0)
                )

        # 4. Sort by merged score descending, take Top-K
        ranked = sorted(merged.items(), key=lambda x: x[1], reverse=True)
        result = [pid for pid, _ in ranked[:K]]

        return result


# ── module-level singleton for convenience ──────────────────────────────
_generator: Optional[CandidateGenerator] = None


def _get_generator() -> CandidateGenerator:
    global _generator
    if _generator is None:
        _generator = CandidateGenerator()
    return _generator


def get_candidates(
    worker_id: int,
    timestamp: datetime,
    K: int = 50,
) -> List[int]:
    """Module-level convenience wrapper.

    Signature matches ``docs/roadmap.md`` section 6.5:
        get_candidates(worker_id: int, timestamp: datetime, K: int = 50) -> list[int]
    """
    # Anti-leakage assertion: timestamp must be a proper datetime
    assert isinstance(timestamp, datetime), (
        f"timestamp must be datetime, got {type(timestamp)}"
    )
    return _get_generator().get_candidates(worker_id, timestamp, K)
