"""Adapter between JOB-06 get_candidates() and evaluate()'s candidate_fn.

evaluate() requires:
    candidate_fn(worker_id, timestamp, project_info, ground_truth_pid, k) -> list[int]

JOB-06 provides:
    get_candidates(worker_id, timestamp, K) -> list[int]

Design choice — ground-truth injection:
    We inject ground_truth_pid into the candidate set when it is missing.
    This matches evaluate()'s default_candidate_fn behaviour, ensuring that
    ranking metrics (HR@K, NDCG@K) are comparable across baselines, DQN,
    and LLM methods.  Without injection, a miss at the recall stage would
    always produce zero ranking scores, confounding recall-stage quality
    with ranking-stage quality.

Performance note:
    The standard CandidateGenerator.get_candidates() scans the full entry
    history per call, which is O(N) per invocation.  For 48k test entries
    this is ~48k * 490k = 24 billion comparisons.  To make baseline
    evaluation tractable, this module provides a cached adapter that
    uses the same recall logic but with pre-indexed data structures.
"""

from __future__ import annotations

import bisect
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Tuple

from src.candidates.recall import load_entry_history, load_project_metadata


class _FastCandidateGenerator:
    """Optimised candidate generator using pre-built indices.

    Same logic as CandidateGenerator but uses binary search on sorted
    timestamps instead of linear scans.
    """

    def __init__(
        self,
        popularity_weight: float = 0.7,
        category_weight: float = 0.3,
        recency_days: int = 60,
    ):
        self.popularity_weight = popularity_weight
        self.category_weight = category_weight
        self.recency_days = recency_days

        self.project_meta = load_project_metadata()
        self.entry_history = load_entry_history()

        # Pre-sort timestamps for binary search
        self._timestamps = [e["_parsed_ts"] for e in self.entry_history]

        # Pre-build worker -> set of entry indices for fast lookup
        self._worker_entries: Dict[int, List[int]] = defaultdict(list)
        for i, e in enumerate(self.entry_history):
            self._worker_entries[e["worker_id"]].append(i)

    def _get_active_projects(self, timestamp: datetime) -> Set[int]:
        active: Set[int] = set()
        for pid, info in self.project_meta.items():
            if info["start_date"] <= timestamp <= info["deadline"]:
                active.add(pid)
        return active

    def _is_cold_start(self, worker_id: int, timestamp: datetime) -> bool:
        indices = self._worker_entries.get(worker_id, [])
        for idx in indices:
            if self._timestamps[idx] < timestamp:
                return False
        return True

    def _popularity_scores(
        self, active: Set[int], timestamp: datetime
    ) -> Dict[int, float]:
        cutoff = timestamp - timedelta(days=self.recency_days)
        # Binary search for cutoff and timestamp positions
        lo = bisect.bisect_left(self._timestamps, cutoff)
        hi = bisect.bisect_left(self._timestamps, timestamp)
        counts: Dict[int, int] = {}
        for i in range(lo, hi):
            pid = self.entry_history[i]["project_id"]
            if pid in active:
                counts[pid] = counts.get(pid, 0) + 1
        return {pid: float(counts.get(pid, 0)) for pid in active}

    def _category_scores(
        self, active: Set[int], worker_id: int, timestamp: datetime
    ) -> Dict[int, float]:
        # Build worker's category histogram
        cat_counts: Dict[int, int] = {}
        for idx in self._worker_entries.get(worker_id, []):
            if self._timestamps[idx] >= timestamp:
                continue
            pid = self.entry_history[idx]["project_id"]
            cat = self.project_meta.get(pid, {}).get("category", -1)
            if cat >= 0:
                cat_counts[cat] = cat_counts.get(cat, 0) + 1
        # Score active projects
        scores: Dict[int, float] = {}
        for pid in active:
            cat = self.project_meta.get(pid, {}).get("category", -1)
            scores[pid] = float(cat_counts.get(cat, 0))
        return scores

    def get_candidates(
        self, worker_id: int, timestamp: datetime, K: int = 50
    ) -> List[int]:
        active = self._get_active_projects(timestamp)
        if not active:
            return []

        pop = self._popularity_scores(active, timestamp)
        max_pop = max(pop.values(), default=1.0)
        if max_pop > 0:
            pop = {pid: s / max_pop for pid, s in pop.items()}

        if self._is_cold_start(worker_id, timestamp):
            merged = pop
        else:
            cat = self._category_scores(active, worker_id, timestamp)
            max_cat = max(cat.values(), default=1.0)
            if max_cat > 0:
                cat = {pid: s / max_cat for pid, s in cat.items()}
            merged = {}
            for pid in active:
                merged[pid] = (
                    self.popularity_weight * pop.get(pid, 0.0)
                    + self.category_weight * cat.get(pid, 0.0)
                )

        ranked = sorted(merged.items(), key=lambda x: x[1], reverse=True)
        return [pid for pid, _ in ranked[:K]]


def make_candidate_fn(
    inject_ground_truth: bool = True,
    use_fast: bool = True,
):
    """Return a candidate_fn compatible with evaluate().

    Parameters
    ----------
    inject_ground_truth : bool
        If True (default), ensure ground_truth_pid is always in the
        returned candidate list.
    use_fast : bool
        If True (default), use the optimised generator with pre-built
        indices.  Otherwise use the standard CandidateGenerator.
    """
    if use_fast:
        gen = _FastCandidateGenerator()
    else:
        from src.candidates.generator import CandidateGenerator
        gen = CandidateGenerator()

    def candidate_fn(
        worker_id: int,
        timestamp: datetime,
        project_info: Dict[int, dict],
        ground_truth_pid: int,
        k: int = 50,
    ) -> List[int]:
        candidates = gen.get_candidates(worker_id, timestamp, K=k)

        if inject_ground_truth and ground_truth_pid not in candidates:
            if len(candidates) >= k:
                candidates[-1] = ground_truth_pid
            else:
                candidates.append(ground_truth_pid)

        return candidates

    return candidate_fn
