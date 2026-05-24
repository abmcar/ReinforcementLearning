"""Sanity tests for evaluation metrics and the evaluate() pipeline.

Uses hand-crafted data only — no dependency on real data files.
"""

from __future__ import annotations

import math
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Set
from unittest.mock import patch

import pytest

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.eval.metrics import (
    avg_award_value_at_k,
    avg_recommender_worker_quality,
    category_match_rate_at_k,
    finalist_rate_at_k,
    hit_rate_at_k,
    mrr,
    ndcg_at_k,
    precision_at_k,
    project_coverage,
    recall_at_k,
    winner_rate_at_k,
)
from src.eval.protocols import HasRecommend

# ===================================================================
# Generic ranking metrics
# ===================================================================


class TestHitRate:
    def test_hit_at_rank_1(self):
        assert hit_rate_at_k([10, 20, 30], 10, 1) == 1.0

    def test_miss_at_rank_1(self):
        assert hit_rate_at_k([10, 20, 30], 20, 1) == 0.0

    def test_hit_at_rank_3(self):
        assert hit_rate_at_k([10, 20, 30], 30, 3) == 1.0

    def test_miss_beyond_k(self):
        assert hit_rate_at_k([10, 20, 30, 40], 40, 3) == 0.0


class TestNDCG:
    def test_perfect_rank(self):
        # Ground truth at rank 1 -> NDCG = 1 / log2(2) = 1.0
        assert ndcg_at_k([10, 20, 30], 10, 1) == pytest.approx(1.0)

    def test_rank_2(self):
        # Ground truth at rank 2 -> 1 / log2(3)
        expected = 1.0 / math.log2(3)
        assert ndcg_at_k([10, 20, 30], 20, 3) == pytest.approx(expected)

    def test_rank_3(self):
        expected = 1.0 / math.log2(4)
        assert ndcg_at_k([10, 20, 30], 30, 3) == pytest.approx(expected)

    def test_miss(self):
        assert ndcg_at_k([10, 20, 30], 99, 3) == 0.0


class TestMRR:
    def test_rank_1(self):
        assert mrr([10, 20, 30], 10) == pytest.approx(1.0)

    def test_rank_2(self):
        assert mrr([10, 20, 30], 20) == pytest.approx(0.5)

    def test_rank_3(self):
        assert mrr([10, 20, 30], 30) == pytest.approx(1.0 / 3)

    def test_miss(self):
        assert mrr([10, 20, 30], 99) == 0.0


class TestPrecisionRecall:
    def test_precision_single_relevant(self):
        assert precision_at_k([1, 2, 3], {2}, 3) == pytest.approx(1.0 / 3)

    def test_precision_all_relevant(self):
        assert precision_at_k([1, 2, 3], {1, 2, 3}, 3) == pytest.approx(1.0)

    def test_recall_single(self):
        assert recall_at_k([1, 2, 3], {2}, 3) == pytest.approx(1.0)

    def test_recall_partial(self):
        assert recall_at_k([1, 2, 3], {2, 4}, 3) == pytest.approx(0.5)

    def test_recall_empty_gt(self):
        assert recall_at_k([1, 2], set(), 2) == 0.0


# ===================================================================
# Worker-objective metrics
# ===================================================================


class TestWorkerMetrics:
    def test_avg_award(self):
        awards = {1: 100.0, 2: 200.0, 3: 300.0}
        result = avg_award_value_at_k([1, 2, 3], awards, 2)
        assert result == pytest.approx(150.0)

    def test_finalist_rate(self):
        finalist_set = {1, 3}
        result = finalist_rate_at_k([1, 2, 3], finalist_set, 3)
        assert result == pytest.approx(2.0 / 3)

    def test_winner_rate(self):
        winner_set = {2}
        result = winner_rate_at_k([1, 2, 3], winner_set, 3)
        assert result == pytest.approx(1.0 / 3)

    def test_category_match(self):
        proj_cat = {1: 10, 2: 20, 3: 10}
        worker_cats = {10}
        result = category_match_rate_at_k([1, 2, 3], proj_cat, worker_cats, 3)
        assert result == pytest.approx(2.0 / 3)


# ===================================================================
# Requester-objective metrics
# ===================================================================


class TestRequesterMetrics:
    def test_worker_quality(self):
        project_workers = {1: [10, 20], 2: [30]}
        wq = {10: 0.8, 20: 0.6, 30: 1.0}
        result = avg_recommender_worker_quality(project_workers, wq)
        # project 1: mean(0.8, 0.6) = 0.7
        # project 2: mean(1.0) = 1.0
        # macro avg = (0.7 + 1.0) / 2 = 0.85
        assert result == pytest.approx(0.85)

    def test_project_coverage_full(self):
        assert project_coverage({1, 2, 3}, {1, 2, 3}) == pytest.approx(1.0)

    def test_project_coverage_partial(self):
        assert project_coverage({1}, {1, 2, 3}) == pytest.approx(1.0 / 3)

    def test_project_coverage_empty(self):
        assert project_coverage(set(), set()) == 0.0


# ===================================================================
# HasRecommend Protocol
# ===================================================================


class TestProtocol:
    def test_dummy_satisfies_protocol(self):
        class Dummy:
            def recommend(
                self, worker_id: int, timestamp: datetime, candidates: List[int]
            ) -> List[int]:
                return candidates

        assert isinstance(Dummy(), HasRecommend)

    def test_non_conforming(self):
        class Bad:
            pass

        assert not isinstance(Bad(), HasRecommend)


# ===================================================================
# End-to-end smoke test with mock data
# ===================================================================


class RandomRecommender:
    """A dummy random recommender for testing."""

    def __init__(self, seed: int = 42):
        import random as _random
        self._rng = _random.Random(seed)

    def recommend(
        self, worker_id: int, timestamp: datetime, candidates: List[int]
    ) -> List[int]:
        shuffled = list(candidates)
        self._rng.shuffle(shuffled)
        return shuffled


class PerfectRecommender:
    """Always puts ground truth first (cheats by looking at candidates)."""

    def __init__(self, ground_truth_map: Dict[int, int]):
        self._gt = ground_truth_map
        self._call_idx = 0
        self._call_order: List[int] = []

    def set_call_order(self, entries):
        self._call_order = [(e.worker_id, e.project_id) for e in entries]
        self._call_idx = 0

    def recommend(
        self, worker_id: int, timestamp: datetime, candidates: List[int]
    ) -> List[int]:
        if self._call_idx < len(self._call_order):
            _, gt_pid = self._call_order[self._call_idx]
            self._call_idx += 1
        else:
            gt_pid = None

        if gt_pid and gt_pid in candidates:
            rest = [c for c in candidates if c != gt_pid]
            return [gt_pid] + rest
        return list(candidates)


def _make_test_entries():
    """Create a small set of hand-crafted entries for testing."""
    from src.data.split import Entry, EntryList

    base_time = datetime(2020, 1, 1)
    entries = [
        Entry(project_id=1, entry_number=1, worker_id=100,
              entry_created_at=base_time + timedelta(days=i),
              award_value=50.0 * (i + 1), finalist=(i % 2 == 0), winner=(i == 0))
        for i in range(5)
    ] + [
        Entry(project_id=2, entry_number=1, worker_id=200,
              entry_created_at=base_time + timedelta(days=5 + i),
              award_value=100.0, finalist=True, winner=False)
        for i in range(3)
    ] + [
        Entry(project_id=3, entry_number=1, worker_id=100,
              entry_created_at=base_time + timedelta(days=8 + i),
              award_value=75.0, finalist=False, winner=(i == 1))
        for i in range(2)
    ]
    time_range = (entries[0].entry_created_at, entries[-1].entry_created_at)
    return EntryList(entries=entries, time_range=time_range)


def _make_test_project_info():
    base_time = datetime(2019, 12, 1)
    return {
        1: {
            "category": 10, "sub_category": 1,
            "start_date": base_time,
            "deadline": datetime(2020, 6, 1),
            "total_awards": 500.0,
        },
        2: {
            "category": 20, "sub_category": 2,
            "start_date": base_time,
            "deadline": datetime(2020, 6, 1),
            "total_awards": 300.0,
        },
        3: {
            "category": 10, "sub_category": 3,
            "start_date": base_time,
            "deadline": datetime(2020, 6, 1),
            "total_awards": 200.0,
        },
    }


def _make_test_worker_quality():
    return {100: 0.85, 200: 0.72}


class TestEvaluateEndToEnd:
    """Smoke test: patch data loading and run evaluate() end-to-end."""

    def _patch_and_run(self, model, n_bootstrap=0):
        """Run evaluate with mocked data sources."""
        test_entries = _make_test_entries()
        project_info = _make_test_project_info()
        wq = _make_test_worker_quality()

        with patch("src.eval.evaluator.load_split", return_value=test_entries), \
             patch("src.eval.evaluator._load_project_info", return_value=project_info), \
             patch("src.eval.evaluator._load_worker_quality", return_value=(wq, 0.785)):
            from src.eval.evaluator import evaluate
            return evaluate(model, "test", n_bootstrap=n_bootstrap)

    def test_random_recommender_returns_all_metrics(self):
        model = RandomRecommender(seed=123)
        results = self._patch_and_run(model)

        # Check that all expected metric keys are present
        expected_keys = set()
        for k in [1, 5, 10]:
            expected_keys |= {
                f"HR@{k}", f"NDCG@{k}", f"Precision@{k}", f"Recall@{k}",
                f"avg_award_value@{k}", f"finalist_rate@{k}",
                f"winner_rate@{k}", f"category_match_rate@{k}",
            }
        expected_keys.add("MRR")
        expected_keys.add("avg_recommender_worker_quality")
        expected_keys.add("project_coverage")

        missing = expected_keys - set(results.keys())
        assert not missing, f"Missing metrics: {missing}"

        # All values should be floats in [0, ...)
        for key, val in results.items():
            assert isinstance(val, float), f"{key} is not float"

    def test_perfect_recommender_scores_high(self):
        test_entries = _make_test_entries()
        gt_map = {e.worker_id: e.project_id for e in test_entries.entries}
        model = PerfectRecommender(gt_map)
        model.set_call_order(test_entries.entries)

        project_info = _make_test_project_info()
        wq = _make_test_worker_quality()

        with patch("src.eval.evaluator.load_split", return_value=test_entries), \
             patch("src.eval.evaluator._load_project_info", return_value=project_info), \
             patch("src.eval.evaluator._load_worker_quality", return_value=(wq, 0.785)):
            from src.eval.evaluator import evaluate
            results = evaluate(model, "test")

        # Perfect recommender should achieve HR@10 = 1.0
        assert results["HR@10"] == pytest.approx(1.0)
        assert results["HR@1"] == pytest.approx(1.0)
        assert results["NDCG@1"] == pytest.approx(1.0)
        assert results["MRR"] == pytest.approx(1.0)

    def test_bootstrap_ci_keys(self):
        model = RandomRecommender(seed=99)
        results = self._patch_and_run(model, n_bootstrap=50)

        # CI keys should exist for per-entry metrics
        assert "HR@10_ci_lower" in results
        assert "HR@10_ci_upper" in results
        assert results["HR@10_ci_lower"] <= results["HR@10"]
        assert results["HR@10_ci_upper"] >= results["HR@10"]
