"""Smoke tests for baseline recommenders (JOB-05).

Verifies Protocol conformance and basic interface correctness using
hand-crafted mock data — no dependency on real data files.
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List
from unittest.mock import patch, MagicMock

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.eval.protocols import HasRecommend


# ---------------------------------------------------------------------------
# Mock data fixtures
# ---------------------------------------------------------------------------

MOCK_ENTRY_HISTORY = [
    {
        "project_id": 1,
        "worker_id": 100,
        "entry_created_at": "2020-01-01T00:00:00",
        "award_value": 50.0,
        "finalist": True,
        "winner": False,
        "_parsed_ts": datetime(2020, 1, 1),
    },
    {
        "project_id": 2,
        "worker_id": 100,
        "entry_created_at": "2020-01-05T00:00:00",
        "award_value": 100.0,
        "finalist": False,
        "winner": False,
        "_parsed_ts": datetime(2020, 1, 5),
    },
    {
        "project_id": 1,
        "worker_id": 200,
        "entry_created_at": "2020-01-10T00:00:00",
        "award_value": 75.0,
        "finalist": True,
        "winner": True,
        "_parsed_ts": datetime(2020, 1, 10),
    },
    {
        "project_id": 3,
        "worker_id": 100,
        "entry_created_at": "2020-01-15T00:00:00",
        "award_value": 200.0,
        "finalist": False,
        "winner": False,
        "_parsed_ts": datetime(2020, 1, 15),
    },
    {
        "project_id": 2,
        "worker_id": 200,
        "entry_created_at": "2020-01-20T00:00:00",
        "award_value": 150.0,
        "finalist": True,
        "winner": False,
        "_parsed_ts": datetime(2020, 1, 20),
    },
]

MOCK_PROJECT_META = {
    1: {
        "start_date": datetime(2019, 12, 1),
        "deadline": datetime(2020, 6, 1),
        "category": 10,
        "sub_category": 1,
    },
    2: {
        "start_date": datetime(2019, 12, 1),
        "deadline": datetime(2020, 6, 1),
        "category": 20,
        "sub_category": 2,
    },
    3: {
        "start_date": datetime(2019, 12, 1),
        "deadline": datetime(2020, 6, 1),
        "category": 10,
        "sub_category": 3,
    },
}

MOCK_SPLIT_CACHE = {
    "train": MOCK_ENTRY_HISTORY[:3],
    "val": MOCK_ENTRY_HISTORY[3:4],
    "test": MOCK_ENTRY_HISTORY[4:],
}

MOCK_WORKER_QUALITY_CSV = "worker_id,worker_quality\n100,80\n200,60\n"


CANDIDATES = [1, 2, 3]
TIMESTAMP = datetime(2020, 2, 1)


# ---------------------------------------------------------------------------
# Test: Protocol conformance
# ---------------------------------------------------------------------------


class TestProtocolConformance:
    """Every baseline must satisfy HasRecommend."""

    def test_random_satisfies_protocol(self):
        from src.baselines.random import RandomRecommender

        model = RandomRecommender()
        assert isinstance(model, HasRecommend)

    def test_popularity_satisfies_protocol(self):
        with patch(
            "src.baselines.popularity.PopularityRecommender._load_entry_history",
            return_value=MOCK_ENTRY_HISTORY,
        ):
            from src.baselines.popularity import PopularityRecommender

            model = PopularityRecommender(recency_days=60)
        assert isinstance(model, HasRecommend)

    def test_category_match_satisfies_protocol(self):
        mock_data = MagicMock()
        mock_data.entry_history = MOCK_ENTRY_HISTORY
        mock_data.project_meta = MOCK_PROJECT_META
        from src.baselines.category_match import CategoryMatchRecommender

        model = CategoryMatchRecommender(shared_data=mock_data)
        assert isinstance(model, HasRecommend)

    def test_quality_weighted_satisfies_protocol(self):
        mock_data = MagicMock()
        mock_data.entry_history = MOCK_ENTRY_HISTORY
        mock_data.project_meta = MOCK_PROJECT_META
        mock_data.worker_quality = {100: 0.8, 200: 0.6}
        mock_data.wq_median = 0.7
        from src.baselines.quality_weighted import (
            WorkerQualityWeightedRecommender,
        )

        model = WorkerQualityWeightedRecommender(
            recency_days=60, alpha=0.5, shared_data=mock_data
        )
        assert isinstance(model, HasRecommend)

    def test_all_return_list_of_ints(self):
        """Each recommend() should return a list[int] of same length."""
        from src.baselines.random import RandomRecommender

        model = RandomRecommender()
        result = model.recommend(100, TIMESTAMP, CANDIDATES)
        assert isinstance(result, list)
        assert all(isinstance(x, int) for x in result)
        assert set(result) == set(CANDIDATES)


# ---------------------------------------------------------------------------
# Test: RandomRecommender
# ---------------------------------------------------------------------------


class TestRandomRecommender:
    def test_deterministic_with_seed(self):
        from src.baselines.random import RandomRecommender

        m1 = RandomRecommender(seed=123)
        m2 = RandomRecommender(seed=123)
        r1 = m1.recommend(100, TIMESTAMP, list(range(20)))
        r2 = m2.recommend(100, TIMESTAMP, list(range(20)))
        assert r1 == r2

    def test_different_seeds_differ(self):
        from src.baselines.random import RandomRecommender

        m1 = RandomRecommender(seed=1)
        m2 = RandomRecommender(seed=2)
        r1 = m1.recommend(100, TIMESTAMP, list(range(100)))
        r2 = m2.recommend(100, TIMESTAMP, list(range(100)))
        assert r1 != r2

    def test_preserves_candidates(self):
        from src.baselines.random import RandomRecommender

        m = RandomRecommender()
        cands = [10, 20, 30, 40, 50]
        result = m.recommend(100, TIMESTAMP, cands)
        assert sorted(result) == sorted(cands)


# ---------------------------------------------------------------------------
# Test: PopularityRecommender
# ---------------------------------------------------------------------------


class TestPopularityRecommender:
    def _make(self):
        """Build with mocked data."""
        with patch(
            "src.baselines.popularity.PopularityRecommender._load_entry_history",
            return_value=MOCK_ENTRY_HISTORY,
        ):
            from src.baselines.popularity import PopularityRecommender

            return PopularityRecommender(recency_days=60)

    def test_interface(self):
        model = self._make()
        assert isinstance(model, HasRecommend)
        result = model.recommend(100, TIMESTAMP, CANDIDATES)
        assert isinstance(result, list)
        assert set(result) == set(CANDIDATES)

    def test_popular_first(self):
        """Project 1 has 2 entries, project 2 has 2, project 3 has 1 before TIMESTAMP."""
        model = self._make()
        result = model.recommend(100, TIMESTAMP, CANDIDATES)
        # Project 1 and 2 each have 2 entries, project 3 has 1
        # Tie broken by -pid (higher pid first in tie)
        assert result[-1] == 3  # least popular should be last


# ---------------------------------------------------------------------------
# Test: CategoryMatchRecommender
# ---------------------------------------------------------------------------


class TestCategoryMatchRecommender:
    def _make(self):
        """Build CategoryMatchRecommender with mock shared_data."""
        mock_data = MagicMock()
        mock_data.entry_history = MOCK_ENTRY_HISTORY
        mock_data.project_meta = MOCK_PROJECT_META
        from src.baselines.category_match import CategoryMatchRecommender
        return CategoryMatchRecommender(shared_data=mock_data)

    def test_interface(self):
        model = self._make()
        assert isinstance(model, HasRecommend)
        result = model.recommend(100, TIMESTAMP, CANDIDATES)
        assert isinstance(result, list)
        assert set(result) == set(CANDIDATES)

    def test_category_preference(self):
        """Worker 100 has entries in projects 1 (cat=10), 2 (cat=20), 3 (cat=10).
        So cat 10 has count 2, cat 20 has count 1.
        Projects 1,3 (cat=10) should rank above project 2 (cat=20)."""
        model = self._make()
        result = model.recommend(100, TIMESTAMP, CANDIDATES)
        # cat 10 projects (1, 3) should come before cat 20 project (2)
        idx_1 = result.index(1)
        idx_2 = result.index(2)
        idx_3 = result.index(3)
        assert idx_1 < idx_2 and idx_3 < idx_2  # both cat-10 projects beat cat-20


# ---------------------------------------------------------------------------
# Test: WorkerQualityWeightedRecommender
# ---------------------------------------------------------------------------


class TestQualityWeightedRecommender:
    def _make(self):
        """Build WorkerQualityWeightedRecommender with mock shared_data."""
        mock_data = MagicMock()
        mock_data.entry_history = MOCK_ENTRY_HISTORY
        mock_data.project_meta = MOCK_PROJECT_META
        mock_data.worker_quality = {100: 0.8, 200: 0.6}
        mock_data.wq_median = 0.7
        from src.baselines.quality_weighted import (
            WorkerQualityWeightedRecommender,
        )
        return WorkerQualityWeightedRecommender(
            recency_days=60, alpha=0.5, shared_data=mock_data
        )

    def test_interface(self):
        model = self._make()
        assert isinstance(model, HasRecommend)
        result = model.recommend(100, TIMESTAMP, CANDIDATES)
        assert isinstance(result, list)
        assert set(result) == set(CANDIDATES)

    def test_returns_all_candidates(self):
        model = self._make()
        cands = [1, 2, 3]
        result = model.recommend(100, TIMESTAMP, cands)
        assert len(result) == len(cands)

    def test_high_vs_low_quality_different_ranking(self):
        """High-quality worker (blended score) should produce a different
        ranking than low-quality worker (pure popularity)."""
        model = self._make()
        result_high = model.recommend(100, TIMESTAMP, CANDIDATES)  # q=0.8 >= median
        result_low = model.recommend(200, TIMESTAMP, CANDIDATES)   # q=0.6 < median
        assert result_high != result_low


# ---------------------------------------------------------------------------
# Test: Anti-leakage — future entries must not affect results
# ---------------------------------------------------------------------------


class TestAntiLeakage:
    """Future entries in entry_history must not leak into recommendations."""

    FUTURE_TS = datetime(2025, 1, 1)
    QUERY_TS = datetime(2020, 2, 1)

    def _make_history_with_future(self):
        """Return mock entry_history with a future entry appended (sorted)."""
        future_entry = {
            "project_id": 99,
            "worker_id": 100,
            "entry_created_at": "2025-01-01T00:00:00",
            "award_value": 9999.0,
            "finalist": True,
            "winner": True,
            "_parsed_ts": self.FUTURE_TS,
        }
        return MOCK_ENTRY_HISTORY + [future_entry]

    def test_popularity_ignores_future(self):
        """PopularityRecommender must not count future entries."""
        history = self._make_history_with_future()
        with patch(
            "src.baselines.popularity.PopularityRecommender._load_entry_history",
            return_value=history,
        ):
            from src.baselines.popularity import PopularityRecommender

            model = PopularityRecommender(recency_days=60)
        candidates = [1, 2, 3, 99]
        result = model.recommend(100, self.QUERY_TS, candidates)
        # project 99 has zero popularity before QUERY_TS, should be last
        assert result[-1] == 99

    def test_category_match_ignores_future(self):
        """CategoryMatchRecommender must not count future entries."""
        history = self._make_history_with_future()
        mock_meta = {**MOCK_PROJECT_META, 99: {
            "start_date": datetime(2024, 1, 1),
            "deadline": datetime(2025, 6, 1),
            "category": 10,
            "sub_category": 99,
        }}
        mock_data = MagicMock()
        mock_data.entry_history = history
        mock_data.project_meta = mock_meta
        from src.baselines.category_match import CategoryMatchRecommender

        model = CategoryMatchRecommender(shared_data=mock_data)
        # Worker 100's future entry for project 99 should not inflate cat-10 count
        hist_before = model._get_worker_category_histogram(100, self.QUERY_TS)
        hist_after = model._get_worker_category_histogram(
            100, datetime(2025, 6, 1)  # after the future entry
        )
        # Before QUERY_TS, cat-10 count should exclude the future entry
        assert hist_before.get(10, 0) < hist_after.get(10, 0)
