"""Smoke tests for the candidate generation module (JOB-06).

Tests verify:
    1. Return type and ordering contract
    2. Active-project filtering (deadline / start_date)
    3. Cold-start worker fallback
    4. Empty active set returns []
    5. Anti-leakage: only uses data <= timestamp
"""

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

import pytest

# Ensure src is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.candidates.recall import (
    get_active_projects,
    popularity_recall,
    category_recall,
)
from src.candidates.generator import CandidateGenerator


# ── fixtures: synthetic data ────────────────────────────────────────────

FAKE_PROJECT_META = {
    100: {
        "start_date": datetime(2010, 1, 1, tzinfo=timezone.utc),
        "deadline": datetime(2010, 2, 1, tzinfo=timezone.utc),
        "category": 1,
        "sub_category": 10,
    },
    200: {
        "start_date": datetime(2010, 1, 10, tzinfo=timezone.utc),
        "deadline": datetime(2010, 3, 1, tzinfo=timezone.utc),
        "category": 2,
        "sub_category": 20,
    },
    300: {
        "start_date": datetime(2010, 3, 1, tzinfo=timezone.utc),
        "deadline": datetime(2010, 4, 1, tzinfo=timezone.utc),
        "category": 1,
        "sub_category": 10,
    },
    400: {
        "start_date": datetime(2010, 1, 5, tzinfo=timezone.utc),
        "deadline": datetime(2010, 1, 10, tzinfo=timezone.utc),
        "category": 3,
        "sub_category": 30,
    },
}

FAKE_ENTRY_HISTORY = [
    {"project_id": 100, "worker_id": 1, "entry_created_at": "2010-01-05T00:00:00Z"},
    {"project_id": 100, "worker_id": 1, "entry_created_at": "2010-01-06T00:00:00Z"},
    {"project_id": 100, "worker_id": 2, "entry_created_at": "2010-01-07T00:00:00Z"},
    {"project_id": 200, "worker_id": 1, "entry_created_at": "2010-01-15T00:00:00Z"},
    {"project_id": 200, "worker_id": 3, "entry_created_at": "2010-01-20T00:00:00Z"},
    {"project_id": 200, "worker_id": 3, "entry_created_at": "2010-01-25T00:00:00Z"},
    {"project_id": 300, "worker_id": 2, "entry_created_at": "2010-03-05T00:00:00Z"},
]


# ── unit tests ──────────────────────────────────────────────────────────

class TestGetActiveProjects:
    def test_filters_by_time_window(self):
        t = datetime(2010, 1, 15, tzinfo=timezone.utc)
        active = get_active_projects(FAKE_PROJECT_META, t)
        # project 100 (Jan 1 – Feb 1): active
        # project 200 (Jan 10 – Mar 1): active
        # project 300 (Mar 1 – Apr 1): not started yet
        # project 400 (Jan 5 – Jan 10): expired
        assert 100 in active
        assert 200 in active
        assert 300 not in active
        assert 400 not in active

    def test_boundary_start_date(self):
        t = datetime(2010, 1, 10, tzinfo=timezone.utc)
        active = get_active_projects(FAKE_PROJECT_META, t)
        # project 200 starts exactly at this time
        assert 200 in active
        # project 400 deadline is exactly this time
        assert 400 in active

    def test_no_active_returns_empty(self):
        t = datetime(2009, 1, 1, tzinfo=timezone.utc)
        active = get_active_projects(FAKE_PROJECT_META, t)
        assert len(active) == 0


class TestPopularityRecall:
    def test_ranks_by_recent_entries(self):
        t = datetime(2010, 1, 20, tzinfo=timezone.utc)
        active = {100, 200}
        results = popularity_recall(FAKE_ENTRY_HISTORY, active, t, recency_days=60)
        ids = [pid for pid, _ in results]
        # project 100 has 3 entries, project 200 has 1 entry before t=Jan20
        assert ids[0] == 100

    def test_respects_timestamp_cutoff(self):
        t = datetime(2010, 1, 6, tzinfo=timezone.utc)
        active = {100, 200}
        results = popularity_recall(FAKE_ENTRY_HISTORY, active, t, recency_days=60)
        scores = {pid: s for pid, s in results}
        # Strict < semantics: only Jan 5 entry counts (Jan 6 excluded)
        assert scores[100] == 1.0
        assert scores[200] == 0.0


class TestCategoryRecall:
    def test_boosts_matching_category(self):
        t = datetime(2010, 1, 20, tzinfo=timezone.utc)
        active = {100, 200}
        results = category_recall(
            FAKE_ENTRY_HISTORY, FAKE_PROJECT_META, active, 1, t
        )
        scores = {pid: s for pid, s in results}
        # Worker 1 has 2 entries in cat 1 (project 100) and 1 in cat 2 (project 200)
        # project 100 is cat 1 -> score 2, project 200 is cat 2 -> score 1
        assert scores[100] > scores[200]

    def test_unknown_worker_gets_zero_scores(self):
        t = datetime(2010, 1, 20, tzinfo=timezone.utc)
        active = {100, 200}
        results = category_recall(
            FAKE_ENTRY_HISTORY, FAKE_PROJECT_META, active, 999, t
        )
        scores = {pid: s for pid, s in results}
        assert all(s == 0.0 for s in scores.values())


class TestCandidateGenerator:
    """Integration tests using mocked data loaders."""

    @pytest.fixture
    def gen(self):
        with patch("src.candidates.generator.load_project_metadata") as mock_pm, \
             patch("src.candidates.generator.load_entry_history") as mock_eh:
            mock_pm.return_value = FAKE_PROJECT_META
            mock_eh.return_value = FAKE_ENTRY_HISTORY
            generator = CandidateGenerator()
        return generator

    def test_return_type_is_list_of_int(self, gen):
        t = datetime(2010, 1, 20, tzinfo=timezone.utc)
        result = gen.get_candidates(worker_id=1, timestamp=t, K=10)
        assert isinstance(result, list)
        assert all(isinstance(x, int) for x in result)

    def test_respects_k_limit(self, gen):
        t = datetime(2010, 1, 20, tzinfo=timezone.utc)
        result = gen.get_candidates(worker_id=1, timestamp=t, K=1)
        assert len(result) <= 1

    def test_empty_active_returns_empty(self, gen):
        t = datetime(2009, 1, 1, tzinfo=timezone.utc)
        result = gen.get_candidates(worker_id=1, timestamp=t, K=50)
        assert result == []

    def test_cold_start_worker_returns_results(self, gen):
        """Cold-start worker (no history) should still get candidates
        via global popularity fallback."""
        t = datetime(2010, 1, 20, tzinfo=timezone.utc)
        result = gen.get_candidates(worker_id=9999, timestamp=t, K=10)
        assert len(result) > 0

    def test_no_future_projects(self, gen):
        """Candidates should not include projects that haven't started yet."""
        t = datetime(2010, 1, 15, tzinfo=timezone.utc)
        result = gen.get_candidates(worker_id=1, timestamp=t, K=50)
        # project 300 starts Mar 1, should not appear
        assert 300 not in result

    def test_no_expired_projects(self, gen):
        """Candidates should not include projects past deadline."""
        t = datetime(2010, 1, 15, tzinfo=timezone.utc)
        result = gen.get_candidates(worker_id=1, timestamp=t, K=50)
        # project 400 deadline is Jan 10, should not appear
        assert 400 not in result

    def test_descending_score_order(self, gen):
        """First candidate should be the most popular/relevant."""
        t = datetime(2010, 1, 20, tzinfo=timezone.utc)
        result = gen.get_candidates(worker_id=1, timestamp=t, K=10)
        # With our fake data, project 100 (3 entries, matching category)
        # should rank above project 200
        if len(result) >= 2:
            assert result[0] == 100


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
