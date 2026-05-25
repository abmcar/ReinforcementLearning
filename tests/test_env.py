from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd

from src.rl.env import OfflineRecommendationEnv

PROJECT_META = {
    100: {
        "start_date": datetime(2020, 1, 1, tzinfo=timezone.utc),
        "deadline": datetime(2020, 2, 1, tzinfo=timezone.utc),
        "category": 1,
        "sub_category": 2,
    },
    200: {
        "start_date": datetime(2020, 1, 1, tzinfo=timezone.utc),
        "deadline": datetime(2020, 2, 1, tzinfo=timezone.utc),
        "category": 3,
        "sub_category": 4,
    },
    300: {
        "start_date": datetime(2020, 1, 1, tzinfo=timezone.utc),
        "deadline": datetime(2020, 2, 1, tzinfo=timezone.utc),
        "category": 5,
        "sub_category": 6,
    },
}


def _fake_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "worker_id": 1,
                "project_id": 100,
                "entry_created_at": datetime(2020, 1, 5, tzinfo=timezone.utc).isoformat(),
                "label_winner": 0,
                "label_finalist": 1,
                "label_award": 50.0,
                "worker_quality": 0.8,
                "hist_entries": 2,
                "hist_wins": 1,
                "hist_win_rate": 0.5,
                "hist_avg_award": 20.0,
                "category": 1,
                "sub_category": 2,
                "industry": 0,
                "duration_days": 30.0,
                "days_remaining": 10.0,
                "current_entries": 5,
            }
        ]
    )


def test_transition_injects_logged_action_when_candidate_misses():
    env = OfflineRecommendationEnv(
        split="train",
        objective="worker",
        candidate_k=3,
        feature_df=_fake_df(),
        candidate_fn=lambda _worker_id, _timestamp, _k: [200, 300],
        project_meta=PROJECT_META,
        entry_history=[],
        max_transitions=1,
    )
    transition = next(env.iter_transitions())

    assert transition.a == 100
    assert 100 in transition.candidates
    assert transition.candidates[-1] == 100
    assert transition.action_index == transition.candidates.index(100)
    assert transition.s_next is None
    assert transition.s.shape == (env.state_dim,)
    assert transition.candidate_features.shape[1] == env.action_dim
    assert transition.r > 0.0


def test_reward_objective_switches():
    worker_env = OfflineRecommendationEnv(
        split="train",
        objective="worker",
        feature_df=_fake_df(),
        candidate_fn=lambda _worker_id, _timestamp, _k: [100],
        project_meta=PROJECT_META,
        entry_history=[],
        max_transitions=1,
    )
    requester_env = OfflineRecommendationEnv(
        split="train",
        objective="requester",
        feature_df=_fake_df(),
        candidate_fn=lambda _worker_id, _timestamp, _k: [100],
        project_meta=PROJECT_META,
        entry_history=[],
        max_transitions=1,
    )

    worker_reward = next(worker_env.iter_transitions()).r
    requester_reward = next(requester_env.iter_transitions()).r
    assert worker_reward != requester_reward
    assert np.isclose(requester_reward, 0.8)


def test_sample_batch_shapes():
    env = OfflineRecommendationEnv(
        split="train",
        objective="worker",
        feature_df=_fake_df(),
        candidate_fn=lambda _worker_id, _timestamp, _k: [100],
        project_meta=PROJECT_META,
        entry_history=[],
        max_transitions=1,
    )
    batch = env.sample_batch(4)
    assert batch["states"].shape == (1, env.state_dim)
    assert batch["candidate_features"].shape == (1, 1, env.action_dim)
    assert batch["candidate_mask"].tolist() == [[True]]


def test_iter_transitions_is_repeatable():
    env = OfflineRecommendationEnv(
        split="train",
        objective="worker",
        feature_df=_fake_df(),
        candidate_fn=lambda _worker_id, _timestamp, _k: [100],
        project_meta=PROJECT_META,
        entry_history=[],
        max_transitions=1,
    )

    first = next(env.iter_transitions())
    second = next(env.iter_transitions())
    assert first.r == second.r
    assert first.info["category_match"] == second.info["category_match"]
