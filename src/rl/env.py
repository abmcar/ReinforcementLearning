"""Offline recommendation environment for DQN training.

The project has historical logs only, so the default environment is a
contextual bandit replay buffer rather than an interactive simulator.
"""

from __future__ import annotations

import bisect
import math
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, Iterator, List, Literal, Mapping, Optional

import numpy as np
import pandas as pd
from dateutil.parser import parse as parse_dt

from src.baselines.adapter import _FastCandidateGenerator
from src.data.split import load_split
from src.features.build import build_features
from src.rl.rewards import RewardConfig, RewardContext, requester_reward_fn, worker_reward_fn

Objective = Literal["worker", "requester"]

BASE_DIR = Path(__file__).resolve().parent.parent.parent
FEATURE_DIR = BASE_DIR / "outputs" / "features"


@dataclass
class Transition:
    """Single logged recommendation event consumed by offline DQN."""

    s: np.ndarray
    a: int
    r: float
    s_next: Optional[np.ndarray]
    candidates: List[int]
    info: Dict[str, object]
    candidate_features: np.ndarray
    action_index: int


RewardFn = Callable[[Mapping[str, object], Optional[RewardContext], RewardConfig], float]


def _load_feature_frame(split: str) -> pd.DataFrame:
    path = FEATURE_DIR / f"{split}_features.csv"
    if not path.exists():
        if split in ("val", "test") and not (FEATURE_DIR / "train_features.csv").exists():
            build_features("train")
        return build_features(split)
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"Feature file is empty: {path}")
    return df


def _timestamp_key(value: object) -> str:
    if isinstance(value, datetime):
        return value.isoformat()
    return parse_dt(str(value)).isoformat()


class OfflineRecommendationEnv:
    """Replay-buffer style environment for logged recommendation events."""

    state_dim = 9
    action_dim = 6

    def __init__(
        self,
        split: Literal["train", "val", "test"] = "train",
        objective: Objective = "worker",
        *,
        candidate_k: int = 50,
        reward_config: RewardConfig = RewardConfig(),
        feature_df: Optional[pd.DataFrame] = None,
        candidate_fn: Optional[Callable[[int, datetime, int], List[int]]] = None,
        project_meta: Optional[Dict[int, dict]] = None,
        entry_history: Optional[list[dict]] = None,
        max_transitions: Optional[int] = None,
    ):
        if split not in ("train", "val", "test"):
            raise ValueError("split must be one of train, val, test")
        if objective not in ("worker", "requester"):
            raise ValueError("objective must be worker or requester")

        self.split = split
        self.objective = objective
        self.candidate_k = candidate_k
        self.reward_config = reward_config
        self.max_transitions = max_transitions
        self.feature_df = (feature_df.copy() if feature_df is not None else _load_feature_frame(split))
        self.feature_df["entry_created_at"] = self.feature_df["entry_created_at"].map(_timestamp_key)
        self.reward_fn: RewardFn = worker_reward_fn if objective == "worker" else requester_reward_fn

        self._fast_candidates: Optional[_FastCandidateGenerator]
        if project_meta is None or entry_history is None or candidate_fn is None:
            self._fast_candidates = _FastCandidateGenerator()
            self.project_meta = project_meta or self._fast_candidates.project_meta
            self.entry_history = entry_history or self._fast_candidates.entry_history
        else:
            self._fast_candidates = None
            self.project_meta = project_meta
            self.entry_history = entry_history
        if candidate_fn is not None:
            self._candidate_fn = candidate_fn
        else:
            assert self._fast_candidates is not None
            self._candidate_fn = (
                lambda worker_id, timestamp, k: self._fast_candidates.get_candidates(
                    worker_id, timestamp, K=k
                )
            )

        self._project_entry_times: Dict[int, List[datetime]] = defaultdict(list)
        for entry in self.entry_history:
            self._project_entry_times[int(entry["project_id"])].append(entry["_parsed_ts"])

        self._state_index: Dict[tuple[int, str], np.ndarray] = {}
        for row in self.feature_df.to_dict("records"):
            key = (int(row["worker_id"]), str(row["entry_created_at"]))
            self._state_index[key] = self.state_from_row(row)

        self._worker_category_hist: Dict[int, Dict[int, int]] = defaultdict(lambda: defaultdict(int))
        self._seed_history(split)
        self._initial_worker_category_hist = {
            worker_id: dict(counts)
            for worker_id, counts in self._worker_category_hist.items()
        }
        self._cache: Optional[List[Transition]] = None

    def _seed_history(self, split: str) -> None:
        prior_splits: list[str]
        if split == "train":
            prior_splits = []
        elif split == "val":
            prior_splits = ["train"]
        else:
            prior_splits = ["train", "val"]
        for prior in prior_splits:
            for entry in load_split(prior).entries:
                self._update_worker_history(entry.worker_id, entry.project_id)

    def _update_worker_history(self, worker_id: int, project_id: int) -> None:
        cat = int(self.project_meta.get(project_id, {}).get("category", -1))
        if cat >= 0:
            self._worker_category_hist[int(worker_id)][cat] += 1

    def _category_match(self, worker_id: int, project_id: int) -> bool:
        hist = self._worker_category_hist.get(int(worker_id), {})
        if not hist:
            return False
        top_category = max(hist, key=hist.get)
        return int(self.project_meta.get(project_id, {}).get("category", -1)) == top_category

    def state_from_row(self, row: Mapping[str, object]) -> np.ndarray:
        ts = parse_dt(str(row["entry_created_at"]))
        day_of_year = ts.timetuple().tm_yday
        day_of_week = ts.weekday()
        values = [
            float(row.get("worker_quality", 0.5) or 0.5),
            math.log1p(float(row.get("hist_entries", 0.0) or 0.0)) / 10.0,
            math.log1p(float(row.get("hist_wins", 0.0) or 0.0)) / 6.0,
            float(row.get("hist_win_rate", 0.0) or 0.0),
            math.log1p(float(row.get("hist_avg_award", 0.0) or 0.0)) / 8.0,
            math.sin(2.0 * math.pi * day_of_year / 366.0),
            math.cos(2.0 * math.pi * day_of_year / 366.0),
            math.sin(2.0 * math.pi * day_of_week / 7.0),
            math.cos(2.0 * math.pi * day_of_week / 7.0),
        ]
        return np.asarray(values, dtype=np.float32)

    def state_for(self, worker_id: int, timestamp: datetime) -> np.ndarray:
        key = (int(worker_id), timestamp.isoformat())
        state = self._state_index.get(key)
        if state is not None:
            return state
        fallback = {
            "worker_quality": 0.5,
            "hist_entries": 0.0,
            "hist_wins": 0.0,
            "hist_win_rate": 0.0,
            "hist_avg_award": 0.0,
            "entry_created_at": timestamp.isoformat(),
        }
        return self.state_from_row(fallback)

    def _current_entries(self, project_id: int, timestamp: datetime) -> int:
        times = self._project_entry_times.get(int(project_id), [])
        return bisect.bisect_left(times, timestamp)

    def action_features_for(self, project_ids: List[int], timestamp: datetime) -> np.ndarray:
        features = [self._action_features(pid, timestamp) for pid in project_ids]
        if not features:
            return np.zeros((0, self.action_dim), dtype=np.float32)
        return np.stack(features).astype(np.float32)

    def _action_features(self, project_id: int, timestamp: datetime) -> np.ndarray:
        meta = self.project_meta.get(int(project_id), {})
        start = meta.get("start_date", timestamp)
        deadline = meta.get("deadline", timestamp)
        duration_days = max((deadline - start).total_seconds() / 86400.0, 0.0)
        days_remaining = (deadline - timestamp).total_seconds() / 86400.0
        is_active = 1.0 if start <= timestamp <= deadline else 0.0
        values = [
            float(meta.get("category", 0)) / 100.0,
            float(meta.get("sub_category", 0)) / 100.0,
            math.log1p(duration_days) / 6.0,
            max(min(days_remaining / 365.0, 1.0), -1.0),
            math.log1p(float(self._current_entries(int(project_id), timestamp))) / 10.0,
            is_active,
        ]
        return np.asarray(values, dtype=np.float32)

    def _candidates_for(self, worker_id: int, timestamp: datetime, project_id: int) -> List[int]:
        candidates = list(self._candidate_fn(int(worker_id), timestamp, self.candidate_k))
        if not candidates:
            candidates = [int(project_id)]
        elif project_id not in candidates:
            if len(candidates) >= self.candidate_k:
                candidates[-1] = int(project_id)
            else:
                candidates.append(int(project_id))
        return [int(pid) for pid in candidates]

    def iter_transitions(
        self,
        split: Optional[Literal["train", "val", "test"]] = None,
    ) -> Iterator[Transition]:
        if split is not None and split != self.split:
            yield from OfflineRecommendationEnv(
                split=split,
                objective=self.objective,
                candidate_k=self.candidate_k,
                reward_config=self.reward_config,
                max_transitions=self.max_transitions,
            ).iter_transitions()
            return

        self._worker_category_hist = defaultdict(lambda: defaultdict(int))
        for worker_id, counts in self._initial_worker_category_hist.items():
            self._worker_category_hist[worker_id].update(counts)

        produced = 0
        for row in self.feature_df.to_dict("records"):
            worker_id = int(row["worker_id"])
            project_id = int(row["project_id"])
            timestamp = parse_dt(str(row["entry_created_at"]))
            candidates = self._candidates_for(worker_id, timestamp, project_id)
            action_index = candidates.index(project_id)
            category_match = self._category_match(worker_id, project_id)
            context = RewardContext(
                category_match=category_match,
                worker_quality=float(row.get("worker_quality", 0.5) or 0.5),
            )
            state = self.state_from_row(row)
            reward = self.reward_fn(row, context, self.reward_config)
            candidate_features = self.action_features_for(candidates, timestamp)
            yield Transition(
                s=state,
                a=project_id,
                r=reward,
                s_next=None,
                candidates=candidates,
                info={
                    "worker_id": worker_id,
                    "project_id": project_id,
                    "timestamp": timestamp,
                    "objective": self.objective,
                    "category_match": category_match,
                },
                candidate_features=candidate_features,
                action_index=action_index,
            )
            self._update_worker_history(worker_id, project_id)
            produced += 1
            if self.max_transitions is not None and produced >= self.max_transitions:
                break

    def materialize(self, max_transitions: Optional[int] = None) -> List[Transition]:
        if self._cache is None:
            if max_transitions is None:
                self._cache = list(self.iter_transitions())
            else:
                original_max = self.max_transitions
                self.max_transitions = max_transitions
                try:
                    self._cache = list(self.iter_transitions())
                finally:
                    self.max_transitions = original_max
        return self._cache if max_transitions is None else self._cache[:max_transitions]

    def sample_batch(self, batch_size: int) -> Dict[str, np.ndarray]:
        transitions = self.materialize()
        if not transitions:
            raise ValueError("No transitions available to sample")
        size = min(batch_size, len(transitions))
        indices = np.random.choice(len(transitions), size=size, replace=False)
        batch = [transitions[i] for i in indices]
        return collate_transitions(batch)


def collate_transitions(transitions: List[Transition]) -> Dict[str, np.ndarray]:
    if not transitions:
        raise ValueError("transitions must be non-empty")
    states = np.stack([t.s for t in transitions]).astype(np.float32)
    rewards = np.asarray([t.r for t in transitions], dtype=np.float32)
    action_indices = np.asarray([t.action_index for t in transitions], dtype=np.int64)
    max_candidates = max(len(t.candidates) for t in transitions)
    action_dim = transitions[0].candidate_features.shape[1]
    state_dim = transitions[0].s.shape[0]
    candidate_features = np.zeros(
        (len(transitions), max_candidates, action_dim), dtype=np.float32
    )
    candidate_mask = np.zeros((len(transitions), max_candidates), dtype=bool)
    next_states = np.zeros((len(transitions), state_dim), dtype=np.float32)
    next_candidate_features = np.zeros(
        (len(transitions), max_candidates, action_dim), dtype=np.float32
    )
    next_candidate_mask = np.zeros((len(transitions), max_candidates), dtype=bool)
    has_next = np.zeros((len(transitions),), dtype=bool)
    for i, transition in enumerate(transitions):
        n = len(transition.candidates)
        candidate_features[i, :n, :] = transition.candidate_features
        candidate_mask[i, :n] = True
        if transition.s_next is not None:
            next_states[i] = transition.s_next
            next_candidate_features[i, :n, :] = transition.candidate_features
            next_candidate_mask[i, :n] = True
            has_next[i] = True
    return {
        "states": states,
        "rewards": rewards,
        "action_indices": action_indices,
        "candidate_features": candidate_features,
        "candidate_mask": candidate_mask,
        "next_states": next_states,
        "next_candidate_features": next_candidate_features,
        "next_candidate_mask": next_candidate_mask,
        "has_next": has_next,
    }


def iter_transitions(split: str) -> Iterator[Transition]:
    """Roadmap-compatible module-level transition iterator."""

    if split not in ("train", "val", "test"):
        raise ValueError("split must be one of train, val, test")
    yield from OfflineRecommendationEnv(split=split).iter_transitions()
