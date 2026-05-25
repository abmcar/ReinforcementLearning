"""Adapters that expose trained DQN models through the evaluator protocol."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple

import numpy as np
import torch

from src.eval.protocols import HasRecommend
from src.rl.env import OfflineRecommendationEnv
from src.rl.models import build_q_network


class DQNRecommender(HasRecommend):
    """Rank candidates by predicted Q value."""

    def __init__(
        self,
        model: torch.nn.Module,
        env: OfflineRecommendationEnv,
        *,
        device: Optional[str] = None,
        feature_cache: Optional[Dict[Tuple[int, str, Tuple[int, ...]], Tuple[np.ndarray, np.ndarray]]] = None,
    ):
        self.model = model
        self.env = env
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model.to(self.device)
        self.model.eval()
        self._feature_cache = feature_cache if feature_cache is not None else {}

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        *,
        split: Literal["train", "val", "test"] = "val",
        objective: Literal["worker", "requester"] = "worker",
        model_kind: Literal["dqn", "double_dqn", "dueling_dqn"] = "dueling_dqn",
        candidate_k: int = 50,
    ) -> "DQNRecommender":
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        env = OfflineRecommendationEnv(
            split=split, objective=objective, candidate_k=candidate_k
        )
        kind = ckpt.get("config", {}).get("model_kind", model_kind)
        model = build_q_network(kind, ckpt["state_dim"], ckpt["action_dim"])
        model.load_state_dict(ckpt["model_state"])
        return cls(model, env)

    def recommend(
        self,
        worker_id: int,
        timestamp: datetime,
        candidates: List[int],
    ) -> List[int]:
        if not candidates:
            return []
        key = (int(worker_id), timestamp.isoformat(), tuple(int(pid) for pid in candidates))
        cached = self._feature_cache.get(key)
        if cached is None:
            state = self.env.state_for(worker_id, timestamp)
            action_features = self.env.action_features_for(candidates, timestamp)
            self._feature_cache[key] = (state, action_features)
        else:
            state, action_features = cached
        with torch.no_grad():
            states = torch.tensor(state, dtype=torch.float32, device=self.device).view(1, -1)
            actions = torch.tensor(
                action_features, dtype=torch.float32, device=self.device
            ).unsqueeze(0)
            mask = torch.ones((1, len(candidates)), dtype=torch.bool, device=self.device)
            scores = self.model.score_candidates(states, actions, mask).squeeze(0)
        ranked = sorted(
            zip(candidates, scores.detach().cpu().tolist()),
            key=lambda item: item[1],
            reverse=True,
        )
        return [int(pid) for pid, _score in ranked]
