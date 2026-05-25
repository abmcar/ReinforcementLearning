"""Dueling DQN for variable-size recommendation candidates."""

from __future__ import annotations

import torch
from torch import nn

from src.rl.models.dqn import _mlp


class DuelingDQN(nn.Module):
    """Dueling decomposition: Q(s,a)=V(s)+A(s,a)-mean_a A(s,a)."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: tuple[int, ...] = (128, 64),
    ):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        hidden = hidden_dims[-1]
        self.value_net = _mlp((state_dim, *hidden_dims, 1))
        self.advantage_net = _mlp((state_dim + action_dim, *hidden_dims, 1))
        self._hidden = hidden

    def forward(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        values = self.value_net(states).squeeze(-1)
        advantages = self.advantage_net(torch.cat([states, actions], dim=-1)).squeeze(-1)
        return values + advantages

    def score_candidates(
        self,
        states: torch.Tensor,
        candidate_features: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch, candidates, _ = candidate_features.shape
        values = self.value_net(states).squeeze(-1).unsqueeze(1)
        expanded_states = states.unsqueeze(1).expand(-1, candidates, -1)
        flat = torch.cat([expanded_states, candidate_features], dim=-1)
        advantages = self.advantage_net(
            flat.reshape(batch * candidates, self.state_dim + self.action_dim)
        ).reshape(batch, candidates)
        if mask is not None:
            safe_advantages = advantages.masked_fill(~mask, 0.0)
            denom = mask.sum(dim=1).clamp_min(1).to(advantages.dtype).unsqueeze(1)
            mean_advantage = safe_advantages.sum(dim=1, keepdim=True) / denom
            scores = values + advantages - mean_advantage
            return scores.masked_fill(~mask, -1.0e9)
        return values + advantages - advantages.mean(dim=1, keepdim=True)
