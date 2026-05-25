"""Variable-candidate DQN scoring networks."""

from __future__ import annotations

from typing import Iterable, Literal

import torch
from torch import nn


def _mlp(sizes: Iterable[int], activation: type[nn.Module] = nn.ReLU) -> nn.Sequential:
    layers: list[nn.Module] = []
    values = list(sizes)
    for in_dim, out_dim in zip(values, values[1:]):
        layers.append(nn.Linear(in_dim, out_dim))
        if out_dim != values[-1]:
            layers.append(activation())
    return nn.Sequential(*layers)


class PairwiseDQN(nn.Module):
    """Q(s, a) network for variable-size candidate sets.

    The model scores each candidate independently from the concatenated state
    and action-feature vector, so it does not require a fixed global action
    dimension.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: tuple[int, ...] = (128, 64),
    ):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.net = _mlp((state_dim + action_dim, *hidden_dims, 1))

    def forward(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        x = torch.cat([states, actions], dim=-1)
        return self.net(x).squeeze(-1)

    def score_candidates(
        self,
        states: torch.Tensor,
        candidate_features: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch, candidates, _ = candidate_features.shape
        expanded_states = states.unsqueeze(1).expand(-1, candidates, -1)
        flat_states = expanded_states.reshape(batch * candidates, self.state_dim)
        flat_actions = candidate_features.reshape(batch * candidates, self.action_dim)
        scores = self.forward(flat_states, flat_actions).reshape(batch, candidates)
        if mask is not None:
            scores = scores.masked_fill(~mask, -1.0e9)
        return scores


def build_q_network(
    kind: Literal["dqn", "double_dqn", "dueling_dqn"],
    state_dim: int,
    action_dim: int,
    hidden_dims: tuple[int, ...] = (128, 64),
) -> nn.Module:
    """Build a Q network by model-family name."""

    if kind in ("dqn", "double_dqn"):
        return PairwiseDQN(state_dim, action_dim, hidden_dims)
    if kind == "dueling_dqn":
        from src.rl.models.dueling_dqn import DuelingDQN

        return DuelingDQN(state_dim, action_dim, hidden_dims)
    raise ValueError(f"Unknown DQN kind: {kind}")
