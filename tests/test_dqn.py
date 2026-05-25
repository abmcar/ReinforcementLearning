from __future__ import annotations

import torch

from src.rl.models import build_q_network
from src.rl.models.cql import cql_penalty
from src.rl.trainer import double_dqn_targets


def test_all_dqn_variants_score_variable_candidates():
    states = torch.randn(2, 9)
    actions = torch.randn(2, 4, 6)
    mask = torch.tensor([[True, True, False, False], [True, True, True, False]])

    for kind in ("dqn", "double_dqn", "dueling_dqn"):
        model = build_q_network(kind, state_dim=9, action_dim=6)
        scores = model.score_candidates(states, actions, mask)
        assert scores.shape == (2, 4)
        assert scores[0, 2].item() < -1.0e8


def test_cql_penalty_is_finite():
    q_values = torch.tensor([[1.0, 2.0, 3.0], [0.5, -0.5, -1.0]])
    action_indices = torch.tensor([2, 0])
    penalty = cql_penalty(q_values, action_indices)
    assert penalty.shape == (2,)
    assert torch.isfinite(penalty).all()


def test_double_dqn_target_bootstraps_when_next_state_exists():
    online = build_q_network("double_dqn", state_dim=9, action_dim=6)
    target = build_q_network("double_dqn", state_dim=9, action_dim=6)
    states = torch.randn(2, 9)
    actions = torch.randn(2, 3, 6)
    mask = torch.ones(2, 3, dtype=torch.bool)
    rewards = torch.ones(2)
    has_next = torch.tensor([True, False])

    targets = double_dqn_targets(
        online,
        target,
        rewards,
        states,
        actions,
        mask,
        has_next,
        gamma=0.5,
    )

    assert targets.shape == (2,)
    assert targets[1].item() == rewards[1].item()
    assert torch.isfinite(targets).all()
