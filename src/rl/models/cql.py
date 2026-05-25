"""Conservative Q-Learning loss helpers for offline DQN."""

from __future__ import annotations

import torch


def cql_penalty(q_values: torch.Tensor, action_indices: torch.Tensor) -> torch.Tensor:
    """Return CQL discrete-action penalty for a batch.

    ``q_values`` has shape ``[batch, candidates]`` and invalid candidates must
    already be masked to a large negative value.
    """

    data_q = q_values.gather(1, action_indices.view(-1, 1)).squeeze(1)
    return torch.logsumexp(q_values, dim=1) - data_q
