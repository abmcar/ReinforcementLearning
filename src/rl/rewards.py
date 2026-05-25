"""Reward functions for the DQN recommendation environment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional


@dataclass(frozen=True)
class RewardConfig:
    """Reward shaping coefficients shared by worker/requester objectives."""

    award_scale: float = 100.0
    finalist_bonus: float = 0.5
    winner_bonus: float = 1.0
    category_match_bonus: float = 0.1


@dataclass(frozen=True)
class RewardContext:
    """Strictly-prior context available when computing per-step rewards."""

    category_match: bool = False
    worker_quality: float = 0.5


def worker_reward_fn(
    row: Mapping[str, object],
    context: Optional[RewardContext] = None,
    config: RewardConfig = RewardConfig(),
) -> float:
    """Reward for the participant-side objective.

    The roadmap formula is award plus finalist/winner/category terms.  The
    implementation scales dollars by ``award_scale`` so gradients remain in a
    small numeric range while preserving ordering.
    """

    ctx = context or RewardContext()
    award = float(row.get("label_award", 0.0) or 0.0) / config.award_scale
    finalist = float(row.get("label_finalist", 0.0) or 0.0)
    winner = float(row.get("label_winner", 0.0) or 0.0)
    category_match = 1.0 if ctx.category_match else 0.0
    return float(
        award
        + config.finalist_bonus * finalist
        + config.winner_bonus * winner
        + config.category_match_bonus * category_match
    )


def requester_reward_fn(
    row: Mapping[str, object],
    context: Optional[RewardContext] = None,
    config: RewardConfig = RewardConfig(),
) -> float:
    """Reward for the requester-side objective.

    A requester benefits when the recommended project receives work from a
    higher-quality worker.  Worker quality is already normalized to [0, 1] in
    JOB-03 features; when missing, the context fallback is used.
    """

    ctx = context or RewardContext()
    value = row.get("worker_quality", ctx.worker_quality)
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(ctx.worker_quality)
