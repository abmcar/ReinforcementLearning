"""Offline DQN training loop."""

from __future__ import annotations

import json
import random
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from src.rl.env import OfflineRecommendationEnv, collate_transitions
from src.rl.env import Transition
from src.rl.models import build_q_network

BASE_DIR = Path(__file__).resolve().parent.parent.parent
OUTPUT_DIR = BASE_DIR / "outputs" / "dqn"


@dataclass
class DQNTrainConfig:
    objective: Literal["worker", "requester"] = "worker"
    split: Literal["train", "val", "test"] = "train"
    model_kind: Literal["dqn", "double_dqn", "dueling_dqn"] = "dueling_dqn"
    candidate_k: int = 50
    max_transitions: int = 5000
    epochs: int = 3
    max_steps: Optional[int] = None
    batch_size: int = 128
    learning_rate: float = 1.0e-3
    gamma: float = 0.0
    target_update_interval: int = 50
    seed: int = 42
    checkpoint: bool = True
    min_candidates: int = 1


@dataclass
class TrainResult:
    config: DQNTrainConfig
    diagnostics: dict[str, float | int | str]
    checkpoint_path: Optional[Path]
    model: nn.Module


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _copy_weights(target: nn.Module, source: nn.Module) -> None:
    target.load_state_dict(source.state_dict())


def double_dqn_targets(
    online: nn.Module,
    target: nn.Module,
    rewards: torch.Tensor,
    next_states: torch.Tensor,
    next_candidate_features: torch.Tensor,
    next_candidate_mask: torch.Tensor,
    has_next: torch.Tensor,
    gamma: float,
) -> torch.Tensor:
    """Compute Double DQN targets for rows with a next state.

    Contextual-bandit rows have ``has_next=False`` and keep the immediate
    reward target.  When a future MDP environment supplies next-state
    candidates, the online network selects the next action and the target
    network evaluates it.
    """

    if gamma <= 0.0 or not bool(has_next.any()):
        return rewards
    with torch.no_grad():
        online_next = online.score_candidates(
            next_states, next_candidate_features, next_candidate_mask
        )
        next_actions = online_next.argmax(dim=1)
        target_next = target.score_candidates(
            next_states, next_candidate_features, next_candidate_mask
        )
        selected_next = target_next.gather(1, next_actions.view(-1, 1)).squeeze(1)
        bootstrapped = rewards + gamma * selected_next * has_next.to(rewards.dtype)
    return bootstrapped


def collect_training_transitions(
    env: OfflineRecommendationEnv,
    max_transitions: int,
    min_candidates: int = 2,
) -> list[Transition]:
    """Collect candidate-rich transitions for DQN training.

    Early historical rows can have no active alternatives, which makes the
    candidate set contain only the injected logged action.  Those rows are valid
    environment transitions but not useful for Q-ranking; training skips them
    and records the number of collected rows in diagnostics.
    """

    transitions: list[Transition] = []
    for transition in env.iter_transitions():
        if len(transition.candidates) < min_candidates:
            continue
        transitions.append(transition)
        if len(transitions) >= max_transitions:
            break
    return transitions


def train_offline_dqn(
    config: DQNTrainConfig,
    *,
    env: Optional[OfflineRecommendationEnv] = None,
    transitions: Optional[list[Transition]] = None,
) -> TrainResult:
    """Train an offline DQN model and return diagnostics/checkpoint path."""

    _set_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if env is None:
        env = OfflineRecommendationEnv(
            split=config.split,
            objective=config.objective,
            candidate_k=config.candidate_k,
        )
    if transitions is None:
        transitions = collect_training_transitions(
            env, config.max_transitions, config.min_candidates
        )
    if not transitions:
        raise ValueError("No transitions produced for training")

    model = build_q_network(config.model_kind, env.state_dim, env.action_dim).to(device)
    target = build_q_network(config.model_kind, env.state_dim, env.action_dim).to(device)
    _copy_weights(target, model)
    target.eval()

    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    losses: list[float] = []
    q_max_history: list[float] = []
    behavior_rank_history: list[float] = []
    behavior_top1_history: list[float] = []
    steps = 0

    while steps < (config.max_steps or config.epochs * max(1, int(np.ceil(len(transitions) / config.batch_size)))):
        indices = np.random.permutation(len(transitions))
        for start in range(0, len(indices), config.batch_size):
            if config.max_steps is not None and steps >= config.max_steps:
                break
            batch_idx = indices[start:start + config.batch_size]
            batch = collate_transitions([transitions[i] for i in batch_idx])
            states = torch.tensor(batch["states"], dtype=torch.float32, device=device)
            rewards = torch.tensor(batch["rewards"], dtype=torch.float32, device=device)
            action_indices = torch.tensor(
                batch["action_indices"], dtype=torch.long, device=device
            )
            candidate_features = torch.tensor(
                batch["candidate_features"], dtype=torch.float32, device=device
            )
            candidate_mask = torch.tensor(
                batch["candidate_mask"], dtype=torch.bool, device=device
            )
            next_states = torch.tensor(
                batch["next_states"], dtype=torch.float32, device=device
            )
            next_candidate_features = torch.tensor(
                batch["next_candidate_features"], dtype=torch.float32, device=device
            )
            next_candidate_mask = torch.tensor(
                batch["next_candidate_mask"], dtype=torch.bool, device=device
            )
            has_next = torch.tensor(batch["has_next"], dtype=torch.bool, device=device)

            q_values = model.score_candidates(states, candidate_features, candidate_mask)
            q_data = q_values.gather(1, action_indices.view(-1, 1)).squeeze(1)

            if config.model_kind == "double_dqn":
                target_q = double_dqn_targets(
                    model,
                    target,
                    rewards,
                    next_states,
                    next_candidate_features,
                    next_candidate_mask,
                    has_next,
                    config.gamma,
                )
            else:
                target_q = rewards
            td_loss = F.smooth_l1_loss(q_data, target_q)
            loss = td_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            losses.append(float(loss.detach().cpu()))
            finite_q = q_values.detach()[candidate_mask]
            q_max_history.append(float(finite_q.max().cpu()))
            ranks = (q_values > q_data.view(-1, 1)).sum(dim=1).to(torch.float32) + 1.0
            behavior_rank_history.append(float(ranks.mean().detach().cpu()))
            best_actions = q_values.argmax(dim=1)
            behavior_top1_history.append(
                float((best_actions == action_indices).to(torch.float32).mean().detach().cpu())
            )

            steps += 1
            if config.model_kind == "double_dqn" and steps % config.target_update_interval == 0:
                _copy_weights(target, model)

    first_window = losses[: max(1, min(10, len(losses)))]
    last_window = losses[-max(1, min(10, len(losses))):]
    q_window = max(1, min(500, len(q_max_history) // 2 if len(q_max_history) > 1 else 1))
    q_initial_mean = float(np.mean(q_max_history[:q_window]))
    q_final_mean = float(np.mean(q_max_history[-q_window:]))
    if len(q_max_history) >= q_window * 2:
        q_prev_tail_mean = float(np.mean(q_max_history[-2 * q_window:-q_window]))
    else:
        q_prev_tail_mean = q_initial_mean
    initial_loss = float(np.mean(first_window))
    final_loss = float(np.mean(last_window))
    diagnostics: dict[str, float | int | str] = {
        "objective": config.objective,
        "model_kind": config.model_kind,
        "seed": config.seed,
        "transitions": len(transitions),
        "steps": steps,
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "loss_drop_fraction": (initial_loss - final_loss) / max(initial_loss, 1.0e-8),
        "q_max": float(np.max(q_max_history)),
        "q_initial_window_mean": q_initial_mean,
        "q_final_window_mean": q_final_mean,
        "q_relative_growth": (q_final_mean - q_initial_mean) / max(abs(q_initial_mean), 1.0e-8),
        "q_prev_tail_window_mean": q_prev_tail_mean,
        "q_tail_relative_growth": (q_final_mean - q_prev_tail_mean) / max(abs(q_prev_tail_mean), 1.0e-8),
        "q_mean_last": float(np.mean(q_max_history[-max(1, min(10, len(q_max_history))):])),
        "behavior_rank_mean_last": float(
            np.mean(behavior_rank_history[-max(1, min(10, len(behavior_rank_history))):])
        ),
        "behavior_top1_overlap_last": float(
            np.mean(behavior_top1_history[-max(1, min(10, len(behavior_top1_history))):])
        ),
        "device": str(device),
    }

    checkpoint_path: Optional[Path] = None
    run_dir = OUTPUT_DIR / "runs" / (
        f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-"
        f"{config.objective}-{config.model_kind}-seed{config.seed}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "diagnostics.json").write_text(
        json.dumps(diagnostics, indent=2, sort_keys=True), encoding="utf-8"
    )
    (run_dir / "config.json").write_text(
        json.dumps(asdict(config), indent=2, sort_keys=True), encoding="utf-8"
    )

    if config.checkpoint:
        checkpoint_path = run_dir / "model.pt"
        torch.save(
            {
                "model_state": model.state_dict(),
                "config": asdict(config),
                "state_dim": env.state_dim,
                "action_dim": env.action_dim,
                "diagnostics": diagnostics,
            },
            checkpoint_path,
        )
        latest_dir = OUTPUT_DIR / "checkpoints"
        latest_dir.mkdir(parents=True, exist_ok=True)
        latest = latest_dir / f"{config.objective}_{config.model_kind}_seed{config.seed}.pt"
        shutil.copy2(checkpoint_path, latest)
        checkpoint_path = latest

    return TrainResult(
        config=config,
        diagnostics=diagnostics,
        checkpoint_path=checkpoint_path,
        model=model,
    )


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Train offline DQN")
    parser.add_argument("--objective", choices=["worker", "requester"], default="worker")
    parser.add_argument(
        "--model-kind",
        choices=["dqn", "double_dqn", "dueling_dqn"],
        default="dueling_dqn",
    )
    parser.add_argument("--split", choices=["train", "val", "test"], default="train")
    parser.add_argument("--candidate-k", type=int, default=50)
    parser.add_argument("--max-transitions", type=int, default=5000)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-candidates", type=int, default=1)
    args = parser.parse_args()

    result = train_offline_dqn(DQNTrainConfig(**vars(args)))
    print(json.dumps(result.diagnostics, indent=2, sort_keys=True))
    if result.checkpoint_path is not None:
        print(f"checkpoint: {result.checkpoint_path}")


if __name__ == "__main__":
    main()
