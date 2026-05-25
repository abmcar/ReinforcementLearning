"""Double-DQN training configuration.

Double DQN is implemented in the trainer by keeping separate online and target
networks.  This module carries the small amount of configuration needed by that
loop while keeping model construction shared with vanilla DQN.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DoubleDQNConfig:
    gamma: float = 0.0
    target_update_interval: int = 50
