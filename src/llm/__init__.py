"""LLM data construction utilities for crowdsourcing recommendation."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path


def build_dataset(objective: str, split: str) -> "Path":
    from src.llm.data_builder import build_dataset as _build_dataset

    return _build_dataset(objective, split)


__all__ = ["build_dataset"]
