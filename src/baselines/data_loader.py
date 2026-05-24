"""Shared data loading for baseline recommenders.

Loads data once and shares across all baselines to avoid redundant
86MB+ JSON parsing.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from dateutil.parser import parse as parse_dt

from src.candidates.recall import load_project_metadata

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data"


class SharedData:
    """Singleton-like shared data container for baseline recommenders.

    Loads split_cache.json, project metadata, and worker quality once.
    """

    _instance: Optional["SharedData"] = None

    def __init__(self):
        self.entry_history = self._load_entry_history()
        self.project_meta = load_project_metadata()
        self.worker_quality, self.wq_median = self._load_worker_quality()

    @staticmethod
    def _load_entry_history() -> list[dict]:
        cache_file = DATA_DIR / "split_cache.json"
        if not cache_file.exists():
            raise FileNotFoundError(
                "split_cache.json not found. Run JOB-02 first."
            )
        with open(cache_file, "r", encoding="utf-8") as f:
            splits = json.load(f)
        all_entries = splits["train"] + splits["val"] + splits["test"]
        for entry in all_entries:
            entry["_parsed_ts"] = parse_dt(entry["entry_created_at"])
        return all_entries

    @staticmethod
    def _load_worker_quality() -> Tuple[Dict[int, float], float]:
        wq: Dict[int, float] = {}
        wq_file = DATA_DIR / "worker_quality.csv"
        if wq_file.exists():
            df = pd.read_csv(wq_file)
            df = df[df["worker_quality"] > 0]
            wq = dict(zip(df["worker_id"], df["worker_quality"] / 100.0))
        median = float(np.median(list(wq.values()))) if wq else 0.5
        return wq, median

    @classmethod
    def get(cls) -> "SharedData":
        """Get or create the shared data instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset(cls):
        """Reset singleton (for testing)."""
        cls._instance = None
