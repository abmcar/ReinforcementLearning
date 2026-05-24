"""Recall strategies for candidate generation.

Each recall function returns a list of (project_id, score) tuples,
where higher score means higher recall priority.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Set, Tuple

from dateutil.parser import parse as parse_dt

# ── data paths ──────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data"


# ── shared loaders ──────────────────────────────────────────────────────

def load_project_metadata() -> Dict[int, dict]:
    """Load project static metadata (start_date, deadline, category, ...).

    Returns a dict keyed by project_id.
    """
    project_dir = DATA_DIR / "project"
    project_info: Dict[int, dict] = {}
    if not project_dir.exists():
        return project_info

    for txt_file in project_dir.glob("project_*.txt"):
        try:
            pid = int(txt_file.stem.split("_")[1])
            with open(txt_file, "r", encoding="utf-8", errors="ignore") as f:
                data = json.loads(f.read())

            raw_start = data.get("start_date")
            raw_deadline = data.get("deadline")
            if not raw_start or not raw_deadline:
                continue

            project_info[pid] = {
                "start_date": parse_dt(raw_start),
                "deadline": parse_dt(raw_deadline),
                "category": int(data.get("category", 0)),
                "sub_category": int(data.get("sub_category", 0)),
            }
        except Exception:
            continue

    return project_info


def load_entry_history() -> list[dict]:
    """Load all entries from split cache (sorted by time).

    Returns a list of dicts with keys:
        project_id, worker_id, entry_created_at (str),
        _parsed_ts (datetime, pre-parsed for performance).
    """
    cache_file = DATA_DIR / "split_cache.json"
    if not cache_file.exists():
        raise FileNotFoundError(
            "split_cache.json not found. Run JOB-02 (src/data/split.py) first."
        )
    with open(cache_file, "r", encoding="utf-8") as f:
        splits = json.load(f)
    # Concatenate all splits in chronological order
    all_entries = splits["train"] + splits["val"] + splits["test"]
    # Pre-parse timestamps once to avoid repeated parse_dt calls
    for entry in all_entries:
        entry["_parsed_ts"] = parse_dt(entry["entry_created_at"])
    return all_entries


# ── recall strategies ───────────────────────────────────────────────────

def get_active_projects(
    project_meta: Dict[int, dict],
    timestamp: datetime,
) -> Set[int]:
    """Return project IDs that are active at *timestamp*.

    Active means ``start_date <= timestamp <= deadline``.
    """
    active: Set[int] = set()
    for pid, info in project_meta.items():
        if info["start_date"] <= timestamp <= info["deadline"]:
            active.add(pid)
    return active


def popularity_recall(
    entry_history: list[dict],
    active_projects: Set[int],
    timestamp: datetime,
    recency_days: int = 60,
) -> List[Tuple[int, float]]:
    """Global popularity recall: rank active projects by recent entry count.

    Only entries whose ``entry_created_at`` is within
    ``[timestamp - recency_days, timestamp]`` are counted.

    Returns (project_id, score) sorted descending by score.
    """
    cutoff = timestamp - timedelta(days=recency_days)
    counts: Dict[int, int] = {}

    for entry in entry_history:
        t = entry.get("_parsed_ts") or parse_dt(entry["entry_created_at"])
        # Anti-leakage: only use entries strictly BEFORE timestamp
        if t >= timestamp:
            break  # entries are sorted chronologically
        if t < cutoff:
            continue
        pid = entry["project_id"]
        if pid in active_projects:
            counts[pid] = counts.get(pid, 0) + 1

    # Active projects with zero recent entries still get score 0
    results: List[Tuple[int, float]] = []
    for pid in active_projects:
        results.append((pid, float(counts.get(pid, 0))))

    results.sort(key=lambda x: x[1], reverse=True)
    return results


def category_recall(
    entry_history: list[dict],
    project_meta: Dict[int, dict],
    active_projects: Set[int],
    worker_id: int,
    timestamp: datetime,
) -> List[Tuple[int, float]]:
    """Category-match recall: boost projects in categories the worker has
    historically participated in.

    Returns (project_id, score) sorted descending by score.
    Score = number of past entries the worker made in the same category.
    """
    # Build worker's category histogram (only entries strictly before timestamp)
    cat_counts: Dict[int, int] = {}
    for entry in entry_history:
        t = entry.get("_parsed_ts") or parse_dt(entry["entry_created_at"])
        if t >= timestamp:
            break
        if entry["worker_id"] != worker_id:
            continue
        pid = entry["project_id"]
        cat = project_meta.get(pid, {}).get("category", -1)
        if cat >= 0:
            cat_counts[cat] = cat_counts.get(cat, 0) + 1

    # Score each active project by worker's affinity to its category
    results: List[Tuple[int, float]] = []
    for pid in active_projects:
        cat = project_meta.get(pid, {}).get("category", -1)
        score = float(cat_counts.get(cat, 0))
        results.append((pid, score))

    results.sort(key=lambda x: x[1], reverse=True)
    return results
