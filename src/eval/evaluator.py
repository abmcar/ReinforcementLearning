"""Unified evaluation entry point for crowdsourcing task recommendation.

Usage
-----
>>> from src.eval import evaluate, HasRecommend
>>> results = evaluate(model, "test")
>>> print(results["HR@10"])
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import (
    Callable,
    Dict,
    List,
    Literal,
    Optional,
    Set,
    Tuple,
)

import numpy as np
from dateutil.parser import parse as dateparse

from src.data.split import Entry, EntryList, load_split
from src.eval.metrics import (
    avg_award_value_at_k,
    avg_recommender_worker_quality,
    category_match_rate_at_k,
    finalist_rate_at_k,
    hit_rate_at_k,
    mrr,
    ndcg_at_k,
    precision_at_k,
    project_coverage,
    recall_at_k,
    winner_rate_at_k,
)
from src.eval.protocols import HasRecommend

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

K_VALUES = [1, 5, 10]
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data"


# ---------------------------------------------------------------------------
# Metadata loaders
# ---------------------------------------------------------------------------


def _load_worker_quality() -> Tuple[Dict[int, float], float]:
    """Load worker quality from CSV.  Returns (mapping, global_median)."""
    import pandas as pd

    wq: Dict[int, float] = {}
    wq_file = DATA_DIR / "worker_quality.csv"
    if wq_file.exists():
        df = pd.read_csv(wq_file)
        df = df[df["worker_quality"] > 0]
        wq = dict(zip(df["worker_id"], df["worker_quality"] / 100.0))
    median = float(np.median(list(wq.values()))) if wq else 0.5
    return wq, median


def _load_project_info() -> Dict[int, dict]:
    """Load project static metadata (category, dates, etc.)."""
    project_info: Dict[int, dict] = {}
    project_dir = DATA_DIR / "project"
    if not project_dir.exists():
        return project_info
    for txt in project_dir.glob("project_*.txt"):
        try:
            pid = int(txt.stem.split("_")[1])
            with open(txt, "r", encoding="utf-8", errors="ignore") as f:
                data = json.loads(f.read())
            raw_start = data.get("start_date")
            raw_deadline = data.get("deadline")
            if not raw_start or not raw_deadline:
                continue
            start_date = dateparse(raw_start)
            deadline = dateparse(raw_deadline)
            project_info[pid] = {
                "category": int(data.get("category", 0)),
                "sub_category": int(data.get("sub_category", 0)),
                "start_date": start_date,
                "deadline": deadline,
                "total_awards": float(data.get("total_awards", 0) or 0),
            }
        except Exception:
            continue
    return project_info


def _get_active_projects(
    project_info: Dict[int, dict],
    time_range: Tuple[datetime, datetime],
) -> Set[int]:
    """Return project IDs whose [start_date, deadline] overlaps time_range."""
    t_start, t_end = time_range
    active = set()
    for pid, info in project_info.items():
        if info["start_date"] <= t_end and info["deadline"] >= t_start:
            active.add(pid)
    return active


# ---------------------------------------------------------------------------
# Default candidate generator (used before JOB-06 is implemented)
# ---------------------------------------------------------------------------


def default_candidate_fn(
    worker_id: int,
    timestamp: datetime,
    project_info: Dict[int, dict],
    ground_truth_pid: int,
    k: int = 50,
) -> List[int]:
    """Fallback candidate generator: active projects at *timestamp*.

    Always includes *ground_truth_pid* so that HR@K can be non-zero.
    Caps the list at *k* items (randomly sampled if more are available).
    """
    candidates = [
        pid
        for pid, info in project_info.items()
        if info["start_date"] <= timestamp <= info["deadline"]
    ]
    # Ensure ground truth is present
    if ground_truth_pid not in candidates:
        candidates.append(ground_truth_pid)
    if len(candidates) > k:
        # Keep ground truth, sample the rest
        candidates_without_gt = [c for c in candidates if c != ground_truth_pid]
        sampled = random.sample(candidates_without_gt, k - 1)
        candidates = sampled + [ground_truth_pid]
    return candidates


# ---------------------------------------------------------------------------
# Worker history builder (time-machine replay for category preferences)
# ---------------------------------------------------------------------------


def _build_worker_histories(
    entries: List[Entry],
    project_info: Dict[int, dict],
) -> Tuple[
    Dict[int, Set[int]],  # worker -> preferred categories
    Dict[int, Set[int]],  # worker -> finalist projects
    Dict[int, Set[int]],  # worker -> winner projects
    Dict[int, float],     # project -> max award_value observed
]:
    """Replay entries in time order to build per-worker lookup tables.

    This mirrors the time-machine replay from ``build_features`` —
    we only use data from the training portion that precedes each entry.
    For simplicity (and because evaluate runs on a single split),
    we accumulate over all provided entries.
    """
    worker_categories: Dict[int, Dict[int, int]] = defaultdict(lambda: defaultdict(int))
    worker_finalist: Dict[int, Set[int]] = defaultdict(set)
    worker_winner: Dict[int, Set[int]] = defaultdict(set)
    project_award: Dict[int, float] = {}

    for entry in entries:
        cat = project_info.get(entry.project_id, {}).get("category", -1)
        if cat != -1:
            worker_categories[entry.worker_id][cat] += 1
        if entry.finalist:
            worker_finalist[entry.worker_id].add(entry.project_id)
        if entry.winner:
            worker_winner[entry.worker_id].add(entry.project_id)
        # Track maximum award per project
        prev = project_award.get(entry.project_id, 0.0)
        project_award[entry.project_id] = max(prev, entry.award_value)

    # Convert category counts to sets of preferred categories (top-3)
    worker_pref_cats: Dict[int, Set[int]] = {}
    for wid, cat_counts in worker_categories.items():
        sorted_cats = sorted(cat_counts, key=cat_counts.get, reverse=True)  # type: ignore[arg-type]
        worker_pref_cats[wid] = set(sorted_cats[:3])

    return worker_pref_cats, worker_finalist, worker_winner, project_award


# ---------------------------------------------------------------------------
# Bootstrap confidence interval
# ---------------------------------------------------------------------------


def _bootstrap_ci(
    values: List[float],
    n_bootstrap: int = 1000,
    ci: float = 0.95,
) -> Tuple[float, float]:
    """Return (lower, upper) bounds of a bootstrap confidence interval."""
    if not values:
        return (0.0, 0.0)
    arr = np.array(values)
    boot_means = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(arr, size=len(arr), replace=True)
        boot_means.append(float(np.mean(sample)))
    alpha = (1 - ci) / 2
    lower = float(np.percentile(boot_means, alpha * 100))
    upper = float(np.percentile(boot_means, (1 - alpha) * 100))
    return lower, upper


# ---------------------------------------------------------------------------
# Main evaluation entry point
# ---------------------------------------------------------------------------


def evaluate(
    model: HasRecommend,
    split: Literal["train", "val", "test"],
    *,
    candidate_fn: Optional[
        Callable[[int, datetime, Dict[int, dict], int, int], List[int]]
    ] = None,
    candidate_k: int = 50,
    n_bootstrap: int = 0,
    seed: int = 42,
) -> Dict[str, float]:
    """Evaluate a recommender model on a data split.

    Parameters
    ----------
    model : HasRecommend
        Any object with ``recommend(worker_id, timestamp, candidates)``.
    split : str
        One of ``"train"``, ``"val"``, ``"test"``.
    candidate_fn : callable, optional
        ``(worker_id, timestamp, project_info, ground_truth_pid, k) -> list[int]``.
        Defaults to an internal generator that returns active projects at
        *timestamp* (see ``default_candidate_fn``).  When JOB-06 lands,
        pass ``get_candidates`` here.
    candidate_k : int
        Number of candidates passed to the model (default 50).
    n_bootstrap : int
        If > 0, also compute bootstrap confidence intervals for each metric
        (adds ``<metric>_ci_lower`` / ``<metric>_ci_upper`` keys).
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    dict[str, float]
        All computed metrics.  Keys include ``HR@1``, ``HR@5``, ``HR@10``,
        ``NDCG@1``, etc., plus worker-objective and requester-objective
        metrics.
    """
    random.seed(seed)
    np.random.seed(seed)

    # Load data
    entry_list: EntryList = load_split(split)
    project_info = _load_project_info()
    worker_quality, wq_median = _load_worker_quality()

    # Build worker histories from training data for category preferences.
    # For val/test we also include earlier splits so that the history is richer.
    history_entries: List[Entry] = []
    if split in ("val", "test"):
        history_entries.extend(load_split("train").entries)
    if split == "test":
        history_entries.extend(load_split("val").entries)
    history_entries.extend(entry_list.entries)

    (
        worker_pref_cats,
        worker_finalist,
        worker_winner,
        project_award,
    ) = _build_worker_histories(history_entries, project_info)

    active_projects = _get_active_projects(
        project_info, entry_list.time_range
    )

    if candidate_fn is None:
        candidate_fn = default_candidate_fn

    # ----- Per-entry evaluation -----
    # Accumulators for generic ranking metrics
    per_entry: Dict[str, List[float]] = defaultdict(list)
    # Requester-side: track which workers are recommended to each project
    project_rec_workers: Dict[int, List[int]] = defaultdict(list)
    recommended_projects: Set[int] = set()

    for entry in entry_list.entries:
        wid = entry.worker_id
        ts = entry.entry_created_at
        gt_pid = entry.project_id  # ground truth

        # Generate candidates
        candidates = candidate_fn(wid, ts, project_info, gt_pid, candidate_k)

        # Get model ranking
        ranked = model.recommend(wid, ts, candidates)

        # --- Generic ranking metrics ---
        gt_set = {gt_pid}
        for k in K_VALUES:
            per_entry[f"HR@{k}"].append(hit_rate_at_k(ranked, gt_pid, k))
            per_entry[f"NDCG@{k}"].append(ndcg_at_k(ranked, gt_pid, k))
            per_entry[f"Precision@{k}"].append(
                precision_at_k(ranked, gt_set, k)
            )
            per_entry[f"Recall@{k}"].append(recall_at_k(ranked, gt_set, k))
        per_entry["MRR"].append(mrr(ranked, gt_pid))

        # --- Worker-objective metrics ---
        w_fin = worker_finalist.get(wid, set())
        w_win = worker_winner.get(wid, set())
        w_cats = worker_pref_cats.get(wid, set())
        proj_cat = {
            pid: info.get("category", -1) for pid, info in project_info.items()
        }
        for k in K_VALUES:
            per_entry[f"avg_award_value@{k}"].append(
                avg_award_value_at_k(ranked, project_award, k)
            )
            per_entry[f"finalist_rate@{k}"].append(
                finalist_rate_at_k(ranked, w_fin, k)
            )
            per_entry[f"winner_rate@{k}"].append(
                winner_rate_at_k(ranked, w_win, k)
            )
            per_entry[f"category_match_rate@{k}"].append(
                category_match_rate_at_k(ranked, proj_cat, w_cats, k)
            )

        # --- Requester-side accumulation ---
        for pid in ranked[:max(K_VALUES)]:
            project_rec_workers[pid].append(wid)
            recommended_projects.add(pid)

    # ----- Aggregate -----
    results: Dict[str, float] = {}

    for metric_name, values in per_entry.items():
        results[metric_name] = float(np.mean(values))
        if n_bootstrap > 0:
            lo, hi = _bootstrap_ci(values, n_bootstrap=n_bootstrap)
            results[f"{metric_name}_ci_lower"] = lo
            results[f"{metric_name}_ci_upper"] = hi

    # Requester-objective metrics (aggregated, not per-entry)
    results["avg_recommender_worker_quality"] = avg_recommender_worker_quality(
        project_rec_workers, worker_quality, default_quality=wq_median
    )
    results["project_coverage"] = project_coverage(
        recommended_projects, active_projects
    )

    return results


# ---------------------------------------------------------------------------
# Convenience: evaluate and pretty-print
# ---------------------------------------------------------------------------


def evaluate_and_print(
    model: HasRecommend,
    split: Literal["train", "val", "test"],
    **kwargs,
) -> Dict[str, float]:
    """Run ``evaluate`` and print a formatted summary table."""
    results = evaluate(model, split, **kwargs)
    print(f"\n{'=' * 56}")
    print(f" Evaluation Results — split={split}")
    print(f"{'=' * 56}")

    # Group metrics
    generic = [k for k in results if any(
        k.startswith(p) for p in ("HR@", "NDCG@", "Precision@", "Recall@", "MRR")
    ) and "ci_" not in k]
    worker = [k for k in results if any(
        k.startswith(p) for p in (
            "avg_award_value@", "finalist_rate@",
            "winner_rate@", "category_match_rate@",
        )
    ) and "ci_" not in k]
    requester = [k for k in results if k in (
        "avg_recommender_worker_quality", "project_coverage",
    )]

    for group_name, keys in [
        ("Generic Ranking Metrics", sorted(generic)),
        ("Worker-Objective Metrics", sorted(worker)),
        ("Requester-Objective Metrics", sorted(requester)),
    ]:
        if keys:
            print(f"\n  {group_name}:")
            for k in keys:
                v = results[k]
                line = f"    {k:<35s} {v:.6f}"
                ci_lo = results.get(f"{k}_ci_lower")
                ci_hi = results.get(f"{k}_ci_upper")
                if ci_lo is not None and ci_hi is not None:
                    line += f"  [{ci_lo:.6f}, {ci_hi:.6f}]"
                print(line)

    print(f"\n{'=' * 56}\n")
    return results
