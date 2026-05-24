"""Evaluate candidate generation Recall@K on the test split.

Uses an incremental approach for efficiency: pre-builds popularity
indexes and scans the test set in chronological order.

Usage:
    python3 scripts/eval_recall.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from collections import defaultdict
from datetime import timedelta
from dateutil.parser import parse as parse_dt

from src.data.split import load_split
from src.candidates.recall import (
    load_project_metadata,
    load_entry_history,
    get_active_projects,
)


def evaluate_recall(K_values: list[int] | None = None):
    if K_values is None:
        K_values = [20, 50, 100, 200]

    max_k = max(K_values)
    recency_days = 60
    pop_weight = 0.7
    cat_weight = 0.3

    print("Loading project metadata...")
    project_meta = load_project_metadata()
    print(f"  Projects: {len(project_meta)}")

    print("Loading entry history...")
    all_entries = load_entry_history()
    print(f"  Total entries: {len(all_entries)}")

    print("Loading test split for ground truth...")
    test_data = load_split("test")
    test_entries = test_data.entries
    print(f"  Test entries: {len(test_entries)}")
    print(f"  Time range: {test_data.time_range[0]} ~ {test_data.time_range[1]}")

    # Pre-compute: for each entry in all_entries, determine which split it belongs to
    # by checking index ranges
    train_split = load_split("train")
    val_split = load_split("val")
    train_end_idx = len(train_split.entries)
    val_end_idx = train_end_idx + len(val_split.entries)

    # Build worker category history incrementally
    # worker_cat_hist[wid][cat] = count of entries in that category
    worker_cat_hist: dict[int, dict[int, int]] = defaultdict(lambda: defaultdict(int))
    # project_recent_entries[pid] = list of timestamps (for recency scoring)
    project_entry_ts: dict[int, list] = defaultdict(list)

    # First, replay all pre-test entries to build the baseline state
    print("Replaying pre-test history to build indexes...")
    for entry in all_entries[:val_end_idx]:
        wid = entry["worker_id"]
        pid = entry["project_id"]
        ts = entry["_parsed_ts"]

        cat = project_meta.get(pid, {}).get("category", -1)
        if cat >= 0:
            worker_cat_hist[wid][cat] += 1

        project_entry_ts[pid].append(ts)

    # Now evaluate on test entries
    hits = {k: 0 for k in K_values}
    total = 0
    cold_start_count = 0

    # Track which workers have been seen before the current test entry
    known_workers = set()
    for entry in all_entries[:val_end_idx]:
        known_workers.add(entry["worker_id"])

    test_entry_idx = val_end_idx  # pointer into all_entries for incremental update

    print(f"Evaluating Recall@K on {len(test_entries)} test entries...")
    for i, test_entry in enumerate(test_entries):
        gt_project = test_entry.project_id
        wid = test_entry.worker_id
        ts = test_entry.entry_created_at

        # Update indexes with any entries between the last test entry and this one
        # (entries that came before this test entry in time but are also test entries)
        while test_entry_idx < len(all_entries):
            e = all_entries[test_entry_idx]
            e_ts = e["_parsed_ts"]
            if e_ts > ts:
                break
            # Only update if this entry is strictly before the current timestamp
            # (same-timestamp entries should not be leaked)
            if e_ts < ts:
                e_wid = e["worker_id"]
                e_pid = e["project_id"]
                known_workers.add(e_wid)
                cat = project_meta.get(e_pid, {}).get("category", -1)
                if cat >= 0:
                    worker_cat_hist[e_wid][cat] += 1
                project_entry_ts[e_pid].append(e_ts)
                test_entry_idx += 1
            else:
                break

        # 1. Get active projects
        active = get_active_projects(project_meta, ts)
        if not active:
            total += 1
            continue

        # 2. Popularity scores (recent entry count)
        cutoff = ts - timedelta(days=recency_days)
        pop_scores: dict[int, float] = {}
        for pid in active:
            count = 0
            for t in project_entry_ts.get(pid, []):
                if cutoff <= t <= ts:
                    count += 1
            pop_scores[pid] = float(count)

        max_pop = max(pop_scores.values(), default=1.0)
        if max_pop > 0:
            pop_scores = {pid: s / max_pop for pid, s in pop_scores.items()}

        # 3. Category scores (for non-cold-start workers)
        is_cold = wid not in known_workers
        if is_cold:
            cold_start_count += 1
            merged = pop_scores
        else:
            cat_scores: dict[int, float] = {}
            wid_cats = worker_cat_hist.get(wid, {})
            for pid in active:
                cat = project_meta.get(pid, {}).get("category", -1)
                cat_scores[pid] = float(wid_cats.get(cat, 0))

            max_cat = max(cat_scores.values(), default=1.0)
            if max_cat > 0:
                cat_scores = {pid: s / max_cat for pid, s in cat_scores.items()}

            merged = {}
            for pid in active:
                merged[pid] = (
                    pop_weight * pop_scores.get(pid, 0.0)
                    + cat_weight * cat_scores.get(pid, 0.0)
                )

        # 4. Rank and check hits
        ranked = sorted(merged.items(), key=lambda x: x[1], reverse=True)
        candidate_ids = [pid for pid, _ in ranked[:max_k]]

        for k in K_values:
            if gt_project in candidate_ids[:k]:
                hits[k] += 1

        total += 1
        if (i + 1) % 500 == 0:
            print(f"  Processed {i + 1}/{len(test_entries)} entries...")

    print(f"\nTotal test entries evaluated: {total}")
    print(f"Cold-start workers: {cold_start_count} ({100 * cold_start_count / max(total, 1):.1f}%)\n")

    print("=" * 50)
    print(f"{'K':>6} | {'Recall@K':>10} | {'Hits':>8} / {'Total':>8}")
    print("-" * 50)
    for k in K_values:
        recall = hits[k] / total if total > 0 else 0.0
        print(f"{k:>6} | {recall:>10.4f} | {hits[k]:>8} / {total:>8}")
    print("=" * 50)

    return {k: hits[k] / total if total > 0 else 0.0 for k in K_values}


if __name__ == "__main__":
    evaluate_recall()
