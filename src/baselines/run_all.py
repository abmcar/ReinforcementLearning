#!/usr/bin/env python3
"""Run all baseline recommenders and produce a comparison report.

Usage:
    python -m src.baselines.run_all [--split test] [--candidate-k 50] [--output docs/baselines.md]
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List

# Ensure project root on path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Metric groups for display
# ---------------------------------------------------------------------------
GENERIC_METRICS = [
    "HR@1", "HR@5", "HR@10",
    "NDCG@1", "NDCG@5", "NDCG@10",
    "MRR",
    "Precision@1", "Precision@5", "Precision@10",
    "Recall@1", "Recall@5", "Recall@10",
]

WORKER_METRICS = [
    "avg_award_value@1", "avg_award_value@5", "avg_award_value@10",
    "finalist_rate@1", "finalist_rate@5", "finalist_rate@10",
    "winner_rate@1", "winner_rate@5", "winner_rate@10",
    "category_match_rate@1", "category_match_rate@5", "category_match_rate@10",
]

REQUESTER_METRICS = [
    "avg_recommender_worker_quality",
    "project_coverage",
]


def _build_baselines(shared_data) -> List[tuple[str, object]]:
    """Instantiate all baselines with shared data."""
    from src.baselines.random import RandomRecommender
    from src.baselines.popularity import PopularityRecommender
    from src.baselines.category_match import CategoryMatchRecommender
    from src.baselines.quality_weighted import WorkerQualityWeightedRecommender

    return [
        ("Random", RandomRecommender(seed=42)),
        ("Popularity", PopularityRecommender(recency_days=60, shared_data=shared_data)),
        ("CategoryMatch", CategoryMatchRecommender(shared_data=shared_data)),
        ("QualityWeighted", WorkerQualityWeightedRecommender(
            recency_days=60, alpha=0.5, shared_data=shared_data
        )),
    ]


def _run_evaluation(
    split: str,
    candidate_k: int,
) -> Dict[str, Dict[str, float]]:
    """Run evaluate() for every baseline.  Returns {name: metrics_dict}."""
    from src.baselines.adapter import make_candidate_fn
    from src.baselines.data_loader import SharedData
    from src.eval.evaluator import evaluate

    # Load shared data once
    print("Loading shared data...")
    t0 = time.time()
    shared_data = SharedData.get()
    print(f"  shared data loaded in {time.time() - t0:.1f}s")

    # Build optimised candidate function (uses pre-indexed data)
    print("Building candidate function (fast mode)...")
    t0 = time.time()
    candidate_fn = make_candidate_fn(inject_ground_truth=True, use_fast=True)
    print(f"  candidate function ready in {time.time() - t0:.1f}s")
    baselines = _build_baselines(shared_data)

    all_results: Dict[str, Dict[str, float]] = {}

    for name, model in baselines:
        print(f"\n>>> Evaluating {name} on split={split} ...", flush=True)
        t0 = time.time()
        results = evaluate(
            model,  # type: ignore[arg-type]
            split,
            candidate_fn=candidate_fn,
            candidate_k=candidate_k,
        )
        elapsed = time.time() - t0
        print(f"    done in {elapsed:.1f}s")
        all_results[name] = results

    return all_results


def _print_table(
    all_results: Dict[str, Dict[str, float]],
    metrics: List[str],
    title: str,
) -> List[str]:
    """Pretty-print a metric group table, also return markdown lines."""
    names = list(all_results.keys())

    header = f"| Metric | {' | '.join(names)} |"
    sep = f"|{'---|' * (len(names) + 1)}"

    lines = [f"\n### {title}\n", header, sep]
    print(f"\n{title}")
    print(header)
    print(sep)

    for m in metrics:
        vals = [all_results[n].get(m, float("nan")) for n in names]
        row_vals = " | ".join(f"{v:.4f}" for v in vals)
        row = f"| {m} | {row_vals} |"
        lines.append(row)
        print(row)

    return lines


def _generate_markdown(
    all_results: Dict[str, Dict[str, float]],
    split: str,
    candidate_k: int,
) -> str:
    """Generate the full docs/baselines.md content."""
    md: List[str] = []

    md.append("# Baseline Results\n")
    md.append(f"- **Split**: {split}")
    md.append(f"- **Candidate K**: {candidate_k}")
    md.append(f"- **Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    md.append(f"- **Ground-truth injection**: enabled (see `src/baselines/adapter.py`)\n")

    md.append("## Baseline Descriptions\n")
    md.append("| Baseline | Description |")
    md.append("|----------|-------------|")
    md.append("| Random | Uniformly random shuffle of candidates (lower bound) |")
    md.append("| Popularity | Rank by recent entry count (global popularity, 60-day window) |")
    md.append("| CategoryMatch | Rank by worker's historical category affinity |")
    md.append("| QualityWeighted | Blend popularity with award density, weighted by worker quality (requester-oriented) |")
    md.append("")

    md.append("## Results\n")

    md.extend(
        _print_table(all_results, GENERIC_METRICS, "Generic Ranking Metrics")
    )
    md.append("")
    md.extend(
        _print_table(all_results, WORKER_METRICS, "Worker-Objective Metrics")
    )
    md.append("")
    md.extend(
        _print_table(all_results, REQUESTER_METRICS, "Requester-Objective Metrics")
    )
    md.append("")

    # Analysis section
    md.append("## Analysis\n")

    names = list(all_results.keys())
    md.append("### Key Findings\n")
    for group_name, metrics in [
        ("Generic Ranking", GENERIC_METRICS),
        ("Worker-Objective", WORKER_METRICS),
        ("Requester-Objective", REQUESTER_METRICS),
    ]:
        best_counts: Dict[str, int] = {n: 0 for n in names}
        for m in metrics:
            vals = {n: all_results[n].get(m, 0.0) for n in names}
            best_name = max(vals, key=lambda n: vals[n])
            best_counts[best_name] += 1

        leader = max(best_counts, key=lambda n: best_counts[n])
        md.append(
            f"- **{group_name}**: {leader} leads on "
            f"{best_counts[leader]}/{len(metrics)} metrics"
        )

    md.append("")
    md.append("### Notes\n")
    md.append(
        "- Random baseline establishes the lower bound. Any meaningful "
        "recommender should significantly outperform it."
    )
    md.append(
        "- All baselines use the same JOB-06 candidate set (K=50) with "
        "ground-truth injection for fair comparison."
    )
    md.append(
        "- Anti-leakage: all history-based baselines only use data "
        "strictly before each evaluation timestamp."
    )

    return "\n".join(md) + "\n"


def main():
    parser = argparse.ArgumentParser(description="Run all baselines")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--candidate-k", type=int, default=50)
    parser.add_argument("--output", default="docs/baselines.md")
    args = parser.parse_args()

    all_results = _run_evaluation(args.split, args.candidate_k)
    md_content = _generate_markdown(all_results, args.split, args.candidate_k)

    out_path = PROJECT_ROOT / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(md_content, encoding="utf-8")
    print(f"\nResults written to {out_path}")


if __name__ == "__main__":
    main()
