"""Run worker/requester DQN smoke or final experiments."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
import sys
from statistics import mean, pstdev
from typing import Dict, Iterable, Literal

from src.baselines.adapter import make_candidate_fn
from src.eval.evaluator import evaluate
from src.rl.recommender import DQNRecommender
from src.rl.env import OfflineRecommendationEnv
from src.rl.trainer import (
    DQNTrainConfig,
    collect_training_transitions,
    train_offline_dqn,
)

BASE_DIR = Path(__file__).resolve().parent.parent.parent

KEY_METRICS = [
    "HR@1",
    "HR@5",
    "HR@10",
    "NDCG@10",
    "MRR",
    "avg_award_value@10",
    "finalist_rate@10",
    "winner_rate@10",
    "category_match_rate@10",
    "avg_recommender_worker_quality",
    "project_coverage",
]


def _display_path(path: object) -> str:
    if path is None:
        return ""
    value = Path(str(path))
    try:
        return str(value.resolve().relative_to(BASE_DIR))
    except ValueError:
        return str(value)


def _baseline_excerpt() -> list[str]:
    path = BASE_DIR / "docs" / "baselines.md"
    if not path.exists():
        return ["Baseline report not found at `docs/baselines.md`."]
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    keep: list[str] = []
    capture = False
    for line in lines:
        if line.startswith("### Generic Ranking Metrics"):
            capture = True
        if capture:
            keep.append(line)
        if capture and line.startswith("## Analysis"):
            break
    return keep[:-1] if keep and keep[-1].startswith("## Analysis") else keep


def _aggregate(results: list[dict[str, float]]) -> dict[str, float]:
    values: dict[str, list[float]] = defaultdict(list)
    for item in results:
        for key, value in item.items():
            values[key].append(float(value))
    out: dict[str, float] = {}
    for key, vals in values.items():
        out[f"{key}_mean"] = mean(vals)
        out[f"{key}_std"] = pstdev(vals) if len(vals) > 1 else 0.0
    return out


def run_objective(
    objective: Literal["worker", "requester"],
    *,
    seeds: Iterable[int],
    model_kind: Literal["dqn", "double_dqn", "dueling_dqn"],
    train_split: Literal["train", "val", "test"],
    max_transitions: int,
    epochs: int,
    max_steps: int | None,
    batch_size: int,
    candidate_k: int,
    eval_split: Literal["train", "val", "test"],
    max_eval_entries: int | None,
    cql_alpha: float,
    min_candidates: int,
) -> dict[str, object]:
    seed_metrics: list[dict[str, float]] = []
    train_diagnostics: list[dict[str, object]] = []
    raw_candidate_fn = make_candidate_fn(inject_ground_truth=True, use_fast=True)
    candidate_cache: dict[tuple[int, str, int, int], list[int]] = {}

    def candidate_fn(worker_id, timestamp, project_info, ground_truth_pid, k):
        key = (int(worker_id), timestamp.isoformat(), int(ground_truth_pid), int(k))
        cached = candidate_cache.get(key)
        if cached is None:
            cached = raw_candidate_fn(worker_id, timestamp, project_info, ground_truth_pid, k)
            candidate_cache[key] = cached
        return list(cached)

    feature_cache = {}
    print(
        f"[{objective}] preparing train transitions: split={train_split}, "
        f"max_transitions={max_transitions}, min_candidates={min_candidates}",
        flush=True,
    )
    train_env = OfflineRecommendationEnv(
        split=train_split,
        objective=objective,
        candidate_k=candidate_k,
    )
    train_transitions = collect_training_transitions(
        train_env, max_transitions, min_candidates
    )
    print(f"[{objective}] collected {len(train_transitions)} transitions", flush=True)
    eval_env = OfflineRecommendationEnv(
        split=eval_split,
        objective=objective,
        candidate_k=candidate_k,
        max_transitions=max_eval_entries,
    )

    for seed in seeds:
        print(f"[{objective}] training seed={seed}", flush=True)
        train_result = train_offline_dqn(
            DQNTrainConfig(
                objective=objective,
                split=train_split,
                model_kind=model_kind,
                candidate_k=candidate_k,
                max_transitions=max_transitions,
                epochs=epochs,
                max_steps=max_steps,
                batch_size=batch_size,
                cql_alpha=cql_alpha,
                seed=int(seed),
                min_candidates=min_candidates,
            ),
            env=train_env,
            transitions=train_transitions,
        )
        print(f"[{objective}] evaluating seed={seed}", flush=True)
        model = DQNRecommender(train_result.model, eval_env, feature_cache=feature_cache)
        metrics = evaluate(
            model,
            eval_split,
            candidate_fn=candidate_fn,
            candidate_k=candidate_k,
            max_entries=max_eval_entries,
        )
        seed_metrics.append(metrics)
        print(f"[{objective}] finished seed={seed}", flush=True)
        train_diagnostics.append(
            dict(train_result.diagnostics)
        )

    return {
        "objective": objective,
        "seed_metrics": seed_metrics,
        "summary": _aggregate(seed_metrics),
        "train_diagnostics": train_diagnostics,
    }


def _markdown(results: dict[str, object], command: str) -> str:
    generated = datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = [
        "# DQN Results",
        "",
        f"- **Generated**: {generated}",
        f"- **Command**: `{command}`",
        "- **Candidate source**: JOB-06 fast candidate adapter with ground-truth injection.",
        "- **Artifacts**: checkpoints and run diagnostics are under `outputs/dqn/` and are not committed.",
        "",
        "## Baseline Comparison",
        "",
        "The DQN runs use the same evaluator metric names and the same JOB-06",
        "candidate adapter policy as the JOB-05 baselines. Existing baseline",
        "results are reproduced below from `docs/baselines.md`.",
        "",
        *_baseline_excerpt(),
        "",
        "## Summary",
        "",
    ]
    for objective, payload in results.items():
        summary: Dict[str, float] = payload["summary"]  # type: ignore[index]
        lines.append(f"### {objective}")
        lines.append("")
        lines.append("| Metric | Mean | Std |")
        lines.append("|---|---:|---:|")
        for metric in KEY_METRICS:
            lines.append(
                f"| {metric} | {summary.get(metric + '_mean', 0.0):.6f} | "
                f"{summary.get(metric + '_std', 0.0):.6f} |"
            )
        lines.append("")
        lines.append("Training diagnostics:")
        lines.append("")
        lines.append("| Seed | Transitions | Final loss | Q max | CQL penalty |")
        lines.append("|---:|---:|---:|---:|---:|")
        for diag in payload["train_diagnostics"]:  # type: ignore[index]
            lines.append(
                f"| {diag.get('seed', '-')} | {diag['transitions']} | "
                f"{float(diag['final_loss']):.6f} | {float(diag['q_max']):.6f} | "
                f"{float(diag['cql_penalty_last']):.6f} |"
            )
        lines.append("")
    lines.extend([
        "## Ablation",
        "",
        "- Offline regularization is controlled by `--cql-alpha`. The checked",
        "  command records the alpha value used for the main run; setting",
        "  `--cql-alpha 0.0` runs the vanilla DQN ablation through the same",
        "  trainer and evaluator.",
        "",
        "## Notes",
        "",
        "- These runs use the same evaluator metric names as JOB-04.",
        "- The checked-in report records the bounded smoke configuration used during implementation.",
        "- Full-size final experiments can reuse the same scripts with larger `--max-transitions` and no `--max-eval-entries` cap.",
    ])
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run DQN dual-objective experiment")
    parser.add_argument("--objective", choices=["worker", "requester", "both"], default="both")
    parser.add_argument(
        "--model-kind",
        choices=["dqn", "double_dqn", "dueling_dqn"],
        default="dueling_dqn",
    )
    parser.add_argument("--seeds", default="42,43,44")
    parser.add_argument("--max-transitions", type=int, default=1000)
    parser.add_argument("--train-split", choices=["train", "val", "test"], default="train")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--candidate-k", type=int, default=50)
    parser.add_argument("--eval-split", choices=["train", "val", "test"], default="val")
    parser.add_argument("--max-eval-entries", type=int, default=1000)
    parser.add_argument("--cql-alpha", type=float, default=0.1)
    parser.add_argument("--min-candidates", type=int, default=1)
    parser.add_argument("--output", default="docs/dqn_results.md")
    parser.add_argument("--json-output", default=None)
    args = parser.parse_args()
    raw_args = vars(args).copy()
    if args.max_eval_entries is not None and args.max_eval_entries <= 0:
        args.max_eval_entries = None

    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
    objectives = ["worker", "requester"] if args.objective == "both" else [args.objective]
    results = {
        objective: run_objective(
            objective,  # type: ignore[arg-type]
            seeds=seeds,
            model_kind=args.model_kind,
            train_split=args.train_split,
            max_transitions=args.max_transitions,
            epochs=args.epochs,
            max_steps=args.max_steps,
            batch_size=args.batch_size,
            candidate_k=args.candidate_k,
            eval_split=args.eval_split,
            max_eval_entries=args.max_eval_entries,
            cql_alpha=args.cql_alpha,
            min_candidates=args.min_candidates,
        )
        for objective in objectives
    }

    output = BASE_DIR / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    command = f"{sys.executable} -m src.rl.run_experiment " + " ".join(
        f"--{key.replace('_', '-')} {value}"
        for key, value in raw_args.items()
    )
    output.write_text(_markdown(results, command), encoding="utf-8")
    if args.json_output:
        json_path = BASE_DIR / args.json_output
    else:
        json_path = BASE_DIR / "outputs" / "dqn" / "results" / f"{output.stem}.json"
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(results, indent=2, sort_keys=True), encoding="utf-8")
    print(f"wrote {output}")
    print(f"wrote {json_path}")


if __name__ == "__main__":
    main()
