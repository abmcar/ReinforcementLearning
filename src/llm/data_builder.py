"""Build binary SFT datasets for the crowdsourcing LLM baseline."""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Literal

import numpy as np
import pandas as pd
import yaml
from dateutil.parser import parse as parse_dt

from src.baselines.adapter import _FastCandidateGenerator
from src.data.split import Entry, load_split
from src.llm.prompts import build_binary_prompt

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs" / "llm_data"
CONFIG_PATH = BASE_DIR / "experiments" / "configs" / "llm_data.yaml"
SUMMARY_PATH = OUTPUT_DIR / "summary.json"

SplitName = Literal["train", "val", "test"]
ObjectiveName = Literal["worker", "requester"]


@dataclass
class BuilderConfig:
    candidate_k: int = 50
    negatives_per_positive: int = 2
    max_history_items: int = 5
    max_brief_items: int = 4
    worker_award_quantile: float = 0.75
    requester_quality_quantile: float = 0.75
    seed: int = 42
    output_dir: str = "outputs/llm_data"


@dataclass
class DatasetStats:
    objective: str
    split: str
    samples_total: int
    positives: int
    negatives: int
    positive_events_seen: int
    positive_events_covered: int
    skipped_positive_events: int
    average_prompt_tokens: float
    p50_prompt_tokens: int
    p90_prompt_tokens: int
    p99_prompt_tokens: int
    max_prompt_tokens: int
    output_path: str


def _estimate_token_count(text: str) -> int:
    return len(re.findall(r"\w+|[^\w\s]", text))


def _format_float(value: float) -> str:
    return f"{value:.4f}"


def _load_config(path: Path = CONFIG_PATH) -> BuilderConfig:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    return BuilderConfig(**raw)


def _load_worker_quality() -> tuple[dict[int, float], float]:
    wq_file = DATA_DIR / "worker_quality.csv"
    worker_quality: dict[int, float] = {}
    if wq_file.exists():
        df = pd.read_csv(wq_file)
        df = df[df["worker_quality"] > 0]
        worker_quality = dict(zip(df["worker_id"], df["worker_quality"] / 100.0))
    median = float(np.median(list(worker_quality.values()))) if worker_quality else 0.5
    return worker_quality, median


def _load_project_text() -> dict[int, dict]:
    project_dir = DATA_DIR / "project"
    project_info: dict[int, dict] = {}
    for txt_file in project_dir.glob("project_*.txt"):
        try:
            pid = int(txt_file.stem.split("_")[1])
            with open(txt_file, "r", encoding="utf-8", errors="ignore") as f:
                data = json.load(f)
        except Exception:
            continue

        start_date = parse_dt(data["start_date"]) if data.get("start_date") else None
        deadline = parse_dt(data["deadline"]) if data.get("deadline") else None
        project_info[pid] = {
            "title": str(data.get("title") or f"Project {pid}"),
            "category": int(data.get("category", 0) or 0),
            "sub_category": int(data.get("sub_category", 0) or 0),
            "industry": str(data.get("industry") or "Unknown"),
            "package_name": str(data.get("package_name") or "Unknown"),
            "total_awards": float(data.get("total_awards", 0.0) or 0.0),
            "participants": data.get("participants") or [],
            "brief_questions": data.get("brief_questions") or [],
            "brief_answers": data.get("brief_answers") or {},
            "start_date": start_date,
            "deadline": deadline,
        }
    return project_info


def _safe_text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        return ", ".join(_safe_text(item) for item in value if _safe_text(item))
    if isinstance(value, dict):
        parts = []
        for key, item in value.items():
            text = _safe_text(item)
            if text:
                parts.append(f"{key}: {text}")
        return "; ".join(parts)
    text = str(value).replace("\n", " ").strip()
    return re.sub(r"\s+", " ", text)


def _project_brief_items(project: dict, max_items: int) -> list[str]:
    question_labels: dict[str, str] = {}
    for question in project.get("brief_questions", []):
        name = str(question.get("name") or "").strip()
        label = str(question.get("answer_label") or question.get("label") or name).strip()
        if name:
            question_labels[name] = label

    items: list[str] = []
    answers = project.get("brief_answers")
    if isinstance(answers, dict):
        for key, value in answers.items():
            text = _safe_text(value)
            if not text:
                continue
            label = question_labels.get(str(key), str(key))
            items.append(f"{label}: {text[:220]}")
            if len(items) >= max_items:
                break

    if not items and project.get("title"):
        items.append(f"Title summary: {project['title']}")
    return items


def _history_outcome(entry: dict) -> str:
    if entry["winner"]:
        return "winner"
    if entry["finalist"]:
        return "finalist"
    if float(entry["award_value"] or 0.0) > 0:
        return f"award={float(entry['award_value']):.2f}"
    return "participated"


def _collect_top_categories(history: Iterable[dict], project_info: dict[int, dict]) -> str:
    counts: Counter[int] = Counter()
    for item in history:
        category = int(project_info.get(item["project_id"], {}).get("category", 0))
        counts[category] += 1
    if not counts:
        return "None"
    return ", ".join(str(cat) for cat, _ in counts.most_common(3))


def _is_worker_positive(entry: dict, award_threshold: float) -> bool:
    award_value = float(entry.get("award_value", 0.0) or 0.0)
    return bool(entry.get("winner") or entry.get("finalist") or award_value >= award_threshold)


def _is_requester_positive(entry: dict, worker_quality: float, quality_threshold: float) -> bool:
    return worker_quality >= quality_threshold


def _compute_thresholds(
    worker_quality_map: dict[int, float],
    quality_default: float,
    config: BuilderConfig,
) -> dict[str, float]:
    train_entries = load_split("train").entries
    award_values = np.array(
        [entry.award_value for entry in train_entries if entry.award_value > 0],
        dtype=float,
    )
    if len(award_values) == 0:
        worker_award_threshold = 0.0
    else:
        worker_award_threshold = float(
            np.quantile(award_values, config.worker_award_quantile)
        )

    requester_qualities = np.array(
        [worker_quality_map.get(entry.worker_id, quality_default) for entry in train_entries],
        dtype=float,
    )
    requester_quality_threshold = float(
        np.quantile(requester_qualities, config.requester_quality_quantile)
    )

    return {
        "worker_award_threshold": worker_award_threshold,
        "requester_quality_threshold": requester_quality_threshold,
    }


class LLMDataBuilder:
    """Builds binary prompt-response samples for one objective and split."""

    def __init__(self, config: BuilderConfig | None = None):
        self.config = config or _load_config()
        self.output_dir = BASE_DIR / self.config.output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.worker_quality, self.worker_quality_default = _load_worker_quality()
        self.project_info = _load_project_text()
        self.thresholds = _compute_thresholds(
            self.worker_quality,
            self.worker_quality_default,
            self.config,
        )
        self.candidate_generator = _FastCandidateGenerator()
        self.rng = random.Random(self.config.seed)
        self.entries = self._all_entries()
        self.ranges = self._build_index_ranges(self.entries)

    def _all_entries(self) -> list[dict]:
        entries: list[dict] = []
        for split in ("train", "val", "test"):
            for entry in load_split(split).entries:
                entries.append(
                    {
                        "project_id": entry.project_id,
                        "worker_id": entry.worker_id,
                        "entry_created_at": entry.entry_created_at,
                        "award_value": float(entry.award_value),
                        "finalist": bool(entry.finalist),
                        "winner": bool(entry.winner),
                    }
                )
        entries.sort(key=lambda item: item["entry_created_at"])
        return entries

    def _build_index_ranges(self, entries: list[dict]) -> dict[str, tuple[int, int]]:
        train_len = len(load_split("train").entries)
        val_len = len(load_split("val").entries)
        test_len = len(load_split("test").entries)
        assert train_len + val_len + test_len == len(entries)
        return {
            "train": (0, train_len),
            "val": (train_len, train_len + val_len),
            "test": (train_len + val_len, len(entries)),
        }

    def _worker_profile(
        self,
        worker_id: int,
        history: list[dict],
    ) -> dict[str, str]:
        wins = sum(1 for item in history if item["winner"])
        total_award = sum(float(item["award_value"] or 0.0) for item in history)
        entries_count = len(history)
        top_categories = _collect_top_categories(history, self.project_info)
        return {
            "worker_quality": _format_float(
                self.worker_quality.get(worker_id, self.worker_quality_default)
            ),
            "hist_entries": str(entries_count),
            "hist_wins": str(wins),
            "hist_win_rate": _format_float(wins / entries_count if entries_count else 0.0),
            "hist_avg_award": _format_float(
                total_award / entries_count if entries_count else 0.0
            ),
            "top_categories": top_categories,
        }

    def _worker_history_items(self, history: list[dict]) -> list[dict[str, str]]:
        items: list[dict[str, str]] = []
        for item in history[-self.config.max_history_items :][::-1]:
            project = self.project_info.get(item["project_id"], {})
            items.append(
                {
                    "title": str(project.get("title") or f"Project {item['project_id']}"),
                    "category": str(project.get("category", 0)),
                    "industry": str(project.get("industry", "Unknown")),
                    "outcome": _history_outcome(item),
                }
            )
        return items

    def _project_profile(self, project_id: int, timestamp: datetime) -> dict[str, object]:
        project = self.project_info.get(project_id, {})
        deadline = project.get("deadline")
        days_until_deadline = 0.0
        if isinstance(deadline, datetime):
            days_until_deadline = (deadline - timestamp).total_seconds() / 86400.0
        participants = project.get("participants") or []
        return {
            "title": str(project.get("title") or f"Project {project_id}"),
            "category": str(project.get("category", 0)),
            "sub_category": str(project.get("sub_category", 0)),
            "industry": str(project.get("industry") or "Unknown"),
            "package_name": str(project.get("package_name") or "Unknown"),
            "total_awards": _format_float(float(project.get("total_awards", 0.0) or 0.0)),
            "days_until_deadline": _format_float(days_until_deadline),
            "participants_count": str(len(participants) if isinstance(participants, list) else 0),
            "brief_items": _project_brief_items(project, self.config.max_brief_items),
        }

    def _make_sample(
        self,
        *,
        objective: ObjectiveName,
        label: int,
        worker_id: int,
        project_id: int,
        timestamp: datetime,
        history: list[dict],
        source: str,
    ) -> dict[str, object]:
        worker_profile = self._worker_profile(worker_id, history)
        worker_history = self._worker_history_items(history)
        project_profile = self._project_profile(project_id, timestamp)
        prompt = build_binary_prompt(
            objective=objective,
            worker_profile=worker_profile,
            worker_history=worker_history,
            project_profile=project_profile,
        )
        response = "Yes" if label == 1 else "No"
        token_count = _estimate_token_count(prompt["prompt"])
        return {
            "objective": objective,
            "label": label,
            "response": response,
            "worker_id": worker_id,
            "project_id": project_id,
            "timestamp": timestamp.isoformat(),
            "source": source,
            "system": prompt["system"],
            "user": prompt["user"],
            "prompt": prompt["prompt"],
            "prompt_tokens_estimate": token_count,
        }

    def build(self, objective: ObjectiveName, split: SplitName) -> Path:
        objective = objective.lower()  # type: ignore[assignment]
        split = split.lower()  # type: ignore[assignment]
        if objective not in {"worker", "requester"}:
            raise ValueError("objective must be 'worker' or 'requester'")
        if split not in {"train", "val", "test"}:
            raise ValueError("split must be 'train', 'val', or 'test'")

        start_idx, end_idx = self.ranges[split]
        output_path = self.output_dir / f"{objective}_{split}.jsonl"

        worker_history: dict[int, list[dict]] = defaultdict(list)
        samples: list[dict[str, object]] = []

        positive_events_seen = 0
        positive_events_covered = 0

        for idx, entry in enumerate(self.entries):
            worker_id = int(entry["worker_id"])
            history = worker_history[worker_id]

            if start_idx <= idx < end_idx:
                timestamp = entry["entry_created_at"]
                project_id = int(entry["project_id"])
                worker_quality = self.worker_quality.get(worker_id, self.worker_quality_default)

                if objective == "worker":
                    is_positive = _is_worker_positive(
                        entry,
                        self.thresholds["worker_award_threshold"],
                    )
                else:
                    is_positive = _is_requester_positive(
                        entry,
                        worker_quality,
                        self.thresholds["requester_quality_threshold"],
                    )

                if is_positive:
                    positive_events_seen += 1
                    candidates = self.candidate_generator.get_candidates(
                        worker_id,
                        timestamp,
                        K=self.config.candidate_k,
                    )
                    if project_id in candidates:
                        positive_events_covered += 1
                        samples.append(
                            self._make_sample(
                                objective=objective,
                                label=1,
                                worker_id=worker_id,
                                project_id=project_id,
                                timestamp=timestamp,
                                history=history,
                                source="observed_positive",
                            )
                        )

                        negative_pool = [pid for pid in candidates if pid != project_id]
                        if negative_pool:
                            negative_count = min(
                                self.config.negatives_per_positive,
                                len(negative_pool),
                            )
                            for negative_pid in self.rng.sample(negative_pool, negative_count):
                                samples.append(
                                    self._make_sample(
                                        objective=objective,
                                        label=0,
                                        worker_id=worker_id,
                                        project_id=negative_pid,
                                        timestamp=timestamp,
                                        history=history,
                                        source="candidate_negative",
                                    )
                                )

            worker_history[worker_id].append(entry)

        with open(output_path, "w", encoding="utf-8") as f:
            for sample in samples:
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")

        stats = self._summarize_samples(
            samples=samples,
            objective=objective,
            split=split,
            positive_events_seen=positive_events_seen,
            positive_events_covered=positive_events_covered,
            output_path=output_path,
        )
        self._write_summary(stats)
        return output_path

    def _summarize_samples(
        self,
        *,
        samples: list[dict[str, object]],
        objective: str,
        split: str,
        positive_events_seen: int,
        positive_events_covered: int,
        output_path: Path,
    ) -> DatasetStats:
        token_counts = [int(sample["prompt_tokens_estimate"]) for sample in samples] or [0]
        positives = sum(1 for sample in samples if int(sample["label"]) == 1)
        negatives = len(samples) - positives
        return DatasetStats(
            objective=objective,
            split=split,
            samples_total=len(samples),
            positives=positives,
            negatives=negatives,
            positive_events_seen=positive_events_seen,
            positive_events_covered=positive_events_covered,
            skipped_positive_events=positive_events_seen - positive_events_covered,
            average_prompt_tokens=round(float(np.mean(token_counts)), 2),
            p50_prompt_tokens=int(np.percentile(token_counts, 50)),
            p90_prompt_tokens=int(np.percentile(token_counts, 90)),
            p99_prompt_tokens=int(np.percentile(token_counts, 99)),
            max_prompt_tokens=int(max(token_counts)),
            output_path=str(output_path.relative_to(BASE_DIR)),
        )

    def _write_summary(self, stats: DatasetStats) -> None:
        summary: dict[str, dict[str, object]] = {}
        if SUMMARY_PATH.exists():
            with open(SUMMARY_PATH, "r", encoding="utf-8") as f:
                summary = json.load(f)
        summary.setdefault(stats.objective, {})
        summary[stats.objective][stats.split] = asdict(stats)
        summary["thresholds"] = self.thresholds
        summary["config"] = asdict(self.config)
        with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)


def build_dataset(objective: ObjectiveName, split: SplitName) -> Path:
    """Build one binary jsonl dataset split and return its output path."""
    builder = LLMDataBuilder()
    return builder.build(objective, split)


def _cli() -> None:
    parser = argparse.ArgumentParser(description="Build JOB-11 LLM jsonl datasets.")
    parser.add_argument("--objective", choices=["worker", "requester", "all"], required=True)
    parser.add_argument("--split", choices=["train", "val", "test", "all"], required=True)
    args = parser.parse_args()

    builder = LLMDataBuilder()
    objectives = ["worker", "requester"] if args.objective == "all" else [args.objective]
    splits = ["train", "val", "test"] if args.split == "all" else [args.split]

    for objective in objectives:
        for split in splits:
            path = builder.build(objective, split)
            print(path)


if __name__ == "__main__":
    _cli()
