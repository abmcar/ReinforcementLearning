import json
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass
from typing import List, Tuple, Literal
from dateutil.parser import parse


# 定义数据结构约束
@dataclass
class Entry:
    project_id: int
    entry_number: int
    worker_id: int
    entry_created_at: datetime
    award_value: float
    finalist: bool
    winner: bool


@dataclass
class EntryList:
    entries: List[Entry]
    time_range: Tuple[datetime, datetime]


# 缓存文件路径
CACHE_FILE = Path(__file__).resolve().parent.parent.parent / "data" / "split_cache.json"


def _build_and_cache_splits():
    base_dir = Path(__file__).resolve().parent.parent.parent / "data"
    entry_dir = base_dir / "entry"

    txt_files = list(entry_dir.glob("*.txt"))
    if not txt_files:
        raise FileNotFoundError(f"找不到任何 txt 文件: {entry_dir}")

    all_entries = []

    for file_path in txt_files:
        project_id = 0
        try:
            parts = file_path.stem.split('_')
            if len(parts) >= 2 and parts[1].isdigit():
                project_id = int(parts[1])
        except Exception:
            pass

        with open(file_path, "r", encoding="utf-8", errors='ignore') as entry_file:
            try:
                content = entry_file.read()
                if not content.strip().startswith("{") and not content.strip().startswith("["):
                    continue
                text = json.loads(content)
            except json.JSONDecodeError:
                continue

            items = text.get("results", []) if isinstance(text, dict) else (text if isinstance(text, list) else [])

            for item in items:
                if not isinstance(item, dict):
                    continue
                try:
                    worker = item.get("worker", item.get("worker_id", 0))
                    created_at = item.get("entry_created_at", "2020-01-01T00:00:00Z")

                    all_entries.append({
                        "project_id": project_id,
                        "entry_number": int(item.get("entry_number", 0)),
                        "worker_id": int(worker) if worker else 0,
                        "entry_created_at": str(created_at),
                        "award_value": float(item.get("award_value", 0.0) or 0.0),
                        "finalist": bool(item.get("finalist", False)),
                        "winner": bool(item.get("winner", False))
                    })
                except Exception:
                    continue

    # 按时间戳严格排序
    all_entries.sort(key=lambda x: parse(x["entry_created_at"]))

    total = len(all_entries)
    train_end = int(total * 0.8)
    val_end = int(total * 0.9)

    splits = {
        "train": all_entries[:train_end],
        "val": all_entries[train_end:val_end],
        "test": all_entries[val_end:]
    }

    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(splits, f)
    return splits


def load_split(name: Literal["train", "val", "test"]) -> EntryList:
    if not CACHE_FILE.exists():
        splits = _build_and_cache_splits()
    else:
        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            splits = json.load(f)

    raw_data = splits.get(name)
    if not raw_data:
        raise ValueError(f"切片 {name} 为空！")

    entries = []
    for row in raw_data:
        entries.append(Entry(
            project_id=row["project_id"],
            entry_number=row["entry_number"],
            worker_id=row["worker_id"],
            entry_created_at=parse(row["entry_created_at"]),
            award_value=row["award_value"],
            finalist=row["finalist"],
            winner=row["winner"]
        ))
    time_range = (entries[0].entry_created_at, entries[-1].entry_created_at)
    return EntryList(entries=entries, time_range=time_range)


if __name__ == "__main__":
    for split_name in ["train", "val", "test"]:
        data = load_split(split_name)
        print(f"[{split_name.upper()}] 样本数: {len(data.entries)}, 时间跨度: {data.time_range[0]} -> {data.time_range[1]}")