import json
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass
from typing import List, Tuple, Literal
from dateutil.parser import parse


@dataclass
class Entry:
    project_id: int
    entry_number: int
    worker_id: int  # 真实的工人 ID
    entry_created_at: datetime
    award_value: float
    finalist: bool
    winner: bool


@dataclass
class EntryList:
    entries: List[Entry]
    time_range: Tuple[datetime, datetime]


CACHE_FILE = Path(__file__).resolve().parent.parent.parent / "data" / "split_cache.json"


def _build_and_cache_splits():
    base_dir = Path(__file__).resolve().parent.parent.parent / "data"
    entry_dir = base_dir / "entry"
    txt_files = list(entry_dir.glob("*.txt"))

    all_entries = []
    for file_path in txt_files:
        project_id = 0
        try:
            project_id = int(file_path.stem.split('_')[1])
        except:
            pass

        with open(file_path, "r", encoding="utf-8", errors='ignore') as f:
            try:
                text = json.loads(f.read())
                items = text.get("results", [])
            except:
                continue

            for item in items:
                # 【核心修复】：将错误的 worker 改为真正的 author
                worker_id = item.get("author")

                # 如果没读到真实的作者ID，说明是废数据，跳过
                if worker_id is None: continue

                all_entries.append({
                    "project_id": project_id,
                    "entry_number": int(item.get("entry_number", 0)),
                    "worker_id": int(worker_id),
                    "entry_created_at": str(item.get("entry_created_at")),
                    "award_value": float(item.get("award_value", 0.0) or 0.0),
                    "finalist": bool(item.get("finalist", False)),
                    "winner": bool(item.get("winner", False))
                })

    all_entries.sort(key=lambda x: parse(x["entry_created_at"]))

    total = len(all_entries)
    splits = {
        "train": all_entries[:int(total * 0.8)],
        "val": all_entries[int(total * 0.8):int(total * 0.9)],
        "test": all_entries[int(total * 0.9):]
    }

    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(splits, f)
    return splits


def load_split(name):
    if not CACHE_FILE.exists():
        splits = _build_and_cache_splits()
    else:
        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            splits = json.load(f)

    raw_data = splits[name]
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
    _build_and_cache_splits()
    print("✅ split_cache.json 已重写，真实的 author_id 提取成功！")