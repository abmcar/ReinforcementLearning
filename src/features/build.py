import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from dateutil.parser import parse
import warnings

warnings.filterwarnings('ignore', category=FutureWarning)

# 定义基础路径
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs" / "features"
CACHE_FILE = DATA_DIR / "split_cache.json"

# 注意：OUTPUT_DIR 在函数内部按需创建，避免 import 时的副作用


def load_static_data():
    """加载不随时间变化的静态字典 (任务元数据与工人静态质量)"""
    print("📦 正在加载静态元数据 (工人质量 & 任务信息)...")

    # 1. 加载 Worker Quality
    worker_quality = {}
    wq_file = DATA_DIR / "worker_quality.csv"
    if wq_file.exists():
        # 【修复点】：去掉 header=None，直接读取，Pandas会自动识别第一行是表头
        df_wq = pd.read_csv(wq_file)
        # 过滤掉异常值 (-1) 并归一化到 [0,1]
        df_wq = df_wq[df_wq['worker_quality'] > 0]
        worker_quality = dict(zip(df_wq['worker_id'], df_wq['worker_quality'] / 100.0))

    # 冷启动备用：全局中位数
    global_wq_median = np.median(list(worker_quality.values())) if worker_quality else 0.5

    # 2. 加载 Project 静态特征 (直接扫项目文件夹)
    project_info = {}
    project_dir = DATA_DIR / "project"
    if project_dir.exists():
        for txt_file in project_dir.glob("project_*.txt"):
            try:
                pid = int(txt_file.stem.split('_')[1])
                with open(txt_file, 'r', encoding='utf-8', errors='ignore') as f:
                    data = json.loads(f.read())
                    raw_start = data.get("start_date")
                    raw_deadline = data.get("deadline")
                    if not raw_start or not raw_deadline:
                        print(f"  WARNING: project_{pid} missing start_date or deadline, skipped")
                        continue
                    start_date = parse(raw_start)
                    deadline = parse(raw_deadline)
                    duration = (deadline - start_date).total_seconds() / (24 * 3600)

                    project_info[pid] = {
                        "category": int(data.get("category", 0)),
                        "sub_category": int(data.get("sub_category", 0)),
                        "industry": str(data.get("industry") or "Unknown"),
                        "start_date": start_date,
                        "deadline": deadline,
                        "duration_days": max(duration, 0.0)
                    }
            except Exception as e:
                print(f"  WARNING: Failed to load {txt_file.stem}: {e}")
                continue

    return worker_quality, global_wq_median, project_info


def build_features(target_split: str):
    """核心推土机：构建无穿越的动态特征"""
    if target_split not in ["train", "val", "test"]:
        raise ValueError("切片名称必须是 train, val 或 test")

    print(f"\n🚀 开始构建【{target_split.upper()}】集特征矩阵...")

    if not CACHE_FILE.exists():
        raise FileNotFoundError("找不到 split_cache.json，请检查 JOB-02 是否完成！")

    with open(CACHE_FILE, "r", encoding="utf-8") as f:
        splits = json.load(f)

    # 核心防穿越逻辑
    replay_data = []
    if target_split == "train":
        replay_data = splits["train"]
        extract_start_idx = 0
    elif target_split == "val":
        replay_data = splits["train"] + splits["val"]
        extract_start_idx = len(splits["train"])
    elif target_split == "test":
        replay_data = splits["train"] + splits["val"] + splits["test"]
        extract_start_idx = len(splits["train"]) + len(splits["val"])

    worker_quality, global_wq_median, project_info = load_static_data()

    worker_stats = {}  # {worker_id: {'entries': 0, 'wins': 0, 'total_award': 0.0}}
    project_stats = {}  # {project_id: {'current_entries': 0}}

    features_list = []

    print(f"⏳ 正在启动时光机进行时序回放，严防信息泄露 (共需回放 {len(replay_data)} 条记录)...")

    for i, row in enumerate(replay_data):
        wid = row["worker_id"]
        pid = row["project_id"]
        created_at = parse(row["entry_created_at"])

        # 动作一：提取当下的特征
        if i >= extract_start_idx:
            w_stat = worker_stats.get(wid, {'entries': 0, 'wins': 0, 'total_award': 0.0})
            w_entries = w_stat['entries']
            w_wins = w_stat['wins']
            w_award = w_stat['total_award']

            p_stat = project_stats.get(pid, {'current_entries': 0})
            p_entries = p_stat['current_entries']

            p_info = project_info.get(pid, {})
            deadline = p_info.get("deadline", created_at)
            days_remaining = (deadline - created_at).total_seconds() / (24 * 3600)

            features_list.append({
                "worker_id": wid,
                "project_id": pid,
                "entry_created_at": str(created_at),

                "label_winner": int(row["winner"]),
                "label_finalist": int(row["finalist"]),
                "label_award": float(row["award_value"]),

                "worker_quality": worker_quality.get(wid, global_wq_median),
                "hist_entries": w_entries,
                "hist_wins": w_wins,
                "hist_win_rate": w_wins / w_entries if w_entries > 0 else 0.0,
                "hist_avg_award": w_award / w_entries if w_entries > 0 else 0.0,

                "category": p_info.get("category", 0),
                "sub_category": p_info.get("sub_category", 0),
                "industry": p_info.get("industry", "Unknown"),
                "duration_days": p_info.get("duration_days", 0.0),
                "days_remaining": days_remaining,
                "current_entries": p_entries
            })

        # 动作二：更新记忆字典
        if wid not in worker_stats:
            worker_stats[wid] = {'entries': 0, 'wins': 0, 'total_award': 0.0}
        worker_stats[wid]['entries'] += 1
        if row["winner"]:
            worker_stats[wid]['wins'] += 1
        worker_stats[wid]['total_award'] += float(row["award_value"])

        if pid not in project_stats:
            project_stats[pid] = {'current_entries': 0}
        project_stats[pid]['current_entries'] += 1

    df = pd.DataFrame(features_list)

    # Industry Label Encoding：在 train 集上拟合映射表，val/test 复用，
    # 确保同一 industry 在所有 split 中编码一致
    industry_map_file = OUTPUT_DIR / "industry_label_map.json"
    if target_split == "train":
        # 从 train 集建立映射表并持久化
        unique_industries = sorted(df['industry'].unique())
        industry_map = {ind: code for code, ind in enumerate(unique_industries)}
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        with open(industry_map_file, "w", encoding="utf-8") as f:
            json.dump(industry_map, f, ensure_ascii=False)
        print(f"📋 已保存 industry 映射表 ({len(industry_map)} 个类别)")
    else:
        # val/test 复用 train 集的映射表
        if not industry_map_file.exists():
            raise FileNotFoundError(
                "找不到 industry_label_map.json，请先构建 train 集特征！"
            )
        with open(industry_map_file, "r", encoding="utf-8") as f:
            industry_map = json.load(f)

    # 未见过的 industry 统一编码为 -1
    df['industry'] = df['industry'].map(industry_map).fillna(-1).astype(int)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_file = OUTPUT_DIR / f"{target_split}_features.csv"
    df.to_csv(out_file, index=False)

    print(f"✅ 【{target_split.upper()}】集特征构建完成！样本数: {len(df)}, 特征维度: {df.shape[1]}")
    return df


if __name__ == "__main__":
    for split in ["train", "val", "test"]:
        df_split = build_features(split)

        if split == "train":
            print("\n👀 偷偷预览一下 Train 集前两行极其丰富的特征：")
            print(df_split.head(2).T)
            print("-" * 50)
