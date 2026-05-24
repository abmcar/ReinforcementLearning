# JOB-06: 候选集生成模块说明

## 1. 模块概述

候选集生成模块是 DQN 主线和 LLM+SFT 主线的**共享前置模块**，负责将约 5300 个 project 的全量空间预召回至 Top-K 候选集（默认 K=50），为下游排序阶段（DQN / LLM）大幅缩减动作空间。

### 接口签名

```python
get_candidates(worker_id: int, timestamp: datetime, K: int = 50) -> list[int]
```

签名严格遵循 `docs/roadmap.md` § 6.5 接口契约表。

## 2. 召回策略

采用**两路召回 + 加权合并**的架构：

### 2.1 热度召回（Popularity Recall）

- 统计每个 active project 在 `[timestamp - 60天, timestamp]` 时间窗内收到的 entry 数量
- 按 entry 数量降序排列
- 权重：**0.7**

### 2.2 类目匹配召回（Category Recall）

- 统计目标 worker 在 `timestamp` 之前的历史参与中，各 category 的参与次数
- 对每个 active project，用其 category 对应的 worker 参与次数作为得分
- 权重：**0.3**

### 2.3 合并策略

1. 两路召回得分分别归一化到 `[0, 1]`
2. 加权求和：`score = 0.7 * pop_norm + 0.3 * cat_norm`
3. 按合并得分降序截取 Top-K

## 3. 冷启动处理

遵循 `docs/data_split.md` § 2 冷启动协议：

| 场景 | 处理方式 |
|------|----------|
| **冷启动 Worker**（timestamp 前无历史 entry） | 跳过类目召回，退化为纯热度召回 |
| **冷启动 Project**（刚发布的新 project） | 只要满足 `start_date <= t <= deadline` 即进入候选池，热度得分为 0 但仍可被类目匹配召回 |
| **空 Active Project**（timestamp 时刻无任何 active project） | 返回空列表 `[]` |

## 4. 防穿越保护

- **Active 判定**：仅包含 `start_date <= timestamp <= deadline` 的 project
- **热度统计**：仅使用 `entry_created_at <= timestamp` 的 entry（按时序排序后 `break` 提前终止）
- **类目历史**：仅统计 `entry_created_at <= timestamp` 的 worker 历史
- **代码断言**：`get_candidates()` 入口处 assert `timestamp` 为 `datetime` 类型

## 5. K 值选型

在 test split（48805 条 entry）上评估不同 K 值的召回命中率：

| K | Recall@K | Hits | Total |
|---|----------|------|-------|
| 20 | 0.6277 | 30635 | 48805 |
| **50** | **0.9754** | **47603** | **48805** |
| 100 | 0.9998 | 48794 | 48805 |
| 200 | 0.9998 | 48794 | 48805 |

**选型结论**：默认 K=50。

- K=20 召回率仅 62.8%，大量 ground-truth 被漏掉，不可接受
- K=50 召回率达 97.5%，绝大多数 ground-truth 都在候选集中
- K=100 和 K=200 召回率几乎相同（99.98%），相对 K=50 提升极小（+2.4pp），但候选集规模翻倍/四倍，增加下游排序计算量
- K=50 是召回率和候选集规模的最佳折中点

冷启动 worker 占比仅 0.1%（28/48805），说明测试集中绝大部分 worker 在之前都有历史记录。

## 6. 文件结构

```
src/candidates/
├── __init__.py      # 模块入口，导出 get_candidates
├── recall.py        # 召回策略实现（热度、类目匹配）
└── generator.py     # CandidateGenerator 类 + get_candidates 接口

scripts/
└── eval_recall.py   # Recall@K 评估脚本

tests/
└── test_candidates.py  # 冒烟测试（14 个测试用例）
```

## 7. 使用示例

```python
from datetime import datetime, timezone
from src.candidates import get_candidates

# 为 worker_id=42 在 2019-01-15 时刻获取 Top-50 候选
candidates = get_candidates(
    worker_id=42,
    timestamp=datetime(2019, 1, 15, tzinfo=timezone.utc),
    K=50,
)
print(candidates)  # [project_id_1, project_id_2, ...]
```

高频调用场景建议直接使用 `CandidateGenerator` 类以复用预加载的数据：

```python
from src.candidates.generator import CandidateGenerator

gen = CandidateGenerator()
for worker_id, timestamp in queries:
    candidates = gen.get_candidates(worker_id, timestamp, K=50)
```
