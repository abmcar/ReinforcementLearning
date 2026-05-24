# 评估框架与指标说明

本文档定义了所有推荐方案（DQN / LLM+SFT / Baseline）共用的离线评估框架和指标口径。

---

## 1. 评估接口

### 1.1 `HasRecommend` Protocol

所有推荐模型必须实现以下接口：

```python
class HasRecommend(Protocol):
    def recommend(
        self,
        worker_id: int,
        timestamp: datetime,
        candidates: list[int],
    ) -> list[int]:
        """返回按预测相关性降序排列的 project_id 列表。"""
        ...
```

### 1.2 `evaluate(model, split)` 主入口

```python
def evaluate(
    model: HasRecommend,
    split: Literal["train", "val", "test"],
    *,
    candidate_fn: Callable | None = None,  # 默认使用内置候选集生成
    candidate_k: int = 50,
    n_bootstrap: int = 0,                  # >0 时计算置信区间
    seed: int = 42,
) -> dict[str, float]
```

**返回值**：包含所有指标的字典，key 为指标名，value 为 float。若启用 bootstrap，额外包含 `<metric>_ci_lower` / `<metric>_ci_upper`。

### 1.3 候选集生成

`candidate_fn` 参数签名：`(worker_id, timestamp, project_info, ground_truth_pid, k) -> list[int]`

- **默认行为**：返回在 `timestamp` 时刻活跃的所有项目（`start_date <= timestamp <= deadline`），始终包含 ground truth project，上限 `candidate_k` 个。
- **JOB-06 完成后**：调用方应传入 `get_candidates` 的适配包装。

### 1.4 Ground Truth 定义

对于 split 中的每条 Entry `(worker_id, timestamp, project_id)`：
- **Ground truth positive** = 该 worker 实际选择的 `project_id`
- HR@K = "模型返回的 top-K 排序列表中是否包含该 project_id"

所有下游 job 继承此假设。

---

## 2. 通用排序指标

以下指标对每条评估事件逐一计算，最终取均值。

### 2.1 Hit Rate @ K (HR@K)

$$\text{HR@K} = \frac{1}{N} \sum_{i=1}^{N} \mathbb{1}[\text{gt}_i \in \text{top-K}_i]$$

ground truth 出现在 top-K 则为 1，否则为 0。

### 2.2 NDCG @ K

$$\text{NDCG@K} = \frac{\text{DCG@K}}{\text{IDCG@K}}$$

单个 relevant item 的情况下：
- $\text{DCG@K} = \frac{1}{\log_2(\text{rank} + 1)}$ （若 gt 在 top-K 内）
- $\text{IDCG@K} = \frac{1}{\log_2(2)} = 1.0$ （最佳情况：gt 在 rank 1）

### 2.3 MRR (Mean Reciprocal Rank)

$$\text{MRR} = \frac{1}{N} \sum_{i=1}^{N} \frac{1}{\text{rank}_i}$$

若 gt 不在列表中，该条贡献 0。

### 2.4 Precision @ K

$$\text{Precision@K} = \frac{|\text{relevant} \cap \text{top-K}|}{K}$$

### 2.5 Recall @ K

$$\text{Recall@K} = \frac{|\text{relevant} \cap \text{top-K}|}{|\text{relevant}|}$$

**K 取值**：{1, 5, 10}

---

## 3. 参与者（Worker）目标指标

这些指标衡量推荐对 worker 的价值——worker 希望被推荐到高奖金、高胜率、与自身技能匹配的项目。

### 3.1 `avg_award_value@K`

推荐 top-K 中各项目 award_value 的均值。

$$\text{avg\_award\_value@K} = \frac{1}{K} \sum_{j=1}^{K} \text{award\_value}(\text{top-K}_j)$$

### 3.2 `finalist_rate@K`

推荐 top-K 中，worker 在历史数据中成为 finalist 的项目的比例。

$$\text{finalist\_rate@K} = \frac{|\{p \in \text{top-K} : \text{worker 曾在 } p \text{ 中为 finalist}\}|}{K}$$

### 3.3 `winner_rate@K`

推荐 top-K 中，worker 在历史数据中成为 winner 的项目的比例。

$$\text{winner\_rate@K} = \frac{|\{p \in \text{top-K} : \text{worker 曾在 } p \text{ 中为 winner}\}|}{K}$$

### 3.4 `category_match_rate@K`

推荐 top-K 中，项目 category 与 worker 历史偏好 category（top-3 频次）匹配的比例。

$$\text{category\_match\_rate@K} = \frac{|\{p \in \text{top-K} : \text{cat}(p) \in \text{worker\_pref\_cats}\}|}{K}$$

---

## 4. 请求者（Requester）目标指标

这些指标以 project 视角统计，衡量推荐对 requester 的价值——requester 希望高质量 worker 被推荐到自己的项目。

### 4.1 `avg_recommender_worker_quality`

每个被推荐过的 project，收集所有被推荐到该 project 的 worker 的 quality，取均值；再在所有 project 上做 macro average。

$$\text{avg\_rec\_wq} = \frac{1}{|P|} \sum_{p \in P} \frac{1}{|W_p|} \sum_{w \in W_p} \text{quality}(w)$$

其中 $P$ 为被推荐过的 project 集合，$W_p$ 为被推荐到 project $p$ 的 worker 集合。

### 4.2 `project_coverage`

测试期内有多少比例的活跃 project 至少被推荐过一次。

$$\text{project\_coverage} = \frac{|\text{recommended\_projects} \cap \text{active\_projects}|}{|\text{active\_projects}|}$$

### 4.3 `entry_count_uplift`（可选，暂未实现）

被推荐到的 project 的 entry 数 vs 同期未被推荐 project 的对照差异。由于离线评估无法观测推荐对 entry 数的因果影响，此指标标注为**不可用**——仅在有在线 A/B 测试环境时可计算。

---

## 5. 双目标总结

| 目标 | 关注的指标 | 含义 |
|------|-----------|------|
| **参与者 (Worker)** | `avg_award_value@K`, `finalist_rate@K`, `winner_rate@K`, `category_match_rate@K` | Worker 从推荐中获得的实际收益和技能匹配度 |
| **请求者 (Requester)** | `avg_recommender_worker_quality`, `project_coverage` | Requester 获得的 worker 质量和项目曝光度 |
| **通用** | `HR@K`, `NDCG@K`, `MRR`, `Precision@K`, `Recall@K` | 推荐排序的准确性 |

---

## 6. Reward (per-step) → Metric (aggregated) 对应表

此表与 `docs/roadmap.md` §6.5 末尾的映射表保持一致。任一处改动需同步另一处。

| 目标 | per-step reward (JOB-07) | aggregated metric (JOB-04) |
|------|--------------------------|------------------------------|
| worker | `award_value + α·finalist + β·winner + γ·category_match` | `avg_award_value@K` / `finalist_rate@K` / `category_match_rate@K` |
| requester | `worker_quality(被推荐 worker 的质量)` | `avg_recommender_worker_quality` / `project_coverage` |

---

## 7. 统计显著性

- DQN 多 seed (>=3) 训练，报 mean +/- std。
- LLM 训练允许 >=2 seed。
- `evaluate()` 支持 `n_bootstrap` 参数，启用后为每个指标计算 95% bootstrap 置信区间（添加 `<metric>_ci_lower` / `<metric>_ci_upper` keys）。

---

## 8. 使用示例

```python
from src.eval import evaluate, HasRecommend

class MyRecommender:
    def recommend(self, worker_id, timestamp, candidates):
        # 你的排序逻辑
        return sorted(candidates)

model = MyRecommender()
results = evaluate(model, "test")

# 查看结果
for k, v in sorted(results.items()):
    print(f"{k}: {v:.4f}")
```

---

## 9. 文件清单

| 文件 | 用途 |
|------|------|
| `src/eval/protocols.py` | `HasRecommend` Protocol 定义 |
| `src/eval/metrics.py` | 所有指标函数实现 |
| `src/eval/evaluator.py` | `evaluate()` 主入口 + 数据加载 + 候选集默认生成 |
| `tests/test_metrics.py` | 指标单元测试 + 端到端冒烟测试 |
| `docs/evaluation.md` | 本文档 |
