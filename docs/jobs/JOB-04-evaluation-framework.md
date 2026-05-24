# JOB-04: 评估框架与指标

**Phase**: A
**依赖**: JOB-02
**Status**: ✅ Completed

---

## 背景

DQN、LLM、所有 baseline 必须用**完全相同**的评估框架才能比较。本 job 实现统一的离线评估接口,负责接受任意推荐模型的输出(给定 worker 和候选集,返回排序),计算所有指标。

## 目标

实现可被所有方案复用的离线评估框架,统一指标计算口径,产出评估接口和指标定义文档。

## 输入

- **依赖产出**:JOB-02 的 `load_split()`
- **抽象约定**:评估接口接受任何实现了 `recommend(worker_id, timestamp, candidates) -> list[int]` 的对象(完整签名见 `docs/roadmap.md` § 6.5)。本 job 在 `src/eval/protocols.py`(或等价位置)定义 `HasRecommend` Protocol 类,DQN/LLM/baseline 通过 duck typing 接入。

## 工作内容

- 实现通用排序指标:HR@K、NDCG@K、MRR、Precision@K、Recall@K(K ∈ {1, 5, 10})
- 实现**双目标**专属指标(命名口径**与 `docs/roadmap.md` § 6 / § 6.5 完全一致**):
  - **参与者(worker)目标**:
    - `avg_award_value@K`:推荐 top-K 中实际 award_value 的均值
    - `finalist_rate@K` / `winner_rate@K`
    - `category_match_rate@K`:与 worker 历史偏好 category 的匹配率
  - **请求者(requester)目标**(在 worker→project 推荐范式下,以 project 视角统计所有指向该 project 的推荐):
    - `avg_recommender_worker_quality`:每个 project 被推荐到的所有 worker 的平均 quality
    - `project_coverage`:测试期内有多少比例的 active project 至少被推荐过一次
    - `entry_count_uplift`(可选):被推荐到的 project 的 entry 数 vs 同期未被推荐 project 的对照差异
- **Reward → Metric 映射**:在 `docs/evaluation.md` 加一节"Reward(per-step,JOB-07) → Metric(aggregated,本文)对应表",**与 `docs/roadmap.md` § 6.5 末尾的映射表一致**。任一处改动需同步另一处。
- 实现统一的 `evaluate(model, split)` 入口,返回 dict 形式的所有指标
- 实现多 seed / bootstrap 的置信区间计算
- 文档化:每个指标的公式、对 ground truth 的依赖、参与者/请求者目标各看哪些指标

## 产出

**提交到 Git**:
- `src/eval/metrics.py`(指标实现)
- `src/eval/evaluator.py`(`evaluate()` 入口)
- `docs/evaluation.md`:指标清单、公式、口径、双目标解释
- `tests/test_metrics.py`:对 HR/NDCG 等用人造数据的 sanity test

**不提交**:
- 评估输出(放 `outputs/eval/`)

## 验收标准

- [ ] `evaluate(model, split)` 接口稳定,文档化
- [ ] HR/NDCG/MRR 单元测试通过(用 hand-crafted 小数据集)
- [ ] `docs/evaluation.md` 说明每个指标的口径,以及参与者目标 / 请求者目标分别关注哪些指标
- [ ] 提供一个 dummy random recommender 跑通完整评估流程的 demo

## 参考资料

- LLM4Rec Survey(评估章节):[arXiv:2305.19860](https://arxiv.org/abs/2305.19860)
- 经典指标定义参考 RecBole 文档

## 备注

- 「请求者目标」的指标设计比「参与者目标」更微妙(因为请求者关心的是被分到的 worker 质量,不是某次推荐的 hit/miss)。允许在本 job 给出**多个候选指标**,JOB-09/JOB-10 训练时再敲定主指标。
- 若发现某些指标无法在离线日志上计算(如真实点击率,本数据没有),在 `docs/evaluation.md` 里明确标注"不可用"。
