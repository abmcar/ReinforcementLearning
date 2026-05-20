# JOB-06: 候选集生成模块

**Phase**: B(同时服务 C)
**依赖**: JOB-03
**Status**: ⬜ Pending

---

## 背景

本 job 实现 **DQN 和 LLM 主线共享**的候选集生成模块,产出统一的 action space / candidate list。「为什么需要两阶段召回」的背景见 `docs/roadmap.md` § 6。

## 目标

实现 `get_candidates(worker_id, timestamp, K)` 接口(签名见 `docs/roadmap.md` § 6.5),返回在该时间点对该 worker 的 Top-K 候选 project 集合,供 DQN 和 LLM 共享使用。

## 输入

- **依赖产出**:JOB-03 的 `build_features()`、JOB-02 的**冷启动协议**(见 `docs/data_split.md`)
- **数据**:project 的 active 时间(`start_date <= t <= deadline`)、worker 历史(仅限 timestamp t 之前)

## 工作内容

- 设计召回策略,至少实现一种,推荐**组合多路召回**:
  - 时间窗:只保留在 `t` 时刻 active 的 project
  - 类目匹配:worker 历史活跃 category 下的 project
  - 热度:全局或近期热门 project
  - (可选)双塔 embedding 召回
- 实现 `get_candidates(worker_id, timestamp, K=50)` 接口
- 给出**候选集质量评估**:在 test split 上,看 ground-truth 的 project 落在 Top-K 召回内的命中率(Recall@K),作为候选集质量的下界
- 文档化:K 的选择理由、各路召回的权重 / 合并策略

## 产出

**提交到 Git**:
- `src/candidates/__init__.py`、`src/candidates/recall.py`(各召回路)、`src/candidates/generator.py`(合并入口)
- `docs/candidates.md`:策略说明、K 的选型、召回命中率统计

**不提交**:
- 候选集缓存(放 `outputs/candidates/`)

## 验收标准

- [ ] `get_candidates()` 接口签名严格匹配 `docs/roadmap.md` § 6.5
- [ ] **召回时使用的 worker 历史严格不晚于 `timestamp t`**(防止 future leakage),在代码中加 assertion 保护
- [ ] 空 active project 和冷启动 worker 按 JOB-02 冷启动协议处理(在 `docs/candidates.md` 注明)
- [ ] 召回 Recall@K 在 test split 上有报告(在 `docs/candidates.md`)
- [ ] K 的选型有定量依据(例如 Recall@50 vs Recall@100 的折中)

## 参考资料

- 两阶段推荐 / 大动作空间相关论文见 `docs/roadmap.md` § 6「DQN 主线」。

## 备注

- 召回阶段的命中率是后续所有方案的上限。如果 Recall@K 太低(如 < 50%),需要回头加强召回。
