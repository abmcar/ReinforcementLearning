# JOB-05: Baseline 推荐策略

**Phase**: A
**依赖**: JOB-03, JOB-04, JOB-06
**Status**: ✅ Completed

---

## 背景

DQN 和 LLM 方法都需要和**非 RL、非 LLM 的简单 baseline** 对比才能证明其价值。本 job 实现一组覆盖不同复杂度的 baseline,作为所有后续方法对比的基线。

## 目标

实现至少 4 个 baseline 推荐策略,并在 JOB-04 评估框架下跑出完整指标,作为后续 DQN/LLM 方案的对比基线。

## 输入

- **依赖产出**:JOB-03 的 `build_features()`、JOB-04 的 `evaluate()`、JOB-06 的 `get_candidates()`
- **接口约定**:每个 baseline 实现 `recommend(worker_id, timestamp, candidates) -> list[int]`(签名见 `docs/roadmap.md` § 6.5),候选集**统一通过 JOB-06 `get_candidates()`** 取,确保和 DQN/LLM 评估条件一致

## 工作内容

实现以下 baseline(可根据数据情况增减,但至少 4 个):

- **Random**:从候选集中随机排序(下界)
- **Popularity**:按 project 的 `entry_count` 或全局热度排序
- **Category-Match**:按 worker 历史最常活跃 category 匹配 project category 排序
- **Worker-Quality-weighted Popularity**:为请求者目标设计,根据 worker quality 加权
- **可选**:Item-CF / Matrix Factorization(用 implicit feedback)等传统协同过滤
- **可选**:CTR-style 监督学习(LightGBM / DeepFM)作为更强 baseline

每个 baseline:
- 在 train split fit、在 val/test split 评估
- **必须在 JOB-06 候选集上排序**,不能直接对全量 project 排序
- 报全部 JOB-04 指标(参与者 + 请求者两组)
- 结果写入 `docs/baselines.md`(表格形式)

## 产出

**提交到 Git**:
- `src/baselines/random.py`、`src/baselines/popularity.py`、`src/baselines/category_match.py`、...
- `src/baselines/run_all.py`:一键跑所有 baseline 并输出结果表
- `docs/baselines.md`:baseline 描述 + 完整指标表 + 简短分析

**不提交**:
- 训练中间产物

## 验收标准

- [ ] 至少 4 个 baseline 实现完成
- [ ] 在 test split 上跑出完整指标
- [ ] `docs/baselines.md` 含指标对比表,标注每个 baseline 在参与者目标和请求者目标上的表现
- [ ] 一键脚本可复现所有 baseline 结果

## 参考资料

- Rendle 等多次实验显示简单 baseline 在合理调参下常逼近复杂模型:[arXiv:1905.01395](https://arxiv.org/abs/1905.01395)("Are We Really Making Much Progress?")

## 备注

- baseline 结果是后续所有 DQN / LLM 实验的对比锚点,**质量比数量重要**。不要为了凑数实现一堆质量很差的 baseline。
- 如果某个 baseline 在某指标上意外强,要在 `docs/baselines.md` 里写明,这是有价值的发现。
