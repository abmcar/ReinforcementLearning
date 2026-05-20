# JOB-16: DQN vs LLM 对比分析

**Phase**: D
**依赖**: JOB-10, JOB-15
**Status**: ⬜ Pending

---

## 背景

DQN 主线(JOB-10)和 LLM 主线(JOB-15)各自产出了双目标实验结果。本 job 把两条线放到一张大表里横向对比,产出有研究价值的洞察,作为 JOB-17 报告的核心素材。

## 目标

整合 DQN、LLM、所有 baseline 的指标,做横向对比分析,产出**统一的对比表**和**关键洞察文档**。

## 输入

- **依赖产出**:JOB-10 的 `docs/dqn_results.md`、JOB-15 的 `docs/llm_results.md`、JOB-05 的 `docs/baselines.md`、JOB-12 的 `docs/llm_zero_shot.md`

## 工作内容

- 制作大对比表:行=**JOB-05 / JOB-10 / JOB-12 / JOB-15 / JOB-14(若有)实际产出的全部方法**,列=指标,分参与者 / 请求者两组
- 做可视化:bar chart / radar chart / scatter(算力 vs 指标)
- 分析维度建议:
  - **方法族对比**:DQN 系列 vs LLM 系列 vs 简单 baseline
  - **目标对比**:同一方法在 worker 目标和 requester 目标上的相对优势
  - **复杂度/收益**:哪个方法 ROI 最高?
  - **失败案例分析**:任何方法在哪些 worker / project 子群上系统性失败?抽几个例子
- 写洞察文档

## 产出

**提交到 Git**:
- `docs/comparison.md`:对比表、可视化、分析、关键洞察
- `docs/figures/`:对比图表

**不提交**:无新增大文件

## 验收标准

- [ ] 对比表覆盖所有方法 × 所有指标 × 两个目标(JOB-14 完成则纳入,否则注明 N/A)
- [ ] 至少 3 张对比图(柱状 / 雷达 / 散点等),保存到 `docs/figures/JOB-16-*.png`
- [ ] 分析里有**具体洞察**(不是泛泛"DQN 更好",而是"在 award_value@10 上 Offline DQN 比 popularity 高 X%,但在 category_match 上输给 category-match baseline")
- [ ] 失败案例至少 2-3 个,带数据
- [ ] **seed 数不同时**(DQN ≥ 3, LLM 可能 ≥ 2)对比表的 std 列保留各自 N,并在表注里注明

## 参考资料

- 项目内所有上游 job 的结果文档

## 备注

- 这一步**不重新训练**,只整合已有结果。如发现某些方法的结果不可信(seed 数太少、口径不一致),先反馈给对应 job owner 补做,而不是强行写报告。
- 对比表口径必须严格一致:同一 split、同一候选集 K、同一指标公式。
