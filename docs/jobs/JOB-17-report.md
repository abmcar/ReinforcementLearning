# JOB-17: 实验报告与答辩材料

**Phase**: D
**依赖**: JOB-16
**Status**: ⬜ Pending

---

## 背景

整个项目的最终交付。把所有 job 的成果整理为一份**课程作业报告**(DQN 主线为主) + 一份**答辩 PPT/slides**。这是评分的直接对象。

## 目标

产出符合课程要求的实验报告(DQN 主线 + LLM 拓展实验)和答辩材料。

## 输入

- **依赖产出**:JOB-01 ~ JOB-16 所有 docs/ 下产出物

## 工作内容

- **报告**:建议结构(可根据课程具体要求调整),严格遵守 `docs/roadmap.md` § 1 的篇幅分配(DQN ≥ 60%,LLM ≤ 30%):
  1. 任务定义与背景(`强化学习大作业.md` 和 `docs/roadmap.md`)
  2. 数据描述(`docs/data_report.md`,JOB-01)
  3. 方法
     - **DQN 主线**(主章节):MDP 定义(`docs/rl_env.md`,JOB-07)、模型(`docs/dqn_models.md` + `docs/offline_dqn.md`,JOB-08/09)、训练(`docs/dqn_results.md`,JOB-10)
     - **LLM 拓展**(单独次级章节,字数 ≤ 总篇幅 30%):数据构造(`docs/llm_data.md`,JOB-11)、训练(`docs/llm_sft.md` + `docs/llm_results.md`,JOB-13/15)
  4. 实验设置(`docs/data_split.md` JOB-02、`docs/features.md` JOB-03、`docs/evaluation.md` JOB-04、`docs/candidates.md` JOB-06)
  5. **主实验结果**:DQN(`docs/dqn_results.md`,JOB-10)
  6. **拓展实验结果**:LLM(`docs/llm_results.md`,JOB-15)
  7. 综合对比分析(`docs/comparison.md`,JOB-16)
  8. 结论与未来工作
- **答辩 PPT**:重点突出
  - 任务定义和挑战(为什么需要 RL + 为什么需要 offline RL)
  - 方法核心(DQN + offline 正则 + LLM 对比)
  - 关键结果(2-3 个 best result 突出展示)
  - 失败案例和反思

## 产出

**提交到 Git**:
- `docs/report.md`(或 `docs/report.pdf` + 源 .tex / .md)
- `docs/slides/`(PPT 源文件 + 导出 PDF)
- 报告里所有图引用 `docs/figures/` 已有的图,不要重复生成

**不提交**:无

## 验收标准

- [ ] 报告完整,涵盖所有作业要求章节(任务、数据、方法、实验、结论)
- [ ] DQN 主线为主交付,LLM 拓展作为单独章节
- [ ] 答辩 PPT 准备完毕,可独立讲 15-20 分钟
- [ ] 所有指标和图表的引用都能在上游 job docs / `docs/figures/` 中找到

## 参考资料

- 项目内所有 docs/

## 备注

- 务必在报告里**说明 LLM 主线的定位**(对比实验 / 方法拓展),解释为什么作业要求 DQN 但同时报了 LLM 结果。参考 `docs/roadmap.md` § 1「方案定位」原文。
- 报告里要**老老实实写失败结果**,不要只报 cherry-picked 的好数。失败分析往往更能体现思考深度。
