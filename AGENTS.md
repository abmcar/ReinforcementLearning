# AGENTS.md

本文件为 Codex、Claude Code 或其他 coding agents 提供项目级操作说明。

`AGENTS.md` 是唯一维护的 agent 说明文件；`CLAUDE.md` 和 `agent.md` 应保持为指向本文件的软链接。

## 基本规则

- 使用中文与用户沟通；代码、命令、变量名和 commit message 使用英文。
- 执行前先检查现有文件和项目结构，不要凭记忆假设。
- 不要删除或覆盖他人改动，除非用户明确要求。
- 修改代码或文档后，运行与改动相关的检查；如果无法运行，说明原因。

## 项目目标

本项目是强化学习课程小组大作业，任务是使用 DQN 系列模型解决众包任务推荐问题，并分别优化：

1. 参与者利益。
2. 请求者利益。

并行做一条 **LLM + SFT** 推荐方案作为对比实验(详见 [`docs/roadmap.md`](./docs/roadmap.md) § 1)。

## 推进计划与 Job 管理

- 项目整体推进计划和进度表在 [`docs/roadmap.md`](./docs/roadmap.md),所有 job 详情在 [`docs/jobs/`](./docs/jobs/)。
- **接手 job 时**:把目标 job 文件 + `docs/roadmap.md` 一起作为上下文,按 job 文件里的「工作内容」「产出」「验收标准」执行。
- **开始 job 时**:在 `docs/roadmap.md` § 4 进度表对应行,把 Status 改为 🔵 In Progress、Owner 填上自己。
- **完成 job 时**:把 Status 改为 ✅ Completed,在 PR 描述里附 job 文件链接和实际产出文件清单。
- **依赖未完成时**:不要提前开始下游 job(依赖关系见 `docs/roadmap.md` § 5)。
- **新增 / 拆分 / 合并 job**:同步更新 `docs/roadmap.md` 进度表、依赖图和 `docs/jobs/` 文件,变更记录写在 commit message。

## 数据与提交约束

- `data.zip` 已提交到 repo，供组员克隆后解压使用。
- 不要提交 `data/entry/`、`data/project/`（解压后的原始数据目录）。
- 不要提交 `outputs/`、`runs/`、`checkpoints/`、`models/` 或模型权重文件。
- 需要保留可复现实验所需的代码、配置、图表和文档。

开发流程详见 [`docs/roadmap.md`](./docs/roadmap.md)。

## 文档要求

- README 记录项目目标、数据说明和协作流程。
- `docs/` 中必须说明训练集、验证集和测试集的划分方式。
- 实验结论必须包含指标、配置和结果来源。
