# 强化学习大作业：众包任务推荐

使用 DQN 系列模型解决众包系统中的任务推荐问题，并分别从参与者利益和请求者利益两个角度设计实验、训练模型并分析结果。

## 作业目标

在众包平台中，当 worker 到达系统时，系统需要从可用 project 中推荐一个任务。推荐策略需要考虑 worker 兴趣、任务时间、任务完成情况、worker 质量等动态因素。

本项目需要完成两个方向：

1. 最大化参与者利益：让 worker 更容易获得相关、感兴趣且收益更高的任务。
2. 最大化请求者利益：让 project 获得更多、更高质量的回答。

模型要求：使用 DQN 系列方法。

## 当前资料

- [强化学习大作业.md](./强化学习大作业.md)：课程作业要求整理版。
- `data/sample_read_data.py`：数据读取参考代码。
- `data/project_list.csv`：project 基础统计信息。
- `data/worker_quality.csv`：worker quality 信息。
- `data/entry/`：worker entry 数据，本地保留，不提交到 Git。
- `data/project/`：project 数据，本地保留，不提交到 Git。
- `data.zip`：原始压缩包，本地保留，不提交到 Git。

说明：`data.zip` 文件超过 GitHub 普通文件大小限制，`data/entry/` 与 `data/project/` 总量较大，因此默认不纳入 Git 版本控制。协作者需要从课程附件或共享网盘获取完整数据，并解压到仓库根目录的 `data/` 下。

## 目录结构

```text
.
├── README.md
├── AGENTS.md           # 唯一维护的 agent 协作说明
├── CLAUDE.md -> AGENTS.md
├── agent.md -> AGENTS.md
├── 强化学习大作业.md
├── data/
│   ├── sample_read_data.py
│   ├── project_list.csv
│   ├── worker_quality.csv
│   ├── entry/          # 本地数据，不提交
│   └── project/        # 本地数据，不提交
├── src/                # 数据处理、环境、模型、训练代码
├── experiments/        # 实验配置和运行脚本
├── docs/               # 设计文档、实验记录、报告素材
└── outputs/            # 本地输出，不提交
```

## 协作流程

1. 每个成员从 `main` 分支拉取最新代码。
2. 新功能或实验使用独立分支，例如 `feature/dqn-worker-reward`。
3. 提交前运行能覆盖本次修改的脚本或测试。
4. 通过 Pull Request 合并到 `main`。
5. 实验结果、关键参数和结论需要记录到 `docs/` 或 Pull Request 描述中。

## 实验推进建议

1. 数据理解：读取并统计 worker、project、entry 数据。
2. 环境建模：定义 state、action、reward、transition。
3. 参与者目标：设计以 worker 收益或匹配度为核心的 reward。
4. 请求者目标：设计以回答数量、回答质量或任务完成度为核心的 reward。
5. 模型训练：实现 DQN、Double DQN 或 Dueling DQN 等模型。
6. 评估对比：与随机推荐、规则推荐等 baseline 对比。
7. 文档整理：说明数据划分、实验设置、结果图表和结论。

## GitHub 上传

首次创建远程仓库后，在本地执行：

```bash
git remote add origin git@github.com:<your-org-or-user>/<repo-name>.git
git push -u origin main
```

如果使用 HTTPS：

```bash
git remote add origin https://github.com/<your-org-or-user>/<repo-name>.git
git push -u origin main
```
