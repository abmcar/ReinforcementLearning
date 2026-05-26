# JOB-11: LLM 数据构造与 Prompt 说明

## 1. 方案概览

本 job 采用 **binary SFT** 范式构造 LLM 训练数据。每个样本只包含一个 `(worker, project, timestamp)` 配对，模型回答固定为 `Yes` 或 `No`。

- `worker` 目标:判断该 project 是否值得推荐给 worker，强调奖励、finalist / winner 信号和历史拟合度。
- `requester` 目标:判断该 worker 是否是 requester 的高质量候选，强调 `worker_quality` 与项目匹配度。
- 候选集统一通过 JOB-06 的候选逻辑生成，保证与 DQN / baseline 的评估口径一致。

代码入口:

- `src/llm/data_builder.py`
- `src/llm/prompts.py`
- `src/llm/__init__.py`

共享接口:

```python
from src.llm import build_dataset

path = build_dataset("worker", "train")
print(path)
```

命令行复现:

```bash
python -m src.llm.data_builder --objective all --split all
```

输出文件写到 `outputs/llm_data/`，不提交到 Git。

## 2. Prompt 模板

Prompt 固定为二分类问答格式，包含四部分:

1. `system`:说明众包设计平台语境，并强制输出 `Yes` / `No`。
2. `worker profile`:`worker_quality`、历史 entry 数、历史 win 数、平均奖金、top categories。
3. `recent worker history`:最近 `N=5` 条严格早于 `timestamp` 的历史参与记录。
4. `candidate project`:标题、category、industry、package、奖金、截止时间、brief 摘要。

当前模板版本是 `v1-binary-crowdsourcing`，默认参数见 [`experiments/configs/llm_data.yaml`](../experiments/configs/llm_data.yaml)。

## 3. 标签定义

### 3.1 Worker 目标

正样本条件:

- `winner=True`，或
- `finalist=True`，或
- `award_value >= 300.0`

其中 `300.0` 来自训练集非零 `award_value` 的 `0.75` 分位数。

### 3.2 Requester 目标

正样本条件:

- `worker_quality >= 0.96`

其中 `0.90` 来自训练集 `worker_quality` 的 `0.75` 分位数。

### 3.3 负样本

- 仅从 JOB-06 候选集中采样。
- 每个正样本默认采 `2` 个 negative。
- 若 ground-truth project 不在候选集中，则该正样本跳过，不做人工注入。

## 4. 构造约束

- **防穿越**:worker 历史只使用严格早于 `timestamp` 的交互。
- **冷启动 worker**:历史为空时写入固定描述 `No prior participation history is available for this worker.`。
- **候选一致性**:使用 JOB-06 同口径的快速候选生成器 `_FastCandidateGenerator`，逻辑与 `get_candidates()` 保持一致，只做性能优化，不重写召回策略。
- **长度控制**:project brief 最多保留 `4` 条，worker 历史最多保留 `5` 条。

Prompt 长度统计使用 `prompt_tokens_estimate`，实现上是基于正则的近似 token 计数，不依赖外部 tokenizer。实测所有 split 的 `p99` 和 `max` 都远低于 2048。

## 5. 数据集统计

### 5.1 Worker 目标

| split | samples | positives | negatives | covered positives | skipped positives | avg tokens | p50 | p90 | p99 | max |
|------|---------:|----------:|----------:|------------------:|------------------:|-----------:|----:|----:|----:|----:|
| train | 9,774 | 3,332 | 6,442 | 3,332 / 3,691 | 359 | 397.57 | 396 | 449 | 512 | 717 |
| val | 1,647 | 549 | 1,098 | 549 / 559 | 10 | 398.13 | 396 | 452 | 499 | 542 |
| test | 1,662 | 554 | 1,108 | 554 / 581 | 27 | 396.89 | 395 | 439 | 489 | 553 |

### 5.2 Requester 目标

| split | samples | positives | negatives | covered positives | skipped positives | avg tokens | p50 | p90 | p99 | max |
|------|---------:|----------:|----------:|------------------:|------------------:|-----------:|----:|----:|----:|----:|
| train | 277,782 | 93,330 | 184,452 | 93,330 / 99,496 | 6,166 | 404.91 | 401 | 455 | 520 | 716 |
| val | 71,217 | 23,739 | 47,478 | 23,739 / 24,056 | 317 | 400.87 | 398 | 454 | 513 | 599 |
| test | 74,871 | 24,957 | 49,914 | 24,957 / 25,625 | 668 | 396.33 | 395 | 439 | 488 | 572 |

### 5.3 观察

- `worker` 目标数据更稀疏，适合直接全量训练。
- `requester` 目标在原始 `0.75` 分位口径下显著大于 `worker` 目标，训练集达到 `277,782` 条样本；后续 JOB-13 建议先跑 `1k` / `5k` 子集 sanity。
- 候选召回对正样本覆盖率整体稳定，`worker_train` 略低于验证和测试，说明训练早期项目的候选 miss 相对更多。

## 6. 真实样本示例

以下示例来自 `outputs/llm_data/*.jsonl` 的真实样本，内容做了截断展示。

### Example 1: Worker Positive

Response: `Yes`

```text
Objective: worker
Worker Profile:
- worker_quality=0.7800
- total_prior_entries=3
- prior_wins=0
- top_categories=7
Recent Worker History:
1. Logo design for Investment Company | category=7 | industry=Unknown | outcome=participated
2. Logo design for Investment Company | category=7 | industry=Unknown | outcome=participated
3. Logo design for Investment Company | category=7 | industry=Unknown | outcome=participated
Candidate Project:
- title=Logo design for Investment Company
- category=7
- total_awards=300.0000
Project Brief:
- legacy_brief_text: Blue Sky Laboratories Ltd is a new investment company in UK...
```

### Example 2: Worker Negative

Response: `No`

```text
Objective: worker
Worker Profile:
- worker_quality=0.7800
- total_prior_entries=3
- top_categories=7
Recent Worker History:
1. Logo Design for new project SkillzBase | category=7 | industry=Unknown | outcome=participated
2. Logo Design for new project SkillzBase | category=7 | industry=Unknown | outcome=participated
3. Logo Design for new project SkillzBase | category=7 | industry=Unknown | outcome=participated
Candidate Project:
- title=Logo for new whisky company
- total_awards=200.0000
Project Brief:
- legacy_brief_text: 'Simply Whisky' is a new company focused on getting younger people...
```

### Example 3: Requester Positive

Response: `Yes`

```text
Objective: requester
Worker Profile:
- worker_quality=0.9600
- total_prior_entries=0
Recent Worker History:
No prior participation history is available for this worker.
Candidate Project:
- title=New logo for Mavenlink, a marketplace for delivering advice and services online
- total_awards=300.0000
Project Brief:
- legacy_brief_text: Mavenlink was founded on the principle of making qualified professional service providers...
```

### Example 4: Requester Negative

Response: `No`

```text
Objective: requester
Worker Profile:
- worker_quality=0.9600
- total_prior_entries=0
Recent Worker History:
No prior participation history is available for this worker.
Candidate Project:
- title=Community Center Logo
- total_awards=200.0000
Project Brief:
- legacy_brief_text: We are a community based organization by the name of Iranian Community Center...
```

### Example 5: Long-History Worker Positive

Response: `Yes`

```text
Objective: worker
Worker Profile:
- worker_quality=1.0000
- total_prior_entries=549
- prior_wins=5
- top_categories=7, 10, 6
Recent Worker History:
1. Law Firm Website Build Out (Richard A. Myers Jr. & Associates) | category=10 | industry=legal | outcome=participated
2. Design an engaging Intranet portal for a Group of companies (employee portal) | category=10 | industry=other | outcome=participated
...
Candidate Project:
- title=Law Firm Website Build Out (Richard A. Myers Jr. & Associates)
- total_awards=540.0000
Project Brief:
- Top 3 Things: Compassion, Knowledge, Professionalism
- Pages: title: Estate Planning, ...
```

## 7. 运行结果文件

本次实际生成的文件:

- `outputs/llm_data/worker_train.jsonl`
- `outputs/llm_data/worker_val.jsonl`
- `outputs/llm_data/worker_test.jsonl`
- `outputs/llm_data/requester_train.jsonl`
- `outputs/llm_data/requester_val.jsonl`
- `outputs/llm_data/requester_test.jsonl`
- `outputs/llm_data/summary.json`

这些文件不提交到 Git，但后续 JOB-12 / JOB-13 可以直接复用。
