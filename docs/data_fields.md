# JOB-01: 数据字段审查与对比清单 (Data Fields)

本文档旨在梳理实际数据文件中的 JSON/CSV 字段，并与 `docs/roadmap.md` §2 中的预期字段进行逐一比对，以防下游任务（如特征工程）产生字段名不对齐的 Error。

## 1. Worker 质量数据 (`worker_quality.csv`)

| 实际字段名 | 数据类型 | Roadmap 对比状态 | 备注说明 |
| :--- | :--- | :--- | :--- |
| `worker_id` | int | 正常匹配 | 用户的唯一标识符 |
| `worker_quality` | float | 正常匹配 | 取值范围含负数，实际处理中需过滤 `<= 0` 的异常值并除以 100 归一化 |

---

## 2. Project 项目详情 (`project/project_<id>.txt`)

| 实际字段名 | 数据类型 | Roadmap 对比状态 | 备注说明 |
| :--- | :--- | :--- | :--- |
| `sub_category` | int | 正常匹配 | 子分类 ID |
| `category` | int | 正常匹配 | 主分类 ID |
| `entry_count` | int | 正常匹配 | 提交的答案/作品总数 |
| `start_date` | string | 正常匹配 | 任务发布时间 (需用 dateutil 解析) |
| `deadline` | string | 正常匹配 | 任务截止时间 |
| `industry` | string | 正常匹配 | 行业领域 (部分数据可能缺失或为空) |
| `title` | string | 正常匹配 | 项目标题文本 |
| `brief_questions` | list | 正常匹配 | 需求描述等文本信息 |
| `id` / `status` / 等 | 混合 | 正常匹配 | 符合 roadmap 预期 |

---

## 3. Entry 提交记录 (`entry/entry_<pid>_<k>.txt`)

> **注意：** 真实的 Entry 数据是嵌套在 JSON 的 `results` 数组中的。

| 实际字段名 | 数据类型 | Roadmap 对比状态 | 备注说明 |
| :--- | :--- | :--- | :--- |
| `entry_number` | int | 正常匹配 | 作品的唯一编号 |
| `entry_created_at` | string | 正常匹配 | 工人提交作品的具体时间 |
| `worker` | int | ⚠️ **roadmap 列出但实际不存在** | Roadmap 中声称该字段名为 `author`，但实际 JSON 解析时该字段名为 `worker`。 |
| `award_value` | float | 正常匹配 | 奖金数值 |
| `finalist` / `winner`| bool | 正常匹配 | 用于判断 worker 是否获胜的核心标签 |
| `offer_value` / `tip` | float | 正常匹配 | 其他激励收益 |
| `entry_type` | string | 正常匹配 | 提交类型 |

## 总结与发现
通过对比发现，整体数据结构与 Roadmap 基本吻合，但下游 **JOB-03 (特征工程)** 在处理 `entry` 数据时，**必须使用 `worker` 键来获取用户 ID，而不是使用 Roadmap 中误写的 `author` 键**，否则会导致 `KeyError`。