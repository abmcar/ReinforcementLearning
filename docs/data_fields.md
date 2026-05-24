# JOB-01: 数据字段审查与对比清单 (Data Fields)

本文档旨在梳理实际数据文件中的 JSON/CSV 字段，并与 `docs/roadmap.md` §2 中的预期字段进行逐一比对，以防下游任务（如特征工程）产生字段名不对齐的 Error。

## 1. Worker 质量数据 (`worker_quality.csv`)

| 实际字段名 | 数据类型 | Roadmap 对比状态 | 备注说明 |
| :--- | :--- | :--- | :--- |
| `worker_id` | int | 正常匹配 | 用户的唯一标识符 |
| `worker_quality` | float | 正常匹配 | 原始取值范围 [-1, 100]。共 154 个 worker 的 quality <= 0（含 -1），实际处理中需过滤并除以 100 归一化到 [0, 1] |

---

## 2. 项目索引 (`project_list.csv`)

该文件无 header 行，两列含义如下：

| 列序号 | 对应字段名 | 数据类型 | 备注说明 |
| :--- | :--- | :--- | :--- |
| 第 1 列 | `project_id` | int | 项目唯一标识符，与 `project/project_<id>.txt` 文件名对应 |
| 第 2 列 | `project_answer_num` | int | 该项目的 entry 提交总数，与 `sample_read_data.py` 中的 `entry_count` 变量含义一致，用于遍历 `entry/entry_<pid>_<k>.txt` 分页文件 |

> **来源确认**：`强化学习大作业.md` 明确记载该文件包含 `project_id` 和 `project_answer_num`；`sample_read_data.py` 第 34-35 行以 `project_id = int(line[0])` / `entry_count = int(line[1])` 读取。

---

## 3. Project 项目详情 (`project/project_<id>.txt`)

| 实际字段名 | 数据类型 | Roadmap 对比状态 | 备注说明 |
| :--- | :--- | :--- | :--- |
| `id` | int | 正常匹配 | 项目 ID |
| `title` | string | 正常匹配 | 项目标题文本 |
| `category` | int | 正常匹配 | 主分类 ID |
| `sub_category` | int | 正常匹配 | 子分类 ID |
| `industry` | string/null | 正常匹配 | 行业领域（部分数据可能为 null） |
| `start_date` | string | 正常匹配 | 任务发布时间（需用 dateutil 解析） |
| `deadline` | string | 正常匹配 | 任务截止时间 |
| `status` | string | 正常匹配 | 项目状态 |
| `entry_count` | int | 正常匹配 | 提交的答案/作品总数 |
| `creative_count` | int | 正常匹配 | 参与创作者数量 |
| `average_score` | float | 正常匹配 | 平均评分 |
| `total_awards` | float | 正常匹配 | 总奖金额 |
| `brief_questions` | list | 正常匹配 | 需求描述问题列表 |
| `brief_answers` | list | 正常匹配 | 需求描述答案列表 |
| `package_name` | string | 正常匹配 | 套餐名称 |
| `participants` | int/list | 正常匹配 | 参与者信息 |

---

## 4. Entry 提交记录 (`entry/entry_<pid>_<k>.txt`)

> **注意：** 每个 entry 文件是一个 JSON 对象，实际的 entry 记录嵌套在 `results` 数组中。文件按每 24 条分页（即 k = 0, 24, 48, ...），与 `sample_read_data.py` 的分页逻辑一致。

**顶层结构**:

| 字段名 | 数据类型 | 备注说明 |
| :--- | :--- | :--- |
| `results` | list[object] | 包含本页所有 entry 记录的数组 |

**`results` 数组中每条 entry 的字段**:

| 实际字段名 | 数据类型 | Roadmap 对比状态 | 备注说明 |
| :--- | :--- | :--- | :--- |
| `entry_number` | int | 正常匹配 | 作品的唯一编号 |
| `entry_created_at` | string | 正常匹配 | 工人提交作品的具体时间 |
| `worker` | int | **未在 roadmap 列出** | 实际数据 JSON key 为 `worker`，即 worker_id。`sample_read_data.py` 第 65 行确认使用 `item["worker"]`。下游 JOB-03 需以实际 key `worker` 为准 |
| (`author`) | - | **roadmap 列出但实际不存在** | Roadmap §2 列为 `author`(=worker_id)，但实际数据中该 key 不存在，对应的实际 key 是 `worker` |
| `award_value` | float | 正常匹配 | 奖金数值 |
| `finalist` | bool | 正常匹配 | 是否进入决赛 |
| `winner` | bool | 正常匹配 | 是否获胜（只有 winner 拿全额奖金） |
| `offer_value` | float | 正常匹配 | 额外报价收益 |
| `tip` | float | 正常匹配 | 小费收益 |
| `eliminated` | bool | 正常匹配 | 是否被淘汰 |
| `withdrawn` | bool | 正常匹配 | 是否被撤回 |
| `entry_type` | string | 正常匹配 | 提交类型 |
| `project` | int | 正常匹配 | 所属项目 ID |

## 总结与发现

1. 整体数据结构与 Roadmap 基本吻合。
2. **关键不一致**：entry 数据中获取 worker ID 的实际 key 是 `worker`，而 Roadmap §2 列为 `author`。`sample_read_data.py` 使用的是 `worker`。下游 JOB-03（特征工程）必须以实际 key `worker` 为准，否则会导致 `KeyError`。
3. `project_list.csv` 无 header，第 2 列（`project_answer_num`）是每个 project 的 entry 总数，用于遍历分页 entry 文件。
