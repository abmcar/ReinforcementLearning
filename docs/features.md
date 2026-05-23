# JOB-03: 特征工程说明文档 (Feature Engineering)

本文档记录了推荐系统（DQN 与 LLM 主线）所使用的统一特征空间。
所有特征严格遵循**“防穿越（No Data Leakage）”**原则：任何在时间点 $t$ 发生的交互，其动态特征均只使用严格早于 $t$ 的历史数据计算。

## 1. Worker 特征 (工人画像)
| 特征名称 | 类型 | 范围/说明 | 来源 |
| :--- | :--- | :--- | :--- |
| `worker_quality` | 静态 | Float [0, 1]。工人的历史评分，缺失值使用全局中位数填充 | `worker_quality.csv` |
| `hist_entries` | 动态 | Int $\ge 0$。截止到 $t$ 时刻，该工人提交过的总作品数 | Entry 日志时序推断 |
| `hist_wins` | 动态 | Int $\ge 0$。截止到 $t$ 时刻，该工人累计获胜 (winner=True) 的次数 | Entry 日志时序推断 |
| `hist_win_rate` | 动态 | Float [0, 1]。`hist_wins / hist_entries`。若无历史则为 0 | 组合推断 |
| `hist_avg_award`| 动态 | Float $\ge 0$。截止到 $t$ 时刻，该工人历史赚取的平均奖金 | 组合推断 |

## 2. Project 特征 (任务画像)
| 特征名称 | 类型 | 范围/说明 | 来源 |
| :--- | :--- | :--- | :--- |
| `category` | 静态 | Int。任务的主分类 ID | `project_*.txt` |
| `sub_category` | 静态 | Int。任务的子分类 ID | `project_*.txt` |
| `industry` | 静态 | Int (Label Encoded)。任务所属行业 | `project_*.txt` |
| `duration_days` | 静态 | Float。任务总生命周期 (`deadline` - `start_date` 的天数) | `project_*.txt` |
| `days_remaining`| 动态 | Float。截止到 $t$ 时刻，距离 `deadline` 还有多少天 | 组合推断 |
| `current_entries`| 动态 | Int $\ge 0$。截止到 $t$ 时刻，该任务已经收到的作品数 | Entry 日志时序推断 |

## 3. 冷启动处理协议
严格遵守 JOB-02 的协议：
* **全新 Worker**：动态历史特征全部置为 `0`，`worker_quality` 置为全局中位数。
* **全新 Project**：`current_entries` 置为 `0`，保留所有的静态特征（分类、行业等）。