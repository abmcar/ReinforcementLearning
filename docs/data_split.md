# JOB-02: 数据集时序划分与冷启动协议 (v1)

本文档定义了众包推荐任务的全局数据集划分基准，以及针对冷启动问题的结构化处理协议。所有下游任务（包括 DQN 主线和 LLM 主线）必须严格遵循此标准，以保证评估的公平性。

## 1. 数据划分方案

### 1.1 划分策略选型
* **采用方案**：**全局时序排序后按比例切分 (Global Time-ordered Proportional Split)**
* **选型理由**：推荐系统具有极强的时效性，必须防范”未来信息泄漏（Data Leakage）”。随机划分或交叉验证在实际生产环境中不可行。按照历史日志（`entry_created_at`）严格排序后按索引位置比例切分，可以模拟推荐系统在线上”用过去的数据训练，在未来的流量上验证”的真实流程。
* **实现细节**：先对所有 entry 按 `entry_created_at` 升序排序，然后按排序后的索引位置按比例划分。注意：同一时刻的多条 entry 可能被分配到不同 split。
* **划分比例**：
  * **Train (训练集)**：排序后前 80% 的日志
  * **Validation (验证集)**：排序后中间 10% 的日志
  * **Test (测试集)**：排序后最后 10% 的日志

### 1.2 接口与结构
代码已在 `src/data/split.py` 实现。提供共享接口 `load_split(name: Literal["train", "val", "test"])`，返回 `EntryList` dataclass（包含强类型的日志列表和当前切片的时间跨度），供下游调用。

---

## 2. 结构化冷启动协议

在严格的时序划分下，验证集和测试集必然会遇到训练集中未见过的实体。所有相关 Job (JOB-03/06/07/11) 遇到此类情况时，必须统一采取以下 fallback 策略：

### 2.1 冷启动 Worker (User Cold-Start)
在测试阶段出现了一个全新的 Worker ID：
* **特征构造 (JOB-03)**：所有需要聚合历史交互计算的特征全部置为 `0`；其静态属性 `worker_quality` 若缺失，则使用训练集中所有已知 `worker_quality` 的**全局中位数**进行插补。
* **推荐召回 (JOB-06)**：失去协同过滤等个性化召回基础，直接降级为“全局热度策略（Popularity-based）”——向其召回近期收到 `entry` 数量最多且尚未截稿的 Top-K Projects。

### 2.2 冷启动 Project (Item Cold-Start)
在测试阶段出现了一个刚刚发布的新 Project ID：
* **特征构造 (JOB-03)**：其动态特征（如已收到的 entry 数）置为 `0`；由于新任务有完备的元数据，必须充分利用 `category`、`sub_category` 和 `title` 提取基于内容的特征（Content-based features）。
* **Env 抽象 (JOB-07)**：冷启动 Project 依然作为合法的可选动作参与大动作空间的推荐。

### 2.3 空 Active Project (无任务可做)
当一个 Worker 访问系统时，如果系统内没有任何处于“未截止且正在进行”的 active project：
* **动作空间约束**：`get_candidates` 接口必须返回空列表 `[]`。
* **Env 状态转移 (JOB-07)**：此步骤直接判定为终止状态或空转，不强制推荐过期任务，也不产生任何 Reward。