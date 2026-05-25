# JOB-07: 推荐环境抽象(Env)

**Phase**: B
**依赖**: JOB-03, JOB-06
**Status**: ✅ Completed

---

## 背景

DQN 训练需要把"推荐任务"形式化为 MDP:state / action / reward / transition。本任务只有离线日志,所以"Env"实际上是一个**离线日志重放器(replay buffer)**,提供 `(s, a, r, s')` 元组,而不是真正的可交互环境。本 job 定义这个抽象,后续 DQN 模型(JOB-08/09)都基于它训练。

## 目标

实现众包推荐场景的 MDP 抽象和离线日志 → transition 转换逻辑,产出 `OfflineRecommendationEnv` 类(或等价模块),支持枚举所有训练 transition、采样 batch、双目标 reward 切换。

## 输入

- **依赖产出**:JOB-03 特征、JOB-06 候选集、JOB-02 冷启动协议(见 `docs/data_split.md`)
- **数据**:train split

## 工作内容

- 定义 MDP 元素(在 `docs/rl_env.md` 写清楚):
  - **MDP 形态默认 = contextual bandit**(roadmap §6 DQN 主线已声明默认形态)。即只学 `Q(s, a)`,不建模 transition `s → s'`。若 owner 选择真 MDP(带 transition),必须在 `docs/rl_env.md` 给出"为什么本数据下适合"的论证。
  - **State**:worker 的静态特征(quality)+ 动态历史(最近 N 个 entry 的 project 特征聚合,**严格限于 timestamp t 之前**)+ 当前时间上下文。给出维度。
  - **Action**:从 JOB-06 候选集中选一个 project。Action space 等价于 `K = len(candidates)`(典型 50)。
  - **Reward**:**预留两套接口**,worker reward 和 requester reward 分开,由 JOB-10 在训练时选用。具体公式建议(字段名以 JOB-01 EDA 报告为准,若 `award_value` / `finalist` / `winner` 任一缺失需相应回滚 reward 设计):
    - **worker reward 默认**:`r = award_value + α·1[finalist] + β·1[winner] + γ·category_match`,系数在 `experiments/configs/` yaml 里
    - **requester reward 默认**:`r = worker_quality(被推荐 worker 的 quality 归一化值)`
    - 与 JOB-04 metric 的对应关系见 `docs/roadmap.md` § 6.5 末尾「Reward → Metric 映射表」
  - **Transition**(若选择真 MDP):`s` = worker 在时刻 t 的状态;`a` = 时刻 t 推荐的 project;`r` = 实际交互后果;`s'` = worker 在下一次到达时刻的状态。冷启动 worker / 空候选集按 JOB-02 协议处理。
- 实现接口(签名见 `docs/roadmap.md` § 6.5):
  - `iter_transitions(split) -> Iterator[Transition]`,`Transition` 是 dataclass,字段 `s, a, r, s_next, candidates, info`(`info` 至少含 `worker_id, project_id, timestamp`)
  - `sample_batch(batch_size)`:供 DQN 训练
  - reward function 可注入(`env = Env(reward_fn=worker_reward_fn)`)
- 给出 sanity check:对若干 worker 打印 trajectory,人工核对合理性

## 产出

**提交到 Git**:
- `src/rl/env.py`(`OfflineRecommendationEnv`)
- `src/rl/rewards.py`(worker_reward_fn、requester_reward_fn 等)
- `docs/rl_env.md`:MDP 定义文档(state、action、reward 公式、transition 处理、设计权衡)
- `tests/test_env.py`:基础 sanity test

**不提交**:
- 缓存的 transition 数据(放 `outputs/replay/`)

## 验收标准

- [ ] MDP 元素在 `docs/rl_env.md` 写清楚,每个选择(尤其 transition 的处理方式 / 是否退化为 contextual bandit)有**明确选定方案和理由**,不能停在"讨论了但没定"
- [ ] `iter_transitions` 能跑出 ≥ 1 万条 transition(具体数取决于数据)
- [ ] reward function 可切换(worker / requester),JOB-10 直接调用
- [ ] 至少一个 sanity test 通过(如 transition 数 > 0、reward 在合理范围)

## 参考资料

- DQN 主线对照论文与 state 设计参考见 `docs/roadmap.md` § 6「DQN 主线」。

## 备注

- 默认 contextual bandit 形态(roadmap §6 已决)。若选真 MDP,后续 JOB-09 CQL/BCQ 实现会复杂得多 — 务必先和 owner 讨论清楚。
- `category_match` 是工程构造量(将 worker 历史最常 category 与 project category 对比的指示函数),**不是数据自带的 ground-truth 字段**。
- "reward 怎么定义最合理"本身就是研究问题。本 job 提供**接口**和**至少一组可工作的 reward**,精确的 reward shaping 留给 JOB-10。
