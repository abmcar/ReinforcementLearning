# JOB-02: 数据集时序划分 + 冷启动协议

**Phase**: A
**依赖**: JOB-01
**Status**: ⬜ Pending

---

## 背景

推荐系统场景下,**不能用随机划分**(否则未来信息泄漏到训练集,评估严重失真)。本任务的数据有明确时间戳(`entry_created_at`、`project.start_date/deadline`),必须按时序划分。本 job 输出的 split 是所有后续训练/评估 job 的公共基线。

## 目标

定义并实现训练集 / 验证集 / 测试集的**时序划分方案**,产出可复用的 split 模块,以及说明文档(`docs/data_split.md`)详细记录划分细节(数据划分文档是项目交付的硬性要求)。

## 输入

- **依赖产出**:JOB-01 的数据报告(确认时间范围、entry 在各时间段的分布)
- **数据**:`data/entry/*` 中的 `entry_created_at` 字段、`data/project/*` 中的 `start_date / deadline`

## 工作内容

- 调研并选定一种时序划分方案。可选方案包括:
  - **按时间点切分**(如 80% 时间训练、10% 验证、10% 测试)
  - **滚动窗口**(time-aware leave-future-out)
  - **按 project deadline 切分**(已结束 project 入训练集,后期 project 入测试集)
- 给出方案选型的理由(在 `docs/data_split.md` 里说明)。
- 实现 `load_split(name)` 接口(签名见 `docs/roadmap.md` § 6.5):接受 `name: Literal["train","val","test"]`,返回 `EntryList`(dataclass,字段 `entries: list[Entry]` 和 `time_range: tuple[datetime, datetime]`)。
- 定义并产出**冷启动协议**(在 `docs/data_split.md` 单独一节):
  - **冷启动 worker**(测试集中出现但训练集没见过)— 给出 fallback 特征构造方案(如:用全局 worker_quality 均值、用 popularity 推荐策略)
  - **冷启动 project**(测试集中出现但训练集没见过)— 同上
  - **空 active project**(某 worker 到达时刻没有任何 active project)— 明确 `get_candidates` 返回 `[]` 还是 fallback
  - 协议必须**结构化**,JOB-03 / JOB-06 / JOB-07 / JOB-11 都按此处理
- 与 JOB-04(评估)和 JOB-07(Env)对齐数据接口约定。

## 产出

**提交到 Git**:
- `src/data/split.py`:划分逻辑 + `load_split()` 接口
- `src/data/__init__.py`(如尚未存在)
- `docs/data_split.md`:划分方案、比例、理由、冷启动处理策略
- 可选:`outputs/splits/split_v1.json`(冻结的 split 索引,如果选择固化)

**不提交**:
- split 后的具体数据切片(过大,运行时生成)

## 验收标准

- [ ] `load_split()` 接口签名严格匹配 `docs/roadmap.md` § 6.5
- [ ] train / val / test 时间区间无重叠(可在测试中 assert)
- [ ] `docs/data_split.md` 说明划分方案、比例、**冷启动协议(完整方案,可被 JOB-03/06/07/11 引用)**、对下游 job 的影响
- [ ] 至少一个冒烟测试:加载 split 并检查时间区间不重叠、样本数符合预期

## 参考资料

- 一般推荐系统时序划分综述,如 Meng et al. "Exploring data splitting strategies for the evaluation of recommendation models" (RecSys 2020 LBR),[arXiv:2007.13237](https://arxiv.org/abs/2007.13237)

## 备注

- 划分方案一旦定下来,**所有 DQN 和 LLM 实验都必须使用同一个 split**,否则 JOB-16 的对比无意义。
- 如果后期方案改动,需要在 `docs/data_split.md` 里加版本号(v1, v2, ...),并通知所有下游 job 重跑。
