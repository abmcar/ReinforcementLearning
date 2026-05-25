# JOB-03: 公共特征工程

**Phase**: A
**依赖**: JOB-01, JOB-02
**Status**: ✅ Completed

---

## 背景

DQN 主线和 LLM 主线都需要从原始数据中提取 worker / project / 交互特征,但提取口径必须一致才能保证对比公平。本 job 实现公共特征工程模块,同时服务两条主线。

## 目标

实现一个统一的特征工程模块,从原始数据(entry / project / worker)中提取 worker 特征、project 特征、worker-project 交互特征,产出可复用的 `build_features()` 接口和缓存机制。

## 输入

- **依赖产出**:JOB-02 的 `load_split()`(在 train 上 fit、在 val/test 上 transform)、JOB-01 的字段语义结论
- **数据**:全部原始数据

## 工作内容

- 设计 worker 特征。建议覆盖维度:
  - 静态:worker_quality
  - 动态:历史 entry 总数、不同 category 上的活跃度、获奖率、平均 award、最近 N 个交互
- 设计 project 特征。建议覆盖维度:
  - 静态:category / sub_category / industry / package(award) / 任务时长(deadline - start_date)
  - 动态(在某时间点 t):已收到的 entry 数、距离 deadline 的天数、当前 winner / finalist 数
- 设计交互特征(可选):worker 在该 category/industry 上的历史成绩、worker 是否参与过该 requester(若数据可推断)
- 实现 `build_features(split_name)` 接口,返回结构化特征(numpy / pandas / tensor),并支持本地缓存避免重复计算。
- 文档说明每个特征的定义、来源、归一化方式。

## 产出

**提交到 Git**:
- `src/features/__init__.py`、`src/features/worker.py`、`src/features/project.py`、`src/features/interaction.py`(或合并到一个文件)
- `src/features/build.py`:`build_features()` 入口
- `docs/features.md`:特征清单(每个特征:名称、定义、类型、范围、来源)

**不提交**:
- 缓存的特征矩阵(放 `outputs/features/`,加入 `.gitignore`)

## 验收标准

- [x] `build_features("train")` 等可一键跑通
- [x] 特征构建无数据泄漏(测试集特征不依赖测试集时间点之后的信息)
- [x] `docs/features.md` 清单完整,所有维度都可追溯到代码
- [x] 提供一个冒烟脚本,加载 train split 跑出特征 shape 并打印若干样本

## 参考资料

- 推荐系统特征工程通用经验:任何 CTR 预估 baseline 实现的特征部分(如 DeepCTR、RecBole)

## 备注

- 特征工程容易出现**数据泄漏**:计算 worker 在时间点 t 的"历史成绩"时,只能用 t 之前的 entry。请在代码里加 assertion 或 unit test 保护。
- LLM 主线(JOB-11)会从这里的 worker / project 特征构造自然语言 prompt,所以**特征要有可解释的 name 和取值**,而不是匿名 embedding。
