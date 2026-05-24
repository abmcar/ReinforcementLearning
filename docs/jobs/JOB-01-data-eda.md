# JOB-01: 数据 EDA 与统计报告

**Phase**: A
**依赖**: 无
**Status**: 🔵 In Progress

---

## 背景

项目数据是 crowdSPRING 设计众包平台的真实历史日志,但目前没有任何统计文档可参考。后续所有 job(特征工程、数据划分、Reward 设计、Prompt 构造)都需要依赖对数据规模、分布、字段语义的清晰理解。本 job 是所有下游工作的事实基础。

## 目标

完成对 `data/` 三类原始文件(`worker_quality.csv` / `project_list.csv` / `project/` / `entry/`)的探索性分析,产出可复现的统计脚本和一份结构化的数据报告。

## 输入

- **数据**:
  - `data/worker_quality.csv`(worker_id, worker_quality)
  - `data/project_list.csv`(project_id, project_answer_num)
  - `data/project/project_<id>.txt`(JSON,含 category / sub_category / industry / start_date / deadline / entry_count / entry_ids 等)
  - `data/entry/entry_<pid>_<k>.txt`(JSON,含 entry_created_at / author(worker_id) / award_value / finalist / winner / entry_type 等)
- **参考**:`data/sample_read_data.py`

## 工作内容

- 统计三类数据的基本规模(数量、时间范围、关键字段的非空率、唯一值数量)。
- 描述 worker / project / entry 的分布特征(per-worker entry 数分布、per-project entry 数分布、category/industry/sub_category 分布、award_value 分布、时间分布)。
- 识别数据质量问题(缺失、异常值、重复、字段语义不明的字段)并记录。
- 提出**对下游 job 有影响的发现**,例如:数据稀疏度、长尾分布、worker_quality 负值的含义、entry 时间和 project deadline 的关系等。
- 产出可复现的 EDA 脚本(可一键 re-run)和数据报告。

## 产出

**提交到 Git**:
- `src/data/eda.py`(或 notebook,如 `notebooks/01_data_eda.ipynb`):可复现的 EDA 脚本
- `docs/data_report.md`:数据报告(含关键统计表和图说明)
- `docs/figures/`:关键统计图(直方图、时间序列、热力图等)

**不提交**:
- 中间临时 cache、原始数据本身

## 验收标准

- [ ] EDA 脚本可在 `data/` 数据齐全的本地环境一键跑通
- [ ] 产出 `docs/data_fields.md`:**完整枚举** entry / project / worker_quality 的所有顶层 JSON key,与 `docs/roadmap.md` § 2 字段清单**逐一比对**,标出"未在 roadmap 列出"和"roadmap 列出但实际不存在"
- [ ] `docs/data_report.md` 至少包含:三类数据规模、关键字段分布、时间范围、与下游有关的发现列表
- [ ] 关键图表保存到 `docs/figures/JOB-01-*.png` 并在报告中嵌入
- [ ] 报告里**明确写出**对下游 job(特别是 JOB-02 数据划分、JOB-04 评估、JOB-07 reward 设计)的建议

## 参考资料

- `data/sample_read_data.py` —— 字段语义和读取方式的权威参考
- 项目根目录的 `强化学习大作业.md` —— 数据字段的官方说明

## 备注

- `entry/` 目录约 22800 个文件,完整解析可能慢。允许 EDA 阶段做采样统计,但最终统计数字需要全量跑一次并归档。
- crowdSPRING 是**设计**类众包(logo、网站等),不是 MTurk 数据标注。reward 模型设计时注意 award_value、finalist、winner 的语义区别(只有 winner 拿全额,finalist 可能拿部分)。
