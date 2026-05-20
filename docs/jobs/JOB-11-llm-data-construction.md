# JOB-11: LLM 数据构造与 Prompt 设计

**Phase**: C
**依赖**: JOB-03, JOB-06
**Status**: ⬜ Pending

---

## 背景

LLM+SFT 主线的第一步是把原始 worker / project 数据转成自然语言形式的(prompt, response)训练样本。文献表明,把 user 历史 → "推荐下一个 item" 的 prompt 模板有多种范式(P5 / TALLRec / BIGRec)。本 job 选定一种范式,实现数据构造管线,产出后续 SFT 训练用的数据集。

## 目标

设计并实现 LLM 训练数据构造模块,把 worker 历史 + 候选 project 信息转成 SFT 训练样本,产出 train/val/test 的 prompt-response 数据集文件。

## 输入

- **依赖产出**:JOB-03 特征(尤其是 worker 历史交互、project 描述字段)、JOB-06 候选集、JOB-02 冷启动协议(见 `docs/data_split.md`)
- **数据**:project 的 `title` / `brief_questions` / `brief_answers` / `category` / `industry` / `package_name` 等自然语言可读字段(字段名以 JOB-01 EDA 报告为准,若有缺失需相应调整 prompt 模板)

## 工作内容

- 选定 prompt 范式(默认建议见 `docs/roadmap.md` § 6 「LLM 主线」)
- 设计 prompt 模板。建议要素包括:
  - 系统提示:介绍任务和数据上下文
  - worker 画像:quality、活跃 category、历史 entry 摘要(近 N 个 project 的 title/category)
  - 候选 project 信息:title、category、industry、deadline 距离 / award 等
  - 问题:"该 worker 是否会接这个 project? Yes / No" 或 "请从候选中选出最适合的 project"
- **双目标正样本定义**(JOB-15 会用):
  - **参与者(worker)目标**:正样本 = worker 实际接且 award 高 / finalist / winner 的 (worker, project) pair
  - **请求者(requester)目标**:正样本 = project 实际接到的 high-quality(worker_quality > 阈值)entry 的 (worker, project) pair
  - 实现为 `build_dataset(objective, split)`,签名严格匹配 `docs/roadmap.md` § 6.5(`objective: Literal["worker","requester"], split: str -> Path`,返回写出的 jsonl 文件路径)
- 从 train split 构造(prompt, response)样本,其中 response 来自真实日志 ground truth
- 负样本采样策略(对 binary 任务):从 JOB-06 候选集中随机采样未被该 worker 选择的 project 作为 Negative
- 控制 prompt 长度(< 2048 tokens 是经验上限,8B 模型友好)
- 文档:prompt 模板的几个 example、两个目标各自的训练样本规模、prompt 长度分布

## 产出

**提交到 Git**:
- `src/llm/data_builder.py`:数据构造主模块
- `src/llm/prompts.py`:prompt 模板和填充函数
- `docs/llm_data.md`:prompt 设计说明、样本范例、规模统计
- `experiments/configs/llm_data.yaml`:数据构造超参(负样本比例、历史长度 N 等)

**不提交**:
- 实际训练数据集 jsonl 文件(放 `outputs/llm_data/`,可能很大)

## 验收标准

- [ ] `build_dataset(objective, split)` 接口签名严格匹配 `docs/roadmap.md` § 6.5
- [ ] 至少一种 prompt 模板实现完成,数据集构造可一键复现
- [ ] train / val / test 三个 split 的样本数、prompt 长度分布在文档中
- [ ] 文档里给出 3-5 个真实样本示例(prompt + response)
- [ ] 候选集**通过 JOB-06 `get_candidates()` 调用**取得(不能自己重写召回),保证和 DQN/baseline 评估口径一致
- [ ] worker 历史构造遵循 timestamp 约束(不晚于 t),冷启动 worker 按 JOB-02 协议处理

## 参考资料

- Prompt 范式与论文清单见 `docs/roadmap.md` § 6「LLM 主线」。

## 备注

- **众包域用 LLM 做推荐是文献空白**(参见 roadmap 调研)。你的 prompt 设计如果想突出贡献,可以在系统提示里强调"crowdsourcing"语境,而不是泛用电商语境。
- prompt 模板**一旦定下来就不要在 JOB-13 训练时频繁改**(影响可复现性)。如有大改动,在 `docs/llm_data.md` 加版本号。
