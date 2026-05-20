# JOB-12: Zero-shot LLM 基线

**Phase**: C
**依赖**: JOB-11, JOB-04
**Status**: ⬜ Pending

---

## 背景

在做 LoRA SFT(JOB-13)之前,先用未微调的 LLM(zero-shot 或 few-shot prompt)跑一遍评估,作为 SFT 的对照基线。这能回答"SFT 真的有用吗?"以及"LLM 推理本身的天花板在哪"。

## 目标

用 JOB-11 设计的 prompt 模板,在 zero-shot / few-shot 模式下用一个开源 LLM 跑完整评估,产出 baseline 指标。

## 输入

- **依赖产出**:JOB-11 的 prompt 模板和 test split 数据、JOB-04 evaluator

## 工作内容

- 选定一个或多个 LLM 做 zero-shot 测试(默认底座 / 降级路径见 `docs/roadmap.md` § 6「LLM 主线」;本地无 GPU 默认走 **OpenAI `gpt-4o-mini` 或 DeepSeek `deepseek-chat`**,API key 走环境变量,不入 Git)
- 实现 LLM 推理接口,适配 JOB-04 evaluator 的 `recommend(worker_id, timestamp, candidates) -> list[int]`(签名见 `docs/roadmap.md` § 6.5)
- 对应两种范式都做(若 JOB-11 实现了多种):
  - Binary 模式:对每个候选 ask "Yes/No",按概率 / logit 排序
  - List-wise 模式:把候选集塞入 prompt,直接让 LLM 输出排序
- 至少做一次 few-shot 实验(在 prompt 里加 1-3 个例子),与 zero-shot 对比
- 评估并写报告

## 产出

**提交到 Git**:
- `src/llm/zero_shot.py`:zero-shot 推理 + 评估
- `experiments/scripts/run_llm_zero_shot.sh`
- `docs/llm_zero_shot.md`:模型选择、推理设置(temperature、max_tokens、batch_size)、zero-shot vs few-shot 结果对比

**不提交**:
- LLM 输出原始日志(放 `outputs/llm_runs/`)

## 验收标准

- [ ] 至少一个 LLM 跑通 zero-shot 评估
- [ ] zero-shot 和(至少一种)few-shot 结果都报
- [ ] 与 JOB-05 baseline、未来 JOB-15 SFT 结果可直接对比(指标口径一致)

## 参考资料

- "Large Language Models are Zero-Shot Rankers for Recommender Systems":[arXiv:2305.08845](https://arxiv.org/abs/2305.08845)

## 备注

- list-wise 模式有 position bias(LLM 倾向于选靠前的候选),需要在 `docs/llm_zero_shot.md` 里写明并尝试缓解(随机化候选顺序、多次推理取均值)。
- 如果完全没有本地 GPU,允许用 API,但需要在文档里写明 cost 估算和样本采样策略(全量推理可能很贵)。
