# JOB-14: 生成式 SFT + grounding(可选)

**Phase**: C(可选)
**依赖**: JOB-13
**Status**: 🟡 Optional

> ⚠️ **本 job 可整体跳过**,跳过不影响 JOB-15/16/17。仅在 JOB-13 顺利完成且时间 / 算力允许时考虑。

---

## 背景

JOB-13 实现的是 binary / discriminative SFT(对每个候选回答 Yes/No 或选 1 个)。生成式范式(BIGRec、P5)让 LLM 直接生成 item 的 title 或 semantic id,然后 grounding 回真实 item id。它的优势是**潜在的零样本能力 + 可解释性**,劣势是**幻觉**(生成不存在的 project)。

## 目标

在 JOB-13 的训练框架基础上,实现 BIGRec 风格的生成式 SFT + grounding,跑通训练和评估,给出与 binary 范式的对比。

## 输入

- **依赖产出**:JOB-13 的 LoRA SFT 框架、JOB-11 的数据构造模块
- **数据**:在 JOB-11 基础上,新增生成式 prompt 模板(response 改成 project title 或 semantic description)

## 工作内容

- 设计生成式 prompt:`"该 worker 适合什么 project? 请描述。"` → response = project title + 关键属性
- 实现 grounding:LLM 生成的描述用 BM25 / 向量检索回到真实 project 候选集
- 统计 grounding 命中率("生成的描述是否能 grounding 到 ground truth project"、"生成的描述 grounding 后落在 JOB-06 候选集内的比例")
- 跑评估:用 grounding 后的 top-K 入 JOB-04 evaluator
- 文档对比 binary(JOB-13) vs generative(JOB-14)的指标和定性差异

## 产出

**提交到 Git**:
- `src/llm/generative_sft.py`(可与 JOB-13 共享主干)
- `src/llm/grounding.py`(BM25 / 向量检索 grounding)
- `experiments/configs/generative_sft.yaml`
- `docs/llm_generative.md`:范式对比、grounding 命中率、最终指标对比

**不提交**:
- 权重和原始生成日志

## 验收标准

- [ ] 生成式 SFT 跑通完整一轮训练
- [ ] grounding 实现,命中率有数据
- [ ] 与 binary 范式在 JOB-04 指标上的对比表

## 参考资料

- BIGRec:[arXiv:2308.08434](https://arxiv.org/abs/2308.08434)、[GitHub](https://github.com/SAI990323/BIGRec)
- P5(text-to-text 统一范式):Geng et al. RecSys 2022, [arXiv:2203.13366](https://arxiv.org/abs/2203.13366)
- LLaRA(hybrid prompt 注入 ID embedding):[arXiv:2312.02445](https://arxiv.org/abs/2312.02445)

## 备注

- **可选 job**,如果时间紧或显卡紧张可跳过,不影响 JOB-15、JOB-16 推进。
- 若效果显著优于 binary 范式,可在最终报告里作为主要 LLM 方法呈现;否则作为对比实验提一句。
