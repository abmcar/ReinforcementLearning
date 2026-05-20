# JOB-15: LLM 双目标训练与评估

**Phase**: C
**依赖**: JOB-13, JOB-04
**Status**: ⬜ Pending

---

## 背景

JOB-13 给了 SFT 训练管线,本 job 类比 JOB-10 之于 DQN:用**两套数据**(参与者目标、请求者目标)分别训 LoRA,跑完整评估,产出 LLM 主线的最终交付。

## 目标

完成 LLM+SFT 主线在参与者和请求者两个目标下的完整训练与评估,产出可被 JOB-16 直接引用的指标表。

## 输入

- **依赖产出**:JOB-13 的训练管线、JOB-11 的数据构造、JOB-04 evaluator、JOB-05 baseline、JOB-12 zero-shot 结果

## 工作内容

- 调用 JOB-11 的 `build_dataset(objective="worker", split=...)` / `build_dataset(objective="requester", split=...)` 产出两份训练集(签名见 `docs/roadmap.md` § 6.5)
- 分别跑 LoRA SFT 训练:`run_lora_sft.sh worker_obj` / `run_lora_sft.sh requester_obj`
- 实现 LLM 模型到 JOB-04 `recommend(worker_id, timestamp, candidates) -> list[int]` 接口的适配层(binary 模式:对每个候选打 Yes/No 概率后排序;签名见 `docs/roadmap.md` § 6.5)
- 多 seed(若资源允许,≥ 2)
- 在 test split 上跑 JOB-04 全指标
- 与 zero-shot(JOB-12)、baselines(JOB-05)对比
- 分析:SFT 提升了哪些指标?哪些没动?两个目标下的策略差异?
- 所有结果写入 `docs/llm_results.md`

**简化路径**(资源不足时):只跑一轮单目标 SFT,在 prompt 里加 system message 切换目标语义(如 `"请优先考虑 worker 收益"` vs `"请优先考虑 project 质量"`)。采用时必须在 `docs/llm_results.md` 里**显著标注是简化方案**,并不在 JOB-16 对比表里被当作严格的双目标实验。

## 产出

**提交到 Git**:
- `experiments/scripts/run_llm_sft_worker.sh`、`run_llm_sft_requester.sh`
- `experiments/configs/lora_worker_obj.yaml` / `lora_requester_obj.yaml`
- `docs/llm_results.md`:结果表、分析、与 baseline / zero-shot 对比

**不提交**:
- LoRA 权重、训练日志(`outputs/`)

## 验收标准

- [ ] 参与者目标和请求者目标分别有完整训练 + 评估结果
- [ ] 与 JOB-05 baseline、JOB-12 zero-shot 对比表
- [ ] `docs/llm_results.md` 含结果表、分析、关键发现
- [ ] 一键脚本可复现 final 实验
- [ ] `docs/llm_results.md` 记录**所有跑过的配置**(不只是 best),包括失败/不显著的结果

## 参考资料

- LLM SFT 选型见 `docs/roadmap.md` § 6「LLM 主线」。

## 备注

- 若采用「简化路径」(见上),需在 `docs/llm_results.md` 显著标注。
