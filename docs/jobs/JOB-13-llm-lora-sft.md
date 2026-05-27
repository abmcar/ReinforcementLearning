# JOB-13: LoRA SFT 训练管线

**Phase**: C
**依赖**: JOB-11
**Status**: ✅ Completed

---

## 背景

LLM+SFT 主线的核心工程是把开源 LLM(7B 量级)在 JOB-11 构造的数据上做 LoRA 微调。学术界证实 LoRA(rank=8/16)在推荐 SFT 场景效果接近 full fine-tune,且显存友好。本 job 完成训练管线本身,JOB-15 负责跑双目标实验和评估。

## 目标

实现 LoRA SFT 训练管线,在 JOB-11 数据上跑通完整训练并产出可复现脚本、sanity checkpoint 和双目标 adapter。正式评估与对比留到 JOB-15。

## 输入

- **依赖产出**:JOB-11 的训练数据(jsonl 格式,通过 `build_dataset(objective, split)` 取)
- **环境**:**默认本地 GPU 12-24GB(3B + LoRA / QLoRA 路径)**,详见 `docs/roadmap.md` § 6 LLM 主线。≥ 24GB 卡可升级到 7B/8B;无本地 GPU 走云端 API,但 SFT 通常需要 GPU,API 路径主要用于 JOB-12 zero-shot 而非本 job

## 工作内容

- 选定训练框架(主流选项见 `docs/roadmap.md` § 6「LLM 主线」)和底座模型,所选项写入 `docs/llm_sft.md`
- 实现训练 pipeline:
  - 数据 loader:读 JOB-11 jsonl,组装成 chat format
  - LoRA / QLoRA 配置(具体 rank / target_modules / 优化器超参写入 yaml,不要 hardcode)
  - checkpoint 保存策略
- 跑一次完整训练(可先在小数据子集上 sanity)
- 验证 sanity:loss 下降、推理结果有变化(对一个 prompt 比对 SFT 前后输出)

## 产出

**提交到 Git**:
- `src/llm/sft_train.py`(或一个 LLaMA-Factory 配置 yaml)
- `experiments/configs/lora_<model>.yaml`(文件名反映实际使用的底座,如 `lora_qwen3b.yaml`)
- `experiments/scripts/run_lora_sft.sh`
- `docs/llm_sft.md`:训练框架、底座选择、LoRA 配置、超参表、显存/时间实测

**不提交**:
- LoRA 权重、训练日志(放 `outputs/checkpoints/lora/`)

## 验收标准

- [x] 训练 pipeline 跑通至少一次:已完成 `worker` 1k+ sanity、`worker` 全量和 `requester` 同预算训练,loss 曲线与日志见 `docs/llm_sft.md`
- [x] Loss 曲线落到 `docs/figures/JOB-13-loss.png`
- [x] 推理脚本可加载 LoRA 权重并做单条 / batch 推理,SFT 前后推理结果在同一 prompt 上有可见差异(示例见 `docs/llm_sft.md`)
- [x] 显存占用 / 训练时长实测数据在 `docs/llm_sft.md`

## 参考资料

- 底座 / LoRA / 框架选型见 `docs/roadmap.md` § 6「LLM 主线」。

## 备注

- 训练**不需要等到最优**,本 job 只确认管线 work。最终调优在 JOB-15。
