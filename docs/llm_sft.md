# JOB-13: LoRA SFT 管线

## 1. 目标

`JOB-13` 的目标是把 `JOB-11 -> QLoRA -> checkpoint -> 推理对比` 这条链路跑通，并沉淀成可复现脚本。最终调优和正式评估留到 `JOB-15`。

本轮已确定的主线配置：

- 底座：`Qwen2.5-7B-Instruct`
- 训练方式：`QLoRA + 4 GPU DDP`
- 训练目标：`worker` 与 `requester`
- 训练范围：`worker` 用全量；`requester` 先与 `worker` 对齐预算，避免在 `JOB-13` 阶段被超大样本拖慢

## 2. 代码入口

- 训练脚本：[src/llm/sft_train.py](/root/ReinforcementLearning/src/llm/sft_train.py:1)
- sanity 配置：[experiments/configs/lora_qwen7b.yaml](/root/ReinforcementLearning/experiments/configs/lora_qwen7b.yaml:1)
- worker 全量配置：[experiments/configs/lora_qwen7b_worker_full.yaml](/root/ReinforcementLearning/experiments/configs/lora_qwen7b_worker_full.yaml:1)
- requester 同预算配置：[experiments/configs/lora_qwen7b_requester_matched.yaml](/root/ReinforcementLearning/experiments/configs/lora_qwen7b_requester_matched.yaml:1)
- 启动脚本：[experiments/scripts/run_lora_sft.sh](/root/ReinforcementLearning/experiments/scripts/run_lora_sft.sh:1)
- loss 曲线：[docs/figures/JOB-13-loss.png](/root/ReinforcementLearning/docs/figures/JOB-13-loss.png)

## 3. 训练配置

统一训练超参如下：

| 项目 | 值 |
|---|---|
| Base model | `/nfs/pretrains/Qwen/Qwen2.5-7B-Instruct` |
| Max length | `1024` |
| Epochs | `1` |
| Per-device batch size | `1` |
| Grad accumulation | `8` |
| LoRA rank | `8` |
| LoRA alpha | `16` |
| LoRA dropout | `0.05` |
| Quantization | `4-bit NF4` |
| DType | `bfloat16` |
| GPUs | `4 x GPU` DDP |

数据预算：

- worker full:
  - train `9774`
  - val `1647`
- requester matched:
  - train `9774`
  - val `1647`

`requester` 原始训练集远大于 `worker`，但在 `JOB-13` 阶段先按相同预算训练，便于验证双目标训练链路，并把正式大规模对比留给 `JOB-15`。

## 4. 运行方式

worker sanity:

```bash
PYTHON_BIN=/home/xiayu/miniconda3/envs/llama_factory_/bin/python \
CUDA_VISIBLE_DEVICES=0,1,2,4 \
NPROC_PER_NODE=4 \
CONFIG=experiments/configs/lora_qwen7b.yaml \
bash experiments/scripts/run_lora_sft.sh
```

worker full:

```bash
PYTHON_BIN=/home/xiayu/miniconda3/envs/llama_factory_/bin/python \
CUDA_VISIBLE_DEVICES=0,1,2,4 \
NPROC_PER_NODE=4 \
CONFIG=experiments/configs/lora_qwen7b_worker_full.yaml \
bash experiments/scripts/run_lora_sft.sh
```

requester matched:

```bash
PYTHON_BIN=/home/xiayu/miniconda3/envs/llama_factory_/bin/python \
CUDA_VISIBLE_DEVICES=0,1,2,4 \
NPROC_PER_NODE=4 \
CONFIG=experiments/configs/lora_qwen7b_requester_matched.yaml \
bash experiments/scripts/run_lora_sft.sh
```

单条推理对比：

```bash
PYTHON_BIN=/home/xiayu/miniconda3/envs/llama_factory_/bin/python \
python -m src.llm.sft_train generate \
  --config experiments/configs/lora_qwen7b_worker_full.yaml \
  --adapter-path outputs/checkpoints/lora/qwen25_7b_worker_full_4gpu/checkpoint-305 \
  --prompt-text "..."
```

## 5. 实测结果

### 5.1 worker sanity

| 指标 | 值 |
|---|---|
| Train samples | `1024` |
| Eval samples | `256` |
| Train runtime | `256.86s` |
| Train loss | `0.1958` |
| Final eval loss | `0.1115` |
| 输出目录 | `outputs/checkpoints/lora/qwen25_7b_worker_sanity_4gpu` |

### 5.2 worker full

| 指标 | 值 |
|---|---|
| Train samples | `9774` |
| Eval samples | `1647` |
| Train runtime | `2556.47s` |
| Wall time | `45m45s` |
| Train loss | `0.1167` |
| Final eval loss | `0.1148` |
| 输出目录 | `outputs/checkpoints/lora/qwen25_7b_worker_full_4gpu` |

训练中观察到的单卡显存大约在 `16.7` 到 `18.1 GiB`。

### 5.3 requester matched

| 指标 | 值 |
|---|---|
| Train samples | `9774` |
| Eval samples | `1647` |
| Train runtime | `2486.39s` |
| Wall time | `41m21s` |
| Train loss | `0.1247` |
| Final eval loss | `0.1382` |
| 输出目录 | `outputs/checkpoints/lora/qwen25_7b_requester_matched_4gpu` |

训练中观察到的单卡显存大约在 `13.6` 到 `13.9 GiB`。

## 6. SFT 前后推理对比

使用验证集中的真实 prompt 做单条生成，对比 base model 与 LoRA adapter。

worker prompt:

- baseline:
  - `Yes. The worker has a high quality rating (0.9500 ...`
- SFT adapter:
  - `Yes. The worker has experience in logo redesigns (category 7) and ...`

requester prompt:

- baseline:
  - `Yes. The worker has a high quality rating and significant experience in logo design, ...`
- SFT adapter:
  - `Yes. The worker has a high quality rating (0.9600 ...`

这两组输出都不是简单的逐字重复。SFT 后的回答更快引用训练样本中的结构化特征，如类别、历史经验和质量分数，说明 adapter 已被正确加载，且对输出模式产生了可见影响。

## 7. 结论

`JOB-13` 已完成其目标：

- `Qwen2.5-7B-Instruct + QLoRA + 4 GPU DDP` 训练链路已跑通
- `worker` 与 `requester` 两个目标都已产出可加载 adapter
- loss 曲线已生成
- SFT 前后输出已验证存在可见差异

`JOB-15` 将在此基础上继续做正式双目标评估与 `JOB-12`/RL baseline 对比。
