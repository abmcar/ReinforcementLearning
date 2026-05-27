#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
CONFIG="${CONFIG:-experiments/configs/lora_qwen7b.yaml}"
MODE="${MODE:-train}"
PROMPT_TEXT="${PROMPT_TEXT:-}"
BASELINE_ONLY="${BASELINE_ONLY:-0}"
ADAPTER_PATH="${ADAPTER_PATH:-}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
MASTER_PORT="${MASTER_PORT:-29517}"

ARGS=("${MODE}" "--config" "${CONFIG}")

if [[ -n "${PROMPT_TEXT}" ]]; then
  ARGS+=("--prompt-text" "${PROMPT_TEXT}")
fi

if [[ "${BASELINE_ONLY}" == "1" ]]; then
  ARGS+=("--baseline-only")
fi

if [[ -n "${ADAPTER_PATH}" ]]; then
  ARGS+=("--adapter-path" "${ADAPTER_PATH}")
fi

if [[ "${NPROC_PER_NODE}" -gt 1 ]]; then
  "${PYTHON_BIN}" -m torch.distributed.run \
    --standalone \
    --nproc_per_node "${NPROC_PER_NODE}" \
    --master_port "${MASTER_PORT}" \
    -m src.llm.sft_train \
    "${ARGS[@]}"
else
  "${PYTHON_BIN}" -m src.llm.sft_train "${ARGS[@]}"
fi
