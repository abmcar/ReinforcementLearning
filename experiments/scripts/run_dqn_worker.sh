#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 0 ]; then
  echo "usage: PYTHON=.venv/bin/python $0" >&2
  exit 2
fi

"${PYTHON:-.venv/bin/python}" -m src.rl.run_experiment \
  --objective worker \
  --model-kind dueling_dqn \
  --seeds 42,43,44 \
  --train-split train \
  --max-transitions 1000 \
  --epochs 2 \
  --max-steps 5000 \
  --batch-size 256 \
  --candidate-k 50 \
  --eval-split test \
  --max-eval-entries 0 \
  --cql-alpha 1.0 \
  --min-candidates 2 \
  --output docs/dqn_results_worker.md
