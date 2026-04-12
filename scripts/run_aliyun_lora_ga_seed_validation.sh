#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
cd "$PROJECT_ROOT"

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_ENDPOINT="${HF_HUB_ENDPOINT:-$HF_ENDPOINT}"
export HF_HUB_DISABLE_TELEMETRY=1
export HF_HOME="${HF_HOME:-$PROJECT_ROOT/.cache/huggingface}"
export PYTHONPATH="${PYTHONPATH:-$PROJECT_ROOT/src}"
export TOKENIZERS_PARALLELISM=false

SEEDS="${SEEDS:-11,42,123,3407,2026}"
RESULTS_ROOT="${RESULTS_ROOT:-results}"
EPOCHS="${EPOCHS:-1}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-16}"
EXPERIMENT_TYPE="${EXPERIMENT_TYPE:-roberta-base_MRPC_LoRA-GA多随机种子验证_A10}"

python scripts/run_baseline_seed_validation.py \
  --method lora_ga \
  --results-root "$RESULTS_ROOT" \
  --epochs "$EPOCHS" \
  --train-batch-size "$TRAIN_BATCH_SIZE" \
  --seeds "$SEEDS" \
  --experiment-type "$EXPERIMENT_TYPE"
