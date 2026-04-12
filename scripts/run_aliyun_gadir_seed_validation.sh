#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

export PYTHONPATH="$PROJECT_ROOT/src"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export HF_HOME="${HF_HOME:-$PROJECT_ROOT/.cache/huggingface}"

CONFIG_PATH="${CONFIG_PATH:-configs/experiment/roberta_gadi_r_aliyun_a10.yaml}"
RESULTS_ROOT="${RESULTS_ROOT:-$PROJECT_ROOT/results}"
EXPERIMENT_TYPE="${EXPERIMENT_TYPE:-roberta-base_MRPC_GADI-R多随机种子验证_A10}"
SEEDS="${SEEDS:-11,42,123,3407,2026}"
LORA_ACCURACY="${LORA_ACCURACY:-0.7818627450980392}"
LORA_GA_ACCURACY="${LORA_GA_ACCURACY:-0.8112745098039216}"
LORA_PATH_LABEL="${LORA_PATH_LABEL:-results/2026-04-04/roberta-base_MRPC_GADI-R修正版完整对比实验/lora_141521}"
LORA_GA_PATH_LABEL="${LORA_GA_PATH_LABEL:-results/2026-04-04/roberta-base_MRPC_GADI-R修正版完整对比实验/lora_ga_143615}"

python scripts/run_gadir_seed_validation.py \
  --config "$CONFIG_PATH" \
  --results-root "$RESULTS_ROOT" \
  --epochs 1 \
  --train-batch-size 16 \
  --seeds "$SEEDS" \
  --lora-accuracy "$LORA_ACCURACY" \
  --lora-ga-accuracy "$LORA_GA_ACCURACY" \
  --lora-path "$LORA_PATH_LABEL" \
  --lora-ga-path "$LORA_GA_PATH_LABEL" \
  --experiment-type "$EXPERIMENT_TYPE"
