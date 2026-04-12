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

python scripts/run_baseline_seed_validation.py \
  --method lora \
  --config configs/experiment/roberta_lora_aliyun_a10_rte.yaml \
  --results-root "$RESULTS_ROOT" \
  --epochs "$EPOCHS" \
  --train-batch-size "$TRAIN_BATCH_SIZE" \
  --seeds "$SEEDS" \
  --experiment-type roberta-base_RTE_LoRA多随机种子验证_A10

python scripts/run_baseline_seed_validation.py \
  --method lora_ga \
  --config configs/experiment/roberta_lora_ga_aliyun_a10_rte.yaml \
  --results-root "$RESULTS_ROOT" \
  --epochs "$EPOCHS" \
  --train-batch-size "$TRAIN_BATCH_SIZE" \
  --seeds "$SEEDS" \
  --experiment-type roberta-base_RTE_LoRA-GA多随机种子验证_A10

python scripts/run_gadir_seed_validation.py \
  --config configs/experiment/roberta_gadi_r_aliyun_a10_rte_query_only_top1_step120.yaml \
  --results-root "$RESULTS_ROOT" \
  --epochs "$EPOCHS" \
  --train-batch-size "$TRAIN_BATCH_SIZE" \
  --seeds "$SEEDS" \
  --experiment-type roberta-base_RTE_GADI-R多随机种子验证_A10

python scripts/generate_task_comparison_report.py \
  --results-root "$RESULTS_ROOT" \
  --task-name RTE \
  --suite-name roberta-base_RTE_A10_公平多随机种子总对比报告 \
  --lora-experiment-type roberta-base_RTE_LoRA多随机种子验证_A10 \
  --lora-ga-experiment-type roberta-base_RTE_LoRA-GA多随机种子验证_A10 \
  --gadi-experiment-type roberta-base_RTE_GADI-R多随机种子验证_A10
