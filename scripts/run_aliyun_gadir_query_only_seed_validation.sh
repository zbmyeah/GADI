#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

export CONFIG_PATH="${CONFIG_PATH:-configs/experiment/roberta_gadi_r_aliyun_a10_query_only.yaml}"
export EXPERIMENT_TYPE="${EXPERIMENT_TYPE:-roberta-base_MRPC_GADI-R多随机种子验证_A10_query_only}"

bash scripts/run_aliyun_gadir_seed_validation.sh
