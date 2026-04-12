#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

echo "[1/2] Running query-only + topk=1 + calibration_batches=8 + step120"
bash scripts/run_aliyun_gadir_query_only_top1_step120_seed_validation.sh

echo "[2/2] Running query-only + topk=1 + calibration_batches=8 + step140"
bash scripts/run_aliyun_gadir_query_only_top1_step140_seed_validation.sh
