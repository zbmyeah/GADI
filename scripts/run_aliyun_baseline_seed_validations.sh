#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
cd "$PROJECT_ROOT"

bash scripts/run_aliyun_lora_seed_validation.sh
bash scripts/run_aliyun_lora_ga_seed_validation.sh
