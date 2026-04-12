#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
cd "$PROJECT_ROOT"

bash scripts/run_aliyun_rte_seed_suite.sh
bash scripts/run_aliyun_sst2_seed_suite.sh
