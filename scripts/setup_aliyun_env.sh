#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

python - <<'PY'
import torch
print("python ready")
print("torch version:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("cuda device count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("cuda device name:", torch.cuda.get_device_name(0))
PY

python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements-aliyun.txt
