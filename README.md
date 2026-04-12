# GADI-R Research Scaffold

This repository is a graduation-project code scaffold for studying parameter-efficient fine-tuning with:

- Vanilla LoRA
- LoRA-GA
- GADI-R (Gradient-Aligned Dynamic Re-basing)

The implementation intentionally reuses the `peft` LoRA stack instead of rebuilding LoRA from scratch. The custom code focuses on:

- collecting full gradients on LoRA target layers,
- LoRA-GA style gradient-aligned initialization,
- GADI-R style drift detection and function-preserving dynamic re-basing,
- a lightweight training loop that leaves room for later large-model optimization.

## Quick Start

Windows:

```powershell
.\.venv\Scripts\Activate.ps1
$env:PYTHONPATH = "$PWD\src"
python scripts\train.py --config configs\experiment\roberta_lora.yaml
```

Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
export PYTHONPATH=$PWD/src
python scripts/train.py --config configs/experiment/roberta_lora.yaml
```

## Aliyun A10

For an Alibaba Cloud GPU server such as `ecs.gn7i-c8g1.2xlarge (8 vCPU, 30 GiB, NVIDIA A10 x1)`, the project can run without algorithmic changes. A GPU-friendly config is provided at:

```text
configs/experiment/roberta_gadi_r_aliyun_a10.yaml
```

This config keeps the training setup comparable to the existing MRPC baselines while enabling:

- CUDA mixed precision with `bfloat16`
- TF32 on NVIDIA Ampere
- faster DataLoader settings for Linux

Recommended multi-seed launch command on Linux:

```bash
bash scripts/setup_aliyun_env.sh
bash scripts/run_aliyun_gadir_seed_validation.sh
```

For a query-only re-basing ablation on the same A10 server:

```bash
bash scripts/run_aliyun_gadir_query_only_seed_validation.sh
```

For the next recommended stabilizing follow-ups:

```bash
bash scripts/run_aliyun_gadir_query_only_top1_step120_seed_validation.sh
bash scripts/run_aliyun_gadir_query_only_top1_step140_seed_validation.sh
```

Or run both sequentially:

```bash
bash scripts/run_aliyun_gadir_query_only_top1_followups.sh
```

For fair multi-seed baseline reruns on the same A10 server:

```bash
bash scripts/run_aliyun_lora_seed_validation.sh
bash scripts/run_aliyun_lora_ga_seed_validation.sh
```

Or run both baselines sequentially:

```bash
bash scripts/run_aliyun_baseline_seed_validations.sh
```

The helper script sets `PYTHONPATH`, a local Hugging Face cache directory, and runs the fixed best-known GADI-R setting:

- `step120`
- `gradient_mix=0.5`
- `query_only`
- `topk_layers=1`

For the full RTE suite on the same A10 server:

```bash
bash scripts/run_aliyun_rte_seed_suite.sh
```

For the full SST-2 suite:

```bash
bash scripts/run_aliyun_sst2_seed_suite.sh
```

Or run both task suites sequentially:

```bash
bash scripts/run_aliyun_rte_sst2_suites.sh
```

## Current Scope

- The scaffold is ready for sequence classification experiments first.
- Large-model memory optimization is separated behind clear modules, so we can later replace the prototype gradient collector with a layer-wise implementation for 7B-scale runs.
- LoRA is provided by `peft`; LoRA-GA and GADI-R only customize initialization and dynamic updates on top of it.

## Directory Overview

```text
configs/            experiment yaml files
scripts/            training and utility entrypoints
src/gadir/          project source package
```
