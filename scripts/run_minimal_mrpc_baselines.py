from __future__ import annotations

import argparse
import copy
import csv
import sys
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from gadir.config import ExperimentConfig, load_experiment_config
from gadir.training.trainer import run_training
from gadir.utils.logging import get_logger
from gadir.utils.results import make_run_directory, write_config_snapshot, write_markdown, write_metrics_json

LOGGER = get_logger("minimal_mrpc_baselines")

BASELINE_CONFIGS = {
    "LoRA": PROJECT_ROOT / "configs" / "experiment" / "roberta_lora.yaml",
    "LoRA-GA": PROJECT_ROOT / "configs" / "experiment" / "roberta_lora_ga.yaml",
    "GADI-R": PROJECT_ROOT / "configs" / "experiment" / "roberta_gadi_r.yaml",
}


def build_minimal_config(config: ExperimentConfig, method_label: str) -> ExperimentConfig:
    cfg = copy.deepcopy(config)
    cfg.seed = 42
    cfg.training.num_epochs = 1
    cfg.training.max_steps = 20
    cfg.training.train_batch_size = 8
    cfg.training.eval_batch_size = 32
    cfg.training.gradient_accumulation_steps = 1
    cfg.training.log_every_steps = 5
    cfg.training.eval_every_steps = 10
    cfg.data.calibration_size = 8
    cfg.lora_ga.calibration_batches = 1
    if method_label == "GADI-R":
        cfg.rebase.enabled = True
        cfg.rebase.interval_steps = 10
        cfg.rebase.drift_threshold = 0.05
        cfg.rebase.topk_layers = 2
        cfg.rebase.calibration_batches = 1
        cfg.rebase.use_residual_gradient = True
    return cfg


def build_experiment_markdown(
    method_label: str,
    config: ExperimentConfig,
    metrics: dict,
    elapsed_seconds: float,
    run_dir: Path,
) -> str:
    return f"""# 实验说明

## 一、实验目的

本实验用于跑通 `roberta-base + MRPC` 的最小可复现实验，并在统一设置下比较 `{method_label}` 方法的效果。

## 二、实验配置

- 实验方法：{method_label}
- 模型：{config.model.model_name_or_path}
- 数据集：{config.data.dataset_name}/{config.data.dataset_config_name}
- 任务类型：{config.model.task_type}
- LoRA Rank：{config.lora.rank}
- LoRA Alpha：{config.lora.alpha}
- 最大训练步数：{config.training.max_steps}
- 训练批大小：{config.training.train_batch_size}
- 校准样本数：{config.data.calibration_size}
- 随机种子：{config.seed}

## 三、实验结果

- 最终验证集 Loss：{metrics.get("loss", 0.0):.6f}
- 最终验证集 Accuracy：{metrics.get("accuracy", 0.0):.6f}
- 总耗时（秒）：{elapsed_seconds:.2f}

## 四、结果文件说明

- 当前实验目录：`{run_dir}`
- 配置快照：`配置快照.yaml`
- 指标结果：`结果.json`
- 模型权重目录：`模型权重/`

## 五、备注

本次实验属于“最小实验”，主要目标是验证代码流程、结果落盘和方法间相对表现。
由于当前设置仅训练少量 step，结果主要用于开发阶段对比，不直接作为最终论文结论。
"""


def build_summary_markdown(summary_rows: list[dict], suite_dir: Path) -> str:
    lines = [
        "# 对比总结",
        "",
        "## 一、实验说明",
        "",
        "本目录汇总了 `roberta-base + MRPC` 最小对比实验的结果。三组方法使用统一的训练步数和数据设置，以便快速比较开发阶段的相对表现。",
        "",
        "## 二、结果汇总",
        "",
        "| 方法 | 验证集 Loss | 验证集 Accuracy | 耗时（秒） | 实验目录 |",
        "| --- | ---: | ---: | ---: | --- |",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['method']} | {row['loss']:.6f} | {row['accuracy']:.6f} | {row['elapsed_seconds']:.2f} | `{row['run_dir']}` |"
        )

    sorted_rows = sorted(summary_rows, key=lambda item: item["accuracy"], reverse=True)
    best_row = sorted_rows[0]
    lines.extend(
        [
            "",
            "## 三、结论",
            "",
            f"- 在本次最小实验中，验证集 Accuracy 最高的方法是 `{best_row['method']}`，Accuracy 为 `{best_row['accuracy']:.6f}`。",
            "- 需要注意：本次实验只跑了少量 step，结论主要用于验证实现是否跑通，以及观察方法早期训练趋势。",
            "",
            "## 四、目录说明",
            "",
            f"- 当前对比实验根目录：`{suite_dir}`",
            "- 每个子目录都包含中文实验说明、配置快照、结果 JSON 和模型权重。",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run minimal RoBERTa-base + MRPC baseline comparisons.")
    parser.add_argument(
        "--results-root",
        default=str(PROJECT_ROOT / "results"),
        help="Root directory used to store experiment outputs.",
    )
    args = parser.parse_args()

    experiment_type = "roberta-base_MRPC_最小对比实验"
    suite_timestamp = datetime.now()
    suite_dir = Path(args.results_root) / suite_timestamp.strftime("%Y-%m-%d") / experiment_type
    suite_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict] = []

    for method_label, config_path in BASELINE_CONFIGS.items():
        LOGGER.info("Running baseline: %s", method_label)
        raw_config = load_experiment_config(config_path)
        config = build_minimal_config(raw_config, method_label)

        run_dir = make_run_directory(
            results_root=args.results_root,
            experiment_type=experiment_type,
            method_name=method_label.lower().replace("-", "_"),
            timestamp=datetime.now(),
        )
        config.training.output_dir = str(run_dir / "模型权重")
        write_config_snapshot(config, run_dir / "配置快照.yaml")

        start_time = time.perf_counter()
        metrics = run_training(config)
        elapsed_seconds = time.perf_counter() - start_time

        result_payload = {
            "method": method_label,
            "metrics": metrics,
            "elapsed_seconds": elapsed_seconds,
            "config": asdict(config),
        }
        write_metrics_json(result_payload, run_dir / "结果.json")
        write_markdown(
            run_dir / "实验说明.md",
            build_experiment_markdown(method_label, config, metrics, elapsed_seconds, run_dir),
        )

        summary_rows.append(
            {
                "method": method_label,
                "loss": float(metrics.get("loss", 0.0)),
                "accuracy": float(metrics.get("accuracy", 0.0)),
                "elapsed_seconds": elapsed_seconds,
                "run_dir": str(run_dir),
            }
        )

    with (suite_dir / "结果汇总.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["method", "loss", "accuracy", "elapsed_seconds", "run_dir"],
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    write_markdown(suite_dir / "对比总结.md", build_summary_markdown(summary_rows, suite_dir))
    LOGGER.info("Finished all baseline runs. Summary directory: %s", suite_dir)


if __name__ == "__main__":
    main()
