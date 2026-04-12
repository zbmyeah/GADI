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

LOGGER = get_logger("full_mrpc_baselines")

BASELINE_CONFIGS = {
    "LoRA": PROJECT_ROOT / "configs" / "experiment" / "roberta_lora.yaml",
    "LoRA-GA": PROJECT_ROOT / "configs" / "experiment" / "roberta_lora_ga.yaml",
    "GADI-R": PROJECT_ROOT / "configs" / "experiment" / "roberta_gadi_r.yaml",
}


def build_full_config(
    config: ExperimentConfig,
    method_label: str,
    num_epochs: int,
    train_batch_size: int,
) -> ExperimentConfig:
    cfg = copy.deepcopy(config)
    cfg.seed = 42
    cfg.training.num_epochs = num_epochs
    cfg.training.max_steps = None
    cfg.training.train_batch_size = train_batch_size
    cfg.training.eval_batch_size = 32
    cfg.training.gradient_accumulation_steps = 1
    cfg.training.log_every_steps = 20
    cfg.training.eval_every_steps = 40
    cfg.data.calibration_size = max(train_batch_size, 32)
    cfg.lora_ga.calibration_batches = 1

    if method_label == "GADI-R":
        cfg.rebase.enabled = True
        cfg.rebase.interval_steps = 80
        cfg.rebase.warmup_steps = 120
        cfg.rebase.max_rebases = 1
        cfg.rebase.drift_threshold = 0.03
        cfg.rebase.topk_layers = 2
        cfg.rebase.calibration_batches = 4
        cfg.rebase.use_residual_gradient = True
        cfg.rebase.gradient_mix = 0.7
        cfg.rebase.calibration_eval_mode = True
        cfg.rebase.reset_optimizer_state = False
    return cfg


def summarize_history(metrics: dict) -> tuple[float, float, int]:
    eval_history = metrics.get("eval_history", [])
    if not eval_history:
        return float(metrics.get("loss", 0.0)), float(metrics.get("accuracy", 0.0)), int(metrics.get("total_steps", 0))

    best_item = max(eval_history, key=lambda item: (item["accuracy"], -item["loss"]))
    return float(best_item["loss"]), float(best_item["accuracy"]), int(best_item["step"])


def build_experiment_markdown(
    method_label: str,
    config: ExperimentConfig,
    metrics: dict,
    elapsed_seconds: float,
    run_dir: Path,
) -> str:
    best_loss, best_accuracy, best_step = summarize_history(metrics)
    final_loss = float(metrics.get("loss", 0.0))
    final_accuracy = float(metrics.get("accuracy", 0.0))

    return f"""# 实验说明

## 一、实验目的

本实验用于在 `roberta-base + MRPC` 设置下进行较完整的基线对比，验证 `{method_label}` 在更长训练过程中的表现，并观察其是否优于 LoRA 与 LoRA-GA。

## 二、实验配置

- 实验方法：{method_label}
- 模型：{config.model.model_name_or_path}
- 数据集：{config.data.dataset_name}/{config.data.dataset_config_name}
- 任务类型：{config.model.task_type}
- LoRA Rank：{config.lora.rank}
- LoRA Alpha：{config.lora.alpha}
- 训练轮数：{config.training.num_epochs}
- 训练批大小：{config.training.train_batch_size}
- 评估批大小：{config.training.eval_batch_size}
- 校准样本数：{config.data.calibration_size}
- 评估间隔步数：{config.training.eval_every_steps}
- 随机种子：{config.seed}

## 三、实验结果

- 最终验证集 Loss：{final_loss:.6f}
- 最终验证集 Accuracy：{final_accuracy:.6f}
- 训练过程最好验证集 Loss：{best_loss:.6f}
- 训练过程最好验证集 Accuracy：{best_accuracy:.6f}
- 最好结果出现步数：{best_step}
- 总耗时（秒）：{elapsed_seconds:.2f}

## 四、结果文件说明

- 当前实验目录：`{run_dir}`
- 配置快照：`配置快照.yaml`
- 指标结果：`结果.json`
- 模型权重目录：`模型权重/`

## 五、备注

本次实验使用完整训练轮次，而非开发阶段的极少量 step。若 GADI-R 在最终指标或最佳中间指标上优于基线，说明动态重基化策略具有实际价值；若没有超过，则说明当前超参数或方法设计仍需继续调整。
"""


def build_summary_markdown(summary_rows: list[dict], suite_dir: Path, num_epochs: int, train_batch_size: int) -> str:
    lines = [
        "# 完整实验总结",
        "",
        "## 一、实验设置",
        "",
        f"- 模型：`roberta-base`",
        f"- 数据集：`GLUE/MRPC`",
        f"- 训练轮数：`{num_epochs}`",
        f"- 训练批大小：`{train_batch_size}`",
        "- 对比方法：`LoRA`、`LoRA-GA`、`GADI-R`",
        "- 随机种子：`42`",
        "",
        "## 二、结果汇总",
        "",
        "| 方法 | 最终 Loss | 最终 Accuracy | 最佳 Loss | 最佳 Accuracy | 最佳步数 | 耗时（秒） |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    for row in summary_rows:
        lines.append(
            f"| {row['method']} | {row['final_loss']:.6f} | {row['final_accuracy']:.6f} | {row['best_loss']:.6f} | {row['best_accuracy']:.6f} | {row['best_step']} | {row['elapsed_seconds']:.2f} |"
        )

    best_final = max(summary_rows, key=lambda item: (item["final_accuracy"], -item["final_loss"]))
    best_peak = max(summary_rows, key=lambda item: (item["best_accuracy"], -item["best_loss"]))

    lines.extend(
        [
            "",
            "## 三、结论",
            "",
            f"- 最终验证集指标最好的是 `{best_final['method']}`，最终 Accuracy 为 `{best_final['final_accuracy']:.6f}`。",
            f"- 训练过程中最好指标最高的是 `{best_peak['method']}`，最佳 Accuracy 为 `{best_peak['best_accuracy']:.6f}`，出现在 step `{best_peak['best_step']}`。",
            "- 如果 GADI-R 没有超过基线，需要优先检查重基化触发频率、候选层选择和 residual gradient 计算方式。",
            "",
            "## 四、目录说明",
            "",
            f"- 当前实验根目录：`{suite_dir}`",
            "- 每个方法子目录都包含中文实验说明、配置快照、结果 JSON 和模型权重。",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a fuller RoBERTa-base + MRPC baseline comparison.")
    parser.add_argument("--results-root", default=str(PROJECT_ROOT / "results"))
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--train-batch-size", type=int, default=16)
    parser.add_argument(
        "--experiment-type",
        default="roberta-base_MRPC_完整对比实验",
        help="Directory label used under results/<date>/",
    )
    args = parser.parse_args()

    experiment_type = args.experiment_type
    suite_timestamp = datetime.now()
    suite_dir = Path(args.results_root) / suite_timestamp.strftime("%Y-%m-%d") / experiment_type
    suite_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict] = []

    for method_label, config_path in BASELINE_CONFIGS.items():
        LOGGER.info("Running full baseline: %s", method_label)
        raw_config = load_experiment_config(config_path)
        config = build_full_config(
            raw_config,
            method_label=method_label,
            num_epochs=args.epochs,
            train_batch_size=args.train_batch_size,
        )

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

        best_loss, best_accuracy, best_step = summarize_history(metrics)
        summary_rows.append(
            {
                "method": method_label,
                "final_loss": float(metrics.get("loss", 0.0)),
                "final_accuracy": float(metrics.get("accuracy", 0.0)),
                "best_loss": best_loss,
                "best_accuracy": best_accuracy,
                "best_step": best_step,
                "elapsed_seconds": elapsed_seconds,
                "run_dir": str(run_dir),
            }
        )

    with (suite_dir / "结果汇总.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "method",
                "final_loss",
                "final_accuracy",
                "best_loss",
                "best_accuracy",
                "best_step",
                "elapsed_seconds",
                "run_dir",
            ],
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    write_markdown(
        suite_dir / "对比总结.md",
        build_summary_markdown(summary_rows, suite_dir, args.epochs, args.train_batch_size),
    )
    LOGGER.info("Finished full experiment suite. Summary directory: %s", suite_dir)


if __name__ == "__main__":
    main()
