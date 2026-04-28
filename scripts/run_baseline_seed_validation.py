from __future__ import annotations

import argparse
import copy
import csv
import json
import statistics
import sys
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from gadir.config import ExperimentConfig, load_experiment_config
from gadir.training.trainer import run_training
from gadir.utils.logging import get_logger
from gadir.utils.results import make_run_directory, write_config_snapshot, write_markdown, write_metrics_json

LOGGER = get_logger("baseline_seed_validation")

METHOD_SPECS = {
    "lora": {
        "method_name": "LoRA",
        "config_path": PROJECT_ROOT / "configs" / "experiment" / "roberta_lora.yaml",
        "experiment_type": "roberta-base_MRPC_LoRA多随机种子验证_A10",
        "run_prefix": "lora_seed",
    },
    "lora_ga": {
        "method_name": "LoRA-GA",
        "config_path": PROJECT_ROOT / "configs" / "experiment" / "roberta_lora_ga.yaml",
        "experiment_type": "roberta-base_MRPC_LoRA-GA多随机种子验证_A10",
        "run_prefix": "lora_ga_seed",
    },
}


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _get_seed(payload: dict[str, Any]) -> int | None:
    seed = payload.get("config", {}).get("seed")
    return None if seed is None else int(seed)


def _get_training_cfg(payload: dict[str, Any]) -> dict[str, Any]:
    return payload.get("config", {}).get("training", {})

# 查找结果目录下是否存在匹配的已经运行的实验结果，返回最新的
def _find_existing_result(
    results_root: Path,
    *,
    predicate,
) -> tuple[Path, dict[str, Any]] | None:
    matches: list[tuple[Path, dict[str, Any]]] = []
    for json_path in results_root.rglob("结果.json"):
        payload = _read_json(json_path)
        if predicate(payload):
            matches.append((json_path, payload))
    if not matches:
        return None
    matches.sort(key=lambda item: item[0].stat().st_mtime, reverse=True)
    return matches[0]


def build_seed_config(
    base_config: ExperimentConfig,
    *,
    seed: int,
    num_epochs: int | None,
    train_batch_size: int | None,
) -> ExperimentConfig:
    cfg = copy.deepcopy(base_config)
    cfg.seed = seed
    cfg.training.max_steps = None
    if num_epochs is not None:
        cfg.training.num_epochs = num_epochs
    if train_batch_size is not None:
        cfg.training.train_batch_size = train_batch_size
        cfg.data.calibration_size = max(train_batch_size, cfg.data.calibration_size)
    return cfg


def summarize_history(metrics: dict[str, Any]) -> tuple[float, float, int]:
    eval_history = metrics.get("eval_history", [])
    if not eval_history:
        return (
            float(metrics.get("loss", 0.0)),
            float(metrics.get("accuracy", 0.0)),
            int(metrics.get("total_steps", 0)),
        )
    best_item = max(eval_history, key=lambda item: (item["accuracy"], -item["loss"]))
    return float(best_item["loss"]), float(best_item["accuracy"]), int(best_item["step"])


def _mean(values: list[float]) -> float:
    return statistics.mean(values) if values else 0.0


def _std(values: list[float]) -> float:
    return statistics.stdev(values) if len(values) > 1 else 0.0


def build_experiment_markdown(
    method_name: str,
    config: ExperimentConfig,
    metrics: dict[str, Any],
    elapsed_seconds: float,
    run_dir: Path,
) -> str:
    best_loss, best_accuracy, best_step = summarize_history(metrics)
    runtime = metrics.get("runtime", {})
    return f"""# 实验说明

## 一、实验目的

本实验用于在阿里云 A10 环境下，对 `{method_name}` 在 `{config.data.dataset_config_name.upper()}` 任务上进行多随机种子复现实验，为后续与 GADI-R 的公平比较提供基线结果。

## 二、实验配置

- 方法：`{method_name}`
- 随机种子：`{config.seed}`
- 模型：`{config.model.model_name_or_path}`
- 数据集：`{config.data.dataset_name}/{config.data.dataset_config_name}`
- LoRA Rank：`{config.lora.rank}`
- LoRA Alpha：`{config.lora.alpha}`
- 训练轮数：`{config.training.num_epochs}`
- 训练 batch size：`{config.training.train_batch_size}`
- 评估 batch size：`{config.training.eval_batch_size}`
- AMP：`{config.training.use_amp}`
- AMP dtype：`{config.training.amp_dtype}`
- TF32：`{config.training.allow_tf32}`

## 三、实验结果

- 最终验证集 Loss：`{float(metrics.get("loss", 0.0)):.6f}`
- 最终验证集 Accuracy：`{float(metrics.get("accuracy", 0.0)):.6f}`
- 训练过程中最佳验证集 Loss：`{best_loss:.6f}`
- 训练过程中最佳验证集 Accuracy：`{best_accuracy:.6f}`
- 最佳结果出现步数：`{best_step}`
- 总运行时间（秒）：`{elapsed_seconds:.2f}`
- 额外初始化时间（秒）：`{float(runtime.get('method_initialization_time_seconds', 0.0)):.2f}`
- 重基化额外耗时（秒）：`{float(runtime.get('rebase_overhead_time_seconds', 0.0)):.2f}`
- 峰值显存（MiB）：`{float(runtime.get('peak_memory_allocated_mb', 0.0)):.2f}`

## 四、结果文件说明

- 当前实验目录：`{run_dir}`
- 配置快照：`配置快照.yaml`
- 指标结果：`结果.json`
- 模型权重目录：`模型权重/`
"""


def build_summary_markdown(
    method_name: str,
    summary_rows: list[dict[str, Any]],
    suite_dir: Path,
    seeds: list[int],
    config: ExperimentConfig,
) -> str:
    accuracies = [row["final_accuracy"] for row in summary_rows]
    losses = [row["final_loss"] for row in summary_rows]
    total_times = [row["total_wall_time_seconds"] for row in summary_rows]
    init_times = [row["initialization_time_seconds"] for row in summary_rows]
    rebase_times = [row["rebase_overhead_time_seconds"] for row in summary_rows]
    peak_memories = [row["peak_memory_allocated_mb"] for row in summary_rows]
    best_row = max(summary_rows, key=lambda item: (item["final_accuracy"], -item["final_loss"]))
    worst_row = min(summary_rows, key=lambda item: (item["final_accuracy"], -item["final_loss"]))

    lines = [
        f"# {method_name} 多随机种子验证总结",
        "",
        "## 一、实验设置",
        "",
        f"- 方法：`{method_name}`",
        f"- 模型：`{config.model.model_name_or_path}`",
        f"- 数据集：`{config.data.dataset_name}/{config.data.dataset_config_name}`",
        f"- 随机种子列表：`{', '.join(str(seed) for seed in seeds)}`",
        "- 环境：`Alibaba Cloud A10`",
        "- 已存在的相同配置结果会直接复用，不重复训练。",
        "",
        "## 二、逐种子结果",
        "",
        "| seed | 是否复用 | 最终 Accuracy | 最终 Loss | 最佳 Accuracy | 最佳步数 | 总时间(s) | 初始化(s) | 重基化(s) | 峰值显存(MiB) | 路径 |",
        "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]

    for row in summary_rows:
        lines.append(
            f"| {row['seed']} | {'是' if row['reused'] else '否'} | {row['final_accuracy']:.6f} | {row['final_loss']:.6f} | {row['best_accuracy']:.6f} | {row['best_step']} | {row['total_wall_time_seconds']:.2f} | {row['initialization_time_seconds']:.2f} | {row['rebase_overhead_time_seconds']:.2f} | {row['peak_memory_allocated_mb']:.2f} | `{row['path']}` |"
        )

    lines.extend(
        [
            "",
            "## 三、统计结果",
            "",
            f"- 最终 Accuracy 均值：`{_mean(accuracies):.6f}`",
            f"- 最终 Accuracy 标准差：`{_std(accuracies):.6f}`",
            f"- 最终 Loss 均值：`{_mean(losses):.6f}`",
            f"- 最终 Loss 标准差：`{_std(losses):.6f}`",
            f"- 总运行时间均值（秒）：`{_mean(total_times):.2f}`",
            f"- 额外初始化时间均值（秒）：`{_mean(init_times):.2f}`",
            f"- 重基化额外耗时均值（秒）：`{_mean(rebase_times):.2f}`",
            f"- 峰值显存均值（MiB）：`{_mean(peak_memories):.2f}`",
            f"- 表现最好的种子：`{best_row['seed']}`，最终 Accuracy `{best_row['final_accuracy']:.6f}`",
            f"- 表现最差的种子：`{worst_row['seed']}`，最终 Accuracy `{worst_row['final_accuracy']:.6f}`",
            "",
            "## 四、结论",
            "",
            f"- 当前 `{method_name}` 在 `{config.data.dataset_config_name.upper()}` 上的 A10 公平复现实验已完成，可直接与 GADI-R 做 `mean ± std` 和开销对比。",
            "",
            "## 五、目录说明",
            "",
            f"- 当前实验根目录：`{suite_dir}`",
            "- 每个子目录都包含中文实验说明、配置快照、结果 JSON 和模型权重；复用结果在总表中保留原路径。",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run multi-seed validation for LoRA or LoRA-GA baselines.")
    parser.add_argument("--method", choices=sorted(METHOD_SPECS.keys()), required=True)
    parser.add_argument("--config", default=None, help="Optional override config path.")
    parser.add_argument("--results-root", default=str(PROJECT_ROOT / "results"))
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--train-batch-size", type=int, default=None)
    parser.add_argument("--seeds", default="11,42,123,3407,2026")
    parser.add_argument("--experiment-type", default=None)
    args = parser.parse_args()

    spec = METHOD_SPECS[args.method]
    method_name = spec["method_name"]
    config_path = Path(args.config) if args.config else spec["config_path"]
    experiment_type = args.experiment_type or spec["experiment_type"]
    seeds = [int(item.strip()) for item in args.seeds.split(",") if item.strip()]
    results_root = Path(args.results_root)

    suite_dir = results_root / datetime.now().strftime("%Y-%m-%d") / experiment_type
    suite_dir.mkdir(parents=True, exist_ok=True)
    LOGGER.info(
        "Starting baseline multi-seed validation | method=%s | config=%s | seeds=%s | experiment_type=%s",
        method_name,
        config_path,
        seeds,
        experiment_type,
    )

    base_config = load_experiment_config(config_path)
    expected_epochs = args.epochs if args.epochs is not None else base_config.training.num_epochs
    expected_batch_size = args.train_batch_size if args.train_batch_size is not None else base_config.training.train_batch_size
    summary_rows: list[dict[str, Any]] = []

    for seed in seeds:
        # 判断是否已经运行过，避免重复进行实验
        existing_run = _find_existing_result(
            results_root,
            predicate=lambda payload, seed=seed: (
                payload.get("method") == method_name
                and _get_seed(payload) == seed
                and payload.get("config", {}).get("model", {}).get("model_name_or_path") == base_config.model.model_name_or_path
                and payload.get("config", {}).get("data", {}).get("dataset_name") == base_config.data.dataset_name
                and payload.get("config", {}).get("data", {}).get("dataset_config_name") == base_config.data.dataset_config_name
                and _get_training_cfg(payload).get("num_epochs") == expected_epochs
                and _get_training_cfg(payload).get("train_batch_size") == expected_batch_size
                and _get_training_cfg(payload).get("max_steps") is None
                and _get_training_cfg(payload).get("use_amp") == base_config.training.use_amp
                and _get_training_cfg(payload).get("allow_tf32") == base_config.training.allow_tf32
                and _get_training_cfg(payload).get("pin_memory") == base_config.training.pin_memory
                and _get_training_cfg(payload).get("persistent_workers") == base_config.training.persistent_workers
                and _get_training_cfg(payload).get("dataloader_num_workers") == base_config.training.dataloader_num_workers
            ),
        )

        if existing_run is not None:
            run_json_path, payload = existing_run
            LOGGER.info("Reusing existing %s seed=%s result: %s", method_name, seed, run_json_path.parent)
            metrics = payload.get("metrics", {})    # 关键运行结果
            elapsed_seconds = float(payload.get("elapsed_seconds", 0.0))
            reused = True
            run_dir = run_json_path.parent
        else:
            LOGGER.info("Running %s seed=%s", method_name, seed)
            config = build_seed_config(
                base_config,
                seed=seed,
                num_epochs=args.epochs,
                train_batch_size=args.train_batch_size,
            )
            run_dir = make_run_directory(
                results_root=results_root,
                experiment_type=experiment_type,
                method_name=f"{spec['run_prefix']}_{seed}",
                timestamp=datetime.now(),
            )
            config.training.output_dir = str(run_dir / "模型权重")
            write_config_snapshot(config, run_dir / "配置快照.yaml")

            start = time.perf_counter()
            metrics = run_training(config)
            elapsed_seconds = time.perf_counter() - start

            payload = {
                "method": method_name,
                "seed": seed,
                "metrics": metrics,
                "elapsed_seconds": elapsed_seconds,
                "config": asdict(config),
            }
            write_metrics_json(payload, run_dir / "结果.json")
            write_markdown(
                run_dir / "实验说明.md",
                build_experiment_markdown(method_name, config, metrics, elapsed_seconds, run_dir),
            )
            reused = False
        # 统计最小损失，最高精确度以及发生的迭代步骤
        best_loss, best_accuracy, best_step = summarize_history(metrics)
        runtime = metrics.get("runtime", {})
        #各个随机种子对比CSV
        summary_rows.append(
            {
                "seed": seed,
                "reused": reused,
                "final_loss": float(metrics.get("loss", 0.0)),
                "final_accuracy": float(metrics.get("accuracy", 0.0)),
                "best_loss": best_loss,
                "best_accuracy": best_accuracy,
                "best_step": best_step,
                "elapsed_seconds": elapsed_seconds,
                "total_wall_time_seconds": float(runtime.get("total_wall_time_seconds", elapsed_seconds)),
                "initialization_time_seconds": float(runtime.get("method_initialization_time_seconds", 0.0)),
                "rebase_overhead_time_seconds": float(runtime.get("rebase_overhead_time_seconds", 0.0)),
                "peak_memory_allocated_mb": float(runtime.get("peak_memory_allocated_mb", 0.0)),
                "path": str(run_dir),
            }
        )
        LOGGER.info(
            "Finished %s seed=%s | reused=%s | final_accuracy=%.6f | final_loss=%.6f",
            method_name,
            seed,
            reused,
            float(metrics.get("accuracy", 0.0)),
            float(metrics.get("loss", 0.0)),
        )

    summary_rows.sort(key=lambda item: item["seed"])

    with (suite_dir / "结果汇总.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "seed",
                "reused",
                "final_loss",
                "final_accuracy",
                "best_loss",
                "best_accuracy",
                "best_step",
                "elapsed_seconds",
                "total_wall_time_seconds",
                "initialization_time_seconds",
                "rebase_overhead_time_seconds",
                "peak_memory_allocated_mb",
                "path",
            ],
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    write_markdown(
        suite_dir / "对比总结.md",
        build_summary_markdown(method_name, summary_rows, suite_dir, seeds, base_config),
    )
    LOGGER.info("Finished %s multi-seed validation. Summary directory: %s", method_name, suite_dir)


if __name__ == "__main__":
    main()
