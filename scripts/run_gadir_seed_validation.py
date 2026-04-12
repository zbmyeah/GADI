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

LOGGER = get_logger("gadir_seed_validation")


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _get_rebase_cfg(payload: dict[str, Any]) -> dict[str, Any]:
    return payload.get("config", {}).get("rebase", {})


def _get_training_cfg(payload: dict[str, Any]) -> dict[str, Any]:
    return payload.get("config", {}).get("training", {})


def _get_seed(payload: dict[str, Any]) -> int | None:
    seed = payload.get("config", {}).get("seed")
    return None if seed is None else int(seed)


def _get_gradient_mix(payload: dict[str, Any]) -> float | None:
    rebase_cfg = _get_rebase_cfg(payload)
    value = rebase_cfg.get("gradient_mix", payload.get("gradient_mix"))
    return None if value is None else float(value)


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
    num_epochs: int,
    train_batch_size: int,
) -> ExperimentConfig:
    cfg = copy.deepcopy(base_config)
    cfg.seed = seed
    cfg.method = "gadi_r"
    cfg.training.num_epochs = num_epochs
    cfg.training.max_steps = None
    cfg.training.train_batch_size = train_batch_size
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


def summarize_rebase(metrics: dict[str, Any]) -> tuple[str, bool, list[str]]:
    history = metrics.get("method_artifacts", {}).get("rebase_history", [])
    if not history:
        return "未发生重基化。", False, []

    lines: list[str] = []
    refreshed_layers: list[str] = []
    value_selected = False
    for item in history:
        current_layers = item.get("refreshed_layers", [])
        refreshed_layers.extend(current_layers)
        if any("value" in layer for layer in current_layers):
            value_selected = True
        lines.append(f"- step {item.get('step')}: {', '.join(current_layers) if current_layers else '无层被刷新'}")
    return "\n".join(lines), value_selected, refreshed_layers


def build_experiment_markdown(
    config: ExperimentConfig,
    metrics: dict[str, Any],
    elapsed_seconds: float,
    run_dir: Path,
    reference_metrics: dict[str, float],
) -> str:
    best_loss, best_accuracy, best_step = summarize_history(metrics)
    rebase_desc, value_selected, refreshed_layers = summarize_rebase(metrics)
    final_accuracy = float(metrics.get("accuracy", 0.0))
    lora_delta = final_accuracy - reference_metrics["lora_accuracy"]
    lora_ga_delta = final_accuracy - reference_metrics["lora_ga_accuracy"]
    return f"""# 实验说明

## 一、实验目的

本实验用于验证固定配置 `step{config.rebase.interval_steps} + gradient_mix={config.rebase.gradient_mix} + {config.rebase.selection_strategy}` 的 GADI-R 在不同随机种子下是否稳定，并观察它相对当前 LoRA / LoRA-GA 参考基线是否仍有优势。

## 二、实验配置

- 实验方法：GADI-R
- 随机种子：{config.seed}
- 模型：{config.model.model_name_or_path}
- 数据集：{config.data.dataset_name}/{config.data.dataset_config_name}
- LoRA Rank：{config.lora.rank}
- LoRA Alpha：{config.lora.alpha}
- 训练轮数：{config.training.num_epochs}
- 训练批大小：{config.training.train_batch_size}
- 重基化步数：{config.rebase.interval_steps}
- 选择策略：{config.rebase.selection_strategy}
- topk_layers：{config.rebase.topk_layers}
- gradient_mix：{config.rebase.gradient_mix}
- 校准 batch 数：{config.rebase.calibration_batches}

## 三、实验结果

- 最终验证集 Loss：{float(metrics.get("loss", 0.0)):.6f}
- 最终验证集 Accuracy：{final_accuracy:.6f}
- 训练过程最好验证集 Loss：{best_loss:.6f}
- 训练过程最好验证集 Accuracy：{best_accuracy:.6f}
- 最好结果出现步数：{best_step}
- 相对 LoRA 参考基线 Accuracy 差值：{lora_delta:+.6f}
- 相对 LoRA-GA 参考基线 Accuracy 差值：{lora_ga_delta:+.6f}
- 总耗时（秒）：{elapsed_seconds:.2f}

## 四、重基化记录

{rebase_desc}

是否有 value 层进入 top-k：{"是" if value_selected else "否"}

本次被刷新的层：{"；".join(refreshed_layers) if refreshed_layers else "无"}

## 五、结果文件说明

- 当前实验目录：`{run_dir}`
- 配置快照：`配置快照.yaml`
- 指标结果：`结果.json`
- 模型权重目录：`模型权重/`
"""


def _mean(values: list[float]) -> float:
    return statistics.mean(values) if values else 0.0


def _std(values: list[float]) -> float:
    return statistics.stdev(values) if len(values) > 1 else 0.0


def build_summary_markdown(
    summary_rows: list[dict[str, Any]],
    suite_dir: Path,
    seeds: list[int],
    reference_rows: dict[str, dict[str, Any]],
    config: ExperimentConfig,
) -> str:
    accuracies = [row["final_accuracy"] for row in summary_rows]
    losses = [row["final_loss"] for row in summary_rows]
    lora_acc = reference_rows["lora"]["final_accuracy"]
    lora_ga_acc = reference_rows["lora_ga"]["final_accuracy"]

    beat_lora = [row for row in summary_rows if row["final_accuracy"] > lora_acc]
    beat_lora_ga = [row for row in summary_rows if row["final_accuracy"] > lora_ga_acc]
    best_row = max(summary_rows, key=lambda item: (item["final_accuracy"], -item["final_loss"]))
    worst_row = min(summary_rows, key=lambda item: (item["final_accuracy"], -item["final_loss"]))

    lines = [
        "# GADI-R 多随机种子验证总结",
        "",
        "## 一、实验设置",
        "",
        f"- 固定配置：`step{config.rebase.interval_steps} + gradient_mix={config.rebase.gradient_mix} + {config.rebase.selection_strategy} + topk_layers={config.rebase.topk_layers}`。",
        "- 模型：`roberta-base`",
        "- 数据集：`GLUE/MRPC`",
        f"- 随机种子列表：`{', '.join(str(seed) for seed in seeds)}`",
        "- 已存在的相同配置结果会直接复用，不重复训练。",
        "",
        "## 二、参考基线",
        "",
        f"- LoRA 参考基线最终 Accuracy：`{lora_acc:.6f}`，路径：`{reference_rows['lora']['path']}`",
        f"- LoRA-GA 参考基线最终 Accuracy：`{lora_ga_acc:.6f}`，路径：`{reference_rows['lora_ga']['path']}`",
        "- 说明：当前参考基线来自已完成的 `seed=42` 实验，因此这里的结论主要用于判断 GADI 配置本身的稳定性和超越潜力。",
        "",
        "## 三、逐种子结果",
        "",
        "| seed | 是否复用 | 最终 Accuracy | 最终 Loss | 最佳 Accuracy | 最佳步数 | 是否超过 LoRA | 是否超过 LoRA-GA | value 是否入选 | 刷新层 |",
        "| ---: | --- | ---: | ---: | ---: | ---: | --- | --- | --- | --- |",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['seed']} | {'是' if row['reused'] else '否'} | {row['final_accuracy']:.6f} | {row['final_loss']:.6f} | {row['best_accuracy']:.6f} | {row['best_step']} | {'是' if row['final_accuracy'] > lora_acc else '否'} | {'是' if row['final_accuracy'] > lora_ga_acc else '否'} | {'是' if row['value_selected'] else '否'} | {row['refreshed_layers_text']} |"
        )

    lines.extend(
        [
            "",
            "## 四、统计结果",
            "",
            f"- 最终 Accuracy 均值：`{_mean(accuracies):.6f}`",
            f"- 最终 Accuracy 标准差：`{_std(accuracies):.6f}`",
            f"- 最终 Accuracy 最小值：`{min(accuracies):.6f}`",
            f"- 最终 Accuracy 最大值：`{max(accuracies):.6f}`",
            f"- 最终 Loss 均值：`{_mean(losses):.6f}`",
            f"- 最终 Loss 标准差：`{_std(losses):.6f}`",
            f"- 超过 LoRA 参考基线的种子数：`{len(beat_lora)}/{len(summary_rows)}`",
            f"- 超过 LoRA-GA 参考基线的种子数：`{len(beat_lora_ga)}/{len(summary_rows)}`",
            f"- 表现最好的种子：`{best_row['seed']}`，最终 Accuracy `{best_row['final_accuracy']:.6f}`",
            f"- 表现最差的种子：`{worst_row['seed']}`，最终 Accuracy `{worst_row['final_accuracy']:.6f}`",
            "",
            "## 五、结论",
            "",
        ]
    )

    mean_accuracy = _mean(accuracies)
    if mean_accuracy > lora_ga_acc and len(beat_lora_ga) >= max(1, len(summary_rows) // 2 + 1):
        lines.append("- 当前固定配置在多随机种子下表现出较稳定的超越 LoRA-GA 的趋势，具有较强的继续推进价值。")
    elif len(beat_lora_ga) >= 1:
        lines.append("- 当前固定配置已经在部分随机种子上超过 LoRA-GA，但多种子均值或多数种子优势还不够稳定，说明它有潜力但仍需继续打磨。")
    else:
        lines.append("- 当前固定配置尚未在多随机种子下体现出稳定超过 LoRA-GA 的能力，后续仍需继续优化。")

    if len(beat_lora) == len(summary_rows):
        lines.append("- 该配置对标准 LoRA 参考基线已经表现出明显而稳定的优势。")
    else:
        lines.append("- 该配置对标准 LoRA 的优势仍需结合更多种子继续确认。")

    lines.extend(
        [
            "",
            "## 六、目录说明",
            "",
            f"- 当前实验根目录：`{suite_dir}`",
            "- 每个子目录都包含中文实验说明、配置快照、结果 JSON 和模型权重；复用结果在总表中保留原路径。",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run multi-seed validation for the best fixed GADI-R configuration.")
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "configs" / "experiment" / "roberta_gadi_r.yaml"),
        help="Path to the base experiment yaml file.",
    )
    parser.add_argument("--results-root", default=str(PROJECT_ROOT / "results"))
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--train-batch-size", type=int, default=16)
    parser.add_argument("--seeds", default="11,42,123,3407,2026")
    parser.add_argument("--lora-accuracy", type=float, default=None)
    parser.add_argument("--lora-ga-accuracy", type=float, default=None)
    parser.add_argument("--lora-path", default="manual_reference")
    parser.add_argument("--lora-ga-path", default="manual_reference")
    parser.add_argument(
        "--experiment-type",
        default="roberta-base_MRPC_GADI-R多随机种子验证",
        help="Directory label used under results/<date>/",
    )
    args = parser.parse_args()

    seeds = [int(item.strip()) for item in args.seeds.split(",") if item.strip()]
    results_root = Path(args.results_root)
    suite_dir = results_root / datetime.now().strftime("%Y-%m-%d") / args.experiment_type
    suite_dir.mkdir(parents=True, exist_ok=True)
    LOGGER.info(
        "Starting multi-seed validation | config=%s | seeds=%s | experiment_type=%s",
        args.config,
        seeds,
        args.experiment_type,
    )

    base_config = load_experiment_config(args.config)

    lora_ref = _find_existing_result(
        results_root,
        predicate=lambda payload: (
            payload.get("method") == "LoRA"
            and payload.get("config", {}).get("model", {}).get("model_name_or_path") == "roberta-base"
            and payload.get("config", {}).get("data", {}).get("dataset_name") == "glue"
            and payload.get("config", {}).get("data", {}).get("dataset_config_name") == "mrpc"
            and _get_training_cfg(payload).get("num_epochs") == args.epochs
            and _get_training_cfg(payload).get("train_batch_size") == args.train_batch_size
            and _get_training_cfg(payload).get("max_steps") is None
        ),
    )
    lora_ga_ref = _find_existing_result(
        results_root,
        predicate=lambda payload: (
            payload.get("method") == "LoRA-GA"
            and payload.get("config", {}).get("model", {}).get("model_name_or_path") == "roberta-base"
            and payload.get("config", {}).get("data", {}).get("dataset_name") == "glue"
            and payload.get("config", {}).get("data", {}).get("dataset_config_name") == "mrpc"
            and _get_training_cfg(payload).get("num_epochs") == args.epochs
            and _get_training_cfg(payload).get("train_batch_size") == args.train_batch_size
            and _get_training_cfg(payload).get("max_steps") is None
        ),
    )
    reference_rows: dict[str, dict[str, Any]] = {}
    if lora_ref is not None:
        lora_path, lora_payload = lora_ref
        reference_rows["lora"] = {
            "final_accuracy": float(lora_payload.get("metrics", {}).get("accuracy", 0.0)),
            "path": str(lora_path.parent),
        }
    elif args.lora_accuracy is not None:
        reference_rows["lora"] = {
            "final_accuracy": float(args.lora_accuracy),
            "path": args.lora_path,
        }

    if lora_ga_ref is not None:
        lora_ga_path, lora_ga_payload = lora_ga_ref
        reference_rows["lora_ga"] = {
            "final_accuracy": float(lora_ga_payload.get("metrics", {}).get("accuracy", 0.0)),
            "path": str(lora_ga_path.parent),
        }
    elif args.lora_ga_accuracy is not None:
        reference_rows["lora_ga"] = {
            "final_accuracy": float(args.lora_ga_accuracy),
            "path": args.lora_ga_path,
        }

    if "lora" not in reference_rows or "lora_ga" not in reference_rows:
        raise RuntimeError("未找到 LoRA / LoRA-GA 参考基线结果，也没有通过命令行提供参考 Accuracy。")

    LOGGER.info(
        "Reference baselines | LoRA=%.6f | LoRA-GA=%.6f",
        reference_rows["lora"]["final_accuracy"],
        reference_rows["lora_ga"]["final_accuracy"],
    )

    summary_rows: list[dict[str, Any]] = []

    for seed in seeds:
        existing_run = _find_existing_result(
            results_root,
            predicate=lambda payload, seed=seed: (
                payload.get("method") == "GADI-R"
                and _get_seed(payload) == seed
                and _get_rebase_cfg(payload).get("interval_steps") == base_config.rebase.interval_steps
                and _get_rebase_cfg(payload).get("warmup_steps") == base_config.rebase.warmup_steps
                and _get_rebase_cfg(payload).get("topk_layers") == base_config.rebase.topk_layers
                and _get_rebase_cfg(payload).get("selection_strategy", "global_topk") == base_config.rebase.selection_strategy
                and _get_gradient_mix(payload) == base_config.rebase.gradient_mix
                and _get_training_cfg(payload).get("num_epochs") == args.epochs
                and _get_training_cfg(payload).get("train_batch_size") == args.train_batch_size
            ),
        )

        if existing_run is not None:
            run_json_path, payload = existing_run
            LOGGER.info("Reusing existing GADI-R seed=%s result: %s", seed, run_json_path.parent)
            metrics = payload.get("metrics", {})
            elapsed_seconds = float(payload.get("elapsed_seconds", 0.0))
            reused = True
            run_dir = run_json_path.parent
        else:
            LOGGER.info("Running GADI-R fixed-config seed=%s", seed)
            config = build_seed_config(
                base_config,
                seed=seed,
                num_epochs=args.epochs,
                train_batch_size=args.train_batch_size,
            )
            run_dir = make_run_directory(
                results_root=results_root,
                experiment_type=args.experiment_type,
                method_name=f"gadi_seed_{seed}",
                timestamp=datetime.now(),
            )
            config.training.output_dir = str(run_dir / "模型权重")

            write_config_snapshot(config, run_dir / "配置快照.yaml")
            start = time.perf_counter()
            metrics = run_training(config)
            elapsed_seconds = time.perf_counter() - start

            payload = {
                "method": "GADI-R",
                "seed": seed,
                "metrics": metrics,
                "elapsed_seconds": elapsed_seconds,
                "config": asdict(config),
            }
            write_metrics_json(payload, run_dir / "结果.json")
            write_markdown(
                run_dir / "实验说明.md",
                build_experiment_markdown(config, metrics, elapsed_seconds, run_dir, {
                    "lora_accuracy": reference_rows["lora"]["final_accuracy"],
                    "lora_ga_accuracy": reference_rows["lora_ga"]["final_accuracy"],
                }),
            )
            reused = False

        best_loss, best_accuracy, best_step = summarize_history(metrics)
        _, value_selected, refreshed_layers = summarize_rebase(metrics)
        summary_rows.append(
            {
                "seed": seed,
                "reused": reused,
                "final_loss": float(metrics.get("loss", 0.0)),
                "final_accuracy": float(metrics.get("accuracy", 0.0)),
                "best_loss": best_loss,
                "best_accuracy": best_accuracy,
                "best_step": best_step,
                "value_selected": value_selected,
                "elapsed_seconds": elapsed_seconds,
                "refreshed_layers_text": "；".join(refreshed_layers) if refreshed_layers else "无",
                "path": str(run_dir),
            }
        )
        LOGGER.info(
            "Finished seed=%s | reused=%s | final_accuracy=%.6f | final_loss=%.6f",
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
                "value_selected",
                "elapsed_seconds",
                "refreshed_layers_text",
                "path",
            ],
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    write_markdown(
        suite_dir / "对比总结.md",
        build_summary_markdown(summary_rows, suite_dir, seeds, reference_rows, base_config),
    )
    LOGGER.info("Finished GADI-R multi-seed validation. Summary directory: %s", suite_dir)


if __name__ == "__main__":
    main()
