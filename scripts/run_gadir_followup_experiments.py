from __future__ import annotations

import argparse
import copy
import csv
import json
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

LOGGER = get_logger("gadir_followup")


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _get_rebase_cfg(payload: dict[str, Any]) -> dict[str, Any]:
    return payload.get("config", {}).get("rebase", {})


def _get_training_cfg(payload: dict[str, Any]) -> dict[str, Any]:
    return payload.get("config", {}).get("training", {})


def _get_gradient_mix(payload: dict[str, Any]) -> float | None:
    rebase_cfg = _get_rebase_cfg(payload)
    value = rebase_cfg.get("gradient_mix", payload.get("gradient_mix"))
    return None if value is None else float(value)


def _summarize_result_payload(payload: dict[str, Any], source_label: str, reused: bool) -> dict[str, Any]:
    metrics = payload.get("metrics", {})
    method_artifacts = metrics.get("method_artifacts", {})
    rebase_history = method_artifacts.get("rebase_history", [])
    refreshed_layers: list[str] = []
    for event in rebase_history:
        refreshed_layers.extend(event.get("refreshed_layers", []))
    value_selected = any("value" in layer for layer in refreshed_layers)

    eval_history = metrics.get("eval_history", [])
    if eval_history:
        best_item = max(eval_history, key=lambda item: (item["accuracy"], -item["loss"]))
        best_loss = float(best_item["loss"])
        best_accuracy = float(best_item["accuracy"])
        best_step = int(best_item["step"])
    else:
        best_loss = float(metrics.get("loss", 0.0))
        best_accuracy = float(metrics.get("accuracy", 0.0))
        best_step = int(metrics.get("total_steps", 0))

    config = payload.get("config", {})
    rebase_cfg = config.get("rebase", {})
    return {
        "source": source_label,
        "reused": reused,
        "selection_strategy": rebase_cfg.get("selection_strategy", "global_topk"),
        "rebase_step": rebase_cfg.get("interval_steps", ""),
        "gradient_mix": rebase_cfg.get("gradient_mix", payload.get("gradient_mix", "")),
        "final_loss": float(metrics.get("loss", 0.0)),
        "final_accuracy": float(metrics.get("accuracy", 0.0)),
        "best_loss": best_loss,
        "best_accuracy": best_accuracy,
        "best_step": best_step,
        "value_selected": value_selected,
        "elapsed_seconds": float(payload.get("elapsed_seconds", 0.0)),
        "refreshed_layers": refreshed_layers,
        "refreshed_layers_text": "；".join(refreshed_layers) if refreshed_layers else "无",
    }


def _find_existing_result(
    results_root: Path,
    *,
    experiment_name_contains: str,
    predicate,
) -> tuple[Path, dict[str, Any]] | None:
    matches: list[tuple[Path, dict[str, Any]]] = []
    for json_path in results_root.rglob("结果.json"):
        if experiment_name_contains not in str(json_path):
            continue
        payload = _read_json(json_path)
        if predicate(payload):
            matches.append((json_path, payload))
    if not matches:
        return None
    matches.sort(key=lambda item: item[0].stat().st_mtime, reverse=True)
    return matches[0]


def build_gadir_config(
    base_config: ExperimentConfig,
    *,
    selection_strategy: str,
    rebase_step: int,
    gradient_mix: float,
    num_epochs: int,
    train_batch_size: int,
) -> ExperimentConfig:
    cfg = copy.deepcopy(base_config)
    cfg.seed = 42
    cfg.method = "gadi_r"
    cfg.training.num_epochs = num_epochs
    cfg.training.max_steps = None
    cfg.training.train_batch_size = train_batch_size
    cfg.training.eval_batch_size = 32
    cfg.training.gradient_accumulation_steps = 1
    cfg.training.log_every_steps = 20
    cfg.training.eval_every_steps = 40

    cfg.data.calibration_size = 32
    cfg.lora_ga.calibration_batches = 1

    cfg.rebase.enabled = True
    cfg.rebase.interval_steps = rebase_step
    cfg.rebase.warmup_steps = rebase_step
    cfg.rebase.max_rebases = 1
    cfg.rebase.drift_threshold = 0.03
    cfg.rebase.topk_layers = 2
    cfg.rebase.selection_strategy = selection_strategy
    cfg.rebase.calibration_batches = 4
    cfg.rebase.use_residual_gradient = True
    cfg.rebase.gradient_mix = gradient_mix
    cfg.rebase.calibration_eval_mode = True
    cfg.rebase.reset_optimizer_state = False
    return cfg


def build_experiment_markdown(
    config: ExperimentConfig,
    metrics: dict[str, Any],
    elapsed_seconds: float,
    run_dir: Path,
) -> str:
    eval_history = metrics.get("eval_history", [])
    if eval_history:
        best_item = max(eval_history, key=lambda item: (item["accuracy"], -item["loss"]))
    else:
        best_item = {
            "loss": float(metrics.get("loss", 0.0)),
            "accuracy": float(metrics.get("accuracy", 0.0)),
            "step": int(metrics.get("total_steps", 0)),
        }
    rebase_history = metrics.get("method_artifacts", {}).get("rebase_history", [])
    if rebase_history:
        rebase_desc = "\n".join(
            f"- step {item.get('step')}: {', '.join(item.get('refreshed_layers', [])) or '无层被刷新'}"
            for item in rebase_history
        )
    else:
        rebase_desc = "未发生重基化。"
    value_selected = any(
        "value" in layer
        for item in rebase_history
        for layer in item.get("refreshed_layers", [])
    )
    return f"""# 实验说明

## 一、实验目的

本实验用于评估修正版 GADI-R 在新的层选择策略或新的重基化步数下的表现，并与已完成实验对比，筛选最有希望超过基线的 GADI 配置。

## 二、实验配置

- 实验方法：GADI-R
- 模型：{config.model.model_name_or_path}
- 数据集：{config.data.dataset_name}/{config.data.dataset_config_name}
- LoRA Rank：{config.lora.rank}
- LoRA Alpha：{config.lora.alpha}
- 训练轮数：{config.training.num_epochs}
- 训练批大小：{config.training.train_batch_size}
- 单次重基化步数：{config.rebase.interval_steps}
- 层选择策略：{config.rebase.selection_strategy}
- topk_layers：{config.rebase.topk_layers}
- gradient_mix：{config.rebase.gradient_mix}
- 校准 batch 数：{config.rebase.calibration_batches}

## 三、实验结果

- 最终验证集 Loss：{float(metrics.get("loss", 0.0)):.6f}
- 最终验证集 Accuracy：{float(metrics.get("accuracy", 0.0)):.6f}
- 最佳验证集 Loss：{float(best_item['loss']):.6f}
- 最佳验证集 Accuracy：{float(best_item['accuracy']):.6f}
- 最佳结果出现步数：{int(best_item['step'])}
- 总耗时（秒）：{elapsed_seconds:.2f}

## 四、重基化记录

{rebase_desc}

是否有 value 层进入 top-k：{"是" if value_selected else "否"}

## 五、结果文件说明

- 当前实验目录：`{run_dir}`
- 配置快照：`配置快照.yaml`
- 指标结果：`结果.json`
- 模型权重目录：`模型权重/`
"""


def build_summary_markdown(rows: list[dict[str, Any]], suite_dir: Path) -> str:
    lines = [
        "# GADI-R 后续实验总结",
        "",
        "## 一、实验说明",
        "",
        "- 本轮包含两组后续实验：层选择机制消融、重基化步数消融。",
        "- 已经做过的相同设置不重复运行，而是直接复用已有结果。",
        "- 同时纳入已有 `LoRA-GA` 结果，作为外部强基线对照。",
        "",
        "## 二、结果汇总",
        "",
        "| 来源 | 是否复用 | 选择策略 | 重基化步数 | gradient_mix | 最终 Accuracy | 最终 Loss | value 是否入选 |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row['source']} | {'是' if row['reused'] else '否'} | {row['selection_strategy']} | {row['rebase_step']} | {row['gradient_mix']} | {row['final_accuracy']:.6f} | {row['final_loss']:.6f} | {'是' if row['value_selected'] else '否'} |"
        )

    gadi_rows = [row for row in rows if row["source"].startswith("GADI")]
    best_gadi = max(gadi_rows, key=lambda item: (item["final_accuracy"], -item["final_loss"]))
    followup_rows = [
        row
        for row in rows
        if row["source"].startswith("GADI层选择") or row["source"].startswith("GADI步数")
    ]
    best_followup = None
    if followup_rows:
        best_followup = max(followup_rows, key=lambda item: (item["final_accuracy"], -item["final_loss"]))
    fresh_gadi_rows = [row for row in gadi_rows if not row["reused"]]
    best_fresh_gadi = None
    if fresh_gadi_rows:
        best_fresh_gadi = max(fresh_gadi_rows, key=lambda item: (item["final_accuracy"], -item["final_loss"]))
    baseline_row = next((row for row in rows if row["source"] == "LoRA-GA基线"), None)

    lines.extend(
        [
            "",
            "## 三、结论",
            "",
            f"- 当前最优的 GADI 配置是 `{best_gadi['source']}`，最终 Accuracy 为 `{best_gadi['final_accuracy']:.6f}`。",
        ]
    )
    if best_followup is not None:
        lines.append(
            f"- 本轮两组后续实验中表现最好的新设置是 `{best_followup['source']}`，最终 Accuracy 为 `{best_followup['final_accuracy']:.6f}`，最终 Loss 为 `{best_followup['final_loss']:.6f}`。"
        )
    if best_fresh_gadi is not None:
        lines.append(
            f"- 本轮新运行设置中最好的配置是 `{best_fresh_gadi['source']}`，最终 Accuracy 为 `{best_fresh_gadi['final_accuracy']:.6f}`。"
        )
    if baseline_row is not None:
        delta = best_gadi["final_accuracy"] - baseline_row["final_accuracy"]
        lines.append(
            f"- 相比 LoRA-GA 基线（Accuracy `{baseline_row['final_accuracy']:.6f}`），该配置的 Accuracy 差值为 `{delta:+.6f}`。"
        )
    lines.extend(
        [
            "- 如果仍然看不到 value 层入选，后续应优先改层打分机制，而不是继续盲目调 gradient_mix。",
            "",
            "## 四、目录说明",
            "",
            f"- 当前实验根目录：`{suite_dir}`",
            "- 新运行的实验目录包含完整中文实验说明；复用结果仅在本汇总中引用其现有路径和指标。",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run follow-up GADI-R experiments without rerunning identical settings.")
    parser.add_argument("--results-root", default=str(PROJECT_ROOT / "results"))
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--train-batch-size", type=int, default=16)
    parser.add_argument(
        "--experiment-type",
        default="roberta-base_MRPC_GADI-R后续实验",
        help="Directory label used under results/<date>/",
    )
    args = parser.parse_args()

    results_root = Path(args.results_root)
    suite_timestamp = datetime.now()
    suite_dir = results_root / suite_timestamp.strftime("%Y-%m-%d") / args.experiment_type
    suite_dir.mkdir(parents=True, exist_ok=True)

    base_config = load_experiment_config(PROJECT_ROOT / "configs" / "experiment" / "roberta_gadi_r.yaml")
    summary_rows: list[dict[str, Any]] = []

    existing_gadi = _find_existing_result(
        results_root,
        experiment_name_contains="roberta-base_MRPC_GADI-R_gradient_mix消融实验",
        predicate=lambda payload: (
            payload.get("method") == "GADI-R"
            and _get_rebase_cfg(payload).get("interval_steps") == 120
            and _get_rebase_cfg(payload).get("selection_strategy", "global_topk") == "global_topk"
            and _get_gradient_mix(payload) == 0.5
        ),
    )
    if existing_gadi is not None:
        path, payload = existing_gadi
        row = _summarize_result_payload(payload, "GADI现有最佳(step120,mix0.5,global)", reused=True)
        row["path"] = str(path.parent)
        summary_rows.append(row)

    existing_lora_ga = _find_existing_result(
        results_root,
        experiment_name_contains="roberta-base_MRPC_GADI-R修正版完整对比实验",
        predicate=lambda payload: payload.get("method") == "LoRA-GA",
    )
    if existing_lora_ga is not None:
        path, payload = existing_lora_ga
        row = _summarize_result_payload(payload, "LoRA-GA基线", reused=True)
        row["path"] = str(path.parent)
        summary_rows.append(row)

    experiment_specs = [
        {
            "label": "GADI层选择_module_normalized",
            "selection_strategy": "module_normalized",
            "rebase_step": 120,
            "gradient_mix": 0.5,
        },
        {
            "label": "GADI层选择_balanced_qv",
            "selection_strategy": "balanced_qv",
            "rebase_step": 120,
            "gradient_mix": 0.5,
        },
        {
            "label": "GADI步数_step100",
            "selection_strategy": "global_topk",
            "rebase_step": 100,
            "gradient_mix": 0.5,
        },
        {
            "label": "GADI步数_step140",
            "selection_strategy": "global_topk",
            "rebase_step": 140,
            "gradient_mix": 0.5,
        },
    ]

    for spec in experiment_specs:
        existing_spec = _find_existing_result(
            results_root,
            experiment_name_contains=args.experiment_type,
            predicate=lambda payload, spec=spec: (
                payload.get("method") == "GADI-R"
                and _get_rebase_cfg(payload).get("interval_steps") == spec["rebase_step"]
                and _get_rebase_cfg(payload).get("selection_strategy", "global_topk") == spec["selection_strategy"]
                and _get_gradient_mix(payload) == spec["gradient_mix"]
                and _get_training_cfg(payload).get("num_epochs") == args.epochs
                and _get_training_cfg(payload).get("train_batch_size") == args.train_batch_size
            ),
        )
        if existing_spec is not None:
            path, payload = existing_spec
            LOGGER.info("Reusing existing follow-up experiment: %s", spec["label"])
            row = _summarize_result_payload(payload, spec["label"], reused=True)
            row["path"] = str(path.parent)
            summary_rows.append(row)
            continue

        LOGGER.info("Running follow-up experiment: %s", spec["label"])
        config = build_gadir_config(
            base_config,
            selection_strategy=spec["selection_strategy"],
            rebase_step=spec["rebase_step"],
            gradient_mix=spec["gradient_mix"],
            num_epochs=args.epochs,
            train_batch_size=args.train_batch_size,
        )
        run_dir = make_run_directory(
            results_root=results_root,
            experiment_type=args.experiment_type,
            method_name=spec["label"].replace("GADI", "gadi").replace("层选择", "selection").replace("步数", "step"),
            timestamp=datetime.now(),
        )
        config.training.output_dir = str(run_dir / "模型权重")

        write_config_snapshot(config, run_dir / "配置快照.yaml")
        start = time.perf_counter()
        metrics = run_training(config)
        elapsed_seconds = time.perf_counter() - start

        payload = {
            "method": "GADI-R",
            "label": spec["label"],
            "metrics": metrics,
            "elapsed_seconds": elapsed_seconds,
            "config": asdict(config),
        }
        write_metrics_json(payload, run_dir / "结果.json")
        write_markdown(run_dir / "实验说明.md", build_experiment_markdown(config, metrics, elapsed_seconds, run_dir))

        row = _summarize_result_payload(payload, spec["label"], reused=False)
        row["path"] = str(run_dir)
        summary_rows.append(row)

    with (suite_dir / "结果汇总.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "source",
                "reused",
                "selection_strategy",
                "rebase_step",
                "gradient_mix",
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
        csv_rows = [{field: row.get(field, "") for field in writer.fieldnames} for row in summary_rows]
        writer.writerows(csv_rows)

    write_markdown(suite_dir / "对比总结.md", build_summary_markdown(summary_rows, suite_dir))
    LOGGER.info("Finished follow-up suite. Summary directory: %s", suite_dir)


if __name__ == "__main__":
    main()
