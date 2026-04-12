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

LOGGER = get_logger("gadir_mix_ablation")


def build_gadir_ablation_config(
    config: ExperimentConfig,
    gradient_mix: float,
    num_epochs: int,
    train_batch_size: int,
) -> ExperimentConfig:
    cfg = copy.deepcopy(config)
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
    cfg.rebase.interval_steps = 120
    cfg.rebase.warmup_steps = 120
    cfg.rebase.max_rebases = 1
    cfg.rebase.drift_threshold = 0.03
    cfg.rebase.topk_layers = 2
    cfg.rebase.calibration_batches = 4
    cfg.rebase.use_residual_gradient = True
    cfg.rebase.gradient_mix = gradient_mix
    cfg.rebase.calibration_eval_mode = True
    cfg.rebase.reset_optimizer_state = False
    return cfg


def summarize_history(metrics: dict) -> tuple[float, float, int]:
    eval_history = metrics.get("eval_history", [])
    if not eval_history:
        return float(metrics.get("loss", 0.0)), float(metrics.get("accuracy", 0.0)), int(metrics.get("total_steps", 0))
    best_item = max(eval_history, key=lambda item: (item["accuracy"], -item["loss"]))
    return float(best_item["loss"]), float(best_item["accuracy"]), int(best_item["step"])


def describe_rebase_history(metrics: dict) -> tuple[str, bool]:
    method_artifacts = metrics.get("method_artifacts", {})
    history = method_artifacts.get("rebase_history", [])
    if not history:
        return "未发生重基化。", False

    lines: list[str] = []
    value_selected = False
    for item in history:
        refreshed_layers = item.get("refreshed_layers", [])
        if any("value" in layer for layer in refreshed_layers):
            value_selected = True
        lines.append(f"- step {item.get('step')}: {', '.join(refreshed_layers) if refreshed_layers else '无层被刷新'}")
    return "\n".join(lines), value_selected


def build_experiment_markdown(
    config: ExperimentConfig,
    metrics: dict,
    elapsed_seconds: float,
    run_dir: Path,
) -> str:
    best_loss, best_accuracy, best_step = summarize_history(metrics)
    rebase_desc, value_selected = describe_rebase_history(metrics)
    return f"""# 实验说明

## 一、实验目的

本实验用于评估修正版 GADI-R 在 `gradient_mix={config.rebase.gradient_mix}` 下的效果，并观察把单次重基化提前到 `step 120` 后，性能是否继续改善。

## 二、实验配置

- 实验方法：GADI-R
- 模型：{config.model.model_name_or_path}
- 数据集：{config.data.dataset_name}/{config.data.dataset_config_name}
- LoRA Rank：{config.lora.rank}
- LoRA Alpha：{config.lora.alpha}
- 训练轮数：{config.training.num_epochs}
- 训练批大小：{config.training.train_batch_size}
- 单次重基化步数：{config.rebase.interval_steps}
- Warmup 步数：{config.rebase.warmup_steps}
- 最大重基化次数：{config.rebase.max_rebases}
- topk_layers：{config.rebase.topk_layers}
- gradient_mix：{config.rebase.gradient_mix}
- 校准 batch 数：{config.rebase.calibration_batches}
- query/value 是否共同参与竞争：是

## 三、实验结果

- 最终验证集 Loss：{float(metrics.get("loss", 0.0)):.6f}
- 最终验证集 Accuracy：{float(metrics.get("accuracy", 0.0)):.6f}
- 训练过程最好验证集 Loss：{best_loss:.6f}
- 训练过程最好验证集 Accuracy：{best_accuracy:.6f}
- 最好结果出现步数：{best_step}
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


def build_summary_markdown(summary_rows: list[dict], suite_dir: Path) -> str:
    lines = [
        "# GADI-R Gradient Mix 消融总结",
        "",
        "## 一、实验设置",
        "",
        "- 只测试修正版 GADI-R，不重复运行 LoRA 与 LoRA-GA。",
        "- 模型：`roberta-base`",
        "- 数据集：`GLUE/MRPC`",
        "- 单次重基化固定在 `step 120`",
        "- `topk_layers=2`",
        "- `query/value` 共同参与候选层竞争",
        "- `gradient_mix` 消融：`0.5 / 0.7 / 0.9`",
        "",
        "## 二、结果汇总",
        "",
        "| gradient_mix | 最终 Loss | 最终 Accuracy | 最佳 Loss | 最佳 Accuracy | 最佳步数 | value 是否入选 | 耗时（秒） |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['gradient_mix']:.1f} | {row['final_loss']:.6f} | {row['final_accuracy']:.6f} | {row['best_loss']:.6f} | {row['best_accuracy']:.6f} | {row['best_step']} | {'是' if row['value_selected'] else '否'} | {row['elapsed_seconds']:.2f} |"
        )

    best_row = max(summary_rows, key=lambda item: (item["final_accuracy"], -item["final_loss"]))
    lines.extend(
        [
            "",
            "## 三、结论",
            "",
            f"- 本轮消融中最终表现最好的 `gradient_mix` 是 `{best_row['gradient_mix']:.1f}`，最终 Accuracy 为 `{best_row['final_accuracy']:.6f}`。",
            "- 如果所有变体都仍然只选中 query 层，说明当前 drift score 对 value 层不敏感，后面需要考虑分模块归一化或强制模块均衡。",
            "",
            "## 四、目录说明",
            "",
            f"- 当前实验根目录：`{suite_dir}`",
            "- 每个子目录都包含中文实验说明、配置快照、结果 JSON 和模型权重。",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run GADI-R gradient_mix ablations only.")
    parser.add_argument("--results-root", default=str(PROJECT_ROOT / "results"))
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--train-batch-size", type=int, default=16)
    parser.add_argument(
        "--experiment-type",
        default="roberta-base_MRPC_GADI-R_gradient_mix消融实验",
        help="Directory label used under results/<date>/",
    )
    args = parser.parse_args()

    suite_timestamp = datetime.now()
    suite_dir = Path(args.results_root) / suite_timestamp.strftime("%Y-%m-%d") / args.experiment_type
    suite_dir.mkdir(parents=True, exist_ok=True)

    base_config = load_experiment_config(PROJECT_ROOT / "configs" / "experiment" / "roberta_gadi_r.yaml")
    summary_rows: list[dict] = []

    for gradient_mix in (0.5, 0.7, 0.9):
        LOGGER.info("Running GADI-R ablation with gradient_mix=%s", gradient_mix)
        config = build_gadir_ablation_config(
            base_config,
            gradient_mix=gradient_mix,
            num_epochs=args.epochs,
            train_batch_size=args.train_batch_size,
        )

        run_dir = make_run_directory(
            results_root=args.results_root,
            experiment_type=args.experiment_type,
            method_name=f"gadi_r_mix_{str(gradient_mix).replace('.', '')}",
            timestamp=datetime.now(),
        )
        config.training.output_dir = str(run_dir / "模型权重")

        write_config_snapshot(config, run_dir / "配置快照.yaml")
        start_time = time.perf_counter()
        metrics = run_training(config)
        elapsed_seconds = time.perf_counter() - start_time

        result_payload = {
            "method": "GADI-R",
            "gradient_mix": gradient_mix,
            "metrics": metrics,
            "elapsed_seconds": elapsed_seconds,
            "config": asdict(config),
        }
        write_metrics_json(result_payload, run_dir / "结果.json")
        write_markdown(run_dir / "实验说明.md", build_experiment_markdown(config, metrics, elapsed_seconds, run_dir))

        best_loss, best_accuracy, best_step = summarize_history(metrics)
        _, value_selected = describe_rebase_history(metrics)
        summary_rows.append(
            {
                "gradient_mix": gradient_mix,
                "final_loss": float(metrics.get("loss", 0.0)),
                "final_accuracy": float(metrics.get("accuracy", 0.0)),
                "best_loss": best_loss,
                "best_accuracy": best_accuracy,
                "best_step": best_step,
                "value_selected": value_selected,
                "elapsed_seconds": elapsed_seconds,
                "run_dir": str(run_dir),
            }
        )

    with (suite_dir / "结果汇总.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "gradient_mix",
                "final_loss",
                "final_accuracy",
                "best_loss",
                "best_accuracy",
                "best_step",
                "value_selected",
                "elapsed_seconds",
                "run_dir",
            ],
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    write_markdown(suite_dir / "对比总结.md", build_summary_markdown(summary_rows, suite_dir))
    LOGGER.info("Finished GADI-R ablation suite. Summary directory: %s", suite_dir)


if __name__ == "__main__":
    main()
