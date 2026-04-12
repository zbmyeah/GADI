from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel
from sklearn.metrics import f1_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from gadir.config import DataConfig, ExperimentConfig, LoraExperimentConfig, LoraGAConfig, ModelConfig, OptimizerConfig, RebaseConfig, TrainingConfig
from gadir.data.loaders import build_data_bundle
from gadir.utils.logging import get_logger
from gadir.utils.results import write_markdown, write_metrics_json

LOGGER = get_logger("evaluate_mrpc_f1")

DEFAULT_EXPERIMENTS = {
    "LoRA": "roberta-base_MRPC_LoRA多随机种子验证_A10",
    "LoRA-GA": "roberta-base_MRPC_LoRA-GA多随机种子验证_A10",
    "GADI-R": "roberta-base_MRPC_GADI-R多随机种子验证_A10_query_only_top1_step120",
}


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_config_from_payload(payload: dict[str, Any]) -> ExperimentConfig:
    raw = payload["config"]
    return ExperimentConfig(
        seed=raw["seed"],
        method=raw["method"],
        model=ModelConfig(**raw["model"]),
        data=DataConfig(**raw["data"]),
        lora=LoraExperimentConfig(**raw["lora"]),
        lora_ga=LoraGAConfig(**raw["lora_ga"]),
        rebase=RebaseConfig(**raw["rebase"]),
        optimizer=OptimizerConfig(**raw["optimizer"]),
        training=TrainingConfig(**raw["training"]),
    )


def _load_adapter_model(checkpoint_dir: Path, config: ExperimentConfig) -> tuple[torch.nn.Module, Any]:
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir, use_fast=True)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForSequenceClassification.from_pretrained(
        config.model.model_name_or_path,
        num_labels=config.model.num_labels,
    )
    model = PeftModel.from_pretrained(base_model, checkpoint_dir)
    return model, tokenizer


@torch.no_grad()
def _evaluate_with_f1(
    model: torch.nn.Module,
    dataloader,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_examples = 0
    all_predictions: list[int] = []
    all_labels: list[int] = []

    for batch in dataloader:
        batch = {key: value.to(device) for key, value in batch.items()}
        outputs = model(**batch)
        predictions = outputs.logits.argmax(dim=-1)
        labels = batch["labels"]

        total_loss += float(outputs.loss) * labels.size(0)
        total_examples += labels.size(0)
        all_predictions.extend(predictions.detach().cpu().tolist())
        all_labels.extend(labels.detach().cpu().tolist())

    if total_examples == 0:
        return {"loss": 0.0, "accuracy": 0.0, "f1": 0.0}

    correct = sum(int(pred == label) for pred, label in zip(all_predictions, all_labels))
    return {
        "loss": total_loss / total_examples,
        "accuracy": correct / total_examples,
        "f1": float(f1_score(all_labels, all_predictions)),
    }


def _find_latest_summary(results_root: Path, experiment_type: str) -> Path:
    matches = [path for path in results_root.rglob("结果汇总.csv") if path.parent.name == experiment_type]
    if not matches:
        raise FileNotFoundError(f"未找到实验类型 {experiment_type} 对应的结果汇总.csv")
    matches.sort(key=lambda item: item.stat().st_mtime, reverse=True)
    return matches[0]


def _mean(values: list[float]) -> float:
    return statistics.mean(values) if values else 0.0


def _std(values: list[float]) -> float:
    return statistics.stdev(values) if len(values) > 1 else 0.0


def _build_markdown(
    suite_dir: Path,
    summary_rows: list[dict[str, Any]],
    method_stats: list[dict[str, Any]],
) -> str:
    lines = [
        "# MRPC F1 补充评估报告",
        "",
        "## 一、说明",
        "",
        "本报告基于已训练完成的 MRPC 模型权重进行离线重评估，补充 `F1` 指标，不重新训练模型。",
        "",
        "## 二、逐 run 结果",
        "",
        "| 方法 | seed | loss | accuracy | f1 | 路径 |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
    ]

    for row in summary_rows:
        lines.append(
            f"| {row['method']} | {row['seed']} | {row['loss']:.6f} | {row['accuracy']:.6f} | {row['f1']:.6f} | `{row['path']}` |"
        )

    lines.extend(
        [
            "",
            "## 三、方法级统计",
            "",
            "| 方法 | Accuracy mean ± std | F1 mean ± std |",
            "| --- | --- | --- |",
        ]
    )
    for row in method_stats:
        lines.append(
            f"| {row['method']} | `{row['accuracy_mean']:.6f} ± {row['accuracy_std']:.6f}` | `{row['f1_mean']:.6f} ± {row['f1_std']:.6f}` |"
        )

    lines.extend(
        [
            "",
            "## 四、结论",
            "",
            "- 本补充实验用于让 MRPC 结果同时具备 `Accuracy` 与 `F1`，更符合该任务常见汇报方式。",
            f"- 当前结果目录：`{suite_dir}`",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate MRPC checkpoints and supplement F1 metrics.")
    parser.add_argument("--results-root", default=str(PROJECT_ROOT / "results"))
    parser.add_argument(
        "--suite-name",
        default="roberta-base_MRPC_A10_F1补充评估",
        help="Directory label used under results/<date>/",
    )
    args = parser.parse_args()

    results_root = Path(args.results_root)
    date_dir = datetime.now().strftime("%Y-%m-%d")
    suite_dir = results_root / date_dir / args.suite_name
    suite_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    LOGGER.info("Starting MRPC F1 supplement evaluation on device=%s", device)

    summary_rows: list[dict[str, Any]] = []

    for method_name, experiment_type in DEFAULT_EXPERIMENTS.items():
        summary_csv = _find_latest_summary(results_root, experiment_type)
        with summary_csv.open("r", encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))

        for row in rows:
            run_dir = Path(row["path"])
            payload = _read_json(run_dir / "结果.json")
            config = _load_config_from_payload(payload)
            checkpoint_dir = run_dir / "模型权重"

            LOGGER.info("Evaluating %s seed=%s | checkpoint=%s", method_name, config.seed, checkpoint_dir)
            model, tokenizer = _load_adapter_model(checkpoint_dir, config)
            model.to(device)
            data_bundle = build_data_bundle(config, tokenizer)
            metrics = _evaluate_with_f1(model, data_bundle.eval_loader, device)

            record = {
                "method": method_name,
                "seed": config.seed,
                "loss": metrics["loss"],
                "accuracy": metrics["accuracy"],
                "f1": metrics["f1"],
                "path": str(run_dir),
            }
            summary_rows.append(record)

    summary_rows.sort(key=lambda item: (item["method"], item["seed"]))

    method_stats: list[dict[str, Any]] = []
    for method_name in ["LoRA", "LoRA-GA", "GADI-R"]:
        rows = [row for row in summary_rows if row["method"] == method_name]
        accuracies = [float(row["accuracy"]) for row in rows]
        f1s = [float(row["f1"]) for row in rows]
        method_stats.append(
            {
                "method": method_name,
                "accuracy_mean": _mean(accuracies),
                "accuracy_std": _std(accuracies),
                "f1_mean": _mean(f1s),
                "f1_std": _std(f1s),
            }
        )

    with (suite_dir / "MRPC_F1逐run结果.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["method", "seed", "loss", "accuracy", "f1", "path"])
        writer.writeheader()
        writer.writerows(summary_rows)

    with (suite_dir / "MRPC_F1方法统计.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["method", "accuracy_mean", "accuracy_std", "f1_mean", "f1_std"])
        writer.writeheader()
        writer.writerows(method_stats)

    write_metrics_json(
        {
            "summary_rows": summary_rows,
            "method_stats": method_stats,
            "generated_at": datetime.now().isoformat(),
        },
        suite_dir / "MRPC_F1补充结果.json",
    )
    write_markdown(suite_dir / "MRPC_F1补充报告.md", _build_markdown(suite_dir, summary_rows, method_stats))
    LOGGER.info("Finished MRPC F1 supplement evaluation. Summary directory: %s", suite_dir)


if __name__ == "__main__":
    main()
