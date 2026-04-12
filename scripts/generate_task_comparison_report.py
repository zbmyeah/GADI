from __future__ import annotations

import argparse
import csv
import statistics
from datetime import datetime
from pathlib import Path


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


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


def _to_float(row: dict[str, str], key: str) -> float:
    value = row.get(key, "")
    return float(value) if value not in ("", None) else 0.0


def _method_stats(method_name: str, rows: list[dict[str, str]]) -> dict[str, float | str]:
    accuracies = [_to_float(row, "final_accuracy") for row in rows]
    losses = [_to_float(row, "final_loss") for row in rows]
    total_times = [_to_float(row, "total_wall_time_seconds") for row in rows]
    init_times = [_to_float(row, "initialization_time_seconds") for row in rows]
    rebase_times = [_to_float(row, "rebase_overhead_time_seconds") for row in rows]
    peak_memories = [_to_float(row, "peak_memory_allocated_mb") for row in rows]
    return {
        "method": method_name,
        "accuracy_mean": _mean(accuracies),
        "accuracy_std": _std(accuracies),
        "loss_mean": _mean(losses),
        "loss_std": _std(losses),
        "total_time_mean": _mean(total_times),
        "total_time_std": _std(total_times),
        "init_time_mean": _mean(init_times),
        "init_time_std": _std(init_times),
        "rebase_time_mean": _mean(rebase_times),
        "rebase_time_std": _std(rebase_times),
        "peak_memory_mean": _mean(peak_memories),
        "peak_memory_std": _std(peak_memories),
        "best_accuracy": max(accuracies) if accuracies else 0.0,
        "worst_accuracy": min(accuracies) if accuracies else 0.0,
    }


def _merge_seed_rows(method_to_rows: dict[str, list[dict[str, str]]]) -> list[dict[str, str | float | int]]:
    seeds = sorted({int(row["seed"]) for rows in method_to_rows.values() for row in rows})
    merged: list[dict[str, str | float | int]] = []
    indexed = {
        method: {int(row["seed"]): row for row in rows}
        for method, rows in method_to_rows.items()
    }
    for seed in seeds:
        lora = indexed["LoRA"][seed]
        lora_ga = indexed["LoRA-GA"][seed]
        gadi = indexed["GADI-R"][seed]
        lora_acc = _to_float(lora, "final_accuracy")
        lora_ga_acc = _to_float(lora_ga, "final_accuracy")
        gadi_acc = _to_float(gadi, "final_accuracy")
        best_method = max(
            [("LoRA", lora_acc), ("LoRA-GA", lora_ga_acc), ("GADI-R", gadi_acc)],
            key=lambda item: item[1],
        )[0]
        merged.append(
            {
                "seed": seed,
                "lora_accuracy": lora_acc,
                "lora_ga_accuracy": lora_ga_acc,
                "gadi_accuracy": gadi_acc,
                "best_method": best_method,
                "gadi_refreshed_layers": gadi.get("refreshed_layers_text", "无"),
            }
        )
    return merged


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        filtered_rows = [
            {field: row.get(field, "") for field in fieldnames}
            for row in rows
        ]
        writer.writerows(filtered_rows)


def build_markdown(
    task_name: str,
    suite_dir: Path,
    stats_rows: list[dict[str, float | str]],
    seed_rows: list[dict[str, str | float | int]],
    source_paths: dict[str, Path],
    gadi_config_desc: str,
) -> str:
    best_method = max(stats_rows, key=lambda row: row["accuracy_mean"])
    lines = [
        f"# RoBERTa-base + {task_name} 在 A10 上的公平多随机种子总对比报告",
        "",
        "## 一、报告目的",
        "",
        f"本报告汇总 `LoRA`、`LoRA-GA` 与固定主配置 `GADI-R` 在 `{task_name}` 任务上的公平对比结果，用于直接纳入论文实验章节。",
        "",
        "当前 GADI-R 主配置为：",
        "",
        f"- `{gadi_config_desc}`",
        "",
        "## 二、统一实验设置",
        "",
        "- 模型：`roberta-base`",
        f"- 数据集：`GLUE/{task_name.lower()}`",
        "- 硬件环境：`Alibaba Cloud ecs.gn7i-c8g1.2xlarge (NVIDIA A10 x1)`",
        "- 随机种子：`11, 42, 123, 3407, 2026`",
        "- 训练轮数：`1`",
        "",
        "对应原始结果目录如下：",
        "",
        f"- `LoRA`：`{source_paths['LoRA'].parent}`",
        f"- `LoRA-GA`：`{source_paths['LoRA-GA'].parent}`",
        f"- `GADI-R`：`{source_paths['GADI-R'].parent}`",
        "",
        "## 三、总体结果对比",
        "",
        "| 方法 | 最终 Accuracy mean ± std | 最终 Loss mean ± std | 总时间 mean ± std (s) | 初始化 mean ± std (s) | 重基化 mean ± std (s) | 峰值显存 mean ± std (MiB) |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]

    for row in stats_rows:
        lines.append(
            f"| {row['method']} | `{row['accuracy_mean']:.6f} ± {row['accuracy_std']:.6f}` | `{row['loss_mean']:.6f} ± {row['loss_std']:.6f}` | `{row['total_time_mean']:.2f} ± {row['total_time_std']:.2f}` | `{row['init_time_mean']:.2f} ± {row['init_time_std']:.2f}` | `{row['rebase_time_mean']:.2f} ± {row['rebase_time_std']:.2f}` | `{row['peak_memory_mean']:.2f} ± {row['peak_memory_std']:.2f}` |"
        )

    lines.extend(
        [
            "",
            "## 四、逐随机种子对比",
            "",
            "| seed | LoRA | LoRA-GA | GADI-R | 当个 seed 最优方法 | GADI 刷新层 |",
            "| ---: | ---: | ---: | ---: | --- | --- |",
        ]
    )
    for row in seed_rows:
        lines.append(
            f"| {row['seed']} | `{row['lora_accuracy']:.6f}` | `{row['lora_ga_accuracy']:.6f}` | `{row['gadi_accuracy']:.6f}` | `{row['best_method']}` | `{row['gadi_refreshed_layers']}` |"
        )

    lines.extend(
        [
            "",
            "## 五、时间与显存开销解读",
            "",
            "- `总时间` 反映一次完整 run 从进入训练流程到保存结果结束的整体开销。",
            "- `额外初始化时间` 对应方法特有初始化步骤。LoRA 近似为 0，LoRA-GA 与 GADI-R 主要来自梯度收集与低秩分解。",
            "- `重基化额外耗时` 主要反映 GADI-R 在训练中进行 drift 检测与重基化的额外开销，LoRA 与 LoRA-GA 为 0。",
            "- `峰值显存` 用于判断 GADI-R 的额外开销是否处于可接受范围。",
            "",
            "## 六、结论",
            "",
            f"- 当前任务上平均 Accuracy 最好的方法是 `{best_method['method']}`，其均值为 `{best_method['accuracy_mean']:.6f}`。",
            "- 若 GADI-R 的平均 Accuracy 同时高于 LoRA 与 LoRA-GA，且额外初始化时间与重基化耗时相对可控，则可说明其具备实际应用价值。",
            "",
            "## 七、目录说明",
            "",
            f"- 当前报告目录：`{suite_dir}`",
            "- 本目录包含总对比报告、方法统计表、逐 seed 对比表、时间显存汇总表。",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a task-level comparison report for LoRA / LoRA-GA / GADI-R.")
    parser.add_argument("--results-root", default="results")
    parser.add_argument("--task-name", required=True)
    parser.add_argument("--suite-name", required=True)
    parser.add_argument("--lora-experiment-type", required=True)
    parser.add_argument("--lora-ga-experiment-type", required=True)
    parser.add_argument("--gadi-experiment-type", required=True)
    parser.add_argument("--gadi-config-desc", default="step120 + gradient_mix=0.5 + query_only + topk_layers=1")
    args = parser.parse_args()

    results_root = Path(args.results_root)
    source_paths = {
        "LoRA": _find_latest_summary(results_root, args.lora_experiment_type),
        "LoRA-GA": _find_latest_summary(results_root, args.lora_ga_experiment_type),
        "GADI-R": _find_latest_summary(results_root, args.gadi_experiment_type),
    }

    method_to_rows = {
        method: _read_csv_rows(path)
        for method, path in source_paths.items()
    }
    stats_rows = [
        _method_stats(method, rows)
        for method, rows in method_to_rows.items()
    ]
    stats_rows.sort(key=lambda item: {"LoRA": 0, "LoRA-GA": 1, "GADI-R": 2}[str(item["method"])])
    seed_rows = _merge_seed_rows(method_to_rows)

    suite_dir = results_root / datetime.now().strftime("%Y-%m-%d") / args.suite_name
    suite_dir.mkdir(parents=True, exist_ok=True)

    _write_csv(
        suite_dir / "方法统计.csv",
        stats_rows,
        [
            "method",
            "accuracy_mean",
            "accuracy_std",
            "loss_mean",
            "loss_std",
            "total_time_mean",
            "total_time_std",
            "init_time_mean",
            "init_time_std",
            "rebase_time_mean",
            "rebase_time_std",
            "peak_memory_mean",
            "peak_memory_std",
            "best_accuracy",
            "worst_accuracy",
        ],
    )
    _write_csv(
        suite_dir / "逐seed对比.csv",
        seed_rows,
        [
            "seed",
            "lora_accuracy",
            "lora_ga_accuracy",
            "gadi_accuracy",
            "best_method",
            "gadi_refreshed_layers",
        ],
    )
    _write_csv(
        suite_dir / "时间显存汇总表.csv",
        stats_rows,
        [
            "method",
            "total_time_mean",
            "total_time_std",
            "init_time_mean",
            "init_time_std",
            "rebase_time_mean",
            "rebase_time_std",
            "peak_memory_mean",
            "peak_memory_std",
        ],
    )
    markdown = build_markdown(
        task_name=args.task_name,
        suite_dir=suite_dir,
        stats_rows=stats_rows,
        seed_rows=seed_rows,
        source_paths=source_paths,
        gadi_config_desc=args.gadi_config_desc,
    )
    (suite_dir / "总对比报告.md").write_text(markdown, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
