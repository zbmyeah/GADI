from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml


def _serialize(value: Any) -> Any:
    if is_dataclass(value):
        return {key: _serialize(item) for key, item in asdict(value).items()}
    if isinstance(value, dict):
        return {key: _serialize(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_serialize(item) for item in value]
    return value


def make_run_directory(
    results_root: str | Path,
    experiment_type: str,
    method_name: str,
    timestamp: datetime | None = None,
) -> Path:
    now = timestamp or datetime.now()
    date_dir = now.strftime("%Y-%m-%d")
    run_stamp = now.strftime("%H%M%S")
    run_dir = Path(results_root) / date_dir / experiment_type / f"{method_name}_{run_stamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def write_config_snapshot(config: Any, path: str | Path) -> None:
    payload = _serialize(config)
    with Path(path).open("w", encoding="utf-8-sig") as handle:
        yaml.safe_dump(payload, handle, allow_unicode=True, sort_keys=False)


def write_metrics_json(metrics: dict[str, Any], path: str | Path) -> None:
    with Path(path).open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, ensure_ascii=False, indent=2)


def write_markdown(path: str | Path, content: str) -> None:
    with Path(path).open("w", encoding="utf-8-sig") as handle:
        handle.write(content)
