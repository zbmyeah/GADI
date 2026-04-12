from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(slots=True)
class ModelConfig:
    model_name_or_path: str
    task_type: str = "SEQ_CLS"
    num_labels: int = 2
    target_modules: list[str] = field(default_factory=list)
    torch_dtype: str | None = None


@dataclass(slots=True)
class DataConfig:
    dataset_name: str
    dataset_config_name: str | None = None
    max_length: int = 128
    calibration_size: int = 8


@dataclass(slots=True)
class LoraExperimentConfig:
    rank: int = 8
    alpha: int = 16
    dropout: float = 0.0
    bias: str = "none"
    use_rslora: bool = False
    adapter_name: str = "default"


@dataclass(slots=True)
class LoraGAConfig:
    gamma: float = 16.0
    calibration_batches: int = 1


@dataclass(slots=True)
class RebaseConfig:
    enabled: bool = False
    interval_steps: int = 200
    warmup_steps: int = 0
    max_rebases: int | None = None
    drift_threshold: float = 0.15
    topk_layers: int = 2
    selection_strategy: str = "global_topk"
    calibration_batches: int = 1
    use_residual_gradient: bool = True
    gradient_mix: float = 0.7
    calibration_eval_mode: bool = True
    reset_optimizer_state: bool = False


@dataclass(slots=True)
class OptimizerConfig:
    learning_rate: float = 1e-4
    weight_decay: float = 0.0


@dataclass(slots=True)
class TrainingConfig:
    output_dir: str = "outputs/default"
    train_batch_size: int = 8
    eval_batch_size: int = 32
    dataloader_num_workers: int = 0
    pin_memory: bool = False
    persistent_workers: bool = False
    gradient_accumulation_steps: int = 1
    num_epochs: int = 1
    max_steps: int | None = None
    log_every_steps: int = 10
    eval_every_steps: int = 100
    use_amp: bool = False
    amp_dtype: str = "bfloat16"
    allow_tf32: bool = False


@dataclass(slots=True)
class ExperimentConfig:
    seed: int
    method: str
    model: ModelConfig
    data: DataConfig
    lora: LoraExperimentConfig = field(default_factory=LoraExperimentConfig)
    lora_ga: LoraGAConfig = field(default_factory=LoraGAConfig)
    rebase: RebaseConfig = field(default_factory=RebaseConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)


def _build_dataclass(dataclass_type: type[Any], payload: dict[str, Any] | None) -> Any:
    return dataclass_type(**(payload or {}))


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)

    return ExperimentConfig(
        seed=raw["seed"],
        method=raw["method"],
        model=_build_dataclass(ModelConfig, raw.get("model")),
        data=_build_dataclass(DataConfig, raw.get("data")),
        lora=_build_dataclass(LoraExperimentConfig, raw.get("lora")),
        lora_ga=_build_dataclass(LoraGAConfig, raw.get("lora_ga")),
        rebase=_build_dataclass(RebaseConfig, raw.get("rebase")),
        optimizer=_build_dataclass(OptimizerConfig, raw.get("optimizer")),
        training=_build_dataclass(TrainingConfig, raw.get("training")),
    )
