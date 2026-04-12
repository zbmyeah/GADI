from __future__ import annotations

from gadir.config import ExperimentConfig
from gadir.methods.base import BaseAdaptationMethod
from gadir.methods.gadi_r import GADIRMethod
from gadir.methods.lora import VanillaLoraMethod
from gadir.methods.lora_ga import LoraGAMethod

METHOD_REGISTRY: dict[str, type[BaseAdaptationMethod]] = {
    "lora": VanillaLoraMethod,
    "lora_ga": LoraGAMethod,
    "gadi_r": GADIRMethod,
}


def build_method(experiment_config: ExperimentConfig) -> BaseAdaptationMethod:
    method_name = experiment_config.method.lower()
    if method_name not in METHOD_REGISTRY:
        raise ValueError(f"Unsupported method: {experiment_config.method}")
    return METHOD_REGISTRY[method_name](experiment_config)


__all__ = ["build_method", "BaseAdaptationMethod", "VanillaLoraMethod", "LoraGAMethod", "GADIRMethod"]
