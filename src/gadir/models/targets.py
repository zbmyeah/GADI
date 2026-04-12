from __future__ import annotations

from gadir.config import ModelConfig

DEFAULT_TARGET_MODULES: dict[str, list[str]] = {
    "bert": ["query", "value"],
    "roberta": ["query", "value"],
    "deberta": ["query_proj", "value_proj"],
    "llama": ["q_proj", "v_proj"],
    "mistral": ["q_proj", "v_proj"],
    "t5": ["q", "v"],
}


def resolve_target_modules(model_config: ModelConfig) -> list[str]:
    if model_config.target_modules:
        return model_config.target_modules

    lower_name = model_config.model_name_or_path.lower()
    for model_key, modules in DEFAULT_TARGET_MODULES.items():
        if model_key in lower_name:
            return modules
    raise ValueError(
        "Could not infer target modules automatically. "
        "Please set model.target_modules explicitly in the config."
    )
