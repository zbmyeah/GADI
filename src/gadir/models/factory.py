from __future__ import annotations

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
)

from gadir.config import ModelConfig
from gadir.models.targets import resolve_target_modules


def _resolve_dtype(dtype_name: str | None) -> torch.dtype | None:
    if dtype_name is None:
        return None
    if not hasattr(torch, dtype_name):
        raise ValueError(f"Unsupported torch dtype: {dtype_name}")
    return getattr(torch, dtype_name)


def load_model_and_tokenizer(model_config: ModelConfig):
    target_modules = resolve_target_modules(model_config)
    torch_dtype = _resolve_dtype(model_config.torch_dtype)

    tokenizer = AutoTokenizer.from_pretrained(model_config.model_name_or_path, use_fast=True)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    task_type = model_config.task_type.upper()
    common_kwargs = {"torch_dtype": torch_dtype} if torch_dtype is not None else {}
    if task_type == "SEQ_CLS":
        model = AutoModelForSequenceClassification.from_pretrained(
            model_config.model_name_or_path,
            num_labels=model_config.num_labels,
            **common_kwargs,
        )
    elif task_type == "SEQ_2_SEQ_LM":
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_config.model_name_or_path,
            **common_kwargs,
        )
    elif task_type == "CAUSAL_LM":
        model = AutoModelForCausalLM.from_pretrained(
            model_config.model_name_or_path,
            **common_kwargs,
        )
    else:
        raise ValueError(f"Unsupported task type: {model_config.task_type}")

    return model, tokenizer, target_modules
