from __future__ import annotations

from collections.abc import Callable

from transformers import PreTrainedTokenizerBase

from gadir.config import DataConfig

GLUE_TEXT_FIELDS: dict[str, tuple[str, str | None]] = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
}


def build_glue_preprocess_fn(
    data_config: DataConfig,
    tokenizer: PreTrainedTokenizerBase,
) -> Callable:
    if data_config.dataset_config_name not in GLUE_TEXT_FIELDS:
        raise ValueError(f"Unsupported GLUE task: {data_config.dataset_config_name}")
    text_key, text_pair_key = GLUE_TEXT_FIELDS[data_config.dataset_config_name]

    def preprocess_fn(batch: dict) -> dict:
        tokenizer_kwargs = {
            "padding": False,
            "max_length": data_config.max_length,
            "truncation": True,
        }
        if text_pair_key is None:
            result = tokenizer(batch[text_key], **tokenizer_kwargs)
        else:
            result = tokenizer(batch[text_key], batch[text_pair_key], **tokenizer_kwargs)
        result["labels"] = batch["label"]
        return result

    return preprocess_fn
