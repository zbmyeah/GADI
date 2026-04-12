from __future__ import annotations

from dataclasses import dataclass

from datasets import Dataset, load_dataset
from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding, PreTrainedTokenizerBase

from gadir.config import ExperimentConfig
from gadir.data.preprocess import build_glue_preprocess_fn


@dataclass(slots=True)
class DataBundle:
    train_loader: DataLoader
    eval_loader: DataLoader
    calibration_loader: DataLoader
    train_dataset: Dataset
    eval_dataset: Dataset


def _select_eval_split(dataset_name: str, dataset_config_name: str | None) -> str:
    if dataset_name == "glue" and dataset_config_name == "mnli":
        return "validation_matched"
    return "validation"


def build_data_bundle(
    experiment_config: ExperimentConfig,
    tokenizer: PreTrainedTokenizerBase,
) -> DataBundle:
    if experiment_config.data.dataset_name != "glue":
        raise NotImplementedError("The first scaffold supports GLUE sequence classification only.")

    raw_dataset = load_dataset(
        experiment_config.data.dataset_name,
        experiment_config.data.dataset_config_name,
    )
    preprocess_fn = build_glue_preprocess_fn(experiment_config.data, tokenizer)
    tokenized_dataset = raw_dataset.map(
        preprocess_fn,
        batched=True,
        remove_columns=raw_dataset["train"].column_names,
    )

    train_dataset = tokenized_dataset["train"]
    eval_dataset = tokenized_dataset[_select_eval_split(
        experiment_config.data.dataset_name,
        experiment_config.data.dataset_config_name,
    )]
    calibration_size = min(experiment_config.data.calibration_size, len(train_dataset))
    calibration_dataset = train_dataset.select(range(calibration_size))

    collator = DataCollatorWithPadding(tokenizer=tokenizer)
    loader_kwargs = {
        "num_workers": experiment_config.training.dataloader_num_workers,
        "pin_memory": experiment_config.training.pin_memory,
    }
    if experiment_config.training.dataloader_num_workers > 0:
        loader_kwargs["persistent_workers"] = experiment_config.training.persistent_workers

    return DataBundle(
        train_loader=DataLoader(
            train_dataset,
            batch_size=experiment_config.training.train_batch_size,
            shuffle=True,
            collate_fn=collator,
            **loader_kwargs,
        ),
        eval_loader=DataLoader(
            eval_dataset,
            batch_size=experiment_config.training.eval_batch_size,
            shuffle=False,
            collate_fn=collator,
            **loader_kwargs,
        ),
        calibration_loader=DataLoader(
            calibration_dataset,
            batch_size=experiment_config.training.train_batch_size,
            shuffle=False,
            collate_fn=collator,
            **loader_kwargs,
        ),
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )
