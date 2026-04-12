from __future__ import annotations

import argparse
import sys
from itertools import cycle
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from gadir.config import load_experiment_config
from gadir.data.loaders import build_data_bundle
from gadir.methods import build_method
from gadir.models.factory import load_model_and_tokenizer
from gadir.utils.logging import get_logger
from gadir.utils.peft import iter_lora_linear_layers


def _move_batch_to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run only the LoRA-GA or GADI-R initialization stage.")
    parser.add_argument("--config", required=True, help="Path to an experiment yaml file.")
    args = parser.parse_args()

    logger = get_logger("run_init_only")
    config = load_experiment_config(args.config)
    model, tokenizer, target_modules = load_model_and_tokenizer(config.model)
    config.model.target_modules = target_modules
    method = build_method(config)
    model = method.wrap_model(model)

    data_bundle = build_data_bundle(config, tokenizer)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    calibration_iterator = cycle(data_bundle.calibration_loader)
    calibration_batch_provider = lambda batch_count: [
        _move_batch_to_device(next(calibration_iterator), device) for _ in range(batch_count)
    ]
    method.initialize(model, calibration_batch_provider, device)

    logger.info("Initialization finished for method=%s", config.method)
    logger.info(
        "LoRA layers discovered: %s",
        [layer_name for layer_name, _ in iter_lora_linear_layers(model, adapter_name=config.lora.adapter_name)],
    )


if __name__ == "__main__":
    main()
