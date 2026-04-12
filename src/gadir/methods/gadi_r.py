from __future__ import annotations

from dataclasses import asdict

import torch

from gadir.methods.base import CalibrationBatchProvider
from gadir.methods.lora_ga import LoraGAMethod
from gadir.rebase.updater import DynamicRebaser


class GADIRMethod(LoraGAMethod):
    def __init__(self, experiment_config):
        super().__init__(experiment_config)
        self.rebaser = DynamicRebaser(experiment_config, adapter_name=self.adapter_name)

    def after_optimizer_step(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        global_step: int,
        calibration_batch_provider: CalibrationBatchProvider,
        device: torch.device,
    ) -> None:
        del device
        self.rebaser.maybe_rebase(
            model=model,
            optimizer=optimizer,
            global_step=global_step,
            calibration_batch_provider=calibration_batch_provider,
        )

    def get_artifacts(self) -> dict:
        return {
            "rebase_history": [asdict(event) for event in self.rebaser.history],
            "rebase_count": self.rebaser.rebase_count,
        }
