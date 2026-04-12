from __future__ import annotations

from dataclasses import asdict
import time

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
        start_time = time.perf_counter()
        self.rebaser.maybe_rebase(
            model=model,
            optimizer=optimizer,
            global_step=global_step,
            calibration_batch_provider=calibration_batch_provider,
        )
        self.add_rebase_time(time.perf_counter() - start_time)

    def get_artifacts(self) -> dict:
        return {
            **super().get_artifacts(),
            "rebase_history": [asdict(event) for event in self.rebaser.history],
            "rebase_count": self.rebaser.rebase_count,
        }
