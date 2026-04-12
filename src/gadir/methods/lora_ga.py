from __future__ import annotations

import torch

from gadir.init.gradient_collector import collect_lora_weight_gradients
from gadir.init.svd import build_lora_ga_factors
from gadir.methods.base import CalibrationBatchProvider
from gadir.methods.lora import VanillaLoraMethod
from gadir.utils.peft import iter_lora_linear_layers, replace_adapter_preserving_function


class LoraGAMethod(VanillaLoraMethod):
    def initialize(
        self,
        model: torch.nn.Module,
        calibration_batch_provider: CalibrationBatchProvider,
        device: torch.device,
    ) -> None:
        del device
        self.logger.info(
            "LoRA-GA initialization started | calibration_batches=%s",
            self.config.lora_ga.calibration_batches,
        )
        calibration_batches = calibration_batch_provider(self.config.lora_ga.calibration_batches)
        gradients = collect_lora_weight_gradients(
            model,
            calibration_batches,
            adapter_name=self.adapter_name,
        )
        self.logger.info(
            "LoRA-GA initialization collected gradients for %s layers. Building low-rank factors...",
            len(gradients),
        )

        updated_layers = 0
        for layer_name, module in iter_lora_linear_layers(model, adapter_name=self.adapter_name):
            gradient = gradients.get(layer_name)
            if gradient is None:
                continue
            factors = build_lora_ga_factors(
                gradient=gradient,
                rank=self.config.lora.rank,
                gamma=self.config.lora_ga.gamma,
            )
            replace_adapter_preserving_function(
                module,
                new_a=factors.a,
                new_b=factors.b,
                adapter_name=self.adapter_name,
            )
            updated_layers += 1
            if updated_layers == 1 or updated_layers % 4 == 0:
                self.logger.info(
                    "LoRA-GA initialization progress | updated_layers=%s | latest_layer=%s",
                    updated_layers,
                    layer_name,
                )

        self.logger.info("LoRA-GA initialization refreshed %s LoRA layers.", updated_layers)
