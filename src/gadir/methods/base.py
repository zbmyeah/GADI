from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable

import torch

from gadir.config import ExperimentConfig
from gadir.utils.logging import get_logger

CalibrationBatchProvider = Callable[[int], list[dict[str, torch.Tensor]]]


class BaseAdaptationMethod(ABC):
    def __init__(self, experiment_config: ExperimentConfig) -> None:
        self.config = experiment_config
        self.adapter_name = experiment_config.lora.adapter_name
        self.logger = get_logger(self.__class__.__name__)

    @abstractmethod
    def wrap_model(self, model: torch.nn.Module) -> torch.nn.Module:
        raise NotImplementedError

    def initialize(
        self,
        model: torch.nn.Module,
        calibration_batch_provider: CalibrationBatchProvider,
        device: torch.device,
    ) -> None:
        del model, calibration_batch_provider, device

    def after_optimizer_step(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        global_step: int,
        calibration_batch_provider: CalibrationBatchProvider,
        device: torch.device,
    ) -> None:
        del model, optimizer, global_step, calibration_batch_provider, device

    def get_artifacts(self) -> dict:
        return {}
