from __future__ import annotations

from dataclasses import dataclass
import time

import torch

from gadir.config import ExperimentConfig
from gadir.init.gradient_collector import collect_lora_weight_gradients
from gadir.init.svd import build_rebase_factors
from gadir.methods.base import CalibrationBatchProvider
from gadir.rebase.drift import compute_drift_score, compute_residual_gradient
from gadir.rebase.scheduler import IntervalRebaseScheduler
from gadir.utils.logging import get_logger
from gadir.utils.peft import get_adapter_weights, iter_lora_linear_layers, replace_adapter_preserving_function


@dataclass(slots=True)
class RebaseEvent:
    step: int
    refreshed_layers: list[str]
    scores: dict[str, float]
    duration_seconds: float = 0.0


@dataclass(slots=True)
class LayerCandidate:
    raw_score: float
    layer_name: str
    module: torch.nn.Module
    gradient_for_refresh: torch.Tensor
    group_name: str


class DynamicRebaser:
    def __init__(self, experiment_config: ExperimentConfig, adapter_name: str = "default") -> None:
        self.config = experiment_config
        self.adapter_name = adapter_name
        self.scheduler = IntervalRebaseScheduler(experiment_config.rebase)
        self.logger = get_logger(self.__class__.__name__)
        self.rebase_count = 0
        self.history: list[RebaseEvent] = []

    def maybe_rebase(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        global_step: int,
        calibration_batch_provider: CalibrationBatchProvider,
    ) -> RebaseEvent | None:
        start_time = time.perf_counter()
        if not self.scheduler.should_rebase(global_step, self.rebase_count):
            return None

        self.logger.info(
            "GADI-R drift check triggered at step=%s | calibration_batches=%s",
            global_step,
            self.config.rebase.calibration_batches,
        )
        calibration_batches = calibration_batch_provider(self.config.rebase.calibration_batches)
        gradients = collect_lora_weight_gradients(
            model,
            calibration_batches,
            adapter_name=self.adapter_name,
            eval_mode=self.config.rebase.calibration_eval_mode,
        )

        candidates: list[LayerCandidate] = []
        layer_scores: dict[str, float] = {}
        for layer_name, module in iter_lora_linear_layers(model, adapter_name=self.adapter_name):
            gradient = gradients.get(layer_name)
            if gradient is None:
                continue
            a_weight, b_weight = get_adapter_weights(module, self.adapter_name)
            drift_score = compute_drift_score(gradient, a_weight, b_weight)
            layer_scores[layer_name] = drift_score

            if self.config.rebase.use_residual_gradient:
                residual_gradient = compute_residual_gradient(gradient, a_weight, b_weight)
                mix = min(max(self.config.rebase.gradient_mix, 0.0), 1.0)
                gradient_for_refresh = mix * gradient + (1.0 - mix) * residual_gradient
            else:
                gradient_for_refresh = gradient
            candidates.append(
                LayerCandidate(
                    raw_score=drift_score,
                    layer_name=layer_name,
                    module=module,
                    gradient_for_refresh=gradient_for_refresh,
                    group_name=self._infer_group_name(layer_name),
                )
            )

        if not candidates:
            self.logger.info("GADI-R found no LoRA layer gradients at step %s.", global_step)
            event = RebaseEvent(
                step=global_step,
                refreshed_layers=[],
                scores=layer_scores,
                duration_seconds=time.perf_counter() - start_time,
            )
            self.history.append(event)
            return event

        max_score = max(layer_scores.values())
        self.logger.info(
            "GADI-R drift summary at step=%s | max_score=%.4f | threshold=%.4f",
            global_step,
            max_score,
            self.config.rebase.drift_threshold,
        )
        if max_score < self.config.rebase.drift_threshold:
            self.logger.info("GADI-R checked drift at step %s but skipped re-basing.", global_step)
            event = RebaseEvent(
                step=global_step,
                refreshed_layers=[],
                scores=layer_scores,
                duration_seconds=time.perf_counter() - start_time,
            )
            self.history.append(event)
            return event

        selected_layers = self._select_candidates(candidates)
        self.logger.info(
            "GADI-R selected layers at step=%s: %s",
            global_step,
            ", ".join(candidate.layer_name for candidate in selected_layers),
        )

        refreshed_layers: list[str] = []
        for candidate in selected_layers:
            new_factors = build_rebase_factors(
                gradient=candidate.gradient_for_refresh,
                rank=self.config.lora.rank,
                gamma=self.config.lora_ga.gamma,
            )
            replace_adapter_preserving_function(
                candidate.module,
                new_a=new_factors.a,
                new_b=new_factors.b,
                adapter_name=self.adapter_name,
            )
            if self.config.rebase.reset_optimizer_state:
                self._reset_optimizer_state(optimizer, candidate.module)
            refreshed_layers.append(candidate.layer_name)

        if refreshed_layers:
            self.rebase_count += 1
            self.logger.info(
                "GADI-R re-based %s layers at step %s (count=%s): %s",
                len(refreshed_layers),
                global_step,
                self.rebase_count,
                ", ".join(refreshed_layers),
            )
        else:
            self.logger.info("GADI-R checked drift at step %s but skipped re-basing.", global_step)

        event = RebaseEvent(
            step=global_step,
            refreshed_layers=refreshed_layers,
            scores=layer_scores,
            duration_seconds=time.perf_counter() - start_time,
        )
        self.history.append(event)
        return event

    def _select_candidates(self, candidates: list[LayerCandidate]) -> list[LayerCandidate]:
        strategy = self.config.rebase.selection_strategy
        topk = max(1, self.config.rebase.topk_layers)
        if strategy == "global_topk":
            return sorted(candidates, key=lambda item: item.raw_score, reverse=True)[:topk]
        if strategy == "query_only":
            query_candidates = sorted(
                [item for item in candidates if item.group_name == "query"],
                key=lambda item: item.raw_score,
                reverse=True,
            )
            if query_candidates:
                return query_candidates[:topk]
            self.logger.info("query_only selection found no query candidates, falling back to global_topk.")
            return sorted(candidates, key=lambda item: item.raw_score, reverse=True)[:topk]
        if strategy == "module_normalized":
            max_by_group: dict[str, float] = {}
            for candidate in candidates:
                max_by_group[candidate.group_name] = max(max_by_group.get(candidate.group_name, 0.0), candidate.raw_score)

            def normalized_key(item: LayerCandidate) -> tuple[float, float]:
                denominator = max(max_by_group.get(item.group_name, 1e-8), 1e-8)
                return (item.raw_score / denominator, item.raw_score)

            return sorted(candidates, key=normalized_key, reverse=True)[:topk]
        if strategy == "balanced_qv":
            selected: list[LayerCandidate] = []
            query_candidates = sorted(
                [item for item in candidates if item.group_name == "query"],
                key=lambda item: item.raw_score,
                reverse=True,
            )
            value_candidates = sorted(
                [item for item in candidates if item.group_name == "value"],
                key=lambda item: item.raw_score,
                reverse=True,
            )
            if query_candidates:
                selected.append(query_candidates[0])
            if len(selected) < topk and value_candidates:
                selected.append(value_candidates[0])
            if len(selected) < topk:
                used_names = {item.layer_name for item in selected}
                remaining = [item for item in sorted(candidates, key=lambda item: item.raw_score, reverse=True) if item.layer_name not in used_names]
                selected.extend(remaining[: topk - len(selected)])
            return selected[:topk]
        raise ValueError(f"Unsupported rebase selection strategy: {strategy}")

    @staticmethod
    def _infer_group_name(layer_name: str) -> str:
        lower_name = layer_name.lower()
        if "query" in lower_name or "q_proj" in lower_name:
            return "query"
        if "value" in lower_name or "v_proj" in lower_name:
            return "value"
        return "other"

    def _reset_optimizer_state(self, optimizer: torch.optim.Optimizer, module: torch.nn.Module) -> None:
        for parameter in (
            module.lora_A[self.adapter_name].weight,
            module.lora_B[self.adapter_name].weight,
        ):
            optimizer.state.pop(parameter, None)
