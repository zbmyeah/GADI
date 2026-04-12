from __future__ import annotations

from collections.abc import Iterator

import torch


def iter_lora_linear_layers(model: torch.nn.Module, adapter_name: str = "default") -> Iterator[tuple[str, torch.nn.Module]]:
    for name, module in model.named_modules():
        if hasattr(module, "lora_A") and hasattr(module, "lora_B"):
            if adapter_name in module.lora_A and adapter_name in module.lora_B:
                yield name, module


def get_adapter_weights(module: torch.nn.Module, adapter_name: str = "default") -> tuple[torch.Tensor, torch.Tensor]:
    a_weight = module.lora_A[adapter_name].weight
    b_weight = module.lora_B[adapter_name].weight
    return a_weight, b_weight


def adapter_scaling(module: torch.nn.Module, adapter_name: str = "default") -> float:
    return float(module.scaling[adapter_name])


def compute_delta_weight(module: torch.nn.Module, adapter_name: str = "default") -> torch.Tensor:
    a_weight, b_weight = get_adapter_weights(module, adapter_name)
    scaling = adapter_scaling(module, adapter_name)
    return (b_weight @ a_weight) * scaling


@torch.no_grad()
def replace_adapter_preserving_function(
    module: torch.nn.Module,
    new_a: torch.Tensor,
    new_b: torch.Tensor,
    adapter_name: str = "default",
) -> None:
    a_weight, b_weight = get_adapter_weights(module, adapter_name)
    base_weight = module.base_layer.weight

    current_effective = base_weight.data + compute_delta_weight(module, adapter_name).to(base_weight.dtype)
    new_delta = (new_b @ new_a) * adapter_scaling(module, adapter_name)

    base_weight.data.copy_(current_effective - new_delta.to(base_weight.dtype))
    a_weight.data.copy_(new_a.to(device=a_weight.device, dtype=a_weight.dtype))
    b_weight.data.copy_(new_b.to(device=b_weight.device, dtype=b_weight.dtype))
