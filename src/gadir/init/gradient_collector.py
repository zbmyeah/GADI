from __future__ import annotations

import torch

from gadir.utils.logging import get_logger
from gadir.utils.peft import iter_lora_linear_layers

LOGGER = get_logger("gradient_collector")


def collect_lora_weight_gradients(
    model: torch.nn.Module,
    batches: list[dict[str, torch.Tensor]],
    adapter_name: str = "default",
    eval_mode: bool = False,
) -> dict[str, torch.Tensor]:
    if not batches:
        raise ValueError("At least one calibration batch is required for gradient collection.")

    lora_layers = list(iter_lora_linear_layers(model, adapter_name=adapter_name))
    previous_training_state = model.training
    requires_grad_state: list[tuple[torch.Tensor, bool]] = []
    LOGGER.info(
        "Collecting gradients | batches=%s | mode=%s | lora_layers=%s",
        len(batches),
        "eval" if eval_mode else "train",
        len(lora_layers),
    )

    for _, module in lora_layers:
        weight = module.base_layer.weight
        requires_grad_state.append((weight, weight.requires_grad))
        weight.requires_grad_(True)

    model.zero_grad(set_to_none=True)
    if eval_mode:
        model.eval()
    else:
        model.train()

    for batch_index, batch in enumerate(batches, start=1):
        outputs = model(**batch)
        (outputs.loss / len(batches)).backward()
        LOGGER.info(
            "Gradient collection progress | batch=%s/%s | loss=%.4f",
            batch_index,
            len(batches),
            float(outputs.loss.detach()),
        )

    gradients: dict[str, torch.Tensor] = {}
    for name, module in lora_layers:
        grad = module.base_layer.weight.grad
        if grad is not None:
            gradients[name] = grad.detach().float().clone()

    model.zero_grad(set_to_none=True)
    for weight, previous_flag in requires_grad_state:
        weight.requires_grad_(previous_flag)
    model.train(previous_training_state)
    LOGGER.info("Finished gradient collection for %s layers.", len(gradients))
    return gradients
