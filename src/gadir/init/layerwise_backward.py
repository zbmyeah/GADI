from __future__ import annotations


def collect_layerwise_gradients(*args, **kwargs):
    raise NotImplementedError(
        "Large-model layer-wise backward optimization is intentionally separated from the "
        "first scaffold. Use collect_lora_weight_gradients for the prototype, and replace "
        "this module when moving to 7B-scale experiments."
    )
