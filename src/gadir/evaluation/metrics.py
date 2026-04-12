from __future__ import annotations

import torch
from torch.utils.data import DataLoader


def _move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


@torch.no_grad()
def evaluate_sequence_classification(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_examples = 0
    total_correct = 0

    for batch in dataloader:
        batch = _move_batch(batch, device)
        outputs = model(**batch)
        predictions = outputs.logits.argmax(dim=-1)
        labels = batch["labels"]

        total_loss += float(outputs.loss) * labels.size(0)
        total_correct += int((predictions == labels).sum().item())
        total_examples += labels.size(0)

    if total_examples == 0:
        return {"loss": 0.0, "accuracy": 0.0}

    return {
        "loss": total_loss / total_examples,
        "accuracy": total_correct / total_examples,
    }
