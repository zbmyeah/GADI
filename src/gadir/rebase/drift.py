from __future__ import annotations

import torch


def _orthonormal_columns(matrix: torch.Tensor) -> torch.Tensor:
    if matrix.numel() == 0:
        return matrix
    q_matrix, _ = torch.linalg.qr(matrix.float(), mode="reduced")
    return q_matrix


def compute_residual_gradient(
    gradient: torch.Tensor,
    a_weight: torch.Tensor,
    b_weight: torch.Tensor,
) -> torch.Tensor:
    gradient = gradient.float()
    q_b = _orthonormal_columns(b_weight.detach())
    q_a = _orthonormal_columns(a_weight.detach().T)

    if q_b.numel() == 0 or q_a.numel() == 0:
        return gradient

    projection_b = q_b @ q_b.T
    projection_a = q_a @ q_a.T
    return gradient - projection_b @ gradient - gradient @ projection_a + projection_b @ gradient @ projection_a


def compute_drift_score(
    gradient: torch.Tensor,
    a_weight: torch.Tensor,
    b_weight: torch.Tensor,
) -> float:
    residual = compute_residual_gradient(gradient, a_weight, b_weight)
    denominator = gradient.float().norm().clamp_min(1e-8)
    return float((residual.norm() / denominator).item())
