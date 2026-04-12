from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(slots=True)
class LowRankFactors:
    a: torch.Tensor
    b: torch.Tensor


def _pad_to_rank(a_factor: torch.Tensor, b_factor: torch.Tensor, target_rank: int) -> LowRankFactors:
    current_rank = a_factor.shape[0]
    if current_rank == target_rank:
        return LowRankFactors(a=a_factor, b=b_factor)
    if current_rank > target_rank:
        return LowRankFactors(a=a_factor[:target_rank], b=b_factor[:, :target_rank])

    pad_rank = target_rank - current_rank
    a_pad = torch.zeros(
        pad_rank,
        a_factor.shape[1],
        device=a_factor.device,
        dtype=a_factor.dtype,
    )
    b_pad = torch.zeros(
        b_factor.shape[0],
        pad_rank,
        device=b_factor.device,
        dtype=b_factor.dtype,
    )
    return LowRankFactors(
        a=torch.cat([a_factor, a_pad], dim=0),
        b=torch.cat([b_factor, b_pad], dim=1),
    )


def build_lora_ga_factors(gradient: torch.Tensor, rank: int, gamma: float) -> LowRankFactors:
    gradient = gradient.float()
    u_matrix, _, vh_matrix = torch.linalg.svd(gradient, full_matrices=False)
    usable_rank = min(rank, vh_matrix.shape[0], u_matrix.shape[1])
    scale = gradient.shape[0] ** 0.25 / max(gamma, 1e-6)

    a_factor = vh_matrix[:usable_rank, :] * scale
    left_start = usable_rank
    left_end = min(left_start + usable_rank, u_matrix.shape[1])
    b_factor = u_matrix[:, left_start:left_end]

    if b_factor.shape[1] < usable_rank:
        missing = usable_rank - b_factor.shape[1]
        b_factor = torch.cat([b_factor, u_matrix[:, :missing]], dim=1)
    b_factor = b_factor * scale
    return _pad_to_rank(a_factor, b_factor, rank)


def build_rebase_factors(gradient: torch.Tensor, rank: int, gamma: float) -> LowRankFactors:
    gradient = gradient.float()
    usable_rank = min(rank, min(gradient.shape))
    u_matrix, _, vh_matrix = torch.linalg.svd(gradient, full_matrices=False)
    u_matrix = u_matrix[:, :usable_rank]
    vh_matrix = vh_matrix[:usable_rank, :]

    scale = gradient.shape[0] ** 0.25 / max(gamma, 1e-6)
    b_factor = u_matrix * scale
    a_factor = vh_matrix * scale
    return _pad_to_rank(a_factor, b_factor, rank)
