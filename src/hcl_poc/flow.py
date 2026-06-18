from __future__ import annotations

import torch
from torch import nn


def flow_matching_loss(model: nn.Module, x_1: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
    eps = torch.randn_like(x_1)
    t = torch.rand(x_1.shape[0], device=x_1.device, dtype=x_1.dtype)
    x_t = (1.0 - t[:, None]) * eps + t[:, None] * x_1
    target = x_1 - eps
    pred = model(x_t, t, cond)
    return torch.mean((pred - target) ** 2)


@torch.inference_mode()
def sample_flow(
    model: nn.Module,
    cond: torch.Tensor,
    steps: int,
    sample_dim: int,
    initial_noise: torch.Tensor | None = None,
) -> torch.Tensor:
    x = (
        torch.randn(cond.shape[0], sample_dim, device=cond.device, dtype=cond.dtype)
        if initial_noise is None
        else initial_noise.to(device=cond.device, dtype=cond.dtype)
    )
    dt = 1.0 / steps
    for i in range(steps):
        t = torch.full((cond.shape[0],), i / steps, device=cond.device, dtype=cond.dtype)
        x = x + dt * model(x, t, cond)
    return x
