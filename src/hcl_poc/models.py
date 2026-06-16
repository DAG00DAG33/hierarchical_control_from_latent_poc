from __future__ import annotations

import math

import torch
from torch import nn


def sinusoidal_time_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    if t.ndim == 1:
        t = t[:, None]
    half = dim // 2
    freqs = torch.exp(
        torch.arange(half, device=t.device, dtype=t.dtype) * (-math.log(10000.0) / max(half - 1, 1))
    )
    args = t * freqs[None, :]
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if emb.shape[-1] < dim:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb


class MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int, depth: int = 3) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        dim = in_dim
        for _ in range(depth):
            layers.extend([nn.Linear(dim, hidden_dim), nn.SiLU()])
            dim = hidden_dim
        layers.append(nn.Linear(dim, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ObservationEncoder(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = MLP(input_dim, latent_dim, hidden_dim, depth=3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ActionSequenceEncoder(nn.Module):
    def __init__(self, action_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.gru = nn.GRU(action_dim, hidden_dim, batch_first=True)
        self.horizon_embed = nn.Embedding(512, hidden_dim)

    def forward(self, actions: torch.Tensor, horizons: torch.Tensor) -> torch.Tensor:
        _, h = self.gru(actions)
        horizons = torch.clamp(horizons.long(), 0, self.horizon_embed.num_embeddings - 1)
        return h[-1] + self.horizon_embed(horizons)


class RepresentationWorldModel(nn.Module):
    def __init__(self, latent_dim: int, action_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.action_encoder = ActionSequenceEncoder(action_dim, hidden_dim)
        self.predictor = MLP(latent_dim + hidden_dim, latent_dim, hidden_dim, depth=3)

    def forward(self, z: torch.Tensor, action_seq: torch.Tensor, horizons: torch.Tensor) -> torch.Tensor:
        action_code = self.action_encoder(action_seq, horizons)
        return self.predictor(torch.cat([z, action_code], dim=-1))


class FlowModel(nn.Module):
    def __init__(self, sample_dim: int, cond_dim: int, hidden_dim: int, time_dim: int = 64) -> None:
        super().__init__()
        self.sample_dim = sample_dim
        self.cond_dim = cond_dim
        self.time_dim = time_dim
        self.net = MLP(sample_dim + cond_dim + time_dim, sample_dim, hidden_dim, depth=4)

    def forward(self, x_t: torch.Tensor, t: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        t_emb = sinusoidal_time_embedding(t, self.time_dim)
        return self.net(torch.cat([x_t, t_emb, cond], dim=-1))

