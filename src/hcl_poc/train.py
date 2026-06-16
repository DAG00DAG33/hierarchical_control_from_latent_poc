from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import torch
from rich.console import Console
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from hcl_poc.config import Config
from hcl_poc.data import (
    Episode,
    fit_action_standardizer,
    fit_input_standardizer,
    load_episodes,
)
from hcl_poc.flow import flow_matching_loss
from hcl_poc.models import FlowModel, ObservationEncoder, RepresentationWorldModel
from hcl_poc.utils import Standardizer, Timer, default_device, ensure_dir, set_seed, write_json

console = Console()


def _obs_input(ep: Episode, standardizer: Standardizer) -> np.ndarray:
    return standardizer.transform(np.concatenate([ep.features, ep.proprio], axis=-1))


class RepresentationDataset(Dataset):
    def __init__(
        self,
        episodes: list[Episode],
        input_norm: Standardizer,
        action_norm: Standardizer,
        horizons: list[int],
        max_horizon: int,
        length: int = 200_000,
    ) -> None:
        self.episodes = episodes
        self.inputs = [_obs_input(ep, input_norm) for ep in episodes]
        self.actions = [action_norm.transform(ep.actions) for ep in episodes]
        self.horizons = horizons
        self.max_horizon = max_horizon
        self.length = length

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, _idx: int) -> dict[str, torch.Tensor]:
        ep_idx = np.random.randint(0, len(self.episodes))
        ep = self.episodes[ep_idx]
        valid_h = [h for h in self.horizons if ep.length > h]
        h = int(valid_h[np.random.randint(0, len(valid_h))])
        t = int(np.random.randint(0, ep.length - h))
        action_seq = np.zeros((self.max_horizon, ep.actions.shape[-1]), dtype=np.float32)
        action_seq[:h] = self.actions[ep_idx][t : t + h]
        return {
            "x_t": torch.from_numpy(self.inputs[ep_idx][t]),
            "x_future": torch.from_numpy(self.inputs[ep_idx][t + h]),
            "actions": torch.from_numpy(action_seq),
            "horizon": torch.tensor(h, dtype=torch.long),
        }


class FlowActionDataset(Dataset):
    def __init__(
        self,
        episodes: list[Episode],
        latents: list[np.ndarray],
        action_norm: Standardizer,
        chunk: int,
        length: int = 200_000,
        goal_horizon: int | None = None,
    ) -> None:
        min_future = chunk if goal_horizon is None else max(chunk, goal_horizon)
        keep = [idx for idx, ep in enumerate(episodes) if ep.length > min_future]
        if not keep:
            raise ValueError(f"No episodes longer than required horizon {min_future}")
        self.episodes = [episodes[idx] for idx in keep]
        self.latents = [latents[idx] for idx in keep]
        self.actions = [action_norm.transform(episodes[idx].actions) for idx in keep]
        self.chunk = chunk
        self.length = length
        self.goal_horizon = goal_horizon

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, _idx: int) -> dict[str, torch.Tensor]:
        ep_idx = np.random.randint(0, len(self.episodes))
        ep = self.episodes[ep_idx]
        min_future = self.chunk if self.goal_horizon is None else max(self.chunk, self.goal_horizon)
        t = int(np.random.randint(0, ep.length - min_future))
        chunk = self.actions[ep_idx][t : t + self.chunk].reshape(-1)
        z_t = self.latents[ep_idx][t]
        if self.goal_horizon is None:
            cond = z_t
        else:
            cond = np.concatenate([z_t, self.latents[ep_idx][t + self.goal_horizon]], axis=-1)
        return {"x": torch.from_numpy(chunk), "cond": torch.from_numpy(cond.astype(np.float32))}


class FlowLatentDataset(Dataset):
    def __init__(
        self,
        episodes: list[Episode],
        latents: list[np.ndarray],
        horizon: int,
        length: int = 200_000,
    ) -> None:
        keep = [idx for idx, ep in enumerate(episodes) if ep.length > horizon]
        if not keep:
            raise ValueError(f"No episodes longer than high-level horizon {horizon}")
        self.episodes = [episodes[idx] for idx in keep]
        self.latents = [latents[idx] for idx in keep]
        self.horizon = horizon
        self.length = length

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, _idx: int) -> dict[str, torch.Tensor]:
        ep_idx = np.random.randint(0, len(self.episodes))
        ep = self.episodes[ep_idx]
        t = int(np.random.randint(0, ep.length - self.horizon))
        return {
            "x": torch.from_numpy(self.latents[ep_idx][t + self.horizon]),
            "cond": torch.from_numpy(self.latents[ep_idx][t]),
        }


def _loader(dataset: Dataset, batch_size: int) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, num_workers=2, pin_memory=torch.cuda.is_available())


def train_representation(config: Config, n_traj: int, seed: int) -> Path:
    set_seed(seed)
    device = default_device()
    artifact_dir = ensure_dir(Path(config.get("paths.artifact_dir")) / f"n{n_traj}" / f"seed{seed}")
    ckpt_path = artifact_dir / "encoder.pt"
    if ckpt_path.exists():
        console.print(f"Encoder exists: {ckpt_path}")
        return ckpt_path

    episodes = load_episodes(config.get("paths.prepared_path"), limit=n_traj)
    input_norm = fit_input_standardizer(episodes)
    action_norm = fit_action_standardizer(episodes)
    input_dim = episodes[0].features.shape[-1] + episodes[0].proprio.shape[-1]
    action_dim = episodes[0].actions.shape[-1]
    latent_dim = int(config.get("representation.latent_dim"))
    hidden_dim = int(config.get("representation.hidden_dim"))
    horizons = [int(h) for h in config.get("representation.horizons_steps")]
    max_horizon = max(horizons)

    encoder = ObservationEncoder(input_dim, latent_dim, hidden_dim).to(device)
    world_model = RepresentationWorldModel(latent_dim, action_dim, hidden_dim).to(device)
    opt = torch.optim.AdamW(list(encoder.parameters()) + list(world_model.parameters()), lr=float(config.get("representation.lr")))
    dataset = RepresentationDataset(
        episodes,
        input_norm,
        action_norm,
        horizons,
        max_horizon,
        length=max(10_000, n_traj * 1000),
    )
    loader = _loader(dataset, int(config.get("representation.batch_size")))
    epochs = int(config.get("representation.epochs"))
    sig_weight = float(config.get("representation.sigreg_weight"))
    timer = Timer()
    last_loss = 0.0
    for _epoch in tqdm(range(epochs), desc=f"train encoder n={n_traj} seed={seed}"):
        for batch in loader:
            x_t = batch["x_t"].to(device)
            x_f = batch["x_future"].to(device)
            action_seq = batch["actions"].to(device)
            horizon = batch["horizon"].to(device)
            z_t = encoder(x_t)
            z_f = encoder(x_f)
            pred = world_model(z_t, action_seq, horizon)
            pred_loss = torch.mean((pred - z_f) ** 2)
            std = torch.sqrt(z_t.var(dim=0) + 1e-4)
            sigreg = torch.mean(torch.relu(1.0 - std))
            loss = pred_loss + sig_weight * sigreg
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            last_loss = float(loss.detach().cpu())

    torch.save(
        {
            "encoder": encoder.state_dict(),
            "world_model": world_model.state_dict(),
            "input_norm": input_norm.state_dict(),
            "action_norm": action_norm.state_dict(),
            "input_dim": input_dim,
            "action_dim": action_dim,
            "latent_dim": latent_dim,
            "hidden_dim": hidden_dim,
            "elapsed_s": timer.elapsed(),
            "last_loss": last_loss,
        },
        ckpt_path,
    )
    write_json(artifact_dir / "encoder_metrics.json", {"elapsed_s": timer.elapsed(), "loss": last_loss})
    return ckpt_path


@torch.inference_mode()
def encode_latents(config: Config, n_traj: int, seed: int) -> tuple[list[Episode], list[np.ndarray], dict]:
    device = default_device()
    ckpt = torch.load(
        Path(config.get("paths.artifact_dir")) / f"n{n_traj}" / f"seed{seed}" / "encoder.pt",
        map_location=device,
        weights_only=False,
    )
    encoder = ObservationEncoder(ckpt["input_dim"], ckpt["latent_dim"], ckpt["hidden_dim"]).to(device)
    encoder.load_state_dict(ckpt["encoder"])
    encoder.eval()
    input_norm = Standardizer.from_state_dict(ckpt["input_norm"])
    episodes = load_episodes(config.get("paths.prepared_path"), limit=n_traj)
    latents: list[np.ndarray] = []
    for ep in episodes:
        x = torch.from_numpy(_obs_input(ep, input_norm)).to(device)
        out = []
        for start in range(0, len(x), 4096):
            out.append(encoder(x[start : start + 4096]).cpu().numpy())
        latents.append(np.concatenate(out, axis=0).astype(np.float32))
    return episodes, latents, ckpt


def train_flow_policy(
    config: Config,
    n_traj: int,
    seed: int,
    kind: Literal["flat", "low", "high"],
    horizon_steps: int | None = None,
) -> Path:
    set_seed(seed)
    device = default_device()
    train_representation(config, n_traj, seed)
    episodes, latents, encoder_ckpt = encode_latents(config, n_traj, seed)
    artifact_dir = ensure_dir(Path(config.get("paths.artifact_dir")) / f"n{n_traj}" / f"seed{seed}")
    name = kind if horizon_steps is None else f"{kind}_h{horizon_steps}"
    ckpt_path = artifact_dir / f"{name}.pt"
    if ckpt_path.exists():
        console.print(f"Policy exists: {ckpt_path}")
        return ckpt_path

    action_norm = Standardizer.from_state_dict(encoder_ckpt["action_norm"])
    chunk = int(config.get("policy.action_chunk_steps"))
    latent_dim = int(encoder_ckpt["latent_dim"])
    hidden_dim = int(config.get("policy.hidden_dim"))

    if kind == "flat":
        dataset = FlowActionDataset(
            episodes,
            latents,
            action_norm,
            chunk,
            length=max(10_000, n_traj * 1000),
        )
        sample_dim = chunk * episodes[0].actions.shape[-1]
        cond_dim = latent_dim
    elif kind == "low":
        if horizon_steps is None:
            raise ValueError("Low-level training requires horizon_steps")
        dataset = FlowActionDataset(
            episodes,
            latents,
            action_norm,
            chunk,
            length=max(10_000, n_traj * 1000),
            goal_horizon=horizon_steps,
        )
        sample_dim = chunk * episodes[0].actions.shape[-1]
        cond_dim = 2 * latent_dim
    elif kind == "high":
        if horizon_steps is None:
            raise ValueError("High-level training requires horizon_steps")
        dataset = FlowLatentDataset(
            episodes,
            latents,
            horizon_steps,
            length=max(10_000, n_traj * 1000),
        )
        sample_dim = latent_dim
        cond_dim = latent_dim
    else:
        raise ValueError(kind)

    model = FlowModel(sample_dim, cond_dim, hidden_dim).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(config.get("policy.lr")))
    loader = _loader(dataset, int(config.get("policy.batch_size")))
    epochs = int(config.get("policy.epochs"))
    timer = Timer()
    last_loss = 0.0
    for _epoch in tqdm(range(epochs), desc=f"train {name} n={n_traj} seed={seed}"):
        for batch in loader:
            x = batch["x"].to(device).float()
            cond = batch["cond"].to(device).float()
            loss = flow_matching_loss(model, x, cond)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            last_loss = float(loss.detach().cpu())

    torch.save(
        {
            "model": model.state_dict(),
            "kind": kind,
            "horizon_steps": horizon_steps,
            "sample_dim": sample_dim,
            "cond_dim": cond_dim,
            "hidden_dim": hidden_dim,
            "latent_dim": latent_dim,
            "action_dim": episodes[0].actions.shape[-1],
            "chunk": chunk,
            "action_norm": action_norm.state_dict(),
            "flow_steps": int(config.get("policy.flow_steps")),
            "elapsed_s": timer.elapsed(),
            "last_loss": last_loss,
        },
        ckpt_path,
    )
    write_json(artifact_dir / f"{name}_metrics.json", {"elapsed_s": timer.elapsed(), "loss": last_loss})
    return ckpt_path
