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
from hcl_poc.eval import extract_runtime_rgb_proprio
from hcl_poc.features import dino_from_config
from hcl_poc.flow import flow_matching_loss, sample_flow
from hcl_poc.models import FlowModel, MLP, ObservationEncoder, RepresentationWorldModel
from hcl_poc.rl import _rl_paths, load_ppo_agent
from hcl_poc.utils import Standardizer, Timer, default_device, ensure_dir, set_seed, write_json

console = Console()


def _obs_input(ep: Episode, standardizer: Standardizer) -> np.ndarray:
    return standardizer.transform(np.concatenate([ep.features, ep.proprio], axis=-1))


def _dino_metadata(config: Config) -> dict[str, object]:
    return {
        "dino_model": config.get("dino.model_name"),
        "dino_feature_type": config.get("dino.feature_type", "cls"),
        "dino_spatial_pool": int(config.get("dino.spatial_pool", 4)),
    }


def _clip_episode_actions(config: Config, episodes: list[Episode]) -> list[Episode]:
    if not bool(config.get("policy.clip_actions_to_env_space", False)):
        return episodes
    low = np.asarray(config.get("policy.action_low"), dtype=np.float32)
    high = np.asarray(config.get("policy.action_high"), dtype=np.float32)
    return [
        Episode(ep.features, ep.proprio, np.clip(ep.actions, low, high).astype(np.float32))
        for ep in episodes
    ]


def _to_numpy(value: object) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _scalar_bool(value: object) -> bool:
    return bool(_to_numpy(value).reshape(-1)[0])


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
        goal_noise_std: float = 0.0,
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
        self.goal_noise_std = goal_noise_std

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
            z_goal = self.latents[ep_idx][t + self.goal_horizon]
            if self.goal_noise_std > 0.0:
                z_goal = z_goal + np.random.randn(*z_goal.shape).astype(np.float32) * self.goal_noise_std
            cond = np.concatenate([z_t, z_goal], axis=-1)
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


class ArrayActionDataset(Dataset):
    def __init__(self, inputs: np.ndarray, actions: np.ndarray, length: int) -> None:
        if len(inputs) != len(actions):
            raise ValueError(f"Input/action length mismatch: {len(inputs)} != {len(actions)}")
        self.inputs = inputs.astype(np.float32)
        self.actions = actions.astype(np.float32)
        self.length = length

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, _idx: int) -> dict[str, torch.Tensor]:
        idx = int(np.random.randint(0, len(self.inputs)))
        return {
            "cond": torch.from_numpy(self.inputs[idx]),
            "x": torch.from_numpy(self.actions[idx]),
        }


def _loader(dataset: Dataset, batch_size: int) -> DataLoader:
    return DataLoader(
        dataset, batch_size=batch_size, num_workers=2, pin_memory=torch.cuda.is_available()
    )


def train_representation(config: Config, n_traj: int, seed: int) -> Path:
    set_seed(seed)
    device = default_device()
    artifact_dir = ensure_dir(Path(config.get("paths.artifact_dir")) / f"n{n_traj}" / f"seed{seed}")
    ckpt_path = artifact_dir / "encoder.pt"
    if ckpt_path.exists():
        console.print(f"Encoder exists: {ckpt_path}")
        return ckpt_path

    episodes = _clip_episode_actions(
        config, load_episodes(config.get("paths.prepared_path"), limit=n_traj)
    )
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
    reconstruction_weight = float(config.get("representation.reconstruction_weight", 0.0))
    decoder = (
        MLP(latent_dim, input_dim, hidden_dim, depth=3).to(device)
        if reconstruction_weight > 0.0
        else None
    )
    params = list(encoder.parameters()) + list(world_model.parameters())
    if decoder is not None:
        params += list(decoder.parameters())
    opt = torch.optim.AdamW(params, lr=float(config.get("representation.lr")))
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
    last_pred_loss = 0.0
    last_reconstruction_loss = 0.0
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
            reconstruction_loss = torch.zeros((), device=device)
            if decoder is not None:
                reconstruction_loss = 0.5 * (
                    torch.mean((decoder(z_t) - x_t) ** 2) + torch.mean((decoder(z_f) - x_f) ** 2)
                )
                loss = loss + reconstruction_weight * reconstruction_loss
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            last_loss = float(loss.detach().cpu())
            last_pred_loss = float(pred_loss.detach().cpu())
            last_reconstruction_loss = float(reconstruction_loss.detach().cpu())

    payload = {
        "encoder": encoder.state_dict(),
        "world_model": world_model.state_dict(),
        "input_norm": input_norm.state_dict(),
        "action_norm": action_norm.state_dict(),
        "input_dim": input_dim,
        "action_dim": action_dim,
        "latent_dim": latent_dim,
        "hidden_dim": hidden_dim,
        "reconstruction_weight": reconstruction_weight,
        **_dino_metadata(config),
        "elapsed_s": timer.elapsed(),
        "last_loss": last_loss,
        "last_pred_loss": last_pred_loss,
        "last_reconstruction_loss": last_reconstruction_loss,
    }
    if decoder is not None:
        payload["decoder"] = decoder.state_dict()
    torch.save(payload, ckpt_path)
    write_json(
        artifact_dir / "encoder_metrics.json",
        {
            "elapsed_s": timer.elapsed(),
            "loss": last_loss,
            "pred_loss": last_pred_loss,
            "reconstruction_loss": last_reconstruction_loss,
            "reconstruction_weight": reconstruction_weight,
        },
    )
    return ckpt_path


@torch.inference_mode()
def encode_latents(
    config: Config, n_traj: int, seed: int
) -> tuple[list[Episode], list[np.ndarray], dict]:
    device = default_device()
    ckpt = torch.load(
        Path(config.get("paths.artifact_dir")) / f"n{n_traj}" / f"seed{seed}" / "encoder.pt",
        map_location=device,
        weights_only=False,
    )
    encoder = ObservationEncoder(ckpt["input_dim"], ckpt["latent_dim"], ckpt["hidden_dim"]).to(
        device
    )
    encoder.load_state_dict(ckpt["encoder"])
    encoder.eval()
    input_norm = Standardizer.from_state_dict(ckpt["input_norm"])
    episodes = _clip_episode_actions(
        config, load_episodes(config.get("paths.prepared_path"), limit=n_traj)
    )
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
    kind: Literal["flat", "flat_obs", "low", "high"],
    horizon_steps: int | None = None,
    force: bool = False,
) -> Path:
    set_seed(seed)
    device = default_device()
    artifact_dir = ensure_dir(Path(config.get("paths.artifact_dir")) / f"n{n_traj}" / f"seed{seed}")
    name = kind if horizon_steps is None else f"{kind}_h{horizon_steps}"
    ckpt_path = artifact_dir / f"{name}.pt"
    if ckpt_path.exists() and not force:
        console.print(f"Policy exists: {ckpt_path}")
        return ckpt_path

    chunk = int(config.get("policy.action_chunk_steps"))
    hidden_dim = int(config.get("policy.hidden_dim"))

    if kind == "flat_obs":
        episodes = _clip_episode_actions(
            config, load_episodes(config.get("paths.prepared_path"), limit=n_traj)
        )
        input_norm = fit_input_standardizer(episodes)
        action_norm = fit_action_standardizer(episodes)
        obs_inputs = [_obs_input(ep, input_norm) for ep in episodes]
        dataset = FlowActionDataset(
            episodes,
            obs_inputs,
            action_norm,
            chunk,
            length=max(10_000, n_traj * 1000),
        )
        sample_dim = chunk * episodes[0].actions.shape[-1]
        cond_dim = obs_inputs[0].shape[-1]
        latent_dim = None
    else:
        train_representation(config, n_traj, seed)
        episodes, latents, encoder_ckpt = encode_latents(config, n_traj, seed)
        action_norm = Standardizer.from_state_dict(encoder_ckpt["action_norm"])
        latent_dim = int(encoder_ckpt["latent_dim"])

    if kind == "flat_obs":
        pass
    elif kind == "flat":
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
            goal_noise_std=float(config.get("policy.low_subgoal_noise_std", 0.0)),
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
            "input_norm": input_norm.state_dict() if kind == "flat_obs" else None,
            "flow_steps": int(config.get("policy.flow_steps")),
            **_dino_metadata(config),
            "elapsed_s": timer.elapsed(),
            "last_loss": last_loss,
        },
        ckpt_path,
    )
    write_json(
        artifact_dir / f"{name}_metrics.json", {"elapsed_s": timer.elapsed(), "loss": last_loss}
    )
    return ckpt_path


def _load_flow_model(path: Path, device: torch.device) -> tuple[FlowModel, dict]:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = FlowModel(ckpt["sample_dim"], ckpt["cond_dim"], ckpt["hidden_dim"]).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, ckpt


def _sample_hierarchy_transitions(
    episodes: list[Episode],
    latents: list[np.ndarray],
    action_norm: Standardizer,
    horizon: int,
    chunk: int,
    samples: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    min_future = max(horizon, chunk)
    keep = [idx for idx, ep in enumerate(episodes) if ep.length > min_future]
    if not keep:
        raise ValueError(f"No episodes longer than required horizon {min_future}")
    norm_actions = [action_norm.transform(ep.actions) for ep in episodes]
    z_now: list[np.ndarray] = []
    z_future: list[np.ndarray] = []
    action_chunks: list[np.ndarray] = []
    for _ in range(samples):
        ep_idx = int(keep[rng.integers(0, len(keep))])
        ep = episodes[ep_idx]
        t = int(rng.integers(0, ep.length - min_future))
        z_now.append(latents[ep_idx][t])
        z_future.append(latents[ep_idx][t + horizon])
        action_chunks.append(norm_actions[ep_idx][t : t + chunk].reshape(-1))
    return (
        np.stack(z_now, axis=0).astype(np.float32),
        np.stack(z_future, axis=0).astype(np.float32),
        np.stack(action_chunks, axis=0).astype(np.float32),
    )


@torch.inference_mode()
def _sample_flow_numpy(
    model: FlowModel,
    cond: np.ndarray,
    steps: int,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    out: list[np.ndarray] = []
    for start in range(0, len(cond), batch_size):
        cond_t = torch.from_numpy(cond[start : start + batch_size]).to(device).float()
        sample = sample_flow(model, cond_t, steps, model.sample_dim).cpu().numpy()
        out.append(sample)
    return np.concatenate(out, axis=0).astype(np.float32)


def _latent_metrics(pred: np.ndarray, target: np.ndarray, current: np.ndarray) -> dict[str, float]:
    pred_mse = float(np.mean((pred - target) ** 2))
    persistence_mse = float(np.mean((current - target) ** 2))
    return {
        "mse": pred_mse,
        "rmse": float(np.sqrt(pred_mse)),
        "persistence_mse": persistence_mse,
        "persistence_rmse": float(np.sqrt(persistence_mse)),
        "mse_vs_persistence": float(pred_mse / max(persistence_mse, 1e-12)),
    }


def _action_metrics(
    pred_norm: np.ndarray, target_norm: np.ndarray, action_norm: Standardizer
) -> dict[str, float]:
    action_dim = int(action_norm.mean.shape[0])
    pred = action_norm.inverse(pred_norm.reshape(-1, action_dim)).reshape(pred_norm.shape)
    target = action_norm.inverse(target_norm.reshape(-1, action_dim)).reshape(target_norm.shape)
    first_pred = pred[:, :action_dim]
    first_target = target[:, :action_dim]
    return {
        "norm_mse": float(np.mean((pred_norm - target_norm) ** 2)),
        "chunk_mae": float(np.mean(np.abs(pred - target))),
        "first_action_mae": float(np.mean(np.abs(first_pred - first_target))),
        "first_action_max_abs_err": float(np.max(np.abs(first_pred - first_target))),
    }


def diagnose_hierarchy(
    config: Config,
    n_traj: int,
    seed: int,
    horizon_s: float,
    samples: int,
    out_path: Path,
) -> Path:
    set_seed(seed)
    device = default_device()
    artifact_dir = Path(config.get("paths.artifact_dir")) / f"n{n_traj}" / f"seed{seed}"
    h_steps = max(1, int(round(float(horizon_s) * float(config.get("control_freq", 20)))))
    chunk = int(config.get("policy.action_chunk_steps"))
    batch_size = int(config.get("policy.batch_size"))

    episodes, latents, encoder_ckpt = encode_latents(config, n_traj, seed)
    action_norm = Standardizer.from_state_dict(encoder_ckpt["action_norm"])
    z_now, z_future, action_target = _sample_hierarchy_transitions(
        episodes,
        latents,
        action_norm,
        h_steps,
        chunk,
        samples,
        seed + 91_000 + h_steps,
    )

    high, high_ckpt = _load_flow_model(artifact_dir / f"high_h{h_steps}.pt", device)
    low, low_ckpt = _load_flow_model(artifact_dir / f"low_h{h_steps}.pt", device)
    flat, flat_ckpt = _load_flow_model(artifact_dir / "flat.pt", device)

    high_pred = _sample_flow_numpy(high, z_now, int(high_ckpt["flow_steps"]), device, batch_size)
    low_oracle = _sample_flow_numpy(
        low,
        np.concatenate([z_now, z_future], axis=-1),
        int(low_ckpt["flow_steps"]),
        device,
        batch_size,
    )
    low_sampled_high = _sample_flow_numpy(
        low,
        np.concatenate([z_now, high_pred], axis=-1),
        int(low_ckpt["flow_steps"]),
        device,
        batch_size,
    )
    flat_pred = _sample_flow_numpy(flat, z_now, int(flat_ckpt["flow_steps"]), device, batch_size)

    metrics = {
        "config": str(config.path),
        "n_traj": int(n_traj),
        "seed": int(seed),
        "horizon_s": float(horizon_s),
        "horizon_steps": int(h_steps),
        "samples": int(samples),
        "high_subgoal": _latent_metrics(high_pred, z_future, z_now),
        "low_oracle_subgoal": _action_metrics(low_oracle, action_target, action_norm),
        "low_sampled_high_subgoal": _action_metrics(low_sampled_high, action_target, action_norm),
        "flat_latent": _action_metrics(flat_pred, action_target, action_norm),
    }
    write_json(out_path, metrics)
    console.print(f"Wrote hierarchy diagnostic: {out_path}")
    return out_path


def train_bc_policy(
    config: Config, n_traj: int, seed: int, force: bool = False, one_step: bool = False
) -> Path:
    set_seed(seed)
    device = default_device()
    artifact_dir = ensure_dir(Path(config.get("paths.artifact_dir")) / f"n{n_traj}" / f"seed{seed}")
    kind = "bc_obs_1step" if one_step else "bc_obs"
    ckpt_path = artifact_dir / f"{kind}.pt"
    if ckpt_path.exists() and not force:
        console.print(f"BC policy exists: {ckpt_path}")
        return ckpt_path

    episodes = _clip_episode_actions(
        config, load_episodes(config.get("paths.prepared_path"), limit=n_traj)
    )
    input_norm = fit_input_standardizer(episodes)
    action_norm = fit_action_standardizer(episodes)
    obs_inputs = [_obs_input(ep, input_norm) for ep in episodes]
    chunk = 1 if one_step else int(config.get("policy.action_chunk_steps"))
    hidden_dim = int(config.get("policy.hidden_dim"))
    sample_dim = chunk * episodes[0].actions.shape[-1]
    cond_dim = obs_inputs[0].shape[-1]

    dataset = FlowActionDataset(
        episodes,
        obs_inputs,
        action_norm,
        chunk,
        length=max(10_000, n_traj * 1000),
    )
    model = MLP(cond_dim, sample_dim, hidden_dim, depth=4).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(config.get("policy.lr")))
    loader = _loader(dataset, int(config.get("policy.batch_size")))
    epochs = int(config.get("policy.epochs"))
    timer = Timer()
    last_loss = 0.0
    for _epoch in tqdm(range(epochs), desc=f"train {kind} n={n_traj} seed={seed}"):
        for batch in loader:
            x = batch["x"].to(device).float()
            cond = batch["cond"].to(device).float()
            pred = model(cond)
            loss = torch.mean((pred - x) ** 2)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            last_loss = float(loss.detach().cpu())

    torch.save(
        {
            "model": model.state_dict(),
            "kind": kind,
            "sample_dim": sample_dim,
            "cond_dim": cond_dim,
            "hidden_dim": hidden_dim,
            "action_dim": episodes[0].actions.shape[-1],
            "chunk": chunk,
            "action_norm": action_norm.state_dict(),
            "input_norm": input_norm.state_dict(),
            **_dino_metadata(config),
            "elapsed_s": timer.elapsed(),
            "last_loss": last_loss,
        },
        ckpt_path,
    )
    write_json(
        artifact_dir / f"{kind}_metrics.json", {"elapsed_s": timer.elapsed(), "loss": last_loss}
    )
    return ckpt_path


def _structured_state_vector(obs: object) -> np.ndarray:
    if not isinstance(obs, dict):
        raise TypeError("DAgger teacher labeling requires structured state observations")
    agent = obs["agent"]
    extra = obs["extra"]
    parts = [
        _to_numpy(agent["qpos"]).reshape(-1),
        _to_numpy(agent["qvel"]).reshape(-1),
        _to_numpy(extra["tcp_pose"]).reshape(-1),
        _to_numpy(extra["goal_pos"]).reshape(-1),
        _to_numpy(extra["obj_pose"]).reshape(-1),
    ]
    return np.concatenate(parts, axis=0).astype(np.float32)


def _obj_pose_label(obs: object) -> np.ndarray:
    if not isinstance(obs, dict):
        raise TypeError("Pose labels require structured state observations")
    obj_pose = _to_numpy(obs["extra"]["obj_pose"]).reshape(-1).astype(np.float32)
    goal_pos = _to_numpy(obs["extra"]["goal_pos"]).reshape(-1).astype(np.float32)
    quat_w = float(obj_pose[3])
    quat_z = float(obj_pose[6])
    yaw = 2.0 * np.arctan2(quat_z, quat_w)
    return np.asarray(
        [obj_pose[0], obj_pose[1], np.sin(yaw), np.cos(yaw), goal_pos[0], goal_pos[1], goal_pos[2]],
        dtype=np.float32,
    )


@torch.inference_mode()
def _collect_pose_probe_samples(
    config: Config,
    seed: int,
    samples: int,
    out_path: Path,
) -> tuple[np.ndarray, np.ndarray]:
    import gymnasium as gym
    import mani_skill  # noqa: F401

    device = default_device()
    teacher = load_ppo_agent(_rl_paths(config).best, device)
    dino = dino_from_config(config, device)
    env = gym.make(
        config.get("env_id"),
        obs_mode=config.get("obs_mode"),
        control_mode=config.get("control_mode"),
        reward_mode="normalized_dense",
        render_mode=None,
    )
    action_low = np.asarray(env.action_space.low, dtype=np.float32)
    action_high = np.asarray(env.action_space.high, dtype=np.float32)
    collect_seed = int(config.get("rl.collect_seed", 70_000)) + 400_000 + seed * 10_000
    features: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    attempts = 0
    pbar = tqdm(total=samples, desc=f"collect pose labels seed={seed}")
    while len(features) < samples:
        attempts += 1
        obs, _info = env.reset(seed=collect_seed + attempts)
        done = False
        truncated = False
        while not (done or truncated) and len(features) < samples:
            rgb, _proprio = extract_runtime_rgb_proprio(obs)
            state = _structured_state_vector(obs)
            features.append(dino.encode_batch(rgb[None])[0])
            labels.append(_obj_pose_label(obs))
            action_t, _logprob, _entropy, _value = teacher.get_action_and_value(
                torch.from_numpy(state[None]).to(device).float(),
                deterministic=True,
            )
            action = action_t.detach().cpu().numpy()[0].astype(np.float32)
            if bool(config.get("policy.clip_actions_to_env_space", False)):
                action = np.clip(action, action_low, action_high).astype(np.float32)
            obs, _reward, done, truncated, _info = env.step(action)
            pbar.update(1)
    pbar.close()
    env.close()
    feature_arr = np.stack(features, axis=0).astype(np.float32)
    label_arr = np.stack(labels, axis=0).astype(np.float32)
    ensure_dir(out_path.parent)
    np.savez_compressed(out_path, features=feature_arr, labels=label_arr)
    return feature_arr, label_arr


def train_pose_predictor(
    config: Config,
    n_traj: int,
    seed: int,
    force: bool = False,
    samples: int = 4000,
) -> Path:
    set_seed(seed)
    device = default_device()
    artifact_dir = ensure_dir(Path(config.get("paths.artifact_dir")) / f"n{n_traj}" / f"seed{seed}")
    ckpt_path = artifact_dir / "pose_predictor.pt"
    if ckpt_path.exists() and not force:
        console.print(f"Pose predictor exists: {ckpt_path}")
        return ckpt_path

    labels_path = artifact_dir / "pose_predictor_labels.npz"
    if labels_path.exists() and not force:
        data = np.load(labels_path)
        features = data["features"].astype(np.float32)
        labels = data["labels"].astype(np.float32)
        if labels.shape[-1] != 7:
            features, labels = _collect_pose_probe_samples(config, seed, samples, labels_path)
    else:
        features, labels = _collect_pose_probe_samples(config, seed, samples, labels_path)

    rng = np.random.default_rng(seed)
    order = rng.permutation(len(features))
    split = int(0.8 * len(order))
    train_idx = order[:split]
    val_idx = order[split:]
    input_norm = Standardizer.fit(features[train_idx])
    label_norm = Standardizer.fit(labels[train_idx])
    train_dataset = ArrayActionDataset(
        input_norm.transform(features[train_idx]),
        label_norm.transform(labels[train_idx]),
        length=max(10_000, len(train_idx) * 50),
    )
    hidden_dim = int(config.get("policy.hidden_dim"))
    model = MLP(features.shape[-1], labels.shape[-1], hidden_dim, depth=3).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(config.get("policy.lr")))
    loader = DataLoader(
        train_dataset,
        batch_size=int(config.get("policy.batch_size")),
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    epochs = int(config.get("policy.epochs"))
    timer = Timer()
    last_loss = 0.0
    for _epoch in tqdm(range(epochs), desc=f"train pose predictor n={n_traj} seed={seed}"):
        for batch in loader:
            x = batch["x"].to(device).float()
            cond = batch["cond"].to(device).float()
            pred = model(cond)
            loss = torch.mean((pred - x) ** 2)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            last_loss = float(loss.detach().cpu())

    model.eval()
    with torch.inference_mode():
        val_x = torch.from_numpy(input_norm.transform(features[val_idx])).to(device).float()
        pred = label_norm.inverse(model(val_x).cpu().numpy())
    target = labels[val_idx]
    pred_yaw = np.arctan2(pred[:, 2], pred[:, 3])
    target_yaw = np.arctan2(target[:, 2], target[:, 3])
    yaw_err = np.arctan2(np.sin(pred_yaw - target_yaw), np.cos(pred_yaw - target_yaw))
    metrics = {
        "elapsed_s": timer.elapsed(),
        "loss": last_loss,
        "samples": int(len(features)),
        "val_samples": int(len(val_idx)),
        "val_pos_mae_x_m": float(np.mean(np.abs(pred[:, 0] - target[:, 0]))),
        "val_pos_mae_y_m": float(np.mean(np.abs(pred[:, 1] - target[:, 1]))),
        "val_yaw_mae_rad": float(np.mean(np.abs(yaw_err))),
        "val_yaw_mae_deg": float(np.degrees(np.mean(np.abs(yaw_err)))),
    }
    torch.save(
        {
            "model": model.state_dict(),
            "input_norm": input_norm.state_dict(),
            "label_norm": label_norm.state_dict(),
            "feature_dim": features.shape[-1],
            "pose_dim": labels.shape[-1],
            "hidden_dim": hidden_dim,
            **_dino_metadata(config),
            **metrics,
        },
        ckpt_path,
    )
    write_json(artifact_dir / "pose_predictor_metrics.json", metrics)
    return ckpt_path


def _pose_probe_metrics(pred: np.ndarray, target: np.ndarray) -> dict[str, float]:
    pred_yaw = np.arctan2(pred[:, 2], pred[:, 3])
    target_yaw = np.arctan2(target[:, 2], target[:, 3])
    yaw_err = np.arctan2(np.sin(pred_yaw - target_yaw), np.cos(pred_yaw - target_yaw))
    return {
        "pos_mae_x_m": float(np.mean(np.abs(pred[:, 0] - target[:, 0]))),
        "pos_mae_y_m": float(np.mean(np.abs(pred[:, 1] - target[:, 1]))),
        "pos_rmse_x_m": float(np.sqrt(np.mean((pred[:, 0] - target[:, 0]) ** 2))),
        "pos_rmse_y_m": float(np.sqrt(np.mean((pred[:, 1] - target[:, 1]) ** 2))),
        "yaw_mae_rad": float(np.mean(np.abs(yaw_err))),
        "yaw_mae_deg": float(np.degrees(np.mean(np.abs(yaw_err)))),
    }


@torch.inference_mode()
def _encode_probe_samples(
    config: Config, n_traj: int, seed: int, sample_path: Path
) -> tuple[np.ndarray, np.ndarray, dict]:
    device = default_device()
    ckpt = torch.load(
        Path(config.get("paths.artifact_dir")) / f"n{n_traj}" / f"seed{seed}" / "encoder.pt",
        map_location=device,
        weights_only=False,
    )
    encoder = ObservationEncoder(ckpt["input_dim"], ckpt["latent_dim"], ckpt["hidden_dim"]).to(
        device
    )
    encoder.load_state_dict(ckpt["encoder"])
    encoder.eval()
    input_norm = Standardizer.from_state_dict(ckpt["input_norm"])
    data = np.load(sample_path)
    raw_inputs = np.concatenate(
        [data["features"].astype(np.float32), data["proprios"].astype(np.float32)], axis=-1
    )
    inputs = input_norm.transform(raw_inputs)
    latents: list[np.ndarray] = []
    for start in range(0, len(inputs), 4096):
        x = torch.from_numpy(inputs[start : start + 4096]).to(device).float()
        latents.append(encoder(x).cpu().numpy())
    return (
        np.concatenate(latents, axis=0).astype(np.float32),
        data["labels"].astype(np.float32),
        ckpt,
    )


def probe_latent_pose(
    config: Config,
    n_traj: int,
    seed: int,
    sample_path: Path,
    out_path: Path,
    epochs: int = 300,
    hidden_dim: int = 256,
    batch_size: int = 256,
) -> Path:
    set_seed(seed)
    device = default_device()
    timer = Timer()
    latents, labels, ckpt = _encode_probe_samples(config, n_traj, seed, sample_path)
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(latents))
    split = int(0.8 * len(order))
    train_idx = order[:split]
    val_idx = order[split:]
    input_norm = Standardizer.fit(latents[train_idx])
    label_norm = Standardizer.fit(labels[train_idx])
    train_x = input_norm.transform(latents[train_idx])
    train_y = label_norm.transform(labels[train_idx])
    val_x = input_norm.transform(latents[val_idx])

    model = MLP(latents.shape[-1], labels.shape[-1], hidden_dim, depth=3).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    dataset = torch.utils.data.TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y))
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    last_loss = 0.0
    for _epoch in tqdm(range(epochs), desc=f"probe latent pose n={n_traj} seed={seed}"):
        for x, y in loader:
            x = x.to(device).float()
            y = y.to(device).float()
            pred = model(x)
            loss = torch.mean((pred - y) ** 2)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            last_loss = float(loss.detach().cpu())

    model.eval()
    with torch.inference_mode():
        train_pred = label_norm.inverse(
            model(torch.from_numpy(train_x).to(device).float()).cpu().numpy()
        )
        val_pred = label_norm.inverse(
            model(torch.from_numpy(val_x).to(device).float()).cpu().numpy()
        )
    baseline = np.broadcast_to(labels[train_idx].mean(axis=0, keepdims=True), labels[val_idx].shape)
    metrics = {
        "elapsed_s": timer.elapsed(),
        "samples": int(len(latents)),
        "train_samples": int(len(train_idx)),
        "val_samples": int(len(val_idx)),
        "latent_dim": int(ckpt["latent_dim"]),
        "hidden_dim": int(ckpt["hidden_dim"]),
        "probe_hidden_dim": int(hidden_dim),
        "probe_epochs": int(epochs),
        "loss": last_loss,
        "train": _pose_probe_metrics(train_pred, labels[train_idx]),
        "val": _pose_probe_metrics(val_pred, labels[val_idx]),
        "mean_pose_baseline_val": _pose_probe_metrics(baseline, labels[val_idx]),
    }
    write_json(out_path, metrics)
    console.print(f"Wrote latent pose probe: {out_path}")
    return out_path


@torch.inference_mode()
def _predict_pose_features(
    pose_ckpt: dict[str, object],
    features: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    model = MLP(
        int(pose_ckpt["feature_dim"]),
        int(pose_ckpt["pose_dim"]),
        int(pose_ckpt["hidden_dim"]),
        depth=3,
    ).to(device)
    model.load_state_dict(pose_ckpt["model"])
    model.eval()
    input_norm = Standardizer.from_state_dict(pose_ckpt["input_norm"])
    label_norm = Standardizer.from_state_dict(pose_ckpt["label_norm"])
    out: list[np.ndarray] = []
    x = input_norm.transform(features)
    for start in range(0, len(x), 4096):
        pred = model(torch.from_numpy(x[start : start + 4096]).to(device).float()).cpu().numpy()
        out.append(label_norm.inverse(pred))
    return np.concatenate(out, axis=0).astype(np.float32)


def train_pose_bc_policy(config: Config, n_traj: int, seed: int, force: bool = False) -> Path:
    set_seed(seed)
    device = default_device()
    artifact_dir = ensure_dir(Path(config.get("paths.artifact_dir")) / f"n{n_traj}" / f"seed{seed}")
    ckpt_path = artifact_dir / "bc_pose.pt"
    if ckpt_path.exists() and not force:
        console.print(f"Pose BC policy exists: {ckpt_path}")
        return ckpt_path

    pose_path = train_pose_predictor(config, n_traj, seed, force=force)
    pose_ckpt = torch.load(pose_path, map_location=device, weights_only=False)
    episodes = _clip_episode_actions(
        config, load_episodes(config.get("paths.prepared_path"), limit=n_traj)
    )
    pose_inputs: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    for ep in episodes:
        pose = _predict_pose_features(pose_ckpt, ep.features, device)
        pose_inputs.append(np.concatenate([pose, ep.proprio], axis=-1))
        actions.append(ep.actions)
    raw_inputs = np.concatenate(pose_inputs, axis=0).astype(np.float32)
    raw_actions = np.concatenate(actions, axis=0).astype(np.float32)
    input_norm = Standardizer.fit(raw_inputs)
    action_norm = Standardizer.fit(raw_actions)
    dataset = ArrayActionDataset(
        input_norm.transform(raw_inputs),
        action_norm.transform(raw_actions),
        length=max(10_000, n_traj * 1000),
    )

    hidden_dim = int(config.get("policy.hidden_dim"))
    action_dim = raw_actions.shape[-1]
    model = MLP(raw_inputs.shape[-1], action_dim, hidden_dim, depth=4).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(config.get("policy.lr")))
    loader = DataLoader(
        dataset,
        batch_size=int(config.get("policy.batch_size")),
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    epochs = int(config.get("policy.epochs"))
    timer = Timer()
    last_loss = 0.0
    for _epoch in tqdm(range(epochs), desc=f"train bc_pose n={n_traj} seed={seed}"):
        for batch in loader:
            x = batch["x"].to(device).float()
            cond = batch["cond"].to(device).float()
            pred = model(cond)
            loss = torch.mean((pred - x) ** 2)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            last_loss = float(loss.detach().cpu())

    torch.save(
        {
            "model": model.state_dict(),
            "kind": "bc_pose",
            "sample_dim": action_dim,
            "cond_dim": raw_inputs.shape[-1],
            "hidden_dim": hidden_dim,
            "action_dim": action_dim,
            "chunk": 1,
            "action_norm": action_norm.state_dict(),
            "input_norm": input_norm.state_dict(),
            "pose_model": pose_ckpt["model"],
            "pose_input_norm": pose_ckpt["input_norm"],
            "pose_label_norm": pose_ckpt["label_norm"],
            "pose_feature_dim": pose_ckpt["feature_dim"],
            "pose_dim": pose_ckpt["pose_dim"],
            "pose_hidden_dim": pose_ckpt["hidden_dim"],
            **_dino_metadata(config),
            "elapsed_s": timer.elapsed(),
            "last_loss": last_loss,
        },
        ckpt_path,
    )
    write_json(
        artifact_dir / "bc_pose_metrics.json", {"elapsed_s": timer.elapsed(), "loss": last_loss}
    )
    return ckpt_path


@torch.inference_mode()
def _collect_dagger_labels(
    config: Config,
    n_traj: int,
    seed: int,
    rollout_episodes: int,
) -> tuple[np.ndarray, np.ndarray]:
    import gymnasium as gym
    import mani_skill  # noqa: F401

    device = default_device()
    artifact_dir = Path(config.get("paths.artifact_dir")) / f"n{n_traj}" / f"seed{seed}"
    init_ckpt_path = artifact_dir / "bc_obs_1step.pt"
    if not init_ckpt_path.exists():
        train_bc_policy(config, n_traj, seed, one_step=True)
    learner_ckpt = torch.load(init_ckpt_path, map_location=device, weights_only=False)
    learner = MLP(
        learner_ckpt["cond_dim"], learner_ckpt["sample_dim"], learner_ckpt["hidden_dim"], depth=4
    ).to(device)
    learner.load_state_dict(learner_ckpt["model"])
    learner.eval()
    input_norm = Standardizer.from_state_dict(learner_ckpt["input_norm"])
    action_norm = Standardizer.from_state_dict(learner_ckpt["action_norm"])

    dino = dino_from_config(config, device)
    teacher = load_ppo_agent(_rl_paths(config).best, device)
    env = gym.make(
        config.get("env_id"),
        obs_mode=config.get("obs_mode"),
        control_mode=config.get("control_mode"),
        reward_mode="normalized_dense",
        render_mode=None,
    )
    action_low = np.asarray(env.action_space.low, dtype=np.float32)
    action_high = np.asarray(env.action_space.high, dtype=np.float32)
    eval_seed = int(config.get("data.eval_seed", 10000)) + 300_000 + seed * 10_000

    inputs: list[np.ndarray] = []
    teacher_actions: list[np.ndarray] = []
    for ep_idx in tqdm(
        range(rollout_episodes), desc=f"collect dagger labels n={n_traj} seed={seed}"
    ):
        obs, _info = env.reset(seed=eval_seed + ep_idx)
        done = False
        truncated = False
        while not (done or truncated):
            rgb, proprio = extract_runtime_rgb_proprio(obs)
            feat = dino.encode_batch(rgb[None])[0]
            learner_input = np.concatenate([feat, proprio], axis=0).astype(np.float32)
            cond = torch.from_numpy(input_norm.transform(learner_input[None])).to(device).float()
            pred = learner(cond).cpu().numpy()[0]
            action = action_norm.inverse(pred.reshape(-1, action_norm.mean.shape[0]))[0]
            if bool(config.get("policy.clip_actions_to_env_space", False)):
                action = np.clip(action, action_low, action_high).astype(np.float32)

            state = _structured_state_vector(obs)
            teacher_action_t, _logprob, _entropy, _value = teacher.get_action_and_value(
                torch.from_numpy(state[None]).to(device).float(),
                deterministic=True,
            )
            teacher_action = teacher_action_t.detach().cpu().numpy()[0].astype(np.float32)
            if bool(config.get("policy.clip_actions_to_env_space", False)):
                teacher_action = np.clip(teacher_action, action_low, action_high).astype(np.float32)
            inputs.append(learner_input)
            teacher_actions.append(teacher_action)

            obs, _reward, done, truncated, _info = env.step(action)
    env.close()
    return np.stack(inputs, axis=0), np.stack(teacher_actions, axis=0)


def train_dagger_bc_policy(
    config: Config,
    n_traj: int,
    seed: int,
    force: bool = False,
    rollout_episodes: int | None = None,
) -> Path:
    set_seed(seed)
    device = default_device()
    artifact_dir = ensure_dir(Path(config.get("paths.artifact_dir")) / f"n{n_traj}" / f"seed{seed}")
    ckpt_path = artifact_dir / "bc_obs_dagger.pt"
    if ckpt_path.exists() and not force:
        console.print(f"DAgger BC policy exists: {ckpt_path}")
        return ckpt_path

    rollout_episodes = rollout_episodes or int(config.get("data.eval_episodes", 50))
    episodes = _clip_episode_actions(
        config, load_episodes(config.get("paths.prepared_path"), limit=n_traj)
    )
    demo_inputs = np.concatenate(
        [np.concatenate([ep.features, ep.proprio], axis=-1) for ep in episodes], axis=0
    )
    demo_actions = np.concatenate([ep.actions for ep in episodes], axis=0)
    labels_path = artifact_dir / "bc_obs_dagger_labels.npz"
    if labels_path.exists() and not force:
        labels = np.load(labels_path)
        dagger_inputs = labels["inputs"]
        dagger_actions = labels["actions"]
    else:
        dagger_inputs, dagger_actions = _collect_dagger_labels(
            config, n_traj, seed, rollout_episodes
        )
        np.savez_compressed(labels_path, inputs=dagger_inputs, actions=dagger_actions)

    raw_inputs = np.concatenate([demo_inputs, dagger_inputs], axis=0).astype(np.float32)
    raw_actions = np.concatenate([demo_actions, dagger_actions], axis=0).astype(np.float32)
    input_norm = Standardizer.fit(raw_inputs)
    action_norm = Standardizer.fit(raw_actions)
    dataset = ArrayActionDataset(
        input_norm.transform(raw_inputs),
        action_norm.transform(raw_actions),
        length=max(10_000, n_traj * 1000),
    )

    hidden_dim = int(config.get("policy.hidden_dim"))
    action_dim = raw_actions.shape[-1]
    model = MLP(raw_inputs.shape[-1], action_dim, hidden_dim, depth=4).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(config.get("policy.lr")))
    loader = DataLoader(
        dataset,
        batch_size=int(config.get("policy.batch_size")),
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    epochs = int(config.get("policy.epochs"))
    timer = Timer()
    last_loss = 0.0
    for _epoch in tqdm(range(epochs), desc=f"train bc_obs_dagger n={n_traj} seed={seed}"):
        for batch in loader:
            x = batch["x"].to(device).float()
            cond = batch["cond"].to(device).float()
            pred = model(cond)
            loss = torch.mean((pred - x) ** 2)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            last_loss = float(loss.detach().cpu())

    torch.save(
        {
            "model": model.state_dict(),
            "kind": "bc_obs_dagger",
            "sample_dim": action_dim,
            "cond_dim": raw_inputs.shape[-1],
            "hidden_dim": hidden_dim,
            "action_dim": action_dim,
            "chunk": 1,
            "action_norm": action_norm.state_dict(),
            "input_norm": input_norm.state_dict(),
            "dagger_rollout_episodes": rollout_episodes,
            "dagger_samples": int(len(dagger_inputs)),
            "demo_samples": int(len(demo_inputs)),
            **_dino_metadata(config),
            "elapsed_s": timer.elapsed(),
            "last_loss": last_loss,
        },
        ckpt_path,
    )
    write_json(
        artifact_dir / "bc_obs_dagger_metrics.json",
        {
            "elapsed_s": timer.elapsed(),
            "loss": last_loss,
            "dagger_rollout_episodes": rollout_episodes,
            "dagger_samples": int(len(dagger_inputs)),
            "demo_samples": int(len(demo_inputs)),
        },
    )
    return ckpt_path


@torch.inference_mode()
def _collect_state_teacher_episodes(
    config: Config,
    n_traj: int,
    seed: int,
) -> tuple[list[np.ndarray], list[np.ndarray], int]:
    import gymnasium as gym
    import mani_skill  # noqa: F401

    device = default_device()
    teacher = load_ppo_agent(_rl_paths(config).best, device)
    env = gym.make(
        config.get("env_id"),
        obs_mode="state",
        control_mode=config.get("control_mode"),
        reward_mode="normalized_dense",
        render_mode=None,
        reconfiguration_freq=config.get("rl.collect_reconfiguration_freq", 1),
    )
    action_low = np.asarray(config.get("policy.action_low"), dtype=np.float32)
    action_high = np.asarray(config.get("policy.action_high"), dtype=np.float32)
    max_attempts = int(config.get("rl.collect_max_attempts", n_traj * 4))
    states: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    attempts = 0
    collect_seed = int(config.get("rl.collect_seed", 70_000)) + 200_000 + seed * max_attempts
    for attempts in tqdm(
        range(1, max_attempts + 1), desc=f"collect state teacher n={n_traj} seed={seed}"
    ):
        obs, _info = env.reset(seed=collect_seed + attempts)
        ep_states: list[np.ndarray] = []
        ep_actions: list[np.ndarray] = []
        done = False
        truncated = False
        success = False
        while not (done or truncated):
            state = _to_numpy(obs).reshape(-1).astype(np.float32)
            state_t = torch.from_numpy(state[None]).to(device).float()
            action_t, _logprob, _entropy, _value = teacher.get_action_and_value(
                state_t, deterministic=True
            )
            action = action_t.detach().cpu().numpy()[0].astype(np.float32)
            if bool(config.get("policy.clip_actions_to_env_space", False)):
                action = np.clip(action, action_low, action_high).astype(np.float32)
            ep_states.append(state)
            ep_actions.append(action)
            obs, _reward, done, truncated, info = env.step(action)
            success = success or _scalar_bool(info.get("success", False))
        if success:
            states.append(np.stack(ep_states, axis=0))
            actions.append(np.stack(ep_actions, axis=0))
            if len(states) >= n_traj:
                break
    env.close()
    if len(states) < n_traj:
        raise RuntimeError(
            f"Collected only {len(states)}/{n_traj} successful state teacher episodes"
        )
    return states, actions, attempts


def train_state_bc_policy(config: Config, n_traj: int, seed: int, force: bool = False) -> Path:
    set_seed(seed)
    device = default_device()
    artifact_dir = ensure_dir(Path(config.get("paths.artifact_dir")) / f"n{n_traj}" / f"seed{seed}")
    ckpt_path = artifact_dir / "bc_state.pt"
    if ckpt_path.exists() and not force:
        console.print(f"State BC policy exists: {ckpt_path}")
        return ckpt_path

    states, actions, attempts = _collect_state_teacher_episodes(config, n_traj, seed)
    input_norm = Standardizer.fit(np.concatenate(states, axis=0))
    action_norm = Standardizer.fit(np.concatenate(actions, axis=0))
    state_inputs = [input_norm.transform(x) for x in states]
    fake_episodes = [
        Episode(
            features=np.empty((len(a), 0), dtype=np.float32),
            proprio=np.empty((len(a), 0), dtype=np.float32),
            actions=a,
        )
        for a in actions
    ]

    chunk = int(config.get("policy.action_chunk_steps"))
    hidden_dim = int(config.get("policy.hidden_dim"))
    sample_dim = chunk * actions[0].shape[-1]
    cond_dim = states[0].shape[-1]
    dataset = FlowActionDataset(
        fake_episodes,
        state_inputs,
        action_norm,
        chunk,
        length=max(10_000, n_traj * 1000),
    )
    model = MLP(cond_dim, sample_dim, hidden_dim, depth=4).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(config.get("policy.lr")))
    loader = DataLoader(
        dataset,
        batch_size=int(config.get("policy.batch_size")),
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    epochs = int(config.get("policy.epochs"))
    timer = Timer()
    last_loss = 0.0
    for _epoch in tqdm(range(epochs), desc=f"train bc_state n={n_traj} seed={seed}"):
        for batch in loader:
            x = batch["x"].to(device).float()
            cond = batch["cond"].to(device).float()
            pred = model(cond)
            loss = torch.mean((pred - x) ** 2)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            last_loss = float(loss.detach().cpu())

    torch.save(
        {
            "model": model.state_dict(),
            "kind": "bc_state",
            "sample_dim": sample_dim,
            "cond_dim": cond_dim,
            "hidden_dim": hidden_dim,
            "action_dim": actions[0].shape[-1],
            "chunk": chunk,
            "action_norm": action_norm.state_dict(),
            "input_norm": input_norm.state_dict(),
            "attempts": attempts,
            "elapsed_s": timer.elapsed(),
            "last_loss": last_loss,
        },
        ckpt_path,
    )
    write_json(
        artifact_dir / "bc_state_metrics.json",
        {"elapsed_s": timer.elapsed(), "loss": last_loss, "attempts": attempts},
    )
    return ckpt_path
