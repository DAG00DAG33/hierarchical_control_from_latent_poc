from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from rich.console import Console
from torch import nn
from tqdm import trange

from hcl_poc.config import Config
from hcl_poc.incremental import _load_phase6_train_episodes
from hcl_poc.learned_interface import (
    _encode_effect_array,
    _load_representation,
    prepare_learned_interface_episodes,
    train_learned_interface_representation,
)
from hcl_poc.models import MLP
from hcl_poc.utils import Standardizer, default_device, ensure_dir, set_seed, write_json

console = Console()


class ReachabilityDistance(nn.Module):
    def __init__(self, latent_dim: int, hidden_dim: int = 512, depth: int = 3) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.net = MLP(latent_dim * 4 + 1, 1, hidden_dim, depth=depth)

    def forward(self, start: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        delta = goal - start
        l2 = torch.linalg.vector_norm(delta, dim=-1, keepdim=True)
        features = torch.cat([start, goal, delta, delta.abs(), l2], dim=-1)
        return torch.sigmoid(self.net(features)).squeeze(-1)


def _artifact_dir(config: Config, candidate: str, seed: int) -> Path:
    return ensure_dir(
        config.path_value("paths.incremental_artifact_dir")
        / "reachability_distance"
        / candidate
        / f"seed{seed}"
    )


def _result_dir(config: Config, candidate: str, seed: int) -> Path:
    return ensure_dir(
        config.path_value("paths.incremental_results_dir")
        / "reachability_distance"
        / candidate
        / f"seed{seed}"
    )


def _valid_latent_episodes(episodes: list[np.ndarray]) -> list[np.ndarray]:
    valid = [
        np.asarray(episode, dtype=np.float32)
        for episode in episodes
        if len(episode) >= 2
    ]
    if not valid:
        raise ValueError("No reachability latent episodes with at least two frames")
    return valid


def _load_encoded_latents(
    config: Config,
    candidate: str,
    seed: int,
    force: bool = False,
) -> tuple[list[np.ndarray], list[np.ndarray], Path]:
    encoded_path = prepare_learned_interface_episodes(
        config, candidate, seed=seed, force=force
    )
    encoded = torch.load(encoded_path, map_location="cpu", weights_only=False)
    if "train_goals" not in encoded or "validation_goals" not in encoded:
        raise ValueError(
            f"Unsupported encoded episode cache format for reachability: {encoded_path}"
        )
    return (
        _valid_latent_episodes(encoded["train_goals"]),
        _valid_latent_episodes(encoded["validation_goals"]),
        encoded_path,
    )


def _effect_progress_cache_path(
    config: Config,
    candidate: str,
    seed: int,
    horizon_steps: int,
    anchor_stride: int,
    max_span: int,
) -> Path:
    return (
        _artifact_dir(config, candidate, seed)
        / f"effect_progress_h{horizon_steps}_stride{anchor_stride}_span{max_span}.pt"
    )


def _build_effect_progress_episodes(
    frame_episodes: list[dict[str, np.ndarray]],
    encoder: nn.Module,
    frame_norm: Standardizer,
    device: torch.device,
    *,
    anchor_stride: int,
    max_span: int,
) -> list[np.ndarray]:
    if anchor_stride <= 0:
        raise ValueError("anchor_stride must be positive")
    if max_span <= 1:
        raise ValueError("max_span must be greater than one")
    progress_episodes: list[np.ndarray] = []
    for episode in frame_episodes:
        frames = np.asarray(episode["frames"], dtype=np.float32)
        if len(frames) < 2:
            continue
        for base in range(0, len(frames) - 1, anchor_stride):
            stop = min(len(frames), base + max_span)
            if stop - base < 2:
                continue
            start_frames = np.repeat(frames[base : base + 1], stop - base, axis=0)
            future_frames = frames[base:stop]
            effects = _encode_effect_array(
                encoder,
                frame_norm,
                start_frames,
                future_frames,
                np.ones(stop - base, dtype=np.float32),
                device,
            )
            progress_episodes.append(effects.astype(np.float32))
    if not progress_episodes:
        raise ValueError("No effect progress episodes could be built")
    return progress_episodes


def _load_effect_progress_latents(
    config: Config,
    candidate: str,
    seed: int,
    horizon_steps: int,
    force: bool = False,
) -> tuple[list[np.ndarray], list[np.ndarray], Path]:
    anchor_stride = int(
        config.get("reachability_distance.effect_anchor_stride", 2)
    )
    max_span = int(
        config.get("reachability_distance.effect_max_span", horizon_steps * 4 + 1)
    )
    cache_path = _effect_progress_cache_path(
        config, candidate, seed, horizon_steps, anchor_stride, max_span
    )
    if cache_path.exists() and not force:
        payload = torch.load(cache_path, map_location="cpu", weights_only=False)
        return (
            _valid_latent_episodes(payload["train_goals"]),
            _valid_latent_episodes(payload["validation_goals"]),
            cache_path,
        )

    representation_path = train_learned_interface_representation(
        config, candidate, seed=seed, force=False
    )
    device = default_device()
    encoder, checkpoint = _load_representation(representation_path, device)
    if checkpoint["encoder_type"] != "effect":
        raise ValueError(
            f"Expected effect representation for {candidate}, got "
            f"{checkpoint['encoder_type']}"
        )
    frame_norm = Standardizer.from_state_dict(checkpoint["frame_norm"])
    train_frames, validation_frames, _metadata = _load_phase6_train_episodes(config)
    train_goals = _build_effect_progress_episodes(
        train_frames,
        encoder,
        frame_norm,
        device,
        anchor_stride=anchor_stride,
        max_span=max_span,
    )
    validation_goals = _build_effect_progress_episodes(
        validation_frames,
        encoder,
        frame_norm,
        device,
        anchor_stride=anchor_stride,
        max_span=max_span,
    )
    payload = {
        "format_version": 1,
        "candidate": candidate,
        "representation_checkpoint": str(representation_path),
        "encoder_type": "effect",
        "horizon_steps": horizon_steps,
        "anchor_stride": anchor_stride,
        "max_span": max_span,
        "train_goals": train_goals,
        "validation_goals": validation_goals,
    }
    torch.save(payload, cache_path)
    return train_goals, validation_goals, cache_path


def _load_reachability_latents(
    config: Config,
    candidate: str,
    seed: int,
    horizon_steps: int,
    force: bool = False,
) -> tuple[list[np.ndarray], list[np.ndarray], Path, str]:
    representation_path = train_learned_interface_representation(
        config, candidate, seed=seed, force=False
    )
    representation = torch.load(
        representation_path, map_location="cpu", weights_only=False
    )
    encoder_type = str(representation["encoder_type"])
    if encoder_type == "effect":
        train_raw, validation_raw, encoded_path = _load_effect_progress_latents(
            config, candidate, seed, horizon_steps, force=force
        )
        return train_raw, validation_raw, encoded_path, encoder_type
    train_raw, validation_raw, encoded_path = _load_encoded_latents(
        config, candidate, seed=seed, force=force
    )
    return train_raw, validation_raw, encoded_path, encoder_type


def _fit_goal_norm(episodes: list[np.ndarray]) -> Standardizer:
    return Standardizer.fit(np.concatenate(episodes, axis=0))


def _normalize_latent_episodes(
    episodes: list[np.ndarray], goal_norm: Standardizer
) -> list[np.ndarray]:
    return [goal_norm.transform(episode).astype(np.float32) for episode in episodes]


def _sample_reachability_batch(
    episodes: list[np.ndarray],
    batch_size: int,
    horizon_steps: int,
    rng: np.random.Generator,
    forward_probability: float = 0.6,
    reverse_probability: float = 0.2,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if horizon_steps <= 0:
        raise ValueError("horizon_steps must be positive")
    latent_dim = int(episodes[0].shape[-1])
    starts = np.empty((batch_size, latent_dim), dtype=np.float32)
    goals = np.empty((batch_size, latent_dim), dtype=np.float32)
    targets = np.empty((batch_size,), dtype=np.float32)
    mode_cut_forward = forward_probability
    mode_cut_reverse = forward_probability + reverse_probability
    for row in range(batch_size):
        mode = float(rng.random())
        ep_i = int(rng.integers(0, len(episodes)))
        episode = episodes[ep_i]
        if mode < mode_cut_forward:
            t0 = int(rng.integers(0, len(episode) - 1))
            max_gap = len(episode) - t0 - 1
            gap = int(rng.integers(1, max_gap + 1))
            starts[row] = episode[t0]
            goals[row] = episode[t0 + gap]
            targets[row] = min(gap / horizon_steps, 1.0)
        elif mode < mode_cut_reverse:
            t1 = int(rng.integers(1, len(episode)))
            t0 = int(rng.integers(0, t1))
            starts[row] = episode[t1]
            goals[row] = episode[t0]
            targets[row] = 1.0
        else:
            other_i = int(rng.integers(0, len(episodes)))
            if len(episodes) > 1:
                while other_i == ep_i:
                    other_i = int(rng.integers(0, len(episodes)))
            starts[row] = episode[int(rng.integers(0, len(episode)))]
            other = episodes[other_i]
            goals[row] = other[int(rng.integers(0, len(other)))]
            targets[row] = 1.0
    return starts, goals, targets


def _rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    ranks[order] = np.arange(len(values), dtype=np.float64)
    sorted_values = values[order]
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        if end - start > 1:
            ranks[order[start:end]] = (start + end - 1) / 2.0
        start = end
    return ranks


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2:
        return float("nan")
    rx = _rankdata(np.asarray(x, dtype=np.float64))
    ry = _rankdata(np.asarray(y, dtype=np.float64))
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    denom = float(np.linalg.norm(rx) * np.linalg.norm(ry))
    if denom == 0.0:
        return float("nan")
    return float(np.dot(rx, ry) / denom)


def _binary_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    labels = labels.astype(bool)
    positives = scores[labels]
    negatives = scores[~labels]
    if len(positives) == 0 or len(negatives) == 0:
        return float("nan")
    return float(
        (
            (positives[:, None] > negatives[None, :]).mean()
            + 0.5 * (positives[:, None] == negatives[None, :]).mean()
        )
    )


@torch.inference_mode()
def _predict_distances(
    model: ReachabilityDistance,
    starts: np.ndarray,
    goals: np.ndarray,
    device: torch.device,
    batch_size: int = 4096,
) -> np.ndarray:
    chunks = []
    for offset in range(0, len(starts), batch_size):
        start = torch.as_tensor(starts[offset : offset + batch_size], device=device)
        goal = torch.as_tensor(goals[offset : offset + batch_size], device=device)
        chunks.append(model(start, goal).detach().cpu().numpy())
    return np.concatenate(chunks, axis=0).astype(np.float32)


def _sample_temporal_eval_pairs(
    episodes: list[np.ndarray],
    samples: int,
    horizon_steps: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    starts, goals, targets = _sample_reachability_batch(
        episodes,
        samples,
        horizon_steps,
        rng,
        forward_probability=1.0,
        reverse_probability=0.0,
    )
    return starts, goals, targets


def _sample_near_far_eval(
    episodes: list[np.ndarray],
    samples: int,
    horizon_steps: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    latent_dim = int(episodes[0].shape[-1])
    near_starts = np.empty((samples, latent_dim), dtype=np.float32)
    near_goals = np.empty((samples, latent_dim), dtype=np.float32)
    far_starts = np.empty((samples, latent_dim), dtype=np.float32)
    far_goals = np.empty((samples, latent_dim), dtype=np.float32)
    eligible = [ep for ep in episodes if len(ep) >= horizon_steps * 2 + 2]
    if not eligible:
        eligible = episodes
    for row in range(samples):
        episode = eligible[int(rng.integers(0, len(eligible)))]
        if len(episode) >= horizon_steps * 2 + 2:
            t0 = int(rng.integers(0, len(episode) - horizon_steps * 2 - 1))
            near_gap = int(rng.integers(1, horizon_steps + 1))
            far_gap = int(rng.integers(horizon_steps + 1, len(episode) - t0))
        else:
            t0 = 0
            near_gap = 1
            far_gap = len(episode) - 1
        near_starts[row] = episode[t0]
        near_goals[row] = episode[t0 + near_gap]
        far_starts[row] = episode[t0]
        far_goals[row] = episode[t0 + far_gap]
    return near_starts, near_goals, far_starts, far_goals


def _sample_shuffled_eval(
    episodes: list[np.ndarray],
    samples: int,
    horizon_steps: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    near_starts, near_goals, _targets = _sample_temporal_eval_pairs(
        episodes, samples, horizon_steps, rng
    )
    shuffled_goals = np.empty_like(near_goals)
    for row in range(samples):
        episode = episodes[int(rng.integers(0, len(episodes)))]
        shuffled_goals[row] = episode[int(rng.integers(0, len(episode)))]
    starts = np.concatenate([near_starts, near_starts], axis=0)
    goals = np.concatenate([near_goals, shuffled_goals], axis=0)
    labels = np.concatenate(
        [
            np.zeros(samples, dtype=np.int64),
            np.ones(samples, dtype=np.int64),
        ],
        axis=0,
    )
    return starts, goals, labels


def _sample_demo_decrease_eval(
    episodes: list[np.ndarray],
    samples: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    eligible = [ep for ep in episodes if len(ep) >= 3]
    if not eligible:
        raise ValueError("Need episodes with at least three frames for demo decrease eval")
    latent_dim = int(eligible[0].shape[-1])
    earlier = np.empty((samples, latent_dim), dtype=np.float32)
    later = np.empty((samples, latent_dim), dtype=np.float32)
    goals = np.empty((samples, latent_dim), dtype=np.float32)
    for row in range(samples):
        episode = eligible[int(rng.integers(0, len(eligible)))]
        t0 = int(rng.integers(0, len(episode) - 2))
        target = int(rng.integers(t0 + 2, len(episode)))
        earlier[row] = episode[t0]
        later[row] = episode[t0 + 1]
        goals[row] = episode[target]
    return earlier, later, goals


def _reachability_checkpoint_path(config: Config, candidate: str, seed: int) -> Path:
    return _artifact_dir(config, candidate, seed) / "d_phi.pt"


def train_reachability_distance(
    config: Config,
    candidate: str = "vae512_w2048_b1e6",
    seed: int = 0,
    epochs: int | None = None,
    batch_size: int | None = None,
    batches_per_epoch: int | None = None,
    hidden_dim: int | None = None,
    depth: int | None = None,
    lr: float | None = None,
    horizon_steps: int | None = None,
    force: bool = False,
) -> Path:
    artifact_dir = _artifact_dir(config, candidate, seed)
    checkpoint_path = artifact_dir / "d_phi.pt"
    if checkpoint_path.exists() and not force:
        console.print(f"Reachability distance exists: {checkpoint_path}")
        return checkpoint_path

    set_seed(seed)
    hidden_dim = int(hidden_dim or config.get("reachability_distance.hidden_dim", 512))
    depth = int(depth or config.get("reachability_distance.depth", 3))
    epochs = int(epochs or config.get("reachability_distance.epochs", 30))
    batch_size = int(batch_size or config.get("reachability_distance.batch_size", 512))
    batches_per_epoch = int(
        batches_per_epoch or config.get("reachability_distance.batches_per_epoch", 200)
    )
    lr = float(lr or config.get("reachability_distance.lr", 3e-4))
    horizon_steps = int(
        horizon_steps or config.get("reachability_distance.horizon_steps", 10)
    )
    train_raw, validation_raw, encoded_path, encoder_type = _load_reachability_latents(
        config, candidate, seed=seed, horizon_steps=horizon_steps, force=False
    )
    goal_norm = _fit_goal_norm(train_raw)
    train = _normalize_latent_episodes(train_raw, goal_norm)
    validation = _normalize_latent_episodes(validation_raw, goal_norm)
    latent_dim = int(train[0].shape[-1])

    device = default_device()
    model = ReachabilityDistance(latent_dim, hidden_dim, depth).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.MSELoss()
    rng = np.random.default_rng(seed + 8171)
    history: list[dict[str, float]] = []
    for epoch in trange(1, epochs + 1, desc=f"train D_phi {candidate}"):
        model.train()
        losses = []
        for _ in range(batches_per_epoch):
            starts, goals, targets = _sample_reachability_batch(
                train, batch_size, horizon_steps, rng
            )
            start_t = torch.as_tensor(starts, device=device)
            goal_t = torch.as_tensor(goals, device=device)
            target_t = torch.as_tensor(targets, device=device)
            pred = model(start_t, goal_t)
            loss = loss_fn(pred, target_t)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            losses.append(float(loss.detach().cpu()))
        if epoch == 1 or epoch == epochs or epoch % max(1, epochs // 5) == 0:
            metrics = evaluate_reachability_distance_model(
                model,
                validation,
                horizon_steps=horizon_steps,
                seed=seed + epoch,
                samples=int(config.get("reachability_distance.eval_samples", 4096)),
                device=device,
            )
            metrics["train_loss"] = float(np.mean(losses))
            metrics["epoch"] = float(epoch)
            history.append(metrics)

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "candidate": candidate,
        "seed": seed,
        "latent_dim": latent_dim,
        "hidden_dim": hidden_dim,
        "depth": depth,
        "horizon_steps": horizon_steps,
        "goal_norm": goal_norm.state_dict(),
        "encoded_path": str(encoded_path),
        "encoder_type": encoder_type,
        "history": history,
    }
    torch.save(checkpoint, checkpoint_path)
    metrics_payload = {
        "checkpoint": str(checkpoint_path),
        "candidate": candidate,
        "seed": seed,
        "encoded_path": str(encoded_path),
        "encoder_type": encoder_type,
        "history": history,
        "final": history[-1] if history else {},
    }
    write_json(artifact_dir / "metrics.json", metrics_payload)
    console.print(f"Wrote reachability distance: {checkpoint_path}")
    return checkpoint_path


def _load_reachability_checkpoint(
    checkpoint_path: Path, device: torch.device
) -> tuple[ReachabilityDistance, dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = ReachabilityDistance(
        int(checkpoint["latent_dim"]),
        int(checkpoint["hidden_dim"]),
        int(checkpoint["depth"]),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint


def load_reachability_distance(
    checkpoint_path: Path, device: torch.device
) -> tuple[ReachabilityDistance, Standardizer, dict[str, Any]]:
    model, checkpoint = _load_reachability_checkpoint(checkpoint_path, device)
    goal_norm = Standardizer.from_state_dict(checkpoint["goal_norm"])
    return model, goal_norm, checkpoint


def evaluate_reachability_distance_model(
    model: ReachabilityDistance,
    episodes: list[np.ndarray],
    horizon_steps: int,
    seed: int,
    samples: int,
    device: torch.device,
) -> dict[str, float]:
    rng = np.random.default_rng(seed + 9137)
    temporal_starts, temporal_goals, temporal_targets = _sample_temporal_eval_pairs(
        episodes, samples, horizon_steps, rng
    )
    temporal_pred = _predict_distances(model, temporal_starts, temporal_goals, device)

    near_starts, near_goals, far_starts, far_goals = _sample_near_far_eval(
        episodes, samples, horizon_steps, rng
    )
    near_pred = _predict_distances(model, near_starts, near_goals, device)
    far_pred = _predict_distances(model, far_starts, far_goals, device)

    shuffled_starts, shuffled_goals, shuffled_labels = _sample_shuffled_eval(
        episodes, samples, horizon_steps, rng
    )
    shuffled_pred = _predict_distances(
        model, shuffled_starts, shuffled_goals, device
    )

    earlier, later, final_goals = _sample_demo_decrease_eval(
        episodes, samples, rng
    )
    earlier_pred = _predict_distances(model, earlier, final_goals, device)
    later_pred = _predict_distances(model, later, final_goals, device)

    return {
        "temporal_mse": float(np.mean((temporal_pred - temporal_targets) ** 2)),
        "temporal_spearman": _spearman(temporal_targets, temporal_pred),
        "near_mean": float(near_pred.mean()),
        "far_mean": float(far_pred.mean()),
        "near_far_accuracy": float((near_pred < far_pred).mean()),
        "shuffled_auc": _binary_auc(shuffled_pred, shuffled_labels),
        "demo_decrease_accuracy": float((later_pred < earlier_pred).mean()),
    }


def evaluate_reachability_distance(
    config: Config,
    candidate: str = "vae512_w2048_b1e6",
    seed: int = 0,
    checkpoint_path: Path | None = None,
    samples: int | None = None,
    output_path: Path | None = None,
    force: bool = False,
) -> Path:
    checkpoint_path = checkpoint_path or _reachability_checkpoint_path(
        config, candidate, seed
    )
    output_path = output_path or _result_dir(config, candidate, seed) / "eval.json"
    if output_path.exists() and not force:
        console.print(f"Reachability eval exists: {output_path}")
        return output_path
    if not checkpoint_path.exists():
        checkpoint_path = train_reachability_distance(
            config, candidate=candidate, seed=seed, force=False
        )
    device = default_device()
    model, checkpoint = _load_reachability_checkpoint(checkpoint_path, device)
    train_raw, validation_raw, encoded_path, encoder_type = _load_reachability_latents(
        config,
        candidate,
        seed=seed,
        horizon_steps=int(checkpoint["horizon_steps"]),
        force=False,
    )
    goal_norm = Standardizer.from_state_dict(checkpoint["goal_norm"])
    validation = _normalize_latent_episodes(validation_raw, goal_norm)
    samples = int(samples or config.get("reachability_distance.eval_samples", 4096))
    metrics = evaluate_reachability_distance_model(
        model,
        validation,
        horizon_steps=int(checkpoint["horizon_steps"]),
        seed=seed,
        samples=samples,
        device=device,
    )
    payload: dict[str, Any] = {
        "checkpoint": str(checkpoint_path),
        "candidate": candidate,
        "seed": seed,
        "samples": samples,
        "encoded_path": str(encoded_path),
        "encoder_type": encoder_type,
        "train_episodes": len(train_raw),
        "validation_episodes": len(validation_raw),
        **metrics,
    }
    write_json(output_path, payload)
    console.print(payload)
    return output_path
