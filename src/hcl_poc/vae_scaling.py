from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import gymnasium as gym
import h5py
import numpy as np
import torch
from rich.console import Console
from torch import nn
from torch.utils.data import DataLoader
from tqdm import trange

from hcl_poc.config import Config
from hcl_poc.flow import flow_matching_loss, sample_flow
from hcl_poc.incremental import (
    _load_phase6_train_episodes,
    _phase4_dino_from_config,
    _phase4_frame_inputs,
    _phase7_obs_state_tensor,
    _rl_backend,
    _runtime_metadata,
    _wilson_interval,
)
from hcl_poc.learned_interface import (
    _HeldGoalDataset,
    _load_hierarchy,
    _load_representation,
    _low_condition_array,
    evaluate_learned_interface_hierarchy,
    prepare_learned_interface_episodes,
    train_learned_interface_hierarchy,
    train_learned_interface_representation,
)
from hcl_poc.models import FlowModel, MLP
from hcl_poc.rl import _rl_paths, load_ppo_agent
from hcl_poc.utils import (
    Standardizer,
    Timer,
    default_device,
    ensure_dir,
    set_seed,
    write_json,
)

console = Console()

VAE_SCALING_BUDGETS = (50, 100, 200, 500, 1000, 1800)
VAE_SCALING_SEEDS = (0, 1, 2)
VAE_CANDIDATE = "vae512_w2048_b1e6"
DEPLOYABLE_METHODS = (
    "deterministic_hierarchy",
    "flow_hierarchy",
    "flat_latent_deterministic",
    "flat_latent_flow",
    "flat_observation_deterministic",
    "flat_observation_flow",
)
ALL_METHODS = (*DEPLOYABLE_METHODS, "oracle_hierarchy")


def vae_scaling_config(config: Config, n_trajectories: int) -> Config:
    if n_trajectories not in VAE_SCALING_BUDGETS:
        raise ValueError(
            f"VAE scaling budget must be one of {VAE_SCALING_BUDGETS}"
        )
    raw = copy.deepcopy(config.raw)
    raw["paths"]["incremental_artifact_dir"] = str(
        config.path_value("paths.incremental_artifact_dir")
        / "vae512_scaling"
        / f"n{n_trajectories}"
    )
    raw["paths"]["incremental_results_dir"] = str(
        config.path_value("paths.incremental_results_dir")
        / "vae512_scaling"
        / f"n{n_trajectories}"
    )
    raw["incremental"]["phase4"]["train_episodes"] = n_trajectories
    raw["incremental"]["phase6"]["train_episodes"] = n_trajectories
    raw["learned_interface"]["evaluation"]["seed_start"] = int(
        config.get("vae_scaling.eval_seed_start", 2_200_000)
    )
    return Config(raw=raw, path=config.path)


def _point_artifact_dir(config: Config, seed: int) -> Path:
    return ensure_dir(
        config.path_value("paths.incremental_artifact_dir") / f"seed{seed}"
    )


def _point_result_dir(config: Config, seed: int) -> Path:
    return ensure_dir(
        config.path_value("paths.incremental_results_dir") / f"seed{seed}"
    )


def write_vae_scaling_manifest(
    config: Config,
    n_trajectories: int,
    force: bool = False,
) -> Path:
    point_config = vae_scaling_config(config, n_trajectories)
    path = ensure_dir(
        point_config.path_value("paths.incremental_artifact_dir")
    ) / "data_manifest.json"
    if path.exists() and not force:
        return path
    dataset_path = Path(config.get("incremental.phase4.prepared_path"))
    validation_count = int(
        config.get("incremental.phase4.validation_episodes", 200)
    )
    with h5py.File(dataset_path, "r") as h5:
        keys = sorted(key for key in h5 if key.startswith("episode_"))
        train_keys = keys[:n_trajectories]
        validation_keys = keys[-validation_count:]
        train_lengths = [int(len(h5[key]["actions"])) for key in train_keys]
        validation_lengths = [
            int(len(h5[key]["actions"])) for key in validation_keys
        ]
    if set(train_keys) & set(validation_keys):
        raise ValueError("VAE scaling train/validation trajectory overlap")
    fingerprint_source = json.dumps(
        {
            "dataset": str(dataset_path.resolve()),
            "train": list(zip(train_keys, train_lengths, strict=True)),
            "validation": list(
                zip(validation_keys, validation_lengths, strict=True)
            ),
        },
        sort_keys=True,
    ).encode()
    payload = {
        "experiment": "vae512_sample_efficiency",
        "dataset": str(dataset_path),
        "selection": "nested prefix for train; fixed final 200 for validation",
        "n_trajectories": n_trajectories,
        "train_keys": train_keys,
        "validation_keys": validation_keys,
        "train_transitions": int(sum(train_lengths)),
        "validation_transitions": int(sum(validation_lengths)),
        "equivalent_behavior_seconds": sum(train_lengths)
        / float(config.get("control_freq", 20)),
        "sha256": hashlib.sha256(fingerprint_source).hexdigest(),
        "metadata": _runtime_metadata(config),
    }
    write_json(path, payload)
    return path


def validate_nested_vae_scaling_manifests(config: Config) -> dict[str, Any]:
    manifests = []
    for budget in VAE_SCALING_BUDGETS:
        path = write_vae_scaling_manifest(config, budget)
        with path.open() as stream:
            manifests.append(json.load(stream))
    validation = manifests[0]["validation_keys"]
    previous: list[str] = []
    for manifest in manifests:
        train = manifest["train_keys"]
        if train[: len(previous)] != previous:
            raise ValueError("VAE scaling trajectory budgets are not nested")
        if manifest["validation_keys"] != validation:
            raise ValueError("VAE scaling validation split changed across budgets")
        previous = train
    return {
        "budgets": list(VAE_SCALING_BUDGETS),
        "nested": True,
        "fixed_validation": True,
        "manifest_sha256": [manifest["sha256"] for manifest in manifests],
    }


def _load_point_episodes(
    config: Config,
    seed: int,
) -> tuple[
    list[dict[str, np.ndarray]],
    list[dict[str, np.ndarray]],
    dict[str, Any],
    dict[str, Any],
]:
    encoded_path = prepare_learned_interface_episodes(
        config, VAE_CANDIDATE, seed, force=False
    )
    encoded = torch.load(encoded_path, map_location="cpu", weights_only=False)
    train_frames, validation_frames, metadata = _load_phase6_train_episodes(
        config
    )

    def combine(
        frame_episodes: list[dict[str, np.ndarray]],
        goals: list[np.ndarray],
    ) -> list[dict[str, np.ndarray]]:
        if len(frame_episodes) != len(goals):
            raise ValueError("VAE scaling encoded episode count mismatch")
        return [
            {
                "frames": frame_episode["frames"],
                "latents": latent,
                "actions": frame_episode["actions"],
            }
            for frame_episode, latent in zip(
                frame_episodes, goals, strict=True
            )
        ]

    return (
        combine(train_frames, encoded["train_goals"]),
        combine(validation_frames, encoded["validation_goals"]),
        metadata,
        encoded,
    )


class _FlatDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        episodes: list[dict[str, np.ndarray]],
        input_key: str,
        input_norm: Standardizer,
        action_norm: Standardizer,
        length: int,
    ) -> None:
        self.episodes = episodes
        self.input_key = input_key
        self.input_norm = input_norm
        self.action_norm = action_norm
        self.length = length
        self.zero_action = action_norm.transform(
            np.zeros((1, 3), dtype=np.float32)
        )[0]

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, _index: int) -> tuple[torch.Tensor, torch.Tensor]:
        episode = self.episodes[np.random.randint(0, len(self.episodes))]
        t = int(np.random.randint(0, len(episode["actions"])))
        previous = (
            self.action_norm.transform(episode["actions"][t - 1 : t])[0]
            if t > 0
            else self.zero_action
        )
        current = self.input_norm.transform(
            episode[self.input_key][t : t + 1]
        )[0]
        target = self.action_norm.transform(
            episode["actions"][t : t + 1]
        )[0]
        return (
            torch.from_numpy(
                np.concatenate([current, previous]).astype(np.float32)
            ),
            torch.from_numpy(target.astype(np.float32)),
        )


def _flat_validation_arrays(
    episodes: list[dict[str, np.ndarray]],
    input_key: str,
    input_norm: Standardizer,
    action_norm: Standardizer,
    samples: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    zero_action = action_norm.transform(
        np.zeros((1, 3), dtype=np.float32)
    )[0]
    conditions = []
    actions = []
    for _ in range(samples):
        episode = episodes[int(rng.integers(0, len(episodes)))]
        t = int(rng.integers(0, len(episode["actions"])))
        previous = (
            action_norm.transform(episode["actions"][t - 1 : t])[0]
            if t > 0
            else zero_action
        )
        current = input_norm.transform(episode[input_key][t : t + 1])[0]
        conditions.append(np.concatenate([current, previous]))
        actions.append(episode["actions"][t])
    return (
        np.asarray(conditions, dtype=np.float32),
        np.asarray(actions, dtype=np.float32),
    )


def train_vae_scaling_flat_policy(
    config: Config,
    n_trajectories: int,
    representation: str,
    policy_type: str,
    seed: int,
    force: bool = False,
) -> Path:
    if representation not in {"latent", "observation"}:
        raise ValueError(f"Unknown flat representation: {representation}")
    if policy_type not in {"deterministic", "flow"}:
        raise ValueError(f"Unknown flat policy type: {policy_type}")
    point_config = vae_scaling_config(config, n_trajectories)
    artifact_dir = ensure_dir(
        _point_artifact_dir(point_config, seed)
        / f"flat_{representation}_{policy_type}"
    )
    checkpoint_path = artifact_dir / "policy.pt"
    if checkpoint_path.exists() and not force:
        return checkpoint_path
    set_seed(seed)
    train, validation, metadata, encoded = _load_point_episodes(
        point_config, seed
    )
    representation_checkpoint = torch.load(
        encoded["representation_checkpoint"],
        map_location="cpu",
        weights_only=False,
    )
    frame_norm = Standardizer.from_state_dict(
        representation_checkpoint["frame_norm"]
    )
    latent_norm = Standardizer.fit(
        np.concatenate([episode["latents"] for episode in train])
    )
    action_norm = Standardizer.fit(
        np.concatenate([episode["actions"] for episode in train])
    )
    input_key = "latents" if representation == "latent" else "frames"
    input_norm = latent_norm if representation == "latent" else frame_norm
    input_dim = train[0][input_key].shape[-1]
    hidden_dim = int(config.get("vae_scaling.policy.hidden_dim", 512))
    batch_size = int(config.get("vae_scaling.policy.batch_size", 512))
    batches_per_epoch = int(
        config.get("vae_scaling.policy.batches_per_epoch", 200)
    )
    epochs = int(config.get("vae_scaling.policy.epochs", 60))
    learning_rate = float(config.get("vae_scaling.policy.lr", 3e-4))
    device = default_device()
    model: nn.Module = (
        MLP(input_dim + 3, 3, hidden_dim, depth=4)
        if policy_type == "deterministic"
        else FlowModel(3, input_dim + 3, hidden_dim)
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    loader = DataLoader(
        _FlatDataset(
            train,
            input_key,
            input_norm,
            action_norm,
            batch_size * batches_per_epoch,
        ),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    validation_conditions, validation_actions = _flat_validation_arrays(
        validation,
        input_key,
        input_norm,
        action_norm,
        int(config.get("vae_scaling.policy.validation_samples", 5000)),
        seed + 4100,
    )
    validation_noise = np.random.default_rng(seed + 4200).standard_normal(
        (len(validation_conditions), 3)
    ).astype(np.float32)
    best_state: dict[str, torch.Tensor] | None = None
    best_mae = float("inf")
    best_epoch = 0
    history = []
    timer = Timer()
    for epoch in trange(
        1,
        epochs + 1,
        desc=f"train VAE scaling flat {representation} {policy_type}",
    ):
        model.train()
        train_loss = 0.0
        for condition, target in loader:
            condition = condition.to(device, non_blocking=True).float()
            target = target.to(device, non_blocking=True).float()
            loss = (
                torch.mean((model(condition) - target) ** 2)
                if policy_type == "deterministic"
                else flow_matching_loss(model, target, condition)
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            train_loss += float(loss.detach().cpu())
        model.eval()
        with torch.inference_mode():
            condition_t = torch.from_numpy(validation_conditions).to(device)
            normalized_prediction = (
                model(condition_t)
                if policy_type == "deterministic"
                else sample_flow(
                    model,
                    condition_t,
                    steps=int(config.get("vae_scaling.flow_steps", 24)),
                    sample_dim=3,
                    initial_noise=torch.from_numpy(validation_noise).to(device),
                )
            )
            prediction = action_norm.inverse(
                normalized_prediction.cpu().numpy()
            )
        action_mae = float(np.mean(np.abs(prediction - validation_actions)))
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss / batches_per_epoch,
                "validation_action_mae": action_mae,
            }
        )
        if action_mae < best_mae:
            best_mae = action_mae
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
    if best_state is None:
        raise RuntimeError("VAE scaling flat policy produced no checkpoint")
    payload = {
        "experiment": "vae512_sample_efficiency",
        "method": f"flat_{representation}_{policy_type}",
        "n_trajectories": n_trajectories,
        "seed": seed,
        "representation": representation,
        "policy_type": policy_type,
        "input_dim": int(input_dim),
        "condition_dim": int(input_dim + 3),
        "hidden_dim": hidden_dim,
        "action_dim": 3,
        "model": best_state,
        "frame_norm": frame_norm.state_dict(),
        "latent_norm": latent_norm.state_dict(),
        "action_norm": action_norm.state_dict(),
        "representation_checkpoint": encoded["representation_checkpoint"],
        "best_epoch": best_epoch,
        "validation_action_mae": best_mae,
        "history": history,
        "parameter_count": int(sum(p.numel() for p in model.parameters())),
        "optimizer": "AdamW",
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "batches_per_epoch": batches_per_epoch,
        "epochs": epochs,
        "flow_steps": int(config.get("vae_scaling.flow_steps", 24)),
        "elapsed_s": timer.elapsed(),
        "data": metadata,
        "metadata": _runtime_metadata(config),
    }
    torch.save(payload, checkpoint_path)
    write_json(
        artifact_dir / "policy_metrics.json",
        {
            key: value
            for key, value in payload.items()
            if key not in {"model", "frame_norm", "latent_norm", "action_norm"}
        },
    )
    return checkpoint_path


def _high_flow_validation_samples(
    episodes: list[dict[str, np.ndarray]],
    frame_norm: Standardizer,
    goal_norm: Standardizer,
    action_norm: Standardizer,
    horizon_steps: int,
    samples: int,
    seed: int,
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    zero_action = action_norm.transform(
        np.zeros((1, 3), dtype=np.float32)
    )[0]
    conditions = []
    goals = []
    current_frames = []
    previous_actions = []
    target_actions = []
    remaining_values = []
    for _ in range(samples):
        episode = episodes[int(rng.integers(0, len(episodes)))]
        base = int(rng.integers(0, len(episode["actions"]) - horizon_steps))
        offset = int(rng.integers(0, horizon_steps))
        current = base + offset
        high_previous = (
            action_norm.transform(episode["actions"][base - 1 : base])[0]
            if base > 0
            else zero_action
        )
        previous = (
            action_norm.transform(
                episode["actions"][current - 1 : current]
            )[0]
            if current > 0
            else zero_action
        )
        conditions.append(
            np.concatenate(
                [
                    frame_norm.transform(episode["frames"][base : base + 1])[0],
                    high_previous,
                ]
            )
        )
        goals.append(
            goal_norm.transform(
                episode["latents"][
                    base + horizon_steps : base + horizon_steps + 1
                ]
            )[0]
        )
        current_frames.append(
            frame_norm.transform(
                episode["frames"][current : current + 1]
            )[0]
        )
        previous_actions.append(previous)
        target_actions.append(episode["actions"][current])
        remaining_values.append((horizon_steps - offset) / horizon_steps)
    return {
        "conditions": np.asarray(conditions, dtype=np.float32),
        "goals": np.asarray(goals, dtype=np.float32),
        "current_frames": np.asarray(current_frames, dtype=np.float32),
        "previous_actions": np.asarray(previous_actions, dtype=np.float32),
        "target_actions": np.asarray(target_actions, dtype=np.float32),
        "remaining": np.asarray(remaining_values, dtype=np.float32)[:, None],
    }


def train_vae_scaling_flow_high_level(
    config: Config,
    n_trajectories: int,
    seed: int,
    force: bool = False,
) -> Path:
    point_config = vae_scaling_config(config, n_trajectories)
    artifact_dir = ensure_dir(
        _point_artifact_dir(point_config, seed) / "flow_hierarchy"
    )
    checkpoint_path = artifact_dir / "high_flow.pt"
    if checkpoint_path.exists() and not force:
        return checkpoint_path
    set_seed(seed)
    hierarchy_path = train_learned_interface_hierarchy(
        point_config, VAE_CANDIDATE, seed, force=False
    )
    hierarchy = torch.load(
        hierarchy_path, map_location="cpu", weights_only=False
    )
    device = default_device()
    _unused_high, low_model = _load_hierarchy(hierarchy, device)
    train, validation, metadata, encoded = _load_point_episodes(
        point_config, seed
    )
    frame_norm = Standardizer.from_state_dict(hierarchy["frame_norm"])
    goal_norm = Standardizer.from_state_dict(hierarchy["goal_norm"])
    action_norm = Standardizer.from_state_dict(hierarchy["action_norm"])
    horizon_steps = int(hierarchy["horizon_steps"])
    hidden_dim = int(config.get("vae_scaling.policy.hidden_dim", 512))
    batch_size = int(config.get("vae_scaling.policy.batch_size", 512))
    batches_per_epoch = int(
        config.get("vae_scaling.policy.batches_per_epoch", 200)
    )
    epochs = int(config.get("vae_scaling.policy.epochs", 60))
    learning_rate = float(config.get("vae_scaling.policy.lr", 3e-4))
    model = FlowModel(
        int(hierarchy["goal_dim"]),
        int(hierarchy["frame_dim"]) + 3,
        hidden_dim,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    loader = DataLoader(
        _HeldGoalDataset(
            [
                {
                    "frames": episode["frames"],
                    "goals": episode["latents"],
                    "actions": episode["actions"],
                }
                for episode in train
            ],
            frame_norm,
            goal_norm,
            action_norm,
            horizon_steps,
            "high",
            batch_size * batches_per_epoch,
            "concat",
        ),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    validation = _high_flow_validation_samples(
        validation,
        frame_norm,
        goal_norm,
        action_norm,
        horizon_steps,
        int(config.get("vae_scaling.policy.validation_samples", 5000)),
        seed + 4300,
    )
    validation_noise = np.random.default_rng(seed + 4400).standard_normal(
        validation["goals"].shape
    ).astype(np.float32)
    best_state: dict[str, torch.Tensor] | None = None
    best_action_mae = float("inf")
    best_epoch = 0
    history = []
    timer = Timer()
    for epoch in trange(1, epochs + 1, desc="train VAE scaling flow high"):
        model.train()
        train_loss = 0.0
        for condition, goal in loader:
            condition = condition.to(device, non_blocking=True).float()
            goal = goal.to(device, non_blocking=True).float()
            loss = flow_matching_loss(model, goal, condition)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            train_loss += float(loss.detach().cpu())
        model.eval()
        with torch.inference_mode():
            prediction = sample_flow(
                model,
                torch.from_numpy(validation["conditions"]).to(device),
                steps=int(config.get("vae_scaling.flow_steps", 24)),
                sample_dim=int(hierarchy["goal_dim"]),
                initial_noise=torch.from_numpy(validation_noise).to(device),
            ).cpu().numpy()
        goal_l2 = float(
            np.mean(
                np.linalg.norm(
                    prediction - validation["goals"], axis=-1
                )
            )
        )
        predicted_condition = _low_condition_array(
            validation["current_frames"],
            np.empty_like(prediction),
            prediction,
            validation["previous_actions"],
            validation["remaining"],
            "concat",
        )
        oracle_condition = _low_condition_array(
            validation["current_frames"],
            np.empty_like(validation["goals"]),
            validation["goals"],
            validation["previous_actions"],
            validation["remaining"],
            "concat",
        )
        with torch.inference_mode():
            predicted_action = action_norm.inverse(
                low_model(
                    torch.from_numpy(predicted_condition).to(device)
                )
                .cpu()
                .numpy()
            )
            oracle_action = action_norm.inverse(
                low_model(torch.from_numpy(oracle_condition).to(device))
                .cpu()
                .numpy()
            )
        predicted_action_mae = float(
            np.mean(
                np.abs(predicted_action - validation["target_actions"])
            )
        )
        induced_action_l2 = float(
            np.mean(
                np.linalg.norm(
                    predicted_action - oracle_action, axis=-1
                )
            )
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss / batches_per_epoch,
                "validation_normalized_goal_l2": goal_l2,
                "validation_predicted_action_mae": predicted_action_mae,
                "validation_prediction_induced_action_l2": induced_action_l2,
            }
        )
        if predicted_action_mae < best_action_mae:
            best_action_mae = predicted_action_mae
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
    if best_state is None:
        raise RuntimeError("VAE scaling flow high level produced no checkpoint")
    payload = {
        "experiment": "vae512_sample_efficiency",
        "method": "flow_hierarchy",
        "n_trajectories": n_trajectories,
        "seed": seed,
        "model": best_state,
        "condition_dim": int(hierarchy["frame_dim"]) + 3,
        "goal_dim": int(hierarchy["goal_dim"]),
        "hidden_dim": hidden_dim,
        "hierarchy_checkpoint": str(hierarchy_path),
        "representation_checkpoint": encoded["representation_checkpoint"],
        "best_epoch": best_epoch,
        "validation_predicted_action_mae": best_action_mae,
        "validation_metrics": history[best_epoch - 1],
        "history": history,
        "parameter_count": int(sum(p.numel() for p in model.parameters())),
        "optimizer": "AdamW",
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "batches_per_epoch": batches_per_epoch,
        "epochs": epochs,
        "flow_steps": int(config.get("vae_scaling.flow_steps", 24)),
        "elapsed_s": timer.elapsed(),
        "data": metadata,
        "metadata": _runtime_metadata(config),
    }
    torch.save(payload, checkpoint_path)
    write_json(
        artifact_dir / "high_flow_metrics.json",
        {key: value for key, value in payload.items() if key != "model"},
    )
    return checkpoint_path


def _deterministic_noise(
    training_seed: int,
    environment_seeds: list[int],
    decision_index: int,
    dimension: int,
) -> np.ndarray:
    rows = []
    for environment_seed in environment_seeds:
        sequence = np.random.SeedSequence(
            [training_seed, environment_seed, decision_index, dimension]
        )
        rows.append(
            np.random.default_rng(sequence).standard_normal(dimension)
        )
    return np.asarray(rows, dtype=np.float32)


def train_vae_scaling_point(
    config: Config,
    n_trajectories: int,
    seed: int,
    force: bool = False,
) -> dict[str, Path]:
    if seed not in VAE_SCALING_SEEDS:
        raise ValueError(f"VAE scaling seed must be one of {VAE_SCALING_SEEDS}")
    point_config = vae_scaling_config(config, n_trajectories)
    manifest = write_vae_scaling_manifest(config, n_trajectories)
    representation = train_learned_interface_representation(
        point_config, VAE_CANDIDATE, seed, force=force
    )
    encoded = prepare_learned_interface_episodes(
        point_config, VAE_CANDIDATE, seed, force=force
    )
    hierarchy = train_learned_interface_hierarchy(
        point_config, VAE_CANDIDATE, seed, force=force
    )
    paths = {
        "manifest": manifest,
        "representation": representation,
        "encoded": encoded,
        "deterministic_hierarchy": hierarchy,
        "flow_hierarchy": train_vae_scaling_flow_high_level(
            config, n_trajectories, seed, force=force
        ),
    }
    for representation_name in ("latent", "observation"):
        for policy_type in ("deterministic", "flow"):
            paths[f"flat_{representation_name}_{policy_type}"] = (
                train_vae_scaling_flat_policy(
                    config,
                    n_trajectories,
                    representation_name,
                    policy_type,
                    seed,
                    force=force,
                )
            )
    write_json(
        _point_artifact_dir(point_config, seed) / "training_manifest.json",
        {
            "experiment": "vae512_sample_efficiency",
            "n_trajectories": n_trajectories,
            "seed": seed,
            "artifacts": {key: str(value) for key, value in paths.items()},
            "metadata": _runtime_metadata(config),
        },
    )
    return paths


def _load_flat_policy(
    checkpoint: dict[str, Any], device: torch.device
) -> nn.Module:
    model: nn.Module = (
        MLP(
            int(checkpoint["condition_dim"]),
            int(checkpoint["action_dim"]),
            int(checkpoint["hidden_dim"]),
            depth=4,
        )
        if checkpoint["policy_type"] == "deterministic"
        else FlowModel(
            int(checkpoint["action_dim"]),
            int(checkpoint["condition_dim"]),
            int(checkpoint["hidden_dim"]),
        )
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model


def _rollout_payload(
    config: Config,
    method: str,
    n_trajectories: int,
    seed: int,
    eval_seed_start: int,
    successes: list[float],
    final_rewards: list[float],
    max_rewards: list[float],
    teacher_maes: list[float],
    checkpoint_path: Path,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    episodes = len(successes)
    success = float(np.mean(successes))
    return {
        "experiment": "vae512_sample_efficiency",
        "method": method,
        "n_trajectories": n_trajectories,
        "training_seed": seed,
        "episodes": episodes,
        "eval_seed_start": eval_seed_start,
        "success": success,
        "success_wilson_95": _wilson_interval(success, episodes),
        "final_reward": float(np.mean(final_rewards)),
        "max_reward": float(np.mean(max_rewards)),
        "teacher_action_mae": float(np.mean(teacher_maes)),
        "episode_success": successes,
        "episode_final_reward": final_rewards,
        "episode_max_reward": max_rewards,
        "checkpoint": str(checkpoint_path),
        **(extra or {}),
        "metadata": _runtime_metadata(config),
    }


@torch.inference_mode()
def evaluate_vae_scaling_flat_policy(
    config: Config,
    n_trajectories: int,
    representation: str,
    policy_type: str,
    seed: int,
    episodes: int | None = None,
    force: bool = False,
) -> Path:
    point_config = vae_scaling_config(config, n_trajectories)
    eval_episodes = int(
        episodes or config.get("vae_scaling.deployable_eval_episodes", 500)
    )
    method = f"flat_{representation}_{policy_type}"
    output_path = (
        ensure_dir(_point_result_dir(point_config, seed) / method)
        / f"eval_{eval_episodes}.json"
    )
    if output_path.exists() and not force:
        return output_path
    checkpoint_path = train_vae_scaling_flat_policy(
        config,
        n_trajectories,
        representation,
        policy_type,
        seed,
        force=False,
    )
    device = default_device()
    checkpoint = torch.load(
        checkpoint_path, map_location=device, weights_only=False
    )
    model = _load_flat_policy(checkpoint, device)
    frame_norm = Standardizer.from_state_dict(checkpoint["frame_norm"])
    latent_norm = Standardizer.from_state_dict(checkpoint["latent_norm"])
    action_norm = Standardizer.from_state_dict(checkpoint["action_norm"])
    encoder = None
    if representation == "latent":
        encoder, _encoder_checkpoint = _load_representation(
            Path(checkpoint["representation_checkpoint"]), device
        )
    dino = _phase4_dino_from_config(config, device)
    teacher = load_ppo_agent(_rl_paths(config).best, device)
    max_steps = int(config.get("env_max_episode_steps", 100))
    num_envs_max = min(
        int(config.get("vae_scaling.eval_num_envs", 64)), eval_episodes
    )
    eval_seed_start = int(config.get("vae_scaling.eval_seed_start", 2_200_000))
    successes: list[float] = []
    final_rewards: list[float] = []
    max_rewards: list[float] = []
    teacher_maes: list[float] = []
    timer = Timer()
    progress = trange(eval_episodes, desc=f"eval {method} n={n_trajectories}")
    for batch_start in range(0, eval_episodes, num_envs_max):
        num_envs = min(num_envs_max, eval_episodes - batch_start)
        reset_seeds = [
            eval_seed_start + batch_start + index for index in range(num_envs)
        ]
        env = gym.make(
            config.get("env_id"),
            obs_mode="rgb+state",
            control_mode=config.get("control_mode"),
            reward_mode="normalized_dense",
            render_mode=None,
            sim_backend=_rl_backend(config),
            num_envs=num_envs,
            reconfiguration_freq=config.get("rl.eval_reconfiguration_freq", 1),
        )
        action_low_np = np.asarray(env.action_space.low, dtype=np.float32)
        action_high_np = np.asarray(env.action_space.high, dtype=np.float32)
        if action_low_np.ndim == 2:
            action_low_np = action_low_np[0]
            action_high_np = action_high_np[0]
        action_low = torch.as_tensor(action_low_np, device=device)
        action_high = torch.as_tensor(action_high_np, device=device)
        previous_action = np.repeat(
            action_norm.transform(np.zeros((1, 3), dtype=np.float32)),
            num_envs,
            axis=0,
        )
        active = np.ones(num_envs, dtype=bool)
        success_once = np.zeros(num_envs, dtype=bool)
        batch_final = np.zeros(num_envs, dtype=np.float32)
        batch_max = np.full(num_envs, -np.inf, dtype=np.float32)
        try:
            obs, _info = env.reset(seed=reset_seeds)
            for step in range(max_steps):
                if not np.any(active):
                    break
                frames = _phase4_frame_inputs(
                    obs, dino, int(config.get("dino.batch_size", 64))
                )
                if representation == "observation":
                    current = frame_norm.transform(frames)
                else:
                    if encoder is None:
                        raise RuntimeError("Missing VAE encoder for latent policy")
                    latent = (
                        encoder(
                            torch.from_numpy(frame_norm.transform(frames))
                            .to(device)
                            .float()
                        )
                        .cpu()
                        .numpy()
                    )
                    current = latent_norm.transform(latent)
                condition = np.concatenate(
                    [current, previous_action], axis=-1
                ).astype(np.float32)
                condition_t = torch.from_numpy(condition).to(device)
                normalized_action = (
                    model(condition_t)
                    if policy_type == "deterministic"
                    else sample_flow(
                        model,
                        condition_t,
                        steps=int(checkpoint["flow_steps"]),
                        sample_dim=3,
                        initial_noise=torch.from_numpy(
                            _deterministic_noise(seed, reset_seeds, step, 3)
                        ).to(device),
                    )
                )
                raw_action = action_norm.inverse(
                    normalized_action.cpu().numpy()
                )
                teacher_action = (
                    torch.clamp(
                        teacher.actor_mean(_phase7_obs_state_tensor(obs, device)),
                        action_low,
                        action_high,
                    )
                    .cpu()
                    .numpy()
                )
                teacher_maes.extend(
                    np.mean(
                        np.abs(raw_action[active] - teacher_action[active]),
                        axis=-1,
                    ).tolist()
                )
                action = torch.clamp(
                    torch.from_numpy(raw_action).to(device).float(),
                    action_low,
                    action_high,
                )
                action[~torch.from_numpy(active).to(device)] = 0.0
                obs, reward, terminated, truncated, info = env.step(action)
                previous_action = action_norm.transform(
                    action.cpu().numpy().astype(np.float32)
                )
                reward_np = reward.detach().cpu().numpy().reshape(-1)
                batch_final[active] = reward_np[active]
                batch_max[active] = np.maximum(
                    batch_max[active], reward_np[active]
                )
                if "success" in info:
                    success_once |= (
                        info["success"]
                        .detach()
                        .cpu()
                        .numpy()
                        .reshape(-1)
                        .astype(bool)
                    )
                done = (
                    torch.logical_or(terminated, truncated)
                    .detach()
                    .cpu()
                    .numpy()
                    .reshape(-1)
                    .astype(bool)
                )
                newly_done = active & done
                if np.any(newly_done):
                    progress.update(int(np.sum(newly_done)))
                    active[newly_done] = False
            if np.any(active):
                progress.update(int(np.sum(active)))
            successes.extend(success_once.astype(float).tolist())
            final_rewards.extend(batch_final.astype(float).tolist())
            max_rewards.extend(batch_max.astype(float).tolist())
        finally:
            env.close()
    progress.close()
    payload = _rollout_payload(
        config,
        method,
        n_trajectories,
        seed,
        eval_seed_start,
        successes,
        final_rewards,
        max_rewards,
        teacher_maes,
        checkpoint_path,
        {
            "policy_type": policy_type,
            "representation": representation,
            "elapsed_s": timer.elapsed(),
            "flow_noise": (
                "deterministic SeedSequence(training_seed, environment_seed, "
                "step, action_dim)"
                if policy_type == "flow"
                else None
            ),
        },
    )
    write_json(output_path, payload)
    return output_path


@torch.inference_mode()
def evaluate_vae_scaling_flow_hierarchy(
    config: Config,
    n_trajectories: int,
    seed: int,
    episodes: int | None = None,
    force: bool = False,
) -> Path:
    point_config = vae_scaling_config(config, n_trajectories)
    eval_episodes = int(
        episodes or config.get("vae_scaling.deployable_eval_episodes", 500)
    )
    output_path = (
        ensure_dir(_point_result_dir(point_config, seed) / "flow_hierarchy")
        / f"eval_{eval_episodes}.json"
    )
    if output_path.exists() and not force:
        return output_path
    checkpoint_path = train_vae_scaling_flow_high_level(
        config, n_trajectories, seed, force=False
    )
    device = default_device()
    checkpoint = torch.load(
        checkpoint_path, map_location=device, weights_only=False
    )
    hierarchy = torch.load(
        checkpoint["hierarchy_checkpoint"],
        map_location=device,
        weights_only=False,
    )
    high_flow = FlowModel(
        int(checkpoint["goal_dim"]),
        int(checkpoint["condition_dim"]),
        int(checkpoint["hidden_dim"]),
    ).to(device)
    high_flow.load_state_dict(checkpoint["model"])
    high_flow.eval()
    _unused_high, low_model = _load_hierarchy(hierarchy, device)
    frame_norm = Standardizer.from_state_dict(hierarchy["frame_norm"])
    action_norm = Standardizer.from_state_dict(hierarchy["action_norm"])
    dino = _phase4_dino_from_config(config, device)
    teacher = load_ppo_agent(_rl_paths(config).best, device)
    horizon_steps = int(hierarchy["horizon_steps"])
    update_period = int(hierarchy["update_period"])
    max_steps = int(config.get("env_max_episode_steps", 100))
    num_envs_max = min(
        int(config.get("vae_scaling.eval_num_envs", 64)), eval_episodes
    )
    eval_seed_start = int(config.get("vae_scaling.eval_seed_start", 2_200_000))
    successes: list[float] = []
    final_rewards: list[float] = []
    max_rewards: list[float] = []
    teacher_maes: list[float] = []
    goal_norms: list[float] = []
    timer = Timer()
    progress = trange(
        eval_episodes, desc=f"eval flow hierarchy n={n_trajectories}"
    )
    for batch_start in range(0, eval_episodes, num_envs_max):
        num_envs = min(num_envs_max, eval_episodes - batch_start)
        reset_seeds = [
            eval_seed_start + batch_start + index for index in range(num_envs)
        ]
        env = gym.make(
            config.get("env_id"),
            obs_mode="rgb+state",
            control_mode=config.get("control_mode"),
            reward_mode="normalized_dense",
            render_mode=None,
            sim_backend=_rl_backend(config),
            num_envs=num_envs,
            reconfiguration_freq=config.get("rl.eval_reconfiguration_freq", 1),
        )
        action_low_np = np.asarray(env.action_space.low, dtype=np.float32)
        action_high_np = np.asarray(env.action_space.high, dtype=np.float32)
        if action_low_np.ndim == 2:
            action_low_np = action_low_np[0]
            action_high_np = action_high_np[0]
        action_low = torch.as_tensor(action_low_np, device=device)
        action_high = torch.as_tensor(action_high_np, device=device)
        previous_action = np.repeat(
            action_norm.transform(np.zeros((1, 3), dtype=np.float32)),
            num_envs,
            axis=0,
        )
        held_goal = np.zeros(
            (num_envs, int(hierarchy["goal_dim"])), dtype=np.float32
        )
        countdown = np.zeros(num_envs, dtype=np.int32)
        decision_count = np.zeros(num_envs, dtype=np.int32)
        active = np.ones(num_envs, dtype=bool)
        success_once = np.zeros(num_envs, dtype=bool)
        batch_final = np.zeros(num_envs, dtype=np.float32)
        batch_max = np.full(num_envs, -np.inf, dtype=np.float32)
        try:
            obs, _info = env.reset(seed=reset_seeds)
            for _step in range(max_steps):
                if not np.any(active):
                    break
                frames = _phase4_frame_inputs(
                    obs, dino, int(config.get("dino.batch_size", 64))
                )
                normalized_frames = frame_norm.transform(frames)
                replan = active & (countdown <= 0)
                if np.any(replan):
                    condition = np.concatenate(
                        [normalized_frames, previous_action], axis=-1
                    ).astype(np.float32)
                    noise = np.zeros_like(held_goal)
                    for index in np.flatnonzero(replan):
                        noise[index] = _deterministic_noise(
                            seed,
                            [reset_seeds[index]],
                            int(decision_count[index]),
                            int(hierarchy["goal_dim"]),
                        )[0]
                    predicted_goal = sample_flow(
                        high_flow,
                        torch.from_numpy(condition).to(device),
                        steps=int(checkpoint["flow_steps"]),
                        sample_dim=int(hierarchy["goal_dim"]),
                        initial_noise=torch.from_numpy(noise).to(device),
                    ).cpu().numpy()
                    held_goal[replan] = predicted_goal[replan]
                    goal_norms.extend(
                        np.linalg.norm(predicted_goal[replan], axis=-1).tolist()
                    )
                    countdown[replan] = update_period
                    decision_count[replan] += 1
                condition = _low_condition_array(
                    normalized_frames,
                    np.empty_like(held_goal),
                    held_goal,
                    previous_action,
                    (
                        np.maximum(countdown, 1).astype(np.float32)
                        / horizon_steps
                    )[:, None],
                    "concat",
                )
                raw_action = action_norm.inverse(
                    low_model(
                        torch.from_numpy(condition).to(device).float()
                    )
                    .cpu()
                    .numpy()
                )
                teacher_action = (
                    torch.clamp(
                        teacher.actor_mean(_phase7_obs_state_tensor(obs, device)),
                        action_low,
                        action_high,
                    )
                    .cpu()
                    .numpy()
                )
                teacher_maes.extend(
                    np.mean(
                        np.abs(raw_action[active] - teacher_action[active]),
                        axis=-1,
                    ).tolist()
                )
                action = torch.clamp(
                    torch.from_numpy(raw_action).to(device).float(),
                    action_low,
                    action_high,
                )
                action[~torch.from_numpy(active).to(device)] = 0.0
                obs, reward, terminated, truncated, info = env.step(action)
                previous_action = action_norm.transform(
                    action.cpu().numpy().astype(np.float32)
                )
                countdown -= 1
                reward_np = reward.detach().cpu().numpy().reshape(-1)
                batch_final[active] = reward_np[active]
                batch_max[active] = np.maximum(
                    batch_max[active], reward_np[active]
                )
                if "success" in info:
                    success_once |= (
                        info["success"]
                        .detach()
                        .cpu()
                        .numpy()
                        .reshape(-1)
                        .astype(bool)
                    )
                done = (
                    torch.logical_or(terminated, truncated)
                    .detach()
                    .cpu()
                    .numpy()
                    .reshape(-1)
                    .astype(bool)
                )
                newly_done = active & done
                if np.any(newly_done):
                    progress.update(int(np.sum(newly_done)))
                    active[newly_done] = False
            if np.any(active):
                progress.update(int(np.sum(active)))
            successes.extend(success_once.astype(float).tolist())
            final_rewards.extend(batch_final.astype(float).tolist())
            max_rewards.extend(batch_max.astype(float).tolist())
        finally:
            env.close()
    progress.close()
    payload = _rollout_payload(
        config,
        "flow_hierarchy",
        n_trajectories,
        seed,
        eval_seed_start,
        successes,
        final_rewards,
        max_rewards,
        teacher_maes,
        checkpoint_path,
        {
            "elapsed_s": timer.elapsed(),
            "mean_normalized_goal_norm": float(np.mean(goal_norms)),
            "high_level_decisions_per_episode": len(goal_norms) / eval_episodes,
            "flow_noise": (
                "deterministic SeedSequence(training_seed, environment_seed, "
                "decision_index, goal_dim)"
            ),
        },
    )
    write_json(output_path, payload)
    return output_path


def evaluate_vae_scaling_point(
    config: Config,
    n_trajectories: int,
    seed: int,
    deployable_episodes: int | None = None,
    oracle_episodes: int | None = None,
    force: bool = False,
) -> Path:
    point_config = vae_scaling_config(config, n_trajectories)
    deployable_count = int(
        deployable_episodes
        or config.get("vae_scaling.deployable_eval_episodes", 500)
    )
    oracle_count = int(
        oracle_episodes or config.get("vae_scaling.oracle_eval_episodes", 50)
    )
    train_vae_scaling_point(config, n_trajectories, seed, force=False)
    result_dir = _point_result_dir(point_config, seed)
    summary_path = result_dir / "summary.json"
    if summary_path.exists() and not force:
        return summary_path
    deterministic_path = evaluate_learned_interface_hierarchy(
        point_config,
        VAE_CANDIDATE,
        "learned",
        seed,
        episodes=deployable_count,
        force=force,
    )
    oracle_path = evaluate_learned_interface_hierarchy(
        point_config,
        VAE_CANDIDATE,
        "oracle",
        seed,
        episodes=oracle_count,
        force=force,
    )
    paths = {
        "deterministic_hierarchy": deterministic_path,
        "flow_hierarchy": evaluate_vae_scaling_flow_hierarchy(
            config,
            n_trajectories,
            seed,
            episodes=deployable_count,
            force=force,
        ),
        "oracle_hierarchy": oracle_path,
    }
    for representation in ("latent", "observation"):
        for policy_type in ("deterministic", "flow"):
            paths[f"flat_{representation}_{policy_type}"] = (
                evaluate_vae_scaling_flat_policy(
                    config,
                    n_trajectories,
                    representation,
                    policy_type,
                    seed,
                    episodes=deployable_count,
                    force=force,
                )
            )
    rows = []
    for method in ALL_METHODS:
        with paths[method].open() as stream:
            result = json.load(stream)
        rows.append(
            {
                "method": method,
                "source": str(paths[method]),
                "success": float(result["success"]),
                "success_wilson_95": result["success_wilson_95"],
                "final_reward": float(result["final_reward"]),
                "max_reward": float(result["max_reward"]),
                "teacher_action_mae": float(result["teacher_action_mae"]),
                "episodes": int(result["episodes"]),
            }
        )
    with write_vae_scaling_manifest(config, n_trajectories).open() as stream:
        manifest = json.load(stream)
    payload = {
        "experiment": "vae512_sample_efficiency",
        "n_trajectories": n_trajectories,
        "training_seed": seed,
        "deployable_evaluation_episodes": deployable_count,
        "oracle_evaluation_episodes": oracle_count,
        "evaluation_seed_start": int(
            config.get("vae_scaling.eval_seed_start", 2_200_000)
        ),
        "data_manifest": str(
            write_vae_scaling_manifest(config, n_trajectories)
        ),
        "data_manifest_sha256": manifest["sha256"],
        "rows": rows,
        "metadata": _runtime_metadata(config),
    }
    write_json(summary_path, payload)
    return summary_path
