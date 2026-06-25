from __future__ import annotations

import csv
import copy
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import torch
from torch import nn
from tqdm import trange

from hcl_poc.config import Config
from hcl_poc.incremental import _phase4_dino_from_config, _phase4_frame_inputs, _rl_backend
from hcl_poc.learned_interface import (
    _encode_effect_array,
    _load_hierarchy,
    _load_representation,
    _low_condition_array,
    train_learned_interface_hierarchy,
)
from hcl_poc.reachability import (
    ReachabilityDistance,
    _load_reachability_latents,
    load_reachability_distance,
)
from hcl_poc.rl import layer_init
from hcl_poc.utils import Standardizer, Timer, default_device, ensure_dir, set_seed, write_json
from hcl_poc.vae_scaling import VAE_CANDIDATE, vae_scaling_config


def _mlp(
    in_dim: int, out_dim: int, width: int, depth: int, output_std: float = 1.0
) -> nn.Sequential:
    layers: list[nn.Module] = []
    dim = in_dim
    for _ in range(depth):
        layers.extend([layer_init(nn.Linear(dim, width)), nn.Tanh()])
        dim = width
    layers.append(layer_init(nn.Linear(dim, out_dim), std=output_std))
    return nn.Sequential(*layers)


class ResidualActorCritic(nn.Module):
    def __init__(
        self,
        condition_dim: int,
        action_dim: int = 3,
        width: int = 256,
        depth: int = 2,
        initial_logstd: float = -2.3,
    ) -> None:
        super().__init__()
        self.condition_dim = condition_dim
        self.action_dim = action_dim
        self.width = width
        self.depth = depth
        self.actor_mean = _mlp(condition_dim, action_dim, width, depth, output_std=1e-4)
        nn.init.zeros_(self.actor_mean[-1].bias)
        self.actor_logstd = nn.Parameter(torch.full((1, action_dim), initial_logstd))
        self.critic = _mlp(condition_dim, 1, width, depth, output_std=1.0)

    def distribution(self, condition: torch.Tensor) -> torch.distributions.Normal:
        mean = self.actor_mean(condition)
        std = torch.exp(self.actor_logstd.expand_as(mean))
        return torch.distributions.Normal(mean, std)

    def get_action_and_value(
        self,
        condition: torch.Tensor,
        raw_action: torch.Tensor | None = None,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        distribution = self.distribution(condition)
        if raw_action is None:
            raw_action = distribution.mean if deterministic else distribution.sample()
        return (
            raw_action,
            distribution.log_prob(raw_action).sum(-1),
            distribution.entropy().sum(-1),
            self.critic(condition).flatten(),
        )


class DirectLowActorCritic(nn.Module):
    def __init__(
        self,
        low_model: nn.Module,
        action_mean: np.ndarray,
        action_std: np.ndarray,
        condition_dim: int,
        width: int = 256,
        depth: int = 2,
        initial_logstd: float = -2.3,
    ) -> None:
        super().__init__()
        self.condition_dim = condition_dim
        self.action_dim = 3
        self.width = width
        self.depth = depth
        self.low_model = copy.deepcopy(low_model)
        for parameter in self.low_model.parameters():
            parameter.requires_grad_(False)
        if hasattr(self.low_model, "output_layer"):
            last = self.low_model.output_layer
        else:
            try:
                last = self.low_model.policy.net[-1]
            except AttributeError as exc:
                raise ValueError(
                    "R3 last-layer tuning requires a low policy with policy.net "
                    "or output_layer"
                ) from exc
        if not isinstance(last, nn.Linear):
            raise ValueError("Expected final low-policy module to be nn.Linear")
        for parameter in last.parameters():
            parameter.requires_grad_(True)
        self.actor_logstd = nn.Parameter(torch.full((1, 3), initial_logstd))
        self.critic = _mlp(condition_dim, 1, width, depth, output_std=1.0)
        self.register_buffer(
            "action_mean", torch.as_tensor(action_mean.reshape(1, -1), dtype=torch.float32)
        )
        self.register_buffer(
            "action_std", torch.as_tensor(action_std.reshape(1, -1), dtype=torch.float32)
        )

    def mean_action(self, condition: torch.Tensor) -> torch.Tensor:
        normalized = self.low_model(condition)
        return normalized * self.action_std + self.action_mean

    def get_action_and_value(
        self,
        condition: torch.Tensor,
        raw_action: torch.Tensor | None = None,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mean = self.mean_action(condition)
        std = torch.exp(self.actor_logstd.expand_as(mean))
        distribution = torch.distributions.Normal(mean, std)
        if raw_action is None:
            raw_action = mean if deterministic else distribution.sample()
        return (
            raw_action,
            distribution.log_prob(raw_action).sum(-1),
            distribution.entropy().sum(-1),
            self.critic(condition).flatten(),
        )


@dataclass
class FrozenHierarchy:
    high_model: nn.Module
    low_model: nn.Module
    encoder: nn.Module
    frame_norm: Standardizer
    representation_frame_norm: Standardizer
    goal_norm: Standardizer
    action_norm: Standardizer
    horizon_steps: int
    update_period: int
    conditioning: str
    encoder_type: str
    frame_dim: int
    goal_dim: int
    checkpoint_path: Path


def _paths(
    config: Config, n_demo: int, seed: int, run_name: str, candidate: str = VAE_CANDIDATE
) -> tuple[Path, Path]:
    demo_label = f"n{n_demo}" if candidate == VAE_CANDIDATE else candidate
    artifact = ensure_dir(
        config.path_value("paths.incremental_artifact_dir")
        / "low_level_rl"
        / demo_label
        / f"seed{seed}"
        / run_name
    )
    result = ensure_dir(
        config.path_value("paths.incremental_results_dir")
        / "low_level_rl"
        / demo_label
        / f"seed{seed}"
        / run_name
    )
    return artifact, result


def _load_frozen(
    config: Config,
    n_demo: int,
    seed: int,
    device: torch.device,
    candidate: str = VAE_CANDIDATE,
) -> FrozenHierarchy:
    point_config = vae_scaling_config(config, n_demo) if candidate == VAE_CANDIDATE else config
    hierarchy_path = train_learned_interface_hierarchy(
        point_config, candidate, seed=seed, force=False
    )
    checkpoint = torch.load(hierarchy_path, map_location="cpu", weights_only=False)
    high_model, low_model = _load_hierarchy(checkpoint, device)
    encoder, representation = _load_representation(
        Path(checkpoint["representation_checkpoint"]), device
    )
    modules = [high_model, low_model, encoder]
    for module in modules:
        module.eval()
        module.requires_grad_(False)
    return FrozenHierarchy(
        high_model=high_model,
        low_model=low_model,
        encoder=encoder,
        frame_norm=Standardizer.from_state_dict(checkpoint["frame_norm"]),
        representation_frame_norm=Standardizer.from_state_dict(representation["frame_norm"]),
        goal_norm=Standardizer.from_state_dict(checkpoint["goal_norm"]),
        action_norm=Standardizer.from_state_dict(checkpoint["action_norm"]),
        horizon_steps=int(checkpoint["horizon_steps"]),
        update_period=int(checkpoint["update_period"]),
        conditioning=str(checkpoint.get("conditioning", "concat")),
        encoder_type=str(representation["encoder_type"]),
        frame_dim=int(checkpoint["frame_dim"]),
        goal_dim=int(checkpoint["goal_dim"]),
        checkpoint_path=Path(hierarchy_path),
    )


@torch.inference_mode()
def _encode_frames(frozen: FrozenHierarchy, frames: np.ndarray, device: torch.device) -> np.ndarray:
    if frozen.encoder_type == "effect":
        raise ValueError("Effect-code representations require an anchor frame")
    normalized = frozen.representation_frame_norm.transform(frames)
    latent = frozen.encoder(torch.from_numpy(normalized).to(device).float())
    return frozen.goal_norm.transform(latent.cpu().numpy().astype(np.float32))


@torch.inference_mode()
def _encode_effect_progress(
    frozen: FrozenHierarchy,
    anchor_frames: np.ndarray,
    frames: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    if frozen.encoder_type != "effect":
        return _encode_frames(frozen, frames, device)
    effect = _encode_effect_array(
        frozen.encoder,
        frozen.representation_frame_norm,
        anchor_frames,
        frames,
        np.ones(len(frames), dtype=np.float32),
        device,
    )
    return frozen.goal_norm.transform(effect.astype(np.float32))


def _visual_env(config: Config, num_envs: int):
    from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv

    base = gym.make(
        config.get("env_id"),
        obs_mode="rgb+state",
        control_mode=config.get("control_mode"),
        reward_mode="normalized_dense",
        render_mode=None,
        sim_backend=_rl_backend(config),
        num_envs=num_envs,
        reconfiguration_freq=config.get("rl.eval_reconfiguration_freq", 1),
    )
    return ManiSkillVectorEnv(
        base,
        num_envs,
        ignore_terminations=True,
        record_metrics=True,
    )


def _default_reachability_checkpoint(config: Config, n_demo: int, seed: int) -> Path:
    point_config = vae_scaling_config(config, n_demo)
    return (
        point_config.path_value("paths.incremental_artifact_dir")
        / "reachability_distance"
        / VAE_CANDIDATE
        / f"seed{seed}"
        / "d_phi.pt"
    )


def _candidate_reachability_checkpoint(config: Config, candidate: str, seed: int) -> Path:
    return (
        config.path_value("paths.incremental_artifact_dir")
        / "reachability_distance"
        / candidate
        / f"seed{seed}"
        / "d_phi.pt"
    )


def _condition_dim(frozen: FrozenHierarchy) -> int:
    if frozen.conditioning == "relation":
        return frozen.frame_dim + 2 * frozen.goal_dim + 4
    return frozen.frame_dim + frozen.goal_dim + 4


@torch.inference_mode()
def _reachability_distance_values(
    model: ReachabilityDistance,
    distance_goal_norm: Standardizer,
    frozen: FrozenHierarchy,
    current_latent_norm: np.ndarray,
    goal_latent_norm: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    current_raw = frozen.goal_norm.inverse(current_latent_norm)
    goal_raw = frozen.goal_norm.inverse(goal_latent_norm)
    current = distance_goal_norm.transform(current_raw).astype(np.float32)
    goal = distance_goal_norm.transform(goal_raw).astype(np.float32)
    values = model(
        torch.from_numpy(current).to(device).float(),
        torch.from_numpy(goal).to(device).float(),
    )
    return values.cpu().numpy().astype(np.float32)


def _binary_auc(scores: np.ndarray, labels: np.ndarray) -> float | None:
    labels_bool = labels.astype(bool)
    positives = scores[labels_bool]
    negatives = scores[~labels_bool]
    if len(positives) == 0 or len(negatives) == 0:
        return None
    return float(
        (
            (positives[:, None] > negatives[None, :]).mean()
            + 0.5 * (positives[:, None] == negatives[None, :]).mean()
        )
    )


def compare_serial_low_level_eval(
    base_json: Path,
    candidate_json: Path,
    output: Path | None = None,
    force: bool = False,
) -> Path:
    output = output or candidate_json.with_name(
        f"{candidate_json.stem}_vs_{base_json.stem}.json"
    )
    if output.exists() and not force:
        return output
    base = json.loads(base_json.read_text())
    candidate = json.loads(candidate_json.read_text())
    base_seeds = base.get("episode_seed")
    candidate_seeds = candidate.get("episode_seed")
    if base_seeds is None or candidate_seeds is None:
        raise ValueError("Both serial eval JSONs must contain episode_seed")
    if base_seeds != candidate_seeds:
        raise ValueError("Serial eval episode_seed arrays do not match")
    base_success = np.asarray(base["episode_success"], dtype=np.float32)
    candidate_success = np.asarray(candidate["episode_success"], dtype=np.float32)
    if len(base_success) != len(candidate_success):
        raise ValueError("Serial eval episode_success arrays have different lengths")
    improvements = (base_success == 0.0) & (candidate_success == 1.0)
    regressions = (base_success == 1.0) & (candidate_success == 0.0)
    payload = {
        "base_json": str(base_json),
        "candidate_json": str(candidate_json),
        "episodes": int(len(base_success)),
        "episode_seed_start": int(base_seeds[0]) if base_seeds else None,
        "episode_seed_end": int(base_seeds[-1]) if base_seeds else None,
        "base_success": float(base_success.mean()) if len(base_success) else None,
        "candidate_success": float(candidate_success.mean())
        if len(candidate_success)
        else None,
        "success_delta": float(candidate_success.mean() - base_success.mean())
        if len(base_success)
        else None,
        "improvements": int(improvements.sum()),
        "regressions": int(regressions.sum()),
        "net_improvements": int(improvements.sum() - regressions.sum()),
        "both_success": int(((base_success == 1.0) & (candidate_success == 1.0)).sum()),
        "both_fail": int(((base_success == 0.0) & (candidate_success == 0.0)).sum()),
        "improvement_seeds": [
            int(seed) for seed, value in zip(base_seeds, improvements, strict=True) if value
        ],
        "regression_seeds": [
            int(seed) for seed, value in zip(base_seeds, regressions, strict=True) if value
        ],
    }
    write_json(output, payload)
    return output


def compare_serial_low_level_segments(
    base_json: Path,
    candidate_json: Path,
    output: Path | None = None,
    force: bool = False,
) -> Path:
    output = output or candidate_json.with_name(
        f"{candidate_json.stem}_segments_vs_{base_json.stem}.json"
    )
    if output.exists() and not force:
        return output
    base = json.loads(base_json.read_text())
    candidate = json.loads(candidate_json.read_text())
    required = [
        "serial_segment_episode_seed",
        "serial_segment_index",
        "serial_segment_raw_distance_reduction",
    ]
    for payload, name in ((base, "base"), (candidate, "candidate")):
        missing = [key for key in required if key not in payload]
        if missing:
            raise ValueError(f"{name} serial eval is missing segment fields: {missing}")

    def segment_map(payload: dict[str, Any]) -> dict[tuple[int, int], int]:
        seeds = payload["serial_segment_episode_seed"]
        indices = payload["serial_segment_index"]
        if len(seeds) != len(indices):
            raise ValueError("serial segment seed/index arrays have different lengths")
        return {
            (int(seed), int(index)): offset
            for offset, (seed, index) in enumerate(zip(seeds, indices, strict=True))
        }

    base_map = segment_map(base)
    candidate_map = segment_map(candidate)
    common_keys = [key for key in base_map if key in candidate_map]
    if not common_keys:
        raise ValueError("No matching serial segments found")
    base_raw_reduction = np.asarray(
        [base["serial_segment_raw_distance_reduction"][base_map[key]] for key in common_keys],
        dtype=np.float32,
    )
    candidate_raw_reduction = np.asarray(
        [
            candidate["serial_segment_raw_distance_reduction"][candidate_map[key]]
            for key in common_keys
        ],
        dtype=np.float32,
    )
    delta = candidate_raw_reduction - base_raw_reduction
    helpful = delta > 0.0
    harmful = delta < 0.0
    feature_names = [
        "serial_segment_initial_selected_distance",
        "serial_segment_initial_raw_distance",
        "serial_segment_initial_base_action_l2",
        "serial_segment_initial_previous_action_norm_l2",
        "serial_segment_residual_l2_mean",
    ]
    feature_diagnostics: dict[str, dict[str, float | None]] = {}
    for name in feature_names:
        if name not in candidate:
            continue
        values = np.asarray(
            [candidate[name][candidate_map[key]] for key in common_keys],
            dtype=np.float32,
        )
        auc = _binary_auc(values, helpful.astype(np.float32))
        corr = (
            float(np.corrcoef(values, delta)[0, 1])
            if len(values) > 1 and np.std(values) > 0.0 and np.std(delta) > 0.0
            else None
        )
        feature_diagnostics[name] = {
            "helpful_auc": auc,
            "oriented_helpful_auc": max(auc, 1.0 - auc) if auc is not None else None,
            "corr_raw_reduction_delta": corr,
            "helpful_mean": float(values[helpful].mean()) if np.any(helpful) else None,
            "harmful_mean": float(values[harmful].mean()) if np.any(harmful) else None,
        }
    payload = {
        "base_json": str(base_json),
        "candidate_json": str(candidate_json),
        "base_segments": int(len(base_map)),
        "candidate_segments": int(len(candidate_map)),
        "common_segments": int(len(common_keys)),
        "aligned_order": [
            (int(seed), int(index))
            for seed, index in zip(
                base["serial_segment_episode_seed"],
                base["serial_segment_index"],
                strict=True,
            )
        ]
        == [
            (int(seed), int(index))
            for seed, index in zip(
                candidate["serial_segment_episode_seed"],
                candidate["serial_segment_index"],
                strict=True,
            )
        ],
        "base_raw_reduction_mean": float(base_raw_reduction.mean()),
        "candidate_raw_reduction_mean": float(candidate_raw_reduction.mean()),
        "raw_reduction_delta_mean": float(delta.mean()),
        "helpful_segments": int(helpful.sum()),
        "harmful_segments": int(harmful.sum()),
        "neutral_segments": int((delta == 0.0).sum()),
        "feature_diagnostics": feature_diagnostics,
    }
    write_json(output, payload)
    return output


def fit_serial_initial_selector(
    base_json: Path,
    candidate_json: Path,
    output: Path,
    validation_base_json: Path | None = None,
    validation_candidate_json: Path | None = None,
    ridge: float = 1.0,
    force: bool = False,
) -> Path:
    if output.exists() and not force:
        return output
    if ridge < 0.0:
        raise ValueError("ridge must be non-negative")
    base = json.loads(base_json.read_text())
    candidate = json.loads(candidate_json.read_text())
    feature_names = [
        "episode_initial_selected_distance",
        "episode_initial_raw_distance",
        "episode_initial_base_action_l2",
    ]

    def load_pair(
        left: dict[str, Any],
        right: dict[str, Any],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[int]]:
        left_seeds = left.get("episode_seed")
        right_seeds = right.get("episode_seed")
        if left_seeds is None or right_seeds is None:
            raise ValueError("Both serial eval JSONs must contain episode_seed")
        if left_seeds != right_seeds:
            raise ValueError("Serial eval episode_seed arrays do not match")
        features = np.stack(
            [np.asarray(left[name], dtype=np.float32) for name in feature_names],
            axis=1,
        )
        left_success = np.asarray(left["episode_success"], dtype=np.float32)
        right_success = np.asarray(right["episode_success"], dtype=np.float32)
        if len(features) != len(left_success) or len(features) != len(right_success):
            raise ValueError("Serial eval feature and success arrays have different lengths")
        return features, left_success, right_success, [int(seed) for seed in left_seeds]

    train_x, train_base_success, train_candidate_success, train_seeds = load_pair(
        base, candidate
    )
    mean = train_x.mean(axis=0)
    std = train_x.std(axis=0) + 1e-6
    train_z = (train_x - mean) / std
    discordant = train_base_success != train_candidate_success
    if not bool(np.any(discordant)):
        raise ValueError("No discordant serial outcomes are available for selector fitting")
    labels = np.where(
        train_candidate_success[discordant] > train_base_success[discordant],
        1.0,
        -1.0,
    ).astype(np.float32)
    design = train_z[discordant]
    weights = np.linalg.solve(
        design.T @ design + float(ridge) * np.eye(design.shape[1], dtype=np.float32),
        design.T @ labels,
    ).astype(np.float32)

    def evaluate_split(
        features: np.ndarray,
        base_success: np.ndarray,
        candidate_success: np.ndarray,
        seeds: list[int],
        threshold: float,
    ) -> dict[str, Any]:
        scores = ((features - mean) / std) @ weights
        use_candidate = scores >= threshold
        mixed = np.where(use_candidate, candidate_success, base_success)
        improvements = (base_success == 0.0) & (mixed == 1.0)
        regressions = (base_success == 1.0) & (mixed == 0.0)
        return {
            "episodes": int(len(base_success)),
            "seed_start": int(seeds[0]) if seeds else None,
            "seed_end": int(seeds[-1]) if seeds else None,
            "base_success": float(base_success.mean()) if len(base_success) else None,
            "candidate_success": float(candidate_success.mean())
            if len(candidate_success)
            else None,
            "selector_success": float(mixed.mean()) if len(mixed) else None,
            "selector_use_candidate_rate": float(use_candidate.mean())
            if len(use_candidate)
            else None,
            "improvements": int(improvements.sum()),
            "regressions": int(regressions.sum()),
            "net_improvements": int(improvements.sum() - regressions.sum()),
        }

    train_scores = train_z @ weights
    best: tuple[float, float] | None = None
    for threshold in np.unique(train_scores):
        mixed = np.where(
            train_scores >= threshold,
            train_candidate_success,
            train_base_success,
        )
        score = float(mixed.mean())
        item = (score, float(threshold))
        if best is None or item[0] > best[0]:
            best = item
    if best is None:
        raise RuntimeError("Failed to choose selector threshold")
    _train_best_success, threshold = best
    payload: dict[str, Any] = {
        "base_json": str(base_json),
        "candidate_json": str(candidate_json),
        "feature_names": feature_names,
        "ridge": float(ridge),
        "weights": weights.astype(float).tolist(),
        "mean": mean.astype(float).tolist(),
        "std": std.astype(float).tolist(),
        "threshold": threshold,
        "train": evaluate_split(
            train_x,
            train_base_success,
            train_candidate_success,
            train_seeds,
            threshold,
        ),
        "validation": None,
    }
    if validation_base_json is not None or validation_candidate_json is not None:
        if validation_base_json is None or validation_candidate_json is None:
            raise ValueError("Provide both validation base and candidate JSONs")
        validation_base = json.loads(validation_base_json.read_text())
        validation_candidate = json.loads(validation_candidate_json.read_text())
        val_x, val_base_success, val_candidate_success, val_seeds = load_pair(
            validation_base, validation_candidate
        )
        payload["validation_base_json"] = str(validation_base_json)
        payload["validation_candidate_json"] = str(validation_candidate_json)
        payload["validation"] = evaluate_split(
            val_x,
            val_base_success,
            val_candidate_success,
            val_seeds,
            threshold,
        )
    write_json(output, payload)
    return output


class HierarchyRollout:
    def __init__(
        self,
        config: Config,
        frozen: FrozenHierarchy,
        num_envs: int,
        seed_start: int,
        device: torch.device,
        distance_metric: str = "raw_l2",
        reachability_checkpoint_path: Path | None = None,
    ) -> None:
        if distance_metric not in {"raw_l2", "reachability"}:
            raise ValueError(f"Unknown low-level distance metric: {distance_metric}")
        self.config = config
        self.frozen = frozen
        self.num_envs = num_envs
        self.seed_start = seed_start
        self.device = device
        self.distance_metric = distance_metric
        self.reachability_model: ReachabilityDistance | None = None
        self.reachability_goal_norm: Standardizer | None = None
        self.reachability_checkpoint_path = reachability_checkpoint_path
        if distance_metric == "reachability":
            if reachability_checkpoint_path is None:
                raise ValueError("reachability distance metric requires a checkpoint")
            model, goal_norm, _checkpoint = load_reachability_distance(
                reachability_checkpoint_path, device
            )
            model.requires_grad_(False)
            self.reachability_model = model
            self.reachability_goal_norm = goal_norm
        self.env = _visual_env(config, num_envs)
        self.dino = _phase4_dino_from_config(config, device)
        low = np.asarray(self.env.single_action_space.low, dtype=np.float32)
        high = np.asarray(self.env.single_action_space.high, dtype=np.float32)
        self.action_low = torch.as_tensor(low, device=device)
        self.action_high = torch.as_tensor(high, device=device)
        self.zero_previous = frozen.action_norm.transform(np.zeros((1, 3), dtype=np.float32))[0]
        self.episode_offset = 0
        self.obs: dict[str, Any]
        self.frames: np.ndarray
        self.normalized_frames: np.ndarray
        self.anchor_frames: np.ndarray
        self.current_latent: np.ndarray
        self.previous_action: np.ndarray
        self.held_goal: np.ndarray
        self.countdown: np.ndarray
        self.previous_env_reward: np.ndarray
        self.reset()

    def distance(self, current_latent: np.ndarray, goal_latent: np.ndarray) -> np.ndarray:
        if self.distance_metric == "raw_l2":
            return self.raw_distance(current_latent, goal_latent)
        if self.reachability_model is None or self.reachability_goal_norm is None:
            raise RuntimeError("Reachability distance model was not initialized")
        return _reachability_distance_values(
            self.reachability_model,
            self.reachability_goal_norm,
            self.frozen,
            current_latent,
            goal_latent,
            self.device,
        )

    @staticmethod
    def raw_distance(current_latent: np.ndarray, goal_latent: np.ndarray) -> np.ndarray:
        return np.mean(np.square(current_latent - goal_latent), axis=-1).astype(np.float32)

    def reset(self) -> None:
        seeds = [self.seed_start + self.episode_offset + index for index in range(self.num_envs)]
        self.episode_offset += self.num_envs
        self.obs, _info = self.env.reset(seed=seeds)
        self.frames = _phase4_frame_inputs(
            self.obs, self.dino, int(self.config.get("dino.batch_size", 64))
        )
        self.normalized_frames = self.frozen.frame_norm.transform(self.frames)
        self.anchor_frames = self.frames.copy()
        self.current_latent = _encode_effect_progress(
            self.frozen, self.anchor_frames, self.frames, self.device
        )
        self.previous_action = np.repeat(self.zero_previous[None], self.num_envs, axis=0)
        self.held_goal = np.zeros((self.num_envs, self.frozen.goal_dim), dtype=np.float32)
        self.countdown = np.zeros(self.num_envs, dtype=np.int32)
        self.previous_env_reward = np.zeros(self.num_envs, dtype=np.float32)

    @torch.inference_mode()
    def condition(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, np.ndarray, np.ndarray]:
        replan = self.countdown <= 0
        if np.any(replan):
            high_condition = np.concatenate([self.normalized_frames, self.previous_action], axis=-1)
            predicted = (
                self.frozen.high_model(torch.from_numpy(high_condition).to(self.device).float())
                .cpu()
                .numpy()
            )
            self.held_goal[replan] = predicted[replan]
            self.anchor_frames[replan] = self.frames[replan]
            self.current_latent[replan] = _encode_effect_progress(
                self.frozen,
                self.anchor_frames[replan],
                self.frames[replan],
                self.device,
            )
            self.countdown[replan] = self.frozen.update_period
        remaining = np.maximum(self.countdown, 1).astype(np.float32)
        condition_np = _low_condition_array(
            self.normalized_frames,
            self.current_latent,
            self.held_goal,
            self.previous_action,
            (remaining / self.frozen.horizon_steps)[:, None],
            self.frozen.conditioning,
        )
        condition = torch.from_numpy(condition_np).to(self.device).float()
        normalized_base = self.frozen.low_model(condition)
        base_action = torch.from_numpy(
            self.frozen.action_norm.inverse(normalized_base.cpu().numpy())
        ).to(self.device)
        distance = self.distance(self.current_latent, self.held_goal)
        return condition, base_action, distance, replan

    @torch.inference_mode()
    def step(
        self,
        executed_action: torch.Tensor,
        previous_distance: np.ndarray,
        terminal_weight: float,
        distance_progress_weight: float,
        task_reward_weight: float,
        task_progress_weight: float,
        residual_penalty: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        next_obs, env_reward, terminated, truncated, info = self.env.step(executed_action)
        done = torch.logical_or(terminated, truncated).detach().cpu().numpy().reshape(-1)
        env_reward_np = env_reward.detach().cpu().numpy().reshape(-1).astype(np.float32)
        next_frames = _phase4_frame_inputs(
            next_obs, self.dino, int(self.config.get("dino.batch_size", 64))
        )
        next_latent = _encode_effect_progress(
            self.frozen, self.anchor_frames, next_frames, self.device
        )
        next_distance = self.distance(next_latent, self.held_goal)
        raw_next_distance = self.raw_distance(next_latent, self.held_goal)
        segment_end = self.countdown == 1
        env_progress = env_reward_np - self.previous_env_reward
        reward = distance_progress_weight * (previous_distance - next_distance)
        reward -= residual_penalty
        reward -= terminal_weight * next_distance * segment_end.astype(np.float32)
        reward += task_reward_weight * env_reward_np
        reward += task_progress_weight * env_progress
        # Auto-reset observations do not belong to the previous held goal.
        reward[done] = (
            task_reward_weight * env_reward_np[done] + task_progress_weight * env_progress[done]
        )
        self.obs = next_obs
        self.frames = next_frames
        self.normalized_frames = self.frozen.frame_norm.transform(next_frames)
        self.current_latent = next_latent
        clipped = torch.clamp(executed_action, self.action_low, self.action_high)
        self.previous_action = self.frozen.action_norm.transform(
            clipped.cpu().numpy().astype(np.float32)
        )
        self.countdown -= 1
        self.previous_env_reward = env_reward_np
        if np.any(done):
            self.previous_action[done] = self.zero_previous
            self.countdown[done] = 0
            self.previous_env_reward[done] = 0.0
        metrics = {
            "next_distance": next_distance,
            "raw_next_distance": raw_next_distance,
            "segment_end": segment_end,
            "done": done,
            "env_reward": env_reward_np,
            "env_progress": env_progress,
            "info": info,
        }
        return reward.astype(np.float32), done.astype(np.float32), metrics

    def close(self) -> None:
        self.env.close()


def _teacher_goal_threshold(
    config: Config, n_demo: int, seed: int, candidate: str = VAE_CANDIDATE
) -> dict[str, float]:
    point = vae_scaling_config(config, n_demo) if candidate == VAE_CANDIDATE else config
    path = (
        point.path_value("paths.incremental_artifact_dir")
        / "learned_interface"
        / candidate
        / f"seed{seed}"
        / "encoded_episodes.pt"
    )
    hierarchy = torch.load(path.parent / "hierarchy.pt", map_location="cpu", weights_only=False)
    norm = Standardizer.from_state_dict(hierarchy["goal_norm"])
    horizon = int(hierarchy["horizon_steps"])
    representation = torch.load(
        hierarchy["representation_checkpoint"], map_location="cpu", weights_only=False
    )
    if representation["encoder_type"] == "effect":
        _train_raw, validation_raw, _encoded_path, _encoder_type = _load_reachability_latents(
            point, candidate, seed=seed, horizon_steps=horizon, force=False
        )
        one_step_distances = []
        initial_distances = []
        for episode in validation_raw:
            z = norm.transform(np.asarray(episode, dtype=np.float32))
            if len(z) < 2:
                continue
            initial_distances.append(
                np.mean((z[0:1] - z[-1:]) ** 2, axis=-1)
            )
            one_step_distances.append(
                np.mean((z[-2:-1] - z[-1:]) ** 2, axis=-1)
            )
        initial = np.concatenate(initial_distances)
        one_step = np.concatenate(one_step_distances)
        return {
            "teacher_initial_distance_mean": float(initial.mean()),
            "teacher_initial_distance_median": float(np.median(initial)),
            "teacher_one_step_distance_mean": float(one_step.mean()),
            "goal_threshold": float(np.quantile(one_step, 0.90)),
            "segments": int(len(initial)),
        }
    payload = torch.load(path, map_location="cpu", weights_only=False)
    one_step_distances: list[np.ndarray] = []
    initial_distances: list[np.ndarray] = []
    for goals in payload["validation_goals"]:
        z = norm.transform(np.asarray(goals, dtype=np.float32))
        if len(z) <= horizon:
            continue
        initial_distances.append(np.mean((z[:-horizon] - z[horizon:]) ** 2, axis=-1))
        one_step_distances.append(np.mean((z[horizon - 1 : -1] - z[horizon:]) ** 2, axis=-1))
    initial = np.concatenate(initial_distances)
    one_step = np.concatenate(one_step_distances)
    return {
        "teacher_initial_distance_mean": float(initial.mean()),
        "teacher_initial_distance_median": float(np.median(initial)),
        "teacher_one_step_distance_mean": float(one_step.mean()),
        "goal_threshold": float(np.quantile(one_step, 0.90)),
        "segments": int(len(initial)),
    }


def audit_low_level_rl(config: Config, n_demo: int, seed: int) -> Path:
    artifact, result = _paths(config, n_demo, seed, "audit")
    device = default_device()
    frozen = _load_frozen(config, n_demo, seed, device)
    threshold = _teacher_goal_threshold(config, n_demo, seed)
    condition_dim = _condition_dim(frozen)
    agent = ResidualActorCritic(condition_dim)
    zero_condition = torch.zeros((32, condition_dim))
    with torch.no_grad():
        residual = torch.tanh(agent.actor_mean(zero_condition))
    frozen_parameter_count = sum(
        parameter.numel()
        for module in (frozen.encoder, frozen.high_model, frozen.low_model)
        for parameter in module.parameters()
    )
    trainable_frozen = sum(
        parameter.requires_grad
        for module in (frozen.encoder, frozen.high_model, frozen.low_model)
        for parameter in module.parameters()
    )
    payload = {
        "n_demo": n_demo,
        "seed": seed,
        "hierarchy_checkpoint": str(frozen.checkpoint_path),
        "horizon_steps": frozen.horizon_steps,
        "update_period": frozen.update_period,
        "condition_dim": condition_dim,
        "frozen_parameter_count": frozen_parameter_count,
        "frozen_trainable_tensors": trainable_frozen,
        "initial_residual_abs_max": float(residual.abs().max()),
        "local_reset_available": False,
        "training_rollout_mode": "full_hierarchy_segment_reward",
        **threshold,
    }
    write_json(result / "audit.json", payload)
    with (artifact / "config.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["key", "value"])
        writer.writerows(payload.items())
    return result / "audit.json"


def _save_checkpoint(
    path: Path,
    agent: ResidualActorCritic | DirectLowActorCritic,
    optimizer: torch.optim.Optimizer,
    global_step: int,
    recipe: dict[str, Any],
    history: list[dict[str, Any]],
) -> None:
    torch.save(
        {
            "agent": agent.state_dict(),
            "condition_dim": agent.condition_dim,
            "action_dim": agent.action_dim,
            "width": agent.width,
            "depth": agent.depth,
            "optimizer": optimizer.state_dict(),
            "global_step": global_step,
            "recipe": recipe,
            "history": history,
        },
        path,
    )


def _load_residual(path: Path, device: torch.device) -> tuple[ResidualActorCritic, dict[str, Any]]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    agent = ResidualActorCritic(
        int(checkpoint["condition_dim"]),
        int(checkpoint["action_dim"]),
        int(checkpoint["width"]),
        int(checkpoint["depth"]),
    ).to(device)
    agent.load_state_dict(checkpoint["agent"])
    agent.eval()
    return agent, checkpoint


def _load_direct(
    path: Path,
    frozen: FrozenHierarchy,
    device: torch.device,
) -> tuple[DirectLowActorCritic, dict[str, Any]]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    agent = DirectLowActorCritic(
        frozen.low_model,
        frozen.action_norm.mean,
        frozen.action_norm.std,
        int(checkpoint["condition_dim"]),
        int(checkpoint["width"]),
        int(checkpoint["depth"]),
    ).to(device)
    agent.load_state_dict(checkpoint["agent"])
    agent.eval()
    return agent, checkpoint


@torch.inference_mode()
def evaluate_residual_rl(
    config: Config,
    n_demo: int,
    seed: int,
    run_name: str,
    episodes: int,
    seed_start: int,
    candidate: str = VAE_CANDIDATE,
    checkpoint_path: Path | None = None,
    ensemble_checkpoint_paths: list[Path] | None = None,
    distance_metric: str = "raw_l2",
    reachability_checkpoint_path: Path | None = None,
    residual_l2_gate_max: float | None = None,
    selected_distance_gate_max: float | None = None,
    initial_selector_weights: list[float] | None = None,
    initial_selector_mean: list[float] | None = None,
    initial_selector_std: list[float] | None = None,
    initial_selector_threshold: float | None = None,
    force: bool = False,
) -> Path:
    if checkpoint_path is not None and ensemble_checkpoint_paths:
        raise ValueError("Use either checkpoint_path or ensemble_checkpoint_paths, not both")
    if residual_l2_gate_max is not None and residual_l2_gate_max < 0.0:
        raise ValueError("residual_l2_gate_max must be non-negative")
    if selected_distance_gate_max is not None and selected_distance_gate_max < 0.0:
        raise ValueError("selected_distance_gate_max must be non-negative")
    selector_parts = [
        initial_selector_weights,
        initial_selector_mean,
        initial_selector_std,
        initial_selector_threshold,
    ]
    has_initial_selector = any(part is not None for part in selector_parts)
    if has_initial_selector and not all(part is not None for part in selector_parts):
        raise ValueError(
            "initial selector requires weights, mean, std, and threshold"
        )
    selector_weights_np: np.ndarray | None = None
    selector_mean_np: np.ndarray | None = None
    selector_std_np: np.ndarray | None = None
    if has_initial_selector:
        if (
            initial_selector_weights is None
            or initial_selector_mean is None
            or initial_selector_std is None
            or initial_selector_threshold is None
        ):
            raise RuntimeError("Initial selector validation failed")
        if (
            len(initial_selector_weights) != 3
            or len(initial_selector_mean) != 3
            or len(initial_selector_std) != 3
        ):
            raise ValueError(
                "initial selector weights, mean, and std must each have three values"
            )
        selector_weights_np = np.asarray(initial_selector_weights, dtype=np.float32)
        selector_mean_np = np.asarray(initial_selector_mean, dtype=np.float32)
        selector_std_np = np.asarray(initial_selector_std, dtype=np.float32)
        if np.any(selector_std_np <= 0.0):
            raise ValueError("initial selector std values must be positive")
    artifact, result = _paths(config, n_demo, seed, run_name, candidate)
    output = result / f"eval_{episodes}_seed{seed_start}.json"
    if output.exists() and not force:
        return output
    device = default_device()
    frozen = _load_frozen(config, n_demo, seed, device, candidate)
    residual_agent: ResidualActorCritic | None = None
    direct_agent: DirectLowActorCritic | None = None
    direct_ensemble: list[DirectLowActorCritic] = []
    alpha = 0.0
    global_step = 0
    if ensemble_checkpoint_paths:
        for index, ensemble_path in enumerate(ensemble_checkpoint_paths):
            if not ensemble_path.exists():
                raise FileNotFoundError(f"Ensemble checkpoint not found: {ensemble_path}")
            raw_checkpoint = torch.load(ensemble_path, map_location="cpu", weights_only=False)
            recipe = raw_checkpoint.get("recipe", {})
            method = str(recipe.get("method", "r1_residual_deterministic"))
            if method != "r3_direct_last_layer":
                raise ValueError(
                    "Low-level ensembles currently support only r3_direct_last_layer "
                    f"checkpoints, got {method} for {ensemble_path}"
                )
            candidate_metric = str(recipe.get("distance_metric", "raw_l2"))
            raw_reachability_checkpoint = recipe.get("reachability_checkpoint")
            candidate_reachability_path = (
                Path(str(raw_reachability_checkpoint))
                if raw_reachability_checkpoint is not None
                else None
            )
            if index == 0:
                distance_metric = candidate_metric
                reachability_checkpoint_path = candidate_reachability_path
            elif (
                candidate_metric != distance_metric
                or candidate_reachability_path != reachability_checkpoint_path
            ):
                raise ValueError("All ensemble checkpoints must use the same distance metric")
            agent, checkpoint = _load_direct(ensemble_path, frozen, device)
            direct_ensemble.append(agent)
            global_step = max(global_step, int(checkpoint["global_step"]))
    else:
        checkpoint_path = checkpoint_path or artifact / "latest.pt"
    if not direct_ensemble and checkpoint_path is not None and checkpoint_path.exists():
        raw_checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        recipe = raw_checkpoint.get("recipe", {})
        method = str(recipe.get("method", "r1_residual_deterministic"))
        distance_metric = str(recipe.get("distance_metric", "raw_l2"))
        raw_reachability_checkpoint = recipe.get("reachability_checkpoint")
        if raw_reachability_checkpoint is not None:
            reachability_checkpoint_path = Path(str(raw_reachability_checkpoint))
        if method == "r3_direct_last_layer":
            direct_agent, checkpoint = _load_direct(checkpoint_path, frozen, device)
        else:
            residual_agent, checkpoint = _load_residual(checkpoint_path, device)
            alpha = float(checkpoint["recipe"]["alpha"])
        global_step = int(checkpoint["global_step"])
    elif distance_metric == "reachability" and reachability_checkpoint_path is None:
        reachability_checkpoint_path = (
            _default_reachability_checkpoint(config, n_demo, seed)
            if candidate == VAE_CANDIDATE
            else _candidate_reachability_checkpoint(config, candidate, seed)
        )
    num_envs = min(int(config.get("low_level_rl.eval_num_envs", 32)), episodes)
    rollout = HierarchyRollout(
        config,
        frozen,
        num_envs,
        seed_start,
        device,
        distance_metric=distance_metric,
        reachability_checkpoint_path=reachability_checkpoint_path,
    )
    threshold = _teacher_goal_threshold(config, n_demo, seed, candidate)["goal_threshold"]
    successes: list[float] = []
    finals: list[float] = []
    maxima: list[float] = []
    initial_distances: list[float] = []
    final_distances: list[float] = []
    raw_initial_distances: list[float] = []
    raw_final_distances: list[float] = []
    reached: list[float] = []
    selected_metric_terminal_scores: list[float] = []
    saturation = 0
    action_count = 0
    residual_magnitudes: list[float] = []
    episode_residual_l2_mean: list[float] = []
    episode_action_saturation_rate: list[float] = []
    episode_raw_distance_reduction_mean: list[float] = []
    episode_selected_distance_reduction_mean: list[float] = []
    episode_goal_reach_rate: list[float] = []
    episode_selected_distance_mean: list[float] = []
    episode_raw_distance_mean: list[float] = []
    episode_base_action_l2_mean: list[float] = []
    episode_previous_action_norm_l2_mean: list[float] = []
    episode_replan_rate: list[float] = []
    episode_initial_selected_distance: list[float] = []
    episode_initial_raw_distance: list[float] = []
    episode_initial_base_action_l2: list[float] = []
    episode_initial_env_reward: list[float] = []
    episode_selected_distance_gate_rate: list[float] = []
    episode_initial_selector_use_tuned: list[float] = []
    current_segment_initial: np.ndarray | None = None
    current_segment_raw_initial: np.ndarray | None = None
    current_final = np.zeros(num_envs, dtype=np.float32)
    current_max = np.full(num_envs, -np.inf, dtype=np.float32)
    current_residual_sum = np.zeros(num_envs, dtype=np.float32)
    current_saturation_sum = np.zeros(num_envs, dtype=np.float32)
    current_step_count = np.zeros(num_envs, dtype=np.float32)
    current_raw_reduction_sum = np.zeros(num_envs, dtype=np.float32)
    current_selected_reduction_sum = np.zeros(num_envs, dtype=np.float32)
    current_reach_sum = np.zeros(num_envs, dtype=np.float32)
    current_segment_count = np.zeros(num_envs, dtype=np.float32)
    current_selected_distance_sum = np.zeros(num_envs, dtype=np.float32)
    current_raw_distance_sum = np.zeros(num_envs, dtype=np.float32)
    current_base_action_l2_sum = np.zeros(num_envs, dtype=np.float32)
    current_previous_action_norm_l2_sum = np.zeros(num_envs, dtype=np.float32)
    current_replan_sum = np.zeros(num_envs, dtype=np.float32)
    current_initial_selected_distance = np.full(num_envs, np.nan, dtype=np.float32)
    current_initial_raw_distance = np.full(num_envs, np.nan, dtype=np.float32)
    current_initial_base_action_l2 = np.full(num_envs, np.nan, dtype=np.float32)
    current_initial_env_reward = np.full(num_envs, np.nan, dtype=np.float32)
    current_distance_gate_sum = np.zeros(num_envs, dtype=np.float32)
    current_initial_selector_use_tuned = np.ones(num_envs, dtype=bool)
    distance_gate_count = 0
    initial_selector_tuned_count = 0
    initial_selector_episode_count = 0
    while len(successes) < episodes:
        condition, base_action, distance, replan = rollout.condition()
        raw_distance = rollout.raw_distance(rollout.current_latent, rollout.held_goal)
        base_action_l2 = (
            torch.linalg.vector_norm(base_action, dim=-1).cpu().numpy().astype(np.float32)
        )
        previous_action_norm_l2 = np.linalg.norm(rollout.previous_action, axis=-1).astype(
            np.float32
        )
        first_step = current_step_count == 0.0
        if np.any(first_step):
            current_initial_selected_distance[first_step] = distance[first_step]
            current_initial_raw_distance[first_step] = raw_distance[first_step]
            current_initial_base_action_l2[first_step] = base_action_l2[first_step]
            current_initial_env_reward[first_step] = rollout.previous_env_reward[first_step]
            if has_initial_selector:
                if (
                    selector_weights_np is None
                    or selector_mean_np is None
                    or selector_std_np is None
                    or initial_selector_threshold is None
                ):
                    raise RuntimeError("Initial selector was not initialized")
                selector_features = np.stack(
                    [
                        distance[first_step],
                        raw_distance[first_step],
                        base_action_l2[first_step],
                    ],
                    axis=1,
                )
                selector_scores = (
                    (selector_features - selector_mean_np) / selector_std_np
                ) @ selector_weights_np
                selected_tuned = selector_scores >= initial_selector_threshold
                current_initial_selector_use_tuned[first_step] = selected_tuned
                initial_selector_tuned_count += int(selected_tuned.sum())
                initial_selector_episode_count += int(first_step.sum())
        current_selected_distance_sum += distance
        current_raw_distance_sum += raw_distance
        current_base_action_l2_sum += base_action_l2
        current_previous_action_norm_l2_sum += previous_action_norm_l2
        current_replan_sum += replan.astype(np.float32)
        if current_segment_initial is None:
            current_segment_initial = distance.copy()
            current_segment_raw_initial = raw_distance.copy()
        else:
            current_segment_initial[replan] = distance[replan]
            if current_segment_raw_initial is None:
                raise RuntimeError("Raw segment distance state was not initialized")
            current_segment_raw_initial[replan] = raw_distance[replan]
        if direct_ensemble:
            unclipped = torch.stack(
                [
                    agent.get_action_and_value(condition, deterministic=True)[0]
                    for agent in direct_ensemble
                ],
                dim=0,
            ).mean(dim=0)
            residual = unclipped - base_action
        elif direct_agent is not None:
            unclipped = direct_agent.get_action_and_value(condition, deterministic=True)[0]
            residual = unclipped - base_action
        else:
            raw_residual = (
                residual_agent.get_action_and_value(condition, deterministic=True)[0]
                if residual_agent is not None
                else torch.zeros_like(base_action)
            )
            residual = alpha * torch.tanh(raw_residual)
            unclipped = base_action + residual
        if has_initial_selector:
            use_base_np = ~current_initial_selector_use_tuned
            if bool(np.any(use_base_np)):
                use_base = torch.from_numpy(use_base_np).to(device)
                unclipped = torch.where(use_base[:, None], base_action, unclipped)
                residual = torch.where(use_base[:, None], torch.zeros_like(residual), residual)
        distance_gate_np = np.zeros(num_envs, dtype=bool)
        if residual_l2_gate_max is not None:
            residual_norm_before_gate = torch.linalg.vector_norm(residual, dim=-1)
            use_base = residual_norm_before_gate > residual_l2_gate_max
            if bool(use_base.any()):
                unclipped = torch.where(use_base[:, None], base_action, unclipped)
                residual = torch.where(use_base[:, None], torch.zeros_like(residual), residual)
        if selected_distance_gate_max is not None:
            distance_gate_np = distance > selected_distance_gate_max
            if bool(np.any(distance_gate_np)):
                use_base = torch.from_numpy(distance_gate_np).to(device)
                unclipped = torch.where(use_base[:, None], base_action, unclipped)
                residual = torch.where(use_base[:, None], torch.zeros_like(residual), residual)
        current_distance_gate_sum += distance_gate_np.astype(np.float32)
        distance_gate_count += int(distance_gate_np.sum())
        action = torch.clamp(unclipped, rollout.action_low, rollout.action_high)
        saturated = torch.any(unclipped != action, dim=-1)
        saturation += int(saturated.sum().cpu())
        action_count += num_envs
        residual_norm = torch.linalg.vector_norm(residual, dim=-1).cpu().numpy().astype(np.float32)
        residual_magnitudes.extend(residual_norm.tolist())
        current_residual_sum += residual_norm
        current_saturation_sum += saturated.cpu().numpy().astype(np.float32)
        current_step_count += 1.0
        _reward, _done, metrics = rollout.step(
            action,
            distance,
            terminal_weight=0.0,
            distance_progress_weight=1.0,
            task_reward_weight=0.0,
            task_progress_weight=0.0,
            residual_penalty=np.zeros(num_envs, dtype=np.float32),
        )
        segment_end = metrics["segment_end"]
        if np.any(segment_end):
            if current_segment_raw_initial is None:
                raise RuntimeError("Raw segment distance state was not initialized")
            initial_distances.extend(current_segment_initial[segment_end].tolist())
            final_distances.extend(metrics["next_distance"][segment_end].tolist())
            raw_initial_distances.extend(
                current_segment_raw_initial[segment_end].tolist()
            )
            raw_final = metrics["raw_next_distance"][segment_end]
            raw_final_distances.extend(raw_final.tolist())
            reached_np = (raw_final < threshold).astype(np.float32)
            reached.extend(reached_np.tolist())
            selected_metric_terminal_scores.extend(
                (-metrics["next_distance"][segment_end]).tolist()
            )
            current_raw_reduction_sum[segment_end] += (
                current_segment_raw_initial[segment_end] - raw_final
            )
            current_selected_reduction_sum[segment_end] += (
                current_segment_initial[segment_end] - metrics["next_distance"][segment_end]
            )
            current_reach_sum[segment_end] += reached_np
            current_segment_count[segment_end] += 1.0
            current_segment_initial[segment_end] = metrics["next_distance"][segment_end]
            current_segment_raw_initial[segment_end] = raw_final
        info = metrics["info"]
        current_final = metrics["env_reward"].astype(np.float32)
        current_max = np.maximum(current_max, current_final)
        if "final_info" in info:
            mask = info["_final_info"]
            if bool(mask.any()):
                episode = info["final_info"]["episode"]
                successes.extend(episode["success_once"][mask].float().cpu().numpy().tolist())
                mask_np = mask.detach().cpu().numpy().astype(bool)
                finals.extend(current_final[mask_np].tolist())
                maxima.extend(current_max[mask_np].tolist())
                step_denominator = np.maximum(current_step_count[mask_np], 1.0)
                segment_denominator = np.maximum(current_segment_count[mask_np], 1.0)
                episode_residual_l2_mean.extend(
                    (current_residual_sum[mask_np] / step_denominator).tolist()
                )
                episode_action_saturation_rate.extend(
                    (current_saturation_sum[mask_np] / step_denominator).tolist()
                )
                episode_raw_distance_reduction_mean.extend(
                    (current_raw_reduction_sum[mask_np] / segment_denominator).tolist()
                )
                episode_selected_distance_reduction_mean.extend(
                    (current_selected_reduction_sum[mask_np] / segment_denominator).tolist()
                )
                episode_goal_reach_rate.extend(
                    (current_reach_sum[mask_np] / segment_denominator).tolist()
                )
                episode_selected_distance_mean.extend(
                    (current_selected_distance_sum[mask_np] / step_denominator).tolist()
                )
                episode_raw_distance_mean.extend(
                    (current_raw_distance_sum[mask_np] / step_denominator).tolist()
                )
                episode_base_action_l2_mean.extend(
                    (current_base_action_l2_sum[mask_np] / step_denominator).tolist()
                )
                episode_previous_action_norm_l2_mean.extend(
                    (
                        current_previous_action_norm_l2_sum[mask_np] / step_denominator
                    ).tolist()
                )
                episode_replan_rate.extend(
                    (current_replan_sum[mask_np] / step_denominator).tolist()
                )
                episode_initial_selected_distance.extend(
                    current_initial_selected_distance[mask_np].tolist()
                )
                episode_initial_raw_distance.extend(
                    current_initial_raw_distance[mask_np].tolist()
                )
                episode_initial_base_action_l2.extend(
                    current_initial_base_action_l2[mask_np].tolist()
                )
                episode_initial_env_reward.extend(current_initial_env_reward[mask_np].tolist())
                episode_selected_distance_gate_rate.extend(
                    (current_distance_gate_sum[mask_np] / step_denominator).tolist()
                )
                episode_initial_selector_use_tuned.extend(
                    current_initial_selector_use_tuned[mask_np].astype(np.float32).tolist()
                )
                current_max[mask_np] = -np.inf
                current_residual_sum[mask_np] = 0.0
                current_saturation_sum[mask_np] = 0.0
                current_step_count[mask_np] = 0.0
                current_raw_reduction_sum[mask_np] = 0.0
                current_selected_reduction_sum[mask_np] = 0.0
                current_reach_sum[mask_np] = 0.0
                current_segment_count[mask_np] = 0.0
                current_selected_distance_sum[mask_np] = 0.0
                current_raw_distance_sum[mask_np] = 0.0
                current_base_action_l2_sum[mask_np] = 0.0
                current_previous_action_norm_l2_sum[mask_np] = 0.0
                current_replan_sum[mask_np] = 0.0
                current_initial_selected_distance[mask_np] = np.nan
                current_initial_raw_distance[mask_np] = np.nan
                current_initial_base_action_l2[mask_np] = np.nan
                current_initial_env_reward[mask_np] = np.nan
                current_distance_gate_sum[mask_np] = 0.0
                current_initial_selector_use_tuned[mask_np] = True
        if len(successes) >= episodes:
            break
    rollout.close()
    count = min(len(successes), episodes)
    initial_np = np.asarray(initial_distances, dtype=np.float32)
    final_np = np.asarray(final_distances, dtype=np.float32)
    raw_initial_np = np.asarray(raw_initial_distances, dtype=np.float32)
    raw_final_np = np.asarray(raw_final_distances, dtype=np.float32)
    reached_np = np.asarray(reached, dtype=np.float32)
    selected_scores_np = np.asarray(selected_metric_terminal_scores, dtype=np.float32)
    payload = {
        "n_demo": n_demo,
        "candidate": candidate,
        "seed": seed,
        "run_name": run_name,
        "checkpoint": str(checkpoint_path)
        if (residual_agent is not None or direct_agent is not None)
        else None,
        "ensemble_checkpoints": [str(path) for path in ensemble_checkpoint_paths]
        if ensemble_checkpoint_paths
        else None,
        "residual_l2_gate_max": residual_l2_gate_max,
        "selected_distance_gate_max": selected_distance_gate_max,
        "initial_selector_feature_order": [
            "initial_selected_distance",
            "initial_raw_distance",
            "initial_base_action_l2",
        ]
        if has_initial_selector
        else None,
        "initial_selector_weights": initial_selector_weights,
        "initial_selector_mean": initial_selector_mean,
        "initial_selector_std": initial_selector_std,
        "initial_selector_threshold": initial_selector_threshold,
        "distance_metric": distance_metric,
        "reachability_checkpoint": str(reachability_checkpoint_path)
        if reachability_checkpoint_path is not None
        else None,
        "rl_steps": global_step,
        "episodes": count,
        "seed_start": seed_start,
        "eval_mode": "vector_auto_reset_unpaired",
        "episode_seed": None,
        "success": float(np.mean(successes[:count])),
        "final_reward": float(np.mean(finals[:count])),
        "max_reward": float(np.mean(maxima[:count])),
        "segment_initial_distance": float(initial_np.mean()),
        "segment_final_distance": float(final_np.mean()),
        "segment_distance_reduction": float((initial_np - final_np).mean()),
        "raw_segment_initial_distance": float(raw_initial_np.mean()),
        "raw_segment_final_distance": float(raw_final_np.mean()),
        "raw_segment_distance_reduction": float((raw_initial_np - raw_final_np).mean()),
        "segment_goal_reach_rate": float(np.mean(reached)) if reached else None,
        "selected_metric_terminal_reach_auc": _binary_auc(
            selected_scores_np, reached_np
        ),
        "goal_threshold": threshold,
        "action_saturation_rate": saturation / max(action_count, 1),
        "residual_l2_mean": float(np.mean(residual_magnitudes)),
        "selected_distance_gate_rate": distance_gate_count / max(action_count, 1),
        "initial_selector_tuned_rate": initial_selector_tuned_count
        / max(initial_selector_episode_count, 1)
        if has_initial_selector
        else None,
        "episode_success": successes[:count],
        "episode_final_reward": finals[:count],
        "episode_max_reward": maxima[:count],
        "episode_residual_l2_mean": episode_residual_l2_mean[:count],
        "episode_action_saturation_rate": episode_action_saturation_rate[:count],
        "episode_raw_segment_distance_reduction": episode_raw_distance_reduction_mean[:count],
        "episode_segment_distance_reduction": episode_selected_distance_reduction_mean[:count],
        "episode_segment_goal_reach_rate": episode_goal_reach_rate[:count],
        "episode_selected_distance_mean": episode_selected_distance_mean[:count],
        "episode_raw_distance_mean": episode_raw_distance_mean[:count],
        "episode_base_action_l2_mean": episode_base_action_l2_mean[:count],
        "episode_previous_action_norm_l2_mean": episode_previous_action_norm_l2_mean[:count],
        "episode_replan_rate": episode_replan_rate[:count],
        "episode_initial_selected_distance": episode_initial_selected_distance[:count],
        "episode_initial_raw_distance": episode_initial_raw_distance[:count],
        "episode_initial_base_action_l2": episode_initial_base_action_l2[:count],
        "episode_initial_env_reward": episode_initial_env_reward[:count],
        "episode_selected_distance_gate_rate": episode_selected_distance_gate_rate[:count],
        "episode_initial_selector_use_tuned": episode_initial_selector_use_tuned[:count],
    }
    write_json(output, payload)
    return output


def _validate_initial_selector(
    initial_selector_weights: list[float] | None,
    initial_selector_mean: list[float] | None,
    initial_selector_std: list[float] | None,
    initial_selector_threshold: float | None,
) -> tuple[bool, np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    selector_parts = [
        initial_selector_weights,
        initial_selector_mean,
        initial_selector_std,
        initial_selector_threshold,
    ]
    has_initial_selector = any(part is not None for part in selector_parts)
    if has_initial_selector and not all(part is not None for part in selector_parts):
        raise ValueError("initial selector requires weights, mean, std, and threshold")
    if not has_initial_selector:
        return False, None, None, None
    if (
        initial_selector_weights is None
        or initial_selector_mean is None
        or initial_selector_std is None
    ):
        raise RuntimeError("Initial selector validation failed")
    if (
        len(initial_selector_weights) != 3
        or len(initial_selector_mean) != 3
        or len(initial_selector_std) != 3
    ):
        raise ValueError("initial selector weights, mean, and std must each have three values")
    selector_weights_np = np.asarray(initial_selector_weights, dtype=np.float32)
    selector_mean_np = np.asarray(initial_selector_mean, dtype=np.float32)
    selector_std_np = np.asarray(initial_selector_std, dtype=np.float32)
    if np.any(selector_std_np <= 0.0):
        raise ValueError("initial selector std values must be positive")
    return has_initial_selector, selector_weights_np, selector_mean_np, selector_std_np


@torch.inference_mode()
def evaluate_residual_rl_serial(
    config: Config,
    n_demo: int,
    seed: int,
    run_name: str,
    episodes: int,
    seed_start: int,
    candidate: str = VAE_CANDIDATE,
    checkpoint_path: Path | None = None,
    distance_metric: str = "raw_l2",
    reachability_checkpoint_path: Path | None = None,
    residual_l2_gate_max: float | None = None,
    selected_distance_gate_max: float | None = None,
    initial_selector_weights: list[float] | None = None,
    initial_selector_mean: list[float] | None = None,
    initial_selector_std: list[float] | None = None,
    initial_selector_threshold: float | None = None,
    force: bool = False,
) -> Path:
    if residual_l2_gate_max is not None and residual_l2_gate_max < 0.0:
        raise ValueError("residual_l2_gate_max must be non-negative")
    if selected_distance_gate_max is not None and selected_distance_gate_max < 0.0:
        raise ValueError("selected_distance_gate_max must be non-negative")
    (
        has_initial_selector,
        selector_weights_np,
        selector_mean_np,
        selector_std_np,
    ) = _validate_initial_selector(
        initial_selector_weights,
        initial_selector_mean,
        initial_selector_std,
        initial_selector_threshold,
    )
    artifact, result = _paths(config, n_demo, seed, run_name, candidate)
    output = result / f"serial_eval_{episodes}_seed{seed_start}.json"
    if output.exists() and not force:
        return output

    device = default_device()
    frozen = _load_frozen(config, n_demo, seed, device, candidate)
    residual_agent: ResidualActorCritic | None = None
    direct_agent: DirectLowActorCritic | None = None
    alpha = 0.0
    global_step = 0
    checkpoint_path = checkpoint_path or artifact / "latest.pt"
    if checkpoint_path is not None and checkpoint_path.exists():
        raw_checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        recipe = raw_checkpoint.get("recipe", {})
        method = str(recipe.get("method", "r1_residual_deterministic"))
        distance_metric = str(recipe.get("distance_metric", "raw_l2"))
        raw_reachability_checkpoint = recipe.get("reachability_checkpoint")
        if raw_reachability_checkpoint is not None:
            reachability_checkpoint_path = Path(str(raw_reachability_checkpoint))
        if method == "r3_direct_last_layer":
            direct_agent, checkpoint = _load_direct(checkpoint_path, frozen, device)
        else:
            residual_agent, checkpoint = _load_residual(checkpoint_path, device)
            alpha = float(checkpoint["recipe"]["alpha"])
        global_step = int(checkpoint["global_step"])
    elif distance_metric == "reachability" and reachability_checkpoint_path is None:
        reachability_checkpoint_path = (
            _default_reachability_checkpoint(config, n_demo, seed)
            if candidate == VAE_CANDIDATE
            else _candidate_reachability_checkpoint(config, candidate, seed)
        )

    reachability_model: ReachabilityDistance | None = None
    reachability_goal_norm: Standardizer | None = None
    if distance_metric == "reachability":
        if reachability_checkpoint_path is None:
            raise ValueError("reachability distance metric requires a checkpoint")
        reachability_model, reachability_goal_norm, _checkpoint = load_reachability_distance(
            reachability_checkpoint_path, device
        )
        reachability_model.requires_grad_(False)
    elif distance_metric != "raw_l2":
        raise ValueError(f"Unknown low-level distance metric: {distance_metric}")

    def selected_distance(current_latent: np.ndarray, goal_latent: np.ndarray) -> np.ndarray:
        if distance_metric == "raw_l2":
            return HierarchyRollout.raw_distance(current_latent, goal_latent)
        if reachability_model is None or reachability_goal_norm is None:
            raise RuntimeError("Reachability distance model was not initialized")
        return _reachability_distance_values(
            reachability_model,
            reachability_goal_norm,
            frozen,
            current_latent,
            goal_latent,
            device,
        )

    env = _visual_env(config, 1)
    action_low_np = np.asarray(env.single_action_space.low, dtype=np.float32)
    action_high_np = np.asarray(env.single_action_space.high, dtype=np.float32)
    action_low = torch.as_tensor(action_low_np, device=device)
    action_high = torch.as_tensor(action_high_np, device=device)
    dino = _phase4_dino_from_config(config, device)
    zero_previous = frozen.action_norm.transform(np.zeros((1, 3), dtype=np.float32))
    max_steps = int(config.get("env_max_episode_steps", 100))
    threshold = _teacher_goal_threshold(config, n_demo, seed, candidate)["goal_threshold"]

    episode_seed: list[int] = []
    successes: list[float] = []
    finals: list[float] = []
    maxima: list[float] = []
    steps_out: list[int] = []
    episode_initial_selected_distance: list[float] = []
    episode_initial_raw_distance: list[float] = []
    episode_initial_base_action_l2: list[float] = []
    episode_initial_selector_use_tuned: list[float] = []
    episode_residual_l2_mean: list[float] = []
    episode_action_saturation_rate: list[float] = []
    episode_selected_distance_gate_rate: list[float] = []
    episode_raw_segment_distance_reduction: list[float] = []
    episode_segment_distance_reduction: list[float] = []
    episode_segment_goal_reach_rate: list[float] = []
    segment_initial_distances: list[float] = []
    segment_final_distances: list[float] = []
    raw_segment_initial_distances: list[float] = []
    raw_segment_final_distances: list[float] = []
    serial_segment_episode_seed: list[int] = []
    serial_segment_index: list[int] = []
    serial_segment_start_step: list[int] = []
    serial_segment_initial_selected_distance: list[float] = []
    serial_segment_initial_raw_distance: list[float] = []
    serial_segment_initial_base_action_l2: list[float] = []
    serial_segment_initial_previous_action_norm_l2: list[float] = []
    serial_segment_final_selected_distance: list[float] = []
    serial_segment_final_raw_distance: list[float] = []
    serial_segment_selected_distance_reduction: list[float] = []
    serial_segment_raw_distance_reduction: list[float] = []
    serial_segment_goal_reached: list[float] = []
    serial_segment_residual_l2_mean: list[float] = []
    serial_segment_action_saturation_rate: list[float] = []
    serial_segment_distance_gate_rate: list[float] = []
    reached: list[float] = []
    selected_metric_terminal_scores: list[float] = []
    total_saturation = 0
    total_actions = 0
    total_distance_gate = 0
    total_selector_tuned = 0
    residual_magnitudes: list[float] = []

    try:
        for episode_index in trange(episodes, desc=f"serial low RL {run_name}"):
            rollout_seed = seed_start + episode_index
            obs, _info = env.reset(seed=[rollout_seed])
            frames = _phase4_frame_inputs(obs, dino, int(config.get("dino.batch_size", 64)))
            normalized_frames = frozen.frame_norm.transform(frames)
            anchor_frames = frames.copy()
            current_latent = _encode_effect_progress(frozen, anchor_frames, frames, device)
            previous_action = zero_previous.copy()
            held_goal = np.zeros((1, frozen.goal_dim), dtype=np.float32)
            countdown = 0
            selected_tuned = True
            selected_at_start = False
            final_reward = 0.0
            max_reward = -float("inf")
            success = False
            residual_sum = 0.0
            saturation_sum = 0.0
            distance_gate_sum = 0.0
            raw_reduction_sum = 0.0
            selected_reduction_sum = 0.0
            reach_sum = 0.0
            segment_count = 0
            segment_initial = np.zeros(1, dtype=np.float32)
            segment_raw_initial = np.zeros(1, dtype=np.float32)
            segment_start_step = 0
            segment_start_base_action_l2 = 0.0
            segment_start_previous_action_norm_l2 = 0.0
            segment_residual_sum = 0.0
            segment_saturation_sum = 0.0
            segment_distance_gate_sum = 0.0
            segment_step_count = 0
            step_count = 0

            for _step in range(max_steps):
                replan = countdown <= 0
                if replan:
                    high_condition = np.concatenate(
                        [normalized_frames, previous_action], axis=-1
                    )
                    held_goal = (
                        frozen.high_model(
                            torch.from_numpy(high_condition).to(device).float()
                        )
                        .cpu()
                        .numpy()
                    )
                    anchor_frames = frames.copy()
                    current_latent = _encode_effect_progress(
                        frozen, anchor_frames, frames, device
                    )
                    countdown = frozen.update_period
                remaining = np.asarray(
                    [[max(countdown, 1) / frozen.horizon_steps]], dtype=np.float32
                )
                condition_np = _low_condition_array(
                    normalized_frames,
                    current_latent,
                    held_goal,
                    previous_action,
                    remaining,
                    frozen.conditioning,
                )
                condition = torch.from_numpy(condition_np).to(device).float()
                normalized_base = frozen.low_model(condition)
                base_action = torch.from_numpy(
                    frozen.action_norm.inverse(normalized_base.cpu().numpy())
                ).to(device)
                distance = selected_distance(current_latent, held_goal)
                raw_distance = HierarchyRollout.raw_distance(current_latent, held_goal)
                base_action_l2 = float(torch.linalg.vector_norm(base_action, dim=-1).cpu()[0])
                if step_count == 0:
                    episode_initial_selected_distance.append(float(distance[0]))
                    episode_initial_raw_distance.append(float(raw_distance[0]))
                    episode_initial_base_action_l2.append(base_action_l2)
                    if has_initial_selector:
                        if (
                            selector_weights_np is None
                            or selector_mean_np is None
                            or selector_std_np is None
                            or initial_selector_threshold is None
                        ):
                            raise RuntimeError("Initial selector was not initialized")
                        selector_features = np.asarray(
                            [[distance[0], raw_distance[0], base_action_l2]],
                            dtype=np.float32,
                        )
                        selector_score = float(
                            (
                                ((selector_features - selector_mean_np) / selector_std_np)
                                @ selector_weights_np
                            )[0]
                        )
                        selected_tuned = selector_score >= initial_selector_threshold
                    selected_at_start = selected_tuned
                    if has_initial_selector:
                        total_selector_tuned += int(selected_tuned)
                if replan:
                    segment_initial = distance.copy()
                    segment_raw_initial = raw_distance.copy()
                    segment_start_step = step_count
                    segment_start_base_action_l2 = base_action_l2
                    segment_start_previous_action_norm_l2 = float(
                        np.linalg.norm(previous_action, axis=-1)[0]
                    )
                    segment_residual_sum = 0.0
                    segment_saturation_sum = 0.0
                    segment_distance_gate_sum = 0.0
                    segment_step_count = 0

                if direct_agent is not None:
                    unclipped = direct_agent.get_action_and_value(
                        condition, deterministic=True
                    )[0]
                    residual = unclipped - base_action
                else:
                    raw_residual = (
                        residual_agent.get_action_and_value(condition, deterministic=True)[0]
                        if residual_agent is not None
                        else torch.zeros_like(base_action)
                    )
                    residual = alpha * torch.tanh(raw_residual)
                    unclipped = base_action + residual
                if has_initial_selector and not selected_tuned:
                    unclipped = base_action
                    residual = torch.zeros_like(residual)
                if residual_l2_gate_max is not None:
                    residual_norm_before_gate = float(
                        torch.linalg.vector_norm(residual, dim=-1).cpu()[0]
                    )
                    if residual_norm_before_gate > residual_l2_gate_max:
                        unclipped = base_action
                        residual = torch.zeros_like(residual)
                if selected_distance_gate_max is not None and float(distance[0]) > (
                    selected_distance_gate_max
                ):
                    unclipped = base_action
                    residual = torch.zeros_like(residual)
                    distance_gate_sum += 1.0
                    segment_distance_gate_sum += 1.0
                    total_distance_gate += 1

                action = torch.clamp(unclipped, action_low, action_high)
                saturated = bool(torch.any(unclipped != action).cpu())
                saturation_sum += float(saturated)
                segment_saturation_sum += float(saturated)
                total_saturation += int(saturated)
                total_actions += 1
                residual_norm = float(torch.linalg.vector_norm(residual, dim=-1).cpu()[0])
                residual_sum += residual_norm
                segment_residual_sum += residual_norm
                residual_magnitudes.append(residual_norm)
                segment_step_count += 1

                next_obs, reward, terminated, truncated, info = env.step(action)
                next_frames = _phase4_frame_inputs(
                    next_obs, dino, int(config.get("dino.batch_size", 64))
                )
                next_latent = _encode_effect_progress(
                    frozen, anchor_frames, next_frames, device
                )
                next_distance = selected_distance(next_latent, held_goal)
                raw_next_distance = HierarchyRollout.raw_distance(next_latent, held_goal)
                if countdown == 1:
                    segment_initial_distances.append(float(segment_initial[0]))
                    segment_final_distances.append(float(next_distance[0]))
                    raw_segment_initial_distances.append(float(segment_raw_initial[0]))
                    raw_segment_final_distances.append(float(raw_next_distance[0]))
                    reached_value = float(raw_next_distance[0] < threshold)
                    reached.append(reached_value)
                    selected_metric_terminal_scores.append(float(-next_distance[0]))
                    raw_reduction_sum += float(segment_raw_initial[0] - raw_next_distance[0])
                    selected_reduction_sum += float(segment_initial[0] - next_distance[0])
                    reach_sum += reached_value
                    segment_count += 1
                    segment_denominator = max(segment_step_count, 1)
                    serial_segment_episode_seed.append(rollout_seed)
                    serial_segment_index.append(segment_count - 1)
                    serial_segment_start_step.append(segment_start_step)
                    serial_segment_initial_selected_distance.append(float(segment_initial[0]))
                    serial_segment_initial_raw_distance.append(float(segment_raw_initial[0]))
                    serial_segment_initial_base_action_l2.append(segment_start_base_action_l2)
                    serial_segment_initial_previous_action_norm_l2.append(
                        segment_start_previous_action_norm_l2
                    )
                    serial_segment_final_selected_distance.append(float(next_distance[0]))
                    serial_segment_final_raw_distance.append(float(raw_next_distance[0]))
                    serial_segment_selected_distance_reduction.append(
                        float(segment_initial[0] - next_distance[0])
                    )
                    serial_segment_raw_distance_reduction.append(
                        float(segment_raw_initial[0] - raw_next_distance[0])
                    )
                    serial_segment_goal_reached.append(reached_value)
                    serial_segment_residual_l2_mean.append(
                        segment_residual_sum / segment_denominator
                    )
                    serial_segment_action_saturation_rate.append(
                        segment_saturation_sum / segment_denominator
                    )
                    serial_segment_distance_gate_rate.append(
                        segment_distance_gate_sum / segment_denominator
                    )

                previous_action = frozen.action_norm.transform(
                    action.cpu().numpy().astype(np.float32)
                )
                countdown -= 1
                obs = next_obs
                frames = next_frames
                normalized_frames = frozen.frame_norm.transform(frames)
                current_latent = next_latent
                final_reward = float(np.asarray(reward.cpu()).reshape(-1)[0])
                max_reward = max(max_reward, final_reward)
                if "success" in info:
                    success = success or bool(np.asarray(info["success"].cpu()).reshape(-1)[0])
                final_info_done = False
                if "final_info" in info:
                    final_mask = info["_final_info"]
                    final_info_done = bool(final_mask.any())
                    if final_info_done:
                        episode = info["final_info"]["episode"]
                        success = success or bool(
                            episode["success_once"][final_mask].float().cpu().numpy()[0]
                        )
                step_count += 1
                if final_info_done or bool(
                    np.asarray(torch.logical_or(terminated, truncated).cpu()).reshape(-1)[0]
                ):
                    break

            episode_seed.append(rollout_seed)
            successes.append(float(success))
            finals.append(final_reward)
            maxima.append(max_reward)
            steps_out.append(step_count)
            step_denominator = max(step_count, 1)
            segment_denominator = max(segment_count, 1)
            episode_residual_l2_mean.append(residual_sum / step_denominator)
            episode_action_saturation_rate.append(saturation_sum / step_denominator)
            episode_selected_distance_gate_rate.append(distance_gate_sum / step_denominator)
            episode_raw_segment_distance_reduction.append(
                raw_reduction_sum / segment_denominator
            )
            episode_segment_distance_reduction.append(
                selected_reduction_sum / segment_denominator
            )
            episode_segment_goal_reach_rate.append(reach_sum / segment_denominator)
            episode_initial_selector_use_tuned.append(float(selected_at_start))
    finally:
        env.close()

    initial_np = np.asarray(segment_initial_distances, dtype=np.float32)
    final_np = np.asarray(segment_final_distances, dtype=np.float32)
    raw_initial_np = np.asarray(raw_segment_initial_distances, dtype=np.float32)
    raw_final_np = np.asarray(raw_segment_final_distances, dtype=np.float32)
    reached_np = np.asarray(reached, dtype=np.float32)
    selected_scores_np = np.asarray(selected_metric_terminal_scores, dtype=np.float32)

    def mean_or_none(values: list[float] | np.ndarray) -> float | None:
        if len(values) == 0:
            return None
        return float(np.mean(values))

    payload = {
        "n_demo": n_demo,
        "candidate": candidate,
        "seed": seed,
        "run_name": run_name,
        "checkpoint": str(checkpoint_path)
        if (residual_agent is not None or direct_agent is not None)
        else None,
        "residual_l2_gate_max": residual_l2_gate_max,
        "selected_distance_gate_max": selected_distance_gate_max,
        "initial_selector_feature_order": [
            "initial_selected_distance",
            "initial_raw_distance",
            "initial_base_action_l2",
        ]
        if has_initial_selector
        else None,
        "initial_selector_weights": initial_selector_weights,
        "initial_selector_mean": initial_selector_mean,
        "initial_selector_std": initial_selector_std,
        "initial_selector_threshold": initial_selector_threshold,
        "distance_metric": distance_metric,
        "reachability_checkpoint": str(reachability_checkpoint_path)
        if reachability_checkpoint_path is not None
        else None,
        "rl_steps": global_step,
        "episodes": episodes,
        "seed_start": seed_start,
        "eval_mode": "serial_explicit_seed",
        "success": float(np.mean(successes)),
        "final_reward": float(np.mean(finals)),
        "max_reward": float(np.mean(maxima)),
        "segment_initial_distance": mean_or_none(initial_np),
        "segment_final_distance": mean_or_none(final_np),
        "segment_distance_reduction": mean_or_none(initial_np - final_np),
        "raw_segment_initial_distance": mean_or_none(raw_initial_np),
        "raw_segment_final_distance": mean_or_none(raw_final_np),
        "raw_segment_distance_reduction": mean_or_none(raw_initial_np - raw_final_np),
        "segment_goal_reach_rate": mean_or_none(reached),
        "selected_metric_terminal_reach_auc": _binary_auc(
            selected_scores_np, reached_np
        )
        if len(reached_np) > 0
        else None,
        "goal_threshold": threshold,
        "action_saturation_rate": total_saturation / max(total_actions, 1),
        "residual_l2_mean": float(np.mean(residual_magnitudes))
        if residual_magnitudes
        else 0.0,
        "selected_distance_gate_rate": total_distance_gate / max(total_actions, 1),
        "initial_selector_tuned_rate": total_selector_tuned / max(episodes, 1)
        if has_initial_selector
        else None,
        "episode_seed": episode_seed,
        "episode_success": successes,
        "episode_final_reward": finals,
        "episode_max_reward": maxima,
        "episode_steps": steps_out,
        "episode_initial_selected_distance": episode_initial_selected_distance,
        "episode_initial_raw_distance": episode_initial_raw_distance,
        "episode_initial_base_action_l2": episode_initial_base_action_l2,
        "episode_initial_selector_use_tuned": episode_initial_selector_use_tuned,
        "episode_residual_l2_mean": episode_residual_l2_mean,
        "episode_action_saturation_rate": episode_action_saturation_rate,
        "episode_selected_distance_gate_rate": episode_selected_distance_gate_rate,
        "episode_raw_segment_distance_reduction": episode_raw_segment_distance_reduction,
        "episode_segment_distance_reduction": episode_segment_distance_reduction,
        "episode_segment_goal_reach_rate": episode_segment_goal_reach_rate,
        "serial_segment_episode_seed": serial_segment_episode_seed,
        "serial_segment_index": serial_segment_index,
        "serial_segment_start_step": serial_segment_start_step,
        "serial_segment_initial_selected_distance": serial_segment_initial_selected_distance,
        "serial_segment_initial_raw_distance": serial_segment_initial_raw_distance,
        "serial_segment_initial_base_action_l2": serial_segment_initial_base_action_l2,
        "serial_segment_initial_previous_action_norm_l2": (
            serial_segment_initial_previous_action_norm_l2
        ),
        "serial_segment_final_selected_distance": serial_segment_final_selected_distance,
        "serial_segment_final_raw_distance": serial_segment_final_raw_distance,
        "serial_segment_selected_distance_reduction": (
            serial_segment_selected_distance_reduction
        ),
        "serial_segment_raw_distance_reduction": serial_segment_raw_distance_reduction,
        "serial_segment_goal_reached": serial_segment_goal_reached,
        "serial_segment_residual_l2_mean": serial_segment_residual_l2_mean,
        "serial_segment_action_saturation_rate": serial_segment_action_saturation_rate,
        "serial_segment_distance_gate_rate": serial_segment_distance_gate_rate,
    }
    write_json(output, payload)
    return output


@torch.inference_mode()
def record_low_level_rl_videos(
    config: Config,
    n_demo: int,
    seed: int,
    run_name: str,
    episodes: int,
    seed_start: int,
    checkpoint_path: Path | None = None,
    force: bool = False,
) -> list[Path]:
    import imageio.v2 as imageio

    device = default_device()
    frozen = _load_frozen(config, n_demo, seed, device)
    artifact, result = _paths(config, n_demo, seed, run_name)
    checkpoint_path = checkpoint_path or artifact / "latest.pt"
    residual_agent: ResidualActorCritic | None = None
    direct_agent: DirectLowActorCritic | None = None
    alpha = 0.0
    global_step = 0
    if checkpoint_path.exists():
        raw_checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        method = str(raw_checkpoint.get("recipe", {}).get("method", "r1_residual_deterministic"))
        if method == "r3_direct_last_layer":
            direct_agent, checkpoint = _load_direct(checkpoint_path, frozen, device)
        else:
            residual_agent, checkpoint = _load_residual(checkpoint_path, device)
            alpha = float(checkpoint["recipe"]["alpha"])
        global_step = int(checkpoint["global_step"])
    output_dir = ensure_dir(result / "videos")
    dino = _phase4_dino_from_config(config, device)
    zero_previous = frozen.action_norm.transform(np.zeros((1, 3), dtype=np.float32))
    max_steps = int(config.get("env_max_episode_steps", 100))
    control_freq = int(config.get("control_freq", 20))
    paths: list[Path] = []

    def render_frame(env: gym.Env) -> np.ndarray:
        rendered = env.render()
        frame = (
            rendered.detach().cpu().numpy()
            if isinstance(rendered, torch.Tensor)
            else np.asarray(rendered)
        )
        if frame.ndim == 4:
            frame = frame[0]
        return frame.astype(np.uint8)

    for episode_index in trange(episodes, desc=f"record low RL {run_name}"):
        rollout_seed = seed_start + episode_index
        existing = sorted(output_dir.glob(f"seed{rollout_seed}_*.mp4"))
        if existing and not force:
            paths.append(existing[0])
            continue
        env = gym.make(
            config.get("env_id"),
            obs_mode="rgb+state",
            control_mode=config.get("control_mode"),
            reward_mode="normalized_dense",
            render_mode="rgb_array",
            sim_backend=_rl_backend(config),
            num_envs=1,
            reconfiguration_freq=config.get("rl.eval_reconfiguration_freq", 1),
        )
        action_low_np = np.asarray(env.action_space.low, dtype=np.float32)
        action_high_np = np.asarray(env.action_space.high, dtype=np.float32)
        if action_low_np.ndim == 2:
            action_low_np = action_low_np[0]
            action_high_np = action_high_np[0]
        action_low = torch.as_tensor(action_low_np, device=device)
        action_high = torch.as_tensor(action_high_np, device=device)
        previous_action = zero_previous.copy()
        held_goal = np.zeros((1, frozen.goal_dim), dtype=np.float32)
        countdown = 0
        frames_out: list[np.ndarray] = []
        success = False
        final_reward = 0.0
        max_reward = -float("inf")
        try:
            obs, _info = env.reset(seed=[rollout_seed])
            frames_out.append(render_frame(env))
            for _step in range(max_steps):
                frames = _phase4_frame_inputs(
                    obs, dino, int(config.get("dino.batch_size", 64))
                )
                normalized_frames = frozen.frame_norm.transform(frames)
                current_latent = _encode_frames(frozen, frames, device)
                if countdown <= 0:
                    high_condition = np.concatenate([normalized_frames, previous_action], axis=-1)
                    held_goal = (
                        frozen.high_model(
                            torch.from_numpy(high_condition).to(device).float()
                        )
                        .cpu()
                        .numpy()
                    )
                    countdown = frozen.update_period
                remaining = np.asarray(
                    [[max(countdown, 1) / frozen.horizon_steps]], dtype=np.float32
                )
                condition_np = _low_condition_array(
                    normalized_frames,
                    current_latent,
                    held_goal,
                    previous_action,
                    remaining,
                    frozen.conditioning,
                )
                condition = torch.from_numpy(condition_np).to(device).float()
                normalized_base = frozen.low_model(condition)
                base_action = torch.from_numpy(
                    frozen.action_norm.inverse(normalized_base.cpu().numpy())
                ).to(device)
                if direct_agent is not None:
                    unclipped = direct_agent.get_action_and_value(
                        condition, deterministic=True
                    )[0]
                else:
                    raw_residual = (
                        residual_agent.get_action_and_value(condition, deterministic=True)[0]
                        if residual_agent is not None
                        else torch.zeros_like(base_action)
                    )
                    unclipped = base_action + alpha * torch.tanh(raw_residual)
                action = torch.clamp(unclipped, action_low, action_high)
                obs, reward, terminated, truncated, info = env.step(action)
                previous_action = frozen.action_norm.transform(
                    action.cpu().numpy().astype(np.float32)
                )
                countdown -= 1
                frames_out.append(render_frame(env))
                final_reward = float(np.asarray(reward.cpu()).reshape(-1)[0])
                max_reward = max(max_reward, final_reward)
                if "success" in info:
                    success = success or bool(np.asarray(info["success"].cpu()).reshape(-1)[0])
                if bool(
                    np.asarray(torch.logical_or(terminated, truncated).cpu()).reshape(-1)[0]
                ):
                    break
        finally:
            env.close()
        path = output_dir / (
            f"seed{rollout_seed}_step{global_step}_success{int(success)}_"
            f"final{final_reward:.3f}_max{max_reward:.3f}.mp4"
        )
        imageio.mimsave(path, frames_out, fps=control_freq, macro_block_size=1)
        paths.append(path)
    return paths


def train_residual_rl(
    config: Config,
    n_demo: int,
    seed: int,
    run_name: str,
    total_steps: int,
    alpha: float,
    terminal_weight: float,
    distance_progress_weight: float = 1.0,
    task_reward_weight: float = 0.0,
    task_progress_weight: float = 0.0,
    distance_metric: str = "raw_l2",
    reachability_checkpoint_path: Path | None = None,
    candidate: str = VAE_CANDIDATE,
    rl_seed_offset: int = 0,
    force: bool = False,
) -> Path:
    if rl_seed_offset < 0:
        raise ValueError("rl_seed_offset must be non-negative")
    if candidate == VAE_CANDIDATE and n_demo not in {500, 1000}:
        raise ValueError("Low-level RL currently supports N_demo in {500, 1000}")
    artifact, result = _paths(config, n_demo, seed, run_name, candidate)
    latest = artifact / "latest.pt"
    best_train_latent = artifact / "best_train_latent.pt"
    if force:
        latest.unlink(missing_ok=True)
        best_train_latent.unlink(missing_ok=True)
    device = default_device()
    set_seed(seed + 50_000 + rl_seed_offset)
    frozen = _load_frozen(config, n_demo, seed, device, candidate)
    if distance_metric == "reachability" and reachability_checkpoint_path is None:
        reachability_checkpoint_path = (
            _default_reachability_checkpoint(config, n_demo, seed)
            if candidate == VAE_CANDIDATE
            else _candidate_reachability_checkpoint(config, candidate, seed)
        )
    if distance_metric == "reachability" and not reachability_checkpoint_path.exists():
        raise FileNotFoundError(
            f"Reachability checkpoint not found: {reachability_checkpoint_path}"
        )
    if any(
        parameter.requires_grad
        for module in (frozen.encoder, frozen.high_model, frozen.low_model)
        for parameter in module.parameters()
    ):
        raise RuntimeError("Frozen hierarchy unexpectedly has trainable parameters")
    condition_dim = _condition_dim(frozen)
    width = int(config.get("low_level_rl.residual_width", 256))
    depth = int(config.get("low_level_rl.residual_depth", 2))
    agent = ResidualActorCritic(
        condition_dim,
        width=width,
        depth=depth,
        initial_logstd=float(config.get("low_level_rl.initial_logstd", -2.3)),
    ).to(device)
    optimizer = torch.optim.Adam(
        agent.parameters(), lr=float(config.get("low_level_rl.learning_rate", 1e-4)), eps=1e-5
    )
    recipe = {
        "method": "r1_residual_deterministic",
        "n_demo": n_demo,
        "candidate": candidate,
        "seed": seed,
        "rl_seed_offset": rl_seed_offset,
        "alpha": alpha,
        "terminal_weight": terminal_weight,
        "distance_progress_weight": distance_progress_weight,
        "task_reward_weight": task_reward_weight,
        "task_progress_weight": task_progress_weight,
        "distance_metric": distance_metric,
        "reachability_checkpoint": str(reachability_checkpoint_path)
        if reachability_checkpoint_path is not None
        else None,
        "residual_penalty_weight": float(config.get("low_level_rl.residual_penalty_weight", 0.01)),
        "segment_terminates_gae": bool(config.get("low_level_rl.segment_terminates_gae", True)),
        "rollout_mode": "full_hierarchy_segment_reward",
        "frozen_hierarchy": str(frozen.checkpoint_path),
    }
    global_step = 0
    history: list[dict[str, Any]] = []
    best_score = -math.inf
    if latest.exists() and not force:
        checkpoint = torch.load(latest, map_location=device, weights_only=False)
        if checkpoint["recipe"] != recipe:
            raise ValueError(f"Existing run {run_name} has a different recipe")
        agent.load_state_dict(checkpoint["agent"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        global_step = int(checkpoint["global_step"])
        history = list(checkpoint["history"])
    if global_step >= total_steps:
        return latest
    num_envs = int(config.get("low_level_rl.num_envs", 32))
    rollout_steps = int(config.get("low_level_rl.rollout_steps", 32))
    batch_size = num_envs * rollout_steps
    minibatches = int(config.get("low_level_rl.num_minibatches", 8))
    if batch_size % minibatches:
        raise ValueError("RL batch size must divide num_minibatches")
    minibatch_size = batch_size // minibatches
    rollout = HierarchyRollout(
        config,
        frozen,
        num_envs,
        int(config.get("low_level_rl.train_seed_start", 3_000_000))
        + seed * 100_000
        + rl_seed_offset * 100_000,
        device,
        distance_metric=distance_metric,
        reachability_checkpoint_path=reachability_checkpoint_path,
    )
    condition_buf = torch.zeros((rollout_steps, num_envs, condition_dim), device=device)
    raw_action_buf = torch.zeros((rollout_steps, num_envs, 3), device=device)
    logprob_buf = torch.zeros((rollout_steps, num_envs), device=device)
    reward_buf = torch.zeros((rollout_steps, num_envs), device=device)
    done_buf = torch.zeros((rollout_steps, num_envs), device=device)
    value_buf = torch.zeros((rollout_steps, num_envs), device=device)
    next_done = torch.zeros(num_envs, device=device)
    gamma = float(config.get("low_level_rl.gamma", 0.99))
    gae_lambda = float(config.get("low_level_rl.gae_lambda", 0.95))
    clip_coef = float(config.get("low_level_rl.clip_coef", 0.2))
    ent_coef = float(config.get("low_level_rl.entropy_coef", 0.0))
    value_coef = float(config.get("low_level_rl.value_coef", 0.5))
    update_epochs = int(config.get("low_level_rl.update_epochs", 4))
    max_grad_norm = float(config.get("low_level_rl.max_grad_norm", 0.5))
    residual_penalty_weight = float(config.get("low_level_rl.residual_penalty_weight", 0.01))
    checkpoint_every = int(config.get("low_level_rl.checkpoint_every_steps", 25_000))
    next_checkpoint = ((global_step // checkpoint_every) + 1) * checkpoint_every
    timer = Timer()
    progress = trange(
        global_step, total_steps, initial=global_step, total=total_steps, desc=run_name
    )
    try:
        while global_step < total_steps:
            agent.eval()
            distance_values: list[float] = []
            terminal_distances: list[float] = []
            residual_values: list[float] = []
            saturation_count = 0
            for step in range(rollout_steps):
                condition, base_action, distance, _replan = rollout.condition()
                condition_buf[step] = condition
                done_buf[step] = next_done
                with torch.no_grad():
                    raw_action, logprob, _entropy, value = agent.get_action_and_value(condition)
                residual = alpha * torch.tanh(raw_action)
                unclipped = base_action + residual
                action = torch.clamp(unclipped, rollout.action_low, rollout.action_high)
                raw_action_buf[step] = raw_action
                logprob_buf[step] = logprob
                value_buf[step] = value
                penalty = (
                    residual_penalty_weight * torch.mean(residual.square(), dim=-1).cpu().numpy()
                )
                reward, done, metrics = rollout.step(
                    action,
                    distance,
                    terminal_weight,
                    distance_progress_weight,
                    task_reward_weight,
                    task_progress_weight,
                    penalty,
                )
                reward_buf[step] = torch.from_numpy(reward).to(device)
                # Each held goal defines a local 10-step MDP. Do not carry
                # advantage estimates into the next, unrelated goal segment.
                local_done = done.astype(bool)
                if bool(config.get("low_level_rl.segment_terminates_gae", True)):
                    local_done = np.logical_or(local_done, metrics["segment_end"])
                next_done = torch.from_numpy(local_done.astype(np.float32)).to(device)
                distance_values.extend(distance.tolist())
                terminal_distances.extend(metrics["next_distance"][metrics["segment_end"]].tolist())
                residual_values.extend(torch.linalg.vector_norm(residual, dim=-1).cpu().tolist())
                saturation_count += int(torch.any(unclipped != action, dim=-1).sum().cpu())
                global_step += num_envs
                progress.update(min(num_envs, total_steps - progress.n))
            with torch.no_grad():
                next_condition, _base, _distance, _replan = rollout.condition()
                next_value = agent.critic(next_condition).flatten()
                advantages = torch.zeros_like(reward_buf)
                last_gae = torch.zeros(num_envs, device=device)
                for step in reversed(range(rollout_steps)):
                    if step == rollout_steps - 1:
                        next_nonterminal = 1.0 - next_done
                        following_value = next_value
                    else:
                        next_nonterminal = 1.0 - done_buf[step + 1]
                        following_value = value_buf[step + 1]
                    delta = (
                        reward_buf[step]
                        + gamma * following_value * next_nonterminal
                        - value_buf[step]
                    )
                    last_gae = delta + gamma * gae_lambda * next_nonterminal * last_gae
                    advantages[step] = last_gae
                returns = advantages + value_buf
            flat_condition = condition_buf.flatten(0, 1)
            flat_action = raw_action_buf.flatten(0, 1)
            flat_logprob = logprob_buf.flatten()
            flat_advantage = advantages.flatten()
            flat_return = returns.flatten()
            flat_value = value_buf.flatten()
            indices = np.arange(batch_size)
            clip_fractions: list[float] = []
            agent.train()
            for _epoch in range(update_epochs):
                np.random.shuffle(indices)
                for start in range(0, batch_size, minibatch_size):
                    selected = indices[start : start + minibatch_size]
                    _action, new_logprob, entropy, new_value = agent.get_action_and_value(
                        flat_condition[selected], flat_action[selected]
                    )
                    log_ratio = new_logprob - flat_logprob[selected]
                    ratio = log_ratio.exp()
                    advantage = flat_advantage[selected]
                    advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
                    policy_loss = torch.maximum(
                        -advantage * ratio,
                        -advantage * torch.clamp(ratio, 1.0 - clip_coef, 1.0 + clip_coef),
                    ).mean()
                    value_prediction = new_value.flatten()
                    value_clipped = flat_value[selected] + torch.clamp(
                        value_prediction - flat_value[selected], -clip_coef, clip_coef
                    )
                    value_loss = (
                        0.5
                        * torch.maximum(
                            (value_prediction - flat_return[selected]).square(),
                            (value_clipped - flat_return[selected]).square(),
                        ).mean()
                    )
                    loss = policy_loss + value_coef * value_loss - ent_coef * entropy.mean()
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    nn.utils.clip_grad_norm_(agent.parameters(), max_grad_norm)
                    optimizer.step()
                    clip_fractions.append(
                        float(((ratio - 1).abs() > clip_coef).float().mean().detach().cpu())
                    )
            row = {
                "global_step": global_step,
                "mean_reward": float(reward_buf.mean().cpu()),
                "mean_latent_distance": float(np.mean(distance_values)),
                "mean_terminal_distance": float(np.mean(terminal_distances))
                if terminal_distances
                else None,
                "mean_residual_l2": float(np.mean(residual_values)),
                "action_saturation_rate": saturation_count / batch_size,
                "policy_loss": float(policy_loss.detach().cpu()),
                "value_loss": float(value_loss.detach().cpu()),
                "clip_fraction": float(np.mean(clip_fractions)),
                "elapsed_s": timer.elapsed(),
            }
            history.append(row)
            score = -float(row["mean_terminal_distance"] or row["mean_latent_distance"])
            if score > best_score:
                best_score = score
                _save_checkpoint(
                    best_train_latent,
                    agent,
                    optimizer,
                    global_step,
                    recipe,
                    history,
                )
            if global_step >= next_checkpoint or global_step >= total_steps:
                _save_checkpoint(latest, agent, optimizer, global_step, recipe, history)
                _save_checkpoint(
                    artifact / f"step_{global_step:09d}.pt",
                    agent,
                    optimizer,
                    global_step,
                    recipe,
                    history,
                )
                write_json(
                    result / "train_metrics.json",
                    {"recipe": recipe, "latest": row, "history": history},
                )
                next_checkpoint += checkpoint_every
    finally:
        progress.close()
        rollout.close()
    return latest


def train_direct_low_rl(
    config: Config,
    n_demo: int,
    seed: int,
    run_name: str,
    total_steps: int,
    bc_weight: float,
    terminal_weight: float,
    distance_progress_weight: float = 1.0,
    task_reward_weight: float = 0.0,
    task_progress_weight: float = 0.0,
    distance_metric: str = "raw_l2",
    reachability_checkpoint_path: Path | None = None,
    candidate: str = VAE_CANDIDATE,
    rl_seed_offset: int = 0,
    force: bool = False,
) -> Path:
    if rl_seed_offset < 0:
        raise ValueError("rl_seed_offset must be non-negative")
    if candidate == VAE_CANDIDATE and n_demo not in {500, 1000}:
        raise ValueError("Low-level RL currently supports N_demo in {500, 1000}")
    artifact, result = _paths(config, n_demo, seed, run_name, candidate)
    latest = artifact / "latest.pt"
    best_train_latent = artifact / "best_train_latent.pt"
    if force:
        latest.unlink(missing_ok=True)
        best_train_latent.unlink(missing_ok=True)
    device = default_device()
    set_seed(seed + 60_000 + rl_seed_offset)
    frozen = _load_frozen(config, n_demo, seed, device, candidate)
    if distance_metric == "reachability" and reachability_checkpoint_path is None:
        reachability_checkpoint_path = (
            _default_reachability_checkpoint(config, n_demo, seed)
            if candidate == VAE_CANDIDATE
            else _candidate_reachability_checkpoint(config, candidate, seed)
        )
    if distance_metric == "reachability" and not reachability_checkpoint_path.exists():
        raise FileNotFoundError(
            f"Reachability checkpoint not found: {reachability_checkpoint_path}"
        )
    if any(
        parameter.requires_grad
        for module in (frozen.encoder, frozen.high_model, frozen.low_model)
        for parameter in module.parameters()
    ):
        raise RuntimeError("Frozen hierarchy unexpectedly has trainable parameters")
    condition_dim = _condition_dim(frozen)
    width = int(config.get("low_level_rl.residual_width", 256))
    depth = int(config.get("low_level_rl.residual_depth", 2))
    agent = DirectLowActorCritic(
        frozen.low_model,
        frozen.action_norm.mean,
        frozen.action_norm.std,
        condition_dim,
        width=width,
        depth=depth,
        initial_logstd=float(config.get("low_level_rl.direct_initial_logstd", -4.0)),
    ).to(device)
    trainable = [parameter for parameter in agent.parameters() if parameter.requires_grad]
    optimizer = torch.optim.Adam(
        trainable,
        lr=float(config.get("low_level_rl.direct_learning_rate", 3e-5)),
        eps=1e-5,
    )
    recipe = {
        "method": "r3_direct_last_layer",
        "n_demo": n_demo,
        "candidate": candidate,
        "seed": seed,
        "rl_seed_offset": rl_seed_offset,
        "bc_weight": bc_weight,
        "terminal_weight": terminal_weight,
        "distance_progress_weight": distance_progress_weight,
        "task_reward_weight": task_reward_weight,
        "task_progress_weight": task_progress_weight,
        "distance_metric": distance_metric,
        "reachability_checkpoint": str(reachability_checkpoint_path)
        if reachability_checkpoint_path is not None
        else None,
        "segment_terminates_gae": bool(config.get("low_level_rl.segment_terminates_gae", True)),
        "rollout_mode": "full_hierarchy_segment_reward",
        "trainable_scope": "low_policy_final_layer_plus_logstd_and_critic",
        "frozen_hierarchy": str(frozen.checkpoint_path),
    }
    global_step = 0
    history: list[dict[str, Any]] = []
    best_score = -math.inf
    if latest.exists() and not force:
        checkpoint = torch.load(latest, map_location=device, weights_only=False)
        if checkpoint["recipe"] != recipe:
            raise ValueError(f"Existing run {run_name} has a different recipe")
        agent.load_state_dict(checkpoint["agent"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        global_step = int(checkpoint["global_step"])
        history = list(checkpoint["history"])
    if global_step >= total_steps:
        return latest
    num_envs = int(config.get("low_level_rl.num_envs", 32))
    rollout_steps = int(config.get("low_level_rl.rollout_steps", 32))
    batch_size = num_envs * rollout_steps
    minibatches = int(config.get("low_level_rl.num_minibatches", 8))
    if batch_size % minibatches:
        raise ValueError("RL batch size must divide num_minibatches")
    minibatch_size = batch_size // minibatches
    rollout = HierarchyRollout(
        config,
        frozen,
        num_envs,
        int(config.get("low_level_rl.train_seed_start", 3_000_000))
        + seed * 100_000
        + rl_seed_offset * 100_000,
        device,
        distance_metric=distance_metric,
        reachability_checkpoint_path=reachability_checkpoint_path,
    )
    condition_buf = torch.zeros((rollout_steps, num_envs, condition_dim), device=device)
    raw_action_buf = torch.zeros((rollout_steps, num_envs, 3), device=device)
    base_action_buf = torch.zeros((rollout_steps, num_envs, 3), device=device)
    logprob_buf = torch.zeros((rollout_steps, num_envs), device=device)
    reward_buf = torch.zeros((rollout_steps, num_envs), device=device)
    done_buf = torch.zeros((rollout_steps, num_envs), device=device)
    value_buf = torch.zeros((rollout_steps, num_envs), device=device)
    next_done = torch.zeros(num_envs, device=device)
    gamma = float(config.get("low_level_rl.gamma", 0.99))
    gae_lambda = float(config.get("low_level_rl.gae_lambda", 0.95))
    clip_coef = float(config.get("low_level_rl.clip_coef", 0.2))
    ent_coef = float(config.get("low_level_rl.entropy_coef", 0.0))
    value_coef = float(config.get("low_level_rl.value_coef", 0.5))
    update_epochs = int(config.get("low_level_rl.update_epochs", 4))
    max_grad_norm = float(config.get("low_level_rl.max_grad_norm", 0.5))
    checkpoint_every = int(config.get("low_level_rl.checkpoint_every_steps", 25_000))
    next_checkpoint = ((global_step // checkpoint_every) + 1) * checkpoint_every
    timer = Timer()
    progress = trange(
        global_step, total_steps, initial=global_step, total=total_steps, desc=run_name
    )
    try:
        while global_step < total_steps:
            agent.eval()
            distance_values: list[float] = []
            terminal_distances: list[float] = []
            delta_values: list[float] = []
            saturation_count = 0
            for step in range(rollout_steps):
                condition, base_action, distance, _replan = rollout.condition()
                condition_buf[step] = condition
                base_action_buf[step] = base_action
                done_buf[step] = next_done
                with torch.no_grad():
                    raw_action, logprob, _entropy, value = agent.get_action_and_value(condition)
                action = torch.clamp(raw_action, rollout.action_low, rollout.action_high)
                raw_action_buf[step] = raw_action
                logprob_buf[step] = logprob
                value_buf[step] = value
                reward, done, metrics = rollout.step(
                    action,
                    distance,
                    terminal_weight,
                    distance_progress_weight,
                    task_reward_weight,
                    task_progress_weight,
                    np.zeros(num_envs, dtype=np.float32),
                )
                reward_buf[step] = torch.from_numpy(reward).to(device)
                local_done = done.astype(bool)
                if bool(config.get("low_level_rl.segment_terminates_gae", True)):
                    local_done = np.logical_or(local_done, metrics["segment_end"])
                next_done = torch.from_numpy(local_done.astype(np.float32)).to(device)
                distance_values.extend(distance.tolist())
                terminal_distances.extend(metrics["next_distance"][metrics["segment_end"]].tolist())
                delta_values.extend(torch.linalg.vector_norm(raw_action - base_action, dim=-1).cpu().tolist())
                saturation_count += int(torch.any(raw_action != action, dim=-1).sum().cpu())
                global_step += num_envs
                progress.update(min(num_envs, total_steps - progress.n))
            with torch.no_grad():
                next_condition, _base, _distance, _replan = rollout.condition()
                next_value = agent.critic(next_condition).flatten()
                advantages = torch.zeros_like(reward_buf)
                last_gae = torch.zeros(num_envs, device=device)
                for step in reversed(range(rollout_steps)):
                    if step == rollout_steps - 1:
                        next_nonterminal = 1.0 - next_done
                        following_value = next_value
                    else:
                        next_nonterminal = 1.0 - done_buf[step + 1]
                        following_value = value_buf[step + 1]
                    delta = (
                        reward_buf[step]
                        + gamma * following_value * next_nonterminal
                        - value_buf[step]
                    )
                    last_gae = delta + gamma * gae_lambda * next_nonterminal * last_gae
                    advantages[step] = last_gae
                returns = advantages + value_buf
            flat_condition = condition_buf.flatten(0, 1)
            flat_action = raw_action_buf.flatten(0, 1)
            flat_base_action = base_action_buf.flatten(0, 1)
            flat_logprob = logprob_buf.flatten()
            flat_advantage = advantages.flatten()
            flat_return = returns.flatten()
            flat_value = value_buf.flatten()
            indices = np.arange(batch_size)
            clip_fractions: list[float] = []
            bc_losses: list[float] = []
            agent.train()
            for _epoch in range(update_epochs):
                np.random.shuffle(indices)
                for start in range(0, batch_size, minibatch_size):
                    selected = indices[start : start + minibatch_size]
                    _action, new_logprob, entropy, new_value = agent.get_action_and_value(
                        flat_condition[selected], raw_action=flat_action[selected]
                    )
                    mean_action = agent.mean_action(flat_condition[selected])
                    bc_loss = torch.mean((mean_action - flat_base_action[selected]).square())
                    log_ratio = new_logprob - flat_logprob[selected]
                    ratio = log_ratio.exp()
                    advantage = flat_advantage[selected]
                    advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
                    policy_loss = torch.maximum(
                        -advantage * ratio,
                        -advantage * torch.clamp(ratio, 1.0 - clip_coef, 1.0 + clip_coef),
                    ).mean()
                    value_prediction = new_value.flatten()
                    value_clipped = flat_value[selected] + torch.clamp(
                        value_prediction - flat_value[selected], -clip_coef, clip_coef
                    )
                    value_loss = (
                        0.5
                        * torch.maximum(
                            (value_prediction - flat_return[selected]).square(),
                            (value_clipped - flat_return[selected]).square(),
                        ).mean()
                    )
                    loss = (
                        policy_loss
                        + value_coef * value_loss
                        - ent_coef * entropy.mean()
                        + bc_weight * bc_loss
                    )
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    nn.utils.clip_grad_norm_(trainable, max_grad_norm)
                    optimizer.step()
                    clip_fractions.append(
                        float(((ratio - 1).abs() > clip_coef).float().mean().detach().cpu())
                    )
                    bc_losses.append(float(bc_loss.detach().cpu()))
            row = {
                "global_step": global_step,
                "mean_reward": float(reward_buf.mean().cpu()),
                "mean_latent_distance": float(np.mean(distance_values)),
                "mean_terminal_distance": float(np.mean(terminal_distances))
                if terminal_distances
                else None,
                "mean_direct_delta_l2": float(np.mean(delta_values)),
                "action_saturation_rate": saturation_count / batch_size,
                "policy_loss": float(policy_loss.detach().cpu()),
                "value_loss": float(value_loss.detach().cpu()),
                "bc_loss": float(np.mean(bc_losses)),
                "clip_fraction": float(np.mean(clip_fractions)),
                "elapsed_s": timer.elapsed(),
            }
            history.append(row)
            score = -float(row["mean_terminal_distance"] or row["mean_latent_distance"])
            if score > best_score:
                best_score = score
                _save_checkpoint(
                    best_train_latent,
                    agent,
                    optimizer,
                    global_step,
                    recipe,
                    history,
                )
            if global_step >= next_checkpoint or global_step >= total_steps:
                _save_checkpoint(latest, agent, optimizer, global_step, recipe, history)
                _save_checkpoint(
                    artifact / f"step_{global_step:09d}.pt",
                    agent,
                    optimizer,
                    global_step,
                    recipe,
                    history,
                )
                write_json(
                    result / "train_metrics.json",
                    {"recipe": recipe, "latest": row, "history": history},
                )
                next_checkpoint += checkpoint_every
    finally:
        progress.close()
        rollout.close()
    return latest
