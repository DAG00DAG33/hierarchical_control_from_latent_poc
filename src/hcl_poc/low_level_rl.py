from __future__ import annotations

import csv
import copy
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
    force: bool = False,
) -> Path:
    if checkpoint_path is not None and ensemble_checkpoint_paths:
        raise ValueError("Use either checkpoint_path or ensemble_checkpoint_paths, not both")
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
    current_segment_initial: np.ndarray | None = None
    current_segment_raw_initial: np.ndarray | None = None
    current_final = np.zeros(num_envs, dtype=np.float32)
    current_max = np.full(num_envs, -np.inf, dtype=np.float32)
    while len(successes) < episodes:
        condition, base_action, distance, replan = rollout.condition()
        raw_distance = rollout.raw_distance(rollout.current_latent, rollout.held_goal)
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
        action = torch.clamp(unclipped, rollout.action_low, rollout.action_high)
        saturation += int(torch.any(unclipped != action, dim=-1).sum().cpu())
        action_count += num_envs
        residual_magnitudes.extend(torch.linalg.vector_norm(residual, dim=-1).cpu().tolist())
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
                current_max[mask_np] = -np.inf
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
        "distance_metric": distance_metric,
        "reachability_checkpoint": str(reachability_checkpoint_path)
        if reachability_checkpoint_path is not None
        else None,
        "rl_steps": global_step,
        "episodes": count,
        "seed_start": seed_start,
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
        "episode_success": successes[:count],
        "episode_final_reward": finals[:count],
        "episode_max_reward": maxima[:count],
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
