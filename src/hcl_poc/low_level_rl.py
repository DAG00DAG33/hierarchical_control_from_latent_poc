from __future__ import annotations

import csv
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
    _load_hierarchy,
    _load_representation,
    _low_condition_array,
    train_learned_interface_hierarchy,
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
    frame_dim: int
    goal_dim: int
    checkpoint_path: Path


def _paths(config: Config, n_demo: int, seed: int, run_name: str) -> tuple[Path, Path]:
    artifact = ensure_dir(
        config.path_value("paths.incremental_artifact_dir")
        / "low_level_rl"
        / f"n{n_demo}"
        / f"seed{seed}"
        / run_name
    )
    result = ensure_dir(
        config.path_value("paths.incremental_results_dir")
        / "low_level_rl"
        / f"n{n_demo}"
        / f"seed{seed}"
        / run_name
    )
    return artifact, result


def _load_frozen(config: Config, n_demo: int, seed: int, device: torch.device) -> FrozenHierarchy:
    point_config = vae_scaling_config(config, n_demo)
    hierarchy_path = train_learned_interface_hierarchy(
        point_config, VAE_CANDIDATE, seed=seed, force=False
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
        frame_dim=int(checkpoint["frame_dim"]),
        goal_dim=int(checkpoint["goal_dim"]),
        checkpoint_path=Path(hierarchy_path),
    )


@torch.inference_mode()
def _encode_frames(frozen: FrozenHierarchy, frames: np.ndarray, device: torch.device) -> np.ndarray:
    normalized = frozen.representation_frame_norm.transform(frames)
    latent = frozen.encoder(torch.from_numpy(normalized).to(device).float())
    return frozen.goal_norm.transform(latent.cpu().numpy().astype(np.float32))


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


class HierarchyRollout:
    def __init__(
        self,
        config: Config,
        frozen: FrozenHierarchy,
        num_envs: int,
        seed_start: int,
        device: torch.device,
    ) -> None:
        self.config = config
        self.frozen = frozen
        self.num_envs = num_envs
        self.seed_start = seed_start
        self.device = device
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
        self.current_latent: np.ndarray
        self.previous_action: np.ndarray
        self.held_goal: np.ndarray
        self.countdown: np.ndarray
        self.reset()

    def reset(self) -> None:
        seeds = [self.seed_start + self.episode_offset + index for index in range(self.num_envs)]
        self.episode_offset += self.num_envs
        self.obs, _info = self.env.reset(seed=seeds)
        self.frames = _phase4_frame_inputs(
            self.obs, self.dino, int(self.config.get("dino.batch_size", 64))
        )
        self.normalized_frames = self.frozen.frame_norm.transform(self.frames)
        self.current_latent = _encode_frames(self.frozen, self.frames, self.device)
        self.previous_action = np.repeat(self.zero_previous[None], self.num_envs, axis=0)
        self.held_goal = np.zeros((self.num_envs, self.frozen.goal_dim), dtype=np.float32)
        self.countdown = np.zeros(self.num_envs, dtype=np.int32)

    @torch.inference_mode()
    def condition(self) -> tuple[torch.Tensor, torch.Tensor, np.ndarray]:
        replan = self.countdown <= 0
        if np.any(replan):
            high_condition = np.concatenate([self.normalized_frames, self.previous_action], axis=-1)
            predicted = (
                self.frozen.high_model(torch.from_numpy(high_condition).to(self.device).float())
                .cpu()
                .numpy()
            )
            self.held_goal[replan] = predicted[replan]
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
        distance = np.mean(np.square(self.current_latent - self.held_goal), axis=-1).astype(
            np.float32
        )
        return condition, base_action, distance

    @torch.inference_mode()
    def step(
        self,
        executed_action: torch.Tensor,
        previous_distance: np.ndarray,
        terminal_weight: float,
        task_reward_weight: float,
        residual_penalty: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        next_obs, env_reward, terminated, truncated, info = self.env.step(executed_action)
        done = torch.logical_or(terminated, truncated).detach().cpu().numpy().reshape(-1)
        next_frames = _phase4_frame_inputs(
            next_obs, self.dino, int(self.config.get("dino.batch_size", 64))
        )
        next_latent = _encode_frames(self.frozen, next_frames, self.device)
        next_distance = np.mean(np.square(next_latent - self.held_goal), axis=-1).astype(np.float32)
        segment_end = self.countdown == 1
        reward = previous_distance - next_distance
        reward -= residual_penalty
        reward -= terminal_weight * next_distance * segment_end.astype(np.float32)
        reward += task_reward_weight * env_reward.detach().cpu().numpy().reshape(-1)
        # Auto-reset observations do not belong to the previous held goal.
        reward[done] = task_reward_weight * env_reward.detach().cpu().numpy().reshape(-1)[done]
        self.obs = next_obs
        self.frames = next_frames
        self.normalized_frames = self.frozen.frame_norm.transform(next_frames)
        self.current_latent = next_latent
        clipped = torch.clamp(executed_action, self.action_low, self.action_high)
        self.previous_action = self.frozen.action_norm.transform(
            clipped.cpu().numpy().astype(np.float32)
        )
        self.countdown -= 1
        if np.any(done):
            self.previous_action[done] = self.zero_previous
            self.countdown[done] = 0
        metrics = {
            "next_distance": next_distance,
            "segment_end": segment_end,
            "done": done,
            "env_reward": env_reward.detach().cpu().numpy().reshape(-1),
            "info": info,
        }
        return reward.astype(np.float32), done.astype(np.float32), metrics

    def close(self) -> None:
        self.env.close()


def _teacher_goal_threshold(config: Config, n_demo: int, seed: int) -> dict[str, float]:
    point = vae_scaling_config(config, n_demo)
    path = (
        point.path_value("paths.incremental_artifact_dir")
        / "learned_interface"
        / VAE_CANDIDATE
        / f"seed{seed}"
        / "encoded_episodes.pt"
    )
    payload = torch.load(path, map_location="cpu", weights_only=False)
    hierarchy = torch.load(path.parent / "hierarchy.pt", map_location="cpu", weights_only=False)
    norm = Standardizer.from_state_dict(hierarchy["goal_norm"])
    horizon = int(hierarchy["horizon_steps"])
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
    condition_dim = frozen.frame_dim + frozen.goal_dim + 4
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
    agent: ResidualActorCritic,
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


@torch.inference_mode()
def evaluate_residual_rl(
    config: Config,
    n_demo: int,
    seed: int,
    run_name: str,
    episodes: int,
    seed_start: int,
    checkpoint_path: Path | None = None,
    force: bool = False,
) -> Path:
    artifact, result = _paths(config, n_demo, seed, run_name)
    output = result / f"eval_{episodes}_seed{seed_start}.json"
    if output.exists() and not force:
        return output
    device = default_device()
    frozen = _load_frozen(config, n_demo, seed, device)
    checkpoint_path = checkpoint_path or artifact / "best.pt"
    agent = None
    alpha = 0.0
    global_step = 0
    if checkpoint_path.exists():
        agent, checkpoint = _load_residual(checkpoint_path, device)
        alpha = float(checkpoint["recipe"]["alpha"])
        global_step = int(checkpoint["global_step"])
    num_envs = min(int(config.get("low_level_rl.eval_num_envs", 32)), episodes)
    rollout = HierarchyRollout(config, frozen, num_envs, seed_start, device)
    threshold = _teacher_goal_threshold(config, n_demo, seed)["goal_threshold"]
    successes: list[float] = []
    finals: list[float] = []
    maxima: list[float] = []
    initial_distances: list[float] = []
    final_distances: list[float] = []
    reached: list[float] = []
    saturation = 0
    action_count = 0
    residual_magnitudes: list[float] = []
    current_segment_initial: np.ndarray | None = None
    current_final = np.zeros(num_envs, dtype=np.float32)
    current_max = np.full(num_envs, -np.inf, dtype=np.float32)
    while len(successes) < episodes:
        condition, base_action, distance = rollout.condition()
        if current_segment_initial is None:
            current_segment_initial = distance.copy()
        raw_residual = (
            agent.get_action_and_value(condition, deterministic=True)[0]
            if agent is not None
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
            task_reward_weight=0.0,
            residual_penalty=np.zeros(num_envs, dtype=np.float32),
        )
        segment_end = metrics["segment_end"]
        if np.any(segment_end):
            initial_distances.extend(current_segment_initial[segment_end].tolist())
            final_distances.extend(metrics["next_distance"][segment_end].tolist())
            reached.extend(
                (metrics["next_distance"][segment_end] < threshold).astype(float).tolist()
            )
            current_segment_initial[segment_end] = metrics["next_distance"][segment_end]
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
    payload = {
        "n_demo": n_demo,
        "seed": seed,
        "run_name": run_name,
        "checkpoint": str(checkpoint_path) if agent is not None else None,
        "rl_steps": global_step,
        "episodes": count,
        "seed_start": seed_start,
        "success": float(np.mean(successes[:count])),
        "final_reward": float(np.mean(finals[:count])),
        "max_reward": float(np.mean(maxima[:count])),
        "segment_initial_distance": float(initial_np.mean()),
        "segment_final_distance": float(final_np.mean()),
        "segment_distance_reduction": float((initial_np - final_np).mean()),
        "segment_goal_reach_rate": float(np.mean(reached)),
        "goal_threshold": threshold,
        "action_saturation_rate": saturation / max(action_count, 1),
        "residual_l2_mean": float(np.mean(residual_magnitudes)),
        "episode_success": successes[:count],
    }
    write_json(output, payload)
    return output


def train_residual_rl(
    config: Config,
    n_demo: int,
    seed: int,
    run_name: str,
    total_steps: int,
    alpha: float,
    terminal_weight: float,
    task_reward_weight: float = 0.0,
    force: bool = False,
) -> Path:
    if n_demo not in {500, 1000}:
        raise ValueError("Low-level RL currently supports N_demo in {500, 1000}")
    artifact, result = _paths(config, n_demo, seed, run_name)
    latest = artifact / "latest.pt"
    best = artifact / "best.pt"
    if force:
        latest.unlink(missing_ok=True)
        best.unlink(missing_ok=True)
    device = default_device()
    set_seed(seed + 50_000)
    frozen = _load_frozen(config, n_demo, seed, device)
    if any(
        parameter.requires_grad
        for module in (frozen.encoder, frozen.high_model, frozen.low_model)
        for parameter in module.parameters()
    ):
        raise RuntimeError("Frozen hierarchy unexpectedly has trainable parameters")
    condition_dim = frozen.frame_dim + frozen.goal_dim + 4
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
        "seed": seed,
        "alpha": alpha,
        "terminal_weight": terminal_weight,
        "task_reward_weight": task_reward_weight,
        "residual_penalty_weight": float(config.get("low_level_rl.residual_penalty_weight", 0.01)),
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
        return best if best.exists() else latest
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
        int(config.get("low_level_rl.train_seed_start", 3_000_000)) + seed * 100_000,
        device,
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
                condition, base_action, distance = rollout.condition()
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
                    action, distance, terminal_weight, task_reward_weight, penalty
                )
                reward_buf[step] = torch.from_numpy(reward).to(device)
                next_done = torch.from_numpy(done).to(device)
                distance_values.extend(distance.tolist())
                terminal_distances.extend(metrics["next_distance"][metrics["segment_end"]].tolist())
                residual_values.extend(torch.linalg.vector_norm(residual, dim=-1).cpu().tolist())
                saturation_count += int(torch.any(unclipped != action, dim=-1).sum().cpu())
                global_step += num_envs
                progress.update(min(num_envs, total_steps - progress.n))
            with torch.no_grad():
                next_condition, _base, _distance = rollout.condition()
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
                _save_checkpoint(best, agent, optimizer, global_step, recipe, history)
            if global_step >= next_checkpoint or global_step >= total_steps:
                _save_checkpoint(latest, agent, optimizer, global_step, recipe, history)
                write_json(
                    result / "train_metrics.json",
                    {"recipe": recipe, "latest": row, "history": history},
                )
                next_checkpoint += checkpoint_every
    finally:
        progress.close()
        rollout.close()
    return best
