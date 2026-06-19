from __future__ import annotations

import copy
import subprocess
from importlib.metadata import version
from pathlib import Path
from typing import Any, Callable

import gymnasium as gym
import h5py
import mani_skill  # noqa: F401
import numpy as np
import torch
from rich.console import Console
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import trange

from hcl_poc.config import Config
from hcl_poc.features import DinoExtractor, batched
from hcl_poc.flow import flow_matching_loss, sample_flow
from hcl_poc.models import FlowModel, MLP, ObservationEncoder, RepresentationWorldModel
from hcl_poc.rl import (
    PPOAgent,
    _make_state_env,
    _rl_backend,
    _rl_paths,
    evaluate_ppo,
    load_ppo_agent,
)
from hcl_poc.utils import Standardizer, Timer, default_device, ensure_dir, set_seed, write_json

console = Console()


def _numpy(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _scalar(value: Any) -> float:
    return float(_numpy(value).reshape(-1)[0])


def _state_obs(obs: Any) -> np.ndarray:
    return _numpy(obs).reshape(-1).astype(np.float32)


def _git_metadata() -> dict[str, Any]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )
    return {"git_commit": commit, "git_dirty": dirty}


def _runtime_metadata(config: Config) -> dict[str, Any]:
    return {
        **_git_metadata(),
        "config": str(config.path),
        "environment": str(config.get("env_id")),
        "controller": str(config.get("control_mode")),
        "control_freq": int(config.get("control_freq")),
        "mani_skill_version": version("mani-skill"),
        "gymnasium_version": version("gymnasium"),
        "torch_version": str(torch.__version__),
        "cuda_available": torch.cuda.is_available(),
        "device": str(default_device()),
    }


def _make_scalar_state_env(config: Config):
    return gym.make(
        config.get("env_id"),
        obs_mode="state",
        control_mode=config.get("control_mode"),
        reward_mode="normalized_dense",
        render_mode=None,
        sim_backend=config.get("incremental.phase0.scalar_sim_backend", "physx_cpu"),
        reconfiguration_freq=config.get("rl.eval_reconfiguration_freq", 1),
    )


def _actor_policy(actor: nn.Module, device: torch.device) -> Callable[[np.ndarray], np.ndarray]:
    @torch.inference_mode()
    def policy(state: np.ndarray) -> np.ndarray:
        state_t = torch.from_numpy(state[None]).to(device).float()
        return actor(state_t).detach().cpu().numpy()[0].astype(np.float32)

    return policy


def copied_actor_student(teacher: PPOAgent, device: torch.device) -> nn.Sequential:
    student = copy.deepcopy(teacher.actor_mean).to(device)
    student.load_state_dict(teacher.actor_mean.state_dict())
    student.eval()
    return student


def _evaluate_scalar_policy(
    config: Config,
    policy: Callable[[np.ndarray], np.ndarray],
    episodes: int,
    seed_start: int,
) -> dict[str, Any]:
    env = _make_scalar_state_env(config)
    action_low = np.asarray(env.action_space.low, dtype=np.float32)
    action_high = np.asarray(env.action_space.high, dtype=np.float32)
    clip_actions = bool(config.get("policy.clip_actions_to_env_space", True))
    successes: list[float] = []
    final_rewards: list[float] = []
    max_rewards: list[float] = []
    episode_lengths: list[int] = []
    latencies: list[float] = []
    saturated: list[float] = []
    for episode in trange(episodes, desc="phase0 scalar evaluation"):
        obs, _info = env.reset(seed=seed_start + episode)
        terminated = truncated = False
        success = False
        final_reward = 0.0
        max_reward = -float("inf")
        steps = 0
        while not (terminated or truncated):
            timer = Timer()
            raw_action = policy(_state_obs(obs))
            latencies.append(timer.elapsed())
            saturated.append(float(np.any((raw_action < action_low) | (raw_action > action_high))))
            action = (
                np.clip(raw_action, action_low, action_high).astype(np.float32)
                if clip_actions
                else raw_action
            )
            obs, reward, terminated, truncated, info = env.step(action)
            final_reward = _scalar(reward)
            max_reward = max(max_reward, final_reward)
            success = success or bool(_scalar(info.get("success", False)))
            steps += 1
        successes.append(float(success))
        final_rewards.append(final_reward)
        max_rewards.append(max_reward)
        episode_lengths.append(steps)
    env.close()
    return {
        "success": float(np.mean(successes)),
        "success_stderr": float(np.std(successes) / np.sqrt(len(successes))),
        "final_reward": float(np.mean(final_rewards)),
        "max_reward": float(np.mean(max_rewards)),
        "mean_episode_length": float(np.mean(episode_lengths)),
        "action_saturation_rate": float(np.mean(saturated)),
        "inference_latency_s": float(np.mean(latencies)),
        "episodes": episodes,
        "seed_start": seed_start,
    }


@torch.inference_mode()
def _collect_phase0_causal_audit(
    config: Config,
    teacher: PPOAgent,
    output_path: Path,
    successful_trajectories: int,
) -> dict[str, Any]:
    env = _make_scalar_state_env(config)
    device = next(teacher.parameters()).device
    action_low = np.asarray(env.action_space.low, dtype=np.float32)
    action_high = np.asarray(env.action_space.high, dtype=np.float32)
    max_attempts = int(config.get("incremental.phase0.audit_max_attempts", 100))
    seed_start = int(config.get("incremental.phase0.audit_seed", 20000))
    ensure_dir(output_path.parent)
    if output_path.exists():
        output_path.unlink()
    successes = 0
    attempts = 0
    with h5py.File(output_path, "w") as h5:
        meta = h5.create_group("meta")
        for key, value in _runtime_metadata(config).items():
            meta.attrs[key] = value
        meta.attrs["dataset_type"] = "causal_dataset"
        meta.attrs["checkpoint"] = str(_rl_paths(config).best)
        meta.attrs["action_semantics"] = "raw teacher output, clipped/executed action"
        for attempts in trange(1, max_attempts + 1, desc="phase0 causal audit"):
            reset_seed = seed_start + attempts
            obs, _info = env.reset(seed=reset_seed)
            observations = [_state_obs(obs)]
            simulator_states = [_numpy(env.unwrapped.get_state()).reshape(-1).astype(np.float32)]
            raw_actions: list[np.ndarray] = []
            clipped_actions: list[np.ndarray] = []
            executed_actions: list[np.ndarray] = []
            rewards: list[float] = []
            terminated_flags: list[bool] = []
            truncated_flags: list[bool] = []
            success_flags: list[bool] = []
            terminated = truncated = False
            success = False
            while not (terminated or truncated):
                state_t = torch.from_numpy(observations[-1][None]).to(device).float()
                raw_action = teacher.actor_mean(state_t).detach().cpu().numpy()[0].astype(np.float32)
                clipped_action = np.clip(raw_action, action_low, action_high).astype(np.float32)
                executed_action = (
                    clipped_action
                    if bool(config.get("policy.clip_actions_to_env_space", True))
                    else raw_action
                )
                next_obs, reward, terminated, truncated, info = env.step(executed_action)
                step_success = bool(_scalar(info.get("success", False)))
                success = success or step_success
                raw_actions.append(raw_action)
                clipped_actions.append(clipped_action)
                executed_actions.append(executed_action)
                rewards.append(_scalar(reward))
                terminated_flags.append(bool(_scalar(terminated)))
                truncated_flags.append(bool(_scalar(truncated)))
                success_flags.append(step_success)
                observations.append(_state_obs(next_obs))
                simulator_states.append(
                    _numpy(env.unwrapped.get_state()).reshape(-1).astype(np.float32)
                )
                obs = next_obs
            if not success:
                continue
            group = h5.create_group(f"episode_{successes:04d}")
            group.attrs["reset_seed"] = reset_seed
            group.attrs["success"] = success
            group.create_dataset("observations", data=np.stack(observations), compression="gzip")
            group.create_dataset(
                "simulator_states", data=np.stack(simulator_states), compression="gzip"
            )
            group.create_dataset("raw_actions", data=np.stack(raw_actions), compression="gzip")
            group.create_dataset(
                "clipped_actions", data=np.stack(clipped_actions), compression="gzip"
            )
            group.create_dataset(
                "executed_actions", data=np.stack(executed_actions), compression="gzip"
            )
            group.create_dataset("rewards", data=np.asarray(rewards, dtype=np.float32))
            group.create_dataset("terminated", data=np.asarray(terminated_flags, dtype=np.bool_))
            group.create_dataset("truncated", data=np.asarray(truncated_flags, dtype=np.bool_))
            group.create_dataset("step_success", data=np.asarray(success_flags, dtype=np.bool_))
            successes += 1
            if successes >= successful_trajectories:
                break
        meta.attrs["attempts"] = attempts
        meta.attrs["successful_trajectories"] = successes
    env.close()
    if successes < successful_trajectories:
        raise RuntimeError(
            f"Collected only {successes}/{successful_trajectories} successful audit trajectories"
        )
    return {
        "path": str(output_path),
        "dataset_type": "causal_dataset",
        "successful_trajectories": successes,
        "attempts": attempts,
    }


def action_alignment_metrics(
    observations: np.ndarray,
    stored_actions: np.ndarray,
    teacher_actions: np.ndarray,
) -> dict[str, float]:
    if len(observations) != len(stored_actions) + 1:
        raise ValueError("Expected one more observation than action")
    if teacher_actions.shape != stored_actions.shape:
        raise ValueError("Teacher and stored action shapes differ")
    metrics = {
        "shift_0_mae": float(np.mean(np.abs(stored_actions - teacher_actions))),
    }
    if len(stored_actions) > 1:
        metrics["shift_minus_1_mae"] = float(
            np.mean(np.abs(stored_actions[1:] - teacher_actions[:-1]))
        )
        metrics["shift_plus_1_mae"] = float(
            np.mean(np.abs(stored_actions[:-1] - teacher_actions[1:]))
        )
    return metrics


@torch.inference_mode()
def _audit_phase0_dataset(
    config: Config,
    teacher: PPOAgent,
    dataset_path: Path,
) -> dict[str, Any]:
    device = next(teacher.parameters()).device
    all_observations: list[np.ndarray] = []
    all_raw: list[np.ndarray] = []
    all_clipped: list[np.ndarray] = []
    all_executed: list[np.ndarray] = []
    per_episode: list[dict[str, Any]] = []
    with h5py.File(dataset_path, "r") as h5:
        for key in sorted(k for k in h5 if k.startswith("episode_")):
            group = h5[key]
            observations = np.asarray(group["observations"], dtype=np.float32)
            stored_raw = np.asarray(group["raw_actions"], dtype=np.float32)
            stored_clipped = np.asarray(group["clipped_actions"], dtype=np.float32)
            stored_executed = np.asarray(group["executed_actions"], dtype=np.float32)
            queried_raw = (
                teacher.actor_mean(torch.from_numpy(observations[:-1]).to(device).float())
                .detach()
                .cpu()
                .numpy()
                .astype(np.float32)
            )
            action_low = np.asarray(config.get("policy.action_low"), dtype=np.float32)
            action_high = np.asarray(config.get("policy.action_high"), dtype=np.float32)
            queried_clipped = np.clip(queried_raw, action_low, action_high).astype(np.float32)
            raw_alignment = action_alignment_metrics(observations, stored_raw, queried_raw)
            clipped_alignment = action_alignment_metrics(
                observations, stored_clipped, queried_clipped
            )
            per_episode.append(
                {
                    "episode": key,
                    "reset_seed": int(group.attrs["reset_seed"]),
                    "length": int(len(stored_executed)),
                    "raw_alignment": raw_alignment,
                    "clipped_alignment": clipped_alignment,
                }
            )
            all_observations.append(observations)
            all_raw.append(stored_raw)
            all_clipped.append(stored_clipped)
            all_executed.append(stored_executed)
    raw = np.concatenate(all_raw)
    clipped = np.concatenate(all_clipped)
    executed = np.concatenate(all_executed)
    action_norm = Standardizer.fit(executed)
    round_trip = action_norm.inverse(action_norm.transform(executed))
    return {
        "episodes": per_episode,
        "transition_count": int(len(executed)),
        "observation_dim": int(all_observations[0].shape[-1]),
        "action_dim": int(executed.shape[-1]),
        "raw_to_clipped_mae": float(np.mean(np.abs(raw - clipped))),
        "raw_out_of_bounds_fraction": float(np.mean(np.any(raw != clipped, axis=-1))),
        "clipped_to_executed_mae": float(np.mean(np.abs(clipped - executed))),
        "normalization_round_trip_max_abs_error": float(np.max(np.abs(round_trip - executed))),
        "global_shift_0_mae": float(
            np.mean([x["clipped_alignment"]["shift_0_mae"] for x in per_episode])
        ),
        "global_shift_minus_1_mae": float(
            np.mean([x["clipped_alignment"]["shift_minus_1_mae"] for x in per_episode])
        ),
        "global_shift_plus_1_mae": float(
            np.mean([x["clipped_alignment"]["shift_plus_1_mae"] for x in per_episode])
        ),
    }


def _replay_causal_transitions(config: Config, dataset_path: Path) -> dict[str, Any]:
    env = _make_scalar_state_env(config)
    episode_errors: list[float] = []
    with h5py.File(dataset_path, "r") as h5:
        for key in sorted(k for k in h5 if k.startswith("episode_")):
            group = h5[key]
            reset_seed = int(group.attrs["reset_seed"])
            expected_states = np.asarray(group["simulator_states"], dtype=np.float32)
            actions = np.asarray(group["executed_actions"], dtype=np.float32)
            env.reset(seed=reset_seed)
            max_error = float(
                np.max(
                    np.abs(
                        _numpy(env.unwrapped.get_state()).reshape(-1).astype(np.float32)
                        - expected_states[0]
                    )
                )
            )
            for step, action in enumerate(actions):
                env.step(action)
                actual = _numpy(env.unwrapped.get_state()).reshape(-1).astype(np.float32)
                max_error = max(max_error, float(np.max(np.abs(actual - expected_states[step + 1]))))
            episode_errors.append(max_error)
    env.close()
    return {
        "episode_max_abs_state_errors": episode_errors,
        "max_abs_state_error": float(max(episode_errors)),
        "tolerance": float(config.get("incremental.phase0.transition_replay_tolerance", 1e-5)),
    }


class _OverfitActor(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


class _TemporalConcatPolicy(nn.Module):
    def __init__(
        self,
        step_dim: int,
        history: int,
        action_dim: int,
        hidden_dim: int,
    ) -> None:
        super().__init__()
        self.step_dim = step_dim
        self.history = history
        self.action_dim = action_dim
        self.net = nn.Sequential(
            nn.Linear(step_dim * history, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        return self.net(history.flatten(start_dim=1))


class _TemporalGruPolicy(nn.Module):
    def __init__(
        self,
        step_dim: int,
        history: int,
        action_dim: int,
        hidden_dim: int,
    ) -> None:
        super().__init__()
        self.step_dim = step_dim
        self.history = history
        self.action_dim = action_dim
        self.input = nn.Linear(step_dim, hidden_dim)
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.output = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        x = self.input(history)
        out, _ = self.gru(x)
        return self.output(out[:, -1])


def _load_audit_queries(dataset_path: Path) -> tuple[list[np.ndarray], list[np.ndarray], list[int]]:
    states: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    reset_seeds: list[int] = []
    with h5py.File(dataset_path, "r") as h5:
        for key in sorted(k for k in h5 if k.startswith("episode_")):
            group = h5[key]
            states.append(np.asarray(group["observations"][:-1], dtype=np.float32))
            actions.append(np.asarray(group["executed_actions"], dtype=np.float32))
            reset_seeds.append(int(group.attrs["reset_seed"]))
    return states, actions, reset_seeds


def _train_overfit_actor(
    config: Config,
    states: np.ndarray,
    actions: np.ndarray,
    input_norm: Standardizer,
    action_norm: Standardizer,
    name: str,
    artifact_dir: Path,
) -> tuple[_OverfitActor, dict[str, Any]]:
    device = default_device()
    x = torch.from_numpy(input_norm.transform(states)).to(device).float()
    y = torch.from_numpy(action_norm.transform(actions)).to(device).float()
    model = _OverfitActor(
        states.shape[-1],
        actions.shape[-1],
        int(config.get("incremental.phase0.overfit_hidden_dim", 256)),
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=float(config.get("incremental.phase0.overfit_lr", 1e-3))
    )
    max_steps = int(config.get("incremental.phase0.overfit_max_steps", 10000))
    target_mae = float(config.get("incremental.phase0.overfit_target_normalized_mae", 1e-3))
    timer = Timer()
    final_loss = float("inf")
    final_mae = float("inf")
    step = 0
    for step in trange(1, max_steps + 1, desc=f"phase0 overfit {name}"):
        pred = model(x)
        loss = torch.mean((pred - y) ** 2)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        final_loss = float(loss.detach().cpu())
        final_mae = float(torch.mean(torch.abs(pred - y)).detach().cpu())
        if final_mae <= target_mae:
            break
    model.eval()
    with torch.inference_mode():
        pred_norm = model(x).detach().cpu().numpy()
    pred_action = action_norm.inverse(pred_norm)
    metrics = {
        "name": name,
        "samples": int(len(states)),
        "steps": step,
        "normalized_mse": final_loss,
        "normalized_mae": final_mae,
        "action_mae": float(np.mean(np.abs(pred_action - actions))),
        "action_max_abs_error": float(np.max(np.abs(pred_action - actions))),
        "elapsed_s": timer.elapsed(),
        "target_normalized_mae": target_mae,
        "passed": final_mae <= target_mae,
    }
    torch.save(
        {
            "model": model.state_dict(),
            "obs_dim": states.shape[-1],
            "action_dim": actions.shape[-1],
            "hidden_dim": int(config.get("incremental.phase0.overfit_hidden_dim", 256)),
            "input_norm": input_norm.state_dict(),
            "action_norm": action_norm.state_dict(),
            "metrics": metrics,
        },
        artifact_dir / f"overfit_{name}.pt",
    )
    return model, metrics


def _evaluate_overfit_on_seeds(
    config: Config,
    model: _OverfitActor,
    input_norm: Standardizer,
    action_norm: Standardizer,
    reset_seeds: list[int],
) -> dict[str, Any]:
    device = next(model.parameters()).device

    @torch.inference_mode()
    def policy(state: np.ndarray) -> np.ndarray:
        x = torch.from_numpy(input_norm.transform(state[None])).to(device).float()
        pred = model(x).detach().cpu().numpy()
        return action_norm.inverse(pred)[0]

    env = _make_scalar_state_env(config)
    action_low = np.asarray(env.action_space.low, dtype=np.float32)
    action_high = np.asarray(env.action_space.high, dtype=np.float32)
    successes = []
    for reset_seed in reset_seeds:
        obs, _info = env.reset(seed=reset_seed)
        terminated = truncated = False
        success = False
        while not (terminated or truncated):
            action = np.clip(policy(_state_obs(obs)), action_low, action_high).astype(np.float32)
            obs, _reward, terminated, truncated, info = env.step(action)
            success = success or bool(_scalar(info.get("success", False)))
        successes.append(float(success))
    env.close()
    return {
        "reset_seeds": reset_seeds,
        "success": float(np.mean(successes)),
        "successes": successes,
    }


def _run_overfit_ladder(
    config: Config,
    dataset_path: Path,
    artifact_dir: Path,
) -> dict[str, Any]:
    states_by_episode, actions_by_episode, reset_seeds = _load_audit_queries(dataset_path)
    all_states = np.concatenate(states_by_episode)
    all_actions = np.concatenate(actions_by_episode)
    input_norm = Standardizer.fit(all_states)
    action_norm = Standardizer.fit(all_actions)
    subsets = {
        "one_state": (states_by_episode[0][:1], actions_by_episode[0][:1], []),
        "one_trajectory": (states_by_episode[0], actions_by_episode[0], reset_seeds[:1]),
        "ten_trajectories": (all_states, all_actions, reset_seeds),
    }
    results: dict[str, Any] = {}
    for index, (name, (states, actions, seeds)) in enumerate(subsets.items()):
        set_seed(int(config.get("seed", 0)) + index)
        model, metrics = _train_overfit_actor(
            config,
            states,
            actions,
            input_norm,
            action_norm,
            name,
            artifact_dir,
        )
        if seeds:
            metrics["closed_loop_training_initializations"] = _evaluate_overfit_on_seeds(
                config,
                model,
                input_norm,
                action_norm,
                seeds,
            )
        results[name] = metrics
    return results


def run_phase0(config: Config, episodes: int | None = None, force: bool = False) -> Path:
    set_seed(int(config.get("seed", 0)))
    device = default_device()
    teacher = load_ppo_agent(_rl_paths(config).best, device)
    student = copied_actor_student(teacher, device)
    results_dir = ensure_dir(config.path_value("paths.incremental_results_dir") / "phase0")
    artifact_dir = ensure_dir(config.path_value("paths.incremental_artifact_dir") / "phase0")
    data_dir = ensure_dir(config.path_value("paths.incremental_data_dir"))
    output_path = results_dir / "phase0_summary.json"
    audit_path = data_dir / "phase0_causal_audit.h5"
    eval_episodes = int(episodes or config.get("incremental.phase0.eval_episodes", 500))
    eval_seed = int(config.get("incremental.phase0.eval_seed", 10000))

    comparison_states = np.random.default_rng(0).normal(
        size=(1024, teacher.obs_dim)
    ).astype(np.float32)
    with torch.inference_mode():
        comparison_t = torch.from_numpy(comparison_states).to(device)
        teacher_outputs = teacher.actor_mean(comparison_t)
        student_outputs = student(comparison_t)
    copied_max_error = float(torch.max(torch.abs(teacher_outputs - student_outputs)).cpu())

    vector_reference = evaluate_ppo(
        config,
        checkpoint=_rl_paths(config).best,
        episodes=eval_episodes,
    )
    teacher_eval = _evaluate_scalar_policy(
        config, _actor_policy(teacher.actor_mean, device), eval_episodes, eval_seed
    )
    student_eval = _evaluate_scalar_policy(
        config, _actor_policy(student, device), eval_episodes, eval_seed
    )

    if force or not audit_path.exists():
        collection = _collect_phase0_causal_audit(
            config,
            teacher,
            audit_path,
            int(config.get("incremental.phase0.audit_successful_trajectories", 10)),
        )
    else:
        with h5py.File(audit_path, "r") as h5:
            collection = {
                "path": str(audit_path),
                "dataset_type": str(h5["meta"].attrs["dataset_type"]),
                "successful_trajectories": int(h5["meta"].attrs["successful_trajectories"]),
                "attempts": int(h5["meta"].attrs["attempts"]),
            }
    alignment = _audit_phase0_dataset(config, teacher, audit_path)
    transition_replay = _replay_causal_transitions(config, audit_path)
    overfit = _run_overfit_ladder(config, audit_path, artifact_dir)

    teacher_student_success_gap = abs(teacher_eval["success"] - student_eval["success"])
    tolerance = float(config.get("incremental.phase0.transition_replay_tolerance", 1e-5))
    target_mae = float(config.get("incremental.phase0.overfit_target_normalized_mae", 1e-3))
    gate_checks = {
        "teacher_downstream_success_at_least_80pct": teacher_eval["success"] >= 0.80,
        "scalar_vector_success_within_5pp": abs(
            teacher_eval["success"] - vector_reference["success"]
        )
        <= 0.05,
        "copied_output_exact": copied_max_error == 0.0,
        "copied_success_within_1pp": teacher_student_success_gap <= 0.01,
        "state_action_alignment": alignment["global_shift_0_mae"] < min(
            alignment["global_shift_minus_1_mae"],
            alignment["global_shift_plus_1_mae"],
        ),
        "causal_transition_replay": transition_replay["max_abs_state_error"] <= tolerance,
        "one_state_overfit": overfit["one_state"]["normalized_mae"] <= target_mae,
        "one_trajectory_overfit": overfit["one_trajectory"]["normalized_mae"] <= target_mae,
    }
    payload = {
        "phase": 0,
        "metadata": _runtime_metadata(config),
        "teacher_checkpoint": str(_rl_paths(config).best),
        "teacher_vector_reference_eval": vector_reference,
        "teacher_scalar_eval": teacher_eval,
        "copied_student_scalar_eval": student_eval,
        "teacher_student_success_gap": teacher_student_success_gap,
        "copied_actor_max_abs_output_error": copied_max_error,
        "causal_collection": collection,
        "action_and_alignment_audit": alignment,
        "causal_transition_replay": transition_replay,
        "overfit_ladder": overfit,
        "gate_checks": gate_checks,
        "gate_passed": all(gate_checks.values()),
    }
    write_json(output_path, payload)
    console.print(payload)
    return output_path


@torch.inference_mode()
def collect_phase1_query_dataset(config: Config, force: bool = False) -> Path:
    output_path = config.path_value("paths.incremental_data_dir") / "phase1_query_dataset.h5"
    required_episodes = int(config.get("incremental.phase1.query_episodes", 2600))
    if output_path.exists() and not force:
        with h5py.File(output_path, "r") as h5:
            existing = int(h5["meta"].attrs["episodes"])
        if existing >= required_episodes:
            console.print(f"Phase 1 query dataset exists: {output_path}")
            return output_path

    device = default_device()
    teacher = load_ppo_agent(_rl_paths(config).best, device)
    num_envs = int(config.get("incremental.phase1.query_num_envs", 256))
    env = _make_state_env(
        config,
        num_envs,
        record_metrics=True,
        ignore_terminations=False,
        reconfiguration_freq=0,
    )
    action_low = torch.as_tensor(env.single_action_space.low, device=device, dtype=torch.float32)
    action_high = torch.as_tensor(env.single_action_space.high, device=device, dtype=torch.float32)
    obs, _info = env.reset(seed=int(config.get("incremental.phase1.query_seed", 30000)))
    obs = obs.to(device).float()
    state_buffers: list[list[np.ndarray]] = [[] for _ in range(num_envs)]
    raw_action_buffers: list[list[np.ndarray]] = [[] for _ in range(num_envs)]
    clipped_action_buffers: list[list[np.ndarray]] = [[] for _ in range(num_envs)]
    episodes: list[tuple[np.ndarray, np.ndarray, np.ndarray, bool]] = []
    progress = trange(required_episodes, desc="collect phase1 query episodes")
    while len(episodes) < required_episodes:
        raw_action = teacher.actor_mean(obs)
        clipped_action = torch.clamp(raw_action, action_low, action_high)
        states_np = obs.detach().cpu().numpy().astype(np.float32)
        raw_np = raw_action.detach().cpu().numpy().astype(np.float32)
        clipped_np = clipped_action.detach().cpu().numpy().astype(np.float32)
        for env_idx in range(num_envs):
            state_buffers[env_idx].append(states_np[env_idx])
            raw_action_buffers[env_idx].append(raw_np[env_idx])
            clipped_action_buffers[env_idx].append(clipped_np[env_idx])
        obs, _reward, _terminated, _truncated, info = env.step(clipped_action)
        obs = obs.to(device).float()
        if "final_info" not in info:
            continue
        final_mask = info["_final_info"].detach().cpu().numpy().astype(bool)
        success_once = (
            info["final_info"]["episode"]["success_once"].detach().cpu().numpy().astype(bool)
        )
        for env_idx in np.flatnonzero(final_mask):
            episodes.append(
                (
                    np.stack(state_buffers[env_idx]),
                    np.stack(raw_action_buffers[env_idx]),
                    np.stack(clipped_action_buffers[env_idx]),
                    bool(success_once[env_idx]),
                )
            )
            state_buffers[env_idx].clear()
            raw_action_buffers[env_idx].clear()
            clipped_action_buffers[env_idx].clear()
            progress.update(1)
            if len(episodes) >= required_episodes:
                break
    progress.close()
    env.close()

    ensure_dir(output_path.parent)
    tmp_path = output_path.with_suffix(".tmp.h5")
    if tmp_path.exists():
        tmp_path.unlink()
    with h5py.File(tmp_path, "w") as h5:
        meta = h5.create_group("meta")
        for key, value in _runtime_metadata(config).items():
            meta.attrs[key] = value
        meta.attrs["dataset_type"] = "query_dataset"
        meta.attrs["episodes"] = len(episodes)
        meta.attrs["successful_episodes"] = sum(int(ep[3]) for ep in episodes)
        meta.attrs["checkpoint"] = str(_rl_paths(config).best)
        meta.attrs["semantics"] = (
            "independent privileged state and deterministic teacher action pairs; "
            "no next-state claim"
        )
        for episode_idx, (states, raw_actions, clipped_actions, success) in enumerate(episodes):
            group = h5.create_group(f"episode_{episode_idx:05d}")
            group.attrs["success"] = success
            group.create_dataset("states", data=states, compression="gzip")
            group.create_dataset("teacher_raw_actions", data=raw_actions, compression="gzip")
            group.create_dataset(
                "teacher_clipped_actions", data=clipped_actions, compression="gzip"
            )
    tmp_path.replace(output_path)
    console.print(f"Wrote Phase 1 query dataset: {output_path}")
    return output_path


def _phase1_episode_keys(
    h5: h5py.File,
    subset: str,
) -> list[str]:
    keys = sorted(key for key in h5 if key.startswith("episode_"))
    if subset == "all":
        return keys
    if subset == "successful":
        return [key for key in keys if bool(h5[key].attrs["success"])]
    raise ValueError(f"Unknown Phase 1 subset: {subset}")


def _load_phase1_queries(
    dataset_path: Path,
    subset: str,
    train_episodes: int,
    validation_episodes: int,
    label_kind: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    label_dataset = {
        "deterministic_clipped": "teacher_clipped_actions",
        "deterministic_raw": "teacher_raw_actions",
    }.get(label_kind)
    if label_dataset is None:
        raise ValueError(f"Unknown Phase 1 label kind: {label_kind}")
    with h5py.File(dataset_path, "r") as h5:
        keys = _phase1_episode_keys(h5, subset)
        required = train_episodes + validation_episodes
        if len(keys) < required:
            raise ValueError(
                f"Phase 1 subset '{subset}' has {len(keys)} episodes, requires {required}"
            )
        train_keys = keys[:train_episodes]
        validation_keys = keys[-validation_episodes:]
        train_states = np.concatenate(
            [np.asarray(h5[key]["states"], dtype=np.float32) for key in train_keys]
        )
        train_actions = np.concatenate(
            [np.asarray(h5[key][label_dataset], dtype=np.float32) for key in train_keys]
        )
        validation_states = np.concatenate(
            [np.asarray(h5[key]["states"], dtype=np.float32) for key in validation_keys]
        )
        validation_actions = np.concatenate(
            [np.asarray(h5[key][label_dataset], dtype=np.float32) for key in validation_keys]
        )
    metadata = {
        "dataset_type": "query_dataset",
        "subset": subset,
        "label_kind": label_kind,
        "train_episodes": train_episodes,
        "validation_episodes": validation_episodes,
        "train_queries": int(len(train_states)),
        "validation_queries": int(len(validation_states)),
    }
    return train_states, train_actions, validation_states, validation_actions, metadata


def _identity_standardizer(dim: int) -> Standardizer:
    return Standardizer(np.zeros(dim, dtype=np.float32), np.ones(dim, dtype=np.float32))


def _action_regression_metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, Any]:
    correlations = []
    for dim in range(target.shape[-1]):
        if np.std(prediction[:, dim]) < 1e-8 or np.std(target[:, dim]) < 1e-8:
            correlations.append(float("nan"))
        else:
            correlations.append(float(np.corrcoef(prediction[:, dim], target[:, dim])[0, 1]))
    return {
        "mae": float(np.mean(np.abs(prediction - target))),
        "rmse": float(np.sqrt(np.mean((prediction - target) ** 2))),
        "max_abs_error": float(np.max(np.abs(prediction - target))),
        "correlation_per_dim": correlations,
        "prediction_out_of_bounds_fraction": float(
            np.mean(np.any((prediction < -1.0) | (prediction > 1.0), axis=-1))
        ),
        "target_near_bounds_fraction": float(
            np.mean(np.any(np.abs(target) >= 0.99, axis=-1))
        ),
    }


def train_phase1_bc(
    config: Config,
    n_episodes: int | None = None,
    seed: int = 0,
    subset: str = "all",
    label_kind: str = "deterministic_clipped",
    force: bool = False,
) -> Path:
    set_seed(seed)
    dataset_path = collect_phase1_query_dataset(config, force=False)
    n_episodes = int(n_episodes or config.get("incremental.phase1.train_episodes", 2000))
    validation_episodes = int(config.get("incremental.phase1.validation_episodes", 200))
    artifact_dir = ensure_dir(
        config.path_value("paths.incremental_artifact_dir")
        / "phase1"
        / f"n{n_episodes}"
        / f"seed{seed}"
    )
    checkpoint_path = artifact_dir / f"bc_{subset}_{label_kind}.pt"
    if checkpoint_path.exists() and not force:
        console.print(f"Phase 1 BC exists: {checkpoint_path}")
        return checkpoint_path
    train_states, train_actions, val_states, val_actions, data_metadata = (
        _load_phase1_queries(
            dataset_path,
            subset,
            n_episodes,
            validation_episodes,
            label_kind,
        )
    )
    normalize_inputs = bool(config.get("incremental.phase1.normalize_inputs", False))
    input_norm = (
        Standardizer.fit(train_states)
        if normalize_inputs
        else _identity_standardizer(train_states.shape[-1])
    )
    x_train = torch.from_numpy(input_norm.transform(train_states)).float()
    y_train = torch.from_numpy(train_actions).float()
    x_val = torch.from_numpy(input_norm.transform(val_states)).float()
    y_val = torch.from_numpy(val_actions).float()
    loader = DataLoader(
        TensorDataset(x_train, y_train),
        batch_size=int(config.get("incremental.phase1.batch_size", 4096)),
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    device = default_device()
    model = PPOAgent(
        train_states.shape[-1],
        train_actions.shape[-1],
        int(config.get("incremental.phase1.hidden_dim", 256)),
    ).actor_mean.to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(config.get("incremental.phase1.lr", 3e-4)),
    )
    epochs = int(config.get("incremental.phase1.epochs", 100))
    timer = Timer()
    best_val_loss = float("inf")
    best_state = None
    history: list[dict[str, float]] = []
    x_val_device = x_val.to(device)
    y_val_device = y_val.to(device)
    for epoch in trange(1, epochs + 1, desc=f"train phase1 BC {subset} n={n_episodes}"):
        model.train()
        train_loss_sum = 0.0
        train_count = 0
        for states, actions in loader:
            states = states.to(device, non_blocking=True)
            actions = actions.to(device, non_blocking=True)
            prediction = model(states)
            loss = torch.mean((prediction - actions) ** 2)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            train_loss_sum += float(loss.detach().cpu()) * len(states)
            train_count += len(states)
        model.eval()
        with torch.inference_mode():
            val_loss = float(torch.mean((model(x_val_device) - y_val_device) ** 2).cpu())
        train_loss = train_loss_sum / train_count
        history.append({"epoch": epoch, "train_mse": train_loss, "validation_mse": val_loss})
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
    if best_state is None:
        raise RuntimeError("Phase 1 BC training produced no checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    with torch.inference_mode():
        val_prediction = model(x_val_device).detach().cpu().numpy()
    validation_metrics = _action_regression_metrics(val_prediction, val_actions)
    payload = {
        "model": model.state_dict(),
        "obs_dim": train_states.shape[-1],
        "action_dim": train_actions.shape[-1],
        "hidden_dim": int(config.get("incremental.phase1.hidden_dim", 256)),
        "input_norm": input_norm.state_dict(),
        "normalize_inputs": normalize_inputs,
        "data": data_metadata,
        "validation_metrics": validation_metrics,
        "best_validation_mse": best_val_loss,
        "history": history,
        "elapsed_s": timer.elapsed(),
        "metadata": _runtime_metadata(config),
    }
    torch.save(payload, checkpoint_path)
    write_json(
        artifact_dir / f"bc_{subset}_{label_kind}_metrics.json",
        {
            "data": data_metadata,
            "validation_metrics": validation_metrics,
            "best_validation_mse": best_val_loss,
            "elapsed_s": timer.elapsed(),
        },
    )
    console.print(f"Wrote Phase 1 BC: {checkpoint_path}")
    return checkpoint_path


def _load_phase1_bc(checkpoint_path: Path, device: torch.device) -> tuple[nn.Module, dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = PPOAgent(
        int(checkpoint["obs_dim"]),
        int(checkpoint["action_dim"]),
        int(checkpoint["hidden_dim"]),
    ).actor_mean.to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, checkpoint


def evaluate_phase1_bc(
    config: Config,
    n_episodes: int | None = None,
    seed: int = 0,
    subset: str = "all",
    label_kind: str = "deterministic_clipped",
    episodes: int | None = None,
) -> Path:
    n_episodes = int(n_episodes or config.get("incremental.phase1.train_episodes", 2000))
    checkpoint_path = train_phase1_bc(
        config,
        n_episodes=n_episodes,
        seed=seed,
        subset=subset,
        label_kind=label_kind,
        force=False,
    )
    device = default_device()
    model, checkpoint = _load_phase1_bc(checkpoint_path, device)
    input_norm = Standardizer.from_state_dict(checkpoint["input_norm"])

    @torch.inference_mode()
    def policy(state: np.ndarray) -> np.ndarray:
        state_t = torch.from_numpy(input_norm.transform(state[None])).to(device).float()
        return model(state_t).detach().cpu().numpy()[0].astype(np.float32)

    eval_episodes = int(episodes or config.get("incremental.phase1.eval_episodes", 100))
    metrics = _evaluate_scalar_policy(
        config,
        policy,
        eval_episodes,
        int(config.get("incremental.phase1.eval_seed", 10000)),
    )
    results_dir = ensure_dir(
        config.path_value("paths.incremental_results_dir")
        / "phase1"
        / f"n{n_episodes}"
        / f"seed{seed}"
    )
    output_path = results_dir / f"bc_{subset}_{label_kind}.json"
    payload = {
        "phase": 1,
        "method": "privileged_bc",
        "subset": subset,
        "label_kind": label_kind,
        "n_episodes": n_episodes,
        "seed": seed,
        "closed_loop": metrics,
        "held_out_action_metrics": checkpoint["validation_metrics"],
        "data": checkpoint["data"],
        "metadata": _runtime_metadata(config),
        "gate_passed": metrics["success"] >= 0.70,
    }
    write_json(output_path, payload)
    console.print(payload)
    return output_path


def _phase2_query_path(config: Config, iteration: int, seed: int) -> Path:
    return (
        config.path_value("paths.incremental_data_dir")
        / "phase2_dagger"
        / f"seed{seed}"
        / f"iteration_{iteration:02d}.npz"
    )


@torch.inference_mode()
def collect_phase2_dagger_queries(
    config: Config,
    iteration: int,
    seed: int = 0,
    force: bool = False,
) -> Path:
    if iteration < 1:
        raise ValueError("DAgger iteration must be at least 1")
    output_path = _phase2_query_path(config, iteration, seed)
    required_queries = int(config.get("incremental.phase2.dagger_queries_per_iteration", 50000))
    if output_path.exists() and not force:
        with np.load(output_path) as data:
            existing = len(data["states"])
        if existing >= required_queries:
            console.print(f"Phase 2 DAgger queries exist: {output_path}")
            return output_path

    if iteration == 1:
        learner_path = (
            config.path_value("paths.incremental_artifact_dir")
            / "phase1"
            / f"n{int(config.get('incremental.phase1.train_episodes', 2000))}"
            / f"seed{seed}"
            / "bc_all_deterministic_raw.pt"
        )
    else:
        learner_path = (
            config.path_value("paths.incremental_artifact_dir")
            / "phase2"
            / f"iteration_{iteration - 1:02d}"
            / f"seed{seed}"
            / "bc_dagger.pt"
        )
    device = default_device()
    learner, learner_checkpoint = _load_phase1_bc(learner_path, device)
    learner_input_norm = Standardizer.from_state_dict(learner_checkpoint["input_norm"])
    teacher = load_ppo_agent(_rl_paths(config).best, device)
    num_envs = int(config.get("incremental.phase2.dagger_num_envs", 256))
    env = _make_state_env(
        config,
        num_envs,
        record_metrics=True,
        ignore_terminations=False,
        reconfiguration_freq=0,
    )
    action_low = torch.as_tensor(env.single_action_space.low, device=device, dtype=torch.float32)
    action_high = torch.as_tensor(env.single_action_space.high, device=device, dtype=torch.float32)
    obs, _info = env.reset(
        seed=int(config.get("incremental.phase2.dagger_seed", 40000)) + seed * 1000 + iteration
    )
    obs = obs.to(device).float()
    states: list[np.ndarray] = []
    teacher_raw_actions: list[np.ndarray] = []
    learner_raw_actions: list[np.ndarray] = []
    completed_successes: list[float] = []
    progress = trange(required_queries, desc=f"collect phase2 DAgger iteration {iteration}")
    collected = 0
    while collected < required_queries:
        normalized = learner_input_norm.transform(obs.detach().cpu().numpy().astype(np.float32))
        learner_raw = learner(torch.from_numpy(normalized).to(device).float())
        learner_executed = torch.clamp(learner_raw, action_low, action_high)
        teacher_raw = teacher.actor_mean(obs)
        take = min(num_envs, required_queries - collected)
        states.append(obs[:take].detach().cpu().numpy().astype(np.float32))
        teacher_raw_actions.append(
            teacher_raw[:take].detach().cpu().numpy().astype(np.float32)
        )
        learner_raw_actions.append(
            learner_raw[:take].detach().cpu().numpy().astype(np.float32)
        )
        collected += take
        progress.update(take)
        obs, _reward, _terminated, _truncated, info = env.step(learner_executed)
        obs = obs.to(device).float()
        if "final_info" in info:
            mask = info["_final_info"]
            if bool(mask.any()):
                success = info["final_info"]["episode"]["success_once"][mask]
                completed_successes.extend(
                    float(value) for value in success.detach().float().cpu().numpy()
                )
    progress.close()
    env.close()
    states_array = np.concatenate(states, axis=0)
    teacher_array = np.concatenate(teacher_raw_actions, axis=0)
    learner_array = np.concatenate(learner_raw_actions, axis=0)
    ensure_dir(output_path.parent)
    np.savez_compressed(
        output_path,
        states=states_array,
        teacher_raw_actions=teacher_array,
        learner_raw_actions=learner_array,
        dataset_type=np.asarray("query_dataset"),
        iteration=np.asarray(iteration),
        seed=np.asarray(seed),
        rollout_episodes=np.asarray(len(completed_successes)),
        rollout_success=np.asarray(
            float(np.mean(completed_successes)) if completed_successes else float("nan")
        ),
        git_commit=np.asarray(_git_metadata()["git_commit"]),
    )
    console.print(
        {
            "path": str(output_path),
            "queries": len(states_array),
            "rollout_episodes": len(completed_successes),
            "rollout_success": (
                float(np.mean(completed_successes)) if completed_successes else float("nan")
            ),
        }
    )
    return output_path


def train_phase2_dagger_bc(
    config: Config,
    iteration: int,
    seed: int = 0,
    force: bool = False,
) -> Path:
    if iteration < 1:
        raise ValueError("DAgger iteration must be at least 1")
    for current_iteration in range(1, iteration + 1):
        collect_phase2_dagger_queries(config, current_iteration, seed=seed, force=False)
    artifact_dir = ensure_dir(
        config.path_value("paths.incremental_artifact_dir")
        / "phase2"
        / f"iteration_{iteration:02d}"
        / f"seed{seed}"
    )
    checkpoint_path = artifact_dir / "bc_dagger.pt"
    if checkpoint_path.exists() and not force:
        console.print(f"Phase 2 DAgger BC exists: {checkpoint_path}")
        return checkpoint_path

    dataset_path = collect_phase1_query_dataset(config, force=False)
    train_states, train_actions, val_states, val_actions, base_metadata = (
        _load_phase1_queries(
            dataset_path,
            "all",
            int(config.get("incremental.phase1.train_episodes", 2000)),
            int(config.get("incremental.phase1.validation_episodes", 200)),
            "deterministic_raw",
        )
    )
    dagger_states = []
    dagger_actions = []
    iteration_metadata = []
    for current_iteration in range(1, iteration + 1):
        path = _phase2_query_path(config, current_iteration, seed)
        with np.load(path) as data:
            dagger_states.append(np.asarray(data["states"], dtype=np.float32))
            dagger_actions.append(np.asarray(data["teacher_raw_actions"], dtype=np.float32))
            iteration_metadata.append(
                {
                    "iteration": current_iteration,
                    "queries": int(len(data["states"])),
                    "rollout_episodes": int(data["rollout_episodes"]),
                    "rollout_success": float(data["rollout_success"]),
                }
            )
    all_train_states = np.concatenate([train_states, *dagger_states])
    all_train_actions = np.concatenate([train_actions, *dagger_actions])
    if iteration == 1:
        previous_path = (
            config.path_value("paths.incremental_artifact_dir")
            / "phase1"
            / f"n{int(config.get('incremental.phase1.train_episodes', 2000))}"
            / f"seed{seed}"
            / "bc_all_deterministic_raw.pt"
        )
    else:
        previous_path = (
            config.path_value("paths.incremental_artifact_dir")
            / "phase2"
            / f"iteration_{iteration - 1:02d}"
            / f"seed{seed}"
            / "bc_dagger.pt"
        )
    previous_model, previous_checkpoint = _load_phase1_bc(previous_path, default_device())
    input_norm = Standardizer.from_state_dict(previous_checkpoint["input_norm"])
    train_dataset = TensorDataset(
        torch.from_numpy(input_norm.transform(all_train_states)).float(),
        torch.from_numpy(all_train_actions).float(),
    )
    loader = DataLoader(
        train_dataset,
        batch_size=int(config.get("incremental.phase2.batch_size", 4096)),
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    device = default_device()
    model = previous_model.to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(config.get("incremental.phase2.lr", 5e-4)),
    )
    x_val = torch.from_numpy(input_norm.transform(val_states)).to(device).float()
    y_val = torch.from_numpy(val_actions).to(device).float()
    epochs = int(config.get("incremental.phase2.epochs", 100))
    model.eval()
    with torch.inference_mode():
        best_val_loss = float(torch.mean((model(x_val) - y_val) ** 2).cpu())
    best_state = copy.deepcopy(model.state_dict())
    history = []
    timer = Timer()
    for epoch in trange(1, epochs + 1, desc=f"train phase2 DAgger iteration {iteration}"):
        model.train()
        loss_sum = 0.0
        count = 0
        for states, actions in loader:
            states = states.to(device, non_blocking=True)
            actions = actions.to(device, non_blocking=True)
            prediction = model(states)
            loss = torch.mean((prediction - actions) ** 2)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach().cpu()) * len(states)
            count += len(states)
        model.eval()
        with torch.inference_mode():
            val_loss = float(torch.mean((model(x_val) - y_val) ** 2).cpu())
        history.append(
            {
                "epoch": epoch,
                "train_mse": loss_sum / count,
            "base_validation_mse": val_loss,
        }
    )
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
    if best_state is None:
        raise RuntimeError("Phase 2 DAgger training produced no checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    with torch.inference_mode():
        base_val_prediction = model(x_val).detach().cpu().numpy()
    payload = {
        "model": model.state_dict(),
        "obs_dim": all_train_states.shape[-1],
        "action_dim": all_train_actions.shape[-1],
        "hidden_dim": int(config.get("incremental.phase1.hidden_dim", 256)),
        "input_norm": input_norm.state_dict(),
        "normalize_inputs": True,
        "data": {
            "dataset_type": "query_dataset",
            "base": base_metadata,
            "dagger_iterations": iteration_metadata,
            "total_train_queries": int(len(all_train_states)),
        },
        "validation_metrics": _action_regression_metrics(base_val_prediction, val_actions),
        "best_validation_mse": best_val_loss,
        "history": history,
        "elapsed_s": timer.elapsed(),
        "metadata": _runtime_metadata(config),
    }
    torch.save(payload, checkpoint_path)
    write_json(
        artifact_dir / "bc_dagger_metrics.json",
        {
            "data": payload["data"],
            "validation_metrics": payload["validation_metrics"],
            "best_validation_mse": best_val_loss,
            "elapsed_s": timer.elapsed(),
        },
    )
    console.print(f"Wrote Phase 2 DAgger BC: {checkpoint_path}")
    return checkpoint_path


def evaluate_phase2_dagger_bc(
    config: Config,
    iteration: int,
    seed: int = 0,
    episodes: int | None = None,
) -> Path:
    checkpoint_path = train_phase2_dagger_bc(config, iteration, seed=seed, force=False)
    device = default_device()
    model, checkpoint = _load_phase1_bc(checkpoint_path, device)
    input_norm = Standardizer.from_state_dict(checkpoint["input_norm"])

    @torch.inference_mode()
    def policy(state: np.ndarray) -> np.ndarray:
        state_t = torch.from_numpy(input_norm.transform(state[None])).to(device).float()
        return model(state_t).detach().cpu().numpy()[0].astype(np.float32)

    eval_episodes = int(episodes or config.get("incremental.phase2.eval_episodes", 100))
    metrics = _evaluate_scalar_policy(
        config,
        policy,
        eval_episodes,
        int(config.get("incremental.phase2.eval_seed", 10000)),
    )
    results_dir = ensure_dir(
        config.path_value("paths.incremental_results_dir")
        / "phase2"
        / f"iteration_{iteration:02d}"
        / f"seed{seed}"
    )
    output_path = results_dir / "bc_dagger.json"
    payload = {
        "phase": 2,
        "method": "privileged_bc_dagger",
        "iteration": iteration,
        "seed": seed,
        "closed_loop": metrics,
        "held_out_action_metrics": checkpoint["validation_metrics"],
        "data": checkpoint["data"],
        "metadata": _runtime_metadata(config),
        "gate_passed": metrics["success"] >= 0.80,
    }
    write_json(output_path, payload)
    console.print(payload)
    return output_path


def _load_phase0_simulator_states(config: Config) -> list[np.ndarray]:
    path = config.path_value("paths.incremental_data_dir") / "phase0_causal_audit.h5"
    states = []
    with h5py.File(path, "r") as h5:
        for key in sorted(k for k in h5 if k.startswith("episode_")):
            episode_states = np.asarray(h5[key]["simulator_states"], dtype=np.float32)
            if len(episode_states) > 6:
                states.extend(episode_states[:-5])
    if not states:
        raise ValueError(f"No usable simulator states in {path}")
    return states


def _perturbed_recovery_states(
    config: Config,
    samples: int,
) -> tuple[torch.Tensor, np.ndarray]:
    rng = np.random.default_rng(int(config.get("incremental.phase2.recovery_seed", 50000)))
    source_states = _load_phase0_simulator_states(config)
    chosen = np.stack(
        [source_states[index] for index in rng.integers(0, len(source_states), size=samples)]
    )
    env = gym.make(
        config.get("env_id"),
        obs_mode="state",
        control_mode=config.get("control_mode"),
        reward_mode="normalized_dense",
        render_mode=None,
        sim_backend="physx_cuda",
        num_envs=samples,
        reconfiguration_freq=0,
    )
    env.reset(seed=int(config.get("incremental.phase2.recovery_seed", 50000)))
    device = default_device()
    env.unwrapped.set_state(torch.from_numpy(chosen).to(device).float())
    state_dict = env.unwrapped.get_state_dict()
    tee = state_dict["actors"]["Tee"].clone()
    xy_std = float(config.get("incremental.phase2.recovery_xy_std_m", 0.01))
    yaw_std = np.deg2rad(float(config.get("incremental.phase2.recovery_yaw_std_deg", 5.0)))
    perturbations = np.zeros((samples, 3), dtype=np.float32)
    perturbations[:, :2] = np.clip(
        rng.normal(0.0, xy_std, size=(samples, 2)),
        -2.0 * xy_std,
        2.0 * xy_std,
    )
    perturbations[:, 2] = np.clip(
        rng.normal(0.0, yaw_std, size=samples),
        -2.0 * yaw_std,
        2.0 * yaw_std,
    )
    tee[:, :2] += torch.from_numpy(perturbations[:, :2]).to(tee.device)
    current_yaw = 2.0 * torch.atan2(tee[:, 6], tee[:, 3])
    perturbed_yaw = current_yaw + torch.from_numpy(perturbations[:, 2]).to(tee.device)
    tee[:, 3] = torch.cos(0.5 * perturbed_yaw)
    tee[:, 4:6] = 0.0
    tee[:, 6] = torch.sin(0.5 * perturbed_yaw)
    state_dict["actors"]["Tee"] = tee
    env.unwrapped.set_state_dict(state_dict)
    perturbed_states = env.unwrapped.get_state().detach().clone()
    env.close()
    return perturbed_states, perturbations


@torch.inference_mode()
def _rollout_recovery_batch(
    config: Config,
    initial_states: torch.Tensor,
    policy: Callable[[torch.Tensor], torch.Tensor],
    collect_trajectories: bool,
) -> tuple[np.ndarray, list[dict[str, np.ndarray]]]:
    samples = len(initial_states)
    env = gym.make(
        config.get("env_id"),
        obs_mode="state",
        control_mode=config.get("control_mode"),
        reward_mode="normalized_dense",
        render_mode=None,
        sim_backend="physx_cuda",
        num_envs=samples,
        reconfiguration_freq=0,
    )
    env.reset(seed=int(config.get("incremental.phase2.recovery_seed", 50000)))
    env.unwrapped.set_state(initial_states)
    obs = env.unwrapped.get_obs().to(default_device()).float()
    action_low = torch.as_tensor(env.action_space.low, device=obs.device, dtype=torch.float32)
    action_high = torch.as_tensor(env.action_space.high, device=obs.device, dtype=torch.float32)
    active = np.ones(samples, dtype=bool)
    success = np.zeros(samples, dtype=bool)
    state_buffers: list[list[np.ndarray]] = [[] for _ in range(samples)]
    simulator_state_buffers: list[list[np.ndarray]] = [[] for _ in range(samples)]
    action_buffers: list[list[np.ndarray]] = [[] for _ in range(samples)]
    final_simulator_states: list[np.ndarray | None] = [None for _ in range(samples)]
    max_steps = int(config.get("incremental.phase2.recovery_max_steps", 100))
    for _step in range(max_steps):
        if not active.any():
            break
        raw_action = policy(obs)
        executed_action = torch.clamp(raw_action, action_low, action_high)
        if collect_trajectories:
            obs_np = obs.detach().cpu().numpy().astype(np.float32)
            sim_np = (
                env.unwrapped.get_state().detach().cpu().numpy().astype(np.float32)
            )
            action_np = executed_action.detach().cpu().numpy().astype(np.float32)
            for index in np.flatnonzero(active):
                state_buffers[index].append(obs_np[index])
                simulator_state_buffers[index].append(sim_np[index])
                action_buffers[index].append(action_np[index])
        obs, _reward, terminated, truncated, info = env.step(executed_action)
        obs = obs.to(default_device()).float()
        next_simulator_states = (
            env.unwrapped.get_state().detach().cpu().numpy().astype(np.float32)
        )
        step_success = _numpy(info.get("success", np.zeros(samples))).reshape(-1).astype(bool)
        success |= step_success
        done = (
            _numpy(terminated).reshape(-1).astype(bool)
            | _numpy(truncated).reshape(-1).astype(bool)
            | step_success
        )
        for index in np.flatnonzero(active & done):
            final_simulator_states[index] = next_simulator_states[index]
        active &= ~done
    for index in np.flatnonzero(active):
        final_simulator_states[index] = next_simulator_states[index]
    env.close()
    trajectories = []
    if collect_trajectories:
        for index in range(samples):
            if not state_buffers[index]:
                continue
            trajectories.append(
                {
                    "sample_index": np.asarray(index),
                    "states": np.stack(state_buffers[index]),
                    "simulator_states": np.concatenate(
                        [
                            np.stack(simulator_state_buffers[index]),
                            np.asarray(final_simulator_states[index], dtype=np.float32)[None],
                        ],
                        axis=0,
                    ),
                    "actions": np.stack(action_buffers[index]),
                    "success": np.asarray(success[index]),
                }
            )
    return success, trajectories


def evaluate_phase2_recovery(
    config: Config,
    iteration: int = 3,
    seed: int = 0,
    samples: int | None = None,
    force: bool = False,
) -> Path:
    samples = int(samples or config.get("incremental.phase2.recovery_samples", 128))
    results_dir = ensure_dir(config.path_value("paths.incremental_results_dir") / "phase2")
    output_path = results_dir / f"recovery_iteration_{iteration:02d}_seed{seed}.json"
    causal_path = (
        config.path_value("paths.incremental_data_dir")
        / "phase2_recovery"
        / f"iteration_{iteration:02d}_seed{seed}.h5"
    )
    if output_path.exists() and causal_path.exists() and not force:
        console.print(f"Phase 2 recovery result exists: {output_path}")
        return output_path
    initial_states, perturbations = _perturbed_recovery_states(config, samples)
    device = default_device()
    teacher = load_ppo_agent(_rl_paths(config).best, device)
    learner_path = train_phase2_dagger_bc(config, iteration, seed=seed, force=False)
    learner, learner_checkpoint = _load_phase1_bc(learner_path, device)
    learner_input_norm = Standardizer.from_state_dict(learner_checkpoint["input_norm"])

    def teacher_policy(obs: torch.Tensor) -> torch.Tensor:
        return teacher.actor_mean(obs)

    def learner_policy(obs: torch.Tensor) -> torch.Tensor:
        normalized = learner_input_norm.transform(obs.detach().cpu().numpy().astype(np.float32))
        return learner(torch.from_numpy(normalized).to(device).float())

    teacher_success, teacher_trajectories = _rollout_recovery_batch(
        config,
        initial_states,
        teacher_policy,
        collect_trajectories=True,
    )
    learner_success, _ = _rollout_recovery_batch(
        config,
        initial_states,
        learner_policy,
        collect_trajectories=False,
    )
    teacher_recoverable = teacher_success
    paired_learner_success = learner_success[teacher_recoverable]
    ensure_dir(causal_path.parent)
    tmp_path = causal_path.with_suffix(".tmp.h5")
    if tmp_path.exists():
        tmp_path.unlink()
    with h5py.File(tmp_path, "w") as h5:
        meta = h5.create_group("meta")
        for key, value in _runtime_metadata(config).items():
            meta.attrs[key] = value
        meta.attrs["dataset_type"] = "causal_dataset"
        meta.attrs["source"] = "perturbed_state_then_deterministic_teacher_rollout"
        meta.attrs["samples"] = samples
        meta.attrs["teacher_recoverable"] = int(teacher_recoverable.sum())
        saved = 0
        for trajectory in teacher_trajectories:
            index = int(trajectory["sample_index"])
            if not bool(trajectory["success"]):
                continue
            group = h5.create_group(f"episode_{saved:04d}")
            group.attrs["source_sample_index"] = index
            group.attrs["success"] = True
            group.create_dataset("perturbation_xy_yaw", data=perturbations[index])
            group.create_dataset("states", data=trajectory["states"], compression="gzip")
            group.create_dataset(
                "simulator_states",
                data=trajectory["simulator_states"],
                compression="gzip",
            )
            group.create_dataset("actions", data=trajectory["actions"], compression="gzip")
            saved += 1
    tmp_path.replace(causal_path)
    payload = {
        "phase": 2,
        "iteration": iteration,
        "seed": seed,
        "samples": samples,
        "perturbation": {
            "xy_std_m": float(config.get("incremental.phase2.recovery_xy_std_m", 0.01)),
            "yaw_std_deg": float(
                config.get("incremental.phase2.recovery_yaw_std_deg", 5.0)
            ),
        },
        "teacher_recovery_success": float(np.mean(teacher_success)),
        "teacher_recoverable_samples": int(teacher_recoverable.sum()),
        "learner_recovery_success_all": float(np.mean(learner_success)),
        "learner_recovery_success_when_teacher_recovers": (
            float(np.mean(paired_learner_success))
            if len(paired_learner_success)
            else float("nan")
        ),
        "causal_dataset": str(causal_path),
        "gate_passed": bool(
            len(paired_learner_success)
            and float(np.mean(paired_learner_success)) >= 0.80
        ),
        "metadata": _runtime_metadata(config),
    }
    write_json(output_path, payload)
    console.print(payload)
    return output_path


def _load_phase3_aggregate_queries(
    config: Config,
    iteration: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    label_kind = str(config.get("incremental.phase3.label_kind", "deterministic_raw"))
    dataset_path = collect_phase1_query_dataset(config, force=False)
    train_states, train_actions, val_states, val_actions, base_metadata = _load_phase1_queries(
        dataset_path,
        "all",
        int(config.get("incremental.phase1.train_episodes", 2000)),
        int(config.get("incremental.phase1.validation_episodes", 200)),
        label_kind,
    )
    dagger_states = []
    dagger_actions = []
    dagger_metadata = []
    for current_iteration in range(1, iteration + 1):
        path = _phase2_query_path(config, current_iteration, seed=0)
        if not path.exists():
            raise FileNotFoundError(f"Missing DAgger query file: {path}")
        with np.load(path) as data:
            dagger_states.append(np.asarray(data["states"], dtype=np.float32))
            teacher_actions = np.asarray(data["teacher_raw_actions"], dtype=np.float32)
            if label_kind == "deterministic_clipped":
                teacher_actions = np.clip(teacher_actions, -1.0, 1.0).astype(np.float32)
            elif label_kind != "deterministic_raw":
                raise ValueError(f"Unsupported Phase 3 label kind: {label_kind}")
            dagger_actions.append(teacher_actions)
            dagger_metadata.append(
                {
                    "iteration": current_iteration,
                    "queries": int(len(data["states"])),
                    "rollout_success": float(data["rollout_success"]),
                }
            )
    all_states = np.concatenate([train_states, *dagger_states], axis=0)
    all_actions = np.concatenate([train_actions, *dagger_actions], axis=0)
    metadata = {
        "dataset_type": "query_dataset",
        "label_kind": label_kind,
        "base": base_metadata,
        "dagger_iterations": dagger_metadata,
        "train_queries": int(len(all_states)),
        "validation_queries": int(len(val_states)),
    }
    return all_states, all_actions, val_states, val_actions, metadata


def _phase3_action_metrics(
    model: FlowModel,
    states: np.ndarray,
    actions: np.ndarray,
    input_norm: Standardizer,
    action_norm: Standardizer,
    flow_steps: int,
    sample_repeats: int = 1,
) -> dict[str, Any]:
    device = next(model.parameters()).device
    preds = []
    batch_size = 8192
    with torch.inference_mode():
        for start in range(0, len(states), batch_size):
            cond_np = input_norm.transform(states[start : start + batch_size])
            cond = torch.from_numpy(cond_np).to(device).float()
            samples = []
            for _ in range(sample_repeats):
                pred_norm = sample_flow(model, cond, flow_steps, model.sample_dim)
                samples.append(pred_norm.detach().cpu().numpy())
            pred_norm_np = np.mean(samples, axis=0)
            preds.append(action_norm.inverse(pred_norm_np))
    prediction = np.concatenate(preds, axis=0)
    metrics = _action_regression_metrics(prediction, actions)
    if sample_repeats > 1:
        metrics["sample_repeats"] = sample_repeats
    return metrics


def _phase3_zero_noise_action_metrics(
    model: FlowModel,
    states: np.ndarray,
    actions: np.ndarray,
    input_norm: Standardizer,
    action_norm: Standardizer,
    flow_steps: int,
) -> dict[str, Any]:
    device = next(model.parameters()).device
    preds = []
    batch_size = 8192
    with torch.inference_mode():
        for start in range(0, len(states), batch_size):
            cond = torch.from_numpy(input_norm.transform(states[start : start + batch_size])).to(
                device
            ).float()
            noise = torch.zeros(cond.shape[0], model.sample_dim, device=device, dtype=cond.dtype)
            pred_norm = sample_flow(
                model,
                cond,
                flow_steps,
                model.sample_dim,
                initial_noise=noise,
            )
            preds.append(action_norm.inverse(pred_norm.detach().cpu().numpy()))
    prediction = np.concatenate(preds, axis=0)
    metrics = _action_regression_metrics(prediction, actions)
    metrics["mode"] = "zero_noise"
    return metrics


def _integrate_flow_train(
    model: FlowModel,
    cond: torch.Tensor,
    steps: int,
    sample_dim: int,
    initial_noise: torch.Tensor,
) -> torch.Tensor:
    x = initial_noise
    dt = 1.0 / steps
    for step in range(steps):
        t = torch.full((cond.shape[0],), step / steps, device=cond.device, dtype=cond.dtype)
        x = x + dt * model(x, t, cond)
    return x


def _flow_sampling_diagnostics(
    model: FlowModel,
    states: np.ndarray,
    actions: np.ndarray,
    input_norm: Standardizer,
    action_norm: Standardizer,
    flow_steps: int,
) -> dict[str, Any]:
    device = next(model.parameters()).device
    count = min(256, len(states))
    cond = torch.from_numpy(input_norm.transform(states[:count])).to(device).float()
    samples = []
    with torch.inference_mode():
        for _ in range(16):
            pred = sample_flow(model, cond, flow_steps, model.sample_dim)
            samples.append(action_norm.inverse(pred.detach().cpu().numpy()))
    sample_arr = np.stack(samples, axis=0)
    mean_action = sample_arr.mean(axis=0)
    return {
        "states": count,
        "sample_action_std_mean": float(sample_arr.std(axis=0).mean()),
        "sample_mean_action_mae": float(np.mean(np.abs(mean_action - actions[:count]))),
        "single_sample_action_mae": float(np.mean(np.abs(sample_arr[0] - actions[:count]))),
        "preclip_out_of_bounds_fraction": float(
            np.mean(np.any((sample_arr < -1.0) | (sample_arr > 1.0), axis=-1))
        ),
    }


def _run_phase3_overfit_diagnostics(
    config: Config,
    states: np.ndarray,
    actions: np.ndarray,
    input_norm: Standardizer,
    action_norm: Standardizer,
) -> dict[str, Any]:
    device = default_device()
    out: dict[str, Any] = {}
    for name, count in [("one_query", 1), ("ten_queries", 10), ("hundred_queries", 100)]:
        set_seed(10_000 + count)
        model = FlowModel(
            sample_dim=actions.shape[-1],
            cond_dim=states.shape[-1],
            hidden_dim=int(config.get("incremental.phase3.hidden_dim", 256)),
        ).to(device)
        optimizer = torch.optim.Adam(
            model.parameters(), lr=float(config.get("incremental.phase3.overfit_lr", 1e-3))
        )
        cond = torch.from_numpy(input_norm.transform(states[:count])).to(device).float()
        target = torch.from_numpy(action_norm.transform(actions[:count])).to(device).float()
        steps = int(config.get("incremental.phase3.overfit_steps", 5000))
        last_loss = 0.0
        for _ in range(steps):
            loss = flow_matching_loss(model, target, cond)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            last_loss = float(loss.detach().cpu())
        metrics = _phase3_action_metrics(
            model,
            states[:count],
            actions[:count],
            input_norm,
            action_norm,
            int(config.get("incremental.phase3.flow_steps", 24)),
            sample_repeats=8,
        )
        metrics["final_flow_loss"] = last_loss
        metrics["queries"] = count
        out[name] = metrics
    return out


def train_phase3_flow(
    config: Config,
    seed: int = 0,
    force: bool = False,
) -> Path:
    set_seed(seed)
    iteration = int(config.get("incremental.phase3.dagger_iteration", 3))
    artifact_dir = ensure_dir(
        config.path_value("paths.incremental_artifact_dir")
        / "phase3"
        / f"seed{seed}"
    )
    checkpoint_path = artifact_dir / "one_step_flow.pt"
    if checkpoint_path.exists() and not force:
        console.print(f"Phase 3 flow exists: {checkpoint_path}")
        return checkpoint_path
    train_states, train_actions, val_states, val_actions, data_metadata = (
        _load_phase3_aggregate_queries(config, iteration)
    )
    input_norm = Standardizer.fit(train_states)
    action_norm = Standardizer.fit(train_actions)
    train_dataset = TensorDataset(
        torch.from_numpy(input_norm.transform(train_states)).float(),
        torch.from_numpy(action_norm.transform(train_actions)).float(),
    )
    loader = DataLoader(
        train_dataset,
        batch_size=int(config.get("incremental.phase3.batch_size", 4096)),
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    device = default_device()
    model = FlowModel(
        sample_dim=train_actions.shape[-1],
        cond_dim=train_states.shape[-1],
        hidden_dim=int(config.get("incremental.phase3.hidden_dim", 256)),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(config.get("incremental.phase3.lr", 5e-4))
    )
    val_cond = torch.from_numpy(input_norm.transform(val_states)).to(device).float()
    val_target = torch.from_numpy(action_norm.transform(val_actions)).to(device).float()
    epochs = int(config.get("incremental.phase3.epochs", 200))
    best_state = None
    best_val_loss = float("inf")
    best_val_action_mae = float("inf")
    history = []
    timer = Timer()
    for epoch in trange(1, epochs + 1, desc="train phase3 one-step flow"):
        model.train()
        train_loss_sum = 0.0
        train_count = 0
        for states, actions in loader:
            states = states.to(device, non_blocking=True)
            actions = actions.to(device, non_blocking=True)
            loss = flow_matching_loss(model, actions, states)
            consistency_weight = float(
                config.get("incremental.phase3.endpoint_consistency_weight", 0.0)
            )
            if consistency_weight > 0.0:
                consistency_count = min(
                    int(config.get("incremental.phase3.endpoint_consistency_batch", 512)),
                    len(states),
                )
                consistency_steps = int(
                    config.get("incremental.phase3.endpoint_consistency_steps", 4)
                )
                zero = torch.zeros(
                    consistency_count,
                    model.sample_dim,
                    device=device,
                    dtype=states.dtype,
                )
                endpoint = _integrate_flow_train(
                    model,
                    states[:consistency_count],
                    consistency_steps,
                    model.sample_dim,
                    zero,
                )
                loss = loss + consistency_weight * torch.mean(
                    (endpoint - actions[:consistency_count]) ** 2
                )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            train_loss_sum += float(loss.detach().cpu()) * len(states)
            train_count += len(states)
        model.eval()
        with torch.inference_mode():
            # Use the loss as a cheap validation criterion. Action MAE is computed after training.
            val_loss = float(flow_matching_loss(model, val_target, val_cond).detach().cpu())
        history.append(
            {
                "epoch": epoch,
                "train_flow_loss": train_loss_sum / train_count,
                "validation_flow_loss": val_loss,
            }
        )
        row = history[-1]
        if epoch % int(config.get("incremental.phase3.validation_action_interval", 10)) == 0:
            subset = min(int(config.get("incremental.phase3.validation_action_subset", 4096)), len(val_states))
            if str(config.get("incremental.phase3.eval_mode", "sample_mean")) == "zero_noise":
                action_metrics = _phase3_zero_noise_action_metrics(
                    model,
                    val_states[:subset],
                    val_actions[:subset],
                    input_norm,
                    action_norm,
                    int(config.get("incremental.phase3.flow_steps", 24)),
                )
            else:
                action_metrics = _phase3_action_metrics(
                    model,
                    val_states[:subset],
                    val_actions[:subset],
                    input_norm,
                    action_norm,
                    int(config.get("incremental.phase3.flow_steps", 24)),
                    sample_repeats=2,
                )
            row["validation_action_mae"] = action_metrics["mae"]
            if action_metrics["mae"] < best_val_action_mae:
                best_val_action_mae = action_metrics["mae"]
                best_val_loss = val_loss
                best_state = copy.deepcopy(model.state_dict())
        elif best_state is None and val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
    if best_state is None:
        raise RuntimeError("Phase 3 flow training produced no checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    flow_steps = int(config.get("incremental.phase3.flow_steps", 24))
    validation_action_metrics = (
        _phase3_zero_noise_action_metrics(
            model, val_states, val_actions, input_norm, action_norm, flow_steps
        )
        if str(config.get("incremental.phase3.eval_mode", "sample_mean")) == "zero_noise"
        else _phase3_action_metrics(
            model, val_states, val_actions, input_norm, action_norm, flow_steps, sample_repeats=4
        )
    )
    sampling = _flow_sampling_diagnostics(
        model, val_states, val_actions, input_norm, action_norm, flow_steps
    )
    overfit = _run_phase3_overfit_diagnostics(
        config,
        train_states,
        train_actions,
        input_norm,
        action_norm,
    )
    payload = {
        "model": model.state_dict(),
        "sample_dim": train_actions.shape[-1],
        "cond_dim": train_states.shape[-1],
        "hidden_dim": int(config.get("incremental.phase3.hidden_dim", 256)),
        "input_norm": input_norm.state_dict(),
        "action_norm": action_norm.state_dict(),
        "flow_steps": flow_steps,
        "data": data_metadata,
        "history": history,
        "best_validation_flow_loss": best_val_loss,
        "best_validation_action_mae": best_val_action_mae,
        "validation_action_metrics": validation_action_metrics,
        "sampling_diagnostics": sampling,
        "overfit_diagnostics": overfit,
        "elapsed_s": timer.elapsed(),
        "metadata": _runtime_metadata(config),
    }
    torch.save(payload, checkpoint_path)
    write_json(
        artifact_dir / "one_step_flow_metrics.json",
        {
            "data": data_metadata,
            "best_validation_flow_loss": best_val_loss,
            "best_validation_action_mae": best_val_action_mae,
            "validation_action_metrics": validation_action_metrics,
            "sampling_diagnostics": sampling,
            "overfit_diagnostics": overfit,
            "elapsed_s": timer.elapsed(),
        },
    )
    console.print(f"Wrote Phase 3 one-step flow: {checkpoint_path}")
    return checkpoint_path


def _load_phase3_flow(path: Path, device: torch.device) -> tuple[FlowModel, dict[str, Any]]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model = FlowModel(
        sample_dim=int(checkpoint["sample_dim"]),
        cond_dim=int(checkpoint["cond_dim"]),
        hidden_dim=int(checkpoint["hidden_dim"]),
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, checkpoint


def evaluate_phase3_flow(
    config: Config,
    seed: int = 0,
    episodes: int | None = None,
) -> Path:
    checkpoint_path = train_phase3_flow(config, seed=seed, force=False)
    device = default_device()
    model, checkpoint = _load_phase3_flow(checkpoint_path, device)
    input_norm = Standardizer.from_state_dict(checkpoint["input_norm"])
    action_norm = Standardizer.from_state_dict(checkpoint["action_norm"])
    flow_steps = int(checkpoint["flow_steps"])
    sample_dim = int(checkpoint["sample_dim"])
    eval_samples = int(config.get("incremental.phase3.eval_samples", 1))
    eval_mode = str(config.get("incremental.phase3.eval_mode", "sample_mean"))

    @torch.inference_mode()
    def policy(state: np.ndarray) -> np.ndarray:
        cond = torch.from_numpy(input_norm.transform(state[None])).to(device).float()
        if eval_mode == "zero_noise":
            noise = torch.zeros(cond.shape[0], sample_dim, device=device, dtype=cond.dtype)
            action_normed = sample_flow(
                model, cond, flow_steps, sample_dim, initial_noise=noise
            ).detach().cpu().numpy()
        elif eval_mode == "sample_mean":
            samples = []
            for _ in range(eval_samples):
                pred_norm = sample_flow(model, cond, flow_steps, sample_dim)
                samples.append(pred_norm.detach().cpu().numpy())
            action_normed = np.mean(samples, axis=0)
        else:
            raise ValueError(f"Unknown Phase 3 eval mode: {eval_mode}")
        return action_norm.inverse(action_normed)[0]

    eval_episodes = int(episodes or config.get("incremental.phase3.eval_episodes", 100))
    metrics = _evaluate_scalar_policy(
        config,
        policy,
        eval_episodes,
        int(config.get("incremental.phase3.eval_seed", 10000)),
    )
    bc_reference_path = (
        config.path_value("paths.incremental_results_dir")
        / "phase2"
        / "iteration_03"
        / f"seed{seed}"
        / "bc_dagger.json"
    )
    if not bc_reference_path.exists():
        raise FileNotFoundError(f"Missing Phase 2 BC reference: {bc_reference_path}")
    import json

    with bc_reference_path.open("r", encoding="utf-8") as f:
        bc_reference = json.load(f)
    bc_success = float(bc_reference["closed_loop"]["success"])
    results_dir = ensure_dir(
        config.path_value("paths.incremental_results_dir") / "phase3" / f"seed{seed}"
    )
    output_path = results_dir / "one_step_flow.json"
    payload = {
        "phase": 3,
        "method": "privileged_one_step_flow",
        "seed": seed,
        "closed_loop": metrics,
        "eval_mode": eval_mode,
        "bc_reference_success": bc_success,
        "held_out_action_metrics": checkpoint["validation_action_metrics"],
        "sampling_diagnostics": checkpoint["sampling_diagnostics"],
        "overfit_diagnostics": checkpoint["overfit_diagnostics"],
        "data": checkpoint["data"],
        "metadata": _runtime_metadata(config),
        "gate_passed": metrics["success"] >= bc_success - 0.05,
    }
    write_json(output_path, payload)
    console.print(payload)
    return output_path


def _phase4_dino_from_config(config: Config, device: torch.device) -> DinoExtractor:
    return DinoExtractor(
        str(config.get("dino.model_name")),
        device,
        feature_type=str(config.get("dino.feature_type", "spatial")),
        spatial_pool=int(config.get("dino.spatial_pool", 4)),
    )


def _phase4_prepared_path(config: Config) -> Path:
    return Path(config.get("incremental.phase4.prepared_path"))


def _load_phase4_episodes(
    config: Config,
) -> tuple[list[dict[str, np.ndarray]], list[dict[str, np.ndarray]], dict[str, Any]]:
    path = _phase4_prepared_path(config)
    if not path.exists():
        raise FileNotFoundError(f"Missing Phase 4 prepared visual dataset: {path}")
    train_episodes = int(config.get("incremental.phase4.train_episodes", 1800))
    validation_episodes = int(config.get("incremental.phase4.validation_episodes", 200))
    with h5py.File(path, "r") as h5:
        keys = sorted(k for k in h5 if k.startswith("episode_"))
        required = train_episodes + validation_episodes
        if len(keys) < required:
            raise ValueError(f"{path} has {len(keys)} episodes, requires {required}")
        train_keys = keys[:train_episodes]
        val_keys = keys[-validation_episodes:]
        action_low = np.asarray(config.get("policy.action_low"), dtype=np.float32)
        action_high = np.asarray(config.get("policy.action_high"), dtype=np.float32)

        def read(keys_in: list[str]) -> list[dict[str, np.ndarray]]:
            episodes = []
            for key in keys_in:
                group = h5[key]
                features = np.asarray(group["dino"], dtype=np.float32)
                proprio = np.asarray(group["proprio"], dtype=np.float32)
                actions = np.asarray(group["actions"], dtype=np.float32)
                actions = np.clip(actions, action_low, action_high).astype(np.float32)
                frames = np.concatenate([features, proprio], axis=-1).astype(np.float32)
                episodes.append({"frames": frames, "actions": actions})
            return episodes

        metadata = {
            "dataset_type": "causal_dataset",
            "path": str(path),
            "source": str(h5["meta"].attrs.get("source", "unknown")) if "meta" in h5 else "unknown",
            "dino_model": (
                str(h5["meta"].attrs.get("dino_model", config.get("dino.model_name")))
                if "meta" in h5
                else str(config.get("dino.model_name"))
            ),
            "dino_feature_type": (
                str(h5["meta"].attrs.get("dino_feature_type", config.get("dino.feature_type")))
                if "meta" in h5
                else str(config.get("dino.feature_type"))
            ),
            "dino_spatial_pool": (
                int(h5["meta"].attrs.get("dino_spatial_pool", config.get("dino.spatial_pool", 4)))
                if "meta" in h5
                else int(config.get("dino.spatial_pool", 4))
            ),
            "train_episodes": train_episodes,
            "validation_episodes": validation_episodes,
        }
        train = read(train_keys)
        val = read(val_keys)
    metadata["train_queries"] = int(sum(len(ep["actions"]) for ep in train))
    metadata["validation_queries"] = int(sum(len(ep["actions"]) for ep in val))
    metadata["frame_dim"] = int(train[0]["frames"].shape[-1])
    metadata["action_dim"] = int(train[0]["actions"].shape[-1])
    return train, val, metadata


def _phase4_fit_standardizers(
    train_episodes: list[dict[str, np.ndarray]],
) -> tuple[Standardizer, Standardizer]:
    frames = np.concatenate([ep["frames"] for ep in train_episodes], axis=0)
    actions = np.concatenate([ep["actions"] for ep in train_episodes], axis=0)
    return Standardizer.fit(frames), Standardizer.fit(actions)


def _phase4_normalize_episodes(
    episodes: list[dict[str, np.ndarray]],
    frame_norm: Standardizer,
    action_norm: Standardizer,
) -> list[dict[str, np.ndarray]]:
    normalized = []
    for ep in episodes:
        normalized.append(
            {
                "frames": frame_norm.transform(ep["frames"]),
                "actions": action_norm.transform(ep["actions"]),
                "raw_actions": ep["actions"],
            }
        )
    return normalized


def _phase4_history(
    frames: np.ndarray,
    actions: np.ndarray,
    t: int,
    history: int,
    zero_action_norm: np.ndarray,
) -> np.ndarray:
    rows = []
    for offset in range(history):
        frame_t = t - history + 1 + offset
        source_t = max(0, frame_t)
        prev_t = frame_t - 1
        prev_action = actions[prev_t] if prev_t >= 0 else zero_action_norm
        rows.append(np.concatenate([frames[source_t], prev_action], axis=0))
    return np.stack(rows, axis=0).astype(np.float32)


class _Phase4HistoryDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        episodes: list[dict[str, np.ndarray]],
        history: int,
        zero_action_norm: np.ndarray,
        length: int,
    ) -> None:
        self.episodes = [ep for ep in episodes if len(ep["actions"]) > 0]
        self.history = history
        self.zero_action_norm = zero_action_norm
        self.length = length
        if not self.episodes:
            raise ValueError("No usable Phase 4 episodes")

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, _index: int) -> tuple[torch.Tensor, torch.Tensor]:
        episode = self.episodes[np.random.randint(0, len(self.episodes))]
        t = int(np.random.randint(0, len(episode["actions"])))
        history = _phase4_history(
            episode["frames"],
            episode["actions"],
            t,
            self.history,
            self.zero_action_norm,
        )
        return torch.from_numpy(history), torch.from_numpy(episode["actions"][t])


def _make_phase4_policy(
    architecture: str,
    step_dim: int,
    history: int,
    action_dim: int,
    hidden_dim: int,
) -> nn.Module:
    if architecture == "concat":
        return _TemporalConcatPolicy(step_dim, history, action_dim, hidden_dim)
    if architecture == "gru":
        return _TemporalGruPolicy(step_dim, history, action_dim, hidden_dim)
    raise ValueError(f"Unknown Phase 4 architecture: {architecture}")


def _phase4_action_metrics(
    model: nn.Module,
    episodes: list[dict[str, np.ndarray]],
    history: int,
    action_norm: Standardizer,
    zero_action_norm: np.ndarray,
    max_queries: int,
) -> dict[str, Any]:
    device = next(model.parameters()).device
    rng = np.random.default_rng(20_000 + history)
    candidates = [(ep_i, t) for ep_i, ep in enumerate(episodes) for t in range(len(ep["actions"]))]
    if len(candidates) > max_queries:
        chosen = rng.choice(len(candidates), size=max_queries, replace=False)
        candidates = [candidates[int(index)] for index in chosen]
    predictions = []
    targets = []
    batch = []
    batch_targets = []
    batch_size = 2048
    with torch.inference_mode():
        for ep_i, t in candidates:
            episode = episodes[ep_i]
            batch.append(
                _phase4_history(
                    episode["frames"],
                    episode["actions"],
                    t,
                    history,
                    zero_action_norm,
                )
            )
            batch_targets.append(episode["raw_actions"][t])
            if len(batch) == batch_size:
                x = torch.from_numpy(np.stack(batch)).to(device).float()
                pred = model(x).detach().cpu().numpy()
                predictions.append(action_norm.inverse(pred))
                targets.append(np.stack(batch_targets))
                batch.clear()
                batch_targets.clear()
        if batch:
            x = torch.from_numpy(np.stack(batch)).to(device).float()
            pred = model(x).detach().cpu().numpy()
            predictions.append(action_norm.inverse(pred))
            targets.append(np.stack(batch_targets))
    return _action_regression_metrics(np.concatenate(predictions), np.concatenate(targets))


def train_phase4_visual_bc(
    config: Config,
    history: int,
    architecture: str | None = None,
    seed: int = 0,
    force: bool = False,
) -> Path:
    set_seed(seed)
    architecture = architecture or str(config.get("incremental.phase4.architecture", "concat"))
    artifact_dir = ensure_dir(
        config.path_value("paths.incremental_artifact_dir")
        / "phase4"
        / f"{architecture}_h{history}"
        / f"seed{seed}"
    )
    checkpoint_path = artifact_dir / "visual_bc.pt"
    if checkpoint_path.exists() and not force:
        console.print(f"Phase 4 visual BC exists: {checkpoint_path}")
        return checkpoint_path
    train_episodes, val_episodes, data_metadata = _load_phase4_episodes(config)
    frame_norm, action_norm = _phase4_fit_standardizers(train_episodes)
    train_norm = _phase4_normalize_episodes(train_episodes, frame_norm, action_norm)
    val_norm = _phase4_normalize_episodes(val_episodes, frame_norm, action_norm)
    zero_action_norm = action_norm.transform(np.zeros((1, data_metadata["action_dim"]), dtype=np.float32))[0]
    step_dim = data_metadata["frame_dim"] + data_metadata["action_dim"]
    hidden_dim = int(config.get("incremental.phase4.hidden_dim", 512))
    model = _make_phase4_policy(
        architecture,
        step_dim,
        history,
        data_metadata["action_dim"],
        hidden_dim,
    ).to(default_device())
    dataset = _Phase4HistoryDataset(
        train_norm,
        history,
        zero_action_norm,
        length=int(config.get("incremental.phase4.batch_size", 512))
        * int(config.get("incremental.phase4.batches_per_epoch", 500)),
    )
    loader = DataLoader(
        dataset,
        batch_size=int(config.get("incremental.phase4.batch_size", 512)),
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config.get("incremental.phase4.lr", 3e-4)),
    )
    epochs = int(config.get("incremental.phase4.epochs", 50))
    timer = Timer()
    best_state = None
    best_mae = float("inf")
    history_rows = []
    device = next(model.parameters()).device
    for epoch in trange(1, epochs + 1, desc=f"train phase4 {architecture} h={history}"):
        model.train()
        loss_sum = 0.0
        count = 0
        for x, y in loader:
            x = x.to(device, non_blocking=True).float()
            y = y.to(device, non_blocking=True).float()
            pred = model(x)
            loss = torch.mean((pred - y) ** 2)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach().cpu()) * len(x)
            count += len(x)
        model.eval()
        metrics = _phase4_action_metrics(
            model,
            val_norm,
            history,
            action_norm,
            zero_action_norm,
            int(config.get("incremental.phase4.validation_queries", 10000)),
        )
        row = {
            "epoch": epoch,
            "train_mse": loss_sum / count,
            "validation_action_mae": metrics["mae"],
            "validation_action_rmse": metrics["rmse"],
        }
        history_rows.append(row)
        if metrics["mae"] < best_mae:
            best_mae = metrics["mae"]
            best_state = copy.deepcopy(model.state_dict())
    if best_state is None:
        raise RuntimeError("Phase 4 visual BC training produced no checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    validation_metrics = _phase4_action_metrics(
        model,
        val_norm,
        history,
        action_norm,
        zero_action_norm,
        int(config.get("incremental.phase4.validation_queries", 10000)),
    )
    payload = {
        "model": model.state_dict(),
        "architecture": architecture,
        "history": history,
        "step_dim": step_dim,
        "frame_dim": data_metadata["frame_dim"],
        "action_dim": data_metadata["action_dim"],
        "hidden_dim": hidden_dim,
        "frame_norm": frame_norm.state_dict(),
        "action_norm": action_norm.state_dict(),
        "zero_action_norm": zero_action_norm,
        "validation_metrics": validation_metrics,
        "data": data_metadata,
        "history_rows": history_rows,
        "elapsed_s": timer.elapsed(),
        "metadata": _runtime_metadata(config),
    }
    torch.save(payload, checkpoint_path)
    write_json(
        artifact_dir / "visual_bc_metrics.json",
        {
            "architecture": architecture,
            "history": history,
            "validation_metrics": validation_metrics,
            "data": data_metadata,
            "elapsed_s": timer.elapsed(),
        },
    )
    console.print(f"Wrote Phase 4 visual BC: {checkpoint_path}")
    return checkpoint_path


def _load_phase4_visual_bc(path: Path, device: torch.device) -> tuple[nn.Module, dict[str, Any]]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model = _make_phase4_policy(
        str(checkpoint["architecture"]),
        int(checkpoint["step_dim"]),
        int(checkpoint["history"]),
        int(checkpoint["action_dim"]),
        int(checkpoint["hidden_dim"]),
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, checkpoint


def _phase4_rgb_state(obs: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    rgb = _numpy(obs["sensor_data"]["base_camera"]["rgb"])
    state = _numpy(obs["state"])
    if rgb.shape[-1] == 4:
        rgb = rgb[..., :3]
    return rgb.astype(np.uint8), state.astype(np.float32)


@torch.inference_mode()
def _phase4_frame_inputs(
    obs: dict[str, Any],
    dino: DinoExtractor,
    batch_size: int,
) -> np.ndarray:
    rgb, state = _phase4_rgb_state(obs)
    features = [dino.encode_batch(chunk) for chunk in batched(rgb, batch_size)]
    dino_features = np.concatenate(features, axis=0)
    proprio = state[:, :21].astype(np.float32)
    return np.concatenate([dino_features, proprio], axis=-1).astype(np.float32)


def _phase4_make_visual_env(config: Config, num_envs: int):
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
        ignore_terminations=not bool(config.get("rl.eval_partial_reset", False)),
        record_metrics=True,
    )


def evaluate_phase4_visual_bc(
    config: Config,
    history: int,
    architecture: str | None = None,
    seed: int = 0,
    episodes: int | None = None,
) -> Path:
    architecture = architecture or str(config.get("incremental.phase4.architecture", "concat"))
    checkpoint_path = train_phase4_visual_bc(
        config,
        history=history,
        architecture=architecture,
        seed=seed,
        force=False,
    )
    device = default_device()
    model, checkpoint = _load_phase4_visual_bc(checkpoint_path, device)
    frame_norm = Standardizer.from_state_dict(checkpoint["frame_norm"])
    action_norm = Standardizer.from_state_dict(checkpoint["action_norm"])
    zero_action_norm = np.asarray(checkpoint["zero_action_norm"], dtype=np.float32)
    dino = _phase4_dino_from_config(config, device)
    eval_episodes = int(episodes or config.get("incremental.phase4.eval_episodes", 100))
    num_envs = min(int(config.get("incremental.phase4.eval_num_envs", 64)), eval_episodes)
    env = _phase4_make_visual_env(config, num_envs)
    action_low = torch.as_tensor(env.single_action_space.low, device=device, dtype=torch.float32)
    action_high = torch.as_tensor(env.single_action_space.high, device=device, dtype=torch.float32)
    obs, _info = env.reset(seed=int(config.get("incremental.phase4.eval_seed", 10000)))
    frames = frame_norm.transform(
        _phase4_frame_inputs(obs, dino, int(config.get("dino.batch_size", 64)))
    )
    frame_history = np.repeat(frames[:, None, :], history, axis=1).astype(np.float32)
    action_history = np.repeat(zero_action_norm[None, None, :], num_envs, axis=0)
    action_history = np.repeat(action_history, history, axis=1).astype(np.float32)
    successes: list[float] = []
    final_rewards: list[float] = []
    max_rewards: list[float] = []
    episode_lengths: list[int] = []
    latencies: list[float] = []
    active_max_reward = np.full(num_envs, -np.inf, dtype=np.float32)
    active_lengths = np.zeros(num_envs, dtype=np.int32)
    while len(successes) < eval_episodes:
        policy_input = np.concatenate([frame_history, action_history], axis=-1)
        timer = Timer()
        with torch.inference_mode():
            pred_norm = model(torch.from_numpy(policy_input).to(device).float())
            raw_action = action_norm.inverse(pred_norm.detach().cpu().numpy())
        latencies.append(timer.elapsed() / num_envs)
        action = torch.from_numpy(raw_action).to(device).float()
        if bool(config.get("policy.clip_actions_to_env_space", True)):
            action = torch.clamp(action, action_low, action_high)
        obs, reward, _terminated, _truncated, info = env.step(action)
        reward_np = _numpy(reward).reshape(-1).astype(np.float32)
        active_max_reward = np.maximum(active_max_reward, reward_np)
        active_lengths += 1
        executed_norm = action_norm.transform(action.detach().cpu().numpy().astype(np.float32))
        next_frames = frame_norm.transform(
            _phase4_frame_inputs(obs, dino, int(config.get("dino.batch_size", 64)))
        )
        frame_history = np.roll(frame_history, shift=-1, axis=1)
        frame_history[:, -1] = next_frames
        action_history = np.roll(action_history, shift=-1, axis=1)
        action_history[:, -1] = executed_norm
        if "final_info" in info:
            mask = _numpy(info["_final_info"]).reshape(-1).astype(bool)
            if mask.any():
                episode_info = info["final_info"]["episode"]
                success_once = _numpy(episode_info["success_once"]).reshape(-1)
                for env_idx in np.flatnonzero(mask):
                    successes.append(float(success_once[env_idx]))
                    final_rewards.append(float(reward_np[env_idx]))
                    max_rewards.append(float(active_max_reward[env_idx]))
                    episode_lengths.append(int(active_lengths[env_idx]))
                    frame_history[env_idx] = np.repeat(
                        next_frames[env_idx][None, :], history, axis=0
                    )
                    action_history[env_idx] = np.repeat(
                        zero_action_norm[None, :], history, axis=0
                    )
                    active_max_reward[env_idx] = -np.inf
                    active_lengths[env_idx] = 0
                    if len(successes) >= eval_episodes:
                        break
    env.close()
    metrics = {
        "success": float(np.mean(successes[:eval_episodes])),
        "success_stderr": float(
            np.std(successes[:eval_episodes]) / np.sqrt(eval_episodes)
        ),
        "final_reward": float(np.mean(final_rewards[:eval_episodes])),
        "max_reward": float(np.mean(max_rewards[:eval_episodes])),
        "mean_episode_length": float(np.mean(episode_lengths[:eval_episodes])),
        "inference_latency_s": float(np.mean(latencies)),
        "episodes": eval_episodes,
        "seed_start": int(config.get("incremental.phase4.eval_seed", 10000)),
        "num_envs": num_envs,
    }
    results_dir = ensure_dir(
        config.path_value("paths.incremental_results_dir")
        / "phase4"
        / f"{architecture}_h{history}"
        / f"seed{seed}"
    )
    output_path = results_dir / "visual_bc.json"
    payload = {
        "phase": 4,
        "method": "visual_deterministic_bc",
        "architecture": architecture,
        "history": history,
        "seed": seed,
        "closed_loop": metrics,
        "held_out_action_metrics": checkpoint["validation_metrics"],
        "data": checkpoint["data"],
        "metadata": _runtime_metadata(config),
        "gate_passed": metrics["success"] >= 0.50,
    }
    write_json(output_path, payload)
    console.print(payload)
    return output_path


def _phase4_obj_yaw(state: np.ndarray) -> np.ndarray:
    return (2.0 * np.arctan2(state[:, 30], state[:, 27])).astype(np.float32)


def _phase4_probe_labels(
    state: np.ndarray,
    prev_motion_state: np.ndarray,
    config: Config,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    obj_xy = state[:, 24:26].astype(np.float32)
    yaw = _phase4_obj_yaw(state)
    tcp_xy = state[:, 14:16].astype(np.float32)
    motion_state = np.concatenate([obj_xy, yaw[:, None], tcp_xy], axis=-1)
    dt_inv = float(config.get("control_freq", 20))
    obj_vel_xy = (motion_state[:, :2] - prev_motion_state[:, :2]) * dt_inv
    yaw_delta = np.arctan2(
        np.sin(motion_state[:, 2] - prev_motion_state[:, 2]),
        np.cos(motion_state[:, 2] - prev_motion_state[:, 2]),
    )
    yaw_vel = yaw_delta[:, None] * dt_inv
    tcp_vel_xy = (motion_state[:, 3:5] - prev_motion_state[:, 3:5]) * dt_inv
    contact_threshold = float(config.get("incremental.phase4.probe_contact_distance_m", 0.08))
    contact = (np.linalg.norm(tcp_xy - obj_xy, axis=-1) < contact_threshold).astype(np.float32)
    labels = np.concatenate(
        [obj_xy, yaw[:, None], obj_vel_xy, yaw_vel, tcp_xy, tcp_vel_xy],
        axis=-1,
    ).astype(np.float32)
    return labels, contact[:, None], motion_state.astype(np.float32)


def _binary_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    labels = labels.astype(bool)
    positives = int(labels.sum())
    negatives = int((~labels).sum())
    if positives == 0 or negatives == 0:
        return float("nan")
    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)
    return float((ranks[labels].sum() - positives * (positives + 1) / 2) / (positives * negatives))


def probe_phase4_visual_history(
    config: Config,
    history: int,
    architecture: str | None = None,
    seed: int = 0,
    samples: int | None = None,
    force: bool = False,
) -> Path:
    set_seed(seed)
    architecture = architecture or str(config.get("incremental.phase4.architecture", "concat"))
    samples = int(samples or config.get("incremental.phase4.probe_samples", 8000))
    results_dir = ensure_dir(
        config.path_value("paths.incremental_results_dir")
        / "phase4"
        / f"{architecture}_h{history}"
        / f"seed{seed}"
    )
    output_path = results_dir / "visual_history_probe.json"
    if output_path.exists() and not force:
        console.print(f"Phase 4 visual-history probe exists: {output_path}")
        return output_path
    checkpoint_path = train_phase4_visual_bc(
        config,
        history=history,
        architecture=architecture,
        seed=seed,
        force=False,
    )
    device = default_device()
    _model, checkpoint = _load_phase4_visual_bc(checkpoint_path, device)
    frame_norm = Standardizer.from_state_dict(checkpoint["frame_norm"])
    action_norm = Standardizer.from_state_dict(checkpoint["action_norm"])
    zero_action_norm = np.asarray(checkpoint["zero_action_norm"], dtype=np.float32)
    dino = _phase4_dino_from_config(config, device)
    teacher = load_ppo_agent(_rl_paths(config).best, device)
    num_envs = min(int(config.get("incremental.phase4.eval_num_envs", 64)), samples)
    env = _phase4_make_visual_env(config, num_envs)
    action_low = torch.as_tensor(env.single_action_space.low, device=device, dtype=torch.float32)
    action_high = torch.as_tensor(env.single_action_space.high, device=device, dtype=torch.float32)
    obs, _info = env.reset(seed=int(config.get("incremental.phase4.eval_seed", 10000)) + 700_000)
    frames = frame_norm.transform(
        _phase4_frame_inputs(obs, dino, int(config.get("dino.batch_size", 64)))
    )
    frame_history = np.repeat(frames[:, None, :], history, axis=1).astype(np.float32)
    action_history = np.repeat(zero_action_norm[None, None, :], num_envs, axis=0)
    action_history = np.repeat(action_history, history, axis=1).astype(np.float32)
    _rgb, state = _phase4_rgb_state(obs)
    prev_motion_state = np.concatenate(
        [state[:, 24:26], _phase4_obj_yaw(state)[:, None], state[:, 14:16]],
        axis=-1,
    )
    inputs = []
    continuous_labels = []
    contact_labels = []
    progress = trange(samples, desc=f"collect phase4 probe h={history}")
    while len(inputs) < samples:
        state_t = torch.from_numpy(state).to(device).float()
        raw_action = teacher.actor_mean(state_t)
        action = torch.clamp(raw_action, action_low, action_high)
        labels, contact, current_motion_state = _phase4_probe_labels(
            state, prev_motion_state, config
        )
        take = min(num_envs, samples - len(inputs))
        current_input = np.concatenate([frame_history, action_history], axis=-1)
        inputs.append(current_input[:take].reshape(take, -1).astype(np.float32))
        continuous_labels.append(labels[:take])
        contact_labels.append(contact[:take])
        progress.update(take)
        obs, _reward, _terminated, _truncated, info = env.step(action)
        executed_norm = action_norm.transform(action.detach().cpu().numpy().astype(np.float32))
        next_frames = frame_norm.transform(
            _phase4_frame_inputs(obs, dino, int(config.get("dino.batch_size", 64)))
        )
        frame_history = np.roll(frame_history, shift=-1, axis=1)
        frame_history[:, -1] = next_frames
        action_history = np.roll(action_history, shift=-1, axis=1)
        action_history[:, -1] = executed_norm
        _rgb, state = _phase4_rgb_state(obs)
        next_motion_state = np.concatenate(
            [state[:, 24:26], _phase4_obj_yaw(state)[:, None], state[:, 14:16]],
            axis=-1,
        )
        if "final_info" in info:
            mask = _numpy(info["_final_info"]).reshape(-1).astype(bool)
            for env_idx in np.flatnonzero(mask):
                frame_history[env_idx] = np.repeat(next_frames[env_idx][None, :], history, axis=0)
                action_history[env_idx] = np.repeat(zero_action_norm[None, :], history, axis=0)
        prev_motion_state = current_motion_state
        if "final_info" in info:
            mask = _numpy(info["_final_info"]).reshape(-1).astype(bool)
            prev_motion_state[mask] = next_motion_state[mask]
        if len(inputs) * num_envs >= samples + num_envs:
            break
    progress.close()
    env.close()
    x_all = np.concatenate(inputs, axis=0)[:samples]
    y_all = np.concatenate(continuous_labels, axis=0)[:samples]
    c_all = np.concatenate(contact_labels, axis=0)[:samples]
    rng = np.random.default_rng(seed)
    order = rng.permutation(samples)
    split = int(0.8 * samples)
    train_idx = order[:split]
    val_idx = order[split:]
    x_norm = Standardizer.fit(x_all[train_idx])
    y_norm = Standardizer.fit(y_all[train_idx])
    train_x = torch.from_numpy(x_norm.transform(x_all[train_idx])).float()
    train_y = torch.from_numpy(y_norm.transform(y_all[train_idx])).float()
    train_c = torch.from_numpy(c_all[train_idx]).float()
    dataset = TensorDataset(train_x, train_y, train_c)
    loader = DataLoader(
        dataset,
        batch_size=int(config.get("incremental.phase4.probe_batch_size", 512)),
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    hidden_dim = int(config.get("incremental.phase4.probe_hidden_dim", 512))
    trunk = nn.Sequential(
        nn.Linear(x_all.shape[-1], hidden_dim),
        nn.SiLU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.SiLU(),
    ).to(device)
    continuous_head = nn.Linear(hidden_dim, y_all.shape[-1]).to(device)
    contact_head = nn.Linear(hidden_dim, 1).to(device)
    optimizer = torch.optim.AdamW(
        list(trunk.parameters()) + list(continuous_head.parameters()) + list(contact_head.parameters()),
        lr=float(config.get("incremental.phase4.probe_lr", 1e-3)),
    )
    epochs = int(config.get("incremental.phase4.probe_epochs", 100))
    timer = Timer()
    for _epoch in trange(epochs, desc=f"train phase4 probe h={history}"):
        for x, y, c in loader:
            x = x.to(device, non_blocking=True).float()
            y = y.to(device, non_blocking=True).float()
            c = c.to(device, non_blocking=True).float()
            hidden = trunk(x)
            pred_y = continuous_head(hidden)
            pred_c = contact_head(hidden)
            loss = torch.mean((pred_y - y) ** 2) + torch.nn.functional.binary_cross_entropy_with_logits(
                pred_c, c
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    with torch.inference_mode():
        val_x = torch.from_numpy(x_norm.transform(x_all[val_idx])).to(device).float()
        hidden = trunk(val_x)
        pred_y = y_norm.inverse(continuous_head(hidden).detach().cpu().numpy())
        contact_logits = contact_head(hidden).detach().cpu().numpy()[:, 0]
    target_y = y_all[val_idx]
    target_c = c_all[val_idx, 0]
    mean_y = np.broadcast_to(y_all[train_idx].mean(axis=0, keepdims=True), target_y.shape)
    contact_prior = float(c_all[train_idx].mean())
    contact_pred = (1.0 / (1.0 + np.exp(-contact_logits))) >= 0.5
    baseline_contact = np.full_like(target_c.astype(bool), contact_prior >= 0.5)
    names = [
        "obj_x_m",
        "obj_y_m",
        "obj_yaw_rad",
        "obj_vx_mps",
        "obj_vy_mps",
        "obj_yaw_rate_rps",
        "tcp_x_m",
        "tcp_y_m",
        "tcp_vx_mps",
        "tcp_vy_mps",
    ]
    mae = np.mean(np.abs(pred_y - target_y), axis=0)
    baseline_mae = np.mean(np.abs(mean_y - target_y), axis=0)
    metrics = {
        "phase": 4,
        "method": "visual_history_probe",
        "architecture": architecture,
        "history": history,
        "samples": samples,
        "train_samples": int(len(train_idx)),
        "validation_samples": int(len(val_idx)),
        "continuous_mae": {name: float(value) for name, value in zip(names, mae, strict=True)},
        "mean_baseline_mae": {
            name: float(value) for name, value in zip(names, baseline_mae, strict=True)
        },
        "contact": {
            "positive_fraction_train": contact_prior,
            "positive_fraction_val": float(target_c.mean()),
            "accuracy": float(np.mean(contact_pred == target_c.astype(bool))),
            "majority_baseline_accuracy": float(np.mean(baseline_contact == target_c.astype(bool))),
            "auroc": _binary_auc(contact_logits, target_c),
        },
        "elapsed_s": timer.elapsed(),
        "metadata": _runtime_metadata(config),
    }
    metrics["gate_support"] = {
        "pose_better_than_mean": bool(mae[0] < baseline_mae[0] and mae[1] < baseline_mae[1]),
        "velocity_better_than_mean": bool(
            mae[3] < baseline_mae[3] and mae[4] < baseline_mae[4]
        ),
        "contact_better_than_majority": bool(
            metrics["contact"]["accuracy"] > metrics["contact"]["majority_baseline_accuracy"]
        ),
    }
    write_json(output_path, metrics)
    console.print(metrics)
    return output_path


def _phase5_flow_action_metrics(
    model: FlowModel,
    episodes: list[dict[str, np.ndarray]],
    history: int,
    action_norm: Standardizer,
    zero_action_norm: np.ndarray,
    flow_steps: int,
    max_queries: int,
) -> dict[str, Any]:
    device = next(model.parameters()).device
    rng = np.random.default_rng(50_000 + history)
    candidates = [(ep_i, t) for ep_i, ep in enumerate(episodes) for t in range(len(ep["actions"]))]
    if len(candidates) > max_queries:
        chosen = rng.choice(len(candidates), size=max_queries, replace=False)
        candidates = [candidates[int(index)] for index in chosen]
    predictions = []
    targets = []
    batch = []
    batch_targets = []
    batch_size = 2048
    with torch.inference_mode():
        for ep_i, t in candidates:
            episode = episodes[ep_i]
            batch.append(
                _phase4_history(
                    episode["frames"],
                    episode["actions"],
                    t,
                    history,
                    zero_action_norm,
                ).reshape(-1)
            )
            batch_targets.append(episode["raw_actions"][t])
            if len(batch) == batch_size:
                cond = torch.from_numpy(np.stack(batch)).to(device).float()
                zero = torch.zeros(cond.shape[0], model.sample_dim, device=device, dtype=cond.dtype)
                pred_norm = sample_flow(
                    model,
                    cond,
                    flow_steps,
                    model.sample_dim,
                    initial_noise=zero,
                )
                predictions.append(action_norm.inverse(pred_norm.detach().cpu().numpy()))
                targets.append(np.stack(batch_targets))
                batch.clear()
                batch_targets.clear()
        if batch:
            cond = torch.from_numpy(np.stack(batch)).to(device).float()
            zero = torch.zeros(cond.shape[0], model.sample_dim, device=device, dtype=cond.dtype)
            pred_norm = sample_flow(
                model,
                cond,
                flow_steps,
                model.sample_dim,
                initial_noise=zero,
            )
            predictions.append(action_norm.inverse(pred_norm.detach().cpu().numpy()))
            targets.append(np.stack(batch_targets))
    metrics = _action_regression_metrics(np.concatenate(predictions), np.concatenate(targets))
    metrics["mode"] = "zero_noise"
    return metrics


def train_phase5_visual_flow(
    config: Config,
    history: int | None = None,
    architecture: str | None = None,
    seed: int = 0,
    force: bool = False,
) -> Path:
    set_seed(seed)
    history = int(history or config.get("incremental.phase5.history", 1))
    architecture = architecture or str(config.get("incremental.phase5.architecture", "concat"))
    artifact_dir = ensure_dir(
        config.path_value("paths.incremental_artifact_dir")
        / "phase5"
        / f"{architecture}_h{history}"
        / f"seed{seed}"
    )
    checkpoint_path = artifact_dir / "visual_flow.pt"
    if checkpoint_path.exists() and not force:
        console.print(f"Phase 5 visual flow exists: {checkpoint_path}")
        return checkpoint_path
    train_episodes, val_episodes, data_metadata = _load_phase4_episodes(config)
    bc_path = train_phase4_visual_bc(
        config,
        history=history,
        architecture=architecture,
        seed=seed,
        force=False,
    )
    _bc_model, bc_checkpoint = _load_phase4_visual_bc(bc_path, default_device())
    frame_norm = Standardizer.from_state_dict(bc_checkpoint["frame_norm"])
    action_norm = Standardizer.from_state_dict(bc_checkpoint["action_norm"])
    zero_action_norm = np.asarray(bc_checkpoint["zero_action_norm"], dtype=np.float32)
    train_norm = _phase4_normalize_episodes(train_episodes, frame_norm, action_norm)
    val_norm = _phase4_normalize_episodes(val_episodes, frame_norm, action_norm)
    dataset = _Phase4HistoryDataset(
        train_norm,
        history,
        zero_action_norm,
        length=int(config.get("incremental.phase5.batch_size", 512))
        * int(config.get("incremental.phase5.batches_per_epoch", 500)),
    )
    loader = DataLoader(
        dataset,
        batch_size=int(config.get("incremental.phase5.batch_size", 512)),
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    cond_dim = int((data_metadata["frame_dim"] + data_metadata["action_dim"]) * history)
    action_dim = int(data_metadata["action_dim"])
    model = FlowModel(
        sample_dim=action_dim,
        cond_dim=cond_dim,
        hidden_dim=int(config.get("incremental.phase5.hidden_dim", 512)),
    ).to(default_device())
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config.get("incremental.phase5.lr", 3e-4)),
    )
    epochs = int(config.get("incremental.phase5.epochs", 80))
    flow_steps = int(config.get("incremental.phase5.flow_steps", 24))
    best_state = None
    best_mae = float("inf")
    history_rows = []
    timer = Timer()
    device = next(model.parameters()).device
    for epoch in trange(1, epochs + 1, desc=f"train phase5 flow {architecture} h={history}"):
        model.train()
        loss_sum = 0.0
        count = 0
        for x, y in loader:
            cond = x.flatten(start_dim=1).to(device, non_blocking=True).float()
            target = y.to(device, non_blocking=True).float()
            loss = flow_matching_loss(model, target, cond)
            consistency_weight = float(
                config.get("incremental.phase5.endpoint_consistency_weight", 0.0)
            )
            if consistency_weight > 0.0:
                consistency_count = min(
                    int(config.get("incremental.phase5.endpoint_consistency_batch", 256)),
                    len(cond),
                )
                consistency_steps = int(
                    config.get("incremental.phase5.endpoint_consistency_steps", 4)
                )
                zero = torch.zeros(
                    consistency_count,
                    model.sample_dim,
                    device=device,
                    dtype=cond.dtype,
                )
                endpoint = _integrate_flow_train(
                    model,
                    cond[:consistency_count],
                    consistency_steps,
                    model.sample_dim,
                    zero,
                )
                loss = loss + consistency_weight * torch.mean(
                    (endpoint - target[:consistency_count]) ** 2
                )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach().cpu()) * len(cond)
            count += len(cond)
        row = {"epoch": epoch, "train_loss": loss_sum / count}
        if epoch % int(config.get("incremental.phase5.validation_action_interval", 5)) == 0:
            model.eval()
            metrics = _phase5_flow_action_metrics(
                model,
                val_norm,
                history,
                action_norm,
                zero_action_norm,
                flow_steps,
                int(config.get("incremental.phase5.validation_queries", 10000)),
            )
            row["validation_action_mae"] = metrics["mae"]
            row["validation_action_rmse"] = metrics["rmse"]
            if metrics["mae"] < best_mae:
                best_mae = metrics["mae"]
                best_state = copy.deepcopy(model.state_dict())
        history_rows.append(row)
    if best_state is None:
        raise RuntimeError("Phase 5 visual flow training produced no checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    validation_metrics = _phase5_flow_action_metrics(
        model,
        val_norm,
        history,
        action_norm,
        zero_action_norm,
        flow_steps,
        int(config.get("incremental.phase5.validation_queries", 10000)),
    )
    payload = {
        "model": model.state_dict(),
        "architecture": architecture,
        "history": history,
        "cond_dim": cond_dim,
        "sample_dim": action_dim,
        "hidden_dim": int(config.get("incremental.phase5.hidden_dim", 512)),
        "flow_steps": flow_steps,
        "frame_norm": frame_norm.state_dict(),
        "action_norm": action_norm.state_dict(),
        "zero_action_norm": zero_action_norm,
        "validation_metrics": validation_metrics,
        "data": data_metadata,
        "history_rows": history_rows,
        "bc_reference_checkpoint": str(bc_path),
        "elapsed_s": timer.elapsed(),
        "metadata": _runtime_metadata(config),
    }
    torch.save(payload, checkpoint_path)
    write_json(
        artifact_dir / "visual_flow_metrics.json",
        {
            "architecture": architecture,
            "history": history,
            "validation_metrics": validation_metrics,
            "data": data_metadata,
            "elapsed_s": timer.elapsed(),
            "bc_reference_checkpoint": str(bc_path),
        },
    )
    console.print(f"Wrote Phase 5 visual flow: {checkpoint_path}")
    return checkpoint_path


def _load_phase5_visual_flow(path: Path, device: torch.device) -> tuple[FlowModel, dict[str, Any]]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model = FlowModel(
        sample_dim=int(checkpoint["sample_dim"]),
        cond_dim=int(checkpoint["cond_dim"]),
        hidden_dim=int(checkpoint["hidden_dim"]),
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, checkpoint


def evaluate_phase5_visual_flow(
    config: Config,
    history: int | None = None,
    architecture: str | None = None,
    seed: int = 0,
    episodes: int | None = None,
) -> Path:
    history = int(history or config.get("incremental.phase5.history", 1))
    architecture = architecture or str(config.get("incremental.phase5.architecture", "concat"))
    checkpoint_path = train_phase5_visual_flow(
        config,
        history=history,
        architecture=architecture,
        seed=seed,
        force=False,
    )
    device = default_device()
    model, checkpoint = _load_phase5_visual_flow(checkpoint_path, device)
    frame_norm = Standardizer.from_state_dict(checkpoint["frame_norm"])
    action_norm = Standardizer.from_state_dict(checkpoint["action_norm"])
    zero_action_norm = np.asarray(checkpoint["zero_action_norm"], dtype=np.float32)
    dino = _phase4_dino_from_config(config, device)
    flow_steps = int(checkpoint["flow_steps"])
    eval_episodes = int(episodes or config.get("incremental.phase5.eval_episodes", 100))
    num_envs = min(int(config.get("incremental.phase5.eval_num_envs", 64)), eval_episodes)
    env = _phase4_make_visual_env(config, num_envs)
    action_low = torch.as_tensor(env.single_action_space.low, device=device, dtype=torch.float32)
    action_high = torch.as_tensor(env.single_action_space.high, device=device, dtype=torch.float32)
    obs, _info = env.reset(seed=int(config.get("incremental.phase5.eval_seed", 10000)))
    frames = frame_norm.transform(
        _phase4_frame_inputs(obs, dino, int(config.get("dino.batch_size", 64)))
    )
    frame_history = np.repeat(frames[:, None, :], history, axis=1).astype(np.float32)
    action_history = np.repeat(zero_action_norm[None, None, :], num_envs, axis=0)
    action_history = np.repeat(action_history, history, axis=1).astype(np.float32)
    successes: list[float] = []
    final_rewards: list[float] = []
    max_rewards: list[float] = []
    episode_lengths: list[int] = []
    latencies: list[float] = []
    active_max_reward = np.full(num_envs, -np.inf, dtype=np.float32)
    active_lengths = np.zeros(num_envs, dtype=np.int32)
    while len(successes) < eval_episodes:
        policy_input = np.concatenate([frame_history, action_history], axis=-1).reshape(num_envs, -1)
        timer = Timer()
        with torch.inference_mode():
            cond = torch.from_numpy(policy_input).to(device).float()
            zero = torch.zeros(cond.shape[0], model.sample_dim, device=device, dtype=cond.dtype)
            pred_norm = sample_flow(
                model,
                cond,
                flow_steps,
                model.sample_dim,
                initial_noise=zero,
            )
            raw_action = action_norm.inverse(pred_norm.detach().cpu().numpy())
        latencies.append(timer.elapsed() / num_envs)
        action = torch.from_numpy(raw_action).to(device).float()
        if bool(config.get("policy.clip_actions_to_env_space", True)):
            action = torch.clamp(action, action_low, action_high)
        obs, reward, _terminated, _truncated, info = env.step(action)
        reward_np = _numpy(reward).reshape(-1).astype(np.float32)
        active_max_reward = np.maximum(active_max_reward, reward_np)
        active_lengths += 1
        executed_norm = action_norm.transform(action.detach().cpu().numpy().astype(np.float32))
        next_frames = frame_norm.transform(
            _phase4_frame_inputs(obs, dino, int(config.get("dino.batch_size", 64)))
        )
        frame_history = np.roll(frame_history, shift=-1, axis=1)
        frame_history[:, -1] = next_frames
        action_history = np.roll(action_history, shift=-1, axis=1)
        action_history[:, -1] = executed_norm
        if "final_info" in info:
            mask = _numpy(info["_final_info"]).reshape(-1).astype(bool)
            if mask.any():
                episode_info = info["final_info"]["episode"]
                success_once = _numpy(episode_info["success_once"]).reshape(-1)
                for env_idx in np.flatnonzero(mask):
                    successes.append(float(success_once[env_idx]))
                    final_rewards.append(float(reward_np[env_idx]))
                    max_rewards.append(float(active_max_reward[env_idx]))
                    episode_lengths.append(int(active_lengths[env_idx]))
                    frame_history[env_idx] = np.repeat(
                        next_frames[env_idx][None, :], history, axis=0
                    )
                    action_history[env_idx] = np.repeat(
                        zero_action_norm[None, :], history, axis=0
                    )
                    active_max_reward[env_idx] = -np.inf
                    active_lengths[env_idx] = 0
                    if len(successes) >= eval_episodes:
                        break
    env.close()
    metrics = {
        "success": float(np.mean(successes[:eval_episodes])),
        "success_stderr": float(
            np.std(successes[:eval_episodes]) / np.sqrt(eval_episodes)
        ),
        "final_reward": float(np.mean(final_rewards[:eval_episodes])),
        "max_reward": float(np.mean(max_rewards[:eval_episodes])),
        "mean_episode_length": float(np.mean(episode_lengths[:eval_episodes])),
        "inference_latency_s": float(np.mean(latencies)),
        "episodes": eval_episodes,
        "seed_start": int(config.get("incremental.phase5.eval_seed", 10000)),
        "num_envs": num_envs,
    }
    bc_result_path = (
        config.path_value("paths.incremental_results_dir")
        / "phase4"
        / f"{architecture}_h{history}"
        / f"seed{seed}"
        / "visual_bc.json"
    )
    if not bc_result_path.exists():
        evaluate_phase4_visual_bc(
            config,
            history=history,
            architecture=architecture,
            seed=seed,
            episodes=eval_episodes,
        )
    import json

    with bc_result_path.open("r", encoding="utf-8") as f:
        bc_result = json.load(f)
    bc_success = float(bc_result["closed_loop"]["success"])
    results_dir = ensure_dir(
        config.path_value("paths.incremental_results_dir")
        / "phase5"
        / f"{architecture}_h{history}"
        / f"seed{seed}"
    )
    output_path = results_dir / "visual_flow.json"
    payload = {
        "phase": 5,
        "method": "visual_one_step_flow",
        "architecture": architecture,
        "history": history,
        "seed": seed,
        "closed_loop": metrics,
        "eval_mode": "zero_noise",
        "bc_reference_success": bc_success,
        "held_out_action_metrics": checkpoint["validation_metrics"],
        "data": checkpoint["data"],
        "metadata": _runtime_metadata(config),
        "gate_passed": metrics["success"] >= bc_success - 0.05,
    }
    write_json(output_path, payload)
    console.print(payload)
    return output_path


class _Phase6RepresentationDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        episodes: list[dict[str, np.ndarray]],
        horizons: list[int],
        max_horizon: int,
        length: int,
    ) -> None:
        self.episodes = [ep for ep in episodes if len(ep["actions"]) > max_horizon]
        self.horizons = horizons
        self.max_horizon = max_horizon
        self.length = length
        if not self.episodes:
            raise ValueError(f"No Phase 6 episodes longer than max horizon {max_horizon}")

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, _index: int) -> dict[str, torch.Tensor]:
        episode = self.episodes[np.random.randint(0, len(self.episodes))]
        valid_horizons = [h for h in self.horizons if len(episode["actions"]) > h]
        horizon = int(valid_horizons[np.random.randint(0, len(valid_horizons))])
        t = int(np.random.randint(0, len(episode["actions"]) - horizon))
        action_seq = np.zeros(
            (self.max_horizon, episode["actions"].shape[-1]),
            dtype=np.float32,
        )
        action_seq[:horizon] = episode["actions"][t : t + horizon]
        return {
            "x_t": torch.from_numpy(episode["frames"][t]),
            "x_future": torch.from_numpy(episode["frames"][t + horizon]),
            "actions": torch.from_numpy(action_seq),
            "horizon": torch.tensor(horizon, dtype=torch.long),
        }


def _phase6_variant_weights(config: Config, variant: str) -> tuple[float, float, float]:
    prediction_weight = float(config.get("incremental.phase6.prediction_weight", 1.0))
    sigreg_weight = float(config.get("incremental.phase6.sigreg_weight", 0.05))
    reconstruction_weight = float(config.get("incremental.phase6.reconstruction_weight", 0.1))
    if variant == "wm_recon":
        return prediction_weight, sigreg_weight, reconstruction_weight
    if variant == "wm_norecon":
        return prediction_weight, sigreg_weight, 0.0
    if variant == "ae_recon":
        return 0.0, 0.0, reconstruction_weight
    raise ValueError(f"Unknown Phase 6 variant: {variant}")


def _load_phase6_train_episodes(
    config: Config,
) -> tuple[list[dict[str, np.ndarray]], list[dict[str, np.ndarray]], dict[str, Any]]:
    train, val, metadata = _load_phase4_episodes(config)
    train_episodes = int(config.get("incremental.phase6.train_episodes", 1800))
    validation_episodes = int(config.get("incremental.phase6.validation_episodes", 200))
    train = train[:train_episodes]
    val = val[:validation_episodes]
    metadata = {
        **metadata,
        "phase6_train_episodes": train_episodes,
        "phase6_validation_episodes": validation_episodes,
        "phase6_train_queries": int(sum(len(ep["actions"]) for ep in train)),
        "phase6_validation_queries": int(sum(len(ep["actions"]) for ep in val)),
    }
    return train, val, metadata


def _phase6_normalize_episodes(
    episodes: list[dict[str, np.ndarray]],
    frame_norm: Standardizer,
    action_norm: Standardizer,
) -> list[dict[str, np.ndarray]]:
    out = []
    for episode in episodes:
        out.append(
            {
                "frames": frame_norm.transform(episode["frames"]),
                "actions": action_norm.transform(episode["actions"]),
                "raw_frames": episode["frames"],
                "raw_actions": episode["actions"],
            }
        )
    return out


def train_phase6_representation(
    config: Config,
    latent_dim: int,
    variant: str | None = None,
    seed: int = 0,
    force: bool = False,
) -> Path:
    set_seed(seed)
    variant = variant or str(config.get("incremental.phase6.default_variant", "wm_recon"))
    artifact_dir = ensure_dir(
        config.path_value("paths.incremental_artifact_dir")
        / "phase6"
        / f"{variant}_z{latent_dim}"
        / f"seed{seed}"
    )
    checkpoint_path = artifact_dir / "encoder.pt"
    if checkpoint_path.exists() and not force:
        console.print(f"Phase 6 representation exists: {checkpoint_path}")
        return checkpoint_path

    train_episodes, val_episodes, data_metadata = _load_phase6_train_episodes(config)
    frame_norm, action_norm = _phase4_fit_standardizers(train_episodes)
    train_norm = _phase6_normalize_episodes(train_episodes, frame_norm, action_norm)
    val_norm = _phase6_normalize_episodes(val_episodes, frame_norm, action_norm)
    input_dim = int(data_metadata["frame_dim"])
    action_dim = int(data_metadata["action_dim"])
    hidden_dim = int(config.get("incremental.phase6.hidden_dim", 512))
    horizons = [int(h) for h in config.get("incremental.phase6.horizons_steps", [1, 2, 4, 8])]
    max_horizon = max(horizons)
    prediction_weight, sigreg_weight, reconstruction_weight = _phase6_variant_weights(
        config, variant
    )

    device = default_device()
    encoder = ObservationEncoder(input_dim, latent_dim, hidden_dim).to(device)
    world_model = RepresentationWorldModel(latent_dim, action_dim, hidden_dim).to(device)
    decoder = (
        MLP(latent_dim, input_dim, hidden_dim, depth=3).to(device)
        if reconstruction_weight > 0.0
        else None
    )
    params = list(encoder.parameters())
    if prediction_weight > 0.0:
        params += list(world_model.parameters())
    if decoder is not None:
        params += list(decoder.parameters())
    optimizer = torch.optim.AdamW(params, lr=float(config.get("incremental.phase6.lr", 3e-4)))
    dataset = _Phase6RepresentationDataset(
        train_norm,
        horizons,
        max_horizon,
        length=int(config.get("incremental.phase6.batch_size", 512))
        * int(config.get("incremental.phase6.batches_per_epoch", 400)),
    )
    loader = DataLoader(
        dataset,
        batch_size=int(config.get("incremental.phase6.batch_size", 512)),
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    epochs = int(config.get("incremental.phase6.epochs", 60))
    history = []
    timer = Timer()
    best_state = None
    best_val = float("inf")
    validation_samples = int(config.get("incremental.phase6.validation_samples", 8192))
    proprio_dim = int(config.get("incremental.phase6.proprio_dim", 21))
    proprio_reconstruction_weight = float(
        config.get("incremental.phase6.proprio_reconstruction_weight", 1.0)
    )

    def reconstruction_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if proprio_dim <= 0 or proprio_reconstruction_weight <= 0.0:
            return torch.mean((pred - target) ** 2)
        dino_loss = torch.mean((pred[:, :-proprio_dim] - target[:, :-proprio_dim]) ** 2)
        proprio_loss = torch.mean((pred[:, -proprio_dim:] - target[:, -proprio_dim:]) ** 2)
        return dino_loss + proprio_reconstruction_weight * proprio_loss

    def validation_metrics() -> dict[str, float]:
        probe_dataset = _Phase6RepresentationDataset(
            val_norm,
            horizons,
            max_horizon,
            length=validation_samples,
        )
        probe_loader = DataLoader(
            probe_dataset,
            batch_size=int(config.get("incremental.phase6.batch_size", 512)),
            shuffle=False,
            num_workers=0,
        )
        pred_losses = []
        recon_losses = []
        dino_recon_losses = []
        proprio_recon_losses = []
        with torch.inference_mode():
            for batch in probe_loader:
                x_t = batch["x_t"].to(device).float()
                x_future = batch["x_future"].to(device).float()
                actions = batch["actions"].to(device).float()
                horizon = batch["horizon"].to(device)
                z_t = encoder(x_t)
                z_future = encoder(x_future)
                if prediction_weight > 0.0:
                    pred = world_model(z_t, actions, horizon)
                    pred_losses.append(float(torch.mean((pred - z_future) ** 2).cpu()))
                if decoder is not None:
                    recon_t = decoder(z_t)
                    recon_future = decoder(z_future)
                    recon = 0.5 * (
                        reconstruction_loss(recon_t, x_t)
                        + reconstruction_loss(recon_future, x_future)
                    )
                    recon_losses.append(float(recon.cpu()))
                    dino_recon_losses.append(
                        float(
                            0.5
                            * (
                                torch.mean((recon_t[:, :-proprio_dim] - x_t[:, :-proprio_dim]) ** 2)
                                + torch.mean(
                                    (
                                        recon_future[:, :-proprio_dim]
                                        - x_future[:, :-proprio_dim]
                                    )
                                    ** 2
                                )
                            ).cpu()
                        )
                    )
                    proprio_recon_losses.append(
                        float(
                            0.5
                            * (
                                torch.mean((recon_t[:, -proprio_dim:] - x_t[:, -proprio_dim:]) ** 2)
                                + torch.mean(
                                    (
                                        recon_future[:, -proprio_dim:]
                                        - x_future[:, -proprio_dim:]
                                    )
                                    ** 2
                                )
                            ).cpu()
                        )
                    )
        return {
            "prediction_mse": float(np.mean(pred_losses)) if pred_losses else 0.0,
            "reconstruction_mse": float(np.mean(recon_losses)) if recon_losses else 0.0,
            "dino_reconstruction_mse": (
                float(np.mean(dino_recon_losses)) if dino_recon_losses else 0.0
            ),
            "proprio_reconstruction_mse": (
                float(np.mean(proprio_recon_losses)) if proprio_recon_losses else 0.0
            ),
        }

    for epoch in trange(1, epochs + 1, desc=f"train phase6 {variant} z={latent_dim}"):
        encoder.train()
        world_model.train()
        if decoder is not None:
            decoder.train()
        loss_sum = 0.0
        pred_sum = 0.0
        sig_sum = 0.0
        recon_sum = 0.0
        count = 0
        for batch in loader:
            x_t = batch["x_t"].to(device, non_blocking=True).float()
            x_future = batch["x_future"].to(device, non_blocking=True).float()
            actions = batch["actions"].to(device, non_blocking=True).float()
            horizon = batch["horizon"].to(device, non_blocking=True)
            z_t = encoder(x_t)
            z_future = encoder(x_future)
            pred_loss = torch.zeros((), device=device)
            if prediction_weight > 0.0:
                pred = world_model(z_t, actions, horizon)
                pred_loss = torch.mean((pred - z_future) ** 2)
            std = torch.sqrt(z_t.var(dim=0) + 1e-4)
            sigreg = torch.mean(torch.relu(1.0 - std))
            recon_loss = torch.zeros((), device=device)
            if decoder is not None:
                recon_t = decoder(z_t)
                recon_future = decoder(z_future)
                recon_loss = 0.5 * (
                    reconstruction_loss(recon_t, x_t)
                    + reconstruction_loss(recon_future, x_future)
                )
            loss = (
                prediction_weight * pred_loss
                + sigreg_weight * sigreg
                + reconstruction_weight * recon_loss
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            batch_count = len(x_t)
            loss_sum += float(loss.detach().cpu()) * batch_count
            pred_sum += float(pred_loss.detach().cpu()) * batch_count
            sig_sum += float(sigreg.detach().cpu()) * batch_count
            recon_sum += float(recon_loss.detach().cpu()) * batch_count
            count += batch_count
        encoder.eval()
        world_model.eval()
        if decoder is not None:
            decoder.eval()
        val = validation_metrics()
        selection = val["prediction_mse"] + val["reconstruction_mse"]
        row = {
            "epoch": epoch,
            "train_loss": loss_sum / count,
            "train_prediction_mse": pred_sum / count,
            "train_sigreg": sig_sum / count,
            "train_reconstruction_mse": recon_sum / count,
            **{f"validation_{key}": value for key, value in val.items()},
        }
        history.append(row)
        if selection < best_val:
            best_val = selection
            best_state = {
                "encoder": copy.deepcopy(encoder.state_dict()),
                "world_model": copy.deepcopy(world_model.state_dict()),
                "decoder": copy.deepcopy(decoder.state_dict()) if decoder is not None else None,
            }
    if best_state is None:
        raise RuntimeError("Phase 6 representation training produced no checkpoint")
    encoder.load_state_dict(best_state["encoder"])
    world_model.load_state_dict(best_state["world_model"])
    if decoder is not None and best_state["decoder"] is not None:
        decoder.load_state_dict(best_state["decoder"])
    final_val = validation_metrics()
    payload = {
        "encoder": encoder.state_dict(),
        "world_model": world_model.state_dict(),
        "decoder": decoder.state_dict() if decoder is not None else None,
        "variant": variant,
        "input_dim": input_dim,
        "action_dim": action_dim,
        "latent_dim": latent_dim,
        "hidden_dim": hidden_dim,
        "horizons_steps": horizons,
        "max_horizon": max_horizon,
        "frame_norm": frame_norm.state_dict(),
        "action_norm": action_norm.state_dict(),
        "prediction_weight": prediction_weight,
        "sigreg_weight": sigreg_weight,
        "reconstruction_weight": reconstruction_weight,
        "validation_metrics": final_val,
        "history": history,
        "data": data_metadata,
        "elapsed_s": timer.elapsed(),
        "metadata": _runtime_metadata(config),
    }
    torch.save(payload, checkpoint_path)
    write_json(
        artifact_dir / "encoder_metrics.json",
        {
            "variant": variant,
            "latent_dim": latent_dim,
            "validation_metrics": final_val,
            "elapsed_s": timer.elapsed(),
            "data": data_metadata,
        },
    )
    console.print(f"Wrote Phase 6 representation: {checkpoint_path}")
    return checkpoint_path


def _load_phase6_encoder(path: Path, device: torch.device) -> tuple[ObservationEncoder, dict[str, Any]]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    encoder = ObservationEncoder(
        int(checkpoint["input_dim"]),
        int(checkpoint["latent_dim"]),
        int(checkpoint["hidden_dim"]),
    ).to(device)
    encoder.load_state_dict(checkpoint["encoder"])
    encoder.eval()
    return encoder, checkpoint


def _phase6_probe_dataset_path(config: Config) -> Path:
    return config.path_value("paths.incremental_data_dir") / "phase6_probe_dataset.npz"


@torch.inference_mode()
def collect_phase6_probe_dataset(config: Config, force: bool = False) -> Path:
    output_path = _phase6_probe_dataset_path(config)
    required = int(config.get("incremental.phase6.probe_samples", 12000))
    if output_path.exists() and not force:
        with np.load(output_path) as data:
            if len(data["inputs"]) >= required:
                console.print(f"Phase 6 probe dataset exists: {output_path}")
                return output_path
    device = default_device()
    dino = _phase4_dino_from_config(config, device)
    teacher = load_ppo_agent(_rl_paths(config).best, device)
    num_envs = min(int(config.get("incremental.phase6.eval_num_envs", 64)), required)
    env = _phase4_make_visual_env(config, num_envs)
    action_low = torch.as_tensor(env.single_action_space.low, device=device, dtype=torch.float32)
    action_high = torch.as_tensor(env.single_action_space.high, device=device, dtype=torch.float32)
    obs, _info = env.reset(seed=int(config.get("incremental.phase6.probe_seed", 720000)))
    inputs = _phase4_frame_inputs(obs, dino, int(config.get("dino.batch_size", 64)))
    _rgb, state = _phase4_rgb_state(obs)
    prev_motion_state = np.concatenate(
        [state[:, 24:26], _phase4_obj_yaw(state)[:, None], state[:, 14:16]],
        axis=-1,
    )
    input_rows = []
    next_input_rows = []
    action_rows = []
    label_rows = []
    next_label_rows = []
    contact_rows = []
    next_contact_rows = []
    reward_rows = []
    progress = trange(required, desc="collect phase6 causal probe")
    collected = 0
    while collected < required:
        state_t = torch.from_numpy(state).to(device).float()
        raw_action = teacher.actor_mean(state_t)
        action = torch.clamp(raw_action, action_low, action_high)
        labels, contact, current_motion_state = _phase4_probe_labels(
            state,
            prev_motion_state,
            config,
        )
        next_obs, reward, _terminated, _truncated, info = env.step(action)
        next_inputs = _phase4_frame_inputs(next_obs, dino, int(config.get("dino.batch_size", 64)))
        _next_rgb, next_state = _phase4_rgb_state(next_obs)
        next_motion_state = np.concatenate(
            [next_state[:, 24:26], _phase4_obj_yaw(next_state)[:, None], next_state[:, 14:16]],
            axis=-1,
        )
        next_labels, next_contact, _ = _phase4_probe_labels(
            next_state,
            current_motion_state,
            config,
        )
        take = min(num_envs, required - collected)
        input_rows.append(inputs[:take])
        next_input_rows.append(next_inputs[:take])
        action_rows.append(action.detach().cpu().numpy().astype(np.float32)[:take])
        label_rows.append(labels[:take])
        next_label_rows.append(next_labels[:take])
        contact_rows.append(contact[:take])
        next_contact_rows.append(next_contact[:take])
        reward_rows.append(_numpy(reward).reshape(-1, 1).astype(np.float32)[:take])
        collected += take
        progress.update(take)
        inputs = next_inputs
        state = next_state
        prev_motion_state = current_motion_state
        if "final_info" in info:
            mask = _numpy(info["_final_info"]).reshape(-1).astype(bool)
            prev_motion_state[mask] = next_motion_state[mask]
    progress.close()
    env.close()
    ensure_dir(output_path.parent)
    np.savez_compressed(
        output_path,
        inputs=np.concatenate(input_rows, axis=0)[:required].astype(np.float32),
        next_inputs=np.concatenate(next_input_rows, axis=0)[:required].astype(np.float32),
        actions=np.concatenate(action_rows, axis=0)[:required].astype(np.float32),
        labels=np.concatenate(label_rows, axis=0)[:required].astype(np.float32),
        next_labels=np.concatenate(next_label_rows, axis=0)[:required].astype(np.float32),
        contact=np.concatenate(contact_rows, axis=0)[:required].astype(np.float32),
        next_contact=np.concatenate(next_contact_rows, axis=0)[:required].astype(np.float32),
        reward=np.concatenate(reward_rows, axis=0)[:required].astype(np.float32),
        dataset_type=np.asarray("causal_dataset"),
        semantics=np.asarray("teacher-executed visual/state/action/next-visual transitions"),
    )
    console.print(f"Wrote Phase 6 probe dataset: {output_path}")
    return output_path


def _phase6_representations(
    config: Config,
    inputs: np.ndarray,
    next_inputs: np.ndarray,
    representation: str,
    latent_dim: int | None,
    variant: str | None,
    seed: int,
    force: bool = False,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    if representation == "raw":
        train_episodes, _val_episodes, _metadata = _load_phase6_train_episodes(config)
        frame_norm, _action_norm = _phase4_fit_standardizers(train_episodes)
        return frame_norm.transform(inputs), frame_norm.transform(next_inputs), {
            "representation": "raw_spatial_dino_proprio",
            "dim": int(inputs.shape[-1]),
        }
    if representation != "latent":
        raise ValueError(f"Unknown Phase 6 representation: {representation}")
    if latent_dim is None:
        raise ValueError("Latent Phase 6 probe requires --latent-dim")
    path = train_phase6_representation(
        config,
        latent_dim=latent_dim,
        variant=variant,
        seed=seed,
        force=force,
    )
    device = default_device()
    encoder, checkpoint = _load_phase6_encoder(path, device)
    frame_norm = Standardizer.from_state_dict(checkpoint["frame_norm"])
    reps = []
    next_reps = []
    x = frame_norm.transform(inputs)
    x_next = frame_norm.transform(next_inputs)
    for start in range(0, len(x), 4096):
        with torch.inference_mode():
            reps.append(
                encoder(torch.from_numpy(x[start : start + 4096]).to(device).float()).cpu().numpy()
            )
            next_reps.append(
                encoder(torch.from_numpy(x_next[start : start + 4096]).to(device).float())
                .cpu()
                .numpy()
            )
    return np.concatenate(reps).astype(np.float32), np.concatenate(next_reps).astype(np.float32), {
        "representation": "latent",
        "variant": checkpoint["variant"],
        "latent_dim": int(checkpoint["latent_dim"]),
        "checkpoint": str(path),
    }


def _phase6_train_probe_heads(
    config: Config,
    reps: np.ndarray,
    next_reps: np.ndarray,
    actions: np.ndarray,
    labels: np.ndarray,
    next_labels: np.ndarray,
    contact: np.ndarray,
    reward: np.ndarray,
    seed: int,
) -> dict[str, Any]:
    set_seed(seed)
    device = default_device()
    rng = np.random.default_rng(seed)
    def encode_probe_labels(raw_labels: np.ndarray) -> np.ndarray:
        yaw = raw_labels[:, 2]
        return np.concatenate(
            [
                raw_labels[:, :2],
                np.sin(yaw)[:, None],
                np.cos(yaw)[:, None],
                raw_labels[:, 3:],
            ],
            axis=-1,
        ).astype(np.float32)

    probe_labels = encode_probe_labels(labels)
    probe_next_labels = encode_probe_labels(next_labels)
    order = rng.permutation(len(reps))
    split = int(0.8 * len(order))
    train_idx = order[:split]
    val_idx = order[split:]
    rep_norm = Standardizer.fit(reps[train_idx])
    action_norm = Standardizer.fit(actions[train_idx])
    label_norm = Standardizer.fit(probe_labels[train_idx])
    reward_norm = Standardizer.fit(reward[train_idx])
    x_train = rep_norm.transform(reps[train_idx])
    x_next_train = rep_norm.transform(next_reps[train_idx])
    action_train = action_norm.transform(actions[train_idx])
    label_train = label_norm.transform(probe_labels[train_idx])
    next_label_train = label_norm.transform(probe_next_labels[train_idx])
    reward_train = reward_norm.transform(reward[train_idx])
    contact_train = contact[train_idx]
    dataset = TensorDataset(
        torch.from_numpy(x_train),
        torch.from_numpy(x_next_train),
        torch.from_numpy(action_train),
        torch.from_numpy(label_train),
        torch.from_numpy(next_label_train),
        torch.from_numpy(contact_train),
        torch.from_numpy(reward_train),
    )
    loader = DataLoader(
        dataset,
        batch_size=int(config.get("incremental.phase6.probe_batch_size", 512)),
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    hidden_dim = int(config.get("incremental.phase6.probe_hidden_dim", 512))
    label_head = MLP(reps.shape[-1], probe_labels.shape[-1], hidden_dim, depth=3).to(device)
    contact_head = MLP(reps.shape[-1], 1, hidden_dim, depth=3).to(device)
    reward_head = MLP(reps.shape[-1], 1, hidden_dim, depth=3).to(device)
    inverse_head = MLP(2 * reps.shape[-1], actions.shape[-1], hidden_dim, depth=3).to(device)
    forward_head = MLP(
        reps.shape[-1] + actions.shape[-1],
        probe_labels.shape[-1],
        hidden_dim,
        depth=3,
    ).to(device)
    params = (
        list(label_head.parameters())
        + list(contact_head.parameters())
        + list(reward_head.parameters())
        + list(inverse_head.parameters())
        + list(forward_head.parameters())
    )
    optimizer = torch.optim.AdamW(params, lr=float(config.get("incremental.phase6.probe_lr", 1e-3)))
    epochs = int(config.get("incremental.phase6.probe_epochs", 120))
    timer = Timer()
    for _epoch in trange(epochs, desc="train phase6 probes"):
        for x, x_next, action, label, next_label, contact_y, reward_y in loader:
            x = x.to(device).float()
            x_next = x_next.to(device).float()
            action = action.to(device).float()
            label = label.to(device).float()
            next_label = next_label.to(device).float()
            contact_y = contact_y.to(device).float()
            reward_y = reward_y.to(device).float()
            pred_label = label_head(x)
            pred_contact = contact_head(x)
            pred_reward = reward_head(x)
            pred_action = inverse_head(torch.cat([x, x_next], dim=-1))
            pred_next_label = forward_head(torch.cat([x, action], dim=-1))
            loss = (
                torch.mean((pred_label - label) ** 2)
                + torch.nn.functional.binary_cross_entropy_with_logits(pred_contact, contact_y)
                + torch.mean((pred_reward - reward_y) ** 2)
                + torch.mean((pred_action - action) ** 2)
                + torch.mean((pred_next_label - next_label) ** 2)
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

    x_val = rep_norm.transform(reps[val_idx])
    x_next_val = rep_norm.transform(next_reps[val_idx])
    action_val = action_norm.transform(actions[val_idx])
    with torch.inference_mode():
        x_val_t = torch.from_numpy(x_val).to(device).float()
        x_next_val_t = torch.from_numpy(x_next_val).to(device).float()
        action_val_t = torch.from_numpy(action_val).to(device).float()
        pred_label_encoded = label_norm.inverse(label_head(x_val_t).cpu().numpy())
        pred_contact_logit = contact_head(x_val_t).cpu().numpy()[:, 0]
        pred_reward = reward_norm.inverse(reward_head(x_val_t).cpu().numpy())
        pred_action = action_norm.inverse(
            inverse_head(torch.cat([x_val_t, x_next_val_t], dim=-1)).cpu().numpy()
        )
        pred_next_label_encoded = label_norm.inverse(
            forward_head(torch.cat([x_val_t, action_val_t], dim=-1)).cpu().numpy()
        )
    target_label = labels[val_idx]
    target_next_label = next_labels[val_idx]
    target_label_encoded = probe_labels[val_idx]
    target_next_label_encoded = probe_next_labels[val_idx]
    target_contact = contact[val_idx, 0]
    target_reward = reward[val_idx]
    target_action = actions[val_idx]
    label_mean_encoded = np.broadcast_to(
        probe_labels[train_idx].mean(axis=0, keepdims=True),
        target_label_encoded.shape,
    )
    next_label_mean = np.broadcast_to(
        probe_next_labels[train_idx].mean(axis=0, keepdims=True),
        target_next_label_encoded.shape,
    )
    reward_mean = np.broadcast_to(reward[train_idx].mean(axis=0, keepdims=True), target_reward.shape)
    action_mean = np.broadcast_to(actions[train_idx].mean(axis=0, keepdims=True), target_action.shape)
    names = [
        "obj_x_m",
        "obj_y_m",
        "obj_yaw_rad",
        "obj_vx_mps",
        "obj_vy_mps",
        "obj_yaw_rate_rps",
        "tcp_x_m",
        "tcp_y_m",
        "tcp_vx_mps",
        "tcp_vy_mps",
    ]
    def structured_errors(pred_encoded: np.ndarray, target_raw: np.ndarray) -> np.ndarray:
        pred_yaw = np.arctan2(pred_encoded[:, 2], pred_encoded[:, 3])
        yaw_err = np.abs(
            np.arctan2(
                np.sin(pred_yaw - target_raw[:, 2]),
                np.cos(pred_yaw - target_raw[:, 2]),
            )
        )
        return np.stack(
            [
                np.abs(pred_encoded[:, 0] - target_raw[:, 0]),
                np.abs(pred_encoded[:, 1] - target_raw[:, 1]),
                yaw_err,
                np.abs(pred_encoded[:, 4] - target_raw[:, 3]),
                np.abs(pred_encoded[:, 5] - target_raw[:, 4]),
                np.abs(pred_encoded[:, 6] - target_raw[:, 5]),
                np.abs(pred_encoded[:, 7] - target_raw[:, 6]),
                np.abs(pred_encoded[:, 8] - target_raw[:, 7]),
                np.abs(pred_encoded[:, 9] - target_raw[:, 8]),
                np.abs(pred_encoded[:, 10] - target_raw[:, 9]),
            ],
            axis=-1,
        )

    label_mae = structured_errors(pred_label_encoded, target_label).mean(axis=0)
    baseline_label_mae = structured_errors(label_mean_encoded, target_label).mean(axis=0)
    next_label_mae = structured_errors(pred_next_label_encoded, target_next_label).mean(axis=0)
    baseline_next_label_mae = structured_errors(next_label_mean, target_next_label).mean(axis=0)
    contact_pred = pred_contact_logit >= 0.0
    contact_prior = float(contact[train_idx].mean())
    contact_baseline = np.full_like(target_contact.astype(bool), contact_prior >= 0.5)
    inverse_metrics = _action_regression_metrics(pred_action, target_action)
    inverse_baseline = _action_regression_metrics(action_mean, target_action)
    reward_mae = float(np.mean(np.abs(pred_reward - target_reward)))
    reward_baseline_mae = float(np.mean(np.abs(reward_mean - target_reward)))
    return {
        "samples": int(len(reps)),
        "train_samples": int(len(train_idx)),
        "validation_samples": int(len(val_idx)),
        "representation_dim": int(reps.shape[-1]),
        "continuous_mae": {name: float(value) for name, value in zip(names, label_mae, strict=True)},
        "mean_baseline_mae": {
            name: float(value) for name, value in zip(names, baseline_label_mae, strict=True)
        },
        "forward_next_label_mae": {
            name: float(value) for name, value in zip(names, next_label_mae, strict=True)
        },
        "forward_mean_baseline_mae": {
            name: float(value) for name, value in zip(names, baseline_next_label_mae, strict=True)
        },
        "contact": {
            "positive_fraction_train": contact_prior,
            "positive_fraction_val": float(target_contact.mean()),
            "accuracy": float(np.mean(contact_pred == target_contact.astype(bool))),
            "majority_baseline_accuracy": float(
                np.mean(contact_baseline == target_contact.astype(bool))
            ),
            "auroc": _binary_auc(pred_contact_logit, target_contact),
        },
        "reward": {
            "mae": reward_mae,
            "mean_baseline_mae": reward_baseline_mae,
        },
        "inverse_dynamics": {
            "action_mae": inverse_metrics["mae"],
            "mean_baseline_action_mae": inverse_baseline["mae"],
            "action_rmse": inverse_metrics["rmse"],
            "mean_baseline_action_rmse": inverse_baseline["rmse"],
        },
        "gate_support": {
            "pose_under_1cm": bool(label_mae[0] <= 0.01 and label_mae[1] <= 0.01),
            "yaw_under_10deg": bool(np.degrees(label_mae[2]) <= 10.0),
            "velocity_better_than_mean": bool(
                label_mae[3] < baseline_label_mae[3]
                and label_mae[4] < baseline_label_mae[4]
                and label_mae[5] < baseline_label_mae[5]
            ),
            "contact_auroc_over_0_80": bool(_binary_auc(pred_contact_logit, target_contact) >= 0.80),
            "inverse_better_than_mean": bool(inverse_metrics["mae"] < inverse_baseline["mae"]),
            "reward_better_than_mean": bool(reward_mae < reward_baseline_mae),
            "forward_better_than_mean": bool(np.mean(next_label_mae) < np.mean(baseline_next_label_mae)),
        },
        "elapsed_s": timer.elapsed(),
    }


def probe_phase6_representation(
    config: Config,
    representation: str = "raw",
    latent_dim: int | None = None,
    variant: str | None = None,
    seed: int = 0,
    force: bool = False,
) -> Path:
    variant_tag = "raw" if representation == "raw" else f"{variant or config.get('incremental.phase6.default_variant', 'wm_recon')}_z{latent_dim}"
    results_dir = ensure_dir(
        config.path_value("paths.incremental_results_dir")
        / "phase6"
        / variant_tag
        / f"seed{seed}"
    )
    output_path = results_dir / "representation_probe.json"
    if output_path.exists() and not force:
        console.print(f"Phase 6 representation probe exists: {output_path}")
        return output_path
    dataset_path = collect_phase6_probe_dataset(config, force=False)
    with np.load(dataset_path) as data:
        inputs = np.asarray(data["inputs"], dtype=np.float32)
        next_inputs = np.asarray(data["next_inputs"], dtype=np.float32)
        actions = np.asarray(data["actions"], dtype=np.float32)
        labels = np.asarray(data["labels"], dtype=np.float32)
        next_labels = np.asarray(data["next_labels"], dtype=np.float32)
        contact = np.asarray(data["contact"], dtype=np.float32)
        reward = np.asarray(data["reward"], dtype=np.float32)
    reps, next_reps, rep_metadata = _phase6_representations(
        config,
        inputs,
        next_inputs,
        representation,
        latent_dim,
        variant,
        seed,
        force=force,
    )
    metrics = _phase6_train_probe_heads(
        config,
        reps,
        next_reps,
        actions,
        labels,
        next_labels,
        contact,
        reward,
        seed,
    )
    payload = {
        "phase": 6,
        "method": "representation_probe",
        "representation": rep_metadata,
        "probe_dataset": str(dataset_path),
        **metrics,
        "metadata": _runtime_metadata(config),
    }
    write_json(output_path, payload)
    console.print(payload)
    return output_path


def _phase6_encode_control_episodes(
    encoder: ObservationEncoder,
    frame_norm: Standardizer,
    action_norm: Standardizer,
    episodes: list[dict[str, np.ndarray]],
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    encoder.eval()
    conds = []
    actions = []
    with torch.inference_mode():
        for episode in episodes:
            x = frame_norm.transform(episode["frames"])
            chunks = []
            for start in range(0, len(x), 4096):
                chunks.append(
                    encoder(torch.from_numpy(x[start : start + 4096]).to(device).float())
                    .cpu()
                    .numpy()
                )
            latents = np.concatenate(chunks, axis=0)
            zero_action = np.zeros((1, episode["actions"].shape[-1]), dtype=np.float32)
            prev_action = np.concatenate([zero_action, episode["actions"][:-1]], axis=0)
            prev_action_norm = action_norm.transform(prev_action)
            conds.append(np.concatenate([latents, prev_action_norm], axis=-1))
            actions.append(episode["actions"])
    return np.concatenate(conds).astype(np.float32), np.concatenate(actions).astype(np.float32)


def _phase6_latent_flow_action_metrics(
    model: FlowModel,
    cond: np.ndarray,
    target_actions: np.ndarray,
    cond_norm: Standardizer,
    action_norm: Standardizer,
    flow_steps: int,
    max_queries: int,
) -> dict[str, Any]:
    device = next(model.parameters()).device
    rng = np.random.default_rng(60_000 + model.cond_dim)
    if len(cond) > max_queries:
        chosen = rng.choice(len(cond), size=max_queries, replace=False)
        cond = cond[chosen]
        target_actions = target_actions[chosen]
    cond = cond_norm.transform(cond)
    predictions = []
    batch_size = 2048
    with torch.inference_mode():
        for start in range(0, len(cond), batch_size):
            cond_t = torch.from_numpy(cond[start : start + batch_size]).to(device).float()
            zero = torch.zeros(cond_t.shape[0], model.sample_dim, device=device, dtype=cond_t.dtype)
            pred_norm = sample_flow(
                model,
                cond_t,
                flow_steps,
                model.sample_dim,
                initial_noise=zero,
            )
            predictions.append(action_norm.inverse(pred_norm.cpu().numpy()))
    metrics = _action_regression_metrics(np.concatenate(predictions), target_actions)
    metrics["mode"] = "zero_noise"
    return metrics


def train_phase6_latent_bc(
    config: Config,
    latent_dim: int,
    variant: str | None = None,
    seed: int = 0,
    force: bool = False,
) -> Path:
    set_seed(seed)
    variant = variant or str(config.get("incremental.phase6.default_variant", "wm_recon"))
    encoder_path = train_phase6_representation(
        config,
        latent_dim=latent_dim,
        variant=variant,
        seed=seed,
        force=False,
    )
    artifact_dir = ensure_dir(
        config.path_value("paths.incremental_artifact_dir")
        / "phase6"
        / f"{variant}_z{latent_dim}"
        / f"seed{seed}"
    )
    checkpoint_path = artifact_dir / "latent_bc.pt"
    if checkpoint_path.exists() and not force:
        console.print(f"Phase 6 latent BC exists: {checkpoint_path}")
        return checkpoint_path
    device = default_device()
    encoder, encoder_checkpoint = _load_phase6_encoder(encoder_path, device)
    encoder.eval()
    frame_norm = Standardizer.from_state_dict(encoder_checkpoint["frame_norm"])
    action_norm = Standardizer.from_state_dict(encoder_checkpoint["action_norm"])
    train_episodes, val_episodes, _data_metadata = _load_phase6_train_episodes(config)
    train_latents, train_actions = _phase6_encode_control_episodes(
        encoder,
        frame_norm,
        action_norm,
        train_episodes,
        device,
    )
    val_latents, val_actions = _phase6_encode_control_episodes(
        encoder,
        frame_norm,
        action_norm,
        val_episodes,
        device,
    )
    train_dataset = TensorDataset(
        torch.from_numpy(train_latents).float(),
        torch.from_numpy(action_norm.transform(train_actions)).float(),
    )
    loader = DataLoader(
        train_dataset,
        batch_size=int(config.get("incremental.phase6.control_batch_size", 512)),
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    cond_dim = train_latents.shape[-1]
    model = MLP(
        cond_dim,
        train_actions.shape[-1],
        int(config.get("incremental.phase6.hidden_dim", 512)),
        depth=4,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config.get("incremental.phase6.control_lr", 3e-4)),
    )
    epochs = int(config.get("incremental.phase6.control_epochs", 80))
    best_state = None
    best_val = float("inf")
    history = []
    timer = Timer()
    x_val = torch.from_numpy(val_latents).to(device).float()
    y_val = torch.from_numpy(action_norm.transform(val_actions)).to(device).float()
    for epoch in trange(1, epochs + 1, desc=f"train phase6 latent BC {variant} z={latent_dim}"):
        model.train()
        loss_sum = 0.0
        count = 0
        for x, y in loader:
            x = x.to(device, non_blocking=True).float()
            y = y.to(device, non_blocking=True).float()
            pred = model(x)
            loss = torch.mean((pred - y) ** 2)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach().cpu()) * len(x)
            count += len(x)
        model.eval()
        with torch.inference_mode():
            val_mse = float(torch.mean((model(x_val) - y_val) ** 2).cpu())
        history.append({"epoch": epoch, "train_mse": loss_sum / count, "validation_mse": val_mse})
        if val_mse < best_val:
            best_val = val_mse
            best_state = copy.deepcopy(model.state_dict())
    if best_state is None:
        raise RuntimeError("Phase 6 latent BC training produced no checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    with torch.inference_mode():
        pred_action = action_norm.inverse(model(x_val).cpu().numpy())
    validation_metrics = _action_regression_metrics(pred_action, val_actions)
    payload = {
        "model": model.state_dict(),
        "variant": variant,
        "latent_dim": latent_dim,
        "cond_dim": cond_dim,
        "hidden_dim": int(config.get("incremental.phase6.hidden_dim", 512)),
        "action_dim": train_actions.shape[-1],
        "encoder_checkpoint": str(encoder_path),
        "action_norm": action_norm.state_dict(),
        "validation_metrics": validation_metrics,
        "best_validation_mse": best_val,
        "history": history,
        "elapsed_s": timer.elapsed(),
        "metadata": _runtime_metadata(config),
    }
    torch.save(payload, checkpoint_path)
    write_json(
        artifact_dir / "latent_bc_metrics.json",
        {
            "variant": variant,
            "latent_dim": latent_dim,
            "validation_metrics": validation_metrics,
            "best_validation_mse": best_val,
            "elapsed_s": timer.elapsed(),
        },
    )
    console.print(f"Wrote Phase 6 latent BC: {checkpoint_path}")
    return checkpoint_path


def evaluate_phase6_latent_bc(
    config: Config,
    latent_dim: int,
    variant: str | None = None,
    seed: int = 0,
    episodes: int | None = None,
    force: bool = False,
) -> Path:
    variant = variant or str(config.get("incremental.phase6.default_variant", "wm_recon"))
    checkpoint_path = train_phase6_latent_bc(
        config,
        latent_dim=latent_dim,
        variant=variant,
        seed=seed,
        force=force,
    )
    device = default_device()
    bc_checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    encoder_path = Path(bc_checkpoint["encoder_checkpoint"])
    encoder, encoder_checkpoint = _load_phase6_encoder(encoder_path, device)
    model = MLP(
        int(bc_checkpoint["cond_dim"]),
        int(bc_checkpoint["action_dim"]),
        int(bc_checkpoint["hidden_dim"]),
        depth=4,
    ).to(device)
    model.load_state_dict(bc_checkpoint["model"])
    model.eval()
    frame_norm = Standardizer.from_state_dict(encoder_checkpoint["frame_norm"])
    action_norm = Standardizer.from_state_dict(bc_checkpoint["action_norm"])
    zero_action_norm = action_norm.transform(
        np.zeros((1, int(bc_checkpoint["action_dim"])), dtype=np.float32)
    )[0]
    dino = _phase4_dino_from_config(config, device)
    action_low = None
    action_high = None
    eval_episodes = int(episodes or config.get("incremental.phase6.eval_episodes", 100))
    num_envs = min(int(config.get("incremental.phase6.eval_num_envs", 64)), eval_episodes)
    env = _phase4_make_visual_env(config, num_envs)
    action_low = torch.as_tensor(env.single_action_space.low, device=device, dtype=torch.float32)
    action_high = torch.as_tensor(env.single_action_space.high, device=device, dtype=torch.float32)
    obs, _info = env.reset(seed=int(config.get("incremental.phase6.eval_seed", 10000)))
    prev_action_norm = np.repeat(zero_action_norm[None, :], num_envs, axis=0).astype(np.float32)
    successes: list[float] = []
    final_rewards: list[float] = []
    max_rewards: list[float] = []
    episode_lengths: list[int] = []
    latencies: list[float] = []
    active_max_reward = np.full(num_envs, -np.inf, dtype=np.float32)
    active_lengths = np.zeros(num_envs, dtype=np.int32)
    while len(successes) < eval_episodes:
        timer = Timer()
        frames = frame_norm.transform(
            _phase4_frame_inputs(obs, dino, int(config.get("dino.batch_size", 64)))
        )
        with torch.inference_mode():
            z = encoder(torch.from_numpy(frames).to(device).float())
            prev_action_t = torch.from_numpy(prev_action_norm).to(device).float()
            pred_norm = model(torch.cat([z, prev_action_t], dim=-1))
            raw_action = action_norm.inverse(
                pred_norm.cpu().numpy()
            )
        latencies.append(timer.elapsed() / num_envs)
        action = torch.from_numpy(raw_action).to(device).float()
        if bool(config.get("policy.clip_actions_to_env_space", True)):
            action = torch.clamp(action, action_low, action_high)
        prev_action_norm = action_norm.transform(action.detach().cpu().numpy().astype(np.float32))
        obs, reward, _terminated, _truncated, info = env.step(action)
        reward_np = _numpy(reward).reshape(-1).astype(np.float32)
        active_max_reward = np.maximum(active_max_reward, reward_np)
        active_lengths += 1
        if "final_info" in info:
            mask = _numpy(info["_final_info"]).reshape(-1).astype(bool)
            if mask.any():
                episode_info = info["final_info"]["episode"]
                success_once = _numpy(episode_info["success_once"]).reshape(-1)
                for env_idx in np.flatnonzero(mask):
                    successes.append(float(success_once[env_idx]))
                    final_rewards.append(float(reward_np[env_idx]))
                    max_rewards.append(float(active_max_reward[env_idx]))
                    episode_lengths.append(int(active_lengths[env_idx]))
                    active_max_reward[env_idx] = -np.inf
                    active_lengths[env_idx] = 0
                    prev_action_norm[env_idx] = zero_action_norm
                    if len(successes) >= eval_episodes:
                        break
    env.close()
    metrics = {
        "success": float(np.mean(successes[:eval_episodes])),
        "success_stderr": float(np.std(successes[:eval_episodes]) / np.sqrt(eval_episodes)),
        "final_reward": float(np.mean(final_rewards[:eval_episodes])),
        "max_reward": float(np.mean(max_rewards[:eval_episodes])),
        "mean_episode_length": float(np.mean(episode_lengths[:eval_episodes])),
        "inference_latency_s": float(np.mean(latencies)),
        "episodes": eval_episodes,
        "seed_start": int(config.get("incremental.phase6.eval_seed", 10000)),
        "num_envs": num_envs,
    }
    visual_flow_path = (
        config.path_value("paths.incremental_results_dir")
        / "phase5"
        / "concat_h1"
        / f"seed{seed}"
        / "visual_flow.json"
    )
    import json

    with visual_flow_path.open("r", encoding="utf-8") as f:
        visual_flow = json.load(f)
    visual_success = float(visual_flow["closed_loop"]["success"])
    results_dir = ensure_dir(
        config.path_value("paths.incremental_results_dir")
        / "phase6"
        / f"{variant}_z{latent_dim}"
        / f"seed{seed}"
    )
    output_path = results_dir / "latent_bc_control.json"
    payload = {
        "phase": 6,
        "method": "latent_deterministic_bc_control",
        "variant": variant,
        "latent_dim": latent_dim,
        "seed": seed,
        "closed_loop": metrics,
        "direct_visual_flow_success": visual_success,
        "control_gate_80pct": metrics["success"] >= 0.8 * visual_success,
        "control_gate_90pct": metrics["success"] >= 0.9 * visual_success,
        "held_out_action_metrics": bc_checkpoint["validation_metrics"],
        "metadata": _runtime_metadata(config),
    }
    write_json(output_path, payload)
    console.print(payload)
    return output_path


def train_phase6_latent_flow(
    config: Config,
    latent_dim: int,
    variant: str | None = None,
    seed: int = 0,
    force: bool = False,
) -> Path:
    set_seed(seed)
    variant = variant or str(config.get("incremental.phase6.default_variant", "wm_recon"))
    encoder_path = train_phase6_representation(
        config,
        latent_dim=latent_dim,
        variant=variant,
        seed=seed,
        force=False,
    )
    artifact_dir = ensure_dir(
        config.path_value("paths.incremental_artifact_dir")
        / "phase6"
        / f"{variant}_z{latent_dim}"
        / f"seed{seed}"
    )
    checkpoint_path = artifact_dir / "latent_flow.pt"
    if checkpoint_path.exists() and not force:
        console.print(f"Phase 6 latent flow exists: {checkpoint_path}")
        return checkpoint_path
    device = default_device()
    encoder, encoder_checkpoint = _load_phase6_encoder(encoder_path, device)
    frame_norm = Standardizer.from_state_dict(encoder_checkpoint["frame_norm"])
    action_norm = Standardizer.from_state_dict(encoder_checkpoint["action_norm"])
    train_episodes, val_episodes, _data_metadata = _load_phase6_train_episodes(config)
    train_cond_raw, train_actions = _phase6_encode_control_episodes(
        encoder,
        frame_norm,
        action_norm,
        train_episodes,
        device,
    )
    val_cond_raw, val_actions = _phase6_encode_control_episodes(
        encoder,
        frame_norm,
        action_norm,
        val_episodes,
        device,
    )
    cond_norm = Standardizer.fit(train_cond_raw)
    train_cond = cond_norm.transform(train_cond_raw)
    train_target = action_norm.transform(train_actions)
    train_dataset = TensorDataset(
        torch.from_numpy(train_cond).float(),
        torch.from_numpy(train_target).float(),
    )
    loader = DataLoader(
        train_dataset,
        batch_size=int(config.get("incremental.phase6.control_batch_size", 512)),
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    action_dim = train_actions.shape[-1]
    model = FlowModel(
        sample_dim=action_dim,
        cond_dim=train_cond.shape[-1],
        hidden_dim=int(config.get("incremental.phase6.control_flow_hidden_dim", 512)),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config.get("incremental.phase6.control_flow_lr", 3e-4)),
    )
    epochs = int(config.get("incremental.phase6.control_flow_epochs", 80))
    flow_steps = int(config.get("incremental.phase6.control_flow_steps", 24))
    validation_queries = int(config.get("incremental.phase6.validation_queries", 10000))
    validation_interval = int(config.get("incremental.phase6.control_flow_validation_interval", 5))
    endpoint_weight = float(config.get("incremental.phase6.control_flow_endpoint_weight", 20.0))
    endpoint_steps = int(config.get("incremental.phase6.control_flow_endpoint_steps", 4))
    endpoint_batch = int(config.get("incremental.phase6.control_flow_endpoint_batch", 256))
    best_state = None
    best_mae = float("inf")
    history = []
    timer = Timer()
    for epoch in trange(1, epochs + 1, desc=f"train phase6 latent flow {variant} z={latent_dim}"):
        model.train()
        loss_sum = 0.0
        count = 0
        for cond, target in loader:
            cond = cond.to(device, non_blocking=True).float()
            target = target.to(device, non_blocking=True).float()
            loss = flow_matching_loss(model, target, cond)
            if endpoint_weight > 0.0:
                consistency_count = min(endpoint_batch, len(cond))
                zero = torch.zeros(
                    consistency_count,
                    model.sample_dim,
                    device=device,
                    dtype=cond.dtype,
                )
                endpoint = _integrate_flow_train(
                    model,
                    cond[:consistency_count],
                    endpoint_steps,
                    model.sample_dim,
                    zero,
                )
                loss = loss + endpoint_weight * torch.mean(
                    (endpoint - target[:consistency_count]) ** 2
                )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach().cpu()) * len(cond)
            count += len(cond)
        row = {"epoch": epoch, "train_loss": loss_sum / count}
        if epoch % validation_interval == 0:
            model.eval()
            metrics = _phase6_latent_flow_action_metrics(
                model,
                val_cond_raw,
                val_actions,
                cond_norm,
                action_norm,
                flow_steps,
                validation_queries,
            )
            row["validation_action_mae"] = metrics["mae"]
            row["validation_action_rmse"] = metrics["rmse"]
            if metrics["mae"] < best_mae:
                best_mae = metrics["mae"]
                best_state = copy.deepcopy(model.state_dict())
        history.append(row)
    if best_state is None:
        raise RuntimeError("Phase 6 latent flow training produced no checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    validation_metrics = _phase6_latent_flow_action_metrics(
        model,
        val_cond_raw,
        val_actions,
        cond_norm,
        action_norm,
        flow_steps,
        validation_queries,
    )
    payload = {
        "model": model.state_dict(),
        "variant": variant,
        "latent_dim": latent_dim,
        "cond_dim": train_cond.shape[-1],
        "sample_dim": action_dim,
        "hidden_dim": int(config.get("incremental.phase6.control_flow_hidden_dim", 512)),
        "flow_steps": flow_steps,
        "encoder_checkpoint": str(encoder_path),
        "cond_norm": cond_norm.state_dict(),
        "action_norm": action_norm.state_dict(),
        "validation_metrics": validation_metrics,
        "best_validation_mae": best_mae,
        "history": history,
        "elapsed_s": timer.elapsed(),
        "metadata": _runtime_metadata(config),
    }
    torch.save(payload, checkpoint_path)
    write_json(
        artifact_dir / "latent_flow_metrics.json",
        {
            "variant": variant,
            "latent_dim": latent_dim,
            "validation_metrics": validation_metrics,
            "best_validation_mae": best_mae,
            "elapsed_s": timer.elapsed(),
        },
    )
    console.print(f"Wrote Phase 6 latent flow: {checkpoint_path}")
    return checkpoint_path


def evaluate_phase6_latent_flow(
    config: Config,
    latent_dim: int,
    variant: str | None = None,
    seed: int = 0,
    episodes: int | None = None,
    force: bool = False,
) -> Path:
    variant = variant or str(config.get("incremental.phase6.default_variant", "wm_recon"))
    checkpoint_path = train_phase6_latent_flow(
        config,
        latent_dim=latent_dim,
        variant=variant,
        seed=seed,
        force=force,
    )
    device = default_device()
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    encoder_path = Path(checkpoint["encoder_checkpoint"])
    encoder, encoder_checkpoint = _load_phase6_encoder(encoder_path, device)
    model = FlowModel(
        sample_dim=int(checkpoint["sample_dim"]),
        cond_dim=int(checkpoint["cond_dim"]),
        hidden_dim=int(checkpoint["hidden_dim"]),
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    frame_norm = Standardizer.from_state_dict(encoder_checkpoint["frame_norm"])
    cond_norm = Standardizer.from_state_dict(checkpoint["cond_norm"])
    action_norm = Standardizer.from_state_dict(checkpoint["action_norm"])
    zero_action_norm = action_norm.transform(
        np.zeros((1, int(checkpoint["sample_dim"])), dtype=np.float32)
    )[0]
    dino = _phase4_dino_from_config(config, device)
    flow_steps = int(checkpoint["flow_steps"])
    eval_episodes = int(episodes or config.get("incremental.phase6.eval_episodes", 100))
    num_envs = min(int(config.get("incremental.phase6.eval_num_envs", 64)), eval_episodes)
    env = _phase4_make_visual_env(config, num_envs)
    action_low = torch.as_tensor(env.single_action_space.low, device=device, dtype=torch.float32)
    action_high = torch.as_tensor(env.single_action_space.high, device=device, dtype=torch.float32)
    obs, _info = env.reset(seed=int(config.get("incremental.phase6.eval_seed", 10000)))
    prev_action_norm = np.repeat(zero_action_norm[None, :], num_envs, axis=0).astype(np.float32)
    successes: list[float] = []
    final_rewards: list[float] = []
    max_rewards: list[float] = []
    episode_lengths: list[int] = []
    latencies: list[float] = []
    active_max_reward = np.full(num_envs, -np.inf, dtype=np.float32)
    active_lengths = np.zeros(num_envs, dtype=np.int32)
    while len(successes) < eval_episodes:
        timer = Timer()
        frames = frame_norm.transform(
            _phase4_frame_inputs(obs, dino, int(config.get("dino.batch_size", 64)))
        )
        with torch.inference_mode():
            z = encoder(torch.from_numpy(frames).to(device).float()).cpu().numpy()
            cond_np = cond_norm.transform(np.concatenate([z, prev_action_norm], axis=-1))
            cond = torch.from_numpy(cond_np).to(device).float()
            zero = torch.zeros(cond.shape[0], model.sample_dim, device=device, dtype=cond.dtype)
            pred_norm = sample_flow(
                model,
                cond,
                flow_steps,
                model.sample_dim,
                initial_noise=zero,
            )
            raw_action = action_norm.inverse(pred_norm.cpu().numpy())
        latencies.append(timer.elapsed() / num_envs)
        action = torch.from_numpy(raw_action).to(device).float()
        if bool(config.get("policy.clip_actions_to_env_space", True)):
            action = torch.clamp(action, action_low, action_high)
        prev_action_norm = action_norm.transform(action.detach().cpu().numpy().astype(np.float32))
        obs, reward, _terminated, _truncated, info = env.step(action)
        reward_np = _numpy(reward).reshape(-1).astype(np.float32)
        active_max_reward = np.maximum(active_max_reward, reward_np)
        active_lengths += 1
        if "final_info" in info:
            mask = _numpy(info["_final_info"]).reshape(-1).astype(bool)
            if mask.any():
                episode_info = info["final_info"]["episode"]
                success_once = _numpy(episode_info["success_once"]).reshape(-1)
                for env_idx in np.flatnonzero(mask):
                    successes.append(float(success_once[env_idx]))
                    final_rewards.append(float(reward_np[env_idx]))
                    max_rewards.append(float(active_max_reward[env_idx]))
                    episode_lengths.append(int(active_lengths[env_idx]))
                    active_max_reward[env_idx] = -np.inf
                    active_lengths[env_idx] = 0
                    prev_action_norm[env_idx] = zero_action_norm
                    if len(successes) >= eval_episodes:
                        break
    env.close()
    metrics = {
        "success": float(np.mean(successes[:eval_episodes])),
        "success_stderr": float(np.std(successes[:eval_episodes]) / np.sqrt(eval_episodes)),
        "final_reward": float(np.mean(final_rewards[:eval_episodes])),
        "max_reward": float(np.mean(max_rewards[:eval_episodes])),
        "mean_episode_length": float(np.mean(episode_lengths[:eval_episodes])),
        "inference_latency_s": float(np.mean(latencies)),
        "episodes": eval_episodes,
        "seed_start": int(config.get("incremental.phase6.eval_seed", 10000)),
        "num_envs": num_envs,
    }
    visual_flow_path = (
        config.path_value("paths.incremental_results_dir")
        / "phase5"
        / "concat_h1"
        / f"seed{seed}"
        / "visual_flow.json"
    )
    import json

    with visual_flow_path.open("r", encoding="utf-8") as f:
        visual_flow = json.load(f)
    visual_success = float(visual_flow["closed_loop"]["success"])
    results_dir = ensure_dir(
        config.path_value("paths.incremental_results_dir")
        / "phase6"
        / f"{variant}_z{latent_dim}"
        / f"seed{seed}"
    )
    output_path = results_dir / "latent_flow_control.json"
    payload = {
        "phase": 6,
        "method": "latent_zero_noise_flow_control",
        "variant": variant,
        "latent_dim": latent_dim,
        "seed": seed,
        "closed_loop": metrics,
        "direct_visual_flow_success": visual_success,
        "control_gate_80pct": metrics["success"] >= 0.8 * visual_success,
        "control_gate_90pct": metrics["success"] >= 0.9 * visual_success,
        "held_out_action_metrics": checkpoint["validation_metrics"],
        "metadata": _runtime_metadata(config),
    }
    write_json(output_path, payload)
    console.print(payload)
    return output_path


def collect_phase6_latent_dagger_queries(
    config: Config,
    latent_dim: int,
    variant: str | None = None,
    iteration: int = 1,
    seed: int = 0,
    episodes: int | None = None,
    force: bool = False,
) -> Path:
    variant = variant or str(config.get("incremental.phase6.default_variant", "wm_recon"))
    artifact_dir = ensure_dir(
        config.path_value("paths.incremental_artifact_dir")
        / "phase6"
        / f"{variant}_z{latent_dim}"
        / f"seed{seed}"
    )
    output_path = artifact_dir / f"latent_dagger_iter{iteration}.npz"
    if output_path.exists() and not force:
        console.print(f"Phase 6 latent DAgger queries exist: {output_path}")
        return output_path

    checkpoint_path = train_phase6_latent_bc(
        config,
        latent_dim=latent_dim,
        variant=variant,
        seed=seed,
        force=False,
    )
    device = default_device()
    bc_checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    encoder_path = Path(bc_checkpoint["encoder_checkpoint"])
    encoder, encoder_checkpoint = _load_phase6_encoder(encoder_path, device)
    model = MLP(
        int(bc_checkpoint["cond_dim"]),
        int(bc_checkpoint["action_dim"]),
        int(bc_checkpoint["hidden_dim"]),
        depth=4,
    ).to(device)
    model.load_state_dict(bc_checkpoint["model"])
    model.eval()
    frame_norm = Standardizer.from_state_dict(encoder_checkpoint["frame_norm"])
    action_norm = Standardizer.from_state_dict(bc_checkpoint["action_norm"])
    zero_action_norm = action_norm.transform(
        np.zeros((1, int(bc_checkpoint["action_dim"])), dtype=np.float32)
    )[0]
    teacher = load_ppo_agent(_rl_paths(config).best, device)
    dino = _phase4_dino_from_config(config, device)
    eval_episodes = int(episodes or config.get("incremental.phase6.dagger_episodes", 200))
    num_envs = min(int(config.get("incremental.phase6.eval_num_envs", 64)), eval_episodes)
    env = _phase4_make_visual_env(config, num_envs)
    action_low = torch.as_tensor(env.single_action_space.low, device=device, dtype=torch.float32)
    action_high = torch.as_tensor(env.single_action_space.high, device=device, dtype=torch.float32)
    obs, _info = env.reset(
        seed=int(config.get("incremental.phase6.dagger_seed", 730000)) + 1000 * iteration
    )
    prev_action_norm = np.repeat(zero_action_norm[None, :], num_envs, axis=0).astype(np.float32)
    frames_rows = []
    prev_action_rows = []
    teacher_action_rows = []
    successes: list[float] = []
    while len(successes) < eval_episodes:
        frames_raw = _phase4_frame_inputs(obs, dino, int(config.get("dino.batch_size", 64)))
        _rgb, state = _phase4_rgb_state(obs)
        with torch.inference_mode():
            teacher_action = torch.clamp(
                teacher.actor_mean(torch.from_numpy(state).to(device).float()),
                action_low,
                action_high,
            )
            frames_norm = frame_norm.transform(frames_raw)
            z = encoder(torch.from_numpy(frames_norm).to(device).float())
            prev_action_t = torch.from_numpy(prev_action_norm).to(device).float()
            pred_norm = model(torch.cat([z, prev_action_t], dim=-1))
            raw_action = action_norm.inverse(pred_norm.cpu().numpy())
        frames_rows.append(frames_raw.astype(np.float32))
        prev_action_rows.append(prev_action_norm.copy())
        teacher_action_rows.append(teacher_action.cpu().numpy().astype(np.float32))
        action = torch.from_numpy(raw_action).to(device).float()
        if bool(config.get("policy.clip_actions_to_env_space", True)):
            action = torch.clamp(action, action_low, action_high)
        prev_action_norm = action_norm.transform(action.detach().cpu().numpy().astype(np.float32))
        obs, _reward, _terminated, _truncated, info = env.step(action)
        if "final_info" in info:
            mask = _numpy(info["_final_info"]).reshape(-1).astype(bool)
            if mask.any():
                episode_info = info["final_info"]["episode"]
                success_once = _numpy(episode_info["success_once"]).reshape(-1)
                for env_idx in np.flatnonzero(mask):
                    successes.append(float(success_once[env_idx]))
                    prev_action_norm[env_idx] = zero_action_norm
                    if len(successes) >= eval_episodes:
                        break
    env.close()
    np.savez_compressed(
        output_path,
        frames=np.concatenate(frames_rows, axis=0).astype(np.float32),
        prev_action_norm=np.concatenate(prev_action_rows, axis=0).astype(np.float32),
        teacher_actions=np.concatenate(teacher_action_rows, axis=0).astype(np.float32),
        dataset_type=np.asarray("state_query_dataset"),
        semantics=np.asarray("latent-policy visited visual states relabeled by privileged teacher"),
        collection_success=np.asarray(successes, dtype=np.float32),
    )
    console.print(f"Wrote Phase 6 latent DAgger queries: {output_path}")
    return output_path


def train_phase6_latent_dagger_bc(
    config: Config,
    latent_dim: int,
    variant: str | None = None,
    iteration: int = 1,
    seed: int = 0,
    force: bool = False,
) -> Path:
    set_seed(seed)
    variant = variant or str(config.get("incremental.phase6.default_variant", "wm_recon"))
    query_path = collect_phase6_latent_dagger_queries(
        config,
        latent_dim=latent_dim,
        variant=variant,
        iteration=iteration,
        seed=seed,
        force=False,
    )
    encoder_path = train_phase6_representation(
        config,
        latent_dim=latent_dim,
        variant=variant,
        seed=seed,
        force=False,
    )
    artifact_dir = ensure_dir(
        config.path_value("paths.incremental_artifact_dir")
        / "phase6"
        / f"{variant}_z{latent_dim}"
        / f"seed{seed}"
    )
    checkpoint_path = artifact_dir / f"latent_dagger_bc_iter{iteration}.pt"
    if checkpoint_path.exists() and not force:
        console.print(f"Phase 6 latent DAgger BC exists: {checkpoint_path}")
        return checkpoint_path
    device = default_device()
    encoder, encoder_checkpoint = _load_phase6_encoder(encoder_path, device)
    frame_norm = Standardizer.from_state_dict(encoder_checkpoint["frame_norm"])
    action_norm = Standardizer.from_state_dict(encoder_checkpoint["action_norm"])
    train_episodes, val_episodes, _data_metadata = _load_phase6_train_episodes(config)
    train_cond, train_actions = _phase6_encode_control_episodes(
        encoder,
        frame_norm,
        action_norm,
        train_episodes,
        device,
    )
    val_cond, val_actions = _phase6_encode_control_episodes(
        encoder,
        frame_norm,
        action_norm,
        val_episodes,
        device,
    )
    with np.load(query_path) as data:
        query_frames = np.asarray(data["frames"], dtype=np.float32)
        query_prev_actions = np.asarray(data["prev_action_norm"], dtype=np.float32)
        query_actions = np.asarray(data["teacher_actions"], dtype=np.float32)
    with torch.inference_mode():
        encoded_chunks = []
        query_norm = frame_norm.transform(query_frames)
        for start in range(0, len(query_norm), 4096):
            encoded_chunks.append(
                encoder(torch.from_numpy(query_norm[start : start + 4096]).to(device).float())
                .cpu()
                .numpy()
            )
    query_cond = np.concatenate([np.concatenate(encoded_chunks), query_prev_actions], axis=-1)
    rng = np.random.default_rng(seed + iteration)
    order = rng.permutation(len(query_cond))
    split = int(0.8 * len(order))
    query_train = order[:split]
    query_val = order[split:]
    repeats = int(config.get("incremental.phase6.dagger_query_repeats", 4))
    repeated_query_train = np.repeat(query_train, repeats)
    train_cond = np.concatenate([train_cond, query_cond[repeated_query_train]], axis=0)
    train_actions = np.concatenate([train_actions, query_actions[repeated_query_train]], axis=0)
    val_cond = np.concatenate([val_cond, query_cond[query_val]], axis=0)
    val_actions = np.concatenate([val_actions, query_actions[query_val]], axis=0)
    train_dataset = TensorDataset(
        torch.from_numpy(train_cond).float(),
        torch.from_numpy(action_norm.transform(train_actions)).float(),
    )
    loader = DataLoader(
        train_dataset,
        batch_size=int(config.get("incremental.phase6.control_batch_size", 512)),
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    model = MLP(
        train_cond.shape[-1],
        train_actions.shape[-1],
        int(config.get("incremental.phase6.hidden_dim", 1024)),
        depth=4,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config.get("incremental.phase6.control_lr", 3e-4)),
    )
    epochs = int(config.get("incremental.phase6.control_epochs", 80))
    x_val = torch.from_numpy(val_cond).to(device).float()
    y_val = torch.from_numpy(action_norm.transform(val_actions)).to(device).float()
    best_state = None
    best_val = float("inf")
    history = []
    timer = Timer()
    for epoch in trange(1, epochs + 1, desc=f"train phase6 latent DAgger BC {iteration}"):
        model.train()
        loss_sum = 0.0
        count = 0
        for x, y in loader:
            x = x.to(device, non_blocking=True).float()
            y = y.to(device, non_blocking=True).float()
            loss = torch.mean((model(x) - y) ** 2)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach().cpu()) * len(x)
            count += len(x)
        model.eval()
        with torch.inference_mode():
            val_mse = float(torch.mean((model(x_val) - y_val) ** 2).cpu())
        history.append({"epoch": epoch, "train_mse": loss_sum / count, "validation_mse": val_mse})
        if val_mse < best_val:
            best_val = val_mse
            best_state = copy.deepcopy(model.state_dict())
    if best_state is None:
        raise RuntimeError("Phase 6 latent DAgger BC training produced no checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    with torch.inference_mode():
        pred_action = action_norm.inverse(model(x_val).cpu().numpy())
    validation_metrics = _action_regression_metrics(pred_action, val_actions)
    payload = {
        "model": model.state_dict(),
        "variant": variant,
        "latent_dim": latent_dim,
        "iteration": iteration,
        "cond_dim": train_cond.shape[-1],
        "hidden_dim": int(config.get("incremental.phase6.hidden_dim", 1024)),
        "action_dim": train_actions.shape[-1],
        "encoder_checkpoint": str(encoder_path),
        "action_norm": action_norm.state_dict(),
        "query_path": str(query_path),
        "query_train_samples": int(len(query_train)),
        "query_validation_samples": int(len(query_val)),
        "query_repeats": repeats,
        "validation_metrics": validation_metrics,
        "best_validation_mse": best_val,
        "history": history,
        "elapsed_s": timer.elapsed(),
        "metadata": _runtime_metadata(config),
    }
    torch.save(payload, checkpoint_path)
    write_json(
        artifact_dir / f"latent_dagger_bc_iter{iteration}_metrics.json",
        {
            "variant": variant,
            "latent_dim": latent_dim,
            "iteration": iteration,
            "validation_metrics": validation_metrics,
            "best_validation_mse": best_val,
            "query_path": str(query_path),
            "elapsed_s": timer.elapsed(),
        },
    )
    console.print(f"Wrote Phase 6 latent DAgger BC: {checkpoint_path}")
    return checkpoint_path


def evaluate_phase6_latent_dagger_bc(
    config: Config,
    latent_dim: int,
    variant: str | None = None,
    iteration: int = 1,
    seed: int = 0,
    episodes: int | None = None,
    force: bool = False,
) -> Path:
    variant = variant or str(config.get("incremental.phase6.default_variant", "wm_recon"))
    checkpoint_path = train_phase6_latent_dagger_bc(
        config,
        latent_dim=latent_dim,
        variant=variant,
        iteration=iteration,
        seed=seed,
        force=force,
    )
    device = default_device()
    bc_checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    encoder_path = Path(bc_checkpoint["encoder_checkpoint"])
    encoder, encoder_checkpoint = _load_phase6_encoder(encoder_path, device)
    model = MLP(
        int(bc_checkpoint["cond_dim"]),
        int(bc_checkpoint["action_dim"]),
        int(bc_checkpoint["hidden_dim"]),
        depth=4,
    ).to(device)
    model.load_state_dict(bc_checkpoint["model"])
    model.eval()
    frame_norm = Standardizer.from_state_dict(encoder_checkpoint["frame_norm"])
    action_norm = Standardizer.from_state_dict(bc_checkpoint["action_norm"])
    zero_action_norm = action_norm.transform(
        np.zeros((1, int(bc_checkpoint["action_dim"])), dtype=np.float32)
    )[0]
    dino = _phase4_dino_from_config(config, device)
    eval_episodes = int(episodes or config.get("incremental.phase6.eval_episodes", 100))
    num_envs = min(int(config.get("incremental.phase6.eval_num_envs", 64)), eval_episodes)
    env = _phase4_make_visual_env(config, num_envs)
    action_low = torch.as_tensor(env.single_action_space.low, device=device, dtype=torch.float32)
    action_high = torch.as_tensor(env.single_action_space.high, device=device, dtype=torch.float32)
    obs, _info = env.reset(seed=int(config.get("incremental.phase6.eval_seed", 10000)))
    prev_action_norm = np.repeat(zero_action_norm[None, :], num_envs, axis=0).astype(np.float32)
    successes: list[float] = []
    final_rewards: list[float] = []
    max_rewards: list[float] = []
    episode_lengths: list[int] = []
    latencies: list[float] = []
    active_max_reward = np.full(num_envs, -np.inf, dtype=np.float32)
    active_lengths = np.zeros(num_envs, dtype=np.int32)
    while len(successes) < eval_episodes:
        timer = Timer()
        frames = frame_norm.transform(
            _phase4_frame_inputs(obs, dino, int(config.get("dino.batch_size", 64)))
        )
        with torch.inference_mode():
            z = encoder(torch.from_numpy(frames).to(device).float())
            prev_action_t = torch.from_numpy(prev_action_norm).to(device).float()
            pred_norm = model(torch.cat([z, prev_action_t], dim=-1))
            raw_action = action_norm.inverse(pred_norm.cpu().numpy())
        latencies.append(timer.elapsed() / num_envs)
        action = torch.from_numpy(raw_action).to(device).float()
        if bool(config.get("policy.clip_actions_to_env_space", True)):
            action = torch.clamp(action, action_low, action_high)
        prev_action_norm = action_norm.transform(action.detach().cpu().numpy().astype(np.float32))
        obs, reward, _terminated, _truncated, info = env.step(action)
        reward_np = _numpy(reward).reshape(-1).astype(np.float32)
        active_max_reward = np.maximum(active_max_reward, reward_np)
        active_lengths += 1
        if "final_info" in info:
            mask = _numpy(info["_final_info"]).reshape(-1).astype(bool)
            if mask.any():
                episode_info = info["final_info"]["episode"]
                success_once = _numpy(episode_info["success_once"]).reshape(-1)
                for env_idx in np.flatnonzero(mask):
                    successes.append(float(success_once[env_idx]))
                    final_rewards.append(float(reward_np[env_idx]))
                    max_rewards.append(float(active_max_reward[env_idx]))
                    episode_lengths.append(int(active_lengths[env_idx]))
                    active_max_reward[env_idx] = -np.inf
                    active_lengths[env_idx] = 0
                    prev_action_norm[env_idx] = zero_action_norm
                    if len(successes) >= eval_episodes:
                        break
    env.close()
    metrics = {
        "success": float(np.mean(successes[:eval_episodes])),
        "success_stderr": float(np.std(successes[:eval_episodes]) / np.sqrt(eval_episodes)),
        "final_reward": float(np.mean(final_rewards[:eval_episodes])),
        "max_reward": float(np.mean(max_rewards[:eval_episodes])),
        "mean_episode_length": float(np.mean(episode_lengths[:eval_episodes])),
        "inference_latency_s": float(np.mean(latencies)),
        "episodes": eval_episodes,
        "seed_start": int(config.get("incremental.phase6.eval_seed", 10000)),
        "num_envs": num_envs,
    }
    visual_flow_path = (
        config.path_value("paths.incremental_results_dir")
        / "phase5"
        / "concat_h1"
        / f"seed{seed}"
        / "visual_flow.json"
    )
    import json

    with visual_flow_path.open("r", encoding="utf-8") as f:
        visual_flow = json.load(f)
    visual_success = float(visual_flow["closed_loop"]["success"])
    results_dir = ensure_dir(
        config.path_value("paths.incremental_results_dir")
        / "phase6"
        / f"{variant}_z{latent_dim}"
        / f"seed{seed}"
    )
    output_path = results_dir / f"latent_dagger_bc_iter{iteration}_control.json"
    payload = {
        "phase": 6,
        "method": "latent_dagger_deterministic_bc_control",
        "variant": variant,
        "latent_dim": latent_dim,
        "iteration": iteration,
        "seed": seed,
        "closed_loop": metrics,
        "direct_visual_flow_success": visual_success,
        "control_gate_80pct": metrics["success"] >= 0.8 * visual_success,
        "control_gate_90pct": metrics["success"] >= 0.9 * visual_success,
        "held_out_action_metrics": bc_checkpoint["validation_metrics"],
        "query_path": bc_checkpoint["query_path"],
        "metadata": _runtime_metadata(config),
    }
    write_json(output_path, payload)
    console.print(payload)
    return output_path


def _phase7_defaults(
    config: Config,
    latent_dim: int | None,
    variant: str | None,
    horizon_steps: int | None,
    action_chunk_steps: int | None,
    goal_encoding: str | None,
    goal_dropout_prob: float | None,
) -> tuple[int, str, int, int, str, float]:
    return (
        int(latent_dim or config.get("incremental.phase7.latent_dim", 256)),
        str(variant or config.get("incremental.phase7.variant", "ae_recon")),
        int(horizon_steps or config.get("incremental.phase7.horizon_steps", 10)),
        int(action_chunk_steps or config.get("incremental.phase7.action_chunk_steps", 1)),
        str(goal_encoding or config.get("incremental.phase7.goal_encoding", "absolute")),
        float(
            config.get("incremental.phase7.goal_dropout_prob", 0.0)
            if goal_dropout_prob is None
            else goal_dropout_prob
        ),
    )


def _phase7_tag(
    variant: str,
    latent_dim: int,
    horizon_steps: int,
    action_chunk_steps: int,
    goal_encoding: str = "absolute",
    goal_dropout_prob: float = 0.0,
) -> str:
    tag = f"{variant}_z{latent_dim}_k{horizon_steps}_h{action_chunk_steps}"
    if goal_encoding != "absolute":
        tag = f"{tag}_{goal_encoding}"
    if goal_dropout_prob > 0.0:
        tag = f"{tag}_gd{int(round(100 * goal_dropout_prob))}"
    return tag


def _phase7_condition(
    z: np.ndarray,
    goal: np.ndarray,
    prev_action_norm: np.ndarray,
    goal_encoding: str,
) -> np.ndarray:
    if goal_encoding == "absolute":
        goal_part = goal
    elif goal_encoding == "delta":
        goal_part = goal - z
    else:
        raise ValueError(f"Unknown Phase 7 goal encoding: {goal_encoding}")
    return np.concatenate([z, goal_part, prev_action_norm], axis=-1).astype(np.float32)


class _Phase7OracleDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        conditions: np.ndarray,
        actions: np.ndarray,
        length: int,
        latent_dim: int,
        goal_dropout_prob: float = 0.0,
    ) -> None:
        if len(conditions) == 0:
            raise ValueError("Phase 7 oracle dataset is empty")
        self.conditions = conditions.astype(np.float32)
        self.actions = actions.astype(np.float32)
        self.length = length
        self.latent_dim = latent_dim
        self.goal_dropout_prob = goal_dropout_prob

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, _index: int) -> tuple[torch.Tensor, torch.Tensor]:
        i = int(np.random.randint(0, len(self.conditions)))
        condition = self.conditions[i].copy()
        if self.goal_dropout_prob > 0.0 and np.random.random() < self.goal_dropout_prob:
            condition[self.latent_dim : 2 * self.latent_dim] = 0.0
        return torch.from_numpy(condition), torch.from_numpy(self.actions[i])


def _phase7_encode_oracle_episodes(
    encoder: ObservationEncoder,
    frame_norm: Standardizer,
    action_norm: Standardizer,
    episodes: list[dict[str, np.ndarray]],
    horizon_steps: int,
    action_chunk_steps: int,
    goal_encoding: str,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    if action_chunk_steps != 1:
        raise NotImplementedError("Phase 7 currently implements H=1 low-level actions")
    conditions = []
    actions = []
    zero_action_norm = action_norm.transform(
        np.zeros((1, episodes[0]["actions"].shape[-1]), dtype=np.float32)
    )[0]
    with torch.inference_mode():
        for episode in episodes:
            if len(episode["actions"]) <= horizon_steps:
                continue
            frames_norm = frame_norm.transform(episode["frames"])
            chunks = []
            for start in range(0, len(frames_norm), 4096):
                chunks.append(
                    encoder(torch.from_numpy(frames_norm[start : start + 4096]).to(device).float())
                    .cpu()
                    .numpy()
                )
            z = np.concatenate(chunks, axis=0).astype(np.float32)
            prev_actions = action_norm.transform(episode["actions"]).astype(np.float32)
            for t in range(len(episode["actions"]) - horizon_steps):
                prev_action = prev_actions[t - 1] if t > 0 else zero_action_norm
                conditions.append(
                    _phase7_condition(z[t], z[t + horizon_steps], prev_action, goal_encoding)
                )
                actions.append(episode["actions"][t])
    if not conditions:
        raise ValueError(f"No Phase 7 samples for horizon {horizon_steps}")
    return np.stack(conditions).astype(np.float32), np.stack(actions).astype(np.float32)


def _phase7_oracle_action_metrics(
    model: nn.Module,
    conditions: np.ndarray,
    actions: np.ndarray,
    action_norm: Standardizer,
    latent_dim: int,
    goal_encoding: str,
    max_queries: int,
    seed: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    indices = np.arange(len(conditions))
    if len(indices) > max_queries:
        indices = rng.choice(indices, size=max_queries, replace=False)
    x = conditions[indices].copy()
    y = actions[indices]
    current = x[:, :latent_dim]
    goal_part = x[:, latent_dim : 2 * latent_dim]
    absolute_goal = current + goal_part if goal_encoding == "delta" else goal_part
    shuffled_order = rng.permutation(len(x))
    shuffled = x.copy()
    shuffled[:, latent_dim : 2 * latent_dim] = (
        absolute_goal[shuffled_order] - current
        if goal_encoding == "delta"
        else absolute_goal[shuffled_order]
    )
    zero_goal = x.copy()
    zero_goal[:, latent_dim : 2 * latent_dim] = -current if goal_encoding == "delta" else 0.0
    device = next(model.parameters()).device

    def predict(raw_conditions: np.ndarray) -> np.ndarray:
        preds = []
        with torch.inference_mode():
            for start in range(0, len(raw_conditions), 4096):
                pred_norm = model(torch.from_numpy(raw_conditions[start : start + 4096]).to(device).float())
                preds.append(action_norm.inverse(pred_norm.cpu().numpy()))
        return np.concatenate(preds).astype(np.float32)

    correct_pred = predict(x)
    shuffled_pred = predict(shuffled)
    zero_pred = predict(zero_goal)
    correct = _action_regression_metrics(correct_pred, y)
    shuffled_metrics = _action_regression_metrics(shuffled_pred, y)
    zero_metrics = _action_regression_metrics(zero_pred, y)
    return {
        "correct_goal": correct,
        "shuffled_goal": shuffled_metrics,
        "zero_goal": zero_metrics,
        "goal_sensitivity_l2": float(np.mean(np.linalg.norm(correct_pred - shuffled_pred, axis=-1))),
        "mae_gap_shuffled_minus_correct": float(shuffled_metrics["mae"] - correct["mae"]),
        "mae_gap_zero_minus_correct": float(zero_metrics["mae"] - correct["mae"]),
        "queries": int(len(indices)),
    }


def train_phase7_oracle_low_level(
    config: Config,
    latent_dim: int | None = None,
    variant: str | None = None,
    horizon_steps: int | None = None,
    action_chunk_steps: int | None = None,
    goal_encoding: str | None = None,
    goal_dropout_prob: float | None = None,
    seed: int = 0,
    force: bool = False,
) -> Path:
    set_seed(seed)
    (
        latent_dim,
        variant,
        horizon_steps,
        action_chunk_steps,
        goal_encoding,
        goal_dropout_prob,
    ) = _phase7_defaults(
        config,
        latent_dim,
        variant,
        horizon_steps,
        action_chunk_steps,
        goal_encoding,
        goal_dropout_prob,
    )
    if action_chunk_steps >= horizon_steps:
        raise ValueError(f"Phase 7 requires H < k, got H={action_chunk_steps}, k={horizon_steps}")
    tag = _phase7_tag(
        variant,
        latent_dim,
        horizon_steps,
        action_chunk_steps,
        goal_encoding,
        goal_dropout_prob,
    )
    artifact_dir = ensure_dir(
        config.path_value("paths.incremental_artifact_dir") / "phase7" / tag / f"seed{seed}"
    )
    checkpoint_path = artifact_dir / "oracle_low_level.pt"
    if checkpoint_path.exists() and not force:
        console.print(f"Phase 7 oracle low-level policy exists: {checkpoint_path}")
        return checkpoint_path
    encoder_path = train_phase6_representation(
        config,
        latent_dim=latent_dim,
        variant=variant,
        seed=seed,
        force=False,
    )
    device = default_device()
    encoder, encoder_checkpoint = _load_phase6_encoder(encoder_path, device)
    frame_norm = Standardizer.from_state_dict(encoder_checkpoint["frame_norm"])
    action_norm = Standardizer.from_state_dict(encoder_checkpoint["action_norm"])
    train_episodes, val_episodes, data_metadata = _load_phase6_train_episodes(config)
    train_cond, train_actions = _phase7_encode_oracle_episodes(
        encoder,
        frame_norm,
        action_norm,
        train_episodes,
        horizon_steps,
        action_chunk_steps,
        goal_encoding,
        device,
    )
    val_cond, val_actions = _phase7_encode_oracle_episodes(
        encoder,
        frame_norm,
        action_norm,
        val_episodes,
        horizon_steps,
        action_chunk_steps,
        goal_encoding,
        device,
    )
    train_dataset = _Phase7OracleDataset(
        train_cond,
        action_norm.transform(train_actions).astype(np.float32),
        length=int(config.get("incremental.phase7.batch_size", 512))
        * int(config.get("incremental.phase7.batches_per_epoch", 300)),
        latent_dim=latent_dim,
        goal_dropout_prob=goal_dropout_prob,
    )
    loader = DataLoader(
        train_dataset,
        batch_size=int(config.get("incremental.phase7.batch_size", 512)),
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    model = MLP(
        train_cond.shape[-1],
        train_actions.shape[-1],
        int(config.get("incremental.phase7.hidden_dim", 1024)),
        depth=4,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config.get("incremental.phase7.lr", 3e-4)),
    )
    x_val = torch.from_numpy(val_cond).to(device).float()
    y_val = torch.from_numpy(action_norm.transform(val_actions)).to(device).float()
    epochs = int(config.get("incremental.phase7.epochs", 80))
    best_state = None
    best_val = float("inf")
    history = []
    timer = Timer()
    for epoch in trange(1, epochs + 1, desc=f"train phase7 oracle {tag}"):
        model.train()
        loss_sum = 0.0
        count = 0
        for x, y in loader:
            x = x.to(device, non_blocking=True).float()
            y = y.to(device, non_blocking=True).float()
            loss = torch.mean((model(x) - y) ** 2)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach().cpu()) * len(x)
            count += len(x)
        model.eval()
        with torch.inference_mode():
            val_mse = float(torch.mean((model(x_val) - y_val) ** 2).cpu())
        history.append({"epoch": epoch, "train_mse": loss_sum / count, "validation_mse": val_mse})
        if val_mse < best_val:
            best_val = val_mse
            best_state = copy.deepcopy(model.state_dict())
    if best_state is None:
        raise RuntimeError("Phase 7 oracle low-level training produced no checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    validation_metrics = _phase7_oracle_action_metrics(
        model,
        val_cond,
        val_actions,
        action_norm,
        latent_dim,
        goal_encoding,
        int(config.get("incremental.phase7.validation_queries", 10000)),
        seed + horizon_steps,
    )
    payload = {
        "model": model.state_dict(),
        "variant": variant,
        "latent_dim": latent_dim,
        "horizon_steps": horizon_steps,
        "action_chunk_steps": action_chunk_steps,
        "goal_encoding": goal_encoding,
        "goal_dropout_prob": goal_dropout_prob,
        "cond_dim": train_cond.shape[-1],
        "hidden_dim": int(config.get("incremental.phase7.hidden_dim", 1024)),
        "action_dim": train_actions.shape[-1],
        "encoder_checkpoint": str(encoder_path),
        "action_norm": action_norm.state_dict(),
        "validation_metrics": validation_metrics,
        "best_validation_mse": best_val,
        "history": history,
        "data": {
            **data_metadata,
            "phase7_train_samples": int(len(train_cond)),
            "phase7_validation_samples": int(len(val_cond)),
        },
        "elapsed_s": timer.elapsed(),
        "metadata": _runtime_metadata(config),
    }
    torch.save(payload, checkpoint_path)
    write_json(
        artifact_dir / "oracle_low_level_metrics.json",
        {
            "variant": variant,
            "latent_dim": latent_dim,
            "horizon_steps": horizon_steps,
            "action_chunk_steps": action_chunk_steps,
            "goal_encoding": goal_encoding,
            "goal_dropout_prob": goal_dropout_prob,
            "validation_metrics": validation_metrics,
            "best_validation_mse": best_val,
            "elapsed_s": timer.elapsed(),
        },
    )
    console.print(f"Wrote Phase 7 oracle low-level policy: {checkpoint_path}")
    return checkpoint_path


@torch.inference_mode()
def _phase7_collect_oracle_frames(
    config: Config,
    dino: DinoExtractor,
    eval_episodes: int,
    seed_start: int,
) -> list[np.ndarray]:
    device = default_device()
    teacher = load_ppo_agent(_rl_paths(config).best, device)
    num_envs = min(int(config.get("incremental.phase7.eval_num_envs", 64)), eval_episodes)
    env = _phase4_make_visual_env(config, num_envs)
    action_low = torch.as_tensor(env.single_action_space.low, device=device, dtype=torch.float32)
    action_high = torch.as_tensor(env.single_action_space.high, device=device, dtype=torch.float32)
    obs, _info = env.reset(seed=seed_start)
    oracle: list[list[np.ndarray] | None] = [[] for _ in range(eval_episodes)]
    active_idx = np.arange(num_envs, dtype=np.int32)
    next_idx = num_envs
    completed = 0
    while completed < eval_episodes:
        frames = _phase4_frame_inputs(obs, dino, int(config.get("dino.batch_size", 64)))
        _rgb, state = _phase4_rgb_state(obs)
        for env_idx, episode_idx in enumerate(active_idx):
            if episode_idx < eval_episodes and oracle[episode_idx] is not None:
                oracle[episode_idx].append(frames[env_idx].copy())
        action = torch.clamp(
            teacher.actor_mean(torch.from_numpy(state).to(device).float()),
            action_low,
            action_high,
        )
        obs, _reward, _terminated, _truncated, info = env.step(action)
        if "final_info" in info:
            mask = _numpy(info["_final_info"]).reshape(-1).astype(bool)
            for env_idx in np.flatnonzero(mask):
                completed += 1
                if next_idx < eval_episodes:
                    active_idx[env_idx] = next_idx
                    next_idx += 1
                else:
                    active_idx[env_idx] = eval_episodes
                if completed >= eval_episodes:
                    break
    env.close()
    out = []
    for episode in oracle:
        if episode is None or not episode:
            raise RuntimeError("Oracle collection produced an empty episode")
        out.append(np.stack(episode).astype(np.float32))
    return out


@torch.inference_mode()
def _phase7_encode_oracle_frame_sequences(
    encoder: ObservationEncoder,
    frame_norm: Standardizer,
    oracle_frames: list[np.ndarray],
    device: torch.device,
) -> list[np.ndarray]:
    encoded = []
    for frames in oracle_frames:
        frame_rows = frame_norm.transform(frames)
        chunks = []
        for start in range(0, len(frame_rows), 4096):
            chunks.append(
                encoder(torch.from_numpy(frame_rows[start : start + 4096]).to(device).float())
                .cpu()
                .numpy()
            )
        encoded.append(np.concatenate(chunks, axis=0).astype(np.float32))
    return encoded


def _phase7_visual_flow_success(config: Config, seed: int) -> float:
    import json

    visual_flow_path = (
        config.path_value("paths.incremental_results_dir")
        / "phase5"
        / "concat_h1"
        / f"seed{seed}"
        / "visual_flow.json"
    )
    with visual_flow_path.open("r", encoding="utf-8") as f:
        return float(json.load(f)["closed_loop"]["success"])


@torch.inference_mode()
def _evaluate_phase7_goal_mode(
    config: Config,
    checkpoint: dict[str, Any],
    encoder: ObservationEncoder,
    frame_norm: Standardizer,
    action_norm: Standardizer,
    model: nn.Module,
    dino: DinoExtractor,
    oracle_latents: list[np.ndarray],
    goal_mode: str,
    seed_start: int,
    eval_episodes: int,
) -> dict[str, Any]:
    device = default_device()
    num_envs = min(int(config.get("incremental.phase7.eval_num_envs", 64)), eval_episodes)
    env = _phase4_make_visual_env(config, num_envs)
    action_low = torch.as_tensor(env.single_action_space.low, device=device, dtype=torch.float32)
    action_high = torch.as_tensor(env.single_action_space.high, device=device, dtype=torch.float32)
    obs, _info = env.reset(seed=seed_start)
    action_dim = int(checkpoint["action_dim"])
    latent_dim = int(checkpoint["latent_dim"])
    horizon_steps = int(checkpoint["horizon_steps"])
    goal_encoding = str(checkpoint.get("goal_encoding", "absolute"))
    zero_action_norm = action_norm.transform(np.zeros((1, action_dim), dtype=np.float32))[0]
    prev_action_norm = np.repeat(zero_action_norm[None, :], num_envs, axis=0).astype(np.float32)
    active_idx = np.arange(num_envs, dtype=np.int32)
    active_t = np.zeros(num_envs, dtype=np.int32)
    next_idx = num_envs
    successes: list[float] = []
    final_rewards: list[float] = []
    max_rewards: list[float] = []
    episode_lengths: list[int] = []
    latencies: list[float] = []
    active_max_reward = np.full(num_envs, -np.inf, dtype=np.float32)
    active_lengths = np.zeros(num_envs, dtype=np.int32)
    goal_distances = []
    while len(successes) < eval_episodes:
        timer = Timer()
        frames = frame_norm.transform(
            _phase4_frame_inputs(obs, dino, int(config.get("dino.batch_size", 64)))
        )
        z = encoder(torch.from_numpy(frames).to(device).float()).cpu().numpy().astype(np.float32)
        goals = np.zeros((num_envs, latent_dim), dtype=np.float32)
        for env_idx in range(num_envs):
            episode_idx = int(active_idx[env_idx])
            t = int(active_t[env_idx])
            if goal_mode == "zero" or episode_idx >= eval_episodes:
                continue
            source_idx = episode_idx
            if goal_mode == "shuffled":
                source_idx = (episode_idx + max(1, eval_episodes // 2)) % eval_episodes
            elif goal_mode != "correct":
                raise ValueError(f"Unknown Phase 7 goal mode: {goal_mode}")
            source = oracle_latents[source_idx]
            goals[env_idx] = source[min(t + horizon_steps, len(source) - 1)]
        cond = _phase7_condition(z, goals, prev_action_norm, goal_encoding)
        pred_norm = model(torch.from_numpy(cond).to(device).float())
        raw_action = action_norm.inverse(pred_norm.cpu().numpy())
        latencies.append(timer.elapsed() / num_envs)
        goal_distances.extend(np.linalg.norm(z - goals, axis=-1).tolist())
        action = torch.from_numpy(raw_action).to(device).float()
        if bool(config.get("policy.clip_actions_to_env_space", True)):
            action = torch.clamp(action, action_low, action_high)
        prev_action_norm = action_norm.transform(action.cpu().numpy().astype(np.float32))
        obs, reward, _terminated, _truncated, info = env.step(action)
        reward_np = _numpy(reward).reshape(-1).astype(np.float32)
        active_max_reward = np.maximum(active_max_reward, reward_np)
        active_lengths += 1
        active_t += 1
        if "final_info" in info:
            mask = _numpy(info["_final_info"]).reshape(-1).astype(bool)
            if mask.any():
                episode_info = info["final_info"]["episode"]
                success_once = _numpy(episode_info["success_once"]).reshape(-1)
                for env_idx in np.flatnonzero(mask):
                    successes.append(float(success_once[env_idx]))
                    final_rewards.append(float(reward_np[env_idx]))
                    max_rewards.append(float(active_max_reward[env_idx]))
                    episode_lengths.append(int(active_lengths[env_idx]))
                    active_max_reward[env_idx] = -np.inf
                    active_lengths[env_idx] = 0
                    active_t[env_idx] = 0
                    prev_action_norm[env_idx] = zero_action_norm
                    if next_idx < eval_episodes:
                        active_idx[env_idx] = next_idx
                        next_idx += 1
                    else:
                        active_idx[env_idx] = eval_episodes
                    if len(successes) >= eval_episodes:
                        break
    env.close()
    return {
        "success": float(np.mean(successes[:eval_episodes])),
        "success_stderr": float(np.std(successes[:eval_episodes]) / np.sqrt(eval_episodes)),
        "final_reward": float(np.mean(final_rewards[:eval_episodes])),
        "max_reward": float(np.mean(max_rewards[:eval_episodes])),
        "mean_episode_length": float(np.mean(episode_lengths[:eval_episodes])),
        "inference_latency_s": float(np.mean(latencies)),
        "latent_current_to_goal_l2": float(np.mean(goal_distances)),
        "episodes": eval_episodes,
        "seed_start": seed_start,
        "num_envs": num_envs,
    }


def evaluate_phase7_oracle_low_level(
    config: Config,
    latent_dim: int | None = None,
    variant: str | None = None,
    horizon_steps: int | None = None,
    action_chunk_steps: int | None = None,
    goal_encoding: str | None = None,
    goal_dropout_prob: float | None = None,
    seed: int = 0,
    episodes: int | None = None,
    goal_mode: str = "all",
    force: bool = False,
) -> Path:
    (
        latent_dim,
        variant,
        horizon_steps,
        action_chunk_steps,
        goal_encoding,
        goal_dropout_prob,
    ) = _phase7_defaults(
        config,
        latent_dim,
        variant,
        horizon_steps,
        action_chunk_steps,
        goal_encoding,
        goal_dropout_prob,
    )
    checkpoint_path = train_phase7_oracle_low_level(
        config,
        latent_dim=latent_dim,
        variant=variant,
        horizon_steps=horizon_steps,
        action_chunk_steps=action_chunk_steps,
        goal_encoding=goal_encoding,
        goal_dropout_prob=goal_dropout_prob,
        seed=seed,
        force=force,
    )
    device = default_device()
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    encoder, encoder_checkpoint = _load_phase6_encoder(Path(checkpoint["encoder_checkpoint"]), device)
    frame_norm = Standardizer.from_state_dict(encoder_checkpoint["frame_norm"])
    action_norm = Standardizer.from_state_dict(checkpoint["action_norm"])
    model = MLP(
        int(checkpoint["cond_dim"]),
        int(checkpoint["action_dim"]),
        int(checkpoint["hidden_dim"]),
        depth=4,
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    dino = _phase4_dino_from_config(config, device)
    eval_episodes = int(episodes or config.get("incremental.phase7.eval_episodes", 100))
    seed_start = int(config.get("incremental.phase7.eval_seed", 10000))
    oracle_frames = _phase7_collect_oracle_frames(config, dino, eval_episodes, seed_start)
    oracle_latents = _phase7_encode_oracle_frame_sequences(encoder, frame_norm, oracle_frames, device)
    modes = ["correct", "shuffled", "zero"] if goal_mode == "all" else [goal_mode]
    closed_loop = {
        mode: _evaluate_phase7_goal_mode(
            config,
            checkpoint,
            encoder,
            frame_norm,
            action_norm,
            model,
            dino,
            oracle_latents,
            mode,
            seed_start,
            eval_episodes,
        )
        for mode in modes
    }
    correct_success = closed_loop["correct"]["success"] if "correct" in closed_loop else None
    visual_success = _phase7_visual_flow_success(config, seed)
    tag = _phase7_tag(
        variant,
        latent_dim,
        horizon_steps,
        action_chunk_steps,
        goal_encoding,
        goal_dropout_prob,
    )
    results_dir = ensure_dir(
        config.path_value("paths.incremental_results_dir") / "phase7" / tag / f"seed{seed}"
    )
    output_path = results_dir / f"oracle_low_level_{goal_mode}.json"
    payload = {
        "phase": 7,
        "method": "oracle_future_latent_low_level",
        "variant": variant,
        "latent_dim": latent_dim,
        "horizon_steps": horizon_steps,
        "action_chunk_steps": action_chunk_steps,
        "goal_encoding": goal_encoding,
        "goal_dropout_prob": goal_dropout_prob,
        "seed": seed,
        "goal_mode": goal_mode,
        "closed_loop": closed_loop,
        "validation_action_metrics": checkpoint["validation_metrics"],
        "direct_visual_flow_success": visual_success,
        "oracle_gate_visual_flow": (
            bool(correct_success >= visual_success) if correct_success is not None else None
        ),
        "oracle_gate_90pct_visual_flow": (
            bool(correct_success >= 0.9 * visual_success) if correct_success is not None else None
        ),
        "metadata": _runtime_metadata(config),
    }
    write_json(output_path, payload)
    console.print(payload)
    return output_path


def _phase7_artifact_dir(
    config: Config,
    variant: str,
    latent_dim: int,
    horizon_steps: int,
    action_chunk_steps: int,
    goal_encoding: str,
    goal_dropout_prob: float,
    seed: int,
) -> Path:
    return ensure_dir(
        config.path_value("paths.incremental_artifact_dir")
        / "phase7"
        / _phase7_tag(
            variant,
            latent_dim,
            horizon_steps,
            action_chunk_steps,
            goal_encoding,
            goal_dropout_prob,
        )
        / f"seed{seed}"
    )


def _phase7_results_dir(
    config: Config,
    variant: str,
    latent_dim: int,
    horizon_steps: int,
    action_chunk_steps: int,
    goal_encoding: str,
    goal_dropout_prob: float,
    seed: int,
) -> Path:
    return ensure_dir(
        config.path_value("paths.incremental_results_dir")
        / "phase7"
        / _phase7_tag(
            variant,
            latent_dim,
            horizon_steps,
            action_chunk_steps,
            goal_encoding,
            goal_dropout_prob,
        )
        / f"seed{seed}"
    )


def _load_phase7_low_level_checkpoint(
    path: Path,
    device: torch.device,
) -> tuple[nn.Module, dict[str, Any]]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model = MLP(
        int(checkpoint["cond_dim"]),
        int(checkpoint["action_dim"]),
        int(checkpoint["hidden_dim"]),
        depth=4,
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, checkpoint


@torch.inference_mode()
def collect_phase7_oracle_dagger_queries(
    config: Config,
    latent_dim: int | None = None,
    variant: str | None = None,
    horizon_steps: int | None = None,
    action_chunk_steps: int | None = None,
    goal_encoding: str | None = None,
    goal_dropout_prob: float | None = None,
    iteration: int = 1,
    seed: int = 0,
    episodes: int | None = None,
    force: bool = False,
) -> Path:
    (
        latent_dim,
        variant,
        horizon_steps,
        action_chunk_steps,
        goal_encoding,
        goal_dropout_prob,
    ) = _phase7_defaults(
        config,
        latent_dim,
        variant,
        horizon_steps,
        action_chunk_steps,
        goal_encoding,
        goal_dropout_prob,
    )
    artifact_dir = _phase7_artifact_dir(
        config,
        variant,
        latent_dim,
        horizon_steps,
        action_chunk_steps,
        goal_encoding,
        goal_dropout_prob,
        seed,
    )
    output_path = artifact_dir / f"oracle_dagger_iter{iteration}.npz"
    if output_path.exists() and not force:
        console.print(f"Phase 7 oracle DAgger queries exist: {output_path}")
        return output_path
    base_checkpoint_path = train_phase7_oracle_low_level(
        config,
        latent_dim=latent_dim,
        variant=variant,
        horizon_steps=horizon_steps,
        action_chunk_steps=action_chunk_steps,
        goal_encoding=goal_encoding,
        goal_dropout_prob=goal_dropout_prob,
        seed=seed,
        force=False,
    )
    rollout_checkpoint_path = base_checkpoint_path
    previous_dagger = artifact_dir / f"oracle_low_level_dagger_iter{iteration - 1}.pt"
    if iteration > 1 and previous_dagger.exists():
        rollout_checkpoint_path = previous_dagger
    device = default_device()
    model, checkpoint = _load_phase7_low_level_checkpoint(rollout_checkpoint_path, device)
    encoder, encoder_checkpoint = _load_phase6_encoder(Path(checkpoint["encoder_checkpoint"]), device)
    frame_norm = Standardizer.from_state_dict(encoder_checkpoint["frame_norm"])
    action_norm = Standardizer.from_state_dict(checkpoint["action_norm"])
    dino = _phase4_dino_from_config(config, device)
    eval_episodes = int(episodes or config.get("incremental.phase7.dagger_episodes", 200))
    seed_start = int(config.get("incremental.phase7.dagger_seed", 740000)) + 1000 * iteration
    oracle_frames = _phase7_collect_oracle_frames(config, dino, eval_episodes, seed_start)
    oracle_latents = _phase7_encode_oracle_frame_sequences(encoder, frame_norm, oracle_frames, device)
    teacher = load_ppo_agent(_rl_paths(config).best, device)
    num_envs = min(int(config.get("incremental.phase7.eval_num_envs", 64)), eval_episodes)
    env = _phase4_make_visual_env(config, num_envs)
    action_low = torch.as_tensor(env.single_action_space.low, device=device, dtype=torch.float32)
    action_high = torch.as_tensor(env.single_action_space.high, device=device, dtype=torch.float32)
    obs, _info = env.reset(seed=seed_start)
    action_dim = int(checkpoint["action_dim"])
    zero_action_norm = action_norm.transform(np.zeros((1, action_dim), dtype=np.float32))[0]
    prev_action_norm = np.repeat(zero_action_norm[None, :], num_envs, axis=0).astype(np.float32)
    active_idx = np.arange(num_envs, dtype=np.int32)
    active_t = np.zeros(num_envs, dtype=np.int32)
    next_idx = num_envs
    successes: list[float] = []
    cond_rows = []
    teacher_action_rows = []
    while len(successes) < eval_episodes:
        frames_raw = _phase4_frame_inputs(obs, dino, int(config.get("dino.batch_size", 64)))
        frames = frame_norm.transform(frames_raw)
        _rgb, state = _phase4_rgb_state(obs)
        z = encoder(torch.from_numpy(frames).to(device).float()).cpu().numpy().astype(np.float32)
        goals = np.zeros((num_envs, latent_dim), dtype=np.float32)
        for env_idx in range(num_envs):
            episode_idx = int(active_idx[env_idx])
            if episode_idx >= eval_episodes:
                continue
            source = oracle_latents[episode_idx]
            goals[env_idx] = source[min(int(active_t[env_idx]) + horizon_steps, len(source) - 1)]
        cond = _phase7_condition(z, goals, prev_action_norm, goal_encoding)
        teacher_action = torch.clamp(
            teacher.actor_mean(torch.from_numpy(state).to(device).float()),
            action_low,
            action_high,
        ).cpu().numpy().astype(np.float32)
        active_mask = active_idx < eval_episodes
        if np.any(active_mask):
            cond_rows.append(cond[active_mask].copy())
            teacher_action_rows.append(teacher_action[active_mask].copy())
        pred_norm = model(torch.from_numpy(cond).to(device).float())
        raw_action = action_norm.inverse(pred_norm.cpu().numpy())
        action = torch.from_numpy(raw_action).to(device).float()
        if bool(config.get("policy.clip_actions_to_env_space", True)):
            action = torch.clamp(action, action_low, action_high)
        prev_action_norm = action_norm.transform(action.cpu().numpy().astype(np.float32))
        obs, _reward, _terminated, _truncated, info = env.step(action)
        active_t += 1
        if "final_info" in info:
            mask = _numpy(info["_final_info"]).reshape(-1).astype(bool)
            if mask.any():
                episode_info = info["final_info"]["episode"]
                success_once = _numpy(episode_info["success_once"]).reshape(-1)
                for env_idx in np.flatnonzero(mask):
                    successes.append(float(success_once[env_idx]))
                    active_t[env_idx] = 0
                    prev_action_norm[env_idx] = zero_action_norm
                    if next_idx < eval_episodes:
                        active_idx[env_idx] = next_idx
                        next_idx += 1
                    else:
                        active_idx[env_idx] = eval_episodes
                    if len(successes) >= eval_episodes:
                        break
    env.close()
    np.savez_compressed(
        output_path,
        conditions=np.concatenate(cond_rows, axis=0).astype(np.float32),
        teacher_actions=np.concatenate(teacher_action_rows, axis=0).astype(np.float32),
        collection_success=np.asarray(successes, dtype=np.float32),
        dataset_type=np.asarray("state_query_dataset"),
        semantics=np.asarray(
            "phase7 low-level visited current latents with oracle future goals relabeled by privileged teacher"
        ),
        goal_encoding=np.asarray(goal_encoding),
        goal_dropout_prob=np.asarray(goal_dropout_prob, dtype=np.float32),
        rollout_checkpoint=np.asarray(str(rollout_checkpoint_path)),
    )
    console.print(f"Wrote Phase 7 oracle DAgger queries: {output_path}")
    return output_path


def train_phase7_oracle_dagger_low_level(
    config: Config,
    latent_dim: int | None = None,
    variant: str | None = None,
    horizon_steps: int | None = None,
    action_chunk_steps: int | None = None,
    goal_encoding: str | None = None,
    goal_dropout_prob: float | None = None,
    iteration: int = 1,
    seed: int = 0,
    force: bool = False,
) -> Path:
    set_seed(seed)
    (
        latent_dim,
        variant,
        horizon_steps,
        action_chunk_steps,
        goal_encoding,
        goal_dropout_prob,
    ) = _phase7_defaults(
        config,
        latent_dim,
        variant,
        horizon_steps,
        action_chunk_steps,
        goal_encoding,
        goal_dropout_prob,
    )
    artifact_dir = _phase7_artifact_dir(
        config,
        variant,
        latent_dim,
        horizon_steps,
        action_chunk_steps,
        goal_encoding,
        goal_dropout_prob,
        seed,
    )
    checkpoint_path = artifact_dir / f"oracle_low_level_dagger_iter{iteration}.pt"
    if checkpoint_path.exists() and not force:
        console.print(f"Phase 7 oracle DAgger low-level policy exists: {checkpoint_path}")
        return checkpoint_path
    query_path = collect_phase7_oracle_dagger_queries(
        config,
        latent_dim=latent_dim,
        variant=variant,
        horizon_steps=horizon_steps,
        action_chunk_steps=action_chunk_steps,
        goal_encoding=goal_encoding,
        goal_dropout_prob=goal_dropout_prob,
        iteration=iteration,
        seed=seed,
        force=False,
    )
    encoder_path = train_phase6_representation(
        config,
        latent_dim=latent_dim,
        variant=variant,
        seed=seed,
        force=False,
    )
    device = default_device()
    encoder, encoder_checkpoint = _load_phase6_encoder(encoder_path, device)
    frame_norm = Standardizer.from_state_dict(encoder_checkpoint["frame_norm"])
    action_norm = Standardizer.from_state_dict(encoder_checkpoint["action_norm"])
    train_episodes, val_episodes, data_metadata = _load_phase6_train_episodes(config)
    train_cond, train_actions = _phase7_encode_oracle_episodes(
        encoder,
        frame_norm,
        action_norm,
        train_episodes,
        horizon_steps,
        action_chunk_steps,
        goal_encoding,
        device,
    )
    val_cond, val_actions = _phase7_encode_oracle_episodes(
        encoder,
        frame_norm,
        action_norm,
        val_episodes,
        horizon_steps,
        action_chunk_steps,
        goal_encoding,
        device,
    )
    with np.load(query_path) as data:
        query_cond = np.asarray(data["conditions"], dtype=np.float32)
        query_actions = np.asarray(data["teacher_actions"], dtype=np.float32)
    rng = np.random.default_rng(seed + iteration)
    order = rng.permutation(len(query_cond))
    split = int(0.8 * len(order))
    query_train = order[:split]
    query_val = order[split:]
    repeats = int(config.get("incremental.phase7.dagger_query_repeats", 4))
    train_cond = np.concatenate([train_cond, query_cond[np.repeat(query_train, repeats)]], axis=0)
    train_actions = np.concatenate(
        [train_actions, query_actions[np.repeat(query_train, repeats)]], axis=0
    )
    val_cond = np.concatenate([val_cond, query_cond[query_val]], axis=0)
    val_actions = np.concatenate([val_actions, query_actions[query_val]], axis=0)
    train_dataset = _Phase7OracleDataset(
        train_cond,
        action_norm.transform(train_actions).astype(np.float32),
        length=int(config.get("incremental.phase7.batch_size", 512))
        * int(config.get("incremental.phase7.batches_per_epoch", 300)),
        latent_dim=latent_dim,
        goal_dropout_prob=goal_dropout_prob,
    )
    loader = DataLoader(
        train_dataset,
        batch_size=int(config.get("incremental.phase7.batch_size", 512)),
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    model = MLP(
        train_cond.shape[-1],
        train_actions.shape[-1],
        int(config.get("incremental.phase7.hidden_dim", 1024)),
        depth=4,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config.get("incremental.phase7.lr", 3e-4)),
    )
    x_val = torch.from_numpy(val_cond).to(device).float()
    y_val = torch.from_numpy(action_norm.transform(val_actions)).to(device).float()
    epochs = int(config.get("incremental.phase7.epochs", 80))
    best_state = None
    best_val = float("inf")
    history = []
    timer = Timer()
    for epoch in trange(1, epochs + 1, desc=f"train phase7 DAgger {iteration}"):
        model.train()
        loss_sum = 0.0
        count = 0
        for x, y in loader:
            x = x.to(device, non_blocking=True).float()
            y = y.to(device, non_blocking=True).float()
            loss = torch.mean((model(x) - y) ** 2)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach().cpu()) * len(x)
            count += len(x)
        model.eval()
        with torch.inference_mode():
            val_mse = float(torch.mean((model(x_val) - y_val) ** 2).cpu())
        history.append({"epoch": epoch, "train_mse": loss_sum / count, "validation_mse": val_mse})
        if val_mse < best_val:
            best_val = val_mse
            best_state = copy.deepcopy(model.state_dict())
    if best_state is None:
        raise RuntimeError("Phase 7 oracle DAgger training produced no checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    validation_metrics = _phase7_oracle_action_metrics(
        model,
        val_cond,
        val_actions,
        action_norm,
        latent_dim,
        goal_encoding,
        int(config.get("incremental.phase7.validation_queries", 10000)),
        seed + 100 * iteration + horizon_steps,
    )
    payload = {
        "model": model.state_dict(),
        "variant": variant,
        "latent_dim": latent_dim,
        "horizon_steps": horizon_steps,
        "action_chunk_steps": action_chunk_steps,
        "goal_encoding": goal_encoding,
        "goal_dropout_prob": goal_dropout_prob,
        "iteration": iteration,
        "cond_dim": train_cond.shape[-1],
        "hidden_dim": int(config.get("incremental.phase7.hidden_dim", 1024)),
        "action_dim": train_actions.shape[-1],
        "encoder_checkpoint": str(encoder_path),
        "action_norm": action_norm.state_dict(),
        "query_path": str(query_path),
        "query_train_samples": int(len(query_train)),
        "query_validation_samples": int(len(query_val)),
        "query_repeats": repeats,
        "validation_metrics": validation_metrics,
        "best_validation_mse": best_val,
        "history": history,
        "data": {
            **data_metadata,
            "phase7_train_samples": int(len(train_cond)),
            "phase7_validation_samples": int(len(val_cond)),
        },
        "elapsed_s": timer.elapsed(),
        "metadata": _runtime_metadata(config),
    }
    torch.save(payload, checkpoint_path)
    write_json(
        artifact_dir / f"oracle_low_level_dagger_iter{iteration}_metrics.json",
        {
            "variant": variant,
            "latent_dim": latent_dim,
            "horizon_steps": horizon_steps,
            "action_chunk_steps": action_chunk_steps,
            "goal_encoding": goal_encoding,
            "goal_dropout_prob": goal_dropout_prob,
            "iteration": iteration,
            "validation_metrics": validation_metrics,
            "best_validation_mse": best_val,
            "query_path": str(query_path),
            "elapsed_s": timer.elapsed(),
        },
    )
    console.print(f"Wrote Phase 7 oracle DAgger low-level policy: {checkpoint_path}")
    return checkpoint_path


def evaluate_phase7_oracle_dagger_low_level(
    config: Config,
    latent_dim: int | None = None,
    variant: str | None = None,
    horizon_steps: int | None = None,
    action_chunk_steps: int | None = None,
    goal_encoding: str | None = None,
    goal_dropout_prob: float | None = None,
    iteration: int = 1,
    seed: int = 0,
    episodes: int | None = None,
    goal_mode: str = "all",
    force: bool = False,
) -> Path:
    (
        latent_dim,
        variant,
        horizon_steps,
        action_chunk_steps,
        goal_encoding,
        goal_dropout_prob,
    ) = _phase7_defaults(
        config,
        latent_dim,
        variant,
        horizon_steps,
        action_chunk_steps,
        goal_encoding,
        goal_dropout_prob,
    )
    checkpoint_path = train_phase7_oracle_dagger_low_level(
        config,
        latent_dim=latent_dim,
        variant=variant,
        horizon_steps=horizon_steps,
        action_chunk_steps=action_chunk_steps,
        goal_encoding=goal_encoding,
        goal_dropout_prob=goal_dropout_prob,
        iteration=iteration,
        seed=seed,
        force=force,
    )
    device = default_device()
    model, checkpoint = _load_phase7_low_level_checkpoint(checkpoint_path, device)
    encoder, encoder_checkpoint = _load_phase6_encoder(Path(checkpoint["encoder_checkpoint"]), device)
    frame_norm = Standardizer.from_state_dict(encoder_checkpoint["frame_norm"])
    action_norm = Standardizer.from_state_dict(checkpoint["action_norm"])
    dino = _phase4_dino_from_config(config, device)
    eval_episodes = int(episodes or config.get("incremental.phase7.eval_episodes", 100))
    seed_start = int(config.get("incremental.phase7.eval_seed", 10000))
    oracle_frames = _phase7_collect_oracle_frames(config, dino, eval_episodes, seed_start)
    oracle_latents = _phase7_encode_oracle_frame_sequences(encoder, frame_norm, oracle_frames, device)
    modes = ["correct", "shuffled", "zero"] if goal_mode == "all" else [goal_mode]
    closed_loop = {
        mode: _evaluate_phase7_goal_mode(
            config,
            checkpoint,
            encoder,
            frame_norm,
            action_norm,
            model,
            dino,
            oracle_latents,
            mode,
            seed_start,
            eval_episodes,
        )
        for mode in modes
    }
    correct_success = closed_loop["correct"]["success"] if "correct" in closed_loop else None
    visual_success = _phase7_visual_flow_success(config, seed)
    results_dir = _phase7_results_dir(
        config,
        variant,
        latent_dim,
        horizon_steps,
        action_chunk_steps,
        goal_encoding,
        goal_dropout_prob,
        seed,
    )
    output_path = results_dir / f"oracle_low_level_dagger_iter{iteration}_{goal_mode}.json"
    payload = {
        "phase": 7,
        "method": "oracle_future_latent_low_level_dagger",
        "variant": variant,
        "latent_dim": latent_dim,
        "horizon_steps": horizon_steps,
        "action_chunk_steps": action_chunk_steps,
        "goal_encoding": goal_encoding,
        "goal_dropout_prob": goal_dropout_prob,
        "iteration": iteration,
        "seed": seed,
        "goal_mode": goal_mode,
        "closed_loop": closed_loop,
        "validation_action_metrics": checkpoint["validation_metrics"],
        "query_path": checkpoint["query_path"],
        "direct_visual_flow_success": visual_success,
        "oracle_gate_visual_flow": (
            bool(correct_success >= visual_success) if correct_success is not None else None
        ),
        "oracle_gate_90pct_visual_flow": (
            bool(correct_success >= 0.9 * visual_success) if correct_success is not None else None
        ),
        "metadata": _runtime_metadata(config),
    }
    write_json(output_path, payload)
    console.print(payload)
    return output_path
