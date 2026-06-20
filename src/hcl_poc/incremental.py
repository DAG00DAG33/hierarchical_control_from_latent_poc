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
from hcl_poc.models import (
    FlowModel,
    MLP,
    ObservationEncoder,
    RepresentationWorldModel,
    VariationalObservationEncoder,
)
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
                raw_action = (
                    teacher.actor_mean(state_t).detach().cpu().numpy()[0].astype(np.float32)
                )
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
                max_error = max(
                    max_error, float(np.max(np.abs(actual - expected_states[step + 1])))
                )
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

    comparison_states = (
        np.random.default_rng(0).normal(size=(1024, teacher.obs_dim)).astype(np.float32)
    )
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
        "state_action_alignment": alignment["global_shift_0_mae"]
        < min(
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
        "target_near_bounds_fraction": float(np.mean(np.any(np.abs(target) >= 0.99, axis=-1))),
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
    train_states, train_actions, val_states, val_actions, data_metadata = _load_phase1_queries(
        dataset_path,
        subset,
        n_episodes,
        validation_episodes,
        label_kind,
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


def _load_phase1_bc(
    checkpoint_path: Path, device: torch.device
) -> tuple[nn.Module, dict[str, Any]]:
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
        teacher_raw_actions.append(teacher_raw[:take].detach().cpu().numpy().astype(np.float32))
        learner_raw_actions.append(learner_raw[:take].detach().cpu().numpy().astype(np.float32))
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
    train_states, train_actions, val_states, val_actions, base_metadata = _load_phase1_queries(
        dataset_path,
        "all",
        int(config.get("incremental.phase1.train_episodes", 2000)),
        int(config.get("incremental.phase1.validation_episodes", 200)),
        "deterministic_raw",
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
            sim_np = env.unwrapped.get_state().detach().cpu().numpy().astype(np.float32)
            action_np = executed_action.detach().cpu().numpy().astype(np.float32)
            for index in np.flatnonzero(active):
                state_buffers[index].append(obs_np[index])
                simulator_state_buffers[index].append(sim_np[index])
                action_buffers[index].append(action_np[index])
        obs, _reward, terminated, truncated, info = env.step(executed_action)
        obs = obs.to(default_device()).float()
        next_simulator_states = env.unwrapped.get_state().detach().cpu().numpy().astype(np.float32)
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
            "yaw_std_deg": float(config.get("incremental.phase2.recovery_yaw_std_deg", 5.0)),
        },
        "teacher_recovery_success": float(np.mean(teacher_success)),
        "teacher_recoverable_samples": int(teacher_recoverable.sum()),
        "learner_recovery_success_all": float(np.mean(learner_success)),
        "learner_recovery_success_when_teacher_recovers": (
            float(np.mean(paired_learner_success)) if len(paired_learner_success) else float("nan")
        ),
        "causal_dataset": str(causal_path),
        "gate_passed": bool(
            len(paired_learner_success) and float(np.mean(paired_learner_success)) >= 0.80
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
            cond = (
                torch.from_numpy(input_norm.transform(states[start : start + batch_size]))
                .to(device)
                .float()
            )
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
        config.path_value("paths.incremental_artifact_dir") / "phase3" / f"seed{seed}"
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
            subset = min(
                int(config.get("incremental.phase3.validation_action_subset", 4096)),
                len(val_states),
            )
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
            action_normed = (
                sample_flow(model, cond, flow_steps, sample_dim, initial_noise=noise)
                .detach()
                .cpu()
                .numpy()
            )
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
    zero_action_norm = action_norm.transform(
        np.zeros((1, data_metadata["action_dim"]), dtype=np.float32)
    )[0]
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
    eval_seed_start: int | None = None,
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
    eval_seed_start = int(
        eval_seed_start
        if eval_seed_start is not None
        else config.get("incremental.phase4.eval_seed", 10000)
    )
    obs, _info = env.reset(seed=eval_seed_start)
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
                    action_history[env_idx] = np.repeat(zero_action_norm[None, :], history, axis=0)
                    active_max_reward[env_idx] = -np.inf
                    active_lengths[env_idx] = 0
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
        "seed_start": eval_seed_start,
        "num_envs": num_envs,
    }
    results_dir = ensure_dir(
        config.path_value("paths.incremental_results_dir")
        / "phase4"
        / f"{architecture}_h{history}"
        / f"seed{seed}"
    )
    default_seed = int(config.get("incremental.phase4.eval_seed", 10000))
    output_path = results_dir / (
        "visual_bc.json"
        if eval_seed_start == default_seed
        else f"visual_bc_eval_seed{eval_seed_start}_{eval_episodes}.json"
    )
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
        list(trunk.parameters())
        + list(continuous_head.parameters())
        + list(contact_head.parameters()),
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
            loss = torch.mean(
                (pred_y - y) ** 2
            ) + torch.nn.functional.binary_cross_entropy_with_logits(pred_c, c)
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
        "velocity_better_than_mean": bool(mae[3] < baseline_mae[3] and mae[4] < baseline_mae[4]),
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
    eval_seed_start: int | None = None,
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
    eval_seed_start = int(
        eval_seed_start
        if eval_seed_start is not None
        else config.get("incremental.phase5.eval_seed", 10000)
    )
    obs, _info = env.reset(seed=eval_seed_start)
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
        policy_input = np.concatenate([frame_history, action_history], axis=-1).reshape(
            num_envs, -1
        )
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
                    action_history[env_idx] = np.repeat(zero_action_norm[None, :], history, axis=0)
                    active_max_reward[env_idx] = -np.inf
                    active_lengths[env_idx] = 0
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
        "seed_start": eval_seed_start,
        "num_envs": num_envs,
    }
    bc_result_path = (
        config.path_value("paths.incremental_results_dir")
        / "phase4"
        / f"{architecture}_h{history}"
        / f"seed{seed}"
        / (
            "visual_bc.json"
            if eval_seed_start == int(config.get("incremental.phase4.eval_seed", 10000))
            else f"visual_bc_eval_seed{eval_seed_start}_{eval_episodes}.json"
        )
    )
    if not bc_result_path.exists():
        evaluate_phase4_visual_bc(
            config,
            history=history,
            architecture=architecture,
            seed=seed,
            episodes=eval_episodes,
            eval_seed_start=eval_seed_start,
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
    default_seed = int(config.get("incremental.phase5.eval_seed", 10000))
    output_path = results_dir / (
        "visual_flow.json"
        if eval_seed_start == default_seed
        else f"visual_flow_eval_seed{eval_seed_start}_{eval_episodes}.json"
    )
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
    if variant == "vae_recon":
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
    encoder_type = "vae" if variant.startswith("vae_") else "deterministic"
    vae_beta = (
        float(config.get("incremental.phase6.vae_beta", 1e-4)) if encoder_type == "vae" else 0.0
    )

    device = default_device()
    if encoder_type == "vae":
        encoder = VariationalObservationEncoder(input_dim, latent_dim, hidden_dim).to(device)
    else:
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

    def vae_kl_loss(mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        return -0.5 * torch.mean(1.0 + logvar - mean.square() - torch.exp(logvar))

    def encode_for_training(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if encoder_type != "vae":
            return encoder(x), torch.zeros((), device=x.device)
        z, mean, logvar = encoder.sample(x)
        return z, vae_kl_loss(mean, logvar)

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
        kl_losses = []
        with torch.inference_mode():
            for batch in probe_loader:
                x_t = batch["x_t"].to(device).float()
                x_future = batch["x_future"].to(device).float()
                actions = batch["actions"].to(device).float()
                horizon = batch["horizon"].to(device)
                z_t = encoder(x_t)
                z_future = encoder(x_future)
                if encoder_type == "vae":
                    mean_t, logvar_t = encoder.encode_stats(x_t)
                    mean_future, logvar_future = encoder.encode_stats(x_future)
                    kl_losses.append(
                        float(
                            0.5
                            * (
                                vae_kl_loss(mean_t, logvar_t)
                                + vae_kl_loss(mean_future, logvar_future)
                            ).cpu()
                        )
                    )
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
                                    (recon_future[:, :-proprio_dim] - x_future[:, :-proprio_dim])
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
                                    (recon_future[:, -proprio_dim:] - x_future[:, -proprio_dim:])
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
            "kl": float(np.mean(kl_losses)) if kl_losses else 0.0,
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
        kl_sum = 0.0
        count = 0
        for batch in loader:
            x_t = batch["x_t"].to(device, non_blocking=True).float()
            x_future = batch["x_future"].to(device, non_blocking=True).float()
            actions = batch["actions"].to(device, non_blocking=True).float()
            horizon = batch["horizon"].to(device, non_blocking=True)
            z_t, kl_t = encode_for_training(x_t)
            z_future, kl_future = encode_for_training(x_future)
            kl_loss = 0.5 * (kl_t + kl_future)
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
                    reconstruction_loss(recon_t, x_t) + reconstruction_loss(recon_future, x_future)
                )
            loss = (
                prediction_weight * pred_loss
                + sigreg_weight * sigreg
                + reconstruction_weight * recon_loss
                + vae_beta * kl_loss
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            batch_count = len(x_t)
            loss_sum += float(loss.detach().cpu()) * batch_count
            pred_sum += float(pred_loss.detach().cpu()) * batch_count
            sig_sum += float(sigreg.detach().cpu()) * batch_count
            recon_sum += float(recon_loss.detach().cpu()) * batch_count
            kl_sum += float(kl_loss.detach().cpu()) * batch_count
            count += batch_count
        encoder.eval()
        world_model.eval()
        if decoder is not None:
            decoder.eval()
        val = validation_metrics()
        selection = val["prediction_mse"] + val["reconstruction_mse"] + vae_beta * val["kl"]
        row = {
            "epoch": epoch,
            "train_loss": loss_sum / count,
            "train_prediction_mse": pred_sum / count,
            "train_sigreg": sig_sum / count,
            "train_reconstruction_mse": recon_sum / count,
            "train_kl": kl_sum / count,
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
        "encoder_type": encoder_type,
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
        "vae_beta": vae_beta,
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
            "encoder_type": encoder_type,
            "latent_dim": latent_dim,
            "validation_metrics": final_val,
            "vae_beta": vae_beta,
            "elapsed_s": timer.elapsed(),
            "data": data_metadata,
        },
    )
    console.print(f"Wrote Phase 6 representation: {checkpoint_path}")
    return checkpoint_path


def _load_phase6_encoder(path: Path, device: torch.device) -> tuple[nn.Module, dict[str, Any]]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    encoder_cls = (
        VariationalObservationEncoder
        if checkpoint.get("encoder_type") == "vae"
        else ObservationEncoder
    )
    encoder = encoder_cls(
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
        return (
            frame_norm.transform(inputs),
            frame_norm.transform(next_inputs),
            {
                "representation": "raw_spatial_dino_proprio",
                "dim": int(inputs.shape[-1]),
            },
        )
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
    return (
        np.concatenate(reps).astype(np.float32),
        np.concatenate(next_reps).astype(np.float32),
        {
            "representation": "latent",
            "variant": checkpoint["variant"],
            "latent_dim": int(checkpoint["latent_dim"]),
            "checkpoint": str(path),
        },
    )


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
    reward_mean = np.broadcast_to(
        reward[train_idx].mean(axis=0, keepdims=True), target_reward.shape
    )
    action_mean = np.broadcast_to(
        actions[train_idx].mean(axis=0, keepdims=True), target_action.shape
    )
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
        "continuous_mae": {
            name: float(value) for name, value in zip(names, label_mae, strict=True)
        },
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
            "contact_auroc_over_0_80": bool(
                _binary_auc(pred_contact_logit, target_contact) >= 0.80
            ),
            "inverse_better_than_mean": bool(inverse_metrics["mae"] < inverse_baseline["mae"]),
            "reward_better_than_mean": bool(reward_mae < reward_baseline_mae),
            "forward_better_than_mean": bool(
                np.mean(next_label_mae) < np.mean(baseline_next_label_mae)
            ),
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
    variant_tag = (
        "raw"
        if representation == "raw"
        else f"{variant or config.get('incremental.phase6.default_variant', 'wm_recon')}_z{latent_dim}"
    )
    results_dir = ensure_dir(
        config.path_value("paths.incremental_results_dir") / "phase6" / variant_tag / f"seed{seed}"
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


def _clone_mani_state_dict(state: dict[str, Any]) -> dict[str, Any]:
    cloned: dict[str, Any] = {}
    for key, value in state.items():
        if isinstance(value, dict):
            cloned[key] = _clone_mani_state_dict(value)
        elif isinstance(value, torch.Tensor):
            cloned[key] = value.detach().clone()
        else:
            raise TypeError(f"Unsupported ManiSkill state value for key {key}: {type(value)}")
    return cloned


def _nested_state_max_abs_errors(
    left: dict[str, Any],
    right: dict[str, Any],
    prefix: str = "",
) -> dict[str, float]:
    errors: dict[str, float] = {}
    if set(left) != set(right):
        raise ValueError(f"State dict keys differ at {prefix}: {set(left)} != {set(right)}")
    for key in left:
        name = f"{prefix}.{key}" if prefix else key
        if isinstance(left[key], dict):
            errors.update(_nested_state_max_abs_errors(left[key], right[key], name))
        else:
            diff = torch.max(torch.abs(left[key] - right[key])).detach().cpu()
            errors[name] = float(diff)
    return errors


@torch.inference_mode()
def run_phase7_branch_audit(
    config: Config,
    latent_dim: int | None = None,
    variant: str | None = None,
    seed: int = 0,
    trials: int | None = None,
    warmup_steps: int | None = None,
    force: bool = False,
) -> Path:
    if _rl_backend(config) != "physx_cuda":
        raise RuntimeError("Phase 7 branch audit requires the canonical CUDA backend")
    latent_dim = int(latent_dim or config.get("incremental.phase7.latent_dim", 256))
    variant = str(variant or config.get("incremental.phase7.variant", "ae_recon"))
    trials = int(trials or config.get("incremental.phase7.branch_audit_trials", 8))
    warmup_steps = int(
        warmup_steps or config.get("incremental.phase7.branch_audit_warmup_steps", 8)
    )
    results_dir = ensure_dir(
        config.path_value("paths.incremental_results_dir")
        / "phase7"
        / "branch_audit"
        / f"{variant}_z{latent_dim}"
        / f"seed{seed}"
    )
    output_path = results_dir / "branch_state_parity.json"
    if output_path.exists() and not force:
        console.print(f"Phase 7 branch audit exists: {output_path}")
        return output_path

    device = default_device()
    teacher = load_ppo_agent(_rl_paths(config).best, device)
    encoder_path = train_phase6_representation(
        config,
        latent_dim=latent_dim,
        variant=variant,
        seed=seed,
        force=False,
    )
    encoder, encoder_checkpoint = _load_phase6_encoder(encoder_path, device)
    frame_norm = Standardizer.from_state_dict(encoder_checkpoint["frame_norm"])
    dino = DinoExtractor(
        str(config.get("dino.model_name")),
        device,
        feature_type=str(config.get("dino.feature_type", "cls")),
        spatial_pool=int(config.get("dino.spatial_pool", 4)),
    )
    dino_batch_size = int(config.get("dino.batch_size", 32))

    def make_env():
        return gym.make(
            config.get("env_id"),
            obs_mode="rgb+state",
            control_mode=config.get("control_mode"),
            reward_mode="normalized_dense",
            render_mode=None,
            sim_backend=_rl_backend(config),
            num_envs=1,
            reconfiguration_freq=config.get("rl.eval_reconfiguration_freq", 1),
        )

    student_env = make_env()
    branch_env = make_env()
    action_low = torch.as_tensor(student_env.action_space.low, device=device, dtype=torch.float32)
    action_high = torch.as_tensor(student_env.action_space.high, device=device, dtype=torch.float32)
    per_trial: list[dict[str, Any]] = []
    max_values = {
        "copied_flat_state_error": 0.0,
        "copied_component_error": 0.0,
        "teacher_action_error": 0.0,
        "transition_state_error": 0.0,
        "reward_error": 0.0,
        "rgb_pixel_error": 0.0,
        "dino_feature_error": 0.0,
        "latent_error": 0.0,
    }
    try:
        for trial in trange(trials, desc="phase7 branch audit"):
            reset_seed = int(config.get("incremental.phase7.branch_audit_seed", 910000))
            student_obs, _student_info = student_env.reset(seed=reset_seed + seed * 1000 + trial)
            branch_env.reset(seed=reset_seed + seed * 1000 + 100000 + trial)
            actual_warmup_steps = warmup_steps + trial % 5
            for _step in range(actual_warmup_steps):
                state_t = student_obs["state"].to(device).float()
                action_t = torch.clamp(teacher.actor_mean(state_t), action_low, action_high)
                student_obs, _reward, terminated, truncated, _info = student_env.step(
                    action_t.detach().cpu().numpy()[0]
                )
                if bool(_numpy(terminated).reshape(-1)[0]) or bool(
                    _numpy(truncated).reshape(-1)[0]
                ):
                    break

            source_state_dict = student_env.unwrapped.get_state_dict()
            branch_env.unwrapped.set_state_dict(_clone_mani_state_dict(source_state_dict))
            student_obs = student_env.unwrapped.get_obs()
            branch_obs = branch_env.unwrapped.get_obs()

            copied_flat_error = float(
                torch.max(
                    torch.abs(student_env.unwrapped.get_state() - branch_env.unwrapped.get_state())
                )
                .detach()
                .cpu()
            )
            component_errors = _nested_state_max_abs_errors(
                source_state_dict,
                branch_env.unwrapped.get_state_dict(),
            )
            copied_component_error = float(max(component_errors.values()))

            student_action = teacher.actor_mean(student_obs["state"].to(device).float())
            branch_action = teacher.actor_mean(branch_obs["state"].to(device).float())
            teacher_action_error = float(
                torch.max(torch.abs(student_action - branch_action)).detach().cpu()
            )
            action = torch.clamp(student_action, action_low, action_high).detach().cpu().numpy()[0]

            student_rgb, _student_state_obs = _phase4_rgb_state(student_obs)
            branch_rgb, _branch_state_obs = _phase4_rgb_state(branch_obs)
            rgb_pixel_error = float(
                np.max(np.abs(student_rgb.astype(np.int16) - branch_rgb.astype(np.int16)))
            )
            rgbs = np.concatenate([student_rgb, branch_rgb], axis=0)
            features = np.concatenate(
                [dino.encode_batch(chunk) for chunk in batched(rgbs, dino_batch_size)],
                axis=0,
            )
            dino_feature_error = float(np.max(np.abs(features[0] - features[1])))
            proprio = (
                torch.cat([student_obs["state"], branch_obs["state"]], dim=0)
                .detach()
                .cpu()
                .numpy()[:, :21]
                .astype(np.float32)
            )
            frame_inputs = np.concatenate([features, proprio], axis=-1).astype(np.float32)
            z = encoder(torch.from_numpy(frame_norm.transform(frame_inputs)).to(device).float())
            latent_error = float(torch.max(torch.abs(z[0] - z[1])).detach().cpu())

            _student_next, student_reward, _student_term, _student_trunc, _student_info = (
                student_env.step(action)
            )
            _branch_next, branch_reward, _branch_term, _branch_trunc, _branch_info = (
                branch_env.step(action)
            )
            transition_state_error = float(
                torch.max(
                    torch.abs(student_env.unwrapped.get_state() - branch_env.unwrapped.get_state())
                )
                .detach()
                .cpu()
            )
            reward_error = float(abs(_scalar(student_reward) - _scalar(branch_reward)))
            row = {
                "trial": trial,
                "warmup_steps": actual_warmup_steps,
                "copied_flat_state_error": copied_flat_error,
                "copied_component_errors": component_errors,
                "copied_component_error": copied_component_error,
                "teacher_action_error": teacher_action_error,
                "transition_state_error": transition_state_error,
                "reward_error": reward_error,
                "rgb_pixel_error": rgb_pixel_error,
                "dino_feature_error": dino_feature_error,
                "latent_error": latent_error,
            }
            per_trial.append(row)
            for key in max_values:
                max_values[key] = max(max_values[key], float(row[key]))
    finally:
        student_env.close()
        branch_env.close()

    tolerances = {
        "copied_flat_state_error": float(
            config.get("incremental.phase7.branch_state_tolerance", 1e-6)
        ),
        "copied_component_error": float(
            config.get("incremental.phase7.branch_state_tolerance", 1e-6)
        ),
        "teacher_action_error": float(
            config.get("incremental.phase7.branch_action_tolerance", 1e-6)
        ),
        "transition_state_error": float(
            config.get("incremental.phase7.branch_transition_tolerance", 1e-5)
        ),
        "reward_error": float(config.get("incremental.phase7.branch_reward_tolerance", 1e-6)),
        "rgb_pixel_error": float(config.get("incremental.phase7.branch_rgb_tolerance", 0.0)),
        "dino_feature_error": float(config.get("incremental.phase7.branch_dino_tolerance", 1e-6)),
        "latent_error": float(config.get("incremental.phase7.branch_latent_tolerance", 1e-6)),
    }
    gates = {key: max_values[key] <= tolerances[key] for key in tolerances}
    payload = {
        "phase": "7B",
        "method": "branch_state_parity_audit",
        "variant": variant,
        "latent_dim": latent_dim,
        "seed": seed,
        "trials": trials,
        "warmup_steps": warmup_steps,
        "checkpoint": str(encoder_path),
        "max": max_values,
        "tolerances": tolerances,
        "gates": gates,
        "passed": bool(all(gates.values())),
        "per_trial": per_trial,
        "metadata": _runtime_metadata(config),
    }
    write_json(output_path, payload)
    console.print(payload)
    return output_path


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


def _phase7_privileged_yaw(state: np.ndarray) -> np.ndarray:
    quat = state[..., 27:31]
    return (2.0 * np.arctan2(quat[..., 3], quat[..., 0])).astype(np.float32)


def _wrap_angle(angle: np.ndarray) -> np.ndarray:
    return ((angle + np.pi) % (2.0 * np.pi) - np.pi).astype(np.float32)


def _phase7_privileged_goal(
    current_state: np.ndarray,
    future_state: np.ndarray,
    horizon_steps: int,
    control_freq: int,
) -> np.ndarray:
    current = np.asarray(current_state, dtype=np.float32)
    future = np.asarray(future_state, dtype=np.float32)
    if current.shape[-1] != 31 or future.shape[-1] != 31:
        raise ValueError(
            f"Expected 31D PushT privileged state, got {current.shape[-1]} and {future.shape[-1]}"
        )
    dt = float(horizon_steps) / float(control_freq)
    obj_xy = future[..., 24:26]
    yaw = _phase7_privileged_yaw(future)
    obj_vel_xy = (future[..., 24:26] - current[..., 24:26]) / dt
    yaw_rate = _wrap_angle(yaw - _phase7_privileged_yaw(current)) / dt
    tcp_pos = future[..., 14:17]
    tcp_vel = (future[..., 14:17] - current[..., 14:17]) / dt
    contact = (
        np.linalg.norm(future[..., 14:16] - future[..., 24:26], axis=-1, keepdims=True)
        < float(0.08)
    ).astype(np.float32)
    return np.concatenate(
        [
            obj_xy,
            np.sin(yaw)[..., None].astype(np.float32),
            np.cos(yaw)[..., None].astype(np.float32),
            obj_vel_xy.astype(np.float32),
            yaw_rate[..., None].astype(np.float32),
            tcp_pos.astype(np.float32),
            tcp_vel.astype(np.float32),
            contact,
        ],
        axis=-1,
    ).astype(np.float32)


PRE_RL_PHASE_B_GOAL_TYPES = ("full", "robot", "tcp", "object", "object_pose")


def _pre_rl_phase_b_goal(
    current_state: np.ndarray,
    future_state: np.ndarray,
    horizon_steps: int,
    control_freq: int,
    goal_type: str,
) -> np.ndarray:
    if goal_type not in PRE_RL_PHASE_B_GOAL_TYPES:
        raise ValueError(f"Unknown Phase B goal type: {goal_type}")
    current = np.asarray(current_state, dtype=np.float32)
    future = np.asarray(future_state, dtype=np.float32)
    if current.shape[-1] != 31 or future.shape[-1] != 31:
        raise ValueError(
            f"Expected 31D PushT privileged state, got {current.shape[-1]} and "
            f"{future.shape[-1]}"
        )
    dt = float(horizon_steps) / float(control_freq)
    object_xy = future[..., 24:26]
    object_yaw = _phase7_privileged_yaw(future)
    object_velocity = (future[..., 24:26] - current[..., 24:26]) / dt
    object_yaw_rate = _wrap_angle(object_yaw - _phase7_privileged_yaw(current)) / dt
    object_pose = np.concatenate(
        [
            object_xy,
            np.sin(object_yaw)[..., None],
            np.cos(object_yaw)[..., None],
        ],
        axis=-1,
    ).astype(np.float32)
    object_goal = np.concatenate(
        [object_pose, object_velocity, object_yaw_rate[..., None]], axis=-1
    ).astype(np.float32)
    tcp_position = future[..., 14:17]
    tcp_velocity = (future[..., 14:17] - current[..., 14:17]) / dt
    tcp_goal = np.concatenate([tcp_position, tcp_velocity], axis=-1).astype(np.float32)
    joint_goal = future[..., :14].astype(np.float32)
    contact = (
        np.linalg.norm(future[..., 14:16] - future[..., 24:26], axis=-1, keepdims=True)
        < float(0.08)
    ).astype(np.float32)
    if goal_type == "object_pose":
        return object_pose
    if goal_type == "object":
        return object_goal
    if goal_type == "tcp":
        return tcp_goal
    if goal_type == "robot":
        return np.concatenate([tcp_goal, joint_goal], axis=-1).astype(np.float32)
    return np.concatenate(
        [object_goal, tcp_goal, joint_goal, contact], axis=-1
    ).astype(np.float32)


def _phase7_privileged_condition(
    current_state: np.ndarray,
    goal: np.ndarray | None,
    prev_action_norm: np.ndarray,
) -> np.ndarray:
    pieces = [current_state.astype(np.float32)]
    if goal is not None:
        pieces.append(goal.astype(np.float32))
    pieces.append(prev_action_norm.astype(np.float32))
    return np.concatenate(pieces, axis=-1).astype(np.float32)


def _phase7_privileged_checkpoint_dir(config: Config, horizon_steps: int, seed: int) -> Path:
    return ensure_dir(
        config.path_value("paths.incremental_artifact_dir")
        / "phase7"
        / "privileged_branch"
        / f"k{horizon_steps}"
        / f"seed{seed}"
    )


def _phase7_privileged_results_dir(config: Config, horizon_steps: int, seed: int) -> Path:
    return ensure_dir(
        config.path_value("paths.incremental_results_dir")
        / "phase7"
        / "privileged_branch"
        / f"k{horizon_steps}"
        / f"seed{seed}"
    )


def _load_phase7_privileged_episodes(
    config: Config,
    horizon_steps: int,
    *,
    cap_train_to_usable: bool = False,
) -> tuple[list[dict[str, np.ndarray]], list[dict[str, np.ndarray]], dict[str, Any]]:
    dataset_path = collect_phase1_query_dataset(config, force=False)
    subset = str(config.get("incremental.phase7.privileged_subset", "successful"))
    train_episodes = int(config.get("incremental.phase7.privileged_train_episodes", 1800))
    validation_episodes = int(config.get("incremental.phase7.privileged_validation_episodes", 200))
    label_kind = str(
        config.get("incremental.phase7.privileged_label_kind", "deterministic_clipped")
    )
    label_dataset = {
        "deterministic_clipped": "teacher_clipped_actions",
        "deterministic_raw": "teacher_raw_actions",
    }.get(label_kind)
    if label_dataset is None:
        raise ValueError(f"Unknown privileged Phase 7 label kind: {label_kind}")
    with h5py.File(dataset_path, "r") as h5:
        keys = _phase1_episode_keys(h5, subset)
        usable = [key for key in keys if len(h5[key]["teacher_clipped_actions"]) > horizon_steps]
        requested_train_episodes = train_episodes
        required = requested_train_episodes + validation_episodes
        if len(usable) < required and not cap_train_to_usable:
            raise ValueError(
                f"Phase 7 privileged subset '{subset}' has {len(usable)} usable episodes, "
                f"requires {required}"
            )
        train_episodes = min(
            requested_train_episodes,
            len(usable) - validation_episodes,
        )
        if train_episodes <= 0:
            raise ValueError(
                f"Phase 7 privileged subset '{subset}' has {len(usable)} usable episodes, "
                f"which cannot provide {validation_episodes} validation episodes and any "
                "training episodes"
            )
        train_keys = usable[:train_episodes]
        validation_keys = usable[-validation_episodes:]

        def read(keys_to_read: list[str]) -> list[dict[str, np.ndarray]]:
            episodes = []
            for key in keys_to_read:
                episodes.append(
                    {
                        "states": np.asarray(h5[key]["states"], dtype=np.float32),
                        "actions": np.asarray(h5[key][label_dataset], dtype=np.float32),
                    }
                )
            return episodes

        train = read(train_keys)
        validation = read(validation_keys)
    metadata = {
        "dataset_path": str(dataset_path),
        "subset": subset,
        "label_kind": label_kind,
        "train_episodes": train_episodes,
        "requested_train_episodes": requested_train_episodes,
        "validation_episodes": validation_episodes,
        "usable_episodes": len(usable),
    }
    return train, validation, metadata


def _phase7_build_privileged_conditions(
    episodes: list[dict[str, np.ndarray]],
    action_norm: Standardizer,
    horizon_steps: int,
    control_freq: int,
    include_goal: bool,
) -> tuple[np.ndarray, np.ndarray]:
    conditions: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    action_dim = episodes[0]["actions"].shape[-1]
    zero_prev = action_norm.transform(np.zeros((1, action_dim), dtype=np.float32))[0]
    for episode in episodes:
        states = episode["states"]
        episode_actions = episode["actions"]
        prev_actions = action_norm.transform(episode_actions)
        for t in range(len(episode_actions) - horizon_steps):
            prev_action = prev_actions[t - 1] if t > 0 else zero_prev
            goal = (
                _phase7_privileged_goal(
                    states[t], states[t + horizon_steps], horizon_steps, control_freq
                )
                if include_goal
                else None
            )
            conditions.append(_phase7_privileged_condition(states[t], goal, prev_action))
            actions.append(episode_actions[t])
    if not conditions:
        raise ValueError("No Phase 7 privileged samples were produced")
    return np.stack(conditions).astype(np.float32), np.stack(actions).astype(np.float32)


def _pre_rl_phase_b_conditions(
    episodes: list[dict[str, np.ndarray]],
    action_norm: Standardizer,
    horizon_steps: int,
    control_freq: int,
    goal_type: str | None,
) -> tuple[np.ndarray, np.ndarray]:
    conditions: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    action_dim = episodes[0]["actions"].shape[-1]
    zero_prev = action_norm.transform(np.zeros((1, action_dim), dtype=np.float32))[0]
    for episode in episodes:
        states = episode["states"]
        episode_actions = episode["actions"]
        previous_actions = action_norm.transform(episode_actions)
        for t in range(len(episode_actions) - horizon_steps):
            previous = previous_actions[t - 1] if t > 0 else zero_prev
            goal = (
                None
                if goal_type is None
                else _pre_rl_phase_b_goal(
                    states[t],
                    states[t + horizon_steps],
                    horizon_steps,
                    control_freq,
                    goal_type,
                )
            )
            conditions.append(_phase7_privileged_condition(states[t], goal, previous))
            actions.append(episode_actions[t])
    if not conditions:
        raise ValueError(f"No Phase B privileged samples for horizon {horizon_steps}")
    return np.stack(conditions).astype(np.float32), np.stack(actions).astype(np.float32)


def _pre_rl_phase_c_time_conditioned_conditions(
    episodes: list[dict[str, np.ndarray]],
    action_norm: Standardizer,
    horizon_steps: int,
    control_freq: int,
    goal_type: str,
) -> tuple[np.ndarray, np.ndarray]:
    conditions: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    action_dim = episodes[0]["actions"].shape[-1]
    zero_prev = action_norm.transform(np.zeros((1, action_dim), dtype=np.float32))[0]
    for episode in episodes:
        states = episode["states"]
        episode_actions = episode["actions"]
        previous_actions = action_norm.transform(episode_actions)
        for t in range(len(episode_actions) - horizon_steps):
            previous = previous_actions[t - 1] if t > 0 else zero_prev
            for offset in range(1, horizon_steps + 1):
                goal = _pre_rl_phase_b_goal(
                    states[t],
                    states[t + offset],
                    offset,
                    control_freq,
                    goal_type,
                )
                base = _phase7_privileged_condition(states[t], goal, previous)
                conditions.append(
                    np.concatenate(
                        [base, np.asarray([offset / horizon_steps], dtype=np.float32)]
                    )
                )
                actions.append(episode_actions[t])
    if not conditions:
        raise ValueError(f"No Phase C time-conditioned samples for horizon {horizon_steps}")
    return np.stack(conditions).astype(np.float32), np.stack(actions).astype(np.float32)


def _phase7_train_privileged_model(
    config: Config,
    name: str,
    train_cond: np.ndarray,
    train_actions: np.ndarray,
    val_cond: np.ndarray,
    val_actions: np.ndarray,
    action_norm: Standardizer,
    seed: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    set_seed(seed)
    device = default_device()
    cond_norm = Standardizer.fit(train_cond)
    x_train = torch.from_numpy(cond_norm.transform(train_cond)).float()
    y_train = torch.from_numpy(action_norm.transform(train_actions)).float()
    loader = DataLoader(
        TensorDataset(x_train, y_train),
        batch_size=int(config.get("incremental.phase7.privileged_batch_size", 4096)),
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    hidden_dim = int(config.get("incremental.phase7.privileged_hidden_dim", 256))
    model = MLP(train_cond.shape[-1], train_actions.shape[-1], hidden_dim, depth=4).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(config.get("incremental.phase7.privileged_lr", 3e-4))
    )
    x_val = torch.from_numpy(cond_norm.transform(val_cond)).to(device).float()
    y_val = torch.from_numpy(action_norm.transform(val_actions)).to(device).float()
    epochs = int(config.get("incremental.phase7.privileged_epochs", 100))
    best_state = None
    best_val = float("inf")
    history = []
    for epoch in trange(1, epochs + 1, desc=f"train phase7D {name}"):
        model.train()
        train_sum = 0.0
        train_count = 0
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            loss = torch.mean((model(x) - y) ** 2)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            train_sum += float(loss.detach().cpu()) * len(x)
            train_count += len(x)
        model.eval()
        with torch.inference_mode():
            val_mse = float(torch.mean((model(x_val) - y_val) ** 2).cpu())
        history.append(
            {"epoch": epoch, "train_mse": train_sum / train_count, "validation_mse": val_mse}
        )
        if val_mse < best_val:
            best_val = val_mse
            best_state = copy.deepcopy(model.state_dict())
    if best_state is None:
        raise RuntimeError(f"Phase 7D {name} training produced no checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    preds = []
    with torch.inference_mode():
        for start in range(0, len(val_cond), 8192):
            x = (
                torch.from_numpy(cond_norm.transform(val_cond[start : start + 8192]))
                .to(device)
                .float()
            )
            pred_norm = model(x).cpu().numpy()
            preds.append(action_norm.inverse(pred_norm))
    validation_metrics = _action_regression_metrics(np.concatenate(preds), val_actions)
    payload = {
        "model": best_state,
        "cond_norm": cond_norm.state_dict(),
        "cond_dim": int(train_cond.shape[-1]),
        "action_dim": int(train_actions.shape[-1]),
        "hidden_dim": hidden_dim,
        "best_validation_mse": best_val,
        "validation_metrics": validation_metrics,
        "history": history,
    }
    summary = {
        "best_validation_mse": best_val,
        "validation_metrics": validation_metrics,
        "train_samples": int(len(train_cond)),
        "validation_samples": int(len(val_cond)),
    }
    return payload, summary


def train_phase7_privileged_branch_baselines(
    config: Config,
    horizon_steps: int | None = None,
    seed: int = 0,
    force: bool = False,
) -> Path:
    horizon_steps = int(horizon_steps or config.get("incremental.phase7.horizon_steps", 2))
    checkpoint_dir = _phase7_privileged_checkpoint_dir(config, horizon_steps, seed)
    checkpoint_path = checkpoint_dir / "privileged_branch_baselines.pt"
    if checkpoint_path.exists() and not force:
        console.print(f"Phase 7D privileged baselines exist: {checkpoint_path}")
        return checkpoint_path
    train_episodes, val_episodes, data_metadata = _load_phase7_privileged_episodes(
        config, horizon_steps
    )
    train_actions = np.concatenate([episode["actions"] for episode in train_episodes], axis=0)
    action_norm = Standardizer.fit(train_actions)
    control_freq = int(config.get("control_freq", 20))
    flat_train_cond, flat_train_actions = _phase7_build_privileged_conditions(
        train_episodes, action_norm, horizon_steps, control_freq, include_goal=False
    )
    flat_val_cond, flat_val_actions = _phase7_build_privileged_conditions(
        val_episodes, action_norm, horizon_steps, control_freq, include_goal=False
    )
    goal_train_cond, goal_train_actions = _phase7_build_privileged_conditions(
        train_episodes, action_norm, horizon_steps, control_freq, include_goal=True
    )
    goal_val_cond, goal_val_actions = _phase7_build_privileged_conditions(
        val_episodes, action_norm, horizon_steps, control_freq, include_goal=True
    )
    timer = Timer()
    flat_payload, flat_summary = _phase7_train_privileged_model(
        config,
        "flat",
        flat_train_cond,
        flat_train_actions,
        flat_val_cond,
        flat_val_actions,
        action_norm,
        seed,
    )
    goal_payload, goal_summary = _phase7_train_privileged_model(
        config,
        "branch-goal",
        goal_train_cond,
        goal_train_actions,
        goal_val_cond,
        goal_val_actions,
        action_norm,
        seed + 1000,
    )
    payload = {
        "phase": "7D",
        "method": "privileged_flat_and_structured_branch_goal",
        "horizon_steps": horizon_steps,
        "control_freq": control_freq,
        "seed": seed,
        "action_norm": action_norm.state_dict(),
        "flat": flat_payload,
        "branch_goal": goal_payload,
        "data": {
            **data_metadata,
            "flat_train_samples": int(len(flat_train_cond)),
            "flat_validation_samples": int(len(flat_val_cond)),
            "goal_train_samples": int(len(goal_train_cond)),
            "goal_validation_samples": int(len(goal_val_cond)),
        },
        "elapsed_s": timer.elapsed(),
        "metadata": _runtime_metadata(config),
    }
    torch.save(payload, checkpoint_path)
    write_json(
        checkpoint_dir / "privileged_branch_baselines_metrics.json",
        {
            "phase": "7D",
            "horizon_steps": horizon_steps,
            "seed": seed,
            "flat": flat_summary,
            "branch_goal": goal_summary,
            "data": payload["data"],
            "elapsed_s": payload["elapsed_s"],
        },
    )
    console.print(f"Wrote Phase 7D privileged baselines: {checkpoint_path}")
    return checkpoint_path


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
                pred_norm = model(
                    torch.from_numpy(raw_conditions[start : start + 4096]).to(device).float()
                )
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
        "goal_sensitivity_l2": float(
            np.mean(np.linalg.norm(correct_pred - shuffled_pred, axis=-1))
        ),
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


def train_phase7_residual_low_level(
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
    checkpoint_path = artifact_dir / "oracle_low_level_residual.pt"
    if checkpoint_path.exists() and not force:
        console.print(f"Phase 7 residual low-level policy exists: {checkpoint_path}")
        return checkpoint_path

    encoder_path = train_phase6_representation(
        config,
        latent_dim=latent_dim,
        variant=variant,
        seed=seed,
        force=False,
    )
    flat_checkpoint_path = train_phase6_latent_bc(
        config,
        latent_dim=latent_dim,
        variant=variant,
        seed=seed,
        force=False,
    )
    device = default_device()
    encoder, encoder_checkpoint = _load_phase6_encoder(encoder_path, device)
    flat_checkpoint = torch.load(flat_checkpoint_path, map_location=device, weights_only=False)
    if Path(flat_checkpoint["encoder_checkpoint"]) != encoder_path:
        raise ValueError("Residual controller flat policy and goal encoder checkpoints differ")
    frame_norm = Standardizer.from_state_dict(encoder_checkpoint["frame_norm"])
    action_norm = Standardizer.from_state_dict(encoder_checkpoint["action_norm"])
    flat_action_norm = Standardizer.from_state_dict(flat_checkpoint["action_norm"])
    if not (
        np.array_equal(action_norm.mean, flat_action_norm.mean)
        and np.array_equal(action_norm.std, flat_action_norm.std)
    ):
        raise ValueError("Residual controller flat and Phase 7 action normalization differ")

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
        action_norm.transform(train_actions),
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
    action_dim = int(flat_checkpoint["action_dim"])
    flat_model = MLP(
        int(flat_checkpoint["cond_dim"]),
        action_dim,
        int(flat_checkpoint["hidden_dim"]),
        depth=4,
    ).to(device)
    flat_model.load_state_dict(flat_checkpoint["model"])
    residual_model = MLP(
        train_cond.shape[-1],
        action_dim,
        int(config.get("incremental.phase7.hidden_dim", 1024)),
        depth=4,
    ).to(device)
    final_layer = residual_model.net[-1]
    if not isinstance(final_layer, nn.Linear):
        raise TypeError("Expected the residual MLP to end with a linear layer")
    nn.init.zeros_(final_layer.weight)
    nn.init.zeros_(final_layer.bias)
    model = _Phase7ResidualController(flat_model, residual_model, latent_dim, action_dim).to(device)
    optimizer = torch.optim.AdamW(
        residual_model.parameters(),
        lr=float(config.get("incremental.phase7.lr", 3e-4)),
    )
    x_val = torch.from_numpy(val_cond).to(device).float()
    y_val = torch.from_numpy(action_norm.transform(val_actions)).to(device).float()
    epochs = int(config.get("incremental.phase7.epochs", 80))
    best_state = None
    best_val = float("inf")
    history = []
    timer = Timer()
    for epoch in trange(1, epochs + 1, desc="train phase7 residual low level"):
        model.train()
        flat_model.eval()
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
            best_state = copy.deepcopy(residual_model.state_dict())
    if best_state is None:
        raise RuntimeError("Phase 7 residual training produced no checkpoint")
    residual_model.load_state_dict(best_state)
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
    with torch.inference_mode():
        residual_norm = residual_model(x_val).cpu().numpy()
    residual_action = residual_norm * action_norm.std
    residual_metrics = {
        "mean_l2": float(np.mean(np.linalg.norm(residual_action, axis=-1))),
        "mean_abs": float(np.mean(np.abs(residual_action))),
        "max_abs": float(np.max(np.abs(residual_action))),
    }
    payload = {
        "controller_type": "residual",
        "flat_model": flat_model.state_dict(),
        "residual_model": residual_model.state_dict(),
        "variant": variant,
        "latent_dim": latent_dim,
        "horizon_steps": horizon_steps,
        "action_chunk_steps": action_chunk_steps,
        "goal_encoding": goal_encoding,
        "goal_dropout_prob": goal_dropout_prob,
        "cond_dim": train_cond.shape[-1],
        "flat_cond_dim": int(flat_checkpoint["cond_dim"]),
        "hidden_dim": int(config.get("incremental.phase7.hidden_dim", 1024)),
        "flat_hidden_dim": int(flat_checkpoint["hidden_dim"]),
        "action_dim": action_dim,
        "encoder_checkpoint": str(encoder_path),
        "flat_checkpoint": str(flat_checkpoint_path),
        "action_norm": action_norm.state_dict(),
        "validation_metrics": validation_metrics,
        "validation_residual_metrics": residual_metrics,
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
        artifact_dir / "oracle_low_level_residual_metrics.json",
        {
            "variant": variant,
            "latent_dim": latent_dim,
            "horizon_steps": horizon_steps,
            "action_chunk_steps": action_chunk_steps,
            "goal_encoding": goal_encoding,
            "goal_dropout_prob": goal_dropout_prob,
            "validation_metrics": validation_metrics,
            "validation_residual_metrics": residual_metrics,
            "best_validation_mse": best_val,
            "elapsed_s": timer.elapsed(),
        },
    )
    console.print(f"Wrote Phase 7 residual low-level policy: {checkpoint_path}")
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

    result_dir = (
        config.path_value("paths.incremental_results_dir")
        / "phase5"
        / "concat_h1"
        / f"seed{seed}"
    )
    visual_flow_path = result_dir / "visual_flow.json"
    if not visual_flow_path.exists():
        evaluated_paths = sorted(result_dir.glob("visual_flow_eval_seed*_*.json"))
        if len(evaluated_paths) != 1:
            raise FileNotFoundError(
                "Expected visual_flow.json or exactly one seed-specific visual flow result in "
                f"{result_dir}, found {len(evaluated_paths)}"
            )
        visual_flow_path = evaluated_paths[0]
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
    encoder, encoder_checkpoint = _load_phase6_encoder(
        Path(checkpoint["encoder_checkpoint"]), device
    )
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
    oracle_latents = _phase7_encode_oracle_frame_sequences(
        encoder, frame_norm, oracle_frames, device
    )
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


class _Phase7ResidualController(nn.Module):
    def __init__(
        self,
        flat_model: nn.Module,
        residual_model: nn.Module,
        latent_dim: int,
        action_dim: int,
    ) -> None:
        super().__init__()
        self.flat_model = flat_model
        self.residual_model = residual_model
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        for parameter in self.flat_model.parameters():
            parameter.requires_grad_(False)

    def forward(self, condition: torch.Tensor) -> torch.Tensor:
        flat_condition = torch.cat(
            [condition[:, : self.latent_dim], condition[:, -self.action_dim :]],
            dim=-1,
        )
        return self.flat_model(flat_condition) + self.residual_model(condition)


def _load_phase7_low_level_checkpoint(
    path: Path,
    device: torch.device,
) -> tuple[nn.Module, dict[str, Any]]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if checkpoint.get("controller_type") == "residual":
        flat_model = MLP(
            int(checkpoint["flat_cond_dim"]),
            int(checkpoint["action_dim"]),
            int(checkpoint["flat_hidden_dim"]),
            depth=4,
        )
        flat_model.load_state_dict(checkpoint["flat_model"])
        residual_model = MLP(
            int(checkpoint["cond_dim"]),
            int(checkpoint["action_dim"]),
            int(checkpoint["hidden_dim"]),
            depth=4,
        )
        residual_model.load_state_dict(checkpoint["residual_model"])
        model = _Phase7ResidualController(
            flat_model,
            residual_model,
            int(checkpoint["latent_dim"]),
            int(checkpoint["action_dim"]),
        ).to(device)
    else:
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
def evaluate_phase7_matched_flat_latent_policy(
    config: Config,
    latent_dim: int | None = None,
    variant: str | None = None,
    seed: int = 0,
    episodes: int | None = None,
    force: bool = False,
) -> Path:
    latent_dim = int(latent_dim or config.get("incremental.phase7.latent_dim", 256))
    variant = str(variant or config.get("incremental.phase7.variant", "ae_recon"))
    # This evaluator is inference-only after checkpoint preparation. Explicitly
    # re-enable autograd when a matched flat checkpoint must be trained lazily.
    with torch.inference_mode(False):
        checkpoint_path = train_phase6_latent_bc(
            config,
            latent_dim=latent_dim,
            variant=variant,
            seed=seed,
            force=False,
        )
    results_dir = ensure_dir(
        config.path_value("paths.incremental_results_dir")
        / "phase7"
        / "matched_flat"
        / f"{variant}_z{latent_dim}"
        / f"seed{seed}"
    )
    eval_episodes = int(episodes or config.get("incremental.phase7.eval_episodes", 100))
    output_path = results_dir / f"matched_flat_latent_eval_{eval_episodes}.json"
    if output_path.exists() and not force:
        console.print(f"Phase 7E matched flat latent eval exists: {output_path}")
        return output_path

    device = default_device()
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    encoder, encoder_checkpoint = _load_phase6_encoder(
        Path(checkpoint["encoder_checkpoint"]), device
    )
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

    def make_env(num_envs: int):
        return gym.make(
            config.get("env_id"),
            obs_mode="rgb+state",
            control_mode=config.get("control_mode"),
            reward_mode="normalized_dense",
            render_mode=None,
            sim_backend=_rl_backend(config),
            num_envs=num_envs,
            reconfiguration_freq=config.get("rl.eval_reconfiguration_freq", 1),
        )

    zero_action_norm = action_norm.transform(
        np.zeros((1, int(checkpoint["action_dim"])), dtype=np.float32)
    )[0]
    seed_start = int(config.get("incremental.phase7.replay_branch_seed", 1_200_000))
    max_num_envs = min(
        int(config.get("incremental.phase7.replay_branch_num_envs", 16)),
        eval_episodes,
    )
    successes: list[float] = []
    final_rewards: list[float] = []
    max_rewards: list[float] = []
    episode_lengths: list[int] = []
    latencies: list[float] = []
    max_episode_steps = int(config.get("env_max_episode_steps", 100))
    progress = trange(eval_episodes, desc="phase7E matched flat latent eval")
    for batch_start in range(0, eval_episodes, max_num_envs):
        num_envs = min(max_num_envs, eval_episodes - batch_start)
        reset_seeds = [seed_start + batch_start + i for i in range(num_envs)]
        env = make_env(num_envs)
        action_low_np = np.asarray(env.action_space.low, dtype=np.float32)
        action_high_np = np.asarray(env.action_space.high, dtype=np.float32)
        if action_low_np.ndim == 2:
            action_low_np = action_low_np[0]
            action_high_np = action_high_np[0]
        action_low = torch.as_tensor(action_low_np, device=device, dtype=torch.float32)
        action_high = torch.as_tensor(action_high_np, device=device, dtype=torch.float32)
        try:
            obs, _info = env.reset(seed=reset_seeds)
            prev_action_norm = np.repeat(zero_action_norm[None, :], num_envs, axis=0).astype(
                np.float32
            )
            active = np.ones(num_envs, dtype=bool)
            success_once = np.zeros(num_envs, dtype=bool)
            batch_final_rewards = np.zeros(num_envs, dtype=np.float32)
            batch_max_rewards = np.full(num_envs, -np.inf, dtype=np.float32)
            batch_lengths = np.zeros(num_envs, dtype=np.int32)
            for _step in range(max_episode_steps):
                if not active.any():
                    break
                active_count = int(active.sum())
                timer = Timer()
                frames = frame_norm.transform(
                    _phase4_frame_inputs(obs, dino, int(config.get("dino.batch_size", 64)))
                )
                z = encoder(torch.from_numpy(frames).to(device).float())
                prev_action_t = torch.from_numpy(prev_action_norm).to(device).float()
                pred_norm = model(torch.cat([z, prev_action_t], dim=-1))
                raw_action = action_norm.inverse(pred_norm.cpu().numpy()).astype(np.float32)
                latencies.append(timer.elapsed() / active_count)
                action_t = torch.from_numpy(raw_action).to(device).float()
                if bool(config.get("policy.clip_actions_to_env_space", True)):
                    action_t = torch.clamp(action_t, action_low, action_high)
                action_t[~torch.from_numpy(active).to(device)] = 0.0
                obs, reward, terminated, truncated, info = env.step(action_t)
                prev_action_norm = action_norm.transform(
                    action_t.detach().cpu().numpy().astype(np.float32)
                )
                reward_np = _numpy(reward).reshape(-1).astype(np.float32)
                batch_final_rewards[active] = reward_np[active]
                batch_max_rewards[active] = np.maximum(batch_max_rewards[active], reward_np[active])
                batch_lengths[active] += 1
                if "success" in info:
                    success_once |= _numpy(info["success"]).reshape(-1).astype(bool)
                done = _numpy(torch.logical_or(terminated, truncated)).reshape(-1).astype(bool)
                newly_done = active & done
                if np.any(newly_done):
                    progress.update(int(newly_done.sum()))
                    active[newly_done] = False
            if np.any(active):
                progress.update(int(active.sum()))
            successes.extend(float(x) for x in success_once)
            final_rewards.extend(float(x) for x in batch_final_rewards)
            max_rewards.extend(float(x) for x in batch_max_rewards)
            episode_lengths.extend(int(x) for x in batch_lengths)
        finally:
            env.close()
    progress.close()
    metrics = {
        "success": float(np.mean(successes)),
        "success_stderr": float(np.std(successes) / np.sqrt(len(successes))),
        "final_reward": float(np.mean(final_rewards)),
        "max_reward": float(np.mean(max_rewards)),
        "mean_episode_length": float(np.mean(episode_lengths)),
        "inference_latency_s": float(np.mean(latencies)),
        "episodes": eval_episodes,
        "num_envs": max_num_envs,
        "seed_start": seed_start,
    }
    payload = {
        "phase": "7E",
        "method": "matched_flat_latent_policy",
        "variant": variant,
        "latent_dim": latent_dim,
        "seed": seed,
        "checkpoint": str(checkpoint_path),
        "closed_loop": metrics,
        "held_out_action_metrics": checkpoint["validation_metrics"],
        "metadata": _runtime_metadata(config),
    }
    write_json(output_path, payload)
    console.print(payload)
    return output_path


def evaluate_phase7_replay_branch_oracle_low_level(
    config: Config,
    latent_dim: int | None = None,
    variant: str | None = None,
    horizon_steps: int | None = None,
    action_chunk_steps: int | None = None,
    goal_encoding: str | None = None,
    goal_dropout_prob: float | None = None,
    seed: int = 0,
    episodes: int | None = None,
    dagger_iteration: int | None = None,
    dagger_query_episodes: int | None = None,
    residual: bool = False,
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
    if action_chunk_steps != 1:
        raise NotImplementedError("Replay branch oracle evaluation currently supports H=1")
    if residual and dagger_iteration is not None:
        raise ValueError("Residual and DAgger checkpoints are separate Phase 7 evaluations")
    if residual:
        checkpoint_path = train_phase7_residual_low_level(
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
        checkpoint_label = "residual"
    elif dagger_iteration is None:
        checkpoint_path = train_phase7_oracle_low_level(
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
        checkpoint_label = "base"
    else:
        checkpoint_path = train_phase7_oracle_dagger_low_level(
            config,
            latent_dim=latent_dim,
            variant=variant,
            horizon_steps=horizon_steps,
            action_chunk_steps=action_chunk_steps,
            goal_encoding=goal_encoding,
            goal_dropout_prob=goal_dropout_prob,
            iteration=dagger_iteration,
            seed=seed,
            query_episodes=dagger_query_episodes,
            force=False,
        )
        query_episode_count = int(
            dagger_query_episodes or config.get("incremental.phase7.dagger_episodes", 200)
        )
        checkpoint_label = f"branch_dagger_iter{dagger_iteration}_e{query_episode_count}"
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
    eval_episodes = int(
        episodes or config.get("incremental.phase7.replay_branch_eval_episodes", 10)
    )
    output_name = (
        f"replay_branch_oracle_eval_{eval_episodes}.json"
        if checkpoint_label == "base"
        else f"replay_branch_oracle_eval_{checkpoint_label}_{eval_episodes}.json"
    )
    output_path = results_dir / output_name
    if output_path.exists() and not force:
        console.print(f"Phase 7 replay branch oracle eval exists: {output_path}")
        return output_path

    device = default_device()
    model, checkpoint = _load_phase7_low_level_checkpoint(checkpoint_path, device)
    encoder, encoder_checkpoint = _load_phase6_encoder(
        Path(checkpoint["encoder_checkpoint"]), device
    )
    frame_norm = Standardizer.from_state_dict(encoder_checkpoint["frame_norm"])
    action_norm = Standardizer.from_state_dict(checkpoint["action_norm"])
    dino = _phase4_dino_from_config(config, device)
    teacher = load_ppo_agent(_rl_paths(config).best, device)

    def make_env(num_envs: int):
        return gym.make(
            config.get("env_id"),
            obs_mode="rgb+state",
            control_mode=config.get("control_mode"),
            reward_mode="normalized_dense",
            render_mode=None,
            sim_backend=_rl_backend(config),
            num_envs=num_envs,
            reconfiguration_freq=config.get("rl.eval_reconfiguration_freq", 1),
        )

    zero_action_norm = action_norm.transform(
        np.zeros((1, int(checkpoint["action_dim"])), dtype=np.float32)
    )[0]
    seed_start = int(config.get("incremental.phase7.replay_branch_seed", 1_200_000))
    max_num_envs = min(
        int(config.get("incremental.phase7.replay_branch_num_envs", 16)),
        eval_episodes,
    )
    replay_state_tolerance = float(
        config.get("incremental.phase7.replay_branch_state_tolerance", 1e-6)
    )
    successes: list[float] = []
    final_rewards: list[float] = []
    max_rewards: list[float] = []
    episode_lengths: list[int] = []
    replay_errors: list[float] = []
    current_to_goal_l2: list[float] = []
    action_maes: list[float] = []
    policy_latencies: list[float] = []
    branch_latencies: list[float] = []
    branch_latencies_per_env: list[float] = []
    failed_replay_steps = 0
    max_episode_steps = int(config.get("env_max_episode_steps", 100))
    progress = trange(eval_episodes, desc="phase7 batched replay branch oracle eval")
    for batch_start in range(0, eval_episodes, max_num_envs):
        num_envs = min(max_num_envs, eval_episodes - batch_start)
        reset_seeds = [seed_start + batch_start + i for i in range(num_envs)]
        student_env = make_env(num_envs)
        branch_env = make_env(num_envs)
        action_low_np = np.asarray(student_env.action_space.low, dtype=np.float32)
        action_high_np = np.asarray(student_env.action_space.high, dtype=np.float32)
        if action_low_np.ndim == 2:
            action_low_np = action_low_np[0]
            action_high_np = action_high_np[0]
        action_low = torch.as_tensor(action_low_np, device=device, dtype=torch.float32)
        action_high = torch.as_tensor(action_high_np, device=device, dtype=torch.float32)
        try:
            obs, _info = student_env.reset(seed=reset_seeds)
            history: list[torch.Tensor] = []
            prev_action_norm = np.repeat(zero_action_norm[None, :], num_envs, axis=0).astype(
                np.float32
            )
            active = np.ones(num_envs, dtype=bool)
            success_once = np.zeros(num_envs, dtype=bool)
            batch_final_rewards = np.zeros(num_envs, dtype=np.float32)
            batch_max_rewards = np.full(num_envs, -np.inf, dtype=np.float32)
            batch_lengths = np.zeros(num_envs, dtype=np.int32)
            for _step in range(max_episode_steps):
                if not active.any():
                    break
                active_count = int(active.sum())
                branch_timer = Timer()
                branch_obs, _branch_info = branch_env.reset(seed=reset_seeds)
                replay_done = torch.zeros(num_envs, device=device, dtype=torch.bool)
                for action_history in history:
                    branch_obs, _branch_reward, branch_term, branch_trunc, _branch_info = (
                        branch_env.step(action_history)
                    )
                    replay_done = replay_done | torch.logical_or(branch_term, branch_trunc).view(-1)
                state_errors = torch.max(
                    torch.abs(student_env.unwrapped.get_state() - branch_env.unwrapped.get_state()),
                    dim=1,
                ).values
                active_state_errors = state_errors.detach().cpu().numpy()[active]
                replay_errors.extend(float(x) for x in active_state_errors)
                replay_done_np = replay_done.detach().cpu().numpy().astype(bool)
                failed_replay_steps += int(
                    np.sum(
                        active
                        & (
                            replay_done_np
                            | (state_errors.detach().cpu().numpy() > replay_state_tolerance)
                        )
                    )
                )

                for _ in range(horizon_steps):
                    teacher_action = torch.clamp(
                        teacher.actor_mean(branch_obs["state"].to(device).float()),
                        action_low,
                        action_high,
                    ).detach()
                    branch_obs, _branch_reward, branch_term, branch_trunc, _branch_info = (
                        branch_env.step(teacher_action)
                    )
                    if bool(torch.all(torch.logical_or(branch_term, branch_trunc))):
                        break
                branch_elapsed = branch_timer.elapsed()
                branch_latencies.append(branch_elapsed)
                branch_latencies_per_env.append(branch_elapsed / active_count)

                policy_timer = Timer()
                current_frame = _phase4_frame_inputs(
                    obs,
                    dino,
                    int(config.get("dino.batch_size", 64)),
                )
                goal_frame = _phase4_frame_inputs(
                    branch_obs,
                    dino,
                    int(config.get("dino.batch_size", 64)),
                )
                frames = frame_norm.transform(np.concatenate([current_frame, goal_frame], axis=0))
                z_pair = encoder(torch.from_numpy(frames).to(device).float()).detach().cpu().numpy()
                z = z_pair[:num_envs].astype(np.float32)
                goals = z_pair[num_envs:].astype(np.float32)
                current_to_goal_l2.extend(
                    np.linalg.norm(z[active] - goals[active], axis=-1).tolist()
                )
                cond = np.stack(
                    [
                        _phase7_condition(z[i], goals[i], prev_action_norm[i], goal_encoding)
                        for i in range(num_envs)
                    ],
                    axis=0,
                )
                pred_norm = model(torch.from_numpy(cond).to(device).float()).detach()
                raw_action = action_norm.inverse(pred_norm.cpu().numpy()).astype(np.float32)
                teacher_now = (
                    torch.clamp(
                        teacher.actor_mean(obs["state"].to(device).float()),
                        action_low,
                        action_high,
                    )
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(np.float32)
                )
                action_maes.extend(
                    np.mean(np.abs(raw_action[active] - teacher_now[active]), axis=-1).tolist()
                )
                policy_latencies.append(policy_timer.elapsed() / active_count)

                action_t = torch.from_numpy(raw_action).to(device).float()
                if bool(config.get("policy.clip_actions_to_env_space", True)):
                    action_t = torch.clamp(action_t, action_low, action_high)
                action_t[~torch.from_numpy(active).to(device)] = 0.0
                obs, reward, terminated, truncated, info = student_env.step(action_t)
                history.append(action_t.detach().clone())
                action_np = action_t.detach().cpu().numpy().astype(np.float32)
                prev_action_norm = action_norm.transform(action_np)
                reward_np = _numpy(reward).reshape(-1).astype(np.float32)
                batch_final_rewards[active] = reward_np[active]
                batch_max_rewards[active] = np.maximum(batch_max_rewards[active], reward_np[active])
                batch_lengths[active] += 1
                if "success" in info:
                    success_once |= _numpy(info["success"]).reshape(-1).astype(bool)
                done = _numpy(torch.logical_or(terminated, truncated)).reshape(-1).astype(bool)
                newly_done = active & done
                if np.any(newly_done):
                    completed = int(newly_done.sum())
                    progress.update(completed)
                    active[newly_done] = False
            still_active = np.flatnonzero(active)
            if len(still_active) > 0:
                progress.update(len(still_active))
            successes.extend(float(x) for x in success_once)
            final_rewards.extend(float(x) for x in batch_final_rewards)
            max_rewards.extend(float(x) for x in batch_max_rewards)
            episode_lengths.extend(int(x) for x in batch_lengths)
        finally:
            student_env.close()
            branch_env.close()
    progress.close()

    metrics = {
        "success": float(np.mean(successes)),
        "success_stderr": float(np.std(successes) / np.sqrt(len(successes))),
        "final_reward": float(np.mean(final_rewards)),
        "max_reward": float(np.mean(max_rewards)),
        "mean_episode_length": float(np.mean(episode_lengths)),
        "replay_current_state_error_mean": float(np.mean(replay_errors)) if replay_errors else 0.0,
        "replay_current_state_error_max": float(np.max(replay_errors)) if replay_errors else 0.0,
        "replay_failed_step_fraction": float(failed_replay_steps / max(1, len(replay_errors))),
        "latent_current_to_branch_goal_l2": (
            float(np.mean(current_to_goal_l2)) if current_to_goal_l2 else 0.0
        ),
        "teacher_action_mae": float(np.mean(action_maes)) if action_maes else 0.0,
        "policy_latency_s": float(np.mean(policy_latencies)) if policy_latencies else 0.0,
        "branch_generation_latency_s": float(np.mean(branch_latencies))
        if branch_latencies
        else 0.0,
        "branch_generation_latency_per_env_s": (
            float(np.mean(branch_latencies_per_env)) if branch_latencies_per_env else 0.0
        ),
        "episodes": eval_episodes,
        "num_envs": max_num_envs,
        "seed_start": seed_start,
    }
    visual_success = _phase7_visual_flow_success(config, seed)
    payload = {
        "phase": "7C",
        "method": "batched_replay_branch_oracle_low_level_eval",
        "variant": variant,
        "latent_dim": latent_dim,
        "horizon_steps": horizon_steps,
        "action_chunk_steps": action_chunk_steps,
        "goal_encoding": goal_encoding,
        "goal_dropout_prob": goal_dropout_prob,
        "dagger_iteration": dagger_iteration,
        "controller_type": "residual" if residual else "monolithic",
        "seed": seed,
        "checkpoint": str(checkpoint_path),
        "closed_loop": metrics,
        "validation_action_metrics": checkpoint["validation_metrics"],
        "direct_visual_flow_success": visual_success,
        "matched_replay_state_gate": metrics["replay_failed_step_fraction"] == 0.0,
        "oracle_gate_visual_flow": bool(metrics["success"] >= visual_success),
        "oracle_gate_90pct_visual_flow": bool(metrics["success"] >= 0.9 * visual_success),
        "metadata": _runtime_metadata(config),
    }
    write_json(output_path, payload)
    console.print(payload)
    return output_path


def _phase7_checkpoint_for_eval(
    config: Config,
    latent_dim: int,
    variant: str,
    horizon_steps: int,
    action_chunk_steps: int,
    goal_encoding: str,
    goal_dropout_prob: float,
    seed: int,
    dagger_iteration: int | None,
    dagger_query_episodes: int | None,
) -> tuple[Path, str]:
    if dagger_iteration is None:
        checkpoint_path = train_phase7_oracle_low_level(
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
        return checkpoint_path, "base"
    checkpoint_path = train_phase7_oracle_dagger_low_level(
        config,
        latent_dim=latent_dim,
        variant=variant,
        horizon_steps=horizon_steps,
        action_chunk_steps=action_chunk_steps,
        goal_encoding=goal_encoding,
        goal_dropout_prob=goal_dropout_prob,
        iteration=dagger_iteration,
        seed=seed,
        query_episodes=dagger_query_episodes,
        force=False,
    )
    query_episode_count = int(
        dagger_query_episodes or config.get("incremental.phase7.dagger_episodes", 200)
    )
    return checkpoint_path, f"branch_dagger_iter{dagger_iteration}_e{query_episode_count}"


@torch.inference_mode()
def evaluate_phase7_valid_goal_use(
    config: Config,
    latent_dim: int | None = None,
    variant: str | None = None,
    horizon_steps: int | None = None,
    action_chunk_steps: int | None = None,
    goal_encoding: str | None = None,
    goal_dropout_prob: float | None = None,
    seed: int = 0,
    episodes: int | None = None,
    dagger_iteration: int | None = None,
    dagger_query_episodes: int | None = None,
    counterfactual_queries: int = 0,
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
    if action_chunk_steps != 1:
        raise NotImplementedError("Phase 7G currently supports H=1")
    horizons = sorted({max(1, horizon_steps - 1), horizon_steps, horizon_steps + 1})
    checkpoint_path, checkpoint_label = _phase7_checkpoint_for_eval(
        config,
        latent_dim,
        variant,
        horizon_steps,
        action_chunk_steps,
        goal_encoding,
        goal_dropout_prob,
        seed,
        dagger_iteration,
        dagger_query_episodes,
    )
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
    eval_episodes = int(episodes or config.get("incremental.phase7.goal_use_eval_episodes", 10))
    counterfactual_suffix = f"_cf{counterfactual_queries}" if counterfactual_queries > 0 else ""
    output_path = (
        results_dir
        / f"valid_goal_use_{checkpoint_label}_{eval_episodes}{counterfactual_suffix}.json"
    )
    if output_path.exists() and not force:
        console.print(f"Phase 7G valid goal-use eval exists: {output_path}")
        return output_path

    device = default_device()
    model, checkpoint = _load_phase7_low_level_checkpoint(checkpoint_path, device)
    encoder, encoder_checkpoint = _load_phase6_encoder(
        Path(checkpoint["encoder_checkpoint"]), device
    )
    frame_norm = Standardizer.from_state_dict(encoder_checkpoint["frame_norm"])
    action_norm = Standardizer.from_state_dict(checkpoint["action_norm"])
    dino = _phase4_dino_from_config(config, device)
    teacher = load_ppo_agent(_rl_paths(config).best, device)

    def make_env(num_envs: int):
        return gym.make(
            config.get("env_id"),
            obs_mode="rgb+state",
            control_mode=config.get("control_mode"),
            reward_mode="normalized_dense",
            render_mode=None,
            sim_backend=_rl_backend(config),
            num_envs=num_envs,
            reconfiguration_freq=config.get("rl.eval_reconfiguration_freq", 1),
        )

    zero_action_norm = action_norm.transform(
        np.zeros((1, int(checkpoint["action_dim"])), dtype=np.float32)
    )[0]
    seed_start = int(config.get("incremental.phase7.replay_branch_seed", 1_200_000))
    max_num_envs = min(
        int(config.get("incremental.phase7.replay_branch_num_envs", 16)),
        eval_episodes,
    )
    replay_state_tolerance = float(
        config.get("incremental.phase7.replay_branch_state_tolerance", 1e-6)
    )
    max_episode_steps = int(config.get("env_max_episode_steps", 100))
    max_steps_per_episode = int(
        config.get("incremental.phase7.goal_use_max_steps_per_episode", max_episode_steps)
    )
    counterfactual_stride = int(config.get("incremental.phase7.counterfactual_query_stride", 10))
    if counterfactual_queries < 0:
        raise ValueError("counterfactual_queries must be non-negative")
    max_horizon = max(horizons)
    replay_errors: list[float] = []
    failed_replay_steps = 0
    action_l2_by_pair: dict[str, list[float]] = {
        f"{a}_to_{b}": [] for a, b in zip(horizons[:-1], horizons[1:])
    }
    latent_l2_by_pair: dict[str, list[float]] = {
        f"{a}_to_{b}": [] for a, b in zip(horizons[:-1], horizons[1:])
    }
    tcp_l2_by_pair: dict[str, list[float]] = {
        f"{a}_to_{b}": [] for a, b in zip(horizons[:-1], horizons[1:])
    }
    action_l2_near_far: list[float] = []
    latent_l2_near_far: list[float] = []
    tcp_l2_near_far: list[float] = []
    directional_cosines: list[float] = []
    positive_directional = []
    farther_projection_ge_near = []
    teacher_action_maes: list[float] = []
    branch_latencies_per_env: list[float] = []
    counterfactual_replay_errors: list[float] = []
    counterfactual_initial_latent_errors: list[float] = []
    counterfactual_final_latent_errors: list[float] = []
    counterfactual_initial_object_position_errors: list[float] = []
    counterfactual_final_object_position_errors: list[float] = []
    counterfactual_initial_yaw_errors: list[float] = []
    counterfactual_final_yaw_errors: list[float] = []
    counterfactual_initial_tcp_errors: list[float] = []
    counterfactual_final_tcp_errors: list[float] = []
    counterfactual_latent_closest: list[float] = []
    counterfactual_physical_closest: list[float] = []
    counterfactual_done_before_goal: list[float] = []
    counterfactual_samples = 0
    sample_count = 0
    successes: list[float] = []
    progress = trange(eval_episodes, desc="phase7G valid reachable goal-use")
    for batch_start in range(0, eval_episodes, max_num_envs):
        num_envs = min(max_num_envs, eval_episodes - batch_start)
        reset_seeds = [seed_start + batch_start + i for i in range(num_envs)]
        student_env = make_env(num_envs)
        branch_env = make_env(num_envs)
        action_low_np = np.asarray(student_env.action_space.low, dtype=np.float32)
        action_high_np = np.asarray(student_env.action_space.high, dtype=np.float32)
        if action_low_np.ndim == 2:
            action_low_np = action_low_np[0]
            action_high_np = action_high_np[0]
        action_low = torch.as_tensor(action_low_np, device=device, dtype=torch.float32)
        action_high = torch.as_tensor(action_high_np, device=device, dtype=torch.float32)
        try:
            obs, _info = student_env.reset(seed=reset_seeds)
            history: list[torch.Tensor] = []
            prev_action_norm = np.repeat(zero_action_norm[None, :], num_envs, axis=0).astype(
                np.float32
            )
            active = np.ones(num_envs, dtype=bool)
            success_once = np.zeros(num_envs, dtype=bool)
            for _step in range(min(max_episode_steps, max_steps_per_episode)):
                if not active.any():
                    break
                active_count = int(active.sum())
                branch_timer = Timer()
                branch_obs, _branch_info = branch_env.reset(seed=reset_seeds)
                replay_done = torch.zeros(num_envs, device=device, dtype=torch.bool)
                for action_history in history:
                    branch_obs, _branch_reward, branch_term, branch_trunc, _branch_info = (
                        branch_env.step(action_history)
                    )
                    replay_done = replay_done | torch.logical_or(branch_term, branch_trunc).view(-1)
                state_errors = torch.max(
                    torch.abs(student_env.unwrapped.get_state() - branch_env.unwrapped.get_state()),
                    dim=1,
                ).values
                state_errors_np = state_errors.detach().cpu().numpy()
                replay_errors.extend(float(x) for x in state_errors_np[active])
                replay_done_np = replay_done.detach().cpu().numpy().astype(bool)
                failed_replay_steps += int(
                    np.sum(active & (replay_done_np | (state_errors_np > replay_state_tolerance)))
                )

                branch_frames: dict[int, np.ndarray] = {}
                branch_states: dict[int, np.ndarray] = {}
                for step_idx in range(1, max_horizon + 1):
                    teacher_action = torch.clamp(
                        teacher.actor_mean(branch_obs["state"].to(device).float()),
                        action_low,
                        action_high,
                    )
                    branch_obs, _branch_reward, branch_term, branch_trunc, _branch_info = (
                        branch_env.step(teacher_action)
                    )
                    if step_idx in horizons:
                        branch_frames[step_idx] = _phase4_frame_inputs(
                            branch_obs,
                            dino,
                            int(config.get("dino.batch_size", 64)),
                        )
                        branch_states[step_idx] = (
                            branch_obs["state"].detach().cpu().numpy().astype(np.float32)
                        )
                    if bool(torch.all(torch.logical_or(branch_term, branch_trunc))):
                        break
                branch_latencies_per_env.append(branch_timer.elapsed() / active_count)
                if set(branch_frames) != set(horizons):
                    break

                current_frame = _phase4_frame_inputs(
                    obs, dino, int(config.get("dino.batch_size", 64))
                )
                all_frames = [current_frame] + [branch_frames[h] for h in horizons]
                frames = frame_norm.transform(np.concatenate(all_frames, axis=0))
                z_all = encoder(torch.from_numpy(frames).to(device).float()).detach().cpu().numpy()
                z = z_all[:num_envs].astype(np.float32)
                goals = {
                    h: z_all[(i + 1) * num_envs : (i + 2) * num_envs].astype(np.float32)
                    for i, h in enumerate(horizons)
                }
                actions_by_h: dict[int, np.ndarray] = {}
                for h in horizons:
                    cond = np.stack(
                        [
                            _phase7_condition(z[i], goals[h][i], prev_action_norm[i], goal_encoding)
                            for i in range(num_envs)
                        ],
                        axis=0,
                    )
                    pred_norm = model(torch.from_numpy(cond).to(device).float()).detach()
                    actions_by_h[h] = action_norm.inverse(pred_norm.cpu().numpy()).astype(
                        np.float32
                    )

                current_state = obs["state"].detach().cpu().numpy().astype(np.float32)
                if (
                    counterfactual_samples < counterfactual_queries
                    and _step % counterfactual_stride == 0
                ):
                    remaining = counterfactual_queries - counterfactual_samples
                    query_indices = np.flatnonzero(active)[:remaining]
                    query_count = len(query_indices)
                    if query_count > 0:
                        target_count = len(horizons)
                        target_goals = np.stack(
                            [goals[h][env_idx] for env_idx in query_indices for h in horizons],
                            axis=0,
                        ).astype(np.float32)
                        target_states = np.stack(
                            [
                                branch_states[h][env_idx]
                                for env_idx in query_indices
                                for h in horizons
                            ],
                            axis=0,
                        ).astype(np.float32)
                        initial_states = current_state[query_indices].repeat(target_count, axis=0)
                        final_states = np.zeros_like(target_states)
                        final_latents = np.zeros_like(target_goals)
                        captured = np.zeros(len(target_goals), dtype=bool)
                        done_before_capture = np.zeros(len(target_goals), dtype=bool)

                        for target_idx, target_horizon in enumerate(horizons):
                            counter_env = make_env(num_envs)
                            try:
                                counter_obs, _counter_info = counter_env.reset(seed=reset_seeds)
                                counter_done = torch.zeros(
                                    num_envs,
                                    device=device,
                                    dtype=torch.bool,
                                )
                                for action_history in history:
                                    (
                                        counter_obs,
                                        _counter_reward,
                                        counter_term,
                                        counter_trunc,
                                        _counter_info,
                                    ) = counter_env.step(action_history)
                                    counter_done |= torch.logical_or(
                                        counter_term, counter_trunc
                                    ).view(-1)
                                counter_replay_error = torch.max(
                                    torch.abs(
                                        student_env.unwrapped.get_state()
                                        - counter_env.unwrapped.get_state()
                                    ),
                                    dim=1,
                                ).values
                                selected_replay_error = counter_replay_error[query_indices]
                                counterfactual_replay_errors.extend(
                                    selected_replay_error.cpu().numpy().tolist()
                                )
                                if bool(torch.any(selected_replay_error > replay_state_tolerance)):
                                    raise RuntimeError(
                                        "Counterfactual replay did not reproduce the learner state: "
                                        f"errors={selected_replay_error.cpu().numpy().tolist()}"
                                    )

                                counter_prev_action_norm = prev_action_norm.copy()
                                counter_frames = frame_norm.transform(
                                    _phase4_frame_inputs(
                                        counter_obs,
                                        dino,
                                        int(config.get("dino.batch_size", 64)),
                                    )
                                )
                                counter_z = (
                                    encoder(torch.from_numpy(counter_frames).to(device).float())
                                    .detach()
                                    .cpu()
                                    .numpy()
                                    .astype(np.float32)
                                )
                                for _counter_step in range(target_horizon):
                                    counter_cond = np.stack(
                                        [
                                            _phase7_condition(
                                                counter_z[i],
                                                goals[target_horizon][i],
                                                counter_prev_action_norm[i],
                                                goal_encoding,
                                            )
                                            for i in range(num_envs)
                                        ],
                                        axis=0,
                                    )
                                    counter_pred_norm = model(
                                        torch.from_numpy(counter_cond).to(device).float()
                                    ).detach()
                                    counter_action_np = action_norm.inverse(
                                        counter_pred_norm.cpu().numpy()
                                    ).astype(np.float32)
                                    counter_action = (
                                        torch.from_numpy(counter_action_np).to(device).float()
                                    )
                                    if bool(config.get("policy.clip_actions_to_env_space", True)):
                                        counter_action = torch.clamp(
                                            counter_action, action_low, action_high
                                        )
                                    counter_action[counter_done] = 0.0
                                    (
                                        counter_obs,
                                        _counter_reward,
                                        counter_term,
                                        counter_trunc,
                                        _counter_info,
                                    ) = counter_env.step(counter_action)
                                    counter_prev_action_norm = action_norm.transform(
                                        counter_action.cpu().numpy().astype(np.float32)
                                    )
                                    counter_done |= torch.logical_or(
                                        counter_term, counter_trunc
                                    ).view(-1)
                                    counter_frames = frame_norm.transform(
                                        _phase4_frame_inputs(
                                            counter_obs,
                                            dino,
                                            int(config.get("dino.batch_size", 64)),
                                        )
                                    )
                                    counter_z = (
                                        encoder(torch.from_numpy(counter_frames).to(device).float())
                                        .detach()
                                        .cpu()
                                        .numpy()
                                        .astype(np.float32)
                                    )

                                flat_indices = (
                                    np.arange(query_count, dtype=np.int32) * target_count
                                    + target_idx
                                )
                                final_states[flat_indices] = (
                                    counter_obs["state"]
                                    .detach()
                                    .cpu()
                                    .numpy()
                                    .astype(np.float32)[query_indices]
                                )
                                final_latents[flat_indices] = counter_z[query_indices]
                                captured[flat_indices] = True
                                done_before_capture[flat_indices] = (
                                    counter_done[query_indices].cpu().numpy()
                                )
                            finally:
                                counter_env.close()
                        valid = captured & ~done_before_capture
                        counterfactual_done_before_goal.extend(
                            done_before_capture.astype(np.float32).tolist()
                        )
                        if np.any(valid):
                            initial_latent_error = np.linalg.norm(
                                z[query_indices].repeat(target_count, axis=0) - target_goals,
                                axis=-1,
                            )
                            final_latent_error = np.linalg.norm(
                                final_latents - target_goals, axis=-1
                            )
                            initial_object_error = np.linalg.norm(
                                initial_states[:, 24:26] - target_states[:, 24:26],
                                axis=-1,
                            )
                            final_object_error = np.linalg.norm(
                                final_states[:, 24:26] - target_states[:, 24:26],
                                axis=-1,
                            )
                            initial_yaw_error = np.abs(
                                _wrap_angle(
                                    _phase7_privileged_yaw(initial_states)
                                    - _phase7_privileged_yaw(target_states)
                                )
                            )
                            final_yaw_error = np.abs(
                                _wrap_angle(
                                    _phase7_privileged_yaw(final_states)
                                    - _phase7_privileged_yaw(target_states)
                                )
                            )
                            initial_tcp_error = np.linalg.norm(
                                initial_states[:, 14:17] - target_states[:, 14:17], axis=-1
                            )
                            final_tcp_error = np.linalg.norm(
                                final_states[:, 14:17] - target_states[:, 14:17], axis=-1
                            )
                            counterfactual_initial_latent_errors.extend(
                                initial_latent_error[valid].tolist()
                            )
                            counterfactual_final_latent_errors.extend(
                                final_latent_error[valid].tolist()
                            )
                            counterfactual_initial_object_position_errors.extend(
                                initial_object_error[valid].tolist()
                            )
                            counterfactual_final_object_position_errors.extend(
                                final_object_error[valid].tolist()
                            )
                            counterfactual_initial_yaw_errors.extend(
                                initial_yaw_error[valid].tolist()
                            )
                            counterfactual_final_yaw_errors.extend(final_yaw_error[valid].tolist())
                            counterfactual_initial_tcp_errors.extend(
                                initial_tcp_error[valid].tolist()
                            )
                            counterfactual_final_tcp_errors.extend(final_tcp_error[valid].tolist())

                            for local_idx in range(query_count):
                                row = slice(
                                    local_idx * target_count,
                                    (local_idx + 1) * target_count,
                                )
                                local_goals = target_goals[row]
                                local_targets = target_states[row]
                                for target_idx in range(target_count):
                                    flat_idx = local_idx * target_count + target_idx
                                    if not valid[flat_idx]:
                                        continue
                                    latent_distances = np.linalg.norm(
                                        local_goals - final_latents[flat_idx], axis=-1
                                    )
                                    counterfactual_latent_closest.append(
                                        float(np.argmin(latent_distances) == target_idx)
                                    )
                                    object_distances = np.linalg.norm(
                                        local_targets[:, 24:26] - final_states[flat_idx, 24:26],
                                        axis=-1,
                                    )
                                    yaw_distances = np.abs(
                                        _wrap_angle(
                                            _phase7_privileged_yaw(local_targets)
                                            - _phase7_privileged_yaw(
                                                final_states[flat_idx : flat_idx + 1]
                                            )[0]
                                        )
                                    )
                                    tcp_distances = np.linalg.norm(
                                        local_targets[:, 14:17] - final_states[flat_idx, 14:17],
                                        axis=-1,
                                    )
                                    physical_distances = (
                                        object_distances + 0.1 * yaw_distances + tcp_distances
                                    )
                                    counterfactual_physical_closest.append(
                                        float(np.argmin(physical_distances) == target_idx)
                                    )
                        counterfactual_samples += query_count
                tcp_by_h = {h: branch_states[h][:, 14:17] for h in horizons}
                for near_h, far_h in zip(horizons[:-1], horizons[1:]):
                    key = f"{near_h}_to_{far_h}"
                    action_delta = actions_by_h[far_h] - actions_by_h[near_h]
                    latent_delta = goals[far_h] - goals[near_h]
                    tcp_delta = tcp_by_h[far_h] - tcp_by_h[near_h]
                    action_l2_by_pair[key].extend(
                        np.linalg.norm(action_delta[active], axis=-1).tolist()
                    )
                    latent_l2_by_pair[key].extend(
                        np.linalg.norm(latent_delta[active], axis=-1).tolist()
                    )
                    tcp_l2_by_pair[key].extend(np.linalg.norm(tcp_delta[active], axis=-1).tolist())
                near_h = horizons[0]
                center_h = horizon_steps
                far_h = horizons[-1]
                near_far_action_delta = actions_by_h[far_h] - actions_by_h[near_h]
                near_far_latent_delta = goals[far_h] - goals[near_h]
                near_far_tcp_delta = tcp_by_h[far_h] - tcp_by_h[near_h]
                action_l2_near_far.extend(
                    np.linalg.norm(near_far_action_delta[active], axis=-1).tolist()
                )
                latent_l2_near_far.extend(
                    np.linalg.norm(near_far_latent_delta[active], axis=-1).tolist()
                )
                tcp_l2_near_far.extend(np.linalg.norm(near_far_tcp_delta[active], axis=-1).tolist())
                action_xyz = near_far_action_delta[:, :3]
                denom = (
                    np.linalg.norm(action_xyz, axis=-1)
                    * np.linalg.norm(near_far_tcp_delta, axis=-1)
                    + 1e-8
                )
                cosine = np.sum(action_xyz * near_far_tcp_delta, axis=-1) / denom
                directional_cosines.extend(cosine[active].tolist())
                positive_directional.extend((cosine[active] > 0.0).astype(np.float32).tolist())
                near_vec = tcp_by_h[near_h] - current_state[:, 14:17]
                far_vec = tcp_by_h[far_h] - current_state[:, 14:17]
                near_proj = np.sum(actions_by_h[near_h][:, :3] * near_vec, axis=-1) / (
                    np.linalg.norm(near_vec, axis=-1) + 1e-8
                )
                far_proj = np.sum(actions_by_h[far_h][:, :3] * far_vec, axis=-1) / (
                    np.linalg.norm(far_vec, axis=-1) + 1e-8
                )
                farther_projection_ge_near.extend(
                    (far_proj[active] >= near_proj[active]).astype(np.float32).tolist()
                )
                teacher_now = (
                    torch.clamp(
                        teacher.actor_mean(obs["state"].to(device).float()),
                        action_low,
                        action_high,
                    )
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(np.float32)
                )
                teacher_action_maes.extend(
                    np.mean(
                        np.abs(actions_by_h[center_h][active] - teacher_now[active]), axis=-1
                    ).tolist()
                )
                sample_count += active_count

                action_t = torch.from_numpy(actions_by_h[center_h]).to(device).float()
                if bool(config.get("policy.clip_actions_to_env_space", True)):
                    action_t = torch.clamp(action_t, action_low, action_high)
                action_t[~torch.from_numpy(active).to(device)] = 0.0
                obs, _reward, terminated, truncated, info = student_env.step(action_t)
                history.append(action_t.detach().clone())
                prev_action_norm = action_norm.transform(
                    action_t.detach().cpu().numpy().astype(np.float32)
                )
                if "success" in info:
                    success_once |= _numpy(info["success"]).reshape(-1).astype(bool)
                done = _numpy(torch.logical_or(terminated, truncated)).reshape(-1).astype(bool)
                newly_done = active & done
                if np.any(newly_done):
                    progress.update(int(newly_done.sum()))
                    active[newly_done] = False
            if np.any(active):
                progress.update(int(active.sum()))
            successes.extend(float(x) for x in success_once)
        finally:
            student_env.close()
            branch_env.close()
    progress.close()

    def mean_or_zero(values: list[float]) -> float:
        return float(np.mean(values)) if values else 0.0

    def reduction_fraction(initial: list[float], final: list[float]) -> float:
        if not initial:
            return 0.0
        return float(np.mean(np.asarray(final) < np.asarray(initial)))

    counterfactual_metrics = {
        "requested_source_queries": counterfactual_queries,
        "completed_source_queries": counterfactual_samples,
        "goal_rollouts": counterfactual_samples * len(horizons),
        "valid_goal_rollouts": len(counterfactual_final_latent_errors),
        "replay_current_state_error_max": (
            float(np.max(counterfactual_replay_errors)) if counterfactual_replay_errors else 0.0
        ),
        "done_before_goal_fraction": mean_or_zero(counterfactual_done_before_goal),
        "initial_latent_error": mean_or_zero(counterfactual_initial_latent_errors),
        "final_latent_error": mean_or_zero(counterfactual_final_latent_errors),
        "latent_error_reduced_fraction": reduction_fraction(
            counterfactual_initial_latent_errors,
            counterfactual_final_latent_errors,
        ),
        "initial_object_position_error": mean_or_zero(
            counterfactual_initial_object_position_errors
        ),
        "final_object_position_error": mean_or_zero(counterfactual_final_object_position_errors),
        "object_position_error_reduced_fraction": reduction_fraction(
            counterfactual_initial_object_position_errors,
            counterfactual_final_object_position_errors,
        ),
        "initial_yaw_error": mean_or_zero(counterfactual_initial_yaw_errors),
        "final_yaw_error": mean_or_zero(counterfactual_final_yaw_errors),
        "yaw_error_reduced_fraction": reduction_fraction(
            counterfactual_initial_yaw_errors,
            counterfactual_final_yaw_errors,
        ),
        "initial_tcp_position_error": mean_or_zero(counterfactual_initial_tcp_errors),
        "final_tcp_position_error": mean_or_zero(counterfactual_final_tcp_errors),
        "tcp_position_error_reduced_fraction": reduction_fraction(
            counterfactual_initial_tcp_errors,
            counterfactual_final_tcp_errors,
        ),
        "assigned_latent_goal_closest_fraction": mean_or_zero(counterfactual_latent_closest),
        "assigned_physical_goal_closest_fraction": mean_or_zero(counterfactual_physical_closest),
    }
    metrics = {
        "episodes": eval_episodes,
        "num_envs": max_num_envs,
        "seed_start": seed_start,
        "samples": sample_count,
        "rollout_success": float(np.mean(successes)) if successes else 0.0,
        "replay_current_state_error_mean": mean_or_zero(replay_errors),
        "replay_current_state_error_max": float(np.max(replay_errors)) if replay_errors else 0.0,
        "replay_failed_step_fraction": float(failed_replay_steps / max(1, len(replay_errors))),
        "branch_generation_latency_per_env_s": mean_or_zero(branch_latencies_per_env),
        "teacher_action_mae_center_goal": mean_or_zero(teacher_action_maes),
        "action_sensitivity_l2_near_to_far": mean_or_zero(action_l2_near_far),
        "latent_goal_l2_near_to_far": mean_or_zero(latent_l2_near_far),
        "tcp_goal_l2_near_to_far": mean_or_zero(tcp_l2_near_far),
        "directional_consistency_cosine": mean_or_zero(directional_cosines),
        "positive_directional_fraction": mean_or_zero(positive_directional),
        "farther_progress_projection_ge_near_fraction": mean_or_zero(farther_projection_ge_near),
        "action_sensitivity_l2_by_pair": {
            key: mean_or_zero(value) for key, value in action_l2_by_pair.items()
        },
        "latent_goal_l2_by_pair": {
            key: mean_or_zero(value) for key, value in latent_l2_by_pair.items()
        },
        "tcp_goal_l2_by_pair": {key: mean_or_zero(value) for key, value in tcp_l2_by_pair.items()},
        "counterfactual": counterfactual_metrics,
    }
    payload = {
        "phase": "7G",
        "method": "valid_reachable_goal_use",
        "variant": variant,
        "latent_dim": latent_dim,
        "horizon_steps": horizon_steps,
        "tested_horizons": horizons,
        "action_chunk_steps": action_chunk_steps,
        "goal_encoding": goal_encoding,
        "goal_dropout_prob": goal_dropout_prob,
        "dagger_iteration": dagger_iteration,
        "dagger_query_episodes": dagger_query_episodes,
        "seed": seed,
        "checkpoint": str(checkpoint_path),
        "closed_loop": metrics,
        "matched_replay_state_gate": metrics["replay_failed_step_fraction"] == 0.0,
        "goal_sensitivity_gate": metrics["action_sensitivity_l2_near_to_far"] > 0.01,
        "directional_consistency_gate": metrics["positive_directional_fraction"] > 0.5,
        "counterfactual_goal_approach_gate": (
            counterfactual_queries == 0
            or (
                counterfactual_metrics["valid_goal_rollouts"] > 0
                and counterfactual_metrics["latent_error_reduced_fraction"] > 0.5
                and counterfactual_metrics["tcp_position_error_reduced_fraction"] > 0.5
            )
        ),
        "metadata": _runtime_metadata(config),
    }
    write_json(output_path, payload)
    console.print(payload)
    return output_path


def _phase7_load_privileged_model(
    checkpoint: dict[str, Any],
    key: str,
    device: torch.device,
) -> tuple[nn.Module, Standardizer]:
    entry = checkpoint[key]
    model = MLP(
        int(entry["cond_dim"]),
        int(entry["action_dim"]),
        int(entry["hidden_dim"]),
        depth=4,
    ).to(device)
    model.load_state_dict(entry["model"])
    model.eval()
    return model, Standardizer.from_state_dict(entry["cond_norm"])


def _phase7_obs_state_tensor(obs: Any, device: torch.device) -> torch.Tensor:
    if isinstance(obs, dict):
        obs = obs["state"]
    if isinstance(obs, torch.Tensor):
        return obs.to(device).float()
    return torch.as_tensor(obs, device=device, dtype=torch.float32)


@torch.inference_mode()
def _evaluate_phase7_privileged_mode(
    config: Config,
    checkpoint: dict[str, Any],
    mode: str,
    model: nn.Module,
    cond_norm: Standardizer,
    action_norm: Standardizer,
    teacher: PPOAgent,
    horizon_steps: int,
    episodes: int,
    seed_start: int,
    goal_update_period: int = 1,
    max_num_envs_override: int | None = None,
) -> dict[str, Any]:
    valid_modes = {"flat", "branch_goal", *PRE_RL_PHASE_B_GOAL_TYPES}
    if mode not in valid_modes:
        raise ValueError(f"Unknown privileged Phase 7D mode: {mode}")
    if goal_update_period < 1:
        raise ValueError("Goal update period must be at least one primitive step")
    device = default_device()

    def make_env(num_envs: int):
        return gym.make(
            config.get("env_id"),
            obs_mode="state",
            control_mode=config.get("control_mode"),
            reward_mode="normalized_dense",
            render_mode=None,
            sim_backend=_rl_backend(config),
            num_envs=num_envs,
            reconfiguration_freq=config.get("rl.eval_reconfiguration_freq", 1),
        )

    max_num_envs = min(
        int(
            max_num_envs_override
            or config.get("incremental.phase7.replay_branch_num_envs", 16)
        ),
        episodes,
    )
    action_dim = int(checkpoint["flat"]["action_dim"])
    zero_action_norm = action_norm.transform(np.zeros((1, action_dim), dtype=np.float32))[0]
    control_freq = int(checkpoint["control_freq"])
    replay_state_tolerance = float(
        config.get("incremental.phase7.replay_branch_state_tolerance", 1e-6)
    )
    max_episode_steps = int(config.get("env_max_episode_steps", 100))
    successes: list[float] = []
    final_rewards: list[float] = []
    max_rewards: list[float] = []
    episode_lengths: list[int] = []
    teacher_action_maes: list[float] = []
    action_saturation: list[float] = []
    policy_latencies: list[float] = []
    branch_latencies: list[float] = []
    branch_latencies_per_env: list[float] = []
    replay_errors: list[float] = []
    failed_replay_steps = 0
    branch_goal_norms: list[float] = []
    object_subgoal_errors: list[float] = []
    yaw_subgoal_errors: list[float] = []
    tcp_subgoal_errors: list[float] = []
    high_level_decisions = 0
    progress = trange(episodes, desc=f"phase7D privileged {mode} eval")
    for batch_start in range(0, episodes, max_num_envs):
        num_envs = min(max_num_envs, episodes - batch_start)
        reset_seeds = [seed_start + batch_start + i for i in range(num_envs)]
        student_env = make_env(num_envs)
        branch_env = make_env(num_envs) if mode != "flat" else None
        action_low_np = np.asarray(student_env.action_space.low, dtype=np.float32)
        action_high_np = np.asarray(student_env.action_space.high, dtype=np.float32)
        if action_low_np.ndim == 2:
            action_low_np = action_low_np[0]
            action_high_np = action_high_np[0]
        action_low = torch.as_tensor(action_low_np, device=device, dtype=torch.float32)
        action_high = torch.as_tensor(action_high_np, device=device, dtype=torch.float32)
        try:
            obs, _info = student_env.reset(seed=reset_seeds)
            history: list[torch.Tensor] = []
            prev_action_norm = np.repeat(zero_action_norm[None, :], num_envs, axis=0).astype(
                np.float32
            )
            active = np.ones(num_envs, dtype=bool)
            held_goal: np.ndarray | None = None
            held_target_future_state: np.ndarray | None = None
            last_replan_step = 0
            success_once = np.zeros(num_envs, dtype=bool)
            batch_final_rewards = np.zeros(num_envs, dtype=np.float32)
            batch_max_rewards = np.full(num_envs, -np.inf, dtype=np.float32)
            batch_lengths = np.zeros(num_envs, dtype=np.int32)
            for _step in range(max_episode_steps):
                if not active.any():
                    break
                active_count = int(active.sum())
                state_t = _phase7_obs_state_tensor(obs, device)
                current_state = state_t.detach().cpu().numpy().astype(np.float32)
                goal = held_goal
                target_future_state = held_target_future_state
                remaining_steps = horizon_steps
                should_replan = mode != "flat" and (
                    held_goal is None or _step % goal_update_period == 0
                )
                if should_replan:
                    if branch_env is None:
                        raise RuntimeError("Branch env missing for branch-goal evaluation")
                    branch_timer = Timer()
                    branch_obs, _branch_info = branch_env.reset(seed=reset_seeds)
                    replay_done = torch.zeros(num_envs, device=device, dtype=torch.bool)
                    for action_history in history:
                        branch_obs, _branch_reward, branch_term, branch_trunc, _branch_info = (
                            branch_env.step(action_history)
                        )
                        replay_done = replay_done | torch.logical_or(
                            branch_term, branch_trunc
                        ).view(-1)
                    state_errors = torch.max(
                        torch.abs(
                            student_env.unwrapped.get_state() - branch_env.unwrapped.get_state()
                        ),
                        dim=1,
                    ).values
                    state_errors_np = state_errors.detach().cpu().numpy()
                    replay_errors.extend(float(x) for x in state_errors_np[active])
                    replay_done_np = replay_done.detach().cpu().numpy().astype(bool)
                    failed_replay_steps += int(
                        np.sum(
                            active & (replay_done_np | (state_errors_np > replay_state_tolerance))
                        )
                    )
                    for _ in range(horizon_steps):
                        teacher_action = torch.clamp(
                            teacher.actor_mean(_phase7_obs_state_tensor(branch_obs, device)),
                            action_low,
                            action_high,
                        )
                        branch_obs, _branch_reward, branch_term, branch_trunc, _branch_info = (
                            branch_env.step(teacher_action)
                        )
                        if bool(torch.all(torch.logical_or(branch_term, branch_trunc))):
                            break
                    branch_elapsed = branch_timer.elapsed()
                    branch_latencies.append(branch_elapsed)
                    branch_latencies_per_env.append(branch_elapsed / active_count)
                    future_state = (
                        _phase7_obs_state_tensor(branch_obs, device).detach().cpu().numpy()
                    )
                    target_future_state = future_state
                    goal = (
                        _phase7_privileged_goal(
                            current_state,
                            future_state,
                            horizon_steps,
                            control_freq,
                        )
                        if mode == "branch_goal"
                        else _pre_rl_phase_b_goal(
                            current_state,
                            future_state,
                            horizon_steps,
                            control_freq,
                            mode,
                        )
                    )
                    held_goal = goal
                    held_target_future_state = target_future_state
                    last_replan_step = _step
                    high_level_decisions += active_count
                    branch_goal_norms.extend(np.linalg.norm(goal[active], axis=-1).tolist())
                elif mode != "flat" and held_target_future_state is not None:
                    remaining_steps = max(1, horizon_steps - (_step - last_replan_step))
                    goal = (
                        _phase7_privileged_goal(
                            current_state,
                            held_target_future_state,
                            remaining_steps,
                            control_freq,
                        )
                        if mode == "branch_goal"
                        else _pre_rl_phase_b_goal(
                            current_state,
                            held_target_future_state,
                            remaining_steps,
                            control_freq,
                            mode,
                        )
                    )
                    held_goal = goal

                policy_timer = Timer()
                cond = _phase7_privileged_condition(current_state, goal, prev_action_norm)
                if bool(checkpoint[mode].get("time_conditioned", False)):
                    time_to_goal = np.full(
                        (len(cond), 1),
                        remaining_steps / horizon_steps,
                        dtype=np.float32,
                    )
                    cond = np.concatenate([cond, time_to_goal], axis=-1)
                cond_t = torch.from_numpy(cond_norm.transform(cond)).to(device).float()
                raw_action = action_norm.inverse(model(cond_t).cpu().numpy()).astype(np.float32)
                policy_latencies.append(policy_timer.elapsed() / active_count)
                teacher_now = (
                    torch.clamp(
                        teacher.actor_mean(state_t),
                        action_low,
                        action_high,
                    )
                    .cpu()
                    .numpy()
                    .astype(np.float32)
                )
                teacher_action_maes.extend(
                    np.mean(np.abs(raw_action[active] - teacher_now[active]), axis=-1).tolist()
                )
                action_saturation.extend(
                    np.any(
                        (raw_action[active] < action_low_np)
                        | (raw_action[active] > action_high_np),
                        axis=-1,
                    )
                    .astype(np.float32)
                    .tolist()
                )
                action_t = torch.from_numpy(raw_action).to(device).float()
                if bool(config.get("policy.clip_actions_to_env_space", True)):
                    action_t = torch.clamp(action_t, action_low, action_high)
                action_t[~torch.from_numpy(active).to(device)] = 0.0
                obs, reward, terminated, truncated, info = student_env.step(action_t)
                if target_future_state is not None:
                    next_state = _phase7_obs_state_tensor(obs, device).detach().cpu().numpy()
                    object_subgoal_errors.extend(
                        np.linalg.norm(
                            next_state[active, 24:26] - target_future_state[active, 24:26],
                            axis=-1,
                        ).tolist()
                    )
                    yaw_error = np.abs(
                        _wrap_angle(
                            _phase7_privileged_yaw(next_state[active])
                            - _phase7_privileged_yaw(target_future_state[active])
                        )
                    )
                    yaw_subgoal_errors.extend(yaw_error.tolist())
                    tcp_subgoal_errors.extend(
                        np.linalg.norm(
                            next_state[active, 14:17] - target_future_state[active, 14:17],
                            axis=-1,
                        ).tolist()
                    )
                history.append(action_t.detach().clone())
                prev_action_norm = action_norm.transform(
                    action_t.detach().cpu().numpy().astype(np.float32)
                )
                reward_np = _numpy(reward).reshape(-1).astype(np.float32)
                batch_final_rewards[active] = reward_np[active]
                batch_max_rewards[active] = np.maximum(batch_max_rewards[active], reward_np[active])
                batch_lengths[active] += 1
                if "success" in info:
                    success_once |= _numpy(info["success"]).reshape(-1).astype(bool)
                done = _numpy(torch.logical_or(terminated, truncated)).reshape(-1).astype(bool)
                newly_done = active & done
                if np.any(newly_done):
                    progress.update(int(newly_done.sum()))
                    active[newly_done] = False
            if np.any(active):
                progress.update(int(active.sum()))
            successes.extend(float(x) for x in success_once)
            final_rewards.extend(float(x) for x in batch_final_rewards)
            max_rewards.extend(float(x) for x in batch_max_rewards)
            episode_lengths.extend(int(x) for x in batch_lengths)
        finally:
            student_env.close()
            if branch_env is not None:
                branch_env.close()
    progress.close()
    return {
        "success": float(np.mean(successes)),
        "success_stderr": float(np.std(successes) / np.sqrt(len(successes))),
        "final_reward": float(np.mean(final_rewards)),
        "max_reward": float(np.mean(max_rewards)),
        "mean_episode_length": float(np.mean(episode_lengths)),
        "teacher_action_mae": float(np.mean(teacher_action_maes)) if teacher_action_maes else 0.0,
        "action_saturation_rate": float(np.mean(action_saturation)) if action_saturation else 0.0,
        "policy_latency_s": float(np.mean(policy_latencies)) if policy_latencies else 0.0,
        "branch_generation_latency_s": float(np.mean(branch_latencies))
        if branch_latencies
        else 0.0,
        "branch_generation_latency_per_env_s": (
            float(np.mean(branch_latencies_per_env)) if branch_latencies_per_env else 0.0
        ),
        "replay_current_state_error_mean": float(np.mean(replay_errors)) if replay_errors else 0.0,
        "replay_current_state_error_max": float(np.max(replay_errors)) if replay_errors else 0.0,
        "replay_failed_step_fraction": float(failed_replay_steps / max(1, len(replay_errors))),
        "branch_goal_l2": float(np.mean(branch_goal_norms)) if branch_goal_norms else 0.0,
        "one_step_object_subgoal_error_m": (
            float(np.mean(object_subgoal_errors)) if object_subgoal_errors else None
        ),
        "one_step_yaw_subgoal_error_rad": (
            float(np.mean(yaw_subgoal_errors)) if yaw_subgoal_errors else None
        ),
        "one_step_tcp_subgoal_error_m": (
            float(np.mean(tcp_subgoal_errors)) if tcp_subgoal_errors else None
        ),
        "episodes": episodes,
        "num_envs": max_num_envs,
        "seed_start": seed_start,
        "goal_update_period": goal_update_period,
        "high_level_decisions_per_episode": float(high_level_decisions / episodes),
    }


def evaluate_phase7_privileged_branch_baselines(
    config: Config,
    horizon_steps: int | None = None,
    seed: int = 0,
    episodes: int | None = None,
    force: bool = False,
) -> Path:
    horizon_steps = int(horizon_steps or config.get("incremental.phase7.horizon_steps", 2))
    checkpoint_path = train_phase7_privileged_branch_baselines(
        config,
        horizon_steps=horizon_steps,
        seed=seed,
        force=False,
    )
    eval_episodes = int(episodes or config.get("incremental.phase7.privileged_eval_episodes", 50))
    results_dir = _phase7_privileged_results_dir(config, horizon_steps, seed)
    output_path = results_dir / f"privileged_branch_baselines_eval_{eval_episodes}.json"
    if output_path.exists() and not force:
        console.print(f"Phase 7D privileged eval exists: {output_path}")
        return output_path
    device = default_device()
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    action_norm = Standardizer.from_state_dict(checkpoint["action_norm"])
    flat_model, flat_cond_norm = _phase7_load_privileged_model(checkpoint, "flat", device)
    goal_model, goal_cond_norm = _phase7_load_privileged_model(checkpoint, "branch_goal", device)
    teacher = load_ppo_agent(_rl_paths(config).best, device)
    seed_start = int(config.get("incremental.phase7.replay_branch_seed", 1_200_000))
    flat = _evaluate_phase7_privileged_mode(
        config,
        checkpoint,
        "flat",
        flat_model,
        flat_cond_norm,
        action_norm,
        teacher,
        horizon_steps,
        eval_episodes,
        seed_start,
    )
    branch_goal = _evaluate_phase7_privileged_mode(
        config,
        checkpoint,
        "branch_goal",
        goal_model,
        goal_cond_norm,
        action_norm,
        teacher,
        horizon_steps,
        eval_episodes,
        seed_start,
    )
    payload = {
        "phase": "7D",
        "method": "privileged_structured_branch_oracle_baseline",
        "horizon_steps": horizon_steps,
        "seed": seed,
        "checkpoint": str(checkpoint_path),
        "closed_loop": {
            "flat": flat,
            "branch_goal": branch_goal,
        },
        "validation_action_metrics": {
            "flat": checkpoint["flat"]["validation_metrics"],
            "branch_goal": checkpoint["branch_goal"]["validation_metrics"],
        },
        "gate_within_flat_5pp": bool(branch_goal["success"] >= flat["success"] - 0.05),
        "preferred_branch_success_gate": bool(branch_goal["success"] >= 0.80),
        "matched_replay_state_gate": branch_goal["replay_failed_step_fraction"] == 0.0,
        "metadata": _runtime_metadata(config),
    }
    write_json(output_path, payload)
    console.print(payload)
    return output_path


def _pre_rl_phase_b_checkpoint_dir(config: Config, horizon_steps: int, seed: int) -> Path:
    return ensure_dir(
        config.path_value("paths.incremental_artifact_dir")
        / "pre_rl"
        / "phase_b"
        / f"k{horizon_steps}"
        / f"seed{seed}"
    )


def _pre_rl_phase_b_results_dir(config: Config, horizon_steps: int, seed: int) -> Path:
    return ensure_dir(
        config.path_value("paths.incremental_results_dir")
        / "pre_rl"
        / "phase_b"
        / f"k{horizon_steps}"
        / f"seed{seed}"
    )


def _pre_rl_phase_b_action_sensitivity(
    model_payload: dict[str, Any],
    conditions: np.ndarray,
    action_norm: Standardizer,
    seed: int,
    max_samples: int = 10000,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    indices = np.arange(len(conditions))
    if len(indices) > max_samples:
        indices = rng.choice(indices, size=max_samples, replace=False)
    correct = conditions[indices].copy()
    shuffled = correct.copy()
    shuffled[:, 31:-3] = shuffled[rng.permutation(len(shuffled)), 31:-3]
    device = default_device()
    model = MLP(
        int(model_payload["cond_dim"]),
        int(model_payload["action_dim"]),
        int(model_payload["hidden_dim"]),
        depth=4,
    ).to(device)
    model.load_state_dict(model_payload["model"])
    model.eval()
    cond_norm = Standardizer.from_state_dict(model_payload["cond_norm"])

    def predict(rows: np.ndarray) -> np.ndarray:
        outputs = []
        with torch.inference_mode():
            for start in range(0, len(rows), 8192):
                x = torch.from_numpy(cond_norm.transform(rows[start : start + 8192])).to(
                    device
                ).float()
                outputs.append(action_norm.inverse(model(x).cpu().numpy()))
        return np.concatenate(outputs)

    correct_actions = predict(correct)
    shuffled_actions = predict(shuffled)
    differences = np.linalg.norm(correct_actions - shuffled_actions, axis=-1)
    return {
        "valid_goal_action_sensitivity_l2": float(np.mean(differences)),
        "valid_goal_action_sensitivity_median_l2": float(np.median(differences)),
        "samples": int(len(indices)),
    }


def train_pre_rl_phase_b_horizon(
    config: Config,
    horizon_steps: int,
    seed: int = 0,
    force: bool = False,
) -> Path:
    horizons = [int(value) for value in config.get("pre_rl.phase_b.horizons")]
    if horizon_steps not in horizons:
        raise ValueError(f"Phase B horizon must be one of {horizons}, got {horizon_steps}")
    checkpoint_dir = _pre_rl_phase_b_checkpoint_dir(config, horizon_steps, seed)
    checkpoint_path = checkpoint_dir / "oracle_goal_decomposition.pt"
    if checkpoint_path.exists() and not force:
        console.print(f"Pre-RL Phase B models exist: {checkpoint_path}")
        return checkpoint_path
    train_episodes, validation_episodes, data_metadata = _load_phase7_privileged_episodes(
        config,
        horizon_steps,
        cap_train_to_usable=True,
    )
    train_actions = np.concatenate([episode["actions"] for episode in train_episodes], axis=0)
    action_norm = Standardizer.fit(train_actions)
    control_freq = int(config.get("control_freq", 20))
    modes: list[str] = ["flat", *PRE_RL_PHASE_B_GOAL_TYPES]
    models = {}
    summaries = {}
    timer = Timer()
    for mode_index, mode in enumerate(modes):
        goal_type = None if mode == "flat" else mode
        train_cond, train_labels = _pre_rl_phase_b_conditions(
            train_episodes,
            action_norm,
            horizon_steps,
            control_freq,
            goal_type,
        )
        validation_cond, validation_labels = _pre_rl_phase_b_conditions(
            validation_episodes,
            action_norm,
            horizon_steps,
            control_freq,
            goal_type,
        )
        model_payload, summary = _phase7_train_privileged_model(
            config,
            f"pre-rl-b-k{horizon_steps}-{mode}",
            train_cond,
            train_labels,
            validation_cond,
            validation_labels,
            action_norm,
            seed + mode_index * 1000,
        )
        model_payload["goal_type"] = goal_type
        model_payload["goal_dim"] = int(train_cond.shape[-1] - 34)
        if goal_type is not None:
            sensitivity = _pre_rl_phase_b_action_sensitivity(
                model_payload,
                validation_cond,
                action_norm,
                seed + horizon_steps * 100 + mode_index,
            )
            model_payload["goal_sensitivity"] = sensitivity
            summary["goal_sensitivity"] = sensitivity
        models[mode] = model_payload
        summaries[mode] = summary
    payload = {
        "phase": "B1-B2",
        "method": "privileged_oracle_goal_decomposition",
        "horizon_steps": horizon_steps,
        "horizon_seconds": horizon_steps / float(control_freq),
        "control_freq": control_freq,
        "seed": seed,
        "action_norm": action_norm.state_dict(),
        "flat": models["flat"],
        **{mode: models[mode] for mode in PRE_RL_PHASE_B_GOAL_TYPES},
        "data": {
            **data_metadata,
            "causal_train_transitions": int(
                sum(len(episode["actions"]) for episode in train_episodes)
            ),
            "state_query_samples": 0,
        },
        "elapsed_s": timer.elapsed(),
        "metadata": _runtime_metadata(config),
    }
    torch.save(payload, checkpoint_path)
    write_json(
        checkpoint_dir / "oracle_goal_decomposition_metrics.json",
        {
            "phase": "B1-B2",
            "horizon_steps": horizon_steps,
            "seed": seed,
            "models": summaries,
            "data": payload["data"],
            "elapsed_s": payload["elapsed_s"],
            "metadata": payload["metadata"],
        },
    )
    console.print(f"Wrote pre-RL Phase B models: {checkpoint_path}")
    return checkpoint_path


def evaluate_pre_rl_phase_b_horizon(
    config: Config,
    horizon_steps: int,
    seed: int = 0,
    episodes: int | None = None,
    force: bool = False,
) -> Path:
    checkpoint_path = train_pre_rl_phase_b_horizon(
        config, horizon_steps=horizon_steps, seed=seed, force=False
    )
    eval_episodes = int(episodes or config.get("pre_rl.phase_b.smoke_eval_episodes", 20))
    output_path = (
        _pre_rl_phase_b_results_dir(config, horizon_steps, seed)
        / f"oracle_goal_decomposition_eval_{eval_episodes}.json"
    )
    if output_path.exists() and not force:
        console.print(f"Pre-RL Phase B evaluation exists: {output_path}")
        return output_path
    device = default_device()
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    action_norm = Standardizer.from_state_dict(checkpoint["action_norm"])
    teacher = load_ppo_agent(_rl_paths(config).best, device)
    eval_seed = int(config.get("pre_rl.phase_b.eval_seed_start", 1_600_000))
    modes: list[str] = ["flat", *PRE_RL_PHASE_B_GOAL_TYPES]
    rows = []
    for mode in modes:
        model, cond_norm = _phase7_load_privileged_model(checkpoint, mode, device)
        metrics = _evaluate_phase7_privileged_mode(
            config,
            checkpoint,
            mode,
            model,
            cond_norm,
            action_norm,
            teacher,
            horizon_steps,
            eval_episodes,
            eval_seed,
        )
        row = {
            "goal_type": mode,
            **metrics,
            "success_wilson_95": _wilson_interval(metrics["success"], eval_episodes),
            "validation_action_mae": float(checkpoint[mode]["validation_metrics"]["mae"]),
            "goal_dim": int(checkpoint[mode].get("goal_dim", 0)),
            "valid_goal_action_sensitivity_l2": (
                checkpoint[mode].get("goal_sensitivity", {}).get(
                    "valid_goal_action_sensitivity_l2"
                )
            ),
        }
        rows.append(row)
    full_success = next(row["success"] for row in rows if row["goal_type"] == "full")
    for row in rows:
        row["success_fraction_of_full"] = float(
            row["success"] / max(float(full_success), 1e-8)
        )
    object_success = next(row["success"] for row in rows if row["goal_type"] == "object")
    object_pose_success = next(
        row["success"] for row in rows if row["goal_type"] == "object_pose"
    )
    payload = {
        "phase": "B1-B2",
        "experiment": "oracle_goal_information_decomposition",
        "command": (
            "uv run hcl-poc incremental pre-rl-b-eval "
            f"--config {config.path} --horizon-steps {horizon_steps} "
            f"--seed {seed} --episodes {eval_episodes}"
        ),
        "horizon_steps": horizon_steps,
        "horizon_seconds": horizon_steps / float(config.get("control_freq", 20)),
        "action_horizon_steps": 1,
        "seed": seed,
        "evaluation_seed_start": eval_seed,
        "episodes": eval_episodes,
        "checkpoint": str(checkpoint_path),
        "data": checkpoint["data"],
        "rows": rows,
        "object_80pct_full_gate": bool(
            max(object_success, object_pose_success) >= 0.8 * full_success
        ),
        "replay_correct": bool(
            all(
                row["replay_failed_step_fraction"] == 0.0
                for row in rows
                if row["goal_type"] != "flat"
            )
        ),
        "metadata": _runtime_metadata(config),
    }
    write_json(output_path, payload)
    console.print(payload)
    return output_path


def aggregate_pre_rl_phase_b(config: Config, episodes: int | None = None) -> Path:
    import csv
    import json
    import matplotlib.pyplot as plt

    eval_episodes = int(episodes or config.get("pre_rl.phase_b.development_eval_episodes", 100))
    horizons = [int(value) for value in config.get("pre_rl.phase_b.horizons")]
    seed = int(config.get("pre_rl.phase_b.policy_seed", 0))
    results = []
    for horizon in horizons:
        path = (
            _pre_rl_phase_b_results_dir(config, horizon, seed)
            / f"oracle_goal_decomposition_eval_{eval_episodes}.json"
        )
        if not path.exists():
            raise FileNotFoundError(f"Missing Phase B evaluation: {path}")
        with path.open("r", encoding="utf-8") as f:
            results.append(json.load(f))
    root = ensure_dir(config.path_value("paths.incremental_results_dir") / "pre_rl" / "phase_b")
    csv_path = root / "oracle_goal_decomposition.csv"
    fields = [
        "goal_type",
        "horizon_steps",
        "horizon_seconds",
        "policy_seed",
        "episodes",
        "evaluation_seed_start",
        "success",
        "success_ci_low",
        "success_ci_high",
        "final_reward",
        "max_reward",
        "teacher_action_mae",
        "validation_action_mae",
        "one_step_object_subgoal_error_m",
        "one_step_yaw_subgoal_error_rad",
        "one_step_tcp_subgoal_error_m",
        "valid_goal_action_sensitivity_l2",
        "replay_current_state_error_max",
        "success_fraction_of_full",
    ]
    csv_rows = []
    for result in results:
        for row in result["rows"]:
            csv_rows.append(
                {
                    "goal_type": row["goal_type"],
                    "horizon_steps": result["horizon_steps"],
                    "horizon_seconds": result["horizon_seconds"],
                    "policy_seed": result["seed"],
                    "episodes": row["episodes"],
                    "evaluation_seed_start": row["seed_start"],
                    "success": row["success"],
                    "success_ci_low": row["success_wilson_95"][0],
                    "success_ci_high": row["success_wilson_95"][1],
                    "final_reward": row["final_reward"],
                    "max_reward": row["max_reward"],
                    "teacher_action_mae": row["teacher_action_mae"],
                    "validation_action_mae": row["validation_action_mae"],
                    "one_step_object_subgoal_error_m": row[
                        "one_step_object_subgoal_error_m"
                    ],
                    "one_step_yaw_subgoal_error_rad": row[
                        "one_step_yaw_subgoal_error_rad"
                    ],
                    "one_step_tcp_subgoal_error_m": row["one_step_tcp_subgoal_error_m"],
                    "valid_goal_action_sensitivity_l2": row[
                        "valid_goal_action_sensitivity_l2"
                    ],
                    "replay_current_state_error_max": row[
                        "replay_current_state_error_max"
                    ],
                    "success_fraction_of_full": row["success_fraction_of_full"],
                }
            )
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(csv_rows)
    plot_path = root / f"oracle_goal_decomposition_{eval_episodes}.png"
    figure, axis = plt.subplots(figsize=(9, 6))
    for mode in ["flat", *PRE_RL_PHASE_B_GOAL_TYPES]:
        mode_rows = [row for row in csv_rows if row["goal_type"] == mode]
        axis.plot(
            [row["horizon_seconds"] for row in mode_rows],
            [row["success"] for row in mode_rows],
            marker="o",
            label=mode.replace("_", " "),
        )
    axis.set_ylim(0.0, 1.0)
    axis.set_xlabel("Future-goal horizon (s)")
    axis.set_ylabel("Success rate")
    axis.set_title("Oracle goal information decomposition")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(plot_path, dpi=180)
    plt.close(figure)
    best_object_fraction = max(
        row["success_fraction_of_full"]
        for row in csv_rows
        if row["goal_type"] in {"object", "object_pose"}
    )
    payload = {
        "phase": "B",
        "episodes_per_setting": eval_episodes,
        "horizons": horizons,
        "policy_seed": seed,
        "rows": csv_rows,
        "best_object_fraction_of_full": float(best_object_fraction),
        "object_80pct_full_gate": bool(best_object_fraction >= 0.8),
        "csv": str(csv_path),
        "plot": str(plot_path),
        "metadata": _runtime_metadata(config),
    }
    output_path = root / f"phase_b_aggregate_{eval_episodes}.json"
    write_json(output_path, payload)
    console.print(payload)
    return output_path


def train_pre_rl_phase_c_time_conditioned(
    config: Config,
    horizon_steps: int,
    seed: int = 0,
    force: bool = False,
) -> Path:
    goal_type = str(config.get("pre_rl.phase_c.goal_type", "tcp"))
    checkpoint_dir = ensure_dir(
        config.path_value("paths.incremental_artifact_dir")
        / "pre_rl"
        / "phase_c"
        / f"k{horizon_steps}"
        / f"seed{seed}"
    )
    checkpoint_path = checkpoint_dir / f"time_conditioned_{goal_type}.pt"
    if checkpoint_path.exists() and not force:
        console.print(f"Pre-RL Phase C time-conditioned model exists: {checkpoint_path}")
        return checkpoint_path
    phase_b_path = train_pre_rl_phase_b_horizon(
        config,
        horizon_steps=horizon_steps,
        seed=seed,
        force=False,
    )
    phase_b = torch.load(phase_b_path, map_location="cpu", weights_only=False)
    train_episodes, validation_episodes, data_metadata = _load_phase7_privileged_episodes(
        config,
        horizon_steps,
        cap_train_to_usable=True,
    )
    action_norm = Standardizer.from_state_dict(phase_b["action_norm"])
    control_freq = int(config.get("control_freq", 20))
    train_cond, train_labels = _pre_rl_phase_c_time_conditioned_conditions(
        train_episodes,
        action_norm,
        horizon_steps,
        control_freq,
        goal_type,
    )
    validation_cond, validation_labels = _pre_rl_phase_c_time_conditioned_conditions(
        validation_episodes,
        action_norm,
        horizon_steps,
        control_freq,
        goal_type,
    )
    model_payload, summary = _phase7_train_privileged_model(
        config,
        f"pre-rl-c-k{horizon_steps}-time-conditioned-{goal_type}",
        train_cond,
        train_labels,
        validation_cond,
        validation_labels,
        action_norm,
        seed,
    )
    model_payload.update(
        {
            "goal_type": goal_type,
            "goal_dim": int(train_cond.shape[-1] - 35),
            "time_conditioned": True,
            "training_offsets": list(range(1, horizon_steps + 1)),
        }
    )
    payload = {
        "phase": "C1",
        "method": "time_conditioned_multi_offset_oracle",
        "horizon_steps": horizon_steps,
        "control_freq": control_freq,
        "seed": seed,
        "action_norm": phase_b["action_norm"],
        "flat": phase_b["flat"],
        goal_type: model_payload,
        "data": {
            **data_metadata,
            "causal_train_transitions": int(
                sum(len(episode["actions"]) for episode in train_episodes)
            ),
            "expanded_train_samples": int(len(train_cond)),
            "state_query_samples": 0,
        },
        "training_summary": summary,
        "metadata": _runtime_metadata(config),
    }
    torch.save(payload, checkpoint_path)
    write_json(
        checkpoint_dir / f"time_conditioned_{goal_type}_metrics.json",
        {
            "phase": payload["phase"],
            "method": payload["method"],
            "horizon_steps": horizon_steps,
            "goal_type": goal_type,
            "data": payload["data"],
            "training_summary": summary,
            "metadata": payload["metadata"],
        },
    )
    console.print(f"Wrote Pre-RL Phase C time-conditioned model: {checkpoint_path}")
    return checkpoint_path


def run_pre_rl_phase_c_oracle_sweep(
    config: Config,
    episodes: int | None = None,
    time_conditioned: bool = False,
    horizons_override: list[int] | None = None,
    force: bool = False,
) -> Path:
    import csv
    import json
    import matplotlib.pyplot as plt

    eval_episodes = int(episodes or config.get("pre_rl.phase_c.smoke_eval_episodes", 20))
    horizons = horizons_override or [
        int(value) for value in config.get("pre_rl.phase_c.horizons")
    ]
    update_periods = [int(value) for value in config.get("pre_rl.phase_c.update_periods")]
    goal_type = str(config.get("pre_rl.phase_c.goal_type", "tcp"))
    if goal_type not in PRE_RL_PHASE_B_GOAL_TYPES:
        raise ValueError(f"Unknown Phase C goal type: {goal_type}")
    seed = int(config.get("pre_rl.phase_b.policy_seed", 0))
    eval_seed = int(config.get("pre_rl.phase_c.eval_seed_start", 1_700_000))
    max_num_envs = int(config.get("pre_rl.phase_c.replay_branch_num_envs", 32))
    root = ensure_dir(config.path_value("paths.incremental_results_dir") / "pre_rl" / "phase_c")
    device = default_device()
    teacher = load_ppo_agent(_rl_paths(config).best, device)
    rows: list[dict[str, Any]] = []
    variant = "time_conditioned" if time_conditioned else "fixed_offset"

    for horizon_steps in horizons:
        checkpoint_path = (
            train_pre_rl_phase_c_time_conditioned(
                config,
                horizon_steps=horizon_steps,
                seed=seed,
                force=False,
            )
            if time_conditioned
            else train_pre_rl_phase_b_horizon(
                config,
                horizon_steps=horizon_steps,
                seed=seed,
                force=False,
            )
        )
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        action_norm = Standardizer.from_state_dict(checkpoint["action_norm"])
        settings = [("flat", 1), *[(goal_type, period) for period in update_periods]]
        for mode, update_period in settings:
            setting_path = root / (
                f"k{horizon_steps}_{mode}_u{update_period}_{variant}_endpoint_"
                f"eval_{eval_episodes}.json"
            )
            if setting_path.exists() and not force:
                with setting_path.open("r", encoding="utf-8") as f:
                    result = json.load(f)
                rows.append(result["row"])
                continue
            model, cond_norm = _phase7_load_privileged_model(checkpoint, mode, device)
            metrics = _evaluate_phase7_privileged_mode(
                config,
                checkpoint,
                mode,
                model,
                cond_norm,
                action_norm,
                teacher,
                horizon_steps,
                eval_episodes,
                eval_seed,
                goal_update_period=update_period,
                max_num_envs_override=max_num_envs,
            )
            row = {
                **metrics,
                "method": "flat" if mode == "flat" else "oracle_held_goal",
                "goal_type": mode,
                "horizon_steps": horizon_steps,
                "horizon_seconds": horizon_steps / float(config.get("control_freq", 20)),
                "goal_update_period": 0 if mode == "flat" else update_period,
                "success_wilson_95": _wilson_interval(metrics["success"], eval_episodes),
            }
            write_json(
                setting_path,
                {
                    "phase": "C1",
                    "experiment": "oracle_goal_hold_sweep",
                    "policy_variant": variant,
                    "held_goal_semantics": "fixed_endpoint_recomputed_features",
                    "checkpoint": str(checkpoint_path),
                    "data": checkpoint["data"],
                    "row": row,
                    "metadata": _runtime_metadata(config),
                },
            )
            rows.append(row)

    fields = [
        "method",
        "goal_type",
        "horizon_steps",
        "horizon_seconds",
        "goal_update_period",
        "episodes",
        "success",
        "success_ci_low",
        "success_ci_high",
        "final_reward",
        "max_reward",
        "teacher_action_mae",
        "one_step_tcp_subgoal_error_m",
        "high_level_decisions_per_episode",
        "branch_generation_latency_s",
        "replay_current_state_error_max",
    ]
    csv_rows = []
    for row in rows:
        csv_rows.append(
            {
                **{field: row.get(field) for field in fields},
                "success_ci_low": row["success_wilson_95"][0],
                "success_ci_high": row["success_wilson_95"][1],
            }
        )
    csv_path = root / f"oracle_goal_hold_sweep_{variant}_{eval_episodes}.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(csv_rows)

    plot_path = root / f"oracle_goal_hold_sweep_{variant}_{eval_episodes}.png"
    figure, axis = plt.subplots(figsize=(8, 5.5))
    for horizon_steps in horizons:
        horizon_rows = [
            row
            for row in csv_rows
            if row["method"] == "oracle_held_goal"
            and row["horizon_steps"] == horizon_steps
        ]
        axis.plot(
            [row["goal_update_period"] for row in horizon_rows],
            [row["success"] for row in horizon_rows],
            marker="o",
            label=f"k={horizon_steps}",
        )
    axis.set_ylim(0.0, 1.0)
    axis.set_xlabel("Goal hold period U (primitive steps)")
    axis.set_ylabel("Success rate")
    axis.set_title(f"Oracle {goal_type.upper()} goal temporal abstraction")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(plot_path, dpi=180)
    plt.close(figure)

    payload = {
        "phase": "C1",
        "experiment": "oracle_goal_hold_sweep",
        "policy_variant": variant,
        "held_goal_semantics": "fixed_endpoint_recomputed_features",
        "goal_type": goal_type,
        "episodes_per_setting": eval_episodes,
        "horizons": horizons,
        "update_periods": update_periods,
        "rows": rows,
        "csv": str(csv_path),
        "plot": str(plot_path),
        "metadata": _runtime_metadata(config),
    }
    output_path = root / f"phase_c_oracle_sweep_{variant}_{eval_episodes}.json"
    write_json(output_path, payload)
    console.print(payload)
    return output_path


PRE_RL_PHASE_D_PERTURBATIONS = {
    1: "directional_bias",
    2: "action_hold",
    3: "action_delay",
    4: "action_scaling",
}


def _pre_rl_phase_d_schedule(
    rng: np.random.Generator,
    episodes: int,
    max_steps: int,
    bursts_min: int,
    bursts_max: int,
) -> list[list[dict[str, Any]]]:
    schedules: list[list[dict[str, Any]]] = []
    durations = np.asarray([2, 4, 8], dtype=np.int32)
    for _ in range(episodes):
        count = int(rng.integers(bursts_min, bursts_max + 1))
        anchors = np.linspace(8, max_steps - 20, count, dtype=np.int32)
        events = []
        for anchor in anchors:
            duration = int(rng.choice(durations))
            start = int(np.clip(anchor + rng.integers(-3, 4), 5, max_steps - duration - 5))
            kind = int(rng.integers(1, len(PRE_RL_PHASE_D_PERTURBATIONS) + 1))
            direction = rng.normal(size=3).astype(np.float32)
            direction /= max(float(np.linalg.norm(direction)), 1e-6)
            events.append(
                {
                    "start": start,
                    "end": start + duration,
                    "kind": kind,
                    "bias_fraction": float(rng.choice([0.05, 0.10, 0.20])),
                    "bias_direction": direction,
                    "delay": int(rng.integers(1, 4)),
                    "scale": float(rng.choice([0.7, 1.3])),
                }
            )
        schedules.append(sorted(events, key=lambda event: event["start"]))
    return schedules


@torch.inference_mode()
def collect_pre_rl_phase_d_recovery_dataset(
    config: Config,
    episodes: int | None = None,
    force: bool = False,
) -> Path:
    requested_episodes = int(episodes or config.get("pre_rl.phase_d.pilot_episodes", 200))
    output_path = (
        config.path_value("paths.incremental_data_dir")
        / f"pre_rl_phase_d_recovery_{requested_episodes}.h5"
    )
    if output_path.exists() and not force:
        console.print(f"Pre-RL Phase D recovery dataset exists: {output_path}")
        return output_path

    seed = int(config.get("pre_rl.phase_d.seed", 1_800_000))
    num_envs_limit = int(config.get("pre_rl.phase_d.num_envs", 16))
    max_steps = int(config.get("env_max_episode_steps", 100))
    bursts_min = int(config.get("pre_rl.phase_d.bursts_min", 1))
    bursts_max = int(config.get("pre_rl.phase_d.bursts_max", 3))
    rng = np.random.default_rng(seed)
    device = default_device()
    teacher = load_ppo_agent(_rl_paths(config).best, device)
    ensure_dir(output_path.parent)
    tmp_path = output_path.with_suffix(".tmp.h5")
    if tmp_path.exists():
        tmp_path.unlink()

    completed = 0
    recovered_episodes = 0
    successful_episodes = 0
    perturbation_steps = 0
    recovery_steps = 0
    progress = trange(requested_episodes, desc="collect pre-RL Phase D recovery episodes")
    with h5py.File(tmp_path, "w") as h5:
        meta = h5.create_group("meta")
        for key, value in _runtime_metadata(config).items():
            meta.attrs[key] = value
        meta.attrs["dataset_type"] = "causal_action_perturbation_recovery"
        meta.attrs["requested_episodes"] = requested_episodes
        meta.attrs["seed"] = seed
        meta.attrs["semantics"] = (
            "executed_actions are causal behavior actions; teacher_actions are a separate "
            "deterministic recovery-query view from the same visited state"
        )
        meta.attrs["perturbation_types"] = ";".join(
            f"{key}:{value}" for key, value in PRE_RL_PHASE_D_PERTURBATIONS.items()
        )

        while completed < requested_episodes:
            batch_size = min(num_envs_limit, requested_episodes - completed)
            env = _phase4_make_visual_env(config, batch_size)
            reset_seeds = [seed + completed + index for index in range(batch_size)]
            obs, _info = env.reset(seed=reset_seeds)
            schedules = _pre_rl_phase_d_schedule(
                rng,
                batch_size,
                max_steps,
                bursts_min,
                bursts_max,
            )
            action_low_np = np.asarray(env.single_action_space.low, dtype=np.float32)
            action_high_np = np.asarray(env.single_action_space.high, dtype=np.float32)
            action_low = torch.as_tensor(action_low_np, device=device)
            action_high = torch.as_tensor(action_high_np, device=device)
            action_range = action_high_np - action_low_np
            previous_executed = np.zeros((batch_size, len(action_low_np)), dtype=np.float32)
            teacher_history: list[np.ndarray] = []
            bias_noise = np.zeros_like(previous_executed)
            recovery_pending = np.zeros(batch_size, dtype=bool)
            recovered = np.zeros(batch_size, dtype=bool)
            success_once = np.zeros(batch_size, dtype=bool)
            buffers: list[dict[str, list[np.ndarray | float | int | bool]]] = [
                {
                    "rgb": [],
                    "states": [],
                    "proprioception": [],
                    "executed_actions": [],
                    "teacher_actions": [],
                    "perturbation_type": [],
                    "burst_id": [],
                    "perturbation_parameter": [],
                    "perturbation_active": [],
                    "perturbation_start": [],
                    "perturbation_end": [],
                    "recovery_active": [],
                    "recovery_completion": [],
                    "reward": [],
                    "success": [],
                }
                for _ in range(batch_size)
            ]
            try:
                for step in range(max_steps):
                    rgb, states = _phase4_rgb_state(obs)
                    state_t = _phase7_obs_state_tensor(obs, device)
                    teacher_action = torch.clamp(
                        teacher.actor_mean(state_t), action_low, action_high
                    )
                    teacher_np = teacher_action.cpu().numpy().astype(np.float32)
                    teacher_history.append(teacher_np.copy())
                    executed = teacher_np.copy()
                    kinds = np.zeros(batch_size, dtype=np.int8)
                    burst_ids = np.full(batch_size, -1, dtype=np.int16)
                    parameters = np.zeros(batch_size, dtype=np.float32)
                    starts = np.zeros(batch_size, dtype=bool)
                    ends = np.zeros(batch_size, dtype=bool)
                    active = np.zeros(batch_size, dtype=bool)
                    for env_index, events in enumerate(schedules):
                        active_event = next(
                            (
                                (event_index, item)
                                for event_index, item in enumerate(events)
                                if item["start"] <= step < item["end"]
                            ),
                            None,
                        )
                        event = None if active_event is None else active_event[1]
                        starts[env_index] = any(item["start"] == step for item in events)
                        ends[env_index] = any(item["end"] == step for item in events)
                        if starts[env_index]:
                            recovery_pending[env_index] = False
                            recovered[env_index] = False
                        if ends[env_index]:
                            recovery_pending[env_index] = True
                        if event is None:
                            continue
                        active[env_index] = True
                        kind = int(event["kind"])
                        kinds[env_index] = kind
                        burst_ids[env_index] = int(active_event[0])
                        if kind == 1:
                            parameters[env_index] = float(event["bias_fraction"])
                            bias_noise[env_index] = (
                                0.7 * bias_noise[env_index]
                                + 0.3
                                * rng.normal(0.0, 0.01, size=len(action_low_np)).astype(np.float32)
                                * action_range
                            )
                            executed[env_index] += (
                                event["bias_fraction"]
                                * action_range
                                * event["bias_direction"]
                                + bias_noise[env_index]
                            )
                        elif kind == 2:
                            parameters[env_index] = 1.0
                            executed[env_index] = previous_executed[env_index]
                        elif kind == 3:
                            parameters[env_index] = float(event["delay"])
                            source_step = max(0, step - int(event["delay"]))
                            executed[env_index] = teacher_history[source_step][env_index]
                        elif kind == 4:
                            parameters[env_index] = float(event["scale"])
                            executed[env_index] *= float(event["scale"])
                    executed = np.clip(executed, action_low_np, action_high_np)
                    next_obs, reward, _terminated, _truncated, info = env.step(
                        torch.from_numpy(executed).to(device)
                    )
                    reward_np = _numpy(reward).reshape(-1).astype(np.float32)
                    step_success = (
                        _numpy(info["success"]).reshape(-1).astype(bool)
                        if "success" in info
                        else np.zeros(batch_size, dtype=bool)
                    )
                    completion = recovery_pending & step_success & ~recovered
                    recovered |= completion
                    recovery_pending &= ~completion
                    success_once |= step_success
                    for env_index in range(batch_size):
                        row = buffers[env_index]
                        row["rgb"].append(rgb[env_index])
                        row["states"].append(states[env_index])
                        row["proprioception"].append(states[env_index, :21])
                        row["executed_actions"].append(executed[env_index])
                        row["teacher_actions"].append(teacher_np[env_index])
                        row["perturbation_type"].append(int(kinds[env_index]))
                        row["burst_id"].append(int(burst_ids[env_index]))
                        row["perturbation_parameter"].append(float(parameters[env_index]))
                        row["perturbation_active"].append(bool(active[env_index]))
                        row["perturbation_start"].append(bool(starts[env_index]))
                        row["perturbation_end"].append(bool(ends[env_index]))
                        row["recovery_active"].append(bool(recovery_pending[env_index]))
                        row["recovery_completion"].append(bool(completion[env_index]))
                        row["reward"].append(float(reward_np[env_index]))
                        row["success"].append(bool(step_success[env_index]))
                    previous_executed = executed
                    obs = next_obs
            finally:
                env.close()

            for env_index, row in enumerate(buffers):
                group = h5.create_group(f"episode_{completed + env_index:05d}")
                group.attrs["seed"] = reset_seeds[env_index]
                group.attrs["success"] = bool(success_once[env_index])
                group.attrs["recovered_after_final_burst"] = bool(recovered[env_index])
                group.attrs["burst_count"] = len(schedules[env_index])
                for name, values in row.items():
                    array = np.asarray(values)
                    compression = "gzip" if name == "rgb" else "lzf"
                    group.create_dataset(name, data=array, compression=compression)
                burst_group = group.create_group("bursts")
                events = schedules[env_index]
                burst_group.create_dataset(
                    "start", data=np.asarray([event["start"] for event in events], dtype=np.int16)
                )
                burst_group.create_dataset(
                    "end", data=np.asarray([event["end"] for event in events], dtype=np.int16)
                )
                burst_group.create_dataset(
                    "type", data=np.asarray([event["kind"] for event in events], dtype=np.int8)
                )
                burst_group.create_dataset(
                    "bias_fraction",
                    data=np.asarray([event["bias_fraction"] for event in events], dtype=np.float32),
                )
                burst_group.create_dataset(
                    "bias_direction",
                    data=np.stack([event["bias_direction"] for event in events]),
                )
                burst_group.create_dataset(
                    "delay", data=np.asarray([event["delay"] for event in events], dtype=np.int8)
                )
                burst_group.create_dataset(
                    "scale", data=np.asarray([event["scale"] for event in events], dtype=np.float32)
                )
                successful_episodes += int(success_once[env_index])
                recovered_episodes += int(recovered[env_index])
                perturbation_steps += int(np.sum(row["perturbation_active"]))
                recovery_steps += int(np.sum(row["recovery_active"]))
                progress.update(1)
            completed += batch_size
        meta.attrs["episodes"] = completed
        meta.attrs["successful_episodes"] = successful_episodes
        meta.attrs["recovered_episodes"] = recovered_episodes
        meta.attrs["perturbation_steps"] = perturbation_steps
        meta.attrs["recovery_steps"] = recovery_steps
    progress.close()
    tmp_path.replace(output_path)
    console.print(f"Wrote Pre-RL Phase D recovery dataset: {output_path}")
    return output_path


@torch.inference_mode()
def prepare_pre_rl_phase_d_features(
    config: Config,
    episodes: int | None = None,
    force: bool = False,
) -> Path:
    requested_episodes = int(episodes or config.get("pre_rl.phase_d.dataset_episodes", 1000))
    source_path = collect_pre_rl_phase_d_recovery_dataset(
        config,
        episodes=requested_episodes,
        force=False,
    )
    output_path = (
        config.path_value("paths.incremental_data_dir")
        / f"pre_rl_phase_d_recovery_dino_{requested_episodes}.h5"
    )
    if output_path.exists() and not force:
        console.print(f"Pre-RL Phase D DINO dataset exists: {output_path}")
        return output_path
    device = default_device()
    extractor = _phase4_dino_from_config(config, device)
    dino_batch_size = int(config.get("dino.batch_size", 64))
    episode_batch_size = int(config.get("pre_rl.phase_d.feature_episode_batch", 8))
    tmp_path = output_path.with_suffix(".tmp.h5")
    if tmp_path.exists():
        tmp_path.unlink()
    progress = trange(requested_episodes, desc="encode Phase D recovery RGB")
    with h5py.File(source_path, "r") as source, h5py.File(tmp_path, "w") as target:
        source_keys = sorted(key for key in source if key.startswith("episode_"))
        if len(source_keys) < requested_episodes:
            raise ValueError(
                f"{source_path} contains {len(source_keys)} episodes, "
                f"requires {requested_episodes}"
            )
        meta = target.create_group("meta")
        for key, value in _runtime_metadata(config).items():
            meta.attrs[key] = value
        meta.attrs["source_h5"] = str(source_path)
        meta.attrs["episodes"] = requested_episodes
        meta.attrs["dino_model"] = str(config.get("dino.model_name"))
        meta.attrs["dino_feature_type"] = str(config.get("dino.feature_type", "spatial"))
        meta.attrs["dino_spatial_pool"] = int(config.get("dino.spatial_pool", 4))
        for start in range(0, requested_episodes, episode_batch_size):
            keys = source_keys[start : start + episode_batch_size]
            rgbs = [np.asarray(source[key]["rgb"], dtype=np.uint8) for key in keys]
            lengths = [len(rgb) for rgb in rgbs]
            all_rgb = np.concatenate(rgbs, axis=0)
            all_features = np.concatenate(
                [extractor.encode_batch(chunk) for chunk in batched(all_rgb, dino_batch_size)],
                axis=0,
            ).astype(np.float32)
            offset = 0
            for key, length in zip(keys, lengths, strict=True):
                source_group = source[key]
                group = target.create_group(key)
                group.create_dataset(
                    "dino",
                    data=all_features[offset : offset + length],
                    compression="lzf",
                )
                offset += length
                for name in (
                    "proprioception",
                    "states",
                    "executed_actions",
                    "teacher_actions",
                    "perturbation_type",
                    "perturbation_active",
                    "burst_id",
                    "recovery_active",
                    "recovery_completion",
                    "reward",
                    "success",
                ):
                    group.create_dataset(
                        name,
                        data=np.asarray(source_group[name]),
                        compression="lzf",
                    )
                for attr_name, attr_value in source_group.attrs.items():
                    group.attrs[attr_name] = attr_value
                progress.update(1)
    progress.close()
    tmp_path.replace(output_path)
    console.print(f"Wrote Pre-RL Phase D DINO dataset: {output_path}")
    return output_path


def create_pre_rl_phase_d_manifests(
    config: Config,
    force: bool = False,
) -> Path:
    recovery_episodes = int(config.get("pre_rl.phase_d.dataset_episodes", 1000))
    recovery_path = prepare_pre_rl_phase_d_features(
        config,
        episodes=recovery_episodes,
        force=False,
    )
    clean_path = _phase4_prepared_path(config)
    budget = int(config.get("pre_rl.phase_d.transition_budget", 80_000))
    recovery_train_episodes = int(config.get("pre_rl.phase_d.recovery_train_episodes", 800))
    seed = int(config.get("pre_rl.phase_d.manifest_seed", 1_810_000))
    output_path = config.path_value("paths.incremental_data_dir") / "pre_rl_phase_d_manifests.h5"
    if output_path.exists() and not force:
        console.print(f"Pre-RL Phase D manifests exist: {output_path}")
        return output_path
    rng = np.random.default_rng(seed)

    with h5py.File(clean_path, "r") as clean, h5py.File(recovery_path, "r") as recovery:
        clean_keys = sorted(key for key in clean if key.startswith("episode_"))
        recovery_keys = sorted(key for key in recovery if key.startswith("episode_"))
        clean_train_keys = clean_keys[: int(config.get("incremental.phase4.train_episodes", 1800))]
        clean_validation_keys = clean_keys[-int(config.get("incremental.phase4.validation_episodes", 200)) :]
        recovery_train_keys = recovery_keys[:recovery_train_episodes]
        recovery_validation_keys = recovery_keys[recovery_train_episodes:]

        def all_indices(h5: h5py.File, keys: list[str], dataset: str) -> np.ndarray:
            rows = [
                np.column_stack(
                    [
                        np.full(len(h5[key][dataset]), int(key.split("_")[-1]), dtype=np.int32),
                        np.arange(len(h5[key][dataset]), dtype=np.int32),
                    ]
                )
                for key in keys
            ]
            return np.concatenate(rows, axis=0)

        clean_indices = all_indices(clean, clean_train_keys, "actions")
        recovery_indices = all_indices(recovery, recovery_train_keys, "teacher_actions")
        off_nominal_mask = np.concatenate(
            [
                np.asarray(recovery[key]["perturbation_active"], dtype=bool)
                | np.asarray(recovery[key]["recovery_active"], dtype=bool)
                for key in recovery_train_keys
            ]
        )
        off_nominal_indices = recovery_indices[off_nominal_mask]
        if len(clean_indices) < budget or len(off_nominal_indices) < 50_000:
            raise ValueError(
                f"Insufficient Phase D candidates: clean={len(clean_indices)}, "
                f"off_nominal={len(off_nominal_indices)}"
            )
        compositions = {
            "clean": (budget, 0),
            "mixed_25": (int(0.75 * budget), budget - int(0.75 * budget)),
            "mixed_50": (budget // 2, budget - budget // 2),
            "recovery_heavy": (budget - 50_000, 50_000),
        }
        tmp_path = output_path.with_suffix(".tmp.h5")
        if tmp_path.exists():
            tmp_path.unlink()
        with h5py.File(tmp_path, "w") as target:
            meta = target.create_group("meta")
            meta.attrs["clean_source"] = str(clean_path)
            meta.attrs["recovery_source"] = str(recovery_path)
            meta.attrs["transition_budget"] = budget
            meta.attrs["manifest_seed"] = seed
            meta.attrs["recovery_label_views"] = "behavior:executed_actions;query:teacher_actions"
            for name, (clean_count, recovery_count) in compositions.items():
                group = target.create_group(name)
                selected_clean = clean_indices[
                    rng.choice(len(clean_indices), size=clean_count, replace=False)
                ]
                selected_recovery = (
                    off_nominal_indices[
                        rng.choice(
                            len(off_nominal_indices),
                            size=recovery_count,
                            replace=False,
                        )
                    ]
                    if recovery_count
                    else np.empty((0, 2), dtype=np.int32)
                )
                group.create_dataset("clean_episode_timestep", data=selected_clean)
                group.create_dataset("recovery_episode_timestep", data=selected_recovery)
                group.attrs["clean_transitions"] = clean_count
                group.attrs["recovery_transitions"] = recovery_count
                group.attrs["total_transitions"] = clean_count + recovery_count
            validation = target.create_group("validation")
            validation.create_dataset(
                "clean_episode_timestep",
                data=all_indices(clean, clean_validation_keys, "actions"),
            )
            validation.create_dataset(
                "recovery_episode_timestep",
                data=all_indices(recovery, recovery_validation_keys, "teacher_actions"),
            )
        tmp_path.replace(output_path)
    console.print(f"Wrote Pre-RL Phase D manifests: {output_path}")
    return output_path


def _load_pre_rl_phase_d_rows(
    manifest_path: Path,
    variant: str,
    label_view: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    if label_view not in {"query", "behavior"}:
        raise ValueError(f"Unknown Phase D label view: {label_view}")
    with h5py.File(manifest_path, "r") as manifest:
        clean_path = Path(str(manifest["meta"].attrs["clean_source"]))
        recovery_path = Path(str(manifest["meta"].attrs["recovery_source"]))
        group = manifest[variant]
        clean_rows = np.asarray(group["clean_episode_timestep"], dtype=np.int32)
        recovery_rows = np.asarray(group["recovery_episode_timestep"], dtype=np.int32)
    total = len(clean_rows) + len(recovery_rows)
    frames = np.empty((total, 6549), dtype=np.float32)
    targets = np.empty((total, 3), dtype=np.float32)
    previous = np.zeros((total, 3), dtype=np.float32)
    cursor = 0

    def read_source(
        path: Path,
        rows: np.ndarray,
        recovery: bool,
        start: int,
    ) -> int:
        position = start
        with h5py.File(path, "r") as h5:
            for episode_index in np.unique(rows[:, 0]):
                timesteps = rows[rows[:, 0] == episode_index, 1]
                key = (
                    f"episode_{int(episode_index):05d}"
                    if recovery
                    else f"episode_{int(episode_index):04d}"
                )
                group = h5[key]
                count = len(timesteps)
                dino = np.asarray(group["dino"], dtype=np.float32)
                proprio_name = "proprioception" if recovery else "proprio"
                proprio = np.asarray(group[proprio_name], dtype=np.float32)
                frames[position : position + count] = np.concatenate(
                    [dino[timesteps], proprio[timesteps]], axis=-1
                )
                action_name = (
                    "teacher_actions"
                    if recovery and label_view == "query"
                    else "executed_actions"
                    if recovery
                    else "actions"
                )
                actions = np.asarray(group[action_name], dtype=np.float32)
                targets[position : position + count] = actions[timesteps]
                previous_source = (
                    np.asarray(group["executed_actions"], dtype=np.float32)
                    if recovery
                    else np.asarray(group["actions"], dtype=np.float32)
                )
                valid_previous = timesteps > 0
                previous[position : position + count][valid_previous] = previous_source[
                    timesteps[valid_previous] - 1
                ]
                position += count
        return position

    cursor = read_source(clean_path, clean_rows, False, cursor)
    cursor = read_source(recovery_path, recovery_rows, True, cursor)
    if cursor != total:
        raise RuntimeError(f"Loaded {cursor} Phase D rows, expected {total}")
    metadata = {
        "variant": variant,
        "label_view": label_view,
        "clean_source": str(clean_path),
        "recovery_source": str(recovery_path),
        "clean_transitions": int(len(clean_rows)),
        "recovery_transitions": int(len(recovery_rows)),
        "total_transitions": total,
    }
    return frames, previous, targets, metadata


def _load_pre_rl_phase_d_validation(
    manifest_path: Path,
    split: str,
    label_view: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if split not in {"clean", "recovery"}:
        raise ValueError(f"Unknown Phase D validation split: {split}")
    temporary_variant = f"__validation_{split}"
    with h5py.File(manifest_path, "r+") as manifest:
        if temporary_variant in manifest:
            del manifest[temporary_variant]
        group = manifest.create_group(temporary_variant)
        source_name = f"{split}_episode_timestep"
        rows = np.asarray(manifest["validation"][source_name], dtype=np.int32)
        if split == "clean":
            group.create_dataset("clean_episode_timestep", data=rows)
            group.create_dataset(
                "recovery_episode_timestep", data=np.empty((0, 2), dtype=np.int32)
            )
        else:
            group.create_dataset("clean_episode_timestep", data=np.empty((0, 2), dtype=np.int32))
            group.create_dataset("recovery_episode_timestep", data=rows)
    try:
        frames, previous, targets, _metadata = _load_pre_rl_phase_d_rows(
            manifest_path,
            temporary_variant,
            label_view,
        )
    finally:
        with h5py.File(manifest_path, "r+") as manifest:
            del manifest[temporary_variant]
    return frames, previous, targets


def train_pre_rl_phase_d_visual_bc(
    config: Config,
    variant: str,
    label_view: str = "query",
    seed: int = 0,
    force: bool = False,
) -> Path:
    valid_variants = {"clean", "mixed_25", "mixed_50", "recovery_heavy"}
    if variant not in valid_variants:
        raise ValueError(f"Unknown Phase D dataset variant: {variant}")
    manifest_path = create_pre_rl_phase_d_manifests(config, force=False)
    artifact_dir = ensure_dir(
        config.path_value("paths.incremental_artifact_dir")
        / "pre_rl"
        / "phase_d"
        / "visual_bc"
        / variant
        / label_view
        / f"seed{seed}"
    )
    checkpoint_path = artifact_dir / "policy.pt"
    if checkpoint_path.exists() and not force:
        console.print(f"Pre-RL Phase D visual BC exists: {checkpoint_path}")
        return checkpoint_path
    set_seed(seed)
    frames, previous, targets, data_metadata = _load_pre_rl_phase_d_rows(
        manifest_path,
        variant,
        label_view,
    )
    clean_val = _load_pre_rl_phase_d_validation(manifest_path, "clean", label_view)
    recovery_val = _load_pre_rl_phase_d_validation(manifest_path, "recovery", label_view)
    frame_norm = Standardizer.fit(frames)
    action_norm = Standardizer.fit(targets)
    frames -= frame_norm.mean
    frames /= frame_norm.std
    previous = action_norm.transform(previous)
    targets = action_norm.transform(targets)
    conditions = np.concatenate([frames, previous], axis=-1).astype(np.float32)
    del frames, previous
    def normalize_validation(
        rows: tuple[np.ndarray, np.ndarray, np.ndarray],
    ) -> tuple[np.ndarray, np.ndarray]:
        val_frames, val_previous, val_targets = rows
        val_cond = np.concatenate(
            [frame_norm.transform(val_frames), action_norm.transform(val_previous)],
            axis=-1,
        ).astype(np.float32)
        return val_cond, val_targets

    clean_val_normalized = normalize_validation(clean_val)
    recovery_val_normalized = normalize_validation(recovery_val)
    del clean_val, recovery_val
    dataset = TensorDataset(torch.from_numpy(conditions), torch.from_numpy(targets))
    batch_size = int(config.get("pre_rl.phase_d.visual_bc_batch_size", 512))
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    hidden_dim = int(config.get("incremental.phase4.hidden_dim", 512))
    model = _make_phase4_policy("concat", conditions.shape[-1], 1, 3, hidden_dim).to(
        default_device()
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config.get("pre_rl.phase_d.visual_bc_lr", 3e-4)),
    )
    epochs = int(config.get("pre_rl.phase_d.visual_bc_epochs", 50))
    device = default_device()
    best_state = None
    best_recovery_mae = float("inf")
    history = []

    def validation_metrics(rows: tuple[np.ndarray, np.ndarray]) -> dict[str, Any]:
        val_cond, val_targets = rows
        predictions = []
        with torch.inference_mode():
            for start in range(0, len(val_cond), 2048):
                x = torch.from_numpy(val_cond[start : start + 2048]).to(device).float()
                predictions.append(action_norm.inverse(model(x[:, None, :]).cpu().numpy()))
        return _action_regression_metrics(np.concatenate(predictions), val_targets)

    for epoch in trange(1, epochs + 1, desc=f"train Phase D visual BC {variant}"):
        model.train()
        train_loss = 0.0
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            prediction = model(x[:, None, :])
            loss = torch.mean((prediction - y) ** 2)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            train_loss += float(loss.detach().cpu()) * len(x)
        model.eval()
        recovery_metrics = validation_metrics(recovery_val_normalized)
        clean_metrics = validation_metrics(clean_val_normalized)
        history.append(
            {
                "epoch": epoch,
                "train_mse": train_loss / len(dataset),
                "clean_validation_mae": clean_metrics["mae"],
                "recovery_validation_mae": recovery_metrics["mae"],
            }
        )
        if recovery_metrics["mae"] < best_recovery_mae:
            best_recovery_mae = recovery_metrics["mae"]
            best_state = copy.deepcopy(model.state_dict())
    if best_state is None:
        raise RuntimeError("Phase D visual BC training produced no checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    final_clean = validation_metrics(clean_val_normalized)
    final_recovery = validation_metrics(recovery_val_normalized)
    payload = {
        "phase": "D6",
        "method": "direct_visual_bc",
        "variant": variant,
        "label_view": label_view,
        "seed": seed,
        "model": best_state,
        "architecture": "concat",
        "history": 1,
        "step_dim": int(conditions.shape[-1]),
        "action_dim": 3,
        "hidden_dim": hidden_dim,
        "frame_norm": frame_norm.state_dict(),
        "action_norm": action_norm.state_dict(),
        "data": data_metadata,
        "clean_validation_metrics": final_clean,
        "recovery_validation_metrics": final_recovery,
        "history_rows": history,
        "metadata": _runtime_metadata(config),
    }
    torch.save(payload, checkpoint_path)
    write_json(
        artifact_dir / "metrics.json",
        {key: value for key, value in payload.items() if key not in {"model", "frame_norm", "action_norm"}},
    )
    console.print(f"Wrote Pre-RL Phase D visual BC: {checkpoint_path}")
    return checkpoint_path


@torch.inference_mode()
def _evaluate_pre_rl_phase_d_visual_bc_distribution(
    config: Config,
    checkpoint: dict[str, Any],
    episodes: int,
    seed_start: int,
    disturbed: bool,
) -> dict[str, Any]:
    device = default_device()
    model = _make_phase4_policy(
        str(checkpoint["architecture"]),
        int(checkpoint["step_dim"]),
        int(checkpoint["history"]),
        int(checkpoint["action_dim"]),
        int(checkpoint["hidden_dim"]),
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    frame_norm = Standardizer.from_state_dict(checkpoint["frame_norm"])
    action_norm = Standardizer.from_state_dict(checkpoint["action_norm"])
    dino = _phase4_dino_from_config(config, device)
    max_num_envs = min(int(config.get("pre_rl.phase_d.eval_num_envs", 32)), episodes)
    max_steps = int(config.get("env_max_episode_steps", 100))
    rng = np.random.default_rng(seed_start + (10_000 if disturbed else 0))
    successes: list[float] = []
    final_rewards: list[float] = []
    max_rewards: list[float] = []
    recovered: list[float] = []
    recovery_times: list[int] = []
    reward_drops: list[float] = []
    family_recovered: dict[int, list[float]] = {
        key: [] for key in PRE_RL_PHASE_D_PERTURBATIONS
    }
    progress = trange(episodes, desc=f"Phase D visual BC {'disturbed' if disturbed else 'clean'}")
    for batch_start in range(0, episodes, max_num_envs):
        batch_size = min(max_num_envs, episodes - batch_start)
        env = _phase4_make_visual_env(config, batch_size)
        reset_seeds = [seed_start + batch_start + index for index in range(batch_size)]
        obs, _info = env.reset(seed=reset_seeds)
        schedules = (
            _pre_rl_phase_d_schedule(rng, batch_size, max_steps, 1, 1)
            if disturbed
            else [[] for _ in range(batch_size)]
        )
        if disturbed:
            for events in schedules:
                event = events[0]
                duration = int(event["end"] - event["start"])
                event["start"] = int(rng.integers(15, max_steps - duration - 20))
                event["end"] = event["start"] + duration
        action_low_np = np.asarray(env.single_action_space.low, dtype=np.float32)
        action_high_np = np.asarray(env.single_action_space.high, dtype=np.float32)
        action_range = action_high_np - action_low_np
        previous_executed = np.zeros((batch_size, 3), dtype=np.float32)
        policy_history: list[np.ndarray] = []
        bias_noise = np.zeros_like(previous_executed)
        success_once = np.zeros(batch_size, dtype=bool)
        recovered_batch = np.zeros(batch_size, dtype=bool)
        recovery_time = np.full(batch_size, -1, dtype=np.int32)
        batch_final_reward = np.zeros(batch_size, dtype=np.float32)
        batch_max_reward = np.full(batch_size, -np.inf, dtype=np.float32)
        pre_burst_max_reward = np.full(batch_size, -np.inf, dtype=np.float32)
        post_burst_min_reward = np.full(batch_size, np.inf, dtype=np.float32)
        try:
            for step in range(max_steps):
                frame = _phase4_frame_inputs(
                    obs,
                    dino,
                    int(config.get("dino.batch_size", 64)),
                )
                condition = np.concatenate(
                    [frame_norm.transform(frame), action_norm.transform(previous_executed)],
                    axis=-1,
                )
                pred_norm = model(torch.from_numpy(condition[:, None, :]).to(device).float())
                policy_action = action_norm.inverse(pred_norm.cpu().numpy()).astype(np.float32)
                policy_history.append(policy_action.copy())
                executed = policy_action.copy()
                for env_index, events in enumerate(schedules):
                    if not events:
                        continue
                    event = events[0]
                    if not event["start"] <= step < event["end"]:
                        continue
                    kind = int(event["kind"])
                    if kind == 1:
                        bias_noise[env_index] = (
                            0.7 * bias_noise[env_index]
                            + 0.3
                            * rng.normal(0.0, 0.01, size=3).astype(np.float32)
                            * action_range
                        )
                        executed[env_index] += (
                            event["bias_fraction"]
                            * action_range
                            * event["bias_direction"]
                            + bias_noise[env_index]
                        )
                    elif kind == 2:
                        executed[env_index] = previous_executed[env_index]
                    elif kind == 3:
                        source_step = max(0, step - int(event["delay"]))
                        executed[env_index] = policy_history[source_step][env_index]
                    else:
                        executed[env_index] *= float(event["scale"])
                executed = np.clip(executed, action_low_np, action_high_np)
                obs, reward, _terminated, _truncated, info = env.step(
                    torch.from_numpy(executed).to(device)
                )
                reward_np = _numpy(reward).reshape(-1).astype(np.float32)
                batch_final_reward = reward_np
                batch_max_reward = np.maximum(batch_max_reward, reward_np)
                step_success = (
                    _numpy(info["success"]).reshape(-1).astype(bool)
                    if "success" in info
                    else np.zeros(batch_size, dtype=bool)
                )
                success_once |= step_success
                if disturbed:
                    for env_index, events in enumerate(schedules):
                        event = events[0]
                        if step < event["start"]:
                            pre_burst_max_reward[env_index] = max(
                                pre_burst_max_reward[env_index], reward_np[env_index]
                            )
                        if step >= event["end"]:
                            post_burst_min_reward[env_index] = min(
                                post_burst_min_reward[env_index], reward_np[env_index]
                            )
                            if step_success[env_index] and not recovered_batch[env_index]:
                                recovered_batch[env_index] = True
                                recovery_time[env_index] = step - int(event["end"]) + 1
                previous_executed = executed
        finally:
            env.close()
        successes.extend(success_once.astype(np.float32).tolist())
        final_rewards.extend(batch_final_reward.tolist())
        max_rewards.extend(batch_max_reward.tolist())
        if disturbed:
            recovered.extend(recovered_batch.astype(np.float32).tolist())
            recovery_times.extend(recovery_time[recovery_time >= 0].tolist())
            finite = np.isfinite(pre_burst_max_reward) & np.isfinite(post_burst_min_reward)
            reward_drops.extend(
                np.maximum(
                    0.0,
                    pre_burst_max_reward[finite] - post_burst_min_reward[finite],
                ).tolist()
            )
            for env_index, events in enumerate(schedules):
                family_recovered[int(events[0]["kind"])].append(
                    float(recovered_batch[env_index])
                )
        progress.update(batch_size)
    progress.close()
    return {
        "episodes": episodes,
        "seed_start": seed_start,
        "success": float(np.mean(successes)),
        "success_wilson_95": _wilson_interval(float(np.mean(successes)), episodes),
        "final_reward": float(np.mean(final_rewards)),
        "max_reward": float(np.mean(max_rewards)),
        "recovery_success": float(np.mean(recovered)) if recovered else None,
        "mean_recovery_steps_when_recovered": (
            float(np.mean(recovery_times)) if recovery_times else None
        ),
        "mean_reward_drop": float(np.mean(reward_drops)) if reward_drops else None,
        "recovery_success_by_family": {
            PRE_RL_PHASE_D_PERTURBATIONS[key]: float(np.mean(values))
            for key, values in family_recovered.items()
            if values
        },
    }


def evaluate_pre_rl_phase_d_visual_bc(
    config: Config,
    variant: str,
    label_view: str = "query",
    seed: int = 0,
    episodes: int | None = None,
    force: bool = False,
) -> Path:
    checkpoint_path = train_pre_rl_phase_d_visual_bc(
        config,
        variant=variant,
        label_view=label_view,
        seed=seed,
        force=False,
    )
    eval_episodes = int(episodes or config.get("pre_rl.phase_d.eval_episodes", 100))
    result_dir = ensure_dir(
        config.path_value("paths.incremental_results_dir")
        / "pre_rl"
        / "phase_d"
        / "visual_bc"
        / variant
        / label_view
        / f"seed{seed}"
    )
    output_path = result_dir / f"eval_{eval_episodes}.json"
    if output_path.exists() and not force:
        console.print(f"Pre-RL Phase D visual BC evaluation exists: {output_path}")
        return output_path
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    seed_start = int(config.get("pre_rl.phase_d.eval_seed_start", 1_820_000))
    clean = _evaluate_pre_rl_phase_d_visual_bc_distribution(
        config,
        checkpoint,
        eval_episodes,
        seed_start,
        disturbed=False,
    )
    disturbed = _evaluate_pre_rl_phase_d_visual_bc_distribution(
        config,
        checkpoint,
        eval_episodes,
        seed_start,
        disturbed=True,
    )
    payload = {
        "phase": "D6-D7",
        "method": "direct_visual_bc",
        "variant": variant,
        "label_view": label_view,
        "seed": seed,
        "checkpoint": str(checkpoint_path),
        "data": checkpoint["data"],
        "clean_validation_metrics": checkpoint["clean_validation_metrics"],
        "recovery_validation_metrics": checkpoint["recovery_validation_metrics"],
        "clean_evaluation": clean,
        "disturbed_evaluation": disturbed,
        "metadata": _runtime_metadata(config),
    }
    write_json(output_path, payload)
    console.print(payload)
    return output_path


def _pre_rl_rank_correlation(x: np.ndarray, y: np.ndarray) -> float:
    x_rank = np.empty_like(x, dtype=np.float64)
    y_rank = np.empty_like(y, dtype=np.float64)
    x_rank[np.argsort(x)] = np.arange(len(x), dtype=np.float64)
    y_rank[np.argsort(y)] = np.arange(len(y), dtype=np.float64)
    return float(np.corrcoef(x_rank, y_rank)[0, 1])


@torch.inference_mode()
def analyze_pre_rl_phase_e_geometry(config: Config) -> Path:
    import csv
    import matplotlib.pyplot as plt

    probe_path = collect_phase6_probe_dataset(config, force=False)
    with np.load(probe_path) as probe:
        inputs = np.asarray(probe["inputs"], dtype=np.float32)
        next_inputs = np.asarray(probe["next_inputs"], dtype=np.float32)
        labels = np.asarray(probe["labels"], dtype=np.float32)
        next_labels = np.asarray(probe["next_labels"], dtype=np.float32)
        actions = np.asarray(probe["actions"], dtype=np.float32)
        contact = np.asarray(probe["contact"], dtype=np.float32).reshape(-1)
    device = default_device()
    rng = np.random.default_rng(int(config.get("pre_rl.phase_e.seed", 1_900_000)))
    variants = [
        ("raw_dino_proprio", None),
        (
            "ae_recon_z256",
            config.path_value("paths.incremental_artifact_dir")
            / "phase6"
            / "ae_recon_z256"
            / "seed0"
            / "encoder.pt",
        ),
        (
            "vae_recon_z256",
            config.path_value("paths.incremental_artifact_dir")
            / "phase6"
            / "vae_recon_z256"
            / "seed0"
            / "encoder.pt",
        ),
    ]
    pair_count = int(config.get("pre_rl.phase_e.pair_samples", 20_000))
    left = rng.integers(0, len(inputs), size=pair_count)
    right = rng.integers(0, len(inputs), size=pair_count)
    object_xy_distance = np.linalg.norm(labels[left, :2] - labels[right, :2], axis=-1)
    yaw_distance = np.abs(
        _wrap_angle(labels[left, 2] - labels[right, 2])
    )
    tcp_distance = np.linalg.norm(labels[left, 6:8] - labels[right, 6:8], axis=-1)
    action_distance = np.linalg.norm(actions[left] - actions[right], axis=-1)
    rows = []
    root = ensure_dir(config.path_value("paths.incremental_results_dir") / "pre_rl" / "phase_e")
    for name, checkpoint_path in variants:
        decoder = None
        checkpoint: dict[str, Any] | None = None
        if checkpoint_path is None:
            representation = inputs
            next_representation = next_inputs
            kl_mean = None
            reconstruction_mse = None
        else:
            encoder, checkpoint = _load_phase6_encoder(checkpoint_path, device)
            frame_norm = Standardizer.from_state_dict(checkpoint["frame_norm"])

            def encode(values: np.ndarray) -> np.ndarray:
                outputs = []
                normalized = frame_norm.transform(values)
                for start in range(0, len(values), 2048):
                    outputs.append(
                        encoder(
                            torch.from_numpy(normalized[start : start + 2048])
                            .to(device)
                            .float()
                        )
                        .cpu()
                        .numpy()
                    )
                return np.concatenate(outputs).astype(np.float32)

            representation = encode(inputs)
            next_representation = encode(next_inputs)
            kl_mean = None
            if isinstance(encoder, VariationalObservationEncoder):
                normalized = frame_norm.transform(inputs)
                kl_values = []
                for start in range(0, len(inputs), 2048):
                    mean, logvar = encoder.encode_stats(
                        torch.from_numpy(normalized[start : start + 2048]).to(device).float()
                    )
                    kl_values.append(
                        (-0.5 * (1.0 + logvar - mean.square() - logvar.exp()).mean(dim=-1))
                        .cpu()
                        .numpy()
                    )
                kl_mean = float(np.mean(np.concatenate(kl_values)))
            reconstruction_mse = None
            if checkpoint.get("decoder") is not None:
                decoder = MLP(
                    int(checkpoint["latent_dim"]),
                    int(checkpoint["input_dim"]),
                    int(checkpoint["hidden_dim"]),
                    depth=3,
                ).to(device)
                decoder.load_state_dict(checkpoint["decoder"])
                decoder.eval()
                interpolation_indices = rng.choice(len(inputs), size=(256, 2), replace=False)
                alpha = np.asarray([0.25, 0.5, 0.75], dtype=np.float32)
                z0 = representation[interpolation_indices[:, 0]]
                z1 = representation[interpolation_indices[:, 1]]
                interpolated = (
                    z0[:, None, :] * (1.0 - alpha[None, :, None])
                    + z1[:, None, :] * alpha[None, :, None]
                ).reshape(-1, representation.shape[-1])
                decoded = decoder(torch.from_numpy(interpolated).to(device).float()).cpu().numpy()
                normalized_inputs = frame_norm.transform(inputs)
                x0 = normalized_inputs[interpolation_indices[:, 0]]
                x1 = normalized_inputs[interpolation_indices[:, 1]]
                linear_target = (
                    x0[:, None, :] * (1.0 - alpha[None, :, None])
                    + x1[:, None, :] * alpha[None, :, None]
                ).reshape(decoded.shape)
                reconstruction_mse = float(np.mean((decoded - linear_target) ** 2))

        representation_std = representation.std(axis=0)
        standardized = (representation - representation.mean(axis=0)) / np.maximum(
            representation_std, 1e-6
        )
        next_standardized = (next_representation - representation.mean(axis=0)) / np.maximum(
            representation_std, 1e-6
        )
        latent_pair_distance = np.linalg.norm(
            standardized[left] - standardized[right], axis=-1
        )
        transition_distance = np.linalg.norm(next_standardized - standardized, axis=-1)
        object_transition = np.linalg.norm(next_labels[:, :2] - labels[:, :2], axis=-1)
        tcp_transition = np.linalg.norm(next_labels[:, 6:8] - labels[:, 6:8], axis=-1)

        reference_count = int(config.get("pre_rl.phase_e.nearest_references", 3000))
        query_count = int(config.get("pre_rl.phase_e.nearest_queries", 500))
        reference = standardized[:reference_count]
        query_start = reference_count
        query = standardized[query_start : query_start + query_count]
        nearest_indices = []
        for start in range(0, len(query), 128):
            distances = torch.cdist(
                torch.from_numpy(query[start : start + 128]).to(device).float(),
                torch.from_numpy(reference).to(device).float(),
            )
            nearest_indices.append(torch.argmin(distances, dim=1).cpu().numpy())
        nearest = np.concatenate(nearest_indices)
        query_indices = np.arange(query_start, query_start + query_count)
        row = {
            "representation": name,
            "dimension": int(representation.shape[-1]),
            "active_dimensions_std_gt_0_1": int(np.sum(representation_std > 0.1)),
            "vae_kl_mean_per_dimension": kl_mean,
            "decoded_interpolation_linear_mse": reconstruction_mse,
            "pair_object_xy_spearman": _pre_rl_rank_correlation(
                latent_pair_distance, object_xy_distance
            ),
            "pair_object_yaw_spearman": _pre_rl_rank_correlation(
                latent_pair_distance, yaw_distance
            ),
            "pair_tcp_xy_spearman": _pre_rl_rank_correlation(
                latent_pair_distance, tcp_distance
            ),
            "pair_teacher_action_spearman": _pre_rl_rank_correlation(
                latent_pair_distance, action_distance
            ),
            "transition_object_xy_spearman": _pre_rl_rank_correlation(
                transition_distance, object_transition
            ),
            "transition_tcp_xy_spearman": _pre_rl_rank_correlation(
                transition_distance, tcp_transition
            ),
            "transition_action_effort_spearman": _pre_rl_rank_correlation(
                transition_distance, np.linalg.norm(actions, axis=-1)
            ),
            "nearest_object_xy_error_m": float(
                np.mean(np.linalg.norm(labels[query_indices, :2] - labels[nearest, :2], axis=-1))
            ),
            "nearest_object_yaw_error_rad": float(
                np.mean(np.abs(_wrap_angle(labels[query_indices, 2] - labels[nearest, 2])))
            ),
            "nearest_tcp_xy_error_m": float(
                np.mean(np.linalg.norm(labels[query_indices, 6:8] - labels[nearest, 6:8], axis=-1))
            ),
            "nearest_teacher_action_mae": float(
                np.mean(np.abs(actions[query_indices] - actions[nearest]))
            ),
            "nearest_contact_match": float(np.mean(contact[query_indices] == contact[nearest])),
        }
        rows.append(row)

    csv_path = root / "representation_geometry.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    plot_path = root / "representation_geometry.png"
    metrics = [
        "pair_object_xy_spearman",
        "pair_tcp_xy_spearman",
        "pair_teacher_action_spearman",
        "transition_action_effort_spearman",
    ]
    x = np.arange(len(metrics))
    figure, axis = plt.subplots(figsize=(9, 5.5))
    width = 0.25
    for index, row in enumerate(rows):
        axis.bar(
            x + (index - 1) * width,
            [row[metric] for metric in metrics],
            width=width,
            label=row["representation"],
        )
    axis.set_xticks(x, [metric.replace("_spearman", "").replace("_", " ") for metric in metrics])
    axis.set_ylabel("Spearman correlation")
    axis.set_title("Representation geometry versus physical/control distance")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(plot_path, dpi=180)
    plt.close(figure)
    payload = {
        "phase": "E3",
        "experiment": "representation_goal_geometry",
        "probe_dataset": str(probe_path),
        "pair_samples": pair_count,
        "rows": rows,
        "csv": str(csv_path),
        "plot": str(plot_path),
        "metadata": _runtime_metadata(config),
    }
    output_path = root / "representation_geometry.json"
    write_json(output_path, payload)
    console.print(payload)
    return output_path


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
    eval_episodes = int(episodes or config.get("incremental.phase7.dagger_episodes", 200))
    output_path = artifact_dir / f"oracle_branch_dagger_iter{iteration}_e{eval_episodes}.npz"
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
    encoder, encoder_checkpoint = _load_phase6_encoder(
        Path(checkpoint["encoder_checkpoint"]), device
    )
    frame_norm = Standardizer.from_state_dict(encoder_checkpoint["frame_norm"])
    action_norm = Standardizer.from_state_dict(checkpoint["action_norm"])
    dino = _phase4_dino_from_config(config, device)
    seed_start = int(config.get("incremental.phase7.dagger_seed", 740000)) + 1000 * iteration
    teacher = load_ppo_agent(_rl_paths(config).best, device)
    action_dim = int(checkpoint["action_dim"])
    zero_action_norm = action_norm.transform(np.zeros((1, action_dim), dtype=np.float32))[0]
    cond_rows = []
    teacher_action_rows = []
    successes: list[float] = []
    replay_errors: list[float] = []
    branch_latencies: list[float] = []
    branch_latencies_per_env: list[float] = []
    failed_replay_steps = 0
    max_episode_steps = int(config.get("env_max_episode_steps", 100))
    replay_state_tolerance = float(
        config.get("incremental.phase7.replay_branch_state_tolerance", 1e-6)
    )
    max_num_envs = min(
        int(config.get("incremental.phase7.replay_branch_num_envs", 16)),
        eval_episodes,
    )

    def make_env(num_envs: int):
        return gym.make(
            config.get("env_id"),
            obs_mode="rgb+state",
            control_mode=config.get("control_mode"),
            reward_mode="normalized_dense",
            render_mode=None,
            sim_backend=_rl_backend(config),
            num_envs=num_envs,
            reconfiguration_freq=config.get("rl.eval_reconfiguration_freq", 1),
        )

    progress = trange(eval_episodes, desc="phase7F collect coherent branch DAgger")
    for batch_start in range(0, eval_episodes, max_num_envs):
        num_envs = min(max_num_envs, eval_episodes - batch_start)
        reset_seeds = [seed_start + batch_start + i for i in range(num_envs)]
        student_env = make_env(num_envs)
        branch_env = make_env(num_envs)
        action_low_np = np.asarray(student_env.action_space.low, dtype=np.float32)
        action_high_np = np.asarray(student_env.action_space.high, dtype=np.float32)
        if action_low_np.ndim == 2:
            action_low_np = action_low_np[0]
            action_high_np = action_high_np[0]
        action_low = torch.as_tensor(action_low_np, device=device, dtype=torch.float32)
        action_high = torch.as_tensor(action_high_np, device=device, dtype=torch.float32)
        try:
            obs, _info = student_env.reset(seed=reset_seeds)
            history: list[torch.Tensor] = []
            prev_action_norm = np.repeat(zero_action_norm[None, :], num_envs, axis=0).astype(
                np.float32
            )
            active = np.ones(num_envs, dtype=bool)
            success_once = np.zeros(num_envs, dtype=bool)
            for _step in range(max_episode_steps):
                if not active.any():
                    break
                active_count = int(active.sum())
                branch_timer = Timer()
                branch_obs, _branch_info = branch_env.reset(seed=reset_seeds)
                replay_done = torch.zeros(num_envs, device=device, dtype=torch.bool)
                for action_history in history:
                    branch_obs, _branch_reward, branch_term, branch_trunc, _branch_info = (
                        branch_env.step(action_history)
                    )
                    replay_done = replay_done | torch.logical_or(branch_term, branch_trunc).view(-1)
                state_errors = torch.max(
                    torch.abs(student_env.unwrapped.get_state() - branch_env.unwrapped.get_state()),
                    dim=1,
                ).values
                state_errors_np = state_errors.detach().cpu().numpy()
                replay_errors.extend(float(x) for x in state_errors_np[active])
                replay_done_np = replay_done.detach().cpu().numpy().astype(bool)
                failed_replay_steps += int(
                    np.sum(active & (replay_done_np | (state_errors_np > replay_state_tolerance)))
                )
                for _ in range(horizon_steps):
                    teacher_action = torch.clamp(
                        teacher.actor_mean(branch_obs["state"].to(device).float()),
                        action_low,
                        action_high,
                    )
                    branch_obs, _branch_reward, branch_term, branch_trunc, _branch_info = (
                        branch_env.step(teacher_action)
                    )
                    if bool(torch.all(torch.logical_or(branch_term, branch_trunc))):
                        break
                branch_elapsed = branch_timer.elapsed()
                branch_latencies.append(branch_elapsed)
                branch_latencies_per_env.append(branch_elapsed / active_count)

                current_frame = _phase4_frame_inputs(
                    obs, dino, int(config.get("dino.batch_size", 64))
                )
                goal_frame = _phase4_frame_inputs(
                    branch_obs,
                    dino,
                    int(config.get("dino.batch_size", 64)),
                )
                frames = frame_norm.transform(np.concatenate([current_frame, goal_frame], axis=0))
                z_pair = encoder(torch.from_numpy(frames).to(device).float()).cpu().numpy()
                z = z_pair[:num_envs].astype(np.float32)
                goals = z_pair[num_envs:].astype(np.float32)
                cond = np.stack(
                    [
                        _phase7_condition(z[i], goals[i], prev_action_norm[i], goal_encoding)
                        for i in range(num_envs)
                    ],
                    axis=0,
                )
                teacher_now = (
                    torch.clamp(
                        teacher.actor_mean(obs["state"].to(device).float()),
                        action_low,
                        action_high,
                    )
                    .cpu()
                    .numpy()
                    .astype(np.float32)
                )
                cond_rows.append(cond[active].copy())
                teacher_action_rows.append(teacher_now[active].copy())

                pred_norm = model(torch.from_numpy(cond).to(device).float())
                raw_action = action_norm.inverse(pred_norm.cpu().numpy()).astype(np.float32)
                action_t = torch.from_numpy(raw_action).to(device).float()
                if bool(config.get("policy.clip_actions_to_env_space", True)):
                    action_t = torch.clamp(action_t, action_low, action_high)
                action_t[~torch.from_numpy(active).to(device)] = 0.0
                obs, _reward, terminated, truncated, info = student_env.step(action_t)
                history.append(action_t.detach().clone())
                prev_action_norm = action_norm.transform(action_t.cpu().numpy().astype(np.float32))
                if "success" in info:
                    success_once |= _numpy(info["success"]).reshape(-1).astype(bool)
                done = _numpy(torch.logical_or(terminated, truncated)).reshape(-1).astype(bool)
                newly_done = active & done
                if np.any(newly_done):
                    progress.update(int(newly_done.sum()))
                    active[newly_done] = False
            if np.any(active):
                progress.update(int(active.sum()))
            successes.extend(float(x) for x in success_once)
        finally:
            student_env.close()
            branch_env.close()
    progress.close()
    np.savez_compressed(
        output_path,
        conditions=np.concatenate(cond_rows, axis=0).astype(np.float32),
        teacher_actions=np.concatenate(teacher_action_rows, axis=0).astype(np.float32),
        collection_success=np.asarray(successes, dtype=np.float32),
        dataset_type=np.asarray("state_query_dataset"),
        semantics=np.asarray(
            "phase7 learner-visited states with exact replay local branch future goals and "
            "privileged teacher actions from the same current state"
        ),
        goal_encoding=np.asarray(goal_encoding),
        goal_dropout_prob=np.asarray(goal_dropout_prob, dtype=np.float32),
        replay_current_state_error_mean=np.asarray(
            float(np.mean(replay_errors)) if replay_errors else 0.0,
            dtype=np.float32,
        ),
        replay_current_state_error_max=np.asarray(
            float(np.max(replay_errors)) if replay_errors else 0.0,
            dtype=np.float32,
        ),
        replay_failed_step_fraction=np.asarray(
            float(failed_replay_steps / max(1, len(replay_errors))),
            dtype=np.float32,
        ),
        branch_generation_latency_s=np.asarray(
            float(np.mean(branch_latencies)) if branch_latencies else 0.0,
            dtype=np.float32,
        ),
        branch_generation_latency_per_env_s=np.asarray(
            float(np.mean(branch_latencies_per_env)) if branch_latencies_per_env else 0.0,
            dtype=np.float32,
        ),
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
    query_episodes: int | None = None,
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
    query_episode_count = int(
        query_episodes or config.get("incremental.phase7.dagger_episodes", 200)
    )
    checkpoint_path = artifact_dir / (
        f"oracle_low_level_branch_dagger_iter{iteration}_e{query_episode_count}.pt"
    )
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
        episodes=query_episodes,
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
        "query_episodes": query_episode_count,
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
        artifact_dir
        / f"oracle_low_level_branch_dagger_iter{iteration}_e{query_episode_count}_metrics.json",
        {
            "variant": variant,
            "latent_dim": latent_dim,
            "horizon_steps": horizon_steps,
            "action_chunk_steps": action_chunk_steps,
            "goal_encoding": goal_encoding,
            "goal_dropout_prob": goal_dropout_prob,
            "iteration": iteration,
            "query_episodes": query_episode_count,
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
    query_episodes: int | None = None,
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
        query_episodes=query_episodes,
        force=force,
    )
    device = default_device()
    model, checkpoint = _load_phase7_low_level_checkpoint(checkpoint_path, device)
    encoder, encoder_checkpoint = _load_phase6_encoder(
        Path(checkpoint["encoder_checkpoint"]), device
    )
    frame_norm = Standardizer.from_state_dict(encoder_checkpoint["frame_norm"])
    action_norm = Standardizer.from_state_dict(checkpoint["action_norm"])
    dino = _phase4_dino_from_config(config, device)
    eval_episodes = int(episodes or config.get("incremental.phase7.eval_episodes", 100))
    seed_start = int(config.get("incremental.phase7.eval_seed", 10000))
    oracle_frames = _phase7_collect_oracle_frames(config, dino, eval_episodes, seed_start)
    oracle_latents = _phase7_encode_oracle_frame_sequences(
        encoder, frame_norm, oracle_frames, device
    )
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


def _phase8_defaults(
    config: Config,
    latent_dim: int | None,
    variant: str | None,
    horizon_steps: int | None,
) -> tuple[int, str, int]:
    return (
        int(latent_dim or config.get("incremental.phase8.latent_dim", 256)),
        str(variant or config.get("incremental.phase8.variant", "ae_recon")),
        int(horizon_steps or config.get("incremental.phase8.horizon_steps", 2)),
    )


def _phase8_latent_cache_path(
    config: Config,
    variant: str,
    latent_dim: int,
    seed: int,
) -> Path:
    return (
        config.path_value("paths.incremental_artifact_dir")
        / "phase8"
        / f"{variant}_z{latent_dim}"
        / f"seed{seed}"
        / "causal_latent_episodes.pt"
    )


@torch.inference_mode()
def prepare_phase8_latent_episodes(
    config: Config,
    latent_dim: int | None = None,
    variant: str | None = None,
    seed: int = 0,
    force: bool = False,
) -> Path:
    latent_dim, variant, _horizon_steps = _phase8_defaults(config, latent_dim, variant, None)
    output_path = _phase8_latent_cache_path(config, variant, latent_dim, seed)
    if output_path.exists() and not force:
        console.print(f"Phase 8 causal latent cache exists: {output_path}")
        return output_path
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
    train_episodes, val_episodes, data_metadata = _load_phase6_train_episodes(config)

    def encode(episodes: list[dict[str, np.ndarray]]) -> list[dict[str, np.ndarray]]:
        encoded = []
        for episode in episodes:
            frames = frame_norm.transform(episode["frames"])
            chunks = []
            for start in range(0, len(frames), 4096):
                chunks.append(
                    encoder(torch.from_numpy(frames[start : start + 4096]).to(device).float())
                    .cpu()
                    .numpy()
                )
            encoded.append(
                {
                    "latents": np.concatenate(chunks).astype(np.float32),
                    "actions": episode["actions"].astype(np.float32),
                }
            )
        return encoded

    payload = {
        "variant": variant,
        "latent_dim": latent_dim,
        "encoder_checkpoint": str(encoder_path),
        "train": encode(train_episodes),
        "validation": encode(val_episodes),
        "data": data_metadata,
        "metadata": _runtime_metadata(config),
    }
    ensure_dir(output_path.parent)
    torch.save(payload, output_path)
    console.print(f"Wrote Phase 8 causal latent cache: {output_path}")
    return output_path


def _phase8_history_condition(
    latents: np.ndarray,
    actions: np.ndarray,
    t: int,
    history: int,
    latent_norm: Standardizer,
    action_norm: Standardizer,
    zero_action_norm: np.ndarray,
) -> np.ndarray:
    rows = []
    for offset in range(history):
        history_t = t - history + 1 + offset
        source_t = max(0, history_t)
        previous_t = history_t - 1
        previous_action = (
            action_norm.transform(actions[previous_t : previous_t + 1])[0]
            if previous_t >= 0
            else zero_action_norm
        )
        rows.append(
            np.concatenate(
                [latent_norm.transform(latents[source_t : source_t + 1])[0], previous_action]
            )
        )
    return np.concatenate(rows).astype(np.float32)


class _Phase8FutureDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        episodes: list[dict[str, np.ndarray]],
        history: int,
        horizon_steps: int,
        latent_norm: Standardizer,
        target_norm: Standardizer,
        target_mode: str,
        action_norm: Standardizer,
        length: int,
    ) -> None:
        self.episodes = [episode for episode in episodes if len(episode["actions"]) > horizon_steps]
        if not self.episodes:
            raise ValueError("No causal episodes are long enough for Phase 8")
        self.history = history
        self.horizon_steps = horizon_steps
        self.latent_norm = latent_norm
        self.target_norm = target_norm
        self.target_mode = target_mode
        self.action_norm = action_norm
        self.zero_action_norm = action_norm.transform(
            np.zeros((1, self.episodes[0]["actions"].shape[-1]), dtype=np.float32)
        )[0]
        self.length = length

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, _index: int) -> tuple[torch.Tensor, torch.Tensor]:
        episode = self.episodes[np.random.randint(0, len(self.episodes))]
        t = int(np.random.randint(0, len(episode["actions"]) - self.horizon_steps))
        condition = _phase8_history_condition(
            episode["latents"],
            episode["actions"],
            t,
            self.history,
            self.latent_norm,
            self.action_norm,
            self.zero_action_norm,
        )
        future = episode["latents"][t + self.horizon_steps]
        target_raw = future if self.target_mode == "absolute" else future - episode["latents"][t]
        target = self.target_norm.transform(target_raw[None, :])[0]
        return torch.from_numpy(condition), torch.from_numpy(target)


def _phase8_validation_samples(
    episodes: list[dict[str, np.ndarray]],
    history: int,
    horizon_steps: int,
    latent_norm: Standardizer,
    action_norm: Standardizer,
    max_samples: int,
    seed: int,
) -> dict[str, np.ndarray]:
    candidates = [
        (episode_idx, t)
        for episode_idx, episode in enumerate(episodes)
        for t in range(len(episode["actions"]) - horizon_steps)
    ]
    if not candidates:
        raise ValueError("No Phase 8 validation samples")
    rng = np.random.default_rng(seed)
    if len(candidates) > max_samples:
        chosen = rng.choice(len(candidates), size=max_samples, replace=False)
        candidates = [candidates[int(i)] for i in chosen]
    zero_action_norm = action_norm.transform(
        np.zeros((1, episodes[0]["actions"].shape[-1]), dtype=np.float32)
    )[0]
    conditions = []
    current_latents = []
    future_latents = []
    previous_actions_norm = []
    teacher_actions = []
    for episode_idx, t in candidates:
        episode = episodes[episode_idx]
        conditions.append(
            _phase8_history_condition(
                episode["latents"],
                episode["actions"],
                t,
                history,
                latent_norm,
                action_norm,
                zero_action_norm,
            )
        )
        current_latents.append(episode["latents"][t])
        future_latents.append(episode["latents"][t + horizon_steps])
        previous_actions_norm.append(
            action_norm.transform(episode["actions"][t - 1 : t])[0] if t > 0 else zero_action_norm
        )
        teacher_actions.append(episode["actions"][t])
    return {
        "conditions": np.stack(conditions).astype(np.float32),
        "current_latents": np.stack(current_latents).astype(np.float32),
        "future_latents": np.stack(future_latents).astype(np.float32),
        "previous_actions_norm": np.stack(previous_actions_norm).astype(np.float32),
        "teacher_actions": np.stack(teacher_actions).astype(np.float32),
    }


def _phase8_nearest_neighbor_metrics(
    predicted: np.ndarray,
    target: np.ndarray,
    references: np.ndarray,
    latent_norm: Standardizer,
) -> dict[str, float]:
    device = default_device()
    reference_t = torch.from_numpy(latent_norm.transform(references)).to(device).float()

    def nearest(values: np.ndarray) -> np.ndarray:
        value_t = torch.from_numpy(latent_norm.transform(values)).to(device).float()
        rows = []
        for start in range(0, len(value_t), 128):
            rows.append(torch.cdist(value_t[start : start + 128], reference_t).min(dim=1).values)
        return torch.cat(rows).cpu().numpy()

    predicted_distance = nearest(predicted)
    target_distance = nearest(target)
    return {
        "predicted_mean": float(np.mean(predicted_distance)),
        "predicted_median": float(np.median(predicted_distance)),
        "target_mean": float(np.mean(target_distance)),
        "target_median": float(np.median(target_distance)),
        "mean_ratio_predicted_to_target": float(
            np.mean(predicted_distance) / max(np.mean(target_distance), 1e-8)
        ),
    }


def train_phase8_deterministic_predictor(
    config: Config,
    history: int,
    latent_dim: int | None = None,
    variant: str | None = None,
    horizon_steps: int | None = None,
    target_mode: str = "absolute",
    seed: int = 0,
    force: bool = False,
) -> Path:
    set_seed(seed)
    latent_dim, variant, horizon_steps = _phase8_defaults(
        config, latent_dim, variant, horizon_steps
    )
    if history < 1:
        raise ValueError("Phase 8 history must be positive")
    if target_mode not in {"absolute", "delta"}:
        raise ValueError(f"Unknown Phase 8 target mode: {target_mode}")
    cache_path = prepare_phase8_latent_episodes(
        config,
        latent_dim=latent_dim,
        variant=variant,
        seed=seed,
        force=False,
    )
    artifact_dir = ensure_dir(
        config.path_value("paths.incremental_artifact_dir")
        / "phase8"
        / f"{variant}_z{latent_dim}_k{horizon_steps}_l{history}"
        / f"seed{seed}"
    )
    checkpoint_name = (
        "deterministic_predictor.pt"
        if target_mode == "absolute"
        else f"deterministic_predictor_{target_mode}.pt"
    )
    checkpoint_path = artifact_dir / checkpoint_name
    if checkpoint_path.exists() and not force:
        console.print(f"Phase 8 deterministic predictor exists: {checkpoint_path}")
        return checkpoint_path
    cached = torch.load(cache_path, map_location="cpu", weights_only=False)
    train_episodes = cached["train"]
    val_episodes = cached["validation"]
    encoder_checkpoint = torch.load(
        cached["encoder_checkpoint"], map_location="cpu", weights_only=False
    )
    action_norm = Standardizer.from_state_dict(encoder_checkpoint["action_norm"])
    latent_norm = Standardizer.fit(
        np.concatenate([episode["latents"] for episode in train_episodes], axis=0)
    )
    target_norm = latent_norm
    if target_mode == "delta":
        target_norm = Standardizer.fit(
            np.concatenate(
                [
                    episode["latents"][horizon_steps:] - episode["latents"][:-horizon_steps]
                    for episode in train_episodes
                ],
                axis=0,
            )
        )
    validation = _phase8_validation_samples(
        val_episodes,
        history,
        horizon_steps,
        latent_norm,
        action_norm,
        int(config.get("incremental.phase8.validation_samples", 10000)),
        seed + history,
    )
    dataset = _Phase8FutureDataset(
        train_episodes,
        history,
        horizon_steps,
        latent_norm,
        target_norm,
        target_mode,
        action_norm,
        length=int(config.get("incremental.phase8.batch_size", 512))
        * int(config.get("incremental.phase8.batches_per_epoch", 300)),
    )
    loader = DataLoader(
        dataset,
        batch_size=int(config.get("incremental.phase8.batch_size", 512)),
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    device = default_device()
    condition_dim = history * (latent_dim + int(action_norm.mean.shape[0]))
    model = MLP(
        condition_dim,
        latent_dim,
        int(config.get("incremental.phase8.hidden_dim", 1024)),
        depth=4,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(config.get("incremental.phase8.lr", 3e-4))
    )
    x_val = torch.from_numpy(validation["conditions"]).to(device).float()
    target_raw = validation["future_latents"]
    if target_mode == "delta":
        target_raw = target_raw - validation["current_latents"]
    y_val = torch.from_numpy(target_norm.transform(target_raw)).to(device).float()
    epochs = int(config.get("incremental.phase8.epochs", 60))
    best_state = None
    best_val = float("inf")
    training_history = []
    timer = Timer()
    for epoch in trange(1, epochs + 1, desc=f"train phase8 deterministic L={history}"):
        model.train()
        loss_sum = 0.0
        count = 0
        for condition, target in loader:
            condition = condition.to(device, non_blocking=True).float()
            target = target.to(device, non_blocking=True).float()
            loss = torch.mean((model(condition) - target) ** 2)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach().cpu()) * len(condition)
            count += len(condition)
        model.eval()
        with torch.inference_mode():
            validation_mse = float(torch.mean((model(x_val) - y_val) ** 2).cpu())
        training_history.append(
            {
                "epoch": epoch,
                "train_mse": loss_sum / count,
                "validation_mse": validation_mse,
            }
        )
        if validation_mse < best_val:
            best_val = validation_mse
            best_state = copy.deepcopy(model.state_dict())
    if best_state is None:
        raise RuntimeError("Phase 8 deterministic training produced no checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    with torch.inference_mode():
        predicted_raw = target_norm.inverse(model(x_val).cpu().numpy())
    predicted = (
        predicted_raw
        if target_mode == "absolute"
        else validation["current_latents"] + predicted_raw
    )
    target = validation["future_latents"]
    persistence = validation["current_latents"]
    prediction_error = np.linalg.norm(predicted - target, axis=-1)
    persistence_error = np.linalg.norm(persistence - target, axis=-1)
    offline_metrics = {
        "normalized_mse": best_val,
        "raw_latent_mae": float(np.mean(np.abs(predicted - target))),
        "raw_latent_l2": float(np.mean(prediction_error)),
        "persistence_raw_latent_l2": float(np.mean(persistence_error)),
        "l2_improvement_over_persistence": float(
            1.0 - np.mean(prediction_error) / max(np.mean(persistence_error), 1e-8)
        ),
        "prediction_better_than_persistence_fraction": float(
            np.mean(prediction_error < persistence_error)
        ),
        "samples": int(len(target)),
    }
    rng = np.random.default_rng(seed + 8000 + history)
    nn_predictions = min(
        int(config.get("incremental.phase8.nearest_neighbor_predictions", 1000)),
        len(predicted),
    )
    nn_references = int(config.get("incremental.phase8.nearest_neighbor_references", 10000))
    pred_idx = rng.choice(len(predicted), size=nn_predictions, replace=False)
    all_train_latents = np.concatenate([episode["latents"] for episode in train_episodes], axis=0)
    if len(all_train_latents) > nn_references:
        ref_idx = rng.choice(len(all_train_latents), size=nn_references, replace=False)
        references = all_train_latents[ref_idx]
    else:
        references = all_train_latents
    nearest_neighbor_metrics = _phase8_nearest_neighbor_metrics(
        predicted[pred_idx], target[pred_idx], references, latent_norm
    )

    phase7_artifact_dir = (
        config.path_value("paths.incremental_artifact_dir")
        / "phase7"
        / _phase7_tag(variant, latent_dim, horizon_steps, 1, "delta", 0.0)
        / f"seed{seed}"
    )
    dagger_candidate = phase7_artifact_dir / "oracle_low_level_branch_dagger_iter1_e10.pt"
    if dagger_candidate.exists():
        low_checkpoint_path = dagger_candidate
    else:
        low_checkpoint_path = train_phase7_oracle_low_level(
            config,
            latent_dim=latent_dim,
            variant=variant,
            horizon_steps=horizon_steps,
            action_chunk_steps=1,
            goal_encoding="delta",
            goal_dropout_prob=0.0,
            seed=seed,
            force=False,
        )
    low_model, low_checkpoint = _load_phase7_low_level_checkpoint(low_checkpoint_path, device)
    low_action_norm = Standardizer.from_state_dict(low_checkpoint["action_norm"])
    true_conditions = np.stack(
        [
            _phase7_condition(current, future, previous, "delta")
            for current, future, previous in zip(
                validation["current_latents"],
                target,
                validation["previous_actions_norm"],
            )
        ]
    )
    predicted_conditions = np.stack(
        [
            _phase7_condition(current, future, previous, "delta")
            for current, future, previous in zip(
                validation["current_latents"],
                predicted,
                validation["previous_actions_norm"],
            )
        ]
    )

    def low_actions(conditions: np.ndarray) -> np.ndarray:
        rows = []
        with torch.inference_mode():
            for start in range(0, len(conditions), 4096):
                pred_norm = low_model(
                    torch.from_numpy(conditions[start : start + 4096]).to(device).float()
                )
                rows.append(low_action_norm.inverse(pred_norm.cpu().numpy()))
        return np.concatenate(rows).astype(np.float32)

    oracle_low_actions = low_actions(true_conditions)
    predicted_low_actions = low_actions(predicted_conditions)
    teacher_actions = validation["teacher_actions"]
    low_level_metrics = {
        "oracle_goal_action_mae": float(np.mean(np.abs(oracle_low_actions - teacher_actions))),
        "predicted_goal_action_mae": float(
            np.mean(np.abs(predicted_low_actions - teacher_actions))
        ),
        "predicted_to_oracle_mae_ratio": float(
            np.mean(np.abs(predicted_low_actions - teacher_actions))
            / max(np.mean(np.abs(oracle_low_actions - teacher_actions)), 1e-8)
        ),
        "predicted_vs_oracle_action_l2": float(
            np.mean(np.linalg.norm(predicted_low_actions - oracle_low_actions, axis=-1))
        ),
    }
    payload = {
        "model": model.state_dict(),
        "variant": variant,
        "latent_dim": latent_dim,
        "horizon_steps": horizon_steps,
        "history": history,
        "target_mode": target_mode,
        "condition_dim": condition_dim,
        "hidden_dim": int(config.get("incremental.phase8.hidden_dim", 1024)),
        "encoder_checkpoint": cached["encoder_checkpoint"],
        "low_level_checkpoint": str(low_checkpoint_path),
        "latent_norm": latent_norm.state_dict(),
        "target_norm": target_norm.state_dict(),
        "action_norm": action_norm.state_dict(),
        "offline_metrics": offline_metrics,
        "nearest_neighbor_metrics": nearest_neighbor_metrics,
        "low_level_metrics": low_level_metrics,
        "best_validation_mse": best_val,
        "training_history": training_history,
        "elapsed_s": timer.elapsed(),
        "data": cached["data"],
        "metadata": _runtime_metadata(config),
    }
    torch.save(payload, checkpoint_path)
    write_json(
        artifact_dir
        / (
            "deterministic_predictor_metrics.json"
            if target_mode == "absolute"
            else f"deterministic_predictor_{target_mode}_metrics.json"
        ),
        {
            key: payload[key]
            for key in [
                "variant",
                "latent_dim",
                "horizon_steps",
                "history",
                "target_mode",
                "offline_metrics",
                "nearest_neighbor_metrics",
                "low_level_metrics",
                "best_validation_mse",
                "elapsed_s",
            ]
        },
    )
    console.print(f"Wrote Phase 8 deterministic predictor: {checkpoint_path}")
    return checkpoint_path


def _phase8_structured_metrics(predicted: np.ndarray, target: np.ndarray) -> dict[str, float]:
    yaw_pred = np.arctan2(predicted[:, 2], predicted[:, 3])
    yaw_target = np.arctan2(target[:, 2], target[:, 3])
    yaw_error = np.abs(_wrap_angle(yaw_pred - yaw_target))
    return {
        "goal_mae": float(np.mean(np.abs(predicted - target))),
        "t_position_l2_m": float(np.mean(np.linalg.norm(predicted[:, :2] - target[:, :2], axis=1))),
        "t_yaw_mae_rad": float(np.mean(yaw_error)),
        "t_velocity_l2_mps": float(
            np.mean(np.linalg.norm(predicted[:, 4:6] - target[:, 4:6], axis=1))
        ),
        "t_yaw_rate_mae_radps": float(np.mean(np.abs(predicted[:, 6] - target[:, 6]))),
        "tcp_position_l2_m": float(
            np.mean(np.linalg.norm(predicted[:, 7:10] - target[:, 7:10], axis=1))
        ),
        "tcp_velocity_l2_mps": float(
            np.mean(np.linalg.norm(predicted[:, 10:13] - target[:, 10:13], axis=1))
        ),
        "contact_accuracy": float(np.mean((predicted[:, 13] >= 0.5) == (target[:, 13] >= 0.5))),
    }


def train_phase8_structured_predictor(
    config: Config,
    horizon_steps: int | None = None,
    seed: int = 0,
    force: bool = False,
) -> Path:
    """Predict the privileged structured goal used by the Phase 7D low level."""
    set_seed(seed)
    horizon_steps = int(horizon_steps or config.get("incremental.phase8.horizon_steps", 2))
    artifact_dir = ensure_dir(
        config.path_value("paths.incremental_artifact_dir")
        / "phase8"
        / f"structured_k{horizon_steps}"
        / f"seed{seed}"
    )
    checkpoint_path = artifact_dir / "deterministic_predictor.pt"
    if checkpoint_path.exists() and not force:
        console.print(f"Phase 8 structured predictor exists: {checkpoint_path}")
        return checkpoint_path

    train_episodes, val_episodes, data_metadata = _load_phase7_privileged_episodes(
        config, horizon_steps
    )
    action_norm = Standardizer.fit(
        np.concatenate([episode["actions"] for episode in train_episodes], axis=0)
    )
    zero_action = action_norm.transform(
        np.zeros((1, train_episodes[0]["actions"].shape[-1]), dtype=np.float32)
    )[0]
    control_freq = int(config.get("control_freq", 20))

    def samples(episodes: list[dict[str, np.ndarray]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        conditions = []
        goals = []
        actions = []
        for episode in episodes:
            states = episode["states"]
            episode_actions = episode["actions"]
            normalized_actions = action_norm.transform(episode_actions)
            for t in range(len(episode_actions) - horizon_steps):
                previous = normalized_actions[t - 1] if t > 0 else zero_action
                conditions.append(np.concatenate([states[t], previous]))
                goals.append(
                    _phase7_privileged_goal(
                        states[t], states[t + horizon_steps], horizon_steps, control_freq
                    )
                )
                actions.append(episode_actions[t])
        return (
            np.asarray(conditions, dtype=np.float32),
            np.asarray(goals, dtype=np.float32),
            np.asarray(actions, dtype=np.float32),
        )

    train_x, train_y, _train_actions = samples(train_episodes)
    val_x, val_y, val_actions = samples(val_episodes)
    max_validation = int(config.get("incremental.phase8.validation_samples", 10000))
    if len(val_x) > max_validation:
        indices = np.random.default_rng(seed + 8100).choice(
            len(val_x), size=max_validation, replace=False
        )
        val_x, val_y, val_actions = val_x[indices], val_y[indices], val_actions[indices]
    input_norm = Standardizer.fit(train_x)
    target_norm = Standardizer.fit(train_y)
    dataset = TensorDataset(
        torch.from_numpy(input_norm.transform(train_x)).float(),
        torch.from_numpy(target_norm.transform(train_y)).float(),
    )
    loader = DataLoader(
        dataset,
        batch_size=int(config.get("incremental.phase8.batch_size", 512)),
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    device = default_device()
    hidden_dim = int(config.get("incremental.phase8.structured_hidden_dim", 512))
    model = MLP(train_x.shape[-1], train_y.shape[-1], hidden_dim, depth=4).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(config.get("incremental.phase8.lr", 3e-4))
    )
    val_x_t = torch.from_numpy(input_norm.transform(val_x)).to(device).float()
    val_y_t = torch.from_numpy(target_norm.transform(val_y)).to(device).float()
    epochs = int(config.get("incremental.phase8.structured_epochs", 30))
    best_loss = float("inf")
    best_state = None
    history = []
    timer = Timer()
    for epoch in trange(1, epochs + 1, desc="train phase8 structured"):
        model.train()
        total = 0.0
        count = 0
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            loss = torch.mean((model(x) - y) ** 2)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += float(loss.detach().cpu()) * len(x)
            count += len(x)
        model.eval()
        with torch.inference_mode():
            val_loss = float(torch.mean((model(val_x_t) - val_y_t) ** 2).cpu())
        history.append({"epoch": epoch, "train_mse": total / count, "validation_mse": val_loss})
        if val_loss < best_loss:
            best_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
    if best_state is None:
        raise RuntimeError("Phase 8 structured training produced no checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    with torch.inference_mode():
        predicted = target_norm.inverse(model(val_x_t).cpu().numpy())
    current_states = val_x[:, :31]
    persistence = _phase7_privileged_goal(
        current_states, current_states, horizon_steps, control_freq
    )

    privileged_path = train_phase7_privileged_branch_baselines(
        config, horizon_steps=horizon_steps, seed=seed, force=False
    )
    privileged = torch.load(privileged_path, map_location=device, weights_only=False)
    low_model, low_cond_norm = _phase7_load_privileged_model(privileged, "branch_goal", device)
    low_action_norm = Standardizer.from_state_dict(privileged["action_norm"])
    previous_actions = val_x[:, 31:]

    def low_actions(goals: np.ndarray) -> np.ndarray:
        cond = _phase7_privileged_condition(current_states, goals, previous_actions)
        with torch.inference_mode():
            output = (
                low_model(torch.from_numpy(low_cond_norm.transform(cond)).to(device).float())
                .cpu()
                .numpy()
            )
        return low_action_norm.inverse(output)

    oracle_action_mae = float(np.mean(np.abs(low_actions(val_y) - val_actions)))
    predicted_action_mae = float(np.mean(np.abs(low_actions(predicted) - val_actions)))
    payload = {
        "model": model.state_dict(),
        "condition_dim": int(train_x.shape[-1]),
        "goal_dim": int(train_y.shape[-1]),
        "hidden_dim": hidden_dim,
        "horizon_steps": horizon_steps,
        "input_norm": input_norm.state_dict(),
        "target_norm": target_norm.state_dict(),
        "action_norm": action_norm.state_dict(),
        "offline_metrics": _phase8_structured_metrics(predicted, val_y),
        "persistence_metrics": _phase8_structured_metrics(persistence, val_y),
        "low_level_metrics": {
            "oracle_goal_action_mae": oracle_action_mae,
            "predicted_goal_action_mae": predicted_action_mae,
            "predicted_to_oracle_mae_ratio": predicted_action_mae / max(oracle_action_mae, 1e-8),
        },
        "best_validation_mse": best_loss,
        "history": history,
        "elapsed_s": timer.elapsed(),
        "data": {**data_metadata, "train_samples": len(train_x), "validation_samples": len(val_x)},
        "metadata": _runtime_metadata(config),
    }
    torch.save(payload, checkpoint_path)
    write_json(
        artifact_dir / "deterministic_predictor_metrics.json",
        {
            key: payload[key]
            for key in [
                "horizon_steps",
                "offline_metrics",
                "persistence_metrics",
                "low_level_metrics",
                "best_validation_mse",
                "elapsed_s",
                "data",
                "metadata",
            ]
        },
    )
    console.print(f"Wrote Phase 8 structured predictor: {checkpoint_path}")
    return checkpoint_path


@torch.inference_mode()
def evaluate_phase8_structured_hierarchy(
    config: Config,
    horizon_steps: int | None = None,
    seed: int = 0,
    episodes: int | None = None,
    force: bool = False,
) -> Path:
    horizon_steps = int(horizon_steps or config.get("incremental.phase8.horizon_steps", 2))
    eval_episodes = int(episodes or config.get("incremental.phase8.eval_episodes", 100))
    output_dir = ensure_dir(
        config.path_value("paths.incremental_results_dir")
        / "phase8"
        / f"structured_k{horizon_steps}"
        / f"seed{seed}"
    )
    output_path = output_dir / f"deterministic_hierarchy_{eval_episodes}.json"
    if output_path.exists() and not force:
        console.print(f"Phase 8 structured hierarchy eval exists: {output_path}")
        return output_path
    predictor_path = train_phase8_structured_predictor(
        config, horizon_steps=horizon_steps, seed=seed, force=False
    )
    device = default_device()
    predictor_checkpoint = torch.load(predictor_path, map_location=device, weights_only=False)
    predictor = MLP(
        int(predictor_checkpoint["condition_dim"]),
        int(predictor_checkpoint["goal_dim"]),
        int(predictor_checkpoint["hidden_dim"]),
        depth=4,
    ).to(device)
    predictor.load_state_dict(predictor_checkpoint["model"])
    predictor.eval()
    input_norm = Standardizer.from_state_dict(predictor_checkpoint["input_norm"])
    target_norm = Standardizer.from_state_dict(predictor_checkpoint["target_norm"])
    action_norm = Standardizer.from_state_dict(predictor_checkpoint["action_norm"])

    privileged_path = train_phase7_privileged_branch_baselines(
        config, horizon_steps=horizon_steps, seed=seed, force=False
    )
    privileged = torch.load(privileged_path, map_location=device, weights_only=False)
    low_model, low_cond_norm = _phase7_load_privileged_model(privileged, "branch_goal", device)
    teacher = load_ppo_agent(_rl_paths(config).best, device)
    num_envs = min(int(config.get("incremental.phase8.eval_num_envs", 64)), eval_episodes)
    env = _make_state_env(
        config,
        num_envs,
        record_metrics=True,
        ignore_terminations=False,
        reconfiguration_freq=0,
    )
    action_low = torch.as_tensor(env.single_action_space.low, device=device, dtype=torch.float32)
    action_high = torch.as_tensor(env.single_action_space.high, device=device, dtype=torch.float32)
    zero_action = action_norm.transform(
        np.zeros((1, int(privileged["branch_goal"]["action_dim"])), dtype=np.float32)
    )[0]
    previous_action = np.repeat(zero_action[None], num_envs, axis=0).astype(np.float32)
    seed_start = int(config.get("incremental.phase8.eval_seed", 1_200_000))
    obs, _info = env.reset(seed=seed_start)
    successes = []
    final_rewards = []
    max_rewards = []
    episode_lengths = []
    teacher_maes = []
    latencies = []
    active_max_reward = np.full(num_envs, -np.inf, dtype=np.float32)
    active_lengths = np.zeros(num_envs, dtype=np.int32)
    while len(successes) < eval_episodes:
        timer = Timer()
        state_t = _phase7_obs_state_tensor(obs, device)
        state = state_t.cpu().numpy().astype(np.float32)
        high_condition = np.concatenate([state, previous_action], axis=-1)
        predicted_goal = target_norm.inverse(
            predictor(torch.from_numpy(input_norm.transform(high_condition)).to(device).float())
            .cpu()
            .numpy()
        )
        low_condition = _phase7_privileged_condition(state, predicted_goal, previous_action)
        raw_action = action_norm.inverse(
            low_model(torch.from_numpy(low_cond_norm.transform(low_condition)).to(device).float())
            .cpu()
            .numpy()
        ).astype(np.float32)
        teacher_action = (
            torch.clamp(teacher.actor_mean(state_t), action_low, action_high)
            .cpu()
            .numpy()
            .astype(np.float32)
        )
        teacher_maes.extend(np.mean(np.abs(raw_action - teacher_action), axis=-1).tolist())
        latencies.append(timer.elapsed() / num_envs)
        action = torch.from_numpy(raw_action).to(device).float()
        if bool(config.get("policy.clip_actions_to_env_space", True)):
            action = torch.clamp(action, action_low, action_high)
        previous_action = action_norm.transform(action.cpu().numpy().astype(np.float32))
        obs, reward, _terminated, _truncated, info = env.step(action)
        reward_np = _numpy(reward).reshape(-1).astype(np.float32)
        active_max_reward = np.maximum(active_max_reward, reward_np)
        active_lengths += 1
        if "final_info" in info:
            mask = _numpy(info["_final_info"]).reshape(-1).astype(bool)
            if mask.any():
                success_once = _numpy(info["final_info"]["episode"]["success_once"]).reshape(-1)
                for env_idx in np.flatnonzero(mask):
                    successes.append(float(success_once[env_idx]))
                    final_rewards.append(float(reward_np[env_idx]))
                    max_rewards.append(float(active_max_reward[env_idx]))
                    episode_lengths.append(int(active_lengths[env_idx]))
                    active_max_reward[env_idx] = -np.inf
                    active_lengths[env_idx] = 0
                    previous_action[env_idx] = zero_action
                    if len(successes) >= eval_episodes:
                        break
    env.close()
    metrics = {
        "success": float(np.mean(successes[:eval_episodes])),
        "success_stderr": float(np.std(successes[:eval_episodes]) / np.sqrt(eval_episodes)),
        "final_reward": float(np.mean(final_rewards[:eval_episodes])),
        "max_reward": float(np.mean(max_rewards[:eval_episodes])),
        "mean_episode_length": float(np.mean(episode_lengths[:eval_episodes])),
        "teacher_action_mae": float(np.mean(teacher_maes)),
        "inference_latency_s": float(np.mean(latencies)),
        "episodes": eval_episodes,
        "num_envs": num_envs,
        "seed_start": seed_start,
    }
    oracle_path = (
        _phase7_privileged_results_dir(config, horizon_steps, seed)
        / "privileged_branch_baselines_eval_100.json"
    )
    import json

    with oracle_path.open("r", encoding="utf-8") as f:
        oracle_success = float(json.load(f)["closed_loop"]["branch_goal"]["success"])
    payload = {
        "phase": "8.1",
        "method": "deterministic_privileged_structured_hierarchy",
        "horizon_steps": horizon_steps,
        "seed": seed,
        "checkpoint": str(predictor_path),
        "closed_loop": metrics,
        "offline_metrics": predictor_checkpoint["offline_metrics"],
        "low_level_metrics": predictor_checkpoint["low_level_metrics"],
        "oracle_result": str(oracle_path),
        "oracle_success": oracle_success,
        "success_fraction_of_oracle": metrics["success"] / max(oracle_success, 1e-8),
        "gate_70pct_oracle": metrics["success"] >= 0.70 * oracle_success,
        "metadata": _runtime_metadata(config),
    }
    write_json(output_path, payload)
    console.print(payload)
    return output_path


def _phase8_pose_probe_metrics(
    predicted_encoded: np.ndarray, target: np.ndarray
) -> dict[str, float]:
    predicted_yaw = np.arctan2(predicted_encoded[:, 2], predicted_encoded[:, 3])
    errors = np.stack(
        [
            np.abs(predicted_encoded[:, 0] - target[:, 0]),
            np.abs(predicted_encoded[:, 1] - target[:, 1]),
            np.abs(_wrap_angle(predicted_yaw - target[:, 2])),
            *[np.abs(predicted_encoded[:, index + 1] - target[:, index]) for index in range(3, 10)],
        ],
        axis=-1,
    )
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
    return {name: float(value) for name, value in zip(names, errors.mean(axis=0), strict=True)}


def probe_phase8_predicted_latents(
    config: Config,
    latent_dim: int | None = None,
    variant: str | None = None,
    horizon_steps: int | None = None,
    seed: int = 0,
    force: bool = False,
) -> Path:
    """Apply one frozen structured probe to real and predicted future latents."""
    latent_dim, variant, horizon_steps = _phase8_defaults(
        config, latent_dim, variant, horizon_steps
    )
    if horizon_steps != 2:
        raise ValueError("The existing time-major probe dataset currently supports k=2")
    results_dir = ensure_dir(
        config.path_value("paths.incremental_results_dir")
        / "phase8"
        / f"{variant}_z{latent_dim}_k{horizon_steps}_l1"
        / f"seed{seed}"
    )
    output_path = results_dir / "predicted_latent_structured_probe.json"
    if output_path.exists() and not force:
        console.print(f"Phase 8 predicted-latent probe exists: {output_path}")
        return output_path
    predictor_path = train_phase8_deterministic_predictor(
        config,
        history=1,
        latent_dim=latent_dim,
        variant=variant,
        horizon_steps=horizon_steps,
        target_mode="absolute",
        seed=seed,
        force=False,
    )
    predictor_checkpoint = torch.load(predictor_path, map_location="cpu", weights_only=False)
    encoder_path = Path(predictor_checkpoint["encoder_checkpoint"])
    device = default_device()
    encoder, encoder_checkpoint = _load_phase6_encoder(encoder_path, device)
    frame_norm = Standardizer.from_state_dict(encoder_checkpoint["frame_norm"])
    latent_norm = Standardizer.from_state_dict(predictor_checkpoint["latent_norm"])
    target_norm = Standardizer.from_state_dict(
        predictor_checkpoint.get("target_norm", predictor_checkpoint["latent_norm"])
    )
    action_norm = Standardizer.from_state_dict(predictor_checkpoint["action_norm"])
    predictor = MLP(
        int(predictor_checkpoint["condition_dim"]),
        latent_dim,
        int(predictor_checkpoint["hidden_dim"]),
        depth=4,
    ).to(device)
    predictor.load_state_dict(predictor_checkpoint["model"])
    predictor.eval()

    probe_path = collect_phase6_probe_dataset(config, force=False)
    with np.load(probe_path) as data:
        inputs = np.asarray(data["inputs"], dtype=np.float32)
        next_inputs = np.asarray(data["next_inputs"], dtype=np.float32)
        actions = np.asarray(data["actions"], dtype=np.float32)
        labels = np.asarray(data["labels"], dtype=np.float32)
        next_labels = np.asarray(data["next_labels"], dtype=np.float32)
        contact = np.asarray(data["contact"], dtype=np.float32)
        next_contact = np.asarray(data["next_contact"], dtype=np.float32)
    num_envs = int(config.get("incremental.phase6.eval_num_envs", 64))
    full_rows = len(inputs) // num_envs * num_envs
    time_steps = full_rows // num_envs

    def blocks(values: np.ndarray) -> np.ndarray:
        return values[:full_rows].reshape(time_steps, num_envs, *values.shape[1:])

    input_blocks = blocks(inputs)
    next_input_blocks = blocks(next_inputs)
    action_blocks = blocks(actions)
    label_blocks = blocks(labels)
    next_label_blocks = blocks(next_labels)
    contact_blocks = blocks(contact)
    next_contact_blocks = blocks(next_contact)

    def encode(values: np.ndarray) -> np.ndarray:
        normalized = frame_norm.transform(values.reshape(-1, values.shape[-1]))
        rows = []
        with torch.inference_mode():
            for start in range(0, len(normalized), 4096):
                rows.append(
                    encoder(torch.from_numpy(normalized[start : start + 4096]).to(device).float())
                    .cpu()
                    .numpy()
                )
        return np.concatenate(rows).reshape(time_steps, num_envs, latent_dim).astype(np.float32)

    latent_blocks = encode(input_blocks)
    next_latent_blocks = encode(next_input_blocks)
    train_envs = int(0.75 * num_envs)
    train_reps = latent_blocks[:, :train_envs].reshape(-1, latent_dim)
    train_labels = label_blocks[:, :train_envs].reshape(-1, labels.shape[-1])
    train_contact = contact_blocks[:, :train_envs].reshape(-1, 1)
    train_encoded_labels = np.concatenate(
        [
            train_labels[:, :2],
            np.sin(train_labels[:, 2:3]),
            np.cos(train_labels[:, 2:3]),
            train_labels[:, 3:],
        ],
        axis=-1,
    ).astype(np.float32)
    rep_norm = Standardizer.fit(train_reps)
    label_norm = Standardizer.fit(train_encoded_labels)
    probe_dataset = TensorDataset(
        torch.from_numpy(rep_norm.transform(train_reps)).float(),
        torch.from_numpy(label_norm.transform(train_encoded_labels)).float(),
        torch.from_numpy(train_contact).float(),
    )
    loader = DataLoader(
        probe_dataset,
        batch_size=int(config.get("incremental.phase6.probe_batch_size", 512)),
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    hidden_dim = int(config.get("incremental.phase6.probe_hidden_dim", 512))
    pose_head = MLP(latent_dim, train_encoded_labels.shape[-1], hidden_dim, depth=3).to(device)
    contact_head = MLP(latent_dim, 1, hidden_dim, depth=3).to(device)
    optimizer = torch.optim.AdamW(
        list(pose_head.parameters()) + list(contact_head.parameters()),
        lr=float(config.get("incremental.phase6.probe_lr", 1e-3)),
    )
    epochs = int(config.get("incremental.phase8.predicted_probe_epochs", 80))
    timer = Timer()
    for _epoch in trange(epochs, desc="train phase8 frozen probe"):
        for x, y, c in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            c = c.to(device, non_blocking=True)
            loss = torch.mean(
                (pose_head(x) - y) ** 2
            ) + torch.nn.functional.binary_cross_entropy_with_logits(contact_head(x), c)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    pose_head.eval()
    contact_head.eval()

    previous_continuity = (
        np.max(np.abs(next_label_blocks[:-2] - label_blocks[1:-1]), axis=-1) < 1e-4
    )
    future_continuity = np.max(np.abs(next_label_blocks[1:-1] - label_blocks[2:]), axis=-1) < 1e-4
    valid = previous_continuity & future_continuity
    valid[:, :train_envs] = False
    time_idx, env_idx = np.nonzero(valid)
    current_t = time_idx + 1
    current_latents = latent_blocks[current_t, env_idx]
    real_future_latents = next_latent_blocks[current_t + 1, env_idx]
    physical_targets = next_label_blocks[current_t + 1, env_idx]
    contact_targets = next_contact_blocks[current_t + 1, env_idx, 0]
    previous_actions = action_blocks[current_t - 1, env_idx]
    high_condition = np.concatenate(
        [latent_norm.transform(current_latents), action_norm.transform(previous_actions)], axis=-1
    )
    with torch.inference_mode():
        predicted_future_latents = target_norm.inverse(
            predictor(torch.from_numpy(high_condition).to(device).float()).cpu().numpy()
        )

    def probe(representations: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x = torch.from_numpy(rep_norm.transform(representations)).to(device).float()
        with torch.inference_mode():
            pose = label_norm.inverse(pose_head(x).cpu().numpy())
            contact_logits = contact_head(x).cpu().numpy()[:, 0]
        return pose, contact_logits

    real_pose, real_contact_logits = probe(real_future_latents)
    predicted_pose, predicted_contact_logits = probe(predicted_future_latents)
    payload = {
        "phase": "8.5",
        "method": "matched_frozen_probe_real_vs_predicted_future_latents",
        "variant": variant,
        "latent_dim": latent_dim,
        "horizon_steps": horizon_steps,
        "seed": seed,
        "predictor": str(predictor_path),
        "probe_dataset": str(probe_path),
        "split": {
            "train_environment_streams": train_envs,
            "evaluation_environment_streams": num_envs - train_envs,
            "train_samples": len(train_reps),
            "valid_k2_evaluation_samples": len(current_latents),
            "continuity_tolerance": 1e-4,
        },
        "real_future_latent_probe_mae": _phase8_pose_probe_metrics(real_pose, physical_targets),
        "predicted_future_latent_probe_mae": _phase8_pose_probe_metrics(
            predicted_pose, physical_targets
        ),
        "contact": {
            "real_accuracy": float(np.mean((real_contact_logits >= 0) == contact_targets)),
            "predicted_accuracy": float(
                np.mean((predicted_contact_logits >= 0) == contact_targets)
            ),
            "real_auroc": _binary_auc(real_contact_logits, contact_targets),
            "predicted_auroc": _binary_auc(predicted_contact_logits, contact_targets),
        },
        "latent": {
            "predicted_to_real_l2": float(
                np.mean(np.linalg.norm(predicted_future_latents - real_future_latents, axis=-1))
            )
        },
        "elapsed_s": timer.elapsed(),
        "metadata": _runtime_metadata(config),
    }
    write_json(output_path, payload)
    console.print(payload)
    return output_path


def train_phase8_action_consistent_predictor(
    config: Config,
    action_consistency_weight: float,
    latent_dim: int | None = None,
    variant: str | None = None,
    horizon_steps: int | None = None,
    seed: int = 0,
    force: bool = False,
) -> Path:
    if action_consistency_weight <= 0:
        raise ValueError("Action-consistency weight must be positive")
    set_seed(seed)
    latent_dim, variant, horizon_steps = _phase8_defaults(
        config, latent_dim, variant, horizon_steps
    )
    weight_label = f"{action_consistency_weight:g}".replace(".", "p")
    artifact_dir = ensure_dir(
        config.path_value("paths.incremental_artifact_dir")
        / "phase8"
        / f"{variant}_z{latent_dim}_k{horizon_steps}_l1"
        / f"seed{seed}"
    )
    output_path = artifact_dir / f"deterministic_predictor_actionw{weight_label}.pt"
    if output_path.exists() and not force:
        console.print(f"Phase 8 action-consistent predictor exists: {output_path}")
        return output_path
    base_path = train_phase8_deterministic_predictor(
        config,
        history=1,
        latent_dim=latent_dim,
        variant=variant,
        horizon_steps=horizon_steps,
        target_mode="absolute",
        seed=seed,
        force=False,
    )
    base = torch.load(base_path, map_location="cpu", weights_only=False)
    cache_path = prepare_phase8_latent_episodes(
        config, latent_dim=latent_dim, variant=variant, seed=seed, force=False
    )
    cached = torch.load(cache_path, map_location="cpu", weights_only=False)
    latent_norm = Standardizer.from_state_dict(base["latent_norm"])
    target_norm = Standardizer.from_state_dict(base.get("target_norm", base["latent_norm"]))
    action_norm = Standardizer.from_state_dict(base["action_norm"])
    dataset = _Phase8FutureDataset(
        cached["train"],
        1,
        horizon_steps,
        latent_norm,
        target_norm,
        "absolute",
        action_norm,
        length=int(config.get("incremental.phase8.batch_size", 512))
        * int(config.get("incremental.phase8.batches_per_epoch", 300)),
    )
    loader = DataLoader(
        dataset,
        batch_size=int(config.get("incremental.phase8.batch_size", 512)),
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    validation = _phase8_validation_samples(
        cached["validation"],
        1,
        horizon_steps,
        latent_norm,
        action_norm,
        int(config.get("incremental.phase8.validation_samples", 10000)),
        seed + 8200,
    )
    device = default_device()
    model = MLP(int(base["condition_dim"]), latent_dim, int(base["hidden_dim"]), depth=4).to(device)
    model.load_state_dict(base["model"])
    low_model, low_checkpoint = _load_phase7_low_level_checkpoint(
        Path(base["low_level_checkpoint"]), device
    )
    for parameter in low_model.parameters():
        parameter.requires_grad_(False)
    latent_mean = torch.from_numpy(latent_norm.mean).to(device).float()
    latent_std = torch.from_numpy(latent_norm.std).to(device).float()
    target_mean = torch.from_numpy(target_norm.mean).to(device).float()
    target_std = torch.from_numpy(target_norm.std).to(device).float()

    def low_condition(condition: torch.Tensor, goal_normalized: torch.Tensor) -> torch.Tensor:
        current = condition[:, :latent_dim] * latent_std + latent_mean
        previous = condition[:, latent_dim:]
        goal = goal_normalized * target_std + target_mean
        return torch.cat([current, goal - current, previous], dim=-1)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config.get("incremental.phase8.action_consistency_lr", 1e-4)),
    )
    epochs = int(config.get("incremental.phase8.action_consistency_epochs", 20))
    history = []
    timer = Timer()
    for epoch in trange(
        1, epochs + 1, desc=f"phase8 action consistency w={action_consistency_weight:g}"
    ):
        model.train()
        totals = np.zeros(3, dtype=np.float64)
        count = 0
        for condition, target in loader:
            condition = condition.to(device, non_blocking=True).float()
            target = target.to(device, non_blocking=True).float()
            predicted = model(condition)
            latent_loss = torch.mean((predicted - target) ** 2)
            with torch.no_grad():
                oracle_action = low_model(low_condition(condition, target))
            predicted_action = low_model(low_condition(condition, predicted))
            action_loss = torch.mean((predicted_action - oracle_action) ** 2)
            loss = latent_loss + action_consistency_weight * action_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            totals += np.asarray(
                [
                    float(loss.detach().cpu()),
                    float(latent_loss.detach().cpu()),
                    float(action_loss.detach().cpu()),
                ]
            ) * len(condition)
            count += len(condition)
        history.append(
            {
                "epoch": epoch,
                "loss": float(totals[0] / count),
                "latent_mse": float(totals[1] / count),
                "action_consistency_mse": float(totals[2] / count),
            }
        )
    model.eval()
    val_condition = torch.from_numpy(validation["conditions"]).to(device).float()
    val_target = (
        torch.from_numpy(target_norm.transform(validation["future_latents"])).to(device).float()
    )
    with torch.inference_mode():
        val_predicted_norm = model(val_condition)
        val_predicted = target_norm.inverse(val_predicted_norm.cpu().numpy())
        oracle_actions_norm = low_model(low_condition(val_condition, val_target)).cpu().numpy()
        predicted_actions_norm = (
            low_model(low_condition(val_condition, val_predicted_norm)).cpu().numpy()
        )
    low_action_norm = Standardizer.from_state_dict(low_checkpoint["action_norm"])
    teacher_actions = validation["teacher_actions"]
    oracle_actions = low_action_norm.inverse(oracle_actions_norm)
    predicted_actions = low_action_norm.inverse(predicted_actions_norm)
    prediction_l2 = np.linalg.norm(val_predicted - validation["future_latents"], axis=-1)
    persistence_l2 = np.linalg.norm(
        validation["current_latents"] - validation["future_latents"], axis=-1
    )
    oracle_mae = float(np.mean(np.abs(oracle_actions - teacher_actions)))
    predicted_mae = float(np.mean(np.abs(predicted_actions - teacher_actions)))
    payload = {
        **base,
        "model": model.state_dict(),
        "action_consistency_weight": action_consistency_weight,
        "base_predictor": str(base_path),
        "offline_metrics": {
            "raw_latent_l2": float(np.mean(prediction_l2)),
            "persistence_raw_latent_l2": float(np.mean(persistence_l2)),
            "l2_improvement_over_persistence": float(
                1.0 - np.mean(prediction_l2) / max(np.mean(persistence_l2), 1e-8)
            ),
        },
        "low_level_metrics": {
            "oracle_goal_action_mae": oracle_mae,
            "predicted_goal_action_mae": predicted_mae,
            "predicted_to_oracle_mae_ratio": predicted_mae / max(oracle_mae, 1e-8),
            "predicted_vs_oracle_action_l2": float(
                np.mean(np.linalg.norm(predicted_actions - oracle_actions, axis=-1))
            ),
        },
        "action_consistency_history": history,
        "action_consistency_elapsed_s": timer.elapsed(),
        "metadata": _runtime_metadata(config),
    }
    torch.save(payload, output_path)
    write_json(
        artifact_dir / f"deterministic_predictor_actionw{weight_label}_metrics.json",
        {
            key: payload[key]
            for key in [
                "action_consistency_weight",
                "base_predictor",
                "offline_metrics",
                "low_level_metrics",
                "action_consistency_history",
                "action_consistency_elapsed_s",
                "metadata",
            ]
        },
    )
    console.print(f"Wrote Phase 8 action-consistent predictor: {output_path}")
    return output_path


def train_phase9_future_flow(
    config: Config,
    latent_dim: int | None = None,
    variant: str | None = None,
    horizon_steps: int | None = None,
    trajectory_limit: int | None = None,
    seed: int = 0,
    force: bool = False,
) -> Path:
    set_seed(seed)
    latent_dim, variant, horizon_steps = _phase8_defaults(
        config, latent_dim, variant, horizon_steps
    )
    endpoint_weight = float(config.get("incremental.phase9.endpoint_consistency_weight", 0.0))
    base_label = "full" if trajectory_limit is None else f"overfit_n{trajectory_limit}"
    weight_label = f"{endpoint_weight:g}".replace(".", "p")
    run_label = base_label if endpoint_weight == 0.0 else f"{base_label}_endpointw{weight_label}"
    artifact_dir = ensure_dir(
        config.path_value("paths.incremental_artifact_dir")
        / "phase9"
        / f"{variant}_z{latent_dim}_k{horizon_steps}_l1"
        / f"seed{seed}"
        / run_label
    )
    output_path = artifact_dir / "future_flow.pt"
    if output_path.exists() and not force:
        console.print(f"Phase 9 future flow exists: {output_path}")
        return output_path
    cache_path = prepare_phase8_latent_episodes(
        config, latent_dim=latent_dim, variant=variant, seed=seed, force=False
    )
    cached = torch.load(cache_path, map_location="cpu", weights_only=False)
    train_episodes = cached["train"]
    if trajectory_limit is not None:
        if trajectory_limit < 1 or trajectory_limit > len(train_episodes):
            raise ValueError(f"Phase 9 trajectory limit must be in [1, {len(train_episodes)}]")
        train_episodes = train_episodes[:trajectory_limit]
        validation_episodes = train_episodes
    else:
        validation_episodes = cached["validation"]
    encoder_checkpoint = torch.load(
        cached["encoder_checkpoint"], map_location="cpu", weights_only=False
    )
    action_norm = Standardizer.from_state_dict(encoder_checkpoint["action_norm"])
    latent_norm = Standardizer.fit(
        np.concatenate([episode["latents"] for episode in train_episodes], axis=0)
    )
    validation = _phase8_validation_samples(
        validation_episodes,
        1,
        horizon_steps,
        latent_norm,
        action_norm,
        int(config.get("incremental.phase9.validation_samples", 5000)),
        seed + 9000 + (trajectory_limit or 0),
    )
    batch_size = int(config.get("incremental.phase9.batch_size", 512))
    batches_per_epoch = int(
        config.get(
            "incremental.phase9.overfit_batches_per_epoch"
            if trajectory_limit is not None
            else "incremental.phase9.batches_per_epoch",
            50 if trajectory_limit is not None else 300,
        )
    )
    dataset = _Phase8FutureDataset(
        train_episodes,
        1,
        horizon_steps,
        latent_norm,
        latent_norm,
        "absolute",
        action_norm,
        length=batch_size * batches_per_epoch,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    device = default_device()
    hidden_dim = int(config.get("incremental.phase9.hidden_dim", 1024))
    model = FlowModel(latent_dim, latent_dim + len(action_norm.mean), hidden_dim).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(config.get("incremental.phase9.lr", 3e-4))
    )
    epochs = int(
        config.get(
            "incremental.phase9.overfit_epochs"
            if trajectory_limit is not None
            else "incremental.phase9.epochs",
            100 if trajectory_limit is not None else 60,
        )
    )
    flow_steps = int(config.get("incremental.phase9.flow_steps", 24))
    validation_interval = int(config.get("incremental.phase9.validation_interval", 10))
    validation_limit = min(
        int(config.get("incremental.phase9.endpoint_validation_samples", 2048)),
        len(validation["conditions"]),
    )
    val_condition = torch.from_numpy(validation["conditions"][:validation_limit]).to(device).float()
    val_target = (
        torch.from_numpy(latent_norm.transform(validation["future_latents"][:validation_limit]))
        .to(device)
        .float()
    )
    best_state = None
    best_endpoint_mse = float("inf")
    history = []
    timer = Timer()
    for epoch in trange(1, epochs + 1, desc=f"train phase9 flow {run_label}"):
        model.train()
        total = 0.0
        count = 0
        for condition, target in loader:
            condition = condition.to(device, non_blocking=True).float()
            target = target.to(device, non_blocking=True).float()
            loss = flow_matching_loss(model, target, condition)
            if endpoint_weight > 0.0:
                endpoint_count = min(
                    int(config.get("incremental.phase9.endpoint_consistency_batch", 128)),
                    len(condition),
                )
                endpoint = _integrate_flow_train(
                    model,
                    condition[:endpoint_count],
                    int(config.get("incremental.phase9.endpoint_consistency_steps", 4)),
                    latent_dim,
                    torch.zeros(
                        endpoint_count,
                        latent_dim,
                        device=device,
                        dtype=condition.dtype,
                    ),
                )
                loss = loss + endpoint_weight * torch.mean(
                    (endpoint - target[:endpoint_count]) ** 2
                )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += float(loss.detach().cpu()) * len(condition)
            count += len(condition)
        row = {"epoch": epoch, "flow_matching_loss": total / count}
        if epoch % validation_interval == 0 or epoch == epochs:
            model.eval()
            with torch.inference_mode():
                endpoint = sample_flow(
                    model,
                    val_condition,
                    flow_steps,
                    latent_dim,
                    initial_noise=torch.zeros_like(val_target),
                )
                endpoint_mse = float(torch.mean((endpoint - val_target) ** 2).cpu())
            row["zero_noise_endpoint_mse"] = endpoint_mse
            if endpoint_mse < best_endpoint_mse:
                best_endpoint_mse = endpoint_mse
                best_state = copy.deepcopy(model.state_dict())
        history.append(row)
    if best_state is None:
        raise RuntimeError("Phase 9 flow training produced no endpoint checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    all_condition = torch.from_numpy(validation["conditions"]).to(device).float()
    zero_predictions = []
    with torch.inference_mode():
        for start in range(0, len(all_condition), 1024):
            condition = all_condition[start : start + 1024]
            zero_predictions.append(
                sample_flow(
                    model,
                    condition,
                    flow_steps,
                    latent_dim,
                    initial_noise=torch.zeros(
                        len(condition), latent_dim, device=device, dtype=condition.dtype
                    ),
                )
                .cpu()
                .numpy()
            )
    predicted = latent_norm.inverse(np.concatenate(zero_predictions))
    target = validation["future_latents"]
    persistence = validation["current_latents"]
    prediction_l2 = np.linalg.norm(predicted - target, axis=-1)
    persistence_l2 = np.linalg.norm(persistence - target, axis=-1)

    low_path = train_phase7_oracle_low_level(
        config,
        latent_dim=latent_dim,
        variant=variant,
        horizon_steps=horizon_steps,
        action_chunk_steps=1,
        goal_encoding="delta",
        goal_dropout_prob=0.0,
        seed=seed,
        force=False,
    )
    low_model, low_checkpoint = _load_phase7_low_level_checkpoint(low_path, device)
    low_action_norm = Standardizer.from_state_dict(low_checkpoint["action_norm"])
    oracle_conditions = np.stack(
        [
            _phase7_condition(current, future, previous, "delta")
            for current, future, previous in zip(
                validation["current_latents"],
                target,
                validation["previous_actions_norm"],
            )
        ]
    )
    predicted_conditions = np.stack(
        [
            _phase7_condition(current, future, previous, "delta")
            for current, future, previous in zip(
                validation["current_latents"],
                predicted,
                validation["previous_actions_norm"],
            )
        ]
    )

    def low_actions(conditions: np.ndarray) -> np.ndarray:
        outputs = []
        with torch.inference_mode():
            for start in range(0, len(conditions), 4096):
                outputs.append(
                    low_model(torch.from_numpy(conditions[start : start + 4096]).to(device).float())
                    .cpu()
                    .numpy()
                )
        return low_action_norm.inverse(np.concatenate(outputs))

    oracle_actions = low_actions(oracle_conditions)
    predicted_actions = low_actions(predicted_conditions)
    teacher_actions = validation["teacher_actions"]
    oracle_mae = float(np.mean(np.abs(oracle_actions - teacher_actions)))
    predicted_mae = float(np.mean(np.abs(predicted_actions - teacher_actions)))
    stochastic_count = min(512, len(all_condition))
    stochastic_samples = []
    with torch.inference_mode():
        for _ in range(4):
            stochastic_samples.append(
                latent_norm.inverse(
                    sample_flow(
                        model,
                        all_condition[:stochastic_count],
                        flow_steps,
                        latent_dim,
                    )
                    .cpu()
                    .numpy()
                )
            )
    stochastic = np.stack(stochastic_samples, axis=1)
    stochastic_l2 = np.linalg.norm(stochastic - target[:stochastic_count, None, :], axis=-1)
    payload = {
        "model": model.state_dict(),
        "variant": variant,
        "latent_dim": latent_dim,
        "horizon_steps": horizon_steps,
        "history": 1,
        "condition_dim": latent_dim + len(action_norm.mean),
        "sample_dim": latent_dim,
        "hidden_dim": hidden_dim,
        "flow_steps": flow_steps,
        "endpoint_consistency_weight": endpoint_weight,
        "trajectory_limit": trajectory_limit,
        "encoder_checkpoint": cached["encoder_checkpoint"],
        "low_level_checkpoint": str(low_path),
        "latent_norm": latent_norm.state_dict(),
        "action_norm": action_norm.state_dict(),
        "offline_metrics": {
            "zero_noise_raw_latent_l2": float(np.mean(prediction_l2)),
            "persistence_raw_latent_l2": float(np.mean(persistence_l2)),
            "zero_noise_better_than_persistence_fraction": float(
                np.mean(prediction_l2 < persistence_l2)
            ),
            "oracle_goal_action_mae": oracle_mae,
            "zero_noise_goal_action_mae": predicted_mae,
            "zero_noise_to_oracle_action_mae_ratio": predicted_mae / max(oracle_mae, 1e-8),
            "stochastic_mean_latent_l2": float(np.mean(stochastic_l2)),
            "stochastic_best_of_4_latent_l2": float(np.mean(np.min(stochastic_l2, axis=1))),
            "stochastic_sample_diversity_l2": float(
                np.mean(np.linalg.norm(stochastic[:, 1:] - stochastic[:, :1], axis=-1))
            ),
            "validation_samples": len(target),
        },
        "best_zero_noise_endpoint_mse": best_endpoint_mse,
        "training_history": history,
        "elapsed_s": timer.elapsed(),
        "data": {
            "cache": str(cache_path),
            "train_trajectories": len(train_episodes),
            "validation_trajectories": len(validation_episodes),
        },
        "metadata": _runtime_metadata(config),
    }
    torch.save(payload, output_path)
    write_json(
        artifact_dir / "future_flow_metrics.json",
        {
            key: payload[key]
            for key in [
                "variant",
                "latent_dim",
                "horizon_steps",
                "trajectory_limit",
                "flow_steps",
                "endpoint_consistency_weight",
                "offline_metrics",
                "best_zero_noise_endpoint_mse",
                "training_history",
                "elapsed_s",
                "data",
                "metadata",
            ]
        },
    )
    console.print(f"Wrote Phase 9 future flow: {output_path}")
    return output_path


@torch.inference_mode()
def evaluate_phase9_future_flow(
    config: Config,
    sample_mode: str = "zero",
    latent_dim: int | None = None,
    variant: str | None = None,
    horizon_steps: int | None = None,
    seed: int = 0,
    episodes: int | None = None,
    force: bool = False,
    robust_low_method: str | None = None,
    interpolation_alpha: float = 0.5,
) -> Path:
    if sample_mode not in {"zero", "random"}:
        raise ValueError(f"Unknown Phase 9 sample mode: {sample_mode}")
    latent_dim, variant, horizon_steps = _phase8_defaults(
        config, latent_dim, variant, horizon_steps
    )
    with torch.inference_mode(False):
        checkpoint_path = train_phase9_future_flow(
            config,
            latent_dim=latent_dim,
            variant=variant,
            horizon_steps=horizon_steps,
            trajectory_limit=None,
            seed=seed,
            force=False,
        )
    eval_episodes = int(episodes or config.get("incremental.phase9.eval_episodes", 100))
    result_phase = "phase10" if robust_low_method is not None else "phase9"
    results_dir = ensure_dir(
        config.path_value("paths.incremental_results_dir")
        / result_phase
        / f"{variant}_z{latent_dim}_k{horizon_steps}_l1"
        / f"seed{seed}"
    )
    method_label = (
        "base"
        if robust_low_method is None
        else (
            f"interpolate_a{interpolation_alpha:g}".replace(".", "p")
            if robust_low_method == "interpolate"
            else robust_low_method
        )
    )
    output_path = results_dir / f"future_flow_{sample_mode}_{method_label}_{eval_episodes}.json"
    if output_path.exists() and not force:
        console.print(f"Phase 9 future-flow eval exists: {output_path}")
        return output_path
    device = default_device()
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    flow = FlowModel(
        int(checkpoint["sample_dim"]),
        int(checkpoint["condition_dim"]),
        int(checkpoint["hidden_dim"]),
    ).to(device)
    flow.load_state_dict(checkpoint["model"])
    flow.eval()
    encoder, encoder_checkpoint = _load_phase6_encoder(
        Path(checkpoint["encoder_checkpoint"]), device
    )
    frame_norm = Standardizer.from_state_dict(encoder_checkpoint["frame_norm"])
    latent_norm = Standardizer.from_state_dict(checkpoint["latent_norm"])
    action_norm = Standardizer.from_state_dict(checkpoint["action_norm"])
    low_path = Path(checkpoint["low_level_checkpoint"])
    if robust_low_method is not None:
        with torch.inference_mode(False):
            low_path = train_phase10_robust_low_level(
                config,
                method=robust_low_method,
                latent_dim=latent_dim,
                variant=variant,
                horizon_steps=horizon_steps,
                interpolation_alpha=interpolation_alpha,
                seed=seed,
                force=False,
            )
    low_model, low_checkpoint = _load_phase7_low_level_checkpoint(low_path, device)
    oracle_low_path = Path(low_checkpoint.get("base_low_checkpoint", low_path))
    low_action_norm = Standardizer.from_state_dict(low_checkpoint["action_norm"])
    if not (
        np.array_equal(action_norm.mean, low_action_norm.mean)
        and np.array_equal(action_norm.std, low_action_norm.std)
    ):
        raise ValueError("Phase 9 flow and low level use different action normalization")
    dino = _phase4_dino_from_config(config, device)
    teacher = load_ppo_agent(_rl_paths(config).best, device)
    num_envs = min(int(config.get("incremental.phase9.eval_num_envs", 64)), eval_episodes)
    env = _phase4_make_visual_env(config, num_envs)
    action_low = torch.as_tensor(env.single_action_space.low, device=device, dtype=torch.float32)
    action_high = torch.as_tensor(env.single_action_space.high, device=device, dtype=torch.float32)
    zero_action = action_norm.transform(
        np.zeros((1, int(low_checkpoint["action_dim"])), dtype=np.float32)
    )[0]
    previous_action = np.repeat(zero_action[None], num_envs, axis=0).astype(np.float32)
    seed_start = int(config.get("incremental.phase9.eval_seed", 1_200_000))
    obs, _info = env.reset(seed=seed_start)
    successes = []
    final_rewards = []
    max_rewards = []
    episode_lengths = []
    teacher_maes = []
    goal_displacements = []
    latencies = []
    active_max_reward = np.full(num_envs, -np.inf, dtype=np.float32)
    active_lengths = np.zeros(num_envs, dtype=np.int32)
    while len(successes) < eval_episodes:
        timer = Timer()
        frames = frame_norm.transform(
            _phase4_frame_inputs(obs, dino, int(config.get("dino.batch_size", 64)))
        )
        z = encoder(torch.from_numpy(frames).to(device).float()).cpu().numpy().astype(np.float32)
        condition = np.concatenate([latent_norm.transform(z), previous_action], axis=-1)
        condition_t = torch.from_numpy(condition).to(device).float()
        initial_noise = (
            torch.zeros(num_envs, latent_dim, device=device, dtype=condition_t.dtype)
            if sample_mode == "zero"
            else None
        )
        predicted_goal = latent_norm.inverse(
            sample_flow(
                flow,
                condition_t,
                int(checkpoint["flow_steps"]),
                latent_dim,
                initial_noise=initial_noise,
            )
            .cpu()
            .numpy()
        )
        goal_displacements.extend(np.linalg.norm(predicted_goal - z, axis=-1).tolist())
        low_condition = np.stack(
            [
                _phase7_condition(z[i], predicted_goal[i], previous_action[i], "delta")
                for i in range(num_envs)
            ]
        )
        raw_action = action_norm.inverse(
            low_model(torch.from_numpy(low_condition).to(device).float()).cpu().numpy()
        ).astype(np.float32)
        teacher_action = (
            torch.clamp(
                teacher.actor_mean(obs["state"].to(device).float()), action_low, action_high
            )
            .cpu()
            .numpy()
            .astype(np.float32)
        )
        teacher_maes.extend(np.mean(np.abs(raw_action - teacher_action), axis=-1).tolist())
        latencies.append(timer.elapsed() / num_envs)
        action = torch.from_numpy(raw_action).to(device).float()
        if bool(config.get("policy.clip_actions_to_env_space", True)):
            action = torch.clamp(action, action_low, action_high)
        previous_action = action_norm.transform(action.cpu().numpy().astype(np.float32))
        obs, reward, _terminated, _truncated, info = env.step(action)
        reward_np = _numpy(reward).reshape(-1).astype(np.float32)
        active_max_reward = np.maximum(active_max_reward, reward_np)
        active_lengths += 1
        if "final_info" in info:
            mask = _numpy(info["_final_info"]).reshape(-1).astype(bool)
            if mask.any():
                success_once = _numpy(info["final_info"]["episode"]["success_once"]).reshape(-1)
                for env_idx in np.flatnonzero(mask):
                    successes.append(float(success_once[env_idx]))
                    final_rewards.append(float(reward_np[env_idx]))
                    max_rewards.append(float(active_max_reward[env_idx]))
                    episode_lengths.append(int(active_lengths[env_idx]))
                    active_max_reward[env_idx] = -np.inf
                    active_lengths[env_idx] = 0
                    previous_action[env_idx] = zero_action
                    if len(successes) >= eval_episodes:
                        break
    env.close()
    metrics = {
        "success": float(np.mean(successes[:eval_episodes])),
        "success_stderr": float(np.std(successes[:eval_episodes]) / np.sqrt(eval_episodes)),
        "final_reward": float(np.mean(final_rewards[:eval_episodes])),
        "max_reward": float(np.mean(max_rewards[:eval_episodes])),
        "mean_episode_length": float(np.mean(episode_lengths[:eval_episodes])),
        "teacher_action_mae": float(np.mean(teacher_maes)),
        "predicted_goal_displacement_l2": float(np.mean(goal_displacements)),
        "inference_latency_s": float(np.mean(latencies)),
        "episodes": eval_episodes,
        "num_envs": num_envs,
        "seed_start": seed_start,
    }
    oracle_result_dir = (
        config.path_value("paths.incremental_results_dir")
        / "phase7"
        / _phase7_tag(variant, latent_dim, horizon_steps, 1, "delta", 0.0)
        / f"seed{seed}"
    )
    import json

    oracle_results = []
    for candidate in oracle_result_dir.glob("replay_branch_oracle_eval*.json"):
        with candidate.open("r", encoding="utf-8") as f:
            result = json.load(f)
        if Path(result.get("checkpoint", "")) == oracle_low_path:
            oracle_results.append((int(result["closed_loop"]["episodes"]), candidate, result))
    if not oracle_results:
        raise FileNotFoundError(f"No Phase 7 oracle result matches {oracle_low_path}")
    _count, oracle_path, oracle_result = max(oracle_results, key=lambda item: item[0])
    oracle_success = float(oracle_result["closed_loop"]["success"])
    payload = {
        "phase": 10 if robust_low_method is not None else 9,
        "method": "conditional_future_latent_flow",
        "variant": variant,
        "latent_dim": latent_dim,
        "horizon_steps": horizon_steps,
        "sample_mode": sample_mode,
        "robust_low_method": robust_low_method,
        "interpolation_alpha": interpolation_alpha,
        "seed": seed,
        "checkpoint": str(checkpoint_path),
        "closed_loop": metrics,
        "offline_metrics": checkpoint["offline_metrics"],
        "oracle_result": str(oracle_path),
        "oracle_success": oracle_success,
        "success_fraction_of_oracle": metrics["success"] / max(oracle_success, 1e-8),
        "gate_vs_deterministic": metrics["success"] >= 0.46,
        "gate_70pct_oracle": metrics["success"] >= 0.70 * oracle_success,
        "gate_80pct_oracle": metrics["success"] >= 0.80 * oracle_success,
        "metadata": _runtime_metadata(config),
    }
    write_json(output_path, payload)
    console.print(payload)
    return output_path


@torch.inference_mode()
def collect_phase10_flow_queries(
    config: Config,
    latent_dim: int | None = None,
    variant: str | None = None,
    horizon_steps: int | None = None,
    seed: int = 0,
    episodes: int | None = None,
    force: bool = False,
) -> Path:
    latent_dim, variant, horizon_steps = _phase8_defaults(
        config, latent_dim, variant, horizon_steps
    )
    episodes = int(episodes or config.get("incremental.phase10.query_episodes", 10))
    flow_path = train_phase9_future_flow(
        config,
        latent_dim=latent_dim,
        variant=variant,
        horizon_steps=horizon_steps,
        seed=seed,
        force=False,
    )
    artifact_dir = ensure_dir(
        config.path_value("paths.incremental_artifact_dir")
        / "phase10"
        / f"{variant}_z{latent_dim}_k{horizon_steps}"
        / f"seed{seed}"
    )
    output_path = artifact_dir / f"flow_queries_e{episodes}.npz"
    if output_path.exists() and not force:
        console.print(f"Phase 10 flow queries exist: {output_path}")
        return output_path
    device = default_device()
    checkpoint = torch.load(flow_path, map_location=device, weights_only=False)
    flow = FlowModel(
        int(checkpoint["sample_dim"]),
        int(checkpoint["condition_dim"]),
        int(checkpoint["hidden_dim"]),
    ).to(device)
    flow.load_state_dict(checkpoint["model"])
    flow.eval()
    encoder, encoder_checkpoint = _load_phase6_encoder(
        Path(checkpoint["encoder_checkpoint"]), device
    )
    frame_norm = Standardizer.from_state_dict(encoder_checkpoint["frame_norm"])
    latent_norm = Standardizer.from_state_dict(checkpoint["latent_norm"])
    action_norm = Standardizer.from_state_dict(checkpoint["action_norm"])
    low_model, low_checkpoint = _load_phase7_low_level_checkpoint(
        Path(checkpoint["low_level_checkpoint"]), device
    )
    teacher = load_ppo_agent(_rl_paths(config).best, device)
    dino = _phase4_dino_from_config(config, device)

    def make_env(num_envs: int):
        return gym.make(
            config.get("env_id"),
            obs_mode="rgb+state",
            control_mode=config.get("control_mode"),
            reward_mode="normalized_dense",
            render_mode=None,
            sim_backend=_rl_backend(config),
            num_envs=num_envs,
            reconfiguration_freq=config.get("rl.eval_reconfiguration_freq", 1),
        )

    zero_action = action_norm.transform(
        np.zeros((1, int(low_checkpoint["action_dim"])), dtype=np.float32)
    )[0]
    max_num_envs = min(int(config.get("incremental.phase10.query_num_envs", 16)), episodes)
    seed_start = int(config.get("incremental.phase10.query_seed", 1_400_000))
    replay_tolerance = float(config.get("incremental.phase10.replay_tolerance", 1e-6))
    max_episode_steps = int(config.get("env_max_episode_steps", 100))
    current_rows = []
    generated_rows = []
    branch_rows = []
    previous_rows = []
    teacher_action_rows = []
    successes = []
    replay_errors = []
    failed_replay_steps = 0
    progress = trange(episodes, desc="phase10 collect flow-goal queries")
    for batch_start in range(0, episodes, max_num_envs):
        num_envs = min(max_num_envs, episodes - batch_start)
        reset_seeds = [seed_start + batch_start + i for i in range(num_envs)]
        student_env = make_env(num_envs)
        branch_env = make_env(num_envs)
        action_low_np = np.asarray(student_env.action_space.low, dtype=np.float32)
        action_high_np = np.asarray(student_env.action_space.high, dtype=np.float32)
        if action_low_np.ndim == 2:
            action_low_np = action_low_np[0]
            action_high_np = action_high_np[0]
        action_low = torch.as_tensor(action_low_np, device=device, dtype=torch.float32)
        action_high = torch.as_tensor(action_high_np, device=device, dtype=torch.float32)
        try:
            obs, _info = student_env.reset(seed=reset_seeds)
            action_history = []
            previous_action = np.repeat(zero_action[None], num_envs, axis=0).astype(np.float32)
            active = np.ones(num_envs, dtype=bool)
            success_once = np.zeros(num_envs, dtype=bool)
            for _step in range(max_episode_steps):
                if not active.any():
                    break
                branch_obs, _branch_info = branch_env.reset(seed=reset_seeds)
                replay_done = torch.zeros(num_envs, device=device, dtype=torch.bool)
                for historical_action in action_history:
                    branch_obs, _reward, term, trunc, _info = branch_env.step(historical_action)
                    replay_done |= torch.logical_or(term, trunc).view(-1)
                state_error = torch.max(
                    torch.abs(student_env.unwrapped.get_state() - branch_env.unwrapped.get_state()),
                    dim=1,
                ).values
                state_error_np = state_error.cpu().numpy()
                replay_errors.extend(state_error_np[active].tolist())
                failed_replay_steps += int(
                    np.sum(
                        active
                        & (
                            replay_done.cpu().numpy().astype(bool)
                            | (state_error_np > replay_tolerance)
                        )
                    )
                )
                for _ in range(horizon_steps):
                    branch_action = torch.clamp(
                        teacher.actor_mean(branch_obs["state"].to(device).float()),
                        action_low,
                        action_high,
                    )
                    branch_obs, _reward, _term, _trunc, _info = branch_env.step(branch_action)
                current_frame = _phase4_frame_inputs(
                    obs, dino, int(config.get("dino.batch_size", 64))
                )
                branch_frame = _phase4_frame_inputs(
                    branch_obs, dino, int(config.get("dino.batch_size", 64))
                )
                frames = frame_norm.transform(np.concatenate([current_frame, branch_frame]))
                pair = encoder(torch.from_numpy(frames).to(device).float()).cpu().numpy()
                current = pair[:num_envs].astype(np.float32)
                branch_goal = pair[num_envs:].astype(np.float32)
                high_condition = np.concatenate(
                    [latent_norm.transform(current), previous_action], axis=-1
                )
                generated_goal = latent_norm.inverse(
                    sample_flow(
                        flow,
                        torch.from_numpy(high_condition).to(device).float(),
                        int(checkpoint["flow_steps"]),
                        latent_dim,
                        initial_noise=torch.zeros(
                            num_envs, latent_dim, device=device, dtype=torch.float32
                        ),
                    )
                    .cpu()
                    .numpy()
                )
                low_condition = np.stack(
                    [
                        _phase7_condition(
                            current[i], generated_goal[i], previous_action[i], "delta"
                        )
                        for i in range(num_envs)
                    ]
                )
                raw_action = action_norm.inverse(
                    low_model(torch.from_numpy(low_condition).to(device).float()).cpu().numpy()
                ).astype(np.float32)
                teacher_action = (
                    torch.clamp(
                        teacher.actor_mean(obs["state"].to(device).float()),
                        action_low,
                        action_high,
                    )
                    .cpu()
                    .numpy()
                    .astype(np.float32)
                )
                current_rows.append(current[active])
                generated_rows.append(generated_goal[active])
                branch_rows.append(branch_goal[active])
                previous_rows.append(previous_action[active])
                teacher_action_rows.append(teacher_action[active])
                action = torch.from_numpy(raw_action).to(device).float()
                if bool(config.get("policy.clip_actions_to_env_space", True)):
                    action = torch.clamp(action, action_low, action_high)
                action[~torch.from_numpy(active).to(device)] = 0.0
                obs, _reward, terminated, truncated, info = student_env.step(action)
                action_history.append(action.detach().clone())
                previous_action = action_norm.transform(action.cpu().numpy().astype(np.float32))
                if "success" in info:
                    success_once |= _numpy(info["success"]).reshape(-1).astype(bool)
                done = _numpy(torch.logical_or(terminated, truncated)).reshape(-1).astype(bool)
                newly_done = active & done
                if np.any(newly_done):
                    progress.update(int(newly_done.sum()))
                    active[newly_done] = False
            if np.any(active):
                progress.update(int(active.sum()))
            successes.extend(success_once.astype(np.float32).tolist())
        finally:
            student_env.close()
            branch_env.close()
    progress.close()
    current = np.concatenate(current_rows).astype(np.float32)
    generated = np.concatenate(generated_rows).astype(np.float32)
    branch = np.concatenate(branch_rows).astype(np.float32)
    np.savez_compressed(
        output_path,
        current_latents=current,
        generated_goals=generated,
        branch_goals=branch,
        previous_actions_norm=np.concatenate(previous_rows).astype(np.float32),
        teacher_actions=np.concatenate(teacher_action_rows).astype(np.float32),
        generated_residuals=(generated - branch).astype(np.float32),
        future_displacements=(branch - current).astype(np.float32),
        collection_success=np.asarray(successes, dtype=np.float32),
        replay_current_state_error_mean=np.asarray(np.mean(replay_errors), dtype=np.float32),
        replay_current_state_error_max=np.asarray(np.max(replay_errors), dtype=np.float32),
        replay_failed_step_fraction=np.asarray(
            failed_replay_steps / max(1, len(replay_errors)), dtype=np.float32
        ),
        flow_checkpoint=np.asarray(str(flow_path)),
        dataset_type=np.asarray("state_query_dataset"),
        semantics=np.asarray(
            "flow-hierarchy states with generated goal, exact local branch goal, and teacher action"
        ),
    )
    console.print(f"Wrote Phase 10 flow queries: {output_path}")
    return output_path


def train_phase10_robust_low_level(
    config: Config,
    method: str,
    latent_dim: int | None = None,
    variant: str | None = None,
    horizon_steps: int | None = None,
    interpolation_alpha: float = 0.5,
    seed: int = 0,
    query_episodes: int | None = None,
    force: bool = False,
) -> Path:
    if method not in {"direct", "interpolate", "empirical", "covariance_diag"}:
        raise ValueError(f"Unknown Phase 10 robustness method: {method}")
    if not 0.0 <= interpolation_alpha <= 1.0:
        raise ValueError("Phase 10 interpolation alpha must be in [0, 1]")
    set_seed(seed)
    latent_dim, variant, horizon_steps = _phase8_defaults(
        config, latent_dim, variant, horizon_steps
    )
    query_episodes = int(query_episodes or config.get("incremental.phase10.query_episodes", 10))
    query_path = collect_phase10_flow_queries(
        config,
        latent_dim=latent_dim,
        variant=variant,
        horizon_steps=horizon_steps,
        seed=seed,
        episodes=query_episodes,
        force=False,
    )
    flow_path = train_phase9_future_flow(
        config,
        latent_dim=latent_dim,
        variant=variant,
        horizon_steps=horizon_steps,
        seed=seed,
        force=False,
    )
    flow_checkpoint = torch.load(flow_path, map_location="cpu", weights_only=False)
    base_low_path = Path(flow_checkpoint["low_level_checkpoint"])
    method_label = (
        f"interpolate_a{interpolation_alpha:g}".replace(".", "p")
        if method == "interpolate"
        else method
    )
    output_path = query_path.parent / f"robust_low_{method_label}.pt"
    if output_path.exists() and not force:
        console.print(f"Phase 10 robust low level exists: {output_path}")
        return output_path
    device = default_device()
    base_model, base_checkpoint = _load_phase7_low_level_checkpoint(base_low_path, device)
    action_norm = Standardizer.from_state_dict(base_checkpoint["action_norm"])
    cache = torch.load(
        _phase8_latent_cache_path(config, variant, latent_dim, seed),
        map_location="cpu",
        weights_only=False,
    )
    nominal_x, nominal_actions = _phase8_nominal_low_level_samples(
        cache["train"], horizon_steps, action_norm
    )
    nominal_val_x, nominal_val_actions = _phase8_nominal_low_level_samples(
        cache["validation"], horizon_steps, action_norm
    )
    with np.load(query_path) as data:
        current = np.asarray(data["current_latents"], dtype=np.float32)
        generated = np.asarray(data["generated_goals"], dtype=np.float32)
        branch = np.asarray(data["branch_goals"], dtype=np.float32)
        previous = np.asarray(data["previous_actions_norm"], dtype=np.float32)
        teacher_actions = np.asarray(data["teacher_actions"], dtype=np.float32)
        residuals = np.asarray(data["generated_residuals"], dtype=np.float32)
    rng = np.random.default_rng(seed + 10_000)
    permutation = rng.permutation(len(current))
    split = max(1, int(0.8 * len(permutation)))
    query_train = permutation[:split]
    query_val = permutation[split:]
    if len(query_val) == 0:
        raise ValueError("Phase 10 requires at least two generated-goal queries")

    if method in {"direct", "interpolate"}:
        robust_goal = (
            generated if method == "direct" else branch + interpolation_alpha * (generated - branch)
        )
        robust_x = np.stack(
            [
                _phase7_condition(current[i], robust_goal[i], previous[i], "delta")
                for i in range(len(current))
            ]
        ).astype(np.float32)
        repeat = max(1, int(np.ceil(len(nominal_x) / len(query_train))))
        robust_train_x = np.tile(robust_x[query_train], (repeat, 1))[: len(nominal_x)]
        robust_train_y = np.tile(action_norm.transform(teacher_actions[query_train]), (repeat, 1))[
            : len(nominal_x)
        ]
    else:
        nominal_current = nominal_x[:, :latent_dim]
        nominal_delta = nominal_x[:, latent_dim : 2 * latent_dim]
        nominal_previous = nominal_x[:, 2 * latent_dim :]
        train_residuals = residuals[query_train]
        if method == "empirical":
            sampled = train_residuals[rng.integers(0, len(train_residuals), size=len(nominal_x))]
        else:
            sampled = rng.normal(
                train_residuals.mean(axis=0),
                np.maximum(train_residuals.std(axis=0), 1e-4),
                size=(len(nominal_x), latent_dim),
            ).astype(np.float32)
        robust_train_x = np.concatenate(
            [nominal_current, nominal_delta + sampled, nominal_previous], axis=-1
        ).astype(np.float32)
        robust_train_y = action_norm.transform(nominal_actions)
    train_x = np.concatenate([nominal_x, robust_train_x])
    train_y = np.concatenate([action_norm.transform(nominal_actions), robust_train_y])
    dataset = TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y))
    loader = DataLoader(
        dataset,
        batch_size=int(config.get("incremental.phase10.batch_size", 512)),
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    model = MLP(
        int(base_checkpoint["cond_dim"]),
        int(base_checkpoint["action_dim"]),
        int(base_checkpoint["hidden_dim"]),
        depth=4,
    ).to(device)
    model.load_state_dict(base_model.state_dict())
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(config.get("incremental.phase10.lr", 1e-4))
    )
    generated_query_x = np.stack(
        [
            _phase7_condition(current[i], generated[i], previous[i], "delta")
            for i in range(len(current))
        ]
    ).astype(np.float32)
    branch_query_x = np.stack(
        [
            _phase7_condition(current[i], branch[i], previous[i], "delta")
            for i in range(len(current))
        ]
    ).astype(np.float32)
    nominal_val_x_t = torch.from_numpy(nominal_val_x).to(device).float()
    nominal_val_y_t = (
        torch.from_numpy(action_norm.transform(nominal_val_actions)).to(device).float()
    )
    generated_val_x_t = torch.from_numpy(generated_query_x[query_val]).to(device).float()
    generated_val_y_t = (
        torch.from_numpy(action_norm.transform(teacher_actions[query_val])).to(device).float()
    )
    best_state = None
    best_selection = float("inf")
    history = []
    timer = Timer()
    epochs = int(config.get("incremental.phase10.epochs", 30))
    for epoch in trange(1, epochs + 1, desc=f"phase10 robust low {method_label}"):
        model.train()
        total = 0.0
        count = 0
        for x, y in loader:
            x = x.to(device, non_blocking=True).float()
            y = y.to(device, non_blocking=True).float()
            loss = torch.mean((model(x) - y) ** 2)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += float(loss.detach().cpu()) * len(x)
            count += len(x)
        model.eval()
        with torch.inference_mode():
            nominal_mse = float(torch.mean((model(nominal_val_x_t) - nominal_val_y_t) ** 2).cpu())
            generated_mse = float(
                torch.mean((model(generated_val_x_t) - generated_val_y_t) ** 2).cpu()
            )
        selection = nominal_mse + generated_mse
        history.append(
            {
                "epoch": epoch,
                "train_mse": total / count,
                "nominal_validation_mse": nominal_mse,
                "generated_query_mse": generated_mse,
                "selection": selection,
            }
        )
        if selection < best_selection:
            best_selection = selection
            best_state = copy.deepcopy(model.state_dict())
    if best_state is None:
        raise RuntimeError("Phase 10 robust low-level training produced no checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    nominal_metrics = _phase7_oracle_action_metrics(
        model,
        nominal_val_x,
        nominal_val_actions,
        action_norm,
        latent_dim,
        "delta",
        int(config.get("incremental.phase7.validation_queries", 10000)),
        seed + 10_001,
    )

    def query_actions(conditions: np.ndarray) -> np.ndarray:
        with torch.inference_mode():
            normalized = model(torch.from_numpy(conditions[query_val]).to(device).float())
        return action_norm.inverse(normalized.cpu().numpy())

    generated_prediction = query_actions(generated_query_x)
    branch_prediction = query_actions(branch_query_x)
    target = teacher_actions[query_val]
    generated_mae = float(np.mean(np.abs(generated_prediction - target)))
    branch_mae = float(np.mean(np.abs(branch_prediction - target)))
    payload = {
        **base_checkpoint,
        "model": model.state_dict(),
        "training_mode": f"balanced_nominal_and_{method_label}",
        "robustness_method": method,
        "interpolation_alpha": interpolation_alpha,
        "base_low_checkpoint": str(base_low_path),
        "query_path": str(query_path),
        "query_episodes": query_episodes,
        "query_samples": len(current),
        "query_validation_samples": len(query_val),
        "validation_metrics": nominal_metrics,
        "generated_query_metrics": {
            "generated_goal_action_mae": generated_mae,
            "branch_goal_action_mae": branch_mae,
            "generated_to_branch_mae_ratio": generated_mae / max(branch_mae, 1e-8),
        },
        "robustness_history": history,
        "robustness_best_selection": best_selection,
        "robustness_elapsed_s": timer.elapsed(),
        "metadata": _runtime_metadata(config),
    }
    torch.save(payload, output_path)
    write_json(
        query_path.parent / f"robust_low_{method_label}_metrics.json",
        {
            key: payload[key]
            for key in [
                "training_mode",
                "robustness_method",
                "interpolation_alpha",
                "query_episodes",
                "query_samples",
                "query_validation_samples",
                "validation_metrics",
                "generated_query_metrics",
                "robustness_best_selection",
                "robustness_elapsed_s",
                "metadata",
            ]
        },
    )
    console.print(f"Wrote Phase 10 robust low level: {output_path}")
    return output_path


def run_phase11_comparison(
    config: Config,
    seed: int = 0,
    episodes: int = 100,
    eval_seed_start: int = 1_200_000,
) -> Path:
    """Build the matched complete-hierarchy comparison and summary plot."""
    visual_bc_path = evaluate_phase4_visual_bc(
        config,
        history=1,
        architecture="concat",
        seed=seed,
        episodes=episodes,
        eval_seed_start=eval_seed_start,
    )
    visual_flow_path = evaluate_phase5_visual_flow(
        config,
        history=1,
        architecture="concat",
        seed=seed,
        episodes=episodes,
        eval_seed_start=eval_seed_start,
    )
    root = config.path_value("paths.incremental_results_dir")
    paths = {
        "privileged_bc": root
        / "phase1"
        / "n2000"
        / f"seed{seed}"
        / "bc_all_deterministic_clipped.json",
        "privileged_flow": root / "phase3" / f"seed{seed}" / "one_step_flow.json",
        "visual_bc": visual_bc_path,
        "visual_flat_flow": visual_flow_path,
        "flat_latent": root
        / "phase7"
        / "matched_flat"
        / "ae_recon_z256"
        / f"seed{seed}"
        / "matched_flat_latent_eval_100.json",
        "oracle_latent_hierarchy": root
        / "phase7"
        / "ae_recon_z256_k2_h1_delta"
        / f"seed{seed}"
        / "replay_branch_oracle_eval_branch_dagger_iter1_e10_100.json",
        "structured_predicted_hierarchy": root
        / "phase8"
        / "structured_k2"
        / f"seed{seed}"
        / "deterministic_hierarchy_100.json",
        "deterministic_latent_hierarchy": root
        / "phase8"
        / "ae_recon_z256_k2_l1"
        / f"seed{seed}"
        / "deterministic_hierarchy_100.json",
        "generative_latent_hierarchy": root
        / "phase9"
        / "ae_recon_z256_k2_l1"
        / f"seed{seed}"
        / "future_flow_zero_100.json",
    }
    import json

    rows = []
    for method, path in paths.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing Phase 11 input for {method}: {path}")
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        closed = payload["closed_loop"]
        if "correct" in closed:
            closed = closed["correct"]
        rows.append(
            {
                "method": method,
                "source": str(path),
                "success": float(closed["success"]),
                "success_stderr": float(closed["success_stderr"]),
                "final_reward": float(closed["final_reward"]),
                "max_reward": float(closed["max_reward"]),
                "inference_latency_s": float(
                    closed.get("inference_latency_s", closed.get("policy_latency_s", 0.0))
                ),
                "episodes": int(closed["episodes"]),
                "eval_seed_start": int(closed["seed_start"]),
                "paired_visual_seed_range": int(closed["seed_start"]) == eval_seed_start,
            }
        )
    by_method = {row["method"]: row for row in rows}
    learned_best = max(
        by_method["deterministic_latent_hierarchy"]["success"],
        by_method["generative_latent_hierarchy"]["success"],
    )
    flat_visual = by_method["visual_flat_flow"]["success"]
    output_dir = ensure_dir(root / "phase11")
    output_path = output_dir / "complete_hierarchy_comparison.json"
    plot_path = output_dir / "complete_hierarchy_comparison.png"
    payload = {
        "phase": 11,
        "method": "complete_hierarchy_comparison",
        "episodes": episodes,
        "policy_seed": seed,
        "paired_eval_seed_start": eval_seed_start,
        "rows": rows,
        "fairness": {
            "visual_methods_paired": all(
                row["paired_visual_seed_range"]
                for row in rows
                if row["method"]
                in {
                    "visual_bc",
                    "visual_flat_flow",
                    "flat_latent",
                    "oracle_latent_hierarchy",
                    "deterministic_latent_hierarchy",
                    "generative_latent_hierarchy",
                }
            ),
            "privileged_reference_seed_caveat": (
                "privileged_bc and privileged_flow retain their original seed-10000 evaluations"
            ),
        },
        "gate": {
            "best_learned_hierarchy_success": learned_best,
            "visual_flat_flow_success": flat_visual,
            "higher_success_than_flat_flow": learned_best > flat_visual,
            "oracle_advantage_over_flat_flow": (
                by_method["oracle_latent_hierarchy"]["success"] > flat_visual
            ),
            "passed": learned_best > flat_visual,
        },
        "plot": str(plot_path),
        "metadata": _runtime_metadata(config),
    }
    write_json(output_path, payload)

    import matplotlib.pyplot as plt

    labels = [row["method"].replace("_", " ") for row in rows]
    success = [row["success"] for row in rows]
    stderr = [row["success_stderr"] for row in rows]
    final_reward = [row["final_reward"] for row in rows]
    max_reward = [row["max_reward"] for row in rows]
    colors = ["#6b7280" if row["method"].startswith("privileged") else "#2563eb" for row in rows]
    for index, row in enumerate(rows):
        if "hierarchy" in row["method"]:
            colors[index] = "#d97706" if "oracle" not in row["method"] else "#059669"
    figure, axes = plt.subplots(1, 2, figsize=(15, 6))
    x = np.arange(len(rows))
    axes[0].bar(x, success, yerr=stderr, capsize=3, color=colors)
    axes[0].set_ylabel("Success rate")
    axes[0].set_ylim(0.0, 1.0)
    axes[0].set_title("Push-T success (100 fixed episodes)")
    width = 0.38
    axes[1].bar(x - width / 2, final_reward, width, label="Final reward", color="#2563eb")
    axes[1].bar(x + width / 2, max_reward, width, label="Maximum reward", color="#d97706")
    axes[1].set_ylabel("Normalized reward")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].set_title("Reward comparison")
    axes[1].legend()
    for axis in axes:
        axis.set_xticks(x)
        axis.set_xticklabels(labels, rotation=35, ha="right")
        axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(plot_path, dpi=180)
    plt.close(figure)
    console.print(payload)
    return output_path


def _phase12_budget_config(config: Config, n_trajectories: int) -> Config:
    raw = copy.deepcopy(config.raw)
    raw["paths"]["incremental_artifact_dir"] = str(
        config.path_value("paths.incremental_artifact_dir") / "phase12" / f"n{n_trajectories}"
    )
    raw["paths"]["incremental_results_dir"] = str(
        config.path_value("paths.incremental_results_dir") / "phase12" / f"n{n_trajectories}"
    )
    raw["incremental"]["phase4"]["train_episodes"] = n_trajectories
    raw["incremental"]["phase6"]["train_episodes"] = n_trajectories
    return Config(raw=raw, path=config.path)


def run_phase12_budget(
    config: Config,
    n_trajectories: int,
    seed: int = 0,
    episodes: int = 100,
    eval_seed_start: int = 1_200_000,
) -> Path:
    budgets = [50, 100, 200, 500, 1000, 1800]
    if n_trajectories not in budgets:
        raise ValueError(f"Phase 12 trajectory budget must be one of {budgets}")
    oracle_episodes = int(config.get("incremental.phase12.oracle_eval_episodes", 10))
    budget_config = _phase12_budget_config(config, n_trajectories)
    output_dir = ensure_dir(
        config.path_value("paths.incremental_results_dir") / "phase12" / f"n{n_trajectories}"
    )
    output_path = output_dir / "sample_efficiency_summary.json"
    if output_path.exists():
        console.print(f"Phase 12 budget summary exists: {output_path}")
        return output_path

    visual_bc_path = evaluate_phase4_visual_bc(
        budget_config,
        history=1,
        architecture="concat",
        seed=seed,
        episodes=episodes,
        eval_seed_start=eval_seed_start,
    )
    visual_flow_path = evaluate_phase5_visual_flow(
        budget_config,
        history=1,
        architecture="concat",
        seed=seed,
        episodes=episodes,
        eval_seed_start=eval_seed_start,
    )
    train_phase6_representation(
        budget_config,
        latent_dim=256,
        variant="ae_recon",
        seed=seed,
        force=False,
    )
    train_phase7_oracle_low_level(
        budget_config,
        latent_dim=256,
        variant="ae_recon",
        horizon_steps=2,
        action_chunk_steps=1,
        goal_encoding="delta",
        goal_dropout_prob=0.0,
        seed=seed,
        force=False,
    )
    oracle_path = evaluate_phase7_replay_branch_oracle_low_level(
        budget_config,
        latent_dim=256,
        variant="ae_recon",
        horizon_steps=2,
        action_chunk_steps=1,
        goal_encoding="delta",
        goal_dropout_prob=0.0,
        seed=seed,
        episodes=oracle_episodes,
        force=False,
    )
    train_phase8_deterministic_predictor(
        budget_config,
        history=1,
        latent_dim=256,
        variant="ae_recon",
        horizon_steps=2,
        target_mode="absolute",
        seed=seed,
        force=False,
    )
    deterministic_path = evaluate_phase8_deterministic_hierarchy(
        budget_config,
        history=1,
        latent_dim=256,
        variant="ae_recon",
        horizon_steps=2,
        target_mode="absolute",
        seed=seed,
        episodes=episodes,
        force=False,
    )
    train_phase9_future_flow(
        budget_config,
        latent_dim=256,
        variant="ae_recon",
        horizon_steps=2,
        seed=seed,
        force=False,
    )
    generative_path = evaluate_phase9_future_flow(
        budget_config,
        sample_mode="zero",
        latent_dim=256,
        variant="ae_recon",
        horizon_steps=2,
        seed=seed,
        episodes=episodes,
        force=False,
    )
    train_episodes, _validation_episodes, _metadata = _load_phase4_episodes(budget_config)
    transitions = int(sum(len(episode["actions"]) for episode in train_episodes))

    import json

    method_paths = {
        "visual_bc": visual_bc_path,
        "visual_flat_flow": visual_flow_path,
        "oracle_hierarchy": oracle_path,
        "deterministic_hierarchy": deterministic_path,
        "generative_hierarchy": generative_path,
    }
    rows = []
    for method, path in method_paths.items():
        with path.open("r", encoding="utf-8") as f:
            result = json.load(f)
        closed = result["closed_loop"]
        rows.append(
            {
                "method": method,
                "source": str(path),
                "success": float(closed["success"]),
                "success_stderr": float(closed["success_stderr"]),
                "final_reward": float(closed["final_reward"]),
                "max_reward": float(closed["max_reward"]),
                "episodes": int(closed["episodes"]),
                "eval_seed_start": int(closed["seed_start"]),
            }
        )
    payload = {
        "phase": 12,
        "n_trajectories": n_trajectories,
        "causal_transitions": transitions,
        "equivalent_behavior_seconds": transitions / float(config.get("control_freq", 20)),
        "state_query_samples": 0,
        "validation_trajectories": 200,
        "policy_seed": seed,
        "evaluation_episodes": episodes,
        "oracle_evaluation_episodes": oracle_episodes,
        "evaluation_seed_start": eval_seed_start,
        "rows": rows,
        "artifact_root": str(budget_config.path_value("paths.incremental_artifact_dir")),
        "metadata": _runtime_metadata(config),
    }
    write_json(output_path, payload)
    console.print(payload)
    return output_path


def plot_phase12_sample_efficiency(config: Config) -> Path:
    import json
    import matplotlib.pyplot as plt

    budgets = [50, 100, 200, 500, 1000, 1800]
    summaries = []
    root = config.path_value("paths.incremental_results_dir") / "phase12"
    for budget in budgets:
        path = root / f"n{budget}" / "sample_efficiency_summary.json"
        if not path.exists():
            raise FileNotFoundError(f"Missing Phase 12 budget result: {path}")
        with path.open("r", encoding="utf-8") as f:
            summaries.append(json.load(f))
    methods = [row["method"] for row in summaries[0]["rows"]]
    transitions = np.asarray([summary["causal_transitions"] for summary in summaries])
    curves = {}
    for method in methods:
        values = []
        errors = []
        for summary in summaries:
            row = next(item for item in summary["rows"] if item["method"] == method)
            values.append(float(row["success"]))
            errors.append(float(row["success_stderr"]))
        success = np.asarray(values)
        curves[method] = {
            "success": values,
            "success_stderr": errors,
            "n50_transitions": (
                int(transitions[np.flatnonzero(success >= 0.5)[0]])
                if np.any(success >= 0.5)
                else None
            ),
            "n70_transitions": (
                int(transitions[np.flatnonzero(success >= 0.7)[0]])
                if np.any(success >= 0.7)
                else None
            ),
            "aulc_log_transitions": float(np.trapezoid(success, np.log(transitions))),
        }
    output_path = root / "sample_efficiency.json"
    plot_path = root / "sample_efficiency.png"
    payload = {
        "phase": 12,
        "trajectory_budgets": budgets,
        "causal_transitions": transitions.tolist(),
        "curves": curves,
        "plot": str(plot_path),
        "protocol": {
            "training_seeds": 1,
            "deployable_evaluation_episodes": 100,
            "oracle_evaluation_episodes": int(
                config.get("incremental.phase12.oracle_eval_episodes", 10)
            ),
            "validation_trajectories": 200,
            "training_seed_robustness_measured": False,
        },
        "metadata": _runtime_metadata(config),
    }
    write_json(output_path, payload)
    figure, axis = plt.subplots(figsize=(9, 6))
    for method in methods:
        label = method.replace("_", " ")
        if method == "oracle_hierarchy":
            label += " (10 eval episodes)"
        axis.errorbar(
            transitions,
            curves[method]["success"],
            yerr=curves[method]["success_stderr"],
            marker="o",
            capsize=3,
            label=label,
        )
    axis.set_xscale("log")
    axis.set_ylim(0.0, 1.0)
    axis.set_xlabel("Causal training transitions")
    axis.set_ylabel("Success rate")
    axis.set_title("Push-T sample efficiency")
    axis.grid(True, which="both", alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(plot_path, dpi=180)
    plt.close(figure)
    console.print(payload)
    return output_path


def _pre_rl_phase_a_config(config: Config) -> Config:
    raw = copy.deepcopy(config.raw)
    raw["paths"]["incremental_artifact_dir"] = str(
        config.path_value("paths.incremental_artifact_dir") / "phase12" / "n1800"
    )
    raw["paths"]["incremental_results_dir"] = str(
        config.path_value("paths.incremental_results_dir") / "pre_rl" / "phase_a"
    )
    raw["incremental"]["phase4"]["train_episodes"] = 1800
    raw["incremental"]["phase6"]["train_episodes"] = 1800
    eval_seed = int(config.get("pre_rl.phase_a.eval_seed_start", 1_500_000))
    raw["incremental"]["phase4"]["eval_seed"] = eval_seed
    raw["incremental"]["phase5"]["eval_seed"] = eval_seed
    raw["incremental"]["phase6"]["eval_seed"] = eval_seed
    raw["incremental"]["phase7"]["eval_seed"] = eval_seed
    raw["incremental"]["phase7"]["replay_branch_seed"] = eval_seed
    raw["incremental"]["phase8"]["eval_seed"] = eval_seed
    raw["incremental"]["phase9"]["eval_seed"] = eval_seed
    return Config(raw=raw, path=config.path)


def _wilson_interval(success: float, episodes: int, z: float = 1.959963984540054) -> list[float]:
    if episodes < 1:
        raise ValueError("Wilson interval requires at least one episode")
    denominator = 1.0 + z * z / episodes
    center = (success + z * z / (2.0 * episodes)) / denominator
    radius = (
        z
        * np.sqrt(success * (1.0 - success) / episodes + z * z / (4.0 * episodes**2))
        / denominator
    )
    return [float(max(0.0, center - radius)), float(min(1.0, center + radius))]


def run_pre_rl_phase_a_seed(config: Config, seed: int) -> Path:
    configured_seeds = [int(value) for value in config.get("pre_rl.phase_a.training_seeds")]
    if seed not in configured_seeds:
        raise ValueError(f"Phase A seed must be one of {configured_seeds}, got {seed}")
    phase_config = _pre_rl_phase_a_config(config)
    eval_episodes = int(config.get("pre_rl.phase_a.eval_episodes", 200))
    oracle_episodes = int(config.get("pre_rl.phase_a.oracle_eval_episodes", 50))
    eval_seed = int(config.get("pre_rl.phase_a.eval_seed_start", 1_500_000))
    result_root = phase_config.path_value("paths.incremental_results_dir")
    output_path = ensure_dir(result_root / "summaries") / f"seed{seed}.json"
    if output_path.exists():
        console.print(f"Pre-RL Phase A seed summary exists: {output_path}")
        return output_path

    visual_bc_path = (
        result_root
        / "phase4"
        / "concat_h1"
        / f"seed{seed}"
        / "visual_bc.json"
    )
    if not visual_bc_path.exists():
        visual_bc_path = evaluate_phase4_visual_bc(
            phase_config,
            history=1,
            architecture="concat",
            seed=seed,
            episodes=eval_episodes,
            eval_seed_start=eval_seed,
        )
    visual_flow_path = (
        result_root
        / "phase5"
        / "concat_h1"
        / f"seed{seed}"
        / "visual_flow.json"
    )
    if not visual_flow_path.exists():
        visual_flow_path = evaluate_phase5_visual_flow(
            phase_config,
            history=1,
            architecture="concat",
            seed=seed,
            episodes=eval_episodes,
            eval_seed_start=eval_seed,
        )
    matched_flat_path = evaluate_phase7_matched_flat_latent_policy(
        phase_config,
        latent_dim=256,
        variant="ae_recon",
        seed=seed,
        episodes=eval_episodes,
    )
    oracle_path = evaluate_phase7_replay_branch_oracle_low_level(
        phase_config,
        latent_dim=256,
        variant="ae_recon",
        horizon_steps=2,
        action_chunk_steps=1,
        goal_encoding="delta",
        goal_dropout_prob=0.0,
        seed=seed,
        episodes=oracle_episodes,
        force=False,
    )
    deterministic_path = evaluate_phase8_deterministic_hierarchy(
        phase_config,
        history=1,
        latent_dim=256,
        variant="ae_recon",
        horizon_steps=2,
        target_mode="absolute",
        seed=seed,
        episodes=eval_episodes,
    )
    generative_path = evaluate_phase9_future_flow(
        phase_config,
        sample_mode="zero",
        latent_dim=256,
        variant="ae_recon",
        horizon_steps=2,
        seed=seed,
        episodes=eval_episodes,
    )

    import json

    method_paths = {
        "visual_bc": visual_bc_path,
        "visual_flat_flow": visual_flow_path,
        "matched_flat_latent": matched_flat_path,
        "oracle_hierarchy": oracle_path,
        "deterministic_hierarchy": deterministic_path,
        "generative_hierarchy": generative_path,
    }
    rows = []
    for method, path in method_paths.items():
        with path.open("r", encoding="utf-8") as f:
            result = json.load(f)
        closed = result["closed_loop"]
        validation_action_mae = None
        rollout_action_mae = closed.get("teacher_action_mae")
        if "held_out_action_metrics" in result:
            validation_action_mae = result["held_out_action_metrics"].get("mae")
        elif method == "oracle_hierarchy":
            validation_action_mae = result["validation_action_metrics"]["correct_goal"]["mae"]
        elif method == "deterministic_hierarchy":
            validation_action_mae = result["low_level_metrics"]["predicted_goal_action_mae"]
        elif method == "generative_hierarchy":
            validation_action_mae = result["offline_metrics"]["zero_noise_goal_action_mae"]
        episodes = int(closed["episodes"])
        success = float(closed["success"])
        rows.append(
            {
                "method": method,
                "source": str(path),
                "success": success,
                "success_wilson_95": _wilson_interval(success, episodes),
                "episodes": episodes,
                "eval_seed_start": int(closed["seed_start"]),
                "final_reward": float(closed["final_reward"]),
                "max_reward": float(closed["max_reward"]),
                "validation_action_mae": (
                    float(validation_action_mae) if validation_action_mae is not None else None
                ),
                "rollout_teacher_action_mae": (
                    float(rollout_action_mae) if rollout_action_mae is not None else None
                ),
            }
        )
    train_episodes, _validation_episodes, _metadata = _load_phase4_episodes(phase_config)
    transitions = int(sum(len(episode["actions"]) for episode in train_episodes))
    payload = {
        "phase": "A",
        "experiment": "full_budget_statistical_replication",
        "command": (
            "uv run hcl-poc incremental pre-rl-a-run "
            f"--config {config.path} --seed {seed}"
        ),
        "policy_seed": seed,
        "evaluation_seed_start": eval_seed,
        "deployable_evaluation_episodes": eval_episodes,
        "oracle_evaluation_episodes": oracle_episodes,
        "causal_trajectories": 1800,
        "causal_transitions": transitions,
        "validation_trajectories": 200,
        "state_query_samples": 0,
        "representation": "ae_recon_z256",
        "rows": rows,
        "artifact_root": str(phase_config.path_value("paths.incremental_artifact_dir")),
        "result_root": str(result_root),
        "metadata": _runtime_metadata(config),
    }
    write_json(output_path, payload)
    console.print(payload)
    return output_path


def aggregate_pre_rl_phase_a(config: Config) -> Path:
    import json
    import matplotlib.pyplot as plt

    phase_config = _pre_rl_phase_a_config(config)
    root = phase_config.path_value("paths.incremental_results_dir")
    seeds = [int(value) for value in config.get("pre_rl.phase_a.training_seeds")]
    summaries = []
    for seed in seeds:
        path = root / "summaries" / f"seed{seed}.json"
        if not path.exists():
            raise FileNotFoundError(f"Missing Phase A seed summary: {path}")
        with path.open("r", encoding="utf-8") as f:
            summaries.append(json.load(f))
    methods = [row["method"] for row in summaries[0]["rows"]]
    rows = []
    for method in methods:
        seed_rows = [
            next(row for row in summary["rows"] if row["method"] == method)
            for summary in summaries
        ]
        successes = np.asarray([row["success"] for row in seed_rows], dtype=np.float64)
        episode_counts = np.asarray([row["episodes"] for row in seed_rows], dtype=np.int64)
        pooled_success = float(
            np.sum(successes * episode_counts) / np.sum(episode_counts)
        )
        rows.append(
            {
                "method": method,
                "seed_success": successes.tolist(),
                "mean_success": float(np.mean(successes)),
                "training_seed_std": float(np.std(successes, ddof=1)),
                "pooled_success": pooled_success,
                "pooled_wilson_95": _wilson_interval(
                    pooled_success, int(np.sum(episode_counts))
                ),
                "mean_final_reward": float(np.mean([row["final_reward"] for row in seed_rows])),
                "mean_max_reward": float(np.mean([row["max_reward"] for row in seed_rows])),
                "seed_rows": seed_rows,
            }
        )
    means = {row["method"]: row["mean_success"] for row in rows}
    best_flat = max(
        means["visual_bc"], means["visual_flat_flow"], means["matched_flat_latent"]
    )
    best_learned_hierarchy = max(
        means["deterministic_hierarchy"], means["generative_hierarchy"]
    )
    ordering_passed = bool(means["oracle_hierarchy"] > best_flat > best_learned_hierarchy)
    plot_path = root / "phase_a_success_across_seeds.png"
    labels = [method.replace("_", " ") for method in methods]
    x = np.arange(len(methods))
    figure, axis = plt.subplots(figsize=(10, 6))
    axis.bar(
        x,
        [row["mean_success"] for row in rows],
        yerr=[row["training_seed_std"] for row in rows],
        capsize=4,
    )
    for method_index, row in enumerate(rows):
        axis.scatter(
            np.full(len(row["seed_success"]), method_index),
            row["seed_success"],
            color="black",
            zorder=3,
        )
    axis.set_xticks(x)
    axis.set_xticklabels(labels, rotation=30, ha="right")
    axis.set_ylim(0.0, 1.0)
    axis.set_ylabel("Success rate")
    axis.set_title("Pre-RL Phase A: success across training seeds")
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(plot_path, dpi=180)
    plt.close(figure)
    output_path = root / "phase_a_aggregate.json"
    payload = {
        "phase": "A",
        "training_seeds": seeds,
        "rows": rows,
        "best_flat_success": best_flat,
        "best_learned_hierarchy_success": best_learned_hierarchy,
        "oracle_success": means["oracle_hierarchy"],
        "gate_ordering": "oracle > flat > learned_hierarchy",
        "gate_passed": ordering_passed,
        "plot": str(plot_path),
        "metadata": _runtime_metadata(config),
    }
    write_json(output_path, payload)
    console.print(payload)
    return output_path


def sweep_phase8_deterministic_predictors(
    config: Config,
    latent_dim: int | None = None,
    variant: str | None = None,
    horizon_steps: int | None = None,
    histories: list[int] | None = None,
    seed: int = 0,
    force: bool = False,
) -> Path:
    latent_dim, variant, horizon_steps = _phase8_defaults(
        config, latent_dim, variant, horizon_steps
    )
    histories = histories or [
        int(value) for value in config.get("incremental.phase8.histories", [1, 2, 4, 8])
    ]
    rows = []
    for history in histories:
        checkpoint_path = train_phase8_deterministic_predictor(
            config,
            history=history,
            latent_dim=latent_dim,
            variant=variant,
            horizon_steps=horizon_steps,
            seed=seed,
            force=force,
        )
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        rows.append(
            {
                "history": history,
                "checkpoint": str(checkpoint_path),
                "offline_metrics": checkpoint["offline_metrics"],
                "nearest_neighbor_metrics": checkpoint["nearest_neighbor_metrics"],
                "low_level_metrics": checkpoint["low_level_metrics"],
            }
        )
    best = min(rows, key=lambda row: row["low_level_metrics"]["predicted_goal_action_mae"])
    output_path = (
        ensure_dir(
            config.path_value("paths.incremental_results_dir")
            / "phase8"
            / f"{variant}_z{latent_dim}_k{horizon_steps}"
            / f"seed{seed}"
        )
        / "deterministic_history_sweep.json"
    )
    payload = {
        "phase": "8.2-8.6",
        "method": "deterministic_future_latent_history_sweep",
        "variant": variant,
        "latent_dim": latent_dim,
        "horizon_steps": horizon_steps,
        "seed": seed,
        "histories": rows,
        "selected_history": best["history"],
        "selection_metric": "predicted_goal_action_mae",
        "metadata": _runtime_metadata(config),
    }
    write_json(output_path, payload)
    console.print(payload)
    return output_path


@torch.inference_mode()
def collect_phase8_dagger_queries(
    config: Config,
    history: int = 1,
    latent_dim: int | None = None,
    variant: str | None = None,
    horizon_steps: int | None = None,
    iteration: int = 1,
    seed: int = 0,
    episodes: int = 10,
    force: bool = False,
) -> Path:
    latent_dim, variant, horizon_steps = _phase8_defaults(
        config, latent_dim, variant, horizon_steps
    )
    base_predictor_path = train_phase8_deterministic_predictor(
        config,
        history=history,
        latent_dim=latent_dim,
        variant=variant,
        horizon_steps=horizon_steps,
        seed=seed,
        force=False,
    )
    artifact_dir = base_predictor_path.parent
    output_path = artifact_dir / f"high_dagger_iter{iteration}_e{episodes}.npz"
    if output_path.exists() and not force:
        console.print(f"Phase 8 DAgger queries exist: {output_path}")
        return output_path
    rollout_predictor_path = base_predictor_path
    if iteration > 1:
        previous = artifact_dir / f"deterministic_predictor_high_dagger_iter{iteration - 1}.pt"
        if not previous.exists():
            raise FileNotFoundError(f"Missing previous Phase 8 DAgger predictor: {previous}")
        rollout_predictor_path = previous
    device = default_device()
    predictor_checkpoint = torch.load(
        rollout_predictor_path, map_location=device, weights_only=False
    )
    predictor = MLP(
        int(predictor_checkpoint["condition_dim"]),
        latent_dim,
        int(predictor_checkpoint["hidden_dim"]),
        depth=4,
    ).to(device)
    predictor.load_state_dict(predictor_checkpoint["model"])
    predictor.eval()
    encoder, encoder_checkpoint = _load_phase6_encoder(
        Path(predictor_checkpoint["encoder_checkpoint"]), device
    )
    frame_norm = Standardizer.from_state_dict(encoder_checkpoint["frame_norm"])
    latent_norm = Standardizer.from_state_dict(predictor_checkpoint["latent_norm"])
    action_norm = Standardizer.from_state_dict(predictor_checkpoint["action_norm"])
    low_model, low_checkpoint = _load_phase7_low_level_checkpoint(
        Path(predictor_checkpoint["low_level_checkpoint"]), device
    )
    dino = _phase4_dino_from_config(config, device)
    teacher = load_ppo_agent(_rl_paths(config).best, device)

    def make_env(num_envs: int):
        return gym.make(
            config.get("env_id"),
            obs_mode="rgb+state",
            control_mode=config.get("control_mode"),
            reward_mode="normalized_dense",
            render_mode=None,
            sim_backend=_rl_backend(config),
            num_envs=num_envs,
            reconfiguration_freq=config.get("rl.eval_reconfiguration_freq", 1),
        )

    zero_action_norm = action_norm.transform(
        np.zeros((1, int(low_checkpoint["action_dim"])), dtype=np.float32)
    )[0]
    seed_start = int(config.get("incremental.phase8.dagger_seed", 1_300_000))
    seed_start += 10_000 * (iteration - 1)
    max_num_envs = min(int(config.get("incremental.phase8.dagger_num_envs", 16)), episodes)
    replay_tolerance = float(config.get("incremental.phase8.dagger_replay_tolerance", 1e-6))
    max_episode_steps = int(config.get("env_max_episode_steps", 100))
    condition_rows = []
    current_latent_rows = []
    branch_goal_rows = []
    previous_action_rows = []
    teacher_action_rows = []
    predicted_goal_rows = []
    successes: list[float] = []
    replay_errors: list[float] = []
    failed_replay_steps = 0
    progress = trange(episodes, desc="phase8 collect learned-hierarchy DAgger")
    for batch_start in range(0, episodes, max_num_envs):
        num_envs = min(max_num_envs, episodes - batch_start)
        reset_seeds = [seed_start + batch_start + i for i in range(num_envs)]
        student_env = make_env(num_envs)
        branch_env = make_env(num_envs)
        action_low_np = np.asarray(student_env.action_space.low, dtype=np.float32)
        action_high_np = np.asarray(student_env.action_space.high, dtype=np.float32)
        if action_low_np.ndim == 2:
            action_low_np = action_low_np[0]
            action_high_np = action_high_np[0]
        action_low = torch.as_tensor(action_low_np, device=device, dtype=torch.float32)
        action_high = torch.as_tensor(action_high_np, device=device, dtype=torch.float32)
        try:
            obs, _info = student_env.reset(seed=reset_seeds)
            history_actions: list[torch.Tensor] = []
            previous_action_norm = np.repeat(zero_action_norm[None, :], num_envs, axis=0).astype(
                np.float32
            )
            step_dim = latent_dim + len(zero_action_norm)
            history_buffer = np.zeros((num_envs, history, step_dim), dtype=np.float32)
            history_initialized = np.zeros(num_envs, dtype=bool)
            active = np.ones(num_envs, dtype=bool)
            success_once = np.zeros(num_envs, dtype=bool)
            for _step in range(max_episode_steps):
                if not active.any():
                    break
                branch_obs, _branch_info = branch_env.reset(seed=reset_seeds)
                replay_done = torch.zeros(num_envs, device=device, dtype=torch.bool)
                for historical_action in history_actions:
                    (
                        branch_obs,
                        _branch_reward,
                        branch_term,
                        branch_trunc,
                        _branch_info,
                    ) = branch_env.step(historical_action)
                    replay_done |= torch.logical_or(branch_term, branch_trunc).view(-1)
                state_error = torch.max(
                    torch.abs(student_env.unwrapped.get_state() - branch_env.unwrapped.get_state()),
                    dim=1,
                ).values
                state_error_np = state_error.cpu().numpy()
                replay_errors.extend(state_error_np[active].tolist())
                replay_done_np = replay_done.cpu().numpy().astype(bool)
                failed_replay_steps += int(
                    np.sum(active & (replay_done_np | (state_error_np > replay_tolerance)))
                )
                for _ in range(horizon_steps):
                    teacher_branch_action = torch.clamp(
                        teacher.actor_mean(branch_obs["state"].to(device).float()),
                        action_low,
                        action_high,
                    )
                    (
                        branch_obs,
                        _branch_reward,
                        _branch_term,
                        _branch_trunc,
                        _branch_info,
                    ) = branch_env.step(teacher_branch_action)

                current_frame = _phase4_frame_inputs(
                    obs, dino, int(config.get("dino.batch_size", 64))
                )
                branch_frame = _phase4_frame_inputs(
                    branch_obs, dino, int(config.get("dino.batch_size", 64))
                )
                frames = frame_norm.transform(np.concatenate([current_frame, branch_frame], axis=0))
                z_pair = encoder(torch.from_numpy(frames).to(device).float()).cpu().numpy()
                current_z = z_pair[:num_envs].astype(np.float32)
                branch_goal = z_pair[num_envs:].astype(np.float32)
                step_rows = np.concatenate(
                    [latent_norm.transform(current_z), previous_action_norm], axis=-1
                )
                uninitialized = ~history_initialized
                if np.any(uninitialized):
                    history_buffer[uninitialized] = np.repeat(
                        step_rows[uninitialized, None, :], history, axis=1
                    )
                    history_initialized[uninitialized] = True
                initialized = ~uninitialized
                if np.any(initialized):
                    history_buffer[initialized, :-1] = history_buffer[initialized, 1:]
                    history_buffer[initialized, -1] = step_rows[initialized]
                high_condition = history_buffer.reshape(num_envs, -1).copy()
                predicted_goal = latent_norm.inverse(
                    predictor(torch.from_numpy(high_condition).to(device).float()).cpu().numpy()
                )
                low_condition = np.stack(
                    [
                        _phase7_condition(
                            current_z[i], predicted_goal[i], previous_action_norm[i], "delta"
                        )
                        for i in range(num_envs)
                    ]
                )
                predicted_action_norm = low_model(
                    torch.from_numpy(low_condition).to(device).float()
                )
                raw_action = action_norm.inverse(predicted_action_norm.cpu().numpy()).astype(
                    np.float32
                )
                teacher_action = (
                    torch.clamp(
                        teacher.actor_mean(obs["state"].to(device).float()),
                        action_low,
                        action_high,
                    )
                    .cpu()
                    .numpy()
                    .astype(np.float32)
                )
                condition_rows.append(high_condition[active])
                current_latent_rows.append(current_z[active])
                branch_goal_rows.append(branch_goal[active])
                previous_action_rows.append(previous_action_norm[active])
                teacher_action_rows.append(teacher_action[active])
                predicted_goal_rows.append(predicted_goal[active])

                action = torch.from_numpy(raw_action).to(device).float()
                if bool(config.get("policy.clip_actions_to_env_space", True)):
                    action = torch.clamp(action, action_low, action_high)
                action[~torch.from_numpy(active).to(device)] = 0.0
                obs, _reward, terminated, truncated, info = student_env.step(action)
                history_actions.append(action.detach().clone())
                previous_action_norm = action_norm.transform(
                    action.cpu().numpy().astype(np.float32)
                )
                if "success" in info:
                    success_once |= _numpy(info["success"]).reshape(-1).astype(bool)
                done = _numpy(torch.logical_or(terminated, truncated)).reshape(-1).astype(bool)
                newly_done = active & done
                if np.any(newly_done):
                    progress.update(int(newly_done.sum()))
                    active[newly_done] = False
            if np.any(active):
                progress.update(int(active.sum()))
            successes.extend(success_once.astype(np.float32).tolist())
        finally:
            student_env.close()
            branch_env.close()
    progress.close()
    if not condition_rows:
        raise RuntimeError("Phase 8 DAgger collection produced no queries")
    np.savez_compressed(
        output_path,
        conditions=np.concatenate(condition_rows).astype(np.float32),
        current_latents=np.concatenate(current_latent_rows).astype(np.float32),
        branch_goals=np.concatenate(branch_goal_rows).astype(np.float32),
        previous_actions_norm=np.concatenate(previous_action_rows).astype(np.float32),
        teacher_actions=np.concatenate(teacher_action_rows).astype(np.float32),
        predicted_goals=np.concatenate(predicted_goal_rows).astype(np.float32),
        collection_success=np.asarray(successes, dtype=np.float32),
        replay_current_state_error_mean=np.asarray(np.mean(replay_errors), dtype=np.float32),
        replay_current_state_error_max=np.asarray(np.max(replay_errors), dtype=np.float32),
        replay_failed_step_fraction=np.asarray(
            failed_replay_steps / max(1, len(replay_errors)), dtype=np.float32
        ),
        dataset_type=np.asarray("state_query_dataset"),
        semantics=np.asarray(
            "learned-hierarchy visited states with exact-replay teacher branch future goals"
        ),
        rollout_predictor=np.asarray(str(rollout_predictor_path)),
        history=np.asarray(history, dtype=np.int32),
        horizon_steps=np.asarray(horizon_steps, dtype=np.int32),
    )
    console.print(f"Wrote Phase 8 DAgger queries: {output_path}")
    return output_path


def train_phase8_dagger_predictor(
    config: Config,
    latent_dim: int | None = None,
    variant: str | None = None,
    horizon_steps: int | None = None,
    query_episodes: int = 10,
    seed: int = 0,
    force: bool = False,
) -> Path:
    set_seed(seed)
    latent_dim, variant, horizon_steps = _phase8_defaults(
        config, latent_dim, variant, horizon_steps
    )
    history = 1
    base_path = train_phase8_deterministic_predictor(
        config,
        history=history,
        latent_dim=latent_dim,
        variant=variant,
        horizon_steps=horizon_steps,
        seed=seed,
        force=False,
    )
    artifact_dir = base_path.parent
    output_path = artifact_dir / f"deterministic_predictor_high_dagger_iter1_e{query_episodes}.pt"
    if output_path.exists() and not force:
        console.print(f"Phase 8 DAgger predictor exists: {output_path}")
        return output_path
    query_path = collect_phase8_dagger_queries(
        config,
        history=history,
        latent_dim=latent_dim,
        variant=variant,
        horizon_steps=horizon_steps,
        iteration=1,
        seed=seed,
        episodes=query_episodes,
        force=False,
    )
    base = torch.load(base_path, map_location="cpu", weights_only=False)
    cache = torch.load(
        _phase8_latent_cache_path(config, variant, latent_dim, seed),
        map_location="cpu",
        weights_only=False,
    )
    latent_norm = Standardizer.from_state_dict(base["latent_norm"])
    action_norm = Standardizer.from_state_dict(base["action_norm"])
    with np.load(query_path) as query_data:
        query_condition = np.asarray(query_data["conditions"], dtype=np.float32)
        query_current = np.asarray(query_data["current_latents"], dtype=np.float32)
        query_goal = np.asarray(query_data["branch_goals"], dtype=np.float32)
        query_previous_action = np.asarray(query_data["previous_actions_norm"], dtype=np.float32)
        query_teacher_actions = np.asarray(query_data["teacher_actions"], dtype=np.float32)
    query_target = latent_norm.transform(query_goal)
    rng = np.random.default_rng(seed + 8100)
    permutation = rng.permutation(len(query_condition))
    split = max(1, int(0.8 * len(permutation)))
    train_indices = permutation[:split]
    val_indices = permutation[split:]
    if len(val_indices) == 0:
        raise ValueError("Phase 8 DAgger requires at least two coherent branch queries")

    train_episodes = cache["train"]
    val_episodes = cache["validation"]
    nominal_validation = _phase8_validation_samples(
        val_episodes,
        history,
        horizon_steps,
        latent_norm,
        action_norm,
        int(config.get("incremental.phase8.validation_samples", 10000)),
        seed + 8101,
    )
    batch_size = int(config.get("incremental.phase8.dagger_batch_size", 512))
    batches_per_epoch = int(config.get("incremental.phase8.dagger_batches_per_epoch", 100))
    nominal_dataset = _Phase8FutureDataset(
        train_episodes,
        history,
        horizon_steps,
        latent_norm,
        latent_norm,
        "absolute",
        action_norm,
        length=(batch_size // 2) * batches_per_epoch,
    )
    nominal_loader = DataLoader(
        nominal_dataset,
        batch_size=batch_size // 2,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    device = default_device()
    model = MLP(
        int(base["condition_dim"]),
        latent_dim,
        int(base["hidden_dim"]),
        depth=4,
    ).to(device)
    model.load_state_dict(base["model"])
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(config.get("incremental.phase8.dagger_lr", 1e-4))
    )
    nominal_x_val = torch.from_numpy(nominal_validation["conditions"]).to(device).float()
    nominal_y_val = (
        torch.from_numpy(latent_norm.transform(nominal_validation["future_latents"]))
        .to(device)
        .float()
    )
    query_x_val = torch.from_numpy(query_condition[val_indices]).to(device).float()
    query_y_val = torch.from_numpy(query_target[val_indices]).to(device).float()
    epochs = int(config.get("incremental.phase8.dagger_epochs", 30))
    best_state = None
    best_selection = float("inf")
    history_rows = []
    timer = Timer()
    for epoch in trange(1, epochs + 1, desc="train phase8 branch DAgger"):
        model.train()
        loss_sum = 0.0
        count = 0
        for nominal_x, nominal_y in nominal_loader:
            chosen = rng.choice(train_indices, size=len(nominal_x), replace=True)
            query_x = torch.from_numpy(query_condition[chosen]).to(device).float()
            query_y = torch.from_numpy(query_target[chosen]).to(device).float()
            nominal_x = nominal_x.to(device, non_blocking=True).float()
            nominal_y = nominal_y.to(device, non_blocking=True).float()
            nominal_loss = torch.mean((model(nominal_x) - nominal_y) ** 2)
            query_loss = torch.mean((model(query_x) - query_y) ** 2)
            loss = 0.5 * nominal_loss + 0.5 * query_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach().cpu()) * len(nominal_x)
            count += len(nominal_x)
        model.eval()
        with torch.inference_mode():
            nominal_mse = float(torch.mean((model(nominal_x_val) - nominal_y_val) ** 2).cpu())
            query_mse = float(torch.mean((model(query_x_val) - query_y_val) ** 2).cpu())
        selection = nominal_mse + query_mse
        history_rows.append(
            {
                "epoch": epoch,
                "train_mse": loss_sum / count,
                "nominal_validation_mse": nominal_mse,
                "query_validation_mse": query_mse,
                "selection": selection,
            }
        )
        if selection < best_selection:
            best_selection = selection
            best_state = copy.deepcopy(model.state_dict())
    if best_state is None:
        raise RuntimeError("Phase 8 DAgger training produced no checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    with torch.inference_mode():
        predicted_query_goal = latent_norm.inverse(
            model(torch.from_numpy(query_condition).to(device).float()).cpu().numpy()
        )
    query_goal_l2 = np.linalg.norm(predicted_query_goal - query_goal, axis=-1)
    low_model, low_checkpoint = _load_phase7_low_level_checkpoint(
        Path(base["low_level_checkpoint"]), device
    )
    low_action_norm = Standardizer.from_state_dict(low_checkpoint["action_norm"])
    predicted_low_condition = np.stack(
        [
            _phase7_condition(current, goal, previous, "delta")
            for current, goal, previous in zip(
                query_current, predicted_query_goal, query_previous_action
            )
        ]
    )
    oracle_low_condition = np.stack(
        [
            _phase7_condition(current, goal, previous, "delta")
            for current, goal, previous in zip(query_current, query_goal, query_previous_action)
        ]
    )

    def predict_low(conditions: np.ndarray) -> np.ndarray:
        with torch.inference_mode():
            normalized = low_model(torch.from_numpy(conditions).to(device).float())
        return low_action_norm.inverse(normalized.cpu().numpy())

    predicted_low_action = predict_low(predicted_low_condition)
    oracle_low_action = predict_low(oracle_low_condition)
    low_level_metrics = {
        "oracle_goal_action_mae": float(np.mean(np.abs(oracle_low_action - query_teacher_actions))),
        "predicted_goal_action_mae": float(
            np.mean(np.abs(predicted_low_action - query_teacher_actions))
        ),
        "predicted_to_oracle_mae_ratio": float(
            np.mean(np.abs(predicted_low_action - query_teacher_actions))
            / max(np.mean(np.abs(oracle_low_action - query_teacher_actions)), 1e-8)
        ),
        "predicted_vs_oracle_action_l2": float(
            np.mean(np.linalg.norm(predicted_low_action - oracle_low_action, axis=-1))
        ),
        "distribution": "coherent_learner_branch_queries",
    }
    payload = {
        **base,
        "model": model.state_dict(),
        "training_mode": "balanced_nominal_and_coherent_branch_dagger",
        "query_path": str(query_path),
        "query_episodes": query_episodes,
        "query_samples": int(len(query_condition)),
        "query_validation_samples": int(len(val_indices)),
        "query_goal_l2": float(np.mean(query_goal_l2[val_indices])),
        "low_level_metrics": low_level_metrics,
        "dagger_training_history": history_rows,
        "dagger_best_selection": best_selection,
        "dagger_elapsed_s": timer.elapsed(),
        "metadata": _runtime_metadata(config),
    }
    torch.save(payload, output_path)
    write_json(
        artifact_dir / f"deterministic_predictor_high_dagger_iter1_e{query_episodes}_metrics.json",
        {
            "query_episodes": query_episodes,
            "query_samples": int(len(query_condition)),
            "query_validation_samples": int(len(val_indices)),
            "query_goal_l2": payload["query_goal_l2"],
            "low_level_metrics": low_level_metrics,
            "dagger_best_selection": best_selection,
            "dagger_elapsed_s": payload["dagger_elapsed_s"],
        },
    )
    console.print(f"Wrote Phase 8 DAgger predictor: {output_path}")
    return output_path


def _phase8_nominal_low_level_samples(
    episodes: list[dict[str, np.ndarray]],
    horizon_steps: int,
    action_norm: Standardizer,
) -> tuple[np.ndarray, np.ndarray]:
    zero_action_norm = action_norm.transform(
        np.zeros((1, episodes[0]["actions"].shape[-1]), dtype=np.float32)
    )[0]
    conditions = []
    actions = []
    for episode in episodes:
        for t in range(len(episode["actions"]) - horizon_steps):
            previous = (
                action_norm.transform(episode["actions"][t - 1 : t])[0]
                if t > 0
                else zero_action_norm
            )
            conditions.append(
                _phase7_condition(
                    episode["latents"][t],
                    episode["latents"][t + horizon_steps],
                    previous,
                    "delta",
                )
            )
            actions.append(episode["actions"][t])
    return np.stack(conditions).astype(np.float32), np.stack(actions).astype(np.float32)


def train_phase8_adapted_low_level(
    config: Config,
    latent_dim: int | None = None,
    variant: str | None = None,
    horizon_steps: int | None = None,
    query_episodes: int = 10,
    seed: int = 0,
    force: bool = False,
) -> Path:
    set_seed(seed)
    latent_dim, variant, horizon_steps = _phase8_defaults(
        config, latent_dim, variant, horizon_steps
    )
    query_path = collect_phase8_dagger_queries(
        config,
        history=1,
        latent_dim=latent_dim,
        variant=variant,
        horizon_steps=horizon_steps,
        seed=seed,
        episodes=query_episodes,
        force=False,
    )
    predictor_path = train_phase8_deterministic_predictor(
        config,
        history=1,
        latent_dim=latent_dim,
        variant=variant,
        horizon_steps=horizon_steps,
        seed=seed,
        force=False,
    )
    predictor_checkpoint = torch.load(predictor_path, map_location="cpu", weights_only=False)
    base_low_path = Path(predictor_checkpoint["low_level_checkpoint"])
    output_path = predictor_path.parent / f"adapted_low_high_dagger_e{query_episodes}.pt"
    if output_path.exists() and not force:
        console.print(f"Phase 8 adapted low level exists: {output_path}")
        return output_path
    device = default_device()
    base_model, base_checkpoint = _load_phase7_low_level_checkpoint(base_low_path, device)
    action_norm = Standardizer.from_state_dict(base_checkpoint["action_norm"])
    cache = torch.load(
        _phase8_latent_cache_path(config, variant, latent_dim, seed),
        map_location="cpu",
        weights_only=False,
    )
    nominal_x, nominal_action = _phase8_nominal_low_level_samples(
        cache["train"], horizon_steps, action_norm
    )
    nominal_val_x, nominal_val_action = _phase8_nominal_low_level_samples(
        cache["validation"], horizon_steps, action_norm
    )
    with np.load(query_path) as query_data:
        query_current = np.asarray(query_data["current_latents"], dtype=np.float32)
        query_goal = np.asarray(query_data["branch_goals"], dtype=np.float32)
        query_previous = np.asarray(query_data["previous_actions_norm"], dtype=np.float32)
        query_action = np.asarray(query_data["teacher_actions"], dtype=np.float32)
    query_x = np.stack(
        [
            _phase7_condition(current, goal, previous, "delta")
            for current, goal, previous in zip(query_current, query_goal, query_previous)
        ]
    )
    rng = np.random.default_rng(seed + 8200)
    permutation = rng.permutation(len(query_x))
    split = max(1, int(0.8 * len(permutation)))
    query_train = permutation[:split]
    query_val = permutation[split:]
    if len(query_val) == 0:
        raise ValueError("Phase 8 low adaptation requires at least two branch queries")
    repeat = max(1, int(np.ceil(len(nominal_x) / len(query_train))))
    train_x = np.concatenate([nominal_x, np.tile(query_x[query_train], (repeat, 1))])
    train_y = np.concatenate(
        [
            action_norm.transform(nominal_action),
            np.tile(action_norm.transform(query_action[query_train]), (repeat, 1)),
        ]
    )
    dataset = TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y))
    loader = DataLoader(
        dataset,
        batch_size=int(config.get("incremental.phase8.dagger_batch_size", 512)),
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    model = MLP(
        int(base_checkpoint["cond_dim"]),
        int(base_checkpoint["action_dim"]),
        int(base_checkpoint["hidden_dim"]),
        depth=4,
    ).to(device)
    model.load_state_dict(base_model.state_dict())
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(config.get("incremental.phase8.dagger_lr", 1e-4))
    )
    nominal_val_x_t = torch.from_numpy(nominal_val_x).to(device).float()
    nominal_val_y_t = torch.from_numpy(action_norm.transform(nominal_val_action)).to(device).float()
    query_val_x_t = torch.from_numpy(query_x[query_val]).to(device).float()
    query_val_y_t = (
        torch.from_numpy(action_norm.transform(query_action[query_val])).to(device).float()
    )
    best_state = None
    best_selection = float("inf")
    rows = []
    timer = Timer()
    for epoch in trange(
        1,
        int(config.get("incremental.phase8.dagger_epochs", 30)) + 1,
        desc="train phase8 adapted low level",
    ):
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
            nominal_mse = float(torch.mean((model(nominal_val_x_t) - nominal_val_y_t) ** 2).cpu())
            query_mse = float(torch.mean((model(query_val_x_t) - query_val_y_t) ** 2).cpu())
        selection = nominal_mse + query_mse
        rows.append(
            {
                "epoch": epoch,
                "train_mse": loss_sum / count,
                "nominal_validation_mse": nominal_mse,
                "query_validation_mse": query_mse,
                "selection": selection,
            }
        )
        if selection < best_selection:
            best_selection = selection
            best_state = copy.deepcopy(model.state_dict())
    if best_state is None:
        raise RuntimeError("Phase 8 low adaptation produced no checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    validation_metrics = _phase7_oracle_action_metrics(
        model,
        nominal_val_x,
        nominal_val_action,
        action_norm,
        latent_dim,
        "delta",
        int(config.get("incremental.phase7.validation_queries", 10000)),
        seed + 8201,
    )
    with torch.inference_mode():
        query_prediction = action_norm.inverse(model(query_val_x_t).cpu().numpy())
    query_metrics = _action_regression_metrics(query_prediction, query_action[query_val])
    payload = {
        **base_checkpoint,
        "model": model.state_dict(),
        "training_mode": "balanced_nominal_and_learned_hierarchy_branch_queries",
        "base_low_checkpoint": str(base_low_path),
        "query_path": str(query_path),
        "query_episodes": query_episodes,
        "query_samples": int(len(query_x)),
        "query_validation_samples": int(len(query_val)),
        "validation_metrics": validation_metrics,
        "query_validation_metrics": query_metrics,
        "adaptation_history": rows,
        "adaptation_best_selection": best_selection,
        "adaptation_elapsed_s": timer.elapsed(),
        "metadata": _runtime_metadata(config),
    }
    torch.save(payload, output_path)
    write_json(
        predictor_path.parent / f"adapted_low_high_dagger_e{query_episodes}_metrics.json",
        {
            "query_episodes": query_episodes,
            "query_samples": int(len(query_x)),
            "query_validation_samples": int(len(query_val)),
            "nominal_validation_metrics": validation_metrics,
            "query_validation_metrics": query_metrics,
            "adaptation_best_selection": best_selection,
            "adaptation_elapsed_s": payload["adaptation_elapsed_s"],
        },
    )
    console.print(f"Wrote Phase 8 adapted low level: {output_path}")
    return output_path


@torch.inference_mode()
def evaluate_phase8_deterministic_hierarchy(
    config: Config,
    history: int,
    latent_dim: int | None = None,
    variant: str | None = None,
    horizon_steps: int | None = None,
    target_mode: str = "absolute",
    seed: int = 0,
    episodes: int | None = None,
    high_dagger_query_episodes: int | None = None,
    adapted_low_query_episodes: int | None = None,
    branch_action_weight: float = 1.0,
    action_consistency_weight: float | None = None,
    force: bool = False,
) -> Path:
    latent_dim, variant, horizon_steps = _phase8_defaults(
        config, latent_dim, variant, horizon_steps
    )
    if not 0.0 <= branch_action_weight <= 1.0:
        raise ValueError("branch_action_weight must be in [0, 1]")
    if action_consistency_weight is not None:
        if high_dagger_query_episodes is not None or target_mode != "absolute" or history != 1:
            raise ValueError(
                "Action-consistent evaluation requires base absolute L=1 without high DAgger"
            )
        with torch.inference_mode(False):
            predictor_path = train_phase8_action_consistent_predictor(
                config,
                action_consistency_weight=action_consistency_weight,
                latent_dim=latent_dim,
                variant=variant,
                horizon_steps=horizon_steps,
                seed=seed,
                force=False,
            )
        predictor_label = f"actionw{action_consistency_weight:g}".replace(".", "p")
    elif high_dagger_query_episodes is None:
        with torch.inference_mode(False):
            predictor_path = train_phase8_deterministic_predictor(
                config,
                history=history,
                latent_dim=latent_dim,
                variant=variant,
                horizon_steps=horizon_steps,
                target_mode=target_mode,
                seed=seed,
                force=False,
            )
        predictor_label = f"base_{target_mode}"
    else:
        if target_mode != "absolute":
            raise ValueError("Phase 8 high DAgger currently supports absolute targets")
        if history != 1:
            raise ValueError("Current Phase 8 DAgger queries support history L=1")
        with torch.inference_mode(False):
            predictor_path = train_phase8_dagger_predictor(
                config,
                latent_dim=latent_dim,
                variant=variant,
                horizon_steps=horizon_steps,
                query_episodes=high_dagger_query_episodes,
                seed=seed,
                force=False,
            )
        predictor_label = f"high_dagger_e{high_dagger_query_episodes}"
    low_level_label = (
        "base_low"
        if adapted_low_query_episodes is None
        else f"adapted_low_e{adapted_low_query_episodes}"
    )
    eval_episodes = int(episodes or config.get("incremental.phase8.eval_episodes", 100))
    results_dir = ensure_dir(
        config.path_value("paths.incremental_results_dir")
        / "phase8"
        / f"{variant}_z{latent_dim}_k{horizon_steps}_l{history}"
        / f"seed{seed}"
    )
    blend_label = f"branchw{int(round(100 * branch_action_weight))}"
    output_path = results_dir / (
        f"deterministic_hierarchy_{predictor_label}_{low_level_label}_"
        f"{blend_label}_{eval_episodes}.json"
    )
    if output_path.exists() and not force:
        console.print(f"Phase 8 deterministic hierarchy eval exists: {output_path}")
        return output_path

    device = default_device()
    predictor_checkpoint = torch.load(predictor_path, map_location=device, weights_only=False)
    predictor = MLP(
        int(predictor_checkpoint["condition_dim"]),
        latent_dim,
        int(predictor_checkpoint["hidden_dim"]),
        depth=4,
    ).to(device)
    predictor.load_state_dict(predictor_checkpoint["model"])
    predictor.eval()
    encoder, encoder_checkpoint = _load_phase6_encoder(
        Path(predictor_checkpoint["encoder_checkpoint"]), device
    )
    frame_norm = Standardizer.from_state_dict(encoder_checkpoint["frame_norm"])
    latent_norm = Standardizer.from_state_dict(predictor_checkpoint["latent_norm"])
    target_norm = Standardizer.from_state_dict(
        predictor_checkpoint.get("target_norm", predictor_checkpoint["latent_norm"])
    )
    checkpoint_target_mode = str(predictor_checkpoint.get("target_mode", "absolute"))
    action_norm = Standardizer.from_state_dict(predictor_checkpoint["action_norm"])
    low_level_path = Path(predictor_checkpoint["low_level_checkpoint"])
    if adapted_low_query_episodes is not None:
        with torch.inference_mode(False):
            low_level_path = train_phase8_adapted_low_level(
                config,
                latent_dim=latent_dim,
                variant=variant,
                horizon_steps=horizon_steps,
                query_episodes=adapted_low_query_episodes,
                seed=seed,
                force=False,
            )
    low_model, low_checkpoint = _load_phase7_low_level_checkpoint(low_level_path, device)
    low_action_norm = Standardizer.from_state_dict(low_checkpoint["action_norm"])
    if not (
        np.array_equal(action_norm.mean, low_action_norm.mean)
        and np.array_equal(action_norm.std, low_action_norm.std)
    ):
        raise ValueError("Phase 8 predictor and low level use different action normalization")
    flat_model = None
    if branch_action_weight < 1.0:
        with torch.inference_mode(False):
            flat_path = train_phase6_latent_bc(
                config,
                latent_dim=latent_dim,
                variant=variant,
                seed=seed,
                force=False,
            )
        flat_checkpoint = torch.load(flat_path, map_location=device, weights_only=False)
        flat_model = MLP(
            int(flat_checkpoint["cond_dim"]),
            int(flat_checkpoint["action_dim"]),
            int(flat_checkpoint["hidden_dim"]),
            depth=4,
        ).to(device)
        flat_model.load_state_dict(flat_checkpoint["model"])
        flat_model.eval()
    dino = _phase4_dino_from_config(config, device)
    teacher = load_ppo_agent(_rl_paths(config).best, device)
    num_envs = min(int(config.get("incremental.phase8.eval_num_envs", 64)), eval_episodes)
    env = _phase4_make_visual_env(config, num_envs)
    action_low = torch.as_tensor(env.single_action_space.low, device=device, dtype=torch.float32)
    action_high = torch.as_tensor(env.single_action_space.high, device=device, dtype=torch.float32)
    zero_action_norm = action_norm.transform(
        np.zeros((1, int(low_checkpoint["action_dim"])), dtype=np.float32)
    )[0]
    seed_start = int(config.get("incremental.phase8.eval_seed", 1_200_000))
    obs, _info = env.reset(seed=seed_start)
    previous_action_norm = np.repeat(zero_action_norm[None, :], num_envs, axis=0).astype(np.float32)
    step_dim = latent_dim + len(zero_action_norm)
    history_buffer = np.zeros((num_envs, history, step_dim), dtype=np.float32)
    history_initialized = np.zeros(num_envs, dtype=bool)
    successes: list[float] = []
    final_rewards: list[float] = []
    max_rewards: list[float] = []
    episode_lengths: list[int] = []
    teacher_action_maes: list[float] = []
    predicted_goal_displacements: list[float] = []
    latencies: list[float] = []
    active_max_reward = np.full(num_envs, -np.inf, dtype=np.float32)
    active_lengths = np.zeros(num_envs, dtype=np.int32)
    while len(successes) < eval_episodes:
        timer = Timer()
        frames = frame_norm.transform(
            _phase4_frame_inputs(obs, dino, int(config.get("dino.batch_size", 64)))
        )
        z = encoder(torch.from_numpy(frames).to(device).float()).cpu().numpy().astype(np.float32)
        z_normalized = latent_norm.transform(z)
        step_rows = np.concatenate([z_normalized, previous_action_norm], axis=-1)
        uninitialized = ~history_initialized
        if np.any(uninitialized):
            history_buffer[uninitialized] = np.repeat(
                step_rows[uninitialized, None, :], history, axis=1
            )
            history_initialized[uninitialized] = True
        initialized = ~uninitialized
        if np.any(initialized):
            history_buffer[initialized, :-1] = history_buffer[initialized, 1:]
            history_buffer[initialized, -1] = step_rows[initialized]
        predicted_target_normalized = (
            predictor(torch.from_numpy(history_buffer.reshape(num_envs, -1)).to(device).float())
            .cpu()
            .numpy()
        )
        predicted_target = target_norm.inverse(predicted_target_normalized)
        predicted_goal = (
            predicted_target if checkpoint_target_mode == "absolute" else z + predicted_target
        )
        predicted_goal_displacements.extend(np.linalg.norm(predicted_goal - z, axis=-1).tolist())
        low_condition = np.stack(
            [
                _phase7_condition(z[i], predicted_goal[i], previous_action_norm[i], "delta")
                for i in range(num_envs)
            ]
        )
        predicted_action_normalized = low_model(torch.from_numpy(low_condition).to(device).float())
        raw_action = action_norm.inverse(predicted_action_normalized.cpu().numpy()).astype(
            np.float32
        )
        if flat_model is not None:
            flat_condition = np.concatenate([z, previous_action_norm], axis=-1)
            flat_action_normalized = flat_model(torch.from_numpy(flat_condition).to(device).float())
            flat_action = action_norm.inverse(flat_action_normalized.cpu().numpy()).astype(
                np.float32
            )
            raw_action = (
                branch_action_weight * raw_action + (1.0 - branch_action_weight) * flat_action
            )
        teacher_action = (
            torch.clamp(
                teacher.actor_mean(obs["state"].to(device).float()), action_low, action_high
            )
            .cpu()
            .numpy()
            .astype(np.float32)
        )
        teacher_action_maes.extend(np.mean(np.abs(raw_action - teacher_action), axis=-1).tolist())
        latencies.append(timer.elapsed() / num_envs)
        action = torch.from_numpy(raw_action).to(device).float()
        if bool(config.get("policy.clip_actions_to_env_space", True)):
            action = torch.clamp(action, action_low, action_high)
        previous_action_norm = action_norm.transform(action.cpu().numpy().astype(np.float32))
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
                    previous_action_norm[env_idx] = zero_action_norm
                    history_initialized[env_idx] = False
                    if len(successes) >= eval_episodes:
                        break
    env.close()
    metrics = {
        "success": float(np.mean(successes[:eval_episodes])),
        "success_stderr": float(np.std(successes[:eval_episodes]) / np.sqrt(eval_episodes)),
        "final_reward": float(np.mean(final_rewards[:eval_episodes])),
        "max_reward": float(np.mean(max_rewards[:eval_episodes])),
        "mean_episode_length": float(np.mean(episode_lengths[:eval_episodes])),
        "teacher_action_mae": float(np.mean(teacher_action_maes)),
        "predicted_goal_displacement_l2": float(np.mean(predicted_goal_displacements)),
        "inference_latency_s": float(np.mean(latencies)),
        "episodes": eval_episodes,
        "num_envs": num_envs,
        "seed_start": seed_start,
    }
    oracle_result_dir = (
        config.path_value("paths.incremental_results_dir")
        / "phase7"
        / _phase7_tag(variant, latent_dim, horizon_steps, 1, "delta", 0.0)
        / f"seed{seed}"
    )
    import json

    oracle_results = []
    for candidate in oracle_result_dir.glob("replay_branch_oracle_eval*.json"):
        with candidate.open("r", encoding="utf-8") as f:
            result = json.load(f)
        if Path(result.get("checkpoint", "")) == low_level_path:
            oracle_results.append((int(result["closed_loop"]["episodes"]), candidate, result))
    if not oracle_results:
        raise FileNotFoundError(
            f"No Phase 7 branch-oracle result matches low-level checkpoint {low_level_path}"
        )
    _oracle_episodes, oracle_result_path, oracle_result = max(
        oracle_results, key=lambda item: item[0]
    )
    oracle_success = float(oracle_result["closed_loop"]["success"])
    payload = {
        "phase": 8,
        "method": "deterministic_future_latent_hierarchy",
        "variant": variant,
        "latent_dim": latent_dim,
        "horizon_steps": horizon_steps,
        "history": history,
        "target_mode": checkpoint_target_mode,
        "high_dagger_query_episodes": high_dagger_query_episodes,
        "adapted_low_query_episodes": adapted_low_query_episodes,
        "branch_action_weight": branch_action_weight,
        "action_consistency_weight": action_consistency_weight,
        "seed": seed,
        "checkpoint": str(predictor_path),
        "closed_loop": metrics,
        "offline_metrics": predictor_checkpoint["offline_metrics"],
        "nearest_neighbor_metrics": predictor_checkpoint["nearest_neighbor_metrics"],
        "low_level_metrics": predictor_checkpoint["low_level_metrics"],
        "oracle_result": str(oracle_result_path),
        "oracle_success": oracle_success,
        "success_fraction_of_oracle": metrics["success"] / max(oracle_success, 1e-8),
        "gate_70pct_oracle": metrics["success"] >= 0.70 * oracle_success,
        "metadata": _runtime_metadata(config),
    }
    write_json(output_path, payload)
    console.print(payload)
    return output_path
