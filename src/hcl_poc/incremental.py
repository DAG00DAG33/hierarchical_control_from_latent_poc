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
from hcl_poc.rl import PPOAgent, _make_state_env, _rl_paths, evaluate_ppo, load_ppo_agent
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
