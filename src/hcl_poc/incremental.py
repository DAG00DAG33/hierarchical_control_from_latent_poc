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
from tqdm import trange

from hcl_poc.config import Config
from hcl_poc.rl import PPOAgent, _rl_paths, evaluate_ppo, load_ppo_agent
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
