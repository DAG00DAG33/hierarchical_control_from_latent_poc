from __future__ import annotations

import csv
import subprocess
import copy
import time
from pathlib import Path
from typing import Any

import gymnasium as gym
import h5py
import mani_skill  # noqa: F401
import numpy as np
import torch
from tqdm import trange

from hcl_poc.config import Config
from hcl_poc.features import batched, dino_from_config
from hcl_poc.rl import _rl_backend, _rl_paths, load_ppo_agent
from hcl_poc.utils import default_device, ensure_dir, write_json


def _to_numpy(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _scalar(value: Any) -> float:
    return float(_to_numpy(value).reshape(-1)[0])


def _bool(value: Any) -> bool:
    return bool(_to_numpy(value).reshape(-1)[0])


def _rgb_and_state(obs: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    rgb = _to_numpy(obs["sensor_data"]["base_camera"]["rgb"])
    state = _to_numpy(obs["state"])
    if rgb.ndim == 4:
        rgb = rgb[0]
    if state.ndim == 2:
        state = state[0]
    if rgb.shape[-1] == 4:
        rgb = rgb[..., :3]
    return rgb.astype(np.uint8), state.astype(np.float32)


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


def _state_dataset_path(config: Config) -> Path:
    return config.path_value("paths.incremental_data_dir").parent / "rl_rerun" / "pusht_state_demos.h5"


def _state_audit_result_dir(config: Config) -> Path:
    return ensure_dir(config.path_value("paths.incremental_results_dir").parent / "rl_rerun")


def _rerun_base_config(config: Config, dataset_path: Path | None = None) -> Config:
    raw = copy.deepcopy(config.raw)
    raw["paths"]["incremental_artifact_dir"] = str(
        config.path_value("paths.incremental_artifact_dir").parent / "rl_rerun"
    )
    raw["paths"]["incremental_results_dir"] = str(
        config.path_value("paths.incremental_results_dir").parent / "rl_rerun"
    )
    raw["incremental"]["phase4"]["prepared_path"] = str(dataset_path or _state_dataset_path(config))
    return Config(raw=raw, path=config.path)


def _make_state_data_env(config: Config):
    return gym.make(
        config.get("env_id"),
        obs_mode="rgb+state",
        control_mode=config.get("control_mode"),
        reward_mode="normalized_dense",
        render_mode=None,
        sim_backend=_rl_backend(config),
        num_envs=1,
        reconfiguration_freq=config.get("rl.collect_reconfiguration_freq", 1),
    )


def _make_benchmark_env(config: Config, num_envs: int, obs_mode: str):
    from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv

    base = gym.make(
        config.get("env_id"),
        obs_mode=obs_mode,
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
        record_metrics=False,
    )


def _parse_int_list(text: str) -> list[int]:
    return [int(item.strip()) for item in text.split(",") if item.strip()]


def _gpu_utilization_percent() -> int | None:
    try:
        output = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except Exception:
        return None
    if not output:
        return None
    return int(output.splitlines()[0].strip())


def _cuda_memory_mib() -> tuple[float | None, float | None]:
    if not torch.cuda.is_available():
        return None, None
    torch.cuda.synchronize()
    return (
        float(torch.cuda.max_memory_allocated() / 2**20),
        float(torch.cuda.max_memory_reserved() / 2**20),
    )


def _safe_close(env: Any | None) -> None:
    if env is not None:
        env.close()


@torch.inference_mode()
def collect_rl_rerun_state_dataset(
    config: Config,
    episodes: int,
    output_path: Path | None = None,
    seed_start: int | None = None,
    max_attempts: int | None = None,
    checkpoint_path: Path | None = None,
    store_rgb: bool = False,
    force: bool = False,
) -> Path:
    out_path = output_path or _state_dataset_path(config)
    if out_path.exists() and not force:
        with h5py.File(out_path, "r") as h5:
            existing = len([key for key in h5.keys() if key.startswith("episode_")])
        if existing >= episodes:
            return out_path
    ensure_dir(out_path.parent)
    tmp_path = out_path.with_suffix(".tmp.h5")
    tmp_path.unlink(missing_ok=True)

    device = default_device()
    teacher_path = checkpoint_path or _rl_paths(config).best
    teacher = load_ppo_agent(teacher_path, device)
    extractor = dino_from_config(config, device)
    dino_batch_size = int(config.get("dino.batch_size", 64))
    env = _make_state_data_env(config)
    action_low = np.asarray(env.action_space.low, dtype=np.float32)
    action_high = np.asarray(env.action_space.high, dtype=np.float32)
    if action_low.ndim == 2:
        action_low = action_low[0]
        action_high = action_high[0]
    start_seed = int(seed_start or config.get("rl.collect_seed", 70_000))
    attempts_limit = int(max_attempts or episodes * 20)
    successes = 0
    attempts = 0
    try:
        with h5py.File(tmp_path, "w") as h5:
            meta = h5.create_group("meta")
            meta.attrs["dataset_type"] = "rl_rerun_state_loadable_pusht"
            meta.attrs["env_id"] = config.get("env_id")
            meta.attrs["obs_mode"] = "rgb+state"
            meta.attrs["control_mode"] = config.get("control_mode")
            meta.attrs["sim_backend"] = _rl_backend(config)
            meta.attrs["control_freq"] = int(config.get("control_freq"))
            meta.attrs["teacher_checkpoint"] = str(teacher_path)
            meta.attrs["dino_model"] = config.get("dino.model_name")
            meta.attrs["dino_feature_type"] = config.get("dino.feature_type", "cls")
            meta.attrs["dino_spatial_pool"] = int(config.get("dino.spatial_pool", 4))
            meta.attrs["store_rgb"] = store_rgb
            for key, value in _git_metadata().items():
                meta.attrs[key] = value
            for attempts in trange(1, attempts_limit + 1, desc="collect RL rerun states"):
                reset_seed = start_seed + attempts
                obs, _info = env.reset(seed=reset_seed)
                rgbs: list[np.ndarray] = []
                observations: list[np.ndarray] = []
                simulator_states: list[np.ndarray] = []
                raw_actions: list[np.ndarray] = []
                clipped_actions: list[np.ndarray] = []
                executed_actions: list[np.ndarray] = []
                previous_actions: list[np.ndarray] = []
                rewards: list[float] = []
                terminated_flags: list[bool] = []
                truncated_flags: list[bool] = []
                success_flags: list[bool] = []
                previous_action = np.zeros(3, dtype=np.float32)
                success = False
                terminated = False
                truncated = False
                while not (terminated or truncated):
                    rgb, state_obs = _rgb_and_state(obs)
                    rgbs.append(rgb)
                    observations.append(state_obs)
                    simulator_states.append(
                        _to_numpy(env.unwrapped.get_state()).reshape(-1).astype(np.float32)
                    )
                    state_tensor = torch.from_numpy(state_obs[None]).to(device).float()
                    raw_action = (
                        teacher.actor_mean(state_tensor).detach().cpu().numpy()[0].astype(np.float32)
                    )
                    clipped_action = np.clip(raw_action, action_low, action_high).astype(np.float32)
                    executed_action = (
                        clipped_action
                        if bool(config.get("policy.clip_actions_to_env_space", True))
                        else raw_action
                    )
                    raw_actions.append(raw_action)
                    clipped_actions.append(clipped_action)
                    executed_actions.append(executed_action)
                    previous_actions.append(previous_action.copy())
                    obs, reward, terminated, truncated, info = env.step(executed_action)
                    step_success = _bool(info.get("success", False))
                    success = success or step_success
                    rewards.append(_scalar(reward))
                    terminated_flags.append(_bool(terminated))
                    truncated_flags.append(_bool(truncated))
                    success_flags.append(step_success)
                    previous_action = executed_action.astype(np.float32)
                if not success:
                    continue
                rgb, state_obs = _rgb_and_state(obs)
                rgbs.append(rgb)
                observations.append(state_obs)
                simulator_states.append(
                    _to_numpy(env.unwrapped.get_state()).reshape(-1).astype(np.float32)
                )
                rgb_array = np.stack(rgbs, axis=0)
                dino = np.concatenate(
                    [extractor.encode_batch(chunk) for chunk in batched(rgb_array, dino_batch_size)],
                    axis=0,
                )
                group = h5.create_group(f"episode_{successes:06d}")
                group.attrs["trajectory_id"] = successes
                group.attrs["reset_seed"] = reset_seed
                group.attrs["success"] = success
                group.attrs["length"] = len(executed_actions)
                group.create_dataset(
                    "timesteps",
                    data=np.arange(len(executed_actions), dtype=np.int32),
                )
                group.create_dataset(
                    "simulator_states",
                    data=np.stack(simulator_states),
                    compression="gzip",
                )
                group.create_dataset(
                    "observations_state",
                    data=np.stack(observations),
                    compression="gzip",
                )
                group.create_dataset(
                    "proprio",
                    data=np.stack(observations)[:, :21].astype(np.float32),
                    compression="gzip",
                )
                group.create_dataset("dino", data=dino, compression="gzip")
                group.create_dataset("raw_actions", data=np.stack(raw_actions), compression="gzip")
                group.create_dataset(
                    "clipped_actions", data=np.stack(clipped_actions), compression="gzip"
                )
                group.create_dataset(
                    "executed_actions", data=np.stack(executed_actions), compression="gzip"
                )
                group["actions"] = group["executed_actions"]
                group.create_dataset(
                    "previous_executed_actions",
                    data=np.stack(previous_actions),
                    compression="gzip",
                )
                group.create_dataset("rewards", data=np.asarray(rewards, dtype=np.float32))
                group.create_dataset("terminated", data=np.asarray(terminated_flags, dtype=np.bool_))
                group.create_dataset("truncated", data=np.asarray(truncated_flags, dtype=np.bool_))
                group.create_dataset("success", data=np.asarray(success_flags, dtype=np.bool_))
                if store_rgb:
                    group.create_dataset("rgb", data=rgb_array, compression="gzip")
                if successes == 0:
                    meta.attrs["state_shape"] = np.stack(simulator_states).shape[1:]
                    meta.attrs["action_shape"] = np.stack(executed_actions).shape[1:]
                    meta.attrs["dino_dim"] = dino.shape[-1]
                    meta.attrs["state_obs_dim"] = np.stack(observations).shape[-1]
                successes += 1
                if successes >= episodes:
                    break
            meta.attrs["attempts"] = attempts
            meta.attrs["successes"] = successes
    finally:
        env.close()
    if successes < episodes:
        raise RuntimeError(f"Collected only {successes}/{episodes} successful trajectories")
    tmp_path.replace(out_path)
    return out_path


def ensure_rl_rerun_action_aliases(
    config: Config,
    dataset_path: Path | None = None,
) -> Path:
    path = dataset_path or _state_dataset_path(config)
    if not path.exists():
        raise FileNotFoundError(path)
    created = 0
    with h5py.File(path, "r+") as h5:
        for key in sorted(k for k in h5.keys() if k.startswith("episode_")):
            group = h5[key]
            if "actions" in group:
                continue
            if "executed_actions" not in group:
                raise KeyError(f"{key} has neither actions nor executed_actions")
            group["actions"] = group["executed_actions"]
            created += 1
        h5["meta"].attrs["actions_alias_created"] = True
        h5["meta"].attrs["actions_alias_count"] = created
    return path


def audit_rl_rerun_state_dataset(
    config: Config,
    dataset_path: Path | None = None,
    samples: int = 100,
    horizon: int = 10,
    seed: int = 0,
    recompute_dino: bool = False,
    warm_start_replay: bool = False,
) -> Path:
    path = dataset_path or _state_dataset_path(config)
    if not path.exists():
        raise FileNotFoundError(path)
    rng = np.random.default_rng(seed)
    device = default_device()
    env = _make_state_data_env(config)
    extractor = dino_from_config(config, device) if recompute_dino else None
    state_errors: list[float] = []
    obs_errors: list[float] = []
    proprio_errors: list[float] = []
    replay_state_errors: list[float] = []
    reward_errors: list[float] = []
    success_mismatches = 0
    dino_errors: list[float] = []
    with h5py.File(path, "r") as h5:
        episode_keys = sorted(key for key in h5.keys() if key.startswith("episode_"))
        if not episode_keys:
            raise RuntimeError(f"No episodes in {path}")
        for _ in trange(samples, desc="audit RL rerun states"):
            key = str(rng.choice(episode_keys))
            group = h5[key]
            length = int(group.attrs["length"])
            max_start = max(0, length - horizon)
            start = int(rng.integers(0, max_start + 1))
            reset_seed = int(group.attrs["reset_seed"])
            states = np.asarray(group["simulator_states"], dtype=np.float32)
            observations = np.asarray(group["observations_state"], dtype=np.float32)
            actions = np.asarray(group["executed_actions"], dtype=np.float32)
            rewards = np.asarray(group["rewards"], dtype=np.float32)
            successes = np.asarray(group["success"], dtype=np.bool_)
            env.reset(seed=reset_seed)
            if warm_start_replay:
                for replay_step in range(start):
                    env.step(actions[replay_step])
            else:
                env.unwrapped.set_state(torch.from_numpy(states[start][None]).to(device).float())
            restored_state = _to_numpy(env.unwrapped.get_state()).reshape(-1).astype(np.float32)
            restored_obs = env.unwrapped.get_obs()
            _rgb, restored_state_obs = _rgb_and_state(restored_obs)
            state_errors.append(float(np.max(np.abs(restored_state - states[start]))))
            obs_errors.append(float(np.max(np.abs(restored_state_obs - observations[start]))))
            proprio_errors.append(
                float(np.max(np.abs(restored_state_obs[:21] - observations[start, :21])))
            )
            if extractor is not None:
                dino = extractor.encode_batch(_rgb[None])[0]
                stored_dino = np.asarray(group["dino"][start], dtype=np.float32)
                dino_errors.append(float(np.mean(np.square(dino - stored_dino))))
            last = min(length, start + horizon)
            for step in range(start, last):
                obs, reward, _terminated, _truncated, info = env.step(actions[step])
                actual_state = _to_numpy(env.unwrapped.get_state()).reshape(-1).astype(np.float32)
                replay_state_errors.append(
                    float(np.max(np.abs(actual_state - states[step + 1])))
                )
                reward_errors.append(float(abs(_scalar(reward) - rewards[step])))
                actual_success = _bool(info.get("success", False))
                success_mismatches += int(actual_success != bool(successes[step]))
    env.close()
    result = {
        "dataset": str(path),
        "samples": samples,
        "horizon": horizon,
        "seed": seed,
        "recompute_dino": recompute_dino,
        "warm_start_replay": warm_start_replay,
        "state_restore_max_abs": float(max(state_errors)),
        "state_restore_mean_abs_max": float(np.mean(state_errors)),
        "observation_restore_max_abs": float(max(obs_errors)),
        "proprio_restore_max_abs": float(max(proprio_errors)),
        "replay_state_max_abs": float(max(replay_state_errors)),
        "reward_max_abs": float(max(reward_errors)),
        "success_mismatches": int(success_mismatches),
        "dino_mse_max": float(max(dino_errors)) if dino_errors else None,
        "dino_mse_mean": float(np.mean(dino_errors)) if dino_errors else None,
    }
    output = _state_audit_result_dir(config) / "state_load_audit.json"
    write_json(output, result)
    return output


def train_rl_rerun_supervised_point(
    config: Config,
    n_demo: int,
    seed: int,
    dataset_path: Path | None = None,
    eval_episodes: int = 100,
    force: bool = False,
) -> dict[str, str]:
    from hcl_poc.learned_interface import (
        evaluate_learned_interface_hierarchy,
        train_learned_interface_hierarchy,
        train_learned_interface_representation,
    )
    from hcl_poc.vae_scaling import VAE_CANDIDATE, vae_scaling_config, write_vae_scaling_manifest

    ensure_rl_rerun_action_aliases(config, dataset_path)
    base = _rerun_base_config(config, dataset_path)
    point = vae_scaling_config(base, n_demo)
    manifest = write_vae_scaling_manifest(base, n_demo, force=force)
    representation = train_learned_interface_representation(
        point, VAE_CANDIDATE, seed=seed, force=force
    )
    hierarchy = train_learned_interface_hierarchy(
        point, VAE_CANDIDATE, seed=seed, force=force
    )
    evaluation = evaluate_learned_interface_hierarchy(
        point,
        VAE_CANDIDATE,
        "learned",
        seed=seed,
        episodes=eval_episodes,
        force=force,
    )
    return {
        "manifest": str(manifest),
        "representation": str(representation),
        "hierarchy": str(hierarchy),
        "evaluation": str(evaluation),
    }


@torch.inference_mode()
def run_rl_rerun_throughput_benchmark(
    config: Config,
    num_envs_values: list[int] | None = None,
    rollout_lens: list[int] | None = None,
    n_demo: int = 1000,
    seed: int = 0,
    output_path: Path | None = None,
) -> Path:
    from hcl_poc.incremental import _phase4_dino_from_config, _phase4_frame_inputs
    from hcl_poc.learned_interface import _low_condition_array
    from hcl_poc.low_level_rl import _load_frozen

    env_values = num_envs_values or [128, 256, 512, 1024, 2048, 4096, 8192, 16384]
    lens = rollout_lens or [10, 16, 32, 64]
    if any(value <= 0 for value in env_values):
        raise ValueError("num_envs values must be positive")
    if any(value <= 0 for value in lens):
        raise ValueError("rollout_lens values must be positive")

    out_path = output_path or (_state_audit_result_dir(config) / "rl_rerun_throughput_benchmark.csv")
    ensure_dir(out_path.parent)
    device = default_device()
    rerun_config = _rerun_base_config(config)
    frozen = _load_frozen(rerun_config, n_demo, seed, device)
    dino = _phase4_dino_from_config(config, device)

    fieldnames = [
        "num_envs",
        "rollout_len",
        "effective_batch",
        "sim_only_steps_per_sec",
        "sim_render_steps_per_sec",
        "sim_render_dino_steps_per_sec",
        "full_stack_steps_per_sec",
        "wall_clock_per_ppo_update_sec",
        "gpu_memory_allocated_mib",
        "gpu_memory_reserved_mib",
        "gpu_utilization_percent",
        "crashed",
        "nan_detected",
        "error",
    ]
    rows: list[dict[str, Any]] = []

    def timed_stage(num_envs: int, rollout_len: int, stage: str) -> tuple[float, bool]:
        obs_mode = "state" if stage == "sim_only" else "rgb+state"
        env = None
        try:
            env = _make_benchmark_env(config, num_envs, obs_mode)
            obs, _info = env.reset(seed=[7_000_000 + i for i in range(num_envs)])
            low = torch.as_tensor(env.single_action_space.low, device=device, dtype=torch.float32)
            high = torch.as_tensor(env.single_action_space.high, device=device, dtype=torch.float32)
            action = torch.zeros((num_envs, 3), device=device, dtype=torch.float32)
            previous = np.repeat(
                frozen.action_norm.transform(np.zeros((1, 3), dtype=np.float32)),
                num_envs,
                axis=0,
            )
            held_goal = np.zeros((num_envs, frozen.goal_dim), dtype=np.float32)
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            start = time.perf_counter()
            nan_detected = False
            for step in range(rollout_len):
                if stage in {"sim_render_dino", "full_stack"}:
                    frames = _phase4_frame_inputs(
                        obs, dino, int(config.get("dino.batch_size", 64))
                    )
                    if not np.isfinite(frames).all():
                        nan_detected = True
                    if stage == "full_stack":
                        normalized_frames = frozen.frame_norm.transform(frames)
                        z = frozen.goal_norm.transform(
                            frozen.encoder(
                                torch.from_numpy(
                                    frozen.representation_frame_norm.transform(frames)
                                )
                                .to(device)
                                .float()
                            )
                            .cpu()
                            .numpy()
                            .astype(np.float32)
                        )
                        high_condition = np.concatenate([normalized_frames, previous], axis=-1)
                        held_goal = (
                            frozen.high_model(
                                torch.from_numpy(high_condition).to(device).float()
                            )
                            .cpu()
                            .numpy()
                            .astype(np.float32)
                        )
                        remaining = np.full(
                            (num_envs, 1),
                            max(frozen.update_period - step % frozen.update_period, 1)
                            / frozen.horizon_steps,
                            dtype=np.float32,
                        )
                        condition = _low_condition_array(
                            normalized_frames,
                            z,
                            held_goal,
                            previous,
                            remaining,
                            frozen.conditioning,
                        )
                        normalized_action = frozen.low_model(
                            torch.from_numpy(condition).to(device).float()
                        )
                        action_np = frozen.action_norm.inverse(
                            normalized_action.cpu().numpy().astype(np.float32)
                        )
                        action = torch.clamp(
                            torch.from_numpy(action_np).to(device).float(), low, high
                        )
                        previous = frozen.action_norm.transform(action.cpu().numpy())
                obs, _reward, _terminated, _truncated, _info = env.step(action)
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            elapsed = time.perf_counter() - start
            return num_envs * rollout_len / elapsed, nan_detected
        finally:
            _safe_close(env)

    for num_envs in env_values:
        for rollout_len in lens:
            row: dict[str, Any] = {
                "num_envs": num_envs,
                "rollout_len": rollout_len,
                "effective_batch": num_envs * rollout_len,
                "sim_only_steps_per_sec": "",
                "sim_render_steps_per_sec": "",
                "sim_render_dino_steps_per_sec": "",
                "full_stack_steps_per_sec": "",
                "wall_clock_per_ppo_update_sec": "",
                "gpu_memory_allocated_mib": "",
                "gpu_memory_reserved_mib": "",
                "gpu_utilization_percent": "",
                "crashed": False,
                "nan_detected": False,
                "error": "",
            }
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
            try:
                for stage, key in [
                    ("sim_only", "sim_only_steps_per_sec"),
                    ("sim_render", "sim_render_steps_per_sec"),
                    ("sim_render_dino", "sim_render_dino_steps_per_sec"),
                    ("full_stack", "full_stack_steps_per_sec"),
                ]:
                    steps_per_sec, nan_detected = timed_stage(num_envs, rollout_len, stage)
                    row[key] = f"{steps_per_sec:.3f}"
                    row["nan_detected"] = bool(row["nan_detected"]) or nan_detected
                full_sps = float(row["full_stack_steps_per_sec"])
                row["wall_clock_per_ppo_update_sec"] = f"{num_envs * rollout_len / full_sps:.3f}"
                allocated, reserved = _cuda_memory_mib()
                row["gpu_memory_allocated_mib"] = (
                    f"{allocated:.1f}" if allocated is not None else ""
                )
                row["gpu_memory_reserved_mib"] = f"{reserved:.1f}" if reserved is not None else ""
                util = _gpu_utilization_percent()
                row["gpu_utilization_percent"] = util if util is not None else ""
            except Exception as exc:
                row["crashed"] = True
                row["error"] = f"{type(exc).__name__}: {exc}"
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            rows.append(row)
            with out_path.open("w", newline="") as file:
                writer = csv.DictWriter(file, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)
    return out_path


def _finite_horizon_gae_returns(
    rewards: np.ndarray,
    values: np.ndarray,
    terminated_after_step: np.ndarray,
    gamma: float,
    gae_lambda: float,
    next_value: float,
) -> np.ndarray:
    if rewards.shape != values.shape or rewards.shape != terminated_after_step.shape:
        raise ValueError("rewards, values, and terminated_after_step must have equal shape")
    advantages = np.zeros_like(rewards, dtype=np.float64)
    last_gae = 0.0
    for step in reversed(range(len(rewards))):
        next_nonterminal = 0.0 if bool(terminated_after_step[step]) else 1.0
        following_value = next_value if step == len(rewards) - 1 else float(values[step + 1])
        delta = (
            float(rewards[step])
            + gamma * following_value * next_nonterminal
            - float(values[step])
        )
        last_gae = delta + gamma * gae_lambda * next_nonterminal * last_gae
        advantages[step] = last_gae
    return advantages + values


@torch.inference_mode()
def run_rl_rerun_algorithm_audit(
    config: Config,
    dataset_path: Path | None = None,
    n_demo: int = 1000,
    seed: int = 0,
    output_path: Path | None = None,
) -> Path:
    from hcl_poc.learned_interface import _low_condition_array
    from hcl_poc.low_level_rl import _load_frozen

    path = dataset_path or _state_dataset_path(config)
    if not path.exists():
        raise FileNotFoundError(path)

    device = default_device()
    rerun_config = _rerun_base_config(config, path)
    frozen = _load_frozen(rerun_config, n_demo, seed, device)

    rewards = np.arange(1, 11, dtype=np.float64)
    values = np.zeros(10, dtype=np.float64)
    terminated_after = np.zeros(10, dtype=bool)
    terminated_after[-1] = True
    returns = _finite_horizon_gae_returns(
        rewards, values, terminated_after, gamma=1.0, gae_lambda=1.0, next_value=999.0
    )
    expected_returns = np.array([55, 54, 52, 49, 45, 40, 34, 27, 19, 10], dtype=np.float64)
    gae_max_abs_error = float(np.max(np.abs(returns - expected_returns)))
    no_bootstrap_last_return = float(returns[-1])

    leaking_returns = _finite_horizon_gae_returns(
        rewards,
        values,
        np.zeros(10, dtype=bool),
        gamma=1.0,
        gae_lambda=1.0,
        next_value=999.0,
    )
    bootstrap_sensitivity = float(leaking_returns[-1] - returns[-1])

    with h5py.File(path, "r") as h5:
        episode_keys = sorted(key for key in h5.keys() if key.startswith("episode_"))
        if not episode_keys:
            raise ValueError(f"No episodes found in {path}")
        selected_key = None
        selected_t = 0
        horizon = int(frozen.horizon_steps)
        for key in episode_keys:
            if len(h5[key]["executed_actions"]) > horizon:
                selected_key = key
                selected_t = min(5, len(h5[key]["executed_actions"]) - horizon - 1)
                break
        if selected_key is None:
            raise ValueError("No episode is long enough for a 10-step local audit")
        group = h5[selected_key]
        dino = np.asarray(group["dino"], dtype=np.float32)
        proprio = np.asarray(group["proprio"], dtype=np.float32)
        previous_actions = np.asarray(group["previous_executed_actions"], dtype=np.float32)
        frame_t = np.concatenate([dino[selected_t], proprio[selected_t]], axis=-1)[None]
        frame_goal = np.concatenate(
            [dino[selected_t + horizon], proprio[selected_t + horizon]], axis=-1
        )[None]
        previous = frozen.action_norm.transform(previous_actions[selected_t : selected_t + 1])

    norm_frame_t = frozen.frame_norm.transform(frame_t)
    z_t = frozen.goal_norm.transform(
        frozen.encoder(
            torch.from_numpy(frozen.representation_frame_norm.transform(frame_t))
            .to(device)
            .float()
        )
        .cpu()
        .numpy()
        .astype(np.float32)
    )
    goal = frozen.goal_norm.transform(
        frozen.encoder(
            torch.from_numpy(frozen.representation_frame_norm.transform(frame_goal))
            .to(device)
            .float()
        )
        .cpu()
        .numpy()
        .astype(np.float32)
    )
    condition = _low_condition_array(
        norm_frame_t,
        z_t,
        goal,
        previous,
        np.ones((1, 1), dtype=np.float32),
        frozen.conditioning,
    )
    normalized_base = frozen.low_model(torch.from_numpy(condition).to(device).float())
    base_action = frozen.action_norm.inverse(normalized_base.cpu().numpy().astype(np.float32))
    clipped_base_action = np.clip(base_action, -1.0, 1.0)
    zero_residual_action = np.clip(
        base_action + 0.1 * np.tanh(np.zeros_like(base_action)), -1.0, 1.0
    )
    zero_residual_max_abs = float(np.max(np.abs(clipped_base_action - zero_residual_action)))
    unclipped_action_box_overshoot = float(np.max(np.abs(base_action - clipped_base_action)))

    result = {
        "dataset": str(path),
        "n_demo": n_demo,
        "seed": seed,
        "horizon_steps": int(frozen.horizon_steps),
        "update_period": int(frozen.update_period),
        "local_episode_length": int(frozen.horizon_steps),
        "selected_episode": selected_key,
        "selected_timestep": selected_t,
        "gae_unit": {
            "rewards": rewards.tolist(),
            "expected_returns": expected_returns.tolist(),
            "computed_returns": returns.tolist(),
            "max_abs_error": gae_max_abs_error,
            "next_value_used_for_terminal_test": 999.0,
            "terminal_last_return": no_bootstrap_last_return,
            "bootstrap_sensitivity_if_not_terminal": bootstrap_sensitivity,
        },
        "checks": {
            "horizon_is_10": int(frozen.horizon_steps) == 10,
            "update_period_is_10": int(frozen.update_period) == 10,
            "gae_matches_hand_computed_returns": gae_max_abs_error < 1e-9,
            "terminal_step_does_not_bootstrap": abs(no_bootstrap_last_return - 10.0) < 1e-9,
            "nonterminal_would_bootstrap": bootstrap_sensitivity > 900.0,
            "zero_residual_matches_frozen_policy": zero_residual_max_abs < 1e-9,
        },
        "zero_residual_max_abs_action_error": zero_residual_max_abs,
        "unclipped_frozen_action_box_overshoot": unclipped_action_box_overshoot,
        "gate_pass": False,
    }
    result["gate_pass"] = bool(all(result["checks"].values()))
    out_path = output_path or (_state_audit_result_dir(config) / "algorithm_audit.json")
    ensure_dir(out_path.parent)
    write_json(out_path, result)
    return out_path


@torch.inference_mode()
def run_rl_rerun_local_reset_audit(
    config: Config,
    dataset_path: Path | None = None,
    n_demo: int = 1000,
    seed: int = 0,
    num_envs: int = 16,
    batches: int = 8,
    output_path: Path | None = None,
) -> Path:
    from hcl_poc.incremental import _phase4_dino_from_config, _phase4_frame_inputs

    path = dataset_path or _state_dataset_path(config)
    if not path.exists():
        raise FileNotFoundError(path)
    if num_envs <= 0 or batches <= 0:
        raise ValueError("num_envs and batches must be positive")

    horizon = 10
    rng = np.random.default_rng(seed)
    device = default_device()
    dino = _phase4_dino_from_config(config, device)
    env = _make_benchmark_env(config, num_envs, "rgb+state")
    max_state_errors: list[float] = []
    max_obs_state_errors: list[float] = []
    max_frame_errors: list[float] = []
    max_previous_action_errors: list[float] = []
    selected_timesteps: list[int] = []
    selected_lengths: list[int] = []
    try:
        with h5py.File(path, "r") as h5:
            episode_keys = sorted(key for key in h5.keys() if key.startswith("episode_"))[:n_demo]
            lengths = {
                key: int(h5[key].attrs["length"])
                for key in episode_keys
                if int(h5[key].attrs["length"]) > horizon
            }
            if not lengths:
                raise ValueError("No train episode is long enough for local reset audit")
            max_t = max(length - horizon - 1 for length in lengths.values())
            candidates_by_t = {
                t: [key for key, length in lengths.items() if length > t + horizon]
                for t in range(max_t + 1)
            }
            for _batch in range(batches):
                valid_t = [t for t, keys in candidates_by_t.items() if len(keys) >= num_envs]
                if not valid_t:
                    raise ValueError(
                        f"No timestep has at least {num_envs} eligible trajectories"
                    )
                t = int(rng.choice(valid_t))
                keys = rng.choice(candidates_by_t[t], size=num_envs, replace=False)
                reset_seeds = [int(h5[key].attrs["reset_seed"]) for key in keys]
                obs, _info = env.reset(seed=reset_seeds)
                for step in range(t):
                    replay_actions = np.stack(
                        [np.asarray(h5[key]["executed_actions"][step], dtype=np.float32) for key in keys]
                    )
                    obs, _reward, _terminated, _truncated, _info = env.step(
                        torch.from_numpy(replay_actions).to(device).float()
                    )
                actual_states = _to_numpy(env.unwrapped.get_state()).astype(np.float32)
                actual_obs_state = _to_numpy(obs["state"]).astype(np.float32)
                if actual_states.ndim == 1:
                    actual_states = actual_states[None]
                if actual_obs_state.ndim == 1:
                    actual_obs_state = actual_obs_state[None]
                expected_states = np.stack(
                    [np.asarray(h5[key]["simulator_states"][t], dtype=np.float32) for key in keys]
                )
                expected_obs_state = np.stack(
                    [np.asarray(h5[key]["observations_state"][t], dtype=np.float32) for key in keys]
                )
                expected_previous = np.stack(
                    [
                        np.asarray(h5[key]["previous_executed_actions"][t], dtype=np.float32)
                        for key in keys
                    ]
                )
                expected_frames = np.stack(
                    [
                        np.concatenate(
                            [
                                np.asarray(h5[key]["dino"][t], dtype=np.float32),
                                np.asarray(h5[key]["proprio"][t], dtype=np.float32),
                            ],
                            axis=-1,
                        )
                        for key in keys
                    ]
                )
                actual_frames = _phase4_frame_inputs(
                    obs, dino, int(config.get("dino.batch_size", 64))
                )
                previous = np.stack(
                    [
                        np.zeros(3, dtype=np.float32)
                        if t == 0
                        else np.asarray(h5[key]["executed_actions"][t - 1], dtype=np.float32)
                        for key in keys
                    ]
                )
                max_state_errors.append(float(np.max(np.abs(actual_states - expected_states))))
                max_obs_state_errors.append(
                    float(np.max(np.abs(actual_obs_state - expected_obs_state)))
                )
                max_frame_errors.append(float(np.mean(np.square(actual_frames - expected_frames))))
                max_previous_action_errors.append(
                    float(np.max(np.abs(previous - expected_previous)))
                )
                selected_timesteps.append(t)
                selected_lengths.extend(int(lengths[str(key)]) for key in keys)
    finally:
        env.close()

    result = {
        "dataset": str(path),
        "n_demo": n_demo,
        "seed": seed,
        "num_envs": num_envs,
        "batches": batches,
        "local_episode_length": horizon,
        "sampled_resets": int(num_envs * batches),
        "state_max_abs_error": float(max(max_state_errors)),
        "obs_state_max_abs_error": float(max(max_obs_state_errors)),
        "frame_mse_max": float(max(max_frame_errors)),
        "frame_mse_mean": float(np.mean(max_frame_errors)),
        "previous_action_max_abs_error": float(max(max_previous_action_errors)),
        "selected_timestep_min": int(min(selected_timesteps)),
        "selected_timestep_max": int(max(selected_timesteps)),
        "selected_episode_length_min": int(min(selected_lengths)),
        "selected_episode_length_max": int(max(selected_lengths)),
        "gate_pass": bool(
            max(max_state_errors) == 0.0
            and max(max_obs_state_errors) == 0.0
            and max(max_previous_action_errors) == 0.0
            and max(max_frame_errors) < 1e-4
        ),
    }
    out_path = output_path or (_state_audit_result_dir(config) / "local_reset_audit.json")
    ensure_dir(out_path.parent)
    write_json(out_path, result)
    return out_path
