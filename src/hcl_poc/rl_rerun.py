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


def _vector_rgb_and_state(obs: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    rgb = _to_numpy(obs["sensor_data"]["base_camera"]["rgb"])
    state = _to_numpy(obs["state"])
    if rgb.ndim != 4:
        raise ValueError(f"Expected vector RGB observation with 4 dims, got {rgb.shape}")
    if state.ndim != 2:
        raise ValueError(f"Expected vector state observation with 2 dims, got {state.shape}")
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


def _vector_dataset_path(config: Config) -> Path:
    return (
        config.path_value("paths.incremental_data_dir").parent
        / "rl_rerun"
        / "pusht_vector_state_demos.h5"
    )


def _state_audit_result_dir(config: Config) -> Path:
    return ensure_dir(config.path_value("paths.incremental_results_dir").parent / "rl_rerun")


def _rl_rerun_artifact_dir(config: Config) -> Path:
    return ensure_dir(config.path_value("paths.incremental_artifact_dir").parent / "rl_rerun")


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


@torch.inference_mode()
def collect_rl_rerun_vector_dataset(
    config: Config,
    output_path: Path | None = None,
    num_envs: int = 16,
    batches: int = 2,
    max_steps: int = 60,
    seed_start: int = 9_500_000,
    checkpoint_path: Path | None = None,
    store_dino: bool = True,
    force: bool = False,
) -> Path:
    from hcl_poc.incremental import _phase4_dino_from_config, _phase4_frame_inputs

    if num_envs <= 0 or batches <= 0 or max_steps <= 10:
        raise ValueError("num_envs and batches must be positive; max_steps must exceed 10")
    out_path = output_path or _vector_dataset_path(config)
    if out_path.exists() and not force:
        with h5py.File(out_path, "r") as h5:
            existing = len([key for key in h5.keys() if key.startswith("batch_")])
        if existing >= batches:
            return out_path
    ensure_dir(out_path.parent)
    tmp_path = out_path.with_suffix(".tmp.h5")
    tmp_path.unlink(missing_ok=True)

    device = default_device()
    teacher_path = checkpoint_path or _rl_paths(config).best
    teacher = load_ppo_agent(teacher_path, device)
    dino = _phase4_dino_from_config(config, device) if store_dino else None
    env = _make_benchmark_env(config, num_envs, "rgb+state")
    action_low = torch.as_tensor(env.single_action_space.low, device=device, dtype=torch.float32)
    action_high = torch.as_tensor(env.single_action_space.high, device=device, dtype=torch.float32)

    try:
        with h5py.File(tmp_path, "w") as h5:
            meta = h5.create_group("meta")
            meta.attrs["source"] = "vector_consistent_privileged_ppo"
            meta.attrs["checkpoint"] = str(teacher_path)
            meta.attrs["num_envs"] = int(num_envs)
            meta.attrs["batches"] = int(batches)
            meta.attrs["max_steps"] = int(max_steps)
            meta.attrs["seed_start"] = int(seed_start)
            meta.attrs["sim_backend"] = _rl_backend(config)
            meta.attrs["control_mode"] = config.get("control_mode")
            meta.attrs["obs_mode"] = "rgb+state"
            meta.attrs["store_dino"] = bool(store_dino)
            meta.attrs["dino_model"] = config.get("dino.model_name")
            meta.attrs["dino_feature_type"] = config.get("dino.feature_type", "cls")
            meta.attrs.update(_git_metadata())
            for batch_index in trange(batches, desc="collect vector PPO batches"):
                batch_seed = int(seed_start + batch_index)
                obs, _info = env.reset(seed=batch_seed)

                states: list[np.ndarray] = []
                obs_states: list[np.ndarray] = []
                proprios: list[np.ndarray] = []
                dinos: list[np.ndarray] = []
                raw_actions: list[np.ndarray] = []
                executed_actions: list[np.ndarray] = []
                previous_actions: list[np.ndarray] = []
                rewards: list[np.ndarray] = []
                terminated_flags: list[np.ndarray] = []
                truncated_flags: list[np.ndarray] = []
                success_flags: list[np.ndarray] = []
                success_once = np.zeros(num_envs, dtype=np.bool_)
                previous = np.zeros((num_envs, 3), dtype=np.float32)

                def store_observation(current_obs: dict[str, Any]) -> None:
                    rgb, state = _vector_rgb_and_state(current_obs)
                    states.append(
                        _to_numpy(env.unwrapped.get_state()).astype(np.float32).copy()
                    )
                    obs_states.append(state.astype(np.float32).copy())
                    proprios.append(state[:, :21].astype(np.float32).copy())
                    if dino is not None:
                        features = [dino.encode_batch(chunk) for chunk in batched(rgb, int(config.get("dino.batch_size", 64)))]
                        dinos.append(np.concatenate(features, axis=0).astype(np.float32))

                store_observation(obs)
                for _step in range(max_steps):
                    state_t = torch.from_numpy(obs_states[-1]).to(device).float()
                    action_t, _logprob, _entropy, _value = teacher.get_action_and_value(
                        state_t,
                        deterministic=True,
                    )
                    raw_action = action_t.detach().cpu().numpy().astype(np.float32)
                    executed = torch.clamp(action_t, action_low, action_high)
                    executed_np = executed.detach().cpu().numpy().astype(np.float32)
                    next_obs, reward, terminated, truncated, info = env.step(executed)
                    success = _to_numpy(info.get("success", np.zeros(num_envs, dtype=np.bool_))).reshape(-1).astype(np.bool_)
                    success_once |= success

                    raw_actions.append(raw_action)
                    executed_actions.append(executed_np)
                    previous_actions.append(previous.copy())
                    rewards.append(_to_numpy(reward).reshape(-1).astype(np.float32))
                    terminated_flags.append(
                        _to_numpy(terminated).reshape(-1).astype(np.bool_)
                    )
                    truncated_flags.append(_to_numpy(truncated).reshape(-1).astype(np.bool_))
                    success_flags.append(success)
                    previous = executed_np
                    obs = next_obs
                    store_observation(obs)

                group = h5.create_group(f"batch_{batch_index:06d}")
                group.attrs["batch_seed"] = batch_seed
                group.attrs["num_envs"] = int(num_envs)
                group.attrs["max_steps"] = int(max_steps)
                group.attrs["success_count"] = int(success_once.sum())
                group.create_dataset("simulator_states", data=np.stack(states), compression="gzip")
                group.create_dataset("observations_state", data=np.stack(obs_states), compression="gzip")
                group.create_dataset("proprio", data=np.stack(proprios), compression="gzip")
                if dinos:
                    group.create_dataset("dino", data=np.stack(dinos), compression="gzip")
                group.create_dataset("raw_actions", data=np.stack(raw_actions), compression="gzip")
                group.create_dataset(
                    "executed_actions", data=np.stack(executed_actions), compression="gzip"
                )
                group.create_dataset(
                    "previous_executed_actions",
                    data=np.stack(previous_actions),
                    compression="gzip",
                )
                group.create_dataset("rewards", data=np.stack(rewards), compression="gzip")
                group.create_dataset(
                    "terminated", data=np.stack(terminated_flags), compression="gzip"
                )
                group.create_dataset(
                    "truncated", data=np.stack(truncated_flags), compression="gzip"
                )
                group.create_dataset("success", data=np.stack(success_flags), compression="gzip")
                group.create_dataset("success_once", data=success_once, compression="gzip")
    finally:
        env.close()
    tmp_path.replace(out_path)
    return out_path


@torch.inference_mode()
def audit_rl_rerun_vector_dataset(
    config: Config,
    dataset_path: Path | None = None,
    batches: int = 4,
    seed: int = 0,
    horizon: int = 10,
    output_path: Path | None = None,
) -> Path:
    from hcl_poc.incremental import _phase4_dino_from_config, _phase4_frame_inputs

    path = dataset_path or _vector_dataset_path(config)
    if not path.exists():
        raise FileNotFoundError(path)
    if batches <= 0 or horizon <= 0:
        raise ValueError("batches and horizon must be positive")

    rng = np.random.default_rng(seed)
    device = default_device()
    with h5py.File(path, "r") as h5:
        meta = h5["meta"].attrs
        num_envs = int(meta["num_envs"])
        max_steps = int(meta["max_steps"])
        store_dino = bool(meta["store_dino"])
        batch_keys = sorted(key for key in h5.keys() if key.startswith("batch_"))
    dino = _phase4_dino_from_config(config, device) if store_dino else None
    env = _make_benchmark_env(config, num_envs, "rgb+state")

    current_state_errors: list[float] = []
    current_obs_errors: list[float] = []
    current_frame_errors: list[float] = []
    goal_state_errors: list[float] = []
    goal_obs_errors: list[float] = []
    goal_frame_errors: list[float] = []
    previous_action_errors: list[float] = []
    chosen_timesteps: list[int] = []
    chosen_batches: list[str] = []
    audit_keys = rng.choice(
        batch_keys,
        size=batches,
        replace=batches > len(batch_keys),
    )

    try:
        with h5py.File(path, "r") as h5:
            for selected_key in audit_keys:
                key = str(selected_key)
                group = h5[key]
                t = int(rng.integers(0, max_steps - horizon + 1))
                obs, _info = env.reset(seed=int(group.attrs["batch_seed"]))
                for step in range(t):
                    action = torch.from_numpy(
                        np.asarray(group["executed_actions"][step], dtype=np.float32)
                    ).to(device)
                    obs, _reward, _terminated, _truncated, _info = env.step(action)

                actual_state = _to_numpy(env.unwrapped.get_state()).astype(np.float32)
                actual_obs_state = _to_numpy(obs["state"]).astype(np.float32)
                expected_state = np.asarray(group["simulator_states"][t], dtype=np.float32)
                expected_obs_state = np.asarray(group["observations_state"][t], dtype=np.float32)
                current_state_errors.append(float(np.max(np.abs(actual_state - expected_state))))
                current_obs_errors.append(
                    float(np.max(np.abs(actual_obs_state - expected_obs_state)))
                )
                previous = (
                    np.zeros((num_envs, 3), dtype=np.float32)
                    if t == 0
                    else np.asarray(group["executed_actions"][t - 1], dtype=np.float32)
                )
                expected_previous = np.asarray(
                    group["previous_executed_actions"][t], dtype=np.float32
                )
                previous_action_errors.append(float(np.max(np.abs(previous - expected_previous))))
                if dino is not None and "dino" in group:
                    expected_frame = np.concatenate(
                        [
                            np.asarray(group["dino"][t], dtype=np.float32),
                            np.asarray(group["proprio"][t], dtype=np.float32),
                        ],
                        axis=-1,
                    )
                    actual_frame = _phase4_frame_inputs(
                        obs, dino, int(config.get("dino.batch_size", 64))
                    )
                    current_frame_errors.append(
                        float(np.mean(np.square(actual_frame - expected_frame)))
                    )

                for step in range(t, t + horizon):
                    action = torch.from_numpy(
                        np.asarray(group["executed_actions"][step], dtype=np.float32)
                    ).to(device)
                    obs, _reward, _terminated, _truncated, _info = env.step(action)
                actual_goal_state = _to_numpy(env.unwrapped.get_state()).astype(np.float32)
                actual_goal_obs_state = _to_numpy(obs["state"]).astype(np.float32)
                expected_goal_state = np.asarray(
                    group["simulator_states"][t + horizon], dtype=np.float32
                )
                expected_goal_obs_state = np.asarray(
                    group["observations_state"][t + horizon], dtype=np.float32
                )
                goal_state_errors.append(
                    float(np.max(np.abs(actual_goal_state - expected_goal_state)))
                )
                goal_obs_errors.append(
                    float(np.max(np.abs(actual_goal_obs_state - expected_goal_obs_state)))
                )
                if dino is not None and "dino" in group:
                    expected_goal_frame = np.concatenate(
                        [
                            np.asarray(group["dino"][t + horizon], dtype=np.float32),
                            np.asarray(group["proprio"][t + horizon], dtype=np.float32),
                        ],
                        axis=-1,
                    )
                    actual_goal_frame = _phase4_frame_inputs(
                        obs, dino, int(config.get("dino.batch_size", 64))
                    )
                    goal_frame_errors.append(
                        float(np.mean(np.square(actual_goal_frame - expected_goal_frame)))
                    )
                chosen_timesteps.append(t)
                chosen_batches.append(key)
    finally:
        env.close()

    result = {
        "dataset": str(path),
        "seed": seed,
        "audited_batches": batches,
        "num_envs": num_envs,
        "horizon": horizon,
        "max_steps": max_steps,
        "chosen_timestep_min": int(min(chosen_timesteps)),
        "chosen_timestep_max": int(max(chosen_timesteps)),
        "chosen_batches": chosen_batches,
        "current_state_max_abs_error": float(max(current_state_errors)),
        "current_obs_state_max_abs_error": float(max(current_obs_errors)),
        "current_frame_mse_max": float(max(current_frame_errors)) if current_frame_errors else None,
        "current_frame_mse_mean": float(np.mean(current_frame_errors)) if current_frame_errors else None,
        "goal_state_max_abs_error": float(max(goal_state_errors)),
        "goal_obs_state_max_abs_error": float(max(goal_obs_errors)),
        "goal_frame_mse_max": float(max(goal_frame_errors)) if goal_frame_errors else None,
        "goal_frame_mse_mean": float(np.mean(goal_frame_errors)) if goal_frame_errors else None,
        "previous_action_max_abs_error": float(max(previous_action_errors)),
        "gate_pass": bool(
            max(current_state_errors) == 0.0
            and max(current_obs_errors) == 0.0
            and max(goal_state_errors) == 0.0
            and max(goal_obs_errors) == 0.0
            and max(previous_action_errors) == 0.0
            and (not current_frame_errors or max(current_frame_errors) < 1e-4)
            and (not goal_frame_errors or max(goal_frame_errors) < 1e-4)
        ),
    }
    out_path = output_path or (_state_audit_result_dir(config) / "vector_state_audit.json")
    ensure_dir(out_path.parent)
    write_json(out_path, result)
    return out_path


def _encode_rerun_frames(frozen: Any, frames: np.ndarray, device: torch.device) -> np.ndarray:
    return frozen.goal_norm.transform(
        frozen.encoder(
            torch.from_numpy(frozen.representation_frame_norm.transform(frames))
            .to(device)
            .float()
        )
        .detach()
        .cpu()
        .numpy()
        .astype(np.float32)
    )


@torch.inference_mode()
def audit_rl_rerun_local_mode_a(
    config: Config,
    dataset_path: Path | None = None,
    n_demo: int = 1000,
    seed: int = 0,
    episodes: int = 4,
    output_path: Path | None = None,
) -> Path:
    from hcl_poc.incremental import _phase4_dino_from_config, _phase4_frame_inputs
    from hcl_poc.learned_interface import _low_condition_array
    from hcl_poc.low_level_rl import _load_frozen

    path = dataset_path or _vector_dataset_path(config)
    if not path.exists():
        raise FileNotFoundError(path)
    if episodes <= 0:
        raise ValueError("episodes must be positive")

    device = default_device()
    rng = np.random.default_rng(seed)
    rerun_config = _rerun_base_config(config)
    frozen = _load_frozen(rerun_config, n_demo, seed, device)
    horizon = int(frozen.horizon_steps)
    if horizon != 10:
        raise ValueError(f"Expected 10-step local horizon, got {horizon}")
    dino = _phase4_dino_from_config(config, device)

    with h5py.File(path, "r") as h5:
        meta = h5["meta"].attrs
        num_envs = int(meta["num_envs"])
        max_steps = int(meta["max_steps"])
        batch_keys = sorted(key for key in h5.keys() if key.startswith("batch_"))
    env = _make_benchmark_env(config, num_envs, "rgb+state")
    action_low = torch.as_tensor(env.single_action_space.low, device=device, dtype=torch.float32)
    action_high = torch.as_tensor(env.single_action_space.high, device=device, dtype=torch.float32)

    initial_distances: list[np.ndarray] = []
    final_distances: list[np.ndarray] = []
    terminal_rewards: list[np.ndarray] = []
    progress_rewards: list[np.ndarray] = []
    saturation_rates: list[float] = []
    task_success_once = np.zeros((episodes, num_envs), dtype=np.bool_)
    chosen_batches: list[str] = []
    chosen_timesteps: list[int] = []
    evaluation_keys = rng.choice(
        batch_keys,
        size=episodes,
        replace=episodes > len(batch_keys),
    )

    try:
        with h5py.File(path, "r") as h5:
            for episode in trange(episodes, desc="audit local Mode-A"):
                key = str(evaluation_keys[episode])
                group = h5[key]
                t = int(rng.integers(0, max_steps - horizon + 1))
                obs, _info = env.reset(seed=int(group.attrs["batch_seed"]))
                for step in range(t):
                    action = torch.from_numpy(
                        np.asarray(group["executed_actions"][step], dtype=np.float32)
                    ).to(device)
                    obs, _reward, _terminated, _truncated, _info = env.step(action)
                goal_frame = np.concatenate(
                    [
                        np.asarray(group["dino"][t + horizon], dtype=np.float32),
                        np.asarray(group["proprio"][t + horizon], dtype=np.float32),
                    ],
                    axis=-1,
                )
                goal_z = _encode_rerun_frames(frozen, goal_frame, device)
                previous = frozen.action_norm.transform(
                    np.asarray(group["previous_executed_actions"][t], dtype=np.float32)
                )
                episode_initial_distance: np.ndarray | None = None
                total_progress = np.zeros(num_envs, dtype=np.float32)
                saturation_count = 0
                for local_step in range(horizon):
                    frames = _phase4_frame_inputs(
                        obs, dino, int(config.get("dino.batch_size", 64))
                    )
                    current_z = _encode_rerun_frames(frozen, frames, device)
                    distance = np.mean(np.square(current_z - goal_z), axis=-1).astype(np.float32)
                    if episode_initial_distance is None:
                        episode_initial_distance = distance.copy()
                    remaining = np.full(
                        (num_envs, 1),
                        (horizon - local_step) / horizon,
                        dtype=np.float32,
                    )
                    condition = _low_condition_array(
                        frozen.frame_norm.transform(frames),
                        current_z,
                        goal_z,
                        previous,
                        remaining,
                        frozen.conditioning,
                    )
                    normalized_action = frozen.low_model(
                        torch.from_numpy(condition).to(device).float()
                    )
                    raw_action = frozen.action_norm.inverse(
                        normalized_action.cpu().numpy().astype(np.float32)
                    )
                    unclipped = torch.from_numpy(raw_action).to(device).float()
                    action = torch.clamp(unclipped, action_low, action_high)
                    saturation_count += int(torch.any(unclipped != action, dim=-1).sum().cpu())
                    obs, _reward, _terminated, _truncated, info = env.step(action)
                    next_frames = _phase4_frame_inputs(
                        obs, dino, int(config.get("dino.batch_size", 64))
                    )
                    next_z = _encode_rerun_frames(frozen, next_frames, device)
                    next_distance = np.mean(
                        np.square(next_z - goal_z), axis=-1
                    ).astype(np.float32)
                    total_progress += distance - next_distance
                    previous = frozen.action_norm.transform(action.cpu().numpy())
                    task_success_once[episode] |= (
                        _to_numpy(info.get("success", np.zeros(num_envs, dtype=np.bool_)))
                        .reshape(-1)
                        .astype(np.bool_)
                    )
                    if local_step == horizon - 1:
                        final_distances.append(next_distance)
                        terminal_rewards.append(-next_distance)
                if episode_initial_distance is None:
                    raise RuntimeError("Local Mode-A audit did not execute any steps")
                initial_distances.append(episode_initial_distance)
                progress_rewards.append(total_progress)
                saturation_rates.append(saturation_count / float(num_envs * horizon))
                chosen_batches.append(key)
                chosen_timesteps.append(t)
    finally:
        env.close()

    initial = np.concatenate(initial_distances)
    final = np.concatenate(final_distances)
    progress = np.concatenate(progress_rewards)
    terminal = np.concatenate(terminal_rewards)
    result = {
        "dataset": str(path),
        "n_demo": n_demo,
        "seed": seed,
        "episodes": episodes,
        "num_envs": num_envs,
        "sampled_local_episodes": int(episodes * num_envs),
        "horizon": horizon,
        "chosen_batches": chosen_batches,
        "chosen_timestep_min": int(min(chosen_timesteps)),
        "chosen_timestep_max": int(max(chosen_timesteps)),
        "initial_distance_mean": float(np.mean(initial)),
        "initial_distance_median": float(np.median(initial)),
        "final_distance_mean": float(np.mean(final)),
        "final_distance_median": float(np.median(final)),
        "distance_reduction_mean": float(np.mean(initial - final)),
        "distance_reduction_median": float(np.median(initial - final)),
        "distance_reduction_fraction": float(np.mean(final < initial)),
        "progress_reward_mean": float(np.mean(progress)),
        "terminal_reward_mean": float(np.mean(terminal)),
        "action_saturation_rate": float(np.mean(saturation_rates)),
        "task_success_once_fraction": float(np.mean(task_success_once)),
        "gate_pass": bool(np.mean(final < initial) > 0.5),
    }
    out_path = output_path or (_state_audit_result_dir(config) / "local_mode_a_audit.json")
    ensure_dir(out_path.parent)
    write_json(out_path, result)
    return out_path


def train_rl_rerun_local_r1(
    config: Config,
    dataset_path: Path | None = None,
    n_demo: int = 1000,
    seed: int = 0,
    run_name: str = "local_r1_mode_a",
    total_steps: int = 32_768,
    alpha: float = 0.1,
    terminal_weight: float = 1.0,
    residual_penalty_weight: float | None = None,
    force: bool = False,
) -> Path:
    from hcl_poc.incremental import _phase4_dino_from_config, _phase4_frame_inputs
    from hcl_poc.learned_interface import _low_condition_array
    from hcl_poc.low_level_rl import ResidualActorCritic, _load_frozen
    from hcl_poc.utils import set_seed

    path = dataset_path or _vector_dataset_path(config)
    if not path.exists():
        raise FileNotFoundError(path)
    if total_steps <= 0:
        raise ValueError("total_steps must be positive")

    artifact = ensure_dir(
        _rl_rerun_artifact_dir(config)
        / "local_r1"
        / f"n{n_demo}"
        / f"seed{seed}"
        / run_name
    )
    result_dir = ensure_dir(
        _state_audit_result_dir(config)
        / "local_r1"
        / f"n{n_demo}"
        / f"seed{seed}"
        / run_name
    )
    latest = artifact / "latest.pt"
    history_path = result_dir / "history.json"
    if force:
        latest.unlink(missing_ok=True)
        history_path.unlink(missing_ok=True)

    device = default_device()
    set_seed(seed + 90_000)
    rerun_config = _rerun_base_config(config)
    frozen = _load_frozen(rerun_config, n_demo, seed, device)
    horizon = int(frozen.horizon_steps)
    if horizon != 10:
        raise ValueError(f"Expected 10-step local horizon, got {horizon}")

    with h5py.File(path, "r") as h5_meta:
        meta = h5_meta["meta"].attrs
        num_envs = int(meta["num_envs"])
        max_steps = int(meta["max_steps"])
        batch_keys = sorted(key for key in h5_meta.keys() if key.startswith("batch_"))
    rollout_steps = horizon
    batch_size = num_envs * rollout_steps
    minibatches = int(config.get("low_level_rl.num_minibatches", 8))
    if batch_size % minibatches:
        raise ValueError("RL batch size must divide num_minibatches")
    minibatch_size = batch_size // minibatches
    if minibatch_size < 4096:
        raise ValueError(
            f"minibatch size {minibatch_size} is below the Phase D minimum of 4096"
        )

    dino = _phase4_dino_from_config(config, device)
    env = _make_benchmark_env(config, num_envs, "rgb+state")
    h5 = h5py.File(path, "r")
    action_low = torch.as_tensor(env.single_action_space.low, device=device, dtype=torch.float32)
    action_high = torch.as_tensor(env.single_action_space.high, device=device, dtype=torch.float32)
    condition_dim = frozen.frame_dim + frozen.goal_dim + 4
    agent = ResidualActorCritic(
        condition_dim,
        width=int(config.get("low_level_rl.residual_width", 256)),
        depth=int(config.get("low_level_rl.residual_depth", 2)),
        initial_logstd=float(config.get("low_level_rl.initial_logstd", -2.3)),
    ).to(device)
    optimizer = torch.optim.Adam(
        agent.parameters(), lr=float(config.get("low_level_rl.learning_rate", 1e-4)), eps=1e-5
    )
    recipe = {
        "method": "r1_residual_deterministic_local_mode_a",
        "dataset": str(path),
        "n_demo": n_demo,
        "seed": seed,
        "run_name": run_name,
        "num_envs": num_envs,
        "rollout_steps": rollout_steps,
        "horizon": horizon,
        "alpha": alpha,
        "terminal_weight": terminal_weight,
        "residual_penalty_weight": float(
            residual_penalty_weight
            if residual_penalty_weight is not None
            else config.get("low_level_rl.residual_penalty_weight", 0.01)
        ),
        "reward": "latent_progress_minus_terminal_distance_minus_residual_penalty",
        "disallowed_training_signals": [
            "mani_skill_reward",
            "task_success",
            "object_pose",
            "task_progress",
        ],
    }
    global_step = 0
    history: list[dict[str, Any]] = []
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

    rng = np.random.default_rng(seed + 123_000)
    current_obs: dict[str, Any]
    current_frames: np.ndarray
    current_z: np.ndarray
    goal_z: np.ndarray
    previous_action: np.ndarray
    local_step = 0
    current_batch = ""
    current_t = 0

    @torch.inference_mode()
    def reset_local_episode() -> None:
        nonlocal current_obs, current_frames, current_z, goal_z, previous_action
        nonlocal local_step, current_batch, current_t
        current_batch = str(rng.choice(batch_keys))
        group = h5[current_batch]
        current_t = int(rng.integers(0, max_steps - horizon + 1))
        current_obs, _info = env.reset(seed=int(group.attrs["batch_seed"]))
        for replay_step in range(current_t):
            replay_action = torch.from_numpy(
                np.asarray(group["executed_actions"][replay_step], dtype=np.float32)
            ).to(device)
            current_obs, _reward, _terminated, _truncated, _info = env.step(replay_action)
        current_frames = _phase4_frame_inputs(
            current_obs, dino, int(config.get("dino.batch_size", 64))
        )
        current_z = _encode_rerun_frames(frozen, current_frames, device)
        goal_frame = np.concatenate(
            [
                np.asarray(group["dino"][current_t + horizon], dtype=np.float32),
                np.asarray(group["proprio"][current_t + horizon], dtype=np.float32),
            ],
            axis=-1,
        )
        goal_z = _encode_rerun_frames(frozen, goal_frame, device)
        previous_action = frozen.action_norm.transform(
            np.asarray(group["previous_executed_actions"][current_t], dtype=np.float32)
        )
        local_step = 0

    @torch.inference_mode()
    def condition_and_base() -> tuple[torch.Tensor, torch.Tensor, np.ndarray]:
        distance = np.mean(np.square(current_z - goal_z), axis=-1).astype(np.float32)
        remaining = np.full(
            (num_envs, 1),
            max(horizon - local_step, 1) / horizon,
            dtype=np.float32,
        )
        condition_np = _low_condition_array(
            frozen.frame_norm.transform(current_frames),
            current_z,
            goal_z,
            previous_action,
            remaining,
            frozen.conditioning,
        )
        condition = torch.from_numpy(condition_np).to(device).float()
        normalized_base = frozen.low_model(condition)
        base_action = torch.from_numpy(
            frozen.action_norm.inverse(normalized_base.cpu().numpy().astype(np.float32))
        ).to(device)
        return condition, base_action, distance

    @torch.inference_mode()
    def local_step_env(action: torch.Tensor, previous_distance: np.ndarray, residual: torch.Tensor) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        nonlocal current_obs, current_frames, current_z, previous_action, local_step
        next_obs, _env_reward, _terminated, _truncated, info = env.step(action)
        next_frames = _phase4_frame_inputs(next_obs, dino, int(config.get("dino.batch_size", 64)))
        next_z = _encode_rerun_frames(frozen, next_frames, device)
        next_distance = np.mean(np.square(next_z - goal_z), axis=-1).astype(np.float32)
        segment_end = local_step == horizon - 1
        penalty_weight = float(recipe["residual_penalty_weight"])
        residual_penalty = penalty_weight * torch.mean(residual.square(), dim=-1).cpu().numpy()
        reward = previous_distance - next_distance - residual_penalty
        if segment_end:
            reward -= terminal_weight * next_distance
        current_obs = next_obs
        current_frames = next_frames
        current_z = next_z
        previous_action = frozen.action_norm.transform(action.cpu().numpy().astype(np.float32))
        local_step += 1
        done = np.full(num_envs, segment_end, dtype=np.bool_)
        metrics = {
            "next_distance": next_distance,
            "segment_end": segment_end,
            "residual_norm": torch.linalg.vector_norm(residual, dim=-1).cpu().numpy(),
            "success": _to_numpy(info.get("success", np.zeros(num_envs, dtype=np.bool_)))
            .reshape(-1)
            .astype(np.bool_),
        }
        if segment_end:
            reset_local_episode()
        return reward.astype(np.float32), done, metrics

    reset_local_episode()
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

    try:
        with trange(global_step, total_steps, initial=global_step, total=total_steps, desc=run_name) as progress:
            while global_step < total_steps:
                distance_values: list[float] = []
                terminal_distances: list[float] = []
                reward_values: list[float] = []
                residual_values: list[float] = []
                saturation_count = 0
                success_count = 0
                agent.eval()
                for step in range(rollout_steps):
                    condition, base_action, distance = condition_and_base()
                    condition_buf[step] = condition
                    done_buf[step] = next_done
                    with torch.no_grad():
                        raw_action, logprob, _entropy, value = agent.get_action_and_value(condition)
                    residual = alpha * torch.tanh(raw_action)
                    unclipped = base_action + residual
                    action = torch.clamp(unclipped, action_low, action_high)
                    raw_action_buf[step] = raw_action
                    logprob_buf[step] = logprob
                    value_buf[step] = value
                    reward, done, metrics = local_step_env(action, distance, residual)
                    reward_buf[step] = torch.from_numpy(reward).to(device)
                    next_done = torch.from_numpy(done.astype(np.float32)).to(device)
                    distance_values.extend(distance.tolist())
                    reward_values.extend(reward.tolist())
                    residual_values.extend(metrics["residual_norm"].tolist())
                    success_count += int(metrics["success"].sum())
                    if metrics["segment_end"]:
                        terminal_distances.extend(metrics["next_distance"].tolist())
                    saturation_count += int(torch.any(unclipped != action, dim=-1).sum().cpu())
                    global_step += num_envs
                    progress.update(min(num_envs, total_steps - progress.n))
                    if global_step >= total_steps and step == rollout_steps - 1:
                        break

                with torch.no_grad():
                    next_condition, _base, _distance = condition_and_base()
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
                flat_raw_action = raw_action_buf.flatten(0, 1)
                flat_logprob = logprob_buf.flatten()
                flat_advantages = advantages.flatten()
                flat_returns = returns.flatten()
                flat_values = value_buf.flatten()
                indices = np.arange(batch_size)
                clipfracs: list[float] = []
                policy_losses: list[float] = []
                value_losses: list[float] = []
                entropies: list[float] = []
                approx_kl = torch.tensor(0.0, device=device)
                agent.train()
                for _epoch in range(update_epochs):
                    np.random.shuffle(indices)
                    for start in range(0, batch_size, minibatch_size):
                        mb = indices[start : start + minibatch_size]
                        _new_action, new_logprob, entropy, new_value = agent.get_action_and_value(
                            flat_condition[mb],
                            flat_raw_action[mb],
                        )
                        logratio = new_logprob - flat_logprob[mb]
                        ratio = logratio.exp()
                        with torch.no_grad():
                            approx_kl = ((ratio - 1.0) - logratio).mean()
                            clipfracs.append(
                                float(((ratio - 1.0).abs() > clip_coef).float().mean().item())
                            )
                        mb_adv = flat_advantages[mb]
                        mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)
                        pg_loss = torch.max(
                            -mb_adv * ratio,
                            -mb_adv * torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef),
                        ).mean()
                        value_loss = 0.5 * (new_value - flat_returns[mb]).square().mean()
                        entropy_loss = entropy.mean()
                        loss = pg_loss - ent_coef * entropy_loss + value_coef * value_loss
                        optimizer.zero_grad(set_to_none=True)
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(agent.parameters(), max_grad_norm)
                        optimizer.step()
                        policy_losses.append(float(pg_loss.detach().cpu()))
                        value_losses.append(float(value_loss.detach().cpu()))
                        entropies.append(float(entropy_loss.detach().cpu()))

                explained_variance = float(
                    1.0
                    - torch.var(flat_returns - flat_values).item()
                    / max(torch.var(flat_returns).item(), 1e-8)
                )
                update_metrics = {
                    "global_step": int(global_step),
                    "mean_return": float(torch.mean(returns).detach().cpu()),
                    "mean_reward": float(np.mean(reward_values)),
                    "mean_distance": float(np.mean(distance_values)),
                    "mean_terminal_distance": float(np.mean(terminal_distances))
                    if terminal_distances
                    else None,
                    "goal_reach_rate_distance_improved_proxy": None,
                    "mean_residual_norm": float(np.mean(residual_values)),
                    "action_saturation_rate": float(saturation_count / batch_size),
                    "task_success_diagnostic_rate": float(success_count / batch_size),
                    "policy_loss": float(np.mean(policy_losses)),
                    "value_loss": float(np.mean(value_losses)),
                    "entropy": float(np.mean(entropies)),
                    "approx_kl": float(approx_kl.detach().cpu()),
                    "clip_fraction": float(np.mean(clipfracs)),
                    "explained_variance": explained_variance,
                    "batch_size": int(batch_size),
                    "minibatch_size": int(minibatch_size),
                }
                history.append(update_metrics)
                write_json(history_path, {"recipe": recipe, "history": history})
                torch.save(
                    {
                        "agent": agent.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "global_step": global_step,
                        "history": history,
                        "recipe": recipe,
                        "condition_dim": condition_dim,
                    },
                    latest,
                )
    finally:
        h5.close()
        env.close()
    return latest


@torch.inference_mode()
def evaluate_rl_rerun_local_r1(
    config: Config,
    checkpoint_path: Path,
    dataset_path: Path | None = None,
    n_demo: int = 1000,
    seed: int = 0,
    episodes: int = 4,
    output_path: Path | None = None,
) -> Path:
    from hcl_poc.incremental import _phase4_dino_from_config, _phase4_frame_inputs
    from hcl_poc.learned_interface import _low_condition_array
    from hcl_poc.low_level_rl import ResidualActorCritic, _load_frozen

    path = dataset_path or _vector_dataset_path(config)
    if not path.exists():
        raise FileNotFoundError(path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(checkpoint_path)
    if episodes <= 0:
        raise ValueError("episodes must be positive")

    device = default_device()
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    recipe = checkpoint["recipe"]
    rng = np.random.default_rng(seed)
    rerun_config = _rerun_base_config(config)
    frozen = _load_frozen(rerun_config, n_demo, seed, device)
    horizon = int(frozen.horizon_steps)
    dino = _phase4_dino_from_config(config, device)
    condition_dim = int(checkpoint["condition_dim"])
    agent = ResidualActorCritic(
        condition_dim,
        width=int(config.get("low_level_rl.residual_width", 256)),
        depth=int(config.get("low_level_rl.residual_depth", 2)),
        initial_logstd=float(config.get("low_level_rl.initial_logstd", -2.3)),
    ).to(device)
    agent.load_state_dict(checkpoint["agent"])
    agent.eval()
    alpha = float(recipe["alpha"])

    with h5py.File(path, "r") as h5:
        meta = h5["meta"].attrs
        num_envs = int(meta["num_envs"])
        max_steps = int(meta["max_steps"])
        batch_keys = sorted(key for key in h5.keys() if key.startswith("batch_"))
    env = _make_benchmark_env(config, num_envs, "rgb+state")
    action_low = torch.as_tensor(env.single_action_space.low, device=device, dtype=torch.float32)
    action_high = torch.as_tensor(env.single_action_space.high, device=device, dtype=torch.float32)

    initial_distances: list[np.ndarray] = []
    final_distances: list[np.ndarray] = []
    residual_norms: list[np.ndarray] = []
    saturation_rates: list[float] = []
    chosen_batches: list[str] = []
    chosen_timesteps: list[int] = []
    evaluation_keys = rng.choice(
        batch_keys,
        size=episodes,
        replace=episodes > len(batch_keys),
    )
    try:
        with h5py.File(path, "r") as h5:
            for episode_index in trange(episodes, desc="eval local R1"):
                key = str(evaluation_keys[episode_index])
                group = h5[key]
                t = int(rng.integers(0, max_steps - horizon + 1))
                obs, _info = env.reset(seed=int(group.attrs["batch_seed"]))
                for replay_step in range(t):
                    replay_action = torch.from_numpy(
                        np.asarray(group["executed_actions"][replay_step], dtype=np.float32)
                    ).to(device)
                    obs, _reward, _terminated, _truncated, _info = env.step(replay_action)
                goal_frame = np.concatenate(
                    [
                        np.asarray(group["dino"][t + horizon], dtype=np.float32),
                        np.asarray(group["proprio"][t + horizon], dtype=np.float32),
                    ],
                    axis=-1,
                )
                goal_z = _encode_rerun_frames(frozen, goal_frame, device)
                previous = frozen.action_norm.transform(
                    np.asarray(group["previous_executed_actions"][t], dtype=np.float32)
                )
                episode_initial_distance: np.ndarray | None = None
                saturation_count = 0
                for local_step in range(horizon):
                    frames = _phase4_frame_inputs(
                        obs, dino, int(config.get("dino.batch_size", 64))
                    )
                    current_z = _encode_rerun_frames(frozen, frames, device)
                    distance = np.mean(np.square(current_z - goal_z), axis=-1).astype(np.float32)
                    if episode_initial_distance is None:
                        episode_initial_distance = distance.copy()
                    remaining = np.full(
                        (num_envs, 1),
                        (horizon - local_step) / horizon,
                        dtype=np.float32,
                    )
                    condition = _low_condition_array(
                        frozen.frame_norm.transform(frames),
                        current_z,
                        goal_z,
                        previous,
                        remaining,
                        frozen.conditioning,
                    )
                    condition_t = torch.from_numpy(condition).to(device).float()
                    normalized_base = frozen.low_model(condition_t)
                    base_action = torch.from_numpy(
                        frozen.action_norm.inverse(
                            normalized_base.cpu().numpy().astype(np.float32)
                        )
                    ).to(device)
                    raw_action, _logprob, _entropy, _value = agent.get_action_and_value(
                        condition_t,
                        deterministic=True,
                    )
                    residual = alpha * torch.tanh(raw_action)
                    residual_norms.append(torch.linalg.vector_norm(residual, dim=-1).cpu().numpy())
                    unclipped = base_action + residual
                    action = torch.clamp(unclipped, action_low, action_high)
                    saturation_count += int(torch.any(unclipped != action, dim=-1).sum().cpu())
                    obs, _reward, _terminated, _truncated, _info = env.step(action)
                    previous = frozen.action_norm.transform(action.cpu().numpy())
                    if local_step == horizon - 1:
                        next_frames = _phase4_frame_inputs(
                            obs, dino, int(config.get("dino.batch_size", 64))
                        )
                        next_z = _encode_rerun_frames(frozen, next_frames, device)
                        final_distances.append(
                            np.mean(np.square(next_z - goal_z), axis=-1).astype(np.float32)
                        )
                if episode_initial_distance is None:
                    raise RuntimeError("Local R1 evaluation did not execute any steps")
                initial_distances.append(episode_initial_distance)
                saturation_rates.append(saturation_count / float(num_envs * horizon))
                chosen_batches.append(key)
                chosen_timesteps.append(t)
    finally:
        env.close()

    initial = np.concatenate(initial_distances)
    final = np.concatenate(final_distances)
    residual = np.concatenate(residual_norms)
    result = {
        "checkpoint": str(checkpoint_path),
        "dataset": str(path),
        "n_demo": n_demo,
        "seed": seed,
        "episodes": episodes,
        "num_envs": num_envs,
        "sampled_local_episodes": int(episodes * num_envs),
        "horizon": horizon,
        "chosen_batches": chosen_batches,
        "chosen_timestep_min": int(min(chosen_timesteps)),
        "chosen_timestep_max": int(max(chosen_timesteps)),
        "initial_distance_mean": float(np.mean(initial)),
        "final_distance_mean": float(np.mean(final)),
        "distance_reduction_mean": float(np.mean(initial - final)),
        "distance_reduction_fraction": float(np.mean(final < initial)),
        "mean_residual_norm": float(np.mean(residual)),
        "action_saturation_rate": float(np.mean(saturation_rates)),
        "recipe": recipe,
    }
    out_path = output_path or (
        _state_audit_result_dir(config)
        / "local_r1"
        / f"n{n_demo}"
        / f"seed{seed}"
        / str(recipe["run_name"])
        / f"eval_local_{episodes}.json"
    )
    write_json(out_path, result)
    return out_path
