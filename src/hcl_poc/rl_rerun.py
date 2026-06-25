from __future__ import annotations

import csv
import copy
import json
import subprocess
import time
from pathlib import Path
from typing import Any

import gymnasium as gym
import h5py
import mani_skill  # noqa: F401
import numpy as np
import torch
from torch import nn
from tqdm import trange

from hcl_poc.config import Config
from hcl_poc.features import batched, dino_from_config
from hcl_poc.flow import flow_matching_loss, sample_flow
from hcl_poc.models import FlowModel
from hcl_poc.rl import _rl_backend, _rl_paths, load_ppo_agent
from hcl_poc.utils import Timer, default_device, ensure_dir, write_json


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


def create_rl_rerun_local_eval_manifest(
    dataset_path: Path,
    output_path: Path,
    episodes: int,
    seed: int,
    horizon: int = 10,
) -> Path:
    if episodes <= 0:
        raise ValueError("episodes must be positive")
    with h5py.File(dataset_path, "r") as h5:
        max_steps = int(h5["meta"].attrs["max_steps"])
        num_envs = int(h5["meta"].attrs["num_envs"])
        batch_keys = sorted(key for key in h5.keys() if key.startswith("batch_"))
        if not batch_keys:
            raise ValueError(f"No vector batches found in {dataset_path}")
        rng = np.random.default_rng(seed)
        selected_keys = rng.choice(
            batch_keys,
            size=episodes,
            replace=episodes > len(batch_keys),
        )
        entries = []
        for key in selected_keys:
            batch_key = str(key)
            entries.append(
                {
                    "batch": batch_key,
                    "batch_seed": int(h5[batch_key].attrs["batch_seed"]),
                    "timestep": int(rng.integers(0, max_steps - horizon + 1)),
                }
            )
    manifest = {
        "dataset": str(dataset_path),
        "num_envs": num_envs,
        "horizon": horizon,
        "seed": seed,
        "sampled_local_episodes": episodes * num_envs,
        "entries": entries,
    }
    ensure_dir(output_path.parent)
    write_json(output_path, manifest)
    return output_path


def _local_eval_entries(
    dataset_path: Path,
    batch_keys: list[str],
    max_steps: int,
    episodes: int,
    seed: int,
    horizon: int,
    manifest_path: Path | None,
) -> list[dict[str, Any]]:
    if manifest_path is None:
        rng = np.random.default_rng(seed)
        selected_keys = rng.choice(
            batch_keys,
            size=episodes,
            replace=episodes > len(batch_keys),
        )
        return [
            {
                "batch": str(key),
                "timestep": int(rng.integers(0, max_steps - horizon + 1)),
            }
            for key in selected_keys
        ]

    manifest = json.loads(manifest_path.read_text())
    if Path(manifest["dataset"]).resolve() != dataset_path.resolve():
        raise ValueError(
            f"Manifest dataset {manifest['dataset']} does not match {dataset_path}"
        )
    if int(manifest["horizon"]) != horizon:
        raise ValueError(
            f"Manifest horizon {manifest['horizon']} does not match {horizon}"
        )
    entries = list(manifest["entries"])
    if episodes != len(entries):
        raise ValueError(
            f"Requested {episodes} episodes but manifest contains {len(entries)}"
        )
    for entry in entries:
        if entry["batch"] not in batch_keys:
            raise ValueError(f"Unknown manifest batch {entry['batch']}")
        timestep = int(entry["timestep"])
        if not 0 <= timestep <= max_steps - horizon:
            raise ValueError(f"Invalid manifest timestep {timestep}")
    return entries


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


def _reset_cuda_peak_memory() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()


def _rl_runtime_metrics(
    *,
    run_start_time: float,
    update_start_time: float,
    run_start_step: int,
    global_step: int,
    batch_size: int,
) -> dict[str, float | None]:
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    update_wall_time_s = time.perf_counter() - update_start_time
    wall_time_s = time.perf_counter() - run_start_time
    trained_steps = max(global_step - run_start_step, 0)
    peak_allocated_mib, peak_reserved_mib = _cuda_memory_mib()
    return {
        "update_wall_time_s": float(update_wall_time_s),
        "wall_time_s": float(wall_time_s),
        "update_samples_per_second": float(batch_size / max(update_wall_time_s, 1e-9)),
        "run_samples_per_second": float(trained_steps / max(wall_time_s, 1e-9)),
        "gpu_peak_memory_allocated_mib": peak_allocated_mib,
        "gpu_peak_memory_reserved_mib": peak_reserved_mib,
    }


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
    disturbed: bool = False,
    force: bool = False,
) -> Path:
    from hcl_poc.incremental import _phase4_dino_from_config, _phase4_frame_inputs
    from hcl_poc.incremental import _pre_rl_phase_d_schedule, PRE_RL_PHASE_D_PERTURBATIONS

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
    action_low_np = np.asarray(env.single_action_space.low, dtype=np.float32)
    action_high_np = np.asarray(env.single_action_space.high, dtype=np.float32)
    action_range_np = action_high_np - action_low_np

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
            meta.attrs["disturbed"] = bool(disturbed)
            meta.attrs["disturbance_family"] = (
                str(PRE_RL_PHASE_D_PERTURBATIONS) if disturbed else ""
            )
            meta.attrs.update(_git_metadata())
            for batch_index in trange(batches, desc="collect vector PPO batches"):
                batch_seed = int(seed_start + batch_index)
                obs, _info = env.reset(seed=batch_seed)
                disturbance_rng = np.random.default_rng(batch_seed + 10_000)
                schedules = (
                    _pre_rl_phase_d_schedule(disturbance_rng, num_envs, max_steps, 1, 1)
                    if disturbed
                    else [[] for _ in range(num_envs)]
                )

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
                bias_noise = np.zeros((num_envs, 3), dtype=np.float32)
                policy_action_history: list[np.ndarray] = []

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
                for step_index in range(max_steps):
                    state_t = torch.from_numpy(obs_states[-1]).to(device).float()
                    action_t, _logprob, _entropy, _value = teacher.get_action_and_value(
                        state_t,
                        deterministic=True,
                    )
                    raw_action = action_t.detach().cpu().numpy().astype(np.float32)
                    executed = torch.clamp(action_t, action_low, action_high)
                    executed_np = executed.detach().cpu().numpy().astype(np.float32)
                    policy_action_history.append(executed_np.copy())
                    if disturbed:
                        for env_index, events in enumerate(schedules):
                            event = events[0]
                            if not event["start"] <= step_index < event["end"]:
                                continue
                            kind = int(event["kind"])
                            if kind == 1:
                                bias_noise[env_index] = (
                                    0.7 * bias_noise[env_index]
                                    + 0.3
                                    * disturbance_rng.normal(0.0, 0.01, size=3).astype(np.float32)
                                    * action_range_np
                                )
                                executed_np[env_index] += (
                                    event["bias_fraction"]
                                    * action_range_np
                                    * event["bias_direction"]
                                    + bias_noise[env_index]
                                )
                            elif kind == 2:
                                executed_np[env_index] = previous[env_index]
                            elif kind == 3:
                                source_step = max(0, step_index - int(event["delay"]))
                                executed_np[env_index] = policy_action_history[source_step][env_index]
                            else:
                                executed_np[env_index] *= float(event["scale"])
                        executed_np = np.clip(executed_np, action_low_np, action_high_np)
                        executed = torch.from_numpy(executed_np).to(device).float()
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
                group.attrs["disturbed"] = bool(disturbed)
                if disturbed:
                    group.create_dataset(
                        "disturbance_kind",
                        data=np.asarray(
                            [int(events[0]["kind"]) for events in schedules],
                            dtype=np.int16,
                        ),
                        compression="gzip",
                    )
                    group.create_dataset(
                        "disturbance_start",
                        data=np.asarray(
                            [int(events[0]["start"]) for events in schedules],
                            dtype=np.int16,
                        ),
                        compression="gzip",
                    )
                    group.create_dataset(
                        "disturbance_end",
                        data=np.asarray(
                            [int(events[0]["end"]) for events in schedules],
                            dtype=np.int16,
                        ),
                        compression="gzip",
                    )
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


def _local_condition_dim(frozen: Any) -> int:
    goal_features = 2 * frozen.goal_dim if frozen.conditioning == "relation" else frozen.goal_dim
    return int(frozen.frame_dim + goal_features + 4)


def _residual_condition_dim(frozen: Any, mode: str) -> int:
    if mode == "full":
        return _local_condition_dim(frozen)
    if mode == "goal_delta":
        return int(3 * frozen.goal_dim + 4)
    raise ValueError(f"Unknown residual condition mode: {mode}")


def _residual_condition_array(
    *,
    mode: str,
    full_condition: np.ndarray,
    current_z: np.ndarray,
    goal_z: np.ndarray,
    previous_action: np.ndarray,
    remaining: np.ndarray,
) -> np.ndarray:
    if mode == "full":
        return full_condition.astype(np.float32)
    if mode == "goal_delta":
        return np.concatenate(
            [
                current_z,
                goal_z,
                goal_z - current_z,
                previous_action,
                remaining,
            ],
            axis=-1,
        ).astype(np.float32)
    raise ValueError(f"Unknown residual condition mode: {mode}")


def _residual_action_from_raw(
    base_action: torch.Tensor,
    raw_residual: torch.Tensor,
    alpha: float,
    action_low: torch.Tensor,
    action_high: torch.Tensor,
    mode: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    unit_residual = torch.tanh(raw_residual)
    if mode == "additive":
        residual = alpha * unit_residual
        unclipped = base_action + residual
    elif mode == "margin_scaled":
        base_anchor = torch.clamp(base_action, action_low, action_high)
        margin = torch.where(
            unit_residual >= 0.0,
            action_high - base_anchor,
            base_anchor - action_low,
        )
        residual = alpha * margin * unit_residual
        unclipped = base_anchor + residual
    else:
        raise ValueError(f"Unknown residual action mode: {mode}")
    action = torch.clamp(unclipped, action_low, action_high)
    return residual, unclipped, action


def _load_low_flow_base(path: Path, device: torch.device) -> tuple[FlowModel, dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(path)
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model = FlowModel(
        int(checkpoint["sample_dim"]),
        int(checkpoint["condition_dim"]),
        int(checkpoint["hidden_dim"]),
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    model.requires_grad_(False)
    return model, checkpoint


@torch.inference_mode()
def _low_flow_base_action(
    model: FlowModel,
    checkpoint: dict[str, Any],
    condition: torch.Tensor,
    frozen: Any,
) -> torch.Tensor:
    normalized = sample_flow(
        model,
        condition,
        steps=int(checkpoint["flow_steps"]),
        sample_dim=int(checkpoint["sample_dim"]),
        initial_noise=torch.zeros(
            (condition.shape[0], int(checkpoint["sample_dim"])),
            device=condition.device,
            dtype=condition.dtype,
        ),
    )
    return torch.from_numpy(
        frozen.action_norm.inverse(normalized.cpu().numpy().astype(np.float32))
    ).to(condition.device)


def train_rl_rerun_low_flow_base(
    config: Config,
    n_demo: int = 500,
    seed: int = 0,
    force: bool = False,
) -> Path:
    from hcl_poc.learned_interface import (
        _HeldGoalDataset,
        _load_phase6_train_episodes,
        _low_condition_array,
        prepare_learned_interface_episodes,
    )
    from hcl_poc.low_level_rl import _load_frozen
    from hcl_poc.utils import set_seed
    from hcl_poc.vae_scaling import VAE_CANDIDATE, vae_scaling_config

    artifact = ensure_dir(
        _rl_rerun_artifact_dir(config)
        / "local_r2"
        / f"n{n_demo}"
        / f"seed{seed}"
        / "low_flow_base"
    )
    checkpoint_path = artifact / "low_flow.pt"
    metrics_path = artifact / "low_flow_metrics.json"
    if checkpoint_path.exists() and not force:
        return checkpoint_path

    set_seed(seed + 170_000)
    device = default_device()
    rerun_config = _rerun_base_config(config)
    frozen = _load_frozen(rerun_config, n_demo, seed, device)
    horizon = int(frozen.horizon_steps)
    if horizon != 10:
        raise ValueError(f"Expected 10-step local horizon, got {horizon}")

    point_config = vae_scaling_config(rerun_config, n_demo)
    encoded_path = prepare_learned_interface_episodes(
        point_config,
        VAE_CANDIDATE,
        seed,
        force=False,
    )
    encoded = torch.load(encoded_path, map_location="cpu", weights_only=False)
    train_frames, validation_frames, data_metadata = _load_phase6_train_episodes(point_config)

    if int(encoded.get("format_version", 1)) != 2:
        raise ValueError("R2 low-flow training expects learned-interface format_version=2")
    if len(train_frames) != len(encoded["train_goals"]):
        raise ValueError("R2 low-flow train frame/goal episode mismatch")
    if len(validation_frames) != len(encoded["validation_goals"]):
        raise ValueError("R2 low-flow validation frame/goal episode mismatch")

    def combine(
        frame_episodes: list[dict[str, np.ndarray]],
        goal_episodes: list[np.ndarray],
    ) -> list[dict[str, np.ndarray]]:
        return [
            {
                "frames": frame_episode["frames"],
                "goals": goals,
                "actions": frame_episode["actions"],
            }
            for frame_episode, goals in zip(frame_episodes, goal_episodes, strict=True)
        ]

    train = combine(train_frames, encoded["train_goals"])
    validation = combine(validation_frames, encoded["validation_goals"])
    batch_size = int(config.get("learned_interface.policy.batch_size", 512))
    batches_per_epoch = int(config.get("learned_interface.policy.batches_per_epoch", 200))
    epochs = int(config.get("learned_interface.policy.epochs", 60))
    hidden_dim = int(config.get("learned_interface.policy.hidden_dim", 512))
    learning_rate = float(config.get("learned_interface.policy.lr", 3e-4))
    flow_steps = int(config.get("vae_scaling.flow_steps", 24))
    validation_samples = int(config.get("learned_interface.policy.validation_samples", 5000))
    train_loader = torch.utils.data.DataLoader(
        _HeldGoalDataset(
            train,
            frozen.frame_norm,
            frozen.goal_norm,
            frozen.action_norm,
            horizon,
            "low",
            batch_size * batches_per_epoch,
            frozen.conditioning,
        ),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    validation_loader = torch.utils.data.DataLoader(
        _HeldGoalDataset(
            validation,
            frozen.frame_norm,
            frozen.goal_norm,
            frozen.action_norm,
            horizon,
            "low",
            validation_samples,
            frozen.conditioning,
        ),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    condition_dim = _local_condition_dim(frozen)
    model = FlowModel(3, condition_dim, hidden_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    history: list[dict[str, Any]] = []
    best_state: dict[str, torch.Tensor] | None = None
    best_mae = float("inf")
    best_epoch = 0
    timer = Timer()

    def validation_action_mae() -> float:
        predictions = []
        targets = []
        model.eval()
        with torch.inference_mode():
            for condition, target in validation_loader:
                condition = condition.to(device, non_blocking=True).float()
                normalized = sample_flow(
                    model,
                    condition,
                    flow_steps,
                    3,
                    initial_noise=torch.zeros((len(condition), 3), device=device),
                )
                predictions.append(
                    frozen.action_norm.inverse(
                        normalized.cpu().numpy().astype(np.float32)
                    )
                )
                targets.append(
                    frozen.action_norm.inverse(target.numpy().astype(np.float32))
                )
        return float(np.mean(np.abs(np.concatenate(predictions) - np.concatenate(targets))))

    for epoch in trange(1, epochs + 1, desc=f"train R2 low flow n={n_demo} seed={seed}"):
        model.train()
        train_loss = 0.0
        for condition, target in train_loader:
            condition = condition.to(device, non_blocking=True).float()
            target = target.to(device, non_blocking=True).float()
            loss = flow_matching_loss(model, target, condition)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            train_loss += float(loss.detach().cpu())
        action_mae = validation_action_mae()
        history.append(
            {
                "epoch": epoch,
                "train_flow_loss": train_loss / batches_per_epoch,
                "validation_zero_noise_action_mae": action_mae,
            }
        )
        if action_mae < best_mae:
            best_mae = action_mae
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())

    if best_state is None:
        raise RuntimeError("R2 low-flow base training produced no checkpoint")
    payload = {
        "method": "r2_low_flow_base",
        "n_demo": n_demo,
        "seed": seed,
        "candidate": "vae512_w2048_b1e6",
        "sample_dim": 3,
        "condition_dim": condition_dim,
        "hidden_dim": hidden_dim,
        "flow_steps": flow_steps,
        "horizon": horizon,
        "conditioning": frozen.conditioning,
        "model": best_state,
        "best_epoch": best_epoch,
        "validation_zero_noise_action_mae": best_mae,
        "history": history,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "batches_per_epoch": batches_per_epoch,
        "epochs": epochs,
        "hierarchy_checkpoint": str(frozen.checkpoint_path),
        "encoded_episodes": str(encoded_path),
        "data": data_metadata,
        "elapsed_s": timer.elapsed(),
    }
    torch.save(payload, checkpoint_path)
    write_json(
        metrics_path,
        {key: value for key, value in payload.items() if key != "model"},
    )
    return checkpoint_path


@torch.inference_mode()
def audit_rl_rerun_local_mode_a(
    config: Config,
    dataset_path: Path | None = None,
    n_demo: int = 1000,
    seed: int = 0,
    episodes: int = 4,
    manifest_path: Path | None = None,
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
    evaluation_entries = _local_eval_entries(
        path,
        batch_keys,
        max_steps,
        episodes,
        seed,
        horizon,
        manifest_path,
    )
    env = _make_benchmark_env(config, num_envs, "rgb+state")
    action_low = torch.as_tensor(env.single_action_space.low, device=device, dtype=torch.float32)
    action_high = torch.as_tensor(env.single_action_space.high, device=device, dtype=torch.float32)

    initial_distances: list[np.ndarray] = []
    final_distances: list[np.ndarray] = []
    terminal_rewards: list[np.ndarray] = []
    progress_rewards: list[np.ndarray] = []
    final_env_rewards: list[np.ndarray] = []
    max_env_rewards: list[np.ndarray] = []
    mean_env_rewards: list[np.ndarray] = []
    saturation_rates: list[float] = []
    task_success_once = np.zeros((episodes, num_envs), dtype=np.bool_)
    chosen_batches: list[str] = []
    chosen_timesteps: list[int] = []
    try:
        with h5py.File(path, "r") as h5:
            for episode in trange(episodes, desc="audit local Mode-A"):
                entry = evaluation_entries[episode]
                key = str(entry["batch"])
                group = h5[key]
                t = int(entry["timestep"])
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
                total_env_reward = np.zeros(num_envs, dtype=np.float32)
                episode_max_env_reward = np.full(num_envs, -np.inf, dtype=np.float32)
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
                    obs, step_reward, _terminated, _truncated, info = env.step(action)
                    step_reward_np = _to_numpy(step_reward).reshape(-1).astype(np.float32)
                    total_env_reward += step_reward_np
                    episode_max_env_reward = np.maximum(
                        episode_max_env_reward,
                        step_reward_np,
                    )
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
                        final_env_rewards.append(step_reward_np)
                        max_env_rewards.append(episode_max_env_reward)
                        mean_env_rewards.append(total_env_reward / horizon)
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
    final_env_reward = np.concatenate(final_env_rewards)
    max_env_reward = np.concatenate(max_env_rewards)
    mean_env_reward = np.concatenate(mean_env_rewards)
    result = {
        "dataset": str(path),
        "n_demo": n_demo,
        "seed": seed,
        "evaluation_manifest": str(manifest_path) if manifest_path else None,
        "evaluation_entries": evaluation_entries,
        "episodes": episodes,
        "num_envs": num_envs,
        "sampled_local_episodes": int(episodes * num_envs),
        "horizon": horizon,
        "chosen_batches": chosen_batches,
        "chosen_timesteps": chosen_timesteps,
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
        "final_env_reward_mean": float(np.mean(final_env_reward)),
        "max_env_reward_mean": float(np.mean(max_env_reward)),
        "mean_env_reward_mean": float(np.mean(mean_env_reward)),
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
    learning_rate: float | None = None,
    num_minibatches: int | None = None,
    checkpoint_every_updates: int = 5,
    initial_logstd: float | None = None,
    force: bool = False,
    base_policy: str = "deterministic",
    flow_checkpoint_path: Path | None = None,
    family_dir: str = "local_r1",
    method_name: str = "r1_residual_deterministic_local_mode_a",
    residual_condition_mode: str = "full",
    residual_action_mode: str = "additive",
) -> Path:
    from hcl_poc.incremental import _phase4_dino_from_config, _phase4_frame_inputs
    from hcl_poc.learned_interface import _low_condition_array
    from hcl_poc.low_level_rl import DirectLowActorCritic, ResidualActorCritic, _load_frozen
    from hcl_poc.utils import set_seed

    path = dataset_path or _vector_dataset_path(config)
    if not path.exists():
        raise FileNotFoundError(path)
    if total_steps <= 0:
        raise ValueError("total_steps must be positive")
    if base_policy not in {"deterministic", "flow"}:
        raise ValueError(f"Unknown residual base policy: {base_policy}")
    if residual_condition_mode not in {"full", "goal_delta"}:
        raise ValueError("residual_condition_mode must be 'full' or 'goal_delta'")
    if residual_action_mode not in {"additive", "margin_scaled"}:
        raise ValueError("residual_action_mode must be 'additive' or 'margin_scaled'")

    artifact = ensure_dir(
        _rl_rerun_artifact_dir(config)
        / family_dir
        / f"n{n_demo}"
        / f"seed{seed}"
        / run_name
    )
    result_dir = ensure_dir(
        _state_audit_result_dir(config)
        / family_dir
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
    if checkpoint_every_updates <= 0:
        raise ValueError("checkpoint_every_updates must be positive")
    flow_model: FlowModel | None = None
    flow_checkpoint: dict[str, Any] | None = None
    resolved_flow_checkpoint_path: Path | None = None
    if base_policy == "flow":
        resolved_flow_checkpoint_path = flow_checkpoint_path or train_rl_rerun_low_flow_base(
            config,
            n_demo=n_demo,
            seed=seed,
            force=False,
        )
        flow_model, flow_checkpoint = _load_low_flow_base(
            resolved_flow_checkpoint_path,
            device,
        )
        if int(flow_checkpoint["condition_dim"]) != _local_condition_dim(frozen):
            raise ValueError("R2 low-flow base condition dimension does not match frozen hierarchy")

    with h5py.File(path, "r") as h5_meta:
        meta = h5_meta["meta"].attrs
        num_envs = int(meta["num_envs"])
        max_steps = int(meta["max_steps"])
        batch_keys = sorted(key for key in h5_meta.keys() if key.startswith("batch_"))
    rollout_steps = horizon
    batch_size = num_envs * rollout_steps
    minibatches = int(
        num_minibatches
        if num_minibatches is not None
        else config.get("low_level_rl.num_minibatches", 8)
    )
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
    condition_dim = _residual_condition_dim(frozen, residual_condition_mode)
    agent = ResidualActorCritic(
        condition_dim,
        width=int(config.get("low_level_rl.residual_width", 256)),
        depth=int(config.get("low_level_rl.residual_depth", 2)),
        initial_logstd=float(
            initial_logstd
            if initial_logstd is not None
            else config.get("low_level_rl.initial_logstd", -2.3)
        ),
    ).to(device)
    resolved_learning_rate = float(
        learning_rate
        if learning_rate is not None
        else config.get("low_level_rl.learning_rate", 1e-4)
    )
    gamma = float(config.get("low_level_rl.gamma", 0.99))
    gae_lambda = float(config.get("low_level_rl.gae_lambda", 0.95))
    clip_coef = float(config.get("low_level_rl.clip_coef", 0.2))
    ent_coef = float(config.get("low_level_rl.entropy_coef", 0.0))
    value_coef = float(config.get("low_level_rl.value_coef", 1.0))
    update_epochs = int(config.get("low_level_rl.update_epochs", 4))
    max_grad_norm = float(config.get("low_level_rl.max_grad_norm", 1.0))
    optimizer = torch.optim.Adam(agent.parameters(), lr=resolved_learning_rate, eps=1e-5)
    recipe = {
        "method": method_name,
        "dataset": str(path),
        "n_demo": n_demo,
        "seed": seed,
        "run_name": run_name,
        "family_dir": family_dir,
        "num_envs": num_envs,
        "rollout_steps": rollout_steps,
        "horizon": horizon,
        "base_policy": base_policy,
        "flow_checkpoint": (
            str(resolved_flow_checkpoint_path) if resolved_flow_checkpoint_path else None
        ),
        "alpha": alpha,
        "terminal_weight": terminal_weight,
        "learning_rate": resolved_learning_rate,
        "minibatches": minibatches,
        "minibatch_size": minibatch_size,
        "update_epochs": update_epochs,
        "gamma": gamma,
        "gae_lambda": gae_lambda,
        "clip_coef": clip_coef,
        "entropy_coef": ent_coef,
        "value_coef": value_coef,
        "max_grad_norm": max_grad_norm,
        "actor_critic_width": int(config.get("low_level_rl.residual_width", 256)),
        "actor_critic_depth": int(config.get("low_level_rl.residual_depth", 2)),
        "initial_logstd": float(
            initial_logstd
            if initial_logstd is not None
            else config.get("low_level_rl.initial_logstd", -2.3)
        ),
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
    if residual_condition_mode != "full":
        recipe["residual_condition_mode"] = residual_condition_mode
    if residual_action_mode != "additive":
        recipe["residual_action_mode"] = residual_action_mode
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
        full_condition_np = _low_condition_array(
            frozen.frame_norm.transform(current_frames),
            current_z,
            goal_z,
            previous_action,
            remaining,
            frozen.conditioning,
        )
        full_condition = torch.from_numpy(full_condition_np).to(device).float()
        if base_policy == "deterministic":
            normalized_base = frozen.low_model(full_condition)
            base_action = torch.from_numpy(
                frozen.action_norm.inverse(normalized_base.cpu().numpy().astype(np.float32))
            ).to(device)
        else:
            if flow_model is None or flow_checkpoint is None:
                raise RuntimeError("R2 flow base was not loaded")
            base_action = _low_flow_base_action(flow_model, flow_checkpoint, full_condition, frozen)
        residual_condition_np = _residual_condition_array(
            mode=residual_condition_mode,
            full_condition=full_condition_np,
            current_z=current_z,
            goal_z=goal_z,
            previous_action=previous_action,
            remaining=remaining,
        )
        residual_condition = torch.from_numpy(residual_condition_np).to(device).float()
        return residual_condition, base_action, distance

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
    run_start_step = int(global_step)
    run_start_time = time.perf_counter()
    _reset_cuda_peak_memory()
    try:
        with trange(global_step, total_steps, initial=global_step, total=total_steps, desc=run_name) as progress:
            while global_step < total_steps:
                update_start_time = time.perf_counter()
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
                    residual, unclipped, action = _residual_action_from_raw(
                        base_action,
                        raw_action,
                        alpha,
                        action_low,
                        action_high,
                        residual_action_mode,
                    )
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
                update_metrics.update(
                    _rl_runtime_metrics(
                        run_start_time=run_start_time,
                        update_start_time=update_start_time,
                        run_start_step=run_start_step,
                        global_step=global_step,
                        batch_size=batch_size,
                    )
                )
                history.append(update_metrics)
                write_json(history_path, {"recipe": recipe, "history": history})
                checkpoint_state = {
                    "agent": agent.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "global_step": global_step,
                    "history": history,
                    "recipe": recipe,
                    "condition_dim": condition_dim,
                }
                torch.save(checkpoint_state, latest)
                if (
                    len(history) % checkpoint_every_updates == 0
                    or global_step >= total_steps
                ):
                    checkpoint_dir = ensure_dir(artifact / "checkpoints")
                    torch.save(
                        checkpoint_state,
                        checkpoint_dir / f"step_{global_step:09d}.pt",
                    )
    finally:
        h5.close()
        env.close()
    return latest


def train_rl_rerun_local_r2(
    config: Config,
    dataset_path: Path | None = None,
    n_demo: int = 500,
    seed: int = 0,
    run_name: str = "local_r2_flow_residual",
    total_steps: int = 32_768,
    alpha: float = 0.1,
    terminal_weight: float = 1.0,
    residual_penalty_weight: float | None = None,
    learning_rate: float | None = None,
    num_minibatches: int | None = None,
    checkpoint_every_updates: int = 5,
    initial_logstd: float | None = None,
    flow_checkpoint_path: Path | None = None,
    force: bool = False,
    residual_condition_mode: str = "full",
    residual_action_mode: str = "additive",
) -> Path:
    return train_rl_rerun_local_r1(
        config,
        dataset_path=dataset_path,
        n_demo=n_demo,
        seed=seed,
        run_name=run_name,
        total_steps=total_steps,
        alpha=alpha,
        terminal_weight=terminal_weight,
        residual_penalty_weight=residual_penalty_weight,
        learning_rate=learning_rate,
        num_minibatches=num_minibatches,
        checkpoint_every_updates=checkpoint_every_updates,
        initial_logstd=initial_logstd,
        force=force,
        base_policy="flow",
        flow_checkpoint_path=flow_checkpoint_path,
        family_dir="local_r2",
        method_name="r2_residual_flow_local_mode_a",
        residual_condition_mode=residual_condition_mode,
        residual_action_mode=residual_action_mode,
    )


def train_rl_rerun_local_r3(
    config: Config,
    dataset_path: Path | None = None,
    n_demo: int = 500,
    seed: int = 0,
    run_name: str = "local_r3_direct_last_layer",
    total_steps: int = 32_768,
    bc_weight: float = 1.0,
    terminal_weight: float = 1.0,
    dense_progress_weight: float = 1.0,
    reward_mode: str = "progress",
    learning_rate: float | None = None,
    num_minibatches: int | None = None,
    initial_logstd: float | None = None,
    checkpoint_every_updates: int = 5,
    goal_sensitivity_weight: float = 0.0,
    goal_sensitivity_margin: float = 0.05,
    force: bool = False,
) -> Path:
    from hcl_poc.incremental import _phase4_dino_from_config, _phase4_frame_inputs
    from hcl_poc.learned_interface import _low_condition_array
    from hcl_poc.low_level_rl import DirectLowActorCritic, _load_frozen
    from hcl_poc.utils import set_seed

    path = dataset_path or _vector_dataset_path(config)
    if not path.exists():
        raise FileNotFoundError(path)
    if total_steps <= 0:
        raise ValueError("total_steps must be positive")
    if checkpoint_every_updates <= 0:
        raise ValueError("checkpoint_every_updates must be positive")
    if goal_sensitivity_weight < 0:
        raise ValueError("goal_sensitivity_weight must be non-negative")
    if goal_sensitivity_margin <= 0:
        raise ValueError("goal_sensitivity_margin must be positive")
    if dense_progress_weight < 0.0:
        raise ValueError("dense_progress_weight must be non-negative")
    if reward_mode not in {"progress", "paired"}:
        raise ValueError("reward_mode must be one of {'progress', 'paired'}")

    artifact = ensure_dir(
        _rl_rerun_artifact_dir(config)
        / "local_r3"
        / f"n{n_demo}"
        / f"seed{seed}"
        / run_name
    )
    result_dir = ensure_dir(
        _state_audit_result_dir(config)
        / "local_r3"
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
    set_seed(seed + 190_000)
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
    minibatches = int(
        num_minibatches
        if num_minibatches is not None
        else config.get("low_level_rl.num_minibatches", 8)
    )
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
    condition_dim = _local_condition_dim(frozen)
    resolved_initial_logstd = float(
        initial_logstd
        if initial_logstd is not None
        else config.get("low_level_rl.direct_initial_logstd", -4.0)
    )
    agent = DirectLowActorCritic(
        frozen.low_model,
        frozen.action_norm.mean,
        frozen.action_norm.std,
        condition_dim,
        width=int(config.get("low_level_rl.residual_width", 256)),
        depth=int(config.get("low_level_rl.residual_depth", 2)),
        initial_logstd=resolved_initial_logstd,
    ).to(device)
    trainable = [parameter for parameter in agent.parameters() if parameter.requires_grad]
    resolved_learning_rate = float(
        learning_rate
        if learning_rate is not None
        else config.get("low_level_rl.direct_learning_rate", 3e-5)
    )
    gamma = float(config.get("low_level_rl.gamma", 0.99))
    gae_lambda = float(config.get("low_level_rl.gae_lambda", 0.95))
    clip_coef = float(config.get("low_level_rl.clip_coef", 0.2))
    ent_coef = float(config.get("low_level_rl.entropy_coef", 0.0))
    value_coef = float(config.get("low_level_rl.value_coef", 1.0))
    update_epochs = int(config.get("low_level_rl.update_epochs", 4))
    max_grad_norm = float(config.get("low_level_rl.max_grad_norm", 1.0))
    optimizer = torch.optim.Adam(trainable, lr=resolved_learning_rate, eps=1e-5)
    recipe = {
        "method": "r3_direct_last_layer_local_mode_a",
        "dataset": str(path),
        "n_demo": n_demo,
        "seed": seed,
        "run_name": run_name,
        "family_dir": "local_r3",
        "num_envs": num_envs,
        "rollout_steps": rollout_steps,
        "horizon": horizon,
        "bc_weight": bc_weight,
        "terminal_weight": terminal_weight,
        "learning_rate": resolved_learning_rate,
        "minibatches": minibatches,
        "minibatch_size": minibatch_size,
        "update_epochs": update_epochs,
        "gamma": gamma,
        "gae_lambda": gae_lambda,
        "clip_coef": clip_coef,
        "entropy_coef": ent_coef,
        "value_coef": value_coef,
        "max_grad_norm": max_grad_norm,
        "actor_critic_width": int(config.get("low_level_rl.residual_width", 256)),
        "actor_critic_depth": int(config.get("low_level_rl.residual_depth", 2)),
        "initial_logstd": resolved_initial_logstd,
        "trainable_scope": "low_policy_final_layer_plus_logstd_and_critic",
        "reward": (
            "latent_progress_minus_terminal_distance_plus_bc_regularization_in_loss"
            if reward_mode == "progress"
            else (
                "latent_progress_plus_cached_base_terminal_improvement"
                "_plus_bc_regularization_in_loss"
            )
        ),
        "disallowed_training_signals": [
            "mani_skill_reward",
            "task_success",
            "object_pose",
            "task_progress",
        ],
    }
    if goal_sensitivity_weight > 0:
        recipe["goal_sensitivity_weight"] = goal_sensitivity_weight
        recipe["goal_sensitivity_margin"] = goal_sensitivity_margin
        recipe["goal_sensitivity_loss"] = (
            "in-batch valid-goal swap hinge on deterministic mean action"
        )
    if reward_mode == "paired":
        recipe["reward_mode"] = reward_mode
    if dense_progress_weight != 1.0:
        recipe["dense_progress_weight"] = dense_progress_weight
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

    rng = np.random.default_rng(seed + 193_000)
    current_obs: dict[str, Any]
    current_frames: np.ndarray
    current_z: np.ndarray
    goal_z: np.ndarray
    previous_action: np.ndarray
    base_terminal_distance: np.ndarray
    base_terminal_distance = np.full(1, np.nan, dtype=np.float32)
    local_step = 0

    @torch.inference_mode()
    def load_local_start(group: h5py.Group, current_t: int) -> None:
        nonlocal current_obs, current_frames, current_z, goal_z, previous_action, local_step
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
    def frozen_base_terminal_distance() -> np.ndarray:
        base_obs = current_obs
        base_frames = current_frames
        base_z = current_z
        base_previous = previous_action.copy()
        for base_step in range(horizon):
            remaining = np.full(
                (num_envs, 1),
                (horizon - base_step) / horizon,
                dtype=np.float32,
            )
            condition_np = _low_condition_array(
                frozen.frame_norm.transform(base_frames),
                base_z,
                goal_z,
                base_previous,
                remaining,
                frozen.conditioning,
            )
            condition = torch.from_numpy(condition_np).to(device).float()
            normalized_base = frozen.low_model(condition)
            unclipped = torch.from_numpy(
                frozen.action_norm.inverse(
                    normalized_base.cpu().numpy().astype(np.float32)
                )
            ).to(device)
            action = torch.clamp(unclipped, action_low, action_high)
            base_obs, _reward, _terminated, _truncated, _info = env.step(action)
            base_frames = _phase4_frame_inputs(
                base_obs,
                dino,
                int(config.get("dino.batch_size", 64)),
            )
            base_z = _encode_rerun_frames(frozen, base_frames, device)
            base_previous = frozen.action_norm.transform(action.cpu().numpy())
        return np.mean(np.square(base_z - goal_z), axis=-1).astype(np.float32)

    @torch.inference_mode()
    def reset_local_episode() -> None:
        nonlocal base_terminal_distance
        group = h5[str(rng.choice(batch_keys))]
        current_t = int(rng.integers(0, max_steps - horizon + 1))
        load_local_start(group, current_t)
        if reward_mode == "paired":
            base_terminal_distance = frozen_base_terminal_distance()
            load_local_start(group, current_t)
        else:
            base_terminal_distance = np.full(num_envs, np.nan, dtype=np.float32)

    @torch.inference_mode()
    def condition_and_bc() -> tuple[torch.Tensor, torch.Tensor, np.ndarray]:
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
    def local_step_env(action: torch.Tensor, previous_distance: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        nonlocal current_obs, current_frames, current_z, previous_action, local_step
        next_obs, _env_reward, _terminated, _truncated, info = env.step(action)
        next_frames = _phase4_frame_inputs(next_obs, dino, int(config.get("dino.batch_size", 64)))
        next_z = _encode_rerun_frames(frozen, next_frames, device)
        next_distance = np.mean(np.square(next_z - goal_z), axis=-1).astype(np.float32)
        segment_end = local_step == horizon - 1
        reward = dense_progress_weight * (previous_distance - next_distance)
        if segment_end:
            if reward_mode == "paired":
                reward += terminal_weight * (base_terminal_distance - next_distance)
            else:
                reward -= terminal_weight * next_distance
        current_obs = next_obs
        current_frames = next_frames
        current_z = next_z
        previous_action = frozen.action_norm.transform(action.cpu().numpy().astype(np.float32))
        local_step += 1
        done = np.full(num_envs, segment_end, dtype=np.bool_)
        metrics = {
            "next_distance": next_distance,
            "base_terminal_distance": base_terminal_distance.copy()
            if segment_end and reward_mode == "paired"
            else None,
            "segment_end": segment_end,
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
    base_action_buf = torch.zeros((rollout_steps, num_envs, 3), device=device)
    logprob_buf = torch.zeros((rollout_steps, num_envs), device=device)
    reward_buf = torch.zeros((rollout_steps, num_envs), device=device)
    done_buf = torch.zeros((rollout_steps, num_envs), device=device)
    value_buf = torch.zeros((rollout_steps, num_envs), device=device)
    next_done = torch.zeros(num_envs, device=device)
    run_start_step = int(global_step)
    run_start_time = time.perf_counter()
    _reset_cuda_peak_memory()
    try:
        with trange(global_step, total_steps, initial=global_step, total=total_steps, desc=run_name) as progress:
            while global_step < total_steps:
                update_start_time = time.perf_counter()
                distance_values: list[float] = []
                terminal_distances: list[float] = []
                base_terminal_distances: list[float] = []
                paired_improvements: list[float] = []
                action_delta_values: list[float] = []
                reward_values: list[float] = []
                saturation_count = 0
                success_count = 0
                agent.eval()
                for step in range(rollout_steps):
                    condition, base_action, distance = condition_and_bc()
                    condition_buf[step] = condition
                    base_action_buf[step] = base_action
                    done_buf[step] = next_done
                    with torch.no_grad():
                        raw_action, logprob, _entropy, value = agent.get_action_and_value(condition)
                    action = torch.clamp(raw_action, action_low, action_high)
                    raw_action_buf[step] = raw_action
                    logprob_buf[step] = logprob
                    value_buf[step] = value
                    reward, done, metrics = local_step_env(action, distance)
                    reward_buf[step] = torch.from_numpy(reward).to(device)
                    next_done = torch.from_numpy(done.astype(np.float32)).to(device)
                    distance_values.extend(distance.tolist())
                    reward_values.extend(reward.tolist())
                    action_delta_values.extend(
                        torch.linalg.vector_norm(action - base_action, dim=-1).cpu().tolist()
                    )
                    success_count += int(metrics["success"].sum())
                    if metrics["segment_end"]:
                        terminal_distances.extend(metrics["next_distance"].tolist())
                        if reward_mode == "paired":
                            cached_base = metrics["base_terminal_distance"]
                            if cached_base is None:
                                raise RuntimeError(
                                    "Paired reward did not report base distance"
                                )
                            improvement = cached_base - metrics["next_distance"]
                            base_terminal_distances.extend(cached_base.tolist())
                            paired_improvements.extend(improvement.tolist())
                    saturation_count += int(torch.any(raw_action != action, dim=-1).sum().cpu())
                    global_step += num_envs
                    progress.update(min(num_envs, total_steps - progress.n))
                    if global_step >= total_steps and step == rollout_steps - 1:
                        break

                with torch.no_grad():
                    next_condition, _base, _distance = condition_and_bc()
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
                flat_base_action = base_action_buf.flatten(0, 1)
                flat_logprob = logprob_buf.flatten()
                flat_advantages = advantages.flatten()
                flat_returns = returns.flatten()
                flat_values = value_buf.flatten()
                indices = np.arange(batch_size)
                clipfracs: list[float] = []
                policy_losses: list[float] = []
                value_losses: list[float] = []
                entropies: list[float] = []
                bc_losses: list[float] = []
                sensitivity_losses: list[float] = []
                sensitivity_values: list[float] = []
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
                        mean_action = agent.mean_action(flat_condition[mb])
                        bc_loss = torch.mean((mean_action - flat_base_action[mb]).square())
                        sensitivity_loss = torch.zeros((), device=device)
                        if goal_sensitivity_weight > 0:
                            swapped_condition = flat_condition[mb].clone()
                            if frozen.conditioning in {"concat", "delta", "film"}:
                                goal_start = frozen.frame_dim
                                goal_stop = goal_start + frozen.goal_dim
                                permutation = torch.randperm(
                                    swapped_condition.shape[0], device=device
                                )
                                swapped_condition[:, goal_start:goal_stop] = swapped_condition[
                                    permutation, goal_start:goal_stop
                                ]
                            elif frozen.conditioning == "relation":
                                future_start = frozen.frame_dim + frozen.goal_dim
                                future_stop = future_start + frozen.goal_dim
                                permutation = torch.randperm(
                                    swapped_condition.shape[0], device=device
                                )
                                swapped_condition[:, future_start:future_stop] = (
                                    swapped_condition[permutation, future_start:future_stop]
                                )
                            else:
                                raise ValueError(
                                    f"Unknown goal conditioning: {frozen.conditioning}"
                                )
                            swapped_mean_action = agent.mean_action(swapped_condition)
                            action_sensitivity = torch.linalg.vector_norm(
                                mean_action - swapped_mean_action, dim=-1
                            )
                            sensitivity_loss = torch.mean(
                                torch.clamp(
                                    goal_sensitivity_margin - action_sensitivity,
                                    min=0.0,
                                ).square()
                            )
                            sensitivity_losses.append(
                                float(sensitivity_loss.detach().cpu())
                            )
                            sensitivity_values.append(
                                float(action_sensitivity.detach().mean().cpu())
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
                        loss = (
                            pg_loss
                            - ent_coef * entropy_loss
                            + value_coef * value_loss
                            + bc_weight * bc_loss
                            + goal_sensitivity_weight * sensitivity_loss
                        )
                        optimizer.zero_grad(set_to_none=True)
                        loss.backward()
                        nn.utils.clip_grad_norm_(trainable, max_grad_norm)
                        optimizer.step()
                        policy_losses.append(float(pg_loss.detach().cpu()))
                        value_losses.append(float(value_loss.detach().cpu()))
                        entropies.append(float(entropy_loss.detach().cpu()))
                        bc_losses.append(float(bc_loss.detach().cpu()))

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
                    "mean_base_terminal_distance": float(np.mean(base_terminal_distances))
                    if base_terminal_distances
                    else None,
                    "mean_paired_improvement": float(np.mean(paired_improvements))
                    if paired_improvements
                    else None,
                    "fraction_paired_improved": float(
                        np.mean(np.asarray(paired_improvements) > 0.0)
                    )
                    if paired_improvements
                    else None,
                    "mean_action_delta_l2": float(np.mean(action_delta_values)),
                    "action_saturation_rate": float(saturation_count / batch_size),
                    "task_success_diagnostic_rate": float(success_count / batch_size),
                    "policy_loss": float(np.mean(policy_losses)),
                    "value_loss": float(np.mean(value_losses)),
                    "bc_loss": float(np.mean(bc_losses)),
                    "goal_sensitivity_loss": (
                        float(np.mean(sensitivity_losses)) if sensitivity_losses else None
                    ),
                    "goal_swap_action_sensitivity_l2": (
                        float(np.mean(sensitivity_values)) if sensitivity_values else None
                    ),
                    "entropy": float(np.mean(entropies)),
                    "approx_kl": float(approx_kl.detach().cpu()),
                    "clip_fraction": float(np.mean(clipfracs)),
                    "explained_variance": explained_variance,
                    "batch_size": int(batch_size),
                    "minibatch_size": int(minibatch_size),
                }
                update_metrics.update(
                    _rl_runtime_metrics(
                        run_start_time=run_start_time,
                        update_start_time=update_start_time,
                        run_start_step=run_start_step,
                        global_step=global_step,
                        batch_size=batch_size,
                    )
                )
                history.append(update_metrics)
                write_json(history_path, {"recipe": recipe, "history": history})
                checkpoint_state = {
                    "agent": agent.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "global_step": global_step,
                    "history": history,
                    "recipe": recipe,
                    "condition_dim": condition_dim,
                }
                torch.save(checkpoint_state, latest)
                if len(history) % checkpoint_every_updates == 0 or global_step >= total_steps:
                    checkpoint_dir = ensure_dir(artifact / "checkpoints")
                    torch.save(checkpoint_state, checkpoint_dir / f"step_{global_step:09d}.pt")
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
    manifest_path: Path | None = None,
    output_path: Path | None = None,
) -> Path:
    from hcl_poc.incremental import _phase4_dino_from_config, _phase4_frame_inputs
    from hcl_poc.learned_interface import _low_condition_array
    from hcl_poc.low_level_rl import DirectLowActorCritic, ResidualActorCritic, _load_frozen

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
    rerun_config = _rerun_base_config(config)
    frozen = _load_frozen(rerun_config, n_demo, seed, device)
    horizon = int(frozen.horizon_steps)
    dino = _phase4_dino_from_config(config, device)
    condition_dim = int(checkpoint["condition_dim"])
    method = str(recipe.get("method", ""))
    is_direct = method.startswith("r3_direct")
    base_policy = str(recipe.get("base_policy", "deterministic"))
    flow_model: FlowModel | None = None
    flow_checkpoint: dict[str, Any] | None = None
    if is_direct:
        base_policy = "deterministic"
    elif base_policy == "flow":
        flow_path = recipe.get("flow_checkpoint")
        if not flow_path:
            raise ValueError("R2 checkpoint is missing flow_checkpoint")
        flow_model, flow_checkpoint = _load_low_flow_base(Path(flow_path), device)
    elif base_policy != "deterministic":
        raise ValueError(f"Unknown residual base policy: {base_policy}")
    if is_direct:
        agent = DirectLowActorCritic(
            frozen.low_model,
            frozen.action_norm.mean,
            frozen.action_norm.std,
            condition_dim,
            width=int(recipe["actor_critic_width"]),
            depth=int(recipe["actor_critic_depth"]),
            initial_logstd=float(recipe["initial_logstd"]),
        ).to(device)
    else:
        agent = ResidualActorCritic(
            condition_dim,
            width=int(recipe["actor_critic_width"]),
            depth=int(recipe["actor_critic_depth"]),
            initial_logstd=float(recipe["initial_logstd"]),
        ).to(device)
    agent.load_state_dict(checkpoint["agent"])
    agent.eval()
    alpha = float(recipe.get("alpha", 0.0))
    residual_condition_mode = str(recipe.get("residual_condition_mode", "full"))
    if residual_condition_mode not in {"full", "goal_delta"}:
        raise ValueError(f"Unknown residual_condition_mode: {residual_condition_mode}")
    residual_action_mode = str(recipe.get("residual_action_mode", "additive"))
    if residual_action_mode not in {"additive", "margin_scaled"}:
        raise ValueError(f"Unknown residual_action_mode: {residual_action_mode}")

    with h5py.File(path, "r") as h5:
        meta = h5["meta"].attrs
        num_envs = int(meta["num_envs"])
        max_steps = int(meta["max_steps"])
        batch_keys = sorted(key for key in h5.keys() if key.startswith("batch_"))
    evaluation_entries = _local_eval_entries(
        path,
        batch_keys,
        max_steps,
        episodes,
        seed,
        horizon,
        manifest_path,
    )
    env = _make_benchmark_env(config, num_envs, "rgb+state")
    action_low = torch.as_tensor(env.single_action_space.low, device=device, dtype=torch.float32)
    action_high = torch.as_tensor(env.single_action_space.high, device=device, dtype=torch.float32)

    initial_distances: list[np.ndarray] = []
    final_distances: list[np.ndarray] = []
    action_delta_norms: list[np.ndarray] = []
    final_env_rewards: list[np.ndarray] = []
    max_env_rewards: list[np.ndarray] = []
    mean_env_rewards: list[np.ndarray] = []
    saturation_rates: list[float] = []
    task_success_once = np.zeros((episodes, num_envs), dtype=np.bool_)
    chosen_batches: list[str] = []
    chosen_timesteps: list[int] = []
    try:
        with h5py.File(path, "r") as h5:
            for episode_index in trange(episodes, desc="eval local R1"):
                entry = evaluation_entries[episode_index]
                key = str(entry["batch"])
                group = h5[key]
                t = int(entry["timestep"])
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
                total_env_reward = np.zeros(num_envs, dtype=np.float32)
                episode_max_env_reward = np.full(num_envs, -np.inf, dtype=np.float32)
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
                    full_condition_np = _low_condition_array(
                        frozen.frame_norm.transform(frames),
                        current_z,
                        goal_z,
                        previous,
                        remaining,
                        frozen.conditioning,
                    )
                    condition_t = torch.from_numpy(full_condition_np).to(device).float()
                    if base_policy == "deterministic":
                        normalized_base = frozen.low_model(condition_t)
                        base_action = torch.from_numpy(
                            frozen.action_norm.inverse(
                                normalized_base.cpu().numpy().astype(np.float32)
                            )
                        ).to(device)
                    else:
                        if flow_model is None or flow_checkpoint is None:
                            raise RuntimeError("R2 flow base was not loaded")
                        base_action = _low_flow_base_action(
                            flow_model,
                            flow_checkpoint,
                            condition_t,
                            frozen,
                        )
                    if is_direct:
                        raw_action, _logprob, _entropy, _value = agent.get_action_and_value(
                            condition_t,
                            deterministic=True,
                        )
                        unclipped = raw_action
                        action_delta_norms.append(
                            torch.linalg.vector_norm(unclipped - base_action, dim=-1)
                            .cpu()
                            .numpy()
                        )
                    else:
                        residual_condition_np = _residual_condition_array(
                            mode=residual_condition_mode,
                            full_condition=full_condition_np,
                            current_z=current_z,
                            goal_z=goal_z,
                            previous_action=previous,
                            remaining=remaining,
                        )
                        residual_condition_t = torch.from_numpy(
                            residual_condition_np
                        ).to(device).float()
                        raw_action, _logprob, _entropy, _value = agent.get_action_and_value(
                            residual_condition_t,
                            deterministic=True,
                        )
                        residual, unclipped, action = _residual_action_from_raw(
                            base_action,
                            raw_action,
                            alpha,
                            action_low,
                            action_high,
                            residual_action_mode,
                        )
                        action_delta_norms.append(
                            torch.linalg.vector_norm(residual, dim=-1).cpu().numpy()
                        )
                    if is_direct:
                        action = torch.clamp(unclipped, action_low, action_high)
                    saturation_count += int(torch.any(unclipped != action, dim=-1).sum().cpu())
                    obs, step_reward, _terminated, _truncated, info = env.step(action)
                    step_reward_np = _to_numpy(step_reward).reshape(-1).astype(np.float32)
                    total_env_reward += step_reward_np
                    episode_max_env_reward = np.maximum(
                        episode_max_env_reward,
                        step_reward_np,
                    )
                    task_success_once[episode_index] |= (
                        _to_numpy(info.get("success", np.zeros(num_envs, dtype=np.bool_)))
                        .reshape(-1)
                        .astype(np.bool_)
                    )
                    previous = frozen.action_norm.transform(action.cpu().numpy())
                    if local_step == horizon - 1:
                        next_frames = _phase4_frame_inputs(
                            obs, dino, int(config.get("dino.batch_size", 64))
                        )
                        next_z = _encode_rerun_frames(frozen, next_frames, device)
                        final_distances.append(
                            np.mean(np.square(next_z - goal_z), axis=-1).astype(np.float32)
                        )
                        final_env_rewards.append(step_reward_np)
                        max_env_rewards.append(episode_max_env_reward)
                        mean_env_rewards.append(total_env_reward / horizon)
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
    action_delta = np.concatenate(action_delta_norms)
    final_env_reward = np.concatenate(final_env_rewards)
    max_env_reward = np.concatenate(max_env_rewards)
    mean_env_reward = np.concatenate(mean_env_rewards)
    result = {
        "checkpoint": str(checkpoint_path),
        "dataset": str(path),
        "n_demo": n_demo,
        "seed": seed,
        "evaluation_manifest": str(manifest_path) if manifest_path else None,
        "evaluation_entries": evaluation_entries,
        "episodes": episodes,
        "num_envs": num_envs,
        "sampled_local_episodes": int(episodes * num_envs),
        "horizon": horizon,
        "chosen_batches": chosen_batches,
        "chosen_timesteps": chosen_timesteps,
        "chosen_timestep_min": int(min(chosen_timesteps)),
        "chosen_timestep_max": int(max(chosen_timesteps)),
        "initial_distance_mean": float(np.mean(initial)),
        "final_distance_mean": float(np.mean(final)),
        "distance_reduction_mean": float(np.mean(initial - final)),
        "distance_reduction_fraction": float(np.mean(final < initial)),
        "mean_residual_norm": float(np.mean(action_delta)),
        "mean_action_delta_l2": float(np.mean(action_delta)),
        "final_env_reward_mean": float(np.mean(final_env_reward)),
        "max_env_reward_mean": float(np.mean(max_env_reward)),
        "mean_env_reward_mean": float(np.mean(mean_env_reward)),
        "task_success_once_fraction": float(np.mean(task_success_once)),
        "action_saturation_rate": float(np.mean(saturation_rates)),
        "recipe": recipe,
    }
    out_path = output_path or (
        _state_audit_result_dir(config)
        / str(recipe.get("family_dir", "local_r1"))
        / f"n{n_demo}"
        / f"seed{seed}"
        / str(recipe["run_name"])
        / f"eval_local_{episodes}.json"
    )
    write_json(out_path, result)
    return out_path


@torch.inference_mode()
def evaluate_rl_rerun_local_r2(
    config: Config,
    checkpoint_path: Path,
    dataset_path: Path | None = None,
    n_demo: int = 500,
    seed: int = 0,
    episodes: int = 4,
    manifest_path: Path | None = None,
    output_path: Path | None = None,
) -> Path:
    return evaluate_rl_rerun_local_r1(
        config,
        checkpoint_path=checkpoint_path,
        dataset_path=dataset_path,
        n_demo=n_demo,
        seed=seed,
        episodes=episodes,
        manifest_path=manifest_path,
        output_path=output_path,
    )


@torch.inference_mode()
def evaluate_rl_rerun_local_r3(
    config: Config,
    checkpoint_path: Path,
    dataset_path: Path | None = None,
    n_demo: int = 500,
    seed: int = 0,
    episodes: int = 4,
    manifest_path: Path | None = None,
    output_path: Path | None = None,
) -> Path:
    return evaluate_rl_rerun_local_r1(
        config,
        checkpoint_path=checkpoint_path,
        dataset_path=dataset_path,
        n_demo=n_demo,
        seed=seed,
        episodes=episodes,
        manifest_path=manifest_path,
        output_path=output_path,
    )


@torch.inference_mode()
def evaluate_rl_rerun_closed_loop_r1(
    config: Config,
    checkpoint_path: Path,
    n_demo: int = 500,
    seed: int = 0,
    episodes: int = 100,
    eval_seed_start: int = 10_000,
    num_envs: int = 64,
    disturbed: bool = False,
    goal_source: str = "learned",
    oracle_copy_mode: str = "replay",
    output_path: Path | None = None,
) -> Path:
    from hcl_poc.incremental import _clone_mani_state_dict
    from hcl_poc.incremental import _phase4_dino_from_config, _phase4_frame_inputs
    from hcl_poc.incremental import _pre_rl_phase_d_schedule, PRE_RL_PHASE_D_PERTURBATIONS
    from hcl_poc.incremental import _phase7_obs_state_tensor
    from hcl_poc.learned_interface import _low_condition_array
    from hcl_poc.low_level_rl import DirectLowActorCritic, ResidualActorCritic, _load_frozen

    if not checkpoint_path.exists():
        raise FileNotFoundError(checkpoint_path)
    if episodes <= 0 or num_envs <= 0:
        raise ValueError("episodes and num_envs must be positive")
    if goal_source not in {"learned", "oracle"}:
        raise ValueError("goal_source must be 'learned' or 'oracle'")
    if oracle_copy_mode not in {"replay", "state_dict"}:
        raise ValueError("oracle_copy_mode must be 'replay' or 'state_dict'")

    device = default_device()
    rerun_config = _rerun_base_config(config)
    frozen = _load_frozen(rerun_config, n_demo, seed, device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    recipe = checkpoint["recipe"]
    if int(recipe["n_demo"]) != n_demo or int(recipe["seed"]) != seed:
        raise ValueError("Residual checkpoint does not match n_demo/seed")
    method = str(recipe.get("method", ""))
    is_direct = method.startswith("r3_direct")
    base_policy = str(recipe.get("base_policy", "deterministic"))
    flow_model: FlowModel | None = None
    flow_checkpoint: dict[str, Any] | None = None
    if is_direct:
        base_policy = "deterministic"
    elif base_policy == "flow":
        flow_path = recipe.get("flow_checkpoint")
        if not flow_path:
            raise ValueError("R2 checkpoint is missing flow_checkpoint")
        flow_model, flow_checkpoint = _load_low_flow_base(Path(flow_path), device)
    elif base_policy != "deterministic":
        raise ValueError(f"Unknown residual base policy: {base_policy}")
    if is_direct:
        agent = DirectLowActorCritic(
            frozen.low_model,
            frozen.action_norm.mean,
            frozen.action_norm.std,
            int(checkpoint["condition_dim"]),
            width=int(recipe["actor_critic_width"]),
            depth=int(recipe["actor_critic_depth"]),
            initial_logstd=float(recipe["initial_logstd"]),
        ).to(device)
    else:
        agent = ResidualActorCritic(
            int(checkpoint["condition_dim"]),
            width=int(recipe["actor_critic_width"]),
            depth=int(recipe["actor_critic_depth"]),
            initial_logstd=float(recipe["initial_logstd"]),
        ).to(device)
    agent.load_state_dict(checkpoint["agent"])
    agent.eval()
    dino = _phase4_dino_from_config(config, device)
    teacher = load_ppo_agent(_rl_paths(config).best, device) if goal_source == "oracle" else None
    alpha = float(recipe.get("alpha", 0.0))
    residual_condition_mode = str(recipe.get("residual_condition_mode", "full"))
    if residual_condition_mode not in {"full", "goal_delta"}:
        raise ValueError(f"Unknown residual_condition_mode: {residual_condition_mode}")
    residual_action_mode = str(recipe.get("residual_action_mode", "additive"))
    if residual_action_mode not in {"additive", "margin_scaled"}:
        raise ValueError(f"Unknown residual_action_mode: {residual_action_mode}")
    max_steps = int(config.get("env_max_episode_steps", 100))
    disturbance_rng = np.random.default_rng(eval_seed_start + 10_000)
    all_schedules = (
        _pre_rl_phase_d_schedule(disturbance_rng, episodes, max_steps, 1, 1)
        if disturbed
        else [[] for _ in range(episodes)]
    )
    if disturbed:
        for events in all_schedules:
            event = events[0]
            duration = int(event["end"] - event["start"])
            event["start"] = int(
                disturbance_rng.integers(15, max_steps - duration - 20)
            )
            event["end"] = event["start"] + duration

    def rollout(use_residual: bool) -> dict[str, Any]:
        rollout_rng = np.random.default_rng(eval_seed_start + 20_000)
        successes: list[float] = []
        final_rewards: list[float] = []
        max_rewards: list[float] = []
        residual_norms: list[float] = []
        recovered: list[float] = []
        recovery_times: list[int] = []
        replay_errors: list[float] = []
        branch_goal_distances: list[float] = []
        branch_latencies: list[float] = []
        saturated_actions = 0
        active_actions = 0
        high_decisions = 0

        for batch_start in range(0, episodes, num_envs):
            batch_envs = min(num_envs, episodes - batch_start)
            reset_seeds = [
                eval_seed_start + batch_start + index for index in range(batch_envs)
            ]
            schedules = all_schedules[batch_start : batch_start + batch_envs]
            env = gym.make(
                config.get("env_id"),
                obs_mode="rgb+state",
                control_mode=config.get("control_mode"),
                reward_mode="normalized_dense",
                render_mode=None,
                sim_backend=_rl_backend(config),
                num_envs=batch_envs,
                reconfiguration_freq=config.get("rl.eval_reconfiguration_freq", 1),
            )
            branch_env = (
                gym.make(
                    config.get("env_id"),
                    obs_mode="rgb+state",
                    control_mode=config.get("control_mode"),
                    reward_mode="normalized_dense",
                    render_mode=None,
                    sim_backend=_rl_backend(config),
                    num_envs=batch_envs,
                    reconfiguration_freq=config.get("rl.eval_reconfiguration_freq", 1),
                )
                if goal_source == "oracle"
                else None
            )
            action_low = torch.as_tensor(
                np.asarray(env.action_space.low, dtype=np.float32),
                device=device,
            )
            action_high = torch.as_tensor(
                np.asarray(env.action_space.high, dtype=np.float32),
                device=device,
            )
            if action_low.ndim == 1:
                action_low = action_low.unsqueeze(0)
                action_high = action_high.unsqueeze(0)
            single_action_space = getattr(env, "single_action_space", env.action_space)
            action_low_np = np.asarray(single_action_space.low, dtype=np.float32)
            action_high_np = np.asarray(single_action_space.high, dtype=np.float32)
            if action_low_np.ndim == 2:
                action_low_np = action_low_np[0]
                action_high_np = action_high_np[0]
            action_range_np = action_high_np - action_low_np
            zero_previous = frozen.action_norm.transform(
                np.zeros((1, 3), dtype=np.float32)
            )[0]
            previous_action = np.repeat(
                zero_previous[None], batch_envs, axis=0
            )
            held_goal = np.zeros(
                (batch_envs, frozen.goal_dim), dtype=np.float32
            )
            countdown = np.zeros(batch_envs, dtype=np.int32)
            active = np.ones(batch_envs, dtype=np.bool_)
            success_once = np.zeros(batch_envs, dtype=np.bool_)
            recovered_batch = np.zeros(batch_envs, dtype=np.bool_)
            recovery_time_batch = np.full(batch_envs, -1, dtype=np.int32)
            batch_final = np.zeros(batch_envs, dtype=np.float32)
            batch_max = np.full(batch_envs, -np.inf, dtype=np.float32)
            policy_action_history: list[np.ndarray] = []
            previous_executed = np.zeros((batch_envs, 3), dtype=np.float32)
            bias_noise = np.zeros_like(previous_executed)
            history: list[torch.Tensor] = []
            try:
                obs, _info = env.reset(seed=reset_seeds)
                if branch_env is not None and oracle_copy_mode == "state_dict":
                    branch_env.reset(seed=reset_seeds)
                for step_index in range(max_steps):
                    if not np.any(active):
                        break
                    frames = _phase4_frame_inputs(
                        obs, dino, int(config.get("dino.batch_size", 64))
                    )
                    normalized_frames = frozen.frame_norm.transform(frames)
                    replan = active & (countdown <= 0)
                    if np.any(replan):
                        high_input = np.concatenate(
                            [normalized_frames, previous_action], axis=-1
                        )
                        predicted_goal = frozen.high_model(
                            torch.from_numpy(high_input).to(device).float()
                        ).cpu().numpy()
                        held_goal[replan] = predicted_goal[replan]
                        if branch_env is not None:
                            branch_timer = Timer()
                            if oracle_copy_mode == "replay":
                                branch_obs, _branch_info = branch_env.reset(seed=reset_seeds)
                                for action_history in history:
                                    branch_step = branch_env.step(action_history)
                                    (
                                        branch_obs,
                                        _branch_reward,
                                        _branch_term,
                                        _branch_trunc,
                                        _branch_info,
                                    ) = branch_step
                            else:
                                branch_env.unwrapped.set_state_dict(
                                    _clone_mani_state_dict(env.unwrapped.get_state_dict())
                                )
                                branch_obs = branch_env.unwrapped.get_obs()
                            replay_error = torch.max(
                                torch.abs(
                                    env.unwrapped.get_state()
                                    - branch_env.unwrapped.get_state()
                                ),
                                dim=1,
                            ).values
                            replay_errors.extend(
                                replay_error.detach().cpu().numpy()[replan].astype(float).tolist()
                            )
                            for _branch_step in range(frozen.horizon_steps):
                                if teacher is None:
                                    raise RuntimeError("Oracle goal source requires teacher")
                                branch_state = _phase7_obs_state_tensor(branch_obs, device)
                                teacher_action = torch.clamp(
                                    teacher.actor_mean(branch_state),
                                    action_low,
                                    action_high,
                                )
                                branch_step = branch_env.step(teacher_action)
                                (
                                    branch_obs,
                                    _branch_reward,
                                    branch_term,
                                    branch_trunc,
                                    _branch_info,
                                ) = branch_step
                                if bool(torch.all(torch.logical_or(branch_term, branch_trunc))):
                                    break
                            branch_frames = _phase4_frame_inputs(
                                branch_obs,
                                dino,
                                int(config.get("dino.batch_size", 64)),
                            )
                            oracle_goal = _encode_rerun_frames(frozen, branch_frames, device)
                            held_goal[replan] = oracle_goal[replan]
                            current_goal_for_distance = _encode_rerun_frames(
                                frozen,
                                frames,
                                device,
                            )
                            branch_goal_distances.extend(
                                np.linalg.norm(
                                    current_goal_for_distance[replan] - oracle_goal[replan],
                                    axis=-1,
                                ).tolist()
                            )
                            branch_latencies.append(
                                branch_timer.elapsed() / int(np.sum(replan))
                            )
                        countdown[replan] = frozen.update_period
                        high_decisions += int(np.sum(replan))

                    if frozen.conditioning in {"delta", "relation"} or (
                        use_residual and not is_direct and residual_condition_mode != "full"
                    ):
                        current_z = _encode_rerun_frames(frozen, frames, device)
                    else:
                        current_z = np.empty_like(held_goal)
                    remaining = np.maximum(countdown, 1).astype(np.float32)
                    condition_np = _low_condition_array(
                        normalized_frames,
                        current_z,
                        held_goal,
                        previous_action,
                        (remaining / frozen.horizon_steps)[:, None],
                        frozen.conditioning,
                    )
                    condition = torch.from_numpy(condition_np).to(device).float()
                    if base_policy == "deterministic":
                        normalized_base = frozen.low_model(condition)
                        base_action = torch.from_numpy(
                            frozen.action_norm.inverse(
                                normalized_base.cpu().numpy().astype(np.float32)
                            )
                        ).to(device)
                    else:
                        if flow_model is None or flow_checkpoint is None:
                            raise RuntimeError("R2 flow base was not loaded")
                        base_action = _low_flow_base_action(
                            flow_model,
                            flow_checkpoint,
                            condition,
                            frozen,
                        )
                    if use_residual and is_direct:
                        raw_action, _logprob, _entropy, _value = agent.get_action_and_value(
                            condition,
                            deterministic=True,
                        )
                        unclipped = raw_action
                        residual_norms.extend(
                            torch.linalg.vector_norm(unclipped - base_action, dim=-1)
                            .cpu()
                            .numpy()[active]
                            .tolist()
                        )
                    elif use_residual:
                        residual_condition_np = _residual_condition_array(
                            mode=residual_condition_mode,
                            full_condition=condition_np,
                            current_z=current_z,
                            goal_z=held_goal,
                            previous_action=previous_action,
                            remaining=(remaining / frozen.horizon_steps)[:, None],
                        )
                        residual_condition = torch.from_numpy(
                            residual_condition_np
                        ).to(device).float()
                        raw_residual, _logprob, _entropy, _value = agent.get_action_and_value(
                            residual_condition,
                            deterministic=True,
                        )
                        residual, unclipped, _action = _residual_action_from_raw(
                            base_action,
                            raw_residual,
                            alpha,
                            action_low,
                            action_high,
                            residual_action_mode,
                        )
                        residual_norms.extend(
                            torch.linalg.vector_norm(residual, dim=-1)
                            .cpu()
                            .numpy()[active]
                            .tolist()
                        )
                    else:
                        unclipped = base_action
                    unclipped_np = unclipped.detach().cpu().numpy().astype(np.float32)
                    executed_np = unclipped_np.copy()
                    policy_action_history.append(unclipped_np.copy())
                    if disturbed:
                        for env_index, events in enumerate(schedules):
                            event = events[0]
                            if not event["start"] <= step_index < event["end"]:
                                continue
                            kind = int(event["kind"])
                            if kind == 1:
                                bias_noise[env_index] = (
                                    0.7 * bias_noise[env_index]
                                    + 0.3
                                    * rollout_rng.normal(0.0, 0.01, size=3).astype(np.float32)
                                    * action_range_np
                                )
                                executed_np[env_index] += (
                                    event["bias_fraction"]
                                    * action_range_np
                                    * event["bias_direction"]
                                    + bias_noise[env_index]
                                )
                            elif kind == 2:
                                executed_np[env_index] = previous_executed[env_index]
                            elif kind == 3:
                                source_step = max(0, step_index - int(event["delay"]))
                                executed_np[env_index] = policy_action_history[source_step][
                                    env_index
                                ]
                            else:
                                executed_np[env_index] *= float(event["scale"])
                        executed_np = np.clip(executed_np, action_low_np, action_high_np)
                        unclipped = torch.from_numpy(executed_np).to(device).float()
                    active_tensor = torch.from_numpy(active).to(device)
                    saturated_actions += int(
                        torch.any(unclipped != torch.clamp(
                            unclipped, action_low, action_high
                        ), dim=-1)[active_tensor].sum().cpu()
                    )
                    active_actions += int(np.sum(active))
                    action = torch.clamp(unclipped, action_low, action_high)
                    action[~active_tensor] = 0.0
                    obs, reward, terminated, truncated, info = env.step(action)
                    if branch_env is not None:
                        history.append(action.detach().clone())
                    executed_after_clip = action.cpu().numpy().astype(np.float32)
                    previous_action = frozen.action_norm.transform(executed_after_clip)
                    previous_executed = executed_after_clip
                    countdown -= 1
                    reward_np = _to_numpy(reward).reshape(-1).astype(np.float32)
                    batch_final[active] = reward_np[active]
                    batch_max[active] = np.maximum(
                        batch_max[active], reward_np[active]
                    )
                    if "success" in info:
                        step_success = (
                            _to_numpy(info["success"])
                            .reshape(-1)
                            .astype(np.bool_)
                        )
                        success_once |= step_success
                        if disturbed:
                            for env_index, events in enumerate(schedules):
                                event = events[0]
                                if (
                                    step_index >= event["end"]
                                    and step_success[env_index]
                                    and not recovered_batch[env_index]
                                ):
                                    recovered_batch[env_index] = True
                                    recovery_time_batch[env_index] = (
                                        step_index - int(event["end"]) + 1
                                    )
                    done = np.logical_or(
                        _to_numpy(terminated).reshape(-1),
                        _to_numpy(truncated).reshape(-1),
                    )
                    active[done.astype(np.bool_)] = False
            finally:
                env.close()
                if branch_env is not None:
                    branch_env.close()
            successes.extend(success_once.astype(float).tolist())
            final_rewards.extend(batch_final.astype(float).tolist())
            max_rewards.extend(batch_max.astype(float).tolist())
            if disturbed:
                recovered.extend(recovered_batch.astype(float).tolist())
                recovery_times.extend(recovery_time_batch[recovery_time_batch >= 0].tolist())

        result = {
            "success": float(np.mean(successes)),
            "final_reward": float(np.mean(final_rewards)),
            "max_reward": float(np.mean(max_rewards)),
            "action_saturation_rate": saturated_actions / max(active_actions, 1),
            "mean_residual_norm": (
                float(np.mean(residual_norms)) if residual_norms else 0.0
            ),
            "high_level_decisions_per_episode": high_decisions / episodes,
            "episode_success": successes,
            "episode_final_reward": final_rewards,
            "episode_max_reward": max_rewards,
        }
        if goal_source == "oracle":
            result.update(
                {
                    "replay_current_state_error_mean": (
                        float(np.mean(replay_errors)) if replay_errors else 0.0
                    ),
                    "replay_current_state_error_max": (
                        float(np.max(replay_errors)) if replay_errors else 0.0
                    ),
                    "branch_goal_l2_mean": (
                        float(np.mean(branch_goal_distances))
                        if branch_goal_distances
                        else None
                    ),
                    "branch_generation_latency_per_replan_s": (
                        float(np.mean(branch_latencies)) if branch_latencies else 0.0
                    ),
                }
            )
        if disturbed:
            result.update(
                {
                    "recovery_success": float(np.mean(recovered)) if recovered else None,
                    "recovery_time_mean": (
                        float(np.mean(recovery_times)) if recovery_times else None
                    ),
                    "episode_recovered": recovered,
                    "episode_recovery_time": recovery_times,
                }
            )
        return result

    frozen_result = rollout(use_residual=False)
    residual_result = rollout(use_residual=True)
    result = {
        "stage": "closed_loop_residual_paired",
        "checkpoint": str(checkpoint_path),
        "n_demo": n_demo,
        "seed": seed,
        "episodes": episodes,
        "eval_seed_start": eval_seed_start,
        "num_envs": num_envs,
        "disturbed": disturbed,
        "disturbance_family": PRE_RL_PHASE_D_PERTURBATIONS if disturbed else None,
        "goal_source": goal_source,
        "oracle_copy_mode": oracle_copy_mode if goal_source == "oracle" else None,
        "horizon": frozen.horizon_steps,
        "update_period": frozen.update_period,
        "base_policy": base_policy,
        "frozen": frozen_result,
        "residual": residual_result,
        "success_delta": residual_result["success"] - frozen_result["success"],
        "final_reward_delta": (
            residual_result["final_reward"] - frozen_result["final_reward"]
        ),
        "max_reward_delta": (
            residual_result["max_reward"] - frozen_result["max_reward"]
        ),
        "recipe": recipe,
    }
    if disturbed:
        result["recovery_success_delta"] = (
            residual_result["recovery_success"] - frozen_result["recovery_success"]
        )
    out_path = output_path or (
        _state_audit_result_dir(config)
        / str(recipe.get("family_dir", "local_r1"))
        / f"n{n_demo}"
        / f"seed{seed}"
        / str(recipe["run_name"])
        / f"closed_loop_{episodes}.json"
    )
    ensure_dir(out_path.parent)
    write_json(out_path, result)
    return out_path


@torch.inference_mode()
def evaluate_rl_rerun_closed_loop_r2(
    config: Config,
    checkpoint_path: Path,
    n_demo: int = 500,
    seed: int = 0,
    episodes: int = 100,
    eval_seed_start: int = 10_000,
    num_envs: int = 64,
    disturbed: bool = False,
    goal_source: str = "learned",
    oracle_copy_mode: str = "replay",
    output_path: Path | None = None,
) -> Path:
    return evaluate_rl_rerun_closed_loop_r1(
        config,
        checkpoint_path=checkpoint_path,
        n_demo=n_demo,
        seed=seed,
        episodes=episodes,
        eval_seed_start=eval_seed_start,
        num_envs=num_envs,
        disturbed=disturbed,
        goal_source=goal_source,
        oracle_copy_mode=oracle_copy_mode,
        output_path=output_path,
    )


@torch.inference_mode()
def evaluate_rl_rerun_closed_loop_r3(
    config: Config,
    checkpoint_path: Path,
    n_demo: int = 500,
    seed: int = 0,
    episodes: int = 100,
    eval_seed_start: int = 10_000,
    num_envs: int = 64,
    disturbed: bool = False,
    goal_source: str = "learned",
    oracle_copy_mode: str = "replay",
    output_path: Path | None = None,
) -> Path:
    return evaluate_rl_rerun_closed_loop_r1(
        config,
        checkpoint_path=checkpoint_path,
        n_demo=n_demo,
        seed=seed,
        episodes=episodes,
        eval_seed_start=eval_seed_start,
        num_envs=num_envs,
        disturbed=disturbed,
        goal_source=goal_source,
        oracle_copy_mode=oracle_copy_mode,
        output_path=output_path,
    )


@torch.inference_mode()
def record_rl_rerun_videos(
    config: Config,
    checkpoint_path: Path,
    n_demo: int = 500,
    seed: int = 0,
    episodes: int = 6,
    eval_seed_start: int = 10_000,
    mode: str = "both",
    output_dir: Path | None = None,
    force: bool = False,
) -> list[Path]:
    import imageio.v2 as imageio
    from hcl_poc.incremental import _phase4_dino_from_config, _phase4_frame_inputs
    from hcl_poc.learned_interface import _low_condition_array
    from hcl_poc.low_level_rl import DirectLowActorCritic, ResidualActorCritic, _load_frozen

    if mode not in {"frozen", "tuned", "both"}:
        raise ValueError("mode must be one of: frozen, tuned, both")
    if episodes <= 0:
        raise ValueError("episodes must be positive")
    if not checkpoint_path.exists():
        raise FileNotFoundError(checkpoint_path)

    device = default_device()
    rerun_config = _rerun_base_config(config)
    frozen = _load_frozen(rerun_config, n_demo, seed, device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    recipe = checkpoint["recipe"]
    if int(recipe["n_demo"]) != n_demo or int(recipe["seed"]) != seed:
        raise ValueError("Checkpoint does not match n_demo/seed")
    method = str(recipe.get("method", ""))
    is_direct = method.startswith("r3_direct")
    base_policy = str(recipe.get("base_policy", "deterministic"))
    flow_model: FlowModel | None = None
    flow_checkpoint: dict[str, Any] | None = None
    if is_direct:
        base_policy = "deterministic"
    elif base_policy == "flow":
        flow_path = recipe.get("flow_checkpoint")
        if not flow_path:
            raise ValueError("R2 checkpoint is missing flow_checkpoint")
        flow_model, flow_checkpoint = _load_low_flow_base(Path(flow_path), device)
    elif base_policy != "deterministic":
        raise ValueError(f"Unknown base policy: {base_policy}")

    if is_direct:
        agent: nn.Module = DirectLowActorCritic(
            frozen.low_model,
            frozen.action_norm.mean,
            frozen.action_norm.std,
            int(checkpoint["condition_dim"]),
            width=int(recipe["actor_critic_width"]),
            depth=int(recipe["actor_critic_depth"]),
            initial_logstd=float(recipe["initial_logstd"]),
        ).to(device)
    else:
        agent = ResidualActorCritic(
            int(checkpoint["condition_dim"]),
            width=int(recipe["actor_critic_width"]),
            depth=int(recipe["actor_critic_depth"]),
            initial_logstd=float(recipe["initial_logstd"]),
        ).to(device)
    agent.load_state_dict(checkpoint["agent"])
    agent.eval()
    alpha = float(recipe.get("alpha", 0.0))
    residual_condition_mode = str(recipe.get("residual_condition_mode", "full"))
    if residual_condition_mode not in {"full", "goal_delta"}:
        raise ValueError(f"Unknown residual_condition_mode: {residual_condition_mode}")
    residual_action_mode = str(recipe.get("residual_action_mode", "additive"))
    if residual_action_mode not in {"additive", "margin_scaled"}:
        raise ValueError(f"Unknown residual_action_mode: {residual_action_mode}")
    dino = _phase4_dino_from_config(config, device)
    max_steps = int(config.get("env_max_episode_steps", 100))
    control_freq = int(config.get("control_freq", 20))
    root = ensure_dir(output_dir or Path("rl_rerun_failure_videos"))
    selected_modes = ["frozen", "tuned"] if mode == "both" else [mode]
    written: list[Path] = []

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

    for selected_mode in selected_modes:
        mode_dir = ensure_dir(root / selected_mode)
        for episode_index in trange(
            episodes,
            desc=f"record rl rerun {selected_mode}",
        ):
            rollout_seed = eval_seed_start + episode_index
            existing = sorted(mode_dir.glob(f"seed{rollout_seed}_*.mp4"))
            if existing and not force:
                written.append(existing[0])
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
            previous_action = frozen.action_norm.transform(
                np.zeros((1, 3), dtype=np.float32)
            )
            held_goal = np.zeros((1, frozen.goal_dim), dtype=np.float32)
            countdown = np.zeros(1, dtype=np.int32)
            frames_out: list[np.ndarray] = []
            success = False
            final_reward = 0.0
            max_reward = -float("inf")
            try:
                obs, _info = env.reset(seed=[rollout_seed])
                frames_out.append(render_frame(env))
                for _step in range(max_steps):
                    frames = _phase4_frame_inputs(
                        obs,
                        dino,
                        int(config.get("dino.batch_size", 64)),
                    )
                    normalized_frames = frozen.frame_norm.transform(frames)
                    if countdown[0] <= 0:
                        high_input = np.concatenate(
                            [normalized_frames, previous_action],
                            axis=-1,
                        )
                        held_goal = (
                            frozen.high_model(
                                torch.from_numpy(high_input).to(device).float()
                            )
                            .cpu()
                            .numpy()
                        )
                        countdown[0] = frozen.update_period

                    if frozen.conditioning in {"delta", "relation"} or (
                        selected_mode == "tuned"
                        and not is_direct
                        and residual_condition_mode != "full"
                    ):
                        current_z = _encode_rerun_frames(frozen, frames, device)
                    else:
                        current_z = np.empty_like(held_goal)
                    condition_np = _low_condition_array(
                        normalized_frames,
                        current_z,
                        held_goal,
                        previous_action,
                        (np.maximum(countdown, 1).astype(np.float32) / frozen.horizon_steps)[
                            :, None
                        ],
                        frozen.conditioning,
                    )
                    condition = torch.from_numpy(condition_np).to(device).float()
                    if base_policy == "deterministic":
                        normalized_base = frozen.low_model(condition)
                        base_action = torch.from_numpy(
                            frozen.action_norm.inverse(
                                normalized_base.cpu().numpy().astype(np.float32)
                            )
                        ).to(device)
                    else:
                        if flow_model is None or flow_checkpoint is None:
                            raise RuntimeError("R2 flow base was not loaded")
                        base_action = _low_flow_base_action(
                            flow_model,
                            flow_checkpoint,
                            condition,
                            frozen,
                        )

                    if selected_mode == "frozen":
                        unclipped = base_action
                    elif is_direct:
                        unclipped = agent.get_action_and_value(
                            condition,
                            deterministic=True,
                        )[0]
                    else:
                        residual_condition_np = _residual_condition_array(
                            mode=residual_condition_mode,
                            full_condition=condition_np,
                            current_z=current_z,
                            goal_z=held_goal,
                            previous_action=previous_action,
                            remaining=(
                                np.maximum(countdown, 1).astype(np.float32)
                                / frozen.horizon_steps
                            )[:, None],
                        )
                        residual_condition = torch.from_numpy(
                            residual_condition_np
                        ).to(device).float()
                        raw_residual = agent.get_action_and_value(
                            residual_condition,
                            deterministic=True,
                        )[0]
                        _residual, unclipped, _action = _residual_action_from_raw(
                            base_action,
                            raw_residual,
                            alpha,
                            action_low,
                            action_high,
                            residual_action_mode,
                        )
                    action = torch.clamp(unclipped, action_low, action_high)
                    obs, reward, terminated, truncated, info = env.step(action)
                    previous_action = frozen.action_norm.transform(
                        action.cpu().numpy().astype(np.float32)
                    )
                    countdown -= 1
                    frames_out.append(render_frame(env))
                    final_reward = float(_to_numpy(reward).reshape(-1)[0])
                    max_reward = max(max_reward, final_reward)
                    if "success" in info:
                        success = success or bool(_to_numpy(info["success"]).reshape(-1)[0])
                    done = bool(
                        _to_numpy(torch.logical_or(terminated, truncated)).reshape(-1)[0]
                    )
                    if done:
                        break
            finally:
                env.close()

            path = mode_dir / (
                f"seed{rollout_seed}_step{int(checkpoint['global_step'])}_"
                f"success{int(success)}_final{final_reward:.3f}_max{max_reward:.3f}.mp4"
            )
            imageio.mimsave(path, frames_out, fps=control_freq, macro_block_size=1)
            written.append(path)
    return written
