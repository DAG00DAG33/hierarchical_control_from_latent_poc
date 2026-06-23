from __future__ import annotations

import subprocess
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
