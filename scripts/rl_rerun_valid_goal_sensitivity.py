from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import h5py
import gymnasium as gym
import mani_skill  # noqa: F401
import numpy as np
import torch

from hcl_poc.config import load_config
from hcl_poc.learned_interface import _low_condition_array
from hcl_poc.low_level_rl import DirectLowActorCritic, ResidualActorCritic, _load_frozen
from hcl_poc.rl_rerun import (
    _encode_rerun_frames,
    _load_low_flow_base,
    _low_flow_base_action,
    _residual_action_from_raw,
    _residual_condition_array,
    _rerun_base_config,
    _vector_dataset_path,
)
from hcl_poc.rl import _rl_backend
from hcl_poc.utils import default_device, write_json


def _summary(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p90": float(np.quantile(values, 0.9)),
        "max": float(np.max(values)),
    }


def _parse_policy(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("policy must be LABEL=CHECKPOINT")
    label, path = value.split("=", 1)
    if not label:
        raise argparse.ArgumentTypeError("policy label must be non-empty")
    checkpoint = Path(path)
    if not checkpoint.exists():
        raise argparse.ArgumentTypeError(f"checkpoint does not exist: {checkpoint}")
    return label, checkpoint


def _load_policy(
    checkpoint_path: Path,
    frozen: Any,
    device: torch.device,
) -> tuple[str, Any, dict[str, Any], Any | None, dict[str, Any] | None]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    recipe = checkpoint["recipe"]
    method = str(recipe.get("method", ""))
    is_direct = method.startswith("r3_direct")
    base_policy = "deterministic" if is_direct else str(recipe.get("base_policy", ""))
    flow_model = None
    flow_checkpoint = None
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
        if base_policy == "flow":
            flow_path = recipe.get("flow_checkpoint")
            if not flow_path:
                raise ValueError("R2 checkpoint is missing flow_checkpoint")
            flow_model, flow_checkpoint = _load_low_flow_base(Path(flow_path), device)
        elif base_policy != "deterministic":
            raise ValueError(f"Unknown base policy: {base_policy}")
        agent = ResidualActorCritic(
            int(checkpoint["condition_dim"]),
            width=int(recipe["actor_critic_width"]),
            depth=int(recipe["actor_critic_depth"]),
            initial_logstd=float(recipe["initial_logstd"]),
        ).to(device)
    agent.load_state_dict(checkpoint["agent"])
    agent.eval()
    return base_policy, agent, recipe, flow_model, flow_checkpoint


@torch.inference_mode()
def _policy_action(
    label: str,
    condition: torch.Tensor,
    base_action: torch.Tensor,
    current_z: np.ndarray,
    goal_z: np.ndarray,
    previous_action: np.ndarray,
    remaining: np.ndarray,
    action_low: torch.Tensor,
    action_high: torch.Tensor,
    policy: tuple[str, Any, dict[str, Any], Any | None, dict[str, Any] | None] | None,
) -> torch.Tensor:
    if policy is None:
        return torch.clamp(base_action, action_low, action_high)
    base_policy, agent, recipe, _flow_model, _flow_checkpoint = policy
    method = str(recipe.get("method", ""))
    if method.startswith("r3_direct"):
        return torch.clamp(agent.mean_action(condition), action_low, action_high)
    residual_condition_mode = str(recipe.get("residual_condition_mode", "full"))
    residual_condition_np = _residual_condition_array(
        mode=residual_condition_mode,
        full_condition=condition.detach().cpu().numpy().astype(np.float32),
        current_z=current_z,
        goal_z=goal_z,
        previous_action=previous_action,
        remaining=remaining,
    )
    residual_condition = torch.from_numpy(residual_condition_np).to(condition.device).float()
    raw_action, _logprob, _entropy, _value = agent.get_action_and_value(
        residual_condition,
        deterministic=True,
    )
    alpha = float(recipe.get("alpha", 0.0))
    residual_action_mode = str(recipe.get("residual_action_mode", "additive"))
    _residual, _unclipped, action = _residual_action_from_raw(
        base_action,
        raw_action,
        alpha,
        action_low,
        action_high,
        residual_action_mode,
    )
    return action


def _sample_batch(
    h5: h5py.File,
    keys: list[str],
    max_steps: int,
    horizons: list[int],
    samples: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, dict[int, np.ndarray], np.ndarray]:
    max_horizon = max(horizons)
    current_frames: list[np.ndarray] = []
    future_frames: dict[int, list[np.ndarray]] = {horizon: [] for horizon in horizons}
    previous_actions: list[np.ndarray] = []
    for _ in range(samples):
        key = str(rng.choice(keys))
        group = h5[key]
        env_index = int(rng.integers(0, int(group.attrs["num_envs"])))
        timestep = int(rng.integers(0, max_steps - max_horizon + 1))
        current_frames.append(
            np.concatenate(
                [
                    np.asarray(group["dino"][timestep, env_index], dtype=np.float32),
                    np.asarray(group["proprio"][timestep, env_index], dtype=np.float32),
                ],
                axis=-1,
            )
        )
        for horizon in horizons:
            future_frames[horizon].append(
                np.concatenate(
                    [
                        np.asarray(
                            group["dino"][timestep + horizon, env_index],
                            dtype=np.float32,
                        ),
                        np.asarray(
                            group["proprio"][timestep + horizon, env_index],
                            dtype=np.float32,
                        ),
                    ],
                    axis=-1,
                )
            )
        previous_actions.append(
            np.asarray(group["previous_executed_actions"][timestep, env_index], dtype=np.float32)
        )
    return (
        np.stack(current_frames, axis=0),
        {horizon: np.stack(frames, axis=0) for horizon, frames in future_frames.items()},
        np.stack(previous_actions, axis=0),
    )


@torch.inference_mode()
def run_valid_goal_sensitivity(args: argparse.Namespace) -> Path:
    config = load_config(args.config)
    dataset_path = Path(args.dataset) if args.dataset else _vector_dataset_path(config)
    if not dataset_path.exists():
        raise FileNotFoundError(dataset_path)
    horizons = [int(part) for part in args.horizons.split(",")]
    if len(horizons) < 2 or any(horizon <= 0 for horizon in horizons):
        raise ValueError("--horizons must contain at least two positive integers")
    if args.samples <= 0 or args.batch_size <= 0:
        raise ValueError("--samples and --batch-size must be positive")

    device = default_device()
    frozen = _load_frozen(_rerun_base_config(config), args.n_demo, args.seed, device)
    env = gym.make(
        config.get("env_id"),
        obs_mode="rgb+state",
        control_mode=config.get("control_mode"),
        reward_mode="normalized_dense",
        render_mode=None,
        sim_backend=_rl_backend(config),
        num_envs=1,
        reconfiguration_freq=config.get("rl.eval_reconfiguration_freq", 1),
    )
    try:
        single_action_space = getattr(env, "single_action_space", env.action_space)
        action_low = torch.as_tensor(
            np.asarray(single_action_space.low, dtype=np.float32),
            device=device,
        )
        action_high = torch.as_tensor(
            np.asarray(single_action_space.high, dtype=np.float32),
            device=device,
        )
    finally:
        env.close()
    policies: dict[str, tuple[str, Any, dict[str, Any], Any | None, dict[str, Any] | None] | None] = {
        "frozen": None
    }
    for label, path in args.policy:
        if label in policies:
            raise ValueError(f"Duplicate policy label: {label}")
        policies[label] = _load_policy(path, frozen, device)

    rng = np.random.default_rng(args.seed + 777_000)
    with h5py.File(dataset_path, "r") as h5:
        meta = h5["meta"].attrs
        max_steps = int(meta["max_steps"])
        keys = sorted(key for key in h5.keys() if key.startswith("batch_"))
        remaining = np.ones((args.batch_size, 1), dtype=np.float32)
        action_by_policy: dict[str, dict[int, list[np.ndarray]]] = {
            label: {horizon: [] for horizon in horizons} for label in policies
        }
        goal_by_horizon: dict[int, list[np.ndarray]] = {horizon: [] for horizon in horizons}
        processed = 0
        while processed < args.samples:
            batch_samples = min(args.batch_size, args.samples - processed)
            current_frames, future_frames, previous_actions = _sample_batch(
                h5,
                keys,
                max_steps,
                horizons,
                batch_samples,
                rng,
            )
            current_z = _encode_rerun_frames(frozen, current_frames, device)
            normalized_frames = frozen.frame_norm.transform(current_frames)
            normalized_previous = frozen.action_norm.transform(previous_actions)
            for horizon in horizons:
                goal_z = _encode_rerun_frames(frozen, future_frames[horizon], device)
                goal_by_horizon[horizon].append(goal_z)
                condition_np = _low_condition_array(
                    normalized_frames,
                    current_z,
                    goal_z,
                    normalized_previous,
                    remaining[:batch_samples],
                    frozen.conditioning,
                )
                condition = torch.from_numpy(condition_np).to(device).float()
                normalized_base = frozen.low_model(condition)
                base_action = torch.from_numpy(
                    frozen.action_norm.inverse(
                        normalized_base.cpu().numpy().astype(np.float32)
                    )
                ).to(device)
                for label, policy in policies.items():
                    action = _policy_action(
                        label,
                        condition,
                        base_action,
                        current_z,
                        goal_z,
                        normalized_previous,
                        remaining[:batch_samples],
                        action_low,
                        action_high,
                        policy,
                    )
                    action_by_policy[label][horizon].append(action.cpu().numpy())
            processed += batch_samples

    goal_arrays = {
        horizon: np.concatenate(chunks, axis=0) for horizon, chunks in goal_by_horizon.items()
    }
    action_arrays = {
        label: {
            horizon: np.concatenate(chunks, axis=0)
            for horizon, chunks in horizon_chunks.items()
        }
        for label, horizon_chunks in action_by_policy.items()
    }
    reference_horizon = args.reference_horizon or horizons[len(horizons) // 2]
    if reference_horizon not in horizons:
        raise ValueError("--reference-horizon must be included in --horizons")

    goal_pairs: dict[str, dict[str, float]] = {}
    for horizon in horizons:
        if horizon == reference_horizon:
            continue
        pair = f"k{horizon}_vs_k{reference_horizon}"
        goal_l2 = np.linalg.norm(goal_arrays[horizon] - goal_arrays[reference_horizon], axis=-1)
        goal_pairs[pair] = _summary(goal_l2)

    policy_results: dict[str, Any] = {}
    for label, horizon_actions in action_arrays.items():
        pair_results: dict[str, Any] = {}
        for horizon in horizons:
            if horizon == reference_horizon:
                continue
            pair = f"k{horizon}_vs_k{reference_horizon}"
            action_l2 = np.linalg.norm(
                horizon_actions[horizon] - horizon_actions[reference_horizon],
                axis=-1,
            )
            goal_l2 = np.linalg.norm(
                goal_arrays[horizon] - goal_arrays[reference_horizon],
                axis=-1,
            )
            ratio = action_l2 / np.maximum(goal_l2, 1e-8)
            pair_results[pair] = {
                "action_l2": _summary(action_l2),
                "action_l2_per_goal_l2": _summary(ratio),
            }
        if len(horizons) >= 3:
            near = horizons[0]
            far = horizons[-1]
            action_l2 = np.linalg.norm(horizon_actions[near] - horizon_actions[far], axis=-1)
            pair_results[f"k{near}_vs_k{far}"] = {
                "action_l2": _summary(action_l2),
                "action_l2_per_goal_l2": _summary(
                    action_l2
                    / np.maximum(
                        np.linalg.norm(goal_arrays[near] - goal_arrays[far], axis=-1),
                        1e-8,
                    )
                ),
            }
        policy_results[label] = pair_results

    result = {
        "method": "rl_rerun_valid_goal_sensitivity",
        "dataset": str(dataset_path),
        "n_demo": args.n_demo,
        "seed": args.seed,
        "samples": args.samples,
        "horizons": horizons,
        "reference_horizon": reference_horizon,
        "remaining_fraction": 1.0,
        "goal_l2": goal_pairs,
        "policies": policy_results,
        "policy_checkpoints": {
            label: str(path) for label, path in args.policy
        },
        "interpretation": (
            "Compares deterministic one-step actions for the same current state "
            "under same-trajectory future goals at nearby horizons."
        ),
    }
    output = args.output or Path("rl_rerun_valid_goal_sensitivity.json")
    write_json(output, result)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/pusht_incremental.yaml")
    parser.add_argument("--dataset")
    parser.add_argument("--n-demo", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--samples", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--horizons", default="9,10,11")
    parser.add_argument("--reference-horizon", type=int)
    parser.add_argument("--policy", action="append", type=_parse_policy, default=[])
    parser.add_argument("--output", type=Path)
    path = run_valid_goal_sensitivity(parser.parse_args())
    print(path)


if __name__ == "__main__":
    main()
