#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
from torch import nn
from tqdm import trange

sys.path.append(str(Path(__file__).resolve().parent))

from rl_reachability_goal_full_success_eval import _load_bc_low, _load_high, _low_action
from rl_reachability_full_deployment_reachability_eval import _oracle_future_state
from rl_reachability_privileged_tcp_ppo import _obs_state_np
from rl_reachability_tcp_full_success_eval import _load_rl_low, _to_numpy

from hcl_poc.config import load_config
from hcl_poc.incremental import _pre_rl_phase_b_goal
from hcl_poc.models import MLP
from hcl_poc.rl import _rl_paths, load_ppo_agent
from hcl_poc.rl_rerun import _make_benchmark_env
from hcl_poc.utils import Standardizer, default_device, ensure_dir


GOAL_TYPE = "full"
POLICY_PATH_DEFAULTS = {
    "phase_c_full_bc": "artifacts/incremental/pre_rl/phase_c/k10/seed0/time_conditioned_full.pt",
    "run22_long_full_ppo": (
        "results/incremental/rl_reachability_debug/run22_full_goal_recomputed_penalty10_continue2_u1000/"
        "privileged_full_ppo_progress_terminal_n4096_seed0/latest.pt"
    ),
    "run25_bc_warm_start_ppo": (
        "results/incremental/rl_reachability_debug/run25_full_bc_warm_start_learned_high_mixture_penalty10_u250/"
        "privileged_full_ppo_progress_terminal_n4096_seed0/latest.pt"
    ),
    "run26_bc_prior_ppo": (
        "results/incremental/rl_reachability_debug/run26_full_bc_prior_learned_high_mixture_u250/"
        "privileged_full_ppo_progress_terminal_n4096_seed0/latest.pt"
    ),
}


def _load_low_policy(
    policy: str,
    path: Path,
    device: torch.device,
) -> tuple[str, nn.Module, dict[str, Any]]:
    if policy == "phase_c_full_bc":
        model, payload = _load_bc_low(path, GOAL_TYPE, device)
        return "phase_b_bc", model, payload
    model, payload = _load_rl_low(path, device)
    if payload.get("goal_type") != GOAL_TYPE:
        raise ValueError(f"{path} is not a full-goal low-level checkpoint")
    return policy, model, payload


def _goal_to_pseudo_future_state(current_state: np.ndarray, full_goal: np.ndarray) -> np.ndarray:
    future = np.asarray(current_state, dtype=np.float32).copy()
    goal = np.asarray(full_goal, dtype=np.float32)
    future[:, 24:26] = goal[:, 0:2]
    yaw = np.arctan2(goal[:, 2], goal[:, 3]).astype(np.float32)
    future[:, 14:17] = goal[:, 7:10]
    future[:, :14] = goal[:, 13:27]
    future[:, 27:31] = 0.0
    future[:, 27] = np.cos(0.5 * yaw)
    future[:, 30] = np.sin(0.5 * yaw)
    return future.astype(np.float32)


@torch.inference_mode()
def _predict_high_goal(
    high_model: nn.Module,
    high_payload: dict[str, Any],
    state: np.ndarray,
    previous_action: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    input_norm = Standardizer.from_state_dict(high_payload["input_norm"])
    target_norm = Standardizer.from_state_dict(high_payload["target_norm"])
    action_norm = Standardizer.from_state_dict(high_payload["action_norm"])
    previous_norm = action_norm.transform(previous_action)
    condition = np.concatenate([state, previous_norm], axis=-1).astype(np.float32)
    predicted = high_model(
        torch.from_numpy(input_norm.transform(condition)).to(device).float()
    ).cpu().numpy()
    return target_norm.inverse(predicted).astype(np.float32)


def _copy_demo_batches(
    source_path: Path,
    h5: h5py.File,
    *,
    output_index: int,
    demo_batches: int,
) -> int:
    with h5py.File(source_path, "r") as source:
        keys = sorted(key for key in source.keys() if key.startswith("batch_"))
        if len(keys) < demo_batches:
            raise ValueError(f"{source_path} has only {len(keys)} batch groups")
        for key in keys[:demo_batches]:
            name = f"batch_{output_index:06d}"
            source.copy(key, h5, name=name)
            group = h5[name]
            group.attrs["source"] = "demo"
            output_index += 1
    return output_index


@torch.inference_mode()
def _collect_policy_batches(
    args: argparse.Namespace,
    h5: h5py.File,
    *,
    output_index: int,
    policy_name: str,
    low_path: Path,
    high_path: Path,
    source_label: str,
) -> int:
    config = load_config(args.config)
    device = default_device()
    high_model = None
    high_payload = None
    teacher = None
    if args.target_source == "learned_high":
        high_model, high_payload = _load_high(high_path, GOAL_TYPE, device)
    elif args.target_source == "teacher_oracle":
        teacher = load_ppo_agent(_rl_paths(config).best, device)
        teacher.eval()
    else:
        raise ValueError(f"Unknown target source: {args.target_source}")
    low_kind, low_model, low_payload = _load_low_policy(policy_name, low_path, device)
    horizon = int(args.horizon)
    max_steps = int(args.max_steps)
    num_envs = int(args.num_envs)
    control_freq = int(config.get("control_freq", 20))
    env = _make_benchmark_env(config, num_envs, "state")
    action_low = torch.as_tensor(env.single_action_space.low, device=device, dtype=torch.float32)
    action_high = torch.as_tensor(env.single_action_space.high, device=device, dtype=torch.float32)
    action_low_np = np.asarray(env.single_action_space.low, dtype=np.float32)
    action_high_np = np.asarray(env.single_action_space.high, dtype=np.float32)
    try:
        for batch in trange(args.deployed_batches_per_policy, desc=f"collect {source_label}"):
            batch_seed = int(args.seed_start + output_index)
            branch_reset_seeds = [batch_seed + env_index for env_index in range(num_envs)]
            obs, _info = env.reset(seed=batch_seed)
            states: list[np.ndarray] = []
            raw_actions: list[np.ndarray] = []
            executed_actions: list[np.ndarray] = []
            previous_actions: list[np.ndarray] = []
            rewards: list[np.ndarray] = []
            terminated_flags: list[np.ndarray] = []
            truncated_flags: list[np.ndarray] = []
            success_flags: list[np.ndarray] = []
            target_future_states = np.zeros((max_steps + 1, num_envs, 31), dtype=np.float32)
            valid_starts: list[int] = []
            previous = np.zeros((num_envs, 3), dtype=np.float32)
            target_future_state = _obs_state_np(obs).copy()
            states.append(target_future_state.copy())
            for step in range(max_steps):
                state = _obs_state_np(obs)
                if step % horizon == 0 and step + horizon <= max_steps:
                    if args.target_source == "learned_high":
                        if high_model is None or high_payload is None:
                            raise RuntimeError("Missing high-level predictor")
                        predicted_goal = _predict_high_goal(
                            high_model,
                            high_payload,
                            state,
                            previous,
                            device,
                        )
                        target_future_state = _goal_to_pseudo_future_state(
                            state,
                            predicted_goal,
                        )
                    else:
                        if teacher is None:
                            raise RuntimeError("Missing teacher policy")
                        target_future_state = _oracle_future_state(
                            config,
                            teacher,
                            env.unwrapped.get_state_dict(),
                            branch_reset_seeds,
                            horizon,
                            action_low,
                            action_high,
                            device,
                        )
                    valid_starts.append(step)
                target_future_states[step] = target_future_state
                remaining_value = max(horizon - (step % horizon), 1)
                remaining = np.full(num_envs, remaining_value, dtype=np.float32)
                goal = _pre_rl_phase_b_goal(
                    state,
                    target_future_state,
                    remaining_value,
                    control_freq,
                    GOAL_TYPE,
                ).astype(np.float32)
                raw_action = _low_action(
                    low_kind,
                    low_model,
                    low_payload,
                    state,
                    goal,
                    previous,
                    remaining,
                    horizon,
                    GOAL_TYPE,
                    device,
                )
                clipped = np.clip(raw_action, action_low_np, action_high_np).astype(np.float32)
                next_obs, reward, terminated, truncated, info = env.step(
                    torch.from_numpy(clipped).to(device).float()
                )
                raw_actions.append(raw_action.astype(np.float32))
                executed_actions.append(clipped)
                previous_actions.append(previous.copy())
                rewards.append(_to_numpy(reward).reshape(-1).astype(np.float32))
                terminated_flags.append(_to_numpy(terminated).reshape(-1).astype(np.bool_))
                truncated_flags.append(_to_numpy(truncated).reshape(-1).astype(np.bool_))
                success_flags.append(
                    _to_numpy(info.get("success", np.zeros(num_envs, dtype=np.bool_)))
                    .reshape(-1)
                    .astype(np.bool_)
                )
                previous = clipped
                obs = next_obs
                states.append(_obs_state_np(obs).copy())
            target_future_states[max_steps] = target_future_state
            name = f"batch_{output_index:06d}"
            group = h5.create_group(name)
            group.attrs["batch_seed"] = batch_seed
            group.attrs["num_envs"] = num_envs
            group.attrs["max_steps"] = max_steps
            group.attrs["source"] = source_label
            group.attrs["collector_policy"] = policy_name
            group.attrs["high_checkpoint"] = str(high_path)
            group.create_dataset("observations_state", data=np.stack(states), compression="gzip")
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
            group.create_dataset("terminated", data=np.stack(terminated_flags), compression="gzip")
            group.create_dataset("truncated", data=np.stack(truncated_flags), compression="gzip")
            group.create_dataset("success", data=np.stack(success_flags), compression="gzip")
            group.create_dataset(
                "target_future_states",
                data=target_future_states,
                compression="gzip",
            )
            group.create_dataset(
                "valid_starts",
                data=np.asarray(valid_starts, dtype=np.int64),
                compression="gzip",
            )
            output_index += 1
    finally:
        env.close()
    return output_index


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/pusht_incremental.yaml")
    parser.add_argument(
        "--demo-dataset",
        default="data/rl_rerun/pusht_vector_state_demos_n4096_b8.h5",
    )
    parser.add_argument(
        "--high-checkpoint",
        default="artifacts/incremental/rl_reachability_debug/full_goal_high_predictor/seed0/predictor.pt",
    )
    parser.add_argument("--phase-c-full-bc", default=POLICY_PATH_DEFAULTS["phase_c_full_bc"])
    parser.add_argument("--run22-low", default=POLICY_PATH_DEFAULTS["run22_long_full_ppo"])
    parser.add_argument("--run25-low", default=POLICY_PATH_DEFAULTS["run25_bc_warm_start_ppo"])
    parser.add_argument("--run26-low", default=POLICY_PATH_DEFAULTS["run26_bc_prior_ppo"])
    parser.add_argument(
        "--collector-policies",
        nargs="+",
        choices=list(POLICY_PATH_DEFAULTS),
        default=["phase_c_full_bc", "run22_long_full_ppo"],
    )
    parser.add_argument(
        "--output",
        default="data/rl_reachability_debug/full_reset_mixture_demo8_bc4_run22_4.h5",
    )
    parser.add_argument("--num-envs", type=int, default=4096)
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=60)
    parser.add_argument("--demo-batches", type=int, default=8)
    parser.add_argument("--deployed-batches-per-policy", type=int, default=4)
    parser.add_argument("--seed-start", type=int, default=4_520_000)
    parser.add_argument(
        "--target-source",
        choices=["learned_high", "teacher_oracle"],
        default="learned_high",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists() and not args.force:
        print(output)
        return
    ensure_dir(output.parent)
    tmp = output.with_suffix(".tmp.h5")
    tmp.unlink(missing_ok=True)
    with h5py.File(tmp, "w") as h5:
        meta = h5.create_group("meta")
        meta.attrs["source"] = "full_state_reset_mixture_learned_high"
        meta.attrs["goal_type"] = GOAL_TYPE
        meta.attrs["num_envs"] = int(args.num_envs)
        meta.attrs["horizon"] = int(args.horizon)
        meta.attrs["max_steps"] = int(args.max_steps)
        meta.attrs["demo_batches"] = int(args.demo_batches)
        meta.attrs["deployed_batches_per_policy"] = int(args.deployed_batches_per_policy)
        meta.attrs["collector_policies"] = ",".join(args.collector_policies)
        meta.attrs["mixture"] = (
            f"{int(args.demo_batches)} demo batches + "
            f"{int(args.deployed_batches_per_policy)} deployed batches per collector"
        )
        meta.attrs["target_source"] = str(args.target_source)
        meta.attrs["high_checkpoint"] = str(args.high_checkpoint)
        output_index = _copy_demo_batches(
            Path(args.demo_dataset),
            h5,
            output_index=0,
            demo_batches=int(args.demo_batches),
        )
        policy_paths = {
            "phase_c_full_bc": Path(args.phase_c_full_bc),
            "run22_long_full_ppo": Path(args.run22_low),
            "run25_bc_warm_start_ppo": Path(args.run25_low),
            "run26_bc_prior_ppo": Path(args.run26_low),
        }
        for policy_name in args.collector_policies:
            output_index = _collect_policy_batches(
                args,
                h5,
                output_index=output_index,
                policy_name=policy_name,
                low_path=policy_paths[policy_name],
                high_path=Path(args.high_checkpoint),
                source_label=f"{policy_name}_deployed",
            )
        meta.attrs["batches"] = int(output_index)
    tmp.replace(output)
    print(output)


if __name__ == "__main__":
    main()
