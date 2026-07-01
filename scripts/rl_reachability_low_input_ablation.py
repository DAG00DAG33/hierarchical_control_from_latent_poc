#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

sys.path.append(str(Path(__file__).resolve().parent))

from rl_reachability_privileged_tcp_ppo import LocalTcpPpo, _distance
from rl_reachability_tcp_full_success_eval import _load_bc_low, _load_rl_low, _low_action
from rl_reachability_tcp_local_policy_compare import _runner_args

from hcl_poc.utils import default_device, ensure_dir, set_seed, write_json


ABLATIONS = (
    "live",
    "cached_start_observation",
    "cached_previous_action",
    "constant_remaining_time",
    "shuffled_goal",
    "shuffled_observation",
)


def _summary(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float32)
    return {
        "mean": float(np.mean(arr)),
        "p50": float(np.quantile(arr, 0.50)),
        "p90": float(np.quantile(arr, 0.90)),
        "p99": float(np.quantile(arr, 0.99)),
    }


def _load_low(
    low_kind: str,
    low_path: Path,
    device: torch.device,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    if low_kind == "bc1800":
        return _load_bc_low(low_path, device)
    return _load_rl_low(low_path, device)


@torch.inference_mode()
def evaluate_ablation(
    runner: LocalTcpPpo,
    refs: list[tuple[str, int]],
    low_kind: str,
    low_path: Path,
    ablation: str,
) -> dict[str, Any]:
    if ablation not in ABLATIONS:
        raise ValueError(f"unknown ablation: {ablation}")
    device = default_device()
    low_model, low_payload = _load_low(low_kind, low_path, device)
    action_low_np = np.asarray(runner.env.single_action_space.low, dtype=np.float32)
    action_high_np = np.asarray(runner.env.single_action_space.high, dtype=np.float32)
    original_initial_distances: list[float] = []
    command_initial_distances: list[float] = []
    original_terminal_distances: list[float] = []
    command_terminal_distances: list[float] = []
    original_tcp_errors: list[float] = []
    command_tcp_errors: list[float] = []
    original_reductions: list[float] = []
    command_reductions: list[float] = []
    action_l2: list[float] = []
    action_saturation: list[float] = []
    action_delta_from_live: list[float] = []
    permutation = np.roll(np.arange(runner.num_envs), 1)
    for ref in refs:
        runner.reset_local_episode(ref)
        original_goal = runner.goal.copy()
        command_goal = original_goal.copy()
        if ablation == "shuffled_goal":
            command_goal = original_goal[permutation].copy()
        start_state = runner.current_state.copy()
        cached_previous_action = runner.action_norm.inverse(runner.previous_action_norm).astype(
            np.float32
        )
        previous_action_raw = cached_previous_action.copy()
        original_start_distance = _distance(start_state, original_goal)
        command_start_distance = _distance(start_state, command_goal)
        original_initial_distances.extend(original_start_distance.tolist())
        command_initial_distances.extend(command_start_distance.tolist())
        for step in range(runner.horizon):
            live_state = runner.current_state.copy()
            live_remaining = np.full(
                runner.num_envs,
                max(runner.horizon - step, 1),
                dtype=np.float32,
            )
            policy_state = live_state
            policy_previous = previous_action_raw
            policy_remaining = live_remaining
            if ablation == "cached_start_observation":
                policy_state = start_state
            elif ablation == "cached_previous_action":
                policy_previous = cached_previous_action
            elif ablation == "constant_remaining_time":
                policy_remaining = np.full(runner.num_envs, runner.horizon, dtype=np.float32)
            elif ablation == "shuffled_observation":
                policy_state = live_state[permutation]
            live_raw_action = _low_action(
                low_kind,
                low_model,
                low_payload,
                live_state,
                original_goal[:, :3],
                previous_action_raw,
                live_remaining,
                runner.horizon,
                runner.control_freq,
                device,
            )
            raw_action = _low_action(
                low_kind,
                low_model,
                low_payload,
                policy_state,
                command_goal[:, :3],
                policy_previous,
                policy_remaining,
                runner.horizon,
                runner.control_freq,
                device,
            )
            action_delta_from_live.extend(
                np.linalg.norm(raw_action - live_raw_action, axis=-1).astype(float).tolist()
            )
            clipped = np.clip(raw_action, action_low_np, action_high_np).astype(np.float32)
            action_saturation.extend(
                np.any((raw_action < action_low_np) | (raw_action > action_high_np), axis=-1)
                .astype(np.float32)
                .tolist()
            )
            action_l2.extend(np.linalg.norm(clipped, axis=-1).astype(float).tolist())
            previous_distance = _distance(runner.current_state, original_goal)
            _reward, _done, _metrics = runner.step_local(
                torch.from_numpy(clipped).to(device).float(),
                previous_distance,
                auto_reset=False,
            )
            previous_action_raw = clipped
        terminal_state = runner.current_state.copy()
        original_terminal = _distance(terminal_state, original_goal)
        command_terminal = _distance(terminal_state, command_goal)
        original_terminal_distances.extend(original_terminal.tolist())
        command_terminal_distances.extend(command_terminal.tolist())
        original_tcp_errors.extend(
            np.linalg.norm(terminal_state[:, 14:17] - original_goal[:, :3], axis=-1)
            .astype(float)
            .tolist()
        )
        command_tcp_errors.extend(
            np.linalg.norm(terminal_state[:, 14:17] - command_goal[:, :3], axis=-1)
            .astype(float)
            .tolist()
        )
        original_reductions.extend((original_start_distance - original_terminal).tolist())
        command_reductions.extend((command_start_distance - command_terminal).tolist())
    original_terminal_np = np.asarray(original_terminal_distances, dtype=np.float32)
    command_terminal_np = np.asarray(command_terminal_distances, dtype=np.float32)
    original_reduction_np = np.asarray(original_reductions, dtype=np.float32)
    command_reduction_np = np.asarray(command_reductions, dtype=np.float32)
    return {
        "low_policy": low_kind,
        "low_checkpoint": str(low_path),
        "ablation": ablation,
        "local_episodes": int(len(original_terminal_np)),
        "original_goal": {
            "initial_distance_sq": _summary(original_initial_distances),
            "terminal_distance_sq": _summary(original_terminal_distances),
            "terminal_tcp_error_m": _summary(original_tcp_errors),
            "distance_sq_reduction_mean": float(np.mean(original_reduction_np)),
            "goal_reach_rate_eps": float(
                np.mean(original_terminal_np <= float(runner.args.success_epsilon))
            ),
            "fraction_improved_from_start": float(np.mean(original_reduction_np > 0.0)),
        },
        "commanded_goal": {
            "initial_distance_sq": _summary(command_initial_distances),
            "terminal_distance_sq": _summary(command_terminal_distances),
            "terminal_tcp_error_m": _summary(command_tcp_errors),
            "distance_sq_reduction_mean": float(np.mean(command_reduction_np)),
            "goal_reach_rate_eps": float(
                np.mean(command_terminal_np <= float(runner.args.success_epsilon))
            ),
            "fraction_improved_from_start": float(np.mean(command_reduction_np > 0.0)),
        },
        "action": {
            "l2": _summary(action_l2),
            "saturation": float(np.mean(action_saturation)),
            "delta_from_live_l2": _summary(action_delta_from_live),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/pusht_incremental.yaml")
    parser.add_argument("--dataset")
    parser.add_argument("--num-envs", type=int, default=4096)
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval-refs", type=int, default=2)
    parser.add_argument("--success-epsilon", type=float, default=0.0025)
    parser.add_argument(
        "--bc-low",
        default="artifacts/incremental/pre_rl/phase_c/k10/seed0/time_conditioned_tcp.pt",
    )
    parser.add_argument(
        "--run2-low",
        default="results/incremental/rl_reachability_debug/run6_true_tcp_b8_u1000/privileged_tcp_ppo_progress_terminal_n4096_seed0/latest.pt",
    )
    parser.add_argument(
        "--run5-low",
        default="results/incremental/rl_reachability_debug/run7_dpsi_progress_b8_u1000/privileged_tcp_ppo_progress_n4096_seed0/latest.pt",
    )
    parser.add_argument(
        "--output",
        default="results/incremental/rl_reachability_debug/run7_low_input_ablation_b8_ref2.json",
    )
    args = parser.parse_args()
    set_seed(args.seed + 4_150_000)
    runner_args = _runner_args(args)
    runner_args.success_epsilon = args.success_epsilon
    runner = LocalTcpPpo(runner_args)
    try:
        refs = runner.sample_references(args.eval_refs, args.seed + 4_125_000)
        policies = [
            ("bc1800", Path(args.bc_low)),
            ("run2_true_tcp_ppo", Path(args.run2_low)),
            ("run5_dpsi_ppo", Path(args.run5_low)),
        ]
        rows = [
            evaluate_ablation(runner, refs, low_kind, low_path, ablation)
            for low_kind, low_path in policies
            for ablation in ABLATIONS
        ]
        payload = {
            "run": "rl_reachability_debug_low_input_ablation",
            "eval_refs": int(args.eval_refs),
            "num_envs": int(runner.num_envs),
            "horizon": int(runner.horizon),
            "success_epsilon": float(args.success_epsilon),
            "ablation_semantics": {
                "original_goal": "reference TCP endpoint from the local reset dataset",
                "commanded_goal": "goal actually given to the policy; differs only for shuffled_goal",
                "delta_from_live_l2": "L2 action difference versus live observation/original-goal condition on the same current state",
            },
            "rows": rows,
        }
        output = Path(args.output)
        ensure_dir(output.parent)
        write_json(output, payload)
        print(output)
    finally:
        runner.close()


if __name__ == "__main__":
    main()
