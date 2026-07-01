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

from hcl_poc.rl import _rl_paths, load_ppo_agent
from hcl_poc.utils import default_device, ensure_dir, set_seed, write_json


def _runner_args(args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(
        config=args.config,
        dataset=args.dataset,
        num_envs=args.num_envs,
        horizon=args.horizon,
        seed=args.seed,
        reward_mode="progress_terminal",
        reward_distance_source="true_tcp",
        dpsi_checkpoint=None,
        dpsi_target_scale=1000.0,
        bc_low_checkpoint=args.bc_low,
        terminal_weight=1.0,
        distance_progress_weight=1.0,
        width=256,
        depth=2,
        initial_logstd=-1.0,
        learning_rate=3e-4,
    )


@torch.inference_mode()
def evaluate_policy(
    runner: LocalTcpPpo,
    refs: list[tuple[str, int]],
    low_kind: str,
    low_path: Path,
    *,
    shuffled_goal: bool,
) -> dict[str, Any]:
    device = default_device()
    if low_kind == "bc1800":
        low_model, low_payload = _load_bc_low(low_path, device)
    else:
        low_model, low_payload = _load_rl_low(low_path, device)
    teacher = load_ppo_agent(_rl_paths(runner.config).best, device)
    action_low_np = np.asarray(runner.env.single_action_space.low, dtype=np.float32)
    action_high_np = np.asarray(runner.env.single_action_space.high, dtype=np.float32)
    action_low = torch.as_tensor(action_low_np, device=device, dtype=torch.float32)
    action_high = torch.as_tensor(action_high_np, device=device, dtype=torch.float32)
    initial_distances = []
    terminal_distances = []
    terminal_errors = []
    reductions = []
    action_l2 = []
    action_saturated = []
    teacher_maes = []
    for ref in refs:
        runner.reset_local_episode(ref)
        goal = runner.goal.copy()
        if shuffled_goal:
            goal = goal[np.roll(np.arange(runner.num_envs), 1)]
        start_distance = _distance(runner.current_state, goal)
        initial_distances.extend(start_distance.tolist())
        previous_action_raw = runner.action_norm.inverse(runner.previous_action_norm).astype(
            np.float32
        )
        for step in range(runner.horizon):
            state = runner.current_state.copy()
            remaining = np.full(
                runner.num_envs,
                max(runner.horizon - step, 1),
                dtype=np.float32,
            )
            raw_action = _low_action(
                low_kind,
                low_model,
                low_payload,
                state,
                goal[:, :3],
                previous_action_raw,
                remaining,
                runner.horizon,
                runner.control_freq,
                device,
            )
            teacher_action = torch.clamp(
                teacher.actor_mean(torch.from_numpy(state).to(device).float()),
                action_low,
                action_high,
            ).cpu().numpy()
            teacher_maes.extend(
                np.mean(np.abs(raw_action - teacher_action), axis=-1).astype(float).tolist()
            )
            clipped = np.clip(raw_action, action_low_np, action_high_np).astype(np.float32)
            action_saturated.extend(
                np.any((raw_action < action_low_np) | (raw_action > action_high_np), axis=-1)
                .astype(np.float32)
                .tolist()
            )
            action_l2.extend(np.linalg.norm(clipped, axis=-1).astype(float).tolist())
            previous_distance = _distance(runner.current_state, goal)
            _reward, _done, _metrics = runner.step_local(
                torch.from_numpy(clipped).to(device).float(),
                previous_distance,
                auto_reset=False,
            )
            previous_action_raw = clipped
        terminal = _distance(runner.current_state, goal)
        terminal_distances.extend(terminal.tolist())
        terminal_errors.extend(
            np.linalg.norm(runner.current_state[:, 14:17] - goal[:, :3], axis=-1)
            .astype(float)
            .tolist()
        )
        reductions.extend((start_distance - terminal).tolist())
    terminal_np = np.asarray(terminal_distances, dtype=np.float32)
    initial_np = np.asarray(initial_distances, dtype=np.float32)
    reduction_np = np.asarray(reductions, dtype=np.float32)
    return {
        "low_policy": low_kind,
        "low_checkpoint": str(low_path),
        "shuffled_goal": bool(shuffled_goal),
        "local_episodes": int(len(terminal_np)),
        "initial_distance_sq_mean": float(np.mean(initial_np)),
        "terminal_distance_sq_mean": float(np.mean(terminal_np)),
        "terminal_tcp_error_m_mean": float(np.mean(terminal_errors)),
        "distance_sq_reduction_mean": float(np.mean(reduction_np)),
        "goal_reach_rate_eps": float(np.mean(terminal_np <= float(runner.args.success_epsilon))),
        "p50_terminal_distance_sq": float(np.quantile(terminal_np, 0.50)),
        "p90_terminal_distance_sq": float(np.quantile(terminal_np, 0.90)),
        "p99_terminal_distance_sq": float(np.quantile(terminal_np, 0.99)),
        "fraction_improved_from_start": float(np.mean(reduction_np > 0.0)),
        "action_saturation": float(np.mean(action_saturated)),
        "action_l2_mean": float(np.mean(action_l2)),
        "teacher_action_mae": float(np.mean(teacher_maes)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/pusht_incremental.yaml")
    parser.add_argument("--dataset")
    parser.add_argument("--num-envs", type=int, default=4096)
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval-refs", type=int, default=1)
    parser.add_argument("--success-epsilon", type=float, default=0.0025)
    parser.add_argument("--include-shuffled", action="store_true")
    parser.add_argument(
        "--bc-low",
        default="artifacts/incremental/pre_rl/phase_c/k10/seed0/time_conditioned_tcp.pt",
    )
    parser.add_argument(
        "--run2-low",
        default="results/incremental/rl_reachability_debug/run2_privileged_tcp/privileged_tcp_ppo_progress_terminal_n4096_seed0/latest.pt",
    )
    parser.add_argument(
        "--run5-low",
        default="results/incremental/rl_reachability_debug/run5_tcp_dpsi_ppo/privileged_tcp_ppo_progress_terminal_n4096_seed0/latest.pt",
    )
    parser.add_argument(
        "--output",
        default="results/incremental/rl_reachability_debug/run5_local_policy_compare.json",
    )
    args = parser.parse_args()
    set_seed(args.seed + 4_130_000)
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
        rows = []
        for shuffled in ([False, True] if args.include_shuffled else [False]):
            for low_kind, low_path in policies:
                rows.append(
                    evaluate_policy(
                        runner,
                        refs,
                        low_kind,
                        low_path,
                        shuffled_goal=shuffled,
                    )
                )
        payload = {
            "run": "rl_reachability_debug_local_low_policy_compare",
            "eval_refs": int(args.eval_refs),
            "num_envs": int(runner.num_envs),
            "horizon": int(runner.horizon),
            "success_epsilon": float(args.success_epsilon),
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
