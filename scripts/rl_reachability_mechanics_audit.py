#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import torch

from hcl_poc.config import load_config
from hcl_poc.low_level_rl import HierarchyRollout, _load_frozen
from hcl_poc.utils import default_device, ensure_dir, write_json


def _action_from_condition(
    frozen: Any,
    condition_np: np.ndarray,
    action_low: torch.Tensor,
    action_high: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    with torch.inference_mode():
        condition = torch.from_numpy(condition_np).to(device).float()
        normalized = frozen.low_model(condition)
        action = torch.from_numpy(
            frozen.action_norm.inverse(normalized.cpu().numpy()).astype(np.float32)
        ).to(device)
        return torch.clamp(action, action_low, action_high)


def _condition_np(
    rollout: HierarchyRollout,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    condition, base_action, distance = rollout.current_condition()
    return (
        condition.detach().cpu().numpy().astype(np.float32),
        base_action.detach().cpu().numpy().astype(np.float32),
        distance.astype(np.float32),
    )


def _mean_l2(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.linalg.norm(left - right, axis=-1).mean())


def _toy_gae(
    rewards: np.ndarray,
    values: np.ndarray,
    next_value: float,
    done_after_step: np.ndarray,
    gamma: float,
    gae_lambda: float,
) -> np.ndarray:
    advantages = np.zeros_like(rewards, dtype=np.float64)
    last_gae = 0.0
    for step in reversed(range(len(rewards))):
        if step == len(rewards) - 1:
            next_nonterminal = 1.0 - float(done_after_step[step])
            following_value = next_value
        else:
            next_nonterminal = 1.0 - float(done_after_step[step])
            following_value = float(values[step + 1])
        delta = rewards[step] + gamma * following_value * next_nonterminal - values[step]
        last_gae = delta + gamma * gae_lambda * next_nonterminal * last_gae
        advantages[step] = last_gae
    return advantages + values


def _manual_local_terminal_returns(gamma: float, gae_lambda: float) -> list[float]:
    rewards = [0.0, 0.0, 1.0]
    values = [0.1, 0.2, 0.3]
    adv2 = rewards[2] - values[2]
    adv1 = rewards[1] + gamma * values[2] - values[1] + gamma * gae_lambda * adv2
    adv0 = rewards[0] + gamma * values[1] - values[0] + gamma * gae_lambda * adv1
    return [adv0 + values[0], adv1 + values[1], adv2 + values[2]]


def run_audit(args: argparse.Namespace) -> Path:
    config = load_config(args.config)
    device = default_device()
    frozen = _load_frozen(config, args.n_demo, args.seed, device, args.candidate)
    rollout = HierarchyRollout(
        config,
        frozen,
        args.num_envs,
        args.seed_start,
        device,
        distance_metric="raw_l2",
    )
    branches: dict[str, HierarchyRollout] = {}
    try:
        condition, _base_action, _distance, replan = rollout.condition()
        if not bool(np.all(replan)):
            raise RuntimeError("Expected all envs to replan at the first audit step")
        initial_condition = condition.detach().cpu().numpy().astype(np.float32)
        frame_dim = frozen.frame_dim
        prev_slice = slice(initial_condition.shape[1] - 4, initial_condition.shape[1] - 1)
        remaining_slice = slice(initial_condition.shape[1] - 1, initial_condition.shape[1])
        cached_start_frames = initial_condition[:, :frame_dim].copy()
        cached_previous = initial_condition[:, prev_slice].copy()
        cached_remaining = initial_condition[:, remaining_slice].copy()
        permutation = np.roll(np.arange(args.num_envs), 1)

        branch_modes = [
            "live",
            "cached_start_observation",
            "cached_previous_action",
            "constant_remaining_time",
            "shuffled_goal",
            "shuffled_observation",
        ]
        for mode in branch_modes:
            branch = HierarchyRollout(
                config,
                frozen,
                args.num_envs,
                args.seed_start + 100_000 + 10_000 * len(branches),
                device,
                distance_metric="raw_l2",
            )
            branch.copy_branch_from(rollout)
            if mode == "shuffled_goal":
                branch.held_goal = branch.held_goal[permutation].copy()
            branches[mode] = branch

        action_sensitivity: dict[str, list[float]] = {
            "cached_start_observation_action_l2": [],
            "cached_previous_action_l2": [],
            "constant_remaining_time_action_l2": [],
            "shuffled_goal_action_l2": [],
            "shuffled_observation_action_l2": [],
        }
        terminal: dict[str, dict[str, float]] = {}

        for _step in range(frozen.update_period):
            live_condition, live_base_action, _live_distance = _condition_np(rollout)
            live_action = torch.as_tensor(live_base_action, device=device)

            modified = live_condition.copy()
            modified[:, :frame_dim] = cached_start_frames
            cached_obs_action = _action_from_condition(
                frozen, modified, rollout.action_low, rollout.action_high, device
            ).detach().cpu().numpy()
            action_sensitivity["cached_start_observation_action_l2"].append(
                _mean_l2(cached_obs_action, live_base_action)
            )

            modified = live_condition.copy()
            modified[:, prev_slice] = cached_previous
            cached_prev_action = _action_from_condition(
                frozen, modified, rollout.action_low, rollout.action_high, device
            ).detach().cpu().numpy()
            action_sensitivity["cached_previous_action_l2"].append(
                _mean_l2(cached_prev_action, live_base_action)
            )

            modified = live_condition.copy()
            modified[:, remaining_slice] = cached_remaining
            const_remaining_action = _action_from_condition(
                frozen, modified, rollout.action_low, rollout.action_high, device
            ).detach().cpu().numpy()
            action_sensitivity["constant_remaining_time_action_l2"].append(
                _mean_l2(const_remaining_action, live_base_action)
            )

            shuffled_goal_condition = live_condition.copy()
            goal_slice = slice(frame_dim, frame_dim + frozen.goal_dim)
            shuffled_goal_condition[:, goal_slice] = shuffled_goal_condition[permutation, goal_slice]
            shuffled_goal_action = _action_from_condition(
                frozen,
                shuffled_goal_condition,
                rollout.action_low,
                rollout.action_high,
                device,
            ).detach().cpu().numpy()
            action_sensitivity["shuffled_goal_action_l2"].append(
                _mean_l2(shuffled_goal_action, live_base_action)
            )

            shuffled_obs_condition = live_condition.copy()
            shuffled_obs_condition[:, :frame_dim] = shuffled_obs_condition[
                permutation, :frame_dim
            ]
            shuffled_obs_action = _action_from_condition(
                frozen,
                shuffled_obs_condition,
                rollout.action_low,
                rollout.action_high,
                device,
            ).detach().cpu().numpy()
            action_sensitivity["shuffled_observation_action_l2"].append(
                _mean_l2(shuffled_obs_action, live_base_action)
            )

            previous_distance = rollout.distance(rollout.current_latent, rollout.held_goal)
            _reward, _done, _metrics = rollout.step(
                live_action,
                previous_distance,
                terminal_weight=0.0,
                distance_progress_weight=0.0,
                task_reward_weight=0.0,
                task_progress_weight=0.0,
                residual_penalty=np.zeros(args.num_envs, dtype=np.float32),
            )

            for mode, branch in branches.items():
                condition_np, base_action_np, distance_np = _condition_np(branch)
                action_np = base_action_np
                if mode == "cached_start_observation":
                    condition_np[:, :frame_dim] = cached_start_frames
                    action_np = _action_from_condition(
                        frozen, condition_np, branch.action_low, branch.action_high, device
                    ).detach().cpu().numpy()
                elif mode == "cached_previous_action":
                    condition_np[:, prev_slice] = cached_previous
                    action_np = _action_from_condition(
                        frozen, condition_np, branch.action_low, branch.action_high, device
                    ).detach().cpu().numpy()
                elif mode == "constant_remaining_time":
                    condition_np[:, remaining_slice] = cached_remaining
                    action_np = _action_from_condition(
                        frozen, condition_np, branch.action_low, branch.action_high, device
                    ).detach().cpu().numpy()
                elif mode == "shuffled_observation":
                    condition_np[:, :frame_dim] = condition_np[permutation, :frame_dim]
                    action_np = _action_from_condition(
                        frozen, condition_np, branch.action_low, branch.action_high, device
                    ).detach().cpu().numpy()
                action = torch.as_tensor(action_np, device=device)
                _reward, _done, metrics = branch.step(
                    action,
                    distance_np,
                    terminal_weight=0.0,
                    distance_progress_weight=0.0,
                    task_reward_weight=0.0,
                    task_progress_weight=0.0,
                    residual_penalty=np.zeros(args.num_envs, dtype=np.float32),
                )
                if bool(np.any(metrics["segment_end"])):
                    mask = metrics["segment_end"].astype(bool)
                    terminal[mode] = {
                        "terminal_selected_distance_mean": float(
                            metrics["next_distance"][mask].mean()
                        ),
                        "terminal_raw_distance_mean": float(
                            metrics["raw_next_distance"][mask].mean()
                        ),
                    }

        sensitivity_summary = {
            key: {
                "mean": float(np.mean(values)),
                "max": float(np.max(values)),
            }
            for key, values in action_sensitivity.items()
        }
    finally:
        rollout.close()
        for branch in branches.values():
            branch.close()

    gamma = 0.99
    gae_lambda = 0.95
    rewards = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
    values = np.asarray([0.1, 0.2, 0.3], dtype=np.float64)
    local_returns = _toy_gae(
        rewards,
        values,
        next_value=10.0,
        done_after_step=np.asarray([False, False, True]),
        gamma=gamma,
        gae_lambda=gae_lambda,
    )
    manual_local = np.asarray(_manual_local_terminal_returns(gamma, gae_lambda))
    trunc_returns = _toy_gae(
        rewards,
        values,
        next_value=10.0,
        done_after_step=np.asarray([False, False, False]),
        gamma=gamma,
        gae_lambda=gae_lambda,
    )

    output = Path(args.output)
    ensure_dir(output.parent)
    payload = {
        "config": args.config,
        "candidate": args.candidate,
        "n_demo": args.n_demo,
        "seed": args.seed,
        "num_envs": args.num_envs,
        "seed_start": args.seed_start,
        "horizon_steps": frozen.horizon_steps,
        "update_period": frozen.update_period,
        "conditioning": frozen.conditioning,
        "action_sensitivity": sensitivity_summary,
        "branch_terminal_distances": terminal,
        "gae_toy": {
            "gamma": gamma,
            "gae_lambda": gae_lambda,
            "local_terminal_returns": local_returns.tolist(),
            "manual_local_terminal_returns": manual_local.tolist(),
            "local_terminal_max_abs_error": float(np.max(np.abs(local_returns - manual_local))),
            "truncated_bootstrap_returns": trunc_returns.tolist(),
            "bootstrap_changes_returns": bool(
                np.max(np.abs(trunc_returns - local_returns)) > 1e-6
            ),
        },
        "pass": {
            "held_goal_input_changes_actions": sensitivity_summary[
                "shuffled_goal_action_l2"
            ]["mean"]
            > args.min_action_change,
            "observation_input_changes_actions": sensitivity_summary[
                "shuffled_observation_action_l2"
            ]["mean"]
            > args.min_action_change,
            "previous_action_changes_actions": sensitivity_summary[
                "cached_previous_action_l2"
            ]["mean"]
            > args.min_action_change,
            "remaining_time_changes_actions": sensitivity_summary[
                "constant_remaining_time_action_l2"
            ]["mean"]
            > args.min_action_change,
            "gae_local_terminal_matches_manual": float(
                np.max(np.abs(local_returns - manual_local))
            )
            < 1e-9,
        },
    }
    write_json(output, payload)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/pusht_incremental.yaml")
    parser.add_argument("--candidate", default="vae512_w2048_b1e6")
    parser.add_argument("--n-demo", type=int, default=1800)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-envs", type=int, default=32)
    parser.add_argument("--seed-start", type=int, default=3_900_000)
    parser.add_argument(
        "--output",
        default="results/incremental/rl_reachability_debug/run1_mechanics_audit.json",
    )
    parser.add_argument("--min-action-change", type=float, default=1e-5)
    args = parser.parse_args()
    print(run_audit(args))


if __name__ == "__main__":
    main()
