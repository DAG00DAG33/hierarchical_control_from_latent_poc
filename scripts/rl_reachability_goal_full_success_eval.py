#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from tqdm import trange

sys.path.append(str(Path(__file__).resolve().parent))

from rl_reachability_privileged_tcp_ppo import _goal_distance, _obs_state_np
from rl_reachability_tcp_full_success_eval import _load_rl_low, _to_numpy

from hcl_poc.config import load_config
from hcl_poc.incremental import PRE_RL_PHASE_B_GOAL_TYPES, _pre_rl_phase_b_goal
from hcl_poc.models import MLP
from hcl_poc.rl import _make_state_env, _rl_paths, load_ppo_agent
from hcl_poc.utils import Standardizer, default_device, ensure_dir, write_json


def _load_bc_low(
    path: Path,
    goal_type: str,
    device: torch.device,
) -> tuple[nn.Module, dict[str, Any]]:
    payload = torch.load(path, map_location=device, weights_only=False)
    entry = payload[goal_type]
    model = MLP(
        int(entry["cond_dim"]),
        int(entry["action_dim"]),
        int(entry["hidden_dim"]),
        depth=4,
    ).to(device)
    model.load_state_dict(entry["model"])
    model.eval()
    return model, payload


def _load_high(path: Path, goal_type: str, device: torch.device) -> tuple[nn.Module, dict[str, Any]]:
    payload = torch.load(path, map_location=device, weights_only=False)
    if payload.get("goal_type") != goal_type:
        raise ValueError(f"{path} is not a {goal_type} high-level checkpoint")
    model = MLP(
        int(payload["condition_dim"]),
        int(payload["target_dim"]),
        int(payload["hidden_dim"]),
        depth=4,
    ).to(device)
    model.load_state_dict(payload["model"])
    model.eval()
    return model, payload


def _goals_to_targets(
    current_state: np.ndarray,
    target_future_state: np.ndarray,
    remaining_steps: np.ndarray,
    control_freq: int,
    goal_type: str,
) -> np.ndarray:
    remaining = np.asarray(remaining_steps, dtype=np.int32).reshape(-1)
    goals = np.zeros((len(current_state), _pre_rl_phase_b_goal(
        current_state[:1],
        target_future_state[:1],
        int(max(1, remaining[0] if len(remaining) else 1)),
        control_freq,
        goal_type,
    ).shape[-1]), dtype=np.float32)
    for value in np.unique(remaining):
        mask = remaining == value
        goals[mask] = _pre_rl_phase_b_goal(
            current_state[mask],
            target_future_state[mask],
            int(max(1, value)),
            control_freq,
            goal_type,
        ).astype(np.float32)
    return goals


def _full_goal_to_pseudo_future_state(
    current_state: np.ndarray,
    full_goal: np.ndarray,
) -> np.ndarray:
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


def _scale_full_future_from_current(
    current_state: np.ndarray,
    future_state: np.ndarray,
    scale: float,
) -> np.ndarray:
    if abs(scale - 1.0) < 1e-8:
        return np.asarray(future_state, dtype=np.float32)
    current = np.asarray(current_state, dtype=np.float32)
    future = np.asarray(future_state, dtype=np.float32)
    scaled = future.copy()
    for slc in (slice(0, 14), slice(14, 17), slice(24, 26)):
        scaled[:, slc] = current[:, slc] + scale * (future[:, slc] - current[:, slc])
    current_yaw = 2.0 * np.arctan2(current[:, 30], current[:, 27])
    future_yaw = 2.0 * np.arctan2(future[:, 30], future[:, 27])
    yaw_delta = np.arctan2(np.sin(future_yaw - current_yaw), np.cos(future_yaw - current_yaw))
    scaled_yaw = current_yaw + scale * yaw_delta
    scaled[:, 27:31] = 0.0
    scaled[:, 27] = np.cos(0.5 * scaled_yaw)
    scaled[:, 30] = np.sin(0.5 * scaled_yaw)
    return scaled.astype(np.float32)


@torch.inference_mode()
def _oracle_future_state(
    config: Any,
    teacher: Any,
    state_dict: dict[str, Any],
    reset_seeds: list[int],
    current_state: np.ndarray,
    horizon: int,
    control_freq: int,
    action_low: torch.Tensor,
    action_high: torch.Tensor,
    device: torch.device,
) -> np.ndarray:
    branch_env = _make_state_env(
        config,
        len(reset_seeds),
        record_metrics=False,
        ignore_terminations=True,
        reconfiguration_freq=0,
    )
    try:
        branch_env.reset(seed=reset_seeds)
        branch_env.unwrapped.set_state_dict(state_dict)
        obs = branch_env.unwrapped.get_obs()
        for _ in range(horizon):
            state = torch.from_numpy(_obs_state_np(obs)).to(device).float()
            action = torch.clamp(teacher.actor_mean(state), action_low, action_high)
            obs, _reward, _terminated, _truncated, _info = branch_env.step(action)
        return _obs_state_np(obs).astype(np.float32)
    finally:
        branch_env.close()


@torch.inference_mode()
def _low_action(
    low_kind: str,
    low_model: nn.Module,
    low_payload: dict[str, Any],
    state: np.ndarray,
    goal: np.ndarray,
    previous_action_raw: np.ndarray,
    remaining: np.ndarray,
    horizon: int,
    goal_type: str,
    device: torch.device,
) -> np.ndarray:
    if low_kind == "phase_b_bc":
        entry = low_payload[goal_type]
        action_norm = Standardizer.from_state_dict(low_payload["action_norm"])
        cond_norm = Standardizer.from_state_dict(entry["cond_norm"])
        previous_norm = action_norm.transform(previous_action_raw)
        condition = np.concatenate([state, goal, previous_norm], axis=-1).astype(np.float32)
        if bool(entry.get("time_conditioned", False)):
            condition = np.concatenate(
                [condition, (remaining / horizon)[:, None]],
                axis=-1,
            ).astype(np.float32)
        normalized = low_model(
            torch.from_numpy(cond_norm.transform(condition)).to(device).float()
        ).cpu().numpy()
        return action_norm.inverse(normalized).astype(np.float32)

    state_norm = Standardizer.from_state_dict(low_payload["state_norm"])
    goal_norm = Standardizer.from_state_dict(low_payload["goal_norm"])
    action_norm = Standardizer.from_state_dict(low_payload["action_norm"])
    previous_norm = action_norm.transform(previous_action_raw)
    condition = np.concatenate(
        [
            state_norm.transform(state),
            goal_norm.transform(goal),
            previous_norm,
            (remaining / horizon)[:, None],
        ],
        axis=-1,
    ).astype(np.float32)
    action, _logprob, _entropy, _value = low_model.get_action_and_value(
        torch.from_numpy(condition).to(device).float(),
        deterministic=True,
    )
    raw_action = action.cpu().numpy().astype(np.float32)
    recipe = low_payload.get("recipe", {})
    if low_payload.get("policy_mode") != "bc_residual" and recipe.get("policy_mode") != "bc_residual":
        return raw_action

    base_model = low_payload["_bc_residual_base_model"]
    base_action_norm = low_payload["_bc_residual_action_norm"]
    base_cond_norm = low_payload["_bc_residual_cond_norm"]
    base_previous_norm = base_action_norm.transform(previous_action_raw)
    base_condition = np.concatenate(
        [
            state,
            goal,
            base_previous_norm,
            (remaining / horizon)[:, None],
        ],
        axis=-1,
    ).astype(np.float32)
    base_normalized = base_model(
        torch.from_numpy(base_cond_norm.transform(base_condition)).to(device).float()
    )
    base_action = base_action_norm.inverse(base_normalized.cpu().numpy()).astype(np.float32)
    alpha = float(recipe.get("residual_alpha", 0.1))
    return (base_action + alpha * np.tanh(raw_action)).astype(np.float32)


@torch.inference_mode()
def evaluate_policy(
    args: argparse.Namespace,
    low_kind: str,
    low_path: Path,
    goal_source: str,
) -> dict[str, Any]:
    config = load_config(args.config)
    device = default_device()
    teacher = load_ppo_agent(_rl_paths(config).best, device)
    goal_type = args.goal_type
    high_model = None
    high_input_norm = None
    high_target_norm = None
    high_action_norm = None
    if goal_source in {"learned", "shuffled_learned"}:
        high_model, high_payload = _load_high(Path(args.high_checkpoint), goal_type, device)
        high_input_norm = Standardizer.from_state_dict(high_payload["input_norm"])
        high_target_norm = Standardizer.from_state_dict(high_payload["target_norm"])
        high_action_norm = Standardizer.from_state_dict(high_payload["action_norm"])
    if low_kind == "phase_b_bc":
        low_model, low_payload = _load_bc_low(low_path, goal_type, device)
    else:
        low_model, low_payload = _load_rl_low(low_path, device)
        if low_payload.get("goal_type") != goal_type:
            raise ValueError(f"{low_path} is not a {goal_type} checkpoint")
    horizon = int(args.horizon)
    update_period = int(args.update_period or horizon)
    control_freq = int(config.get("control_freq", 20))
    num_envs = min(int(args.num_envs), int(args.episodes))
    env = _make_state_env(
        config,
        num_envs,
        record_metrics=True,
        ignore_terminations=False,
        reconfiguration_freq=0,
    )
    action_low = torch.as_tensor(env.single_action_space.low, device=device, dtype=torch.float32)
    action_high = torch.as_tensor(env.single_action_space.high, device=device, dtype=torch.float32)
    action_low_np = np.asarray(env.single_action_space.low, dtype=np.float32)
    action_high_np = np.asarray(env.single_action_space.high, dtype=np.float32)
    successes: list[float] = []
    final_rewards: list[float] = []
    max_rewards: list[float] = []
    episode_lengths: list[int] = []
    action_saturation: list[float] = []
    action_l2: list[float] = []
    teacher_maes: list[float] = []
    hold_goal_distances: list[float] = []
    selected_goal_initial_distances: list[float] = []
    high_decisions = 0
    if (
        args.recompute_held_goal_features
        and goal_source in {"learned", "shuffled_learned"}
        and goal_type != "full"
    ):
        raise ValueError(
            "--recompute-held-goal-features with learned high-level goals is only "
            "implemented for full goals"
        )
    progress = trange(args.episodes, desc=f"{low_kind} {goal_type} {goal_source}")
    try:
        for batch_start in range(0, args.episodes, num_envs):
            batch_envs = min(num_envs, args.episodes - batch_start)
            if batch_envs != num_envs:
                break
            reset_seeds = [args.seed_start + batch_start + i for i in range(batch_envs)]
            obs, _info = env.reset(seed=reset_seeds)
            previous_action = np.zeros((batch_envs, 3), dtype=np.float32)
            high_previous_norm = (
                high_action_norm.transform(previous_action)
                if high_action_norm is not None
                else None
            )
            goal = np.zeros((batch_envs, int(args.goal_dim)), dtype=np.float32)
            target_future_state = np.zeros((batch_envs, 31), dtype=np.float32)
            countdown = np.zeros(batch_envs, dtype=np.int32)
            active = np.ones(batch_envs, dtype=bool)
            success_once = np.zeros(batch_envs, dtype=bool)
            active_lengths = np.zeros(batch_envs, dtype=np.int32)
            active_max_reward = np.full(batch_envs, -np.inf, dtype=np.float32)
            final_reward = np.zeros(batch_envs, dtype=np.float32)
            while np.any(active):
                state = _obs_state_np(obs)
                replan = active & (countdown <= 0)
                if np.any(replan):
                    if goal_source in {"oracle", "shuffled_oracle"}:
                        selected_future_state = _oracle_future_state(
                            config,
                            teacher,
                            env.unwrapped.get_state_dict(),
                            reset_seeds,
                            state,
                            horizon,
                            control_freq,
                            action_low,
                            action_high,
                            device,
                        )
                        if goal_source.startswith("shuffled_"):
                            selected_future_state = selected_future_state[
                                np.roll(np.arange(batch_envs), 1)
                            ]
                        selected = _pre_rl_phase_b_goal(
                            state,
                            selected_future_state,
                            horizon,
                            control_freq,
                            goal_type,
                        ).astype(np.float32)
                    elif goal_source in {"learned", "shuffled_learned"}:
                        if (
                            high_model is None
                            or high_input_norm is None
                            or high_target_norm is None
                            or high_previous_norm is None
                        ):
                            raise RuntimeError("Missing high-level predictor")
                        selected = high_target_norm.inverse(
                            high_model(
                                torch.from_numpy(
                                    high_input_norm.transform(
                                        np.concatenate([state, high_previous_norm], axis=-1)
                                    )
                                )
                                .to(device)
                                .float()
                            )
                            .cpu()
                            .numpy()
                        ).astype(np.float32)
                    else:
                        raise ValueError(f"Unknown goal_source: {goal_source}")
                    if goal_source.startswith("shuffled_") and goal_source not in {
                        "shuffled_oracle"
                    }:
                        selected = selected[np.roll(np.arange(batch_envs), 1)]
                    goal[replan] = selected[replan]
                    if goal_source in {"oracle", "shuffled_oracle"}:
                        target_future_state[replan] = selected_future_state[replan]
                    elif args.recompute_held_goal_features:
                        pseudo_future = _full_goal_to_pseudo_future_state(state, selected)
                        if args.learned_robot_target_mode == "current":
                            pseudo_future[:, :14] = state[:, :14]
                        pseudo_future = _scale_full_future_from_current(
                            state,
                            pseudo_future,
                            float(args.learned_goal_scale),
                        )
                        selected = _pre_rl_phase_b_goal(
                            state,
                            pseudo_future,
                            horizon,
                            control_freq,
                            goal_type,
                        ).astype(np.float32)
                        goal[replan] = selected[replan]
                        target_future_state[replan] = pseudo_future[replan]
                    selected_goal_initial_distances.extend(
                        _goal_distance(
                            state[replan],
                            selected[replan],
                            goal_type,
                            horizon,
                            control_freq,
                        ).astype(float).tolist()
                    )
                    countdown[replan] = update_period
                    high_decisions += int(np.sum(replan))
                remaining = np.maximum(countdown, 1).astype(np.float32)
                if args.recompute_held_goal_features:
                    goal[active] = _goals_to_targets(
                        state[active],
                        target_future_state[active],
                        remaining[active].astype(np.int32),
                        control_freq,
                        goal_type,
                    )
                raw_action = _low_action(
                    low_kind,
                    low_model,
                    low_payload,
                    state,
                    goal,
                    previous_action,
                    remaining,
                    horizon,
                    goal_type,
                    device,
                )
                teacher_action = torch.clamp(
                    teacher.actor_mean(torch.from_numpy(state).to(device).float()),
                    action_low,
                    action_high,
                ).cpu().numpy()
                teacher_maes.extend(
                    np.mean(np.abs(raw_action[active] - teacher_action[active]), axis=-1)
                    .astype(float)
                    .tolist()
                )
                action_saturation.extend(
                    np.any(
                        (raw_action[active] < action_low_np)
                        | (raw_action[active] > action_high_np),
                        axis=-1,
                    ).astype(np.float32).tolist()
                )
                clipped = np.clip(raw_action, action_low_np, action_high_np).astype(np.float32)
                action_l2.extend(
                    np.linalg.norm(clipped[active], axis=-1).astype(float).tolist()
                )
                obs, reward, terminated, truncated, info = env.step(
                    torch.from_numpy(clipped).to(device).float()
                )
                countdown -= 1
                previous_action = clipped
                if high_action_norm is not None:
                    high_previous_norm = high_action_norm.transform(previous_action)
                next_state = _obs_state_np(obs)
                completed_hold = active & (countdown <= 0)
                if np.any(completed_hold):
                    completed_goal = (
                        _goals_to_targets(
                            next_state[completed_hold],
                            target_future_state[completed_hold],
                            np.ones(int(np.sum(completed_hold)), dtype=np.int32),
                            control_freq,
                            goal_type,
                        )
                        if args.recompute_held_goal_features
                        else goal[completed_hold]
                    )
                    hold_goal_distances.extend(
                        _goal_distance(
                            next_state[completed_hold],
                            completed_goal,
                            goal_type,
                            horizon,
                            control_freq,
                        ).astype(float).tolist()
                    )
                reward_np = reward.detach().cpu().numpy().reshape(-1).astype(np.float32)
                final_reward[active] = reward_np[active]
                active_max_reward[active] = np.maximum(active_max_reward[active], reward_np[active])
                active_lengths[active] += 1
                if "success" in info:
                    success_once |= _to_numpy(info["success"]).reshape(-1).astype(bool)
                final_done = np.zeros(batch_envs, dtype=bool)
                final_success = success_once.copy()
                if "final_info" in info:
                    final_done = _to_numpy(info["_final_info"]).reshape(-1).astype(bool)
                    if np.any(final_done) and "episode" in info["final_info"]:
                        final_success = _to_numpy(
                            info["final_info"]["episode"]["success_once"]
                        ).reshape(-1).astype(bool)
                done = torch.logical_or(terminated, truncated).detach().cpu().numpy().reshape(-1).astype(bool)
                newly_done = active & (done | final_done)
                if np.any(newly_done):
                    for idx in np.flatnonzero(newly_done):
                        successes.append(float(final_success[idx]))
                        final_rewards.append(float(final_reward[idx]))
                        max_rewards.append(float(active_max_reward[idx]))
                        episode_lengths.append(int(active_lengths[idx]))
                    active[newly_done] = False
                    progress.update(int(np.sum(newly_done)))
    finally:
        progress.close()
        env.close()
    successes_np = np.asarray(successes[: args.episodes], dtype=np.float32)
    return {
        "low_policy": low_kind,
        "low_checkpoint": str(low_path),
        "goal_type": goal_type,
        "goal_source": goal_source,
        "shuffled_goal": bool(goal_source.startswith("shuffled_")),
        "episodes": int(len(successes_np)),
        "seed_start": int(args.seed_start),
        "num_envs": int(num_envs),
        "success": float(np.mean(successes_np)),
        "final_reward": float(np.mean(final_rewards[: len(successes_np)])),
        "max_reward": float(np.mean(max_rewards[: len(successes_np)])),
        "mean_episode_length": float(np.mean(episode_lengths[: len(successes_np)])),
        "teacher_action_mae": float(np.mean(teacher_maes)),
        "action_saturation_rate": float(np.mean(action_saturation)),
        "action_l2_mean": float(np.mean(action_l2)),
        "hold_goal_distance": float(np.mean(hold_goal_distances)),
        "selected_goal_initial_distance": float(np.mean(selected_goal_initial_distances)),
        "high_level_decisions_per_episode": float(high_decisions / max(len(successes_np), 1)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/pusht_incremental.yaml")
    parser.add_argument("--goal-type", choices=PRE_RL_PHASE_B_GOAL_TYPES, required=True)
    parser.add_argument("--goal-dim", type=int, required=True)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--num-envs", type=int, default=10)
    parser.add_argument("--seed-start", type=int, default=2_380_000)
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument("--update-period", type=int)
    parser.add_argument("--recompute-held-goal-features", action="store_true")
    parser.add_argument("--learned-goal-scale", type=float, default=1.0)
    parser.add_argument(
        "--learned-robot-target-mode",
        choices=["predicted", "current"],
        default="predicted",
    )
    parser.add_argument("--high-checkpoint")
    parser.add_argument(
        "--bc-low",
        default="artifacts/incremental/pre_rl/phase_b/k10/seed0/oracle_goal_decomposition.pt",
    )
    parser.add_argument("--rl-low", required=True)
    parser.add_argument("--rl-low-name", required=True)
    parser.add_argument(
        "--goal-sources",
        nargs="+",
        choices=["oracle", "learned", "shuffled_oracle", "shuffled_learned"],
        default=["oracle", "learned", "shuffled_learned"],
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if any(source in {"learned", "shuffled_learned"} for source in args.goal_sources):
        if not args.high_checkpoint:
            raise ValueError("--high-checkpoint is required for learned goal sources")
    policies = [
        ("phase_b_bc", Path(args.bc_low)),
        (args.rl_low_name, Path(args.rl_low)),
    ]
    rows = [
        evaluate_policy(args, low_kind, path, goal_source)
        for goal_source in args.goal_sources
        for low_kind, path in policies
    ]
    payload = {
        "run": "rl_reachability_debug_goal_full_success",
        "episodes_per_setting": int(args.episodes),
        "goal_type": args.goal_type,
        "learned_goal_scale": float(args.learned_goal_scale),
        "learned_robot_target_mode": str(args.learned_robot_target_mode),
        "high_checkpoint": str(args.high_checkpoint) if args.high_checkpoint else None,
        "rows": rows,
    }
    output = Path(args.output)
    ensure_dir(output.parent)
    write_json(output, payload)
    print(output)


if __name__ == "__main__":
    main()
