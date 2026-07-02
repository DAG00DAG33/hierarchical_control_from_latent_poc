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

from rl_reachability_goal_full_success_eval import (
    _full_goal_to_pseudo_future_state,
    _goals_to_targets,
    _load_bc_low,
    _load_high,
    _low_action,
    _oracle_future_state,
)
from rl_reachability_privileged_tcp_ppo import _goal_distance, _obs_state_np
from rl_reachability_tcp_full_success_eval import _load_rl_low, _to_numpy

from hcl_poc.config import load_config
from hcl_poc.incremental import _pre_rl_phase_b_goal
from hcl_poc.models import MLP
from hcl_poc.rl import _make_state_env, _rl_paths, load_ppo_agent
from hcl_poc.utils import Standardizer, default_device, ensure_dir, write_json


GOAL_TYPE = "full"


def _summary(values: list[float]) -> dict[str, float | int]:
    if not values:
        return {"count": 0}
    x = np.asarray(values, dtype=np.float32)
    return {
        "count": int(len(x)),
        "mean": float(np.mean(x)),
        "p50": float(np.quantile(x, 0.50)),
        "p90": float(np.quantile(x, 0.90)),
        "p99": float(np.quantile(x, 0.99)),
        "max": float(np.max(x)),
    }


def _metric_features(goal: np.ndarray) -> np.ndarray:
    return np.concatenate([goal[:, :4], goal[:, 7:10], goal[:, 13:28]], axis=-1).astype(
        np.float32
    )


def _component_l2(predicted: np.ndarray, oracle: np.ndarray) -> dict[str, list[float]]:
    return {
        "object_pose_l2": np.linalg.norm(predicted[:, :4] - oracle[:, :4], axis=-1)
        .astype(np.float32)
        .tolist(),
        "tcp_l2": np.linalg.norm(predicted[:, 7:10] - oracle[:, 7:10], axis=-1)
        .astype(np.float32)
        .tolist(),
        "robot_l2": np.linalg.norm(predicted[:, 13:28] - oracle[:, 13:28], axis=-1)
        .astype(np.float32)
        .tolist(),
        "full_metric_l2": np.linalg.norm(
            _metric_features(predicted) - _metric_features(oracle),
            axis=-1,
        )
        .astype(np.float32)
        .tolist(),
    }


def _load_low(
    low_kind: str,
    low_path: Path,
    device: torch.device,
) -> tuple[str, nn.Module, dict[str, Any]]:
    if low_kind == "phase_c_full_bc":
        model, payload = _load_bc_low(low_path, GOAL_TYPE, device)
        return "phase_b_bc", model, payload
    model, payload = _load_rl_low(low_path, device)
    if payload.get("goal_type") != GOAL_TYPE:
        raise ValueError(f"{low_path} is not a full-goal low-level checkpoint")
    return low_kind, model, payload


@torch.inference_mode()
def run_audit(args: argparse.Namespace) -> dict[str, Any]:
    config = load_config(args.config)
    device = default_device()
    teacher = load_ppo_agent(_rl_paths(config).best, device)
    teacher.eval()
    high_model, high_payload = _load_high(Path(args.high_checkpoint), GOAL_TYPE, device)
    high_input_norm = Standardizer.from_state_dict(high_payload["input_norm"])
    high_target_norm = Standardizer.from_state_dict(high_payload["target_norm"])
    high_action_norm = Standardizer.from_state_dict(high_payload["action_norm"])
    low_kind, low_model, low_payload = _load_low(args.low_kind, Path(args.low_path), device)
    horizon = int(args.horizon)
    update_period = int(args.update_period or horizon)
    control_freq = int(config.get("control_freq", 20))
    num_envs = min(int(args.num_envs), int(args.episodes))
    env = _make_state_env(
        config,
        num_envs,
        record_metrics=False,
        ignore_terminations=False,
        reconfiguration_freq=0,
    )
    action_low = torch.as_tensor(env.single_action_space.low, device=device, dtype=torch.float32)
    action_high = torch.as_tensor(env.single_action_space.high, device=device, dtype=torch.float32)
    action_low_np = np.asarray(env.single_action_space.low, dtype=np.float32)
    action_high_np = np.asarray(env.single_action_space.high, dtype=np.float32)
    learned_current_distance: list[float] = []
    oracle_current_distance: list[float] = []
    selected_goal_initial_distance: list[float] = []
    component_values: dict[str, list[float]] = {
        "object_pose_l2": [],
        "tcp_l2": [],
        "robot_l2": [],
        "full_metric_l2": [],
    }
    high_decisions = 0
    progress = trange(args.episodes, desc=f"{args.low_kind} learned-high audit")
    try:
        for batch_start in range(0, args.episodes, num_envs):
            batch_envs = min(num_envs, args.episodes - batch_start)
            if batch_envs != num_envs:
                break
            reset_seeds = [args.seed_start + batch_start + i for i in range(batch_envs)]
            obs, _info = env.reset(seed=reset_seeds)
            previous_action = np.zeros((batch_envs, 3), dtype=np.float32)
            high_previous_norm = high_action_norm.transform(previous_action)
            target_future_state = np.zeros((batch_envs, 31), dtype=np.float32)
            goal = np.zeros((batch_envs, 28), dtype=np.float32)
            countdown = np.zeros(batch_envs, dtype=np.int32)
            active = np.ones(batch_envs, dtype=bool)
            active_max_reward = np.full(batch_envs, -np.inf, dtype=np.float32)
            final_reward = np.zeros(batch_envs, dtype=np.float32)
            while np.any(active):
                state = _obs_state_np(obs)
                replan = active & (countdown <= 0)
                if np.any(replan):
                    learned_goal = high_target_norm.inverse(
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
                    oracle_future = _oracle_future_state(
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
                    oracle_goal = _pre_rl_phase_b_goal(
                        state,
                        oracle_future,
                        horizon,
                        control_freq,
                        GOAL_TYPE,
                    ).astype(np.float32)
                    learned_future = _full_goal_to_pseudo_future_state(state, learned_goal)
                    target_future_state[replan] = learned_future[replan]
                    goal[replan] = learned_goal[replan]
                    selected_goal_initial_distance.extend(
                        _goal_distance(
                            state[replan],
                            learned_goal[replan],
                            GOAL_TYPE,
                            horizon,
                            control_freq,
                        )
                        .astype(float)
                        .tolist()
                    )
                    learned_current_distance.extend(
                        _goal_distance(
                            state[replan],
                            learned_goal[replan],
                            GOAL_TYPE,
                            horizon,
                            control_freq,
                        )
                        .astype(float)
                        .tolist()
                    )
                    oracle_current_distance.extend(
                        _goal_distance(
                            state[replan],
                            oracle_goal[replan],
                            GOAL_TYPE,
                            horizon,
                            control_freq,
                        )
                        .astype(float)
                        .tolist()
                    )
                    components = _component_l2(learned_goal[replan], oracle_goal[replan])
                    for key, values in components.items():
                        component_values[key].extend(values)
                    countdown[replan] = update_period
                    high_decisions += int(np.sum(replan))
                remaining = np.maximum(countdown, 1).astype(np.float32)
                goal[active] = _goals_to_targets(
                    state[active],
                    target_future_state[active],
                    remaining[active].astype(np.int32),
                    control_freq,
                    GOAL_TYPE,
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
                    GOAL_TYPE,
                    device,
                )
                clipped = np.clip(raw_action, action_low_np, action_high_np).astype(np.float32)
                obs, reward, terminated, truncated, info = env.step(
                    torch.from_numpy(clipped).to(device).float()
                )
                previous_action = clipped
                high_previous_norm = high_action_norm.transform(previous_action)
                countdown -= 1
                reward_np = _to_numpy(reward).reshape(-1).astype(np.float32)
                final_reward[active] = reward_np[active]
                active_max_reward[active] = np.maximum(active_max_reward[active], reward_np[active])
                final_done = np.zeros(batch_envs, dtype=bool)
                if "final_info" in info:
                    final_done = _to_numpy(info["_final_info"]).reshape(-1).astype(bool)
                done = torch.logical_or(terminated, truncated).detach().cpu().numpy().reshape(-1).astype(bool)
                newly_done = active & (done | final_done)
                if np.any(newly_done):
                    active[newly_done] = False
                    progress.update(int(np.sum(newly_done)))
    finally:
        progress.close()
        env.close()
    completed_episodes = int(progress.n)
    return {
        "run": "rl_reachability_full_high_goal_quality_audit",
        "low_kind": args.low_kind,
        "low_path": str(args.low_path),
        "high_checkpoint": str(args.high_checkpoint),
        "episodes": completed_episodes,
        "seed_start": int(args.seed_start),
        "num_envs": int(num_envs),
        "horizon": int(horizon),
        "update_period": int(update_period),
        "high_decisions": int(high_decisions),
        "high_decisions_per_episode": float(high_decisions / max(completed_episodes, 1)),
        "learned_current_distance": _summary(learned_current_distance),
        "oracle_current_distance": _summary(oracle_current_distance),
        "selected_goal_initial_distance": _summary(selected_goal_initial_distance),
        "learned_vs_oracle": {
            key: _summary(values) for key, values in component_values.items()
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/pusht_incremental.yaml")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--num-envs", type=int, default=10)
    parser.add_argument("--seed-start", type=int, default=2_380_000)
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument("--update-period", type=int)
    parser.add_argument(
        "--high-checkpoint",
        default="artifacts/incremental/rl_reachability_debug/full_goal_high_predictor/seed0/predictor.pt",
    )
    parser.add_argument(
        "--low-kind",
        choices=["phase_c_full_bc", "run30_residual_bc_ppo"],
        required=True,
    )
    parser.add_argument("--low-path", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    ensure_dir(output.parent)
    write_json(output, run_audit(args))
    print(output)


if __name__ == "__main__":
    main()
