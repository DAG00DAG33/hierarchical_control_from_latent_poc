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
from hcl_poc.incremental import _pre_rl_phase_b_goal
from hcl_poc.models import MLP
from hcl_poc.rl import _make_state_env, _rl_paths, load_ppo_agent
from hcl_poc.utils import Standardizer, default_device, ensure_dir, write_json


GOAL_TYPE = "object_pose"


def _load_phase_b_low(path: Path, device: torch.device) -> tuple[nn.Module, dict[str, Any]]:
    payload = torch.load(path, map_location=device, weights_only=False)
    entry = payload[GOAL_TYPE]
    model = MLP(
        int(entry["cond_dim"]),
        int(entry["action_dim"]),
        int(entry["hidden_dim"]),
        depth=4,
    ).to(device)
    model.load_state_dict(entry["model"])
    model.eval()
    return model, payload


def _load_high(path: Path, device: torch.device) -> tuple[nn.Module, dict[str, Any]]:
    payload = torch.load(path, map_location=device, weights_only=False)
    if payload.get("goal_type") != GOAL_TYPE:
        raise ValueError(f"{path} is not an object_pose high-level checkpoint")
    model = MLP(
        int(payload["condition_dim"]),
        int(payload["target_dim"]),
        int(payload["hidden_dim"]),
        depth=4,
    ).to(device)
    model.load_state_dict(payload["model"])
    model.eval()
    return model, payload


@torch.inference_mode()
def _oracle_goal(
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
        future_state = _obs_state_np(obs)
        return _pre_rl_phase_b_goal(
            current_state,
            future_state,
            horizon,
            control_freq,
            GOAL_TYPE,
        ).astype(np.float32)
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
    device: torch.device,
) -> np.ndarray:
    if low_kind == "phase_b_object_pose_bc":
        entry = low_payload[GOAL_TYPE]
        action_norm = Standardizer.from_state_dict(low_payload["action_norm"])
        cond_norm = Standardizer.from_state_dict(entry["cond_norm"])
        previous_norm = action_norm.transform(previous_action_raw)
        condition = np.concatenate([state, goal, previous_norm], axis=-1).astype(np.float32)
        normalized = low_model(
            torch.from_numpy(cond_norm.transform(condition)).to(device).float()
        ).cpu().numpy()
        return action_norm.inverse(normalized).astype(np.float32)

    if low_kind != "run8_object_pose_ppo":
        raise ValueError(f"Unknown low policy: {low_kind}")
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
    return action.cpu().numpy().astype(np.float32)


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
    high_model = None
    high_payload = None
    high_input_norm = None
    high_target_norm = None
    high_action_norm = None
    if goal_source in {"learned", "shuffled_learned"}:
        high_model, high_payload = _load_high(Path(args.high_checkpoint), device)
        high_input_norm = Standardizer.from_state_dict(high_payload["input_norm"])
        high_target_norm = Standardizer.from_state_dict(high_payload["target_norm"])
        high_action_norm = Standardizer.from_state_dict(high_payload["action_norm"])
    base_low_kind = "run8_object_pose_ppo" if low_kind.startswith("run") else low_kind
    if low_kind == "phase_b_object_pose_bc":
        low_model, low_payload = _load_phase_b_low(low_path, device)
    else:
        low_model, low_payload = _load_rl_low(low_path, device)
        if low_payload.get("goal_type") != GOAL_TYPE:
            raise ValueError(f"{low_path} is not an object_pose checkpoint")
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
    learned_high_goal_distances: list[float] = []
    high_decisions = 0
    progress = trange(args.episodes, desc=f"{low_kind} {goal_source}")
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
            goal = np.zeros((batch_envs, 4), dtype=np.float32)
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
                        selected = _oracle_goal(
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
                    elif goal_source in {"learned", "shuffled_learned"}:
                        if (
                            high_model is None
                            or high_input_norm is None
                            or high_target_norm is None
                            or high_previous_norm is None
                        ):
                            raise RuntimeError("Missing high-level object-pose predictor")
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
                    if goal_source.startswith("shuffled_"):
                        selected = selected[np.roll(np.arange(batch_envs), 1)]
                    goal[replan] = selected[replan]
                    learned_high_goal_distances.extend(
                        _goal_distance(
                            state[replan],
                            selected[replan],
                            GOAL_TYPE,
                            horizon,
                            control_freq,
                        ).astype(float).tolist()
                    )
                    countdown[replan] = update_period
                    high_decisions += int(np.sum(replan))
                remaining = np.maximum(countdown, 1).astype(np.float32)
                raw_action = _low_action(
                    base_low_kind,
                    low_model,
                    low_payload,
                    state,
                    goal,
                    previous_action,
                    remaining,
                    horizon,
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
                    hold_goal_distances.extend(
                        _goal_distance(
                            next_state[completed_hold],
                            goal[completed_hold],
                            GOAL_TYPE,
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
        "goal_type": GOAL_TYPE,
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
        "hold_object_pose_distance": float(np.mean(hold_goal_distances)),
        "selected_goal_initial_distance": float(np.mean(learned_high_goal_distances)),
        "high_level_decisions_per_episode": float(high_decisions / max(len(successes_np), 1)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/pusht_incremental.yaml")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--num-envs", type=int, default=10)
    parser.add_argument("--seed-start", type=int, default=2_180_000)
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument("--update-period", type=int)
    parser.add_argument(
        "--high-checkpoint",
        default="artifacts/incremental/rl_reachability_debug/object_pose_high_predictor/seed0/predictor.pt",
    )
    parser.add_argument(
        "--phase-b-low",
        default="artifacts/incremental/pre_rl/phase_b/k10/seed0/oracle_goal_decomposition.pt",
    )
    parser.add_argument(
        "--run8-low",
        default=(
            "results/incremental/rl_reachability_debug/run8_object_pose_b8_u250/"
            "privileged_object_pose_ppo_progress_terminal_n4096_seed0/latest.pt"
        ),
    )
    parser.add_argument("--run8-low-name", default="run8_object_pose_ppo")
    parser.add_argument(
        "--goal-sources",
        nargs="+",
        choices=["oracle", "learned", "shuffled_oracle", "shuffled_learned"],
        default=["oracle", "learned", "shuffled_oracle"],
    )
    parser.add_argument(
        "--output",
        default="results/incremental/rl_reachability_debug/run8_object_pose_full_success_100.json",
    )
    args = parser.parse_args()
    policies = [
        ("phase_b_object_pose_bc", Path(args.phase_b_low)),
        (args.run8_low_name, Path(args.run8_low)),
    ]
    rows = [
        evaluate_policy(args, low_kind, path, goal_source)
        for goal_source in args.goal_sources
        for low_kind, path in policies
    ]
    payload = {
        "run": "rl_reachability_debug_object_pose_full_success",
        "episodes_per_setting": int(args.episodes),
        "goal_type": GOAL_TYPE,
        "high_checkpoint": str(args.high_checkpoint),
        "rows": rows,
    }
    output = Path(args.output)
    ensure_dir(output.parent)
    write_json(output, payload)
    print(output)


if __name__ == "__main__":
    main()
