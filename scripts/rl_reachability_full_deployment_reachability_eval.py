#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from tqdm import trange

sys.path.append(str(Path(__file__).resolve().parent))

from rl_reachability_goal_full_success_eval import _load_bc_low, _low_action
from rl_reachability_privileged_tcp_ppo import _goal_distance, _obs_state_np
from rl_reachability_tcp_full_success_eval import _load_rl_low, _to_numpy

from hcl_poc.config import load_config
from hcl_poc.incremental import _pre_rl_phase_b_goal
from hcl_poc.privileged_z import _clone_mani_state_dict
from hcl_poc.rl import _make_state_env, _rl_paths, load_ppo_agent
from hcl_poc.utils import default_device, ensure_dir, write_json


GOAL_TYPE = "full"

POLICY_PATH_DEFAULTS = {
    "phase_c_full_bc": "artifacts/incremental/pre_rl/phase_c/k10/seed0/time_conditioned_full.pt",
    "run20_full_ppo": (
        "results/incremental/rl_reachability_debug/run20_full_goal_recomputed_teacher_penalty10_b8_u250/"
        "privileged_full_ppo_progress_terminal_n4096_seed0/latest.pt"
    ),
    "run21_long_full_ppo": (
        "results/incremental/rl_reachability_debug/run21_full_goal_recomputed_penalty10_continue_u1000/"
        "privileged_full_ppo_progress_terminal_n4096_seed0/latest.pt"
    ),
    "run22_long_full_ppo": (
        "results/incremental/rl_reachability_debug/run22_full_goal_recomputed_penalty10_continue2_u1000/"
        "privileged_full_ppo_progress_terminal_n4096_seed0/latest.pt"
    ),
    "run23_reset_mixture_ppo": (
        "results/incremental/rl_reachability_debug/run23_full_reset_mixture_learned_high_penalty10_u250/"
        "privileged_full_ppo_progress_terminal_n4096_seed0/latest.pt"
    ),
    "run24_oracle_reset_mixture_ppo": (
        "results/incremental/rl_reachability_debug/run24_full_reset_mixture_oracle_target_penalty10_u250/"
        "privileged_full_ppo_progress_terminal_n4096_seed0/latest.pt"
    ),
}


def _load_low_policy(
    policy: str,
    paths: dict[str, Path],
    device: torch.device,
) -> tuple[str, nn.Module, dict[str, Any]]:
    if policy == "phase_c_full_bc":
        model, payload = _load_bc_low(paths[policy], GOAL_TYPE, device)
        return "phase_b_bc", model, payload
    model, payload = _load_rl_low(paths[policy], device)
    if payload.get("goal_type") != GOAL_TYPE:
        raise ValueError(f"{paths[policy]} is not a full-goal low-level checkpoint")
    return policy, model, payload


def _full_goal(
    state: np.ndarray,
    target_future_state: np.ndarray,
    remaining: int,
    control_freq: int,
) -> np.ndarray:
    return _pre_rl_phase_b_goal(
        state,
        target_future_state,
        int(max(1, remaining)),
        control_freq,
        GOAL_TYPE,
    ).astype(np.float32)


@torch.inference_mode()
def _oracle_future_state(
    config: Any,
    teacher: Any,
    state_dict: dict[str, Any],
    reset_seeds: list[int],
    horizon: int,
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
        branch_env.unwrapped.set_state_dict(_clone_mani_state_dict(state_dict))
        obs = branch_env.unwrapped.get_obs()
        for _ in range(horizon):
            state = torch.from_numpy(_obs_state_np(obs)).to(device).float()
            action = torch.clamp(teacher.actor_mean(state), action_low, action_high)
            obs, _reward, _terminated, _truncated, _info = branch_env.step(action)
        return _obs_state_np(obs).astype(np.float32)
    finally:
        branch_env.close()


@torch.inference_mode()
def _branch_rollout(
    branch_env: Any,
    reset_seeds: list[int],
    state_dict: dict[str, Any],
    low_base_kind: str,
    low_model: nn.Module,
    low_payload: dict[str, Any],
    target_future_state: np.ndarray,
    previous_action: np.ndarray,
    active: np.ndarray,
    *,
    shuffled_goal: bool,
    device: torch.device,
    horizon: int,
    control_freq: int,
    action_low_np: np.ndarray,
    action_high_np: np.ndarray,
) -> dict[str, np.ndarray]:
    branch_env.reset(seed=reset_seeds)
    branch_env.unwrapped.set_state_dict(_clone_mani_state_dict(state_dict))
    obs = branch_env.unwrapped.get_obs()
    target = target_future_state.copy()
    if shuffled_goal:
        target = target[np.roll(np.arange(len(target)), 1)]
    start_state = _obs_state_np(obs)
    start_goal = _full_goal(start_state, target, horizon, control_freq)
    start_distance = _goal_distance(start_state, start_goal, GOAL_TYPE, horizon, control_freq)
    prev = previous_action.copy()
    action_saturation = np.zeros(len(target), dtype=np.float32)
    action_l2 = np.zeros(len(target), dtype=np.float32)
    terminal_reward = np.zeros(len(target), dtype=np.float32)
    max_reward = np.full(len(target), -np.inf, dtype=np.float32)
    for step in range(horizon):
        state = _obs_state_np(obs)
        remaining = np.full(len(target), max(horizon - step, 1), dtype=np.float32)
        goal = _full_goal(state, target, int(max(horizon - step, 1)), control_freq)
        raw_action = _low_action(
            low_base_kind,
            low_model,
            low_payload,
            state,
            goal,
            prev,
            remaining,
            horizon,
            GOAL_TYPE,
            device,
        )
        saturated = np.any(
            (raw_action < action_low_np) | (raw_action > action_high_np),
            axis=-1,
        ).astype(np.float32)
        clipped = np.clip(raw_action, action_low_np, action_high_np).astype(np.float32)
        action_saturation += saturated
        action_l2 += np.linalg.norm(clipped, axis=-1).astype(np.float32)
        obs, reward, _terminated, _truncated, _info = branch_env.step(
            torch.from_numpy(clipped).to(device).float()
        )
        reward_np = _to_numpy(reward).reshape(-1).astype(np.float32)
        terminal_reward = reward_np
        max_reward = np.maximum(max_reward, reward_np)
        prev = clipped
    terminal_state = _obs_state_np(obs)
    terminal_goal = _full_goal(terminal_state, target, 1, control_freq)
    terminal_distance = _goal_distance(
        terminal_state,
        terminal_goal,
        GOAL_TYPE,
        horizon,
        control_freq,
    )
    mask = active.astype(bool)
    return {
        "initial_distance": start_distance[mask],
        "terminal_distance": terminal_distance[mask],
        "improved": (terminal_distance[mask] < start_distance[mask]).astype(np.float32),
        "action_saturation": (action_saturation[mask] / float(horizon)).astype(np.float32),
        "action_l2": (action_l2[mask] / float(horizon)).astype(np.float32),
        "terminal_reward": terminal_reward[mask],
        "max_reward": max_reward[mask],
    }


def _extend(store: dict[str, list[float]], metrics: dict[str, np.ndarray]) -> None:
    for key, values in metrics.items():
        store[key].extend(np.asarray(values, dtype=np.float32).reshape(-1).astype(float).tolist())


def _summary(
    *,
    collector_policy: str,
    candidate_policy: str,
    shuffled_goal: bool,
    store: dict[str, list[float]],
    success_epsilon: float,
) -> dict[str, Any]:
    terminal = np.asarray(store["terminal_distance"], dtype=np.float32)
    initial = np.asarray(store["initial_distance"], dtype=np.float32)
    return {
        "collector_policy": collector_policy,
        "candidate_policy": candidate_policy,
        "goal_type": GOAL_TYPE,
        "goal_source": "oracle",
        "shuffled_goal": bool(shuffled_goal),
        "decisions": int(len(terminal)),
        "initial_distance_mean": float(np.mean(initial)),
        "terminal_distance_mean": float(np.mean(terminal)),
        "distance_reduction_mean": float(np.mean(initial - terminal)),
        "goal_reach_rate_eps": float(np.mean(terminal <= success_epsilon)),
        "p50_terminal_distance": float(np.quantile(terminal, 0.50)),
        "p90_terminal_distance": float(np.quantile(terminal, 0.90)),
        "p99_terminal_distance": float(np.quantile(terminal, 0.99)),
        "fraction_improved_from_start": float(np.mean(store["improved"])),
        "action_saturation": float(np.mean(store["action_saturation"])),
        "action_l2_mean": float(np.mean(store["action_l2"])),
        "terminal_reward": float(np.mean(store["terminal_reward"])),
        "max_reward": float(np.mean(store["max_reward"])),
    }


@torch.inference_mode()
def evaluate_collector(
    args: argparse.Namespace,
    collector_policy: str,
    paths: dict[str, Path],
) -> list[dict[str, Any]]:
    config = load_config(args.config)
    device = default_device()
    teacher = load_ppo_agent(_rl_paths(config).best, device)
    horizon = int(args.horizon)
    update_period = int(args.update_period or horizon)
    control_freq = int(config.get("control_freq", 20))
    collector_base_kind, collector_model, collector_payload = _load_low_policy(
        collector_policy,
        paths,
        device,
    )
    candidates = {
        key: _load_low_policy(key, paths, device)
        for key in args.candidate_policies
    }
    num_envs = int(args.num_envs)
    env = _make_state_env(
        config,
        num_envs,
        record_metrics=True,
        ignore_terminations=False,
        reconfiguration_freq=0,
    )
    branch_env = _make_state_env(
        config,
        num_envs,
        record_metrics=False,
        ignore_terminations=True,
        reconfiguration_freq=0,
    )
    action_low = torch.as_tensor(env.single_action_space.low, device=device, dtype=torch.float32)
    action_high = torch.as_tensor(env.single_action_space.high, device=device, dtype=torch.float32)
    action_low_np = np.asarray(env.single_action_space.low, dtype=np.float32)
    action_high_np = np.asarray(env.single_action_space.high, dtype=np.float32)
    rows: dict[tuple[str, bool], dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    collected = 0
    progress = trange(args.decisions, desc=f"full-deploy-reach {collector_policy}")
    try:
        batch_index = 0
        while collected < args.decisions:
            reset_seeds = [
                args.seed_start + batch_index * num_envs + index
                for index in range(num_envs)
            ]
            batch_index += 1
            obs, _info = env.reset(seed=reset_seeds)
            previous_action = np.zeros((num_envs, 3), dtype=np.float32)
            target_future_state = np.zeros((num_envs, 31), dtype=np.float32)
            countdown = np.zeros(num_envs, dtype=np.int32)
            active = np.ones(num_envs, dtype=bool)
            steps = 0
            while np.any(active) and collected < args.decisions and steps < args.max_steps:
                state = _obs_state_np(obs)
                replan = active & (countdown <= 0)
                if np.any(replan):
                    selected_target = _oracle_future_state(
                        config,
                        teacher,
                        env.unwrapped.get_state_dict(),
                        reset_seeds,
                        horizon,
                        action_low,
                        action_high,
                        device,
                    )
                    target_future_state[replan] = selected_target[replan]
                    countdown[replan] = update_period
                    state_dict = _clone_mani_state_dict(env.unwrapped.get_state_dict())
                    for candidate_policy, (base_kind, low_model, low_payload) in candidates.items():
                        for shuffled in (False, True):
                            metrics = _branch_rollout(
                                branch_env,
                                reset_seeds,
                                state_dict,
                                base_kind,
                                low_model,
                                low_payload,
                                target_future_state,
                                previous_action,
                                replan,
                                shuffled_goal=shuffled,
                                device=device,
                                horizon=horizon,
                                control_freq=control_freq,
                                action_low_np=action_low_np,
                                action_high_np=action_high_np,
                            )
                            _extend(rows[(candidate_policy, shuffled)], metrics)
                    newly_collected = int(np.sum(replan))
                    remaining = args.decisions - progress.n
                    collected += newly_collected
                    progress.update(min(newly_collected, remaining))
                remaining_steps = np.maximum(countdown, 1).astype(np.float32)
                goal = _full_goal(
                    state,
                    target_future_state,
                    int(max(1, np.max(remaining_steps))),
                    control_freq,
                )
                raw_action = _low_action(
                    collector_base_kind,
                    collector_model,
                    collector_payload,
                    state,
                    goal,
                    previous_action,
                    remaining_steps,
                    horizon,
                    GOAL_TYPE,
                    device,
                )
                clipped = np.clip(raw_action, action_low_np, action_high_np).astype(np.float32)
                obs, _reward, terminated, truncated, info = env.step(
                    torch.from_numpy(clipped).to(device).float()
                )
                countdown -= 1
                previous_action = clipped
                done = torch.logical_or(terminated, truncated).detach().cpu().numpy().reshape(-1).astype(bool)
                if "final_info" in info:
                    done |= _to_numpy(info["_final_info"]).reshape(-1).astype(bool)
                active[done] = False
                steps += 1
    finally:
        progress.close()
        env.close()
        branch_env.close()

    return [
        _summary(
            collector_policy=collector_policy,
            candidate_policy=candidate_policy,
            shuffled_goal=shuffled,
            store=store,
            success_epsilon=float(args.success_epsilon),
        )
        for (candidate_policy, shuffled), store in sorted(rows.items())
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/pusht_incremental.yaml")
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument("--update-period", type=int)
    parser.add_argument("--decisions", type=int, default=512)
    parser.add_argument("--num-envs", type=int, default=16)
    parser.add_argument("--seed-start", type=int, default=2_480_000)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--success-epsilon", type=float, default=0.0025)
    parser.add_argument(
        "--collector-policies",
        nargs="+",
        choices=list(POLICY_PATH_DEFAULTS),
        default=["phase_c_full_bc", "run21_long_full_ppo"],
    )
    parser.add_argument(
        "--candidate-policies",
        nargs="+",
        choices=list(POLICY_PATH_DEFAULTS),
        default=list(POLICY_PATH_DEFAULTS),
    )
    parser.add_argument("--phase-c-full-bc", default=POLICY_PATH_DEFAULTS["phase_c_full_bc"])
    parser.add_argument("--run20-low", default=POLICY_PATH_DEFAULTS["run20_full_ppo"])
    parser.add_argument("--run21-low", default=POLICY_PATH_DEFAULTS["run21_long_full_ppo"])
    parser.add_argument("--run22-low", default=POLICY_PATH_DEFAULTS["run22_long_full_ppo"])
    parser.add_argument("--run23-low", default=POLICY_PATH_DEFAULTS["run23_reset_mixture_ppo"])
    parser.add_argument("--run24-low", default=POLICY_PATH_DEFAULTS["run24_oracle_reset_mixture_ppo"])
    parser.add_argument(
        "--output",
        default="results/incremental/rl_reachability_debug/run21_full_deployment_reachability_512.json",
    )
    args = parser.parse_args()
    paths = {
        "phase_c_full_bc": Path(args.phase_c_full_bc),
        "run20_full_ppo": Path(args.run20_low),
        "run21_long_full_ppo": Path(args.run21_low),
        "run22_long_full_ppo": Path(args.run22_low),
        "run23_reset_mixture_ppo": Path(args.run23_low),
        "run24_oracle_reset_mixture_ppo": Path(args.run24_low),
    }
    rows = []
    for collector_policy in args.collector_policies:
        rows.extend(evaluate_collector(args, collector_policy, paths))
    payload = {
        "run": "rl_reachability_debug_full_deployment_reachability",
        "goal_type": GOAL_TYPE,
        "goal_source": "oracle",
        "held_goal_semantics": "fixed_target_recomputed_features",
        "decisions_per_collector": int(args.decisions),
        "rows": rows,
    }
    output = Path(args.output)
    ensure_dir(output.parent)
    write_json(output, payload)
    print(output)


if __name__ == "__main__":
    main()
