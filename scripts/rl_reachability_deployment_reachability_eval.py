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

from rl_reachability_tcp_full_success_eval import (
    _load_bc_low,
    _load_high,
    _load_rl_low,
    _low_action,
    _to_numpy,
)
from rl_reachability_privileged_tcp_ppo import _obs_state_np

from hcl_poc.config import load_config
from hcl_poc.privileged_z import _clone_mani_state_dict
from hcl_poc.rl import _make_state_env
from hcl_poc.utils import Standardizer, default_device, ensure_dir, write_json


POLICY_KEYS = ("bc1800", "run2_true_tcp_ppo", "run5_dpsi_ppo")


def _sq_tcp_distance(state: np.ndarray, endpoint: np.ndarray) -> np.ndarray:
    delta = state[:, 14:17] - endpoint
    return np.sum(delta * delta, axis=-1).astype(np.float32)


def _load_low_policy(
    low_kind: str,
    paths: dict[str, Path],
    device: torch.device,
) -> tuple[nn.Module, dict[str, Any]]:
    if low_kind == "bc1800":
        return _load_bc_low(paths[low_kind], device)
    return _load_rl_low(paths[low_kind], device)


@torch.inference_mode()
def _branch_rollout(
    args: argparse.Namespace,
    config: Any,
    branch_env: Any,
    reset_seeds: list[int],
    state_dict: dict[str, Any],
    low_kind: str,
    low_model: nn.Module,
    low_payload: dict[str, Any],
    endpoint: np.ndarray,
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
    del config
    branch_env.reset(seed=reset_seeds)
    branch_env.unwrapped.set_state_dict(_clone_mani_state_dict(state_dict))
    obs = branch_env.unwrapped.get_obs()
    goal_endpoint = endpoint.copy()
    if shuffled_goal:
        goal_endpoint = goal_endpoint[np.roll(np.arange(len(goal_endpoint)), 1)]
    start_state = _obs_state_np(obs)
    start_distance = _sq_tcp_distance(start_state, goal_endpoint)
    prev = previous_action.copy()
    action_saturation = np.zeros(len(goal_endpoint), dtype=np.float32)
    action_l2 = np.zeros(len(goal_endpoint), dtype=np.float32)
    terminal_reward = np.zeros(len(goal_endpoint), dtype=np.float32)
    max_reward = np.full(len(goal_endpoint), -np.inf, dtype=np.float32)
    for step in range(horizon):
        state = _obs_state_np(obs)
        remaining = np.full(len(goal_endpoint), max(horizon - step, 1), dtype=np.float32)
        raw_action = _low_action(
            low_kind,
            low_model,
            low_payload,
            state,
            goal_endpoint,
            prev,
            remaining,
            horizon,
            control_freq,
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
    terminal_distance = _sq_tcp_distance(terminal_state, goal_endpoint)
    mask = active.astype(bool)
    return {
        "initial_distance": start_distance[mask],
        "terminal_distance": terminal_distance[mask],
        "tcp_error_m": np.sqrt(np.maximum(terminal_distance[mask], 0.0)),
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
        "shuffled_goal": bool(shuffled_goal),
        "decisions": int(len(terminal)),
        "initial_distance_sq_mean": float(np.mean(initial)),
        "terminal_distance_sq_mean": float(np.mean(terminal)),
        "terminal_tcp_error_m_mean": float(np.mean(store["tcp_error_m"])),
        "distance_sq_reduction_mean": float(np.mean(initial - terminal)),
        "goal_reach_rate_eps": float(np.mean(terminal <= success_epsilon)),
        "p50_terminal_distance_sq": float(np.quantile(terminal, 0.50)),
        "p90_terminal_distance_sq": float(np.quantile(terminal, 0.90)),
        "p99_terminal_distance_sq": float(np.quantile(terminal, 0.99)),
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
    high_model, high_payload = _load_high(Path(args.high_checkpoint), device)
    high_input_norm = Standardizer.from_state_dict(high_payload["input_norm"])
    high_target_norm = Standardizer.from_state_dict(high_payload["target_norm"])
    high_action_norm = Standardizer.from_state_dict(high_payload["action_norm"])
    horizon = int(high_payload["horizon_steps"])
    update_period = int(args.update_period or horizon)
    control_freq = int(config.get("control_freq", 20))
    collector_model, collector_payload = _load_low_policy(collector_policy, paths, device)
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
    action_low_np = np.asarray(env.single_action_space.low, dtype=np.float32)
    action_high_np = np.asarray(env.single_action_space.high, dtype=np.float32)
    rows: dict[tuple[str, bool], dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    collected = 0
    progress = trange(args.decisions, desc=f"deploy-reach {collector_policy}")
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
            high_previous_norm = high_action_norm.transform(previous_action)
            endpoint = np.zeros((num_envs, 3), dtype=np.float32)
            countdown = np.zeros(num_envs, dtype=np.int32)
            active = np.ones(num_envs, dtype=bool)
            steps = 0
            while np.any(active) and collected < args.decisions and steps < args.max_steps:
                state = _obs_state_np(obs)
                replan = active & (countdown <= 0)
                if np.any(replan):
                    predicted_endpoint = high_target_norm.inverse(
                        high_model(
                            torch.from_numpy(
                                high_input_norm.transform(
                                    np.concatenate([state, high_previous_norm], axis=-1)
                                )
                            ).to(device).float()
                        ).cpu().numpy()
                    )
                    endpoint[replan] = predicted_endpoint[replan]
                    countdown[replan] = update_period
                    state_dict = _clone_mani_state_dict(env.unwrapped.get_state_dict())
                    for low_kind, (low_model, low_payload) in candidates.items():
                        for shuffled in (False, True):
                            metrics = _branch_rollout(
                                args,
                                config,
                                branch_env,
                                reset_seeds,
                                state_dict,
                                low_kind,
                                low_model,
                                low_payload,
                                endpoint,
                                previous_action,
                                replan,
                                shuffled_goal=shuffled,
                                device=device,
                                horizon=horizon,
                                control_freq=control_freq,
                                action_low_np=action_low_np,
                                action_high_np=action_high_np,
                            )
                            _extend(rows[(low_kind, shuffled)], metrics)
                    newly_collected = int(np.sum(replan))
                    collected += newly_collected
                    progress.update(min(newly_collected, args.decisions - progress.n))
                remaining = np.maximum(countdown, 1).astype(np.float32)
                raw_action = _low_action(
                    collector_policy,
                    collector_model,
                    collector_payload,
                    state,
                    endpoint,
                    previous_action,
                    remaining,
                    horizon,
                    control_freq,
                    device,
                )
                clipped = np.clip(raw_action, action_low_np, action_high_np).astype(np.float32)
                obs, _reward, terminated, truncated, info = env.step(
                    torch.from_numpy(clipped).to(device).float()
                )
                del info
                previous_action = clipped
                high_previous_norm = high_action_norm.transform(previous_action)
                countdown -= 1
                done = (
                    torch.logical_or(terminated, truncated)
                    .detach()
                    .cpu()
                    .numpy()
                    .reshape(-1)
                    .astype(bool)
                )
                active[done] = False
                steps += 1
    finally:
        progress.close()
        branch_env.close()
        env.close()
    return [
        _summary(
            collector_policy=collector_policy,
            candidate_policy=low_kind,
            shuffled_goal=shuffled,
            store=store,
            success_epsilon=float(args.success_epsilon),
        )
        for (low_kind, shuffled), store in sorted(rows.items())
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/pusht_incremental.yaml")
    parser.add_argument("--num-envs", type=int, default=10)
    parser.add_argument("--decisions", type=int, default=200)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--seed-start", type=int, default=2_070_000)
    parser.add_argument("--update-period", type=int)
    parser.add_argument("--success-epsilon", type=float, default=0.0025)
    parser.add_argument(
        "--high-checkpoint",
        default="artifacts/incremental/pre_rl/phase_f/privileged_tcp/seed0/predictor.pt",
    )
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
        default="results/incremental/rl_reachability_debug/run6_dpsi_b8_u1000/privileged_tcp_ppo_progress_terminal_n4096_seed0/latest.pt",
    )
    parser.add_argument(
        "--collector-policies",
        nargs="+",
        choices=POLICY_KEYS,
        default=["bc1800", "run5_dpsi_ppo"],
    )
    parser.add_argument(
        "--candidate-policies",
        nargs="+",
        choices=POLICY_KEYS,
        default=list(POLICY_KEYS),
    )
    parser.add_argument(
        "--output",
        default="results/incremental/rl_reachability_debug/deployment_reachability_eval.json",
    )
    args = parser.parse_args()
    paths = {
        "bc1800": Path(args.bc_low),
        "run2_true_tcp_ppo": Path(args.run2_low),
        "run5_dpsi_ppo": Path(args.run5_low),
    }
    rows: list[dict[str, Any]] = []
    for collector in args.collector_policies:
        rows.extend(evaluate_collector(args, collector, paths))
    payload = {
        "run": "rl_reachability_debug_deployment_reachability_eval",
        "high_checkpoint": str(args.high_checkpoint),
        "collector_policies": list(args.collector_policies),
        "candidate_policies": list(args.candidate_policies),
        "decisions_requested_per_collector": int(args.decisions),
        "num_envs": int(args.num_envs),
        "success_epsilon": float(args.success_epsilon),
        "rows": rows,
    }
    output = Path(args.output)
    ensure_dir(output.parent)
    write_json(output, payload)
    print(output)


if __name__ == "__main__":
    main()
