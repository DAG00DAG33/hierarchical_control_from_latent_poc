#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
from torch import nn

from hcl_poc.config import load_config
from hcl_poc.incremental import _pre_rl_phase_b_goal
from hcl_poc.models import MLP
from hcl_poc.rl_rerun import _make_benchmark_env
from hcl_poc.utils import Standardizer, default_device, ensure_dir, write_json

from rl_reachability_privileged_tcp_ppo import _batch_keys, _goal_distance, _obs_state_np


def _load_phase_c_full(path: Path, device: torch.device) -> tuple[nn.Module, dict[str, Any]]:
    payload = torch.load(path, map_location=device, weights_only=False)
    entry = payload["full"]
    model = MLP(
        int(entry["cond_dim"]),
        int(entry["action_dim"]),
        int(entry["hidden_dim"]),
        depth=4,
    ).to(device)
    model.load_state_dict(entry["model"])
    model.eval()
    return model, payload


def _sample_references(
    h5: h5py.File,
    keys: list[str],
    horizon: int,
    count: int,
    seed: int,
) -> list[tuple[str, int]]:
    rng = np.random.default_rng(seed)
    refs = []
    for _ in range(count):
        key = str(rng.choice(keys))
        group = h5[key]
        max_start = int(group["executed_actions"].shape[0]) - horizon + 1
        refs.append((key, int(rng.integers(0, max_start))))
    return refs


@torch.inference_mode()
def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    config = load_config(args.config)
    device = default_device()
    horizon = int(args.horizon)
    control_freq = int(config.get("control_freq", 20))
    checkpoint_path = Path(args.checkpoint)
    model, payload = _load_phase_c_full(checkpoint_path, device)
    entry = payload["full"]
    action_norm = Standardizer.from_state_dict(payload["action_norm"])
    cond_norm = Standardizer.from_state_dict(entry["cond_norm"])
    env = _make_benchmark_env(config, int(args.num_envs), "state")
    action_low = np.asarray(env.single_action_space.low, dtype=np.float32)
    action_high = np.asarray(env.single_action_space.high, dtype=np.float32)
    terminal_distances: list[float] = []
    initial_distances: list[float] = []
    reductions: list[float] = []
    action_l2: list[float] = []
    action_saturation: list[float] = []
    with h5py.File(args.dataset, "r") as h5:
        keys = _batch_keys(h5)
        refs = _sample_references(h5, keys, horizon, int(args.references), int(args.seed))
        for key, t in refs:
            group = h5[key]
            obs, _info = env.reset(seed=int(group.attrs["batch_seed"]))
            for replay_step in range(t):
                replay_action = torch.from_numpy(
                    np.asarray(group["executed_actions"][replay_step], dtype=np.float32)
                ).to(device)
                obs, _reward, _terminated, _truncated, _info = env.step(replay_action)
            state = _obs_state_np(obs)
            previous_action_raw = np.asarray(
                group["previous_executed_actions"][t],
                dtype=np.float32,
            )
            target_future_state = np.asarray(
                group["observations_state"][t + horizon],
                dtype=np.float32,
            )
            start_goal = _pre_rl_phase_b_goal(
                state,
                target_future_state,
                horizon,
                control_freq,
                "full",
            )
            start_distance = _goal_distance(
                state,
                start_goal,
                "full",
                horizon,
                control_freq,
            )
            initial_distances.extend(start_distance.astype(float).tolist())
            for step in range(horizon):
                state = _obs_state_np(obs)
                remaining = np.full(
                    int(args.num_envs),
                    max(horizon - step, 1),
                    dtype=np.int32,
                )
                goal = _pre_rl_phase_b_goal(
                    state,
                    target_future_state,
                    int(max(horizon - step, 1)),
                    control_freq,
                    "full",
                )
                previous_norm = action_norm.transform(previous_action_raw)
                condition = np.concatenate(
                    [
                        state,
                        goal,
                        previous_norm,
                        (remaining.astype(np.float32) / horizon)[:, None],
                    ],
                    axis=-1,
                ).astype(np.float32)
                normalized = model(
                    torch.from_numpy(cond_norm.transform(condition)).to(device).float()
                )
                raw_action = action_norm.inverse(normalized.cpu().numpy()).astype(np.float32)
                action_saturation.extend(
                    np.any((raw_action < action_low) | (raw_action > action_high), axis=-1)
                    .astype(np.float32)
                    .tolist()
                )
                clipped = np.clip(raw_action, action_low, action_high).astype(np.float32)
                action_l2.extend(np.linalg.norm(clipped, axis=-1).astype(float).tolist())
                obs, _reward, _terminated, _truncated, _info = env.step(
                    torch.from_numpy(clipped).to(device).float()
                )
                previous_action_raw = clipped
            final_state = _obs_state_np(obs)
            final_goal = _pre_rl_phase_b_goal(
                final_state,
                target_future_state,
                1,
                control_freq,
                "full",
            )
            terminal = _goal_distance(final_state, final_goal, "full", horizon, control_freq)
            terminal_distances.extend(terminal.astype(float).tolist())
            reductions.extend((start_distance - terminal).astype(float).tolist())
    env.close()
    terminal_np = np.asarray(terminal_distances, dtype=np.float32)
    initial_np = np.asarray(initial_distances, dtype=np.float32)
    reduction_np = np.asarray(reductions, dtype=np.float32)
    return {
        "checkpoint": str(checkpoint_path),
        "dataset": str(args.dataset),
        "references": int(args.references),
        "local_episodes": int(len(terminal_np)),
        "horizon": horizon,
        "goal_type": "full",
        "initial_distance_mean": float(np.mean(initial_np)),
        "terminal_distance_mean": float(np.mean(terminal_np)),
        "distance_reduction_mean": float(np.mean(reduction_np)),
        "fraction_improved_from_start": float(np.mean(reduction_np > 0.0)),
        "goal_reach_rate_eps": float(np.mean(terminal_np <= float(args.success_epsilon))),
        "p50_terminal_distance": float(np.quantile(terminal_np, 0.50)),
        "p90_terminal_distance": float(np.quantile(terminal_np, 0.90)),
        "p99_terminal_distance": float(np.quantile(terminal_np, 0.99)),
        "action_saturation": float(np.mean(action_saturation)),
        "action_l2_mean": float(np.mean(action_l2)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/pusht_incremental.yaml")
    parser.add_argument(
        "--dataset",
        default="data/rl_rerun/pusht_vector_state_demos_n4096_b8.h5",
    )
    parser.add_argument(
        "--checkpoint",
        default="artifacts/incremental/pre_rl/phase_c/k10/seed0/time_conditioned_full.pt",
    )
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument("--num-envs", type=int, default=4096)
    parser.add_argument("--references", type=int, default=8)
    parser.add_argument("--seed", type=int, default=4_125_000)
    parser.add_argument("--success-epsilon", type=float, default=0.0025)
    parser.add_argument(
        "--output",
        default="results/incremental/rl_reachability_debug/phasec_full_bc_local_eval_ref8.json",
    )
    args = parser.parse_args()
    payload = evaluate(args)
    output = Path(args.output)
    ensure_dir(output.parent)
    write_json(output, payload)
    print(output)


if __name__ == "__main__":
    main()
