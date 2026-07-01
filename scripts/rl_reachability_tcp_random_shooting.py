#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

sys.path.append(str(Path(__file__).resolve().parent))

from rl_reachability_privileged_tcp_ppo import LocalTcpPpo, _distance, _obs_state_np

from hcl_poc.privileged_z import _clone_mani_state_dict
from hcl_poc.rl_rerun import _make_benchmark_env
from hcl_poc.utils import ensure_dir, set_seed, write_json


@torch.inference_mode()
def _policy_sequence(
    runner: LocalTcpPpo,
    env: Any,
    start_state: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    env.unwrapped.set_state_dict(_clone_mani_state_dict(start_state))
    obs = env.unwrapped.get_obs()
    previous = runner.previous_action_norm.copy()
    actions = []
    for step in range(runner.horizon):
        state = _obs_state_np(obs)
        state_norm = runner.state_norm.transform(state)
        remaining = np.full(
            (runner.num_envs, 1),
            max(runner.horizon - step, 1) / runner.horizon,
            dtype=np.float32,
        )
        condition = np.concatenate(
            [state_norm, runner.goal_normed, previous, remaining],
            axis=-1,
        ).astype(np.float32)
        action, _logprob, _entropy, _value = runner.agent.get_action_and_value(
            torch.from_numpy(condition).to(runner.device).float(),
            deterministic=True,
        )
        action = torch.clamp(action, runner.action_low, runner.action_high)
        actions.append(action.detach().cpu().numpy().astype(np.float32))
        obs, _reward, _terminated, _truncated, _info = env.step(action)
        previous = runner.action_norm.transform(actions[-1])
    terminal = _distance(_obs_state_np(obs), runner.goal)
    return np.stack(actions, axis=0), terminal


@torch.inference_mode()
def _sequence_terminal_distance(
    runner: LocalTcpPpo,
    env: Any,
    start_state: dict[str, Any],
    actions: np.ndarray,
) -> np.ndarray:
    env.unwrapped.set_state_dict(_clone_mani_state_dict(start_state))
    obs = env.unwrapped.get_obs()
    for step in range(runner.horizon):
        action = torch.from_numpy(actions[step]).to(runner.device).float()
        action = torch.clamp(action, runner.action_low, runner.action_high)
        obs, _reward, _terminated, _truncated, _info = env.step(action)
    return _distance(_obs_state_np(obs), runner.goal)


def run(args: argparse.Namespace) -> Path:
    set_seed(args.seed + 4_300_000)
    runner_args = argparse.Namespace(
        config=args.config,
        dataset=args.dataset,
        output_dir=args.output_dir,
        num_envs=None,
        horizon=args.horizon,
        goal_type="tcp",
        updates=1,
        seed=args.seed,
        reward_mode="progress_terminal",
        reward_distance_source="true_tcp",
        dpsi_checkpoint=None,
        init_checkpoint=None,
        bc_low_checkpoint="artifacts/incremental/pre_rl/phase_c/k10/seed0/time_conditioned_tcp.pt",
        dpsi_target_scale=1000.0,
        terminal_weight=1.0,
        distance_progress_weight=1.0,
        success_epsilon=args.success_epsilon,
        num_minibatches=8,
        update_epochs=3,
        learning_rate=3e-4,
        initial_logstd=-1.0,
        width=256,
        depth=2,
        gamma=0.99,
        gae_lambda=0.95,
        clip_coef=0.2,
        entropy_coef=0.0,
        value_coef=1.0,
        max_grad_norm=1.0,
        checkpoint_every_updates=1,
        eval_episodes=args.eval_refs,
        force=False,
    )
    runner = LocalTcpPpo(runner_args)
    checkpoint = torch.load(args.checkpoint, map_location=runner.device, weights_only=False)
    runner.agent.load_state_dict(checkpoint["agent"])
    runner.agent.eval()
    search_env = _make_benchmark_env(runner.config, runner.num_envs, "state")
    search_env.reset(seed=args.seed + 4_305_000)
    rng = np.random.default_rng(args.seed + 4_310_000)
    references = runner.sample_references(args.eval_refs, args.seed + 4_125_000)
    initial_distances = []
    ppo_distances = []
    best_by_setting: dict[str, list[np.ndarray]] = {}
    try:
        for reference in references:
            runner.reset_local_episode(reference)
            start_state = _clone_mani_state_dict(runner.env.unwrapped.get_state_dict())
            initial_distances.append(_distance(runner.current_state, runner.goal))
            base_actions, base_distance = _policy_sequence(runner, search_env, start_state)
            ppo_distances.append(base_distance)
            for candidates in args.random_candidates:
                for noise_std in args.noise_stds:
                    key = f"c{candidates}_std{noise_std:g}"
                    best = base_distance.copy()
                    for _candidate in range(candidates):
                        noise = rng.normal(
                            0.0,
                            noise_std,
                            size=base_actions.shape,
                        ).astype(np.float32)
                        candidate_actions = np.clip(
                            base_actions + noise,
                            runner.action_low.detach().cpu().numpy(),
                            runner.action_high.detach().cpu().numpy(),
                        )
                        distance = _sequence_terminal_distance(
                            runner,
                            search_env,
                            start_state,
                            candidate_actions,
                        )
                        best = np.minimum(best, distance)
                    best_by_setting.setdefault(key, []).append(best)
    finally:
        search_env.close()
        runner.close()

    initial = np.concatenate(initial_distances)
    ppo = np.concatenate(ppo_distances)
    settings = {}
    for key, values in best_by_setting.items():
        best = np.concatenate(values)
        settings[key] = {
            "terminal_distance_mean": float(np.mean(best)),
            "goal_reach_rate_eps": float(np.mean(best <= args.success_epsilon)),
            "paired_improvement_vs_ppo": float(np.mean(ppo - best)),
            "fraction_improved_vs_ppo": float(np.mean(best < ppo)),
            "p50_terminal_distance": float(np.quantile(best, 0.50)),
            "p90_terminal_distance": float(np.quantile(best, 0.90)),
            "p99_terminal_distance": float(np.quantile(best, 0.99)),
        }
    payload = {
        "run": "rl_reachability_debug_run3_tcp_random_shooting_pilot",
        "checkpoint": str(args.checkpoint),
        "dataset": str(args.dataset),
        "eval_refs": int(args.eval_refs),
        "local_episodes": int(len(ppo)),
        "horizon": int(args.horizon),
        "success_epsilon": float(args.success_epsilon),
        "baseline": {
            "initial_distance_mean": float(np.mean(initial)),
            "ppo_terminal_distance_mean": float(np.mean(ppo)),
            "ppo_goal_reach_rate_eps": float(np.mean(ppo <= args.success_epsilon)),
            "ppo_p50_terminal_distance": float(np.quantile(ppo, 0.50)),
            "ppo_p90_terminal_distance": float(np.quantile(ppo, 0.90)),
            "ppo_p99_terminal_distance": float(np.quantile(ppo, 0.99)),
        },
        "random_shooting": settings,
    }
    output = Path(args.output)
    ensure_dir(output.parent)
    write_json(output, payload)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/pusht_incremental.yaml")
    parser.add_argument(
        "--dataset",
        default="data/rl_rerun/pusht_vector_state_demos_n4096_b2.h5",
    )
    parser.add_argument(
        "--checkpoint",
        default=(
            "results/incremental/rl_reachability_debug/run2_privileged_tcp/"
            "privileged_tcp_ppo_progress_terminal_n4096_seed0/latest.pt"
        ),
    )
    parser.add_argument(
        "--output",
        default="results/incremental/rl_reachability_debug/run3_tcp_random_shooting_pilot.json",
    )
    parser.add_argument("--output-dir", default="results/incremental/rl_reachability_debug/run3_tmp")
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument("--eval-refs", type=int, default=1)
    parser.add_argument("--random-candidates", type=int, nargs="+", default=[32])
    parser.add_argument("--noise-stds", type=float, nargs="+", default=[0.05])
    parser.add_argument("--success-epsilon", type=float, default=0.0025)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    print(run(args))


if __name__ == "__main__":
    main()
