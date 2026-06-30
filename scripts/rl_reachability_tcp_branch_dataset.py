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
def _policy_sequence_state_and_distance(
    runner: LocalTcpPpo,
    env: Any,
    start_state: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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
        action_np = action.detach().cpu().numpy().astype(np.float32)
        actions.append(action_np)
        obs, _reward, _terminated, _truncated, _info = env.step(action)
        previous = runner.action_norm.transform(action_np)
    terminal_state = _obs_state_np(obs)
    return np.stack(actions, axis=0), terminal_state, _distance(terminal_state, runner.goal)


@torch.inference_mode()
def _sequence_terminal_state_and_distance(
    runner: LocalTcpPpo,
    env: Any,
    start_state: dict[str, Any],
    actions: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    env.unwrapped.set_state_dict(_clone_mani_state_dict(start_state))
    obs = env.unwrapped.get_obs()
    for step in range(runner.horizon):
        action = torch.from_numpy(actions[step]).to(runner.device).float()
        action = torch.clamp(action, runner.action_low, runner.action_high)
        obs, _reward, _terminated, _truncated, _info = env.step(action)
    terminal_state = _obs_state_np(obs)
    return terminal_state, _distance(terminal_state, runner.goal)


def _runner_args(args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(
        config=args.config,
        dataset=args.dataset,
        output_dir=str(Path(args.output).parent / "_tmp"),
        num_envs=None,
        horizon=args.horizon,
        updates=1,
        seed=args.seed,
        reward_mode="progress_terminal",
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


def run(args: argparse.Namespace) -> Path:
    set_seed(args.seed + 4_400_000)
    runner = LocalTcpPpo(_runner_args(args))
    checkpoint = torch.load(args.checkpoint, map_location=runner.device, weights_only=False)
    runner.agent.load_state_dict(checkpoint["agent"])
    runner.agent.eval()
    search_env = _make_benchmark_env(runner.config, runner.num_envs, "state")
    search_env.reset(seed=args.seed + 4_405_000)
    rng = np.random.default_rng(args.seed + 4_410_000)
    references = runner.sample_references(args.eval_refs, args.seed + 4_125_000)

    start_states = []
    goals = []
    ppo_terminal_states = []
    ppo_terminal_distances = []
    candidate_terminal_states = []
    candidate_terminal_distances = []
    best_indices = []
    best_terminal_states = []
    best_terminal_distances = []
    try:
        for reference in references:
            runner.reset_local_episode(reference)
            start_state = _clone_mani_state_dict(runner.env.unwrapped.get_state_dict())
            start_states.append(runner.current_state.copy())
            goals.append(runner.goal.copy())
            base_actions, ppo_state, ppo_distance = _policy_sequence_state_and_distance(
                runner,
                search_env,
                start_state,
            )
            ppo_terminal_states.append(ppo_state)
            ppo_terminal_distances.append(ppo_distance)

            ref_candidate_states = []
            ref_candidate_distances = []
            best_distance = ppo_distance.copy()
            best_state = ppo_state.copy()
            best_index = np.full(runner.num_envs, -1, dtype=np.int32)
            action_low = runner.action_low.detach().cpu().numpy()
            action_high = runner.action_high.detach().cpu().numpy()
            for candidate in range(args.random_candidates):
                noise = rng.normal(
                    0.0,
                    args.noise_std,
                    size=base_actions.shape,
                ).astype(np.float32)
                actions = np.clip(base_actions + noise, action_low, action_high)
                terminal_state, distance = _sequence_terminal_state_and_distance(
                    runner,
                    search_env,
                    start_state,
                    actions,
                )
                ref_candidate_states.append(terminal_state)
                ref_candidate_distances.append(distance)
                improved = distance < best_distance
                best_distance = np.where(improved, distance, best_distance)
                best_state[improved] = terminal_state[improved]
                best_index[improved] = candidate
            candidate_terminal_states.append(np.stack(ref_candidate_states, axis=0))
            candidate_terminal_distances.append(np.stack(ref_candidate_distances, axis=0))
            best_indices.append(best_index)
            best_terminal_states.append(best_state)
            best_terminal_distances.append(best_distance)
    finally:
        search_env.close()
        runner.close()

    output = Path(args.output)
    ensure_dir(output.parent)
    start_arr = np.concatenate(start_states, axis=0).astype(np.float32)
    goal_arr = np.concatenate(goals, axis=0).astype(np.float32)
    ppo_state_arr = np.concatenate(ppo_terminal_states, axis=0).astype(np.float32)
    ppo_distance_arr = np.concatenate(ppo_terminal_distances, axis=0).astype(np.float32)
    candidate_state_arr = np.concatenate(candidate_terminal_states, axis=1).astype(np.float32)
    candidate_distance_arr = np.concatenate(candidate_terminal_distances, axis=1).astype(
        np.float32
    )
    best_state_arr = np.concatenate(best_terminal_states, axis=0).astype(np.float32)
    best_distance_arr = np.concatenate(best_terminal_distances, axis=0).astype(np.float32)
    best_index_arr = np.concatenate(best_indices, axis=0).astype(np.int32)
    shuffled_goal_arr = goal_arr[np.roll(np.arange(len(goal_arr)), 1)].astype(np.float32)
    np.savez_compressed(
        output,
        start_state=start_arr,
        goal=goal_arr,
        shuffled_goal=shuffled_goal_arr,
        ppo_terminal_state=ppo_state_arr,
        ppo_terminal_distance=ppo_distance_arr,
        candidate_terminal_state=candidate_state_arr,
        candidate_terminal_distance=candidate_distance_arr,
        best_candidate_index=best_index_arr,
        best_terminal_state=best_state_arr,
        best_terminal_distance=best_distance_arr,
        horizon=np.asarray(runner.horizon, dtype=np.int32),
        success_epsilon=np.asarray(args.success_epsilon, dtype=np.float32),
        noise_std=np.asarray(args.noise_std, dtype=np.float32),
        random_candidates=np.asarray(args.random_candidates, dtype=np.int32),
    )
    summary = {
        "dataset": str(output),
        "local_episodes": int(len(goal_arr)),
        "random_candidates": int(args.random_candidates),
        "noise_std": float(args.noise_std),
        "ppo_terminal_distance_mean": float(np.mean(ppo_distance_arr)),
        "best_terminal_distance_mean": float(np.mean(best_distance_arr)),
        "candidate_terminal_distance_mean": float(np.mean(candidate_distance_arr)),
        "ppo_reach_rate": float(np.mean(ppo_distance_arr <= args.success_epsilon)),
        "best_reach_rate": float(np.mean(best_distance_arr <= args.success_epsilon)),
        "candidate_reach_rate": float(np.mean(candidate_distance_arr <= args.success_epsilon)),
        "fraction_best_improved_vs_ppo": float(np.mean(best_distance_arr < ppo_distance_arr)),
    }
    write_json(output.with_suffix(".json"), summary)
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
        default="data/rl_reachability_debug/run4_tcp_branch_dataset_c64_ref2.npz",
    )
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument("--eval-refs", type=int, default=2)
    parser.add_argument("--random-candidates", type=int, default=64)
    parser.add_argument("--noise-std", type=float, default=0.05)
    parser.add_argument("--success-epsilon", type=float, default=0.0025)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    print(run(args))


if __name__ == "__main__":
    main()
