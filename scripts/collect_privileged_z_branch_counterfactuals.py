#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import torch

from hcl_poc.config import load_config
from hcl_poc.low_level_rl import ResidualActorCritic
from hcl_poc.privileged_z import (
    PrivilegedZDirectActorCritic,
    _clone_mani_state_dict,
    _model_from_payload,
    _obs_state_np,
    _predict_loaded_payload,
)
from hcl_poc.rl_rerun import _make_benchmark_env, _residual_action_from_raw, _to_numpy
from hcl_poc.utils import Standardizer, default_device


def _load_branch_bank(path: Path, state_dim: int, action_dim: int) -> dict[str, np.ndarray]:
    data = np.load(path, allow_pickle=True)
    horizon_steps = int(np.asarray(data["horizon_steps"]).item())
    conditions = np.asarray(data["conditions"], dtype=np.float32)
    branch_count = len(np.asarray(data["selected_return_delta"]))
    expected_dim = state_dim * 2 + action_dim + 1
    if conditions.shape != (branch_count * horizon_steps, expected_dim):
        raise ValueError(
            f"Expected branch conditions {(branch_count * horizon_steps, expected_dim)}, "
            f"got {conditions.shape}"
        )
    first_conditions = conditions.reshape(horizon_steps, branch_count, expected_dim)[0]
    return {
        "states": first_conditions[:, :state_dim].copy(),
        "goals": first_conditions[:, state_dim : state_dim * 2].copy(),
        "previous": first_conditions[:, state_dim * 2 : state_dim * 2 + action_dim].copy(),
        "return_delta": np.asarray(data["selected_return_delta"], dtype=np.float32),
        "success_delta": np.asarray(data["selected_success_delta"], dtype=np.float32),
    }


def _nearest_candidates(
    query_state: np.ndarray,
    query_goal: np.ndarray,
    bank_states: np.ndarray,
    bank_goals: np.ndarray,
    candidates_per_query: int,
) -> tuple[np.ndarray, np.ndarray]:
    state_diff = query_state[:, None, :] - bank_states[None, :, :]
    goal_diff = query_goal[:, None, :] - bank_goals[None, :, :]
    score = 0.5 * np.mean(state_diff * state_diff, axis=-1) + 0.5 * np.mean(
        goal_diff * goal_diff,
        axis=-1,
    )
    k = min(candidates_per_query, score.shape[1])
    indices = np.argpartition(score, kth=k - 1, axis=1)[:, :k]
    row = np.arange(len(score))[:, None]
    order = np.argsort(score[row, indices], axis=1)
    sorted_indices = indices[row, order]
    sorted_score = score[row, sorted_indices]
    return sorted_indices.astype(np.int64), sorted_score.astype(np.float32)


def _load_tuned_agent(
    residual_checkpoint: Path | None,
    base_payload: dict[str, Any],
    goal_model: torch.nn.Module,
    action_norm: Standardizer,
    device: torch.device,
) -> tuple[ResidualActorCritic | None, PrivilegedZDirectActorCritic | None, dict[str, Any] | None]:
    if residual_checkpoint is None:
        return None, None, None
    tuned_payload = torch.load(residual_checkpoint, map_location=device, weights_only=False)
    recipe = dict(tuned_payload["recipe"])
    method = str(recipe.get("method", ""))
    if method == "privileged_z_residual_r1":
        agent = ResidualActorCritic(
            int(tuned_payload["condition_dim"]),
            action_dim=int(tuned_payload["action_dim"]),
            width=int(recipe["actor_critic_width"]),
            depth=int(recipe["actor_critic_depth"]),
            initial_logstd=float(recipe["initial_logstd"]),
        ).to(device)
        agent.load_state_dict(tuned_payload["agent"])
        agent.eval()
        return agent, None, recipe
    if method in {"privileged_z_direct_r3", "privileged_z_direct_distill"}:
        agent = PrivilegedZDirectActorCritic(
            goal_model,
            base_payload["goal"],
            action_norm.mean,
            action_norm.std,
            int(tuned_payload["condition_dim"]),
            action_dim=int(tuned_payload["action_dim"]),
            train_scope=str(recipe["train_scope"]),
            width=int(recipe["actor_critic_width"]),
            depth=int(recipe["actor_critic_depth"]),
            initial_logstd=float(recipe["initial_logstd"]),
        ).to(device)
        agent.load_state_dict(tuned_payload["agent"])
        agent.eval()
        return None, agent, recipe
    raise ValueError(f"Unsupported tuned checkpoint method: {method}")


def collect_counterfactuals(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    device = default_device()
    payload = torch.load(args.checkpoint, map_location=device, weights_only=False)
    state_norm = Standardizer.from_state_dict(payload["state_norm"])
    action_norm = Standardizer.from_state_dict(payload["action_norm"])
    high_model = _model_from_payload(payload["high"], device)
    goal_model = _model_from_payload(payload["goal"], device)
    horizon_steps = int(payload["horizon_steps"])
    state_dim = int(payload["state_dim"])
    action_dim = int(payload["action_dim"])
    bank = _load_branch_bank(args.branch_bank, state_dim, action_dim)
    residual_agent, direct_agent, tuned_recipe = _load_tuned_agent(
        args.residual_checkpoint,
        payload,
        goal_model,
        action_norm,
        device,
    )

    env = _make_benchmark_env(config, args.num_envs, "rgb+state")
    rollout_env = _make_benchmark_env(config, args.num_envs, "rgb+state")
    rollout_env.reset(seed=args.seed_start)
    action_low = torch.as_tensor(env.single_action_space.low, device=device, dtype=torch.float32)
    action_high = torch.as_tensor(env.single_action_space.high, device=device, dtype=torch.float32)
    zero_previous = action_norm.transform(np.zeros((1, action_dim), dtype=np.float32))[0]

    def low_action(
        state_np: np.ndarray,
        previous_norm: np.ndarray,
        goal_norm: np.ndarray,
        remaining_count: np.ndarray,
        *,
        use_tuned: bool,
    ) -> tuple[torch.Tensor, np.ndarray]:
        normalized_state = state_norm.transform(state_np)
        remaining = np.maximum(remaining_count, 1).astype(np.float32)[:, None] / float(
            horizon_steps
        )
        condition = np.concatenate(
            [normalized_state, goal_norm, previous_norm, remaining],
            axis=-1,
        ).astype(np.float32)
        with torch.inference_mode():
            normalized_action = _predict_loaded_payload(
                goal_model,
                payload["goal"],
                condition,
                device,
            )
        base_action = torch.as_tensor(
            action_norm.inverse(normalized_action),
            device=device,
            dtype=torch.float32,
        )
        if use_tuned and direct_agent is not None:
            with torch.inference_mode():
                raw_action, _logprob, _entropy, _value = direct_agent.get_action_and_value(
                    torch.from_numpy(condition).to(device).float(),
                    deterministic=True,
                )
            action = torch.clamp(raw_action, action_low, action_high)
        elif use_tuned and residual_agent is not None and tuned_recipe is not None:
            with torch.inference_mode():
                raw_residual, _logprob, _entropy, _value = residual_agent.get_action_and_value(
                    torch.from_numpy(condition).to(device).float(),
                    deterministic=True,
                )
                _residual, _unclipped, action = _residual_action_from_raw(
                    base_action,
                    raw_residual,
                    float(tuned_recipe["alpha"]),
                    action_low,
                    action_high,
                    str(tuned_recipe.get("residual_action_mode", "additive")),
                )
        else:
            action = torch.clamp(base_action, action_low, action_high)
        return action, condition

    def high_goal(current_norm: np.ndarray, previous_norm: np.ndarray) -> np.ndarray:
        with torch.inference_mode():
            return _predict_loaded_payload(
                high_model,
                payload["high"],
                np.concatenate([current_norm, previous_norm], axis=-1).astype(np.float32),
                device,
            ).astype(np.float32)

    def run_rollout(
        start_state: dict[str, Any],
        previous_start: np.ndarray,
        first_goal: np.ndarray,
        *,
        first_segment_tuned: bool,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        rollout_env.unwrapped.set_state_dict(_clone_mani_state_dict(start_state))
        obs = rollout_env.unwrapped.get_obs()
        previous = previous_start.copy()
        held_goal = first_goal.copy()
        countdown = np.full(args.num_envs, horizon_steps, dtype=np.int32)
        completed = np.zeros(args.num_envs, dtype=np.bool_)
        success_once = np.zeros(args.num_envs, dtype=np.bool_)
        returns = np.zeros(args.num_envs, dtype=np.float32)
        final_success = np.zeros(args.num_envs, dtype=np.float32)
        final_return = np.zeros(args.num_envs, dtype=np.float32)

        def record(reward: Any, info: dict[str, Any]) -> None:
            reward_np = _to_numpy(reward).reshape(-1).astype(np.float32)
            returns[:] += reward_np
            if "success" in info:
                success_once[:] |= _to_numpy(info["success"]).reshape(-1).astype(np.bool_)
            if "final_info" not in info:
                return
            mask = info["_final_info"]
            mask_np = mask.detach().cpu().numpy().astype(np.bool_)
            if not np.any(mask_np):
                return
            done_indices = np.flatnonzero(mask_np)
            final_info = info["final_info"]
            if "episode" in final_info:
                ep = final_info["episode"]
                success_values = ep["success_once"][mask].detach().float().cpu().numpy()
                return_values = ep["return"][mask].detach().float().cpu().numpy()
            else:
                success_values = success_once[done_indices].astype(np.float32)
                return_values = returns[done_indices]
            for local_idx, env_idx in enumerate(done_indices):
                if completed[env_idx]:
                    continue
                completed[env_idx] = True
                final_success[env_idx] = float(success_values[local_idx])
                final_return[env_idx] = float(return_values[local_idx])

        for step in range(args.max_rollout_steps):
            current_norm = state_norm.transform(_obs_state_np(obs))
            replan = countdown <= 0
            if np.any(replan):
                next_goal = high_goal(current_norm, previous)
                held_goal[replan] = next_goal[replan]
                countdown[replan] = horizon_steps
            use_tuned = first_segment_tuned and step < horizon_steps
            action, _condition = low_action(
                _obs_state_np(obs),
                previous,
                held_goal,
                countdown,
                use_tuned=use_tuned,
            )
            countdown -= 1
            obs, reward, _terminated, _truncated, info = rollout_env.step(action)
            record(reward, info)
            previous = action_norm.transform(action.detach().cpu().numpy().astype(np.float32))
            previous[completed] = zero_previous
            held_goal[completed] = 0.0
            countdown[completed] = 0
            if bool(np.all(completed)):
                break
        final_success[~completed] = success_once[~completed].astype(np.float32)
        final_return[~completed] = returns[~completed]
        return final_success, final_return, completed

    obs, _info = env.reset(seed=args.seed_start)
    previous = np.repeat(zero_previous[None], args.num_envs, axis=0)
    held_goal = np.zeros((args.num_envs, state_dim), dtype=np.float32)
    countdown = np.zeros(args.num_envs, dtype=np.int32)

    query_states: list[np.ndarray] = []
    query_goals: list[np.ndarray] = []
    query_previous: list[np.ndarray] = []
    candidate_indices: list[np.ndarray] = []
    candidate_scores: list[np.ndarray] = []
    base_successes: list[np.ndarray] = []
    base_returns: list[np.ndarray] = []
    candidate_successes: list[np.ndarray] = []
    candidate_returns: list[np.ndarray] = []
    completed_values: list[np.ndarray] = []

    try:
        for _batch in range(args.query_batches):
            state_np = _obs_state_np(obs)
            current_norm = state_norm.transform(state_np)
            query_goal = high_goal(current_norm, previous)
            start_state = _clone_mani_state_dict(env.unwrapped.get_state_dict())
            indices, scores = _nearest_candidates(
                current_norm,
                query_goal,
                bank["states"],
                bank["goals"],
                args.candidates_per_query,
            )
            base_success, base_return, base_completed = run_rollout(
                start_state,
                previous.copy(),
                query_goal.copy(),
                first_segment_tuned=False,
            )
            cand_success_blocks: list[np.ndarray] = []
            cand_return_blocks: list[np.ndarray] = []
            cand_completed_blocks: list[np.ndarray] = []
            for candidate_slot in range(indices.shape[1]):
                candidate_goal = bank["goals"][indices[:, candidate_slot]]
                cand_success, cand_return, cand_completed = run_rollout(
                    start_state,
                    previous.copy(),
                    candidate_goal.copy(),
                    first_segment_tuned=direct_agent is not None or residual_agent is not None,
                )
                cand_success_blocks.append(cand_success)
                cand_return_blocks.append(cand_return)
                cand_completed_blocks.append(cand_completed)
            query_states.append(current_norm.copy())
            query_goals.append(query_goal.copy())
            query_previous.append(previous.copy())
            candidate_indices.append(indices.copy())
            candidate_scores.append(scores.copy())
            base_successes.append(base_success.copy())
            base_returns.append(base_return.copy())
            candidate_successes.append(np.stack(cand_success_blocks, axis=1))
            candidate_returns.append(np.stack(cand_return_blocks, axis=1))
            completed_values.append(
                np.stack(cand_completed_blocks, axis=1) & base_completed[:, None]
            )

            held_goal[:] = query_goal
            countdown[:] = horizon_steps
            for _step in range(horizon_steps):
                action, _condition = low_action(
                    _obs_state_np(obs),
                    previous,
                    held_goal,
                    countdown,
                    use_tuned=False,
                )
                countdown -= 1
                obs, _reward, _terminated, _truncated, _info = env.step(action)
                previous = action_norm.transform(
                    action.detach().cpu().numpy().astype(np.float32)
                )
    finally:
        env.close()
        rollout_env.close()

    q_state = np.concatenate(query_states, axis=0).astype(np.float32)
    q_goal = np.concatenate(query_goals, axis=0).astype(np.float32)
    q_prev = np.concatenate(query_previous, axis=0).astype(np.float32)
    cand_idx = np.concatenate(candidate_indices, axis=0).astype(np.int64)
    cand_score = np.concatenate(candidate_scores, axis=0).astype(np.float32)
    base_success = np.concatenate(base_successes, axis=0).astype(np.float32)
    base_return = np.concatenate(base_returns, axis=0).astype(np.float32)
    cand_success = np.concatenate(candidate_successes, axis=0).astype(np.float32)
    cand_return = np.concatenate(candidate_returns, axis=0).astype(np.float32)
    completed = np.concatenate(completed_values, axis=0).astype(np.bool_)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        query_states=q_state,
        query_goals=q_goal,
        query_previous=q_prev,
        candidate_indices=cand_idx,
        candidate_goals=bank["goals"][cand_idx].astype(np.float32),
        candidate_nearest_scores=cand_score,
        candidate_source_return_delta=bank["return_delta"][cand_idx].astype(np.float32),
        candidate_source_success_delta=bank["success_delta"][cand_idx].astype(np.float32),
        base_success=base_success,
        base_return=base_return,
        candidate_success=cand_success,
        candidate_return=cand_return,
        candidate_success_delta=(cand_success - base_success[:, None]).astype(np.float32),
        candidate_return_delta=(cand_return - base_return[:, None]).astype(np.float32),
        completed=completed,
        checkpoint=np.asarray(str(args.checkpoint)),
        residual_checkpoint=np.asarray("" if args.residual_checkpoint is None else str(args.residual_checkpoint)),
        branch_bank=np.asarray(str(args.branch_bank)),
        seed_start=np.asarray(args.seed_start, dtype=np.int64),
        num_envs=np.asarray(args.num_envs, dtype=np.int64),
        query_batches=np.asarray(args.query_batches, dtype=np.int64),
        candidates_per_query=np.asarray(args.candidates_per_query, dtype=np.int64),
        max_rollout_steps=np.asarray(args.max_rollout_steps, dtype=np.int64),
    )
    best_delta = np.max(cand_return - base_return[:, None], axis=1)
    first_delta = cand_return[:, 0] - base_return
    print(
        {
            "output": str(args.output),
            "queries": int(len(q_state)),
            "candidates_per_query": int(cand_idx.shape[1]),
            "base_success": float(np.mean(base_success)),
            "base_return": float(np.mean(base_return)),
            "nearest_return_delta_mean": float(np.mean(first_delta)),
            "best_candidate_return_delta_mean": float(np.mean(best_delta)),
            "positive_best_return_delta_fraction": float(np.mean(best_delta > 5.0)),
            "best_candidate_success_delta_mean": float(
                np.mean(np.max(cand_success - base_success[:, None], axis=1))
            ),
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/pusht_incremental.yaml")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--residual-checkpoint", type=Path)
    parser.add_argument("--branch-bank", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed-start", type=int, default=9_960_000)
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--query-batches", type=int, default=2)
    parser.add_argument("--candidates-per-query", type=int, default=8)
    parser.add_argument("--max-rollout-steps", type=int, default=120)
    args = parser.parse_args()
    collect_counterfactuals(args)


if __name__ == "__main__":
    main()
