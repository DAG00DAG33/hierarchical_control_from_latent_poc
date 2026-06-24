from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import gymnasium as gym
import mani_skill  # noqa: F401
import numpy as np
import torch

from hcl_poc.config import load_config
from hcl_poc.incremental import (
    _phase4_dino_from_config,
    _phase4_frame_inputs,
    _phase7_obs_state_tensor,
)
from hcl_poc.learned_interface import _low_condition_array
from hcl_poc.low_level_rl import DirectLowActorCritic, ResidualActorCritic, _load_frozen
from hcl_poc.rl import _rl_backend, _rl_paths, load_ppo_agent
from hcl_poc.rl_rerun import (
    _encode_rerun_frames,
    _load_low_flow_base,
    _low_flow_base_action,
    _residual_action_from_raw,
    _residual_condition_array,
    _rerun_base_config,
    _to_numpy,
)
from hcl_poc.utils import default_device, ensure_dir, write_json


def _summary(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"mean": None, "median": None, "p90": None, "max": None}
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p90": float(np.quantile(array, 0.9)),
        "max": float(np.max(array)),
    }


def _make_env(config: Any, num_envs: int):
    return gym.make(
        config.get("env_id"),
        obs_mode="rgb+state",
        control_mode=config.get("control_mode"),
        reward_mode="normalized_dense",
        render_mode=None,
        sim_backend=_rl_backend(config),
        num_envs=num_envs,
        reconfiguration_freq=config.get("rl.eval_reconfiguration_freq", 1),
    )


@torch.inference_mode()
def run_goal_mismatch_audit(args: argparse.Namespace) -> Path:
    config = load_config(args.config)
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(checkpoint_path)
    if args.episodes <= 0 or args.num_envs <= 0:
        raise ValueError("episodes and num_envs must be positive")

    device = default_device()
    frozen = _load_frozen(_rerun_base_config(config), args.n_demo, args.seed, device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    recipe = checkpoint["recipe"]
    if int(recipe["n_demo"]) != args.n_demo or int(recipe["seed"]) != args.seed:
        raise ValueError("Checkpoint recipe does not match --n-demo/--seed")

    method = str(recipe.get("method", ""))
    is_direct = method.startswith("r3_direct")
    base_policy = str(recipe.get("base_policy", ""))
    flow_model = None
    flow_checkpoint = None
    if is_direct:
        base_policy = "deterministic"
    elif base_policy == "flow":
        flow_path = recipe.get("flow_checkpoint")
        if not flow_path:
            raise ValueError("R2 checkpoint is missing flow_checkpoint")
        flow_model, flow_checkpoint = _load_low_flow_base(Path(flow_path), device)
    elif base_policy != "deterministic":
        raise ValueError(f"Unknown base policy: {base_policy}")

    if is_direct:
        agent = DirectLowActorCritic(
            frozen.low_model,
            frozen.action_norm.mean,
            frozen.action_norm.std,
            int(checkpoint["condition_dim"]),
            width=int(recipe["actor_critic_width"]),
            depth=int(recipe["actor_critic_depth"]),
            initial_logstd=float(recipe["initial_logstd"]),
        ).to(device)
    else:
        agent = ResidualActorCritic(
            int(checkpoint["condition_dim"]),
            width=int(recipe["actor_critic_width"]),
            depth=int(recipe["actor_critic_depth"]),
            initial_logstd=float(recipe["initial_logstd"]),
        ).to(device)
    agent.load_state_dict(checkpoint["agent"])
    agent.eval()

    teacher = load_ppo_agent(_rl_paths(config).best, device)
    dino = _phase4_dino_from_config(config, device)
    max_steps = int(config.get("env_max_episode_steps", 100))
    alpha = float(recipe.get("alpha", 0.0))
    residual_condition_mode = str(recipe.get("residual_condition_mode", "full"))
    if residual_condition_mode not in {"full", "goal_delta"}:
        raise ValueError(f"Unknown residual_condition_mode: {residual_condition_mode}")
    residual_action_mode = str(recipe.get("residual_action_mode", "additive"))
    if residual_action_mode not in {"additive", "margin_scaled"}:
        raise ValueError(f"Unknown residual_action_mode: {residual_action_mode}")

    metrics: dict[str, list[float]] = {
        "learned_oracle_goal_l2": [],
        "learned_oracle_goal_mae": [],
        "oracle_goal_displacement_l2": [],
        "learned_goal_displacement_l2": [],
        "frozen_action_l2_learned_vs_oracle": [],
        "tuned_action_l2_learned_vs_oracle": [],
        "tuned_action_l2_learned_vs_frozen": [],
        "tuned_action_l2_oracle_vs_frozen": [],
        "replay_state_error": [],
    }
    raw_rows: list[dict[str, float | int]] = []
    successes: list[float] = []
    final_rewards: list[float] = []
    max_rewards: list[float] = []

    for batch_start in range(0, args.episodes, args.num_envs):
        batch_envs = min(args.num_envs, args.episodes - batch_start)
        reset_seeds = [args.eval_seed_start + batch_start + i for i in range(batch_envs)]
        student = _make_env(config, batch_envs)
        branch = _make_env(config, batch_envs)
        action_low = torch.as_tensor(
            np.asarray(student.action_space.low, dtype=np.float32), device=device
        )
        action_high = torch.as_tensor(
            np.asarray(student.action_space.high, dtype=np.float32), device=device
        )
        if action_low.ndim == 1:
            action_low = action_low.unsqueeze(0)
            action_high = action_high.unsqueeze(0)
        zero_previous = frozen.action_norm.transform(np.zeros((1, 3), dtype=np.float32))[0]
        previous_action = np.repeat(zero_previous[None], batch_envs, axis=0)
        held_goal = np.zeros((batch_envs, frozen.goal_dim), dtype=np.float32)
        countdown = np.zeros(batch_envs, dtype=np.int32)
        active = np.ones(batch_envs, dtype=np.bool_)
        success_once = np.zeros(batch_envs, dtype=np.bool_)
        batch_final = np.zeros(batch_envs, dtype=np.float32)
        batch_max = np.full(batch_envs, -np.inf, dtype=np.float32)
        history: list[torch.Tensor] = []
        try:
            obs, _info = student.reset(seed=reset_seeds)
            for step_index in range(max_steps):
                if not np.any(active):
                    break
                frames = _phase4_frame_inputs(obs, dino, int(config.get("dino.batch_size", 64)))
                normalized_frames = frozen.frame_norm.transform(frames)
                replan = active & (countdown <= 0)
                current_z = _encode_rerun_frames(frozen, frames, device)

                if np.any(replan):
                    high_input = np.concatenate([normalized_frames, previous_action], axis=-1)
                    learned_goal = (
                        frozen.high_model(torch.from_numpy(high_input).to(device).float())
                        .cpu()
                        .numpy()
                    )

                    branch_obs, _branch_info = branch.reset(seed=reset_seeds)
                    for action_history in history:
                        branch_obs, _reward, _term, _trunc, _info = branch.step(action_history)
                    replay_error = torch.max(
                        torch.abs(student.unwrapped.get_state() - branch.unwrapped.get_state()),
                        dim=1,
                    ).values
                    for _ in range(frozen.horizon_steps):
                        teacher_action = torch.clamp(
                            teacher.actor_mean(_phase7_obs_state_tensor(branch_obs, device)),
                            action_low,
                            action_high,
                        )
                        branch_obs, _reward, term, trunc, _info = branch.step(teacher_action)
                        if bool(torch.all(torch.logical_or(term, trunc))):
                            break
                    branch_frames = _phase4_frame_inputs(
                        branch_obs, dino, int(config.get("dino.batch_size", 64))
                    )
                    oracle_goal = _encode_rerun_frames(frozen, branch_frames, device)

                    remaining = np.full(
                        (batch_envs, 1), frozen.update_period / frozen.horizon_steps, dtype=np.float32
                    )
                    if frozen.conditioning in {"delta", "relation"}:
                        cond_current_z = current_z
                    else:
                        cond_current_z = np.empty_like(learned_goal)
                    learned_condition_np = _low_condition_array(
                        normalized_frames,
                        cond_current_z,
                        learned_goal,
                        previous_action,
                        remaining,
                        frozen.conditioning,
                    )
                    oracle_condition_np = _low_condition_array(
                        normalized_frames,
                        cond_current_z,
                        oracle_goal,
                        previous_action,
                        remaining,
                        frozen.conditioning,
                    )
                    learned_condition = torch.from_numpy(learned_condition_np).to(device).float()
                    oracle_condition = torch.from_numpy(oracle_condition_np).to(device).float()
                    if base_policy == "deterministic":
                        learned_base_norm = frozen.low_model(learned_condition)
                        oracle_base_norm = frozen.low_model(oracle_condition)
                        learned_base = torch.from_numpy(
                            frozen.action_norm.inverse(
                                learned_base_norm.cpu().numpy().astype(np.float32)
                            )
                        ).to(device)
                        oracle_base = torch.from_numpy(
                            frozen.action_norm.inverse(
                                oracle_base_norm.cpu().numpy().astype(np.float32)
                            )
                        ).to(device)
                    else:
                        if flow_model is None or flow_checkpoint is None:
                            raise RuntimeError("Flow base was not loaded")
                        learned_base = _low_flow_base_action(
                            flow_model, flow_checkpoint, learned_condition, frozen
                        )
                        oracle_base = _low_flow_base_action(
                            flow_model, flow_checkpoint, oracle_condition, frozen
                        )

                    if is_direct:
                        learned_tuned, *_ = agent.get_action_and_value(
                            learned_condition, deterministic=True
                        )
                        oracle_tuned, *_ = agent.get_action_and_value(
                            oracle_condition, deterministic=True
                        )
                    else:
                        learned_residual_condition_np = _residual_condition_array(
                            mode=residual_condition_mode,
                            full_condition=learned_condition_np,
                            current_z=current_z,
                            goal_z=learned_goal,
                            previous_action=previous_action,
                            remaining=remaining,
                        )
                        oracle_residual_condition_np = _residual_condition_array(
                            mode=residual_condition_mode,
                            full_condition=oracle_condition_np,
                            current_z=current_z,
                            goal_z=oracle_goal,
                            previous_action=previous_action,
                            remaining=remaining,
                        )
                        learned_residual_condition = torch.from_numpy(
                            learned_residual_condition_np
                        ).to(device).float()
                        oracle_residual_condition = torch.from_numpy(
                            oracle_residual_condition_np
                        ).to(device).float()
                        learned_residual, *_ = agent.get_action_and_value(
                            learned_residual_condition, deterministic=True
                        )
                        oracle_residual, *_ = agent.get_action_and_value(
                            oracle_residual_condition, deterministic=True
                        )
                        _learned_delta, learned_tuned, _learned_action = _residual_action_from_raw(
                            learned_base,
                            learned_residual,
                            alpha,
                            action_low,
                            action_high,
                            residual_action_mode,
                        )
                        _oracle_delta, oracle_tuned, _oracle_action = _residual_action_from_raw(
                            oracle_base,
                            oracle_residual,
                            alpha,
                            action_low,
                            action_high,
                            residual_action_mode,
                        )

                    selected = np.flatnonzero(replan)
                    learned_base_np = learned_base.cpu().numpy()
                    oracle_base_np = oracle_base.cpu().numpy()
                    learned_tuned_np = learned_tuned.cpu().numpy()
                    oracle_tuned_np = oracle_tuned.cpu().numpy()
                    replay_error_np = replay_error.cpu().numpy()
                    for index in selected:
                        row = {
                            "episode_index": int(batch_start + index),
                            "env_index": int(index),
                            "step": int(step_index),
                            "learned_oracle_goal_l2": float(
                                np.linalg.norm(learned_goal[index] - oracle_goal[index])
                            ),
                            "learned_oracle_goal_mae": float(
                                np.mean(np.abs(learned_goal[index] - oracle_goal[index]))
                            ),
                            "oracle_goal_displacement_l2": float(
                                np.linalg.norm(oracle_goal[index] - current_z[index])
                            ),
                            "learned_goal_displacement_l2": float(
                                np.linalg.norm(learned_goal[index] - current_z[index])
                            ),
                            "frozen_action_l2_learned_vs_oracle": float(
                                np.linalg.norm(learned_base_np[index] - oracle_base_np[index])
                            ),
                            "tuned_action_l2_learned_vs_oracle": float(
                                np.linalg.norm(learned_tuned_np[index] - oracle_tuned_np[index])
                            ),
                            "tuned_action_l2_learned_vs_frozen": float(
                                np.linalg.norm(learned_tuned_np[index] - learned_base_np[index])
                            ),
                            "tuned_action_l2_oracle_vs_frozen": float(
                                np.linalg.norm(oracle_tuned_np[index] - oracle_base_np[index])
                            ),
                            "replay_state_error": float(replay_error_np[index]),
                        }
                        raw_rows.append(row)
                        for key in metrics:
                            metrics[key].append(float(row[key]))

                    held_goal[replan] = learned_goal[replan]
                    countdown[replan] = frozen.update_period

                if frozen.conditioning in {"delta", "relation"}:
                    condition_current_z = current_z
                else:
                    condition_current_z = np.empty_like(held_goal)
                remaining_steps = np.maximum(countdown, 1).astype(np.float32)
                condition_np = _low_condition_array(
                    normalized_frames,
                    condition_current_z,
                    held_goal,
                    previous_action,
                    (remaining_steps / frozen.horizon_steps)[:, None],
                    frozen.conditioning,
                )
                condition = torch.from_numpy(condition_np).to(device).float()
                if is_direct:
                    action, *_ = agent.get_action_and_value(condition, deterministic=True)
                elif base_policy == "deterministic":
                    normalized_base = frozen.low_model(condition)
                    base_action = torch.from_numpy(
                        frozen.action_norm.inverse(
                            normalized_base.cpu().numpy().astype(np.float32)
                        )
                    ).to(device)
                    residual_condition_np = _residual_condition_array(
                        mode=residual_condition_mode,
                        full_condition=condition_np,
                        current_z=current_z,
                        goal_z=held_goal,
                        previous_action=previous_action,
                        remaining=(remaining_steps / frozen.horizon_steps)[:, None],
                    )
                    residual_condition = torch.from_numpy(
                        residual_condition_np
                    ).to(device).float()
                    residual, *_ = agent.get_action_and_value(
                        residual_condition, deterministic=True
                    )
                    _delta, action, _clipped_action = _residual_action_from_raw(
                        base_action,
                        residual,
                        alpha,
                        action_low,
                        action_high,
                        residual_action_mode,
                    )
                else:
                    if flow_model is None or flow_checkpoint is None:
                        raise RuntimeError("Flow base was not loaded")
                    base_action = _low_flow_base_action(flow_model, flow_checkpoint, condition, frozen)
                    residual_condition_np = _residual_condition_array(
                        mode=residual_condition_mode,
                        full_condition=condition_np,
                        current_z=current_z,
                        goal_z=held_goal,
                        previous_action=previous_action,
                        remaining=(remaining_steps / frozen.horizon_steps)[:, None],
                    )
                    residual_condition = torch.from_numpy(
                        residual_condition_np
                    ).to(device).float()
                    residual, *_ = agent.get_action_and_value(
                        residual_condition, deterministic=True
                    )
                    _delta, action, _clipped_action = _residual_action_from_raw(
                        base_action,
                        residual,
                        alpha,
                        action_low,
                        action_high,
                        residual_action_mode,
                    )
                active_tensor = torch.from_numpy(active).to(device)
                action = torch.clamp(action, action_low, action_high)
                action[~active_tensor] = 0.0
                obs, reward, terminated, truncated, info = student.step(action)
                history.append(action.detach().clone())
                executed = action.cpu().numpy().astype(np.float32)
                previous_action = frozen.action_norm.transform(executed)
                countdown -= 1
                reward_np = _to_numpy(reward).reshape(-1).astype(np.float32)
                batch_final[active] = reward_np[active]
                batch_max[active] = np.maximum(batch_max[active], reward_np[active])
                if "success" in info:
                    success_once |= _to_numpy(info["success"]).reshape(-1).astype(np.bool_)
                done = np.logical_or(
                    _to_numpy(terminated).reshape(-1), _to_numpy(truncated).reshape(-1)
                )
                active[done.astype(np.bool_)] = False
        finally:
            student.close()
            branch.close()
        successes.extend(success_once.astype(float).tolist())
        final_rewards.extend(batch_final.astype(float).tolist())
        max_rewards.extend(batch_max.astype(float).tolist())

    payload = {
        "method": "rl_rerun_goal_mismatch_audit",
        "checkpoint": str(checkpoint_path),
        "n_demo": args.n_demo,
        "seed": args.seed,
        "episodes": args.episodes,
        "eval_seed_start": args.eval_seed_start,
        "num_envs": args.num_envs,
        "horizon": frozen.horizon_steps,
        "update_period": frozen.update_period,
        "base_policy": base_policy,
        "summary": {key: _summary(values) for key, values in metrics.items()},
        "closed_loop": {
            "success": float(np.mean(successes)),
            "final_reward": float(np.mean(final_rewards)),
            "max_reward": float(np.mean(max_rewards)),
        },
        "replan_count": len(raw_rows),
        "rows": raw_rows,
    }
    output = Path(args.output)
    ensure_dir(output.parent)
    write_json(output, payload)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/pusht_incremental.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--n-demo", type=int, choices=[500, 1000], default=500)
    parser.add_argument("--seed", type=int, choices=[0, 1, 2], default=0)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--eval-seed-start", type=int, default=50_000)
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--output", required=True)
    path = run_goal_mismatch_audit(parser.parse_args())
    print(path)


if __name__ == "__main__":
    main()
