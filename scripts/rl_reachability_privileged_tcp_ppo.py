#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
from torch import nn
from tqdm import trange

from hcl_poc.config import load_config
from hcl_poc.incremental import PRE_RL_PHASE_B_GOAL_TYPES, _pre_rl_phase_b_goal
from hcl_poc.low_level_rl import ScratchLowActorCritic
from hcl_poc.models import MLP
from hcl_poc.privileged_z import _clone_mani_state_dict
from hcl_poc.rl import _rl_paths, load_ppo_agent
from hcl_poc.rl_rerun import _make_benchmark_env, _to_numpy
from hcl_poc.utils import Standardizer, default_device, ensure_dir, set_seed, write_json

sys.path.append(str(Path(__file__).resolve().parent))
from rl_reachability_tcp_dpsi_ensemble import TcpDpsi, _features, _target_inverse


TCP_SLICE = slice(14, 17)


def _obs_state_np(obs: Any) -> np.ndarray:
    state = obs["state"] if isinstance(obs, dict) else obs
    if isinstance(state, torch.Tensor):
        return state.detach().cpu().numpy().astype(np.float32)
    return np.asarray(state, dtype=np.float32)


def _dataset_path(config: Any) -> Path:
    return (
        config.path_value("paths.incremental_data_dir").parent
        / "rl_rerun"
        / "pusht_vector_state_demos_n4096_b2.h5"
    )


def _batch_keys(h5: h5py.File) -> list[str]:
    keys = sorted(key for key in h5.keys() if key.startswith("batch_"))
    if not keys:
        raise ValueError("Vector local-reset dataset contains no batch_* groups")
    return keys


def _fit_normalizers(
    h5: h5py.File,
    batch_keys: list[str],
    horizon: int,
    control_freq: int,
    goal_type: str,
) -> tuple[Standardizer, Standardizer, Standardizer, dict[str, Any]]:
    states = []
    actions = []
    goals = []
    for key in batch_keys:
        group = h5[key]
        state = np.asarray(group["observations_state"], dtype=np.float32)
        action = np.asarray(group["executed_actions"], dtype=np.float32)
        max_start = action.shape[0] - horizon + 1
        if max_start <= 0:
            raise ValueError(f"{key} is shorter than horizon={horizon}")
        starts = np.arange(max_start)
        current = state[starts]
        future = state[starts + horizon]
        states.append(state.reshape(-1, state.shape[-1]))
        actions.append(action.reshape(-1, action.shape[-1]))
        goals.append(
            _pre_rl_phase_b_goal(current, future, horizon, control_freq, goal_type).reshape(
                -1,
                _goal_dim(goal_type),
            )
        )
    state_all = np.concatenate(states, axis=0).astype(np.float32)
    action_all = np.concatenate(actions, axis=0).astype(np.float32)
    goal_all = np.concatenate(goals, axis=0).astype(np.float32)
    return (
        Standardizer.fit(state_all),
        Standardizer.fit(action_all),
        Standardizer.fit(goal_all),
        {
            "normalizer_fit_states": int(len(state_all)),
            "normalizer_fit_actions": int(len(action_all)),
            "normalizer_fit_goals": int(len(goal_all)),
            "state_dim": int(state_all.shape[-1]),
            "action_dim": int(action_all.shape[-1]),
            "goal_dim": int(goal_all.shape[-1]),
        },
    )


def _distance(current_state: np.ndarray, goal: np.ndarray) -> np.ndarray:
    delta = current_state[:, TCP_SLICE] - goal[:, :3]
    return np.sum(delta * delta, axis=-1).astype(np.float32)


def _goal_dim(goal_type: str) -> int:
    if goal_type == "object_pose":
        return 4
    if goal_type == "object":
        return 7
    if goal_type == "tcp":
        return 6
    if goal_type == "robot":
        return 20
    if goal_type == "full":
        return 28
    raise ValueError(f"Unknown goal_type: {goal_type}")


def _goal_metric_features(goal: np.ndarray, goal_type: str) -> np.ndarray:
    if goal_type == "tcp":
        return goal[:, :3]
    if goal_type == "object_pose":
        return goal[:, :4]
    if goal_type == "object":
        return goal[:, :4]
    if goal_type == "robot":
        return np.concatenate([goal[:, :3], goal[:, 6:20]], axis=-1).astype(np.float32)
    if goal_type == "full":
        return np.concatenate(
            [goal[:, :4], goal[:, 7:10], goal[:, 13:28]],
            axis=-1,
        ).astype(np.float32)
    raise ValueError(f"Unknown goal_type: {goal_type}")


def _goal_state_features(
    current_state: np.ndarray,
    goal_type: str,
    horizon: int,
    control_freq: int,
) -> np.ndarray:
    goal = _pre_rl_phase_b_goal(
        current_state,
        current_state,
        horizon,
        control_freq,
        goal_type,
    )
    return _goal_metric_features(goal, goal_type)


def _goal_distance(
    current_state: np.ndarray,
    goal: np.ndarray,
    goal_type: str,
    horizon: int,
    control_freq: int,
) -> np.ndarray:
    if goal_type == "tcp":
        return _distance(current_state, goal)
    achieved = _goal_state_features(current_state, goal_type, horizon, control_freq)
    target = _goal_metric_features(goal, goal_type)
    delta = achieved - target
    return np.sum(delta * delta, axis=-1).astype(np.float32)


class LocalTcpPpo:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.config = load_config(args.config)
        self.device = default_device()
        self.dataset = Path(args.dataset) if args.dataset else _dataset_path(self.config)
        if not self.dataset.exists():
            raise FileNotFoundError(self.dataset)
        self.h5 = h5py.File(self.dataset, "r")
        self.batch_keys = _batch_keys(self.h5)
        meta = self.h5["meta"].attrs if "meta" in self.h5 else {}
        self.num_envs = int(args.num_envs or meta.get("num_envs", 0))
        if self.num_envs <= 0:
            first = self.h5[self.batch_keys[0]]["observations_state"]
            self.num_envs = int(first.shape[1])
        self.max_steps = int(meta.get("max_steps", self.h5[self.batch_keys[0]]["executed_actions"].shape[0]))
        self.horizon = int(args.horizon)
        if self.max_steps < self.horizon:
            raise ValueError(f"Dataset max_steps={self.max_steps} is shorter than horizon={self.horizon}")
        self.control_freq = int(self.config.get("control_freq", 20))
        self.goal_type = str(getattr(args, "goal_type", "tcp"))
        if self.goal_type not in PRE_RL_PHASE_B_GOAL_TYPES:
            raise ValueError(f"Unknown goal_type: {self.goal_type}")
        self.state_norm, self.action_norm, self.goal_norm, norm_meta = _fit_normalizers(
            self.h5,
            self.batch_keys,
            self.horizon,
            self.control_freq,
            self.goal_type,
        )
        self.state_dim = int(norm_meta["state_dim"])
        self.action_dim = int(norm_meta["action_dim"])
        self.goal_dim = int(norm_meta["goal_dim"])
        self.condition_dim = self.state_dim + self.goal_dim + self.action_dim + 1
        if self.num_envs != int(self.h5[self.batch_keys[0]]["observations_state"].shape[1]):
            raise ValueError(
                "Run 2 currently expects --num-envs to match the vector dataset num_envs"
            )
        self.env = _make_benchmark_env(self.config, self.num_envs, "state")
        self.action_low = torch.as_tensor(
            self.env.single_action_space.low,
            device=self.device,
            dtype=torch.float32,
        )
        self.action_high = torch.as_tensor(
            self.env.single_action_space.high,
            device=self.device,
            dtype=torch.float32,
        )
        self.agent = ScratchLowActorCritic(
            self.condition_dim,
            action_dim=self.action_dim,
            width=int(args.width),
            depth=int(args.depth),
            initial_logstd=float(args.initial_logstd),
        ).to(self.device)
        self.optimizer = torch.optim.Adam(
            self.agent.parameters(),
            lr=float(args.learning_rate),
            eps=1e-5,
        )
        self.teacher = None
        self.teacher_action_penalty_weight = float(
            getattr(args, "teacher_action_penalty_weight", 0.0)
        )
        if self.teacher_action_penalty_weight:
            self.teacher = load_ppo_agent(_rl_paths(self.config).best, self.device)
            self.teacher.eval()
        self.rng = np.random.default_rng(args.seed + 4_110_000)
        self.dpsi_models: list[TcpDpsi] = []
        self.dpsi_input_norm: Standardizer | None = None
        self.dpsi_target_norm: Standardizer | None = None
        if args.reward_distance_source == "dpsi":
            if self.goal_type != "tcp":
                raise ValueError("D_psi reward currently requires --goal-type tcp")
            if not args.dpsi_checkpoint:
                raise ValueError("--dpsi-checkpoint is required when using D_psi reward")
            dpsi_checkpoint = torch.load(
                args.dpsi_checkpoint,
                map_location=self.device,
                weights_only=False,
            )
            self.dpsi_input_norm = Standardizer.from_state_dict(dpsi_checkpoint["input_norm"])
            self.dpsi_target_norm = Standardizer.from_state_dict(dpsi_checkpoint["target_norm"])
            for state_dict in dpsi_checkpoint["models"]:
                model = TcpDpsi(
                    int(dpsi_checkpoint["input_dim"]),
                    int(dpsi_checkpoint["hidden_dim"]),
                    int(dpsi_checkpoint["depth"]),
                ).to(self.device)
                model.load_state_dict(state_dict)
                model.eval()
                model.requires_grad_(False)
                self.dpsi_models.append(model)
        self.obs: Any = None
        self.current_state = np.zeros((self.num_envs, self.state_dim), dtype=np.float32)
        self.current_state_norm = np.zeros_like(self.current_state)
        self.goal = np.zeros((self.num_envs, self.goal_dim), dtype=np.float32)
        self.goal_normed = np.zeros_like(self.goal)
        self.previous_action_norm = self.action_norm.transform(
            np.zeros((self.num_envs, self.action_dim), dtype=np.float32)
        )
        self.bc_terminal_reward_distance = np.zeros(self.num_envs, dtype=np.float32)
        self.bc_model: nn.Module | None = None
        self.bc_action_norm: Standardizer | None = None
        self.bc_cond_norm: Standardizer | None = None
        self.bc_branch_env: Any | None = None
        if args.reward_mode == "bc_advantage_terminal":
            if self.goal_type != "tcp":
                raise ValueError("bc_advantage_terminal currently requires --goal-type tcp")
            bc_path = Path(args.bc_low_checkpoint)
            if not bc_path.exists():
                raise FileNotFoundError(bc_path)
            bc_payload = torch.load(bc_path, map_location=self.device, weights_only=False)
            bc_entry = bc_payload["tcp"]
            bc_model = MLP(
                int(bc_entry["cond_dim"]),
                int(bc_entry["action_dim"]),
                int(bc_entry["hidden_dim"]),
                depth=4,
            ).to(self.device)
            bc_model.load_state_dict(bc_entry["model"])
            bc_model.eval()
            bc_model.requires_grad_(False)
            self.bc_model = bc_model
            self.bc_action_norm = Standardizer.from_state_dict(bc_payload["action_norm"])
            self.bc_cond_norm = Standardizer.from_state_dict(bc_entry["cond_norm"])
            self.bc_branch_env = _make_benchmark_env(self.config, self.num_envs, "state")
            self.bc_branch_env.reset(seed=args.seed + 4_115_000)
        self.local_step = 0
        self.reset_errors: list[float] = []
        self.normalizer_meta = norm_meta

    def close(self) -> None:
        if self.bc_branch_env is not None:
            self.bc_branch_env.close()
        self.h5.close()
        self.env.close()

    @torch.inference_mode()
    def sample_references(self, count: int, seed: int) -> list[tuple[str, int]]:
        rng = np.random.default_rng(seed)
        refs = []
        for _ in range(count):
            key = str(rng.choice(self.batch_keys))
            group = self.h5[key]
            max_start = int(group["executed_actions"].shape[0]) - self.horizon + 1
            refs.append((key, int(rng.integers(0, max_start))))
        return refs

    @torch.inference_mode()
    def reset_local_episode(self, reference: tuple[str, int] | None = None) -> None:
        if reference is None:
            key = str(self.rng.choice(self.batch_keys))
            group = self.h5[key]
            max_start = int(group["executed_actions"].shape[0]) - self.horizon + 1
            t = int(self.rng.integers(0, max_start))
        else:
            key, t = reference
            group = self.h5[key]
        self.obs, _info = self.env.reset(seed=int(group.attrs["batch_seed"]))
        for replay_step in range(t):
            replay_action = torch.from_numpy(
                np.asarray(group["executed_actions"][replay_step], dtype=np.float32)
            ).to(self.device)
            self.obs, _reward, _terminated, _truncated, _info = self.env.step(replay_action)
        live_state = _obs_state_np(self.obs)
        reference_state = np.asarray(group["observations_state"][t], dtype=np.float32)
        self.reset_errors.append(float(np.linalg.norm(live_state - reference_state, axis=-1).mean()))
        self.current_state = live_state
        self.current_state_norm = self.state_norm.transform(live_state)
        self.previous_action_norm = self.action_norm.transform(
            np.asarray(group["previous_executed_actions"][t], dtype=np.float32)
        )
        future_state = np.asarray(
            group["observations_state"][t + self.horizon],
            dtype=np.float32,
        )
        self.goal = _pre_rl_phase_b_goal(
            live_state,
            future_state,
            self.horizon,
            self.control_freq,
            self.goal_type,
        ).astype(np.float32)
        self.goal_normed = self.goal_norm.transform(self.goal)
        self.local_step = 0
        if self.args.reward_mode == "bc_advantage_terminal":
            self.bc_terminal_reward_distance = self.bc_terminal_distance()

    @torch.inference_mode()
    def bc_terminal_distance(self) -> np.ndarray:
        if (
            self.bc_branch_env is None
            or self.bc_model is None
            or self.bc_action_norm is None
            or self.bc_cond_norm is None
        ):
            raise RuntimeError("BC advantage reward requested before BC baseline was loaded")
        self.bc_branch_env.unwrapped.set_state_dict(
            _clone_mani_state_dict(self.env.unwrapped.get_state_dict())
        )
        obs = self.bc_branch_env.unwrapped.get_obs()
        previous_action_raw = self.action_norm.inverse(self.previous_action_norm)
        action_low = self.action_low.detach().cpu().numpy()
        action_high = self.action_high.detach().cpu().numpy()
        for step in range(self.horizon):
            state = _obs_state_np(obs)
            remaining = np.full(
                self.num_envs,
                max(self.horizon - step, 1),
                dtype=np.float32,
            )
            tcp_velocity = (self.goal[:, :3] - state[:, TCP_SLICE]) / (
                remaining[:, None] / float(self.control_freq)
            )
            goal = np.concatenate([self.goal[:, :3], tcp_velocity], axis=-1).astype(
                np.float32
            )
            previous_norm = self.bc_action_norm.transform(previous_action_raw)
            condition = np.concatenate(
                [state, goal, previous_norm, (remaining / self.horizon)[:, None]],
                axis=-1,
            ).astype(np.float32)
            normalized = self.bc_model(
                torch.from_numpy(self.bc_cond_norm.transform(condition)).to(self.device).float()
            )
            raw_action = self.bc_action_norm.inverse(normalized.detach().cpu().numpy())
            clipped = np.clip(raw_action, action_low, action_high).astype(np.float32)
            obs, _reward, _terminated, _truncated, _info = self.bc_branch_env.step(
                torch.from_numpy(clipped).to(self.device).float()
            )
            previous_action_raw = clipped
        return self.reward_distance(_obs_state_np(obs), self.goal)

    def condition(self, *, shuffled_goal: bool = False) -> tuple[torch.Tensor, np.ndarray]:
        goal = self.goal_normed
        if shuffled_goal:
            goal = goal[np.roll(np.arange(self.num_envs), 1)]
        remaining = np.full(
            (self.num_envs, 1),
            max(self.horizon - self.local_step, 1) / self.horizon,
            dtype=np.float32,
        )
        condition = np.concatenate(
            [self.current_state_norm, goal, self.previous_action_norm, remaining],
            axis=-1,
        ).astype(np.float32)
        return torch.from_numpy(condition).to(self.device), self.reward_distance(
            self.current_state,
            self.goal,
        )

    @torch.inference_mode()
    def reward_distance(self, state: np.ndarray, goal: np.ndarray) -> np.ndarray:
        if self.args.reward_distance_source == "true_tcp":
            return _distance(state, goal)
        if self.args.reward_distance_source == "true_goal":
            return _goal_distance(
                state,
                goal,
                self.goal_type,
                self.horizon,
                self.control_freq,
            )
        if (
            self.dpsi_input_norm is None
            or self.dpsi_target_norm is None
            or not self.dpsi_models
        ):
            raise RuntimeError("D_psi reward requested before the ensemble was loaded")
        features = self.dpsi_input_norm.transform(_features(state, goal, tau=0.0)).astype(
            np.float32
        )
        x = torch.from_numpy(features).to(self.device)
        predictions = []
        for model in self.dpsi_models:
            pred_transformed = self.dpsi_target_norm.inverse(
                model(x).detach().cpu().numpy()[:, None]
            ).reshape(-1)
            predictions.append(_target_inverse(pred_transformed, self.args.dpsi_target_scale))
        return np.maximum(np.mean(np.stack(predictions, axis=0), axis=0), 0.0).astype(
            np.float32
        )

    @torch.inference_mode()
    def step_local(
        self,
        action: torch.Tensor,
        previous_distance: np.ndarray,
        *,
        auto_reset: bool = True,
    ) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        raw_action = action
        clipped = torch.clamp(raw_action, self.action_low, self.action_high)
        next_obs, _env_reward, _terminated, _truncated, info = self.env.step(clipped)
        next_state = _obs_state_np(next_obs)
        next_reward_distance = self.reward_distance(next_state, self.goal)
        next_distance = (
            next_reward_distance
            if self.args.reward_distance_source == "true_goal"
            else _distance(next_state, self.goal)
        )
        next_tcp_distance = (
            _distance(next_state, self.goal)
            if self.goal_type == "tcp"
            else np.full(self.num_envs, np.nan, dtype=np.float32)
        )
        progress = previous_distance - next_reward_distance
        segment_end = self.local_step == self.horizon - 1
        if self.args.reward_mode == "progress":
            reward = float(self.args.distance_progress_weight) * progress
        elif self.args.reward_mode == "terminal":
            reward = np.zeros(self.num_envs, dtype=np.float32)
            if segment_end:
                reward = reward - float(self.args.terminal_weight) * next_reward_distance
        elif self.args.reward_mode == "progress_terminal":
            reward = float(self.args.distance_progress_weight) * progress
            if segment_end:
                reward = reward - float(self.args.terminal_weight) * next_reward_distance
        elif self.args.reward_mode == "bc_advantage_terminal":
            reward = np.zeros(self.num_envs, dtype=np.float32)
            if segment_end:
                reward = float(self.args.terminal_weight) * (
                    self.bc_terminal_reward_distance - next_reward_distance
                )
        else:
            raise ValueError(f"Unknown reward mode: {self.args.reward_mode}")
        teacher_action_mae = np.full(self.num_envs, np.nan, dtype=np.float32)
        if self.teacher is not None and self.teacher_action_penalty_weight:
            teacher_action = torch.clamp(
                self.teacher.actor_mean(
                    torch.from_numpy(self.current_state).to(self.device).float()
                ),
                self.action_low,
                self.action_high,
            )
            teacher_action_mae = (
                torch.mean(torch.abs(clipped - teacher_action), dim=-1)
                .detach()
                .cpu()
                .numpy()
                .astype(np.float32)
            )
            reward = reward - self.teacher_action_penalty_weight * teacher_action_mae
        self.obs = next_obs
        self.current_state = next_state
        self.current_state_norm = self.state_norm.transform(next_state)
        self.previous_action_norm = self.action_norm.transform(
            clipped.detach().cpu().numpy().astype(np.float32)
        )
        self.local_step += 1
        done = np.full(self.num_envs, segment_end, dtype=np.bool_)
        metrics = {
            "next_distance": next_distance,
            "next_tcp_distance": next_tcp_distance,
            "next_reward_distance": next_reward_distance,
            "success": _to_numpy(info.get("success", np.zeros(self.num_envs, dtype=np.bool_)))
            .reshape(-1)
            .astype(np.bool_),
            "saturated": torch.any(raw_action != clipped, dim=-1).detach().cpu().numpy(),
            "action_l2": torch.linalg.vector_norm(clipped, dim=-1).detach().cpu().numpy(),
            "teacher_action_mae": teacher_action_mae,
        }
        if segment_end and auto_reset:
            self.reset_local_episode()
        return reward.astype(np.float32), done, metrics

    @torch.inference_mode()
    def evaluate(
        self,
        episodes: int,
        *,
        references: list[tuple[str, int]] | None = None,
        shuffled_goal: bool = False,
        deterministic: bool = True,
    ) -> dict[str, float | int | list[float]]:
        self.agent.eval()
        if references is None:
            references = self.sample_references(episodes, self.args.seed + 4_120_000)
        initial_distances = []
        terminal_distances = []
        reductions = []
        action_l2 = []
        saturated = 0
        total_actions = 0
        for reference in references:
            self.reset_local_episode(reference)
            start_distance = None
            for _step in range(self.horizon):
                condition, distance = self.condition(shuffled_goal=shuffled_goal)
                if start_distance is None:
                    start_distance = self.reward_distance(self.current_state, self.goal)
                    initial_distances.extend(start_distance.tolist())
                action, _logprob, _entropy, _value = self.agent.get_action_and_value(
                    condition.float(),
                    deterministic=deterministic,
                )
                _reward, _done, metrics = self.step_local(
                    action,
                    distance,
                    auto_reset=False,
                )
                action_l2.extend(metrics["action_l2"].tolist())
                saturated += int(np.sum(metrics["saturated"]))
                total_actions += self.num_envs
            terminal = metrics["next_distance"]
            terminal_distances.extend(terminal.tolist())
            reductions.extend((start_distance - terminal).tolist())
        count = min(len(terminal_distances), len(references) * self.num_envs)
        terminal_np = np.asarray(terminal_distances[:count], dtype=np.float32)
        initial_np = np.asarray(initial_distances[:count], dtype=np.float32)
        reduction_np = np.asarray(reductions[:count], dtype=np.float32)
        return {
            "local_episodes": int(count),
            "initial_distance_mean": float(np.mean(initial_np)),
            "terminal_distance_mean": float(np.mean(terminal_np)),
            "distance_reduction_mean": float(np.mean(reduction_np)),
            "goal_reach_rate_eps": float(np.mean(terminal_np <= float(self.args.success_epsilon))),
            "p50_terminal_distance": float(np.quantile(terminal_np, 0.50)),
            "p90_terminal_distance": float(np.quantile(terminal_np, 0.90)),
            "p99_terminal_distance": float(np.quantile(terminal_np, 0.99)),
            "fraction_improved_from_start": float(np.mean(reduction_np > 0.0)),
            "action_saturation": float(saturated / max(total_actions, 1)),
            "action_l2_mean": float(np.mean(action_l2)),
            "reset_state_l2_mean": float(np.mean(self.reset_errors[-max(episodes, 1) :])),
        }

    def save_checkpoint(
        self,
        path: Path,
        *,
        history: list[dict[str, Any]],
        global_step: int,
        recipe: dict[str, Any],
    ) -> None:
        ensure_dir(path.parent)
        torch.save(
            {
                "agent": self.agent.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "global_step": int(global_step),
                "history": history,
                "recipe": recipe,
                "state_norm": self.state_norm.state_dict(),
                "action_norm": self.action_norm.state_dict(),
                "goal_norm": self.goal_norm.state_dict(),
                "condition_dim": self.condition_dim,
                "state_dim": self.state_dim,
                "goal_dim": self.goal_dim,
                "goal_type": self.goal_type,
                "action_dim": self.action_dim,
            },
            path,
        )

    def train(self) -> Path:
        args = self.args
        run_dir = ensure_dir(
            Path(args.output_dir)
            / (
                f"privileged_{self.goal_type}_ppo_{args.reward_mode}_"
                f"n{self.num_envs}_seed{args.seed}"
            )
        )
        checkpoint_dir = ensure_dir(run_dir / "checkpoints")
        history_path = run_dir / "history.json"
        metrics_path = run_dir / "metrics.json"
        latest_path = run_dir / "latest.pt"
        if latest_path.exists() and not args.force:
            return latest_path
        init_global_step = 0
        init_history: list[dict[str, Any]] = []
        if args.init_checkpoint:
            payload = torch.load(args.init_checkpoint, map_location=self.device, weights_only=False)
            self.agent.load_state_dict(payload["agent"])
            if "optimizer" in payload:
                self.optimizer.load_state_dict(payload["optimizer"])
            init_global_step = int(payload.get("global_step", 0))
            init_history = list(payload.get("history", []))

        batch_size = self.num_envs * self.horizon
        if batch_size % args.num_minibatches:
            raise ValueError(
                f"batch size {batch_size} must divide num_minibatches={args.num_minibatches}"
            )
        minibatch_size = batch_size // args.num_minibatches
        gamma = float(args.gamma)
        gae_lambda = float(args.gae_lambda)
        clip_coef = float(args.clip_coef)
        ent_coef = float(args.entropy_coef)
        value_coef = float(args.value_coef)
        max_grad_norm = float(args.max_grad_norm)
        recipe = {
            "run": "rl_reachability_debug_run2_privileged_tcp_scratch_ppo",
            "dataset": str(self.dataset.resolve()),
            "num_envs": self.num_envs,
            "horizon": self.horizon,
            "goal_type": self.goal_type,
            "updates": int(args.updates),
            "samples_per_update": int(batch_size),
            "total_env_steps": int(batch_size * args.updates),
            "num_minibatches": int(args.num_minibatches),
            "minibatch_size": int(minibatch_size),
            "update_epochs": int(args.update_epochs),
            "reward_mode": args.reward_mode,
            "reward_distance_source": args.reward_distance_source,
            "dpsi_checkpoint": str(args.dpsi_checkpoint) if args.dpsi_checkpoint else None,
            "init_checkpoint": str(args.init_checkpoint) if args.init_checkpoint else None,
            "init_global_step": int(init_global_step),
            "init_history_updates": int(len(init_history)),
            "bc_low_checkpoint": (
                str(args.bc_low_checkpoint)
                if args.reward_mode == "bc_advantage_terminal"
                else None
            ),
            "dpsi_target_scale": float(args.dpsi_target_scale),
                "reward": args.reward_mode,
                "terminal_weight": float(args.terminal_weight),
                "distance_progress_weight": float(args.distance_progress_weight),
                "teacher_action_penalty_weight": float(args.teacher_action_penalty_weight),
                "success_epsilon": float(args.success_epsilon),
            "learning_rate": float(args.learning_rate),
            "initial_logstd": float(args.initial_logstd),
            "gamma": gamma,
            "gae_lambda": gae_lambda,
            "clip_coef": clip_coef,
            "entropy_coef": ent_coef,
            "value_coef": value_coef,
            "max_grad_norm": max_grad_norm,
            **self.normalizer_meta,
        }

        history: list[dict[str, Any]] = []
        eval_refs = self.sample_references(args.eval_episodes, args.seed + 4_125_000)
        initial_eval = self.evaluate(
            args.eval_episodes,
            references=eval_refs,
            deterministic=True,
        )
        self.reset_local_episode()
        condition_buf = torch.zeros((self.horizon, self.num_envs, self.condition_dim), device=self.device)
        raw_action_buf = torch.zeros((self.horizon, self.num_envs, self.action_dim), device=self.device)
        logprob_buf = torch.zeros((self.horizon, self.num_envs), device=self.device)
        reward_buf = torch.zeros((self.horizon, self.num_envs), device=self.device)
        done_buf = torch.zeros((self.horizon, self.num_envs), device=self.device)
        value_buf = torch.zeros((self.horizon, self.num_envs), device=self.device)
        next_done = torch.zeros(self.num_envs, device=self.device)
        global_step = init_global_step
        start_time = time.perf_counter()

        with trange(args.updates, desc="run2 privileged TCP PPO") as progress:
            for update in progress:
                distances = []
                terminal_distances = []
                rewards = []
                action_l2 = []
                teacher_action_maes = []
                saturated = 0
                nan_count = 0
                self.agent.eval()
                for step in range(self.horizon):
                    condition, distance = self.condition()
                    condition_buf[step] = condition.float()
                    done_buf[step] = next_done
                    with torch.no_grad():
                        raw_action, logprob, _entropy, value = self.agent.get_action_and_value(
                            condition.float()
                        )
                    if torch.isnan(raw_action).any() or torch.isnan(value).any():
                        nan_count += int(torch.isnan(raw_action).sum().item())
                    raw_action_buf[step] = raw_action
                    logprob_buf[step] = logprob
                    value_buf[step] = value
                    reward, done, metrics = self.step_local(raw_action, distance)
                    reward_buf[step] = torch.from_numpy(reward).to(self.device)
                    next_done = torch.from_numpy(done.astype(np.float32)).to(self.device)
                    distances.extend(distance.tolist())
                    rewards.extend(reward.tolist())
                    action_l2.extend(metrics["action_l2"].tolist())
                    teacher_action_maes.extend(
                        np.asarray(metrics["teacher_action_mae"], dtype=np.float32)
                        .reshape(-1)
                        .tolist()
                    )
                    saturated += int(np.sum(metrics["saturated"]))
                    if step == self.horizon - 1:
                        terminal_distances.extend(metrics["next_distance"].tolist())
                    global_step += self.num_envs

                with torch.no_grad():
                    next_value = torch.zeros(self.num_envs, device=self.device)
                    advantages = torch.zeros_like(reward_buf)
                    lastgaelam = torch.zeros(self.num_envs, device=self.device)
                    for t in reversed(range(self.horizon)):
                        if t == self.horizon - 1:
                            nextnonterminal = 1.0 - next_done
                            nextvalues = next_value
                        else:
                            nextnonterminal = 1.0 - done_buf[t + 1]
                            nextvalues = value_buf[t + 1]
                        delta = reward_buf[t] + gamma * nextvalues * nextnonterminal - value_buf[t]
                        lastgaelam = delta + gamma * gae_lambda * nextnonterminal * lastgaelam
                        advantages[t] = lastgaelam
                    returns = advantages + value_buf

                b_conditions = condition_buf.reshape((-1, self.condition_dim))
                b_raw_actions = raw_action_buf.reshape((-1, self.action_dim))
                b_logprobs = logprob_buf.reshape(-1)
                b_advantages = advantages.reshape(-1)
                b_returns = returns.reshape(-1)
                b_values = value_buf.reshape(-1)
                b_advantages = (b_advantages - b_advantages.mean()) / (
                    b_advantages.std() + 1e-8
                )
                batch_indices = np.arange(batch_size)
                policy_losses = []
                value_losses = []
                entropies = []
                clipfracs = []
                approx_kls = []
                self.agent.train()
                for _epoch in range(args.update_epochs):
                    self.rng.shuffle(batch_indices)
                    for start in range(0, batch_size, minibatch_size):
                        mb = batch_indices[start : start + minibatch_size]
                        _action, newlogprob, entropy, newvalue = self.agent.get_action_and_value(
                            b_conditions[mb],
                            raw_action=b_raw_actions[mb],
                        )
                        logratio = newlogprob - b_logprobs[mb]
                        ratio = logratio.exp()
                        with torch.no_grad():
                            approx_kls.append(float(((ratio - 1.0) - logratio).mean().cpu()))
                            clipfracs.append(
                                float(((ratio - 1.0).abs() > clip_coef).float().mean().cpu())
                            )
                        pg_loss1 = -b_advantages[mb] * ratio
                        pg_loss2 = -b_advantages[mb] * torch.clamp(
                            ratio,
                            1.0 - clip_coef,
                            1.0 + clip_coef,
                        )
                        pg_loss = torch.max(pg_loss1, pg_loss2).mean()
                        value_loss = 0.5 * ((newvalue - b_returns[mb]) ** 2).mean()
                        entropy_loss = entropy.mean()
                        loss = pg_loss - ent_coef * entropy_loss + value_coef * value_loss
                        self.optimizer.zero_grad(set_to_none=True)
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(self.agent.parameters(), max_grad_norm)
                        self.optimizer.step()
                        policy_losses.append(float(pg_loss.detach().cpu()))
                        value_losses.append(float(value_loss.detach().cpu()))
                        entropies.append(float(entropy_loss.detach().cpu()))

                y_true = b_returns.detach().cpu().numpy()
                y_pred = b_values.detach().cpu().numpy()
                explained_variance = 1.0 - float(np.var(y_true - y_pred) / max(np.var(y_true), 1e-8))
                row = {
                    "update": int(update + 1),
                    "global_step": int(global_step),
                    "mean_return_per_step": float(np.mean(rewards)),
                    "mean_initial_distance": float(np.mean(distances[: self.num_envs])),
                    "mean_distance": float(np.mean(distances)),
                    "mean_terminal_distance": float(np.mean(terminal_distances)),
                    "goal_reach_rate_eps": float(
                        np.mean(np.asarray(terminal_distances) <= float(args.success_epsilon))
                    ),
                    "action_saturation": float(saturated / float(batch_size)),
                    "action_l2_mean": float(np.mean(action_l2)),
                    "teacher_action_mae": (
                        float(np.nanmean(teacher_action_maes))
                        if np.any(~np.isnan(np.asarray(teacher_action_maes, dtype=np.float32)))
                        else None
                    ),
                    "policy_kl": float(np.mean(approx_kls)),
                    "clip_fraction": float(np.mean(clipfracs)),
                    "entropy": float(np.mean(entropies)),
                    "policy_loss": float(np.mean(policy_losses)),
                    "value_loss": float(np.mean(value_losses)),
                    "explained_variance": explained_variance,
                    "nan_count": int(nan_count),
                    "reset_state_l2_recent_mean": float(np.mean(self.reset_errors[-16:])),
                    "elapsed_s": float(time.perf_counter() - start_time),
                }
                history.append(row)
                if (update + 1) % args.checkpoint_every_updates == 0 or update + 1 == args.updates:
                    self.save_checkpoint(
                        latest_path,
                        history=history,
                        global_step=global_step,
                        recipe=recipe,
                    )
                    self.save_checkpoint(
                        checkpoint_dir / f"update_{update + 1:04d}.pt",
                        history=history,
                        global_step=global_step,
                        recipe=recipe,
                    )
                write_json(history_path, {"recipe": recipe, "initial_eval": initial_eval, "history": history})

        trained_eval = self.evaluate(
            args.eval_episodes,
            references=eval_refs,
            deterministic=True,
        )
        shuffled_eval = self.evaluate(
            args.eval_episodes,
            references=eval_refs,
            shuffled_goal=True,
            deterministic=True,
        )
        metrics = {
            "recipe": recipe,
            "initial_eval": initial_eval,
            "trained_eval": trained_eval,
            "trained_shuffled_goal_eval": shuffled_eval,
            "history_first": history[0] if history else None,
            "history_last": history[-1] if history else None,
            "checkpoint": str(latest_path),
        }
        write_json(metrics_path, metrics)
        self.save_checkpoint(latest_path, history=history, global_step=global_step, recipe=recipe)
        return latest_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/pusht_incremental.yaml")
    parser.add_argument("--dataset")
    parser.add_argument("--output-dir", default="results/incremental/rl_reachability_debug/run2_privileged_tcp")
    parser.add_argument("--num-envs", type=int)
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument("--goal-type", choices=PRE_RL_PHASE_B_GOAL_TYPES, default="tcp")
    parser.add_argument("--updates", type=int, default=250)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--reward-mode",
        choices=["progress_terminal", "terminal", "progress", "bc_advantage_terminal"],
        default="progress_terminal",
    )
    parser.add_argument(
        "--reward-distance-source",
        choices=["true_tcp", "true_goal", "dpsi"],
        default="true_tcp",
    )
    parser.add_argument("--dpsi-checkpoint")
    parser.add_argument("--init-checkpoint")
    parser.add_argument(
        "--bc-low-checkpoint",
        default="artifacts/incremental/pre_rl/phase_c/k10/seed0/time_conditioned_tcp.pt",
    )
    parser.add_argument("--dpsi-target-scale", type=float, default=1000.0)
    parser.add_argument("--terminal-weight", type=float, default=1.0)
    parser.add_argument("--distance-progress-weight", type=float, default=1.0)
    parser.add_argument("--teacher-action-penalty-weight", type=float, default=0.0)
    parser.add_argument("--success-epsilon", type=float, default=0.0025)
    parser.add_argument("--num-minibatches", type=int, default=8)
    parser.add_argument("--update-epochs", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--initial-logstd", type=float, default=-1.0)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-coef", type=float, default=0.2)
    parser.add_argument("--entropy-coef", type=float, default=0.0)
    parser.add_argument("--value-coef", type=float, default=1.0)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--checkpoint-every-updates", type=int, default=25)
    parser.add_argument("--eval-episodes", type=int, default=2)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    set_seed(args.seed + 4_100_000)
    runner = LocalTcpPpo(args)
    try:
        print(runner.train())
    finally:
        runner.close()


if __name__ == "__main__":
    main()
