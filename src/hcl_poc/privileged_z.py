from __future__ import annotations

import copy
import json
import time
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
from rich.console import Console
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import trange

from hcl_poc.config import Config
from hcl_poc.flow import flow_matching_loss, sample_flow
from hcl_poc.models import FlowModel, MLP
from hcl_poc.utils import Standardizer, Timer, default_device, ensure_dir, set_seed, write_json

console = Console()


def _ppo_mlp(
    in_dim: int,
    out_dim: int,
    width: int,
    depth: int,
    output_std: float = 1.0,
) -> nn.Sequential:
    layers: list[nn.Module] = []
    dim = in_dim
    for _ in range(depth):
        linear = nn.Linear(dim, width)
        nn.init.orthogonal_(linear.weight, np.sqrt(2.0))
        nn.init.zeros_(linear.bias)
        layers.extend([linear, nn.Tanh()])
        dim = width
    out = nn.Linear(dim, out_dim)
    nn.init.orthogonal_(out.weight, output_std)
    nn.init.zeros_(out.bias)
    layers.append(out)
    return nn.Sequential(*layers)


class PrivilegedZDirectActorCritic(nn.Module):
    def __init__(
        self,
        low_model: nn.Module,
        low_payload: dict[str, Any],
        action_mean: np.ndarray,
        action_std: np.ndarray,
        condition_dim: int,
        action_dim: int,
        *,
        train_scope: str = "final_layer",
        width: int = 256,
        depth: int = 2,
        initial_logstd: float = -4.0,
    ) -> None:
        super().__init__()
        if low_payload.get("model_type") == "flow":
            raise ValueError("Direct privileged-z PPO currently requires an MLP low policy")
        if train_scope not in {"final_layer", "all"}:
            raise ValueError(f"Unknown privileged-z direct train scope: {train_scope}")
        self.condition_dim = condition_dim
        self.action_dim = action_dim
        self.train_scope = train_scope
        self.width = width
        self.depth = depth
        self.low_model = copy.deepcopy(low_model)
        if train_scope == "final_layer":
            for parameter in self.low_model.parameters():
                parameter.requires_grad_(False)
            last = self.low_model.net[-1]
            if not isinstance(last, nn.Linear):
                raise ValueError("Expected final privileged-z low-policy module to be nn.Linear")
            for parameter in last.parameters():
                parameter.requires_grad_(True)
        else:
            for parameter in self.low_model.parameters():
                parameter.requires_grad_(True)
        self.actor_logstd = nn.Parameter(torch.full((1, action_dim), initial_logstd))
        self.critic = _ppo_mlp(condition_dim, 1, width, depth, output_std=1.0)
        self.register_buffer(
            "action_mean",
            torch.as_tensor(action_mean.reshape(1, -1), dtype=torch.float32),
        )
        self.register_buffer(
            "action_std",
            torch.as_tensor(action_std.reshape(1, -1), dtype=torch.float32),
        )

    def mean_action(self, condition: torch.Tensor) -> torch.Tensor:
        normalized = self.low_model(condition)
        return normalized * self.action_std + self.action_mean

    def get_action_and_value(
        self,
        condition: torch.Tensor,
        raw_action: torch.Tensor | None = None,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mean = self.mean_action(condition)
        std = torch.exp(self.actor_logstd.expand_as(mean))
        distribution = torch.distributions.Normal(mean, std)
        if raw_action is None:
            raw_action = mean if deterministic else distribution.sample()
        return (
            raw_action,
            distribution.log_prob(raw_action).sum(-1),
            distribution.entropy().sum(-1),
            self.critic(condition).flatten(),
        )


def _default_dataset(config: Config) -> Path:
    return (
        config.path_value("paths.incremental_data_dir").parent
        / "rl_rerun"
        / "pusht_vector_state_demos_n4096_b2.h5"
    )


def _artifact_dir(
    config: Config,
    n_trajectories: int,
    seed: int,
    run_tag: str | None = None,
) -> Path:
    root = config.path_value("paths.incremental_artifact_dir") / "privileged_z"
    if run_tag:
        root = root / run_tag
    return ensure_dir(root / f"n{n_trajectories}" / f"seed{seed}")


def _result_dir(
    config: Config,
    n_trajectories: int,
    seed: int,
    run_tag: str | None = None,
) -> Path:
    root = config.path_value("paths.incremental_results_dir") / "privileged_z"
    if run_tag:
        root = root / run_tag
    return ensure_dir(root / f"n{n_trajectories}" / f"seed{seed}")


def _trajectory_group_keys(h5: h5py.File) -> list[str]:
    keys: list[str] = []
    for key in sorted(h5.keys()):
        if key == "meta":
            continue
        group = h5[key]
        if not isinstance(group, h5py.Group):
            continue
        if "success_once" in group and "observations_state" in group:
            keys.append(key)
    return keys


def _select_successful_streams(
    h5: h5py.File,
    n_trajectories: int,
    validation_trajectories: int,
    seed: int,
    selection_mode: str = "any_success",
    train_per_expert: int | None = None,
    validation_per_expert: int | None = None,
    expert_attr: str = "expert_index",
) -> tuple[list[tuple[str, int]], list[tuple[str, int]]]:
    if selection_mode not in {"any_success", "balanced_experts"}:
        raise ValueError(f"Unknown privileged-z selection mode: {selection_mode}")
    streams: list[tuple[str, int]] = []
    by_expert: dict[int, list[tuple[str, int]]] = {}
    for key in _trajectory_group_keys(h5):
        group = h5[key]
        success_once = np.asarray(h5[key]["success_once"], dtype=np.bool_)
        for env_index in np.flatnonzero(success_once):
            stream = (key, int(env_index))
            streams.append(stream)
            if selection_mode == "balanced_experts":
                if expert_attr not in group.attrs:
                    raise ValueError(
                        f"Group {key} is missing required expert attr {expert_attr!r}"
                    )
                expert_index = int(group.attrs[expert_attr])
                by_expert.setdefault(expert_index, []).append(stream)
    required = n_trajectories + validation_trajectories
    rng = np.random.default_rng(seed + 5_000_000)
    if selection_mode == "balanced_experts":
        if not by_expert:
            raise ValueError("No expert-indexed successful streams were found")
        expert_ids = sorted(by_expert)
        if train_per_expert is None:
            if n_trajectories % len(expert_ids) != 0:
                raise ValueError(
                    "n_trajectories is not divisible by the number of experts; "
                    "pass --train-per-expert explicitly"
                )
            train_per_expert = n_trajectories // len(expert_ids)
        if validation_per_expert is None:
            if validation_trajectories % len(expert_ids) != 0:
                raise ValueError(
                    "validation_trajectories is not divisible by the number of experts; "
                    "pass --validation-per-expert explicitly"
                )
            validation_per_expert = validation_trajectories // len(expert_ids)
        train: list[tuple[str, int]] = []
        validation: list[tuple[str, int]] = []
        for expert_id in expert_ids:
            expert_streams = by_expert[expert_id]
            required_expert = train_per_expert + validation_per_expert
            if len(expert_streams) < required_expert:
                raise ValueError(
                    f"Expert {expert_id} needs {required_expert} successful streams, "
                    f"found {len(expert_streams)}"
                )
            chosen = [
                expert_streams[index]
                for index in rng.permutation(len(expert_streams))[:required_expert]
            ]
            train.extend(chosen[:train_per_expert])
            validation.extend(chosen[train_per_expert:])
        rng.shuffle(train)
        rng.shuffle(validation)
        return train, validation
    if len(streams) < required:
        raise ValueError(
            f"Need {required} successful vector streams, found {len(streams)}"
        )
    chosen = [streams[index] for index in rng.permutation(len(streams))[:required]]
    return chosen[:n_trajectories], chosen[n_trajectories:]


def _read_streams(
    h5: h5py.File,
    streams: list[tuple[str, int]],
) -> list[dict[str, np.ndarray]]:
    episodes: list[dict[str, np.ndarray]] = []
    for key, env_index in streams:
        group = h5[key]
        expert_index = group.attrs.get("expert_index")
        episodes.append(
            {
                "states": np.asarray(group["observations_state"][:, env_index], dtype=np.float32),
                "actions": np.asarray(group["executed_actions"][:, env_index], dtype=np.float32),
                "previous_actions": np.asarray(
                    group["previous_executed_actions"][:, env_index],
                    dtype=np.float32,
                ),
                "batch": key,
                "env_index": np.asarray(env_index, dtype=np.int64),
                "expert_index": np.asarray(-1 if expert_index is None else int(expert_index), dtype=np.int64),
            }
        )
    return episodes


def _load_episodes(
    dataset_path: Path,
    n_trajectories: int,
    validation_trajectories: int,
    seed: int,
    selection_mode: str = "any_success",
    train_per_expert: int | None = None,
    validation_per_expert: int | None = None,
) -> tuple[list[dict[str, np.ndarray]], list[dict[str, np.ndarray]], dict[str, Any]]:
    if not dataset_path.exists():
        raise FileNotFoundError(dataset_path)
    with h5py.File(dataset_path, "r") as h5:
        meta = dict(h5["meta"].attrs)
        train_streams, val_streams = _select_successful_streams(
            h5,
            n_trajectories,
            validation_trajectories,
            seed,
            selection_mode,
            train_per_expert,
            validation_per_expert,
        )
        train = _read_streams(h5, train_streams)
        validation = _read_streams(h5, val_streams)
    train_experts = [
        int(episode["expert_index"])
        for episode in train
        if int(episode["expert_index"]) >= 0
    ]
    validation_experts = [
        int(episode["expert_index"])
        for episode in validation
        if int(episode["expert_index"]) >= 0
    ]
    return (
        train,
        validation,
        {
            "dataset_path": str(dataset_path),
            "n_trajectories": n_trajectories,
            "validation_trajectories": validation_trajectories,
            "selection_mode": selection_mode,
            "train_per_expert": train_per_expert,
            "validation_per_expert": validation_per_expert,
            "train_expert_counts": {
                str(expert): train_experts.count(expert)
                for expert in sorted(set(train_experts))
            },
            "validation_expert_counts": {
                str(expert): validation_experts.count(expert)
                for expert in sorted(set(validation_experts))
            },
            "h5_meta": {key: str(value) for key, value in meta.items()},
        },
    )


def _privileged_z_samples(
    episodes: list[dict[str, np.ndarray]],
    state_norm: Standardizer,
    action_norm: Standardizer,
    horizon_steps: int,
    include_goal: bool,
    for_high: bool,
) -> tuple[np.ndarray, np.ndarray]:
    rows: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    for episode in episodes:
        states = episode["states"]
        actions = episode["actions"]
        previous_actions = action_norm.transform(episode["previous_actions"])
        normalized_states = state_norm.transform(states)
        normalized_actions = action_norm.transform(actions)
        for t in range(len(actions) - horizon_steps):
            if for_high:
                previous = previous_actions[t]
                rows.append(np.concatenate([normalized_states[t], previous], axis=-1))
                labels.append(normalized_states[t + horizon_steps])
                continue

            if not include_goal:
                remaining = np.asarray([1.0], dtype=np.float32)
                condition = np.concatenate(
                    [normalized_states[t], previous_actions[t], remaining],
                    axis=-1,
                )
                rows.append(condition)
                labels.append(normalized_actions[t])
                continue

            goal = normalized_states[t + horizon_steps]
            for local_step in range(horizon_steps):
                current_t = t + local_step
                remaining = np.asarray(
                    [(horizon_steps - local_step) / float(horizon_steps)],
                    dtype=np.float32,
                )
                condition = np.concatenate(
                    [
                        normalized_states[current_t],
                        goal,
                        previous_actions[current_t],
                        remaining,
                    ],
                    axis=-1,
                )
                rows.append(condition)
                labels.append(normalized_actions[current_t])
    if not rows:
        raise ValueError("No privileged-z samples were produced")
    return np.stack(rows).astype(np.float32), np.stack(labels).astype(np.float32)


def _train_mlp(
    name: str,
    train_x: np.ndarray,
    train_y: np.ndarray,
    val_x: np.ndarray,
    val_y: np.ndarray,
    *,
    hidden_dim: int,
    depth: int,
    batch_size: int,
    epochs: int,
    lr: float,
    seed: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    set_seed(seed)
    device = default_device()
    model = MLP(train_x.shape[-1], train_y.shape[-1], hidden_dim, depth=depth).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y)),
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    rng = np.random.default_rng(seed + 12345)
    val_indices = np.arange(len(val_x))
    if len(val_indices) > 8192:
        val_indices = rng.choice(val_indices, size=8192, replace=False)
    val_x_t = torch.from_numpy(val_x[val_indices]).to(device).float()
    val_y_t = torch.from_numpy(val_y[val_indices]).to(device).float()
    best_state = None
    best_val = float("inf")
    history: list[dict[str, float | int]] = []
    for epoch in trange(1, epochs + 1, desc=f"train privileged-z {name}"):
        model.train()
        total = 0.0
        count = 0
        for x, y in loader:
            x = x.to(device, non_blocking=True).float()
            y = y.to(device, non_blocking=True).float()
            loss = torch.mean((model(x) - y) ** 2)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += float(loss.detach().cpu()) * len(x)
            count += len(x)
        model.eval()
        with torch.inference_mode():
            val_mse = float(torch.mean((model(val_x_t) - val_y_t) ** 2).cpu())
        history.append(
            {
                "epoch": epoch,
                "train_mse": total / max(count, 1),
                "validation_mse": val_mse,
            }
        )
        if val_mse < best_val:
            best_val = val_mse
            best_state = copy.deepcopy(model.state_dict())
    if best_state is None:
        raise RuntimeError(f"{name} produced no checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    predictions: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(val_x), 8192):
            predictions.append(
                model(torch.from_numpy(val_x[start : start + 8192]).to(device).float())
                .cpu()
                .numpy()
            )
    pred = np.concatenate(predictions, axis=0)
    error = pred - val_y
    metrics = {
        "best_validation_mse": best_val,
        "validation_l2_mean": float(np.mean(np.linalg.norm(error, axis=-1))),
        "validation_mae": float(np.mean(np.abs(error))),
        "train_samples": int(len(train_x)),
        "validation_samples": int(len(val_x)),
    }
    return (
        {
            "model_type": "mlp",
            "model": best_state,
            "input_dim": int(train_x.shape[-1]),
            "output_dim": int(train_y.shape[-1]),
            "hidden_dim": hidden_dim,
            "depth": depth,
            "history": history,
        },
        metrics,
    )


def _train_flow(
    name: str,
    train_x: np.ndarray,
    train_y: np.ndarray,
    val_x: np.ndarray,
    val_y: np.ndarray,
    *,
    hidden_dim: int,
    batch_size: int,
    epochs: int,
    lr: float,
    flow_steps: int,
    seed: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    set_seed(seed)
    device = default_device()
    model = FlowModel(train_y.shape[-1], train_x.shape[-1], hidden_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y)),
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    val_x_t = torch.from_numpy(val_x).to(device).float()
    val_y_t = torch.from_numpy(val_y).to(device).float()
    best_state = None
    best_val = float("inf")
    history: list[dict[str, float | int]] = []
    for epoch in trange(1, epochs + 1, desc=f"train privileged-z flow {name}"):
        model.train()
        total = 0.0
        count = 0
        for x, y in loader:
            x = x.to(device, non_blocking=True).float()
            y = y.to(device, non_blocking=True).float()
            loss = flow_matching_loss(model, y, x)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += float(loss.detach().cpu()) * len(x)
            count += len(x)
        model.eval()
        with torch.inference_mode():
            pred = sample_flow(
                model,
                val_x_t,
                flow_steps,
                train_y.shape[-1],
                initial_noise=torch.zeros_like(val_y_t),
            )
            val_mse = float(torch.mean((pred - val_y_t) ** 2).cpu())
        history.append(
            {
                "epoch": epoch,
                "train_flow_loss": total / max(count, 1),
                "validation_zero_noise_mse": val_mse,
            }
        )
        if val_mse < best_val:
            best_val = val_mse
            best_state = copy.deepcopy(model.state_dict())
    if best_state is None:
        raise RuntimeError(f"{name} flow produced no checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    predictions: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(val_x), 8192):
            condition = torch.from_numpy(val_x[start : start + 8192]).to(device).float()
            predictions.append(
                sample_flow(
                    model,
                    condition,
                    flow_steps,
                    train_y.shape[-1],
                    initial_noise=torch.zeros(
                        len(condition),
                        train_y.shape[-1],
                        device=device,
                        dtype=condition.dtype,
                    ),
                )
                .cpu()
                .numpy()
            )
    pred = np.concatenate(predictions, axis=0)
    error = pred - val_y
    metrics = {
        "best_validation_mse": best_val,
        "validation_l2_mean": float(np.mean(np.linalg.norm(error, axis=-1))),
        "validation_mae": float(np.mean(np.abs(error))),
        "train_samples": int(len(train_x)),
        "validation_samples": int(len(val_x)),
        "flow_steps": int(flow_steps),
    }
    return (
        {
            "model_type": "flow",
            "model": best_state,
            "condition_dim": int(train_x.shape[-1]),
            "sample_dim": int(train_y.shape[-1]),
            "hidden_dim": hidden_dim,
            "flow_steps": int(flow_steps),
            "history": history,
        },
        metrics,
    )


def _predict(
    payload: dict[str, Any],
    x: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    if payload.get("model_type") == "flow":
        model = FlowModel(
            int(payload["sample_dim"]),
            int(payload["condition_dim"]),
            int(payload["hidden_dim"]),
        ).to(device)
        model.load_state_dict(payload["model"])
        model.eval()
        outputs: list[np.ndarray] = []
        with torch.inference_mode():
            for start in range(0, len(x), 8192):
                condition = torch.from_numpy(x[start : start + 8192]).to(device).float()
                outputs.append(
                    sample_flow(
                        model,
                        condition,
                        int(payload["flow_steps"]),
                        int(payload["sample_dim"]),
                        initial_noise=torch.zeros(
                            len(condition),
                            int(payload["sample_dim"]),
                            device=device,
                            dtype=condition.dtype,
                        ),
                    )
                    .cpu()
                    .numpy()
                )
        return np.concatenate(outputs, axis=0)
    model = MLP(
        int(payload["input_dim"]),
        int(payload["output_dim"]),
        int(payload["hidden_dim"]),
        depth=int(payload["depth"]),
    ).to(device)
    model.load_state_dict(payload["model"])
    model.eval()
    outputs: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(x), 8192):
            outputs.append(model(torch.from_numpy(x[start : start + 8192]).to(device).float()).cpu().numpy())
    return np.concatenate(outputs, axis=0)


def _model_from_payload(payload: dict[str, Any], device: torch.device) -> MLP | FlowModel:
    if payload.get("model_type") == "flow":
        model = FlowModel(
            int(payload["sample_dim"]),
            int(payload["condition_dim"]),
            int(payload["hidden_dim"]),
        ).to(device)
        model.load_state_dict(payload["model"])
        model.eval()
        return model
    model = MLP(
        int(payload["input_dim"]),
        int(payload["output_dim"]),
        int(payload["hidden_dim"]),
        depth=int(payload["depth"]),
    ).to(device)
    model.load_state_dict(payload["model"])
    model.eval()
    return model


def _predict_loaded_payload(
    model: MLP | FlowModel,
    payload: dict[str, Any],
    x: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    x_t = torch.from_numpy(x).to(device).float()
    if payload.get("model_type") == "flow":
        return (
            sample_flow(
                model,
                x_t,
                int(payload["flow_steps"]),
                int(payload["sample_dim"]),
                initial_noise=torch.zeros(
                    len(x_t),
                    int(payload["sample_dim"]),
                    device=device,
                    dtype=x_t.dtype,
                ),
            )
            .cpu()
            .numpy()
        )
    return model(x_t).cpu().numpy()


def _clone_mani_state_dict(state: dict[str, Any]) -> dict[str, Any]:
    cloned: dict[str, Any] = {}
    for key, value in state.items():
        if isinstance(value, dict):
            cloned[key] = _clone_mani_state_dict(value)
        elif isinstance(value, torch.Tensor):
            cloned[key] = value.detach().clone()
        else:
            raise TypeError(f"Unsupported ManiSkill state value for key {key}: {type(value)}")
    return cloned


def _obs_state_np(obs: Any) -> np.ndarray:
    state = obs["state"] if isinstance(obs, dict) else obs
    if isinstance(state, torch.Tensor):
        return state.detach().cpu().numpy().astype(np.float32)
    return np.asarray(state, dtype=np.float32)


@torch.inference_mode()
def evaluate_privileged_z_hierarchy(
    config: Config,
    checkpoint_path: Path,
    *,
    mode: str = "hierarchy",
    episodes: int = 100,
    seed_start: int = 9_900_000,
    num_envs: int = 64,
    output_path: Path | None = None,
    residual_checkpoint_path: Path | None = None,
    tuned_gate_mode: str = "always",
    tuned_gate_max_degradation_mse: float = 0.0,
    high_goal_delta_scale: float = 1.0,
    high_goal_projection: str = "none",
    high_goal_branch_bank_path: Path | None = None,
    high_goal_branch_selector_path: Path | None = None,
    high_goal_projection_state_weight: float = 0.5,
    high_goal_projection_goal_weight: float = 0.5,
    high_goal_bank_episodes: int = 200,
    high_goal_bank_seed_start: int = 9_800_000,
    high_goal_bank_num_envs: int = 200,
    force: bool = False,
) -> Path:
    from hcl_poc.rl_rerun import _make_benchmark_env, _to_numpy
    from hcl_poc.low_level_rl import ResidualActorCritic
    from hcl_poc.rl_rerun import _residual_action_from_raw

    if mode not in {"flat", "hierarchy", "oracle_hierarchy"}:
        raise ValueError(f"Unknown privileged-z eval mode: {mode}")
    if high_goal_projection not in {
        "none",
        "nearest_oracle_bank",
        "nearest_branch_goal_bank",
        "learned_branch_goal_selector",
    }:
        raise ValueError(f"Unknown high-goal projection: {high_goal_projection}")
    if tuned_gate_mode not in {"always", "local_oracle"}:
        raise ValueError(f"Unknown tuned gate mode: {tuned_gate_mode}")
    if residual_checkpoint_path is not None and mode == "flat":
        raise ValueError("Privileged-z tuned checkpoints require hierarchy or oracle_hierarchy mode")
    if high_goal_delta_scale <= 0.0 or not np.isfinite(high_goal_delta_scale):
        raise ValueError("high_goal_delta_scale must be positive and finite")
    if high_goal_projection_state_weight < 0.0 or high_goal_projection_goal_weight < 0.0:
        raise ValueError("High-goal projection weights must be non-negative")
    if high_goal_projection_state_weight + high_goal_projection_goal_weight <= 0.0:
        raise ValueError("At least one high-goal projection weight must be positive")
    if high_goal_projection != "none":
        if mode != "hierarchy":
            raise ValueError("High-goal projection is only defined for learned hierarchy mode")
        if high_goal_projection == "nearest_branch_goal_bank" and high_goal_branch_bank_path is None:
            raise ValueError("--high-goal-branch-bank is required for nearest_branch_goal_bank")
        if high_goal_projection == "learned_branch_goal_selector" and high_goal_branch_selector_path is None:
            raise ValueError(
                "--high-goal-branch-selector is required for learned_branch_goal_selector"
            )
        if high_goal_bank_episodes <= 0:
            raise ValueError("high_goal_bank_episodes must be positive")
        if high_goal_bank_num_envs <= 0:
            raise ValueError("high_goal_bank_num_envs must be positive")
    if tuned_gate_mode != "always":
        if residual_checkpoint_path is None:
            raise ValueError("A tuned checkpoint is required for tuned gate modes")
        if mode == "flat":
            raise ValueError("Tuned gate modes require hierarchy or oracle_hierarchy mode")
        if tuned_gate_max_degradation_mse < 0.0:
            raise ValueError("tuned_gate_max_degradation_mse must be non-negative")
    out_path = output_path or (
        residual_checkpoint_path.with_name(
            f"{residual_checkpoint_path.stem}_eval_{mode}_n{episodes}.json"
        )
        if residual_checkpoint_path is not None
        else checkpoint_path.with_name(f"{checkpoint_path.stem}_eval_{mode}_n{episodes}.json")
    )
    if out_path.exists() and not force:
        return out_path
    device = default_device()
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_norm = Standardizer.from_state_dict(payload["state_norm"])
    action_norm = Standardizer.from_state_dict(payload["action_norm"])
    flat_model = _model_from_payload(payload["flat"], device)
    high_model = _model_from_payload(payload["high"], device)
    goal_model = _model_from_payload(payload["goal"], device)
    horizon_steps = int(payload["horizon_steps"])
    teacher = None
    if mode == "oracle_hierarchy" or high_goal_projection == "nearest_oracle_bank":
        from hcl_poc.rl import _rl_paths, load_ppo_agent

        teacher = load_ppo_agent(_rl_paths(config).best, device)
    residual_agent = None
    direct_agent: PrivilegedZDirectActorCritic | None = None
    residual_recipe: dict[str, Any] | None = None
    tuned_recipe: dict[str, Any] | None = None
    residual_alpha = 0.0
    residual_action_mode = "additive"
    if residual_checkpoint_path is not None:
        tuned_payload = torch.load(
            residual_checkpoint_path,
            map_location=device,
            weights_only=False,
        )
        tuned_recipe = dict(tuned_payload["recipe"])
        if Path(tuned_recipe["base_checkpoint"]).resolve() != checkpoint_path.resolve():
            raise ValueError("Tuned checkpoint was trained against a different base checkpoint")
        method = str(tuned_recipe.get("method", ""))
        if method == "privileged_z_residual_r1":
            residual_recipe = tuned_recipe
            residual_agent = ResidualActorCritic(
                int(tuned_payload["condition_dim"]),
                action_dim=int(tuned_payload["action_dim"]),
                width=int(tuned_recipe["actor_critic_width"]),
                depth=int(tuned_recipe["actor_critic_depth"]),
                initial_logstd=float(tuned_recipe["initial_logstd"]),
            ).to(device)
            residual_agent.load_state_dict(tuned_payload["agent"])
            residual_agent.eval()
            residual_alpha = float(tuned_recipe["alpha"])
            residual_action_mode = str(tuned_recipe.get("residual_action_mode", "additive"))
        elif method in {"privileged_z_direct_r3", "privileged_z_direct_distill"}:
            direct_agent = PrivilegedZDirectActorCritic(
                goal_model,
                payload["goal"],
                action_norm.mean,
                action_norm.std,
                int(tuned_payload["condition_dim"]),
                action_dim=int(tuned_payload["action_dim"]),
                train_scope=str(tuned_recipe["train_scope"]),
                width=int(tuned_recipe["actor_critic_width"]),
                depth=int(tuned_recipe["actor_critic_depth"]),
                initial_logstd=float(tuned_recipe["initial_logstd"]),
            ).to(device)
            direct_agent.load_state_dict(tuned_payload["agent"])
            direct_agent.eval()
        else:
            raise ValueError(f"Unknown privileged-z tuned checkpoint method: {method}")

    env = _make_benchmark_env(config, num_envs, "rgb+state")
    branch_env = None
    gate_env = None
    if mode == "oracle_hierarchy":
        branch_env = _make_benchmark_env(config, num_envs, "rgb+state")
        branch_env.reset(seed=seed_start)
    if tuned_gate_mode == "local_oracle":
        gate_env = _make_benchmark_env(config, num_envs, "rgb+state")
        gate_env.reset(seed=seed_start)
    action_low = torch.as_tensor(env.single_action_space.low, device=device, dtype=torch.float32)
    action_high = torch.as_tensor(env.single_action_space.high, device=device, dtype=torch.float32)

    high_goal_bank: np.ndarray | None = None
    high_goal_branch_state_bank: np.ndarray | None = None
    high_goal_branch_goal_bank: np.ndarray | None = None
    high_goal_branch_previous_bank: np.ndarray | None = None
    high_goal_branch_outcome_bank: np.ndarray | None = None
    high_goal_selector: nn.Module | None = None
    high_goal_selector_feature_mean: np.ndarray | None = None
    high_goal_selector_feature_std: np.ndarray | None = None
    if high_goal_projection == "nearest_oracle_bank":
        if teacher is None:
            raise RuntimeError("Oracle teacher was not initialized for high-goal projection")
        bank_env = _make_benchmark_env(config, high_goal_bank_num_envs, "rgb+state")
        bank_action_low = torch.as_tensor(
            bank_env.single_action_space.low,
            device=device,
            dtype=torch.float32,
        )
        bank_action_high = torch.as_tensor(
            bank_env.single_action_space.high,
            device=device,
            dtype=torch.float32,
        )
        bank_obs, _bank_info = bank_env.reset(seed=high_goal_bank_seed_start)
        bank_done_episodes = 0
        bank_step = 0
        bank_goals: list[np.ndarray] = []
        try:
            while bank_done_episodes < high_goal_bank_episodes:
                bank_state = torch.as_tensor(
                    _to_numpy(bank_obs["state"]),
                    device=device,
                    dtype=torch.float32,
                )
                bank_action = torch.clamp(
                    teacher.actor_mean(bank_state),
                    bank_action_low,
                    bank_action_high,
                )
                bank_obs, _reward, _terminated, _truncated, bank_info = bank_env.step(
                    bank_action
                )
                bank_step += 1
                if bank_step % horizon_steps == 0:
                    bank_goals.append(state_norm.transform(_obs_state_np(bank_obs)))
                if "final_info" in bank_info:
                    done_mask = bank_info["_final_info"]
                    if bool(done_mask.any()):
                        bank_done_episodes += int(done_mask.detach().sum().cpu())
        finally:
            bank_env.close()
        if not bank_goals:
            raise RuntimeError("High-goal oracle bank collection produced no goals")
        high_goal_bank = np.concatenate(bank_goals, axis=0).astype(np.float32)
    elif high_goal_projection == "nearest_branch_goal_bank":
        if high_goal_branch_bank_path is None:
            raise RuntimeError("Branch goal bank path was not provided")
        branch_bank = _load_branch_goal_projection_bank(
            high_goal_branch_bank_path,
            state_dim=int(payload["state_dim"]),
            action_dim=int(payload["action_dim"]),
        )
        high_goal_branch_state_bank = branch_bank["states"]
        high_goal_branch_goal_bank = branch_bank["goals"]
    elif high_goal_projection == "learned_branch_goal_selector":
        if high_goal_branch_selector_path is None:
            raise RuntimeError("Branch goal selector path was not provided")
        selector_payload = torch.load(
            high_goal_branch_selector_path,
            map_location=device,
            weights_only=False,
        )
        high_goal_selector = _branch_goal_selector_from_payload(selector_payload, device)
        high_goal_branch_state_bank = np.asarray(
            selector_payload["bank_states"],
            dtype=np.float32,
        )
        high_goal_branch_goal_bank = np.asarray(
            selector_payload["bank_goals"],
            dtype=np.float32,
        )
        high_goal_branch_previous_bank = np.asarray(
            selector_payload["bank_previous"],
            dtype=np.float32,
        )
        high_goal_branch_outcome_bank = np.asarray(
            selector_payload["bank_outcome_features"],
            dtype=np.float32,
        )
        high_goal_selector_feature_mean = np.asarray(
            selector_payload["feature_mean"],
            dtype=np.float32,
        )
        high_goal_selector_feature_std = np.asarray(
            selector_payload["feature_std"],
            dtype=np.float32,
        )

    zero_previous = action_norm.transform(np.zeros((1, int(payload["action_dim"])), dtype=np.float32))[0]
    previous = np.repeat(zero_previous[None], num_envs, axis=0)
    held_goal = np.zeros((num_envs, int(payload["state_dim"])), dtype=np.float32)
    countdown = np.zeros(num_envs, dtype=np.int32)
    segment_use_tuned = np.ones(num_envs, dtype=np.bool_)
    successes: list[float] = []
    returns: list[float] = []
    cumulative_returns = np.zeros(num_envs, dtype=np.float32)
    success_once = np.zeros(num_envs, dtype=np.bool_)
    max_rewards = np.full(num_envs, -np.inf, dtype=np.float32)
    decisions = 0
    residual_norms: list[float] = []
    gate_base_mse: list[np.ndarray] = []
    gate_tuned_mse: list[np.ndarray] = []
    gate_use_tuned: list[np.ndarray] = []
    projected_goal_mse: list[np.ndarray] = []
    projected_goal_l2: list[np.ndarray] = []

    def low_level_action_from_state(
        state_np: np.ndarray,
        previous_norm: np.ndarray,
        goal_norm: np.ndarray,
        remaining_norm: np.ndarray,
        *,
        use_tuned: bool,
    ) -> tuple[torch.Tensor, np.ndarray]:
        normalized_state = state_norm.transform(state_np)
        low_input = np.concatenate(
            [
                normalized_state,
                goal_norm,
                previous_norm,
                remaining_norm,
            ],
            axis=-1,
        )
        normalized_action = _predict_loaded_payload(
            goal_model,
            payload["goal"],
            low_input,
            device,
        )
        base_action = torch.as_tensor(
            action_norm.inverse(normalized_action),
            device=device,
            dtype=torch.float32,
        )
        if use_tuned and residual_agent is not None:
            residual_condition = torch.from_numpy(low_input).to(device).float()
            raw_residual, _logprob, _entropy, _value = residual_agent.get_action_and_value(
                residual_condition,
                deterministic=True,
            )
            residual, _unclipped, action = _residual_action_from_raw(
                base_action,
                raw_residual,
                residual_alpha,
                action_low,
                action_high,
                residual_action_mode,
            )
        elif use_tuned and direct_agent is not None:
            condition = torch.from_numpy(low_input).to(device).float()
            raw_action, _logprob, _entropy, _value = direct_agent.get_action_and_value(
                condition,
                deterministic=True,
            )
            action = torch.clamp(raw_action, action_low, action_high)
            residual = action - torch.clamp(base_action, action_low, action_high)
        else:
            action = torch.clamp(base_action, action_low, action_high)
            residual = torch.zeros_like(action)
        residual_norm = torch.linalg.vector_norm(residual, dim=-1).cpu().numpy()
        return action, residual_norm.astype(np.float32)

    def segment_terminal_mse_from_state(
        start_state: dict[str, Any],
        previous_norm: np.ndarray,
        goal_norm: np.ndarray,
        *,
        use_tuned: bool,
    ) -> np.ndarray:
        if gate_env is None:
            raise RuntimeError("Segment gate env was not initialized")
        gate_env.unwrapped.set_state_dict(_clone_mani_state_dict(start_state))
        gate_obs = gate_env.unwrapped.get_obs()
        rollout_previous = previous_norm.copy()
        for gate_step in range(horizon_steps):
            remaining_norm = np.full(
                (num_envs, 1),
                max(horizon_steps - gate_step, 1) / float(horizon_steps),
                dtype=np.float32,
            )
            action, _residual_norm = low_level_action_from_state(
                _obs_state_np(gate_obs),
                rollout_previous,
                goal_norm,
                remaining_norm,
                use_tuned=use_tuned,
            )
            gate_obs, _reward, _terminated, _truncated, _info = gate_env.step(action)
            rollout_previous = action_norm.transform(
                action.detach().cpu().numpy().astype(np.float32)
            )
        terminal_norm = state_norm.transform(_obs_state_np(gate_obs))
        return np.mean((terminal_norm - goal_norm) ** 2, axis=-1).astype(np.float32)

    obs, _info = env.reset(seed=seed_start)
    try:
        while len(successes) < episodes:
            state_np = _to_numpy(obs["state"]).astype(np.float32)
            normalized_state = state_norm.transform(state_np)
            if mode in {"hierarchy", "oracle_hierarchy"}:
                replan = countdown <= 0
                if np.any(replan):
                    replan_state = _clone_mani_state_dict(env.unwrapped.get_state_dict())
                    if mode == "oracle_hierarchy":
                        if branch_env is None or teacher is None:
                            raise RuntimeError("Oracle privileged-z eval was not initialized")
                        branch_env.unwrapped.set_state_dict(
                            _clone_mani_state_dict(replan_state)
                        )
                        branch_obs = branch_env.unwrapped.get_obs()
                        for _step in range(horizon_steps):
                            branch_state = torch.as_tensor(
                                _to_numpy(branch_obs["state"]),
                                device=device,
                                dtype=torch.float32,
                            )
                            branch_action = torch.clamp(
                                teacher.actor_mean(branch_state),
                                action_low,
                                action_high,
                            )
                            branch_obs, _reward, _terminated, _truncated, _info = (
                                branch_env.step(branch_action)
                            )
                        branch_state_np = _to_numpy(branch_obs["state"]).astype(np.float32)
                        high_goal_np = state_norm.transform(branch_state_np)
                    else:
                        high_input = np.concatenate([normalized_state, previous], axis=-1)
                        high_goal_np = _predict_loaded_payload(
                            high_model,
                            payload["high"],
                            high_input,
                            device,
                        )
                        if high_goal_delta_scale != 1.0:
                            high_goal_np = normalized_state + high_goal_delta_scale * (
                                high_goal_np - normalized_state
                            )
                        if high_goal_projection == "nearest_oracle_bank":
                            if high_goal_bank is None:
                                raise RuntimeError("High-goal projection bank was not initialized")
                            projected_goal, projection_mse, projection_l2 = (
                                _nearest_goal_prototypes(high_goal_np, high_goal_bank)
                            )
                            projected_goal_mse.append(projection_mse[replan].copy())
                            projected_goal_l2.append(projection_l2[replan].copy())
                            high_goal_np = projected_goal
                        elif high_goal_projection == "nearest_branch_goal_bank":
                            if (
                                high_goal_branch_state_bank is None
                                or high_goal_branch_goal_bank is None
                            ):
                                raise RuntimeError("High-goal branch bank was not initialized")
                            projected_goal, projection_mse, projection_l2 = (
                                _nearest_branch_goal_prototypes(
                                    normalized_state,
                                    high_goal_np,
                                    high_goal_branch_state_bank,
                                    high_goal_branch_goal_bank,
                                    state_weight=high_goal_projection_state_weight,
                                    goal_weight=high_goal_projection_goal_weight,
                                )
                            )
                            projected_goal_mse.append(projection_mse[replan].copy())
                            projected_goal_l2.append(projection_l2[replan].copy())
                            high_goal_np = projected_goal
                        elif high_goal_projection == "learned_branch_goal_selector":
                            if (
                                high_goal_selector is None
                                or high_goal_branch_state_bank is None
                                or high_goal_branch_goal_bank is None
                                or high_goal_branch_previous_bank is None
                                or high_goal_branch_outcome_bank is None
                                or high_goal_selector_feature_mean is None
                                or high_goal_selector_feature_std is None
                            ):
                                raise RuntimeError("High-goal learned selector was not initialized")
                            projected_goal, projection_mse, projection_l2 = (
                                _select_branch_goals_with_learned_selector(
                                    high_goal_selector,
                                    normalized_state,
                                    high_goal_np,
                                    previous,
                                    high_goal_branch_state_bank,
                                    high_goal_branch_goal_bank,
                                    high_goal_branch_previous_bank,
                                    high_goal_branch_outcome_bank,
                                    high_goal_selector_feature_mean,
                                    high_goal_selector_feature_std,
                                    device,
                                )
                            )
                            projected_goal_mse.append(projection_mse[replan].copy())
                            projected_goal_l2.append(projection_l2[replan].copy())
                            high_goal_np = projected_goal
                    held_goal[replan] = high_goal_np[replan]
                    countdown[replan] = horizon_steps
                    if tuned_gate_mode == "local_oracle":
                        base_mse = segment_terminal_mse_from_state(
                            replan_state,
                            previous,
                            held_goal,
                            use_tuned=False,
                        )
                        tuned_mse = segment_terminal_mse_from_state(
                            replan_state,
                            previous,
                            held_goal,
                            use_tuned=True,
                        )
                        use_tuned = tuned_mse <= (
                            base_mse + float(tuned_gate_max_degradation_mse)
                        )
                        segment_use_tuned[replan] = use_tuned[replan]
                        gate_base_mse.append(base_mse[replan].copy())
                        gate_tuned_mse.append(tuned_mse[replan].copy())
                        gate_use_tuned.append(segment_use_tuned[replan].copy())
                    else:
                        segment_use_tuned[replan] = True
                    decisions += int(np.sum(replan))
                remaining = np.maximum(countdown, 1).astype(np.float32)[:, None]
                low_input = np.concatenate(
                    [
                        normalized_state,
                        held_goal,
                        previous,
                        remaining / float(horizon_steps),
                    ],
                    axis=-1,
                )
                normalized_action = _predict_loaded_payload(
                    goal_model,
                    payload["goal"],
                    low_input,
                    device,
                )
                countdown -= 1
            else:
                flat_input = np.concatenate(
                    [
                        normalized_state,
                        previous,
                        np.ones((num_envs, 1), dtype=np.float32),
                    ],
                    axis=-1,
                )
                normalized_action = _predict_loaded_payload(
                    flat_model,
                    payload["flat"],
                    flat_input,
                    device,
                )
            action_np = action_norm.inverse(normalized_action)
            base_action = torch.as_tensor(action_np, device=device, dtype=torch.float32)
            if residual_agent is not None:
                residual_condition = torch.from_numpy(low_input).to(device).float()
                raw_residual, _logprob, _entropy, _value = residual_agent.get_action_and_value(
                    residual_condition,
                    deterministic=True,
                )
                residual, unclipped, action = _residual_action_from_raw(
                    base_action,
                    raw_residual,
                    residual_alpha,
                    action_low,
                    action_high,
                    residual_action_mode,
                )
                if tuned_gate_mode != "always":
                    gate_mask = torch.from_numpy(segment_use_tuned).to(device).bool()
                    base_clipped = torch.clamp(base_action, action_low, action_high)
                    action = torch.where(gate_mask[:, None], action, base_clipped)
                    residual = torch.where(gate_mask[:, None], residual, torch.zeros_like(residual))
                residual_norms.extend(torch.linalg.vector_norm(residual, dim=-1).cpu().tolist())
            elif direct_agent is not None:
                condition = torch.from_numpy(low_input).to(device).float()
                raw_action, _logprob, _entropy, _value = direct_agent.get_action_and_value(
                    condition,
                    deterministic=True,
                )
                tuned_action = torch.clamp(raw_action, action_low, action_high)
                base_clipped = torch.clamp(base_action, action_low, action_high)
                if tuned_gate_mode == "always":
                    action = tuned_action
                else:
                    gate_mask = torch.from_numpy(segment_use_tuned).to(device).bool()
                    action = torch.where(gate_mask[:, None], tuned_action, base_clipped)
                action_delta = action - base_clipped
                residual_norms.extend(
                    torch.linalg.vector_norm(action_delta, dim=-1).cpu().tolist()
                )
            else:
                action = torch.clamp(base_action, action_low, action_high)
            obs, reward, _terminated, _truncated, info = env.step(action)
            previous = action_norm.transform(action.cpu().numpy().astype(np.float32))
            reward_np = _to_numpy(reward).reshape(-1).astype(np.float32)
            cumulative_returns += reward_np
            max_rewards = np.maximum(max_rewards, reward_np)
            if "success" in info:
                success_once |= _to_numpy(info["success"]).reshape(-1).astype(np.bool_)
            if "final_info" in info:
                mask = info["_final_info"]
                if bool(mask.any()):
                    final_info = info["final_info"]
                    mask_np = mask.detach().cpu().numpy().astype(np.bool_)
                    if "episode" in final_info:
                        ep = final_info["episode"]
                        success_values = (
                            ep["success_once"][mask].detach().float().cpu().numpy()
                        )
                        return_values = ep["return"][mask].detach().float().cpu().numpy()
                    else:
                        success_values = success_once[mask_np].astype(np.float32)
                        return_values = cumulative_returns[mask_np].copy()
                    successes.extend(float(x) for x in success_values)
                    returns.extend(float(x) for x in return_values)
                    previous[mask_np] = zero_previous
                    held_goal[mask_np] = 0.0
                    countdown[mask_np] = 0
                    segment_use_tuned[mask_np] = True
                    cumulative_returns[mask_np] = 0.0
                    success_once[mask_np] = False
                    max_rewards[mask_np] = -np.inf
    finally:
        env.close()
        if branch_env is not None:
            branch_env.close()
        if gate_env is not None:
            gate_env.close()
    if gate_use_tuned:
        gate_used = np.concatenate(gate_use_tuned, axis=0).astype(np.bool_)
        gate_base = np.concatenate(gate_base_mse, axis=0).astype(np.float32)
        gate_tuned = np.concatenate(gate_tuned_mse, axis=0).astype(np.float32)
        gate_summary = {
            "num_decisions": int(len(gate_used)),
            "fraction_tuned": float(np.mean(gate_used)),
            "base_terminal_mse": _summarize_array(gate_base),
            "tuned_terminal_mse": _summarize_array(gate_tuned),
            "paired_improvement_mse": _summarize_array(gate_base - gate_tuned),
        }
    else:
        gate_summary = None
    if projected_goal_mse:
        if high_goal_projection == "nearest_branch_goal_bank":
            bank_size = (
                int(len(high_goal_branch_goal_bank))
                if high_goal_branch_goal_bank is not None
                else 0
            )
        elif high_goal_projection == "learned_branch_goal_selector":
            bank_size = (
                int(len(high_goal_branch_goal_bank))
                if high_goal_branch_goal_bank is not None
                else 0
            )
        else:
            bank_size = int(len(high_goal_bank)) if high_goal_bank is not None else 0
        projection_summary = {
            "bank_size": bank_size,
            "branch_bank_path": str(high_goal_branch_bank_path)
            if high_goal_branch_bank_path is not None
            else None,
            "branch_selector_path": str(high_goal_branch_selector_path)
            if high_goal_branch_selector_path is not None
            else None,
            "state_weight": float(high_goal_projection_state_weight),
            "goal_weight": float(high_goal_projection_goal_weight),
            "bank_episodes": int(high_goal_bank_episodes),
            "bank_seed_start": int(high_goal_bank_seed_start),
            "bank_num_envs": int(high_goal_bank_num_envs),
            "predicted_to_projected_goal_mse": _summarize_array(
                np.concatenate(projected_goal_mse, axis=0)
            ),
            "predicted_to_projected_goal_l2": _summarize_array(
                np.concatenate(projected_goal_l2, axis=0)
            ),
        }
    else:
        projection_summary = None
    result = {
        "checkpoint": str(checkpoint_path),
        "mode": mode,
        "episodes": int(episodes),
        "seed_start": int(seed_start),
        "num_envs": int(num_envs),
        "success": float(np.mean(successes[:episodes])),
        "return": float(np.mean(returns[:episodes])),
        "high_level_decisions_per_episode": decisions / max(len(successes), 1),
        "residual_checkpoint": str(residual_checkpoint_path) if residual_checkpoint_path else None,
        "tuned_checkpoint": str(residual_checkpoint_path) if residual_checkpoint_path else None,
        "tuned_gate_mode": tuned_gate_mode,
        "tuned_gate_max_degradation_mse": float(tuned_gate_max_degradation_mse),
        "high_goal_delta_scale": float(high_goal_delta_scale),
        "high_goal_projection": high_goal_projection,
        "high_goal_branch_bank": str(high_goal_branch_bank_path)
        if high_goal_branch_bank_path is not None
        else None,
        "high_goal_branch_selector": str(high_goal_branch_selector_path)
        if high_goal_branch_selector_path is not None
        else None,
        "high_goal_projection_state_weight": float(high_goal_projection_state_weight),
        "high_goal_projection_goal_weight": float(high_goal_projection_goal_weight),
        "high_goal_projection_summary": projection_summary,
        "tuned_gate_summary": gate_summary,
        "mean_residual_norm": float(np.mean(residual_norms)) if residual_norms else 0.0,
        "residual_recipe": residual_recipe,
        "tuned_recipe": tuned_recipe,
    }
    write_json(out_path, result)
    console.print(result)
    return out_path


def _nearest_goal_prototypes(
    queries: np.ndarray,
    bank: np.ndarray,
    *,
    exclude_self: bool = False,
    chunk_size: int = 512,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if queries.ndim != 2 or bank.ndim != 2:
        raise ValueError("queries and bank must be 2D arrays")
    if queries.shape[1] != bank.shape[1]:
        raise ValueError("queries and bank must have matching feature dimensions")
    if len(bank) == 0:
        raise ValueError("bank must not be empty")
    if exclude_self and len(queries) != len(bank):
        raise ValueError("exclude_self requires queries and bank to have the same length")
    prototype_values: list[np.ndarray] = []
    mse_values: list[np.ndarray] = []
    l2_values: list[np.ndarray] = []
    bank_f = np.asarray(bank, dtype=np.float32)
    for start in range(0, len(queries), chunk_size):
        end = min(start + chunk_size, len(queries))
        diff = np.asarray(queries[start:end], dtype=np.float32)[:, None, :] - bank_f[None, :, :]
        sq = np.mean(diff * diff, axis=-1)
        if exclude_self:
            row = np.arange(start, end)
            sq[np.arange(end - start), row] = np.inf
        nearest = np.argmin(sq, axis=1)
        min_mse = sq[np.arange(end - start), nearest].astype(np.float32)
        prototype_values.append(bank_f[nearest].copy())
        mse_values.append(min_mse)
        l2_values.append(np.sqrt(min_mse * queries.shape[1]).astype(np.float32))
    return (
        np.concatenate(prototype_values, axis=0),
        np.concatenate(mse_values, axis=0),
        np.concatenate(l2_values, axis=0),
    )


def _nearest_goal_distances(
    queries: np.ndarray,
    bank: np.ndarray,
    *,
    exclude_self: bool = False,
    chunk_size: int = 512,
) -> tuple[np.ndarray, np.ndarray]:
    _prototypes, mse, l2 = _nearest_goal_prototypes(
        queries,
        bank,
        exclude_self=exclude_self,
        chunk_size=chunk_size,
    )
    return mse, l2


def _load_branch_goal_projection_bank(
    bank_path: Path,
    *,
    state_dim: int,
    action_dim: int,
) -> dict[str, np.ndarray]:
    data = np.load(bank_path, allow_pickle=True)
    if "conditions" not in data:
        raise ValueError(f"Branch goal bank has no conditions array: {bank_path}")
    if "horizon_steps" not in data:
        raise ValueError(f"Branch goal bank has no horizon_steps field: {bank_path}")
    conditions = np.asarray(data["conditions"], dtype=np.float32)
    horizon_steps = int(np.asarray(data["horizon_steps"]).item())
    expected_dim = state_dim * 2 + action_dim + 1
    if conditions.ndim != 2 or conditions.shape[1] != expected_dim:
        raise ValueError(
            f"Expected branch conditions with shape (*, {expected_dim}), "
            f"got {conditions.shape}"
        )
    if horizon_steps <= 0:
        raise ValueError("Branch goal bank horizon_steps must be positive")
    if len(conditions) % horizon_steps != 0:
        raise ValueError(
            "Branch goal bank conditions length must be divisible by horizon_steps"
        )
    branch_conditions = conditions[::horizon_steps]
    return {
        "states": branch_conditions[:, :state_dim].copy(),
        "goals": branch_conditions[:, state_dim : state_dim * 2].copy(),
    }


def _nearest_branch_goal_prototypes(
    current_states: np.ndarray,
    predicted_goals: np.ndarray,
    bank_states: np.ndarray,
    bank_goals: np.ndarray,
    *,
    state_weight: float = 0.5,
    goal_weight: float = 0.5,
    chunk_size: int = 512,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if current_states.ndim != 2 or predicted_goals.ndim != 2:
        raise ValueError("current_states and predicted_goals must be 2D arrays")
    if bank_states.ndim != 2 or bank_goals.ndim != 2:
        raise ValueError("bank_states and bank_goals must be 2D arrays")
    if current_states.shape != predicted_goals.shape:
        raise ValueError("current_states and predicted_goals must have matching shapes")
    if bank_states.shape != bank_goals.shape:
        raise ValueError("bank_states and bank_goals must have matching shapes")
    if current_states.shape[1] != bank_states.shape[1]:
        raise ValueError("query and bank dimensions must match")
    if len(bank_goals) == 0:
        raise ValueError("bank_goals must not be empty")
    if state_weight < 0.0 or goal_weight < 0.0:
        raise ValueError("state_weight and goal_weight must be non-negative")
    weight_sum = state_weight + goal_weight
    if weight_sum <= 0.0:
        raise ValueError("At least one nearest-branch weight must be positive")
    state_coeff = float(state_weight) / float(weight_sum)
    goal_coeff = float(goal_weight) / float(weight_sum)

    state_bank_f = np.asarray(bank_states, dtype=np.float32)
    goal_bank_f = np.asarray(bank_goals, dtype=np.float32)
    prototype_values: list[np.ndarray] = []
    mse_values: list[np.ndarray] = []
    l2_values: list[np.ndarray] = []
    for start in range(0, len(current_states), chunk_size):
        end = min(start + chunk_size, len(current_states))
        state_diff = (
            np.asarray(current_states[start:end], dtype=np.float32)[:, None, :]
            - state_bank_f[None, :, :]
        )
        goal_diff = (
            np.asarray(predicted_goals[start:end], dtype=np.float32)[:, None, :]
            - goal_bank_f[None, :, :]
        )
        sq = state_coeff * np.mean(state_diff * state_diff, axis=-1) + goal_coeff * np.mean(
            goal_diff * goal_diff,
            axis=-1,
        )
        nearest = np.argmin(sq, axis=1)
        min_mse = sq[np.arange(end - start), nearest].astype(np.float32)
        prototype_values.append(goal_bank_f[nearest].copy())
        mse_values.append(min_mse)
        l2_values.append(np.sqrt(min_mse * current_states.shape[1] * 2).astype(np.float32))
    return (
        np.concatenate(prototype_values, axis=0),
        np.concatenate(mse_values, axis=0),
        np.concatenate(l2_values, axis=0),
    )


def _branch_goal_selector_from_payload(
    payload: dict[str, Any],
    device: torch.device,
) -> nn.Sequential:
    input_dim = int(payload["input_dim"])
    hidden_dim = int(payload["hidden_dim"])
    depth = int(payload["depth"])
    layers: list[nn.Module] = []
    dim = input_dim
    for _ in range(depth):
        layers.extend([nn.Linear(dim, hidden_dim), nn.ReLU()])
        dim = hidden_dim
    layers.append(nn.Linear(dim, 1))
    model = nn.Sequential(*layers).to(device)
    model.load_state_dict(payload["model"])
    model.eval()
    return model


def _branch_selector_features(
    query_states: np.ndarray,
    query_goals: np.ndarray,
    query_previous: np.ndarray,
    candidate_states: np.ndarray,
    candidate_goals: np.ndarray,
    candidate_previous: np.ndarray,
    candidate_outcomes: np.ndarray,
) -> np.ndarray:
    query_count = len(query_states)
    candidate_count = len(candidate_states)
    state_dim = query_states.shape[1]
    action_dim = query_previous.shape[1]
    q_state = np.repeat(query_states[:, None, :], candidate_count, axis=1)
    q_goal = np.repeat(query_goals[:, None, :], candidate_count, axis=1)
    q_prev = np.repeat(query_previous[:, None, :], candidate_count, axis=1)
    c_state = np.repeat(candidate_states[None, :, :], query_count, axis=0)
    c_goal = np.repeat(candidate_goals[None, :, :], query_count, axis=0)
    c_prev = np.repeat(candidate_previous[None, :, :], query_count, axis=0)
    c_outcomes = np.repeat(candidate_outcomes[None, :, :], query_count, axis=0)
    state_delta = q_state - c_state
    goal_delta = q_goal - c_goal
    state_mse = np.mean(state_delta * state_delta, axis=-1, keepdims=True)
    goal_mse = np.mean(goal_delta * goal_delta, axis=-1, keepdims=True)
    prev_delta = q_prev - c_prev
    prev_mse = np.mean(prev_delta * prev_delta, axis=-1, keepdims=True)
    features = np.concatenate(
        [
            q_state,
            q_goal,
            q_prev,
            c_state,
            c_goal,
            c_prev,
            state_delta,
            goal_delta,
            prev_delta,
            state_mse,
            goal_mse,
            prev_mse,
            0.5 * state_mse + 0.5 * goal_mse,
            c_outcomes,
        ],
        axis=-1,
    )
    expected_dim = state_dim * 4 + action_dim * 3 + state_dim * 2 + 4 + candidate_outcomes.shape[1]
    if features.shape[-1] != expected_dim:
        raise RuntimeError("Unexpected branch-selector feature dimension")
    return features.reshape(query_count * candidate_count, features.shape[-1]).astype(np.float32)


def _select_branch_goals_with_learned_selector(
    selector: nn.Module,
    query_states: np.ndarray,
    query_goals: np.ndarray,
    query_previous: np.ndarray,
    bank_states: np.ndarray,
    bank_goals: np.ndarray,
    bank_previous: np.ndarray,
    bank_outcomes: np.ndarray,
    feature_mean: np.ndarray,
    feature_std: np.ndarray,
    device: torch.device,
    *,
    query_chunk_size: int = 16,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if len(bank_goals) == 0:
        raise ValueError("bank_goals must not be empty")
    selected_goals: list[np.ndarray] = []
    selected_mse: list[np.ndarray] = []
    selected_l2: list[np.ndarray] = []
    bank_states_f = np.asarray(bank_states, dtype=np.float32)
    bank_goals_f = np.asarray(bank_goals, dtype=np.float32)
    bank_previous_f = np.asarray(bank_previous, dtype=np.float32)
    bank_outcomes_f = np.asarray(bank_outcomes, dtype=np.float32)
    mean_f = np.asarray(feature_mean, dtype=np.float32)
    std_f = np.maximum(np.asarray(feature_std, dtype=np.float32), 1e-6)
    for start in range(0, len(query_states), query_chunk_size):
        end = min(start + query_chunk_size, len(query_states))
        features = _branch_selector_features(
            np.asarray(query_states[start:end], dtype=np.float32),
            np.asarray(query_goals[start:end], dtype=np.float32),
            np.asarray(query_previous[start:end], dtype=np.float32),
            bank_states_f,
            bank_goals_f,
            bank_previous_f,
            bank_outcomes_f,
        )
        features = (features - mean_f) / std_f
        with torch.inference_mode():
            scores = (
                selector(torch.from_numpy(features).to(device).float())
                .reshape(end - start, len(bank_goals_f))
                .detach()
                .cpu()
                .numpy()
            )
        nearest = np.argmax(scores, axis=1)
        goals = bank_goals_f[nearest].copy()
        goal_delta = np.asarray(query_goals[start:end], dtype=np.float32) - goals
        mse = np.mean(goal_delta * goal_delta, axis=-1).astype(np.float32)
        selected_goals.append(goals)
        selected_mse.append(mse)
        selected_l2.append(np.sqrt(mse * query_goals.shape[1]).astype(np.float32))
    return (
        np.concatenate(selected_goals, axis=0),
        np.concatenate(selected_mse, axis=0),
        np.concatenate(selected_l2, axis=0),
    )


@torch.inference_mode()
def evaluate_privileged_z_goal_validity(
    config: Config,
    checkpoint_path: Path,
    *,
    episodes: int = 200,
    seed_start: int = 9_900_000,
    num_envs: int = 200,
    output_path: Path | None = None,
    force: bool = False,
) -> Path:
    from hcl_poc.rl import _rl_paths, load_ppo_agent
    from hcl_poc.rl_rerun import _make_benchmark_env, _to_numpy

    out_path = output_path or checkpoint_path.with_name(
        f"{checkpoint_path.stem}_goal_validity_n{episodes}.json"
    )
    if out_path.exists() and not force:
        return out_path
    if episodes <= 0:
        raise ValueError("episodes must be positive")
    if num_envs <= 0:
        raise ValueError("num_envs must be positive")

    device = default_device()
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_norm = Standardizer.from_state_dict(payload["state_norm"])
    action_norm = Standardizer.from_state_dict(payload["action_norm"])
    high_model = _model_from_payload(payload["high"], device)
    goal_model = _model_from_payload(payload["goal"], device)
    horizon_steps = int(payload["horizon_steps"])

    env = _make_benchmark_env(config, num_envs, "rgb+state")
    branch_env = _make_benchmark_env(config, num_envs, "rgb+state")
    pred_rollout_env = _make_benchmark_env(config, num_envs, "rgb+state")
    oracle_rollout_env = _make_benchmark_env(config, num_envs, "rgb+state")
    teacher = load_ppo_agent(_rl_paths(config).best, device)
    action_low = torch.as_tensor(env.single_action_space.low, device=device, dtype=torch.float32)
    action_high = torch.as_tensor(env.single_action_space.high, device=device, dtype=torch.float32)
    zero_previous = action_norm.transform(
        np.zeros((1, int(payload["action_dim"])), dtype=np.float32)
    )[0]
    previous = np.repeat(zero_previous[None], num_envs, axis=0)
    held_goal = np.zeros((num_envs, int(payload["state_dim"])), dtype=np.float32)
    countdown = np.zeros(num_envs, dtype=np.int32)
    successes: list[float] = []
    returns: list[float] = []
    cumulative_returns = np.zeros(num_envs, dtype=np.float32)
    success_once = np.zeros(num_envs, dtype=np.bool_)
    max_rewards = np.full(num_envs, -np.inf, dtype=np.float32)

    current_goals: list[np.ndarray] = []
    predicted_goals: list[np.ndarray] = []
    oracle_goals: list[np.ndarray] = []
    pred_action_l2: list[np.ndarray] = []
    pred_policy_to_pred_mse: list[np.ndarray] = []
    pred_policy_to_oracle_mse: list[np.ndarray] = []
    oracle_policy_to_oracle_mse: list[np.ndarray] = []
    oracle_policy_to_pred_mse: list[np.ndarray] = []

    def low_action(
        state_np: np.ndarray,
        previous_norm: np.ndarray,
        goal_norm: np.ndarray,
        remaining_norm: np.ndarray,
    ) -> torch.Tensor:
        low_input = np.concatenate(
            [
                state_norm.transform(state_np),
                goal_norm,
                previous_norm,
                remaining_norm,
            ],
            axis=-1,
        ).astype(np.float32)
        normalized_action = _predict_loaded_payload(
            goal_model,
            payload["goal"],
            low_input,
            device,
        )
        action = torch.as_tensor(
            action_norm.inverse(normalized_action),
            device=device,
            dtype=torch.float32,
        )
        return torch.clamp(action, action_low, action_high)

    def oracle_goal_from_state(start_state: dict[str, Any]) -> np.ndarray:
        branch_env.unwrapped.set_state_dict(_clone_mani_state_dict(start_state))
        branch_obs = branch_env.unwrapped.get_obs()
        for _step in range(horizon_steps):
            branch_state = torch.as_tensor(
                _to_numpy(branch_obs["state"]),
                device=device,
                dtype=torch.float32,
            )
            branch_action = torch.clamp(
                teacher.actor_mean(branch_state),
                action_low,
                action_high,
            )
            branch_obs, _reward, _terminated, _truncated, _info = branch_env.step(
                branch_action
            )
        return state_norm.transform(_obs_state_np(branch_obs))

    def rollout_low_to_goal(
        rollout_env: Any,
        start_state: dict[str, Any],
        previous_norm: np.ndarray,
        policy_goal_norm: np.ndarray,
    ) -> np.ndarray:
        rollout_env.unwrapped.set_state_dict(_clone_mani_state_dict(start_state))
        rollout_obs = rollout_env.unwrapped.get_obs()
        rollout_previous = previous_norm.copy()
        for step in range(horizon_steps):
            remaining_norm = np.full(
                (num_envs, 1),
                max(horizon_steps - step, 1) / float(horizon_steps),
                dtype=np.float32,
            )
            action = low_action(
                _obs_state_np(rollout_obs),
                rollout_previous,
                policy_goal_norm,
                remaining_norm,
            )
            rollout_obs, _reward, _terminated, _truncated, _info = rollout_env.step(action)
            rollout_previous = action_norm.transform(
                action.detach().cpu().numpy().astype(np.float32)
            )
        return state_norm.transform(_obs_state_np(rollout_obs))

    obs, _info = env.reset(seed=seed_start)
    branch_env.reset(seed=seed_start)
    pred_rollout_env.reset(seed=seed_start)
    oracle_rollout_env.reset(seed=seed_start)
    try:
        while len(successes) < episodes:
            state_np = _to_numpy(obs["state"]).astype(np.float32)
            normalized_state = state_norm.transform(state_np)
            replan = countdown <= 0
            if np.any(replan):
                replan_state = _clone_mani_state_dict(env.unwrapped.get_state_dict())
                high_input = np.concatenate([normalized_state, previous], axis=-1)
                predicted_goal = _predict_loaded_payload(
                    high_model,
                    payload["high"],
                    high_input,
                    device,
                )
                oracle_goal = oracle_goal_from_state(replan_state)
                remaining_start = np.ones((num_envs, 1), dtype=np.float32)
                pred_action = low_action(
                    state_np,
                    previous,
                    predicted_goal,
                    remaining_start,
                )
                oracle_action = low_action(
                    state_np,
                    previous,
                    oracle_goal,
                    remaining_start,
                )
                pred_terminal = rollout_low_to_goal(
                    pred_rollout_env,
                    replan_state,
                    previous,
                    predicted_goal,
                )
                oracle_terminal = rollout_low_to_goal(
                    oracle_rollout_env,
                    replan_state,
                    previous,
                    oracle_goal,
                )
                current_goals.append(normalized_state[replan].copy())
                predicted_goals.append(predicted_goal[replan].copy())
                oracle_goals.append(oracle_goal[replan].copy())
                pred_action_l2.append(
                    torch.linalg.vector_norm(pred_action - oracle_action, dim=-1)
                    .cpu()
                    .numpy()
                    .astype(np.float32)[replan]
                    .copy()
                )
                pred_policy_to_pred_mse.append(
                    np.mean((pred_terminal - predicted_goal) ** 2, axis=-1).astype(
                        np.float32
                    )[replan]
                    .copy()
                )
                pred_policy_to_oracle_mse.append(
                    np.mean((pred_terminal - oracle_goal) ** 2, axis=-1).astype(
                        np.float32
                    )[replan]
                    .copy()
                )
                oracle_policy_to_oracle_mse.append(
                    np.mean((oracle_terminal - oracle_goal) ** 2, axis=-1).astype(
                        np.float32
                    )[replan]
                    .copy()
                )
                oracle_policy_to_pred_mse.append(
                    np.mean((oracle_terminal - predicted_goal) ** 2, axis=-1).astype(
                        np.float32
                    )[replan]
                    .copy()
                )
                held_goal[replan] = predicted_goal[replan]
                countdown[replan] = horizon_steps
            remaining = np.maximum(countdown, 1).astype(np.float32)[:, None]
            action = low_action(state_np, previous, held_goal, remaining / float(horizon_steps))
            countdown -= 1
            obs, reward, _terminated, _truncated, info = env.step(action)
            previous = action_norm.transform(action.detach().cpu().numpy().astype(np.float32))
            reward_np = _to_numpy(reward).reshape(-1).astype(np.float32)
            cumulative_returns += reward_np
            max_rewards = np.maximum(max_rewards, reward_np)
            if "success" in info:
                success_once |= _to_numpy(info["success"]).reshape(-1).astype(np.bool_)
            if "final_info" in info:
                mask = info["_final_info"]
                if bool(mask.any()):
                    final_info = info["final_info"]
                    mask_np = mask.detach().cpu().numpy().astype(np.bool_)
                    if "episode" in final_info:
                        ep = final_info["episode"]
                        success_values = (
                            ep["success_once"][mask].detach().float().cpu().numpy()
                        )
                        return_values = ep["return"][mask].detach().float().cpu().numpy()
                    else:
                        success_values = success_once[mask_np].astype(np.float32)
                        return_values = cumulative_returns[mask_np].copy()
                    successes.extend(float(x) for x in success_values)
                    returns.extend(float(x) for x in return_values)
                    previous[mask_np] = zero_previous
                    held_goal[mask_np] = 0.0
                    countdown[mask_np] = 0
                    cumulative_returns[mask_np] = 0.0
                    success_once[mask_np] = False
                    max_rewards[mask_np] = -np.inf
    finally:
        env.close()
        branch_env.close()
        pred_rollout_env.close()
        oracle_rollout_env.close()

    current = np.concatenate(current_goals, axis=0).astype(np.float32)
    predicted = np.concatenate(predicted_goals, axis=0).astype(np.float32)
    oracle = np.concatenate(oracle_goals, axis=0).astype(np.float32)
    rng = np.random.default_rng(seed_start)
    random_goals = rng.standard_normal(size=predicted.shape).astype(np.float32)
    predicted_nn_mse, predicted_nn_l2 = _nearest_goal_distances(predicted, oracle)
    oracle_nn_mse, oracle_nn_l2 = _nearest_goal_distances(
        oracle,
        oracle,
        exclude_self=len(oracle) > 1,
    )
    random_nn_mse, random_nn_l2 = _nearest_goal_distances(random_goals, oracle)
    pred_oracle_diff = predicted - oracle
    current_pred_diff = predicted - current
    current_oracle_diff = oracle - current
    result = {
        "method": "privileged_z_goal_validity",
        "checkpoint": str(checkpoint_path),
        "episodes": int(episodes),
        "seed_start": int(seed_start),
        "num_envs": int(num_envs),
        "horizon_steps": horizon_steps,
        "closed_loop_hierarchy_success": float(np.mean(successes[:episodes])),
        "closed_loop_hierarchy_return": float(np.mean(returns[:episodes])),
        "num_high_level_decisions": int(len(predicted)),
        "predicted_to_oracle_goal_mse": _summarize_array(
            np.mean(pred_oracle_diff * pred_oracle_diff, axis=-1)
        ),
        "predicted_to_oracle_goal_l2": _summarize_array(
            np.linalg.norm(pred_oracle_diff, axis=-1)
        ),
        "current_to_predicted_goal_mse": _summarize_array(
            np.mean(current_pred_diff * current_pred_diff, axis=-1)
        ),
        "current_to_oracle_goal_mse": _summarize_array(
            np.mean(current_oracle_diff * current_oracle_diff, axis=-1)
        ),
        "predicted_goal_nearest_oracle_mse": _summarize_array(predicted_nn_mse),
        "predicted_goal_nearest_oracle_l2": _summarize_array(predicted_nn_l2),
        "oracle_goal_leave_one_out_nearest_mse": _summarize_array(oracle_nn_mse),
        "oracle_goal_leave_one_out_nearest_l2": _summarize_array(oracle_nn_l2),
        "random_goal_nearest_oracle_mse": _summarize_array(random_nn_mse),
        "random_goal_nearest_oracle_l2": _summarize_array(random_nn_l2),
        "predicted_vs_oracle_first_action_l2": _summarize_array(
            np.concatenate(pred_action_l2, axis=0)
        ),
        "predicted_goal_policy_terminal_mse_to_predicted": _summarize_array(
            np.concatenate(pred_policy_to_pred_mse, axis=0)
        ),
        "predicted_goal_policy_terminal_mse_to_oracle": _summarize_array(
            np.concatenate(pred_policy_to_oracle_mse, axis=0)
        ),
        "oracle_goal_policy_terminal_mse_to_oracle": _summarize_array(
            np.concatenate(oracle_policy_to_oracle_mse, axis=0)
        ),
        "oracle_goal_policy_terminal_mse_to_predicted": _summarize_array(
            np.concatenate(oracle_policy_to_pred_mse, axis=0)
        ),
    }
    write_json(out_path, result)
    console.print(result)
    return out_path


@torch.inference_mode()
def collect_privileged_z_closed_loop_preserve_bank(
    config: Config,
    checkpoint_path: Path,
    *,
    mode: str = "hierarchy",
    episodes: int = 512,
    seed_start: int = 9_900_000,
    num_envs: int = 64,
    output_path: Path | None = None,
    force: bool = False,
) -> Path:
    from hcl_poc.rl_rerun import _make_benchmark_env, _to_numpy

    if mode not in {"hierarchy", "oracle_hierarchy"}:
        raise ValueError(f"Unknown closed-loop preserve-bank mode: {mode}")
    if episodes <= 0:
        raise ValueError("episodes must be positive")
    if num_envs <= 0:
        raise ValueError("num_envs must be positive")
    out_path = output_path or checkpoint_path.with_name(
        f"{checkpoint_path.stem}_closed_loop_preserve_{mode}_n{episodes}_seed{seed_start}.npz"
    )
    if out_path.exists() and not force:
        return out_path
    ensure_dir(out_path.parent)

    device = default_device()
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_norm = Standardizer.from_state_dict(payload["state_norm"])
    action_norm = Standardizer.from_state_dict(payload["action_norm"])
    high_model = _model_from_payload(payload["high"], device)
    goal_model = _model_from_payload(payload["goal"], device)
    horizon_steps = int(payload["horizon_steps"])
    state_dim = int(payload["state_dim"])
    action_dim = int(payload["action_dim"])

    env = _make_benchmark_env(config, num_envs, "rgb+state")
    branch_env = None
    teacher = None
    if mode == "oracle_hierarchy":
        from hcl_poc.rl import _rl_paths, load_ppo_agent

        branch_env = _make_benchmark_env(config, num_envs, "rgb+state")
        branch_env.reset(seed=seed_start)
        teacher = load_ppo_agent(_rl_paths(config).best, device)
    action_low = torch.as_tensor(env.single_action_space.low, device=device, dtype=torch.float32)
    action_high = torch.as_tensor(env.single_action_space.high, device=device, dtype=torch.float32)
    zero_previous = action_norm.transform(np.zeros((1, action_dim), dtype=np.float32))[0]
    previous = np.repeat(zero_previous[None], num_envs, axis=0)
    held_goal = np.zeros((num_envs, state_dim), dtype=np.float32)
    countdown = np.zeros(num_envs, dtype=np.int32)
    conditions: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    successes: list[float] = []
    success_once = np.zeros(num_envs, dtype=np.bool_)
    decisions = 0
    obs, _info = env.reset(seed=seed_start)
    try:
        while len(successes) < episodes:
            state_np = _to_numpy(obs["state"]).astype(np.float32)
            normalized_state = state_norm.transform(state_np)
            replan = countdown <= 0
            if np.any(replan):
                if mode == "oracle_hierarchy":
                    if branch_env is None or teacher is None:
                        raise RuntimeError("Oracle preserve-bank collection was not initialized")
                    branch_env.unwrapped.set_state_dict(
                        _clone_mani_state_dict(env.unwrapped.get_state_dict())
                    )
                    branch_obs = branch_env.unwrapped.get_obs()
                    for _step in range(horizon_steps):
                        branch_state = torch.as_tensor(
                            _to_numpy(branch_obs["state"]),
                            device=device,
                            dtype=torch.float32,
                        )
                        branch_action = torch.clamp(
                            teacher.actor_mean(branch_state),
                            action_low,
                            action_high,
                        )
                        branch_obs, _reward, _terminated, _truncated, _info = (
                            branch_env.step(branch_action)
                        )
                    branch_state_np = _to_numpy(branch_obs["state"]).astype(np.float32)
                    high_goal_np = state_norm.transform(branch_state_np)
                else:
                    high_input = np.concatenate([normalized_state, previous], axis=-1)
                    high_goal_np = _predict_loaded_payload(
                        high_model,
                        payload["high"],
                        high_input,
                        device,
                    )
                held_goal[replan] = high_goal_np[replan]
                countdown[replan] = horizon_steps
                decisions += int(np.sum(replan))
            remaining = np.maximum(countdown, 1).astype(np.float32)[:, None]
            low_input = np.concatenate(
                [
                    normalized_state,
                    held_goal,
                    previous,
                    remaining / float(horizon_steps),
                ],
                axis=-1,
            ).astype(np.float32)
            normalized_action = _predict_loaded_payload(
                goal_model,
                payload["goal"],
                low_input,
                device,
            )
            base_action = torch.as_tensor(
                action_norm.inverse(normalized_action),
                device=device,
                dtype=torch.float32,
            )
            action = torch.clamp(base_action, action_low, action_high)
            conditions.append(low_input.copy())
            actions.append(action.cpu().numpy().astype(np.float32))
            countdown -= 1
            obs, _reward, _terminated, _truncated, info = env.step(action)
            previous = action_norm.transform(action.cpu().numpy().astype(np.float32))
            if "success" in info:
                success_once |= _to_numpy(info["success"]).reshape(-1).astype(np.bool_)
            if "final_info" in info:
                mask = info["_final_info"]
                if bool(mask.any()):
                    final_info = info["final_info"]
                    mask_np = mask.detach().cpu().numpy().astype(np.bool_)
                    if "episode" in final_info:
                        ep = final_info["episode"]
                        success_values = (
                            ep["success_once"][mask].detach().float().cpu().numpy()
                        )
                    else:
                        success_values = success_once[mask_np].astype(np.float32)
                    successes.extend(float(x) for x in success_values)
                    previous[mask_np] = zero_previous
                    held_goal[mask_np] = 0.0
                    countdown[mask_np] = 0
                    success_once[mask_np] = False
    finally:
        env.close()
        if branch_env is not None:
            branch_env.close()

    condition_array = np.concatenate(conditions, axis=0).astype(np.float32)
    action_array = np.concatenate(actions, axis=0).astype(np.float32)
    np.savez_compressed(
        out_path,
        conditions=condition_array,
        actions=action_array,
        mode=np.asarray(mode),
        episodes=np.asarray(episodes, dtype=np.int64),
        seed_start=np.asarray(seed_start, dtype=np.int64),
        num_envs=np.asarray(num_envs, dtype=np.int64),
        horizon_steps=np.asarray(horizon_steps, dtype=np.int64),
        decisions=np.asarray(decisions, dtype=np.int64),
        collected_episodes=np.asarray(len(successes), dtype=np.int64),
        base_success=np.asarray(float(np.mean(successes[:episodes])), dtype=np.float32),
    )
    console.print(
        {
            "output": str(out_path),
            "mode": mode,
            "samples": int(len(condition_array)),
            "episodes": int(episodes),
            "base_success": float(np.mean(successes[:episodes])),
        }
    )
    return out_path


@torch.inference_mode()
def collect_privileged_z_closed_loop_action_search_bank(
    config: Config,
    checkpoint_path: Path,
    *,
    mode: str = "hierarchy",
    episodes: int = 256,
    seed_start: int = 9_900_000,
    num_envs: int = 64,
    random_candidates: int = 32,
    random_noise_std: float = 0.05,
    min_improvement_mse: float = 0.01,
    max_base_mse: float | None = None,
    max_action_delta_l2: float | None = None,
    oracle_gate_max_degradation_mse: float | None = None,
    success_epsilon: float = 0.05,
    max_search_batches: int | None = None,
    output_path: Path | None = None,
    force: bool = False,
) -> Path:
    from hcl_poc.rl_rerun import _make_benchmark_env, _to_numpy

    if mode not in {"hierarchy", "oracle_hierarchy"}:
        raise ValueError(f"Unknown closed-loop action-search mode: {mode}")
    if episodes <= 0:
        raise ValueError("episodes must be positive")
    if num_envs <= 0:
        raise ValueError("num_envs must be positive")
    if random_candidates <= 0:
        raise ValueError("random_candidates must be positive")
    if random_noise_std <= 0.0:
        raise ValueError("random_noise_std must be positive")
    if min_improvement_mse < 0.0:
        raise ValueError("min_improvement_mse must be non-negative")
    if max_base_mse is not None and max_base_mse <= 0.0:
        raise ValueError("max_base_mse must be positive when provided")
    if max_action_delta_l2 is not None and max_action_delta_l2 <= 0.0:
        raise ValueError("max_action_delta_l2 must be positive when provided")
    if (
        oracle_gate_max_degradation_mse is not None
        and oracle_gate_max_degradation_mse < 0.0
    ):
        raise ValueError("oracle_gate_max_degradation_mse must be non-negative")
    if success_epsilon <= 0.0:
        raise ValueError("success_epsilon must be positive")
    if max_search_batches is not None and max_search_batches <= 0:
        raise ValueError("max_search_batches must be positive when provided")
    out_path = output_path or checkpoint_path.with_name(
        f"{checkpoint_path.stem}_closed_loop_action_search_{mode}_n{episodes}_seed{seed_start}.npz"
    )
    if out_path.exists() and not force:
        return out_path
    ensure_dir(out_path.parent)

    device = default_device()
    rng = np.random.default_rng(seed_start + 8_900_000)
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_norm = Standardizer.from_state_dict(payload["state_norm"])
    action_norm = Standardizer.from_state_dict(payload["action_norm"])
    high_model = _model_from_payload(payload["high"], device)
    goal_model = _model_from_payload(payload["goal"], device)
    horizon_steps = int(payload["horizon_steps"])
    state_dim = int(payload["state_dim"])
    action_dim = int(payload["action_dim"])
    condition_dim = state_dim + state_dim + action_dim + 1

    env = _make_benchmark_env(config, num_envs, "rgb+state")
    search_env = _make_benchmark_env(config, num_envs, "rgb+state")
    goal_env = None
    teacher = None
    if mode == "oracle_hierarchy" or oracle_gate_max_degradation_mse is not None:
        from hcl_poc.rl import _rl_paths, load_ppo_agent

        goal_env = _make_benchmark_env(config, num_envs, "rgb+state")
        goal_env.reset(seed=seed_start)
        teacher = load_ppo_agent(_rl_paths(config).best, device)
    action_low = torch.as_tensor(env.single_action_space.low, device=device, dtype=torch.float32)
    action_high = torch.as_tensor(env.single_action_space.high, device=device, dtype=torch.float32)
    zero_previous = action_norm.transform(np.zeros((1, action_dim), dtype=np.float32))[0]
    previous = np.repeat(zero_previous[None], num_envs, axis=0)
    held_goal = np.zeros((num_envs, state_dim), dtype=np.float32)
    countdown = np.zeros(num_envs, dtype=np.int32)
    success_once = np.zeros(num_envs, dtype=np.bool_)
    successes: list[float] = []
    condition_chunks: list[np.ndarray] = []
    action_chunks: list[np.ndarray] = []
    selected_base_mse: list[np.ndarray] = []
    selected_best_mse: list[np.ndarray] = []
    selected_action_delta_l2: list[np.ndarray] = []
    selected_oracle_base_mse: list[np.ndarray] = []
    selected_oracle_candidate_mse: list[np.ndarray] = []
    selected_branch_indices: list[np.ndarray] = []
    all_base_mse: list[np.ndarray] = []
    all_best_mse: list[np.ndarray] = []
    all_action_delta_l2: list[np.ndarray] = []
    all_oracle_base_mse: list[np.ndarray] = []
    all_oracle_candidate_mse: list[np.ndarray] = []
    search_batches = 0
    decisions = 0

    def predict_base_action(
        current_norm: np.ndarray,
        goal_norm: np.ndarray,
        previous_norm: np.ndarray,
        remaining_value: int | np.ndarray,
    ) -> tuple[np.ndarray, torch.Tensor]:
        if isinstance(remaining_value, np.ndarray):
            remaining = np.maximum(remaining_value, 1).astype(np.float32)[:, None]
            remaining = remaining / float(horizon_steps)
        else:
            remaining = np.full(
                (len(current_norm), 1),
                max(horizon_steps - remaining_value, 1) / horizon_steps,
                dtype=np.float32,
            )
        condition_np = np.concatenate(
            [current_norm, goal_norm, previous_norm, remaining],
            axis=-1,
        ).astype(np.float32)
        normalized_action = _predict_loaded_payload(
            goal_model,
            payload["goal"],
            condition_np,
            device,
        )
        action = torch.from_numpy(action_norm.inverse(normalized_action)).to(device)
        return condition_np, torch.clamp(action, action_low, action_high)

    def oracle_goal_from_state(start_state: dict[str, Any]) -> np.ndarray:
        if goal_env is None or teacher is None:
            raise RuntimeError("Oracle gate requested without an oracle goal environment")
        goal_env.unwrapped.set_state_dict(_clone_mani_state_dict(start_state))
        goal_obs = goal_env.unwrapped.get_obs()
        for _step in range(horizon_steps):
            goal_state = torch.as_tensor(
                _to_numpy(goal_obs["state"]),
                device=device,
                dtype=torch.float32,
            )
            goal_action = torch.clamp(
                teacher.actor_mean(goal_state),
                action_low,
                action_high,
            )
            goal_obs, _reward, _terminated, _truncated, _info = goal_env.step(goal_action)
        return state_norm.transform(_to_numpy(goal_obs["state"]).astype(np.float32))

    def rollout_search_candidate(
        start_state: dict[str, Any],
        goal_start: np.ndarray,
        previous_start: np.ndarray,
        noise: np.ndarray | None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        search_env.unwrapped.set_state_dict(_clone_mani_state_dict(start_state))
        branch_obs = search_env.unwrapped.get_obs()
        branch_previous = previous_start.copy()
        condition_steps: list[np.ndarray] = []
        action_steps: list[np.ndarray] = []
        for step in range(horizon_steps):
            current_norm = state_norm.transform(_obs_state_np(branch_obs))
            condition_np, action = predict_base_action(
                current_norm,
                goal_start,
                branch_previous,
                step,
            )
            if noise is not None:
                action = torch.clamp(
                    action + torch.from_numpy(noise[step]).to(device),
                    action_low,
                    action_high,
                )
            condition_steps.append(condition_np)
            action_steps.append(action.cpu().numpy().astype(np.float32))
            branch_obs, _reward, _terminated, _truncated, _info = search_env.step(action)
            branch_previous = action_norm.transform(action.cpu().numpy().astype(np.float32))
        final_norm = state_norm.transform(_obs_state_np(branch_obs))
        terminal_mse = np.mean((final_norm - goal_start) ** 2, axis=-1).astype(np.float32)
        return terminal_mse, np.stack(condition_steps), np.stack(action_steps)

    def rollout_action_sequence_terminal_mse(
        start_state: dict[str, Any],
        goal_start: np.ndarray,
        action_sequence: np.ndarray,
    ) -> np.ndarray:
        search_env.unwrapped.set_state_dict(_clone_mani_state_dict(start_state))
        branch_obs = search_env.unwrapped.get_obs()
        for step in range(horizon_steps):
            action = torch.as_tensor(
                action_sequence[step],
                device=device,
                dtype=torch.float32,
            )
            branch_obs, _reward, _terminated, _truncated, _info = search_env.step(
                torch.clamp(action, action_low, action_high)
            )
        final_norm = state_norm.transform(_obs_state_np(branch_obs))
        return np.mean((final_norm - goal_start) ** 2, axis=-1).astype(np.float32)

    obs, _info = env.reset(seed=seed_start)
    search_env.reset(seed=seed_start)
    try:
        while len(successes) < episodes:
            state_np = _to_numpy(obs["state"]).astype(np.float32)
            normalized_state = state_norm.transform(state_np)
            replan = countdown <= 0
            if np.any(replan):
                if mode == "oracle_hierarchy":
                    high_goal_np = oracle_goal_from_state(
                        _clone_mani_state_dict(env.unwrapped.get_state_dict())
                    )
                else:
                    high_input = np.concatenate([normalized_state, previous], axis=-1)
                    high_goal_np = _predict_loaded_payload(
                        high_model,
                        payload["high"],
                        high_input,
                        device,
                    )
                held_goal[replan] = high_goal_np[replan]
                countdown[replan] = horizon_steps
                decisions += int(np.sum(replan))

                if max_search_batches is None or search_batches < max_search_batches:
                    start_state = _clone_mani_state_dict(env.unwrapped.get_state_dict())
                    base_mse, _base_conditions, base_actions = rollout_search_candidate(
                        start_state,
                        held_goal.copy(),
                        previous.copy(),
                        None,
                    )
                    best_mse = base_mse.copy()
                    best_conditions = _base_conditions.copy()
                    best_actions = base_actions.copy()
                    for _candidate in range(random_candidates):
                        noise = rng.normal(
                            0.0,
                            random_noise_std,
                            size=(horizon_steps, num_envs, action_dim),
                        ).astype(np.float32)
                        candidate_mse, candidate_conditions, candidate_actions = (
                            rollout_search_candidate(
                                start_state,
                                held_goal.copy(),
                                previous.copy(),
                                noise,
                            )
                        )
                        improved = candidate_mse < best_mse
                        if np.any(improved):
                            best_mse[improved] = candidate_mse[improved]
                            best_conditions[:, improved] = candidate_conditions[:, improved]
                            best_actions[:, improved] = candidate_actions[:, improved]
                    improvement = base_mse - best_mse
                    action_delta_l2 = np.linalg.norm(
                        best_actions - base_actions,
                        axis=-1,
                    ).mean(axis=0)
                    selected = (
                        replan
                        & (improvement >= min_improvement_mse)
                        & (best_mse < base_mse)
                    )
                    if max_action_delta_l2 is not None:
                        selected &= action_delta_l2 <= max_action_delta_l2
                    if max_base_mse is not None:
                        selected &= base_mse <= max_base_mse
                    oracle_base_mse = None
                    oracle_candidate_mse = None
                    if oracle_gate_max_degradation_mse is not None:
                        oracle_goal = oracle_goal_from_state(start_state)
                        oracle_base_mse, _oracle_base_conditions, _oracle_base_actions = (
                            rollout_search_candidate(
                                start_state,
                                oracle_goal,
                                previous.copy(),
                                None,
                            )
                        )
                        oracle_candidate_mse = rollout_action_sequence_terminal_mse(
                            start_state,
                            oracle_goal,
                            best_actions,
                        )
                        selected &= (
                            oracle_candidate_mse
                            <= oracle_base_mse + oracle_gate_max_degradation_mse
                        )
                    all_base_mse.append(base_mse[replan])
                    all_best_mse.append(best_mse[replan])
                    all_action_delta_l2.append(action_delta_l2[replan])
                    if oracle_base_mse is not None and oracle_candidate_mse is not None:
                        all_oracle_base_mse.append(oracle_base_mse[replan])
                        all_oracle_candidate_mse.append(oracle_candidate_mse[replan])
                    if np.any(selected):
                        condition_chunks.append(
                            best_conditions[:, selected].reshape(-1, condition_dim)
                        )
                        action_chunks.append(best_actions[:, selected].reshape(-1, action_dim))
                        selected_base_mse.append(base_mse[selected])
                        selected_best_mse.append(best_mse[selected])
                        selected_action_delta_l2.append(action_delta_l2[selected])
                        if (
                            oracle_base_mse is not None
                            and oracle_candidate_mse is not None
                        ):
                            selected_oracle_base_mse.append(oracle_base_mse[selected])
                            selected_oracle_candidate_mse.append(
                                oracle_candidate_mse[selected]
                            )
                        selected_branch_indices.append(np.flatnonzero(selected).astype(np.int64))
                    search_batches += 1

            current_norm = state_norm.transform(_obs_state_np(obs))
            _condition_np, action = predict_base_action(
                current_norm,
                held_goal,
                previous,
                countdown,
            )
            countdown -= 1
            obs, _reward, _terminated, _truncated, info = env.step(action)
            previous = action_norm.transform(action.cpu().numpy().astype(np.float32))
            if "success" in info:
                success_once |= _to_numpy(info["success"]).reshape(-1).astype(np.bool_)
            if "final_info" in info:
                mask = info["_final_info"]
                if bool(mask.any()):
                    final_info = info["final_info"]
                    mask_np = mask.detach().cpu().numpy().astype(np.bool_)
                    if "episode" in final_info:
                        ep = final_info["episode"]
                        success_values = (
                            ep["success_once"][mask].detach().float().cpu().numpy()
                        )
                    else:
                        success_values = success_once[mask_np].astype(np.float32)
                    successes.extend(float(x) for x in success_values)
                    previous[mask_np] = zero_previous
                    held_goal[mask_np] = 0.0
                    countdown[mask_np] = 0
                    success_once[mask_np] = False
    finally:
        env.close()
        search_env.close()
        if goal_env is not None:
            goal_env.close()

    if not condition_chunks:
        raise ValueError("Closed-loop action search found no improved branches")
    conditions = np.concatenate(condition_chunks, axis=0).astype(np.float32)
    actions = np.concatenate(action_chunks, axis=0).astype(np.float32)
    base_selected = np.concatenate(selected_base_mse, axis=0).astype(np.float32)
    best_selected = np.concatenate(selected_best_mse, axis=0).astype(np.float32)
    action_delta_selected = np.concatenate(selected_action_delta_l2, axis=0).astype(
        np.float32
    )
    base_all = np.concatenate(all_base_mse, axis=0).astype(np.float32)
    best_all = np.concatenate(all_best_mse, axis=0).astype(np.float32)
    action_delta_all = np.concatenate(all_action_delta_l2, axis=0).astype(np.float32)
    output_arrays: dict[str, Any] = {
        "conditions": conditions,
        "actions": actions,
        "selected_base_mse": base_selected,
        "selected_best_mse": best_selected,
        "selected_improvement_mse": (base_selected - best_selected).astype(np.float32),
        "selected_action_delta_l2": action_delta_selected,
        "searched_base_mse": base_all,
        "searched_best_mse": best_all,
        "searched_action_delta_l2": action_delta_all,
        "selected_branch_indices": np.concatenate(selected_branch_indices, axis=0),
        "mode": np.asarray(mode),
        "episodes": np.asarray(episodes, dtype=np.int64),
        "seed_start": np.asarray(seed_start, dtype=np.int64),
        "num_envs": np.asarray(num_envs, dtype=np.int64),
        "horizon_steps": np.asarray(horizon_steps, dtype=np.int64),
        "random_candidates": np.asarray(random_candidates, dtype=np.int64),
        "random_noise_std": np.asarray(random_noise_std, dtype=np.float32),
        "min_improvement_mse": np.asarray(min_improvement_mse, dtype=np.float32),
        "max_base_mse": np.asarray(
            np.nan if max_base_mse is None else max_base_mse,
            dtype=np.float32,
        ),
        "max_action_delta_l2": np.asarray(
            np.nan if max_action_delta_l2 is None else max_action_delta_l2,
            dtype=np.float32,
        ),
        "oracle_gate_max_degradation_mse": np.asarray(
            np.nan
            if oracle_gate_max_degradation_mse is None
            else oracle_gate_max_degradation_mse,
            dtype=np.float32,
        ),
        "success_epsilon": np.asarray(success_epsilon, dtype=np.float32),
        "search_batches": np.asarray(search_batches, dtype=np.int64),
        "decisions": np.asarray(decisions, dtype=np.int64),
        "collected_episodes": np.asarray(len(successes), dtype=np.int64),
        "base_success": np.asarray(float(np.mean(successes[:episodes])), dtype=np.float32),
    }
    oracle_base_selected = None
    oracle_candidate_selected = None
    oracle_base_all = None
    oracle_candidate_all = None
    if selected_oracle_base_mse:
        oracle_base_selected = np.concatenate(selected_oracle_base_mse, axis=0).astype(
            np.float32
        )
        oracle_candidate_selected = np.concatenate(
            selected_oracle_candidate_mse,
            axis=0,
        ).astype(np.float32)
        oracle_base_all = np.concatenate(all_oracle_base_mse, axis=0).astype(np.float32)
        oracle_candidate_all = np.concatenate(all_oracle_candidate_mse, axis=0).astype(
            np.float32
        )
        output_arrays.update(
            {
                "selected_oracle_base_mse": oracle_base_selected,
                "selected_oracle_candidate_mse": oracle_candidate_selected,
                "selected_oracle_delta_mse": (
                    oracle_candidate_selected - oracle_base_selected
                ).astype(np.float32),
                "searched_oracle_base_mse": oracle_base_all,
                "searched_oracle_candidate_mse": oracle_candidate_all,
                "searched_oracle_delta_mse": (
                    oracle_candidate_all - oracle_base_all
                ).astype(np.float32),
            }
        )
    np.savez_compressed(out_path, **output_arrays)
    summary = {
        "output": str(out_path),
        "mode": mode,
        "condition_rows": int(len(conditions)),
        "selected_branches": int(len(base_selected)),
        "searched_branches": int(len(base_all)),
        "selected_fraction": float(len(base_selected) / max(len(base_all), 1)),
        "selected_base_mse": _summarize_array(base_selected),
        "selected_best_mse": _summarize_array(best_selected),
        "selected_improvement_mse": _summarize_array(base_selected - best_selected),
        "selected_action_delta_l2": _summarize_array(action_delta_selected),
        "searched_base_success_within_epsilon": float(np.mean(base_all < success_epsilon)),
        "searched_best_success_within_epsilon": float(np.mean(best_all < success_epsilon)),
        "max_base_mse": max_base_mse,
        "max_action_delta_l2": max_action_delta_l2,
        "oracle_gate_max_degradation_mse": oracle_gate_max_degradation_mse,
        "episodes": int(episodes),
        "base_success": float(np.mean(successes[:episodes])),
    }
    if oracle_base_selected is not None and oracle_candidate_selected is not None:
        summary["selected_oracle_base_mse"] = _summarize_array(oracle_base_selected)
        summary["selected_oracle_candidate_mse"] = _summarize_array(
            oracle_candidate_selected
        )
        summary["selected_oracle_delta_mse"] = _summarize_array(
            oracle_candidate_selected - oracle_base_selected
        )
        if oracle_base_all is not None and oracle_candidate_all is not None:
            summary["searched_oracle_gate_pass_fraction"] = float(
                np.mean(
                    oracle_candidate_all
                    <= oracle_base_all + float(oracle_gate_max_degradation_mse)
                )
            )
    console.print(summary)
    return out_path


def _summarize_array(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p90": float(np.quantile(values, 0.9)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


def filter_privileged_z_action_search_bank(
    input_path: Path,
    *,
    output_path: Path,
    min_base_mse: float | None = None,
    max_base_mse: float | None = None,
    min_best_mse: float | None = None,
    max_best_mse: float | None = None,
    min_improvement_mse: float | None = None,
    max_improvement_mse: float | None = None,
    max_action_delta_l2: float | None = None,
    max_oracle_delta_mse: float | None = None,
    force: bool = False,
) -> Path:
    if output_path.exists() and not force:
        return output_path
    for name, value in {
        "min_base_mse": min_base_mse,
        "max_base_mse": max_base_mse,
        "min_best_mse": min_best_mse,
        "max_best_mse": max_best_mse,
        "min_improvement_mse": min_improvement_mse,
        "max_improvement_mse": max_improvement_mse,
        "max_action_delta_l2": max_action_delta_l2,
    }.items():
        if value is not None and value < 0.0:
            raise ValueError(f"{name} must be non-negative")
    if max_oracle_delta_mse is not None and not np.isfinite(max_oracle_delta_mse):
        raise ValueError("max_oracle_delta_mse must be finite when provided")

    with np.load(input_path) as bank:
        arrays = {key: bank[key] for key in bank.files}

    required = {
        "conditions",
        "actions",
        "selected_base_mse",
        "selected_best_mse",
        "selected_improvement_mse",
        "selected_action_delta_l2",
        "horizon_steps",
    }
    missing = sorted(required - set(arrays))
    if missing:
        raise ValueError(f"Action-search bank is missing required arrays: {missing}")

    conditions = np.asarray(arrays["conditions"], dtype=np.float32)
    actions = np.asarray(arrays["actions"], dtype=np.float32)
    horizon_steps = int(np.asarray(arrays["horizon_steps"]).item())
    branch_count = int(np.asarray(arrays["selected_base_mse"]).shape[0])
    if branch_count <= 0:
        raise ValueError(f"Action-search bank has no selected branches: {input_path}")
    if conditions.ndim != 2 or actions.ndim != 2:
        raise ValueError("Action-search conditions/actions must be 2D arrays")
    if len(conditions) != branch_count * horizon_steps:
        raise ValueError(
            f"Condition row count {len(conditions)} does not match "
            f"{branch_count} branches * {horizon_steps} horizon steps"
        )
    if len(actions) != branch_count * horizon_steps:
        raise ValueError(
            f"Action row count {len(actions)} does not match "
            f"{branch_count} branches * {horizon_steps} horizon steps"
        )

    selected = np.ones(branch_count, dtype=np.bool_)
    base_mse = np.asarray(arrays["selected_base_mse"], dtype=np.float32)
    best_mse = np.asarray(arrays["selected_best_mse"], dtype=np.float32)
    improvement_mse = np.asarray(arrays["selected_improvement_mse"], dtype=np.float32)
    action_delta_l2 = np.asarray(arrays["selected_action_delta_l2"], dtype=np.float32)
    if min_base_mse is not None:
        selected &= base_mse >= min_base_mse
    if max_base_mse is not None:
        selected &= base_mse <= max_base_mse
    if min_best_mse is not None:
        selected &= best_mse >= min_best_mse
    if max_best_mse is not None:
        selected &= best_mse <= max_best_mse
    if min_improvement_mse is not None:
        selected &= improvement_mse >= min_improvement_mse
    if max_improvement_mse is not None:
        selected &= improvement_mse <= max_improvement_mse
    if max_action_delta_l2 is not None:
        selected &= action_delta_l2 <= max_action_delta_l2
    if max_oracle_delta_mse is not None:
        if "selected_oracle_delta_mse" not in arrays:
            raise ValueError(
                "max_oracle_delta_mse requested, but bank has no selected_oracle_delta_mse"
            )
        selected &= (
            np.asarray(arrays["selected_oracle_delta_mse"], dtype=np.float32)
            <= max_oracle_delta_mse
        )
    if not np.any(selected):
        raise ValueError("Action-search bank filter selected no branches")

    output_arrays: dict[str, Any] = {}
    for key, value in arrays.items():
        array = np.asarray(value)
        if key == "conditions":
            output_arrays[key] = (
                array.reshape(horizon_steps, branch_count, array.shape[-1])[:, selected]
                .reshape(-1, array.shape[-1])
                .astype(array.dtype, copy=False)
            )
        elif key == "actions":
            output_arrays[key] = (
                array.reshape(horizon_steps, branch_count, array.shape[-1])[:, selected]
                .reshape(-1, array.shape[-1])
                .astype(array.dtype, copy=False)
            )
        elif array.shape[:1] == (branch_count,) and key.startswith("selected_"):
            output_arrays[key] = array[selected]
        else:
            output_arrays[key] = array

    output_arrays.update(
        {
            "filter_source_path": np.asarray(str(input_path)),
            "filter_min_base_mse": np.asarray(
                np.nan if min_base_mse is None else min_base_mse,
                dtype=np.float32,
            ),
            "filter_max_base_mse": np.asarray(
                np.nan if max_base_mse is None else max_base_mse,
                dtype=np.float32,
            ),
            "filter_min_best_mse": np.asarray(
                np.nan if min_best_mse is None else min_best_mse,
                dtype=np.float32,
            ),
            "filter_max_best_mse": np.asarray(
                np.nan if max_best_mse is None else max_best_mse,
                dtype=np.float32,
            ),
            "filter_min_improvement_mse": np.asarray(
                np.nan if min_improvement_mse is None else min_improvement_mse,
                dtype=np.float32,
            ),
            "filter_max_improvement_mse": np.asarray(
                np.nan if max_improvement_mse is None else max_improvement_mse,
                dtype=np.float32,
            ),
            "filter_max_action_delta_l2": np.asarray(
                np.nan if max_action_delta_l2 is None else max_action_delta_l2,
                dtype=np.float32,
            ),
            "filter_max_oracle_delta_mse": np.asarray(
                np.nan if max_oracle_delta_mse is None else max_oracle_delta_mse,
                dtype=np.float32,
            ),
            "filter_selected_branches": np.asarray(int(np.sum(selected)), dtype=np.int64),
            "filter_input_branches": np.asarray(branch_count, dtype=np.int64),
        }
    )
    ensure_dir(output_path.parent)
    np.savez_compressed(output_path, **output_arrays)
    summary = {
        "input": str(input_path),
        "output": str(output_path),
        "input_branches": branch_count,
        "selected_branches": int(np.sum(selected)),
        "condition_rows": int(len(output_arrays["conditions"])),
        "selected_base_mse": _summarize_array(base_mse[selected]),
        "selected_best_mse": _summarize_array(best_mse[selected]),
        "selected_improvement_mse": _summarize_array(improvement_mse[selected]),
        "selected_action_delta_l2": _summarize_array(action_delta_l2[selected]),
    }
    if "selected_oracle_delta_mse" in output_arrays:
        summary["selected_oracle_delta_mse"] = _summarize_array(
            np.asarray(output_arrays["selected_oracle_delta_mse"], dtype=np.float32)
        )
    console.print(summary)
    return output_path


def reweight_privileged_z_action_search_bank(
    input_path: Path,
    *,
    output_path: Path,
    mode: str = "base_x_improvement",
    success_epsilon: float = 0.05,
    improvement_scale: float = 0.05,
    min_weight: float = 0.25,
    max_weight: float = 4.0,
    normalize_mean: bool = True,
    force: bool = False,
) -> Path:
    if output_path.exists() and not force:
        return output_path
    if mode not in {"base_mse", "improvement_mse", "base_x_improvement"}:
        raise ValueError(f"Unknown action-search weight mode: {mode}")
    if success_epsilon <= 0.0:
        raise ValueError("success_epsilon must be positive")
    if improvement_scale <= 0.0:
        raise ValueError("improvement_scale must be positive")
    if min_weight <= 0.0:
        raise ValueError("min_weight must be positive")
    if max_weight < min_weight:
        raise ValueError("max_weight must be >= min_weight")

    with np.load(input_path) as bank:
        arrays = {key: bank[key] for key in bank.files}

    required = {
        "conditions",
        "actions",
        "selected_base_mse",
        "selected_improvement_mse",
        "horizon_steps",
    }
    missing = sorted(required - set(arrays))
    if missing:
        raise ValueError(f"Action-search bank is missing required arrays: {missing}")
    conditions = np.asarray(arrays["conditions"])
    horizon_steps = int(np.asarray(arrays["horizon_steps"]).item())
    branch_count = int(np.asarray(arrays["selected_base_mse"]).shape[0])
    if branch_count <= 0:
        raise ValueError(f"Action-search bank has no selected branches: {input_path}")
    if len(conditions) != branch_count * horizon_steps:
        raise ValueError(
            f"Condition row count {len(conditions)} does not match "
            f"{branch_count} branches * {horizon_steps} horizon steps"
        )

    base_ratio = np.maximum(
        np.asarray(arrays["selected_base_mse"], dtype=np.float32) / success_epsilon,
        1e-6,
    )
    improvement_ratio = np.maximum(
        np.asarray(arrays["selected_improvement_mse"], dtype=np.float32)
        / improvement_scale,
        1e-6,
    )
    if mode == "base_mse":
        branch_weights = np.sqrt(base_ratio)
    elif mode == "improvement_mse":
        branch_weights = np.sqrt(improvement_ratio)
    else:
        branch_weights = np.sqrt(base_ratio * improvement_ratio)
    branch_weights = np.clip(branch_weights, min_weight, max_weight).astype(np.float32)
    if normalize_mean:
        mean_weight = float(np.mean(branch_weights))
        if mean_weight <= 0.0:
            raise ValueError("Cannot normalize zero-mean branch weights")
        branch_weights = (branch_weights / mean_weight).astype(np.float32)
    sample_weights = (
        np.repeat(branch_weights[None, :], horizon_steps, axis=0)
        .reshape(-1)
        .astype(np.float32)
    )
    output_arrays = {key: np.asarray(value) for key, value in arrays.items()}
    output_arrays.update(
        {
            "sample_weights": sample_weights,
            "branch_sample_weights": branch_weights,
            "weight_source_path": np.asarray(str(input_path)),
            "weight_mode": np.asarray(mode),
            "weight_success_epsilon": np.asarray(success_epsilon, dtype=np.float32),
            "weight_improvement_scale": np.asarray(improvement_scale, dtype=np.float32),
            "weight_min_weight": np.asarray(min_weight, dtype=np.float32),
            "weight_max_weight": np.asarray(max_weight, dtype=np.float32),
            "weight_normalize_mean": np.asarray(bool(normalize_mean)),
        }
    )
    ensure_dir(output_path.parent)
    np.savez_compressed(output_path, **output_arrays)
    summary = {
        "input": str(input_path),
        "output": str(output_path),
        "branches": branch_count,
        "condition_rows": int(len(conditions)),
        "mode": mode,
        "normalize_mean": bool(normalize_mean),
        "branch_weights": _summarize_array(branch_weights),
        "sample_weights": _summarize_array(sample_weights),
    }
    console.print(summary)
    return output_path


@torch.inference_mode()
def evaluate_privileged_z_branch_outcomes(
    config: Config,
    checkpoint_path: Path,
    *,
    episodes: int = 100,
    seed_start: int = 9_900_000,
    num_envs: int = 100,
    random_candidates: int = 16,
    random_noise_std: float = 0.05,
    branch_source: str = "random_search",
    branch_condition_goal_source: str = "learned_high",
    min_improvement_mse: float = 0.01,
    max_action_delta_l2: float | None = 0.25,
    max_branch_batches: int = 4,
    max_rollout_steps: int = 120,
    bank_output_path: Path | None = None,
    bank_min_success_delta: float | None = None,
    bank_min_return_delta: float | None = None,
    output_path: Path | None = None,
    force: bool = False,
) -> Path:
    from hcl_poc.rl_rerun import _make_benchmark_env, _to_numpy

    if episodes <= 0:
        raise ValueError("episodes must be positive")
    if num_envs <= 0:
        raise ValueError("num_envs must be positive")
    if random_candidates <= 0:
        raise ValueError("random_candidates must be positive")
    if random_noise_std <= 0.0:
        raise ValueError("random_noise_std must be positive")
    if branch_source not in {"random_search", "oracle_low_level"}:
        raise ValueError("branch_source must be one of: random_search, oracle_low_level")
    if branch_condition_goal_source not in {"learned_high", "oracle_goal"}:
        raise ValueError(
            "branch_condition_goal_source must be one of: learned_high, oracle_goal"
        )
    if branch_condition_goal_source == "oracle_goal" and branch_source != "oracle_low_level":
        raise ValueError("oracle_goal branch conditions require branch_source=oracle_low_level")
    if min_improvement_mse < 0.0:
        raise ValueError("min_improvement_mse must be non-negative")
    if max_action_delta_l2 is not None and max_action_delta_l2 <= 0.0:
        raise ValueError("max_action_delta_l2 must be positive when provided")
    if max_branch_batches <= 0:
        raise ValueError("max_branch_batches must be positive")
    if max_rollout_steps <= 0:
        raise ValueError("max_rollout_steps must be positive")
    out_path = output_path or checkpoint_path.with_name(
        f"{checkpoint_path.stem}_branch_outcomes_n{episodes}_seed{seed_start}.json"
    )
    if bank_output_path is not None and bank_output_path.exists() and not force:
        return out_path
    if out_path.exists() and not force:
        return out_path

    device = default_device()
    rng = np.random.default_rng(seed_start + 9_700_000)
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_norm = Standardizer.from_state_dict(payload["state_norm"])
    action_norm = Standardizer.from_state_dict(payload["action_norm"])
    high_model = _model_from_payload(payload["high"], device)
    goal_model = _model_from_payload(payload["goal"], device)
    horizon_steps = int(payload["horizon_steps"])
    state_dim = int(payload["state_dim"])
    action_dim = int(payload["action_dim"])

    env = _make_benchmark_env(config, num_envs, "rgb+state")
    search_env = _make_benchmark_env(config, num_envs, "rgb+state")
    base_outcome_env = _make_benchmark_env(config, num_envs, "rgb+state")
    candidate_outcome_env = _make_benchmark_env(config, num_envs, "rgb+state")
    goal_env = None
    teacher = None
    if branch_source == "oracle_low_level":
        from hcl_poc.rl import _rl_paths, load_ppo_agent

        goal_env = _make_benchmark_env(config, num_envs, "rgb+state")
        teacher = load_ppo_agent(_rl_paths(config).best, device)
    action_low = torch.as_tensor(env.single_action_space.low, device=device, dtype=torch.float32)
    action_high = torch.as_tensor(env.single_action_space.high, device=device, dtype=torch.float32)
    zero_previous = action_norm.transform(np.zeros((1, action_dim), dtype=np.float32))[0]
    previous = np.repeat(zero_previous[None], num_envs, axis=0)
    held_goal = np.zeros((num_envs, state_dim), dtype=np.float32)
    countdown = np.zeros(num_envs, dtype=np.int32)
    success_once = np.zeros(num_envs, dtype=np.bool_)
    successes: list[float] = []
    returns: list[float] = []
    cumulative_returns = np.zeros(num_envs, dtype=np.float32)

    local_base_mse_values: list[np.ndarray] = []
    local_best_mse_values: list[np.ndarray] = []
    local_improvement_values: list[np.ndarray] = []
    action_delta_values: list[np.ndarray] = []
    selected_values: list[np.ndarray] = []
    condition_values: list[np.ndarray] = []
    action_values: list[np.ndarray] = []
    base_success_values: list[np.ndarray] = []
    candidate_success_values: list[np.ndarray] = []
    base_return_values: list[np.ndarray] = []
    candidate_return_values: list[np.ndarray] = []
    outcome_completed_values: list[np.ndarray] = []
    branch_batches = 0

    def predict_low_action(
        current_norm: np.ndarray,
        goal_norm: np.ndarray,
        previous_norm: np.ndarray,
        remaining_value: int | np.ndarray,
    ) -> torch.Tensor:
        if isinstance(remaining_value, np.ndarray):
            remaining = np.maximum(remaining_value, 1).astype(np.float32)[:, None]
            remaining = remaining / float(horizon_steps)
        else:
            remaining = np.full(
                (len(current_norm), 1),
                max(horizon_steps - remaining_value, 1) / horizon_steps,
                dtype=np.float32,
            )
        condition_np = np.concatenate(
            [current_norm, goal_norm, previous_norm, remaining],
            axis=-1,
        ).astype(np.float32)
        normalized_action = _predict_loaded_payload(
            goal_model,
            payload["goal"],
            condition_np,
            device,
        )
        action = torch.from_numpy(action_norm.inverse(normalized_action)).to(device)
        return torch.clamp(action, action_low, action_high)

    def predict_high_goal(current_norm: np.ndarray, previous_norm: np.ndarray) -> np.ndarray:
        high_input = np.concatenate([current_norm, previous_norm], axis=-1)
        return _predict_loaded_payload(high_model, payload["high"], high_input, device)

    def oracle_goal_from_state(start_state: dict[str, Any]) -> np.ndarray:
        if goal_env is None or teacher is None:
            raise RuntimeError("Oracle branch source requested without an oracle goal env")
        goal_env.unwrapped.set_state_dict(_clone_mani_state_dict(start_state))
        goal_obs = goal_env.unwrapped.get_obs()
        for _step in range(horizon_steps):
            goal_state = torch.as_tensor(
                _to_numpy(goal_obs["state"]),
                device=device,
                dtype=torch.float32,
            )
            goal_action = torch.clamp(
                teacher.actor_mean(goal_state),
                action_low,
                action_high,
            )
            goal_obs, _reward, _terminated, _truncated, _info = goal_env.step(goal_action)
        return state_norm.transform(_obs_state_np(goal_obs))

    def rollout_search_candidate(
        start_state: dict[str, Any],
        goal_start: np.ndarray,
        previous_start: np.ndarray,
        noise: np.ndarray | None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        search_env.unwrapped.set_state_dict(_clone_mani_state_dict(start_state))
        branch_obs = search_env.unwrapped.get_obs()
        branch_previous = previous_start.copy()
        condition_steps: list[np.ndarray] = []
        action_steps: list[np.ndarray] = []
        for step in range(horizon_steps):
            current_norm = state_norm.transform(_obs_state_np(branch_obs))
            remaining = np.full(
                (len(current_norm), 1),
                max(horizon_steps - step, 1) / horizon_steps,
                dtype=np.float32,
            )
            condition_np = np.concatenate(
                [current_norm, goal_start, branch_previous, remaining],
                axis=-1,
            ).astype(np.float32)
            normalized_action = _predict_loaded_payload(
                goal_model,
                payload["goal"],
                condition_np,
                device,
            )
            action = torch.from_numpy(action_norm.inverse(normalized_action)).to(device)
            action = torch.clamp(action, action_low, action_high)
            if noise is not None:
                action = torch.clamp(
                    action + torch.from_numpy(noise[step]).to(device),
                    action_low,
                    action_high,
                )
            condition_steps.append(condition_np)
            action_steps.append(action.cpu().numpy().astype(np.float32))
            branch_obs, _reward, _terminated, _truncated, _info = search_env.step(action)
            branch_previous = action_norm.transform(action.cpu().numpy().astype(np.float32))
        final_norm = state_norm.transform(_obs_state_np(branch_obs))
        terminal_mse = np.mean((final_norm - goal_start) ** 2, axis=-1).astype(np.float32)
        return terminal_mse, np.stack(condition_steps, axis=0), np.stack(action_steps, axis=0)

    def rollout_oracle_low_level_candidate(
        start_state: dict[str, Any],
        learned_goal_start: np.ndarray,
        oracle_goal_start: np.ndarray,
        condition_goal_start: np.ndarray,
        previous_start: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        search_env.unwrapped.set_state_dict(_clone_mani_state_dict(start_state))
        branch_obs = search_env.unwrapped.get_obs()
        branch_previous = previous_start.copy()
        condition_steps: list[np.ndarray] = []
        action_steps: list[np.ndarray] = []
        for step in range(horizon_steps):
            current_norm = state_norm.transform(_obs_state_np(branch_obs))
            remaining = np.full(
                (len(current_norm), 1),
                max(horizon_steps - step, 1) / horizon_steps,
                dtype=np.float32,
            )
            train_condition_np = np.concatenate(
                [current_norm, condition_goal_start, branch_previous, remaining],
                axis=-1,
            ).astype(np.float32)
            oracle_condition_np = np.concatenate(
                [current_norm, oracle_goal_start, branch_previous, remaining],
                axis=-1,
            ).astype(np.float32)
            normalized_action = _predict_loaded_payload(
                goal_model,
                payload["goal"],
                oracle_condition_np,
                device,
            )
            action = torch.from_numpy(action_norm.inverse(normalized_action)).to(device)
            action = torch.clamp(action, action_low, action_high)
            condition_steps.append(train_condition_np)
            action_steps.append(action.cpu().numpy().astype(np.float32))
            branch_obs, _reward, _terminated, _truncated, _info = search_env.step(action)
            branch_previous = action_norm.transform(action.cpu().numpy().astype(np.float32))
        final_norm = state_norm.transform(_obs_state_np(branch_obs))
        terminal_mse = np.mean(
            (final_norm - learned_goal_start) ** 2,
            axis=-1,
        ).astype(np.float32)
        return terminal_mse, np.stack(condition_steps, axis=0), np.stack(action_steps, axis=0)

    def run_outcome_rollout(
        outcome_env: Any,
        start_state: dict[str, Any],
        previous_start: np.ndarray,
        segment_actions: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        outcome_env.unwrapped.set_state_dict(_clone_mani_state_dict(start_state))
        obs_branch = outcome_env.unwrapped.get_obs()
        branch_previous = previous_start.copy()
        branch_held_goal = np.zeros((num_envs, state_dim), dtype=np.float32)
        branch_countdown = np.zeros(num_envs, dtype=np.int32)
        branch_success_once = np.zeros(num_envs, dtype=np.bool_)
        branch_return = np.zeros(num_envs, dtype=np.float32)
        completed = np.zeros(num_envs, dtype=np.bool_)
        final_success = np.zeros(num_envs, dtype=np.float32)
        final_return = np.zeros(num_envs, dtype=np.float32)

        def record_step(step_reward: Any, info: dict[str, Any]) -> None:
            reward_np = _to_numpy(step_reward).reshape(-1).astype(np.float32)
            branch_return[:] += reward_np
            if "success" in info:
                branch_success_once[:] |= (
                    _to_numpy(info["success"]).reshape(-1).astype(np.bool_)
                )
            if "final_info" not in info:
                return
            mask_tensor = info["_final_info"]
            mask_np = mask_tensor.detach().cpu().numpy().astype(np.bool_)
            if not np.any(mask_np):
                return
            final_info = info["final_info"]
            done_indices = np.flatnonzero(mask_np)
            if "episode" in final_info:
                ep = final_info["episode"]
                success_all = ep["success_once"][mask_tensor].detach().float().cpu().numpy()
                return_all = ep["return"][mask_tensor].detach().float().cpu().numpy()
            else:
                success_all = branch_success_once[done_indices].astype(np.float32)
                return_all = branch_return[done_indices]
            for local_index, env_index in enumerate(done_indices):
                if completed[env_index]:
                    continue
                final_success[env_index] = float(success_all[local_index])
                final_return[env_index] = float(return_all[local_index])
                completed[env_index] = True

        for step in range(horizon_steps):
            action = torch.as_tensor(
                segment_actions[step],
                device=device,
                dtype=torch.float32,
            )
            action = torch.clamp(action, action_low, action_high)
            obs_branch, reward, _terminated, _truncated, info = outcome_env.step(action)
            record_step(reward, info)
            branch_previous = action_norm.transform(action.cpu().numpy().astype(np.float32))
            branch_previous[completed] = zero_previous
            branch_held_goal[completed] = 0.0
            branch_countdown[completed] = 0

        rollout_steps = horizon_steps
        while not bool(np.all(completed)) and rollout_steps < max_rollout_steps:
            current_norm = state_norm.transform(_obs_state_np(obs_branch))
            replan = branch_countdown <= 0
            if np.any(replan):
                high_goal = predict_high_goal(current_norm, branch_previous)
                branch_held_goal[replan] = high_goal[replan]
                branch_countdown[replan] = horizon_steps
            action = predict_low_action(
                current_norm,
                branch_held_goal,
                branch_previous,
                branch_countdown,
            )
            branch_countdown -= 1
            obs_branch, reward, _terminated, _truncated, info = outcome_env.step(action)
            record_step(reward, info)
            branch_previous = action_norm.transform(action.cpu().numpy().astype(np.float32))
            branch_previous[completed] = zero_previous
            branch_held_goal[completed] = 0.0
            branch_countdown[completed] = 0
            rollout_steps += 1
        final_success[~completed] = branch_success_once[~completed].astype(np.float32)
        final_return[~completed] = branch_return[~completed]
        return final_success, final_return, completed

    obs, _info = env.reset(seed=seed_start)
    search_env.reset(seed=seed_start)
    if goal_env is not None:
        goal_env.reset(seed=seed_start)
    base_outcome_env.reset(seed=seed_start)
    candidate_outcome_env.reset(seed=seed_start)
    try:
        while len(successes) < episodes:
            state_np = _to_numpy(obs["state"]).astype(np.float32)
            normalized_state = state_norm.transform(state_np)
            replan = countdown <= 0
            if np.any(replan):
                high_goal_np = predict_high_goal(normalized_state, previous)
                held_goal[replan] = high_goal_np[replan]
                countdown[replan] = horizon_steps
                if branch_batches < max_branch_batches:
                    start_state = _clone_mani_state_dict(env.unwrapped.get_state_dict())
                    base_mse, _base_conditions, base_actions = rollout_search_candidate(
                        start_state,
                        held_goal.copy(),
                        previous.copy(),
                        None,
                    )
                    best_mse = base_mse.copy()
                    best_conditions = _base_conditions.copy()
                    best_actions = base_actions.copy()
                    if branch_source == "random_search":
                        for _candidate in range(random_candidates):
                            noise = rng.normal(
                                0.0,
                                random_noise_std,
                                size=(horizon_steps, num_envs, action_dim),
                            ).astype(np.float32)
                            candidate_mse, candidate_conditions, candidate_actions = rollout_search_candidate(
                                start_state,
                                held_goal.copy(),
                                previous.copy(),
                                noise,
                            )
                            improved = candidate_mse < best_mse
                            if np.any(improved):
                                best_mse[improved] = candidate_mse[improved]
                                best_conditions[:, improved] = candidate_conditions[:, improved]
                                best_actions[:, improved] = candidate_actions[:, improved]
                    else:
                        oracle_goal = oracle_goal_from_state(start_state)
                        condition_goal = (
                            oracle_goal
                            if branch_condition_goal_source == "oracle_goal"
                            else held_goal.copy()
                        )
                        candidate_mse, candidate_conditions, candidate_actions = (
                            rollout_oracle_low_level_candidate(
                                start_state,
                                held_goal.copy(),
                                oracle_goal,
                                condition_goal,
                                previous.copy(),
                            )
                        )
                        best_mse = candidate_mse
                        best_conditions = candidate_conditions
                        best_actions = candidate_actions
                    improvement = base_mse - best_mse
                    action_delta_l2 = np.linalg.norm(
                        best_actions - base_actions,
                        axis=-1,
                    ).mean(axis=0)
                    selected = (
                        replan
                        & (improvement >= min_improvement_mse)
                        & (best_mse < base_mse)
                    )
                    if max_action_delta_l2 is not None:
                        selected &= action_delta_l2 <= max_action_delta_l2
                    base_success, base_return, base_completed = run_outcome_rollout(
                        base_outcome_env,
                        start_state,
                        previous.copy(),
                        base_actions,
                    )
                    candidate_success, candidate_return, candidate_completed = (
                        run_outcome_rollout(
                            candidate_outcome_env,
                            start_state,
                            previous.copy(),
                            best_actions,
                        )
                    )
                    local_base_mse_values.append(base_mse[replan].copy())
                    local_best_mse_values.append(best_mse[replan].copy())
                    local_improvement_values.append(improvement[replan].copy())
                    action_delta_values.append(action_delta_l2[replan].copy())
                    selected_values.append(selected[replan].copy())
                    condition_values.append(best_conditions[:, replan].copy())
                    action_values.append(best_actions[:, replan].copy())
                    base_success_values.append(base_success[replan].copy())
                    candidate_success_values.append(candidate_success[replan].copy())
                    base_return_values.append(base_return[replan].copy())
                    candidate_return_values.append(candidate_return[replan].copy())
                    outcome_completed_values.append(
                        (base_completed & candidate_completed)[replan].copy()
                    )
                    branch_batches += 1

            current_norm = state_norm.transform(_obs_state_np(obs))
            action = predict_low_action(current_norm, held_goal, previous, countdown)
            countdown -= 1
            obs, reward, _terminated, _truncated, info = env.step(action)
            previous = action_norm.transform(action.cpu().numpy().astype(np.float32))
            reward_np = _to_numpy(reward).reshape(-1).astype(np.float32)
            cumulative_returns += reward_np
            if "success" in info:
                success_once |= _to_numpy(info["success"]).reshape(-1).astype(np.bool_)
            if "final_info" in info:
                mask = info["_final_info"]
                if bool(mask.any()):
                    mask_np = mask.detach().cpu().numpy().astype(np.bool_)
                    final_info = info["final_info"]
                    if "episode" in final_info:
                        ep = final_info["episode"]
                        success_values = (
                            ep["success_once"][mask].detach().float().cpu().numpy()
                        )
                        return_values = ep["return"][mask].detach().float().cpu().numpy()
                    else:
                        success_values = success_once[mask_np].astype(np.float32)
                        return_values = cumulative_returns[mask_np].copy()
                    successes.extend(float(x) for x in success_values)
                    returns.extend(float(x) for x in return_values)
                    previous[mask_np] = zero_previous
                    held_goal[mask_np] = 0.0
                    countdown[mask_np] = 0
                    success_once[mask_np] = False
                    cumulative_returns[mask_np] = 0.0
    finally:
        env.close()
        search_env.close()
        if goal_env is not None:
            goal_env.close()
        base_outcome_env.close()
        candidate_outcome_env.close()

    if not local_base_mse_values:
        raise RuntimeError("No branch outcome batches were evaluated")
    local_base = np.concatenate(local_base_mse_values, axis=0).astype(np.float32)
    local_best = np.concatenate(local_best_mse_values, axis=0).astype(np.float32)
    local_improvement = np.concatenate(local_improvement_values, axis=0).astype(np.float32)
    action_delta = np.concatenate(action_delta_values, axis=0).astype(np.float32)
    selected = np.concatenate(selected_values, axis=0).astype(np.bool_)
    condition_blocks = condition_values
    action_blocks = action_values
    base_success = np.concatenate(base_success_values, axis=0).astype(np.float32)
    candidate_success = np.concatenate(candidate_success_values, axis=0).astype(np.float32)
    base_return = np.concatenate(base_return_values, axis=0).astype(np.float32)
    candidate_return = np.concatenate(candidate_return_values, axis=0).astype(np.float32)
    completed = np.concatenate(outcome_completed_values, axis=0).astype(np.bool_)
    success_delta = candidate_success - base_success
    return_delta = candidate_return - base_return
    bank_written = None
    if bank_output_path is not None:
        bank_mask = np.ones_like(selected, dtype=np.bool_)
        if bank_min_success_delta is not None:
            bank_mask &= success_delta >= float(bank_min_success_delta)
        if bank_min_return_delta is not None:
            bank_mask &= return_delta >= float(bank_min_return_delta)
        if not np.any(bank_mask):
            raise ValueError("Outcome-attributed branch bank selected no branches")
        conditions_all = np.concatenate(condition_blocks, axis=1).astype(np.float32)
        actions_all = np.concatenate(action_blocks, axis=1).astype(np.float32)
        bank_conditions = conditions_all[:, bank_mask].reshape(
            -1,
            conditions_all.shape[-1],
        )
        bank_actions = actions_all[:, bank_mask].reshape(-1, action_dim)
        ensure_dir(bank_output_path.parent)
        np.savez_compressed(
            bank_output_path,
            conditions=bank_conditions.astype(np.float32),
            actions=bank_actions.astype(np.float32),
            selected_base_mse=local_base[bank_mask].astype(np.float32),
            selected_best_mse=local_best[bank_mask].astype(np.float32),
            selected_improvement_mse=local_improvement[bank_mask].astype(np.float32),
            selected_action_delta_l2=action_delta[bank_mask].astype(np.float32),
            selected_base_success=base_success[bank_mask].astype(np.float32),
            selected_candidate_success=candidate_success[bank_mask].astype(np.float32),
            selected_success_delta=success_delta[bank_mask].astype(np.float32),
            selected_base_return=base_return[bank_mask].astype(np.float32),
            selected_candidate_return=candidate_return[bank_mask].astype(np.float32),
            selected_return_delta=return_delta[bank_mask].astype(np.float32),
            selected_completed=completed[bank_mask].astype(np.bool_),
            sample_weights=np.repeat(
                np.maximum(return_delta[bank_mask], 1.0)[None, :],
                horizon_steps,
                axis=0,
            )
            .reshape(-1)
            .astype(np.float32),
            horizon_steps=np.asarray(horizon_steps, dtype=np.int64),
            episodes=np.asarray(episodes, dtype=np.int64),
            seed_start=np.asarray(seed_start, dtype=np.int64),
            num_envs=np.asarray(num_envs, dtype=np.int64),
            random_candidates=np.asarray(random_candidates, dtype=np.int64),
            random_noise_std=np.asarray(random_noise_std, dtype=np.float32),
            branch_source=np.asarray(branch_source),
            branch_condition_goal_source=np.asarray(branch_condition_goal_source),
            bank_min_success_delta=np.asarray(
                np.nan if bank_min_success_delta is None else bank_min_success_delta,
                dtype=np.float32,
            ),
            bank_min_return_delta=np.asarray(
                np.nan if bank_min_return_delta is None else bank_min_return_delta,
                dtype=np.float32,
            ),
        )
        bank_written = {
            "path": str(bank_output_path),
            "branches": int(np.sum(bank_mask)),
            "condition_rows": int(len(bank_conditions)),
            "success_delta": _summarize_array(success_delta[bank_mask]),
            "return_delta": _summarize_array(return_delta[bank_mask]),
        }

    def summarize_mask(mask: np.ndarray) -> dict[str, Any]:
        if not np.any(mask):
            return {"count": 0}
        return {
            "count": int(np.sum(mask)),
            "local_base_mse": _summarize_array(local_base[mask]),
            "local_best_mse": _summarize_array(local_best[mask]),
            "local_improvement_mse": _summarize_array(local_improvement[mask]),
            "action_delta_l2": _summarize_array(action_delta[mask]),
            "base_success": float(np.mean(base_success[mask])),
            "candidate_success": float(np.mean(candidate_success[mask])),
            "success_delta": float(np.mean(success_delta[mask])),
            "base_return": float(np.mean(base_return[mask])),
            "candidate_return": float(np.mean(candidate_return[mask])),
            "return_delta": _summarize_array(return_delta[mask]),
            "candidate_better_success_fraction": float(np.mean(success_delta[mask] > 0.0)),
            "candidate_worse_success_fraction": float(np.mean(success_delta[mask] < 0.0)),
            "candidate_better_return_fraction": float(np.mean(return_delta[mask] > 0.0)),
            "completed_fraction": float(np.mean(completed[mask])),
        }

    result = {
        "method": "privileged_z_branch_outcome_attribution",
        "checkpoint": str(checkpoint_path),
        "episodes": int(episodes),
        "seed_start": int(seed_start),
        "num_envs": int(num_envs),
        "horizon_steps": horizon_steps,
        "random_candidates": int(random_candidates),
        "random_noise_std": float(random_noise_std),
        "branch_source": branch_source,
        "branch_condition_goal_source": branch_condition_goal_source,
        "min_improvement_mse": float(min_improvement_mse),
        "max_action_delta_l2": max_action_delta_l2,
        "max_branch_batches": int(max_branch_batches),
        "max_rollout_steps": int(max_rollout_steps),
        "branch_batches": int(branch_batches),
        "base_closed_loop_success": float(np.mean(successes[:episodes])),
        "base_closed_loop_return": float(np.mean(returns[:episodes])),
        "bank_output": bank_written,
        "all_branches": summarize_mask(np.ones_like(selected, dtype=np.bool_)),
        "locally_selected_branches": summarize_mask(selected),
        "locally_rejected_branches": summarize_mask(~selected),
    }
    write_json(out_path, result)
    console.print(result)
    return out_path


def _entry_env_indices(entry: dict[str, Any], num_envs: int) -> np.ndarray | None:
    if "env_indices" not in entry:
        return None
    indices = np.asarray(entry["env_indices"], dtype=np.int64)
    if indices.ndim != 1:
        raise ValueError("Manifest env_indices must be a 1D list")
    if len(indices) == 0:
        raise ValueError("Manifest env_indices must not be empty")
    if np.any(indices < 0) or np.any(indices >= num_envs):
        raise ValueError(
            f"Manifest env_indices out of bounds for vector batch with {num_envs} envs"
        )
    return indices


def _filter_local_rollout(
    rollout: dict[str, np.ndarray],
    env_indices: np.ndarray | None,
) -> dict[str, np.ndarray]:
    if env_indices is None:
        return rollout
    filtered: dict[str, np.ndarray] = {}
    for key, value in rollout.items():
        if key in {"actions", "residual_norm", "saturation_frac"}:
            filtered[key] = value[:, env_indices]
        else:
            filtered[key] = value[env_indices]
    return filtered


@torch.inference_mode()
def evaluate_privileged_z_local_paired(
    config: Config,
    checkpoint_path: Path,
    *,
    manifest_path: Path,
    residual_checkpoint_path: Path | None = None,
    output_path: Path | None = None,
    goal_source: str = "replay",
    success_epsilon: float = 0.05,
    force: bool = False,
) -> Path:
    from hcl_poc.low_level_rl import ResidualActorCritic
    from hcl_poc.rl_rerun import _make_benchmark_env, _residual_action_from_raw

    if goal_source not in {"replay", "predicted"}:
        raise ValueError(f"Unknown privileged-z local goal source: {goal_source}")
    if success_epsilon <= 0:
        raise ValueError("success_epsilon must be positive")
    out_path = output_path or (
        checkpoint_path.with_name(
            f"{checkpoint_path.stem}_local_paired_{manifest_path.stem}.json"
        )
    )
    if out_path.exists() and not force:
        return out_path
    manifest = json.loads(manifest_path.read_text())
    dataset_path = Path(manifest["dataset"])
    if not dataset_path.exists():
        raise FileNotFoundError(dataset_path)
    entries = list(manifest["entries"])
    if not entries:
        raise ValueError(f"Manifest has no entries: {manifest_path}")

    device = default_device()
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_norm = Standardizer.from_state_dict(payload["state_norm"])
    action_norm = Standardizer.from_state_dict(payload["action_norm"])
    high_model = _model_from_payload(payload["high"], device)
    goal_model = _model_from_payload(payload["goal"], device)
    horizon_steps = int(payload["horizon_steps"])
    action_dim = int(payload["action_dim"])
    if int(manifest["horizon"]) != horizon_steps:
        raise ValueError(
            f"Manifest horizon {manifest['horizon']} does not match checkpoint "
            f"horizon {horizon_steps}"
        )

    residual_agent = None
    direct_agent: PrivilegedZDirectActorCritic | None = None
    residual_recipe: dict[str, Any] | None = None
    tuned_recipe: dict[str, Any] | None = None
    residual_alpha = 0.0
    residual_action_mode = "additive"
    if residual_checkpoint_path is not None:
        tuned_payload = torch.load(
            residual_checkpoint_path,
            map_location=device,
            weights_only=False,
        )
        tuned_recipe = dict(tuned_payload["recipe"])
        if Path(tuned_recipe["base_checkpoint"]).resolve() != checkpoint_path.resolve():
            raise ValueError("Tuned checkpoint was trained against a different base checkpoint")
        method = str(tuned_recipe.get("method", ""))
        if method == "privileged_z_residual_r1":
            residual_recipe = tuned_recipe
            residual_agent = ResidualActorCritic(
                int(tuned_payload["condition_dim"]),
                action_dim=int(tuned_payload["action_dim"]),
                width=int(tuned_recipe["actor_critic_width"]),
                depth=int(tuned_recipe["actor_critic_depth"]),
                initial_logstd=float(tuned_recipe["initial_logstd"]),
            ).to(device)
            residual_agent.load_state_dict(tuned_payload["agent"])
            residual_agent.eval()
            residual_alpha = float(tuned_recipe["alpha"])
            residual_action_mode = str(tuned_recipe.get("residual_action_mode", "additive"))
        elif method in {"privileged_z_direct_r3", "privileged_z_direct_distill"}:
            direct_agent = PrivilegedZDirectActorCritic(
                goal_model,
                payload["goal"],
                action_norm.mean,
                action_norm.std,
                int(tuned_payload["condition_dim"]),
                action_dim=int(tuned_payload["action_dim"]),
                train_scope=str(tuned_recipe["train_scope"]),
                width=int(tuned_recipe["actor_critic_width"]),
                depth=int(tuned_recipe["actor_critic_depth"]),
                initial_logstd=float(tuned_recipe["initial_logstd"]),
            ).to(device)
            direct_agent.load_state_dict(tuned_payload["agent"])
            direct_agent.eval()
        else:
            raise ValueError(f"Unknown privileged-z tuned checkpoint method: {method}")

    def rollout_from_entry(
        h5: h5py.File,
        entry: dict[str, Any],
        *,
        use_tuned: bool,
    ) -> dict[str, np.ndarray]:
        batch_key = str(entry["batch"])
        if batch_key not in h5:
            raise ValueError(f"Unknown manifest batch {batch_key}")
        group = h5[batch_key]
        num_envs = int(group.attrs["num_envs"])
        env_indices = _entry_env_indices(entry, num_envs)
        timestep = int(entry["timestep"])
        if not 0 <= timestep <= int(group.attrs["max_steps"]) - horizon_steps:
            raise ValueError(f"Invalid manifest timestep {timestep}")
        env = _make_benchmark_env(config, num_envs, "state")
        action_low = torch.as_tensor(
            env.single_action_space.low,
            device=device,
            dtype=torch.float32,
        )
        action_high = torch.as_tensor(
            env.single_action_space.high,
            device=device,
            dtype=torch.float32,
        )
        obs, _info = env.reset(seed=int(group.attrs["batch_seed"]))
        try:
            for replay_step in range(timestep):
                replay_action = torch.from_numpy(
                    np.asarray(group["executed_actions"][replay_step], dtype=np.float32)
                ).to(device)
                obs, _reward, _terminated, _truncated, _info = env.step(replay_action)

            state_np = _obs_state_np(obs)
            current_norm = state_norm.transform(state_np)
            previous_norm = action_norm.transform(
                np.asarray(group["previous_executed_actions"][timestep], dtype=np.float32)
            )
            replay_goal_norm = state_norm.transform(
                np.asarray(
                    group["observations_state"][timestep + horizon_steps],
                    dtype=np.float32,
                )
            )
            if goal_source == "predicted":
                high_input = np.concatenate([current_norm, previous_norm], axis=-1)
                goal_norm = _predict_loaded_payload(
                    high_model,
                    payload["high"],
                    high_input,
                    device,
                )
            else:
                goal_norm = replay_goal_norm

            initial_mse = np.mean((current_norm - goal_norm) ** 2, axis=-1).astype(np.float32)
            initial_l2 = np.linalg.norm(current_norm - goal_norm, axis=-1).astype(np.float32)
            action_steps: list[np.ndarray] = []
            residual_norms: list[np.ndarray] = []
            saturation_fracs: list[np.ndarray] = []
            for step in range(horizon_steps):
                remaining = np.full(
                    (num_envs, 1),
                    max(horizon_steps - step, 1) / horizon_steps,
                    dtype=np.float32,
                )
                condition_np = np.concatenate(
                    [current_norm, goal_norm, previous_norm, remaining],
                    axis=-1,
                ).astype(np.float32)
                normalized_action = _predict_loaded_payload(
                    goal_model,
                    payload["goal"],
                    condition_np,
                    device,
                )
                base_action = torch.from_numpy(
                    action_norm.inverse(normalized_action)
                ).to(device)
                unclipped = base_action
                residual = torch.zeros((num_envs, action_dim), device=device)
                if use_tuned and residual_agent is not None:
                    condition = torch.from_numpy(condition_np).to(device).float()
                    raw_residual, _logprob, _entropy, _value = (
                        residual_agent.get_action_and_value(condition, deterministic=True)
                    )
                    residual, unclipped, action = _residual_action_from_raw(
                        base_action,
                        raw_residual,
                        residual_alpha,
                        action_low,
                        action_high,
                        residual_action_mode,
                    )
                elif use_tuned and direct_agent is not None:
                    condition = torch.from_numpy(condition_np).to(device).float()
                    raw_action, _logprob, _entropy, _value = (
                        direct_agent.get_action_and_value(condition, deterministic=True)
                    )
                    unclipped = raw_action
                    action = torch.clamp(raw_action, action_low, action_high)
                    residual = action - torch.clamp(base_action, action_low, action_high)
                else:
                    action = torch.clamp(base_action, action_low, action_high)
                saturated = ((unclipped < action_low) | (unclipped > action_high)).float()
                action_steps.append(action.detach().cpu().numpy().astype(np.float32))
                residual_norms.append(
                    torch.linalg.vector_norm(residual, dim=-1).cpu().numpy().astype(np.float32)
                )
                saturation_fracs.append(saturated.mean(dim=-1).cpu().numpy().astype(np.float32))
                obs, _reward, _terminated, _truncated, _info = env.step(action)
                current_norm = state_norm.transform(
                    _obs_state_np(obs)
                )
                previous_norm = action_norm.transform(
                    action.detach().cpu().numpy().astype(np.float32)
                )
            terminal_mse = np.mean((current_norm - goal_norm) ** 2, axis=-1).astype(np.float32)
            terminal_l2 = np.linalg.norm(current_norm - goal_norm, axis=-1).astype(np.float32)
            return {
                "initial_mse": initial_mse,
                "initial_l2": initial_l2,
                "terminal_mse": terminal_mse,
                "terminal_l2": terminal_l2,
                "actions": np.stack(action_steps, axis=0),
                "residual_norm": np.stack(residual_norms, axis=0),
                "saturation_frac": np.stack(saturation_fracs, axis=0),
            }
        finally:
            env.close()

    base_terminal_mse: list[np.ndarray] = []
    tuned_terminal_mse: list[np.ndarray] = []
    base_terminal_l2: list[np.ndarray] = []
    tuned_terminal_l2: list[np.ndarray] = []
    initial_mse_values: list[np.ndarray] = []
    initial_l2_values: list[np.ndarray] = []
    action_delta_l2: list[np.ndarray] = []
    residual_norm_values: list[np.ndarray] = []
    saturation_values: list[np.ndarray] = []
    with h5py.File(dataset_path, "r") as h5:
        for entry in entries:
            batch_key = str(entry["batch"])
            group = h5[batch_key]
            env_indices = _entry_env_indices(entry, int(group.attrs["num_envs"]))
            base = _filter_local_rollout(
                rollout_from_entry(h5, entry, use_tuned=False),
                env_indices,
            )
            tuned = (
                _filter_local_rollout(
                    rollout_from_entry(h5, entry, use_tuned=True),
                    env_indices,
                )
                if (residual_agent is not None or direct_agent is not None)
                else base
            )
            initial_mse_values.append(base["initial_mse"])
            initial_l2_values.append(base["initial_l2"])
            base_terminal_mse.append(base["terminal_mse"])
            tuned_terminal_mse.append(tuned["terminal_mse"])
            base_terminal_l2.append(base["terminal_l2"])
            tuned_terminal_l2.append(tuned["terminal_l2"])
            action_delta_l2.append(
                np.linalg.norm(tuned["actions"] - base["actions"], axis=-1).reshape(-1)
            )
            residual_norm_values.append(tuned["residual_norm"].reshape(-1))
            saturation_values.append(tuned["saturation_frac"].reshape(-1))

    base_mse = np.concatenate(base_terminal_mse, axis=0)
    tuned_mse = np.concatenate(tuned_terminal_mse, axis=0)
    base_l2 = np.concatenate(base_terminal_l2, axis=0)
    tuned_l2 = np.concatenate(tuned_terminal_l2, axis=0)
    improvement = base_mse - tuned_mse
    result = {
        "method": "privileged_z_local_paired_eval",
        "checkpoint": str(checkpoint_path),
        "residual_checkpoint": str(residual_checkpoint_path) if residual_checkpoint_path else None,
        "tuned_checkpoint": str(residual_checkpoint_path) if residual_checkpoint_path else None,
        "manifest": str(manifest_path),
        "dataset": str(dataset_path),
        "goal_source": goal_source,
        "horizon_steps": horizon_steps,
        "entries": len(entries),
        "num_local_episodes": int(len(improvement)),
        "success_epsilon_mse": success_epsilon,
        "base_terminal_mse": _summarize_array(base_mse),
        "tuned_terminal_mse": _summarize_array(tuned_mse),
        "base_terminal_l2": _summarize_array(base_l2),
        "tuned_terminal_l2": _summarize_array(tuned_l2),
        "initial_mse": _summarize_array(np.concatenate(initial_mse_values, axis=0)),
        "initial_l2": _summarize_array(np.concatenate(initial_l2_values, axis=0)),
        "paired_improvement_mse": _summarize_array(improvement),
        "fraction_improved": float(np.mean(improvement > 0.0)),
        "base_success_within_epsilon": float(np.mean(base_mse < success_epsilon)),
        "tuned_success_within_epsilon": float(np.mean(tuned_mse < success_epsilon)),
        "action_delta_l2": _summarize_array(np.concatenate(action_delta_l2, axis=0)),
        "residual_norm": _summarize_array(np.concatenate(residual_norm_values, axis=0)),
        "action_saturation_frac": _summarize_array(np.concatenate(saturation_values, axis=0)),
        "residual_recipe": residual_recipe,
        "tuned_recipe": tuned_recipe,
    }
    write_json(out_path, result)
    console.print(result)
    return out_path


@torch.inference_mode()
def create_privileged_z_hard_case_manifest(
    config: Config,
    checkpoint_path: Path,
    *,
    manifest_path: Path,
    output_path: Path,
    goal_source: str = "replay",
    threshold_mse: float = 0.05,
    max_envs_per_entry: int | None = None,
    seed: int = 0,
    force: bool = False,
) -> Path:
    from hcl_poc.rl_rerun import _make_benchmark_env

    if goal_source not in {"replay", "predicted"}:
        raise ValueError(f"Unknown privileged-z local goal source: {goal_source}")
    if threshold_mse < 0:
        raise ValueError("threshold_mse must be non-negative")
    if max_envs_per_entry is not None and max_envs_per_entry <= 0:
        raise ValueError("max_envs_per_entry must be positive when provided")
    if output_path.exists() and not force:
        return output_path

    manifest = json.loads(manifest_path.read_text())
    dataset_path = Path(manifest["dataset"])
    if not dataset_path.exists():
        raise FileNotFoundError(dataset_path)
    entries = list(manifest["entries"])
    if not entries:
        raise ValueError(f"Manifest has no entries: {manifest_path}")

    device = default_device()
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_norm = Standardizer.from_state_dict(payload["state_norm"])
    action_norm = Standardizer.from_state_dict(payload["action_norm"])
    high_model = _model_from_payload(payload["high"], device)
    goal_model = _model_from_payload(payload["goal"], device)
    horizon_steps = int(payload["horizon_steps"])
    if int(manifest["horizon"]) != horizon_steps:
        raise ValueError(
            f"Manifest horizon {manifest['horizon']} does not match checkpoint "
            f"horizon {horizon_steps}"
        )
    rng = np.random.default_rng(seed + 8_200_000)
    hard_entries: list[dict[str, Any]] = []
    base_mse_values: list[np.ndarray] = []

    with h5py.File(dataset_path, "r") as h5:
        for entry in entries:
            batch_key = str(entry["batch"])
            if batch_key not in h5:
                raise ValueError(f"Unknown manifest batch {batch_key}")
            group = h5[batch_key]
            num_envs = int(group.attrs["num_envs"])
            parent_indices = _entry_env_indices(entry, num_envs)
            candidate_indices = (
                parent_indices
                if parent_indices is not None
                else np.arange(num_envs, dtype=np.int64)
            )
            timestep = int(entry["timestep"])
            if not 0 <= timestep <= int(group.attrs["max_steps"]) - horizon_steps:
                raise ValueError(f"Invalid manifest timestep {timestep}")

            env = _make_benchmark_env(config, num_envs, "state")
            action_low = torch.as_tensor(
                env.single_action_space.low,
                device=device,
                dtype=torch.float32,
            )
            action_high = torch.as_tensor(
                env.single_action_space.high,
                device=device,
                dtype=torch.float32,
            )
            try:
                obs, _info = env.reset(seed=int(group.attrs["batch_seed"]))
                for replay_step in range(timestep):
                    replay_action = torch.from_numpy(
                        np.asarray(group["executed_actions"][replay_step], dtype=np.float32)
                    ).to(device)
                    obs, _reward, _terminated, _truncated, _info = env.step(replay_action)
                current_norm = state_norm.transform(_obs_state_np(obs))
                previous_norm = action_norm.transform(
                    np.asarray(group["previous_executed_actions"][timestep], dtype=np.float32)
                )
                replay_goal_norm = state_norm.transform(
                    np.asarray(
                        group["observations_state"][timestep + horizon_steps],
                        dtype=np.float32,
                    )
                )
                if goal_source == "predicted":
                    high_input = np.concatenate([current_norm, previous_norm], axis=-1)
                    goal_norm = _predict_loaded_payload(
                        high_model,
                        payload["high"],
                        high_input,
                        device,
                    )
                else:
                    goal_norm = replay_goal_norm
                for step in range(horizon_steps):
                    remaining = np.full(
                        (num_envs, 1),
                        max(horizon_steps - step, 1) / horizon_steps,
                        dtype=np.float32,
                    )
                    condition_np = np.concatenate(
                        [current_norm, goal_norm, previous_norm, remaining],
                        axis=-1,
                    ).astype(np.float32)
                    normalized_action = _predict_loaded_payload(
                        goal_model,
                        payload["goal"],
                        condition_np,
                        device,
                    )
                    action = torch.from_numpy(action_norm.inverse(normalized_action)).to(device)
                    action = torch.clamp(action, action_low, action_high)
                    obs, _reward, _terminated, _truncated, _info = env.step(action)
                    current_norm = state_norm.transform(_obs_state_np(obs))
                    previous_norm = action_norm.transform(
                        action.cpu().numpy().astype(np.float32)
                    )
                terminal_mse = np.mean((current_norm - goal_norm) ** 2, axis=-1).astype(
                    np.float32
                )
            finally:
                env.close()

            candidate_mse = terminal_mse[candidate_indices]
            selected = candidate_indices[candidate_mse >= threshold_mse]
            if max_envs_per_entry is not None and len(selected) > max_envs_per_entry:
                selected = np.asarray(
                    rng.choice(selected, size=max_envs_per_entry, replace=False),
                    dtype=np.int64,
                )
                selected.sort()
            if len(selected):
                hard_entry = dict(entry)
                hard_entry["env_indices"] = [int(index) for index in selected.tolist()]
                hard_entry["parent_num_envs"] = num_envs
                hard_entries.append(hard_entry)
                base_mse_values.append(terminal_mse[selected])

    if not hard_entries:
        raise ValueError(
            f"No hard cases selected from {manifest_path} at threshold {threshold_mse}"
        )
    selected_mse = np.concatenate(base_mse_values, axis=0)
    hard_manifest = {
        **{key: value for key, value in manifest.items() if key != "entries"},
        "source_manifest": str(manifest_path),
        "hard_case_selector": {
            "method": "frozen_base_terminal_mse_threshold",
            "checkpoint": str(checkpoint_path),
            "goal_source": goal_source,
            "threshold_mse": threshold_mse,
            "max_envs_per_entry": max_envs_per_entry,
            "seed": seed,
            "selected_local_episodes": int(len(selected_mse)),
            "base_terminal_mse": _summarize_array(selected_mse),
        },
        "entries": hard_entries,
    }
    write_json(output_path, hard_manifest)
    console.print(hard_manifest["hard_case_selector"])
    return output_path


@torch.inference_mode()
def evaluate_privileged_z_local_action_search(
    config: Config,
    checkpoint_path: Path,
    *,
    manifest_path: Path,
    output_path: Path | None = None,
    goal_source: str = "replay",
    random_candidates: int = 32,
    random_noise_std: float = 0.05,
    success_epsilon: float = 0.05,
    seed: int = 0,
    force: bool = False,
) -> Path:
    from hcl_poc.rl_rerun import _make_benchmark_env

    if goal_source not in {"replay", "predicted"}:
        raise ValueError(f"Unknown privileged-z local goal source: {goal_source}")
    if random_candidates < 0:
        raise ValueError("random_candidates must be non-negative")
    if random_noise_std < 0:
        raise ValueError("random_noise_std must be non-negative")
    if success_epsilon <= 0:
        raise ValueError("success_epsilon must be positive")
    out_path = output_path or checkpoint_path.with_name(
        f"{checkpoint_path.stem}_local_action_search_{manifest_path.stem}.json"
    )
    if out_path.exists() and not force:
        return out_path

    manifest = json.loads(manifest_path.read_text())
    dataset_path = Path(manifest["dataset"])
    if not dataset_path.exists():
        raise FileNotFoundError(dataset_path)
    entries = list(manifest["entries"])
    if not entries:
        raise ValueError(f"Manifest has no entries: {manifest_path}")

    device = default_device()
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_norm = Standardizer.from_state_dict(payload["state_norm"])
    action_norm = Standardizer.from_state_dict(payload["action_norm"])
    high_model = _model_from_payload(payload["high"], device)
    goal_model = _model_from_payload(payload["goal"], device)
    horizon_steps = int(payload["horizon_steps"])
    if int(manifest["horizon"]) != horizon_steps:
        raise ValueError(
            f"Manifest horizon {manifest['horizon']} does not match checkpoint "
            f"horizon {horizon_steps}"
        )

    rng = np.random.default_rng(seed + 8_100_000)
    base_terminal_mse: list[np.ndarray] = []
    replay_terminal_mse: list[np.ndarray] = []
    random_best_terminal_mse: list[np.ndarray] = []

    def base_action(
        current_norm: np.ndarray,
        goal_norm: np.ndarray,
        previous_norm: np.ndarray,
        step: int,
    ) -> torch.Tensor:
        num_envs = current_norm.shape[0]
        remaining = np.full(
            (num_envs, 1),
            max(horizon_steps - step, 1) / horizon_steps,
            dtype=np.float32,
        )
        condition_np = np.concatenate(
            [current_norm, goal_norm, previous_norm, remaining],
            axis=-1,
        ).astype(np.float32)
        normalized_action = _predict_loaded_payload(
            goal_model,
            payload["goal"],
            condition_np,
            device,
        )
        return torch.from_numpy(action_norm.inverse(normalized_action)).to(device)

    def terminal_mse(obs: Any, goal_norm: np.ndarray) -> np.ndarray:
        final_norm = state_norm.transform(_obs_state_np(obs))
        return np.mean((final_norm - goal_norm) ** 2, axis=-1).astype(np.float32)

    with h5py.File(dataset_path, "r") as h5:
        for entry in entries:
            batch_key = str(entry["batch"])
            if batch_key not in h5:
                raise ValueError(f"Unknown manifest batch {batch_key}")
            group = h5[batch_key]
            num_envs = int(group.attrs["num_envs"])
            env_indices = _entry_env_indices(entry, num_envs)
            timestep = int(entry["timestep"])
            if not 0 <= timestep <= int(group.attrs["max_steps"]) - horizon_steps:
                raise ValueError(f"Invalid manifest timestep {timestep}")
            env = _make_benchmark_env(config, num_envs, "state")
            action_low = torch.as_tensor(
                env.single_action_space.low,
                device=device,
                dtype=torch.float32,
            )
            action_high = torch.as_tensor(
                env.single_action_space.high,
                device=device,
                dtype=torch.float32,
            )
            try:
                obs, _info = env.reset(seed=int(group.attrs["batch_seed"]))
                for replay_step in range(timestep):
                    replay_action = torch.from_numpy(
                        np.asarray(group["executed_actions"][replay_step], dtype=np.float32)
                    ).to(device)
                    obs, _reward, _terminated, _truncated, _info = env.step(replay_action)
                start_state = _clone_mani_state_dict(env.unwrapped.get_state_dict())
                start_obs = env.unwrapped.get_obs()
                start_norm = state_norm.transform(_obs_state_np(start_obs))
                previous_start = action_norm.transform(
                    np.asarray(group["previous_executed_actions"][timestep], dtype=np.float32)
                )
                replay_goal_norm = state_norm.transform(
                    np.asarray(
                        group["observations_state"][timestep + horizon_steps],
                        dtype=np.float32,
                    )
                )
                if goal_source == "predicted":
                    high_input = np.concatenate([start_norm, previous_start], axis=-1)
                    goal_norm = _predict_loaded_payload(
                        high_model,
                        payload["high"],
                        high_input,
                        device,
                    )
                else:
                    goal_norm = replay_goal_norm

                obs = start_obs
                previous_norm = previous_start.copy()
                for step in range(horizon_steps):
                    current_norm = state_norm.transform(_obs_state_np(obs))
                    action = torch.clamp(
                        base_action(current_norm, goal_norm, previous_norm, step),
                        action_low,
                        action_high,
                    )
                    obs, _reward, _terminated, _truncated, _info = env.step(action)
                    previous_norm = action_norm.transform(
                        action.cpu().numpy().astype(np.float32)
                    )
                base_mse = terminal_mse(obs, goal_norm)
                if env_indices is not None:
                    base_mse = base_mse[env_indices]
                base_terminal_mse.append(base_mse)

                env.unwrapped.set_state_dict(_clone_mani_state_dict(start_state))
                obs = env.unwrapped.get_obs()
                for step in range(horizon_steps):
                    action = torch.from_numpy(
                        np.asarray(
                            group["executed_actions"][timestep + step],
                            dtype=np.float32,
                        )
                    ).to(device)
                    action = torch.clamp(action, action_low, action_high)
                    obs, _reward, _terminated, _truncated, _info = env.step(action)
                replay_mse = terminal_mse(obs, goal_norm)
                if env_indices is not None:
                    replay_mse = replay_mse[env_indices]
                replay_terminal_mse.append(replay_mse)

                if random_candidates:
                    best = base_mse.copy()
                    for _candidate in range(random_candidates):
                        env.unwrapped.set_state_dict(_clone_mani_state_dict(start_state))
                        obs = env.unwrapped.get_obs()
                        previous_norm = previous_start.copy()
                        for step in range(horizon_steps):
                            current_norm = state_norm.transform(_obs_state_np(obs))
                            action = base_action(current_norm, goal_norm, previous_norm, step)
                            noise = torch.from_numpy(
                                rng.normal(
                                    0.0,
                                    random_noise_std,
                                    size=(num_envs, int(payload["action_dim"])),
                                ).astype(np.float32)
                            ).to(device)
                            action = torch.clamp(action + noise, action_low, action_high)
                            obs, _reward, _terminated, _truncated, _info = env.step(action)
                            previous_norm = action_norm.transform(
                                action.cpu().numpy().astype(np.float32)
                            )
                        candidate_mse = terminal_mse(obs, goal_norm)
                        if env_indices is not None:
                            candidate_mse = candidate_mse[env_indices]
                        best = np.minimum(best, candidate_mse)
                    random_best_terminal_mse.append(best)
                else:
                    random_best_terminal_mse.append(base_mse.copy())
            finally:
                env.close()

    base_mse = np.concatenate(base_terminal_mse, axis=0)
    replay_mse = np.concatenate(replay_terminal_mse, axis=0)
    random_mse = np.concatenate(random_best_terminal_mse, axis=0)
    replay_improvement = base_mse - replay_mse
    random_improvement = base_mse - random_mse
    result = {
        "method": "privileged_z_local_action_search",
        "checkpoint": str(checkpoint_path),
        "manifest": str(manifest_path),
        "dataset": str(dataset_path),
        "goal_source": goal_source,
        "horizon_steps": horizon_steps,
        "entries": len(entries),
        "num_local_episodes": int(len(base_mse)),
        "random_candidates": int(random_candidates),
        "random_noise_std": float(random_noise_std),
        "success_epsilon_mse": success_epsilon,
        "base_terminal_mse": _summarize_array(base_mse),
        "replay_terminal_mse": _summarize_array(replay_mse),
        "random_best_terminal_mse": _summarize_array(random_mse),
        "replay_improvement_mse": _summarize_array(replay_improvement),
        "random_best_improvement_mse": _summarize_array(random_improvement),
        "replay_fraction_improved": float(np.mean(replay_improvement > 0.0)),
        "random_best_fraction_improved": float(np.mean(random_improvement > 0.0)),
        "base_success_within_epsilon": float(np.mean(base_mse < success_epsilon)),
        "replay_success_within_epsilon": float(np.mean(replay_mse < success_epsilon)),
        "random_best_success_within_epsilon": float(np.mean(random_mse < success_epsilon)),
    }
    write_json(out_path, result)
    console.print(result)
    return out_path


def train_privileged_z_local_replay_distill(
    config: Config,
    checkpoint_path: Path,
    *,
    manifest_path: Path,
    preserve_manifest_path: Path | None = None,
    preserve_npz_path: Path | None = None,
    improve_npz_path: Path | None = None,
    replay_weight: float = 1.0,
    preserve_weight: float = 0.0,
    preserve_npz_weight: float = 0.0,
    improve_npz_weight: float = 0.0,
    run_tag: str,
    seed: int = 0,
    epochs: int = 200,
    batch_size: int = 512,
    learning_rate: float = 1e-4,
    train_scope: str = "all",
    initial_logstd: float = -4.0,
    force: bool = False,
) -> Path:
    if epochs <= 0:
        raise ValueError("epochs must be positive")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive")
    if train_scope not in {"final_layer", "all"}:
        raise ValueError(f"Unknown privileged-z distill train scope: {train_scope}")
    if replay_weight <= 0.0:
        raise ValueError("replay_weight must be positive")
    if preserve_weight < 0.0:
        raise ValueError("preserve_weight must be non-negative")
    if preserve_npz_weight < 0.0:
        raise ValueError("preserve_npz_weight must be non-negative")
    if improve_npz_weight < 0.0:
        raise ValueError("improve_npz_weight must be non-negative")
    if preserve_manifest_path is None and preserve_weight > 0.0:
        raise ValueError("preserve_manifest_path is required when preserve_weight > 0")
    if preserve_manifest_path is not None and preserve_weight <= 0.0:
        raise ValueError("preserve_weight must be positive when preserve_manifest_path is set")
    if preserve_npz_path is None and preserve_npz_weight > 0.0:
        raise ValueError("preserve_npz_path is required when preserve_npz_weight > 0")
    if preserve_npz_path is not None and preserve_npz_weight <= 0.0:
        raise ValueError("preserve_npz_weight must be positive when preserve_npz_path is set")
    if improve_npz_path is None and improve_npz_weight > 0.0:
        raise ValueError("improve_npz_path is required when improve_npz_weight > 0")
    if improve_npz_path is not None and improve_npz_weight <= 0.0:
        raise ValueError("improve_npz_weight must be positive when improve_npz_path is set")

    manifest = json.loads(manifest_path.read_text())
    dataset_path = Path(manifest["dataset"])
    if not dataset_path.exists():
        raise FileNotFoundError(dataset_path)
    entries = list(manifest["entries"])
    if not entries:
        raise ValueError(f"Manifest has no entries: {manifest_path}")

    artifact = ensure_dir(
        config.path_value("paths.incremental_artifact_dir")
        / "privileged_z_direct_distill"
        / run_tag
        / f"seed{seed}"
    )
    result_dir = ensure_dir(
        config.path_value("paths.incremental_results_dir")
        / "privileged_z_direct_distill"
        / run_tag
        / f"seed{seed}"
    )
    checkpoint_out = artifact / "latest.pt"
    history_path = result_dir / "history.json"
    if checkpoint_out.exists() and not force:
        return checkpoint_out
    if force:
        checkpoint_out.unlink(missing_ok=True)
        history_path.unlink(missing_ok=True)

    device = default_device()
    set_seed(seed + 8_300_000)
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if payload["goal"].get("model_type") == "flow":
        raise ValueError("Replay distillation currently requires an MLP low policy")
    state_norm = Standardizer.from_state_dict(payload["state_norm"])
    action_norm = Standardizer.from_state_dict(payload["action_norm"])
    goal_model = _model_from_payload(payload["goal"], device)
    horizon_steps = int(payload["horizon_steps"])
    state_dim = int(payload["state_dim"])
    action_dim = int(payload["action_dim"])
    condition_dim = state_dim + state_dim + action_dim + 1
    if int(manifest["horizon"]) != horizon_steps:
        raise ValueError(
            f"Manifest horizon {manifest['horizon']} does not match checkpoint "
            f"horizon {horizon_steps}"
        )

    conditions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    sample_weights: list[np.ndarray] = []
    target_sample_counts: dict[str, int] = {}

    def append_manifest_rows(
        path: Path,
        *,
        target_mode: str,
        weight: float,
    ) -> Path:
        local_manifest = json.loads(path.read_text())
        local_dataset_path = Path(local_manifest["dataset"])
        if not local_dataset_path.exists():
            raise FileNotFoundError(local_dataset_path)
        local_entries = list(local_manifest["entries"])
        if not local_entries:
            raise ValueError(f"Manifest has no entries: {path}")
        if int(local_manifest["horizon"]) != horizon_steps:
            raise ValueError(
                f"Manifest horizon {local_manifest['horizon']} does not match checkpoint "
                f"horizon {horizon_steps}: {path}"
            )
        count_before = sum(len(chunk) for chunk in targets)
        with h5py.File(local_dataset_path, "r") as h5:
            for entry in local_entries:
                batch_key = str(entry["batch"])
                if batch_key not in h5:
                    raise ValueError(f"Unknown manifest batch {batch_key}")
                group = h5[batch_key]
                num_envs = int(group.attrs["num_envs"])
                env_indices = _entry_env_indices(entry, num_envs)
                if env_indices is None:
                    env_indices = np.arange(num_envs, dtype=np.int64)
                timestep = int(entry["timestep"])
                if not 0 <= timestep <= int(group.attrs["max_steps"]) - horizon_steps:
                    raise ValueError(f"Invalid manifest timestep {timestep}")
                goal_norm = state_norm.transform(
                    np.asarray(
                        group["observations_state"][timestep + horizon_steps, env_indices],
                        dtype=np.float32,
                    )
                )
                for offset in range(horizon_steps):
                    states = state_norm.transform(
                        np.asarray(
                            group["observations_state"][timestep + offset, env_indices],
                            dtype=np.float32,
                        )
                    )
                    previous = action_norm.transform(
                        np.asarray(
                            group["previous_executed_actions"][timestep + offset, env_indices],
                            dtype=np.float32,
                        )
                    )
                    remaining = np.full(
                        (len(env_indices), 1),
                        max(horizon_steps - offset, 1) / horizon_steps,
                        dtype=np.float32,
                    )
                    condition = np.concatenate(
                        [states, goal_norm, previous, remaining],
                        axis=-1,
                    ).astype(np.float32)
                    if target_mode == "replay":
                        target = np.asarray(
                            group["executed_actions"][timestep + offset, env_indices],
                            dtype=np.float32,
                        )
                    elif target_mode == "base":
                        with torch.inference_mode():
                            normalized_action = _predict_loaded_payload(
                                goal_model,
                                payload["goal"],
                                condition,
                                device,
                            )
                        target = action_norm.inverse(normalized_action).astype(np.float32)
                    else:
                        raise ValueError(f"Unknown distillation target mode: {target_mode}")
                    conditions.append(condition)
                    targets.append(target)
                    sample_weights.append(
                        np.full((len(env_indices),), float(weight), dtype=np.float32)
                    )
        count_after = sum(len(chunk) for chunk in targets)
        target_sample_counts[target_mode] = target_sample_counts.get(target_mode, 0) + (
            count_after - count_before
        )
        return local_dataset_path

    append_manifest_rows(manifest_path, target_mode="replay", weight=replay_weight)
    preserve_dataset_path = None
    if preserve_manifest_path is not None:
        preserve_dataset_path = append_manifest_rows(
            preserve_manifest_path,
            target_mode="base",
            weight=preserve_weight,
        )
    def append_npz_rows(path: Path, *, target_name: str, weight: float) -> None:
        with np.load(path) as target_npz:
            npz_conditions = np.asarray(target_npz["conditions"], dtype=np.float32)
            npz_actions = np.asarray(target_npz["actions"], dtype=np.float32)
            npz_sample_weights = (
                np.asarray(target_npz["sample_weights"], dtype=np.float32)
                if "sample_weights" in target_npz
                else None
            )
        if npz_conditions.ndim != 2 or npz_conditions.shape[-1] != condition_dim:
            raise ValueError(
                f"Unexpected {target_name} NPZ condition shape {npz_conditions.shape}; "
                f"expected (*, {condition_dim})"
            )
        if npz_actions.ndim != 2 or npz_actions.shape[-1] != action_dim:
            raise ValueError(
                f"Unexpected {target_name} NPZ action shape {npz_actions.shape}; "
                f"expected (*, {action_dim})"
            )
        if npz_sample_weights is None:
            row_weights = np.full((len(npz_conditions),), float(weight), dtype=np.float32)
        else:
            if npz_sample_weights.ndim != 1 or len(npz_sample_weights) != len(npz_conditions):
                raise ValueError(
                    f"Unexpected {target_name} NPZ sample_weights shape "
                    f"{npz_sample_weights.shape}; expected ({len(npz_conditions)},)"
                )
            if np.any(npz_sample_weights <= 0.0):
                raise ValueError(f"{target_name} NPZ sample_weights must be positive")
            row_weights = (float(weight) * npz_sample_weights).astype(np.float32)
        conditions.append(npz_conditions)
        targets.append(npz_actions)
        sample_weights.append(row_weights)
        target_sample_counts[target_name] = target_sample_counts.get(target_name, 0) + int(
            npz_conditions.shape[0]
        )

    if preserve_npz_path is not None:
        append_npz_rows(preserve_npz_path, target_name="base_npz", weight=preserve_npz_weight)
    if improve_npz_path is not None:
        append_npz_rows(improve_npz_path, target_name="improve_npz", weight=improve_npz_weight)

    train_x = np.concatenate(conditions, axis=0).astype(np.float32)
    train_y = np.concatenate(targets, axis=0).astype(np.float32)
    train_w = np.concatenate(sample_weights, axis=0).astype(np.float32)
    if train_x.shape[-1] != condition_dim:
        raise ValueError(f"Unexpected condition dim {train_x.shape[-1]} != {condition_dim}")
    if train_y.shape[-1] != action_dim:
        raise ValueError(f"Unexpected action dim {train_y.shape[-1]} != {action_dim}")
    if len(train_w) != len(train_x):
        raise ValueError(f"Unexpected weight count {len(train_w)} != {len(train_x)}")

    agent = PrivilegedZDirectActorCritic(
        goal_model,
        payload["goal"],
        action_norm.mean,
        action_norm.std,
        condition_dim,
        action_dim,
        train_scope=train_scope,
        width=int(config.get("low_level_rl.residual_width", 256)),
        depth=int(config.get("low_level_rl.residual_depth", 2)),
        initial_logstd=initial_logstd,
    ).to(device)
    trainable = [parameter for parameter in agent.parameters() if parameter.requires_grad]
    optimizer = torch.optim.Adam(trainable, lr=learning_rate)
    generator = torch.Generator()
    generator.manual_seed(seed + 8_300_100)
    loader = DataLoader(
        TensorDataset(
            torch.from_numpy(train_x),
            torch.from_numpy(train_y),
            torch.from_numpy(train_w),
        ),
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
    )

    history: list[dict[str, Any]] = []
    for epoch in trange(1, epochs + 1, desc=run_tag):
        losses: list[float] = []
        delta_l2: list[float] = []
        agent.train()
        for condition_cpu, target_cpu, weight_cpu in loader:
            condition = condition_cpu.to(device).float()
            target = target_cpu.to(device).float()
            weight = weight_cpu.to(device).float()
            pred = agent.mean_action(condition)
            per_sample_mse = torch.mean((pred - target) ** 2, dim=-1)
            loss = torch.sum(weight * per_sample_mse) / torch.sum(weight)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            delta_l2.extend(torch.linalg.vector_norm(pred.detach() - target, dim=-1).cpu().tolist())
        row = {
            "epoch": epoch,
            "train_mse": float(np.mean(losses)),
            "train_action_l2": float(np.mean(delta_l2)),
        }
        history.append(row)
        if epoch == 1 or epoch == epochs or epoch % max(epochs // 10, 1) == 0:
            write_json(history_path, {"recipe": {}, "history": history})

    recipe = {
        "method": "privileged_z_direct_distill",
        "base_checkpoint": str(checkpoint_path.resolve()),
        "manifest": str(manifest_path.resolve()),
        "preserve_manifest": str(preserve_manifest_path.resolve())
        if preserve_manifest_path is not None
        else None,
        "preserve_npz": str(preserve_npz_path.resolve())
        if preserve_npz_path is not None
        else None,
        "improve_npz": str(improve_npz_path.resolve())
        if improve_npz_path is not None
        else None,
        "dataset": str(dataset_path.resolve()),
        "preserve_dataset": str(preserve_dataset_path.resolve())
        if preserve_dataset_path is not None
        else None,
        "run_tag": run_tag,
        "seed": seed,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "train_scope": train_scope,
        "condition_dim": condition_dim,
        "action_dim": action_dim,
        "actor_critic_width": int(config.get("low_level_rl.residual_width", 256)),
        "actor_critic_depth": int(config.get("low_level_rl.residual_depth", 2)),
        "initial_logstd": initial_logstd,
        "target": "weighted_replay_actions_plus_optional_base_preservation",
        "replay_weight": replay_weight,
        "preserve_weight": preserve_weight,
        "preserve_npz_weight": preserve_npz_weight,
        "improve_npz_weight": improve_npz_weight,
        "num_train_samples": int(len(train_x)),
        "target_sample_counts": target_sample_counts,
        "horizon_steps": horizon_steps,
    }
    checkpoint_payload = {
        "agent": agent.state_dict(),
        "optimizer": optimizer.state_dict(),
        "global_step": int(epochs * len(train_x)),
        "condition_dim": condition_dim,
        "action_dim": action_dim,
        "recipe": recipe,
        "history": history,
    }
    torch.save(checkpoint_payload, checkpoint_out)
    write_json(history_path, {"recipe": recipe, "history": history})
    console.print(f"Wrote privileged-z replay distillation checkpoint: {checkpoint_out}")
    return checkpoint_out


def _goal_sensitivity(
    goal_payload: dict[str, Any],
    validation: list[dict[str, np.ndarray]],
    state_norm: Standardizer,
    action_norm: Standardizer,
    horizons: tuple[int, ...],
    samples: int,
    seed: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed + 6_000_000)
    max_horizon = max(horizons)
    candidates: list[tuple[int, int]] = []
    for episode_index, episode in enumerate(validation):
        for t in range(len(episode["actions"]) - max_horizon):
            candidates.append((episode_index, t))
    if not candidates:
        raise ValueError("No validation samples for privileged-z goal sensitivity")
    indices = rng.choice(len(candidates), size=min(samples, len(candidates)), replace=False)
    device = default_device()
    actions_by_horizon: dict[int, np.ndarray] = {}
    for horizon in horizons:
        rows = []
        for choice in indices:
            episode_index, t = candidates[int(choice)]
            episode = validation[episode_index]
            states = state_norm.transform(episode["states"])
            previous = action_norm.transform(episode["previous_actions"][t : t + 1])[0]
            rows.append(
                np.concatenate(
                    [
                        states[t],
                        states[t + horizon],
                        previous,
                        np.asarray([1.0], dtype=np.float32),
                    ],
                    axis=-1,
                )
            )
        normalized_action = _predict(goal_payload, np.stack(rows).astype(np.float32), device)
        actions_by_horizon[horizon] = action_norm.inverse(normalized_action)
    reference = horizons[len(horizons) // 2]
    pairs = {}
    for horizon in horizons:
        if horizon == reference:
            continue
        l2 = np.linalg.norm(actions_by_horizon[horizon] - actions_by_horizon[reference], axis=-1)
        pairs[f"k{horizon}_vs_k{reference}"] = {
            "action_l2_mean": float(np.mean(l2)),
            "action_l2_median": float(np.median(l2)),
            "action_l2_p90": float(np.quantile(l2, 0.9)),
        }
    near_far = (horizons[0], horizons[-1])
    l2 = np.linalg.norm(actions_by_horizon[near_far[0]] - actions_by_horizon[near_far[1]], axis=-1)
    pairs[f"k{near_far[0]}_vs_k{near_far[1]}"] = {
        "action_l2_mean": float(np.mean(l2)),
        "action_l2_median": float(np.median(l2)),
        "action_l2_p90": float(np.quantile(l2, 0.9)),
    }
    return {"samples": int(len(indices)), "reference_horizon": reference, "pairs": pairs}


def train_privileged_z_hierarchy(
    config: Config,
    dataset_path: Path | None = None,
    n_trajectories: int = 500,
    validation_trajectories: int = 200,
    horizon_steps: int = 10,
    seed: int = 0,
    epochs: int = 40,
    batch_size: int = 4096,
    hidden_dim: int = 512,
    lr: float = 3e-4,
    model_family: str = "mlp",
    flow_steps: int = 24,
    selection_mode: str = "any_success",
    train_per_expert: int | None = None,
    validation_per_expert: int | None = None,
    run_tag: str | None = None,
    force: bool = False,
) -> Path:
    path = dataset_path or _default_dataset(config)
    artifact = _artifact_dir(config, n_trajectories, seed, run_tag)
    checkpoint_path = artifact / f"privileged_z_k{horizon_steps}.pt"
    metrics_path = artifact / f"privileged_z_k{horizon_steps}_metrics.json"
    if checkpoint_path.exists() and not force:
        console.print(f"Privileged-z hierarchy exists: {checkpoint_path}")
        return checkpoint_path
    if model_family not in {"mlp", "flow"}:
        raise ValueError(f"Unknown privileged-z model family: {model_family}")
    timer = Timer()
    train, validation, data = _load_episodes(
        path,
        n_trajectories,
        validation_trajectories,
        seed,
        selection_mode,
        train_per_expert,
        validation_per_expert,
    )
    all_train_states = np.concatenate([episode["states"] for episode in train], axis=0)
    all_train_actions = np.concatenate([episode["actions"] for episode in train], axis=0)
    state_norm = Standardizer.fit(all_train_states)
    action_norm = Standardizer.fit(all_train_actions)

    high_train_x, high_train_y = _privileged_z_samples(
        train, state_norm, action_norm, horizon_steps, include_goal=True, for_high=True
    )
    high_val_x, high_val_y = _privileged_z_samples(
        validation, state_norm, action_norm, horizon_steps, include_goal=True, for_high=True
    )
    flat_train_x, flat_train_y = _privileged_z_samples(
        train, state_norm, action_norm, horizon_steps, include_goal=False, for_high=False
    )
    flat_val_x, flat_val_y = _privileged_z_samples(
        validation, state_norm, action_norm, horizon_steps, include_goal=False, for_high=False
    )
    goal_train_x, goal_train_y = _privileged_z_samples(
        train, state_norm, action_norm, horizon_steps, include_goal=True, for_high=False
    )
    goal_val_x, goal_val_y = _privileged_z_samples(
        validation, state_norm, action_norm, horizon_steps, include_goal=True, for_high=False
    )
    if model_family == "flow":
        high_payload, high_metrics = _train_flow(
            "high",
            high_train_x,
            high_train_y,
            high_val_x,
            high_val_y,
            hidden_dim=hidden_dim,
            batch_size=batch_size,
            epochs=epochs,
            lr=lr,
            flow_steps=flow_steps,
            seed=seed + 10,
        )
        flat_payload, flat_metrics = _train_flow(
            "flat",
            flat_train_x,
            flat_train_y,
            flat_val_x,
            flat_val_y,
            hidden_dim=hidden_dim,
            batch_size=batch_size,
            epochs=epochs,
            lr=lr,
            flow_steps=flow_steps,
            seed=seed + 20,
        )
        goal_payload, goal_metrics = _train_flow(
            "goal",
            goal_train_x,
            goal_train_y,
            goal_val_x,
            goal_val_y,
            hidden_dim=hidden_dim,
            batch_size=batch_size,
            epochs=epochs,
            lr=lr,
            flow_steps=flow_steps,
            seed=seed + 30,
        )
    else:
        high_payload, high_metrics = _train_mlp(
            "high",
            high_train_x,
            high_train_y,
            high_val_x,
            high_val_y,
            hidden_dim=hidden_dim,
            depth=4,
            batch_size=batch_size,
            epochs=epochs,
            lr=lr,
            seed=seed + 10,
        )
        flat_payload, flat_metrics = _train_mlp(
            "flat",
            flat_train_x,
            flat_train_y,
            flat_val_x,
            flat_val_y,
            hidden_dim=hidden_dim,
            depth=4,
            batch_size=batch_size,
            epochs=epochs,
            lr=lr,
            seed=seed + 20,
        )
        goal_payload, goal_metrics = _train_mlp(
            "goal",
            goal_train_x,
            goal_train_y,
            goal_val_x,
            goal_val_y,
            hidden_dim=hidden_dim,
            depth=4,
            batch_size=batch_size,
            epochs=epochs,
            lr=lr,
            seed=seed + 30,
        )
    sensitivity = _goal_sensitivity(
        goal_payload,
        validation,
        state_norm,
        action_norm,
        horizons=(2, 5, horizon_steps, 20),
        samples=4096,
        seed=seed,
    )
    payload = {
        "method": "privileged_z_hierarchy",
        "model_family": model_family,
        "dataset": str(path),
        "n_trajectories": n_trajectories,
        "validation_trajectories": validation_trajectories,
        "horizon_steps": horizon_steps,
        "seed": seed,
        "run_tag": run_tag,
        "selection_mode": selection_mode,
        "goal_low_sample_mode": "held_goal_all_remaining_offsets",
        "flow_steps": int(flow_steps) if model_family == "flow" else None,
        "state_dim": int(all_train_states.shape[-1]),
        "action_dim": int(all_train_actions.shape[-1]),
        "state_norm": state_norm.state_dict(),
        "action_norm": action_norm.state_dict(),
        "high": high_payload,
        "flat": flat_payload,
        "goal": goal_payload,
        "metrics": {
            "high": high_metrics,
            "flat": flat_metrics,
            "goal": goal_metrics,
            "goal_sensitivity": sensitivity,
        },
        "data": data,
        "elapsed_s": timer.elapsed(),
    }
    torch.save(payload, checkpoint_path)
    write_json(
        metrics_path,
        {
            "method": payload["method"],
            "checkpoint": str(checkpoint_path),
            "dataset": str(path),
            "n_trajectories": n_trajectories,
            "validation_trajectories": validation_trajectories,
            "horizon_steps": horizon_steps,
            "seed": seed,
            "run_tag": run_tag,
            "selection_mode": selection_mode,
            "goal_low_sample_mode": payload["goal_low_sample_mode"],
            "model_family": payload["model_family"],
            "flow_steps": payload["flow_steps"],
            "metrics": payload["metrics"],
            "data": data,
            "elapsed_s": payload["elapsed_s"],
        },
    )
    console.print(f"Wrote privileged-z hierarchy: {checkpoint_path}")
    return checkpoint_path


def train_privileged_z_residual_rl(
    config: Config,
    checkpoint_path: Path,
    init_dataset_path: Path,
    *,
    run_tag: str,
    seed: int = 0,
    total_steps: int = 32_768,
    alpha: float = 0.1,
    terminal_weight: float = 1.0,
    residual_penalty_weight: float = 0.01,
    learning_rate: float = 1e-4,
    num_minibatches: int = 8,
    update_epochs: int = 4,
    checkpoint_every_updates: int = 5,
    initial_logstd: float = -2.3,
    residual_action_mode: str = "additive",
    residual_goal_source: str = "oracle",
    reward_mode: str = "progress",
    dense_progress_weight: float = 0.0,
    force: bool = False,
) -> Path:
    from hcl_poc.low_level_rl import ResidualActorCritic
    from hcl_poc.rl_rerun import _make_benchmark_env, _residual_action_from_raw, _to_numpy

    if total_steps <= 0:
        raise ValueError("total_steps must be positive")
    if residual_action_mode not in {"additive", "margin_scaled"}:
        raise ValueError(f"Unknown residual action mode: {residual_action_mode}")
    if residual_goal_source not in {"oracle", "predicted", "oracle_to_predicted"}:
        raise ValueError(f"Unknown residual goal source: {residual_goal_source}")
    if reward_mode not in {"progress", "paired"}:
        raise ValueError(f"Unknown privileged-z residual reward mode: {reward_mode}")
    if dense_progress_weight < 0:
        raise ValueError("dense_progress_weight must be non-negative")
    if not init_dataset_path.exists():
        raise FileNotFoundError(init_dataset_path)
    artifact = ensure_dir(
        config.path_value("paths.incremental_artifact_dir")
        / "privileged_z_residual"
        / run_tag
        / f"seed{seed}"
    )
    result_dir = ensure_dir(
        config.path_value("paths.incremental_results_dir")
        / "privileged_z_residual"
        / run_tag
        / f"seed{seed}"
    )
    checkpoint_out = artifact / "latest.pt"
    checkpoint_dir = ensure_dir(artifact / "checkpoints")
    history_path = result_dir / "history.json"
    if force:
        checkpoint_out.unlink(missing_ok=True)
        history_path.unlink(missing_ok=True)
        for checkpoint_file in checkpoint_dir.glob("step_*.pt"):
            checkpoint_file.unlink()
    device = default_device()
    set_seed(seed + 770_000)
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_norm = Standardizer.from_state_dict(payload["state_norm"])
    action_norm = Standardizer.from_state_dict(payload["action_norm"])
    high_model = _model_from_payload(payload["high"], device)
    goal_model = _model_from_payload(payload["goal"], device)
    horizon_steps = int(payload["horizon_steps"])
    state_dim = int(payload["state_dim"])
    action_dim = int(payload["action_dim"])
    condition_dim = state_dim + state_dim + action_dim + 1

    with h5py.File(init_dataset_path, "r") as h5_meta:
        batch_keys = sorted(key for key in h5_meta.keys() if key.startswith("batch_"))
        if not batch_keys:
            raise ValueError(f"No vector batches in {init_dataset_path}")
        first_states = h5_meta[batch_keys[0]]["observations_state"]
        first_actions = h5_meta[batch_keys[0]]["executed_actions"]
        if first_states.ndim != 3 or first_actions.ndim != 3:
            raise ValueError(f"Unexpected vector batch shapes in {init_dataset_path}")
        meta = h5_meta["meta"].attrs if "meta" in h5_meta else {}
        num_envs = int(meta.get("num_envs", first_states.shape[1]))
        max_steps = int(meta.get("max_steps", first_actions.shape[0]))
        for key in batch_keys:
            batch_num_envs = int(
                h5_meta[key].attrs.get(
                    "num_envs",
                    h5_meta[key]["observations_state"].shape[1],
                )
            )
            batch_max_steps = int(
                h5_meta[key].attrs.get(
                    "max_steps",
                    h5_meta[key]["executed_actions"].shape[0],
                )
            )
            if batch_num_envs != num_envs or batch_max_steps != max_steps:
                raise ValueError("Privileged-z residual init dataset has mixed batch shapes")
    rollout_steps = horizon_steps
    batch_size = num_envs * rollout_steps
    if batch_size % num_minibatches:
        raise ValueError(
            f"batch size {batch_size} must divide num_minibatches {num_minibatches}"
        )
    minibatch_size = batch_size // num_minibatches

    env = _make_benchmark_env(config, num_envs, "state")
    base_env = _make_benchmark_env(config, num_envs, "state") if reward_mode == "paired" else None
    if base_env is not None:
        base_env.reset(seed=0)
    h5 = h5py.File(init_dataset_path, "r")
    action_low = torch.as_tensor(env.single_action_space.low, device=device, dtype=torch.float32)
    action_high = torch.as_tensor(env.single_action_space.high, device=device, dtype=torch.float32)
    agent = ResidualActorCritic(
        condition_dim,
        action_dim=action_dim,
        width=int(config.get("low_level_rl.residual_width", 256)),
        depth=int(config.get("low_level_rl.residual_depth", 2)),
        initial_logstd=initial_logstd,
    ).to(device)
    optimizer = torch.optim.Adam(agent.parameters(), lr=learning_rate, eps=1e-5)
    gamma = float(config.get("low_level_rl.gamma", 0.99))
    gae_lambda = float(config.get("low_level_rl.gae_lambda", 0.95))
    clip_coef = float(config.get("low_level_rl.clip_coef", 0.2))
    ent_coef = float(config.get("low_level_rl.entropy_coef", 0.0))
    value_coef = float(config.get("low_level_rl.value_coef", 1.0))
    max_grad_norm = float(config.get("low_level_rl.max_grad_norm", 1.0))
    recipe = {
        "method": "privileged_z_residual_r1",
        "base_checkpoint": str(checkpoint_path.resolve()),
        "init_dataset": str(init_dataset_path.resolve()),
        "run_tag": run_tag,
        "seed": seed,
        "num_envs": num_envs,
        "rollout_steps": rollout_steps,
        "horizon_steps": horizon_steps,
        "condition_dim": condition_dim,
        "alpha": alpha,
        "terminal_weight": terminal_weight,
        "residual_penalty_weight": residual_penalty_weight,
        "learning_rate": learning_rate,
        "num_minibatches": num_minibatches,
        "minibatch_size": minibatch_size,
        "update_epochs": update_epochs,
        "gamma": gamma,
        "gae_lambda": gae_lambda,
        "clip_coef": clip_coef,
        "entropy_coef": ent_coef,
        "value_coef": value_coef,
        "max_grad_norm": max_grad_norm,
        "actor_critic_width": int(config.get("low_level_rl.residual_width", 256)),
        "actor_critic_depth": int(config.get("low_level_rl.residual_depth", 2)),
        "initial_logstd": initial_logstd,
        "residual_action_mode": residual_action_mode,
        "residual_goal_source": residual_goal_source,
        "reward_mode": reward_mode,
        "dense_progress_weight": dense_progress_weight,
        "reward": (
            "privileged_state_progress_minus_terminal_distance_minus_residual_penalty"
            if reward_mode == "progress"
            else "privileged_state_paired_terminal_improvement_minus_residual_penalty"
        ),
        "goal_source": residual_goal_source,
    }
    history: list[dict[str, Any]] = []
    global_step = 0
    if checkpoint_out.exists() and not force:
        existing = torch.load(checkpoint_out, map_location=device, weights_only=False)
        if existing["recipe"] != recipe:
            raise ValueError(f"Existing residual run {checkpoint_out} has a different recipe")
        agent.load_state_dict(existing["agent"])
        optimizer.load_state_dict(existing["optimizer"])
        global_step = int(existing["global_step"])
        history = list(existing["history"])
    if global_step >= total_steps:
        h5.close()
        env.close()
        return checkpoint_out

    rng = np.random.default_rng(seed + 771_000)
    obs: dict[str, Any]
    current_state_norm: np.ndarray
    goal_state_norm: np.ndarray
    previous_action_norm: np.ndarray
    base_terminal_distance: np.ndarray
    goal_prediction_weight = 0.0
    local_step = 0

    def current_goal_prediction_weight() -> float:
        if residual_goal_source == "oracle":
            return 0.0
        if residual_goal_source == "predicted":
            return 1.0
        progress = min(max(global_step / float(total_steps), 0.0), 1.0)
        if progress < 1.0 / 3.0:
            return 0.0
        if progress < 2.0 / 3.0:
            return (progress - 1.0 / 3.0) * 3.0
        return 1.0

    @torch.inference_mode()
    def rollout_base_terminal_distance() -> np.ndarray:
        if base_env is None:
            raise RuntimeError("paired reward requested without a base environment")
        base_env.unwrapped.set_state_dict(_clone_mani_state_dict(env.unwrapped.get_state_dict()))
        base_obs = base_env.unwrapped.get_obs()
        base_previous = previous_action_norm.copy()
        for step in range(horizon_steps):
            base_state = _obs_state_np(base_obs)
            base_state_norm = state_norm.transform(base_state)
            remaining = np.full(
                (num_envs, 1),
                max(horizon_steps - step, 1) / horizon_steps,
                dtype=np.float32,
            )
            condition_np = np.concatenate(
                [base_state_norm, goal_state_norm, base_previous, remaining],
                axis=-1,
            ).astype(np.float32)
            normalized_action = _predict_loaded_payload(
                goal_model,
                payload["goal"],
                condition_np,
                device,
            )
            base_action = torch.from_numpy(action_norm.inverse(normalized_action)).to(device)
            base_action = torch.clamp(base_action, action_low, action_high)
            base_obs, _reward, _terminated, _truncated, _info = base_env.step(base_action)
            base_previous = action_norm.transform(
                base_action.cpu().numpy().astype(np.float32)
            )
        base_final_state = _obs_state_np(base_obs)
        base_final_norm = state_norm.transform(base_final_state)
        return np.mean((base_final_norm - goal_state_norm) ** 2, axis=-1).astype(np.float32)

    @torch.inference_mode()
    def reset_local_episode() -> None:
        nonlocal obs, current_state_norm, goal_state_norm, previous_action_norm
        nonlocal base_terminal_distance
        nonlocal goal_prediction_weight, local_step
        batch_key = str(rng.choice(batch_keys))
        group = h5[batch_key]
        t = int(rng.integers(0, max_steps - horizon_steps + 1))
        obs, _info = env.reset(seed=int(group.attrs["batch_seed"]))
        for replay_step in range(t):
            replay_action = torch.from_numpy(
                np.asarray(group["executed_actions"][replay_step], dtype=np.float32)
            ).to(device)
            obs, _reward, _terminated, _truncated, _info = env.step(replay_action)
        state_np = np.asarray(group["observations_state"][t], dtype=np.float32)
        live_state = _obs_state_np(obs)
        if state_np.shape == live_state.shape:
            current_state_norm = state_norm.transform(live_state)
        else:
            current_state_norm = state_norm.transform(state_np)
        previous_action_norm = action_norm.transform(
            np.asarray(group["previous_executed_actions"][t], dtype=np.float32)
        )
        oracle_goal_norm = state_norm.transform(
            np.asarray(group["observations_state"][t + horizon_steps], dtype=np.float32)
        )
        high_input = np.concatenate([current_state_norm, previous_action_norm], axis=-1)
        predicted_goal_norm = _predict_loaded_payload(
            high_model,
            payload["high"],
            high_input,
            device,
        )
        goal_prediction_weight = current_goal_prediction_weight()
        goal_state_norm = (
            (1.0 - goal_prediction_weight) * oracle_goal_norm
            + goal_prediction_weight * predicted_goal_norm
        ).astype(np.float32)
        base_terminal_distance = (
            rollout_base_terminal_distance()
            if reward_mode == "paired"
            else np.zeros(num_envs, dtype=np.float32)
        )
        local_step = 0

    @torch.inference_mode()
    def condition_and_base() -> tuple[torch.Tensor, torch.Tensor, np.ndarray]:
        remaining = np.full(
            (num_envs, 1),
            max(horizon_steps - local_step, 1) / horizon_steps,
            dtype=np.float32,
        )
        condition_np = np.concatenate(
            [current_state_norm, goal_state_norm, previous_action_norm, remaining],
            axis=-1,
        ).astype(np.float32)
        normalized_action = _predict_loaded_payload(
            goal_model,
            payload["goal"],
            condition_np,
            device,
        )
        base_action = torch.from_numpy(action_norm.inverse(normalized_action)).to(device)
        distance = np.mean((current_state_norm - goal_state_norm) ** 2, axis=-1).astype(np.float32)
        return torch.from_numpy(condition_np).to(device).float(), base_action, distance

    @torch.inference_mode()
    def step_local(action: torch.Tensor, previous_distance: np.ndarray, residual: torch.Tensor) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        nonlocal obs, current_state_norm, previous_action_norm, local_step
        next_obs, _reward, _terminated, _truncated, info = env.step(action)
        next_state = _obs_state_np(next_obs)
        next_state_norm = state_norm.transform(next_state)
        next_distance = np.mean((next_state_norm - goal_state_norm) ** 2, axis=-1).astype(
            np.float32
        )
        residual_penalty = residual_penalty_weight * residual.square().mean(dim=-1).cpu().numpy()
        dense_progress = previous_distance - next_distance
        if reward_mode == "paired":
            reward = dense_progress_weight * dense_progress - residual_penalty
        else:
            reward = dense_progress - residual_penalty
        segment_end = local_step == horizon_steps - 1
        if segment_end:
            if reward_mode == "paired":
                reward += terminal_weight * (base_terminal_distance - next_distance)
            else:
                reward -= terminal_weight * next_distance
        obs = next_obs
        current_state_norm = next_state_norm
        clipped = torch.clamp(action, action_low, action_high)
        previous_action_norm = action_norm.transform(clipped.cpu().numpy().astype(np.float32))
        local_step += 1
        done = np.full(num_envs, segment_end, dtype=np.bool_)
        metrics = {
            "next_distance": next_distance,
            "residual_norm": torch.linalg.vector_norm(residual, dim=-1).cpu().numpy(),
            "success": _to_numpy(info.get("success", np.zeros(num_envs, dtype=np.bool_)))
            .reshape(-1)
            .astype(np.bool_),
        }
        if segment_end:
            metrics["base_terminal_distance"] = base_terminal_distance
            metrics["paired_improvement"] = base_terminal_distance - next_distance
        if segment_end:
            reset_local_episode()
        return reward.astype(np.float32), done, metrics

    reset_local_episode()
    condition_buf = torch.zeros((rollout_steps, num_envs, condition_dim), device=device)
    raw_action_buf = torch.zeros((rollout_steps, num_envs, action_dim), device=device)
    logprob_buf = torch.zeros((rollout_steps, num_envs), device=device)
    reward_buf = torch.zeros((rollout_steps, num_envs), device=device)
    done_buf = torch.zeros((rollout_steps, num_envs), device=device)
    value_buf = torch.zeros((rollout_steps, num_envs), device=device)
    next_done = torch.zeros(num_envs, device=device)
    update_index = len(history)
    start_time = time.perf_counter()
    try:
        with trange(global_step, total_steps, initial=global_step, total=total_steps, desc=run_tag) as progress:
            while global_step < total_steps:
                distances: list[float] = []
                terminal_distances: list[float] = []
                rewards: list[float] = []
                residual_norms: list[float] = []
                base_terminal_distances: list[float] = []
                paired_improvements: list[float] = []
                successes = 0
                agent.eval()
                for step in range(rollout_steps):
                    condition, base_action, distance = condition_and_base()
                    condition_buf[step] = condition
                    done_buf[step] = next_done
                    with torch.no_grad():
                        raw_action, logprob, _entropy, value = agent.get_action_and_value(condition)
                    residual, _unclipped, action = _residual_action_from_raw(
                        base_action,
                        raw_action,
                        alpha,
                        action_low,
                        action_high,
                        residual_action_mode,
                    )
                    raw_action_buf[step] = raw_action
                    logprob_buf[step] = logprob
                    value_buf[step] = value
                    reward, done, metrics = step_local(action, distance, residual)
                    reward_buf[step] = torch.from_numpy(reward).to(device)
                    next_done = torch.from_numpy(done.astype(np.float32)).to(device)
                    distances.extend(distance.tolist())
                    if step == rollout_steps - 1:
                        terminal_distances.extend(metrics["next_distance"].tolist())
                        if "base_terminal_distance" in metrics:
                            base_terminal_distances.extend(
                                metrics["base_terminal_distance"].tolist()
                            )
                        if "paired_improvement" in metrics:
                            paired_improvements.extend(metrics["paired_improvement"].tolist())
                    rewards.extend(reward.tolist())
                    residual_norms.extend(metrics["residual_norm"].tolist())
                    successes += int(np.sum(metrics["success"]))
                    global_step += num_envs
                    progress.update(num_envs)
                with torch.no_grad():
                    next_value = torch.zeros(num_envs, device=device)
                    advantages = torch.zeros_like(reward_buf, device=device)
                    lastgaelam = torch.zeros(num_envs, device=device)
                    for t in reversed(range(rollout_steps)):
                        if t == rollout_steps - 1:
                            nextnonterminal = 1.0 - next_done
                            nextvalues = next_value
                        else:
                            nextnonterminal = 1.0 - done_buf[t + 1]
                            nextvalues = value_buf[t + 1]
                        delta = reward_buf[t] + gamma * nextvalues * nextnonterminal - value_buf[t]
                        lastgaelam = delta + gamma * gae_lambda * nextnonterminal * lastgaelam
                        advantages[t] = lastgaelam
                    returns = advantages + value_buf
                b_conditions = condition_buf.reshape((-1, condition_dim))
                b_raw_actions = raw_action_buf.reshape((-1, action_dim))
                b_logprobs = logprob_buf.reshape(-1)
                b_advantages = advantages.reshape(-1)
                b_returns = returns.reshape(-1)
                b_advantages = (b_advantages - b_advantages.mean()) / (
                    b_advantages.std() + 1e-8
                )
                agent.train()
                batch_indices = np.arange(batch_size)
                policy_losses: list[float] = []
                value_losses: list[float] = []
                entropy_values: list[float] = []
                clipfracs: list[float] = []
                for _epoch in range(update_epochs):
                    rng.shuffle(batch_indices)
                    for start in range(0, batch_size, minibatch_size):
                        mb = batch_indices[start : start + minibatch_size]
                        _action, newlogprob, entropy, newvalue = agent.get_action_and_value(
                            b_conditions[mb],
                            raw_action=b_raw_actions[mb],
                        )
                        logratio = newlogprob - b_logprobs[mb]
                        ratio = logratio.exp()
                        with torch.no_grad():
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
                        v_loss = 0.5 * ((newvalue - b_returns[mb]) ** 2).mean()
                        entropy_loss = entropy.mean()
                        loss = pg_loss - ent_coef * entropy_loss + value_coef * v_loss
                        optimizer.zero_grad(set_to_none=True)
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(agent.parameters(), max_grad_norm)
                        optimizer.step()
                        policy_losses.append(float(pg_loss.detach().cpu()))
                        value_losses.append(float(v_loss.detach().cpu()))
                        entropy_values.append(float(entropy_loss.detach().cpu()))
                update_index += 1
                row = {
                    "update": update_index,
                    "global_step": int(global_step),
                    "mean_reward": float(np.mean(rewards)),
                    "mean_initial_distance": float(np.mean(distances)),
                    "mean_terminal_distance": float(np.mean(terminal_distances)),
                    "mean_base_terminal_distance": float(np.mean(base_terminal_distances))
                    if base_terminal_distances
                    else None,
                    "mean_paired_improvement": float(np.mean(paired_improvements))
                    if paired_improvements
                    else None,
                    "fraction_improved": float(np.mean(np.asarray(paired_improvements) > 0.0))
                    if paired_improvements
                    else None,
                    "mean_residual_norm": float(np.mean(residual_norms)),
                    "goal_prediction_weight": float(goal_prediction_weight),
                    "policy_loss": float(np.mean(policy_losses)),
                    "value_loss": float(np.mean(value_losses)),
                    "entropy": float(np.mean(entropy_values)),
                    "clipfrac": float(np.mean(clipfracs)),
                    "success_fraction_seen": successes / float(batch_size),
                    "elapsed_s": time.perf_counter() - start_time,
                }
                history.append(row)
                write_json(history_path, {"recipe": recipe, "history": history})
                if update_index % checkpoint_every_updates == 0 or global_step >= total_steps:
                    checkpoint_payload = {
                        "agent": agent.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "global_step": int(global_step),
                        "condition_dim": condition_dim,
                        "action_dim": action_dim,
                        "recipe": recipe,
                        "history": history,
                    }
                    torch.save(checkpoint_payload, checkpoint_out)
                    torch.save(
                        checkpoint_payload,
                        checkpoint_dir / f"step_{global_step:09d}.pt",
                    )
    finally:
        h5.close()
        env.close()
        if base_env is not None:
            base_env.close()
    return checkpoint_out


def train_privileged_z_direct_rl(
    config: Config,
    checkpoint_path: Path,
    init_dataset_path: Path,
    *,
    run_tag: str,
    direct_init_checkpoint_path: Path | None = None,
    seed: int = 0,
    total_steps: int = 32_768,
    terminal_weight: float = 1.0,
    learning_rate: float = 3e-5,
    num_minibatches: int = 8,
    update_epochs: int = 4,
    checkpoint_every_updates: int = 5,
    initial_logstd: float = -4.0,
    train_scope: str = "final_layer",
    goal_source: str = "oracle",
    reward_mode: str = "paired",
    dense_progress_weight: float = 0.0,
    bc_weight: float = 0.0,
    min_base_terminal_mse: float | None = None,
    force: bool = False,
) -> Path:
    from hcl_poc.rl_rerun import _make_benchmark_env, _to_numpy

    if total_steps <= 0:
        raise ValueError("total_steps must be positive")
    if checkpoint_every_updates <= 0:
        raise ValueError("checkpoint_every_updates must be positive")
    if train_scope not in {"final_layer", "all"}:
        raise ValueError(f"Unknown privileged-z direct train scope: {train_scope}")
    if goal_source not in {"oracle", "predicted", "oracle_to_predicted"}:
        raise ValueError(f"Unknown privileged-z direct goal source: {goal_source}")
    if reward_mode not in {"progress", "paired"}:
        raise ValueError(f"Unknown privileged-z direct reward mode: {reward_mode}")
    if dense_progress_weight < 0:
        raise ValueError("dense_progress_weight must be non-negative")
    if bc_weight < 0:
        raise ValueError("bc_weight must be non-negative")
    if min_base_terminal_mse is not None and min_base_terminal_mse < 0.0:
        raise ValueError("min_base_terminal_mse must be non-negative")
    if min_base_terminal_mse is not None and reward_mode != "paired":
        raise ValueError("min_base_terminal_mse requires paired reward mode")
    if not init_dataset_path.exists():
        raise FileNotFoundError(init_dataset_path)
    if direct_init_checkpoint_path is not None and not direct_init_checkpoint_path.exists():
        raise FileNotFoundError(direct_init_checkpoint_path)

    artifact = ensure_dir(
        config.path_value("paths.incremental_artifact_dir")
        / "privileged_z_direct"
        / run_tag
        / f"seed{seed}"
    )
    result_dir = ensure_dir(
        config.path_value("paths.incremental_results_dir")
        / "privileged_z_direct"
        / run_tag
        / f"seed{seed}"
    )
    checkpoint_out = artifact / "latest.pt"
    checkpoint_dir = ensure_dir(artifact / "checkpoints")
    history_path = result_dir / "history.json"
    if force:
        checkpoint_out.unlink(missing_ok=True)
        history_path.unlink(missing_ok=True)
        for checkpoint_file in checkpoint_dir.glob("step_*.pt"):
            checkpoint_file.unlink()

    device = default_device()
    set_seed(seed + 780_000)
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if payload["goal"].get("model_type") == "flow":
        raise ValueError("Direct privileged-z PPO currently requires an MLP low policy")
    state_norm = Standardizer.from_state_dict(payload["state_norm"])
    action_norm = Standardizer.from_state_dict(payload["action_norm"])
    high_model = _model_from_payload(payload["high"], device)
    goal_model = _model_from_payload(payload["goal"], device)
    horizon_steps = int(payload["horizon_steps"])
    state_dim = int(payload["state_dim"])
    action_dim = int(payload["action_dim"])
    condition_dim = state_dim + state_dim + action_dim + 1

    with h5py.File(init_dataset_path, "r") as h5_meta:
        batch_keys = sorted(key for key in h5_meta.keys() if key.startswith("batch_"))
        if not batch_keys:
            raise ValueError(f"No vector batches in {init_dataset_path}")
        first_states = h5_meta[batch_keys[0]]["observations_state"]
        first_actions = h5_meta[batch_keys[0]]["executed_actions"]
        if first_states.ndim != 3 or first_actions.ndim != 3:
            raise ValueError(f"Unexpected vector batch shapes in {init_dataset_path}")
        meta = h5_meta["meta"].attrs if "meta" in h5_meta else {}
        num_envs = int(meta.get("num_envs", first_states.shape[1]))
        max_steps = int(meta.get("max_steps", first_actions.shape[0]))
        for key in batch_keys:
            batch_num_envs = int(
                h5_meta[key].attrs.get(
                    "num_envs",
                    h5_meta[key]["observations_state"].shape[1],
                )
            )
            batch_max_steps = int(
                h5_meta[key].attrs.get(
                    "max_steps",
                    h5_meta[key]["executed_actions"].shape[0],
                )
            )
            if batch_num_envs != num_envs or batch_max_steps != max_steps:
                raise ValueError("Privileged-z direct init dataset has mixed batch shapes")

    rollout_steps = horizon_steps
    batch_size = num_envs * rollout_steps
    if batch_size % num_minibatches:
        raise ValueError(
            f"batch size {batch_size} must divide num_minibatches {num_minibatches}"
        )
    minibatch_size = batch_size // num_minibatches

    env = _make_benchmark_env(config, num_envs, "state")
    base_env = _make_benchmark_env(config, num_envs, "state") if reward_mode == "paired" else None
    if base_env is not None:
        base_env.reset(seed=0)
    h5 = h5py.File(init_dataset_path, "r")
    action_low = torch.as_tensor(env.single_action_space.low, device=device, dtype=torch.float32)
    action_high = torch.as_tensor(env.single_action_space.high, device=device, dtype=torch.float32)
    agent = PrivilegedZDirectActorCritic(
        goal_model,
        payload["goal"],
        action_norm.mean,
        action_norm.std,
        condition_dim,
        action_dim,
        train_scope=train_scope,
        width=int(config.get("low_level_rl.residual_width", 256)),
        depth=int(config.get("low_level_rl.residual_depth", 2)),
        initial_logstd=initial_logstd,
    ).to(device)
    if direct_init_checkpoint_path is not None:
        direct_init_payload = torch.load(
            direct_init_checkpoint_path,
            map_location=device,
            weights_only=False,
        )
        direct_init_recipe = dict(direct_init_payload["recipe"])
        if Path(direct_init_recipe["base_checkpoint"]).resolve() != checkpoint_path.resolve():
            raise ValueError(
                "Direct initialization checkpoint was trained against a different base checkpoint"
            )
        if int(direct_init_payload["condition_dim"]) != condition_dim:
            raise ValueError("Direct initialization checkpoint has a different condition_dim")
        if int(direct_init_payload["action_dim"]) != action_dim:
            raise ValueError("Direct initialization checkpoint has a different action_dim")
        agent.load_state_dict(direct_init_payload["agent"])
    trainable = [parameter for parameter in agent.parameters() if parameter.requires_grad]
    optimizer = torch.optim.Adam(trainable, lr=learning_rate, eps=1e-5)
    gamma = float(config.get("low_level_rl.gamma", 0.99))
    gae_lambda = float(config.get("low_level_rl.gae_lambda", 0.95))
    clip_coef = float(config.get("low_level_rl.clip_coef", 0.2))
    ent_coef = float(config.get("low_level_rl.entropy_coef", 0.0))
    value_coef = float(config.get("low_level_rl.value_coef", 1.0))
    max_grad_norm = float(config.get("low_level_rl.max_grad_norm", 1.0))
    recipe = {
        "method": "privileged_z_direct_r3",
        "base_checkpoint": str(checkpoint_path.resolve()),
        "init_dataset": str(init_dataset_path.resolve()),
        "direct_init_checkpoint": str(direct_init_checkpoint_path.resolve())
        if direct_init_checkpoint_path is not None
        else None,
        "run_tag": run_tag,
        "seed": seed,
        "num_envs": num_envs,
        "rollout_steps": rollout_steps,
        "horizon_steps": horizon_steps,
        "condition_dim": condition_dim,
        "terminal_weight": terminal_weight,
        "learning_rate": learning_rate,
        "num_minibatches": num_minibatches,
        "minibatch_size": minibatch_size,
        "update_epochs": update_epochs,
        "gamma": gamma,
        "gae_lambda": gae_lambda,
        "clip_coef": clip_coef,
        "entropy_coef": ent_coef,
        "value_coef": value_coef,
        "max_grad_norm": max_grad_norm,
        "actor_critic_width": int(config.get("low_level_rl.residual_width", 256)),
        "actor_critic_depth": int(config.get("low_level_rl.residual_depth", 2)),
        "initial_logstd": initial_logstd,
        "train_scope": train_scope,
        "goal_source": goal_source,
        "reward_mode": reward_mode,
        "dense_progress_weight": dense_progress_weight,
        "bc_weight": bc_weight,
        "min_base_terminal_mse": min_base_terminal_mse,
        "reward": (
            "privileged_state_progress_minus_terminal_distance"
            if reward_mode == "progress"
            else "privileged_state_paired_terminal_improvement"
        ),
    }

    history: list[dict[str, Any]] = []
    global_step = 0
    if checkpoint_out.exists() and not force:
        existing = torch.load(checkpoint_out, map_location=device, weights_only=False)
        if existing["recipe"] != recipe:
            raise ValueError(f"Existing direct run {checkpoint_out} has a different recipe")
        agent.load_state_dict(existing["agent"])
        optimizer.load_state_dict(existing["optimizer"])
        global_step = int(existing["global_step"])
        history = list(existing["history"])
    if global_step >= total_steps:
        h5.close()
        env.close()
        if base_env is not None:
            base_env.close()
        return checkpoint_out

    rng = np.random.default_rng(seed + 781_000)
    obs: dict[str, Any]
    current_state_norm: np.ndarray
    goal_state_norm: np.ndarray
    previous_action_norm: np.ndarray
    base_terminal_distance: np.ndarray
    active_mask: np.ndarray
    goal_prediction_weight = 0.0
    local_step = 0

    def current_goal_prediction_weight() -> float:
        if goal_source == "oracle":
            return 0.0
        if goal_source == "predicted":
            return 1.0
        progress = min(max(global_step / float(total_steps), 0.0), 1.0)
        if progress < 1.0 / 3.0:
            return 0.0
        if progress < 2.0 / 3.0:
            return (progress - 1.0 / 3.0) * 3.0
        return 1.0

    @torch.inference_mode()
    def rollout_base_terminal_distance() -> np.ndarray:
        if base_env is None:
            raise RuntimeError("paired reward requested without a base environment")
        base_env.unwrapped.set_state_dict(_clone_mani_state_dict(env.unwrapped.get_state_dict()))
        base_obs = base_env.unwrapped.get_obs()
        base_previous = previous_action_norm.copy()
        for step in range(horizon_steps):
            base_state = _obs_state_np(base_obs)
            base_state_norm = state_norm.transform(base_state)
            remaining = np.full(
                (num_envs, 1),
                max(horizon_steps - step, 1) / horizon_steps,
                dtype=np.float32,
            )
            condition_np = np.concatenate(
                [base_state_norm, goal_state_norm, base_previous, remaining],
                axis=-1,
            ).astype(np.float32)
            normalized_action = _predict_loaded_payload(
                goal_model,
                payload["goal"],
                condition_np,
                device,
            )
            base_action = torch.from_numpy(action_norm.inverse(normalized_action)).to(device)
            base_action = torch.clamp(base_action, action_low, action_high)
            base_obs, _reward, _terminated, _truncated, _info = base_env.step(base_action)
            base_previous = action_norm.transform(
                base_action.cpu().numpy().astype(np.float32)
            )
        base_final_state = _obs_state_np(base_obs)
        base_final_norm = state_norm.transform(base_final_state)
        return np.mean((base_final_norm - goal_state_norm) ** 2, axis=-1).astype(np.float32)

    @torch.inference_mode()
    def reset_local_episode() -> None:
        nonlocal obs, current_state_norm, goal_state_norm, previous_action_norm
        nonlocal base_terminal_distance, active_mask, goal_prediction_weight, local_step
        batch_key = str(rng.choice(batch_keys))
        group = h5[batch_key]
        t = int(rng.integers(0, max_steps - horizon_steps + 1))
        obs, _info = env.reset(seed=int(group.attrs["batch_seed"]))
        for replay_step in range(t):
            replay_action = torch.from_numpy(
                np.asarray(group["executed_actions"][replay_step], dtype=np.float32)
            ).to(device)
            obs, _reward, _terminated, _truncated, _info = env.step(replay_action)
        state_np = np.asarray(group["observations_state"][t], dtype=np.float32)
        live_state = _obs_state_np(obs)
        if state_np.shape == live_state.shape:
            current_state_norm = state_norm.transform(live_state)
        else:
            current_state_norm = state_norm.transform(state_np)
        previous_action_norm = action_norm.transform(
            np.asarray(group["previous_executed_actions"][t], dtype=np.float32)
        )
        oracle_goal_norm = state_norm.transform(
            np.asarray(group["observations_state"][t + horizon_steps], dtype=np.float32)
        )
        high_input = np.concatenate([current_state_norm, previous_action_norm], axis=-1)
        predicted_goal_norm = _predict_loaded_payload(
            high_model,
            payload["high"],
            high_input,
            device,
        )
        goal_prediction_weight = current_goal_prediction_weight()
        goal_state_norm = (
            (1.0 - goal_prediction_weight) * oracle_goal_norm
            + goal_prediction_weight * predicted_goal_norm
        ).astype(np.float32)
        base_terminal_distance = (
            rollout_base_terminal_distance()
            if reward_mode == "paired"
            else np.zeros(num_envs, dtype=np.float32)
        )
        if min_base_terminal_mse is None:
            active_mask = np.ones(num_envs, dtype=np.bool_)
        else:
            active_mask = base_terminal_distance >= float(min_base_terminal_mse)
            if not np.any(active_mask):
                raise RuntimeError(
                    "No active local starts met min_base_terminal_mse in sampled vector batch"
                )
        local_step = 0

    @torch.inference_mode()
    def condition_and_base() -> tuple[torch.Tensor, torch.Tensor, np.ndarray]:
        remaining = np.full(
            (num_envs, 1),
            max(horizon_steps - local_step, 1) / horizon_steps,
            dtype=np.float32,
        )
        condition_np = np.concatenate(
            [current_state_norm, goal_state_norm, previous_action_norm, remaining],
            axis=-1,
        ).astype(np.float32)
        normalized_action = _predict_loaded_payload(
            goal_model,
            payload["goal"],
            condition_np,
            device,
        )
        base_action = torch.from_numpy(action_norm.inverse(normalized_action)).to(device)
        distance = np.mean((current_state_norm - goal_state_norm) ** 2, axis=-1).astype(np.float32)
        return torch.from_numpy(condition_np).to(device).float(), base_action, distance

    @torch.inference_mode()
    def step_local(action: torch.Tensor, previous_distance: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        nonlocal obs, current_state_norm, previous_action_norm, local_step
        segment_active_mask = active_mask.copy()
        next_obs, _reward, _terminated, _truncated, info = env.step(action)
        next_state = _obs_state_np(next_obs)
        next_state_norm = state_norm.transform(next_state)
        next_distance = np.mean((next_state_norm - goal_state_norm) ** 2, axis=-1).astype(
            np.float32
        )
        dense_progress = previous_distance - next_distance
        if reward_mode == "paired":
            reward = dense_progress_weight * dense_progress
        else:
            reward = dense_progress
        segment_end = local_step == horizon_steps - 1
        if segment_end:
            if reward_mode == "paired":
                reward += terminal_weight * (base_terminal_distance - next_distance)
            else:
                reward -= terminal_weight * next_distance
        if min_base_terminal_mse is not None:
            reward = np.where(segment_active_mask, reward, 0.0)
        obs = next_obs
        current_state_norm = next_state_norm
        clipped = torch.clamp(action, action_low, action_high)
        previous_action_norm = action_norm.transform(clipped.cpu().numpy().astype(np.float32))
        local_step += 1
        done = np.full(num_envs, segment_end, dtype=np.bool_)
        metrics = {
            "active_mask": segment_active_mask,
            "next_distance": next_distance,
            "success": _to_numpy(info.get("success", np.zeros(num_envs, dtype=np.bool_)))
            .reshape(-1)
            .astype(np.bool_),
        }
        if segment_end:
            metrics["base_terminal_distance"] = base_terminal_distance
            metrics["paired_improvement"] = base_terminal_distance - next_distance
            reset_local_episode()
        return reward.astype(np.float32), done, metrics

    reset_local_episode()
    condition_buf = torch.zeros((rollout_steps, num_envs, condition_dim), device=device)
    raw_action_buf = torch.zeros((rollout_steps, num_envs, action_dim), device=device)
    base_action_buf = torch.zeros((rollout_steps, num_envs, action_dim), device=device)
    active_buf = torch.ones((rollout_steps, num_envs), device=device, dtype=torch.bool)
    logprob_buf = torch.zeros((rollout_steps, num_envs), device=device)
    reward_buf = torch.zeros((rollout_steps, num_envs), device=device)
    done_buf = torch.zeros((rollout_steps, num_envs), device=device)
    value_buf = torch.zeros((rollout_steps, num_envs), device=device)
    next_done = torch.zeros(num_envs, device=device)
    update_index = len(history)
    start_time = time.perf_counter()
    try:
        with trange(global_step, total_steps, initial=global_step, total=total_steps, desc=run_tag) as progress:
            while global_step < total_steps:
                distances: list[float] = []
                terminal_distances: list[float] = []
                rewards: list[float] = []
                action_delta_values: list[float] = []
                saturation_count = 0
                active_count = 0
                base_terminal_distances: list[float] = []
                paired_improvements: list[float] = []
                successes = 0
                agent.eval()
                for step in range(rollout_steps):
                    condition, base_action, distance = condition_and_base()
                    step_active = active_mask.copy()
                    condition_buf[step] = condition
                    base_action_buf[step] = torch.clamp(base_action, action_low, action_high)
                    active_buf[step] = torch.from_numpy(step_active).to(device)
                    done_buf[step] = next_done
                    with torch.no_grad():
                        raw_action, logprob, _entropy, value = agent.get_action_and_value(condition)
                    action = torch.clamp(raw_action, action_low, action_high)
                    raw_action_buf[step] = raw_action
                    logprob_buf[step] = logprob
                    value_buf[step] = value
                    reward, done, metrics = step_local(action, distance)
                    reward_buf[step] = torch.from_numpy(reward).to(device)
                    next_done = torch.from_numpy(done.astype(np.float32)).to(device)
                    distances.extend(distance[step_active].tolist())
                    rewards.extend(reward[step_active].tolist())
                    action_delta_step = (
                        torch.linalg.vector_norm(
                            action - base_action_buf[step],
                            dim=-1,
                        )
                        .cpu()
                        .numpy()
                    )
                    action_delta_values.extend(action_delta_step[step_active].tolist())
                    saturated_step = torch.any(raw_action != action, dim=-1).cpu().numpy()
                    saturation_count += int(np.sum(saturated_step[step_active]))
                    active_count += int(np.sum(step_active))
                    successes += int(np.sum(metrics["success"][step_active]))
                    if step == rollout_steps - 1:
                        terminal_active = metrics["active_mask"]
                        terminal_distances.extend(
                            metrics["next_distance"][terminal_active].tolist()
                        )
                        if "base_terminal_distance" in metrics:
                            base_terminal_distances.extend(
                                metrics["base_terminal_distance"][terminal_active].tolist()
                            )
                        if "paired_improvement" in metrics:
                            paired_improvements.extend(
                                metrics["paired_improvement"][terminal_active].tolist()
                            )
                    global_step += num_envs
                    progress.update(num_envs)

                with torch.no_grad():
                    next_value = torch.zeros(num_envs, device=device)
                    advantages = torch.zeros_like(reward_buf, device=device)
                    lastgaelam = torch.zeros(num_envs, device=device)
                    for t in reversed(range(rollout_steps)):
                        if t == rollout_steps - 1:
                            nextnonterminal = 1.0 - next_done
                            nextvalues = next_value
                        else:
                            nextnonterminal = 1.0 - done_buf[t + 1]
                            nextvalues = value_buf[t + 1]
                        delta = reward_buf[t] + gamma * nextvalues * nextnonterminal - value_buf[t]
                        lastgaelam = delta + gamma * gae_lambda * nextnonterminal * lastgaelam
                        advantages[t] = lastgaelam
                    returns = advantages + value_buf

                b_conditions = condition_buf.reshape((-1, condition_dim))
                b_raw_actions = raw_action_buf.reshape((-1, action_dim))
                b_base_actions = base_action_buf.reshape((-1, action_dim))
                b_active = active_buf.reshape(-1)
                b_logprobs = logprob_buf.reshape(-1)
                b_advantages = advantages.reshape(-1)
                b_returns = returns.reshape(-1)
                b_values = value_buf.reshape(-1)
                active_indices = torch.nonzero(b_active, as_tuple=False).flatten()
                if len(active_indices) == 0:
                    raise RuntimeError("No active PPO samples in rollout batch")
                active_advantages = b_advantages[active_indices]
                b_advantages = b_advantages.clone()
                b_advantages[active_indices] = (
                    active_advantages - active_advantages.mean()
                ) / (
                    active_advantages.std() + 1e-8
                )
                agent.train()
                batch_indices = active_indices.cpu().numpy()
                active_minibatch_size = max(1, len(batch_indices) // num_minibatches)
                policy_losses: list[float] = []
                value_losses: list[float] = []
                entropy_values: list[float] = []
                clipfracs: list[float] = []
                bc_losses: list[float] = []
                approx_kl = torch.tensor(0.0, device=device)
                for _epoch in range(update_epochs):
                    rng.shuffle(batch_indices)
                    for start in range(0, len(batch_indices), active_minibatch_size):
                        mb = batch_indices[start : start + active_minibatch_size]
                        _action, newlogprob, entropy, newvalue = agent.get_action_and_value(
                            b_conditions[mb],
                            raw_action=b_raw_actions[mb],
                        )
                        mean_action = agent.mean_action(b_conditions[mb])
                        bc_loss = torch.mean((mean_action - b_base_actions[mb]).square())
                        logratio = newlogprob - b_logprobs[mb]
                        ratio = logratio.exp()
                        with torch.no_grad():
                            approx_kl = ((ratio - 1.0) - logratio).mean()
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
                        v_loss = 0.5 * ((newvalue - b_returns[mb]) ** 2).mean()
                        entropy_loss = entropy.mean()
                        loss = (
                            pg_loss
                            - ent_coef * entropy_loss
                            + value_coef * v_loss
                            + bc_weight * bc_loss
                        )
                        optimizer.zero_grad(set_to_none=True)
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(trainable, max_grad_norm)
                        optimizer.step()
                        policy_losses.append(float(pg_loss.detach().cpu()))
                        value_losses.append(float(v_loss.detach().cpu()))
                        entropy_values.append(float(entropy_loss.detach().cpu()))
                        bc_losses.append(float(bc_loss.detach().cpu()))

                explained_variance = float(
                    1.0
                    - torch.var(b_returns - b_values).item()
                    / max(torch.var(b_returns).item(), 1e-8)
                )
                update_index += 1
                row = {
                    "update": update_index,
                    "global_step": int(global_step),
                    "mean_reward": float(np.mean(rewards)),
                    "mean_initial_distance": float(np.mean(distances)),
                    "mean_terminal_distance": float(np.mean(terminal_distances)),
                    "mean_base_terminal_distance": float(np.mean(base_terminal_distances))
                    if base_terminal_distances
                    else None,
                    "mean_paired_improvement": float(np.mean(paired_improvements))
                    if paired_improvements
                    else None,
                    "fraction_improved": float(np.mean(np.asarray(paired_improvements) > 0.0))
                    if paired_improvements
                    else None,
                    "mean_action_delta_l2": float(np.mean(action_delta_values)),
                    "active_fraction": float(active_count / batch_size),
                    "action_saturation_rate": float(
                        saturation_count / max(active_count, 1)
                    ),
                    "goal_prediction_weight": float(goal_prediction_weight),
                    "policy_loss": float(np.mean(policy_losses)),
                    "value_loss": float(np.mean(value_losses)),
                    "bc_loss": float(np.mean(bc_losses)),
                    "entropy": float(np.mean(entropy_values)),
                    "approx_kl": float(approx_kl.detach().cpu()),
                    "clipfrac": float(np.mean(clipfracs)),
                    "explained_variance": explained_variance,
                    "success_fraction_seen": successes / float(max(active_count, 1)),
                    "elapsed_s": time.perf_counter() - start_time,
                }
                history.append(row)
                write_json(history_path, {"recipe": recipe, "history": history})
                checkpoint_payload = {
                    "agent": agent.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "global_step": int(global_step),
                    "condition_dim": condition_dim,
                    "action_dim": action_dim,
                    "recipe": recipe,
                    "history": history,
                }
                torch.save(checkpoint_payload, checkpoint_out)
                if update_index % checkpoint_every_updates == 0 or global_step >= total_steps:
                    torch.save(
                        checkpoint_payload,
                        checkpoint_dir / f"step_{global_step:09d}.pt",
                    )
    finally:
        h5.close()
        env.close()
        if base_env is not None:
            base_env.close()
    return checkpoint_out
