from __future__ import annotations

import copy
import time
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
from rich.console import Console
from torch.utils.data import DataLoader, TensorDataset
from tqdm import trange

from hcl_poc.config import Config
from hcl_poc.flow import flow_matching_loss, sample_flow
from hcl_poc.models import FlowModel, MLP
from hcl_poc.utils import Standardizer, Timer, default_device, ensure_dir, set_seed, write_json

console = Console()


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
    force: bool = False,
) -> Path:
    from hcl_poc.rl_rerun import _make_benchmark_env, _to_numpy
    from hcl_poc.low_level_rl import ResidualActorCritic
    from hcl_poc.rl_rerun import _residual_action_from_raw

    if mode not in {"flat", "hierarchy", "oracle_hierarchy"}:
        raise ValueError(f"Unknown privileged-z eval mode: {mode}")
    if residual_checkpoint_path is not None and mode == "flat":
        raise ValueError("Privileged-z residual checkpoints require hierarchy or oracle_hierarchy mode")
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
    residual_agent = None
    residual_recipe: dict[str, Any] | None = None
    residual_alpha = 0.0
    residual_action_mode = "additive"
    if residual_checkpoint_path is not None:
        residual_payload = torch.load(
            residual_checkpoint_path,
            map_location=device,
            weights_only=False,
        )
        residual_recipe = dict(residual_payload["recipe"])
        if Path(residual_recipe["base_checkpoint"]).resolve() != checkpoint_path.resolve():
            raise ValueError("Residual checkpoint was trained against a different base checkpoint")
        residual_agent = ResidualActorCritic(
            int(residual_payload["condition_dim"]),
            action_dim=int(residual_payload["action_dim"]),
            width=int(residual_recipe["actor_critic_width"]),
            depth=int(residual_recipe["actor_critic_depth"]),
            initial_logstd=float(residual_recipe["initial_logstd"]),
        ).to(device)
        residual_agent.load_state_dict(residual_payload["agent"])
        residual_agent.eval()
        residual_alpha = float(residual_recipe["alpha"])
        residual_action_mode = str(residual_recipe.get("residual_action_mode", "additive"))

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
    zero_previous = action_norm.transform(np.zeros((1, int(payload["action_dim"])), dtype=np.float32))[0]
    previous = np.repeat(zero_previous[None], num_envs, axis=0)
    held_goal = np.zeros((num_envs, int(payload["state_dim"])), dtype=np.float32)
    countdown = np.zeros(num_envs, dtype=np.int32)
    successes: list[float] = []
    returns: list[float] = []
    cumulative_returns = np.zeros(num_envs, dtype=np.float32)
    success_once = np.zeros(num_envs, dtype=np.bool_)
    max_rewards = np.full(num_envs, -np.inf, dtype=np.float32)
    decisions = 0
    residual_norms: list[float] = []
    obs, _info = env.reset(seed=seed_start)
    try:
        while len(successes) < episodes:
            state_np = _to_numpy(obs["state"]).astype(np.float32)
            normalized_state = state_norm.transform(state_np)
            if mode in {"hierarchy", "oracle_hierarchy"}:
                replan = countdown <= 0
                if np.any(replan):
                    if mode == "oracle_hierarchy":
                        if branch_env is None or teacher is None:
                            raise RuntimeError("Oracle privileged-z eval was not initialized")
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
                residual_norms.extend(torch.linalg.vector_norm(residual, dim=-1).cpu().tolist())
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
                    cumulative_returns[mask_np] = 0.0
                    success_once[mask_np] = False
                    max_rewards[mask_np] = -np.inf
    finally:
        env.close()
        if branch_env is not None:
            branch_env.close()
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
        "mean_residual_norm": float(np.mean(residual_norms)) if residual_norms else 0.0,
    }
    write_json(out_path, result)
    console.print(result)
    return out_path


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
    history_path = result_dir / "history.json"
    if force:
        checkpoint_out.unlink(missing_ok=True)
        history_path.unlink(missing_ok=True)
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

    env = _make_benchmark_env(config, num_envs, "rgb+state")
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
        "reward": "privileged_state_progress_minus_terminal_distance_minus_residual_penalty",
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
    def reset_local_episode() -> None:
        nonlocal obs, current_state_norm, goal_state_norm, previous_action_norm
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
        live_state = np.asarray(obs["state"].detach().cpu().numpy(), dtype=np.float32)
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
        next_state = np.asarray(next_obs["state"].detach().cpu().numpy(), dtype=np.float32)
        next_state_norm = state_norm.transform(next_state)
        next_distance = np.mean((next_state_norm - goal_state_norm) ** 2, axis=-1).astype(
            np.float32
        )
        residual_penalty = residual_penalty_weight * residual.square().mean(dim=-1).cpu().numpy()
        reward = previous_distance - next_distance - residual_penalty
        segment_end = local_step == horizon_steps - 1
        if segment_end:
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
                    torch.save(
                        {
                            "agent": agent.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "global_step": int(global_step),
                            "condition_dim": condition_dim,
                            "action_dim": action_dim,
                            "recipe": recipe,
                            "history": history,
                        },
                        checkpoint_out,
                    )
    finally:
        h5.close()
        env.close()
    return checkpoint_out
