from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
from rich.console import Console
from torch.utils.data import DataLoader, TensorDataset
from tqdm import trange

from hcl_poc.config import Config
from hcl_poc.models import MLP
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
    zero_previous = action_norm.transform(np.zeros((1, 3), dtype=np.float32))[0]
    for episode in episodes:
        states = episode["states"]
        actions = episode["actions"]
        previous_actions = action_norm.transform(episode["previous_actions"])
        normalized_states = state_norm.transform(states)
        normalized_actions = action_norm.transform(actions)
        for t in range(len(actions) - horizon_steps):
            previous = previous_actions[t - 1] if t > 0 else zero_previous
            if for_high:
                rows.append(np.concatenate([normalized_states[t], previous], axis=-1))
                labels.append(normalized_states[t + horizon_steps])
                continue
            remaining = np.asarray([1.0], dtype=np.float32)
            if include_goal:
                condition = np.concatenate(
                    [
                        normalized_states[t],
                        normalized_states[t + horizon_steps],
                        previous,
                        remaining,
                    ],
                    axis=-1,
                )
            else:
                condition = np.concatenate(
                    [normalized_states[t], previous, remaining],
                    axis=-1,
                )
            rows.append(condition)
            labels.append(normalized_actions[t])
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
    val_x_t = torch.from_numpy(val_x).to(device).float()
    val_y_t = torch.from_numpy(val_y).to(device).float()
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
    with torch.inference_mode():
        pred = model(val_x_t).cpu().numpy()
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
            "model": best_state,
            "input_dim": int(train_x.shape[-1]),
            "output_dim": int(train_y.shape[-1]),
            "hidden_dim": hidden_dim,
            "depth": depth,
            "history": history,
        },
        metrics,
    )


def _predict(
    payload: dict[str, Any],
    x: np.ndarray,
    device: torch.device,
) -> np.ndarray:
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


def _model_from_payload(payload: dict[str, Any], device: torch.device) -> MLP:
    model = MLP(
        int(payload["input_dim"]),
        int(payload["output_dim"]),
        int(payload["hidden_dim"]),
        depth=int(payload["depth"]),
    ).to(device)
    model.load_state_dict(payload["model"])
    model.eval()
    return model


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
    force: bool = False,
) -> Path:
    from hcl_poc.rl_rerun import _make_benchmark_env, _to_numpy

    if mode not in {"flat", "hierarchy"}:
        raise ValueError(f"Unknown privileged-z eval mode: {mode}")
    out_path = output_path or checkpoint_path.with_name(
        f"{checkpoint_path.stem}_eval_{mode}_n{episodes}.json"
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

    env = _make_benchmark_env(config, num_envs, "rgb+state")
    action_low = torch.as_tensor(env.single_action_space.low, device=device, dtype=torch.float32)
    action_high = torch.as_tensor(env.single_action_space.high, device=device, dtype=torch.float32)
    zero_previous = action_norm.transform(np.zeros((1, int(payload["action_dim"])), dtype=np.float32))[0]
    previous = np.repeat(zero_previous[None], num_envs, axis=0)
    held_goal = np.zeros((num_envs, int(payload["state_dim"])), dtype=np.float32)
    countdown = np.zeros(num_envs, dtype=np.int32)
    successes: list[float] = []
    returns: list[float] = []
    cumulative_returns = np.zeros(num_envs, dtype=np.float32)
    max_rewards = np.full(num_envs, -np.inf, dtype=np.float32)
    decisions = 0
    obs, _info = env.reset(seed=seed_start)
    try:
        while len(successes) < episodes:
            state_np = _to_numpy(obs["state"]).astype(np.float32)
            normalized_state = state_norm.transform(state_np)
            if mode == "hierarchy":
                replan = countdown <= 0
                if np.any(replan):
                    high_input = np.concatenate([normalized_state, previous], axis=-1)
                    high_goal = high_model(torch.from_numpy(high_input).to(device).float())
                    held_goal[replan] = high_goal.cpu().numpy()[replan]
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
                normalized_action = goal_model(
                    torch.from_numpy(low_input).to(device).float()
                ).cpu().numpy()
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
                normalized_action = flat_model(
                    torch.from_numpy(flat_input).to(device).float()
                ).cpu().numpy()
            action_np = action_norm.inverse(normalized_action)
            action = torch.as_tensor(action_np, device=device, dtype=torch.float32)
            action = torch.clamp(action, action_low, action_high)
            obs, reward, _terminated, _truncated, info = env.step(action)
            previous = action_norm.transform(action.cpu().numpy().astype(np.float32))
            reward_np = _to_numpy(reward).reshape(-1).astype(np.float32)
            cumulative_returns += reward_np
            max_rewards = np.maximum(max_rewards, reward_np)
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
                        success_values = (
                            final_info["success"][mask].detach().float().cpu().numpy()
                        )
                        return_values = cumulative_returns[mask_np].copy()
                    successes.extend(float(x) for x in success_values)
                    returns.extend(float(x) for x in return_values)
                    previous[mask_np] = zero_previous
                    held_goal[mask_np] = 0.0
                    countdown[mask_np] = 0
                    cumulative_returns[mask_np] = 0.0
                    max_rewards[mask_np] = -np.inf
    finally:
        env.close()
    result = {
        "checkpoint": str(checkpoint_path),
        "mode": mode,
        "episodes": int(episodes),
        "seed_start": int(seed_start),
        "num_envs": int(num_envs),
        "success": float(np.mean(successes[:episodes])),
        "return": float(np.mean(returns[:episodes])),
        "high_level_decisions_per_episode": decisions / max(len(successes), 1),
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
    zero_previous = action_norm.transform(np.zeros((1, 3), dtype=np.float32))[0]
    for horizon in horizons:
        rows = []
        for choice in indices:
            episode_index, t = candidates[int(choice)]
            episode = validation[episode_index]
            states = state_norm.transform(episode["states"])
            previous = (
                action_norm.transform(episode["previous_actions"][t - 1 : t])[0]
                if t > 0
                else zero_previous
            )
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
        "dataset": str(path),
        "n_trajectories": n_trajectories,
        "validation_trajectories": validation_trajectories,
        "horizon_steps": horizon_steps,
        "seed": seed,
        "run_tag": run_tag,
        "selection_mode": selection_mode,
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
            "metrics": payload["metrics"],
            "data": data,
            "elapsed_s": payload["elapsed_s"],
        },
    )
    console.print(f"Wrote privileged-z hierarchy: {checkpoint_path}")
    return checkpoint_path
