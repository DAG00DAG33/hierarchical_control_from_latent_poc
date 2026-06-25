#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from hcl_poc.privileged_z import (
    _model_from_payload,
    _predict_loaded_payload,
)
from hcl_poc.utils import default_device, set_seed


def _selector_model(input_dim: int, hidden_dim: int, depth: int) -> nn.Sequential:
    layers: list[nn.Module] = []
    dim = input_dim
    for _ in range(depth):
        layers.extend([nn.Linear(dim, hidden_dim), nn.ReLU()])
        dim = hidden_dim
    layers.append(nn.Linear(dim, 1))
    return nn.Sequential(*layers)


def _load_branch_bank(path: Path, state_dim: int, action_dim: int) -> dict[str, np.ndarray]:
    data = np.load(path, allow_pickle=True)
    horizon_steps = int(np.asarray(data["horizon_steps"]).item())
    conditions = np.asarray(data["conditions"], dtype=np.float32)
    branch_count = len(np.asarray(data["selected_return_delta"]))
    expected_dim = state_dim * 2 + action_dim + 1
    if conditions.shape != (branch_count * horizon_steps, expected_dim):
        raise ValueError(
            f"Expected conditions shape {(branch_count * horizon_steps, expected_dim)}, "
            f"got {conditions.shape}"
        )
    branch_conditions = conditions.reshape(horizon_steps, branch_count, expected_dim)[0]
    return_delta = np.asarray(data["selected_return_delta"], dtype=np.float32)
    success_delta = np.asarray(data["selected_success_delta"], dtype=np.float32)
    improvement = np.asarray(data["selected_improvement_mse"], dtype=np.float32)
    action_delta = np.asarray(data["selected_action_delta_l2"], dtype=np.float32)
    completed = np.asarray(data["selected_completed"], dtype=np.float32)

    def zscore(values: np.ndarray) -> np.ndarray:
        std = float(np.std(values))
        if std < 1e-6:
            std = 1.0
        return ((values - float(np.mean(values))) / std).astype(np.float32)

    return {
        "states": branch_conditions[:, :state_dim].copy(),
        "goals": branch_conditions[:, state_dim : state_dim * 2].copy(),
        "previous": branch_conditions[:, state_dim * 2 : state_dim * 2 + action_dim].copy(),
        "return_delta": return_delta,
        "outcome_features": np.stack(
            [
                zscore(return_delta),
                success_delta,
                zscore(improvement),
                zscore(action_delta),
                completed,
            ],
            axis=-1,
        ).astype(np.float32),
    }


def _explicit_pair_features(
    query_states: np.ndarray,
    query_goals: np.ndarray,
    query_previous: np.ndarray,
    candidate_states: np.ndarray,
    candidate_goals: np.ndarray,
    candidate_previous: np.ndarray,
    candidate_outcomes: np.ndarray,
) -> np.ndarray:
    state_delta = query_states - candidate_states
    goal_delta = query_goals - candidate_goals
    previous_delta = query_previous - candidate_previous
    state_mse = np.mean(state_delta * state_delta, axis=-1, keepdims=True)
    goal_mse = np.mean(goal_delta * goal_delta, axis=-1, keepdims=True)
    previous_mse = np.mean(previous_delta * previous_delta, axis=-1, keepdims=True)
    return np.concatenate(
        [
            query_states,
            query_goals,
            query_previous,
            candidate_states,
            candidate_goals,
            candidate_previous,
            state_delta,
            goal_delta,
            previous_delta,
            state_mse,
            goal_mse,
            previous_mse,
            0.5 * state_mse + 0.5 * goal_mse,
            candidate_outcomes,
        ],
        axis=-1,
    ).astype(np.float32)


def _pair_features_and_targets(
    rng: np.random.Generator,
    query_indices: np.ndarray,
    bank: dict[str, np.ndarray],
    predicted_goals: np.ndarray,
    *,
    candidates_per_query: int,
    distance_penalty: float,
) -> tuple[np.ndarray, np.ndarray]:
    branch_count = len(bank["states"])
    sampled_candidates = rng.integers(
        0,
        branch_count,
        size=(len(query_indices), candidates_per_query),
    )
    sampled_candidates[:, 0] = query_indices
    flat_query = np.repeat(query_indices, candidates_per_query)
    flat_candidate = sampled_candidates.reshape(-1)
    features = _explicit_pair_features(
        bank["states"][flat_query],
        predicted_goals[flat_query],
        bank["previous"][flat_query],
        bank["states"][flat_candidate],
        bank["goals"][flat_candidate],
        bank["previous"][flat_candidate],
        bank["outcome_features"][flat_candidate],
    )
    state_delta = bank["states"][flat_query] - bank["states"][flat_candidate]
    goal_delta = predicted_goals[flat_query] - bank["goals"][flat_candidate]
    previous_delta = bank["previous"][flat_query] - bank["previous"][flat_candidate]
    distance = (
        0.45 * np.mean(state_delta * state_delta, axis=-1)
        + 0.45 * np.mean(goal_delta * goal_delta, axis=-1)
        + 0.10 * np.mean(previous_delta * previous_delta, axis=-1)
    )
    return_z = bank["outcome_features"][flat_candidate, 0]
    target = return_z - float(distance_penalty) * distance
    return features.astype(np.float32), target.astype(np.float32)[:, None]


def train_selector(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = default_device()
    base = torch.load(args.base_checkpoint, map_location=device, weights_only=False)
    state_dim = int(base["state_dim"])
    action_dim = int(base["action_dim"])
    bank = _load_branch_bank(args.branch_bank, state_dim, action_dim)
    high_model = _model_from_payload(base["high"], device)
    with torch.inference_mode():
        predicted_goals = _predict_loaded_payload(
            high_model,
            base["high"],
            np.concatenate([bank["states"], bank["previous"]], axis=-1).astype(np.float32),
            device,
        ).astype(np.float32)

    indices = np.arange(len(bank["states"]))
    rng.shuffle(indices)
    val_count = max(1, int(round(len(indices) * args.validation_fraction)))
    val_indices = indices[:val_count]
    train_indices = indices[val_count:]
    train_x, train_y = _pair_features_and_targets(
        rng,
        train_indices,
        bank,
        predicted_goals,
        candidates_per_query=args.candidates_per_query,
        distance_penalty=args.distance_penalty,
    )
    val_x, val_y = _pair_features_and_targets(
        rng,
        val_indices,
        bank,
        predicted_goals,
        candidates_per_query=args.candidates_per_query,
        distance_penalty=args.distance_penalty,
    )
    feature_mean = train_x.mean(axis=0).astype(np.float32)
    feature_std = np.maximum(train_x.std(axis=0), 1e-6).astype(np.float32)
    train_x = ((train_x - feature_mean) / feature_std).astype(np.float32)
    val_x = ((val_x - feature_mean) / feature_std).astype(np.float32)

    model = _selector_model(train_x.shape[1], args.hidden_dim, args.depth).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y)),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    val_x_t = torch.from_numpy(val_x).to(device).float()
    val_y_t = torch.from_numpy(val_y).to(device).float()
    best_state = None
    best_val = float("inf")
    history: list[dict[str, float | int]] = []
    for epoch in range(1, args.epochs + 1):
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
        train_mse = total / max(count, 1)
        history.append({"epoch": epoch, "train_mse": train_mse, "validation_mse": val_mse})
        if val_mse < best_val:
            best_val = val_mse
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    if best_state is None:
        raise RuntimeError("Selector training did not produce a checkpoint")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "method": "privileged_z_branch_goal_selector",
            "base_checkpoint": str(args.base_checkpoint),
            "branch_bank": str(args.branch_bank),
            "input_dim": int(train_x.shape[1]),
            "hidden_dim": int(args.hidden_dim),
            "depth": int(args.depth),
            "model": best_state,
            "feature_mean": feature_mean,
            "feature_std": feature_std,
            "bank_states": bank["states"],
            "bank_goals": bank["goals"],
            "bank_previous": bank["previous"],
            "bank_outcome_features": bank["outcome_features"],
            "return_delta": bank["return_delta"],
            "predicted_goals": predicted_goals,
            "distance_penalty": float(args.distance_penalty),
            "candidates_per_query": int(args.candidates_per_query),
            "history": history,
            "best_validation_mse": float(best_val),
        },
        args.output,
    )
    print(
        {
            "output": str(args.output),
            "branches": int(len(bank["states"])),
            "train_pairs": int(len(train_x)),
            "validation_pairs": int(len(val_x)),
            "best_validation_mse": float(best_val),
            "final_train_mse": float(history[-1]["train_mse"]),
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--branch-bank", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--candidates-per-query", type=int, default=64)
    parser.add_argument("--distance-penalty", type=float, default=1.0)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    args = parser.parse_args()
    train_selector(args)


if __name__ == "__main__":
    main()
