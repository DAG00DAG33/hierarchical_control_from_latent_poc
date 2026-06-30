#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import trange

from hcl_poc.models import MLP
from hcl_poc.utils import Standardizer, default_device, ensure_dir, set_seed, write_json


TCP_SLICE = slice(14, 17)


class TcpDpsi(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, depth: int) -> None:
        super().__init__()
        self.net = MLP(input_dim, 1, hidden_dim, depth=depth)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def _rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    ranks[order] = np.arange(len(values), dtype=np.float64)
    sorted_values = values[order]
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        if end - start > 1:
            ranks[order[start:end]] = (start + end - 1) / 2.0
        start = end
    return ranks


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2:
        return float("nan")
    rx = _rankdata(np.asarray(x, dtype=np.float64))
    ry = _rankdata(np.asarray(y, dtype=np.float64))
    rx -= rx.mean()
    ry -= ry.mean()
    denom = float(np.linalg.norm(rx) * np.linalg.norm(ry))
    if denom == 0.0:
        return float("nan")
    return float(np.dot(rx, ry) / denom)


def _binary_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    labels = labels.astype(bool)
    positives = scores[labels]
    negatives = scores[~labels]
    if len(positives) == 0 or len(negatives) == 0:
        return float("nan")
    return float(
        (
            (positives[:, None] > negatives[None, :]).mean()
            + 0.5 * (positives[:, None] == negatives[None, :]).mean()
        )
    )


def _true_tcp_distance(state: np.ndarray, goal: np.ndarray) -> np.ndarray:
    delta = state[..., TCP_SLICE] - goal[..., :3]
    return np.sum(delta * delta, axis=-1).astype(np.float32)


def _features(state: np.ndarray, goal: np.ndarray, tau: float) -> np.ndarray:
    if state.shape[0] != goal.shape[0]:
        raise ValueError("state and goal batch sizes must match")
    tcp_delta = goal[:, :3] - state[:, TCP_SLICE]
    tcp_l2 = np.linalg.norm(tcp_delta, axis=-1, keepdims=True)
    tau_col = np.full((len(state), 1), tau, dtype=np.float32)
    return np.concatenate([state, goal, tcp_delta, tcp_delta * tcp_delta, tcp_l2, tau_col], axis=-1).astype(np.float32)


def _target_transform(target: np.ndarray, scale: float) -> np.ndarray:
    return np.log1p(np.asarray(target, dtype=np.float32) * scale).astype(np.float32)


def _target_inverse(target: np.ndarray, scale: float) -> np.ndarray:
    return (np.expm1(np.asarray(target, dtype=np.float32)) / scale).astype(np.float32)


def _build_samples(
    data: dict[str, np.ndarray],
    env_indices: np.ndarray,
    *,
    include_shuffled: bool,
    max_candidate_samples: int | None,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    goal = data["goal"][env_indices]
    shuffled_goal = data["shuffled_goal"][env_indices]
    states = [
        data["start_state"][env_indices],
        data["ppo_terminal_state"][env_indices],
        data["best_terminal_state"][env_indices],
    ]
    goals = [goal, goal, goal]
    targets = [
        _true_tcp_distance(data["start_state"][env_indices], goal),
        data["ppo_terminal_distance"][env_indices],
        data["best_terminal_distance"][env_indices],
    ]
    candidate_states = data["candidate_terminal_state"][:, env_indices].reshape(-1, data["start_state"].shape[-1])
    candidate_goals = np.repeat(goal[None], data["candidate_terminal_state"].shape[0], axis=0).reshape(-1, goal.shape[-1])
    candidate_targets = data["candidate_terminal_distance"][:, env_indices].reshape(-1)
    if max_candidate_samples is not None and len(candidate_targets) > max_candidate_samples:
        choice = rng.choice(len(candidate_targets), size=max_candidate_samples, replace=False)
        candidate_states = candidate_states[choice]
        candidate_goals = candidate_goals[choice]
        candidate_targets = candidate_targets[choice]
    states.append(candidate_states)
    goals.append(candidate_goals)
    targets.append(candidate_targets)
    if include_shuffled:
        base_states = np.concatenate(
            [
                data["start_state"][env_indices],
                data["ppo_terminal_state"][env_indices],
                data["best_terminal_state"][env_indices],
            ],
            axis=0,
        )
        shuffled = np.concatenate([shuffled_goal, shuffled_goal, shuffled_goal], axis=0)
        states.append(base_states)
        goals.append(shuffled)
        targets.append(_true_tcp_distance(base_states, shuffled))
    state_all = np.concatenate(states, axis=0)
    goal_all = np.concatenate(goals, axis=0)
    target_all = np.concatenate(targets, axis=0).astype(np.float32)
    return _features(state_all, goal_all, tau=0.0), target_all


@torch.inference_mode()
def _predict_ensemble(
    models: list[TcpDpsi],
    input_norm: Standardizer,
    target_norm: Standardizer,
    features: np.ndarray,
    device: torch.device,
    batch_size: int,
    target_scale: float,
) -> tuple[np.ndarray, np.ndarray]:
    x = input_norm.transform(features).astype(np.float32)
    all_predictions = []
    for model in models:
        chunks = []
        model.eval()
        for start in range(0, len(x), batch_size):
            pred_norm = model(torch.from_numpy(x[start : start + batch_size]).to(device))
            chunks.append(pred_norm.detach().cpu().numpy())
        pred_transformed = target_norm.inverse(np.concatenate(chunks, axis=0)[:, None]).reshape(-1)
        pred = _target_inverse(pred_transformed, target_scale)
        all_predictions.append(np.maximum(pred, 0.0))
    stacked = np.stack(all_predictions, axis=0)
    return stacked.mean(axis=0), stacked.std(axis=0)


def _candidate_features(data: dict[str, np.ndarray], env_indices: np.ndarray) -> np.ndarray:
    c, n, state_dim = data["candidate_terminal_state"][:, env_indices].shape
    states = data["candidate_terminal_state"][:, env_indices].reshape(c * n, state_dim)
    goals = np.repeat(data["goal"][env_indices][None], c, axis=0).reshape(c * n, data["goal"].shape[-1])
    return _features(states, goals, tau=0.0)


def _train_one(
    train_x: np.ndarray,
    train_y: np.ndarray,
    val_x: np.ndarray,
    val_y: np.ndarray,
    *,
    seed: int,
    input_norm: Standardizer,
    target_norm: Standardizer,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[TcpDpsi, dict[str, Any]]:
    set_seed(seed)
    x = torch.from_numpy(input_norm.transform(train_x).astype(np.float32))
    y = torch.from_numpy(
        target_norm.transform(_target_transform(train_y, args.target_scale)[:, None]).astype(np.float32)
    ).squeeze(-1)
    generator = torch.Generator()
    generator.manual_seed(seed + 17)
    loader = DataLoader(
        TensorDataset(x, y),
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    model = TcpDpsi(train_x.shape[-1], args.hidden_dim, args.depth).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    loss_fn = nn.MSELoss()
    val_x_norm = torch.from_numpy(input_norm.transform(val_x).astype(np.float32)).to(device)
    val_y_norm = torch.from_numpy(
        target_norm.transform(_target_transform(val_y, args.target_scale)[:, None]).astype(np.float32)
    ).squeeze(-1).to(device)
    best_loss = float("inf")
    best_state = None
    history = []
    for epoch in trange(1, args.epochs + 1, desc=f"train TCP D_psi seed{seed}"):
        model.train()
        losses = []
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        model.eval()
        with torch.inference_mode():
            val_loss = float(loss_fn(model(val_x_norm), val_y_norm).detach().cpu())
        history.append(
            {
                "epoch": epoch,
                "train_loss": float(np.mean(losses)),
                "validation_loss": val_loss,
            }
        )
        if val_loss < best_loss:
            best_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
    if best_state is None:
        raise RuntimeError("D_psi training did not produce a checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    return model, {"seed": seed, "best_validation_loss": best_loss, "history": history}


def run(args: argparse.Namespace) -> Path:
    data_npz = np.load(args.dataset)
    data = {key: data_npz[key] for key in data_npz.files}
    n_env = int(data["goal"].shape[0])
    rng = np.random.default_rng(args.seed + 4_500_000)
    permutation = rng.permutation(n_env)
    val_count = max(1, int(round(n_env * args.validation_fraction)))
    val_indices = permutation[:val_count]
    train_indices = permutation[val_count:]
    train_x, train_y = _build_samples(
        data,
        train_indices,
        include_shuffled=True,
        max_candidate_samples=args.max_train_candidate_samples,
        seed=args.seed + 1,
    )
    val_x, val_y = _build_samples(
        data,
        val_indices,
        include_shuffled=True,
        max_candidate_samples=args.max_val_candidate_samples,
        seed=args.seed + 2,
    )
    input_norm = Standardizer.fit(train_x)
    target_norm = Standardizer.fit(_target_transform(train_y, args.target_scale)[:, None])
    device = default_device()
    models = []
    members = []
    for member in range(args.members):
        model, summary = _train_one(
            train_x,
            train_y,
            val_x,
            val_y,
            seed=args.seed + member * 101,
            input_norm=input_norm,
            target_norm=target_norm,
            args=args,
            device=device,
        )
        models.append(model)
        members.append(summary)

    pred_val, std_val = _predict_ensemble(
        models, input_norm, target_norm, val_x, device, args.eval_batch_size, args.target_scale
    )
    val_reachable = val_y <= float(data["success_epsilon"])
    candidate_features = _candidate_features(data, val_indices)
    candidate_pred, candidate_std = _predict_ensemble(
        models,
        input_norm,
        target_norm,
        candidate_features,
        device,
        args.eval_batch_size,
        args.target_scale,
    )
    c = int(data["candidate_terminal_state"].shape[0])
    n_val = len(val_indices)
    candidate_pred = candidate_pred.reshape(c, n_val)
    candidate_std = candidate_std.reshape(c, n_val)
    candidate_true = data["candidate_terminal_distance"][:, val_indices]
    selected = np.argmin(candidate_pred, axis=0)
    selected_true = candidate_true[selected, np.arange(n_val)]

    ppo_features = _features(data["ppo_terminal_state"][val_indices], data["goal"][val_indices], tau=0.0)
    ppo_pred, ppo_std = _predict_ensemble(
        models, input_norm, target_norm, ppo_features, device, args.eval_batch_size, args.target_scale
    )
    best_features = _features(data["best_terminal_state"][val_indices], data["goal"][val_indices], tau=0.0)
    best_pred, best_std = _predict_ensemble(
        models, input_norm, target_norm, best_features, device, args.eval_batch_size, args.target_scale
    )
    shuffled_features = _features(
        data["ppo_terminal_state"][val_indices],
        data["shuffled_goal"][val_indices],
        tau=0.0,
    )
    shuffled_pred, shuffled_std = _predict_ensemble(
        models,
        input_norm,
        target_norm,
        shuffled_features,
        device,
        args.eval_batch_size,
        args.target_scale,
    )
    shuffled_true = _true_tcp_distance(
        data["ppo_terminal_state"][val_indices],
        data["shuffled_goal"][val_indices],
    )
    ppo_true = data["ppo_terminal_distance"][val_indices]
    best_true = data["best_terminal_distance"][val_indices]
    payload = {
        "run": "rl_reachability_debug_run4_tcp_dpsi_ensemble",
        "dataset": str(args.dataset),
        "members": int(args.members),
        "train_envs": int(len(train_indices)),
        "validation_envs": int(len(val_indices)),
        "train_samples": int(len(train_x)),
        "validation_samples": int(len(val_x)),
        "success_epsilon": float(data["success_epsilon"]),
        "target_transform": f"log1p(distance * {args.target_scale:g})",
        "members_summary": members,
        "metrics": {
            "validation_mse": float(np.mean((pred_val - val_y) ** 2)),
            "validation_spearman": _spearman(val_y, pred_val),
            "reachability_auc": _binary_auc(-pred_val, val_reachable.astype(np.int64)),
            "best_vs_ppo_rank_accuracy": float(np.mean(best_pred < ppo_pred)),
            "best_vs_ppo_true_fraction": float(np.mean(best_true < ppo_true)),
            "shuffled_pred_greater_than_goal_accuracy": float(np.mean(shuffled_pred > ppo_pred)),
            "shuffled_true_greater_than_goal_accuracy": float(np.mean(shuffled_true > ppo_true)),
            "candidate_selection_actual_distance_mean": float(np.mean(selected_true)),
            "candidate_selection_reach_rate": float(
                np.mean(selected_true <= float(data["success_epsilon"]))
            ),
            "random_candidate_actual_distance_mean": float(np.mean(candidate_true)),
            "ppo_actual_distance_mean": float(np.mean(ppo_true)),
            "oracle_best_actual_distance_mean": float(np.mean(best_true)),
            "candidate_selection_improvement_vs_ppo": float(np.mean(ppo_true - selected_true)),
            "candidate_selection_fraction_improved_vs_ppo": float(np.mean(selected_true < ppo_true)),
            "candidate_selection_oracle_gap": float(np.mean(selected_true - best_true)),
            "ensemble_std_validation_mean": float(np.mean(std_val)),
            "ensemble_std_ppo_mean": float(np.mean(ppo_std)),
            "ensemble_std_best_mean": float(np.mean(best_std)),
            "ensemble_std_candidate_mean": float(np.mean(candidate_std)),
            "ensemble_std_shuffled_mean": float(np.mean(shuffled_std)),
            "uncertainty_shuffled_over_ppo_ratio": float(
                np.mean(shuffled_std) / max(float(np.mean(ppo_std)), 1e-12)
            ),
        },
        "gate_thresholds": {
            "best_vs_ppo_rank_accuracy": 0.85,
            "validation_spearman": 0.8,
            "reachability_auc": 0.9,
        },
        "gates": {},
    }
    metrics = payload["metrics"]
    payload["gates"] = {
        "ranks_random_search_selected_better_than_ppo": bool(
            metrics["best_vs_ppo_rank_accuracy"]
            >= payload["gate_thresholds"]["best_vs_ppo_rank_accuracy"]
        ),
        "correlates_with_actual_terminal_distance": bool(
            metrics["validation_spearman"] >= 0.8
        ),
        "separates_reachable_from_unreachable": bool(
            metrics["reachability_auc"] >= 0.9
        ),
        "candidate_selection_improves_actual_distance": bool(
            metrics["candidate_selection_actual_distance_mean"]
            < metrics["ppo_actual_distance_mean"]
        ),
        "candidate_selection_near_oracle_best": bool(
            metrics["candidate_selection_actual_distance_mean"]
            <= metrics["oracle_best_actual_distance_mean"] * 2.0 + 1e-6
        ),
        "uncertainty_rises_on_shuffled_goals": bool(
            metrics["uncertainty_shuffled_over_ppo_ratio"] > 1.0
        ),
    }
    artifact_dir = ensure_dir(Path(args.output_dir))
    checkpoint_path = artifact_dir / "tcp_dpsi_ensemble.pt"
    torch.save(
        {
            "models": [model.state_dict() for model in models],
            "input_norm": input_norm.state_dict(),
            "target_norm": target_norm.state_dict(),
            "input_dim": int(train_x.shape[-1]),
            "hidden_dim": int(args.hidden_dim),
            "depth": int(args.depth),
            "payload": payload,
        },
        checkpoint_path,
    )
    payload["checkpoint"] = str(checkpoint_path)
    output = Path(args.output)
    ensure_dir(output.parent)
    write_json(output, payload)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        default="data/rl_reachability_debug/run4_tcp_branch_dataset_c64_ref2.npz",
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts/incremental/rl_reachability_debug/run4_tcp_dpsi_ensemble",
    )
    parser.add_argument(
        "--output",
        default="results/incremental/rl_reachability_debug/run4_tcp_dpsi_ensemble.json",
    )
    parser.add_argument("--members", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--eval-batch-size", type=int, default=32768)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--max-train-candidate-samples", type=int, default=262144)
    parser.add_argument("--max-val-candidate-samples", type=int)
    parser.add_argument("--target-scale", type=float, default=1000.0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    print(run(args))


if __name__ == "__main__":
    main()
