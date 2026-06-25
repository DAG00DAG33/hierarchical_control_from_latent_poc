#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from hcl_poc.utils import default_device, set_seed


def _features(data: np.lib.npyio.NpzFile) -> np.ndarray:
    query_states = np.asarray(data["query_states"], dtype=np.float32)
    query_goals = np.asarray(data["query_goals"], dtype=np.float32)
    query_previous = np.asarray(data["query_previous"], dtype=np.float32)
    candidate_goals = np.asarray(data["candidate_goals"], dtype=np.float32)
    nearest_scores = np.asarray(data["candidate_nearest_scores"], dtype=np.float32)
    source_return = np.asarray(data["candidate_source_return_delta"], dtype=np.float32)
    source_success = np.asarray(data["candidate_source_success_delta"], dtype=np.float32)
    query_count, candidate_count, state_dim = candidate_goals.shape
    query_states_expanded = np.repeat(query_states[:, None, :], candidate_count, axis=1)
    query_goals_expanded = np.repeat(query_goals[:, None, :], candidate_count, axis=1)
    query_previous_expanded = np.repeat(query_previous[:, None, :], candidate_count, axis=1)
    goal_delta = query_goals_expanded - candidate_goals
    goal_mse = np.mean(goal_delta * goal_delta, axis=-1, keepdims=True)
    return np.concatenate(
        [
            query_states_expanded,
            query_goals_expanded,
            query_previous_expanded,
            candidate_goals,
            goal_delta,
            goal_mse,
            nearest_scores[:, :, None],
            source_return[:, :, None],
            source_success[:, :, None],
        ],
        axis=-1,
    ).reshape(query_count * candidate_count, state_dim * 4 + query_previous.shape[1] + 4)


def _model(input_dim: int, hidden_dim: int, depth: int) -> nn.Sequential:
    layers: list[nn.Module] = []
    dim = input_dim
    for _ in range(depth):
        layers.extend([nn.Linear(dim, hidden_dim), nn.ReLU()])
        dim = hidden_dim
    layers.append(nn.Linear(dim, 1))
    return nn.Sequential(*layers)


def _selection_metrics(
    scores: np.ndarray,
    return_delta: np.ndarray,
    success_delta: np.ndarray,
    source_return_delta: np.ndarray,
    nearest_scores: np.ndarray,
    indices: np.ndarray,
) -> dict[str, float]:
    chosen = np.argmax(scores[indices], axis=1)
    row = np.arange(len(indices))
    source_chosen = np.argmax(source_return_delta[indices], axis=1)
    nearest_chosen = np.argmin(nearest_scores[indices], axis=1)
    return {
        "selected_return_delta": float(np.mean(return_delta[indices][row, chosen])),
        "selected_success_delta": float(np.mean(success_delta[indices][row, chosen])),
        "nearest_return_delta": float(np.mean(return_delta[indices, 0])),
        "nearest_success_delta": float(np.mean(success_delta[indices, 0])),
        "nearest_argmin_return_delta": float(
            np.mean(return_delta[indices][row, nearest_chosen])
        ),
        "source_return_argmax_return_delta": float(
            np.mean(return_delta[indices][row, source_chosen])
        ),
        "source_return_argmax_success_delta": float(
            np.mean(success_delta[indices][row, source_chosen])
        ),
        "oracle_return_delta": float(np.mean(np.max(return_delta[indices], axis=1))),
        "oracle_success_delta": float(
            np.mean(success_delta[indices][row, np.argmax(return_delta[indices], axis=1)])
        ),
    }


def train(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = default_device()
    data = np.load(args.input, allow_pickle=True)
    query_count = int(data["query_states"].shape[0])
    candidate_count = int(data["candidate_goals"].shape[1])
    x = _features(data).astype(np.float32)
    return_delta = np.asarray(data["candidate_return_delta"], dtype=np.float32)
    success_delta = np.asarray(data["candidate_success_delta"], dtype=np.float32)
    source_return_delta = np.asarray(data["candidate_source_return_delta"], dtype=np.float32)
    nearest_scores = np.asarray(data["candidate_nearest_scores"], dtype=np.float32)
    y = return_delta.reshape(-1, 1)
    query_indices = np.arange(query_count)
    rng.shuffle(query_indices)
    val_count = max(1, int(round(query_count * args.validation_fraction)))
    val_queries = query_indices[:val_count]
    train_queries = query_indices[val_count:]
    train_rows = (train_queries[:, None] * candidate_count + np.arange(candidate_count)[None]).reshape(-1)
    val_rows = (val_queries[:, None] * candidate_count + np.arange(candidate_count)[None]).reshape(-1)
    feature_mean = x[train_rows].mean(axis=0).astype(np.float32)
    feature_std = np.maximum(x[train_rows].std(axis=0), 1e-6).astype(np.float32)
    x_norm = ((x - feature_mean) / feature_std).astype(np.float32)

    model = _model(x.shape[1], args.hidden_dim, args.depth).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    if args.loss == "mse":
        loader = DataLoader(
            TensorDataset(torch.from_numpy(x_norm[train_rows]), torch.from_numpy(y[train_rows])),
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=torch.cuda.is_available(),
        )
        val_x = torch.from_numpy(x_norm[val_rows]).to(device).float()
        val_y = torch.from_numpy(y[val_rows]).to(device).float()
    else:
        grouped_x = x_norm.reshape(query_count, candidate_count, -1)
        best_candidate = np.argmax(return_delta, axis=1).astype(np.int64)
        loader = DataLoader(
            TensorDataset(
                torch.from_numpy(grouped_x[train_queries]),
                torch.from_numpy(best_candidate[train_queries]),
            ),
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=torch.cuda.is_available(),
        )
        val_x = torch.from_numpy(grouped_x[val_queries]).to(device).float()
        val_y = torch.from_numpy(best_candidate[val_queries]).to(device).long()
    best_state = None
    best_val = float("inf")
    history: list[dict[str, float | int]] = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        count = 0
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device, non_blocking=True).float()
            if args.loss == "mse":
                batch_y = batch_y.to(device, non_blocking=True).float()
                loss = torch.mean((model(batch_x) - batch_y) ** 2)
                batch_count = len(batch_x)
            else:
                batch_y = batch_y.to(device, non_blocking=True).long()
                logits = model(batch_x.reshape(-1, batch_x.shape[-1])).reshape(
                    batch_x.shape[0],
                    candidate_count,
                )
                loss = torch.nn.functional.cross_entropy(logits, batch_y)
                batch_count = int(batch_x.shape[0])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += float(loss.detach().cpu()) * batch_count
            count += batch_count
        model.eval()
        with torch.inference_mode():
            if args.loss == "mse":
                val_metric = float(torch.mean((model(val_x) - val_y) ** 2).cpu())
            else:
                val_logits = model(val_x.reshape(-1, val_x.shape[-1])).reshape(
                    val_x.shape[0],
                    candidate_count,
                )
                val_metric = float(torch.nn.functional.cross_entropy(val_logits, val_y).cpu())
        history.append(
            {
                "epoch": epoch,
                "train_mse": total / max(count, 1),
                "validation_mse": val_metric,
            }
        )
        if val_metric < best_val:
            best_val = val_metric
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    if best_state is None:
        raise RuntimeError("No selector checkpoint was produced")
    model.load_state_dict(best_state)
    model.eval()
    with torch.inference_mode():
        scores = (
            model(torch.from_numpy(x_norm).to(device).float())
            .reshape(query_count, candidate_count)
            .cpu()
            .numpy()
        )
    train_metrics = _selection_metrics(
        scores,
        return_delta,
        success_delta,
        source_return_delta,
        nearest_scores,
        train_queries,
    )
    val_metrics = _selection_metrics(
        scores,
        return_delta,
        success_delta,
        source_return_delta,
        nearest_scores,
        val_queries,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "method": "privileged_z_counterfactual_branch_selector",
            "source": str(args.input),
            "checkpoint": str(data["checkpoint"]),
            "residual_checkpoint": str(data["residual_checkpoint"]),
            "branch_bank": str(data["branch_bank"]),
            "seed_start": int(np.asarray(data["seed_start"]).item()),
            "num_envs": int(np.asarray(data["num_envs"]).item()),
            "query_batches": int(np.asarray(data["query_batches"]).item()),
            "candidates_per_query": int(np.asarray(data["candidates_per_query"]).item()),
            "max_rollout_steps": int(np.asarray(data["max_rollout_steps"]).item()),
            "input_dim": int(x.shape[1]),
            "hidden_dim": int(args.hidden_dim),
            "depth": int(args.depth),
            "model": best_state,
            "loss": args.loss,
            "feature_mean": feature_mean,
            "feature_std": feature_std,
            "history": history,
            "best_validation_mse": float(best_val),
            "train_metrics": train_metrics,
            "validation_metrics": val_metrics,
        },
        args.output,
    )
    print(
        {
            "output": str(args.output),
            "queries": query_count,
            "candidates_per_query": candidate_count,
            "best_validation_mse": float(best_val),
            "train": train_metrics,
            "validation": val_metrics,
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--validation-fraction", type=float, default=0.25)
    parser.add_argument("--loss", choices=["mse", "best_ce"], default="mse")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
