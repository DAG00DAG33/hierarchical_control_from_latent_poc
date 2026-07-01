#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import trange

from hcl_poc.config import load_config
from hcl_poc.incremental import (
    PRE_RL_PHASE_B_GOAL_TYPES,
    _load_phase7_privileged_episodes,
    _pre_rl_phase_b_goal,
    _runtime_metadata,
    train_pre_rl_phase_c_time_conditioned,
)
from hcl_poc.models import MLP
from hcl_poc.utils import Standardizer, default_device, ensure_dir, write_json


def _samples(
    episodes: list[dict[str, np.ndarray]],
    action_norm: Standardizer,
    horizon: int,
    control_freq: int,
    goal_type: str,
) -> tuple[np.ndarray, np.ndarray]:
    zero_action = action_norm.transform(np.zeros((1, 3), dtype=np.float32))[0]
    conditions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    for episode in episodes:
        states = episode["states"]
        actions = episode["actions"]
        normalized_actions = action_norm.transform(actions)
        for t in range(len(actions) - horizon):
            previous = normalized_actions[t - 1] if t > 0 else zero_action
            conditions.append(np.concatenate([states[t], previous]))
            targets.append(
                _pre_rl_phase_b_goal(
                    states[t : t + 1],
                    states[t + horizon : t + horizon + 1],
                    horizon,
                    control_freq,
                    goal_type,
                )[0]
            )
    return np.asarray(conditions, dtype=np.float32), np.asarray(targets, dtype=np.float32)


def _evaluate(
    model: MLP,
    x: np.ndarray,
    y: np.ndarray,
    input_norm: Standardizer,
    target_norm: Standardizer,
    device: torch.device,
) -> dict[str, float]:
    with torch.inference_mode():
        prediction = target_norm.inverse(
            model(torch.from_numpy(input_norm.transform(x)).to(device).float())
            .cpu()
            .numpy()
        )
    l2 = np.linalg.norm(prediction - y, axis=-1)
    return {
        "validation_goal_l2": float(np.mean(l2)),
        "validation_goal_mse": float(np.mean((prediction - y) ** 2)),
        "validation_goal_p90_l2": float(np.quantile(l2, 0.90)),
    }


def train(args: argparse.Namespace) -> Path:
    config = load_config(args.config)
    horizon = int(args.horizon)
    seed = int(args.seed)
    output = Path(args.output)
    if output.exists() and not args.force:
        print(output)
        return output

    low_path = train_pre_rl_phase_c_time_conditioned(
        config,
        horizon_steps=horizon,
        seed=seed,
        force=False,
    )
    low_checkpoint = torch.load(low_path, map_location="cpu", weights_only=False)
    action_norm = Standardizer.from_state_dict(low_checkpoint["action_norm"])
    train_episodes, val_episodes, data_metadata = _load_phase7_privileged_episodes(
        config,
        horizon,
        cap_train_to_usable=True,
    )
    control_freq = int(config.get("control_freq", 20))
    train_x, train_y = _samples(
        train_episodes,
        action_norm,
        horizon,
        control_freq,
        args.goal_type,
    )
    val_x, val_y = _samples(
        val_episodes,
        action_norm,
        horizon,
        control_freq,
        args.goal_type,
    )
    input_norm = Standardizer.fit(train_x)
    target_norm = Standardizer.fit(train_y)
    dataset = TensorDataset(
        torch.from_numpy(input_norm.transform(train_x)),
        torch.from_numpy(target_norm.transform(train_y)),
    )
    loader = DataLoader(
        dataset,
        batch_size=int(args.batch_size),
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    device = default_device()
    model = MLP(train_x.shape[-1], train_y.shape[-1], int(args.hidden_dim), depth=4).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr))
    best_state: dict[str, Any] | None = None
    best_l2 = float("inf")
    history = []
    for epoch in trange(1, int(args.epochs) + 1, desc=f"train {args.goal_type} high predictor"):
        model.train()
        total = 0.0
        for x, y in loader:
            x = x.to(device, non_blocking=True).float()
            y = y.to(device, non_blocking=True).float()
            loss = torch.mean((model(x) - y) ** 2)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += float(loss.detach().cpu()) * len(x)
        model.eval()
        metrics = _evaluate(model, val_x, val_y, input_norm, target_norm, device)
        history.append(
            {
                "epoch": epoch,
                "train_normalized_mse": float(total / len(dataset)),
                **metrics,
            }
        )
        if metrics["validation_goal_l2"] < best_l2:
            best_l2 = metrics["validation_goal_l2"]
            best_state = copy.deepcopy(model.state_dict())
    if best_state is None:
        raise RuntimeError(f"{args.goal_type} high predictor produced no checkpoint")

    model.load_state_dict(best_state)
    model.eval()
    metrics = _evaluate(model, val_x, val_y, input_norm, target_norm, device)
    current_goal = _pre_rl_phase_b_goal(
        val_x[:, :31],
        val_x[:, :31],
        horizon,
        control_freq,
        args.goal_type,
    )
    persistence_l2 = np.linalg.norm(current_goal - val_y, axis=-1)
    payload = {
        "method": "rl_reachability_goal_high_predictor",
        "goal_type": args.goal_type,
        "horizon_steps": horizon,
        "model": best_state,
        "condition_dim": int(train_x.shape[-1]),
        "target_dim": int(train_y.shape[-1]),
        "hidden_dim": int(args.hidden_dim),
        "input_norm": input_norm.state_dict(),
        "target_norm": target_norm.state_dict(),
        "action_norm": low_checkpoint["action_norm"],
        "low_checkpoint": str(low_path),
        **metrics,
        "persistence_goal_l2": float(np.mean(persistence_l2)),
        "persistence_goal_p90_l2": float(np.quantile(persistence_l2, 0.90)),
        "history": history,
        "data": {
            **data_metadata,
            "train_samples": int(len(train_x)),
            "validation_samples": int(len(val_x)),
        },
        "metadata": _runtime_metadata(config),
    }
    ensure_dir(output.parent)
    torch.save(payload, output)
    write_json(
        output.parent / "metrics.json",
        {
            key: value
            for key, value in payload.items()
            if key not in {"model", "input_norm", "target_norm", "action_norm", "history"}
        },
    )
    print(output)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/pusht_incremental.yaml")
    parser.add_argument("--goal-type", choices=PRE_RL_PHASE_B_GOAL_TYPES, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--force", action="store_true")
    train(parser.parse_args())


if __name__ == "__main__":
    main()
