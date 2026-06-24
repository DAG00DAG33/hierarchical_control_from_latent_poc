from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
from torch import nn

from hcl_poc.config import load_config
from hcl_poc.learned_interface import _low_condition_array
from hcl_poc.low_level_rl import _load_frozen
from hcl_poc.rl_rerun import _encode_rerun_frames, _rerun_base_config, _vector_dataset_path
from hcl_poc.utils import default_device, write_json


class _ActionMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, depth: int) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        current = input_dim
        for _ in range(depth):
            layers.extend([nn.Linear(current, hidden_dim), nn.SiLU()])
            current = hidden_dim
        layers.append(nn.Linear(current, 3))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _summary(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p90": float(np.quantile(values, 0.9)),
        "max": float(np.max(values)),
    }


def _sample_rows(
    h5: h5py.File,
    frozen: Any,
    device: torch.device,
    samples: int,
    horizon: int,
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    keys = sorted(key for key in h5.keys() if key.startswith("batch_"))
    max_steps = int(h5["meta"].attrs["max_steps"])
    current_frames: list[np.ndarray] = []
    future_frames: list[np.ndarray] = []
    previous_actions: list[np.ndarray] = []
    target_actions: list[np.ndarray] = []
    for _ in range(samples):
        key = str(rng.choice(keys))
        group = h5[key]
        env_index = int(rng.integers(0, int(group.attrs["num_envs"])))
        timestep = int(rng.integers(0, max_steps - horizon + 1))
        current_frames.append(
            np.concatenate(
                [
                    np.asarray(group["dino"][timestep, env_index], dtype=np.float32),
                    np.asarray(group["proprio"][timestep, env_index], dtype=np.float32),
                ],
                axis=-1,
            )
        )
        future_frames.append(
            np.concatenate(
                [
                    np.asarray(group["dino"][timestep + horizon, env_index], dtype=np.float32),
                    np.asarray(group["proprio"][timestep + horizon, env_index], dtype=np.float32),
                ],
                axis=-1,
            )
        )
        previous_actions.append(
            np.asarray(group["previous_executed_actions"][timestep, env_index], dtype=np.float32)
        )
        target_actions.append(
            np.asarray(group["executed_actions"][timestep, env_index], dtype=np.float32)
        )
    current = np.stack(current_frames, axis=0)
    future = np.stack(future_frames, axis=0)
    previous = np.stack(previous_actions, axis=0)
    target = np.stack(target_actions, axis=0)
    current_z = _encode_rerun_frames(frozen, current, device)
    future_z = _encode_rerun_frames(frozen, future, device)
    normalized_obs = frozen.frame_norm.transform(current)
    normalized_previous = frozen.action_norm.transform(previous)
    normalized_target = frozen.action_norm.transform(target)
    remaining = np.ones((samples, 1), dtype=np.float32)
    full_condition = _low_condition_array(
        normalized_obs,
        current_z,
        future_z,
        normalized_previous,
        remaining,
        frozen.conditioning,
    )
    if frozen.conditioning == "concat" or frozen.conditioning == "film":
        goal_features = future_z
    elif frozen.conditioning == "delta":
        goal_features = future_z - current_z
    elif frozen.conditioning == "relation":
        goal_features = np.concatenate([current_z, future_z], axis=-1)
    else:
        raise ValueError(f"Unknown conditioning: {frozen.conditioning}")
    return {
        "obs": normalized_obs.astype(np.float32),
        "goal": goal_features.astype(np.float32),
        "prev": normalized_previous.astype(np.float32),
        "remaining": remaining.astype(np.float32),
        "target_norm": normalized_target.astype(np.float32),
        "target_raw": target.astype(np.float32),
        "full_condition": full_condition.astype(np.float32),
    }


def _concat_blocks(data: dict[str, np.ndarray], blocks: list[str]) -> np.ndarray:
    return np.concatenate([data[block] for block in blocks], axis=-1).astype(np.float32)


def _train_variant(
    train_x: np.ndarray,
    train_y: np.ndarray,
    val_x: np.ndarray,
    val_y: np.ndarray,
    val_raw_y: np.ndarray,
    frozen: Any,
    device: torch.device,
    hidden_dim: int,
    depth: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
) -> dict[str, Any]:
    torch.manual_seed(seed)
    model = _ActionMLP(train_x.shape[1], hidden_dim, depth).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    x = torch.from_numpy(train_x).to(device)
    y = torch.from_numpy(train_y).to(device)
    indices = np.arange(train_x.shape[0])
    losses: list[float] = []
    for _epoch in range(epochs):
        np.random.shuffle(indices)
        for start in range(0, len(indices), batch_size):
            mb = indices[start : start + batch_size]
            pred = model(x[mb])
            loss = torch.mean((pred - y[mb]).square())
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
    with torch.inference_mode():
        pred_norm = model(torch.from_numpy(val_x).to(device)).cpu().numpy().astype(np.float32)
    pred_raw = frozen.action_norm.inverse(pred_norm)
    error = pred_raw - val_raw_y
    return {
        "input_dim": int(train_x.shape[1]),
        "train_loss_last": float(losses[-1]),
        "val_norm_mse": float(np.mean((pred_norm - val_y) ** 2)),
        "val_raw_mae": float(np.mean(np.abs(error))),
        "val_raw_l2": _summary(np.linalg.norm(error, axis=-1)),
    }


@torch.inference_mode()
def _frozen_low_reference(
    val_data: dict[str, np.ndarray],
    frozen: Any,
    device: torch.device,
) -> dict[str, Any]:
    condition = torch.from_numpy(val_data["full_condition"]).to(device).float()
    pred_norm = frozen.low_model(condition).cpu().numpy().astype(np.float32)
    pred_raw = frozen.action_norm.inverse(pred_norm)
    error = pred_raw - val_data["target_raw"]
    return {
        "input_dim": int(val_data["full_condition"].shape[1]),
        "val_norm_mse": float(np.mean((pred_norm - val_data["target_norm"]) ** 2)),
        "val_raw_mae": float(np.mean(np.abs(error))),
        "val_raw_l2": _summary(np.linalg.norm(error, axis=-1)),
    }


def run_action_block_prediction(args: argparse.Namespace) -> Path:
    config = load_config(args.config)
    dataset_path = Path(args.dataset) if args.dataset else _vector_dataset_path(config)
    if not dataset_path.exists():
        raise FileNotFoundError(dataset_path)
    if args.train_samples <= 0 or args.val_samples <= 0:
        raise ValueError("--train-samples and --val-samples must be positive")

    device = default_device()
    frozen = _load_frozen(_rerun_base_config(config), args.n_demo, args.seed, device)
    rng = np.random.default_rng(args.seed + 991_000)
    with h5py.File(dataset_path, "r") as h5:
        train = _sample_rows(h5, frozen, device, args.train_samples, args.horizon, rng)
        val = _sample_rows(h5, frozen, device, args.val_samples, args.horizon, rng)

    variants = {
        "obs": ["obs"],
        "goal": ["goal"],
        "prev": ["prev"],
        "obs_prev": ["obs", "prev"],
        "goal_prev": ["goal", "prev"],
        "obs_goal": ["obs", "goal"],
        "full": ["obs", "goal", "prev", "remaining"],
    }
    results: dict[str, Any] = {
        "frozen_low_policy": _frozen_low_reference(val, frozen, device)
    }
    for name, blocks in variants.items():
        results[name] = _train_variant(
            _concat_blocks(train, blocks),
            train["target_norm"],
            _concat_blocks(val, blocks),
            val["target_norm"],
            val["target_raw"],
            frozen,
            device,
            hidden_dim=args.hidden_dim,
            depth=args.depth,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            seed=args.seed + len(results),
        )
        results[name]["blocks"] = blocks

    output = args.output or Path("rl_rerun_action_block_prediction.json")
    write_json(
        output,
        {
            "method": "rl_rerun_action_block_prediction",
            "dataset": str(dataset_path),
            "n_demo": args.n_demo,
            "seed": args.seed,
            "horizon": args.horizon,
            "train_samples": args.train_samples,
            "val_samples": args.val_samples,
            "hidden_dim": args.hidden_dim,
            "depth": args.depth,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "conditioning": frozen.conditioning,
            "results": results,
            "interpretation": (
                "Compares one-step teacher-action predictability from low-level "
                "condition blocks. If obs-only is close to full and goal-only is poor, "
                "the supervised label itself is current-state dominated."
            ),
        },
    )
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/pusht_incremental.yaml")
    parser.add_argument("--dataset")
    parser.add_argument("--n-demo", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument("--train-samples", type=int, default=8192)
    parser.add_argument("--val-samples", type=int, default=2048)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--output", type=Path)
    path = run_action_block_prediction(parser.parse_args())
    print(path)


if __name__ == "__main__":
    main()
