from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch

from hcl_poc.config import load_config
from hcl_poc.learned_interface import _low_condition_array
from hcl_poc.low_level_rl import DirectLowActorCritic, _load_frozen
from hcl_poc.rl_rerun import _encode_rerun_frames, _rerun_base_config, _vector_dataset_path
from hcl_poc.utils import default_device, write_json


def _summary(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p90": float(np.quantile(values, 0.9)),
        "max": float(np.max(values)),
    }


def _parse_policy(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("policy must be LABEL=CHECKPOINT")
    label, path = value.split("=", 1)
    checkpoint = Path(path)
    if not label:
        raise argparse.ArgumentTypeError("policy label must be non-empty")
    if not checkpoint.exists():
        raise argparse.ArgumentTypeError(f"checkpoint does not exist: {checkpoint}")
    return label, checkpoint


def _load_r3_policy(
    checkpoint_path: Path,
    frozen: Any,
    device: torch.device,
) -> DirectLowActorCritic:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    recipe = checkpoint["recipe"]
    method = str(recipe.get("method", ""))
    if not method.startswith("r3_direct"):
        raise ValueError("This diagnostic currently expects R3 direct checkpoints")
    agent = DirectLowActorCritic(
        frozen.low_model,
        frozen.action_norm.mean,
        frozen.action_norm.std,
        int(checkpoint["condition_dim"]),
        width=int(recipe["actor_critic_width"]),
        depth=int(recipe["actor_critic_depth"]),
        initial_logstd=float(recipe["initial_logstd"]),
    ).to(device)
    agent.load_state_dict(checkpoint["agent"])
    agent.eval()
    return agent


def _sample_conditions(
    h5: h5py.File,
    frozen: Any,
    device: torch.device,
    samples: int,
    horizon: int,
    rng: np.random.Generator,
) -> np.ndarray:
    keys = sorted(key for key in h5.keys() if key.startswith("batch_"))
    max_steps = int(h5["meta"].attrs["max_steps"])
    current_frames: list[np.ndarray] = []
    future_frames: list[np.ndarray] = []
    previous_actions: list[np.ndarray] = []
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
    current = np.stack(current_frames, axis=0)
    future = np.stack(future_frames, axis=0)
    previous = np.stack(previous_actions, axis=0)
    current_z = _encode_rerun_frames(frozen, current, device)
    future_z = _encode_rerun_frames(frozen, future, device)
    return _low_condition_array(
        frozen.frame_norm.transform(current),
        current_z,
        future_z,
        frozen.action_norm.transform(previous),
        np.ones((samples, 1), dtype=np.float32),
        frozen.conditioning,
    )


def _block_ranges(frozen: Any) -> dict[str, tuple[int, int]]:
    goal_features = 2 * frozen.goal_dim if frozen.conditioning == "relation" else frozen.goal_dim
    frame_start = 0
    frame_stop = frozen.frame_dim
    goal_start = frame_stop
    goal_stop = goal_start + goal_features
    previous_start = goal_stop
    previous_stop = previous_start + 3
    remaining_start = previous_stop
    remaining_stop = remaining_start + 1
    return {
        "observation": (frame_start, frame_stop),
        "goal": (goal_start, goal_stop),
        "previous_action": (previous_start, previous_stop),
        "remaining": (remaining_start, remaining_stop),
    }


@torch.inference_mode()
def _policy_actions(
    condition: torch.Tensor,
    frozen: Any,
    policy: DirectLowActorCritic | None,
) -> torch.Tensor:
    if policy is None:
        normalized = frozen.low_model(condition)
        return torch.from_numpy(
            frozen.action_norm.inverse(normalized.cpu().numpy().astype(np.float32))
        ).to(condition.device)
    return policy.mean_action(condition)


@torch.inference_mode()
def run_block_sensitivity(args: argparse.Namespace) -> Path:
    config = load_config(args.config)
    dataset_path = Path(args.dataset) if args.dataset else _vector_dataset_path(config)
    if not dataset_path.exists():
        raise FileNotFoundError(dataset_path)
    if args.samples <= 0 or args.batch_size <= 0:
        raise ValueError("--samples and --batch-size must be positive")
    if args.horizon <= 0:
        raise ValueError("--horizon must be positive")

    device = default_device()
    frozen = _load_frozen(_rerun_base_config(config), args.n_demo, args.seed, device)
    policies: dict[str, DirectLowActorCritic | None] = {"frozen": None}
    for label, checkpoint in args.policy:
        if label in policies:
            raise ValueError(f"Duplicate policy label: {label}")
        policies[label] = _load_r3_policy(checkpoint, frozen, device)

    rng = np.random.default_rng(args.seed + 881_000)
    conditions: list[np.ndarray] = []
    with h5py.File(dataset_path, "r") as h5:
        processed = 0
        while processed < args.samples:
            batch_samples = min(args.batch_size, args.samples - processed)
            conditions.append(
                _sample_conditions(h5, frozen, device, batch_samples, args.horizon, rng)
            )
            processed += batch_samples
    condition_np = np.concatenate(conditions, axis=0)
    condition = torch.from_numpy(condition_np).to(device).float()
    ranges = _block_ranges(frozen)
    rng = np.random.default_rng(args.seed + 882_000)

    result: dict[str, Any] = {
        "method": "rl_rerun_condition_block_sensitivity",
        "dataset": str(dataset_path),
        "n_demo": args.n_demo,
        "seed": args.seed,
        "samples": args.samples,
        "horizon": args.horizon,
        "condition_dim": int(condition_np.shape[1]),
        "block_ranges": {name: [int(a), int(b)] for name, (a, b) in ranges.items()},
        "policies": {},
        "policy_checkpoints": {label: str(path) for label, path in args.policy},
        "interpretation": (
            "Measures deterministic action changes when condition blocks are zeroed "
            "or shuffled across same-batch samples."
        ),
    }
    for label, policy in policies.items():
        base_action = _policy_actions(condition, frozen, policy)
        metrics: dict[str, Any] = {}
        for block, (start, stop) in ranges.items():
            zeroed = condition.clone()
            zeroed[:, start:stop] = 0.0
            zero_action = _policy_actions(zeroed, frozen, policy)
            zero_l2 = torch.linalg.vector_norm(base_action - zero_action, dim=-1).cpu().numpy()

            shuffled = condition.clone()
            permutation = torch.as_tensor(
                rng.permutation(condition.shape[0]),
                device=device,
                dtype=torch.long,
            )
            shuffled[:, start:stop] = shuffled[permutation, start:stop]
            shuffle_action = _policy_actions(shuffled, frozen, policy)
            shuffle_l2 = (
                torch.linalg.vector_norm(base_action - shuffle_action, dim=-1).cpu().numpy()
            )

            block_norm = np.linalg.norm(condition_np[:, start:stop], axis=-1)
            metrics[block] = {
                "zero_action_l2": _summary(zero_l2),
                "shuffle_action_l2": _summary(shuffle_l2),
                "block_l2": _summary(block_norm),
            }
        result["policies"][label] = metrics

    output = args.output or Path("rl_rerun_condition_block_sensitivity.json")
    write_json(output, result)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/pusht_incremental.yaml")
    parser.add_argument("--dataset")
    parser.add_argument("--n-demo", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--samples", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument("--policy", action="append", type=_parse_policy, default=[])
    parser.add_argument("--output", type=Path)
    path = run_block_sensitivity(parser.parse_args())
    print(path)


if __name__ == "__main__":
    main()
