from __future__ import annotations

import glob
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
from rich.console import Console

from hcl_poc.config import Config
from hcl_poc.incremental import _load_phase6_train_episodes
from hcl_poc.learned_interface import (
    _encode_effect_array,
    _load_hierarchy,
    _load_representation,
    _low_condition_array,
    prepare_learned_interface_episodes,
    train_learned_interface_hierarchy,
)
from hcl_poc.low_level_rl import _load_frozen
from hcl_poc.utils import Standardizer, default_device, ensure_dir, write_json
from hcl_poc.vae_scaling import VAE_CANDIDATE, vae_scaling_config

console = Console()


def summarize_goal_diagnostics(
    payload: dict[str, Any],
    *,
    min_goal_shuffle_l2: float = 0.1,
    min_goal_sensitivity_l2: float = 0.1,
) -> dict[str, Any]:
    summary = dict(payload["summary"])
    frame_shuffle = float(summary["frame_shuffle_action_change_l2"])
    goal_shuffle = float(summary["goal_shuffle_action_change_l2"])
    max_goal_sensitivity = float(summary["max_goal_sensitivity_l2"])
    goal_to_frame_ratio = goal_shuffle / max(frame_shuffle, 1e-8)
    offline_goal_use_pass = (
        goal_shuffle >= min_goal_shuffle_l2
        or max_goal_sensitivity >= min_goal_sensitivity_l2
    )
    gate_status = (
        "offline_goal_use_pass"
        if offline_goal_use_pass
        else "reject_low_goal_use"
    )
    return {
        "representation": payload["representation"],
        "n_demo": payload.get("n_demo"),
        "seed": payload.get("seed"),
        "samples": payload.get("samples"),
        "conditioning": payload.get("conditioning"),
        "hierarchy_checkpoint": payload.get("hierarchy_checkpoint"),
        "goal_shuffle_action_change_l2": goal_shuffle,
        "frame_shuffle_action_change_l2": frame_shuffle,
        "goal_to_frame_action_change_ratio": float(goal_to_frame_ratio),
        "previous_action_shuffle_action_change_l2": float(
            summary["previous_action_shuffle_action_change_l2"]
        ),
        "max_goal_sensitivity_l2": max_goal_sensitivity,
        "goal_shuffle_mae_gap": float(summary["goal_shuffle_mae_gap"]),
        "offline_goal_use_pass": offline_goal_use_pass,
        "gate_status": gate_status,
        "gate_note": (
            "Goal-use diagnostics are only a rejection gate; candidates that pass "
            "still need closed-loop imitation quality and local-to-task transfer."
        ),
    }


def _diagnostic_sort_key(row: dict[str, Any]) -> tuple[int, str]:
    n_demo = row["n_demo"]
    n_value = int(n_demo) if n_demo is not None else -1
    return n_value, str(row["representation"])


def _write_goal_diagnostics_markdown(output_path: Path, rows: list[dict[str, Any]]) -> None:
    markdown_path = output_path.with_suffix(".md")
    lines = [
        "# Goal Diagnostics Gate",
        "",
        "| representation | N | status | goal shuffle L2 | frame shuffle L2 | goal/frame | max horizon sensitivity | goal MAE gap |",
        "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {representation} | {n_demo} | {gate_status} | {goal:.4f} | {frame:.4f} | "
            "{ratio:.4f} | {sensitivity:.4f} | {mae:.4f} |".format(
                representation=row["representation"],
                n_demo=row["n_demo"],
                gate_status=row["gate_status"],
                goal=row["goal_shuffle_action_change_l2"],
                frame=row["frame_shuffle_action_change_l2"],
                ratio=row["goal_to_frame_action_change_ratio"],
                sensitivity=row["max_goal_sensitivity_l2"],
                mae=row["goal_shuffle_mae_gap"],
            )
        )
    lines.extend(
        [
            "",
            "Passing this offline gate is not a promotion criterion. It only means the low-level policy reacts enough to goals to justify later closed-loop checks.",
            "",
        ]
    )
    markdown_path.write_text("\n".join(lines))


def aggregate_goal_diagnostics(
    input_glob: str,
    *,
    output_path: Path,
    min_goal_shuffle_l2: float = 0.1,
    min_goal_sensitivity_l2: float = 0.1,
    force: bool = False,
) -> Path:
    if output_path.exists() and not force:
        console.print(f"Goal diagnostics aggregate exists: {output_path}")
        return output_path
    input_paths = sorted(Path(path) for path in glob.glob(input_glob, recursive=True))
    if not input_paths:
        raise ValueError(f"No goal diagnostics matched: {input_glob}")
    rows = []
    for path in input_paths:
        payload = json.loads(path.read_text())
        if payload.get("method") != "learned_interface_goal_diagnostics":
            raise ValueError(f"Unexpected goal diagnostics method in {path}")
        row = summarize_goal_diagnostics(
            payload,
            min_goal_shuffle_l2=min_goal_shuffle_l2,
            min_goal_sensitivity_l2=min_goal_sensitivity_l2,
        )
        row["path"] = str(path)
        rows.append(row)
    rows.sort(key=_diagnostic_sort_key)
    output_payload = {
        "method": "aggregate_goal_diagnostics",
        "input_glob": input_glob,
        "min_goal_shuffle_l2": min_goal_shuffle_l2,
        "min_goal_sensitivity_l2": min_goal_sensitivity_l2,
        "rows": rows,
        "counts": {
            "total": len(rows),
            "offline_goal_use_pass": int(
                sum(bool(row["offline_goal_use_pass"]) for row in rows)
            ),
            "reject_low_goal_use": int(
                sum(row["gate_status"] == "reject_low_goal_use" for row in rows)
            ),
        },
        "note": (
            "This aggregate is a hard rejection gate for low goal-use only. "
            "Passing candidates still require closed-loop imitation and local-to-task "
            "transfer checks before PPO scaling."
        ),
    }
    write_json(output_path, output_payload)
    _write_goal_diagnostics_markdown(output_path, rows)
    console.print(output_payload)
    return output_path


def _result_dir(config: Config, n_demo: int, seed: int, candidate: str) -> Path:
    return ensure_dir(
        config.path_value("paths.incremental_results_dir")
        / "goal_diagnostics"
        / f"n{n_demo}"
        / f"seed{seed}"
        / candidate
    )


def _load_encoded_validation_goals(
    config: Config, n_demo: int, seed: int, candidate: str
) -> list[np.ndarray]:
    point_config = (
        vae_scaling_config(config, n_demo)
        if candidate == VAE_CANDIDATE
        else config
    )
    encoded_path = (
        point_config.path_value("paths.incremental_artifact_dir")
        / "learned_interface"
        / candidate
        / f"seed{seed}"
        / "encoded_episodes.pt"
    )
    if not encoded_path.exists():
        encoded_path = prepare_learned_interface_episodes(
            point_config, candidate, seed=seed, force=False
        )
    payload = torch.load(encoded_path, map_location="cpu", weights_only=False)
    if "validation_goals" in payload:
        return [
            np.asarray(goals, dtype=np.float32)
            for goals in payload["validation_goals"]
        ]
    if "validation" in payload:
        validation_goals = []
        for episode in payload["validation"]:
            if "goals" not in episode:
                raise ValueError(
                    f"Missing goals in validation episode from {encoded_path}"
                )
            validation_goals.append(np.asarray(episode["goals"], dtype=np.float32))
        return validation_goals
    raise ValueError(f"Missing validation goals in {encoded_path}")


def _load_candidate_frozen(
    config: Config,
    n_demo: int,
    candidate: str,
    seed: int,
    device: torch.device,
):
    if candidate == VAE_CANDIDATE:
        return _load_frozen(config, n_demo, seed, device), vae_scaling_config(config, n_demo)
    point_config = config
    hierarchy_path = train_learned_interface_hierarchy(
        point_config, candidate, seed=seed, force=False
    )
    checkpoint = torch.load(hierarchy_path, map_location="cpu", weights_only=False)
    _high_model, low_model = _load_hierarchy(checkpoint, device)
    encoder, representation = _load_representation(
        Path(checkpoint["representation_checkpoint"]), device
    )
    low_model.eval()
    low_model.requires_grad_(False)
    encoder.eval()
    encoder.requires_grad_(False)
    return (
        SimpleNamespace(
            low_model=low_model,
            encoder=encoder,
            encoder_type=str(representation["encoder_type"]),
            frame_norm=Standardizer.from_state_dict(checkpoint["frame_norm"]),
            representation_frame_norm=Standardizer.from_state_dict(
                representation["frame_norm"]
            ),
            goal_norm=Standardizer.from_state_dict(checkpoint["goal_norm"]),
            action_norm=Standardizer.from_state_dict(checkpoint["action_norm"]),
            horizon_steps=int(checkpoint["horizon_steps"]),
            conditioning=str(checkpoint.get("conditioning", "concat")),
            frame_dim=int(checkpoint["frame_dim"]),
            goal_dim=int(checkpoint["goal_dim"]),
            checkpoint_path=Path(hierarchy_path),
        ),
        point_config,
    )


def _sample_validation_rows(
    frame_episodes: list[dict[str, np.ndarray]],
    goal_episodes: list[np.ndarray],
    *,
    count: int,
    max_horizon: int,
    rng: np.random.Generator,
) -> list[tuple[int, int]]:
    candidates: list[tuple[int, int]] = []
    for ep_index, (frames, goals) in enumerate(
        zip(frame_episodes, goal_episodes, strict=True)
    ):
        limit = min(len(frames["actions"]), len(frames["frames"]), len(goals) - max_horizon)
        candidates.extend((ep_index, t) for t in range(max(0, limit)))
    if not candidates:
        raise ValueError("No validation rows support requested goal horizons")
    indices = rng.choice(len(candidates), size=min(count, len(candidates)), replace=False)
    return [candidates[int(index)] for index in indices]


@torch.inference_mode()
def _predict_actions(
    low_model: torch.nn.Module,
    action_mean: np.ndarray,
    action_std: np.ndarray,
    condition_np: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    normalized = low_model(torch.from_numpy(condition_np).to(device).float())
    return (
        normalized.cpu().numpy().astype(np.float32) * action_std.reshape(1, -1)
        + action_mean.reshape(1, -1)
    )


def _mean_l2(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.linalg.norm(left - right, axis=-1).mean())


def _mean_mae(prediction: np.ndarray, target: np.ndarray) -> float:
    return float(np.abs(prediction - target).mean())


def _condition_blocks(
    frame_dim: int, goal_dim: int, action_dim: int, conditioning: str
) -> dict[str, slice]:
    start = 0
    blocks = {"frame": slice(start, start + frame_dim)}
    start += frame_dim
    if conditioning in {"concat", "film", "goal_residual"}:
        blocks["goal"] = slice(start, start + goal_dim)
        start += goal_dim
    elif conditioning == "delta":
        blocks["goal_delta"] = slice(start, start + goal_dim)
        start += goal_dim
    elif conditioning == "relation":
        blocks["current_latent"] = slice(start, start + goal_dim)
        start += goal_dim
        blocks["goal"] = slice(start, start + goal_dim)
        start += goal_dim
    else:
        raise ValueError(f"Unknown goal conditioning: {conditioning}")
    blocks["previous_action"] = slice(start, start + action_dim)
    start += action_dim
    blocks["remaining"] = slice(start, start + 1)
    return blocks


def _build_conditions(
    frame_norm,
    goal_norm,
    action_norm,
    frame_episodes: list[dict[str, np.ndarray]],
    goal_episodes: list[np.ndarray],
    rows: list[tuple[int, int]],
    *,
    horizon: int,
    base_horizon: int,
    conditioning: str,
    encoder_type: str = "state",
    encoder: torch.nn.Module | None = None,
    representation_frame_norm: Standardizer | None = None,
    device: torch.device | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if encoder_type == "effect" and conditioning not in {
        "concat",
        "film",
        "goal_residual",
    }:
        raise ValueError("Effect-code diagnostics require absolute-goal conditioning")
    frames = []
    current_latents = []
    goals = []
    effect_start_frames = []
    effect_future_frames = []
    previous_actions = []
    remaining = []
    targets = []
    for ep_index, t in rows:
        episode = frame_episodes[ep_index]
        goal_episode = goal_episodes[ep_index]
        goal_t = min(t + horizon, len(goal_episode) - 1)
        frames.append(episode["frames"][t])
        if encoder_type == "effect":
            effect_start_frames.append(episode["frames"][t])
            effect_future_frames.append(episode["frames"][goal_t])
            current_latents.append(np.zeros_like(goal_episode[goal_t]))
        else:
            current_latents.append(goal_episode[t])
            goals.append(goal_episode[goal_t])
        if t == 0:
            previous_actions.append(np.zeros_like(episode["actions"][0]))
        else:
            previous_actions.append(episode["actions"][t - 1])
        targets.append(episode["actions"][t])
        remaining.append(
            [np.clip(max(horizon, 1) / max(base_horizon, 1), 0.0, 1.0)]
        )
    frame_np = frame_norm.transform(np.asarray(frames, dtype=np.float32))
    current_np = goal_norm.transform(np.asarray(current_latents, dtype=np.float32))
    if encoder_type == "effect":
        if encoder is None or representation_frame_norm is None or device is None:
            raise ValueError("Effect-code diagnostics require encoder and frame norm")
        goals_np_raw = _encode_effect_array(
            encoder,
            representation_frame_norm,
            np.asarray(effect_start_frames, dtype=np.float32),
            np.asarray(effect_future_frames, dtype=np.float32),
            np.ones(len(effect_start_frames), dtype=np.float32),
            device,
        )
        goals = [row for row in goals_np_raw]
    goal_np = goal_norm.transform(np.asarray(goals, dtype=np.float32))
    previous_np = action_norm.transform(np.asarray(previous_actions, dtype=np.float32))
    remaining_np = np.asarray(remaining, dtype=np.float32)
    target_np = np.asarray(targets, dtype=np.float32)
    condition_np = _low_condition_array(
        frame_np,
        current_np,
        goal_np,
        previous_np,
        remaining_np,
        conditioning,
    )
    return condition_np.astype(np.float32), target_np


def _build_mixed_horizon_conditions(
    frame_norm,
    goal_norm,
    action_norm,
    frame_episodes: list[dict[str, np.ndarray]],
    goal_episodes: list[np.ndarray],
    rows: list[tuple[int, int]],
    *,
    horizons: tuple[int, ...],
    base_horizon: int,
    conditioning: str,
    rng: np.random.Generator,
    encoder_type: str = "state",
    encoder: torch.nn.Module | None = None,
    representation_frame_norm: Standardizer | None = None,
    device: torch.device | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if encoder_type == "effect" and conditioning not in {
        "concat",
        "film",
        "goal_residual",
    }:
        raise ValueError("Effect-code diagnostics require absolute-goal conditioning")
    choices = rng.choice(np.asarray(horizons, dtype=np.int64), size=len(rows))
    frames = []
    current_latents = []
    goals = []
    effect_start_frames = []
    effect_future_frames = []
    previous_actions = []
    remaining = []
    targets = []
    for (ep_index, t), horizon_raw in zip(rows, choices, strict=True):
        horizon = int(horizon_raw)
        episode = frame_episodes[ep_index]
        goal_episode = goal_episodes[ep_index]
        goal_t = min(t + horizon, len(goal_episode) - 1)
        frames.append(episode["frames"][t])
        if encoder_type == "effect":
            effect_start_frames.append(episode["frames"][t])
            effect_future_frames.append(episode["frames"][goal_t])
            current_latents.append(np.zeros_like(goal_episode[goal_t]))
        else:
            current_latents.append(goal_episode[t])
            goals.append(goal_episode[goal_t])
        if t == 0:
            previous_actions.append(np.zeros_like(episode["actions"][0]))
        else:
            previous_actions.append(episode["actions"][t - 1])
        targets.append(episode["actions"][t])
        remaining.append(
            [np.clip(max(horizon, 1) / max(base_horizon, 1), 0.0, 1.0)]
        )
    frame_np = frame_norm.transform(np.asarray(frames, dtype=np.float32))
    current_np = goal_norm.transform(np.asarray(current_latents, dtype=np.float32))
    if encoder_type == "effect":
        if encoder is None or representation_frame_norm is None or device is None:
            raise ValueError("Effect-code diagnostics require encoder and frame norm")
        goals_np_raw = _encode_effect_array(
            encoder,
            representation_frame_norm,
            np.asarray(effect_start_frames, dtype=np.float32),
            np.asarray(effect_future_frames, dtype=np.float32),
            np.ones(len(effect_start_frames), dtype=np.float32),
            device,
        )
        goals = [row for row in goals_np_raw]
    goal_np = goal_norm.transform(np.asarray(goals, dtype=np.float32))
    previous_np = action_norm.transform(np.asarray(previous_actions, dtype=np.float32))
    remaining_np = np.asarray(remaining, dtype=np.float32)
    target_np = np.asarray(targets, dtype=np.float32)
    condition_np = _low_condition_array(
        frame_np,
        current_np,
        goal_np,
        previous_np,
        remaining_np,
        conditioning,
    )
    return condition_np.astype(np.float32), target_np


def learned_interface_goal_diagnostics(
    config: Config,
    *,
    n_demo: int,
    candidate: str = VAE_CANDIDATE,
    seed: int = 0,
    samples: int = 5000,
    horizons: tuple[int, ...] = (2, 5, 10),
    output_path: Path | None = None,
    force: bool = False,
) -> Path:
    output_path = output_path or _result_dir(config, n_demo, seed, candidate) / "diagnostics.json"
    if output_path.exists() and not force:
        console.print(f"Goal diagnostics exist: {output_path}")
        return output_path
    if not horizons:
        raise ValueError("At least one horizon is required")
    device = default_device()
    frozen, point_config = _load_candidate_frozen(config, n_demo, candidate, seed, device)
    encoder_type = str(getattr(frozen, "encoder_type", "state"))
    encoder = getattr(frozen, "encoder", None)
    representation_frame_norm = getattr(frozen, "representation_frame_norm", None)
    _train, validation, _metadata = _load_phase6_train_episodes(point_config)
    validation_goals_raw = _load_encoded_validation_goals(config, n_demo, seed, candidate)
    validation_goals = [
        frozen.goal_norm.transform(goals).astype(np.float32)
        for goals in validation_goals_raw
    ]
    rng = np.random.default_rng(seed + 24_681)
    rows = _sample_validation_rows(
        validation,
        validation_goals_raw,
        count=samples,
        max_horizon=max(horizons),
        rng=rng,
    )

    action_mean = frozen.action_norm.mean.astype(np.float32)
    action_std = frozen.action_norm.std.astype(np.float32)
    by_horizon: dict[str, Any] = {}
    predictions: dict[int, np.ndarray] = {}
    conditions: dict[int, np.ndarray] = {}
    targets: dict[int, np.ndarray] = {}
    for horizon in horizons:
        condition_np, target_np = _build_conditions(
            frozen.frame_norm,
            frozen.goal_norm,
            frozen.action_norm,
            validation,
            validation_goals_raw,
            rows,
            horizon=horizon,
            base_horizon=frozen.horizon_steps,
            conditioning=frozen.conditioning,
            encoder_type=encoder_type,
            encoder=encoder,
            representation_frame_norm=representation_frame_norm,
            device=device,
        )
        pred = _predict_actions(
            frozen.low_model, action_mean, action_std, condition_np, device
        )
        conditions[horizon] = condition_np
        predictions[horizon] = pred
        targets[horizon] = target_np
        by_horizon[str(horizon)] = {
            "action_mae": _mean_mae(pred, target_np),
            "action_l2_mean": float(np.linalg.norm(pred, axis=-1).mean()),
        }

    sensitivity: dict[str, float] = {}
    for left_index, left in enumerate(horizons):
        for right in horizons[left_index + 1 :]:
            sensitivity[f"{left}_vs_{right}"] = _mean_l2(
                predictions[left], predictions[right]
            )

    reference_horizon = min(frozen.horizon_steps, max(horizons))
    if reference_horizon not in conditions:
        reference_horizon = horizons[-1]
    reference_condition, reference_target = _build_mixed_horizon_conditions(
        frozen.frame_norm,
        frozen.goal_norm,
        frozen.action_norm,
        validation,
        validation_goals_raw,
        rows,
        horizons=horizons,
        base_horizon=frozen.horizon_steps,
        conditioning=frozen.conditioning,
        rng=rng,
        encoder_type=encoder_type,
        encoder=encoder,
        representation_frame_norm=representation_frame_norm,
        device=device,
    )
    reference_prediction = _predict_actions(
        frozen.low_model, action_mean, action_std, reference_condition, device
    )
    blocks = _condition_blocks(
        frozen.frame_dim,
        frozen.goal_dim,
        action_mean.shape[0],
        frozen.conditioning,
    )
    shuffle: dict[str, Any] = {}
    for name, block in blocks.items():
        shuffled = reference_condition.copy()
        order = rng.permutation(len(shuffled))
        shuffled[:, block] = shuffled[order, block]
        pred = _predict_actions(
            frozen.low_model, action_mean, action_std, shuffled, device
        )
        shuffle[name] = {
            "action_change_l2": _mean_l2(reference_prediction, pred),
            "action_mae": _mean_mae(pred, reference_target),
            "mae_gap_vs_correct": _mean_mae(pred, reference_target)
            - _mean_mae(reference_prediction, reference_target),
        }

    payload: dict[str, Any] = {
        "method": "learned_interface_goal_diagnostics",
        "representation": candidate,
        "n_demo": n_demo,
        "seed": seed,
        "samples": len(rows),
        "horizons": list(horizons),
        "reference_horizon": reference_horizon,
        "shuffle_reference": "mixed_horizons",
        "hierarchy_checkpoint": str(frozen.checkpoint_path),
        "conditioning": frozen.conditioning,
        "by_horizon": by_horizon,
        "same_state_goal_sensitivity_l2": sensitivity,
        "condition_block_shuffle": shuffle,
        "summary": {
            "max_goal_sensitivity_l2": float(max(sensitivity.values()))
            if sensitivity
            else 0.0,
            "goal_shuffle_action_change_l2": shuffle["goal"][
                "action_change_l2"
            ]
            if "goal" in shuffle
            else shuffle["goal_delta"]["action_change_l2"],
            "goal_shuffle_mae_gap": (
                shuffle["goal"]["mae_gap_vs_correct"]
                if "goal" in shuffle
                else shuffle["goal_delta"]["mae_gap_vs_correct"]
            ),
            "frame_shuffle_action_change_l2": shuffle["frame"][
                "action_change_l2"
            ],
            "previous_action_shuffle_action_change_l2": shuffle[
                "previous_action"
            ]["action_change_l2"],
        },
    }
    write_json(output_path, payload)
    console.print(payload)
    return output_path
