from __future__ import annotations

from pathlib import Path
from typing import Any

import gymnasium as gym
import imageio.v2 as imageio
import mani_skill  # noqa: F401
import numpy as np
import torch
from rich.console import Console
from tqdm import trange

from hcl_poc.config import Config
from hcl_poc.features import DinoExtractor
from hcl_poc.flow import sample_flow
from hcl_poc.models import FlowModel, ObservationEncoder
from hcl_poc.utils import Standardizer, Timer, default_device, ensure_dir, set_seed, write_json

console = Console()


def horizon_steps(config: Config, horizon_s: float) -> int:
    control_freq = float(config.get("control_freq", 20))
    return max(1, int(round(horizon_s * control_freq)))


def _flatten_obs(obs: Any) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}

    def visit(prefix: str, node: Any) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                visit(f"{prefix}/{key}" if prefix else str(key), value)
        else:
            try:
                if isinstance(node, torch.Tensor):
                    out[prefix] = node.detach().cpu().numpy()
                else:
                    out[prefix] = np.asarray(node)
            except Exception:
                pass

    visit("", obs)
    return out


def extract_runtime_rgb_proprio(obs: Any) -> tuple[np.ndarray, np.ndarray]:
    flat = _flatten_obs(obs)
    rgb_candidates = [
        value
        for key, value in flat.items()
        if "rgb" in key.lower() and value.ndim >= 3 and value.shape[-1] in (3, 4)
    ]
    if not rgb_candidates:
        raise KeyError(f"No RGB image found in observation keys: {sorted(flat)[:80]}")
    rgb = rgb_candidates[0]
    if rgb.ndim == 4:
        rgb = rgb[0]
    if rgb.shape[-1] == 4:
        rgb = rgb[..., :3]

    qpos = None
    qvel = None
    tcp_pose = None
    for key, value in flat.items():
        low = key.lower()
        if low.endswith("qpos") or "/qpos" in low:
            qpos = value.reshape(-1)
        if low.endswith("qvel") or "/qvel" in low:
            qvel = value.reshape(-1)
        if low.endswith("tcp_pose") or "/tcp_pose" in low:
            tcp_pose = value.reshape(-1)
    if qpos is None or qvel is None or tcp_pose is None:
        raise KeyError(f"Missing qpos/qvel/tcp_pose in observation keys: {sorted(flat)[:80]}")
    return rgb.astype(np.uint8), np.concatenate([qpos, qvel, tcp_pose]).astype(np.float32)


@torch.inference_mode()
def encode_obs(
    obs: Any,
    dino: DinoExtractor,
    encoder: ObservationEncoder,
    input_norm: Standardizer,
    device: torch.device,
) -> torch.Tensor:
    rgb, proprio = extract_runtime_rgb_proprio(obs)
    feat = dino.encode_batch(rgb[None])[0]
    x = input_norm.transform(np.concatenate([feat, proprio], axis=0)[None])
    return encoder(torch.from_numpy(x).to(device).float())


@torch.inference_mode()
def encode_obs_direct(
    obs: Any,
    dino: DinoExtractor,
    input_norm: Standardizer,
    device: torch.device,
) -> torch.Tensor:
    rgb, proprio = extract_runtime_rgb_proprio(obs)
    feat = dino.encode_batch(rgb[None])[0]
    x = input_norm.transform(np.concatenate([feat, proprio], axis=0)[None])
    return torch.from_numpy(x).to(device).float()


def _load_encoder(config: Config, n_traj: int, seed: int, device: torch.device) -> tuple[ObservationEncoder, Standardizer]:
    ckpt_path = Path(config.get("paths.artifact_dir")) / f"n{n_traj}" / f"seed{seed}" / "encoder.pt"
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    encoder = ObservationEncoder(ckpt["input_dim"], ckpt["latent_dim"], ckpt["hidden_dim"]).to(device)
    encoder.load_state_dict(ckpt["encoder"])
    encoder.eval()
    return encoder, Standardizer.from_state_dict(ckpt["input_norm"])


def _load_flow(path: Path, device: torch.device) -> tuple[FlowModel, dict[str, Any]]:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = FlowModel(ckpt["sample_dim"], ckpt["cond_dim"], ckpt["hidden_dim"]).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, ckpt


def _render_frame(env: gym.Env) -> np.ndarray:
    frame = env.render()
    if isinstance(frame, torch.Tensor):
        frame = frame.detach().cpu().numpy()
    frame = np.asarray(frame)
    if frame.ndim == 4:
        frame = frame[0]
    return frame.astype(np.uint8)


@torch.inference_mode()
def _sample_action(
    obs: Any,
    method: str,
    dino: DinoExtractor,
    device: torch.device,
    action_norm: Standardizer,
    steps: int,
    flat: FlowModel | None,
    input_norm: Standardizer,
    encoder: ObservationEncoder | None = None,
    high: FlowModel | None = None,
    high_ckpt: dict[str, Any] | None = None,
    low: FlowModel | None = None,
    subgoal: torch.Tensor | None = None,
) -> tuple[np.ndarray, torch.Tensor | None]:
    if method == "flat_obs":
        assert flat is not None
        cond_obs = encode_obs_direct(obs, dino, input_norm, device)
        action_chunk = sample_flow(flat, cond_obs, steps, flat.sample_dim).cpu().numpy()[0]
    else:
        assert encoder is not None
        z = encode_obs(obs, dino, encoder, input_norm, device)
        if method == "flat":
            assert flat is not None
            action_chunk = sample_flow(flat, z, steps, flat.sample_dim).cpu().numpy()[0]
        elif method == "hier":
            assert high is not None and low is not None and high_ckpt is not None and subgoal is not None
            cond = torch.cat([z, subgoal], dim=-1)
            action_chunk = sample_flow(low, cond, steps, low.sample_dim).cpu().numpy()[0]
        else:
            raise ValueError(method)
    action = action_norm.inverse(action_chunk.reshape(-1, action_norm.mean.shape[0]))[0]
    return action, subgoal


def evaluate(config: Config, n_traj: int, seed: int, method: str, horizon_s: float | None = None) -> Path:
    set_seed(seed)
    device = default_device()
    dino = DinoExtractor(config.get("dino.model_name"), device)
    artifact_dir = Path(config.get("paths.artifact_dir")) / f"n{n_traj}" / f"seed{seed}"
    results_dir = ensure_dir(Path(config.get("paths.results_dir")) / f"n{n_traj}" / f"seed{seed}")
    result_name = method if horizon_s is None else f"{method}_{horizon_s:g}s"
    out_path = results_dir / f"{result_name}.json"
    encoder = None
    input_norm = None

    if method == "flat":
        encoder, input_norm = _load_encoder(config, n_traj, seed, device)
        flat, flat_ckpt = _load_flow(artifact_dir / "flat.pt", device)
        action_norm = Standardizer.from_state_dict(flat_ckpt["action_norm"])
        low = high = None
        steps = int(flat_ckpt["flow_steps"])
    elif method == "flat_obs":
        flat, flat_ckpt = _load_flow(artifact_dir / "flat_obs.pt", device)
        action_norm = Standardizer.from_state_dict(flat_ckpt["action_norm"])
        input_norm = Standardizer.from_state_dict(flat_ckpt["input_norm"])
        low = high = None
        steps = int(flat_ckpt["flow_steps"])
    elif method == "hier":
        encoder, input_norm = _load_encoder(config, n_traj, seed, device)
        if horizon_s is None:
            raise ValueError("Hierarchy evaluation requires horizon_s")
        h_steps = horizon_steps(config, horizon_s)
        high, high_ckpt = _load_flow(artifact_dir / f"high_h{h_steps}.pt", device)
        low, low_ckpt = _load_flow(artifact_dir / f"low_h{h_steps}.pt", device)
        action_norm = Standardizer.from_state_dict(low_ckpt["action_norm"])
        flat = None
        steps = int(low_ckpt["flow_steps"])
    else:
        raise ValueError(method)

    env = gym.make(
        config.get("env_id"),
        obs_mode=config.get("obs_mode"),
        control_mode=config.get("control_mode"),
        render_mode=None,
    )
    eval_episodes = int(config.get("data.eval_episodes", 50))
    eval_seed = int(config.get("data.eval_seed", 10000))
    successes: list[float] = []
    final_rewards: list[float] = []
    max_rewards: list[float] = []
    latencies: list[float] = []

    for ep_idx in trange(eval_episodes, desc=f"eval {result_name} n={n_traj} seed={seed}"):
        obs, _info = env.reset(seed=eval_seed + ep_idx)
        done = False
        truncated = False
        max_reward = -float("inf")
        final_reward = 0.0
        subgoal = None
        refresh = 1
        if method == "hier":
            refresh = max(1, int(round(horizon_steps(config, float(horizon_s)) * float(config.get("policy.high_level_refresh_fraction")))))
        step_idx = 0
        success = False
        while not (done or truncated):
            timer = Timer()
            assert input_norm is not None
            if method == "hier":
                assert high is not None and high_ckpt is not None and encoder is not None
                if subgoal is None or step_idx % refresh == 0:
                    z = encode_obs(obs, dino, encoder, input_norm, device)
                    subgoal = sample_flow(high, z, int(high_ckpt["flow_steps"]), high.sample_dim)
            action, subgoal = _sample_action(
                obs,
                method,
                dino,
                device,
                action_norm,
                steps,
                flat,
                input_norm,
                encoder=encoder,
                high=high,
                high_ckpt=high_ckpt if method == "hier" else None,
                low=low,
                subgoal=subgoal,
            )
            latencies.append(timer.elapsed())
            obs, reward, done, truncated, info = env.step(action)
            reward_f = float(np.asarray(reward).reshape(-1)[0])
            final_reward = reward_f
            max_reward = max(max_reward, reward_f)
            success = success or bool(np.asarray(info.get("success", False)).reshape(-1)[0])
            step_idx += 1
        successes.append(float(success))
        final_rewards.append(final_reward)
        max_rewards.append(max_reward)
    env.close()

    payload = {
        "method": method,
        "horizon_s": horizon_s,
        "n_traj": n_traj,
        "seed": seed,
        "success": float(np.mean(successes)),
        "success_stderr": float(np.std(successes) / max(len(successes), 1) ** 0.5),
        "final_reward": float(np.mean(final_rewards)),
        "max_reward": float(np.mean(max_rewards)),
        "inference_latency_s": float(np.mean(latencies)),
        "episodes": eval_episodes,
    }
    write_json(out_path, payload)
    console.print(payload)
    return out_path


def record_videos(
    config: Config,
    n_traj: int,
    seed: int,
    method: str,
    episodes: int,
    horizon_s: float | None = None,
) -> list[Path]:
    if method != "flat_obs":
        raise ValueError("Video recording currently supports flat_obs only")
    set_seed(seed)
    device = default_device()
    dino = DinoExtractor(config.get("dino.model_name"), device)
    artifact_dir = Path(config.get("paths.artifact_dir")) / f"n{n_traj}" / f"seed{seed}"
    flat, flat_ckpt = _load_flow(artifact_dir / "flat_obs.pt", device)
    action_norm = Standardizer.from_state_dict(flat_ckpt["action_norm"])
    input_norm = Standardizer.from_state_dict(flat_ckpt["input_norm"])
    steps = int(flat_ckpt["flow_steps"])
    out_dir = ensure_dir(Path(config.get("paths.results_dir")).parent / "videos")

    env = gym.make(
        config.get("env_id"),
        obs_mode=config.get("obs_mode"),
        control_mode=config.get("control_mode"),
        render_mode="rgb_array",
    )
    eval_seed = int(config.get("data.eval_seed", 10000))
    paths: list[Path] = []
    for ep_idx in trange(episodes, desc=f"record {method} n={n_traj} seed={seed}"):
        rollout_seed = eval_seed + ep_idx
        obs, _info = env.reset(seed=rollout_seed)
        frames = [_render_frame(env)]
        done = False
        truncated = False
        success = False
        max_reward = -float("inf")
        final_reward = 0.0
        while not (done or truncated):
            action, _subgoal = _sample_action(
                obs,
                method,
                dino,
                device,
                action_norm,
                steps,
                flat,
                input_norm,
            )
            obs, reward, done, truncated, info = env.step(action)
            frames.append(_render_frame(env))
            reward_f = float(np.asarray(reward).reshape(-1)[0])
            final_reward = reward_f
            max_reward = max(max_reward, reward_f)
            success = success or bool(np.asarray(info.get("success", False)).reshape(-1)[0])
        path = out_dir / (
            f"{method}_n{n_traj}_seed{seed}_evalseed{rollout_seed}_"
            f"success{int(success)}_final{final_reward:.3f}_max{max_reward:.3f}.mp4"
        )
        imageio.mimsave(path, frames, fps=int(config.get("control_freq", 20)), macro_block_size=1)
        paths.append(path)
    env.close()
    for path in paths:
        console.print(f"Wrote {path}")
    return paths
