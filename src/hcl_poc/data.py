from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
from rich.console import Console
from tqdm import tqdm

from hcl_poc.config import Config
from hcl_poc.features import DinoExtractor, batched
from hcl_poc.h5util import as_array, episode_groups, list_datasets, pick_dataset
from hcl_poc.utils import Standardizer, default_device, ensure_dir

console = Console()


@dataclass(frozen=True)
class Episode:
    features: np.ndarray
    proprio: np.ndarray
    actions: np.ndarray

    @property
    def length(self) -> int:
        return int(self.actions.shape[0])


def download_demo(config: Config) -> None:
    raw_dir = ensure_dir(config.path_value("paths.raw_demo_dir"))
    cmd = [sys.executable, "-m", "mani_skill.utils.download_demo", config.get("env_id"), "-o", str(raw_dir)]
    console.print(f"[bold]Downloading demos:[/bold] {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def find_h5_candidates(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*.h5") if "trajectory" in p.name.lower() or "demo" in p.name.lower())


def replay_demo(config: Config, source_h5: Path) -> Path:
    raw_dir = ensure_dir(config.path_value("paths.raw_demo_dir"))
    out_path = raw_dir / "pusht_rgb_state_pd_ee_delta_pos.h5"
    if out_path.exists():
        return out_path
    cmd = [
        sys.executable,
        "-m",
        "mani_skill.trajectory.replay_trajectory",
        "--traj-path",
        str(source_h5),
        "--save-traj",
        "--obs-mode",
        str(config.get("obs_mode")),
        "--target-control-mode",
        str(config.get("control_mode")),
        "--use-env-states",
        "--output-name",
        str(out_path),
    ]
    console.print(f"[bold]Replaying demos:[/bold] {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    return out_path


def _pick_rgb(datasets: dict[str, h5py.Dataset]) -> h5py.Dataset:
    return pick_dataset(
        datasets,
        lambda name, ds: "rgb" in name.lower()
        and len(ds.shape) >= 4
        and ds.shape[-1] in (3, 4),
        "RGB observation",
    )


def _pick_named(datasets: dict[str, h5py.Dataset], suffix: str) -> h5py.Dataset:
    return pick_dataset(
        datasets,
        lambda name, _ds: name.lower().endswith(suffix) or f"/{suffix}" in name.lower(),
        suffix,
    )


def _extract_raw_episode(group: h5py.Group) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    datasets = list_datasets(group)
    actions = as_array(_pick_named(datasets, "actions")).astype(np.float32)
    length = actions.shape[0]
    rgb = as_array(_pick_rgb(datasets), length=length)
    if rgb.shape[-1] == 4:
        rgb = rgb[..., :3]
    qpos = as_array(_pick_named(datasets, "qpos"), length=length).astype(np.float32)
    qvel = as_array(_pick_named(datasets, "qvel"), length=length).astype(np.float32)
    proprio = np.concatenate([qpos.reshape(length, -1), qvel.reshape(length, -1)], axis=-1)
    return rgb, proprio, actions


def prepare_dataset(config: Config, force: bool = False) -> Path:
    out_path = config.path_value("paths.prepared_path")
    if out_path.exists() and not force:
        console.print(f"Prepared dataset already exists: {out_path}")
        return out_path

    raw_dir = ensure_dir(config.path_value("paths.raw_demo_dir"))
    candidates = find_h5_candidates(raw_dir)
    if not candidates:
        download_demo(config)
        candidates = find_h5_candidates(raw_dir)
    if not candidates:
        raise FileNotFoundError(f"No ManiSkill HDF5 demos found under {raw_dir}")

    replayed = replay_demo(config, candidates[0])
    device = default_device()
    extractor = DinoExtractor(config.get("dino.model_name"), device)
    batch_size = int(config.get("dino.batch_size", 32))
    max_trajectories = int(config.get("data.max_trajectories", 206))
    ensure_dir(out_path.parent)

    with h5py.File(replayed, "r") as src, h5py.File(out_path, "w") as dst:
        episodes = episode_groups(src)[:max_trajectories]
        meta = dst.create_group("meta")
        meta.attrs["source_h5"] = str(replayed)
        meta.attrs["dino_model"] = config.get("dino.model_name")
        meta.attrs["control_mode"] = config.get("control_mode")
        meta.attrs["obs_mode"] = config.get("obs_mode")
        for idx, group in enumerate(tqdm(episodes, desc="Extract DINO features")):
            rgb, proprio, actions = _extract_raw_episode(group)
            feats = [extractor.encode_batch(chunk) for chunk in batched(rgb, batch_size)]
            features = np.concatenate(feats, axis=0)
            ep = dst.create_group(f"episode_{idx:04d}")
            ep.create_dataset("dino", data=features, compression="gzip")
            ep.create_dataset("proprio", data=proprio, compression="gzip")
            ep.create_dataset("actions", data=actions, compression="gzip")
        console.print(f"Wrote prepared dataset: {out_path}")
    return out_path


def load_episodes(path: str | Path, limit: int | None = None) -> list[Episode]:
    episodes: list[Episode] = []
    with h5py.File(path, "r") as h5:
        keys = sorted(k for k in h5 if k.startswith("episode_"))
        if limit is not None:
            keys = keys[:limit]
        for key in keys:
            group = h5[key]
            episodes.append(
                Episode(
                    features=np.asarray(group["dino"], dtype=np.float32),
                    proprio=np.asarray(group["proprio"], dtype=np.float32),
                    actions=np.asarray(group["actions"], dtype=np.float32),
                )
            )
    if not episodes:
        raise ValueError(f"No prepared episodes found in {path}")
    return episodes


def fit_input_standardizer(episodes: list[Episode]) -> Standardizer:
    x = np.concatenate([np.concatenate([ep.features, ep.proprio], axis=-1) for ep in episodes], axis=0)
    return Standardizer.fit(x)


def fit_action_standardizer(episodes: list[Episode]) -> Standardizer:
    x = np.concatenate([ep.actions for ep in episodes], axis=0)
    return Standardizer.fit(x)


class RandomTupleDataset(torch.utils.data.Dataset):
    def __init__(self, episodes: list[Episode], length: int = 200_000) -> None:
        self.episodes = [ep for ep in episodes if ep.length > 2]
        self.length = length
        if not self.episodes:
            raise ValueError("No usable episodes")

    def __len__(self) -> int:
        return self.length

    def sample_ep_t(self, min_future: int = 1) -> tuple[Episode, int]:
        ep = self.episodes[np.random.randint(0, len(self.episodes))]
        t = np.random.randint(0, ep.length - min_future)
        return ep, int(t)

