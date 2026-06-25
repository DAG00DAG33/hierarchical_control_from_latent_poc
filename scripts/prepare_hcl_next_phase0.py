from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n")


def create_split_manifests(
    *,
    source_manifest: Path,
    output_dir: Path,
    num_train_trajectories: int,
    seed: int,
    eval_seed_start: int,
    eval_episodes: int,
) -> list[Path]:
    manifest = json.loads(source_manifest.read_text())
    base = {
        "experiment": "hcl_next_phase0",
        "source_manifest": str(source_manifest),
        "dataset": manifest["dataset"],
        "num_train_trajectories": num_train_trajectories,
        "seed": seed,
        "sha256": manifest["sha256"],
        "selection": manifest["selection"],
    }

    written: list[Path] = []
    for split, key in (("train", "train_keys"), ("val", "validation_keys")):
        out = dict(base)
        out.update(
            {
                "split": split,
                "episode_keys": manifest[key],
                "num_episodes": len(manifest[key]),
            }
        )
        path = output_dir / f"pusht_n{num_train_trajectories}_seed{seed}_{split}.json"
        write_json(path, out)
        written.append(path)

    eval_manifest = dict(base)
    eval_manifest.update(
        {
            "split": "eval",
            "type": "simulator_seed_bank",
            "seed_start": eval_seed_start,
            "episodes": eval_episodes,
            "note": (
                "Fresh closed-loop simulator seed bank. The source trajectory "
                "dataset has fixed train keys and validation keys, but no "
                "separate held-out eval trajectory split."
            ),
        }
    )
    path = output_dir / f"pusht_n{num_train_trajectories}_seed{seed}_eval.json"
    write_json(path, eval_manifest)
    written.append(path)
    return written


def create_local_reset_banks(
    *,
    dataset_path: Path,
    output_dir: Path,
    regimes: list[int],
    horizons: list[int],
    seed: int,
    entries_per_bank: int,
) -> list[Path]:
    written: list[Path] = []
    with h5py.File(dataset_path, "r") as h5:
        max_steps = int(h5["meta"].attrs["max_steps"])
        num_envs = int(h5["meta"].attrs["num_envs"])
        batch_keys = sorted(key for key in h5.keys() if key.startswith("batch_"))
        if not batch_keys:
            raise ValueError(f"No vector batches found in {dataset_path}")

        for num_train_trajectories in regimes:
            for horizon in horizons:
                if horizon > max_steps:
                    raise ValueError(
                        f"horizon {horizon} exceeds dataset max_steps {max_steps}"
                    )
                rng = np.random.default_rng(seed)
                selected = rng.choice(
                    batch_keys,
                    size=entries_per_bank,
                    replace=entries_per_bank > len(batch_keys),
                )
                entries = []
                for key in selected:
                    batch_key = str(key)
                    entries.append(
                        {
                            "batch": batch_key,
                            "batch_seed": int(h5[batch_key].attrs["batch_seed"]),
                            "timestep": int(rng.integers(0, max_steps - horizon + 1)),
                        }
                    )
                manifest = {
                    "experiment": "hcl_next_phase0",
                    "dataset": str(dataset_path),
                    "num_envs": num_envs,
                    "horizon": horizon,
                    "seed": seed,
                    "num_train_trajectories_regime": num_train_trajectories,
                    "sampled_local_episodes": num_envs * len(entries),
                    "entries": entries,
                    "note": (
                        "Fixed local reset bank for comparing imitation/RL "
                        "checkpoints in this N-demo regime; reset data source "
                        "is shared held-out 4096-env validation vector dataset."
                    ),
                }
                path = (
                    output_dir
                    / f"local_reset_bank_n{num_train_trajectories}_seed{seed}_k{horizon}.json"
                )
                write_json(path, manifest)
                written.append(path)
    return written


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="data/manifests")
    parser.add_argument(
        "--local-reset-dataset",
        default="data/rl_rerun/pusht_vector_state_demos_n4096_val_b1.h5",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval-seed-start", type=int, default=2026062500)
    parser.add_argument("--eval-episodes", type=int, default=500)
    parser.add_argument("--entries-per-bank", type=int, default=1)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    written: list[Path] = []
    for num_train_trajectories in (500, 1800):
        written.extend(
            create_split_manifests(
                source_manifest=Path(
                    f"artifacts/incremental/vae512_scaling/n{num_train_trajectories}/data_manifest.json"
                ),
                output_dir=output_dir,
                num_train_trajectories=num_train_trajectories,
                seed=args.seed,
                eval_seed_start=args.eval_seed_start,
                eval_episodes=args.eval_episodes,
            )
        )

    written.extend(
        create_local_reset_banks(
            dataset_path=Path(args.local_reset_dataset),
            output_dir=output_dir,
            regimes=[500, 1800],
            horizons=[2, 5, 10, 20],
            seed=args.seed,
            entries_per_bank=args.entries_per_bank,
        )
    )

    print(f"wrote {len(written)} files")
    for path in written:
        print(path)


if __name__ == "__main__":
    main()
