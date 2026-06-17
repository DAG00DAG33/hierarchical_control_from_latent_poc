from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from hcl_poc.config import Config
from hcl_poc.utils import ensure_dir, read_json


def _label(row: pd.Series) -> str:
    if row["method"] == "flat":
        return "flat latent"
    if row["method"] == "flat_obs":
        return "flat obs"
    return f"hier {row['horizon_s']:g}s"


def build_report(config: Config) -> Path:
    results_dir = ensure_dir(config.get("paths.results_dir"))
    records = []
    for path in results_dir.rglob("*.json"):
        records.append(read_json(path))
    if not records:
        raise FileNotFoundError(f"No result JSON files found in {results_dir}")
    df = pd.DataFrame.from_records(records)
    csv_path = results_dir / "summary.csv"
    df.to_csv(csv_path, index=False)

    plot_df = df.copy()
    plot_df["label"] = plot_df.apply(_label, axis=1)
    fig, ax = plt.subplots(figsize=(7, 4))
    for label, group in plot_df.groupby("label"):
        agg = group.groupby("n_traj", as_index=False)["success"].mean().sort_values("n_traj")
        ax.plot(agg["n_traj"], agg["success"], marker="o", label=label)
    ax.set_xlabel("training trajectories")
    ax.set_ylabel("final-state success")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(results_dir / "success_vs_trajectories.png", dpi=180)
    plt.close(fig)

    for metric in ["final_reward", "max_reward", "inference_latency_s"]:
        fig, ax = plt.subplots(figsize=(7, 4))
        for label, group in plot_df.groupby("label"):
            agg = group.groupby("n_traj", as_index=False)[metric].mean().sort_values("n_traj")
            ax.plot(agg["n_traj"], agg[metric], marker="o", label=label)
        ax.set_xlabel("training trajectories")
        ax.set_ylabel(metric)
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(results_dir / f"{metric}_vs_trajectories.png", dpi=180)
        plt.close(fig)
    return csv_path
