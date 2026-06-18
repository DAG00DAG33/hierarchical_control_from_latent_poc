from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from hcl_poc.config import Config
from hcl_poc.utils import ensure_dir, read_json


def _label(row: pd.Series) -> str:
    if row["method"] == "flat":
        return "flat latent"
    if row["method"] == "flat_obs":
        return "flat obs"
    if row["method"] == "bc_obs":
        return "BC obs"
    if row["method"] == "bc_obs_1step":
        return "BC obs 1-step"
    if row["method"] == "bc_obs_dagger":
        return "BC obs DAgger"
    if row["method"] == "bc_pose":
        return "BC predicted pose"
    if row["method"] == "bc_state":
        return "BC privileged state"
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
    metrics = ["success", "final_reward", "max_reward", "inference_latency_s"]
    aggregate = (
        plot_df.groupby(["label", "method", "horizon_s", "n_traj"], dropna=False)[metrics]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    aggregate.columns = [
        "_".join(str(part) for part in column if part).rstrip("_")
        if isinstance(column, tuple)
        else str(column)
        for column in aggregate.columns
    ]
    aggregate.to_csv(results_dir / "summary_by_method.csv", index=False)

    for metric in metrics:
        fig, ax = plt.subplots(figsize=(7, 4))
        for label, group in plot_df.groupby("label"):
            agg = (
                group.groupby("n_traj")[metric]
                .agg(["mean", "std", "count"])
                .reset_index()
                .sort_values("n_traj")
            )
            x = agg["n_traj"].to_numpy()
            mean = agg["mean"].to_numpy()
            std = np.nan_to_num(agg["std"].to_numpy(), nan=0.0)
            ax.plot(x, mean, marker="o", label=label)
            ax.fill_between(x, mean - std, mean + std, alpha=0.16)
        ax.set_xlabel("training trajectories")
        ax.set_ylabel("final-state success" if metric == "success" else metric)
        if metric == "success":
            ax.set_ylim(-0.02, 1.02)
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(results_dir / f"{metric}_vs_trajectories.png", dpi=180)
        plt.close(fig)
    return csv_path
