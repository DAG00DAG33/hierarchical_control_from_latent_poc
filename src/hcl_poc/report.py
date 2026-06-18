from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import PercentFormatter

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

    seed_counts = (
        plot_df.groupby("n_traj")["seed"].nunique().sort_index()
        if "seed" in plot_df
        else pd.Series(dtype=int)
    )
    seed_note = ", ".join(
        f"{int(n_traj)}: {int(count)}" for n_traj, count in seed_counts.items()
    )
    x_ticks = sorted(int(value) for value in plot_df["n_traj"].unique())
    metric_labels = {
        "success": "success rate",
        "final_reward": "final normalized reward",
        "max_reward": "maximum normalized reward",
        "inference_latency_s": "inference latency (s/action)",
    }

    for metric in metrics:
        fig, ax = plt.subplots(figsize=(7, 4))
        upper_values = []
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
            lower = np.maximum(0.0, mean - std)
            upper = mean + std
            if metric == "success":
                upper = np.minimum(1.0, upper)
            upper_values.extend(upper)
            ax.fill_between(x, lower, upper, alpha=0.16)
        ax.set_xscale("log")
        ax.set_xticks(x_ticks)
        ax.set_xticklabels([str(value) for value in x_ticks])
        ax.set_xlabel("training trajectories")
        ax.set_ylabel(metric_labels[metric])
        if metric == "success":
            upper_limit = max(0.05, 1.2 * max(upper_values))
            ax.set_ylim(0.0, min(1.0, upper_limit))
            ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.suptitle("Mean +/- sample SD across policy seeds", fontsize=10)
        if seed_note:
            fig.text(0.5, 0.015, f"Seeds per trajectory count: {seed_note}", ha="center", fontsize=8)
        fig.tight_layout(rect=(0, 0.055, 1, 0.96))
        fig.savefig(results_dir / f"{metric}_vs_trajectories.png", dpi=180)
        plt.close(fig)
    return csv_path
