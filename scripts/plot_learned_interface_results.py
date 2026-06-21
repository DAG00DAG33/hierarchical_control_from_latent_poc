from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "incremental"
OUTPUT = ROOT / "docs" / "results" / "learned_interface"


def read(path: Path) -> dict:
    with path.open() as stream:
        return json.load(stream)


def learned_result(candidate: str, goal_source: str, episodes: int) -> dict:
    return read(
        RESULTS
        / "learned_interface"
        / candidate
        / "seed0"
        / f"{goal_source}_hierarchy_eval_{episodes}.json"
    )


def tcp_result(goal_source: str) -> dict:
    payload = read(
        RESULTS
        / "pre_rl"
        / "phase_f"
        / "raw_tcp"
        / "seed0"
        / f"{goal_source}_hierarchy_eval_100.json"
    )
    return payload["closed_loop"]


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    methods = [
        ("TCP endpoint", tcp_result("learned"), tcp_result("oracle")),
        (
            "VAE-512",
            learned_result("vae512_w2048_b1e6", "learned", 100),
            learned_result("vae512_w2048_b1e6", "oracle", 100),
        ),
        (
            "JEPA weak recon",
            learned_result("jepa256_r001_v1_c001", "learned", 100),
            learned_result("jepa256_r001_v1_c001", "oracle", 100),
        ),
        (
            "Effect-32 concat",
            learned_result("effect32", "learned", 100),
            learned_result("effect32", "oracle", 100),
        ),
        (
            "Effect-32 FiLM",
            learned_result("effect32_film", "learned", 100),
            learned_result("effect32_film", "oracle", 100),
        ),
    ]
    labels = [row[0] for row in methods]
    learned = np.asarray([row[1]["success"] for row in methods])
    oracle = np.asarray([row[2]["success"] for row in methods])
    x = np.arange(len(methods))
    width = 0.36
    fig, axis = plt.subplots(figsize=(9.2, 4.8))
    axis.bar(
        x - width / 2,
        learned,
        width,
        label="Learned high level",
        color="#2563eb",
    )
    axis.bar(
        x + width / 2,
        oracle,
        width,
        label="Branch oracle",
        color="#f59e0b",
    )
    axis.axhline(0.71, color="#111827", linestyle="--", linewidth=1)
    axis.set_ylabel("Success rate (100 episodes)")
    axis.set_ylim(0.0, 0.9)
    axis.set_xticks(x, labels, rotation=18, ha="right")
    axis.legend(frameon=False, ncols=2, loc="upper center")
    axis.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(OUTPUT / "closed_loop_comparison.png", dpi=180)
    plt.close(fig)

    dimensions = [16, 32, 64]
    learned_screen = [
        learned_result(f"effect{dim}", "learned", 20)["success"]
        for dim in dimensions
    ]
    oracle_screen = [
        learned_result(f"effect{dim}", "oracle", 20)["success"]
        for dim in dimensions
    ]
    fig, axis = plt.subplots(figsize=(6.8, 4.4))
    axis.plot(
        dimensions,
        learned_screen,
        marker="o",
        linewidth=2,
        label="Learned high level",
        color="#2563eb",
    )
    axis.plot(
        dimensions,
        oracle_screen,
        marker="o",
        linewidth=2,
        label="Branch oracle",
        color="#f59e0b",
    )
    axis.set_xlabel("Effect-code dimension")
    axis.set_ylabel("Success rate (20-episode screen)")
    axis.set_xticks(dimensions)
    axis.set_ylim(0.35, 0.9)
    axis.grid(alpha=0.2)
    axis.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUTPUT / "effect_dimension_sweep.png", dpi=180)
    plt.close(fig)

    summary = {
        "protocol": {
            "training_trajectories": 1800,
            "validation_trajectories": 200,
            "evaluation_seed_start": 2100000,
            "horizon_steps": 10,
            "update_period": 10,
            "action_horizon": 1,
        },
        "comparison_100_episodes": {
            label: {
                "learned": learned_row,
                "oracle": oracle_row,
            }
            for label, learned_row, oracle_row in methods
        },
        "effect_dimension_screen_20_episodes": {
            str(dim): {
                "learned": learned_result(
                    f"effect{dim}", "learned", 20
                ),
                "oracle": learned_result(
                    f"effect{dim}", "oracle", 20
                ),
            }
            for dim in dimensions
        },
        "scene_only_screen_20_episodes": {
            "learned": learned_result(
                "effect32_scene_film", "learned", 20
            ),
            "oracle": learned_result(
                "effect32_scene_film", "oracle", 20
            ),
        },
    }
    with (OUTPUT / "summary.json").open("w") as stream:
        json.dump(summary, stream, indent=2, sort_keys=True)


if __name__ == "__main__":
    main()
