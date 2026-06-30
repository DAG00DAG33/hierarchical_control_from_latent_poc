#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


VARIANTS = {
    "terminal": "run7_dpsi_terminal_b8_u1000",
    "progress": "run7_dpsi_progress_b8_u1000",
    "bc_advantage": "run7_dpsi_bc_advantage_b8_u1000",
}


def _metrics_path(root: Path, run_dir: str) -> Path:
    return root / run_dir / "privileged_tcp_ppo_progress_terminal_n4096_seed0" / "metrics.json"


def _row(name: str, metrics: dict[str, Any]) -> dict[str, float | str]:
    trained = metrics["trained_eval"]
    shuffled = metrics["trained_shuffled_goal_eval"]
    last = metrics["history_last"]
    return {
        "variant": name,
        "terminal_distance": float(trained["terminal_distance_mean"]),
        "reach": float(trained["goal_reach_rate_eps"]),
        "p90": float(trained["p90_terminal_distance"]),
        "p99": float(trained["p99_terminal_distance"]),
        "runner_shuffled_reach": float(shuffled["goal_reach_rate_eps"]),
        "action_saturation": float(trained["action_saturation"]),
        "last_train_terminal_distance": float(last["mean_terminal_distance"]),
        "last_train_reach": float(last["goal_reach_rate_eps"]),
        "elapsed_s": float(last["elapsed_s"]),
    }


def _score(row: dict[str, float | str]) -> float:
    # Local candidate score only. Full-task success remains the deciding metric.
    return (
        float(row["terminal_distance"])
        + 0.25 * float(row["p90"])
        + 0.10 * float(row["p99"])
        + 0.001 * float(row["action_saturation"])
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        default="results/incremental/rl_reachability_debug",
    )
    parser.add_argument(
        "--output",
        default="results/incremental/rl_reachability_debug/run7_reward_variants_summary.json",
    )
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()
    root = Path(args.root)
    rows = []
    missing = []
    for name, run_dir in VARIANTS.items():
        path = _metrics_path(root, run_dir)
        if not path.exists():
            missing.append(str(path))
            continue
        with path.open("r", encoding="utf-8") as f:
            rows.append(_row(name, json.load(f)))
    if missing and not args.allow_incomplete:
        raise FileNotFoundError("Missing Run 7 metrics:\n" + "\n".join(missing))
    ranked = sorted(rows, key=_score)
    payload = {
        "run": "rl_reachability_debug_run7_reward_variant_summary",
        "missing": missing,
        "rows": rows,
        "ranked_by_local_score": ranked,
        "selected_by_local_score": ranked[0]["variant"] if ranked else None,
        "selection_note": (
            "This is only a local reachability ranking. Use full-task success "
            "and deployment-distribution reachability before deciding what to train longer."
        ),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    print(output)
    for row in ranked:
        print(
            row["variant"],
            "terminal",
            f"{float(row['terminal_distance']):.6f}",
            "reach",
            f"{float(row['reach']):.4f}",
            "p90",
            f"{float(row['p90']):.6f}",
            "score",
            f"{_score(row):.6f}",
        )


if __name__ == "__main__":
    main()
