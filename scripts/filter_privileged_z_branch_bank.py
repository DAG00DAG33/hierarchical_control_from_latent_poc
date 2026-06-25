#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def _branch_mask(args: argparse.Namespace, data: np.lib.npyio.NpzFile) -> np.ndarray:
    return_delta = np.asarray(data["selected_return_delta"], dtype=np.float32)
    mask = np.ones(len(return_delta), dtype=np.bool_)
    if args.min_return_delta is not None:
        mask &= return_delta >= float(args.min_return_delta)
    if args.top_return_k is not None:
        if args.top_return_k <= 0:
            raise ValueError("--top-return-k must be positive")
        top_k = min(int(args.top_return_k), len(return_delta))
        top_mask = np.zeros(len(return_delta), dtype=np.bool_)
        top_mask[np.argsort(-return_delta)[:top_k]] = True
        mask &= top_mask
    if not np.any(mask):
        raise ValueError("Filter selected no branches")
    return mask


def filter_bank(args: argparse.Namespace) -> None:
    data = np.load(args.input, allow_pickle=True)
    horizon_steps = int(np.asarray(data["horizon_steps"]).item())
    return_delta = np.asarray(data["selected_return_delta"], dtype=np.float32)
    branch_count = len(return_delta)
    mask = _branch_mask(args, data)
    output: dict[str, np.ndarray] = {}
    for key in data.files:
        array = np.asarray(data[key])
        if key in {"conditions", "actions"}:
            output[key] = (
                array.reshape(horizon_steps, branch_count, *array.shape[1:])[:, mask]
                .reshape(-1, *array.shape[1:])
                .astype(array.dtype, copy=False)
            )
        elif key == "sample_weights" and array.shape[:1] == (branch_count * horizon_steps,):
            output[key] = (
                array.reshape(horizon_steps, branch_count)[:, mask]
                .reshape(-1)
                .astype(array.dtype, copy=False)
            )
        elif array.shape[:1] == (branch_count,) and key.startswith("selected_"):
            output[key] = array[mask]
        else:
            output[key] = array
    output["filter_source_path"] = np.asarray(str(args.input))
    output["filter_min_return_delta"] = np.asarray(
        np.nan if args.min_return_delta is None else args.min_return_delta,
        dtype=np.float32,
    )
    output["filter_top_return_k"] = np.asarray(
        -1 if args.top_return_k is None else args.top_return_k,
        dtype=np.int64,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **output)
    selected_return = return_delta[mask]
    selected_success = np.asarray(data["selected_success_delta"], dtype=np.float32)[mask]
    print(
        {
            "input": str(args.input),
            "output": str(args.output),
            "input_branches": int(branch_count),
            "selected_branches": int(np.sum(mask)),
            "condition_rows": int(output["conditions"].shape[0]),
            "return_delta_mean": float(np.mean(selected_return)),
            "success_delta_mean": float(np.mean(selected_success)),
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-return-delta", type=float)
    parser.add_argument("--top-return-k", type=int)
    args = parser.parse_args()
    filter_bank(args)


if __name__ == "__main__":
    main()
