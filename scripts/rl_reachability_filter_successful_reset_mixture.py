#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from hcl_poc.utils import ensure_dir


def _batch_keys(h5: h5py.File) -> list[str]:
    return sorted(key for key in h5.keys() if key.startswith("batch_"))


def _success_score(group: h5py.Group, mode: str) -> float:
    if "success" not in group:
        return 0.0
    success = np.asarray(group["success"], dtype=np.bool_)
    if success.ndim != 2:
        raise ValueError(f"{group.name}/success must have shape [time, env]")
    if mode == "any":
        return float(np.mean(np.any(success, axis=0)))
    if mode == "final":
        return float(np.mean(success[-1]))
    raise ValueError(f"Unknown score mode: {mode}")


def _copy_group(
    source: h5py.File,
    target: h5py.File,
    *,
    source_key: str,
    output_index: int,
    score: float | None = None,
) -> int:
    name = f"batch_{output_index:06d}"
    source.copy(source_key, target, name=name)
    group = target[name]
    group.attrs["source_group"] = source_key
    if score is not None:
        group.attrs["success_filter_score"] = float(score)
    return output_index + 1


def _copy_attrs(source: h5py.AttributeManager, target: h5py.AttributeManager) -> None:
    for key, value in source.items():
        target[key] = value


def filter_dataset(args: argparse.Namespace) -> dict[str, Any]:
    input_path = Path(args.input)
    output_path = Path(args.output)
    if output_path.exists() and not args.force:
        raise FileExistsError(output_path)
    ensure_dir(output_path.parent)
    tmp_path = output_path.with_suffix(".tmp.h5")
    tmp_path.unlink(missing_ok=True)

    summary: dict[str, Any] = {
        "input": str(input_path),
        "output": str(output_path),
        "score_mode": str(args.score_mode),
        "demo_batches": int(args.demo_batches),
        "top_deployed_batches_per_policy": int(args.top_deployed_batches_per_policy),
        "selected": {},
    }

    with h5py.File(input_path, "r") as src, h5py.File(tmp_path, "w") as dst:
        if "meta" in src:
            meta = dst.create_group("meta")
            _copy_attrs(src["meta"].attrs, meta.attrs)
        else:
            meta = dst.create_group("meta")
        meta.attrs["success_filtered_from"] = str(input_path)
        meta.attrs["success_filter_score_mode"] = str(args.score_mode)
        meta.attrs["success_filter_demo_batches"] = int(args.demo_batches)
        meta.attrs["success_filter_top_deployed_batches_per_policy"] = int(
            args.top_deployed_batches_per_policy
        )

        output_index = 0
        deployed: dict[str, list[tuple[float, str]]] = defaultdict(list)
        demo_keys: list[str] = []
        for key in _batch_keys(src):
            group = src[key]
            source = str(group.attrs.get("source", ""))
            if source == "demo":
                demo_keys.append(key)
                continue
            policy = str(group.attrs.get("collector_policy", source or "unknown"))
            deployed[policy].append((_success_score(group, args.score_mode), key))

        for key in demo_keys[: int(args.demo_batches)]:
            output_index = _copy_group(src, dst, source_key=key, output_index=output_index)

        selected_summary = {}
        for policy in sorted(deployed):
            ranked = sorted(deployed[policy], key=lambda item: (-item[0], item[1]))
            chosen = ranked[: int(args.top_deployed_batches_per_policy)]
            selected_summary[policy] = [
                {"group": key, "score": float(score)} for score, key in chosen
            ]
            for score, key in chosen:
                output_index = _copy_group(
                    src,
                    dst,
                    source_key=key,
                    output_index=output_index,
                    score=score,
                )
        meta.attrs["batches"] = int(output_index)
        meta.attrs["success_filter_selected"] = str(selected_summary)
        summary["selected"] = selected_summary
        summary["batches"] = int(output_index)
    tmp_path.replace(output_path)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--demo-batches", type=int, default=8)
    parser.add_argument("--top-deployed-batches-per-policy", type=int, default=4)
    parser.add_argument("--score-mode", choices=["any", "final"], default="any")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    summary = filter_dataset(args)
    print(summary)


if __name__ == "__main__":
    main()
