from __future__ import annotations

import argparse
from pathlib import Path

import h5py


def _copy_group(src: h5py.Group, dst: h5py.Group) -> None:
    for key, value in src.attrs.items():
        dst.attrs[key] = value
    for name in src:
        src.copy(src[name], dst, name=name)


def merge(output: Path, inputs: list[str], force: bool) -> Path:
    if output.exists() and not force:
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_suffix(".tmp.h5")
    tmp.unlink(missing_ok=True)
    with h5py.File(tmp, "w") as out:
        meta = out.create_group("meta")
        meta.attrs["source"] = "merged_privileged_z_vector_dataset"
        meta.attrs["inputs"] = "\n".join(inputs)
        group_index = 0
        for spec in inputs:
            parts = spec.split(":", 2)
            if len(parts) not in {2, 3}:
                raise ValueError(
                    "Input specs must be LABEL:PATH or LABEL:PATH:CHECKPOINT"
                )
            label = int(parts[0])
            src_path = Path(parts[1])
            checkpoint = parts[2] if len(parts) == 3 else ""
            with h5py.File(src_path, "r") as src:
                for key in sorted(src):
                    if key == "meta" or "success_once" not in src[key]:
                        continue
                    dst = out.create_group(f"batch_{group_index:06d}")
                    _copy_group(src[key], dst)
                    dst.attrs["expert_index"] = label
                    dst.attrs["source_file"] = str(src_path)
                    dst.attrs["expert_checkpoint"] = checkpoint
                    group_index += 1
        meta.attrs["groups"] = group_index
    tmp.replace(output)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--input",
        action="append",
        required=True,
        help="LABEL:PATH or LABEL:PATH:CHECKPOINT. Repeat once per source file.",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    print(merge(Path(args.output), args.input, args.force))


if __name__ == "__main__":
    main()
