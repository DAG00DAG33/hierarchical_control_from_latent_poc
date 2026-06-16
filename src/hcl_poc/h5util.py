from __future__ import annotations

from collections.abc import Callable
from typing import Any

import h5py
import numpy as np


def list_datasets(group: h5py.Group) -> dict[str, h5py.Dataset]:
    datasets: dict[str, h5py.Dataset] = {}

    def visit(name: str, obj: Any) -> None:
        if isinstance(obj, h5py.Dataset):
            datasets[name] = obj

    group.visititems(visit)
    return datasets


def pick_dataset(
    datasets: dict[str, h5py.Dataset],
    predicate: Callable[[str, h5py.Dataset], bool],
    description: str,
) -> h5py.Dataset:
    matches = [(name, ds) for name, ds in datasets.items() if predicate(name, ds)]
    if not matches:
        names = "\n".join(sorted(datasets)[:80])
        raise KeyError(f"Could not find {description}. Available datasets include:\n{names}")
    matches.sort(key=lambda item: (len(item[0]), item[0]))
    return matches[0][1]


def episode_groups(h5: h5py.File) -> list[h5py.Group]:
    candidates = []
    for key, value in h5.items():
        if isinstance(value, h5py.Group) and (key.startswith("traj_") or key.startswith("episode_")):
            candidates.append(value)
    if candidates:
        return candidates
    if any(isinstance(v, h5py.Dataset) for v in h5.values()):
        return [h5]
    raise KeyError("No trajectory groups found in HDF5 file")


def as_array(dataset: h5py.Dataset, length: int | None = None) -> np.ndarray:
    arr = np.asarray(dataset)
    if length is not None and arr.shape[0] != length:
        if arr.shape[0] == length + 1:
            arr = arr[:-1]
        elif arr.shape[0] > length:
            arr = arr[:length]
        else:
            raise ValueError(f"Dataset {dataset.name} has length {arr.shape[0]}, expected {length}")
    return arr

