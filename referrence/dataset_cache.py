# dataset_cache.py

import json
import pickle
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import grelu.data.dataset


TargetMode = Literal[
    "poisson",
    "poisson_multinomial",
    "log1p_mse",
    "borzoi_squash_mse",
]


STORE_DIR = Path("dataset_store")
SPLIT_DIR = STORE_DIR / "splits"
CACHE_DIR = STORE_DIR / "caches"


def borzoi_like_squash(y):
    y = np.maximum(y, 0)
    z = y ** 0.75
    return np.where(z <= 384, z, 384 + np.sqrt(z - 384))


def get_label_settings(target_mode: TargetMode):
    if target_mode in {"poisson", "poisson_multinomial"}:
        return {
            "label_aggfunc": np.sum,
            "label_transform_func": None,
            "min_label_clip": 0,
            "max_label_clip": None,
        }

    if target_mode == "log1p_mse":
        return {
            "label_aggfunc": np.mean,
            "label_transform_func": np.log1p,
            "min_label_clip": 0,
            "max_label_clip": None,
        }

    if target_mode == "borzoi_squash_mse":
        return {
            "label_aggfunc": np.sum,
            "label_transform_func": borzoi_like_squash,
            "min_label_clip": 0,
            "max_label_clip": None,
        }

    raise ValueError(f"Unknown target_mode: {target_mode}")


def read_intervals(path: Path) -> pd.DataFrame:
    return pd.read_csv(
        path,
        sep="\t",
        names=["chrom", "start", "end"],
    )


def save_pickle(obj, path: Path) -> None:
    with open(path, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_pickle(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)


def build_dataset_pair(
    split_name: str,
    target_mode: TargetMode,
    bw_files: list[str],
    task_names: list[str],
    genome: str = "mm10",
    seq_len: int = 524_288,
    label_len: int = 196_608,
    bin_size: int = 32,
    force_rebuild: bool = False,
):
    if len(bw_files) != len(task_names):
        raise ValueError("bw_files and task_names must have the same length.")

    split_path = SPLIT_DIR / split_name
    if not split_path.exists():
        raise FileNotFoundError(f"Split not found: {split_path}")

    cache_path = CACHE_DIR / f"{split_name}__{target_mode}"
    train_pkl = cache_path / "train_dataset.pkl"
    val_pkl = cache_path / "val_dataset.pkl"

    if train_pkl.exists() and val_pkl.exists() and not force_rebuild:
        print(f"Loading cached datasets: {cache_path}")
        train_dataset = load_pickle(train_pkl)
        val_dataset = load_pickle(val_pkl)
        return train_dataset, val_dataset

    print(f"Building datasets: split={split_name}, target_mode={target_mode}")
    cache_path.mkdir(parents=True, exist_ok=True)

    train_intervals = read_intervals(split_path / "train_intervals.bed")
    val_intervals = read_intervals(split_path / "val_intervals.bed")

    label_settings = get_label_settings(target_mode)

    train_dataset = grelu.data.dataset.BigWigSeqDataset(
        intervals=train_intervals,
        bw_files=bw_files,
        tasks=task_names,
        seq_len=seq_len,
        genome=genome,
        label_len=label_len,
        bin_size=bin_size,
        label_aggfunc=label_settings["label_aggfunc"],
        rc=True,
        max_seq_shift=0,
        max_pair_shift=0,
        min_label_clip=label_settings["min_label_clip"],
        max_label_clip=label_settings["max_label_clip"],
        label_transform_func=label_settings["label_transform_func"],
        augment_mode="random",
    )

    val_dataset = grelu.data.dataset.BigWigSeqDataset(
        intervals=val_intervals,
        bw_files=bw_files,
        tasks=task_names,
        seq_len=seq_len,
        genome=genome,
        label_len=label_len,
        bin_size=bin_size,
        label_aggfunc=label_settings["label_aggfunc"],
        rc=False,
        max_seq_shift=0,
        max_pair_shift=0,
        min_label_clip=label_settings["min_label_clip"],
        max_label_clip=label_settings["max_label_clip"],
        label_transform_func=label_settings["label_transform_func"],
    )

    save_pickle(train_dataset, train_pkl)
    save_pickle(val_dataset, val_pkl)

    manifest = {
        "split_name": split_name,
        "target_mode": target_mode,
        "genome": genome,
        "seq_len": seq_len,
        "label_len": label_len,
        "bin_size": bin_size,
        "bw_files": bw_files,
        "task_names": task_names,
        "n_tasks": len(task_names),
        "train_pickle": str(train_pkl),
        "val_pickle": str(val_pkl),
        "note": "Pickle cache is for local reuse only. BED and manifest are the reproducible source.",
    }

    with open(cache_path / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Saved cached datasets: {cache_path}")
    return train_dataset, val_dataset
