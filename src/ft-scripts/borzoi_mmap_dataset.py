"""Borzoi fine-tuning dataset backed by 32 bp binned label memmaps."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import pyBigWig
import torch
from pyfaidx import Fasta

from grelu.data.augment import Augmenter, _split_overall_idx
from grelu.data.dataset import LabeledSeqDataset
from grelu.sequence.format import BASE_TO_INDEX_HASH, indices_to_one_hot
from grelu.sequence.utils import resize


TargetMode = Literal[
    "poisson",
    "poisson_multinomial",
    "log1p_mse",
    "borzoi_squash_mse",
]


def borzoi_like_squash(y: np.ndarray) -> np.ndarray:
    y = np.maximum(y, 0)
    z = y**0.75
    return np.where(z <= 384, z, 384 + np.sqrt(z - 384))


def transform_binned_signal(raw: np.ndarray, target_mode: TargetMode, bin_size: int) -> np.ndarray:
    raw = np.nan_to_num(raw)
    raw = np.maximum(raw, 0)
    raw = raw.reshape(-1, bin_size)

    if target_mode in {"poisson", "poisson_multinomial"}:
        out = raw.sum(axis=1)
    elif target_mode == "log1p_mse":
        out = np.log1p(raw.mean(axis=1))
    elif target_mode == "borzoi_squash_mse":
        out = borzoi_like_squash(raw.sum(axis=1))
    else:
        raise ValueError(target_mode)

    return out.astype(np.float32, copy=False)


def read_intervals(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t", names=["chrom", "start", "end"])


def build_label_mmap(
    intervals: pd.DataFrame,
    bw_files: list[str],
    out_path: Path,
    target_mode: TargetMode,
    label_len: int,
    bin_size: int,
) -> None:
    label_intervals = resize(intervals, seq_len=label_len, input_type="intervals")
    n_bins = label_len // bin_size
    labels = np.lib.format.open_memmap(
        out_path,
        mode="w+",
        dtype=np.float32,
        shape=(len(intervals), len(bw_files), n_bins),
    )

    for task_idx, bw_file in enumerate(bw_files):
        with pyBigWig.open(str(bw_file), "r") as bw:
            for row_idx, row in enumerate(label_intervals.itertuples(index=False)):
                raw = bw.values(row.chrom, int(row.start), int(row.end), numpy=True)
                labels[row_idx, task_idx, :] = transform_binned_signal(
                    raw, target_mode=target_mode, bin_size=bin_size
                )
    labels.flush()


def cache_path(cache_dir: Path, split_name: str, target_mode: TargetMode, bin_size: int) -> Path:
    return cache_dir / f"{split_name}__{target_mode}__bin{bin_size}"


def sized_cache_path(
    cache_dir: Path,
    split_name: str,
    target_mode: TargetMode,
    seq_len: int,
    label_len: int,
    bin_size: int,
) -> Path:
    return cache_dir / f"{split_name}__{target_mode}__seq{seq_len}__label{label_len}__bin{bin_size}"


def manifest_matches(
    manifest_path: Path,
    split_name: str,
    target_mode: TargetMode,
    genome: str,
    seq_len: int,
    label_len: int,
    bin_size: int,
    bw_files: list[str],
    task_names: list[str],
) -> bool:
    if not manifest_path.exists():
        return False
    with manifest_path.open() as handle:
        manifest = json.load(handle)
    expected = {
        "split_name": split_name,
        "target_mode": target_mode,
        "genome": genome,
        "seq_len": seq_len,
        "label_len": label_len,
        "bin_size": bin_size,
        "n_bins": label_len // bin_size,
        "bw_files": bw_files,
        "task_names": task_names,
    }
    return all(manifest.get(key) == value for key, value in expected.items())


def ensure_borzoi_mmap_cache(
    split_name: str,
    target_mode: TargetMode,
    bw_files: list[str],
    task_names: list[str],
    genome: str,
    split_dir: Path,
    cache_dir: Path,
    seq_len: int = 524_288,
    label_len: int = 196_608,
    bin_size: int = 32,
    force_rebuild: bool = False,
) -> Path:
    if len(bw_files) != len(task_names):
        raise ValueError("bw_files and task_names must have the same length.")
    if label_len % bin_size != 0:
        raise ValueError("label_len must be divisible by bin_size.")

    split_path = split_dir / split_name
    if not split_path.exists():
        raise FileNotFoundError(f"Split not found: {split_path}")

    legacy_dir = cache_path(cache_dir, split_name, target_mode, bin_size)
    legacy_manifest = legacy_dir / "manifest.json"
    if not force_rebuild and manifest_matches(
        legacy_manifest,
        split_name=split_name,
        target_mode=target_mode,
        genome=genome,
        seq_len=seq_len,
        label_len=label_len,
        bin_size=bin_size,
        bw_files=bw_files,
        task_names=task_names,
    ):
        out_dir = legacy_dir
    else:
        out_dir = sized_cache_path(
            cache_dir,
            split_name=split_name,
            target_mode=target_mode,
            seq_len=seq_len,
            label_len=label_len,
            bin_size=bin_size,
        )
    train_labels = out_dir / "train_labels.npy"
    val_labels = out_dir / "val_labels.npy"
    manifest = out_dir / "manifest.json"

    if train_labels.exists() and val_labels.exists() and manifest.exists() and not force_rebuild:
        return out_dir

    out_dir.mkdir(parents=True, exist_ok=True)
    train_intervals = read_intervals(split_path / "train_intervals.bed")
    val_intervals = read_intervals(split_path / "val_intervals.bed")

    build_label_mmap(train_intervals, bw_files, train_labels, target_mode, label_len, bin_size)
    build_label_mmap(val_intervals, bw_files, val_labels, target_mode, label_len, bin_size)

    with open(manifest, "w") as handle:
        json.dump(
            {
                "split_name": split_name,
                "target_mode": target_mode,
                "genome": genome,
                "seq_len": seq_len,
                "label_len": label_len,
                "bin_size": bin_size,
                "n_bins": label_len // bin_size,
                "bw_files": bw_files,
                "task_names": task_names,
                "train_shape": list(np.load(train_labels, mmap_mode="r").shape),
                "val_shape": list(np.load(val_labels, mmap_mode="r").shape),
                "note": "Labels are pre-aggregated to bin_size and stored as mmap-compatible .npy arrays.",
            },
            handle,
            indent=2,
        )
    return out_dir


class BorzoiMmapSeqDataset(LabeledSeqDataset):
    """Labeled sequence dataset with lazy FASTA reads and mmap-backed labels.

    This class intentionally does not call ``LabeledSeqDataset.__init__``,
    because the base implementation materializes all sequences and raw labels in
    memory. It still inherits from LabeledSeqDataset to satisfy gReLU Lightning
    loader checks.
    """

    def __init__(
        self,
        intervals: pd.DataFrame,
        labels_path: Path,
        genome: str,
        tasks: list[str],
        seq_len: int,
        label_len: int,
        bin_size: int,
        rc: bool,
        augment_mode: str = "serial",
    ) -> None:
        self.intervals = intervals.reset_index(drop=True)
        self.sequence_intervals = resize(
            self.intervals,
            seq_len=seq_len,
            input_type="intervals",
        ).reset_index(drop=True)
        self.labels_path = Path(labels_path)
        self.genome_file = genome
        self.seq_len = seq_len
        self.label_len = label_len
        self.bin_size = bin_size
        self.binned_label_len = label_len // bin_size
        self.tasks = pd.DataFrame(index=tasks)
        self.n_tasks = len(tasks)
        self.rc = rc
        self.max_seq_shift = 0
        self.max_pair_shift = 0
        self.padded_seq_len = seq_len
        self.padded_label_len = label_len
        self.n_seqs = len(self.intervals)
        self.n_alleles = 1
        self.predict = False
        self._labels = None
        self._genome = None
        self.augmenter = Augmenter(
            rc=rc,
            max_seq_shift=0,
            max_pair_shift=0,
            seq_len=seq_len,
            label_len=self.binned_label_len,
            mode=augment_mode,
        )
        self.n_augmented = len(self.augmenter)

    @property
    def labels(self):
        if self._labels is None:
            self._labels = np.load(self.labels_path, mmap_mode="r")
        return self._labels

    @property
    def genome(self):
        if self._genome is None:
            self._genome = Fasta(self.genome_file, rebuild=False)
        return self._genome

    def close_handles(self) -> None:
        self._labels = None
        self._genome = None

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state["_labels"] = None
        state["_genome"] = None
        return state

    def __len__(self) -> int:
        return self.n_seqs * self.n_augmented

    def _sequence_indices(self, seq_idx: int) -> np.ndarray:
        row = self.sequence_intervals.iloc[seq_idx]
        seq_record = self.genome.get_seq(row["chrom"], int(row["start"]) + 1, int(row["end"]))
        seq = getattr(seq_record, "seq", str(seq_record)).upper()
        if len(seq) != self.seq_len:
            if len(seq) < self.seq_len:
                seq = seq + ("N" * (self.seq_len - len(seq)))
            else:
                seq = seq[: self.seq_len]
        n_index = BASE_TO_INDEX_HASH["N"]
        return np.fromiter(
            (BASE_TO_INDEX_HASH.get(base, n_index) for base in seq),
            dtype=np.int8,
            count=self.seq_len,
        )

    def __getitem__(self, idx: int):
        seq_idx, augment_idx = _split_overall_idx(idx, (self.n_seqs, self.n_augmented))
        seq = self._sequence_indices(seq_idx)
        label = np.array(self.labels[seq_idx], dtype=np.float32, copy=True)
        seq, label = self.augmenter(seq=seq, label=label, idx=augment_idx)
        seq = indices_to_one_hot(seq)
        if self.predict:
            return seq
        return seq, torch.as_tensor(label, dtype=torch.float32)


def build_dataset_pair(
    split_name: str,
    target_mode: TargetMode,
    bw_files: list[str],
    task_names: list[str],
    genome: str,
    split_dir: Path,
    cache_dir: Path,
    seq_len: int = 524_288,
    label_len: int = 196_608,
    bin_size: int = 32,
    force_rebuild: bool = False,
):
    cache = ensure_borzoi_mmap_cache(
        split_name=split_name,
        target_mode=target_mode,
        bw_files=bw_files,
        task_names=task_names,
        genome=genome,
        split_dir=split_dir,
        cache_dir=cache_dir,
        seq_len=seq_len,
        label_len=label_len,
        bin_size=bin_size,
        force_rebuild=force_rebuild,
    )
    split_path = split_dir / split_name
    train_intervals = read_intervals(split_path / "train_intervals.bed")
    val_intervals = read_intervals(split_path / "val_intervals.bed")
    train_dataset = BorzoiMmapSeqDataset(
        intervals=train_intervals,
        labels_path=cache / "train_labels.npy",
        genome=genome,
        tasks=task_names,
        seq_len=seq_len,
        label_len=label_len,
        bin_size=bin_size,
        rc=True,
        augment_mode="random",
    )
    val_dataset = BorzoiMmapSeqDataset(
        intervals=val_intervals,
        labels_path=cache / "val_labels.npy",
        genome=genome,
        tasks=task_names,
        seq_len=seq_len,
        label_len=label_len,
        bin_size=bin_size,
        rc=False,
    )
    return train_dataset, val_dataset
