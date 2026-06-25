# make_borzoi_splits.py

import json
from pathlib import Path

import pandas as pd


SEQ_LEN = 524_288
LABEL_LEN = 196_608
STRIDE = LABEL_LEN

STORE_DIR = Path("dataset_store")
SPLIT_DIR = STORE_DIR / "splits"

CHROM_SIZES_FILE = "/work/Database/Database_fromDocker/Referencedata_mm10/genometable.txt"

AUTOSOMES = {f"chr{i}" for i in range(1, 20)}

SPLIT_PATTERNS = {
    "split_chr10_chr11": {
        "val_chroms": ["chr10"],
        "test_chroms": ["chr11"],
    },
    "split_chr8_chr9": {
        "val_chroms": ["chr8"],
        "test_chroms": ["chr9"],
    },
    "split_chr14_chr15": {
        "val_chroms": ["chr14"],
        "test_chroms": ["chr15"],
    },
    "split_chr5_chr6": {
        "val_chroms": ["chr5"],
        "test_chroms": ["chr6"],
    },
}

TINY_SPLIT_PATTERNS = {
    "split_chr10_chr11_tiny50": {
        "base_split": "split_chr10_chr11",
        "step": 50,
    },
}


def read_chrom_sizes(path: str) -> pd.DataFrame:
    chrom_sizes = pd.read_csv(
        path,
        sep="\t",
        header=None,
        names=["chrom", "size"],
    )
    chrom_sizes = chrom_sizes[chrom_sizes["chrom"].isin(AUTOSOMES)].copy()
    return chrom_sizes


def make_borzoi_intervals(
    chroms: set[str],
    chrom_sizes: pd.DataFrame,
    seq_len: int,
    label_len: int,
    stride: int,
) -> pd.DataFrame:
    flank = (seq_len - label_len) // 2
    rows = []

    for _, row in chrom_sizes.iterrows():
        chrom = row["chrom"]
        chrom_size = int(row["size"])

        if chrom not in chroms:
            continue

        label_start = 0
        while label_start + label_len <= chrom_size:
            label_end = label_start + label_len

            input_start = label_start - flank
            input_end = label_end + flank

            if input_start >= 0 and input_end <= chrom_size:
                rows.append((chrom, input_start, input_end))

            label_start += stride

    return pd.DataFrame(rows, columns=["chrom", "start", "end"])


def write_bed(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, sep="\t", header=False, index=False)


def downsample_intervals(df: pd.DataFrame, step: int) -> pd.DataFrame:
    return df.iloc[::step, :].reset_index(drop=True)


def main() -> None:
    SPLIT_DIR.mkdir(parents=True, exist_ok=True)

    chrom_sizes = read_chrom_sizes(CHROM_SIZES_FILE)
    split_records = {}

    for split_name, spec in SPLIT_PATTERNS.items():
        val_chroms = set(spec["val_chroms"])
        test_chroms = set(spec["test_chroms"])
        train_chroms = AUTOSOMES - val_chroms - test_chroms

        outdir = SPLIT_DIR / split_name
        outdir.mkdir(parents=True, exist_ok=True)

        train_intervals = make_borzoi_intervals(
            train_chroms, chrom_sizes, SEQ_LEN, LABEL_LEN, STRIDE
        )
        val_intervals = make_borzoi_intervals(
            val_chroms, chrom_sizes, SEQ_LEN, LABEL_LEN, STRIDE
        )
        test_intervals = make_borzoi_intervals(
            test_chroms, chrom_sizes, SEQ_LEN, LABEL_LEN, STRIDE
        )
        split_records[split_name] = {
            "train_intervals": train_intervals,
            "val_intervals": val_intervals,
            "test_intervals": test_intervals,
            "train_chroms": sorted(train_chroms),
            "val_chroms": sorted(val_chroms),
            "test_chroms": sorted(test_chroms),
        }

        write_bed(train_intervals, outdir / "train_intervals.bed")
        write_bed(val_intervals, outdir / "val_intervals.bed")
        write_bed(test_intervals, outdir / "test_intervals.bed")

        manifest = {
            "split_name": split_name,
            "genome": "mm10",
            "seq_len": SEQ_LEN,
            "label_len": LABEL_LEN,
            "bin_size": 32,
            "stride": STRIDE,
            "autosomes": sorted(AUTOSOMES),
            "train_chroms": sorted(train_chroms),
            "val_chroms": sorted(val_chroms),
            "test_chroms": sorted(test_chroms),
            "n_train_intervals": int(len(train_intervals)),
            "n_val_intervals": int(len(val_intervals)),
            "n_test_intervals": int(len(test_intervals)),
        }

        with open(outdir / "manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)

        print(
            split_name,
            "train", len(train_intervals),
            "val", len(val_intervals),
            "test", len(test_intervals),
        )

    for split_name, spec in TINY_SPLIT_PATTERNS.items():
        base_split = spec["base_split"]
        step = int(spec["step"])
        base = split_records[base_split]

        outdir = SPLIT_DIR / split_name
        outdir.mkdir(parents=True, exist_ok=True)

        train_intervals = downsample_intervals(base["train_intervals"], step)
        val_intervals = downsample_intervals(base["val_intervals"], step)
        test_intervals = downsample_intervals(base["test_intervals"], step)

        write_bed(train_intervals, outdir / "train_intervals.bed")
        write_bed(val_intervals, outdir / "val_intervals.bed")
        write_bed(test_intervals, outdir / "test_intervals.bed")

        manifest = {
            "split_name": split_name,
            "base_split": base_split,
            "downsample_step": step,
            "genome": "mm10",
            "seq_len": SEQ_LEN,
            "label_len": LABEL_LEN,
            "bin_size": 32,
            "stride": STRIDE,
            "autosomes": sorted(AUTOSOMES),
            "train_chroms": base["train_chroms"],
            "val_chroms": base["val_chroms"],
            "test_chroms": base["test_chroms"],
            "n_train_intervals": int(len(train_intervals)),
            "n_val_intervals": int(len(val_intervals)),
            "n_test_intervals": int(len(test_intervals)),
            "note": "Every 50th interval from the base split for quick memory and pipeline tests.",
        }

        with open(outdir / "manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)

        print(
            split_name,
            "train", len(train_intervals),
            "val", len(val_intervals),
            "test", len(test_intervals),
        )


if __name__ == "__main__":
    main()
