#!/usr/bin/env python
"""Create the approved one-window-per-split Borzoi Saijou smoke profile."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


ASSEMBLY = "mm10"
COORDINATE_CONVENTION = "BED 0-based, half-open"
SEQ_LEN = 524_288
LABEL_LEN = 196_608
BIN_SIZE = 32
FLANK = (SEQ_LEN - LABEL_LEN) // 2

# Approved contract: each input interval is centered on the label interval.
PROFILE_INTERVALS = {
    "train": [("chr1", 32_768, 557_056)],
    "val": [("chr10", 32_768, 557_056)],
    "test": [("chr11", 32_768, 557_056)],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument(
        "--chrom_sizes",
        required=True,
        help="mm10 two-column chromosome-size table used as coordinate authority.",
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def read_chrom_sizes(path: Path) -> dict[str, int]:
    table = pd.read_csv(path, sep="\t", header=None, names=["chrom", "size"])
    return {str(row.chrom): int(row.size) for row in table.itertuples(index=False)}


def label_interval(start: int, end: int) -> tuple[int, int]:
    if end - start != SEQ_LEN:
        raise ValueError(f"Expected a {SEQ_LEN}-bp input window, got {end - start}")
    label_start = start + FLANK
    return label_start, label_start + LABEL_LEN


def validate_profile(chrom_sizes: dict[str, int]) -> dict[str, list[dict[str, int | str]]]:
    validated: dict[str, list[dict[str, int | str]]] = {}
    for split, intervals in PROFILE_INTERVALS.items():
        rows = []
        for chrom, start, end in intervals:
            if chrom not in chrom_sizes:
                raise ValueError(f"{chrom} is absent from the chromosome-size table")
            if start < 0 or end > chrom_sizes[chrom]:
                raise ValueError(f"{chrom}:{start}-{end} lies outside chromosome bounds")
            label_start, label_end = label_interval(start, end)
            rows.append(
                {
                    "chrom": chrom,
                    "input_start": start,
                    "input_end": end,
                    "label_start": label_start,
                    "label_end": label_end,
                }
            )
        validated[split] = rows
    return validated


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    manifest_path = out_dir / "manifest.json"
    if manifest_path.exists() and not args.force:
        print(f"Profile already exists: {manifest_path}")
        return

    chrom_sizes = read_chrom_sizes(Path(args.chrom_sizes))
    validated = validate_profile(chrom_sizes)
    out_dir.mkdir(parents=True, exist_ok=True)
    for split, rows in validated.items():
        pd.DataFrame(
            [(row["chrom"], row["input_start"], row["input_end"]) for row in rows]
        ).to_csv(out_dir / f"{split}_intervals.bed", sep="\t", header=False, index=False)

    manifest = {
        "profile_name": "example_smoke_v1",
        "purpose": "One-epoch LoRA integration smoke test; not a scientific evaluation.",
        "assembly": ASSEMBLY,
        "coordinate_convention": COORDINATE_CONVENTION,
        "reporting_orientation": "reference-forward genomic coordinates",
        "transcript": None,
        "input_window_bp": SEQ_LEN,
        "label_window_bp": LABEL_LEN,
        "label_bin_bp": BIN_SIZE,
        "coordinate_authority": str(Path(args.chrom_sizes).resolve()),
        "intervals": validated,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Wrote approved smoke profile: {out_dir}")


if __name__ == "__main__":
    main()
