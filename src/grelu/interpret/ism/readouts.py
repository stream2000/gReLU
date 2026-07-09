"""Readout interval helpers for profile-based ISM."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class ReadoutWindow:
    """A genomic readout interval to summarize model output."""

    readout_id: str
    chrom: str
    start: int
    end: int
    anchor: int
    gene: str = ""
    role: str = ""


def map_readouts_to_bins(
    readouts: pd.DataFrame,
    *,
    output_start: int,
    n_bins: int,
    bin_size: int,
    chrom: str | None = None,
) -> dict[str, tuple[int, int]]:
    """Map genomic readout intervals to half-open output-bin ranges."""

    mapping: dict[str, tuple[int, int]] = {}
    output_end = int(output_start) + int(n_bins) * int(bin_size)
    for row in readouts.itertuples(index=False):
        if chrom is not None and str(getattr(row, "chrom")) != str(chrom):
            continue
        readout_id = str(getattr(row, "readout_id"))
        start = int(getattr(row, "start"))
        end = int(getattr(row, "end"))
        if end <= start:
            raise ValueError(f"Invalid readout interval for {readout_id}: {start}-{end}")
        if start < output_start or end > output_end:
            raise ValueError(
                f"Readout {readout_id} ({start}-{end}) is outside model output "
                f"{output_start}-{output_end}"
            )
        left = (start - output_start) // bin_size
        right = (end - output_start + bin_size - 1) // bin_size
        if left >= right:
            raise ValueError(f"Readout {readout_id} maps to no output bins")
        mapping[readout_id] = (int(left), int(right))
    return mapping


def summarize_profile_pair(
    ref: pd.Series | object,
    alt: pd.Series | object,
    *,
    pseudocount: float = 1.0,
) -> dict[str, float]:
    """Summarize one REF/ALT profile slice."""

    import numpy as np

    ref_arr = np.maximum(np.asarray(ref, dtype=float), 0.0)
    alt_arr = np.maximum(np.asarray(alt, dtype=float), 0.0)
    delta = alt_arr - ref_arr
    log2fc = np.log2((alt_arr + pseudocount) / (ref_arr + pseudocount))
    peak_ref = int(np.argmax(ref_arr)) if ref_arr.size else -1
    peak_alt = int(np.argmax(alt_arr)) if alt_arr.size else -1
    return {
        "n_bins": int(ref_arr.size),
        "ref_mean": float(ref_arr.mean()),
        "alt_mean": float(alt_arr.mean()),
        "signed_delta_mean": float(delta.mean()),
        "absolute_delta_mean": float(np.abs(delta).mean()),
        "log2fc_mean": float(log2fc.mean()),
        "absolute_log2fc_mean": float(np.abs(log2fc).mean()),
        "delta_peak": float(np.max(np.abs(delta))),
        "ref_peak_bin": peak_ref,
        "alt_peak_bin": peak_alt,
    }
