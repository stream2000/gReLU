#!/usr/bin/env python
"""Plot MREG CTCF motif mutation multi-resolution topology profiles."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ONE_BP_TYPES = ("atac", "dnase", "cage", "rna_seq")
TF_GROUPS = ("CTCF", "RAD21", "SMC3")
HISTONE_GROUPS = ("H3K4me3", "H3K4me2", "H3K27ac", "H2AFZ")

COLORS = {
    "atac": "#1f77b4",
    "dnase": "#2ca02c",
    "cage": "#d62728",
    "rna_seq": "#9467bd",
    "CTCF": "#111111",
    "RAD21": "#ff7f0e",
    "SMC3": "#8c564b",
    "H3K4me3": "#e377c2",
    "H3K4me2": "#bcbd22",
    "H3K27ac": "#17becf",
    "H2AFZ": "#7f7f7f",
}


def _track_label(row: pd.Series) -> str:
    if row.get("output_type") == "chip_tf":
        value = row.get("transcription_factor", "")
        if pd.notna(value) and str(value):
            return str(value)
    if row.get("output_type") == "chip_histone":
        value = row.get("histone_mark", "")
        if pd.notna(value) and str(value):
            return str(value)
    for col in ("track_name", "assay_title"):
        value = row.get(col, "")
        if pd.notna(value) and str(value):
            return str(value)
    return ""


def _fwhm(offsets: np.ndarray, profile: np.ndarray) -> tuple[int | None, int | None, int | None]:
    if profile.size == 0 or float(np.nanmax(profile)) <= 0:
        return None, None, None
    peak = float(np.nanmax(profile))
    idx = np.flatnonzero(profile >= peak / 2.0)
    if idx.size == 0:
        return None, None, None
    left = int(offsets[idx[0]])
    right = int(offsets[idx[-1]])
    return left, right, right - left + 1


def one_bp_profiles(run_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    subdir = run_dir / "resolution_1bp"
    ref = np.load(subdir / "all_track_ref_window.npy", mmap_mode="r")
    alt = np.load(subdir / "all_track_alt_window.npy", mmap_mode="r")
    delta = np.load(subdir / "all_track_delta_window.npy", mmap_mode="r")
    meta = pd.read_csv(subdir / "all_track_metadata.tsv", sep="\t")
    offsets = pd.read_csv(subdir / "all_track_window_bins.tsv", sep="\t")["offset_bp"].to_numpy()

    profile_rows: list[dict] = []
    metric_rows: list[dict] = []
    for output_type in ONE_BP_TYPES:
        idx = np.flatnonzero(meta["output_type"].to_numpy() == output_type)
        if idx.size == 0:
            continue
        ref_block = np.asarray(ref[0, idx, :], dtype=np.float32)
        alt_block = np.asarray(alt[0, idx, :], dtype=np.float32)
        delta_block = np.asarray(delta[0, idx, :], dtype=np.float32)
        signed = delta_block.mean(axis=0)
        abs_mean = np.abs(delta_block).mean(axis=0)
        ref_mean = ref_block.mean(axis=0)
        alt_mean = alt_block.mean(axis=0)
        peak_idx = int(np.argmax(abs_mean))
        left, right, width = _fwhm(offsets, abs_mean)
        metric_rows.append(
            {
                "resolution": "1bp",
                "group": output_type,
                "n_tracks": int(idx.size),
                "peak_offset_bp": int(offsets[peak_idx]),
                "peak_abs_delta_mean": float(abs_mean[peak_idx]),
                "peak_signed_delta_mean": float(signed[peak_idx]),
                "fwhm_left_bp": left,
                "fwhm_right_bp": right,
                "fwhm_bp": width,
                "center_25bp_signed_delta_mean": float(signed[(offsets >= -25) & (offsets <= 25)].mean()),
                "center_25bp_abs_delta_mean": float(abs_mean[(offsets >= -25) & (offsets <= 25)].mean()),
            }
        )
        for offset, ref_v, alt_v, signed_v, abs_v in zip(offsets, ref_mean, alt_mean, signed, abs_mean):
            profile_rows.append(
                {
                    "resolution": "1bp",
                    "group": output_type,
                    "offset_bp": int(offset),
                    "ref_mean": float(ref_v),
                    "alt_mean": float(alt_v),
                    "signed_delta_mean": float(signed_v),
                    "abs_delta_mean": float(abs_v),
                    "normalized_abs_delta": float(abs_v / abs_mean.max()) if abs_mean.max() > 0 else 0.0,
                }
            )
    return pd.DataFrame(profile_rows), pd.DataFrame(metric_rows)


def bp128_profiles(run_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    subdir = run_dir / "resolution_128bp"
    ref = np.load(subdir / "all_track_ref_window.npy", mmap_mode="r")
    alt = np.load(subdir / "all_track_alt_window.npy", mmap_mode="r")
    delta = np.load(subdir / "all_track_delta_window.npy", mmap_mode="r")
    meta = pd.read_csv(subdir / "all_track_metadata.tsv", sep="\t")
    meta["label"] = meta.apply(_track_label, axis=1)
    offsets = pd.read_csv(subdir / "all_track_window_bins.tsv", sep="\t")["offset_bp"].to_numpy()

    wanted = list(TF_GROUPS) + list(HISTONE_GROUPS)
    profile_rows: list[dict] = []
    metric_rows: list[dict] = []
    for group in wanted:
        mask = meta["label"].astype(str).str.upper() == group.upper()
        if group in TF_GROUPS:
            mask &= meta["output_type"].eq("chip_tf")
        else:
            mask &= meta["output_type"].eq("chip_histone")
        idx = np.flatnonzero(mask.to_numpy())
        if idx.size == 0:
            continue
        ref_block = np.asarray(ref[0, idx, :], dtype=np.float32)
        alt_block = np.asarray(alt[0, idx, :], dtype=np.float32)
        delta_block = np.asarray(delta[0, idx, :], dtype=np.float32)
        signed = delta_block.mean(axis=0)
        abs_mean = np.abs(delta_block).mean(axis=0)
        ref_mean = ref_block.mean(axis=0)
        alt_mean = alt_block.mean(axis=0)
        peak_idx = int(np.argmax(abs_mean))
        metric_rows.append(
            {
                "resolution": "128bp",
                "group": group,
                "n_tracks": int(idx.size),
                "peak_offset_bp": int(offsets[peak_idx]),
                "peak_abs_delta_mean": float(abs_mean[peak_idx]),
                "peak_signed_delta_mean": float(signed[peak_idx]),
            }
        )
        for offset, ref_v, alt_v, signed_v, abs_v in zip(offsets, ref_mean, alt_mean, signed, abs_mean):
            profile_rows.append(
                {
                    "resolution": "128bp",
                    "group": group,
                    "offset_bp": int(offset),
                    "ref_mean": float(ref_v),
                    "alt_mean": float(alt_v),
                    "signed_delta_mean": float(signed_v),
                    "abs_delta_mean": float(abs_v),
                    "normalized_abs_delta": float(abs_v / abs_mean.max()) if abs_mean.max() > 0 else 0.0,
                }
            )
    return pd.DataFrame(profile_rows), pd.DataFrame(metric_rows)


def style_axis(ax, title: str, ylabel: str) -> None:
    ax.axvline(0, color="#222222", lw=1.0, alpha=0.8)
    ax.axhline(0, color="#444444", lw=0.8, alpha=0.5)
    ax.axvspan(-5, 7, color="#bbbbbb", alpha=0.18, lw=0)
    ax.set_title(title, loc="left", fontsize=11, fontweight="bold")
    ax.set_xlabel("Offset from edited base (bp)")
    ax.set_ylabel(ylabel)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="#dddddd", lw=0.6, alpha=0.7)


def make_plot(profile_1bp: pd.DataFrame, profile_128: pd.DataFrame, out_prefix: Path) -> None:
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 11,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "figure.dpi": 150,
            "savefig.dpi": 300,
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)

    ax = axes[0, 0]
    for group in ONE_BP_TYPES:
        data = profile_1bp[profile_1bp["group"] == group]
        ax.plot(data["offset_bp"], data["signed_delta_mean"], label=group, color=COLORS[group], lw=1.8)
    style_axis(ax, "A. True 1bp signed effects", "mean alt-ref")
    ax.legend(frameon=False, ncol=2)

    ax = axes[0, 1]
    for group in ONE_BP_TYPES:
        data = profile_1bp[profile_1bp["group"] == group]
        ax.plot(
            data["offset_bp"],
            data["normalized_abs_delta"],
            label=group,
            color=COLORS[group],
            lw=1.8,
        )
    style_axis(ax, "B. True 1bp effect shape", "abs(delta), normalized")
    ax.set_ylim(-0.04, 1.05)
    ax.legend(frameon=False, ncol=2)

    ax = axes[1, 0]
    for group in TF_GROUPS:
        data = profile_128[profile_128["group"] == group]
        ax.plot(
            data["offset_bp"],
            data["signed_delta_mean"],
            marker="o",
            label=group,
            color=COLORS[group],
            lw=1.8,
        )
    style_axis(ax, "C. 128bp CTCF/cohesin ChIP-TF", "mean alt-ref")
    ax.legend(frameon=False)

    ax = axes[1, 1]
    for group in HISTONE_GROUPS:
        data = profile_128[profile_128["group"] == group]
        if data.empty:
            continue
        ax.plot(
            data["offset_bp"],
            data["signed_delta_mean"],
            marker="o",
            label=group,
            color=COLORS[group],
            lw=1.8,
        )
    style_axis(ax, "D. 128bp histone redistribution", "mean alt-ref")
    ax.legend(frameon=False, ncol=2)

    fig.suptitle(
        "MREG CTCF motif SNV: true 1bp accessibility/CAGE/RNA vs 128bp ChIP/histone effects",
        fontsize=13,
        fontweight="bold",
    )
    fig.text(
        0.01,
        0.01,
        "Gray band marks motif span relative to edited base. 1bp panels exclude chip_tf/chip_histone; 128bp panels include ChIP/histone heads.",
        fontsize=8,
        color="#444444",
    )
    for ext in ("png", "svg", "pdf"):
        fig.savefig(out_prefix.with_suffix(f".{ext}"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run_dir",
        default="agent-doc/ism_context/mreg_ctcf_mvp_multires_center_1kb",
        help="Directory containing resolution_1bp and resolution_128bp subdirectories.",
    )
    parser.add_argument("--output_dir", default=None)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    output_dir = Path(args.output_dir) if args.output_dir else run_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    profile_1bp, metrics_1bp = one_bp_profiles(run_dir)
    profile_128, metrics_128 = bp128_profiles(run_dir)

    profile_1bp.to_csv(output_dir / "plot_source_true_1bp_profiles.tsv", sep="\t", index=False)
    profile_128.to_csv(output_dir / "plot_source_128bp_profiles.tsv", sep="\t", index=False)
    pd.concat([metrics_1bp, metrics_128], ignore_index=True).to_csv(
        output_dir / "plot_profile_metrics.tsv", sep="\t", index=False
    )

    out_prefix = output_dir / "mreg_ctcf_multires_topology_overview"
    make_plot(profile_1bp, profile_128, out_prefix)
    print(f"wrote: {out_prefix.with_suffix('.png')}")
    print(f"wrote: {out_prefix.with_suffix('.svg')}")
    print(f"wrote: {out_prefix.with_suffix('.pdf')}")
    print(f"wrote: {output_dir / 'plot_profile_metrics.tsv'}")


if __name__ == "__main__":
    main()
