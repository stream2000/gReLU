#!/usr/bin/env python
"""Summarize per-track variant effects in a centered local mask.

This is the downstream MVP view for ``run_tf_context.py --feature_mode
track_window``. It keeps all AlphaGenome tracks separable and summarizes the
ref/alt windows in a center mask, defaulting to the central 1 kb.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def track_label(row: pd.Series) -> str:
    for col in ["transcription_factor", "histone_mark", "track_name", "assay_title"]:
        value = row.get(col, "")
        if pd.notna(value) and str(value) != "":
            return str(value)
    return ""


def center_mask(window_bins: pd.DataFrame, center_bp: int) -> np.ndarray:
    if "offset_bp" not in window_bins.columns:
        raise ValueError("all_track_window_bins.tsv must contain offset_bp")
    half = float(center_bp) / 2.0
    mask = window_bins["offset_bp"].abs().to_numpy() <= half
    if not np.any(mask):
        center_idx = int(np.argmin(np.abs(window_bins["offset_bp"].to_numpy())))
        mask[center_idx] = True
    return mask


def infer_bin_size_bp(window_bins: pd.DataFrame) -> int:
    """Infer model bin size from adjacent ``offset_bp`` values."""

    offsets = np.sort(window_bins["offset_bp"].astype(int).unique())
    if len(offsets) < 2:
        raise ValueError("Need at least two window bins to infer bin size")
    diffs = np.diff(offsets)
    return int(np.median(diffs))


def center_bp_signal_coordinates(
    window_bins: pd.DataFrame,
    center_bp: int,
    bin_size_bp: int | None = None,
) -> pd.DataFrame:
    """Map each requested center bp to the nearest saved model bin."""

    if "offset_bp" not in window_bins.columns:
        raise ValueError("all_track_window_bins.tsv must contain offset_bp")
    bin_size_bp = infer_bin_size_bp(window_bins) if bin_size_bp is None else int(bin_size_bp)
    if bin_size_bp <= 0:
        raise ValueError(f"bin_size_bp must be positive, got {bin_size_bp}")

    start_offset = -(int(center_bp) // 2)
    bp_offsets = start_offset + np.arange(int(center_bp), dtype=int)
    bin_offsets = window_bins["offset_bp"].to_numpy(dtype=int)
    nearest = np.abs(bp_offsets[:, None] - bin_offsets[None, :]).argmin(axis=1)

    if bin_size_bp == 1:
        coverage_left = int(bin_offsets.min())
        coverage_right = int(bin_offsets.max() + 1)
    else:
        coverage_left = int(np.floor(bin_offsets.min() - bin_size_bp / 2.0))
        coverage_right = int(np.ceil(bin_offsets.max() + bin_size_bp / 2.0))
    if bp_offsets.min() < coverage_left or bp_offsets.max() >= coverage_right:
        raise ValueError(
            f"Requested center_bp={center_bp} exceeds saved window coverage "
            f"[{coverage_left}, {coverage_right}) bp. Rerun track_window with a wider window."
        )

    coords = pd.DataFrame(
        {
            "bp_index": np.arange(int(center_bp), dtype=int),
            "bp_offset": bp_offsets,
            "source_window_bin_index": nearest.astype(int),
            "source_bin_offset_bp": bin_offsets[nearest].astype(int),
        }
    )
    if "model_bin_index" in window_bins.columns:
        model_bins = window_bins["model_bin_index"].to_numpy(dtype=int)
        coords["source_model_bin_index"] = model_bins[nearest].astype(int)
    coords["bin_size_bp"] = bin_size_bp
    return coords


def expand_center_bp_signals(
    ref: np.ndarray,
    alt: np.ndarray,
    window_bins: pd.DataFrame,
    center_bp: int = 1000,
    bin_size_bp: int | None = None,
    pseudocount: float = 1.0,
) -> tuple[dict[str, np.ndarray], pd.DataFrame]:
    """Expand saved model-bin signals onto a 1-bp axis inside the center mask."""

    if ref.shape != alt.shape:
        raise ValueError(f"Shape mismatch: ref={ref.shape}, alt={alt.shape}")
    if ref.ndim != 3:
        raise ValueError(f"Expected sites x tracks x bins arrays, got {ref.shape}")
    if len(window_bins) != ref.shape[2]:
        raise ValueError(f"Window-bin rows {len(window_bins)} != n_bins {ref.shape[2]}")

    coords = center_bp_signal_coordinates(window_bins, center_bp=center_bp, bin_size_bp=bin_size_bp)
    source_bins = coords["source_window_bin_index"].to_numpy(dtype=int)
    ref_bp = np.asarray(ref[:, :, source_bins], dtype=np.float32)
    alt_bp = np.asarray(alt[:, :, source_bins], dtype=np.float32)
    delta_bp = (alt_bp - ref_bp).astype(np.float32, copy=False)
    log2fc_bp = (
        np.log2(float(pseudocount) + np.maximum(alt_bp, 0.0))
        - np.log2(float(pseudocount) + np.maximum(ref_bp, 0.0))
    ).astype(np.float32, copy=False)
    return (
        {
            "ref": ref_bp,
            "alt": alt_bp,
            "raw_change": delta_bp,
            "log2fc": log2fc_bp,
        },
        coords,
    )


def _variant_columns(variants: pd.DataFrame) -> list[str]:
    keep = [
        "site_id",
        "paired_site_id",
        "gene",
        "dic_class",
        "control_type",
        "mutation_strategy",
        "chrom",
        "position",
        "ref",
        "alt",
        "variant_start",
        "variant_end",
        "motif_start",
        "motif_end",
        "matched_seq",
        "strand",
        "peak_score",
        "motif_score",
        "distance_to_gene_bp",
    ]
    return [col for col in keep if col in variants.columns]


def summarize_center_track_effects(
    ref: np.ndarray,
    alt: np.ndarray,
    metadata: pd.DataFrame,
    variants: pd.DataFrame,
    window_bins: pd.DataFrame,
    center_bp: int = 1000,
    pseudocount: float = 1.0,
) -> pd.DataFrame:
    """Return long-form per-site, per-track center-mask effect metrics."""

    if ref.shape != alt.shape:
        raise ValueError(f"Shape mismatch: ref={ref.shape}, alt={alt.shape}")
    if ref.ndim != 3:
        raise ValueError(f"Expected sites x tracks x bins arrays, got {ref.shape}")
    n_sites, n_tracks, n_bins = ref.shape
    if len(metadata) != n_tracks:
        raise ValueError(f"Track metadata rows {len(metadata)} != n_tracks {n_tracks}")
    if len(variants) != n_sites:
        raise ValueError(f"Variant rows {len(variants)} != n_sites {n_sites}")
    if len(window_bins) != n_bins:
        raise ValueError(f"Window-bin rows {len(window_bins)} != n_bins {n_bins}")

    mask = center_mask(window_bins, center_bp)
    ref_center = np.asarray(ref[:, :, mask], dtype=np.float32)
    alt_center = np.asarray(alt[:, :, mask], dtype=np.float32)
    raw_change = alt_center - ref_center
    log2fc = np.log2(float(pseudocount) + np.maximum(alt_center, 0.0)) - np.log2(
        float(pseudocount) + np.maximum(ref_center, 0.0)
    )

    meta = metadata.copy().reset_index(drop=True)
    if "global_track_index" not in meta.columns:
        meta["global_track_index"] = np.arange(n_tracks)
    meta["target"] = meta.apply(track_label, axis=1)

    variant_meta = variants[_variant_columns(variants)].reset_index(drop=True)
    site_index = np.repeat(np.arange(n_sites), n_tracks)
    track_index = np.tile(np.arange(n_tracks), n_sites)
    out = pd.DataFrame(
        {
            "site_row_index": site_index,
            "track_row_index": track_index,
            "center_bp": int(center_bp),
            "n_center_bins": int(mask.sum()),
            "ref_mean": ref_center.mean(axis=2).reshape(-1),
            "alt_mean": alt_center.mean(axis=2).reshape(-1),
            "pseudocount": float(pseudocount),
            "raw_signed_change": raw_change.mean(axis=2).reshape(-1),
            "raw_change_magnitude": np.abs(raw_change).mean(axis=2).reshape(-1),
            "raw_max_change_magnitude": np.abs(raw_change).max(axis=2).reshape(-1),
            "raw_change_l2": np.linalg.norm(raw_change, axis=2).reshape(-1),
            "signed_log2_fold_change": log2fc.mean(axis=2).reshape(-1),
            "log2_fold_change_magnitude": np.abs(log2fc).mean(axis=2).reshape(-1),
            "loss_log2_fold_change_magnitude": np.maximum(-log2fc, 0.0).mean(axis=2).reshape(-1),
            "gain_log2_fold_change_magnitude": np.maximum(log2fc, 0.0).mean(axis=2).reshape(-1),
        }
    )
    out = out.join(variant_meta.iloc[site_index].reset_index(drop=True))

    meta_cols = [
        "global_track_index",
        "output_type",
        "track_index",
        "target",
        "transcription_factor",
        "histone_mark",
        "biosample_name",
        "assay_title",
        "track_name",
    ]
    meta_cols = [col for col in meta_cols if col in meta.columns]
    out = out.join(meta.loc[track_index, meta_cols].reset_index(drop=True))
    return out


def write_center_track_outputs(
    run_dir: str | Path,
    center_bp: int = 1000,
    top_n: int = 50,
    output_prefix: str = "center_1kb",
    save_bp_signals: bool = True,
    pseudocount: float = 1.0,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    run_dir = Path(run_dir)
    ref = np.load(run_dir / "all_track_ref_window.npy", mmap_mode="r")
    alt = np.load(run_dir / "all_track_alt_window.npy", mmap_mode="r")
    metadata = pd.read_csv(run_dir / "all_track_metadata.tsv", sep="\t", low_memory=False)
    variants = pd.read_csv(run_dir / "variants.tsv", sep="\t")
    bins = pd.read_csv(run_dir / "all_track_window_bins.tsv", sep="\t")

    effects = summarize_center_track_effects(
        ref,
        alt,
        metadata=metadata,
        variants=variants,
        window_bins=bins,
        center_bp=center_bp,
        pseudocount=pseudocount,
    )
    effects_path = run_dir / f"{output_prefix}_track_effects.tsv"
    effects.to_csv(effects_path, sep="\t", index=False)

    top = (
        effects.sort_values(["site_id", "log2_fold_change_magnitude"], ascending=[True, False])
        .groupby("site_id", dropna=False)
        .head(top_n)
        .copy()
    )
    top["rank_in_site"] = top.groupby("site_id", dropna=False)["log2_fold_change_magnitude"].rank(
        ascending=False, method="first"
    ).astype(int)
    top.to_csv(run_dir / f"{output_prefix}_top{top_n}_tracks_by_site.tsv", sep="\t", index=False)

    summary = (
        effects.groupby(["output_type", "target"], dropna=False)
        .agg(
            track_count=("global_track_index", "nunique"),
            typical_log2fc_magnitude=("log2_fold_change_magnitude", "median"),
            top_log2fc_magnitude=("log2_fold_change_magnitude", "max"),
            total_log2fc_magnitude=("log2_fold_change_magnitude", "sum"),
            typical_loss_log2fc_magnitude=("loss_log2_fold_change_magnitude", "median"),
            typical_gain_log2fc_magnitude=("gain_log2_fold_change_magnitude", "median"),
            typical_raw_change_magnitude=("raw_change_magnitude", "median"),
            top_raw_change_magnitude=("raw_change_magnitude", "max"),
        )
        .reset_index()
        .sort_values("top_log2fc_magnitude", ascending=False)
    )
    summary.to_csv(run_dir / f"{output_prefix}_track_effect_summary.tsv", sep="\t", index=False)
    if save_bp_signals:
        bp_signals, bp_coords = expand_center_bp_signals(
            ref,
            alt,
            window_bins=bins,
            center_bp=center_bp,
            pseudocount=pseudocount,
        )
        for name, array in bp_signals.items():
            np.save(run_dir / f"{output_prefix}_{name}_bp_signal.npy", array)
        bp_coords.to_csv(run_dir / f"{output_prefix}_bp_signal_coordinates.tsv", sep="\t", index=False)
    return effects, top, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--center_bp", type=int, default=1000)
    parser.add_argument("--top_n", type=int, default=50)
    parser.add_argument("--output_prefix", default="center_1kb")
    parser.add_argument(
        "--pseudocount",
        type=float,
        default=1.0,
        help="Pseudocount for log2((mut+p)/(ref+p)) fold-change metrics.",
    )
    parser.add_argument(
        "--no_save_bp_signals",
        action="store_true",
        help="Skip dense site x track x bp raw-signal arrays.",
    )
    args = parser.parse_args()

    effects, top, summary = write_center_track_outputs(
        args.run_dir,
        center_bp=args.center_bp,
        top_n=args.top_n,
        output_prefix=args.output_prefix,
        save_bp_signals=not args.no_save_bp_signals,
        pseudocount=args.pseudocount,
    )
    run_dir = Path(args.run_dir)
    print(f"wrote: {run_dir / f'{args.output_prefix}_track_effects.tsv'} ({len(effects)} rows)")
    print(f"wrote: {run_dir / f'{args.output_prefix}_top{args.top_n}_tracks_by_site.tsv'}")
    print(f"wrote: {run_dir / f'{args.output_prefix}_track_effect_summary.tsv'}")
    if not args.no_save_bp_signals:
        print(f"wrote: {run_dir / f'{args.output_prefix}_ref_bp_signal.npy'}")
        print(f"wrote: {run_dir / f'{args.output_prefix}_alt_bp_signal.npy'}")
        print(f"wrote: {run_dir / f'{args.output_prefix}_raw_change_bp_signal.npy'}")
        print(f"wrote: {run_dir / f'{args.output_prefix}_log2fc_bp_signal.npy'}")
        print(f"wrote: {run_dir / f'{args.output_prefix}_bp_signal_coordinates.tsv'}")
    print("\nTop tracks:")
    view_cols = [
        "site_id",
        "rank_in_site",
        "global_track_index",
        "output_type",
        "target",
        "biosample_name",
        "signed_log2_fold_change",
        "log2_fold_change_magnitude",
        "loss_log2_fold_change_magnitude",
        "gain_log2_fold_change_magnitude",
        "raw_signed_change",
        "raw_change_magnitude",
        "raw_max_change_magnitude",
    ]
    print(top[[col for col in view_cols if col in top.columns]].head(args.top_n).to_string(index=False))
    print("\nTop output/target summary:")
    print(summary.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
