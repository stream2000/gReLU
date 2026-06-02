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
    delta = alt_center - ref_center
    log_delta = np.log2(1.0 + np.maximum(alt_center, 0.0)) - np.log2(
        1.0 + np.maximum(ref_center, 0.0)
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
            "delta_mean": delta.mean(axis=2).reshape(-1),
            "delta_abs_mean": np.abs(delta).mean(axis=2).reshape(-1),
            "delta_abs_max": np.abs(delta).max(axis=2).reshape(-1),
            "delta_l2": np.linalg.norm(delta, axis=2).reshape(-1),
            "log1p_delta_mean": log_delta.mean(axis=2).reshape(-1),
            "log1p_abs_mean": np.abs(log_delta).mean(axis=2).reshape(-1),
            "log1p_loss_mean": np.maximum(-log_delta, 0.0).mean(axis=2).reshape(-1),
            "log1p_gain_mean": np.maximum(log_delta, 0.0).mean(axis=2).reshape(-1),
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
    )
    effects_path = run_dir / f"{output_prefix}_track_effects.tsv"
    effects.to_csv(effects_path, sep="\t", index=False)

    top = (
        effects.sort_values(["site_id", "delta_abs_mean"], ascending=[True, False])
        .groupby("site_id", dropna=False)
        .head(top_n)
        .copy()
    )
    top["rank_in_site"] = top.groupby("site_id", dropna=False)["delta_abs_mean"].rank(
        ascending=False, method="first"
    ).astype(int)
    top.to_csv(run_dir / f"{output_prefix}_top{top_n}_tracks_by_site.tsv", sep="\t", index=False)

    summary = (
        effects.groupby(["output_type", "target"], dropna=False)
        .agg(
            n_tracks=("global_track_index", "nunique"),
            mean_delta_abs_mean=("delta_abs_mean", "mean"),
            max_delta_abs_mean=("delta_abs_mean", "max"),
            mean_log1p_loss=("log1p_loss_mean", "mean"),
            mean_log1p_gain=("log1p_gain_mean", "mean"),
        )
        .reset_index()
        .sort_values("max_delta_abs_mean", ascending=False)
    )
    summary.to_csv(run_dir / f"{output_prefix}_track_effect_summary.tsv", sep="\t", index=False)
    return effects, top, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--center_bp", type=int, default=1000)
    parser.add_argument("--top_n", type=int, default=50)
    parser.add_argument("--output_prefix", default="center_1kb")
    args = parser.parse_args()

    effects, top, summary = write_center_track_outputs(
        args.run_dir,
        center_bp=args.center_bp,
        top_n=args.top_n,
        output_prefix=args.output_prefix,
    )
    run_dir = Path(args.run_dir)
    print(f"wrote: {run_dir / f'{args.output_prefix}_track_effects.tsv'} ({len(effects)} rows)")
    print(f"wrote: {run_dir / f'{args.output_prefix}_top{args.top_n}_tracks_by_site.tsv'}")
    print(f"wrote: {run_dir / f'{args.output_prefix}_track_effect_summary.tsv'}")
    print("\nTop tracks:")
    view_cols = [
        "site_id",
        "rank_in_site",
        "global_track_index",
        "output_type",
        "target",
        "biosample_name",
        "delta_mean",
        "delta_abs_mean",
        "log1p_loss_mean",
        "log1p_gain_mean",
    ]
    print(top[[col for col in view_cols if col in top.columns]].head(args.top_n).to_string(index=False))
    print("\nTop output/target summary:")
    print(summary.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
