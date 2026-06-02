#!/usr/bin/env python
"""Rank AlphaGenome tracks by sensitivity to random CTCF motif replacement.

The input is a ref/alt track-window run with one perturbation per CTCF site.
This script streams over sites, computes raw and log1p deltas, and writes
track-level sensitivity summaries for heuristic track selection.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_WINDOWS = {
    "full_100kb": None,
    "local_10kb": 10_000,
    "local_5kb": 5_000,
}


def _track_label(row: pd.Series) -> str:
    target = row.get("transcription_factor")
    if pd.isna(target) or target == "":
        target = row.get("histone_mark")
    if pd.isna(target) or target == "":
        target = row.get("track_name")
    return "" if pd.isna(target) else str(target)


def _trimmed_mean(values: np.ndarray, trim: float = 0.1) -> np.ndarray:
    if not 0 <= trim < 0.5:
        raise ValueError(f"trim must be in [0, 0.5), got {trim}")
    n = values.shape[0]
    lo = int(np.floor(n * trim))
    hi = n - lo
    if hi <= lo:
        return values.mean(axis=0)
    sorted_values = np.sort(values, axis=0)
    return sorted_values[lo:hi].mean(axis=0)


def _window_masks(bins: pd.DataFrame) -> dict[str, np.ndarray]:
    offsets = bins["offset_bp"].to_numpy()
    masks: dict[str, np.ndarray] = {}
    for name, radius_bp in DEFAULT_WINDOWS.items():
        if radius_bp is None:
            masks[name] = np.ones(len(offsets), dtype=bool)
        else:
            masks[name] = np.abs(offsets) <= radius_bp
    return masks


def _metric_summary(values: np.ndarray) -> dict[str, np.ndarray]:
    q75, q90, q95, q99 = np.quantile(values, [0.75, 0.90, 0.95, 0.99], axis=0)
    mean = values.mean(axis=0)
    median = np.median(values, axis=0)
    return {
        "mean": mean,
        "median": median,
        "trimmed_mean_10pct": _trimmed_mean(values, trim=0.1),
        "q75": q75,
        "q90": q90,
        "q95": q95,
        "q99": q99,
        "frac_gt_0": (values > 0).mean(axis=0),
        "frac_gt_median_track_mean": (values > np.median(mean)).mean(axis=0),
    }


def _write_table(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, sep="\t", index=False)


def _markdown_table(df: pd.DataFrame) -> list[str]:
    view = df.copy()
    for col in view.columns:
        if pd.api.types.is_float_dtype(view[col]):
            view[col] = view[col].map(lambda x: "" if pd.isna(x) else f"{x:.6g}")
    headers = [str(col) for col in view.columns]
    rows = view.astype(str).to_numpy().tolist()
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return lines


def _add_recommended_full_window_outputs(all_rankings: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    full = all_rankings[all_rankings["ranking_window"] == "full_100kb"].copy()
    score = "full_100kb__primary_sensitivity_score"
    is_padding = (
        full.get("target", pd.Series("", index=full.index))
        .fillna("")
        .astype(str)
        .str.lower()
        .eq("padding")
        | full.get("track_name", pd.Series("", index=full.index))
        .fillna("")
        .astype(str)
        .str.lower()
        .str.contains("padding")
    )
    full["is_padding_track"] = is_padding

    no_padding = full.loc[~full["is_padding_track"]].copy()
    no_padding["full_100kb__recommended_no_padding_rank"] = (
        no_padding[score].rank(ascending=False, method="min").astype(int)
    )
    no_padding["full_100kb__rank_within_output_type"] = (
        no_padding.groupby("output_type", dropna=False)[score]
        .rank(ascending=False, method="min")
        .astype(int)
    )
    no_padding = no_padding.sort_values("full_100kb__recommended_no_padding_rank")
    _write_table(no_padding, output_dir / "track_sensitivity_rank_full_100kb_no_padding.tsv")
    _write_table(
        no_padding.sort_values(["output_type", "full_100kb__rank_within_output_type"]),
        output_dir / "track_sensitivity_rank_full_100kb_by_output_type.tsv",
    )

    practical = no_padding[no_padding["full_100kb__rank_within_output_type"] <= 100].copy()
    tf_targets = {"CTCF", "RAD21", "SMC3"}
    target_upper = no_padding["target"].fillna("").astype(str).str.upper()
    cohesion_tf = no_padding[
        (no_padding["output_type"] == "chip_tf") & target_upper.isin(tf_targets)
    ].copy()
    practical = (
        pd.concat([practical, cohesion_tf], ignore_index=True)
        .drop_duplicates(subset=["global_track_index"])
        .sort_values(["output_type", "full_100kb__rank_within_output_type"])
    )
    _write_table(practical, output_dir / "heuristic_candidate_tracks_full_100kb.tsv")

    no_padding.head(200).groupby(["output_type", "target"], dropna=False).size().rename(
        "n_tracks"
    ).reset_index().sort_values("n_tracks", ascending=False).to_csv(
        output_dir / "top200_full_100kb_target_counts_no_padding.tsv",
        sep="\t",
        index=False,
    )
    return no_padding


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--site_chunk_size", type=int, default=1)
    parser.add_argument("--write_site_track_parquet", action="store_true")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ref = np.load(run_dir / "all_track_ref_window.npy", mmap_mode="r")
    alt = np.load(run_dir / "all_track_alt_window.npy", mmap_mode="r")
    if ref.shape != alt.shape:
        raise ValueError(f"Shape mismatch: ref={ref.shape}, alt={alt.shape}")
    if ref.ndim != 3:
        raise ValueError(f"Expected sites x tracks x bins arrays, got {ref.shape}")

    n_sites, n_tracks, n_bins = ref.shape
    metadata = pd.read_csv(run_dir / "all_track_metadata.tsv", sep="\t", low_memory=False)
    variants = pd.read_csv(run_dir / "variants.tsv", sep="\t")
    bins = pd.read_csv(run_dir / "all_track_window_bins.tsv", sep="\t")
    if len(metadata) != n_tracks:
        raise ValueError(f"Track metadata rows {len(metadata)} != n_tracks {n_tracks}")
    if len(variants) != n_sites:
        raise ValueError(f"Variant rows {len(variants)} != n_sites {n_sites}")
    if len(bins) != n_bins:
        raise ValueError(f"Bin rows {len(bins)} != n_bins {n_bins}")

    windows = _window_masks(bins)
    raw_abs_by_window = {name: np.empty((n_sites, n_tracks), dtype=np.float32) for name in windows}
    raw_signed_by_window = {name: np.empty((n_sites, n_tracks), dtype=np.float32) for name in windows}
    log_abs_by_window = {name: np.empty((n_sites, n_tracks), dtype=np.float32) for name in windows}
    log_signed_by_window = {name: np.empty((n_sites, n_tracks), dtype=np.float32) for name in windows}
    log_loss_by_window = {name: np.empty((n_sites, n_tracks), dtype=np.float32) for name in windows}
    log_gain_by_window = {name: np.empty((n_sites, n_tracks), dtype=np.float32) for name in windows}

    chunk_size = max(1, int(args.site_chunk_size))
    for start in range(0, n_sites, chunk_size):
        stop = min(start + chunk_size, n_sites)
        ref_chunk = np.asarray(ref[start:stop], dtype=np.float32)
        alt_chunk = np.asarray(alt[start:stop], dtype=np.float32)
        delta = alt_chunk - ref_chunk
        log_delta = np.log2(1.0 + np.maximum(alt_chunk, 0.0)) - np.log2(
            1.0 + np.maximum(ref_chunk, 0.0)
        )
        for name, mask in windows.items():
            d = delta[:, :, mask]
            ld = log_delta[:, :, mask]
            raw_abs_by_window[name][start:stop] = np.mean(np.abs(d), axis=2)
            raw_signed_by_window[name][start:stop] = np.mean(d, axis=2)
            log_abs_by_window[name][start:stop] = np.mean(np.abs(ld), axis=2)
            log_signed_by_window[name][start:stop] = np.mean(ld, axis=2)
            log_loss_by_window[name][start:stop] = np.mean(np.maximum(-ld, 0.0), axis=2)
            log_gain_by_window[name][start:stop] = np.mean(np.maximum(ld, 0.0), axis=2)

    track_table = metadata.copy().reset_index(drop=True)
    if "global_track_index" not in track_table.columns:
        track_table["global_track_index"] = np.arange(n_tracks)
    track_table["target"] = track_table.apply(_track_label, axis=1)

    ranking_frames: list[pd.DataFrame] = []
    metric_arrays = {
        "raw_abs_mean": raw_abs_by_window,
        "raw_signed_mean": raw_signed_by_window,
        "log1p_abs_mean": log_abs_by_window,
        "log1p_signed_mean": log_signed_by_window,
        "log1p_loss_mean": log_loss_by_window,
        "log1p_gain_mean": log_gain_by_window,
    }
    for window_name in windows:
        rec = track_table.copy()
        for metric_name, per_window in metric_arrays.items():
            stats = _metric_summary(per_window[window_name])
            for stat_name, values in stats.items():
                rec[f"{window_name}__{metric_name}__{stat_name}"] = values
        primary = rec[f"{window_name}__log1p_abs_mean__median"]
        consistency = rec[f"{window_name}__log1p_abs_mean__frac_gt_median_track_mean"]
        tail = rec[f"{window_name}__log1p_abs_mean__q90"]
        rec[f"{window_name}__primary_sensitivity_score"] = (
            primary * (0.5 + consistency) + 0.25 * tail
        )
        rec[f"{window_name}__primary_rank"] = (
            rec[f"{window_name}__primary_sensitivity_score"]
            .rank(ascending=False, method="min")
            .astype(int)
        )
        rec["ranking_window"] = window_name
        ranking_frames.append(
            rec.sort_values(f"{window_name}__primary_rank").reset_index(drop=True)
        )

    all_rankings = pd.concat(ranking_frames, ignore_index=True)
    _write_table(all_rankings, output_dir / "track_sensitivity_rank_all_windows.tsv")

    for window_name in windows:
        sub = all_rankings[all_rankings["ranking_window"] == window_name].copy()
        _write_table(sub, output_dir / f"track_sensitivity_rank_{window_name}.tsv")

    site_summary_frames: list[pd.DataFrame] = []
    for window_name in windows:
        for metric_name, per_window in metric_arrays.items():
            values = per_window[window_name]
            site_mean = values.mean(axis=1)
            site_q95 = np.quantile(values, 0.95, axis=1)
            site_summary_frames.append(
                pd.DataFrame(
                    {
                        "site_id": variants["site_id"],
                        "window": window_name,
                        "metric": metric_name,
                        "track_mean": site_mean,
                        "track_q95": site_q95,
                    }
                )
            )
    pd.concat(site_summary_frames, ignore_index=True).to_parquet(
        output_dir / "site_level_track_response_summary.parquet", index=False
    )

    if args.write_site_track_parquet:
        site_track_dir = output_dir / "site_track_metric_parquet"
        site_track_dir.mkdir(parents=True, exist_ok=True)
        base = pd.DataFrame(
            {
                "site_id": np.repeat(variants["site_id"].to_numpy(), n_tracks),
                "global_track_index": np.tile(np.arange(n_tracks, dtype=int), n_sites),
            }
        )
        for window_name in windows:
            frame = base.copy()
            for metric_name, per_window in metric_arrays.items():
                frame[metric_name] = per_window[window_name].reshape(-1)
            frame.to_parquet(site_track_dir / f"site_track_metrics_{window_name}.parquet", index=False)

    top200 = (
        all_rankings[all_rankings["ranking_window"] == "full_100kb"]
        .sort_values("full_100kb__primary_rank")
        .head(200)
    )
    target_counts = top200.groupby(["output_type", "target"], dropna=False).size().rename("n_tracks")
    target_counts.reset_index().sort_values("n_tracks", ascending=False).to_csv(
        output_dir / "top200_full_100kb_target_counts.tsv", sep="\t", index=False
    )
    no_padding_full = _add_recommended_full_window_outputs(all_rankings, output_dir)

    provenance = {
        "input_run_dir": str(run_dir),
        "raw_ref_window": str(run_dir / "all_track_ref_window.npy"),
        "raw_alt_window": str(run_dir / "all_track_alt_window.npy"),
        "variants": str(run_dir / "variants.tsv"),
        "track_metadata": str(run_dir / "all_track_metadata.tsv"),
        "window_bins": str(run_dir / "all_track_window_bins.tsv"),
        "input_shape": list(ref.shape),
        "dtype": str(ref.dtype),
        "delta_storage_policy": "full delta matrix was computed streamingly and not stored",
        "ranking_algorithm": (
            "primary_sensitivity_score = median(log1p_abs_mean) * "
            "(0.5 + fraction of sites above median track mean) + 0.25 * q90(log1p_abs_mean)"
        ),
        "recommended_filtering": (
            "Use track_sensitivity_rank_full_100kb_no_padding.tsv or "
            "track_sensitivity_rank_full_100kb_by_output_type.tsv for heuristic selection; "
            "the unfiltered full-window rank can be dominated by RNA-seq and Padding tracks."
        ),
        "windows": {
            name: {
                "radius_bp": DEFAULT_WINDOWS[name],
                "n_bins": int(mask.sum()),
            }
            for name, mask in windows.items()
        },
    }
    (output_dir / "raw_data_provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")

    summary_lines = [
        "# Random CTCF whole-motif replacement track sensitivity ranking",
        "",
        "## Raw Data",
        "",
        f"- ref window: `{run_dir / 'all_track_ref_window.npy'}`",
        f"- alt window: `{run_dir / 'all_track_alt_window.npy'}`",
        f"- variants: `{run_dir / 'variants.tsv'}`",
        f"- shape: `{tuple(ref.shape)}`",
        "",
        "The full delta matrix was computed streamingly from ref/alt and was not stored as a 30 GB `.npy` file.",
        "",
        "## Ranking Algorithm",
        "",
        "`log1p_delta = log2(1 + alt) - log2(1 + ref)` was computed per site, track, and bin.",
        "",
        "Primary score per track:",
        "",
        "```text",
        "median(log1p_abs_mean across sites) * (0.5 + fraction_above_global_track_median)",
        "+ 0.25 * q90(log1p_abs_mean across sites)",
        "```",
        "",
        "This favors tracks that respond consistently across many random CTCF motif replacements while still allowing strong upper-tail response to contribute.",
        "",
        "## Main Outputs",
        "",
        "- `track_sensitivity_rank_full_100kb.tsv`",
        "- `track_sensitivity_rank_local_10kb.tsv`",
        "- `track_sensitivity_rank_local_5kb.tsv`",
        "- `track_sensitivity_rank_all_windows.tsv`",
        "- `track_sensitivity_rank_full_100kb_no_padding.tsv`",
        "- `track_sensitivity_rank_full_100kb_by_output_type.tsv`",
        "- `heuristic_candidate_tracks_full_100kb.tsv`",
        "- `site_level_track_response_summary.parquet`",
        "- `top200_full_100kb_target_counts.tsv`",
        "- `top200_full_100kb_target_counts_no_padding.tsv`",
        "- `raw_data_provenance.json`",
        "",
        "The unfiltered full-window rank is useful as a raw record, but the top rows can be dominated by RNA-seq and Padding tracks. For heuristic track selection, prefer the no-padding table and the output-type-stratified table.",
        "",
        "## Top 20 Full-Window Tracks",
        "",
    ]
    top_cols = [
        "full_100kb__primary_rank",
        "global_track_index",
        "output_type",
        "target",
        "biosample_name",
        "full_100kb__primary_sensitivity_score",
        "full_100kb__log1p_abs_mean__median",
        "full_100kb__log1p_abs_mean__q90",
    ]
    top20 = all_rankings[all_rankings["ranking_window"] == "full_100kb"].head(20)[top_cols].copy()
    for col in top20.columns:
        if pd.api.types.is_float_dtype(top20[col]):
            top20[col] = top20[col].map(lambda x: f"{x:.6g}")
    summary_lines.extend(_markdown_table(top20))
    summary_lines.append("")
    summary_lines.append("## Top 20 Full-Window Tracks, No Padding")
    summary_lines.append("")
    recommended_cols = [
        "full_100kb__recommended_no_padding_rank",
        "full_100kb__primary_rank",
        "global_track_index",
        "output_type",
        "target",
        "biosample_name",
        "full_100kb__primary_sensitivity_score",
        "full_100kb__log1p_abs_mean__median",
        "full_100kb__log1p_abs_mean__q90",
    ]
    recommended_top20 = no_padding_full.head(20)[recommended_cols].copy()
    for col in recommended_top20.columns:
        if pd.api.types.is_float_dtype(recommended_top20[col]):
            recommended_top20[col] = recommended_top20[col].map(lambda x: f"{x:.6g}")
    summary_lines.extend(_markdown_table(recommended_top20))
    summary_lines.append("")
    chip_tf = no_padding_full[no_padding_full["output_type"] == "chip_tf"].head(20).copy()
    if len(chip_tf) > 0:
        summary_lines.append("## Top 20 Full-Window chip_tf Tracks, No Padding")
        summary_lines.append("")
        chip_cols = recommended_cols + ["transcription_factor"]
        for col in chip_tf.columns:
            if pd.api.types.is_float_dtype(chip_tf[col]):
                chip_tf[col] = chip_tf[col].map(lambda x: f"{x:.6g}")
        summary_lines.extend(_markdown_table(chip_tf[chip_cols]))
        summary_lines.append("")
    (output_dir / "track_sensitivity_summary.md").write_text(
        "\n".join(summary_lines), encoding="utf-8"
    )

    print(f"shape: {ref.shape}")
    print(f"wrote: {output_dir / 'track_sensitivity_rank_full_100kb.tsv'}")
    print(f"wrote: {output_dir / 'track_sensitivity_rank_all_windows.tsv'}")
    print(f"wrote: {output_dir / 'track_sensitivity_rank_full_100kb_no_padding.tsv'}")
    print(f"wrote: {output_dir / 'track_sensitivity_rank_full_100kb_by_output_type.tsv'}")
    print(f"wrote: {output_dir / 'heuristic_candidate_tracks_full_100kb.tsv'}")
    print(f"wrote: {output_dir / 'track_sensitivity_summary.md'}")
    print(f"wrote: {output_dir / 'raw_data_provenance.json'}")


if __name__ == "__main__":
    main()
