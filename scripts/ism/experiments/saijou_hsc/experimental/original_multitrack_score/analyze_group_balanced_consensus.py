#!/usr/bin/env python
"""Evaluate an unweighted, group-balanced AlphaGenome multitrack baseline.

This first experimental score deliberately does not use motif annotations or
positive-control labels for fitting:

1. take the median absolute log2FC across the three strict shuffles for every
   gene, center and track;
2. convert the effect to an empirical percentile within gene and track;
3. take the median percentile within each track group;
4. take the median across track groups and rank centers within each gene.

The final 0--100 score is a within-gene empirical rank.  A score above 95
therefore means the top five percent of scanned centers for that gene; it is
not a confidence probability or false-discovery estimate.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import experiment_config as config
import score_calculations as scoring


DEFAULT_ROOT = config.DEFAULT_ROOT
GENE_RUNS = config.ALPHAGENOME_RUNS
CENTER_KEYS = scoring.CENTER_KEYS
TRACK_KEYS = [*CENTER_KEYS, "track_id", "track_group"]
THRESHOLD = scoring.THRESHOLD


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--threshold", type=float, default=THRESHOLD)
    return parser.parse_args()


def load_alphagenome_track_features(root: Path) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    required = {
        "gene",
        "mutation_id",
        "replacement_replicate",
        "variant_offset_from_tss_transcription_bp",
        "track_id",
        "track_group",
        "readout_role",
        "ref_sum",
        "log2fc_ratio_of_sums",
    }
    for gene, run_name in GENE_RUNS.items():
        path = (
            root
            / "runs"
            / run_name
            / "features"
            / "combined_mutation_features.parquet"
        )
        frame = pd.read_parquet(path)
        missing = sorted(required - set(frame.columns))
        if missing:
            raise ValueError(f"{path} is missing columns: {missing}")
        frame = frame.loc[
            frame.gene.eq(gene) & frame.readout_role.eq("tss_1024bp"),
            sorted(required),
        ].copy()
        frames.append(frame)
    features = pd.concat(frames, ignore_index=True)
    numeric = ["ref_sum", "log2fc_ratio_of_sums"]
    if not np.isfinite(features[numeric].to_numpy()).all():
        raise ValueError("Input features contain non-finite core values")
    return features


def score_group_balanced_consensus(
    features: pd.DataFrame,
    threshold: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    features = features.copy()
    features["absolute_log2fc"] = features.log2fc_ratio_of_sums.abs()

    track_centers = (
        features.groupby(TRACK_KEYS, sort=False)
        .agg(
            replacements=("replacement_replicate", "nunique"),
            median_signed_log2fc=("log2fc_ratio_of_sums", "median"),
            median_absolute_log2fc=("absolute_log2fc", "median"),
            median_ref_sum=("ref_sum", "median"),
        )
        .reset_index()
    )
    track_centers["track_effect_percentile"] = track_centers.groupby(
        ["gene", "track_id"], sort=False
    ).median_absolute_log2fc.rank(method="average", pct=True)

    group_centers = (
        track_centers.groupby([*CENTER_KEYS, "track_group"], sort=False)
        .agg(
            tracks=("track_id", "nunique"),
            group_effect_percentile=("track_effect_percentile", "median"),
            group_signed_log2fc=("median_signed_log2fc", "median"),
            group_absolute_log2fc=("median_absolute_log2fc", "median"),
            group_ref_sum=("median_ref_sum", "median"),
        )
        .reset_index()
    )
    group_centers["group_loss_direction"] = (
        group_centers.group_signed_log2fc < 0
    )

    centers = (
        group_centers.groupby(CENTER_KEYS, sort=False)
        .agg(
            track_groups=("track_group", "nunique"),
            consensus_percentile=("group_effect_percentile", "median"),
            upper_quartile_percentile=(
                "group_effect_percentile",
                lambda values: float(np.quantile(values, 0.75)),
            ),
            median_group_signed_log2fc=("group_signed_log2fc", "median"),
            median_group_absolute_log2fc=(
                "group_absolute_log2fc", "median"
            ),
            loss_direction_group_fraction=("group_loss_direction", "mean"),
        )
        .reset_index()
    )
    centers["importance_score"] = (
        100.0
        * centers.groupby("gene", sort=False).consensus_percentile.rank(
            method="average", pct=True
        )
    )
    centers["top_five_percent"] = centers.importance_score > threshold
    return track_centers, group_centers, centers


def plot_consensus_scores(
    centers: pd.DataFrame,
    controls: pd.DataFrame,
    holdout: pd.DataFrame,
    threshold: float,
    out: Path,
) -> None:
    genes = list(GENE_RUNS)
    fig, axes = plt.subplots(
        len(genes), 1, figsize=(12, 8.8), sharex=True, constrained_layout=True
    )
    colors = {"Mdk": "#326891", "Col1a1": "#d17a22", "Acta2": "#6a7d3b"}
    for ax, gene in zip(axes, genes, strict=True):
        subset = centers.loc[centers.gene.eq(gene)].sort_values(
            "variant_offset_from_tss_transcription_bp"
        )
        ax.plot(
            subset.variant_offset_from_tss_transcription_bp,
            subset.importance_score,
            color=colors[gene],
            linewidth=1.0,
        )
        ax.axhline(threshold, color="#333333", linestyle="--", linewidth=0.9)
        intervals = pd.concat(
            [
                controls.loc[controls.gene.eq(gene)],
                holdout.loc[holdout.gene.eq(gene)],
            ],
            ignore_index=True,
            sort=False,
        )
        for interval in intervals.itertuples(index=False):
            ax.axvspan(
                int(interval.start),
                int(interval.end),
                color="#d9a441",
                alpha=0.22,
                linewidth=0,
            )
        ax.set_ylabel(f"{gene}\nscore")
        ax.set_ylim(0, 101)
        ax.grid(axis="y", color="#dddddd", linewidth=0.5)
    axes[-1].set_xlabel(
        "Offset from TSS in transcription direction (bp)"
    )
    fig.suptitle(
        "Method 1: group-balanced multitrack consensus score",
        fontsize=13,
    )
    fig.savefig(out / "method1_score_profiles.png", dpi=180)
    fig.savefig(out / "method1_score_profiles.pdf")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    out = root / "analysis" / "method1_group_balanced_consensus"
    out.mkdir(parents=True, exist_ok=True)

    features = load_alphagenome_track_features(root)
    track_centers, group_centers, centers = score_group_balanced_consensus(
        features, args.threshold
    )
    controls, holdout = config.load_benchmark_intervals(root)
    control_recall = scoring.evaluate_interval_recall(
        centers, controls, args.threshold, interval_kind="positive_control"
    )
    holdout_recall = scoring.evaluate_interval_recall(
        centers, holdout, args.threshold, interval_kind="mdk_holdout"
    )
    regions = scoring.call_threshold_regions(centers, args.threshold)

    track_centers.to_csv(
        out / "track_center_scores.tsv", sep="\t", index=False
    )
    group_centers.to_csv(
        out / "group_center_scores.tsv", sep="\t", index=False
    )
    centers.to_csv(out / "center_scores.tsv", sep="\t", index=False)
    control_recall.to_csv(
        out / "positive_control_recall.tsv", sep="\t", index=False
    )
    holdout_recall.to_csv(
        out / "mdk_holdout_recall.tsv", sep="\t", index=False
    )
    regions.to_csv(out / "score95_regions.tsv", sep="\t", index=False)
    plot_consensus_scores(
        centers, controls, holdout, args.threshold, out
    )

    expected_centers = {"Mdk": 1485, "Col1a1": 1494, "Acta2": 1495}
    center_counts = (
        centers.groupby("gene")
        .size()
        .reindex(GENE_RUNS)
        .astype(int)
        .to_dict()
    )
    track_counts = (
        track_centers.groupby("gene").track_id.nunique().astype(int).to_dict()
    )
    group_counts = (
        group_centers.groupby("gene")
        .track_group.nunique()
        .astype(int)
        .to_dict()
    )
    validation = {
        "status": "ok",
        "method": "method1_group_balanced_consensus",
        "score_definition": (
            "within-gene empirical percentile rank of the median "
            "track-group percentile; strictly above 95 is the top five percent"
        ),
        "score_uses_labels": False,
        "threshold": float(args.threshold),
        "centers_per_gene": center_counts,
        "expected_centers_per_gene": expected_centers,
        "tracks_per_gene": track_counts,
        "track_groups_per_gene": group_counts,
        "replacement_counts": sorted(
            track_centers.replacements.unique().astype(int).tolist()
        ),
        "nonfinite_track_values": int(
            (~np.isfinite(
                track_centers[
                    [
                        "median_signed_log2fc",
                        "median_absolute_log2fc",
                        "median_ref_sum",
                        "track_effect_percentile",
                    ]
                ].to_numpy()
            )).sum()
        ),
        "nonfinite_center_values": int(
            (~np.isfinite(
                centers[
                    [
                        "consensus_percentile",
                        "importance_score",
                        "median_group_signed_log2fc",
                        "loss_direction_group_fraction",
                    ]
                ].to_numpy()
            )).sum()
        ),
        "duplicate_track_center_keys": int(
            track_centers.duplicated(TRACK_KEYS).sum()
        ),
        "duplicate_center_keys": int(centers.duplicated(CENTER_KEYS).sum()),
        "positive_controls": int(len(control_recall)),
        "positive_controls_recalled": int(
            control_recall.recalled_at_threshold.sum()
        ),
        "positive_control_windows_recalled": int(
            control_recall.window_recalled_at_threshold.sum()
        ),
        "mdk_holdout_recalled": bool(
            holdout_recall.recalled_at_threshold.iloc[0]
        ),
        "mdk_holdout_window_recalled": bool(
            holdout_recall.window_recalled_at_threshold.iloc[0]
        ),
        "score95_centers_per_gene": (
            centers.loc[centers.importance_score > args.threshold]
            .groupby("gene")
            .size()
            .reindex(GENE_RUNS, fill_value=0)
            .astype(int)
            .to_dict()
        ),
        "score95_regions": int(len(regions)),
    }
    if center_counts != expected_centers:
        raise RuntimeError(validation)
    if set(track_counts.values()) != {28}:
        raise RuntimeError(validation)
    if set(group_counts.values()) != {12}:
        raise RuntimeError(validation)
    if validation["replacement_counts"] != [3]:
        raise RuntimeError(validation)
    if validation["nonfinite_track_values"]:
        raise RuntimeError(validation)
    if validation["nonfinite_center_values"]:
        raise RuntimeError(validation)
    if validation["duplicate_track_center_keys"]:
        raise RuntimeError(validation)
    if validation["duplicate_center_keys"]:
        raise RuntimeError(validation)
    (out / "validation_summary.json").write_text(
        json.dumps(validation, indent=2) + "\n"
    )
    print(json.dumps(validation, indent=2))
    print(control_recall.to_string(index=False))
    print(holdout_recall.to_string(index=False))


if __name__ == "__main__":
    main()
