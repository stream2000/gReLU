#!/usr/bin/env python
"""Evaluate a spatially supported three-track-group AlphaGenome score.

Method 2 responds to two failures of the all-group median baseline:

* an edit can be important in a subset of biologically distinct track groups;
* overlapping 10-bp edits should support a region, not isolated 2-bp centers.

No motif or positive-control label is used to calculate the score.  Effects
are collapsed within track group, calibrated within gene and group, and the
third-highest group percentile is retained.  That statistic means at least
three track groups must support the center.  Its median within +/-4 bp (one
10-bp edit footprint on the 2-bp scan grid) is ranked within each gene.
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
THRESHOLD = scoring.THRESHOLD


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--threshold", type=float, default=THRESHOLD)
    return parser.parse_args()


def score_spatial_group_support(
    track_centers: pd.DataFrame,
    threshold: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    groups = (
        track_centers.groupby([*CENTER_KEYS, "track_group"], sort=False)
        .agg(
            tracks=("track_id", "nunique"),
            group_absolute_log2fc=("median_absolute_log2fc", "median"),
            group_signed_log2fc=("median_signed_log2fc", "median"),
            group_ref_sum=("median_ref_sum", "median"),
        )
        .reset_index()
    )
    groups["group_effect_percentile"] = groups.groupby(
        ["gene", "track_group"], sort=False
    ).group_absolute_log2fc.rank(method="average", pct=True)
    groups["group_loss_direction"] = groups.group_signed_log2fc < 0

    centers = (
        groups.groupby(CENTER_KEYS, sort=False)
        .agg(
            track_groups=("track_group", "nunique"),
            third_highest_group_percentile=(
                "group_effect_percentile",
                lambda values: float(values.nlargest(3).iloc[-1]),
            ),
            top_three_group_mean_percentile=(
                "group_effect_percentile",
                lambda values: float(values.nlargest(3).mean()),
            ),
            loss_direction_group_fraction=("group_loss_direction", "mean"),
            median_group_signed_log2fc=("group_signed_log2fc", "median"),
        )
        .reset_index()
    )
    centers = centers.merge(
        scoring.compute_spatial_median_support(
            centers, "third_highest_group_percentile"
        ),
        on=CENTER_KEYS,
        how="left",
        validate="one_to_one",
    )
    centers["importance_score"] = (
        100.0
        * centers.groupby("gene", sort=False).spatial_support.rank(
            method="average", pct=True
        )
    )
    centers["top_five_percent"] = centers.importance_score > threshold
    return groups, centers


def plot_spatial_support_comparison(
    method1_centers: pd.DataFrame,
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
        before = method1_centers.loc[method1_centers.gene.eq(gene)].sort_values(
            "variant_offset_from_tss_transcription_bp"
        )
        after = centers.loc[centers.gene.eq(gene)].sort_values(
            "variant_offset_from_tss_transcription_bp"
        )
        ax.plot(
            before.variant_offset_from_tss_transcription_bp,
            before.importance_score,
            color="#aaaaaa",
            linewidth=0.7,
            alpha=0.65,
            label="Method 1",
        )
        ax.plot(
            after.variant_offset_from_tss_transcription_bp,
            after.importance_score,
            color=colors[gene],
            linewidth=1.15,
            label="Method 2",
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
    axes[0].legend(loc="lower right", frameon=False, ncol=2)
    axes[-1].set_xlabel(
        "Offset from TSS in transcription direction (bp)"
    )
    fig.suptitle(
        "Method 2: three-group support with one-edit spatial aggregation",
        fontsize=13,
    )
    fig.savefig(out / "method2_vs_method1_score_profiles.png", dpi=180)
    fig.savefig(out / "method2_vs_method1_score_profiles.pdf")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    method1_dir = root / "analysis" / "method1_group_balanced_consensus"
    out = root / "analysis" / "method2_spatial_three_group_support"
    out.mkdir(parents=True, exist_ok=True)

    track_centers = pd.read_csv(
        method1_dir / "track_center_scores.tsv", sep="\t"
    )
    method1_centers = pd.read_csv(
        method1_dir / "center_scores.tsv", sep="\t"
    )
    groups, centers = score_spatial_group_support(
        track_centers, args.threshold
    )

    controls, holdout = config.load_benchmark_intervals(root)
    control_recall = scoring.evaluate_interval_recall(
        centers,
        controls,
        args.threshold,
        interval_kind="positive_control",
        window_statistic_column="spatial_support",
    )
    holdout_recall = scoring.evaluate_interval_recall(
        centers,
        holdout,
        args.threshold,
        interval_kind="mdk_holdout",
        window_statistic_column="spatial_support",
    )
    regions = scoring.call_threshold_regions(
        centers,
        args.threshold,
        score_statistic_column="spatial_support",
    )

    groups.to_csv(out / "group_center_scores.tsv", sep="\t", index=False)
    centers.to_csv(out / "center_scores.tsv", sep="\t", index=False)
    control_recall.to_csv(
        out / "positive_control_recall.tsv", sep="\t", index=False
    )
    holdout_recall.to_csv(
        out / "mdk_holdout_recall.tsv", sep="\t", index=False
    )
    regions.to_csv(out / "score95_regions.tsv", sep="\t", index=False)
    plot_spatial_support_comparison(
        method1_centers,
        centers,
        controls,
        holdout,
        args.threshold,
        out,
    )

    center_counts = (
        centers.groupby("gene")
        .size()
        .reindex(GENE_RUNS)
        .astype(int)
        .to_dict()
    )
    region_counts = (
        regions.groupby("gene")
        .size()
        .reindex(GENE_RUNS, fill_value=0)
        .astype(int)
        .to_dict()
    )
    singleton_regions = (
        regions.loc[regions.centers.eq(1)]
        .groupby("gene")
        .size()
        .reindex(GENE_RUNS, fill_value=0)
        .astype(int)
        .to_dict()
    )
    validation = {
        "status": "ok",
        "method": "method2_spatial_three_group_support",
        "score_definition": (
            "within-gene empirical rank of the +/-4-bp median of the "
            "third-highest gene-by-track-group effect percentile"
        ),
        "score_uses_labels": False,
        "support_groups_required": 3,
        "spatial_radius_bp": 4,
        "threshold": float(args.threshold),
        "centers_per_gene": center_counts,
        "tracks_per_gene": (
            track_centers.groupby("gene")
            .track_id.nunique()
            .astype(int)
            .to_dict()
        ),
        "track_groups_per_gene": (
            groups.groupby("gene")
            .track_group.nunique()
            .astype(int)
            .to_dict()
        ),
        "spatial_support_center_counts": sorted(
            centers.spatial_support_centers.unique().astype(int).tolist()
        ),
        "nonfinite_group_values": int(
            (~np.isfinite(
                groups[
                    [
                        "group_absolute_log2fc",
                        "group_signed_log2fc",
                        "group_ref_sum",
                        "group_effect_percentile",
                    ]
                ].to_numpy()
            )).sum()
        ),
        "nonfinite_center_values": int(
            (~np.isfinite(
                centers[
                    [
                        "third_highest_group_percentile",
                        "spatial_support",
                        "importance_score",
                    ]
                ].to_numpy()
            )).sum()
        ),
        "duplicate_group_center_keys": int(
            groups.duplicated([*CENTER_KEYS, "track_group"]).sum()
        ),
        "duplicate_center_keys": int(centers.duplicated(CENTER_KEYS).sum()),
        "positive_controls": int(len(control_recall)),
        "positive_controls_recalled_any_center": int(
            control_recall.recalled_at_threshold.sum()
        ),
        "positive_control_windows_recalled": int(
            control_recall.window_recalled_at_threshold.sum()
        ),
        "mdk_holdout_recalled_any_center": bool(
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
        "score95_regions_per_gene": region_counts,
        "score95_singleton_regions_per_gene": singleton_regions,
        "score95_regions": int(len(regions)),
    }
    if center_counts != {"Mdk": 1485, "Col1a1": 1494, "Acta2": 1495}:
        raise RuntimeError(validation)
    if set(validation["tracks_per_gene"].values()) != {28}:
        raise RuntimeError(validation)
    if set(validation["track_groups_per_gene"].values()) != {12}:
        raise RuntimeError(validation)
    if validation["nonfinite_group_values"]:
        raise RuntimeError(validation)
    if validation["nonfinite_center_values"]:
        raise RuntimeError(validation)
    if validation["duplicate_group_center_keys"]:
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
