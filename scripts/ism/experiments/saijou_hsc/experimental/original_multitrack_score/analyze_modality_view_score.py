#!/usr/bin/env python
"""Evaluate a modality-aware two-view AlphaGenome sensitivity score.

RNA and CAGE tracks use the fixed TSS 1024-bp output statistic from the
validated feature tables.  Accessibility, histone and TF tracks instead use
the strongest signed log2FC in the mutation bin and its two neighboring
128-bp bins.  Each view requires support from at least three track groups,
then receives the same one-edit spatial aggregation used by Method 2.

The two view-specific empirical scores are retained for interpretation.  Their
maximum is ranked once more within gene to produce the final 0--100 score, so
the final score above 95 still denotes the overall top five percent rather
than the union of two separate top-five-percent lists.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import experiment_config as config
import profile_features
import score_calculations as scoring


DEFAULT_ROOT = config.DEFAULT_ROOT
GENE_RUNS = config.ALPHAGENOME_RUNS
CENTER_KEYS = scoring.CENTER_KEYS
OUTPUT_MODALITIES = {"rna_seq", "cage"}
THRESHOLD = scoring.THRESHOLD


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--threshold", type=float, default=THRESHOLD)
    return parser.parse_args()


def build_alphagenome_view_features(root: Path) -> pd.DataFrame:
    method1_tracks = pd.read_csv(
        root
        / "analysis"
        / "method1_group_balanced_consensus"
        / "track_center_scores.tsv",
        sep="\t",
    )
    frames: list[pd.DataFrame] = []
    for gene, run_name in GENE_RUNS.items():
        manifest = profile_features.load_validated_track_manifest(
            root, run_name
        )
        metadata = manifest[
            ["track_id", "track_group", "modality"]
        ].drop_duplicates("track_id")
        output = method1_tracks.loc[
            method1_tracks.gene.eq(gene)
        ].merge(
            metadata,
            on=["track_id", "track_group"],
            how="left",
            validate="many_to_one",
        )
        output = output.loc[
            output.modality.isin(OUTPUT_MODALITIES),
            [
                *CENTER_KEYS,
                "track_id",
                "track_group",
                "modality",
                "replacements",
                "median_absolute_log2fc",
                "median_signed_log2fc",
            ],
        ].copy()
        output["score_view"] = "output"
        output["readout_definition"] = "tss_1024bp_ratio_of_sums"
        frames.extend(
            [
                output,
                profile_features.summarize_local_profile_effects(
                    root,
                    gene=gene,
                    run_name=run_name,
                    output_modalities=OUTPUT_MODALITIES,
                ),
            ]
        )
    result = pd.concat(frames, ignore_index=True)
    return result.sort_values(
        [*CENTER_KEYS, "score_view", "track_group", "track_id"]
    ).reset_index(drop=True)


def plot_modality_view_scores(
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
    for ax, gene in zip(axes, genes, strict=True):
        subset = centers.loc[centers.gene.eq(gene)].sort_values(
            "variant_offset_from_tss_transcription_bp"
        )
        x = subset.variant_offset_from_tss_transcription_bp
        ax.plot(
            x,
            subset.importance_score,
            color="#222222",
            linewidth=1.3,
            label="overall calibrated",
        )
        ax.plot(
            x,
            subset.view_score_output,
            color="#326891",
            linewidth=0.9,
            alpha=0.8,
            label="output view",
        )
        ax.plot(
            x,
            subset.view_score_local_regulatory,
            color="#d17a22",
            linewidth=0.9,
            alpha=0.8,
            label="local regulatory view",
        )
        ax.axhline(threshold, color="#555555", linestyle="--", linewidth=0.9)
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
                alpha=0.20,
                linewidth=0,
            )
        ax.set_ylabel(f"{gene}\nscore")
        ax.set_ylim(0, 101)
        ax.grid(axis="y", color="#dddddd", linewidth=0.5)
    axes[0].legend(loc="lower right", frameon=False, ncol=3)
    axes[-1].set_xlabel(
        "Offset from TSS in transcription direction (bp)"
    )
    fig.suptitle(
        "Method 3: modality-aware output and local regulatory views",
        fontsize=13,
    )
    fig.savefig(out / "method3_modality_view_profiles.png", dpi=180)
    fig.savefig(out / "method3_modality_view_profiles.pdf")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    out = root / "analysis" / "method3_modality_views"
    out.mkdir(parents=True, exist_ok=True)

    tracks = build_alphagenome_view_features(root)
    groups, view_centers, centers = scoring.score_modality_views(
        tracks, args.threshold
    )
    controls, holdout = config.load_benchmark_intervals(root)
    control_recall = scoring.evaluate_interval_recall(
        centers,
        controls,
        args.threshold,
        interval_kind="positive_control",
        window_statistic_column="max_view_score_raw",
    )
    holdout_recall = scoring.evaluate_interval_recall(
        centers,
        holdout,
        args.threshold,
        interval_kind="mdk_holdout",
        window_statistic_column="max_view_score_raw",
    )
    view_control_recall = scoring.evaluate_modality_view_recall(
        centers,
        controls,
        args.threshold,
        interval_kind="positive_control",
    )
    view_holdout_recall = scoring.evaluate_modality_view_recall(
        centers,
        holdout,
        args.threshold,
        interval_kind="mdk_holdout",
    )
    regions = scoring.call_threshold_regions(
        centers,
        args.threshold,
        score_statistic_column="max_view_score_raw",
    )

    tracks.to_csv(out / "track_view_scores.tsv", sep="\t", index=False)
    groups.to_csv(out / "group_view_scores.tsv", sep="\t", index=False)
    view_centers.to_csv(
        out / "view_center_scores.tsv", sep="\t", index=False
    )
    centers.to_csv(out / "center_scores.tsv", sep="\t", index=False)
    control_recall.to_csv(
        out / "positive_control_recall.tsv", sep="\t", index=False
    )
    view_control_recall.to_csv(
        out / "positive_control_view_recall.tsv", sep="\t", index=False
    )
    holdout_recall.to_csv(
        out / "mdk_holdout_recall.tsv", sep="\t", index=False
    )
    view_holdout_recall.to_csv(
        out / "mdk_holdout_view_recall.tsv", sep="\t", index=False
    )
    regions.to_csv(out / "score95_regions.tsv", sep="\t", index=False)
    plot_modality_view_scores(
        centers, controls, holdout, args.threshold, out
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
        "method": "method3_modality_views",
        "score_definition": (
            "within-gene empirical rank of the maximum of independently "
            "ranked output-view and local-regulatory-view spatial support"
        ),
        "score_uses_labels": False,
        "views": {
            "output": {
                "modalities": sorted(OUTPUT_MODALITIES),
                "readout": "tss_1024bp_ratio_of_sums",
            },
            "local_regulatory": {
                "modalities": sorted(
                    set(tracks.modality) - OUTPUT_MODALITIES
                ),
                "readout": (
                    "strongest signed log2FC across edit bin and one "
                    "neighboring 128-bp bin on each side"
                ),
            },
        },
        "support_groups_required_per_view": 3,
        "spatial_radius_bp": 4,
        "threshold": float(args.threshold),
        "centers_per_gene": (
            centers.groupby("gene")
            .size()
            .reindex(GENE_RUNS)
            .astype(int)
            .to_dict()
        ),
        "tracks_per_gene": (
            tracks.groupby("gene")
            .track_id.nunique()
            .astype(int)
            .to_dict()
        ),
        "track_groups_per_gene_and_view": {
            gene: {
                view: int(count)
                for view, count in view_counts.items()
            }
            for gene, view_counts in (
                groups.groupby(["gene", "score_view"])
                .track_group.nunique()
                .unstack()
                .astype(int)
                .to_dict(orient="index")
                .items()
            )
        },
        "replacement_counts": sorted(
            tracks.replacements.unique().astype(int).tolist()
        ),
        "nonfinite_track_values": int(
            (~np.isfinite(
                tracks[
                    [
                        "median_absolute_log2fc",
                        "median_signed_log2fc",
                    ]
                ].to_numpy()
            )).sum()
        ),
        "nonfinite_center_values": int(
            (~np.isfinite(
                centers[
                    [
                        "view_score_local_regulatory",
                        "view_score_output",
                        "max_view_score_raw",
                        "importance_score",
                    ]
                ].to_numpy()
            )).sum()
        ),
        "duplicate_track_view_keys": int(
            tracks.duplicated(
                [*CENTER_KEYS, "score_view", "track_id"]
            ).sum()
        ),
        "duplicate_center_keys": int(centers.duplicated(CENTER_KEYS).sum()),
        "positive_controls": int(len(control_recall)),
        "positive_controls_recalled_overall_any_center": int(
            control_recall.recalled_at_threshold.sum()
        ),
        "positive_control_windows_recalled_overall": int(
            control_recall.window_recalled_at_threshold.sum()
        ),
        "positive_controls_recalled_in_either_view_any_center": int(
            view_control_recall.either_view_recalled_any_center.sum()
        ),
        "mdk_holdout_recalled_overall_any_center": bool(
            holdout_recall.recalled_at_threshold.iloc[0]
        ),
        "mdk_holdout_window_recalled_overall": bool(
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
    if validation["centers_per_gene"] != {
        "Mdk": 1485,
        "Col1a1": 1494,
        "Acta2": 1495,
    }:
        raise RuntimeError(validation)
    if set(validation["tracks_per_gene"].values()) != {28}:
        raise RuntimeError(validation)
    if {
        count
        for view_counts in validation[
            "track_groups_per_gene_and_view"
        ].values()
        for count in view_counts.values()
    } != {6}:
        raise RuntimeError(validation)
    if validation["replacement_counts"] != [3]:
        raise RuntimeError(validation)
    if validation["nonfinite_track_values"]:
        raise RuntimeError(validation)
    if validation["nonfinite_center_values"]:
        raise RuntimeError(validation)
    if validation["duplicate_track_view_keys"]:
        raise RuntimeError(validation)
    if validation["duplicate_center_keys"]:
        raise RuntimeError(validation)
    (out / "validation_summary.json").write_text(
        json.dumps(validation, indent=2) + "\n"
    )
    print(json.dumps(validation, indent=2, default=str))
    print(control_recall.to_string(index=False))
    print(view_control_recall.to_string(index=False))
    print(holdout_recall.to_string(index=False))
    print(view_holdout_recall.to_string(index=False))


if __name__ == "__main__":
    main()
