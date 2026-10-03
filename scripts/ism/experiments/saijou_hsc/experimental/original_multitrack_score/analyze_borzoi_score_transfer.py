#!/usr/bin/env python
"""Apply the frozen modality-view score to original Borzoi without refitting."""

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
BORZOI_RUNS = {
    "Mdk": "borzoi_original_mdk",
    "Col1a1": "borzoi_original_col1a1",
    "Acta2": "borzoi_original_acta2",
}
BORZOI_OUTPUT_MODALITIES = {"rna", "cage"}
CENTER_KEYS = scoring.CENTER_KEYS
THRESHOLD = scoring.THRESHOLD


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--threshold", type=float, default=THRESHOLD)
    return parser.parse_args()


def build_borzoi_view_features(root: Path) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    required = {
        "gene",
        "variant_offset_from_tss_transcription_bp",
        "replacement_replicate",
        "track_id",
        "track_group",
        "readout_role",
        "log2fc_ratio_of_sums",
    }
    for gene, run_name in BORZOI_RUNS.items():
        run = root / "runs" / run_name
        features = pd.read_parquet(
            run / "features" / "combined_mutation_features.parquet"
        )
        missing = sorted(required - set(features.columns))
        if missing:
            raise ValueError(f"{run}: missing feature columns {missing}")
        tracks = profile_features.load_validated_track_manifest(
            root, run_name
        )
        output_tracks = tracks.loc[
            tracks.modality.isin(BORZOI_OUTPUT_MODALITIES),
            ["track_id", "track_group", "modality"],
        ]
        output = features.loc[
            features.gene.eq(gene)
            & features.readout_role.eq("tss_1024bp")
            & features.track_id.isin(output_tracks.track_id)
        ].copy()
        output["absolute_log2fc"] = (
            output.log2fc_ratio_of_sums.abs()
        )
        output = (
            output.groupby(
                [*CENTER_KEYS, "track_id", "track_group"], sort=False
            )
            .agg(
                replacements=("replacement_replicate", "nunique"),
                median_absolute_log2fc=("absolute_log2fc", "median"),
                median_signed_log2fc=(
                    "log2fc_ratio_of_sums", "median"
                ),
            )
            .reset_index()
            .merge(
                output_tracks,
                on=["track_id", "track_group"],
                how="left",
                validate="many_to_one",
            )
        )
        output["score_view"] = "output"
        output["readout_definition"] = "tss_1024bp_ratio_of_sums"
        local = profile_features.summarize_local_profile_effects(
            root,
            gene=gene,
            run_name=run_name,
            output_modalities=BORZOI_OUTPUT_MODALITIES,
        )
        frames.extend([output, local])
    result = pd.concat(frames, ignore_index=True)
    return result.sort_values(
        [*CENTER_KEYS, "score_view", "track_group", "track_id"]
    ).reset_index(drop=True)


def compare_alphagenome_borzoi_scores(
    ag: pd.DataFrame,
    bz: pd.DataFrame,
    threshold: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    columns = [
        *CENTER_KEYS,
        "importance_score",
        "view_score_local_regulatory",
        "view_score_output",
        "driving_view",
        "median_group_signed_log2fc",
        "loss_direction_group_fraction",
    ]
    merged = ag[columns].merge(
        bz[columns],
        on=CENTER_KEYS,
        how="inner",
        validate="one_to_one",
        suffixes=("_alphagenome", "_borzoi"),
    )
    rows: list[dict[str, object]] = []
    for gene, group in merged.groupby("gene", sort=False):
        ag_selected = set(
            group.loc[
                group.importance_score_alphagenome > threshold,
                "variant_offset_from_tss_transcription_bp",
            ]
        )
        bz_selected = set(
            group.loc[
                group.importance_score_borzoi > threshold,
                "variant_offset_from_tss_transcription_bp",
            ]
        )
        union = ag_selected | bz_selected
        rows.append(
            {
                "gene": gene,
                "aligned_centers": int(len(group)),
                "overall_score_spearman": float(
                    group.importance_score_alphagenome.corr(
                        group.importance_score_borzoi, method="spearman"
                    )
                ),
                "output_view_score_spearman": float(
                    group.view_score_output_alphagenome.corr(
                        group.view_score_output_borzoi, method="spearman"
                    )
                ),
                "local_view_score_spearman": float(
                    group.view_score_local_regulatory_alphagenome.corr(
                        group.view_score_local_regulatory_borzoi,
                        method="spearman",
                    )
                ),
                "alphagenome_score95_centers": int(len(ag_selected)),
                "borzoi_score95_centers": int(len(bz_selected)),
                "score95_center_intersection": int(
                    len(ag_selected & bz_selected)
                ),
                "score95_center_union": int(len(union)),
                "score95_center_jaccard": (
                    float(len(ag_selected & bz_selected) / len(union))
                    if union
                    else np.nan
                ),
                "driving_view_agreement_fraction": float(
                    (
                        group.driving_view_alphagenome
                        == group.driving_view_borzoi
                    ).mean()
                ),
            }
        )
    return merged, pd.DataFrame.from_records(rows)


def plot_cross_model_scores(
    merged: pd.DataFrame,
    controls: pd.DataFrame,
    holdout: pd.DataFrame,
    threshold: float,
    out: Path,
) -> None:
    fig, axes = plt.subplots(
        3, 1, figsize=(12, 8.8), sharex=True, constrained_layout=True
    )
    for ax, gene in zip(axes, BORZOI_RUNS, strict=True):
        group = merged.loc[merged.gene.eq(gene)].sort_values(
            "variant_offset_from_tss_transcription_bp"
        )
        x = group.variant_offset_from_tss_transcription_bp
        ax.plot(
            x,
            group.importance_score_alphagenome,
            color="#326891",
            linewidth=1.0,
            label="AlphaGenome original",
        )
        ax.plot(
            x,
            group.importance_score_borzoi,
            color="#d17a22",
            linewidth=1.0,
            label="Borzoi original",
        )
        ax.axhline(threshold, color="#444444", linestyle="--", linewidth=0.9)
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
    axes[0].legend(loc="lower right", frameon=False, ncol=2)
    axes[-1].set_xlabel(
        "Offset from TSS in transcription direction (bp)"
    )
    fig.suptitle(
        "Frozen modality-view score on original AlphaGenome and Borzoi",
        fontsize=13,
    )
    fig.savefig(out / "alphagenome_borzoi_score_profiles.png", dpi=180)
    fig.savefig(out / "alphagenome_borzoi_score_profiles.pdf")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    out = root / "analysis" / "borzoi_frozen_score"
    out.mkdir(parents=True, exist_ok=True)

    contract = json.loads(
        (
            root
            / "analysis"
            / "exploration_summary"
            / "frozen_score_contract.json"
        ).read_text()
    )
    if contract["selected_method"] != "method3_modality_views":
        raise ValueError("Frozen score contract is not Method 3")

    tracks = build_borzoi_view_features(root)
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

    ag_centers = pd.read_csv(
        root
        / "analysis"
        / "method3_modality_views"
        / "center_scores.tsv",
        sep="\t",
    )
    cross_model, concordance = compare_alphagenome_borzoi_scores(
        ag_centers, centers, args.threshold
    )
    tracks.to_csv(out / "track_view_scores.tsv", sep="\t", index=False)
    groups.to_csv(out / "group_view_scores.tsv", sep="\t", index=False)
    view_centers.to_csv(
        out / "view_center_scores.tsv", sep="\t", index=False
    )
    centers.to_csv(out / "center_scores.tsv", sep="\t", index=False)
    regions.to_csv(out / "score95_regions.tsv", sep="\t", index=False)
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
    cross_model.to_parquet(
        out / "alphagenome_borzoi_center_scores.parquet", index=False
    )
    concordance.to_csv(
        out / "alphagenome_borzoi_concordance.tsv",
        sep="\t",
        index=False,
    )
    plot_cross_model_scores(
        cross_model,
        controls,
        holdout,
        args.threshold,
        out,
    )

    group_counts = (
        groups.groupby(["gene", "score_view"])
        .track_group.nunique()
        .unstack()
        .astype(int)
    )
    validation = {
        "status": "ok",
        "method": "frozen_method3_modality_views",
        "score_contract_source": str(
            root
            / "analysis"
            / "exploration_summary"
            / "frozen_score_contract.json"
        ),
        "score_rule_changed_after_alphagenome": False,
        "centers_per_gene": (
            centers.groupby("gene").size().astype(int).to_dict()
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
                for view, count in counts.items()
            }
            for gene, counts in group_counts.to_dict(
                orient="index"
            ).items()
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
                        "importance_score",
                    ]
                ].to_numpy()
            )).sum()
        ),
        "duplicate_center_keys": int(
            centers.duplicated(CENTER_KEYS).sum()
        ),
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
        "score95_regions": int(len(regions)),
        "score95_singleton_regions": int(regions.centers.eq(1).sum()),
        "cross_model_aligned_centers": int(len(cross_model)),
    }
    if validation["centers_per_gene"] != {
        "Mdk": 1485,
        "Col1a1": 1494,
        "Acta2": 1495,
    }:
        raise RuntimeError(validation)
    if set(validation["tracks_per_gene"].values()) != {65}:
        raise RuntimeError(validation)
    if {
        tuple(sorted(counts.values()))
        for counts in validation[
            "track_groups_per_gene_and_view"
        ].values()
    } != {(6, 7)}:
        raise RuntimeError(validation)
    if validation["replacement_counts"] != [3]:
        raise RuntimeError(validation)
    if validation["nonfinite_track_values"]:
        raise RuntimeError(validation)
    if validation["nonfinite_center_values"]:
        raise RuntimeError(validation)
    if validation["duplicate_center_keys"]:
        raise RuntimeError(validation)
    (out / "validation_summary.json").write_text(
        json.dumps(validation, indent=2) + "\n"
    )
    print(json.dumps(validation, indent=2))
    print(control_recall.to_string(index=False))
    print(view_control_recall.to_string(index=False))
    print(holdout_recall.to_string(index=False))
    print(concordance.to_string(index=False))


if __name__ == "__main__":
    main()
