#!/usr/bin/env python
"""Freeze the label-free AlphaGenome multitrack exploration contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


REPO = Path(__file__).resolve().parents[6]
DEFAULT_ROOT = (
    REPO / "experiments/ism/original_multitrack_score_20260728"
)
DEFAULT_NINE_GENE_ROOT = (
    REPO
    / "experiments/ism/saijou_nine_gene_tss_3kb_strict_shuffle_pdf_transcripts"
)
METHOD_DIRS = {
    "method1_group_balanced_consensus": "method1_group_balanced_consensus",
    "method2_spatial_three_group_support": (
        "method2_spatial_three_group_support"
    ),
    "method3_modality_views": "method3_modality_views",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument(
        "--nine-gene-root", type=Path, default=DEFAULT_NINE_GENE_ROOT
    )
    return parser.parse_args()


def read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text())


def build_method_comparison(root: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for method, directory in METHOD_DIRS.items():
        path = root / "analysis" / directory
        validation = read_json(path / "validation_summary.json")
        regions = pd.read_csv(path / "score95_regions.tsv", sep="\t")
        if method == "method1_group_balanced_consensus":
            any_center = validation["positive_controls_recalled"]
            windows = validation["positive_control_windows_recalled"]
            mdk_any = validation["mdk_holdout_recalled"]
            mdk_window = validation["mdk_holdout_window_recalled"]
            either_view = np.nan
        elif method == "method2_spatial_three_group_support":
            any_center = validation[
                "positive_controls_recalled_any_center"
            ]
            windows = validation["positive_control_windows_recalled"]
            mdk_any = validation["mdk_holdout_recalled_any_center"]
            mdk_window = validation["mdk_holdout_window_recalled"]
            either_view = np.nan
        else:
            any_center = validation[
                "positive_controls_recalled_overall_any_center"
            ]
            windows = validation[
                "positive_control_windows_recalled_overall"
            ]
            mdk_any = validation[
                "mdk_holdout_recalled_overall_any_center"
            ]
            mdk_window = validation[
                "mdk_holdout_window_recalled_overall"
            ]
            either_view = validation[
                "positive_controls_recalled_in_either_view_any_center"
            ]
        rows.append(
            {
                "method": method,
                "uses_labels": bool(validation["score_uses_labels"]),
                "positive_controls": int(validation["positive_controls"]),
                "positive_controls_recalled_any_center": int(any_center),
                "positive_control_windows_recalled": int(windows),
                "positive_controls_recalled_either_view_any_center": (
                    either_view
                ),
                "mdk_holdout_recalled_any_center": bool(mdk_any),
                "mdk_holdout_window_recalled": bool(mdk_window),
                "score95_regions": int(len(regions)),
                "score95_singleton_regions": int(
                    regions.centers.eq(1).sum()
                ),
                "score_definition": validation["score_definition"],
            }
        )
    return pd.DataFrame.from_records(rows)


def build_control_evidence(
    root: Path, nine_gene_root: Path
) -> pd.DataFrame:
    controls = pd.read_csv(
        root
        / "prepared_full3kb_three_gene"
        / "positive_control_registry.tsv",
        sep="\t",
    )
    motif = pd.read_csv(
        nine_gene_root / "analysis" / "region_motif_annotations.tsv",
        sep="\t",
    )
    motif = motif.loc[
        motif.scan_kind.eq("registered_positive_control")
        & motif.gene.isin(controls.gene),
        [
            "gene",
            "region_label",
            "expected_family",
            "family",
            "annotation_status",
            "matches_expected_family",
            "affected_centers",
            "supporting_motifs",
            "motif_position",
            "median_score_loss",
            "top_motif",
        ],
    ].rename(columns={"region_label": "label"})
    result = controls.merge(
        motif,
        on=["gene", "label"],
        how="left",
        validate="one_to_one",
    )

    for prefix, directory in [
        ("m1", "method1_group_balanced_consensus"),
        ("m2", "method2_spatial_three_group_support"),
        ("m3", "method3_modality_views"),
    ]:
        recall = pd.read_csv(
            root
            / "analysis"
            / directory
            / "positive_control_recall.tsv",
            sep="\t",
        )
        recall = recall[
            [
                "control_id",
                "best_center",
                "best_score",
                "recalled_at_threshold",
                "window_percentile_score",
                "window_recalled_at_threshold",
                "best_median_group_signed_log2fc",
                "best_loss_direction_group_fraction",
            ]
        ].rename(
            columns={
                column: f"{prefix}_{column}"
                for column in recall.columns
                if column != "control_id"
            }
        )
        result = result.merge(
            recall, on="control_id", how="left", validate="one_to_one"
        )

    view_recall = pd.read_csv(
        root
        / "analysis"
        / "method3_modality_views"
        / "positive_control_view_recall.tsv",
        sep="\t",
    )
    view_columns = [
        "control_id",
        "local_regulatory_best_center",
        "local_regulatory_best_score",
        "local_regulatory_recalled_any_center",
        "local_regulatory_window_percentile_score",
        "output_best_center",
        "output_best_score",
        "output_recalled_any_center",
        "output_window_percentile_score",
        "either_view_recalled_any_center",
    ]
    result = result.merge(
        view_recall[view_columns],
        on="control_id",
        how="left",
        validate="one_to_one",
    )

    view_centers = pd.read_csv(
        root
        / "analysis"
        / "method3_modality_views"
        / "view_center_scores.tsv",
        sep="\t",
    )
    for view in ["local_regulatory", "output"]:
        best_center_column = f"{view}_best_center"
        direction = view_centers.loc[
            view_centers.score_view.eq(view),
            [
                "gene",
                "variant_offset_from_tss_transcription_bp",
                "median_group_signed_log2fc",
                "loss_direction_group_fraction",
            ],
        ].rename(
            columns={
                "variant_offset_from_tss_transcription_bp": (
                    best_center_column
                ),
                "median_group_signed_log2fc": (
                    f"{view}_best_median_group_signed_log2fc"
                ),
                "loss_direction_group_fraction": (
                    f"{view}_best_loss_direction_group_fraction"
                ),
            }
        )
        result = result.merge(
            direction,
            on=["gene", best_center_column],
            how="left",
            validate="one_to_one",
        )

    result["native_motif_loss_confirmed"] = (
        result.annotation_status.eq("consistent_native_motif_loss")
        & result.matches_expected_family.eq(1)
    )
    result["m3_overall_majority_negative_response"] = (
        result.m3_best_loss_direction_group_fraction >= (2 / 3)
    ) & (result.m3_best_median_group_signed_log2fc < 0)
    result["m3_output_majority_negative_response"] = (
        result.output_best_loss_direction_group_fraction >= (2 / 3)
    ) & (result.output_best_median_group_signed_log2fc < 0)
    result["m3_local_majority_negative_response"] = (
        result.local_regulatory_best_loss_direction_group_fraction >= (2 / 3)
    ) & (result.local_regulatory_best_median_group_signed_log2fc < 0)
    return result


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    nine_gene_root = args.nine_gene_root.resolve()
    out = root / "analysis" / "exploration_summary"
    out.mkdir(parents=True, exist_ok=True)

    comparison = build_method_comparison(root)
    controls = build_control_evidence(root, nine_gene_root)
    comparison.to_csv(
        out / "method_comparison.tsv", sep="\t", index=False
    )
    controls.to_csv(
        out / "positive_control_evidence.tsv", sep="\t", index=False
    )

    frozen_contract = {
        "status": "frozen_for_borzoi_adaptation",
        "selected_method": "method3_modality_views",
        "selection_used_positive_control_labels_for_weights": False,
        "selection_rationale": [
            (
                "Separates gene-output evidence from mutation-local "
                "regulatory evidence using modality semantics."
            ),
            (
                "Requires at least three track groups per view and one-edit "
                "spatial support; no score>95 region is a singleton."
            ),
            (
                "Preserves a single final within-gene percentile while "
                "retaining interpretable view-specific components."
            ),
            (
                "Was selected for portability and failure control, not for "
                "maximizing four positive-control labels."
            ),
        ],
        "score_semantics": {
            "final_importance_score": (
                "within-gene empirical percentile rank of max(output view "
                "score, local regulatory view score)"
            ),
            "strict_threshold": ">95",
            "strict_threshold_meaning": (
                "overall top five percent of scanned centers for that gene; "
                "not confidence, effect size, recovery probability, or FDR"
            ),
            "signed_effect": (
                "reported separately; negative majority response is described "
                "as loss-like, not required for importance ranking"
            ),
        },
        "view_contract": {
            "output": {
                "modalities": ["rna_seq", "cage"],
                "readout": "TSS 1024-bp log2FC ratio of sums",
                "support": "third-highest of six track-group percentiles",
            },
            "local_regulatory": {
                "modalities": [
                    "atac",
                    "dnase",
                    "chip_histone",
                    "chip_tf",
                ],
                "readout": (
                    "maximum absolute signed log2FC across mutation bin and "
                    "one adjacent 128-bp bin on each side"
                ),
                "support": "third-highest of six track-group percentiles",
            },
            "spatial_support": (
                "median across centers within +/-4 bp, matching one 10-bp "
                "edit footprint on the 2-bp scan grid"
            ),
        },
        "recall_contract": {
            "primary": (
                "same-width-window percentile above 95 for the final overall "
                "score statistic"
            ),
            "secondary": [
                "any overlapping center with final overall score above 95",
                (
                    "any overlapping center with a view-specific score above "
                    "95, reported by view"
                ),
            ],
            "positive_control_units": 4,
            "positive_control_centers_are_not_independent_units": True,
            "mdk_interval_is_holdout_not_positive_control": True,
        },
        "observed_alphagenome_original": {
            "overall_any_center_recall": "1/4",
            "overall_window_recall": "1/4",
            "either_view_any_center_recall": "3/4",
            "gene_stratified_overall_window_recall": {
                "Acta2": "1/1",
                "Col1a1": "0/3",
                "Mdk": "N/A",
            },
            "mdk_holdout_overall_window_recalled": True,
        },
        "borzoi_adaptation_rule": (
            "Map Borzoi tracks into the same modality views. Preserve the "
            "within-view third-highest-group and spatial rules when each view "
            "has at least three groups. If a view has fewer than three groups, "
            "mark that view unavailable rather than silently changing k."
        ),
    }
    (out / "frozen_score_contract.json").write_text(
        json.dumps(frozen_contract, indent=2) + "\n"
    )

    validation = {
        "status": "ok",
        "methods": comparison.method.tolist(),
        "selected_method": frozen_contract["selected_method"],
        "positive_controls": int(len(controls)),
        "native_motif_loss_confirmed": int(
            controls.native_motif_loss_confirmed.sum()
        ),
        "duplicate_control_ids": int(
            controls.control_id.duplicated().sum()
        ),
        "nonfinite_method_control_counts": int(
            comparison[
                [
                    "positive_controls",
                    "positive_controls_recalled_any_center",
                    "positive_control_windows_recalled",
                    "score95_regions",
                    "score95_singleton_regions",
                ]
            ].isna().sum().sum()
        ),
        "contract_status": frozen_contract["status"],
    }
    if validation["positive_controls"] != 4:
        raise RuntimeError(validation)
    if validation["native_motif_loss_confirmed"] != 4:
        raise RuntimeError(validation)
    if validation["duplicate_control_ids"]:
        raise RuntimeError(validation)
    if validation["nonfinite_method_control_counts"]:
        raise RuntimeError(validation)
    (out / "validation_summary.json").write_text(
        json.dumps(validation, indent=2) + "\n"
    )
    print(json.dumps(validation, indent=2))
    print(comparison.to_string(index=False))
    print(
        controls[
            [
                "control_id",
                "native_motif_loss_confirmed",
                "m3_best_score",
                "m3_window_percentile_score",
                "local_regulatory_best_score",
                "output_best_score",
                "m3_best_median_group_signed_log2fc",
                "m3_best_loss_direction_group_fraction",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
