#!/usr/bin/env python
"""Stage frozen modality-view score regions for canonical motif annotation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


REPO = Path(__file__).resolve().parents[6]
DEFAULT_ROOT = (
    REPO / "experiments/ism/original_multitrack_score_20260728"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument(
        "--model",
        choices=["alphagenome", "borzoi"],
        default="alphagenome",
        help="Frozen-score result to stage; defaults to AlphaGenome.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    if args.model == "alphagenome":
        source = root / "analysis" / "method3_modality_views"
    else:
        source = root / "analysis" / "borzoi_frozen_score"
    staged = root / f"motif_annotation_method3_{args.model}"
    prepared = staged / "prepared"
    analysis = staged / "analysis"
    prepared.mkdir(parents=True, exist_ok=True)
    analysis.mkdir(parents=True, exist_ok=True)

    manifest = pd.read_csv(
        root
        / "prepared_full3kb_three_gene"
        / "mutation_manifest.tsv",
        sep="\t",
    )
    regions = pd.read_csv(source / "score95_regions.tsv", sep="\t")
    centers = pd.read_csv(source / "center_scores.tsv", sep="\t")
    controls = pd.read_csv(
        source / "positive_control_recall.tsv", sep="\t"
    )

    regions = regions.sort_values(
        ["peak_score", "centers", "gene", "region_start"],
        ascending=[False, False, True, True],
    ).reset_index(drop=True)
    regions.insert(0, "screen_rank", range(1, len(regions) + 1))
    regions["region_score"] = regions.peak_score
    regions["peak_offset"] = regions.peak_center
    regions["region_label"] = regions.apply(
        lambda row: (
            f"{row.gene} {int(row.region_start):+d}.."
            f"{int(row.region_end):+d}"
        ),
        axis=1,
    )
    peak_context = centers[
        [
            "gene",
            "variant_offset_from_tss_transcription_bp",
            "driving_view",
        ]
    ].rename(
        columns={
            "variant_offset_from_tss_transcription_bp": "peak_center",
            "driving_view": "driving_track",
        }
    )
    regions = regions.merge(
        peak_context,
        on=["gene", "peak_center"],
        how="left",
        validate="many_to_one",
    )
    regions["driving_readout"] = "frozen_method3_modality_view"
    regions["evidence_tier"] = (
        f"{args.model}_original_score_gt_95"
    )

    control_context = centers[
        [
            "gene",
            "variant_offset_from_tss_transcription_bp",
            "driving_view",
        ]
    ].rename(
        columns={
            "variant_offset_from_tss_transcription_bp": "best_center",
            "driving_view": "driving_track",
        }
    )
    controls = controls.merge(
        control_context,
        on=["gene", "best_center"],
        how="left",
        validate="many_to_one",
    )
    controls["driving_readout"] = "frozen_method3_modality_view"

    manifest.to_csv(
        prepared / "mutation_manifest.tsv", sep="\t", index=False
    )
    regions.to_csv(
        analysis / "top_important_regions.tsv", sep="\t", index=False
    )
    controls.to_csv(
        analysis / "positive_control_recall.tsv", sep="\t", index=False
    )
    validation = {
        "status": "ok",
        "source_method": "method3_modality_views",
        "model": args.model,
        "regions": int(len(regions)),
        "genes": int(regions.gene.nunique()),
        "positive_controls": int(len(controls)),
        "manifest_mutations": int(len(manifest)),
        "duplicate_screen_ranks": int(
            regions.screen_rank.duplicated().sum()
        ),
        "regions_without_manifest_centers": int(
            sum(
                not (
                    manifest.gene.eq(row.gene)
                    & manifest.variant_offset_from_tss_transcription_bp.between(
                        int(row.region_start), int(row.region_end)
                    )
                ).any()
                for row in regions.itertuples(index=False)
            )
        ),
    }
    if validation["regions"] < 1:
        raise RuntimeError(validation)
    if validation["genes"] != 3:
        raise RuntimeError(validation)
    if validation["positive_controls"] != 4:
        raise RuntimeError(validation)
    if validation["duplicate_screen_ranks"]:
        raise RuntimeError(validation)
    if validation["regions_without_manifest_centers"]:
        raise RuntimeError(validation)
    (analysis / "staging_validation_summary.json").write_text(
        json.dumps(validation, indent=2) + "\n"
    )
    print(json.dumps(validation, indent=2))


if __name__ == "__main__":
    main()
