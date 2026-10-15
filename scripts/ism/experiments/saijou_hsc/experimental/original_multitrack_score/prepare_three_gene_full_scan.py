#!/usr/bin/env python
"""Filter the validated nine-gene 3-kb manifest to three complete gene scans."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


REPO = Path(__file__).resolve().parents[6]
DEFAULT_SOURCE = (
    REPO
    / "experiments/ism/saijou_nine_gene_tss_3kb_strict_shuffle_pdf_transcripts"
    / "prepared"
)
DEFAULT_CONTROLS = (
    REPO
    / "scripts/ism/experiments/saijou_hsc/configs/"
    "positive_control_registry.tsv"
)
DEFAULT_OUT = (
    REPO
    / "experiments/ism/original_multitrack_score_20260728"
    / "prepared_full3kb_three_gene"
)
DEFAULT_GENES = ("Mdk", "Col1a1", "Acta2")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-prepared", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--controls", type=Path, default=DEFAULT_CONTROLS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--genes", default=",".join(DEFAULT_GENES))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    requested = [gene.strip() for gene in args.genes.split(",") if gene.strip()]
    if not requested:
        raise ValueError("At least one gene is required")

    source = args.source_prepared.resolve()
    out = args.out_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)

    genes = pd.read_csv(source / "genes.tsv", sep="\t")
    loci = pd.read_csv(source / "loci.tsv", sep="\t")
    readouts = pd.read_csv(source / "readouts.tsv", sep="\t")
    manifest = pd.read_csv(source / "mutation_manifest.tsv", sep="\t")
    controls = pd.read_csv(args.controls, sep="\t")

    unknown = sorted(set(requested) - set(genes.gene))
    if unknown:
        raise ValueError(f"Unknown genes: {unknown}")

    selected_genes = genes.loc[genes.gene.isin(requested)].copy()
    selected_genes["_order"] = selected_genes.gene.map(
        {gene: index for index, gene in enumerate(requested)}
    )
    selected_genes = selected_genes.sort_values("_order").drop(columns="_order")
    selected_loci = loci.loc[loci.gene.isin(requested)].copy()
    selected_readouts = readouts.loc[
        readouts.gene.isin(requested) & readouts.role.eq("tss_1024bp")
    ].copy()
    selected_manifest = manifest.loc[manifest.gene.isin(requested)].copy()
    selected_manifest["_order"] = selected_manifest.gene.map(
        {gene: index for index, gene in enumerate(requested)}
    )
    selected_manifest = selected_manifest.sort_values(
        [
            "_order",
            "variant_offset_from_tss_transcription_bp",
            "replacement_replicate",
        ]
    ).drop(columns="_order")

    centers = selected_manifest[
        ["gene", "variant_offset_from_tss_transcription_bp"]
    ].drop_duplicates()
    center_roles = centers.copy()
    center_roles["positive_control"] = False
    center_roles["control_ids"] = ""
    center_roles["mdk_holdout"] = (
        center_roles.gene.eq("Mdk")
        & center_roles.variant_offset_from_tss_transcription_bp.between(
            463 - 5, 531 + 5
        )
    )
    for row in controls.loc[controls.gene.isin(requested)].itertuples(
        index=False
    ):
        mask = center_roles.gene.eq(row.gene) & (
            center_roles.variant_offset_from_tss_transcription_bp.between(
                int(row.start) - 5, int(row.end) + 5
            )
        )
        center_roles.loc[mask, "positive_control"] = True
        existing = center_roles.loc[mask, "control_ids"].astype(str)
        center_roles.loc[mask, "control_ids"] = existing.map(
            lambda value: (
                f"{value};{row.control_id}".strip(";")
                if value
                else str(row.control_id)
            )
        )
    center_roles["benchmark_role"] = "full_3kb_scan"
    center_roles.loc[
        center_roles.positive_control, "benchmark_role"
    ] += ";positive_control"
    center_roles.loc[
        center_roles.mdk_holdout, "benchmark_role"
    ] += ";mdk_holdout"

    selected_genes.to_csv(out / "genes.tsv", sep="\t", index=False)
    selected_loci.to_csv(out / "loci.tsv", sep="\t", index=False)
    selected_readouts.to_csv(out / "readouts.tsv", sep="\t", index=False)
    selected_manifest.to_csv(
        out / "mutation_manifest.tsv", sep="\t", index=False
    )
    center_roles.to_csv(out / "center_roles.tsv", sep="\t", index=False)
    controls.loc[controls.gene.isin(requested)].to_csv(
        out / "positive_control_registry.tsv", sep="\t", index=False
    )

    replicate_counts = (
        selected_manifest.groupby(
            ["gene", "variant_offset_from_tss_transcription_bp"]
        ).replacement_replicate.nunique()
    )
    per_gene = (
        selected_manifest.groupby("gene")
        .agg(
            mutations=("mutation_id", "size"),
            centers=(
                "variant_offset_from_tss_transcription_bp",
                "nunique",
            ),
        )
        .reindex(requested)
    )
    expected_mutations = int((per_gene.centers * 3).sum())
    validation = {
        "status": "ok",
        "source_prepared": str(source),
        "genes": requested,
        "centers_per_gene": per_gene.centers.astype(int).to_dict(),
        "mutations_per_gene": per_gene.mutations.astype(int).to_dict(),
        "centers": int(per_gene.centers.sum()),
        "mutations": int(len(selected_manifest)),
        "expected_mutations": expected_mutations,
        "replicates_per_center": sorted(
            replicate_counts.unique().astype(int).tolist()
        ),
        "duplicate_mutation_ids": int(
            selected_manifest.mutation_id.duplicated().sum()
        ),
        "positive_controls_in_scope": int(
            controls.gene.isin(requested).sum()
        ),
        "positive_control_centers": int(
            center_roles.positive_control.sum()
        ),
        "mdk_holdout_centers": int(center_roles.mdk_holdout.sum()),
        "readout_roles": selected_readouts.role.unique().tolist(),
        "coordinate_contract": {
            "species": "Mus musculus",
            "assembly": "mm10",
            "orientation": (
                "transcription-direction offset from boss transcript TSS"
            ),
            "scan_window_bp": [-1500, 1500],
            "actual_center_range_bp": [-1495, 1495],
            "edit_span_bp": 10,
            "center_stride_bp": 2,
            "replacement_replicates": 3,
        },
    }
    if validation["replicates_per_center"] != [3]:
        raise RuntimeError(validation)
    if validation["mutations"] != expected_mutations:
        raise RuntimeError(validation)
    if validation["duplicate_mutation_ids"]:
        raise RuntimeError(validation)
    (out / "validation_summary.json").write_text(
        json.dumps(validation, indent=2) + "\n"
    )
    print(json.dumps(validation, indent=2))


if __name__ == "__main__":
    main()
