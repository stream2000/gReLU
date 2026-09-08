#!/usr/bin/env python
"""Annotate selected nine-gene candidate regions with motif disruption."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from grelu.interpret.motifs import scan_sequences  # noqa: E402
from grelu.io.motifs import get_jaspar  # noqa: E402

if __package__:
    from .tools.genomics import (
        build_edit_context_sequences,
        canonical_motif_family,
    )
else:
    from tools.genomics import build_edit_context_sequences, canonical_motif_family


DEFAULT_ROOT = REPO_ROOT / "experiments/ism/saijou_all_genes_10bp_scan"
DEFAULT_FASTA = "/work/Database/Database_fromDocker/Referencedata_mm10/genome.fa"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--fasta", type=Path, default=DEFAULT_FASTA)
    parser.add_argument("--context-bp", type=int, default=80)
    parser.add_argument("--pthresh", type=float, default=1e-3)
    parser.add_argument(
        "--annotation-profile",
        choices=("candidate_disruption", "native_loss"),
        default="candidate_disruption",
    )
    parser.add_argument("--motif-loss-delta", type=float, default=5.0)
    parser.add_argument("--motif-loss-replicates", type=int, default=2)
    parser.add_argument("--max-families-per-region", type=int, default=1)
    return parser.parse_args()


def _scan_candidate_motifs(
    manifest: pd.DataFrame, args: argparse.Namespace
) -> tuple[list[str], object, pd.DataFrame]:
    sequences, ids, sequence_rows = build_edit_context_sequences(
        manifest,
        args.fasta,
        args.context_bp,
        metadata_columns={"candidate_segment_id": "locus_id", "gene": "gene"},
    )
    motifs = get_jaspar(release="JASPAR2024", tax_group="vertebrates")
    hits = scan_sequences(
        sequences,
        motifs,
        seq_ids=ids,
        pthresh=args.pthresh,
        rc=True,
    )
    hits = hits.merge(sequence_rows, on="sequence", how="left", validate="many_to_one")
    hits["overlaps_edit"] = (hits.start < hits.edit_rel_end) & (hits.end > hits.edit_rel_start)
    hits["family"] = hits.motif.map(canonical_motif_family)
    hits = hits.sort_values(
        ["sequence", "motif", "start", "end", "strand"], kind="stable"
    ).reset_index(drop=True)
    return sequences, motifs, hits


def _disruption_scores(hits: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    overlap = hits.loc[hits.overlaps_edit].copy()
    scores = overlap.pivot_table(
        index=["mutation_id", "candidate_segment_id", "gene", "center", "motif", "family"],
        columns="state",
        values="fimo_score",
        aggfunc="max",
        fill_value=0.0,
    ).reset_index()
    for state in ["ref", "alt"]:
        if state not in scores:
            scores[state] = 0.0
    scores["signed_score_change"] = scores.ref - scores.alt
    scores["absolute_score_change"] = scores.signed_score_change.abs()
    return overlap, scores


def _summarize_candidate_motifs(
    scores: pd.DataFrame, centers: pd.DataFrame
) -> pd.DataFrame:
    center_metrics = centers[
        [
            "candidate_segment_id",
            "gene",
            "variant_offset_from_tss_transcription_bp",
            "ag_hsc_margin",
            "bz_hsc_margin",
            "ag_generic_effect",
            "bz_generic_effect",
            "discovery_class",
            "region_type",
        ]
    ].drop_duplicates(
        ["candidate_segment_id", "gene", "variant_offset_from_tss_transcription_bp"]
    )
    center_motif = (
        scores.groupby(["candidate_segment_id", "gene", "center", "motif", "family"])
        .agg(
            median_abs_score_change=("absolute_score_change", "median"),
            median_signed_score_change=("signed_score_change", "median"),
            shuffle_replicates=("mutation_id", "nunique"),
        )
        .reset_index()
        .merge(
            center_metrics,
            left_on=["candidate_segment_id", "gene", "center"],
            right_on=[
                "candidate_segment_id",
                "gene",
                "variant_offset_from_tss_transcription_bp",
            ],
            how="left",
            validate="many_to_one",
        )
    )
    summary = (
        center_motif.groupby(["candidate_segment_id", "gene", "motif", "family"])
        .agg(
            affected_centers=("center", "nunique"),
            max_abs_score_change=("median_abs_score_change", "max"),
            median_abs_score_change=("median_abs_score_change", "median"),
            median_ag_hsc_margin=("ag_hsc_margin", "median"),
            median_bz_hsc_margin=("bz_hsc_margin", "median"),
            median_ag_generic_effect=("ag_generic_effect", "median"),
            median_bz_generic_effect=("bz_generic_effect", "median"),
        )
        .reset_index()
        .sort_values(
            ["candidate_segment_id", "affected_centers", "max_abs_score_change"],
            ascending=[True, False, False],
        )
    )
    return summary


def _validation_summary(
    *,
    sequences: list[str],
    motifs,
    hits: pd.DataFrame,
    overlap: pd.DataFrame,
    scores: pd.DataFrame,
    summary: pd.DataFrame,
    manifest: pd.DataFrame,
) -> dict[str, object]:
    validation = {
        "status": "ok",
        "sequences": len(sequences),
        "jaspar_motifs": len(motifs),
        "hits": len(hits),
        "overlapping_hits": len(overlap),
        "disruption_rows": len(scores),
        "summary_rows": len(summary),
        "segments": int(manifest.locus_id.nunique()),
        "genes": int(manifest.gene.nunique()),
        "nonfinite_score_changes": int(
            (~np.isfinite(scores[["signed_score_change", "absolute_score_change"]])).sum().sum()
        ),
    }
    if validation["nonfinite_score_changes"]:
        raise RuntimeError(validation)
    return validation


def _native_loss_scan_regions(
    regions: pd.DataFrame, controls: pd.DataFrame
) -> pd.DataFrame:
    candidate = regions.copy()
    candidate["scan_kind"] = "ranked_candidate"
    candidate["expected_family"] = ""
    control_rows = []
    for screen_rank, row in enumerate(controls.itertuples(index=False), start=1001):
        control_rows.append(
            {
                "screen_rank": screen_rank,
                "gene": row.gene,
                "region_start": int(row.start),
                "region_end": int(row.end),
                "region_label": row.label,
                "region_score": float(row.best_score),
                "peak_score": float(row.best_score),
                "peak_offset": int(row.best_center),
                "driving_track": str(row.driving_track),
                "driving_readout": str(row.driving_readout),
                "evidence_tier": str(row.evidence),
                "scan_kind": "registered_positive_control",
                "expected_family": str(row.family),
            }
        )
    control = pd.DataFrame.from_records(control_rows)
    columns = sorted(set(candidate.columns).union(control.columns))
    return pd.concat(
        [candidate.reindex(columns=columns), control.reindex(columns=columns)],
        ignore_index=True,
    )


def _assign_native_loss_regions(
    manifest: pd.DataFrame, regions: pd.DataFrame
) -> pd.DataFrame:
    pieces = []
    for region in regions.itertuples(index=False):
        selected = manifest.loc[
            manifest.gene.eq(region.gene)
            & manifest.variant_offset_from_tss_transcription_bp.between(
                int(region.region_start), int(region.region_end)
            )
        ].copy()
        if selected.empty:
            raise ValueError(f"No mutations cover screen rank {region.screen_rank}")
        selected["locus_id"] = f"region_{int(region.screen_rank):02d}"
        pieces.append(selected)
    return pd.concat(pieces, ignore_index=True)


def _native_loss_annotations(
    scan_regions: pd.DataFrame,
    scores: pd.DataFrame,
    *,
    delta: float,
    replicates: int,
    max_families: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    scores = scores.rename(columns={"signed_score_change": "signed_score_loss"})
    scores["passes_native_loss"] = (
        scores.ref.gt(0) & scores.signed_score_loss.ge(delta)
    )
    center = (
        scores.groupby(
            ["candidate_segment_id", "gene", "center", "motif", "family"],
            sort=False,
        )
        .agg(
            replacement_replicates=("mutation_id", "nunique"),
            loss_replicates=("passes_native_loss", "sum"),
            median_signed_score_loss=("signed_score_loss", "median"),
            max_signed_score_loss=("signed_score_loss", "max"),
        )
        .reset_index()
    )
    eligible = center.loc[
        center.loss_replicates.ge(replicates)
        & center.median_signed_score_loss.ge(delta)
        & center.family.ne("OTHER")
    ].copy()
    motif_rank = (
        eligible.groupby(
            ["candidate_segment_id", "gene", "family", "motif"], sort=False
        )
        .agg(
            affected_centers=("center", "nunique"),
            motif_position=("center", "median"),
            median_score_loss=("median_signed_score_loss", "median"),
            max_score_loss=("max_signed_score_loss", "max"),
        )
        .reset_index()
        .sort_values(
            [
                "candidate_segment_id",
                "affected_centers",
                "median_score_loss",
                "max_score_loss",
                "motif",
            ],
            ascending=[True, False, False, False, True],
        )
    )
    family = (
        motif_rank.groupby(
            ["candidate_segment_id", "gene", "family"], sort=False
        )
        .agg(
            affected_centers=("affected_centers", "max"),
            supporting_motifs=("motif", "nunique"),
            motif_position=("motif_position", "median"),
            median_score_loss=("median_score_loss", "max"),
            max_score_loss=("max_score_loss", "max"),
        )
        .reset_index()
    )
    top_motifs = motif_rank.drop_duplicates(
        ["candidate_segment_id", "gene", "family"], keep="first"
    )[["candidate_segment_id", "gene", "family", "motif"]].rename(
        columns={"motif": "top_motif"}
    )
    family = family.merge(
        top_motifs,
        on=["candidate_segment_id", "gene", "family"],
        how="left",
        validate="one_to_one",
    ).sort_values(
        [
            "candidate_segment_id",
            "affected_centers",
            "median_score_loss",
            "max_score_loss",
            "family",
        ],
        ascending=[True, False, False, False, True],
    )
    family["family_rank"] = family.groupby("candidate_segment_id").cumcount() + 1
    family = family.loc[family.family_rank.le(max_families)].copy()

    base = scan_regions.copy()
    base["candidate_segment_id"] = base.screen_rank.map(
        lambda value: f"region_{int(value):02d}"
    )
    annotations = base.merge(
        family,
        on=["candidate_segment_id", "gene"],
        how="left",
        validate="one_to_many",
    )
    annotations["region_id"] = annotations.candidate_segment_id
    annotations["family"] = annotations.family.fillna("UNRESOLVED")
    annotations["top_motif"] = annotations.top_motif.fillna("")
    annotations["family_rank"] = annotations.family_rank.fillna(1).astype(int)
    annotations["annotation_status"] = np.where(
        annotations.family.eq("UNRESOLVED"),
        "no_consistent_native_motif_loss",
        "consistent_native_motif_loss",
    )
    annotations["matches_expected_family"] = np.where(
        annotations.scan_kind.eq("registered_positive_control"),
        annotations.family.eq(annotations.expected_family),
        np.nan,
    )
    return center, annotations


def _run_native_loss(root: Path, args: argparse.Namespace) -> dict[str, object]:
    analysis = root / "analysis"
    regions = pd.read_csv(analysis / "top_important_regions.tsv", sep="\t")
    controls = pd.read_csv(analysis / "positive_control_recall.tsv", sep="\t")
    manifest = pd.read_csv(root / "prepared/mutation_manifest.tsv", sep="\t")
    scan_regions = _native_loss_scan_regions(regions, controls)

    assigned_parts = []
    hit_parts = []
    motif_count = 0
    for _, group in scan_regions.groupby("scan_kind", sort=False):
        assigned = _assign_native_loss_regions(manifest, group)
        _, motifs, hits = _scan_candidate_motifs(assigned, args)
        assigned_parts.append(assigned)
        hit_parts.append(hits)
        motif_count = motif_count or len(motifs)
        if motif_count != len(motifs):
            raise ValueError("JASPAR motif count changed between scan groups")
    assigned = pd.concat(assigned_parts, ignore_index=True)
    hits = pd.concat(hit_parts, ignore_index=True)
    overlap, scores = _disruption_scores(hits)
    center, annotations = _native_loss_annotations(
        scan_regions,
        scores,
        delta=args.motif_loss_delta,
        replicates=args.motif_loss_replicates,
        max_families=args.max_families_per_region,
    )
    family_columns = [
        "candidate_segment_id",
        "gene",
        "family",
        "affected_centers",
        "supporting_motifs",
        "motif_position",
        "median_score_loss",
        "max_score_loss",
        "top_motif",
        "family_rank",
    ]
    families = annotations.loc[
        annotations.family.ne("UNRESOLVED"), family_columns
    ].drop_duplicates()
    annotations.to_csv(
        analysis / "region_motif_annotations.tsv", sep="\t", index=False
    )
    scores.to_parquet(analysis / "motif_loss_by_mutation.parquet", index=False)
    center.to_csv(analysis / "motif_loss_by_center.tsv", sep="\t", index=False)
    families.to_csv(
        analysis / "motif_loss_by_region_family.tsv", sep="\t", index=False
    )
    scan_regions.to_csv(
        analysis / "motif_scan_regions.tsv", sep="\t", index=False
    )

    candidate_annotations = annotations.loc[
        annotations.scan_kind.eq("ranked_candidate")
        & annotations.annotation_status.eq("consistent_native_motif_loss")
    ]
    control_annotations = annotations.loc[
        annotations.scan_kind.eq("registered_positive_control")
    ]
    validation = {
        "status": "ok",
        "motif_release": "JASPAR2024 vertebrates",
        "motif_pthresh": float(args.pthresh),
        "motif_loss_delta": float(args.motif_loss_delta),
        "motif_loss_replicates": int(args.motif_loss_replicates),
        "jaspar_motifs": int(motif_count),
        "scan_regions": int(len(scan_regions)),
        "candidate_regions": int(len(regions)),
        "registered_positive_controls": int(len(controls)),
        "manifest_mutations_scanned": int(len(assigned)),
        "overlapping_motif_hits": int(len(overlap)),
        "candidate_regions_with_motif_loss": int(
            candidate_annotations.region_id.nunique()
        ),
        "controls_matching_expected_family": int(
            control_annotations.matches_expected_family.fillna(False).sum()
        ),
        "nonfinite_center_loss_values": int(
            center[["median_signed_score_loss", "max_signed_score_loss"]]
            .isna()
            .sum()
            .sum()
        ),
    }
    if validation["nonfinite_center_loss_values"]:
        validation["status"] = "failed"
    (analysis / "motif_validation_summary.json").write_text(
        json.dumps(validation, indent=2) + "\n"
    )
    print(json.dumps(validation, indent=2))
    if validation["status"] != "ok":
        raise RuntimeError(validation)
    return validation


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    if args.annotation_profile == "native_loss":
        _run_native_loss(root, args)
        return
    analysis = root / "analysis"
    manifest = pd.read_csv(
        root / "prepared_original_candidates/mutation_manifest.tsv", sep="\t"
    )
    centers = pd.read_csv(analysis / "candidate_centers.tsv", sep="\t")
    sequences, motifs, hits = _scan_candidate_motifs(manifest, args)
    overlap, scores = _disruption_scores(hits)
    summary = _summarize_candidate_motifs(scores, centers)
    hits.to_csv(analysis / "candidate_motif_hits.tsv", sep="\t", index=False)
    scores.to_csv(
        analysis / "candidate_motif_disruption_by_mutation.tsv", sep="\t", index=False
    )
    summary.to_csv(
        analysis / "candidate_motif_disruption_summary.tsv", sep="\t", index=False
    )
    validation = _validation_summary(
        sequences=sequences,
        motifs=motifs,
        hits=hits,
        overlap=overlap,
        scores=scores,
        summary=summary,
        manifest=manifest,
    )
    (analysis / "motif_validation_summary.json").write_text(
        json.dumps(validation, indent=2) + "\n"
    )
    print(json.dumps(validation, indent=2))


if __name__ == "__main__":
    main()
