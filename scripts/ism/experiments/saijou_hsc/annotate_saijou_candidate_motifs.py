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


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
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
