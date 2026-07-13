#!/usr/bin/env python
"""Score nine-gene 10-bp scans and prepare non-splice candidate regions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

if __package__:
    from .tools.genomics import (
        add_genomic_annotation,
        load_gtf_annotations,
    )
    from .tools.candidates import cluster_candidates
else:
    from tools.genomics import add_genomic_annotation, load_gtf_annotations
    from tools.candidates import cluster_candidates


REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_ROOT = REPO_ROOT / "experiments/ism/saijou_all_genes_10bp_scan"
DEFAULT_GTF = "/work/Database/Database_fromDocker/Referencedata_mm10/gtf_chrUCSC/chr.gtf"
MODELS = {
    "AlphaGenome e19": "runs/alphagenome_finetuned",
    "Borzoi e39": "runs/borzoi_finetuned",
}
CELLS = ["hsc", "mac", "lsec", "chol"]
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--gtf", default=DEFAULT_GTF)
    parser.add_argument("--segments-per-gene", type=int, default=2)
    parser.add_argument("--max-centers-per-segment", type=int, default=20)
    parser.add_argument("--candidate-quantile", type=float, default=0.95)
    parser.add_argument(
        "--splice-buffer-bp",
        type=int,
        default=6,
        help=(
            "Protect the annotated boundary plus extended core splice grammar. "
            "With a 10-bp edit this excludes centers within 11 bp of a boundary."
        ),
    )
    return parser.parse_args()


def read_model_features(root: Path, model: str, relpath: str) -> pd.DataFrame:
    path = root / relpath / "features/combined_mutation_features.tsv"
    data = pd.read_csv(path, sep="\t")
    data = data.loc[data.track_id.isin(CELLS)].copy()
    data["model"] = model
    data["cell_type"] = data.track_id
    data["abs_effect"] = data.log2fc_ratio_of_sums.abs()
    return data


def center_scores(data: pd.DataFrame) -> pd.DataFrame:
    center_cell = (
        data.groupby(
            [
                "model",
                "gene",
                "readout_role",
                "variant_offset_from_tss_transcription_bp",
                "cell_type",
            ],
            sort=False,
        )
        .agg(
            median_abs_effect=("abs_effect", "median"),
            median_signed_effect=("log2fc_ratio_of_sums", "median"),
            shuffle_sd=("log2fc_ratio_of_sums", "std"),
            shuffles=("mutation_id", "nunique"),
        )
        .reset_index()
    )
    wide = center_cell.pivot_table(
        index=["model", "gene", "readout_role", "variant_offset_from_tss_transcription_bp"],
        columns="cell_type",
        values="median_abs_effect",
    ).reset_index()
    signed = center_cell.pivot_table(
        index=["model", "gene", "readout_role", "variant_offset_from_tss_transcription_bp"],
        columns="cell_type",
        values="median_signed_effect",
    ).reset_index()
    signed = signed.rename(columns={cell: f"{cell}_signed_effect" for cell in CELLS})
    wide = wide.merge(
        signed,
        on=["model", "gene", "readout_role", "variant_offset_from_tss_transcription_bp"],
        how="left",
        validate="one_to_one",
    )
    missing = [cell for cell in CELLS if cell not in wide.columns]
    if missing:
        raise ValueError(f"Missing cell heads: {missing}")
    wide["max_other_effect"] = wide[["mac", "lsec", "chol"]].max(axis=1)
    wide["generic_effect"] = wide[CELLS].max(axis=1)
    wide["hsc_margin"] = wide.hsc - wide.max_other_effect
    wide["hsc_ratio"] = wide.hsc / wide.max_other_effect.clip(lower=1e-12)
    wide["top_cell"] = wide[CELLS].idxmax(axis=1)
    for field in ["generic_effect", "hsc_margin", "hsc_ratio"]:
        wide[f"{field}_percentile"] = wide.groupby(
            ["model", "gene", "readout_role"]
        )[field].rank(method="average", pct=True)
    return wide


def combined_center_scores(scores: pd.DataFrame) -> pd.DataFrame:
    keys = ["gene", "readout_role", "variant_offset_from_tss_transcription_bp"]
    rows = []
    for key, group in scores.groupby(keys, sort=False):
        by_model = group.set_index("model")
        if set(by_model.index) != set(MODELS):
            continue
        generic_pct = by_model.generic_effect_percentile
        hsc_pct = by_model.hsc_margin_percentile
        rows.append(
            {
                "gene": key[0],
                "readout_role": key[1],
                "variant_offset_from_tss_transcription_bp": int(key[2]),
                "ag_generic_effect": by_model.loc["AlphaGenome e19", "generic_effect"],
                "bz_generic_effect": by_model.loc["Borzoi e39", "generic_effect"],
                "ag_hsc_margin": by_model.loc["AlphaGenome e19", "hsc_margin"],
                "bz_hsc_margin": by_model.loc["Borzoi e39", "hsc_margin"],
                "ag_hsc_signed_effect": by_model.loc[
                    "AlphaGenome e19", "hsc_signed_effect"
                ],
                "bz_hsc_signed_effect": by_model.loc[
                    "Borzoi e39", "hsc_signed_effect"
                ],
                "ag_hsc_ratio": by_model.loc["AlphaGenome e19", "hsc_ratio"],
                "bz_hsc_ratio": by_model.loc["Borzoi e39", "hsc_ratio"],
                "ag_top_cell": by_model.loc["AlphaGenome e19", "top_cell"],
                "bz_top_cell": by_model.loc["Borzoi e39", "top_cell"],
                "concordant_generic_percentile": float(generic_pct.min()),
                "single_model_generic_percentile": float(generic_pct.max()),
                "concordant_hsc_margin_percentile": float(hsc_pct.min()),
                "single_model_hsc_margin_percentile": float(hsc_pct.max()),
                "both_hsc_top": bool((by_model.top_cell == "hsc").all()),
            }
        )
    combined = pd.DataFrame.from_records(rows)
    # A high HSC ratio/margin percentile is not useful when every cell head has
    # a near-zero edit effect.  Require at least a 75th-percentile generic
    # effect in one model before the HSC-specific component can drive ranking.
    hsc_component = combined.concordant_hsc_margin_percentile.where(
        combined.single_model_generic_percentile.ge(0.75), 0.0
    )
    combined["hsc_specific_component_percentile"] = hsc_component
    combined["discovery_score"] = pd.concat(
        [
            combined.concordant_generic_percentile,
            hsc_component,
            combined.single_model_generic_percentile,
        ],
        axis=1,
    ).max(axis=1)
    combined["discovery_class"] = np.select(
        [
            combined.both_hsc_top
            & combined.concordant_hsc_margin_percentile.ge(0.90)
            & combined.single_model_generic_percentile.ge(0.75),
            combined.concordant_generic_percentile.ge(0.95),
            combined.single_model_generic_percentile.ge(0.98),
        ],
        ["cross_model_hsc_specific", "cross_model_generic", "single_model_strong"],
        default="background",
    )
    combined["selection_priority"] = (
        combined.discovery_score
        + 0.15 * combined.discovery_class.eq("cross_model_hsc_specific")
        + 0.08 * combined.discovery_class.eq("cross_model_generic")
        + 0.03 * combined.concordant_hsc_margin_percentile
    )
    return combined


def prepare_original_manifest(
    root: Path,
    genes: pd.DataFrame,
    readouts: pd.DataFrame,
    source_manifest: pd.DataFrame,
    segments: pd.DataFrame,
    candidate_centers: pd.DataFrame,
) -> dict[str, object]:
    out = root / "prepared_original_candidates"
    out.mkdir(parents=True, exist_ok=True)
    keys = candidate_centers[
        ["gene", "variant_offset_from_tss_transcription_bp", "candidate_segment_id"]
    ].drop_duplicates()
    manifest = source_manifest.merge(
        keys,
        on=["gene", "variant_offset_from_tss_transcription_bp"],
        how="inner",
        validate="many_to_one",
    )
    manifest["locus_id"] = manifest.candidate_segment_id
    manifest["locus_role"] = "non_splice_candidate_region"
    manifest["source"] = "selected from cross-model fine-tuned nine-gene 10-bp scan"
    manifest = manifest.drop(columns="candidate_segment_id")
    selected_genes = genes.loc[genes.gene.isin(manifest.gene.unique())].copy()
    selected_readouts = readouts.loc[readouts.gene.isin(manifest.gene.unique())].copy()
    loci = segments.rename(
        columns={"candidate_segment_id": "locus_id", "peak_region_type": "role"}
    ).copy()
    loci["analysis_tss"] = loci.gene.map(selected_genes.set_index("gene").analysis_tss)
    loci["chrom"] = loci.gene.map(selected_genes.set_index("gene").chrom)
    loci["strand"] = loci.gene.map(selected_genes.set_index("gene").strand)
    loci["tx_span"] = loci.apply(lambda r: f"{int(r.tx_start):+d}..{int(r.tx_end):+d}", axis=1)
    loci["span_bp"] = 10
    loci["replicates"] = 3
    loci["evidence"] = loci.peak_discovery_class
    selected_genes.to_csv(out / "genes.tsv", sep="\t", index=False)
    selected_readouts.to_csv(out / "readouts.tsv", sep="\t", index=False)
    loci.to_csv(out / "loci.tsv", sep="\t", index=False)
    manifest.to_csv(out / "mutation_manifest.tsv", sep="\t", index=False)
    validation = {
        "status": "ok",
        "genes": selected_genes.gene.tolist(),
        "segments": int(len(segments)),
        "selected_centers": int(len(keys)),
        "mutations": int(len(manifest)),
        "replicates_per_center": sorted(manifest.replacement_replicate.unique().tolist()),
        "duplicate_mutation_ids": int(manifest.mutation_id.duplicated().sum()),
        "splice_overlapping_selected_centers": int(candidate_centers.overlaps_splice_site.sum()),
    }
    if validation["duplicate_mutation_ids"] or validation["splice_overlapping_selected_centers"]:
        raise RuntimeError(validation)
    (out / "validation_summary.json").write_text(json.dumps(validation, indent=2) + "\n")
    return validation


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    prepared = root / "prepared"
    analysis = root / "analysis"
    analysis.mkdir(parents=True, exist_ok=True)
    genes = pd.read_csv(prepared / "genes.tsv", sep="\t")
    readouts = pd.read_csv(prepared / "readouts.tsv", sep="\t")
    source_manifest = pd.read_csv(prepared / "mutation_manifest.tsv", sep="\t")
    (
        transcript_ids,
        exon_intervals,
        splice_sites,
        transcript_features,
        transcript_metadata,
    ) = load_gtf_annotations(args.gtf, genes)

    model_data = [read_model_features(root, model, relpath) for model, relpath in MODELS.items()]
    data = pd.concat(model_data, ignore_index=True)
    scores = center_scores(data)
    combined = combined_center_scores(scores)
    annotated = add_genomic_annotation(
        combined,
        genes,
        exon_intervals,
        splice_sites,
        args.splice_buffer_bp,
        transcript_features,
    )
    segments, candidates = cluster_candidates(
        annotated,
        segments_per_gene=args.segments_per_gene,
        max_centers_per_segment=args.max_centers_per_segment,
        candidate_quantile=args.candidate_quantile,
    )
    scores.to_csv(analysis / "model_center_scores.tsv", sep="\t", index=False)
    annotated.to_csv(analysis / "combined_center_scores_annotated.tsv", sep="\t", index=False)
    segments.to_csv(analysis / "candidate_segments.tsv", sep="\t", index=False)
    candidates.to_csv(analysis / "candidate_centers.tsv", sep="\t", index=False)
    pd.DataFrame(
        [{"gene": gene, **transcript_metadata[gene]} for gene in genes.gene]
    ).to_csv(analysis / "representative_transcripts.tsv", sep="\t", index=False)
    validation = prepare_original_manifest(
        root, genes, readouts, source_manifest, segments, candidates
    )
    validation.update(
        {
            "model_center_score_rows": int(len(scores)),
            "combined_annotated_rows": int(len(annotated)),
            "splice_overlapping_center_rows": int(annotated.overlaps_splice_site.sum()),
            "candidate_segments_per_gene": segments.groupby("gene").size().to_dict(),
        }
    )
    (analysis / "validation_summary.json").write_text(json.dumps(validation, indent=2) + "\n")
    print(json.dumps(validation, indent=2))


if __name__ == "__main__":
    main()
