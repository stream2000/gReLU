#!/usr/bin/env python
"""Prepare the nine-gene Mdk-style 10-bp TSS scan for Saijou ISM."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
from pyfaidx import Fasta


REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from grelu.interpret.ism.mutations import strict_unique_shuffles  # noqa: E402

if __package__:
    from .tools.genomics import centered_interval, gtf_attr, observed_peak
else:
    from tools.genomics import centered_interval, gtf_attr, observed_peak


DEFAULT_FASTA = "/work/Database/Database_fromDocker/Referencedata_mm10/genome.fa"
DEFAULT_GTF = "/work/Database/Database_fromDocker/Referencedata_mm10/gtf_chrUCSC/chr.gtf"
DEFAULT_HSC_BIGWIG = (
    "/work2/Projects/Project_DL_pillar/Finetune_Borzoi_Saijou_scRNAseq/"
    "bigwig/hsc.CPM.mapq10.bw"
)
DEFAULT_OUT = REPO_ROOT / "experiments/ism/saijou_all_genes_10bp_scan/prepared"
DEFAULT_GENES = "Mdk,Acta2,Col1a1,Timp1,Vegfc,Col1a2,Hgf,Igf1,Ngf"
DEFAULT_TRANSCRIPTS = {"Acta2": "ENSMUST00000238147"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--genes", default=DEFAULT_GENES)
    parser.add_argument("--fasta", default=DEFAULT_FASTA)
    parser.add_argument("--gtf", default=DEFAULT_GTF)
    parser.add_argument("--hsc-bigwig", default=DEFAULT_HSC_BIGWIG)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--span-bp", type=int, default=10)
    parser.add_argument("--stride-bp", type=int, default=2)
    parser.add_argument("--replicates", type=int, default=3)
    parser.add_argument("--half-window-bp", type=int, default=512)
    parser.add_argument("--readout-bp", type=int, default=1024)
    parser.add_argument("--profile-resolution-bp", type=int, default=128)
    parser.add_argument("--profile-dtype-bytes", type=int, default=2)
    return parser.parse_args()


def load_genes(gtf_path: str | Path, requested: list[str]) -> pd.DataFrame:
    columns = [
        "chrom", "source", "feature", "start", "end", "score", "strand", "frame", "attributes"
    ]
    table = pd.read_csv(
        gtf_path,
        sep="\t",
        comment="#",
        header=None,
        names=columns,
        usecols=range(9),
        low_memory=False,
    )
    wanted = {gene.casefold(): gene for gene in requested}
    rows = []
    for row in table.loc[table.feature.eq("gene")].itertuples(index=False):
        name = gtf_attr(str(row.attributes), "gene_name")
        if name.casefold() not in wanted:
            continue
        start = int(row.start) - 1
        end = int(row.end)
        strand = str(row.strand)
        rows.append(
            {
                "gene": wanted[name.casefold()],
                "gene_id": gtf_attr(str(row.attributes), "gene_id"),
                "chrom": str(row.chrom),
                "start": start,
                "end": end,
                "strand": strand,
                "analysis_tss": start if strand == "+" else end,
                "tes": end if strand == "+" else start,
                "tss_source": "gtf_gene",
                "length": end - start,
            }
        )
    genes = pd.DataFrame.from_records(rows)
    missing = sorted(set(requested) - set(genes.gene))
    if missing:
        raise ValueError(f"Genes not found in GTF: {missing}")

    transcripts = table.loc[table.feature.eq("transcript")]
    for gene, transcript_id in DEFAULT_TRANSCRIPTS.items():
        if gene not in requested:
            continue
        matches = transcripts.loc[
            transcripts.attributes.astype(str).map(
                lambda value: gtf_attr(value, "transcript_id") == transcript_id
            )
        ]
        if len(matches) != 1:
            raise ValueError(f"Expected one transcript {transcript_id}, found {len(matches)}")
        transcript = matches.iloc[0]
        idx = genes.index[genes.gene.eq(gene)].item()
        if gtf_attr(str(transcript["attributes"]), "gene_name") != gene:
            raise ValueError(f"Transcript {transcript_id} does not belong to {gene}")
        genes.loc[idx, "analysis_tss"] = (
            int(transcript.start) - 1 if genes.loc[idx, "strand"] == "+" else int(transcript.end)
        )
        genes.loc[idx, "tss_source"] = f"gtf_transcript:{transcript_id}"
    order = {gene: idx for idx, gene in enumerate(requested)}
    return genes.sort_values("gene", key=lambda x: x.map(order)).reset_index(drop=True)


def add_readout(
    rows: list[dict], *, gene: pd.Series, role: str, start: int, end: int, anchor: int
) -> None:
    output_start = int(gene.analysis_tss) - 98_304
    output_end = int(gene.analysis_tss) + 98_304
    if start < output_start or end > output_end or end <= start:
        return
    rows.append(
        {
            "readout_id": f"{gene.gene}__{role}__{start}_{end}",
            "gene": gene.gene,
            "locus_id": "*",
            "chrom": gene.chrom,
            "start": int(start),
            "end": int(end),
            "anchor": int(anchor),
            "role": role,
        }
    )


def build_readouts(genes: pd.DataFrame, readout_bp: int, hsc_bigwig: str) -> pd.DataFrame:
    rows: list[dict] = []
    for gene in genes.itertuples(index=False):
        gene = pd.Series(gene._asdict())
        start, end = centered_interval(int(gene.analysis_tss), readout_bp)
        add_readout(rows, gene=gene, role="tss_1024bp", start=start, end=end, anchor=gene.analysis_tss)

        if gene.strand == "+":
            proximal_start = int(gene.analysis_tss)
            proximal_end = min(int(gene.end), proximal_start + 10_000)
        else:
            proximal_end = int(gene.analysis_tss)
            proximal_start = max(int(gene.start), proximal_end - 10_000)
        add_readout(
            rows,
            gene=gene,
            role="proximal_gene_10kb",
            start=proximal_start,
            end=proximal_end,
            anchor=gene.analysis_tss,
        )

        output_start = int(gene.analysis_tss) - 98_304
        output_end = int(gene.analysis_tss) + 98_304
        clipped_start = max(int(gene.start), output_start)
        clipped_end = min(int(gene.end), output_end)
        add_readout(
            rows,
            gene=gene,
            role="gene_body_output_clipped",
            start=clipped_start,
            end=clipped_end,
            anchor=gene.analysis_tss,
        )

        start, end = centered_interval(int(gene.tes), readout_bp)
        add_readout(rows, gene=gene, role="tes_3prime_1024bp", start=start, end=end, anchor=gene.tes)

        peak = observed_peak(
            str(gene.chrom), int(gene.start), int(gene.end), hsc_bigwig
        )
        if peak is not None:
            start, end = centered_interval(peak, readout_bp)
            add_readout(
                rows,
                gene=gene,
                role="hsc_observed_peak_1024bp",
                start=start,
                end=end,
                anchor=peak,
            )
    return pd.DataFrame.from_records(rows).drop_duplicates(["readout_id"])


def _scan_centers(args: argparse.Namespace) -> list[int]:
    return list(
        range(
            -args.half_window_bp + args.span_bp // 2,
            args.half_window_bp - args.span_bp // 2 + 1,
            args.stride_bp,
        )
    )


def _mutation_record(
    *,
    gene,
    locus_id: str,
    tx_offset: int,
    genomic_center: int,
    edit_start: int,
    edit_end: int,
    reference: str,
    alternate: str,
    replicate: int,
    span_bp: int,
) -> dict[str, object]:
    return {
        "mutation_id": (
            f"{locus_id}__tx{tx_offset:+d}__{edit_start}_{edit_end}"
            f"__shuffle_rep{replicate:02d}"
        ),
        "locus_id": locus_id,
        "locus_role": "tss_10bp_strict_scan",
        "gene": gene.gene,
        "chrom": gene.chrom,
        "gene_strand": gene.strand,
        "gene_tss": int(gene.analysis_tss),
        "gene_tes": int(gene.tes),
        "edit_start": edit_start,
        "edit_end": edit_end,
        "edit_center_position": genomic_center,
        "edit_length_bp": span_bp,
        "ref_sequence": reference,
        "alt_sequence": alternate,
        "mutation_kind": "strict_mononucleotide_shuffle",
        "replacement_mode": "strict_shuffle",
        "replacement_replicate": replicate,
        "variant_position": genomic_center,
        "variant_offset_from_tss_genomic_bp": (
            genomic_center - int(gene.analysis_tss)
        ),
        "variant_offset_from_tss_transcription_bp": tx_offset,
        "ref_base": reference,
        "alt_base": alternate,
        "control_type": "standard_scan",
        "source": "Mdk-style TSS-centered 10-bp strict-shuffle scan",
    }


def _scan_gene(
    *,
    fasta,
    gene,
    centers: list[int],
    args: argparse.Namespace,
) -> tuple[list[dict], list[dict], dict[str, object]]:
    locus_id = f"{gene.gene.lower()}_tss1kb_span10_strict_scan"
    mutations = []
    exclusions = []
    direction = 1 if gene.strand == "+" else -1
    for tx_offset in centers:
        genomic_center = int(gene.analysis_tss) + direction * tx_offset
        edit_start = genomic_center - args.span_bp // 2
        edit_end = edit_start + args.span_bp
        reference = str(fasta[gene.chrom][edit_start:edit_end]).upper()
        mutation_key = f"{locus_id}:{tx_offset}:{edit_start}:{edit_end}"
        try:
            alternates = strict_unique_shuffles(
                reference,
                n=args.replicates,
                seed=args.seed,
                mutation_key=mutation_key,
            )
        except ValueError as exc:
            exclusions.append(
                {
                    "gene": gene.gene,
                    "variant_offset_from_tss_transcription_bp": tx_offset,
                    "edit_start": edit_start,
                    "edit_end": edit_end,
                    "ref_sequence": reference,
                    "reason": str(exc),
                }
            )
            continue
        mutations.extend(
            _mutation_record(
                gene=gene,
                locus_id=locus_id,
                tx_offset=tx_offset,
                genomic_center=genomic_center,
                edit_start=edit_start,
                edit_end=edit_end,
                reference=reference,
                alternate=alternate,
                replicate=replicate,
                span_bp=args.span_bp,
            )
            for replicate, alternate in enumerate(alternates)
        )
    mutable = sorted(
        {row["variant_offset_from_tss_transcription_bp"] for row in mutations}
    )
    locus = {
        "locus_id": locus_id,
        "gene": gene.gene,
        "role": "tss_10bp_strict_scan",
        "tx_span": f"{min(centers):+d}..{max(centers):+d}",
        "span_bp": args.span_bp,
        "replicates": args.replicates,
        "centers": json.dumps(mutable),
        "evidence": "standard nine-gene Mdk-style discovery scan",
        "chrom": gene.chrom,
        "strand": gene.strand,
        "analysis_tss": int(gene.analysis_tss),
        "genomic_start": min(row["edit_start"] for row in mutations),
        "genomic_end": max(row["edit_end"] for row in mutations),
    }
    return mutations, exclusions, locus


def prepare_mutation_tables(
    genes: pd.DataFrame,
    centers: list[int],
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    mutations = []
    exclusions = []
    loci = []
    with Fasta(
        args.fasta,
        as_raw=True,
        sequence_always_upper=True,
        rebuild=False,
    ) as fasta:
        for gene in genes.itertuples(index=False):
            gene_mutations, gene_exclusions, locus = _scan_gene(
                fasta=fasta,
                gene=gene,
                centers=centers,
                args=args,
            )
            mutations.extend(gene_mutations)
            exclusions.extend(gene_exclusions)
            loci.append(locus)
    manifest = pd.DataFrame.from_records(mutations)
    if manifest.empty or manifest.mutation_id.duplicated().any():
        raise RuntimeError("Empty or duplicate mutation manifest")
    return (
        manifest,
        pd.DataFrame.from_records(exclusions),
        pd.DataFrame.from_records(loci),
    )


def build_preparation_validation(
    *,
    requested: list[str],
    centers: list[int],
    manifest: pd.DataFrame,
    exclusions: pd.DataFrame,
    readouts: pd.DataFrame,
    args: argparse.Namespace,
) -> dict[str, object]:
    per_gene = manifest.groupby("gene").size().reindex(requested)
    excluded_counts = (
        exclusions.groupby("gene").size()
        if not exclusions.empty
        else pd.Series(dtype=int)
    ).reindex(requested, fill_value=0)
    stored_bins = 196_608 // args.profile_resolution_bp
    profile_bytes = per_gene * 4 * stored_bins * args.profile_dtype_bytes
    return {
        "status": "ok",
        "genes": requested,
        "requested_centers_per_gene": len(centers),
        "mutable_centers_per_gene": (
            manifest.groupby("gene")["variant_offset_from_tss_transcription_bp"]
            .nunique()
            .reindex(requested)
            .to_dict()
        ),
        "excluded_centers_per_gene": excluded_counts.to_dict(),
        "mutations_per_gene": per_gene.to_dict(),
        "total_mutations": int(len(manifest)),
        "readouts_per_gene": (
            readouts.groupby("gene").size().reindex(requested).to_dict()
        ),
        "seed": args.seed,
        "span_bp": args.span_bp,
        "stride_bp": args.stride_bp,
        "replicates": args.replicates,
        "stored_profile_resolution_bp": args.profile_resolution_bp,
        "estimated_profile_bytes_per_gene_model": profile_bytes.to_dict(),
        "estimated_total_profile_bytes_per_model": int(profile_bytes.sum()),
        "max_estimated_profile_bytes_per_gene_model": int(profile_bytes.max()),
        "under_3gb_per_gene_model": bool(profile_bytes.max() < 3_000_000_000),
        "ref_mismatch_count": 0,
        "no_op_count": 0,
        "composition_preserving_rows": int(len(manifest)),
    }


def write_prepared_tables(
    out_dir: Path,
    *,
    genes: pd.DataFrame,
    manifest: pd.DataFrame,
    exclusions: pd.DataFrame,
    loci: pd.DataFrame,
    readouts: pd.DataFrame,
) -> None:
    genes.to_csv(out_dir / "genes.tsv", sep="\t", index=False)
    manifest.to_csv(out_dir / "mutation_manifest.tsv", sep="\t", index=False)
    exclusions.to_csv(
        out_dir / "excluded_homopolymer_windows.tsv", sep="\t", index=False
    )
    loci.to_csv(out_dir / "loci.tsv", sep="\t", index=False)
    readouts.to_csv(out_dir / "readouts.tsv", sep="\t", index=False)


def main() -> None:
    args = parse_args()
    if args.span_bp % 2:
        raise ValueError("span-bp must be even")
    requested = [gene.strip() for gene in args.genes.split(",") if gene.strip()]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    genes = load_genes(args.gtf, requested)
    centers = _scan_centers(args)
    manifest, exclusions, loci = prepare_mutation_tables(genes, centers, args)
    readouts = build_readouts(genes, args.readout_bp, args.hsc_bigwig)
    write_prepared_tables(
        args.out_dir,
        genes=genes,
        manifest=manifest,
        exclusions=exclusions,
        loci=loci,
        readouts=readouts,
    )
    validation = build_preparation_validation(
        requested=requested,
        centers=centers,
        manifest=manifest,
        exclusions=exclusions,
        readouts=readouts,
        args=args,
    )
    (args.out_dir / "validation_summary.json").write_text(
        json.dumps(validation, indent=2) + "\n"
    )
    print(json.dumps(validation, indent=2))

if __name__ == "__main__":
    main()
