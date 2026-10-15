#!/usr/bin/env python
"""Prepare the nine-gene Mdk-style 10-bp TSS scan for Saijou ISM."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Mapping

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from grelu.interpret.ism.fasta import FastaReference  # noqa: E402
from grelu.interpret.ism.mutations import (  # noqa: E402
    AnchoredScan,
    anchored_scan_centers,
    scan_anchored_strict_shuffles,
)

if __package__:
    from .tools.genomics import (
        TRANSCRIPT_OVERRIDES,
        centered_interval,
        gtf_attr,
        observed_peak,
    )
    from .tools.manifests import (
        ManifestLocus,
        readout_row,
        scan_exclusion_rows,
        scan_manifest_rows,
    )
else:
    from tools.genomics import (
        TRANSCRIPT_OVERRIDES,
        centered_interval,
        gtf_attr,
        observed_peak,
    )
    from tools.manifests import (
        ManifestLocus,
        readout_row,
        scan_exclusion_rows,
        scan_manifest_rows,
    )


DEFAULT_FASTA = "/work/Database/Database_fromDocker/Referencedata_mm10/genome.fa"
DEFAULT_GTF = "/work/Database/Database_fromDocker/Referencedata_mm10/gtf_chrUCSC/chr.gtf"
DEFAULT_HSC_BIGWIG = (
    "/work2/Projects/Project_DL_pillar/Finetune_Borzoi_Saijou_scRNAseq/"
    "bigwig/hsc.CPM.mapq10.bw"
)
DEFAULT_OUT = REPO_ROOT / "experiments/ism/saijou_all_genes_10bp_scan/prepared"
DEFAULT_GENES = "Mdk,Acta2,Col1a1,Timp1,Vegfc,Col1a2,Hgf,Igf1,Ngf"


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
    parser.add_argument(
        "--transcript-authority",
        type=Path,
        help=(
            "Optional TSV that preregisters exact transcript IDs, PDF intervals, "
            "strand and TSS. When supplied, its gene order replaces --genes."
        ),
    )
    parser.add_argument(
        "--authority-pdf-flank-bp",
        type=int,
        default=5_000,
        help="Flank included on each side of transcript intervals in the authority TSV.",
    )
    return parser.parse_args()


def load_genes(
    gtf_path: str | Path,
    requested: list[str],
    *,
    transcript_overrides: Mapping[str, str] | None = None,
) -> pd.DataFrame:
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
    if transcript_overrides is None:
        transcript_overrides = TRANSCRIPT_OVERRIDES
    for gene, transcript_id in transcript_overrides.items():
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


def load_authoritative_genes(
    gtf_path: str | Path,
    authority_path: Path,
    *,
    pdf_flank_bp: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Resolve and audit exact boss-supplied mouse transcripts against mm10."""

    authority = pd.read_csv(authority_path, sep="\t")
    required = {
        "gene",
        "pdf_filename",
        "transcript_name",
        "transcript_id",
        "chrom",
        "strand",
        "pdf_plot_start",
        "pdf_plot_end",
        "analysis_tss_0based",
    }
    missing = sorted(required.difference(authority.columns))
    if missing:
        raise ValueError(f"Transcript authority missing columns: {missing}")
    if authority.gene.duplicated().any() or authority.transcript_id.duplicated().any():
        raise ValueError("Transcript authority genes and transcript IDs must be unique")

    requested = authority.gene.astype(str).tolist()
    genes = load_genes(
        gtf_path,
        requested,
        transcript_overrides={},
    ).set_index("gene")
    columns = [
        "chrom",
        "source",
        "feature",
        "start",
        "end",
        "score",
        "strand",
        "frame",
        "attributes",
    ]
    transcripts = pd.read_csv(
        gtf_path,
        sep="\t",
        comment="#",
        header=None,
        names=columns,
        usecols=range(9),
        low_memory=False,
    )
    transcripts = transcripts.loc[transcripts.feature.eq("transcript")].copy()
    for field in ("transcript_id", "transcript_name", "gene_name"):
        transcripts[field] = transcripts.attributes.astype(str).map(
            lambda value, name=field: gtf_attr(value, name)
        )

    audit_rows: list[dict[str, object]] = []
    for row in authority.itertuples(index=False):
        selected = transcripts.loc[
            transcripts.transcript_id.eq(str(row.transcript_id))
        ]
        if len(selected) != 1:
            raise ValueError(
                f"Expected one mm10 transcript {row.transcript_id}, found {len(selected)}"
            )
        transcript = selected.iloc[0]
        start = int(transcript.start) - 1
        end = int(transcript.end)
        strand = str(transcript.strand)
        tss = start if strand == "+" else end
        checks = {
            "gene_name_match": str(transcript.gene_name) == str(row.gene),
            "transcript_name_match": (
                str(transcript.transcript_name) == str(row.transcript_name)
            ),
            "chrom_match": str(transcript.chrom) == str(row.chrom),
            "strand_match": strand == str(row.strand),
            "pdf_start_match": (
                start == int(row.pdf_plot_start) + int(pdf_flank_bp)
            ),
            "pdf_end_match": end == int(row.pdf_plot_end) - int(pdf_flank_bp),
            "tss_match": tss == int(row.analysis_tss_0based),
        }
        if not all(checks.values()):
            raise ValueError(
                {
                    "gene": row.gene,
                    "transcript_id": row.transcript_id,
                    "checks": checks,
                }
            )
        genes.loc[row.gene, ["chrom", "strand", "start", "end"]] = [
            str(row.chrom),
            strand,
            start,
            end,
        ]
        genes.loc[row.gene, "analysis_tss"] = tss
        genes.loc[row.gene, "tes"] = end if strand == "+" else start
        genes.loc[row.gene, "tss_source"] = (
            f"boss_pdf_transcript:{row.transcript_id}"
        )
        genes.loc[row.gene, "length"] = end - start
        audit_rows.append(
            {
                "species": "Mus musculus",
                "assembly": "mm10",
                "gene": row.gene,
                "pdf_filename": row.pdf_filename,
                "transcript_name": row.transcript_name,
                "transcript_id": row.transcript_id,
                "chrom": row.chrom,
                "strand": strand,
                "transcript_start_0based": start,
                "transcript_end_0based_exclusive": end,
                "analysis_tss_0based": tss,
                "tes_0based": end if strand == "+" else start,
                **checks,
            }
        )
    return (
        genes.loc[requested].reset_index(),
        pd.DataFrame.from_records(audit_rows),
    )


def add_readout(
    rows: list[dict], *, gene: pd.Series, role: str, start: int, end: int, anchor: int
) -> None:
    output_start = int(gene.analysis_tss) - 98_304
    output_end = int(gene.analysis_tss) + 98_304
    if start < output_start or end > output_end or end <= start:
        return
    rows.append(
        readout_row(
            readout_id=f"{gene.gene}__{role}__{start}_{end}",
            gene=gene.gene,
            chrom=gene.chrom,
            start=start,
            end=end,
            anchor=anchor,
            role=role,
        )
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


def _scan_gene(
    *,
    fasta,
    gene,
    centers: list[int],
    args: argparse.Namespace,
) -> tuple[list[dict], list[dict], dict[str, object]]:
    window_tag = (
        "tss1kb"
        if int(args.half_window_bp) == 512
        else f"tss{2 * int(args.half_window_bp)}bp"
    )
    locus_id = (
        f"{gene.gene.lower()}_{window_tag}_span{int(args.span_bp)}_strict_scan"
    )
    scan = AnchoredScan(
        locus_id=locus_id,
        chrom=gene.chrom,
        anchor=int(gene.analysis_tss),
        strand=gene.strand,
    )
    locus = ManifestLocus(
        scan=scan,
        locus_role="tss_10bp_strict_scan",
        gene=gene.gene,
        tes=int(gene.tes),
        control_type="standard_scan",
        source=(
            f"TSS-centered +/-{int(args.half_window_bp)}-bp "
            f"{int(args.span_bp)}-bp strict-shuffle scan"
        ),
    )
    edits, excluded = scan_anchored_strict_shuffles(
        fasta=fasta,
        scan=scan,
        centers=centers,
        span_bp=args.span_bp,
        replicates=args.replicates,
        seed=args.seed,
    )
    mutations = scan_manifest_rows(locus, edits)
    exclusions = scan_exclusion_rows(excluded, gene=gene.gene)
    mutable = sorted({edit.tx_offset for edit in edits})
    locus = {
        "locus_id": locus_id,
        "gene": gene.gene,
        "role": "tss_10bp_strict_scan",
        "tx_span": f"{min(centers):+d}..{max(centers):+d}",
        "span_bp": args.span_bp,
        "replicates": args.replicates,
        "centers": json.dumps(mutable),
        "evidence": (
            f"nine-gene TSS-centered +/-{int(args.half_window_bp)}-bp "
            "discovery scan"
        ),
        "chrom": gene.chrom,
        "strand": gene.strand,
        "analysis_tss": int(gene.analysis_tss),
        "genomic_start": min(edit.edit_start for edit in edits),
        "genomic_end": max(edit.edit_end for edit in edits),
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
    with FastaReference(args.fasta) as fasta:
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
    coordinate_audit = None
    if args.transcript_authority:
        genes, coordinate_audit = load_authoritative_genes(
            args.gtf,
            args.transcript_authority,
            pdf_flank_bp=args.authority_pdf_flank_bp,
        )
        requested = genes.gene.astype(str).tolist()
    else:
        requested = [gene.strip() for gene in args.genes.split(",") if gene.strip()]
        genes = load_genes(args.gtf, requested)
    centers = anchored_scan_centers(
        half_window_bp=args.half_window_bp,
        span_bp=args.span_bp,
        stride_bp=args.stride_bp,
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
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
    if coordinate_audit is not None:
        coordinate_audit.to_csv(
            args.out_dir / "coordinate_authority_audit.tsv",
            sep="\t",
            index=False,
        )
        check_columns = [column for column in coordinate_audit if column.endswith("_match")]
        validation.update(
            {
                "species": "Mus musculus",
                "assembly": "mm10",
                "coordinate_authority": str(args.transcript_authority.resolve()),
                "coordinate_authority_rows": int(len(coordinate_audit)),
                "coordinate_checks_passed": bool(
                    coordinate_audit[check_columns].all().all()
                ),
            }
        )
    validation.update(
        {
            "scan_tx_offset_start": int(min(centers)),
            "scan_tx_offset_end": int(max(centers)),
            "scan_requested_span_bp": int(2 * args.half_window_bp),
            "edit_windows_within_requested_span": bool(
                min(centers) - args.span_bp // 2 == -args.half_window_bp
                and max(centers) + args.span_bp // 2 == args.half_window_bp
            ),
        }
    )
    (args.out_dir / "validation_summary.json").write_text(
        json.dumps(validation, indent=2) + "\n"
    )
    print(json.dumps(validation, indent=2))

if __name__ == "__main__":
    main()
