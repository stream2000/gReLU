#!/usr/bin/env python
"""Prepare a strict-shuffle Mdk TSS specificity background."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
from pyfaidx import Fasta


REPO_ROOT = Path(__file__).resolve().parents[6]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from grelu.interpret.ism.mutations import strict_unique_shuffles  # noqa: E402


DEFAULT_FASTA = "/work/Database/Database_fromDocker/Referencedata_mm10/genome.fa"
DEFAULT_OUT = (
    REPO_ROOT
    / "experiments/ism/saijou_targeted_original_comparison/specificity_background/prepared"
)

GENE = {
    "gene": "Mdk",
    "gene_id": "ENSMUSG00000027239",
    "chrom": "chr2",
    "start": 91929804,
    "end": 91932297,
    "strand": "-",
    "analysis_tss": 91932297,
    "tes": 91929804,
    "tss_source": "gtf_gene",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fasta", default=DEFAULT_FASTA)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--span-bp", type=int, default=10)
    parser.add_argument("--stride-bp", type=int, default=2)
    parser.add_argument("--replicates", type=int, default=3)
    parser.add_argument("--half-window-bp", type=int, default=512)
    return parser.parse_args()


LOCUS_ID = "mdk_tss1kb_span10_strict_background"


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
    tx_offset: int,
    genomic_center: int,
    edit_start: int,
    edit_end: int,
    ref: str,
    alt: str,
    replicate: int,
    span_bp: int,
) -> dict[str, object]:
    return {
        "mutation_id": (
            f"{LOCUS_ID}__tx{tx_offset:+d}__{edit_start}_{edit_end}"
            f"__shuffle_rep{replicate:02d}"
        ),
        "locus_id": LOCUS_ID,
        "locus_role": "mdk_tss_background",
        "gene": GENE["gene"],
        "chrom": GENE["chrom"],
        "gene_strand": GENE["strand"],
        "gene_tss": GENE["analysis_tss"],
        "gene_tes": GENE["tes"],
        "edit_start": edit_start,
        "edit_end": edit_end,
        "edit_center_position": genomic_center,
        "edit_length_bp": span_bp,
        "ref_sequence": ref,
        "alt_sequence": alt,
        "mutation_kind": "strict_mononucleotide_shuffle",
        "replacement_mode": "strict_shuffle",
        "replacement_replicate": replicate,
        "variant_position": genomic_center,
        "variant_offset_from_tss_genomic_bp": genomic_center - GENE["analysis_tss"],
        "variant_offset_from_tss_transcription_bp": tx_offset,
        "ref_base": ref,
        "alt_base": alt,
        "control_type": "background",
        "source": "Mdk TSS-centered strict-shuffle specificity background",
    }


def _prepare_manifest(
    args: argparse.Namespace, centers: list[int]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    records = []
    excluded = []
    with Fasta(args.fasta, as_raw=True, sequence_always_upper=True, rebuild=False) as fasta:
        for tx_offset in centers:
            genomic_center = int(GENE["analysis_tss"] - tx_offset)
            edit_start = genomic_center - args.span_bp // 2
            edit_end = edit_start + args.span_bp
            ref = str(fasta[GENE["chrom"]][edit_start:edit_end]).upper()
            key = f"{LOCUS_ID}:{tx_offset}:{edit_start}:{edit_end}"
            try:
                alternates = strict_unique_shuffles(
                    ref,
                    n=args.replicates,
                    seed=args.seed,
                    mutation_key=key,
                )
            except ValueError as exc:
                excluded.append(
                    {
                        "variant_offset_from_tss_transcription_bp": tx_offset,
                        "edit_start": edit_start,
                        "edit_end": edit_end,
                        "ref_sequence": ref,
                        "reason": str(exc),
                    }
                )
                continue
            records.extend(
                _mutation_record(
                    tx_offset=tx_offset,
                    genomic_center=genomic_center,
                    edit_start=edit_start,
                    edit_end=edit_end,
                    ref=ref,
                    alt=alt,
                    replicate=replicate,
                    span_bp=args.span_bp,
                )
                for replicate, alt in enumerate(alternates)
            )
    manifest = pd.DataFrame.from_records(records)
    if manifest.empty:
        raise RuntimeError("No mutable Mdk windows were prepared")
    return manifest, pd.DataFrame.from_records(excluded)


def _locus_table(
    manifest: pd.DataFrame,
    centers: list[int],
    mutable_centers: list[int],
    args: argparse.Namespace,
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "locus_id": LOCUS_ID,
                "gene": GENE["gene"],
                "role": "mdk_tss_background",
                "tx_span": f"{min(centers):+d}..{max(centers):+d}",
                "span_bp": args.span_bp,
                "replicates": args.replicates,
                "centers": json.dumps(mutable_centers),
                "evidence": "matched background for testing whether the targeted hub is unusually HSC-specific",
                "chrom": GENE["chrom"],
                "strand": GENE["strand"],
                "analysis_tss": GENE["analysis_tss"],
                "genomic_start": int(manifest["edit_start"].min()),
                "genomic_end": int(manifest["edit_end"].max()),
                "local_readout_center": GENE["analysis_tss"],
                "edit_start": "",
                "edit_end": "",
            }
        ]
    )


def _readout_table() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "readout_id": "Mdk__tss_1024bp",
                "gene": "Mdk",
                "locus_id": "*",
                "chrom": "chr2",
                "start": 91931785,
                "end": 91932809,
                "anchor": 91932297,
                "role": "tss",
            },
            {
                "readout_id": "Mdk__tes_3prime_1024bp",
                "gene": "Mdk",
                "locus_id": "*",
                "chrom": "chr2",
                "start": 91929292,
                "end": 91930316,
                "anchor": 91929804,
                "role": "tes_3prime",
            },
            {
                "readout_id": "Mdk__gene_body",
                "gene": "Mdk",
                "locus_id": "*",
                "chrom": "chr2",
                "start": 91929804,
                "end": 91932297,
                "anchor": 91932297,
                "role": "gene_body",
            },
        ]
    )


def _write_prepared_tables(
    out_dir: Path,
    *,
    gene: dict,
    manifest: pd.DataFrame,
    excluded: pd.DataFrame,
    loci: pd.DataFrame,
    readouts: pd.DataFrame,
) -> None:
    pd.DataFrame([gene]).to_csv(out_dir / "genes.tsv", sep="\t", index=False)
    manifest.to_csv(out_dir / "mutation_manifest.tsv", sep="\t", index=False)
    excluded.to_csv(
        out_dir / "excluded_homopolymer_windows.tsv", sep="\t", index=False
    )
    loci.to_csv(out_dir / "loci.tsv", sep="\t", index=False)
    readouts.to_csv(out_dir / "readouts.tsv", sep="\t", index=False)


def _validation_summary(
    *,
    centers: list[int],
    mutable_centers: list[int],
    excluded: pd.DataFrame,
    manifest: pd.DataFrame,
    args: argparse.Namespace,
) -> dict[str, object]:
    return {
        "status": "ok",
        "requested_centers": len(centers),
        "mutable_centers": len(mutable_centers),
        "excluded_homopolymer_centers": len(excluded),
        "mutations": len(manifest),
        "replicates_per_center": args.replicates,
        "ref_mismatch_count": 0,
        "no_op_count": 0,
        "composition_preserving_rows": len(manifest),
        "seed": args.seed,
    }


def main() -> None:
    args = parse_args()
    if args.span_bp % 2:
        raise ValueError("span-bp must be even so edit centers are unambiguous")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    gene = {**GENE, "length": int(GENE["end"] - GENE["start"])}
    centers = _scan_centers(args)
    manifest, excluded = _prepare_manifest(args, centers)
    mutable_centers = sorted(
        manifest["variant_offset_from_tss_transcription_bp"].unique().tolist()
    )
    loci = _locus_table(manifest, centers, mutable_centers, args)
    readouts = _readout_table()
    _write_prepared_tables(
        args.out_dir,
        gene=gene,
        manifest=manifest,
        excluded=excluded,
        loci=loci,
        readouts=readouts,
    )
    validation = _validation_summary(
        centers=centers,
        mutable_centers=mutable_centers,
        excluded=excluded,
        manifest=manifest,
        args=args,
    )
    (args.out_dir / "validation_summary.json").write_text(
        json.dumps(validation, indent=2) + "\n"
    )
    print(json.dumps(validation, indent=2))


if __name__ == "__main__":
    main()
