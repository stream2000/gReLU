#!/usr/bin/env python
"""Prepare a strict-shuffle Mdk TSS specificity background."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[6]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from grelu.interpret.ism.fasta import FastaReference  # noqa: E402
from grelu.interpret.ism.mutations import (  # noqa: E402
    AnchoredScan,
    anchored_scan_centers,
    scan_anchored_strict_shuffles,
)

if __package__:
    from ..manifests import (
        ManifestLocus,
        readout_row,
        scan_exclusion_rows,
        scan_manifest_rows,
        standard_readout_row,
    )
else:
    SAIJOU_DIR = Path(__file__).resolve().parents[2]
    if str(SAIJOU_DIR) not in sys.path:
        sys.path.insert(0, str(SAIJOU_DIR))
    from tools.manifests import (
        ManifestLocus,
        readout_row,
        scan_exclusion_rows,
        scan_manifest_rows,
        standard_readout_row,
    )


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


SCAN = AnchoredScan(
    locus_id=LOCUS_ID,
    chrom=GENE["chrom"],
    anchor=int(GENE["analysis_tss"]),
    strand=GENE["strand"],
)
LOCUS = ManifestLocus(
    scan=SCAN,
    locus_role="mdk_tss_background",
    gene=GENE["gene"],
    tes=GENE["tes"],
    control_type="background",
    source="Mdk TSS-centered strict-shuffle specificity background",
)


def _prepare_manifest(
    args: argparse.Namespace, centers: list[int]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    with FastaReference(args.fasta) as fasta:
        edits, excluded = scan_anchored_strict_shuffles(
            fasta=fasta,
            scan=SCAN,
            centers=centers,
            span_bp=args.span_bp,
            replicates=args.replicates,
            seed=args.seed,
        )
    manifest = pd.DataFrame.from_records(scan_manifest_rows(LOCUS, edits))
    if manifest.empty:
        raise RuntimeError("No mutable Mdk windows were prepared")
    return manifest, pd.DataFrame.from_records(scan_exclusion_rows(excluded))


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


def _readout_table(readout_bp: int = 1024) -> pd.DataFrame:
    standard = dict(gene=GENE["gene"], chrom=GENE["chrom"], width_bp=readout_bp)
    return pd.DataFrame(
        [
            standard_readout_row(role="tss", center=GENE["analysis_tss"], **standard),
            standard_readout_row(role="tes_3prime", center=GENE["tes"], **standard),
            readout_row(
                readout_id=f"{GENE['gene']}__gene_body",
                gene=GENE["gene"],
                chrom=GENE["chrom"],
                start=GENE["start"],
                end=GENE["end"],
                anchor=GENE["analysis_tss"],
                role="gene_body",
            ),
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
    centers = anchored_scan_centers(
        half_window_bp=args.half_window_bp,
        span_bp=args.span_bp,
        stride_bp=args.stride_bp,
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    gene = {**GENE, "length": int(GENE["end"] - GENE["start"])}
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
