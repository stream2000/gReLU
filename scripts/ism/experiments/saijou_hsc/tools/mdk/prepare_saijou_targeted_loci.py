#!/usr/bin/env python
"""Prepare the focused Mdk-control original-vs-finetuned manifest."""

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

if __package__:
    from ..genomics import centered_interval, observed_peak
else:
    SAIJOU_DIR = Path(__file__).resolve().parents[2]
    if str(SAIJOU_DIR) not in sys.path:
        sys.path.insert(0, str(SAIJOU_DIR))
    from tools.genomics import centered_interval, observed_peak

DEFAULT_FASTA = "/work/Database/Database_fromDocker/Referencedata_mm10/genome.fa"
DEFAULT_HSC_BIGWIG = (
    "/work2/Projects/Project_DL_pillar/Finetune_Borzoi_Saijou_scRNAseq/"
    "bigwig/hsc.CPM.mapq10.bw"
)
DEFAULT_OUT = REPO_ROOT / "experiments/ism/saijou_targeted_original_comparison/prepared"


GENES = [
    {
        "gene": "Mdk",
        "gene_id": "ENSMUSG00000027239",
        "chrom": "chr2",
        "start": 91929804,
        "end": 91932297,
        "strand": "-",
        "analysis_tss": 91932297,
        "tes": 91929804,
        "tss_source": "gtf_gene",
    },
    {
        "gene": "Acta2",
        "gene_id": "ENSMUSG00000035783",
        "chrom": "chr19",
        "start": 34241089,
        "end": 34255590,
        "strand": "-",
        "analysis_tss": 34254230,
        "tes": 34241089,
        "tss_source": "gtf_transcript:ENSMUST00000238147",
    },
    {
        "gene": "Col1a1",
        "gene_id": "ENSMUSG00000001506",
        "chrom": "chr11",
        "start": 94936223,
        "end": 94953042,
        "strand": "+",
        "analysis_tss": 94936223,
        "tes": 94953042,
        "tss_source": "gtf_gene",
    },
    {
        "gene": "Timp1",
        "gene_id": "ENSMUSG00000001131",
        "chrom": "chrX",
        "start": 20870165,
        "end": 20874735,
        "strand": "+",
        "analysis_tss": 20870165,
        "tes": 20874735,
        "tss_source": "gtf_gene",
    },
]


LOCUS_TEMPLATES = [
    {
        "locus_id": "mdk_intron_hub_441_507",
        "gene": "Mdk",
        "role": "candidate_intron_regulatory_hub",
        "tx_span": "+441..+507",
        "span_bp": 10,
        "replicates": 3,
        "centers": list(range(441, 508, 2)),
        "evidence": "shared fine-tuned Borzoi/AlphaGenome hotspot",
    },
    {
        "locus_id": "mdk_splice_donor_369",
        "gene": "Mdk",
        "role": "splice_donor_comparator",
        "tx_span": "+364..+373",
        "span_bp": 10,
        "replicates": 20,
        "centers": [369],
        "evidence": "first exon/intron AG|GT donor",
    },
    {
        "locus_id": "acta2_202_promoter_carg",
        "gene": "Acta2",
        "role": "known_promoter_CArG",
        "tx_span": "-66..-57",
        "span_bp": 10,
        "replicates": 20,
        "edit_start": 34254287,
        "edit_end": 34254297,
        "evidence": "literature-backed Acta2-202 proximal CArG",
    },
    {
        "locus_id": "acta2_202_intron1_carg",
        "gene": "Acta2",
        "role": "known_intron1_CArG",
        "tx_span": "+1029..+1038",
        "span_bp": 10,
        "replicates": 20,
        "edit_start": 34253192,
        "edit_end": 34253202,
        "evidence": "literature-backed Acta2 first-intron CArG",
    },
    {
        "locus_id": "col1a1_promoter_inverted_ccaat",
        "gene": "Col1a1",
        "role": "known_promoter_NFY_CCAAT",
        "tx_span": "-73..-69",
        "span_bp": 5,
        "replicates": 20,
        "edit_start": 94936150,
        "edit_end": 94936155,
        "evidence": "conserved inverted CCAAT/NFY positive control",
    },
    {
        "locus_id": "timp1_promoter_ets_like_43",
        "gene": "Timp1",
        "role": "candidate_promoter_ETS_like",
        "tx_span": "+40..+46",
        "span_bp": 7,
        "replicates": 20,
        "edit_start": 20870205,
        "edit_end": 20870212,
        "evidence": "shared fine-tuned hotspot overlapping AGGAAGC",
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fasta", default=DEFAULT_FASTA)
    parser.add_argument("--hsc-bigwig", default=DEFAULT_HSC_BIGWIG)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--local-readout-bp", type=int, default=256)
    parser.add_argument("--standard-readout-bp", type=int, default=1024)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    return parser.parse_args()


def tx_offset(position: int, gene: dict) -> int:
    genomic = int(position) - int(gene["analysis_tss"])
    return genomic if gene["strand"] == "+" else -genomic


def _template_centers(template: dict, gene: dict) -> list[int]:
    centers = template.get("centers")
    if centers is not None:
        return centers
    edit_start = int(template["edit_start"])
    edit_end = int(template["edit_end"])
    return [tx_offset(edit_start + (edit_end - edit_start) // 2, gene)]


def _edit_interval(template: dict, gene: dict, center_offset: int) -> tuple[int, int]:
    if "edit_start" in template:
        return int(template["edit_start"]), int(template["edit_end"])
    genomic_center = (
        int(gene["analysis_tss"]) + int(center_offset)
        if gene["strand"] == "+"
        else int(gene["analysis_tss"]) - int(center_offset)
    )
    edit_start = genomic_center - int(template["span_bp"]) // 2
    return edit_start, edit_start + int(template["span_bp"])


def _mutation_record(
    template: dict,
    gene: dict,
    *,
    edit_start: int,
    edit_end: int,
    ref: str,
    alt: str,
    replicate: int,
) -> dict[str, object]:
    edit_center = edit_start + (edit_end - edit_start) // 2
    return {
        "mutation_id": (
            f"{template['locus_id']}__{edit_start}_{edit_end}"
            f"__shuffle_rep{replicate:02d}"
        ),
        "locus_id": template["locus_id"],
        "locus_role": template["role"],
        "gene": gene["gene"],
        "chrom": gene["chrom"],
        "gene_strand": gene["strand"],
        "gene_tss": gene["analysis_tss"],
        "gene_tes": gene["tes"],
        "edit_start": edit_start,
        "edit_end": edit_end,
        "edit_center_position": edit_center,
        "edit_length_bp": edit_end - edit_start,
        "ref_sequence": ref,
        "alt_sequence": alt,
        "mutation_kind": "strict_mononucleotide_shuffle",
        "replacement_mode": "strict_shuffle",
        "replacement_replicate": replicate,
        "variant_position": edit_center,
        "variant_offset_from_tss_genomic_bp": edit_center - int(gene["analysis_tss"]),
        "variant_offset_from_tss_transcription_bp": tx_offset(edit_center, gene),
        "ref_base": ref,
        "alt_base": alt,
        "control_type": "experimental",
        "source": template["evidence"],
    }


def _prepare_locus(fasta, template: dict, gene: dict, seed: int) -> tuple[dict, list[dict]]:
    intervals = []
    mutations = []
    for center_offset in _template_centers(template, gene):
        edit_start, edit_end = _edit_interval(template, gene, center_offset)
        intervals.append((edit_start, edit_end))
        ref = str(fasta[gene["chrom"]][edit_start:edit_end]).upper()
        if len(ref) != int(template["span_bp"]):
            raise ValueError(f"Reference length mismatch for {template['locus_id']}")
        alternates = strict_unique_shuffles(
            ref,
            n=int(template["replicates"]),
            seed=seed,
            mutation_key=f"{template['locus_id']}:{edit_start}:{edit_end}",
        )
        mutations.extend(
            _mutation_record(
                template,
                gene,
                edit_start=edit_start,
                edit_end=edit_end,
                ref=ref,
                alt=alt,
                replicate=replicate,
            )
            for replicate, alt in enumerate(alternates)
        )
    locus_start = min(start for start, _ in intervals)
    locus_end = max(end for _, end in intervals)
    locus = {
        **template,
        "chrom": gene["chrom"],
        "strand": gene["strand"],
        "analysis_tss": gene["analysis_tss"],
        "genomic_start": locus_start,
        "genomic_end": locus_end,
        "local_readout_center": (locus_start + locus_end) // 2,
    }
    return locus, mutations


def _prepare_mutation_tables(
    fasta_path: str, genes: dict[str, dict], seed: int
) -> tuple[pd.DataFrame, list[dict]]:
    loci = []
    mutations = []
    with Fasta(fasta_path, as_raw=True, sequence_always_upper=True, rebuild=False) as fasta:
        for template in LOCUS_TEMPLATES:
            locus, rows = _prepare_locus(fasta, template, genes[template["gene"]], seed)
            loci.append(locus)
            mutations.extend(rows)
        manifest = pd.DataFrame.from_records(mutations)
        if manifest["mutation_id"].duplicated().any():
            raise ValueError("Duplicate targeted mutation IDs")
        for row in manifest.itertuples(index=False):
            observed = str(
                fasta[row.chrom][int(row.edit_start) : int(row.edit_end)]
            ).upper()
            if observed != row.ref_sequence:
                raise ValueError(f"REF mismatch for {row.mutation_id}")
    return manifest, loci


def _standard_readout(
    gene: dict, role: str, center: int, width: int
) -> dict[str, object]:
    start, end = centered_interval(center, width)
    return {
        "readout_id": f"{gene['gene']}__{role}_{width}bp",
        "gene": gene["gene"],
        "locus_id": "*",
        "chrom": gene["chrom"],
        "start": start,
        "end": end,
        "anchor": center,
        "role": role,
    }


def _gene_readouts(gene: dict, args: argparse.Namespace) -> list[dict]:
    rows = [
        _standard_readout(gene, "tss", gene["analysis_tss"], args.standard_readout_bp),
        _standard_readout(gene, "tes_3prime", gene["tes"], args.standard_readout_bp),
    ]
    peak = observed_peak(gene["chrom"], gene["start"], gene["end"], args.hsc_bigwig)
    if peak is not None:
        row = _standard_readout(
            gene, "hsc_observed_peak", peak, args.standard_readout_bp
        )
        row["role"] = "hsc_observed_peak"
        rows.append(row)
    rows.append(
        {
            "readout_id": f"{gene['gene']}__gene_body",
            "gene": gene["gene"],
            "locus_id": "*",
            "chrom": gene["chrom"],
            "start": gene["start"],
            "end": gene["end"],
            "anchor": gene["analysis_tss"],
            "role": "gene_body",
        }
    )
    return rows


def _build_readouts(
    genes: dict[str, dict], loci: list[dict], args: argparse.Namespace
) -> pd.DataFrame:
    rows = [row for gene in genes.values() for row in _gene_readouts(gene, args)]
    for locus in loci:
        start, end = centered_interval(
            locus["local_readout_center"], args.local_readout_bp
        )
        rows.append(
            {
                "readout_id": f"{locus['locus_id']}__local_{args.local_readout_bp}bp",
                "gene": locus["gene"],
                "locus_id": locus["locus_id"],
                "chrom": locus["chrom"],
                "start": start,
                "end": end,
                "anchor": locus["local_readout_center"],
                "role": "local_edit",
            }
        )
    return pd.DataFrame.from_records(rows)


def _write_tables(
    out_dir: Path,
    *,
    genes: pd.DataFrame,
    loci: pd.DataFrame,
    readouts: pd.DataFrame,
    manifest: pd.DataFrame,
) -> None:
    genes.to_csv(out_dir / "genes.tsv", sep="\t", index=False)
    loci.to_csv(out_dir / "loci.tsv", sep="\t", index=False)
    readouts.to_csv(out_dir / "readouts.tsv", sep="\t", index=False)
    manifest.to_csv(out_dir / "mutation_manifest.tsv", sep="\t", index=False)


def _validation_summary(
    *,
    genes: pd.DataFrame,
    loci: pd.DataFrame,
    readouts: pd.DataFrame,
    manifest: pd.DataFrame,
    seed: int,
) -> dict[str, object]:
    return {
        "status": "ok",
        "genes": int(len(genes)),
        "loci": int(len(loci)),
        "mutations": int(len(manifest)),
        "mutations_by_locus": manifest.groupby("locus_id").size().astype(int).to_dict(),
        "readouts": int(len(readouts)),
        "ref_mismatch_count": 0,
        "no_op_count": 0,
        "composition_preserving_rows": int(len(manifest)),
        "seed": int(seed),
    }


def prepare(args: argparse.Namespace) -> dict[str, object]:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    genes = {row["gene"]: dict(row) for row in GENES}
    manifest, locus_rows = _prepare_mutation_tables(args.fasta, genes, args.seed)
    genes_table = pd.DataFrame.from_records(GENES)
    genes_table["length"] = genes_table["end"] - genes_table["start"]
    loci_table = pd.DataFrame.from_records(locus_rows)
    readouts = _build_readouts(genes, locus_rows, args)
    _write_tables(
        args.out_dir,
        genes=genes_table,
        loci=loci_table,
        readouts=readouts,
        manifest=manifest,
    )
    validation = _validation_summary(
        genes=genes_table,
        loci=loci_table,
        readouts=readouts,
        manifest=manifest,
        seed=args.seed,
    )
    (args.out_dir / "validation_summary.json").write_text(
        json.dumps(validation, indent=2) + "\n"
    )
    return validation


def main() -> None:
    args = parse_args()
    print(json.dumps(prepare(args), indent=2))


if __name__ == "__main__":
    main()
