#!/usr/bin/env python
"""Create an MVP CTCF motif perturbation input table near MREG.

The output schema is compatible with ``scripts/ism/run_tf_context.py``. By
default the script finds the hg38 MREG gene interval from the local GTF, queries
UCSC ENCODE4 CTCF rPeak clusters in a flanked region, rescans the peak sequence
with the CTCF PWM, and writes one strand-aware max-IC motif-disrupting SNV.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd
import requests
from pyfaidx import Fasta

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from scripts.ism.make_matched_boundary_pilot import (  # noqa: E402
    AG_INPUT_LEN,
    choose_max_ic_variant,
    choose_nearby_control,
    get_ctcf_peaks,
)
from scripts.ism.prepare_ctcf_sites import (  # noqa: E402
    CANONICAL_HG38,
    DEFAULT_FASTA,
    DEFAULT_MEME,
    DEFAULT_UCSC_API,
    best_motif_hit,
    load_meme_pwm,
    pwm_to_pssm,
)

DEFAULT_GTF = Path("/home/fqijun/.local/share/genomes/hg38/hg38.annotation.gtf")


def parse_region(value: str) -> tuple[str, int, int]:
    """Parse ``chr:start-end`` into 0-based half-open coordinates."""

    match = re.fullmatch(r"([^:]+):([0-9,]+)-([0-9,]+)", value.strip())
    if not match:
        raise ValueError(f"Region must look like chr2:215939308-216034096, got {value!r}")
    chrom = match.group(1)
    start = int(match.group(2).replace(",", ""))
    end = int(match.group(3).replace(",", ""))
    if end <= start:
        raise ValueError(f"Region end must be greater than start: {value}")
    return chrom, start, end


def load_gene_region_from_gtf(gtf_path: str | Path, gene_name: str) -> tuple[str, int, int, str]:
    """Return the union of transcript rows for ``gene_name`` from a GTF."""

    gene_pat = re.compile(r'gene_name "([^"]+)"')
    rows: list[tuple[str, int, int, str]] = []
    with Path(gtf_path).open() as handle:
        for line in handle:
            if not line or line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 9 or fields[2] != "transcript":
                continue
            match = gene_pat.search(fields[8])
            if match is None or match.group(1) != gene_name:
                continue
            chrom = fields[0]
            # GTF is 1-based inclusive; convert to BED-style 0-based half-open.
            start = int(fields[3]) - 1
            end = int(fields[4])
            strand = fields[6]
            rows.append((chrom, start, end, strand))

    if not rows:
        raise ValueError(f"Could not find transcript rows for gene_name={gene_name!r} in {gtf_path}")
    chroms = {row[0] for row in rows}
    if len(chroms) != 1:
        raise ValueError(f"Gene {gene_name} spans multiple chromosomes in {gtf_path}: {sorted(chroms)}")
    strands = {row[3] for row in rows}
    chrom = rows[0][0]
    return chrom, min(row[1] for row in rows), max(row[2] for row in rows), ",".join(sorted(strands))


def _candidate_rows(
    peaks: list[dict],
    fasta: Fasta,
    pwm: list[list[float]],
    pssm: list[list[float]],
    gene_name: str,
    gene_chrom: str,
    gene_start: int,
    gene_end: int,
    gene_strand: str,
    flank_bp: int,
    min_motif_score: float | None,
) -> list[dict]:
    records: list[dict] = []
    half = AG_INPUT_LEN // 2
    for peak in peaks:
        chrom = peak["chrom"]
        if chrom not in fasta:
            continue
        seq = str(fasta[chrom][peak["start"] : peak["end"]]).upper()
        if len(seq) != peak["end"] - peak["start"]:
            continue
        rel_start, rel_end, strand, motif_score = best_motif_hit(seq, pssm)
        if min_motif_score is not None and motif_score < min_motif_score:
            continue
        motif_start = peak["start"] + rel_start
        motif_end = peak["start"] + rel_end
        motif_center = (motif_start + motif_end) // 2
        if motif_center - half < 0 or motif_center + half > len(fasta[chrom]):
            continue
        matched_seq = seq[rel_start:rel_end]
        try:
            variant = choose_max_ic_variant(matched_seq, pwm, motif_start, strand)
        except ValueError:
            continue
        distance_to_gene = 0
        if motif_center < gene_start:
            distance_to_gene = gene_start - motif_center
        elif motif_center > gene_end:
            distance_to_gene = motif_center - gene_end
        records.append(
            {
                "site_id": "",
                "paired_site_id": "",
                "chrom": chrom,
                "start": peak["start"],
                "end": peak["end"],
                "gene": gene_name,
                "gene_chrom": gene_chrom,
                "gene_start": gene_start,
                "gene_end": gene_end,
                "gene_strand": gene_strand,
                "gene_flank_bp": flank_bp,
                "paper_context": "MREG-region CTCF motif perturbation MVP",
                "dic_class": "MREG_example",
                "motif_start": motif_start,
                "motif_end": motif_end,
                "matched_seq": matched_seq,
                "paper_note": "Selected from UCSC ENCODE4 CTCF rPeak clusters around MREG",
                "control_type": "ctcf_motif_max_ic_disruption",
                "mutation_strategy": "ctcf_pwm_max_ic_to_min_prob",
                "variant_position": variant[0],
                "variant_ref": variant[1],
                "variant_alt": variant[2],
                "variant_start": variant[0] - 1,
                "variant_end": variant[0],
                "variant_ref_seq": variant[1],
                "variant_alt_seq": variant[2],
                "variant_offset": variant[3],
                "variant_ref_pwm_prob": variant[4],
                "variant_alt_pwm_prob": variant[5],
                "variant_delta_pwm_prob": variant[6],
                "variant_pwm_offset": variant[7],
                "variant_oriented_ref": variant[8],
                "variant_oriented_alt": variant[9],
                "peak_score": peak["peak_score"],
                "motif_score": motif_score,
                "motif_center": motif_center,
                "distance_to_gene_bp": distance_to_gene,
                "strand": strand,
                "rpeak_ubiquity": peak.get("rpeak_ubiquity", ""),
                "cCRE": peak.get("cCRE", ""),
                "peak_name": peak.get("name", ""),
            }
        )
    return records


def build_mreg_example(
    fasta_path: str | Path,
    gtf_path: str | Path,
    output: str | Path,
    gene_name: str = "MREG",
    region: str | None = None,
    flank_bp: int = 500_000,
    n_sites: int = 1,
    min_motif_score: float | None = None,
    include_control: bool = False,
    meme_path: str | Path = DEFAULT_MEME,
    api_url: str = DEFAULT_UCSC_API,
) -> pd.DataFrame:
    fasta = Fasta(str(fasta_path), as_raw=True, sequence_always_upper=True)
    pwm = load_meme_pwm(meme_path, motif_name="CTCF")
    pssm = pwm_to_pssm(pwm)

    if region:
        gene_chrom, gene_start, gene_end = parse_region(region)
        gene_strand = "."
    else:
        gene_chrom, gene_start, gene_end, gene_strand = load_gene_region_from_gtf(gtf_path, gene_name)
    if gene_chrom not in CANONICAL_HG38:
        raise ValueError(f"Unsupported chromosome for hg38 canonical region: {gene_chrom}")

    query_start = max(0, gene_start - flank_bp)
    query_end = min(CANONICAL_HG38[gene_chrom], gene_end + flank_bp)
    session = requests.Session()
    peaks = get_ctcf_peaks(session, gene_chrom, query_start, query_end, api_url=api_url)
    if not peaks:
        raise ValueError(f"No UCSC ENCODE4 CTCF peaks found in {gene_chrom}:{query_start}-{query_end}")

    records = _candidate_rows(
        peaks,
        fasta=fasta,
        pwm=pwm,
        pssm=pssm,
        gene_name=gene_name,
        gene_chrom=gene_chrom,
        gene_start=gene_start,
        gene_end=gene_end,
        gene_strand=gene_strand,
        flank_bp=flank_bp,
        min_motif_score=min_motif_score,
    )
    if not records:
        raise ValueError(
            "CTCF peaks were found, but none produced a valid motif-disrupting variant "
            f"in {gene_chrom}:{query_start}-{query_end}"
        )

    candidates = pd.DataFrame.from_records(records)
    candidates = candidates.sort_values(
        ["distance_to_gene_bp", "peak_score", "motif_score"],
        ascending=[True, False, False],
    ).reset_index(drop=True)

    selected_rows: list[dict] = []
    for idx, row in candidates.head(max(1, n_sites)).iterrows():
        base_id = f"{gene_name.lower()}_ctcf_example_{idx + 1:02d}"
        rec = row.to_dict()
        rec["site_id"] = f"{base_id}_motif_snv"
        rec["paired_site_id"] = base_id
        selected_rows.append(rec)
        if include_control:
            try:
                control_pos, control_ref, control_alt = choose_nearby_control(
                    fasta,
                    str(row["chrom"]),
                    int(row["start"]),
                    int(row["end"]),
                    int(row["motif_start"]),
                    int(row["motif_end"]),
                )
            except ValueError:
                continue
            control = rec.copy()
            control.update(
                {
                    "site_id": f"{base_id}_nearby_control",
                    "control_type": "same_peak_non_motif_nearby_base",
                    "mutation_strategy": "same_peak_non_motif_control",
                    "variant_position": control_pos,
                    "variant_ref": control_ref,
                    "variant_alt": control_alt,
                    "variant_start": control_pos - 1,
                    "variant_end": control_pos,
                    "variant_ref_seq": control_ref,
                    "variant_alt_seq": control_alt,
                    "variant_offset": control_pos - int(row["motif_start"]) - 1,
                    "variant_ref_pwm_prob": pd.NA,
                    "variant_alt_pwm_prob": pd.NA,
                    "variant_delta_pwm_prob": pd.NA,
                    "variant_pwm_offset": pd.NA,
                    "variant_oriented_ref": pd.NA,
                    "variant_oriented_alt": pd.NA,
                }
            )
            selected_rows.append(control)

    out = pd.DataFrame.from_records(selected_rows)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output, sep="\t", index=False)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fasta", default=str(DEFAULT_FASTA))
    parser.add_argument("--gtf", default=str(DEFAULT_GTF))
    parser.add_argument("--gene_name", default="MREG")
    parser.add_argument("--region", default=None, help="Override gene lookup with chr:start-end")
    parser.add_argument("--flank_bp", type=int, default=500_000)
    parser.add_argument("--n_sites", type=int, default=1)
    parser.add_argument("--min_motif_score", type=float, default=None)
    parser.add_argument("--include_control", action="store_true")
    parser.add_argument("--meme", default=str(DEFAULT_MEME))
    parser.add_argument("--api_url", default=DEFAULT_UCSC_API)
    parser.add_argument(
        "--output",
        default="agent-doc/ism_context/mreg_ctcf_mvp/input_sites.tsv",
    )
    args = parser.parse_args()

    out = build_mreg_example(
        fasta_path=args.fasta,
        gtf_path=args.gtf,
        output=args.output,
        gene_name=args.gene_name,
        region=args.region,
        flank_bp=args.flank_bp,
        n_sites=args.n_sites,
        min_motif_score=args.min_motif_score,
        include_control=args.include_control,
        meme_path=args.meme,
        api_url=args.api_url,
    )
    print(f"Wrote {args.output} ({len(out)} rows)")
    print(
        out[
            [
                "site_id",
                "chrom",
                "start",
                "end",
                "motif_start",
                "motif_end",
                "matched_seq",
                "strand",
                "variant_position",
                "variant_ref",
                "variant_alt",
                "peak_score",
                "motif_score",
                "distance_to_gene_bp",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
