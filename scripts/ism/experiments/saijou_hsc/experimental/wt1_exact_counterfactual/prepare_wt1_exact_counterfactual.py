#!/usr/bin/env python
"""Prepare exhaustive exact-site counterfactuals for the two Mdk WT1 hits.

This is an experimental, hypothesis-discrimination manifest.  It keeps the
validated Mdk-201 coordinates fixed and enumerates every non-reference
mononucleotide-composition-preserving permutation of each 10-bp site, plus all
single-base substitutions represented in the same 10-bp interval.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from pyfaidx import Fasta


REPO = Path(__file__).resolve().parents[6]
SAIJOU = REPO / "scripts/ism/experiments/saijou_hsc"
for path in (REPO / "src", SAIJOU):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from grelu.interpret.ism.mutations import AnchoredScan  # noqa: E402
from grelu.io.motifs import get_jaspar  # noqa: E402
from tools.genomics import centered_interval  # noqa: E402
from tools.manifests import (  # noqa: E402
    ManifestLocus,
    readout_row,
    standard_readout_row,
    strict_shuffle_mutation_row,
)


DEFAULT_FASTA = "/work/Database/Database_fromDocker/Referencedata_mm10/genome.fa"
DEFAULT_OUT = REPO / "experiments/ism/20260918_1349_wt1_exact_counterfactual/prepared"

GENE = {
    "gene": "Mdk",
    "gene_id": "ENSMUSG00000027239",
    "transcript_name": "Mdk-201",
    "transcript_id": "ENSMUST00000028672",
    "chrom": "chr2",
    "start": 91929804,
    "end": 91932297,
    "strand": "-",
    "analysis_tss": 91932297,
    "tes": 91929804,
    "assembly": "mm10",
}

SITES = (
    {
        "locus_id": "wt1_0001",
        "role": "promoter_wt1_pwm_hit",
        "edit_start": 91933588,
        "edit_end": 91933598,
        "tx_start": -1300,
        "tx_end": -1290,
        "expected_ref": "GAGGGGGAGG",
    },
    {
        "locus_id": "wt1_0002",
        "role": "terminal_exon_wt1_pwm_hit",
        "edit_start": 91930028,
        "edit_end": 91930038,
        "tx_start": 2260,
        "tx_end": 2270,
        "expected_ref": "GTGGGAGAGG",
    },
)

BASES = "ACGT"
BASE_INDEX = {base: i for i, base in enumerate(BASES)}
COMPLEMENT = str.maketrans("ACGT", "TGCA")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fasta", default=DEFAULT_FASTA)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--local-readout-bp", type=int, default=256)
    parser.add_argument("--standard-readout-bp", type=int, default=1024)
    return parser.parse_args()


def unique_permutations(sequence: str) -> list[str]:
    """Enumerate unique permutations without materializing 10! tuples."""

    counts = Counter(sequence)
    output: list[str] = []

    def visit(prefix: list[str]) -> None:
        if len(prefix) == len(sequence):
            output.append("".join(prefix))
            return
        for base in sorted(counts):
            if counts[base] == 0:
                continue
            counts[base] -= 1
            prefix.append(base)
            visit(prefix)
            prefix.pop()
            counts[base] += 1

    visit([])
    return output


def reverse_complement(sequence: str) -> str:
    return sequence.translate(COMPLEMENT)[::-1]


def pwm_window_score(sequence: str, pwm: np.ndarray) -> float:
    values = [math.log2(max(float(pwm[BASE_INDEX[base], i]), 1e-8) / 0.25)
              for i, base in enumerate(sequence)]
    return float(sum(values))


def exact_bidirectional_score(sequence: str, pwm: np.ndarray) -> float:
    if len(sequence) != pwm.shape[1]:
        raise ValueError("Exact PWM score requires matching lengths")
    return max(
        pwm_window_score(sequence, pwm),
        pwm_window_score(reverse_complement(sequence), pwm),
    )


def overlapping_family_score(
    context: str,
    site_start: int,
    site_end: int,
    motifs: dict[str, np.ndarray],
) -> float:
    scores: list[float] = []
    for pwm in motifs.values():
        width = int(pwm.shape[1])
        if width > len(context):
            continue
        for start in range(len(context) - width + 1):
            end = start + width
            if max(0, min(end, site_end) - max(start, site_start)) < min(5, width):
                continue
            window = context[start:end]
            scores.append(exact_bidirectional_score(window, pwm))
    return max(scores) if scores else float("nan")


def dinucleotide_distance(reference: str, alternate: str) -> int:
    ref = Counter(reference[i : i + 2] for i in range(len(reference) - 1))
    alt = Counter(alternate[i : i + 2] for i in range(len(alternate) - 1))
    return int(sum(abs(ref[key] - alt[key]) for key in set(ref) | set(alt)))


def design_variants(reference: str) -> list[tuple[str, str]]:
    rows = [
        ("composition_permutation", sequence)
        for sequence in unique_permutations(reference)
        if sequence != reference
    ]
    for position, ref_base in enumerate(reference):
        for alternate in BASES:
            if alternate == ref_base:
                continue
            sequence = reference[:position] + alternate + reference[position + 1 :]
            rows.append((f"single_base_substitution_pos{position + 1:02d}", sequence))
    if len({sequence for _, sequence in rows}) != len(rows):
        raise ValueError("Duplicate alternate sequences within site")
    return rows


def motif_family(motifs: dict[str, np.ndarray], prefixes: tuple[str, ...]) -> dict[str, np.ndarray]:
    selected = {}
    for motif_id, pwm in motifs.items():
        factors = motif_id.upper().split("_", 1)[-1].replace("::", " ").split()
        if any(any(factor.startswith(prefix) for prefix in prefixes) for factor in factors):
            selected[motif_id] = pwm
    return selected


def prepare(args: argparse.Namespace) -> dict[str, object]:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    motifs = get_jaspar(release="JASPAR2024", tax_group="vertebrates")
    wt1_pwm = np.asarray(motifs["MA1627.2_Wt1"], dtype=float)
    confounders = motif_family(motifs, ("SP", "KLF", "EGR"))
    if not confounders:
        raise ValueError("No SP/KLF/EGR confounder PWMs")

    scan = AnchoredScan(
        locus_id="unused",
        chrom=GENE["chrom"],
        anchor=GENE["analysis_tss"],
        strand=GENE["strand"],
    )
    manifest_rows: list[dict[str, object]] = []
    design_rows: list[dict[str, object]] = []
    locus_rows: list[dict[str, object]] = []

    with Fasta(args.fasta, as_raw=True, sequence_always_upper=True, rebuild=False) as fasta:
        for site in SITES:
            reference = str(
                fasta[GENE["chrom"]][site["edit_start"] : site["edit_end"]]
            ).upper()
            if reference != site["expected_ref"]:
                raise ValueError(
                    f"{site['locus_id']} REF mismatch: {reference} != {site['expected_ref']}"
                )
            context_start = site["edit_start"] - 15
            context_end = site["edit_end"] + 15
            native_context = str(fasta[GENE["chrom"]][context_start:context_end]).upper()
            local_site_start = site["edit_start"] - context_start
            local_site_end = site["edit_end"] - context_start
            native_wt1 = exact_bidirectional_score(reference, wt1_pwm)
            native_confounder = overlapping_family_score(
                native_context, local_site_start, local_site_end, confounders
            )

            locus_scan = AnchoredScan(
                locus_id=site["locus_id"],
                chrom=GENE["chrom"],
                anchor=GENE["analysis_tss"],
                strand=GENE["strand"],
            )
            locus = ManifestLocus(
                scan=locus_scan,
                locus_role=site["role"],
                gene=GENE["gene"],
                tes=GENE["tes"],
                control_type="experimental_exact_site_counterfactual",
                source="JASPAR2024 MA1627.2_Wt1 exact-site exhaustive design",
            )
            variants = design_variants(reference)
            for replicate, (design_class, alternate) in enumerate(variants):
                mutation_id = f"{site['locus_id']}__{design_class}__{replicate:04d}"
                row = strict_shuffle_mutation_row(
                    locus,
                    mutation_id=mutation_id,
                    edit_start=site["edit_start"],
                    edit_end=site["edit_end"],
                    ref_sequence=reference,
                    alt_sequence=alternate,
                    replicate=replicate,
                )
                row["mutation_kind"] = "targeted_exact_site_counterfactual"
                row["replacement_mode"] = design_class
                manifest_rows.append(row)

                alt_context = (
                    native_context[:local_site_start]
                    + alternate
                    + native_context[local_site_end:]
                )
                alt_wt1 = exact_bidirectional_score(alternate, wt1_pwm)
                alt_confounder = overlapping_family_score(
                    alt_context, local_site_start, local_site_end, confounders
                )
                design_rows.append(
                    {
                        "mutation_id": mutation_id,
                        "locus_id": site["locus_id"],
                        "design_class": design_class,
                        "ref_sequence": reference,
                        "alt_sequence": alternate,
                        "changed_bases": sum(a != b for a, b in zip(reference, alternate)),
                        "composition_preserved": Counter(reference) == Counter(alternate),
                        "dinucleotide_l1_distance": dinucleotide_distance(reference, alternate),
                        "wt1_score_native": native_wt1,
                        "wt1_score_alt": alt_wt1,
                        "wt1_score_delta": alt_wt1 - native_wt1,
                        "sp_klf_egr_score_native": native_confounder,
                        "sp_klf_egr_score_alt": alt_confounder,
                        "sp_klf_egr_score_delta": alt_confounder - native_confounder,
                    }
                )

            locus_rows.append(
                {
                    **site,
                    "gene": GENE["gene"],
                    "chrom": GENE["chrom"],
                    "strand": GENE["strand"],
                    "analysis_tss": GENE["analysis_tss"],
                    "tes": GENE["tes"],
                    "local_readout_center": (site["edit_start"] + site["edit_end"]) // 2,
                    "variants": len(variants),
                }
            )

    manifest = pd.DataFrame.from_records(manifest_rows)
    design = pd.DataFrame.from_records(design_rows)
    loci = pd.DataFrame.from_records(locus_rows)
    genes = pd.DataFrame.from_records([{**GENE, "length": GENE["end"] - GENE["start"]}])

    readouts = [
        standard_readout_row(
            gene=GENE["gene"], chrom=GENE["chrom"], role="tss",
            center=GENE["analysis_tss"], width_bp=args.standard_readout_bp
        ),
        standard_readout_row(
            gene=GENE["gene"], chrom=GENE["chrom"], role="tes_3prime",
            center=GENE["tes"], width_bp=args.standard_readout_bp
        ),
        readout_row(
            readout_id="Mdk__gene_body", gene=GENE["gene"], chrom=GENE["chrom"],
            start=GENE["start"], end=GENE["end"], anchor=GENE["analysis_tss"],
            role="gene_body"
        ),
    ]
    for site in SITES:
        center = (site["edit_start"] + site["edit_end"]) // 2
        start, end = centered_interval(center, args.local_readout_bp)
        readouts.append(
            readout_row(
                readout_id=f"{site['locus_id']}__local_{args.local_readout_bp}bp",
                gene=GENE["gene"], locus_id=site["locus_id"], chrom=GENE["chrom"],
                start=start, end=end, anchor=center, role="local_edit"
            )
        )
    readouts = pd.DataFrame.from_records(readouts)

    expected_permutations = {
        site["locus_id"]: math.factorial(10)
        // math.prod(math.factorial(v) for v in Counter(site["expected_ref"]).values())
        - 1
        for site in SITES
    }
    observed_permutations = (
        design[design.design_class.eq("composition_permutation")]
        .groupby("locus_id").size().astype(int).to_dict()
    )
    checks = {
        "unique_mutation_ids": bool(manifest.mutation_id.is_unique),
        "no_noop": bool((manifest.ref_sequence != manifest.alt_sequence).all()),
        "design_manifest_keys_match": set(manifest.mutation_id) == set(design.mutation_id),
        "expected_permutation_counts": observed_permutations == expected_permutations,
        "finite_motif_scores": bool(np.isfinite(design.filter(regex="score").to_numpy()).all()),
    }
    if not all(checks.values()):
        raise ValueError(f"Preparation checks failed: {checks}")

    genes.to_csv(args.out_dir / "genes.tsv", sep="\t", index=False)
    loci.to_csv(args.out_dir / "loci.tsv", sep="\t", index=False)
    readouts.to_csv(args.out_dir / "readouts.tsv", sep="\t", index=False)
    manifest.to_csv(args.out_dir / "mutation_manifest.tsv", sep="\t", index=False)
    design.to_csv(args.out_dir / "variant_design.tsv", sep="\t", index=False)
    validation = {
        "status": "ok",
        "coordinate_contract": {
            "species_assembly": "Mus musculus mm10",
            "gene_transcript": "Mdk-201 ENSMUST00000028672",
            "strand": "-",
            "coordinate_convention": "0-based half-open genomic; transcription-oriented offsets",
            "sites": {
                site["locus_id"]: [site["edit_start"], site["edit_end"]]
                for site in SITES
            },
        },
        "mutations": int(len(manifest)),
        "mutations_by_locus": manifest.groupby("locus_id").size().astype(int).to_dict(),
        "composition_permutations_by_locus": observed_permutations,
        "single_base_substitutions_by_locus": (
            design[design.design_class.str.startswith("single_base_substitution")]
            .groupby("locus_id").size().astype(int).to_dict()
        ),
        "confounder_pwm_count": int(len(confounders)),
        "readouts": int(len(readouts)),
        "checks": checks,
    }
    (args.out_dir / "validation_summary.json").write_text(
        json.dumps(validation, indent=2, sort_keys=True) + "\n"
    )
    return validation


def main() -> None:
    print(json.dumps(prepare(parse_args()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
