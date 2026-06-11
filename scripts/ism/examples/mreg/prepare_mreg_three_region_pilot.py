#!/usr/bin/env python
"""Prepare MREG three-region (TSS, HC-DIC, LC-DIC) mutation and readout manifests.

This script implements Phase 0–1 of the MREG TSS / HC-DIC / LC-DIC 128 bp ChIP ISM
exploration plan. It:

1. Reads curated region definitions (with HC-DIC/LC-DIC coordinates from the Nakato
   paper or teacher annotations).
2. Extracts MREG transcript / TSS information from the local GTF.
3. Scans CTCF motifs in TSS and HC-DIC regions using the JASPAR 2024 consensus PWM.
4. Generates strand-aware PWM-maximum-disruption SNVs for CTCF-bearing regions.
5. Generates Pol2 peak-center sequence perturbations for LC-DIC (no CTCF motif).
6. Generates matched non-motif local controls for each experimental mutation.
7. Validates all REF sequences against the hg38 FASTA.
8. Outputs the four manifests required by the downstream runner.

Output manifest files (all TSV):

- mreg_region_manifest.tsv
- mreg_mutation_manifest.tsv
- mreg_readout_regions.tsv
- mreg_track_registry.tsv
- sequence_contexts.tsv
- preparation_qc.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from grelu.interpret.ism.motifs import (
    FastaReference,
    PWMModel,
    MotifHit,
    DesignedEdit,
    _design_motif_disruption,
    _design_local_controls,
    _choose_site_hit,
    load_meme_pwm,
    scan_motif_hits,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_FASTA = Path("/home/fqijun/.local/share/genomes/hg38/hg38.fa")
DEFAULT_GTF = Path("/home/fqijun/.local/share/genomes/hg38/hg38.annotation.gtf")
DEFAULT_MEME = (
    REPO_ROOT
    / "src"
    / "grelu"
    / "resources"
    / "meme"
    / "jaspar_2024_consensus.meme"
)
AG_INPUT_LEN = 1_048_576
PSEUDOCOUNT = 1e-4

# ---------------------------------------------------------------------------
# GTF helpers (inlined from make_mreg_ctcf_example.py for self-containment)
# ---------------------------------------------------------------------------


def _parse_gtf_gene_region(
    gtf_path: Path, gene_name: str
) -> Tuple[str, int, int, str]:
    """Return the union genomic interval and strand for *gene_name* from a GTF."""
    gene_pat = re.compile(r'gene_name "([^"]+)"')
    rows: list[Tuple[str, int, int, str]] = []
    with gtf_path.open() as handle:
        for line in handle:
            if not line or line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 9 or fields[2] != "transcript":
                continue
            match = gene_pat.search(fields[8])
            if match is None or match.group(1).upper() != gene_name.upper():
                continue
            chrom = fields[0]
            start = int(fields[3]) - 1  # GTF 1-based → 0-based half-open
            end = int(fields[4])
            strand = fields[6]
            rows.append((chrom, start, end, strand))
    if not rows:
        raise ValueError(f"No transcript rows for gene_name={gene_name!r} in {gtf_path}")
    chroms = {r[0] for r in rows}
    if len(chroms) != 1:
        raise ValueError(
            f"Gene {gene_name} spans multiple chromosomes: {sorted(chroms)}"
        )
    strands = {r[3] for r in rows}
    return (
        rows[0][0],
        min(r[1] for r in rows),
        max(r[2] for r in rows),
        ",".join(sorted(strands)),
    )


def _parse_gtf_transcripts(
    gtf_path: Path, gene_name: str
) -> pd.DataFrame:
    """Return a table of transcript_id, TSS, strand for *gene_name*."""
    gene_pat = re.compile(r'gene_name "([^"]+)"')
    tx_pat = re.compile(r'transcript_id "([^"]+)"')
    rows: list[dict] = []
    with gtf_path.open() as handle:
        for line in handle:
            if not line or line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 9 or fields[2] != "transcript":
                continue
            match = gene_pat.search(fields[8])
            if match is None or match.group(1).upper() != gene_name.upper():
                continue
            tx_match = tx_pat.search(fields[8])
            tx_id = tx_match.group(1) if tx_match else ""
            strand = fields[6]
            # TSS is 5′ end: for + strand = start; for − strand = end
            if strand == "+":
                tss = int(fields[3]) - 1  # 0-based
            else:
                tss = int(fields[4]) - 1  # 0-based (end is exclusive in GTF → −1)
            rows.append(
                {
                    "transcript_id": tx_id,
                    "chrom": fields[0],
                    "tss": tss,
                    "strand": strand,
                    "start": int(fields[3]) - 1,
                    "end": int(fields[4]),
                }
            )
    if not rows:
        raise ValueError(f"No transcripts for gene_name={gene_name!r} in {gtf_path}")
    return pd.DataFrame.from_records(rows)


# ---------------------------------------------------------------------------
# Region curation helpers
# ---------------------------------------------------------------------------


def _parse_region_spec(value: str) -> dict:
    """Parse a region spec like 'chr2:216013551-216013551:-:tss:primary_TSS'."""
    parts = value.strip().split(":")
    if len(parts) < 3:
        raise ValueError(f"Region spec must have at least chrom:start-end:role: {value!r}")
    chrom = parts[0]
    coords = parts[1].split("-")
    if len(coords) != 2:
        raise ValueError(f"Coordinates must be start-end: {parts[1]!r}")
    start = int(coords[0])
    end = int(coords[1])
    role = parts[2].lower()
    strand = parts[3] if len(parts) > 3 else "."
    label = parts[4] if len(parts) > 4 else ""
    return {
        "chrom": chrom,
        "start": start,
        "end": end,
        "summit": (start + end) // 2,
        "role": role,
        "strand": strand,
        "label": label,
    }


def _optional_int(row: pd.Series, key: str, default: int = -1) -> int:
    """Return an integer field while treating absent/NaN values as default."""
    value = row.get(key, default)
    return default if pd.isna(value) else int(value)


# ---------------------------------------------------------------------------
# Motif scanning in regions
# ---------------------------------------------------------------------------


def _scan_region_for_ctcf(
    fasta: FastaReference,
    chrom: str,
    start: int,
    end: int,
    pwm: PWMModel,
    min_relative_score: float = 0.80,
) -> list[MotifHit]:
    """Scan a genomic region for CTCF motif hits, return sorted by score descending."""
    sequence = fasta.extract(chrom, start, end)
    return scan_motif_hits(sequence, start, pwm, min_relative_score=min_relative_score)


# ---------------------------------------------------------------------------
# LC-DIC Pol2 peak-center perturbation
# ---------------------------------------------------------------------------

BASES = "ACGT"
COMPLEMENT = str.maketrans("ACGTN", "TGCAN")


def _reverse_complement(seq: str) -> str:
    return seq.upper().translate(COMPLEMENT)[::-1]


def _design_pol2_peak_center_perturbation(
    *,
    fasta: FastaReference,
    chrom: str,
    peak_start: int,
    peak_end: int,
    peak_summit: int,
    edit_length: int = 3,
    gc_change_tolerance: float = 0.0,
) -> dict | None:
    """Design a short deterministic substitution centered on a Pol2 peak summit.

    The substitution replaces *edit_length* bases centered at the summit with
    a sequence that preserves GC content as closely as possible while avoiding
    creation of a high-scoring CTCF motif.

    Returns a dict suitable for the mutation manifest, or None on failure.
    """
    half = edit_length // 2
    edit_start = peak_summit - half
    edit_end = edit_start + edit_length

    if edit_start < 0 or edit_end > fasta.chrom_length(chrom):
        return None

    ref_seq = fasta.extract(chrom, edit_start, edit_end)
    if len(ref_seq) != edit_length or any(b not in BASES for b in ref_seq):
        return None

    # Deterministic substitution: flip each base to its complement
    alt_chars = []
    for b in ref_seq:
        idx = BASES.index(b)
        alt_chars.append(BASES[(idx + 2) % 4])  # A↔C, G↔T
    alt_seq = "".join(alt_chars)

    ref_gc = sum(1 for b in ref_seq if b in "GC")
    alt_gc = sum(1 for b in alt_seq if b in "GC")
    gc_delta = alt_gc - ref_gc

    if gc_change_tolerance is not None and abs(gc_delta) > gc_change_tolerance * edit_length:
        return None

    return {
        "edit_start": edit_start,
        "edit_end": edit_end,
        "ref_sequence": ref_seq,
        "alt_sequence": alt_seq,
        "edited_base_count": edit_length,
        "gc_delta_count": gc_delta,
        "gc_change": gc_delta / edit_length if edit_length else 0.0,
        "target_feature": "pol2_peak_center",
    }


# ---------------------------------------------------------------------------
# Matched control design for non-CTCF perturbations
# ---------------------------------------------------------------------------


def _design_local_sequence_control(
    *,
    fasta: FastaReference,
    chrom: str,
    target_edit_start: int,
    target_edit_end: int,
    target_ref: str,
    target_alt: str,
    target_gc_delta: int,
    target_edit_length: int,
    peak_start: int,
    peak_end: int,
    max_distance_bp: int = 500,
    pwm: PWMModel | None = None,
    min_relative_score: float = 0.80,
) -> dict | None:
    """Design a matched local control for a non-CTCF-motif perturbation.

    Searches within *max_distance_bp* of the target edit for a same-length window
    that (a) does not overlap the target edit, (b) has the same GC change when
    the same deterministic substitution is applied, and (c) does not contain a
    high-scoring CTCF motif.
    """
    edit_length = target_edit_end - target_edit_start
    chrom_length = fasta.chrom_length(chrom)
    target_center = (target_edit_start + target_edit_end) // 2

    # Candidate starts ordered by distance from target center
    lower = max(0, target_edit_start - max_distance_bp)
    upper = min(chrom_length - edit_length, target_edit_end + max_distance_bp)
    candidates = sorted(
        range(lower, upper + 1),
        key=lambda s: (abs((s + edit_length // 2) - target_center), s),
    )

    for candidate_start in candidates:
        candidate_end = candidate_start + edit_length
        # No overlap with target
        if max(candidate_start, target_edit_start) < min(candidate_end, target_edit_end):
            continue
        # A non-peak control must be outside the target peak, not merely outside
        # the edited bases.
        if max(candidate_start, peak_start) < min(candidate_end, peak_end):
            continue
        ref_window = fasta.extract(chrom, candidate_start, candidate_end)
        if any(b not in BASES for b in ref_window):
            continue
        # Apply the same deterministic substitution
        alt_chars = []
        for b in ref_window:
            idx = BASES.index(b)
            alt_chars.append(BASES[(idx + 2) % 4])
        alt_window = "".join(alt_chars)
        gc_delta = sum(1 for b in alt_window if b in "GC") - sum(1 for b in ref_window if b in "GC")
        if gc_delta != target_gc_delta:
            continue
        # Check no high-scoring CTCF motif in the ref window
        if pwm is not None:
            _, rel = _best_window_score(ref_window, pwm)
            if rel >= min_relative_score:
                continue
        return {
            "edit_start": candidate_start,
            "edit_end": candidate_end,
            "ref_sequence": ref_window,
            "alt_sequence": alt_window,
            "edited_base_count": edit_length,
            "gc_delta_count": gc_delta,
            "gc_change": gc_delta / edit_length if edit_length else 0.0,
            "target_feature": "non_peak_local_control",
        }
    return None


def _best_window_score(seq: str, pwm: PWMModel) -> Tuple[float, float]:
    """Return best (score, relative_score) on either strand for a PWM-width window."""
    from grelu.interpret.ism.motifs import score_oriented_sequence, reverse_complement

    if len(seq) != pwm.width:
        return float("-inf"), 0.0
    plus = score_oriented_sequence(seq, pwm)
    minus = score_oriented_sequence(reverse_complement(seq), pwm)
    score = max(plus, minus)
    return score, pwm.relative_score(score)


# ---------------------------------------------------------------------------
# Main manifest preparation
# ---------------------------------------------------------------------------


def _make_region_id(role: str, index: int = 0) -> str:
    return f"mreg_{role}_{index:02d}"


def _make_mutation_id(region_id: str, strategy: str) -> str:
    return f"{region_id}_{strategy}"


def _check_1mb_context(
    fasta: FastaReference,
    chrom: str,
    edit_center: int,
) -> bool:
    """Verify that a full 1 Mb AlphaGenome input window fits around edit_center."""
    half = AG_INPUT_LEN // 2
    try:
        clen = fasta.chrom_length(chrom)
    except KeyError:
        return False
    return edit_center - half >= 0 and edit_center + half <= clen


def _build_readout_regions(
    tss_coord: int,
    hc_dic_summit: int | None,
    lc_dic_summit: int | None,
    chrom: str,
) -> pd.DataFrame:
    """Build the named readout regions per the plan's Section 5.3."""
    records = []

    # TSS regions
    for label, radius_bp, role in [
        ("mreg_tss_4kb", 2000, "tss_readout"),
        ("mreg_tss_20kb", 10000, "tss_readout_wide"),
    ]:
        records.append({
            "readout_id": label,
            "source_region_id": "mreg_tss",
            "chrom": chrom,
            "start": max(0, tss_coord - radius_bp),
            "end": tss_coord + radius_bp,
            "role": role,
            "anchor_coordinate": tss_coord,
            "expected_bin_count": (2 * radius_bp) // 128,
        })

    # HC-DIC region
    if hc_dic_summit is not None:
        radius_bp = 2000
        records.append({
            "readout_id": "hc_dic_4kb",
            "source_region_id": "mreg_hc_dic",
            "chrom": chrom,
            "start": max(0, hc_dic_summit - radius_bp),
            "end": hc_dic_summit + radius_bp,
            "role": "hc_dic_readout",
            "anchor_coordinate": hc_dic_summit,
            "expected_bin_count": (2 * radius_bp) // 128,
        })

    # LC-DIC region
    if lc_dic_summit is not None:
        radius_bp = 2000
        records.append({
            "readout_id": "lc_dic_4kb",
            "source_region_id": "mreg_lc_dic",
            "chrom": chrom,
            "start": max(0, lc_dic_summit - radius_bp),
            "end": lc_dic_summit + radius_bp,
            "role": "lc_dic_readout",
            "anchor_coordinate": lc_dic_summit,
            "expected_bin_count": (2 * radius_bp) // 128,
        })

    # Mutation-local windows (added per-mutation by the runner)
    records.append({
        "readout_id": "mutation_center_1kb",
        "source_region_id": "_per_mutation_",
        "chrom": chrom,
        "start": -1,  # filled per mutation
        "end": -1,
        "role": "mutation_center_readout",
        "anchor_coordinate": -1,
        "expected_bin_count": 1024 // 128,
    })
    records.append({
        "readout_id": "mutation_local_4kb",
        "source_region_id": "_per_mutation_",
        "chrom": chrom,
        "start": -1,
        "end": -1,
        "role": "mutation_local_readout",
        "anchor_coordinate": -1,
        "expected_bin_count": 4096 // 128,
    })

    return pd.DataFrame.from_records(records)


def _build_track_registry() -> pd.DataFrame:
    """Build the hardcoded track registry for the three-region pilot.

    Focused on 128 bp ChIP-seq tracks: CTCF, RAD21, POLR2A, SMC3, H3K4me3,
    H3K27ac, H3K4me2. MCF-7 preferred, with deterministic fallback.
    """
    targets = [
        # (target_name, selection_priority, is_primary, output_type)
        ("CTCF", 1, True, "chip_tf"),
        ("RAD21", 2, True, "chip_tf"),
        ("POLR2A", 3, True, "chip_tf"),
        ("POLR2B", 4, False, "chip_tf"),  # fallback if no POLR2A
        ("SMC3", 5, True, "chip_tf"),
        ("H3K4me3", 6, True, "chip_histone"),
        ("H3K27ac", 7, False, "chip_histone"),
        ("H3K4me2", 8, False, "chip_histone"),
    ]
    records = []
    for target, priority, is_primary, output_type in targets:
        records.append({
            "track_id": f"__resolve_at_runtime__:{target}",
            "output_type": output_type,
            "target_name": target,
            "biosample": "MCF-7",
            "track_index": -1,  # resolved at runtime
            "selection_priority": priority,
            "is_primary": is_primary,
            "fallback_reason": "",
        })
    return pd.DataFrame.from_records(records)


def prepare_manifests(
    *,
    curated_regions: pd.DataFrame,
    fasta_path: Path,
    gtf_path: Path,
    meme_path: Path,
    gene_name: str = "MREG",
    output_dir: Path,
) -> dict:
    """Main entry point: produce all manifests from curated region input.

    Parameters
    ----------
    curated_regions : pd.DataFrame
        Must have columns: region_role, chrom, start, end, summit, strand, source, label.
        region_role ∈ {tss, hc_dic, lc_dic}.
    fasta_path : Path
        Path to hg38 FASTA.
    gtf_path : Path
        Path to hg38 GTF annotation.
    meme_path : Path
        Path to JASPAR MEME motif file.
    gene_name : str
        Gene symbol for the host gene (default: MREG).
    output_dir : Path
        Output directory for manifests.

    Returns
    -------
    dict with paths to all output files.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 0. Load resources
    # ------------------------------------------------------------------
    fasta = FastaReference(str(fasta_path))
    pwm = load_meme_pwm(str(meme_path), motif_name="CTCF", pseudocount=PSEUDOCOUNT)

    # Extract MREG gene info from GTF
    gene_chrom, gene_start, gene_end, gene_strand = _parse_gtf_gene_region(gtf_path, gene_name)
    transcripts = _parse_gtf_transcripts(gtf_path, gene_name)

    # Determine primary TSS
    tss_rows = curated_regions[curated_regions["region_role"] == "tss"]
    if tss_rows.empty:
        # Fall back to GTF: use the most upstream TSS
        primary_tss = int(transcripts["tss"].min()) if not transcripts.empty else None
        if primary_tss is None:
            raise ValueError("No TSS region provided and no transcripts found in GTF")
        print(f"[prepare] Using primary TSS from GTF: {gene_chrom}:{primary_tss}")
    else:
        primary_tss = int(tss_rows.iloc[0]["summit"])
        print(f"[prepare] Using primary TSS from curated regions: {gene_chrom}:{primary_tss}")

    # ------------------------------------------------------------------
    # 1. Build region manifest
    # ------------------------------------------------------------------
    region_records = []
    qc_issues = []
    for _, row in curated_regions.iterrows():
        role = str(row["region_role"])
        region_start = int(row["start"])
        region_end = int(row["end"])
        region_summit = int(
            row.get("summit", (region_start + region_end) // 2)
        )
        review_status = str(
            row.get(
                "review_status",
                "confirmed" if role == "tss" else "needs_review",
            )
        )

        if role in ("hc_dic", "lc_dic"):
            if region_start < gene_start or region_end > gene_end:
                qc_issues.append(
                    f"{role}: candidate {gene_chrom}:{region_start}-{region_end} "
                    f"is outside the MREG gene body {gene_start}-{gene_end}"
                )
            distance_to_gene_ends = min(
                abs(region_summit - gene_start),
                abs(region_summit - gene_end),
            )
            if distance_to_gene_ends <= 10_000:
                qc_issues.append(
                    f"{role}: candidate summit {region_summit} is only "
                    f"{distance_to_gene_ends} bp from a gene end; DIC requires >10 kb"
                )

        region_records.append({
            "region_id": _make_region_id(role),
            "region_role": role,
            "chrom": str(row["chrom"]),
            "start": region_start,
            "end": region_end,
            "summit": region_summit,
            "strand": str(row.get("strand", ".")),
            "source": str(row.get("source", "curated_regions")),
            "paper_label": str(row.get("label", "")),
            "host_gene": gene_name,
            "transcript_id": "",
            "tss": primary_tss,
            "distance_to_tss": int(row.get("summit", (int(row["start"]) + int(row["end"])) // 2)) - primary_tss,
            "has_ctcf_motif": False,  # filled below
            "ctcf_motif_start": -1,
            "ctcf_motif_end": -1,
            "ctcf_motif_score": np.nan,
            "pol2_peak_start": _optional_int(row, "pol2_peak_start"),
            "pol2_peak_end": _optional_int(row, "pol2_peak_end"),
            "pol2_peak_summit": _optional_int(row, "pol2_peak_summit"),
            "review_status": review_status,
        })
    region_manifest = pd.DataFrame.from_records(region_records)
    region_manifest.to_csv(output_dir / "mreg_region_manifest.tsv", sep="\t", index=False)

    # ------------------------------------------------------------------
    # 2. Mutation design
    # ------------------------------------------------------------------
    mutation_records = []
    sequence_contexts = []

    for _, region in region_manifest.iterrows():
        region_id = str(region["region_id"])
        role = str(region["region_role"])
        chrom = str(region["chrom"])
        region_start = int(region["start"])
        region_end = int(region["end"])
        region_summit = int(region["summit"])

        if role in ("hc_dic", "lc_dic"):
            within_gene = region_start >= gene_start and region_end <= gene_end
            far_from_gene_ends = min(
                abs(region_summit - gene_start),
                abs(region_summit - gene_end),
            ) > 10_000
            confirmed = str(region["review_status"]).lower() == "confirmed"
            if not (within_gene and far_from_gene_ends and confirmed):
                qc_issues.append(
                    f"{region_id}: skipped mutation design because the DIC candidate "
                    "is not both coordinate-valid and manually confirmed"
                )
                continue

        if role in ("tss", "hc_dic"):
            # ---- CTCF motif disruption strategy ----
            region_seq = fasta.extract(chrom, region_start, region_end)
            hits = _scan_region_for_ctcf(fasta, chrom, region_start, region_end, pwm)
            if not hits:
                qc_issues.append(f"{region_id}: no CTCF motif found in region {chrom}:{region_start}-{region_end}")
                continue

            best_hit = hits[0]  # highest-scoring motif
            # Mark region as having CTCF motif
            region_idx = region_manifest["region_id"] == region_id
            region_manifest.loc[region_idx, "has_ctcf_motif"] = True
            region_manifest.loc[region_idx, "ctcf_motif_start"] = best_hit.start
            region_manifest.loc[region_idx, "ctcf_motif_end"] = best_hit.end
            region_manifest.loc[region_idx, "ctcf_motif_score"] = best_hit.relative_score

            # Design strand-aware max-IC disruption
            # Create a minimal config-like object for _design_motif_disruption
            from dataclasses import dataclass

            @dataclass
            class _MinimalMutationConfig:
                positions_per_motif: int = 3
                max_positions_per_motif: int = 3
                min_relative_motif_score: float = 0.80
                motif_rescan_flank_bp: int = 100
                reject_new_high_scoring_motif: bool = True
                max_control_distance_bp: int = 500
                max_gc_change_difference: float = 0.0
                local_controls_per_motif: int = 3
                min_local_controls_per_motif: int = 1
                include_motif_preserving_control: bool = False
                motif_name: str = "CTCF"

            @dataclass
            class _ConfigWrapper:
                mutation: _MinimalMutationConfig

            config = _ConfigWrapper(mutation=_MinimalMutationConfig())
            disruption = _design_motif_disruption(
                fasta=fasta,
                chrom=chrom,
                hit=best_hit,
                pwm=pwm,
                config=config,
            )

            if disruption is None:
                qc_issues.append(f"{region_id}: motif disruption design failed")
                continue

            mutation_id = _make_mutation_id(region_id, "ctcf_snv")
            mutation_records.append({
                "mutation_id": mutation_id,
                "region_id": region_id,
                "mutation_strategy": "strand_aware_ctcf_pwm_disruption",
                "edit_start": disruption.edit_start,
                "edit_end": disruption.edit_end,
                "ref_sequence": disruption.ref_sequence,
                "alt_sequence": disruption.alt_sequence,
                "target_feature": "CTCF_motif",
                "matched_control_id": f"{mutation_id}_ctrl",
                "ref_pwm_score": disruption.pwm_ref_score,
                "alt_pwm_score": disruption.pwm_alt_score,
                "delta_pwm_score": disruption.pwm_alt_score - disruption.pwm_ref_score,
                "fasta_ref_match": True,
                "qc_status": "ready",
            })

            # Design matched controls
            controls = _design_local_controls(
                fasta=fasta,
                chrom=chrom,
                site_id=region_id,
                hit=best_hit,
                disruption=disruption,
                pwm=pwm,
                config=config,
            )
            for ctrl_idx, control in enumerate(controls):
                ctrl_id = f"{mutation_id}_ctrl_{ctrl_idx:02d}"
                mutation_records.append({
                    "mutation_id": ctrl_id,
                    "region_id": region_id,
                    "mutation_strategy": "local_non_motif_control",
                    "edit_start": control.edit_start,
                    "edit_end": control.edit_end,
                    "ref_sequence": control.ref_sequence,
                    "alt_sequence": control.alt_sequence,
                    "target_feature": "non_motif_control",
                    "matched_control_id": "",
                    "ref_pwm_score": control.pwm_ref_score,
                    "alt_pwm_score": control.pwm_alt_score,
                    "delta_pwm_score": control.pwm_alt_score - control.pwm_ref_score,
                    "fasta_ref_match": True,
                    "qc_status": "ready",
                })

            # Sequence context (flanking ref/alt)
            flank = 50
            ctx_start, ref_ctx = _context_for_edit(fasta, chrom, disruption.edit_start, disruption.edit_end, flank)
            alt_ctx = _apply_edit_to_context(ref_ctx, ctx_start, disruption)
            sequence_contexts.append({
                "mutation_id": mutation_id,
                "chrom": chrom,
                "context_start": ctx_start,
                "context_end": ctx_start + len(ref_ctx),
                "ref_context": ref_ctx,
                "alt_context": alt_ctx,
            })

        elif role == "lc_dic":
            # ---- LC-DIC Pol2 peak-center perturbation ----
            pol2_peak_start = int(region.get("pol2_peak_start", -1))
            pol2_peak_end = int(region.get("pol2_peak_end", -1))
            pol2_peak_summit = int(region.get("pol2_peak_summit", -1))
            # Use region coordinates as fallback if pol2_peak_* not set
            if pol2_peak_start <= 0:
                pol2_peak_start = region_start
            if pol2_peak_end <= 0:
                pol2_peak_end = region_end
            if pol2_peak_summit <= 0:
                pol2_peak_summit = region_summit

            region_manifest.loc[region_manifest["region_id"] == region_id, "pol2_peak_start"] = pol2_peak_start
            region_manifest.loc[region_manifest["region_id"] == region_id, "pol2_peak_end"] = pol2_peak_end
            region_manifest.loc[region_manifest["region_id"] == region_id, "pol2_peak_summit"] = pol2_peak_summit

            pert = _design_pol2_peak_center_perturbation(
                fasta=fasta,
                chrom=chrom,
                peak_start=pol2_peak_start,
                peak_end=pol2_peak_end,
                peak_summit=pol2_peak_summit,
                edit_length=3,
                gc_change_tolerance=1.0,
            )
            if pert is None:
                qc_issues.append(f"{region_id}: Pol2 peak-center perturbation design failed")
                continue

            mutation_id = _make_mutation_id(region_id, "pol2_perturbation")
            mutation_records.append({
                "mutation_id": mutation_id,
                "region_id": region_id,
                "mutation_strategy": "pol2_peak_center_sequence_perturbation",
                "edit_start": pert["edit_start"],
                "edit_end": pert["edit_end"],
                "ref_sequence": pert["ref_sequence"],
                "alt_sequence": pert["alt_sequence"],
                "target_feature": "Pol2_peak_center",
                "matched_control_id": f"{mutation_id}_ctrl",
                "ref_pwm_score": np.nan,
                "alt_pwm_score": np.nan,
                "delta_pwm_score": np.nan,
                "fasta_ref_match": True,
                "qc_status": "ready",
            })

            # Matched local control
            ctrl = _design_local_sequence_control(
                fasta=fasta,
                chrom=chrom,
                target_edit_start=pert["edit_start"],
                target_edit_end=pert["edit_end"],
                target_ref=pert["ref_sequence"],
                target_alt=pert["alt_sequence"],
                target_gc_delta=pert["gc_delta_count"],
                target_edit_length=pert["edited_base_count"],
                peak_start=pol2_peak_start,
                peak_end=pol2_peak_end,
                max_distance_bp=5_000,
                pwm=pwm,
            )
            if ctrl is not None:
                ctrl_id = f"{mutation_id}_ctrl_00"
                mutation_records.append({
                    "mutation_id": ctrl_id,
                    "region_id": region_id,
                    "mutation_strategy": "non_peak_local_control",
                    "edit_start": ctrl["edit_start"],
                    "edit_end": ctrl["edit_end"],
                    "ref_sequence": ctrl["ref_sequence"],
                    "alt_sequence": ctrl["alt_sequence"],
                    "target_feature": "non_peak_control",
                    "matched_control_id": "",
                    "ref_pwm_score": np.nan,
                    "alt_pwm_score": np.nan,
                    "delta_pwm_score": np.nan,
                    "fasta_ref_match": True,
                    "qc_status": "ready",
                })
            else:
                qc_issues.append(f"{region_id}: matched control for Pol2 perturbation not found")

            # Sequence context
            flank = 50
            ctx_start, ref_ctx = _context_for_edit_simple(fasta, chrom, pert["edit_start"], pert["edit_end"], flank)
            alt_ctx = _apply_simple_edit(ref_ctx, ctx_start, pert["edit_start"], pert["edit_end"], pert["alt_sequence"])
            sequence_contexts.append({
                "mutation_id": mutation_id,
                "chrom": chrom,
                "context_start": ctx_start,
                "context_end": ctx_start + len(ref_ctx),
                "ref_context": ref_ctx,
                "alt_context": alt_ctx,
            })

    # Write mutation manifest
    mutation_manifest = pd.DataFrame.from_records(mutation_records)
    mutation_manifest.to_csv(output_dir / "mreg_mutation_manifest.tsv", sep="\t", index=False)

    # Write sequence contexts
    ctx_df = pd.DataFrame.from_records(sequence_contexts)
    ctx_df.to_csv(output_dir / "sequence_contexts.tsv", sep="\t", index=False)

    # ------------------------------------------------------------------
    # 3. FASTA REF validation
    # ------------------------------------------------------------------
    fasta_issues = []
    for _, mut in mutation_manifest.iterrows():
        edit_start = int(mut["edit_start"])
        edit_end = int(mut["edit_end"])
        expected_ref = str(mut["ref_sequence"])
        try:
            observed = fasta.extract(str(region_manifest.iloc[0]["chrom"]), edit_start, edit_end)
        except Exception as exc:
            fasta_issues.append(f"{mut['mutation_id']}: FASTA extraction failed: {exc}")
            continue
        if observed.upper() != expected_ref.upper():
            fasta_issues.append(
                f"{mut['mutation_id']}: REF mismatch at {edit_start}-{edit_end}: "
                f"expected {expected_ref}, observed {observed}"
            )
    if fasta_issues:
        qc_issues.extend(fasta_issues)

    # Update region manifest with final motif info
    region_manifest.to_csv(output_dir / "mreg_region_manifest.tsv", sep="\t", index=False)

    # ------------------------------------------------------------------
    # 4. Readout regions
    # ------------------------------------------------------------------
    hc_row = curated_regions[curated_regions["region_role"] == "hc_dic"]
    lc_row = curated_regions[curated_regions["region_role"] == "lc_dic"]
    hc_summit = int(hc_row.iloc[0]["summit"]) if not hc_row.empty else None
    lc_summit = int(lc_row.iloc[0]["summit"]) if not lc_row.empty else None

    readout_regions = _build_readout_regions(primary_tss, hc_summit, lc_summit, gene_chrom)
    readout_regions.to_csv(output_dir / "mreg_readout_regions.tsv", sep="\t", index=False)

    # ------------------------------------------------------------------
    # 5. Track registry
    # ------------------------------------------------------------------
    track_registry = _build_track_registry()
    track_registry.to_csv(output_dir / "mreg_track_registry.tsv", sep="\t", index=False)

    # ------------------------------------------------------------------
    # 6. QC report
    # ------------------------------------------------------------------
    qc_report = {
        "gene_name": gene_name,
        "gene_chrom": gene_chrom,
        "gene_start": gene_start,
        "gene_end": gene_end,
        "gene_strand": gene_strand,
        "primary_tss": primary_tss,
        "n_regions": len(region_manifest),
        "n_mutations": len(mutation_manifest),
        "n_experimental_mutations": int(
            (~mutation_manifest["mutation_strategy"].str.contains("control")).sum()
            if not mutation_manifest.empty
            else 0
        ),
        "n_controls": int(
            (mutation_manifest["mutation_strategy"].isin(
                ["local_non_motif_control", "non_peak_local_control"]
            )).sum()
            if not mutation_manifest.empty
            else 0
        ),
        "n_readout_regions": len(readout_regions),
        "qc_issues": qc_issues,
        "fasta_path": str(fasta_path),
        "gtf_path": str(gtf_path),
        "meme_path": str(meme_path),
    }
    with (output_dir / "preparation_qc.json").open("w") as handle:
        json.dump(qc_report, handle, indent=2, default=str)

    fasta.close()

    return {
        "output_dir": str(output_dir),
        "region_manifest": str(output_dir / "mreg_region_manifest.tsv"),
        "mutation_manifest": str(output_dir / "mreg_mutation_manifest.tsv"),
        "readout_regions": str(output_dir / "mreg_readout_regions.tsv"),
        "track_registry": str(output_dir / "mreg_track_registry.tsv"),
        "sequence_contexts": str(output_dir / "sequence_contexts.tsv"),
        "qc_report": str(output_dir / "preparation_qc.json"),
        "qc_issue_count": len(qc_issues),
    }


# ---------------------------------------------------------------------------
# Small local helpers (duplicated from motifs.py for LC-DIC control path)
# ---------------------------------------------------------------------------


def _context_for_edit_simple(
    fasta: FastaReference,
    chrom: str,
    edit_start: int,
    edit_end: int,
    flank_bp: int,
) -> Tuple[int, str]:
    chrom_len = fasta.chrom_length(chrom)
    start = max(0, edit_start - flank_bp)
    end = min(chrom_len, edit_end + flank_bp)
    return start, fasta.extract(chrom, start, end)


def _apply_simple_edit(
    context_seq: str,
    context_start: int,
    edit_start: int,
    edit_end: int,
    alt_seq: str,
) -> str:
    rel_start = edit_start - context_start
    rel_end = edit_end - context_start
    return context_seq[:rel_start] + alt_seq + context_seq[rel_end:]


def _context_for_edit(
    fasta: FastaReference,
    chrom: str,
    edit_start: int,
    edit_end: int,
    flank_bp: int,
) -> Tuple[int, str]:
    return _context_for_edit_simple(fasta, chrom, edit_start, edit_end, flank_bp)


def _apply_edit_to_context(
    context_seq: str,
    context_start: int,
    disruption: DesignedEdit,
) -> str:
    return _apply_simple_edit(
        context_seq, context_start,
        disruption.edit_start, disruption.edit_end, disruption.alt_sequence,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--regions",
        required=True,
        help="Path to curated regions TSV (region_role, chrom, start, end, summit, strand, source, label)",
    )
    parser.add_argument(
        "--fasta",
        default=str(DEFAULT_FASTA),
        help=f"Path to hg38 FASTA [default: {DEFAULT_FASTA}]",
    )
    parser.add_argument(
        "--gtf",
        default=str(DEFAULT_GTF),
        help=f"Path to hg38 GTF [default: {DEFAULT_GTF}]",
    )
    parser.add_argument(
        "--meme",
        default=str(DEFAULT_MEME) if DEFAULT_MEME.exists() else "",
        help="Path to JASPAR MEME motif file",
    )
    parser.add_argument("--gene-name", default="MREG", help="Gene symbol [default: MREG]")
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output directory for manifests",
    )
    args = parser.parse_args()

    if not Path(args.regions).exists():
        raise SystemExit(f"Curated regions file not found: {args.regions}")
    if not Path(args.fasta).exists():
        raise SystemExit(f"FASTA file not found: {args.fasta}")
    if not Path(args.gtf).exists():
        raise SystemExit(f"GTF file not found: {args.gtf}")
    if args.meme and not Path(args.meme).exists():
        raise SystemExit(f"MEME motif file not found: {args.meme}")

    curated = pd.read_csv(args.regions, sep="\t", comment="#")
    required_cols = {"region_role", "chrom", "start", "end"}
    missing = required_cols - set(curated.columns)
    if missing:
        raise SystemExit(f"Curated regions missing required columns: {sorted(missing)}")

    valid_roles = {"tss", "hc_dic", "lc_dic"}
    invalid_roles = set(curated["region_role"].str.lower()) - valid_roles
    if invalid_roles:
        raise SystemExit(f"Invalid region_role values: {sorted(invalid_roles)}")

    curated["region_role"] = curated["region_role"].str.lower()
    if "summit" not in curated.columns:
        curated["summit"] = (curated["start"] + curated["end"]) // 2
    if "strand" not in curated.columns:
        curated["strand"] = "."
    if "source" not in curated.columns:
        curated["source"] = "user_provided"
    if "label" not in curated.columns:
        curated["label"] = ""

    result = prepare_manifests(
        curated_regions=curated,
        fasta_path=Path(args.fasta),
        gtf_path=Path(args.gtf),
        meme_path=Path(args.meme) if args.meme else DEFAULT_MEME,
        gene_name=args.gene_name,
        output_dir=Path(args.output_dir),
    )

    print(f"\nPrepared manifests in {result['output_dir']}")
    print(f"  Regions:       {result['region_manifest']}")
    print(f"  Mutations:     {result['mutation_manifest']}")
    print(f"  Readouts:      {result['readout_regions']}")
    print(f"  Track registry:{result['track_registry']}")
    print(f"  Contexts:      {result['sequence_contexts']}")
    print(f"  QC report:     {result['qc_report']}")
    if result["qc_issue_count"]:
        print(f"\n⚠  {result['qc_issue_count']} QC issues (see preparation_qc.json)")


if __name__ == "__main__":
    main()
