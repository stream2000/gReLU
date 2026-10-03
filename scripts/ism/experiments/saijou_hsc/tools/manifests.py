"""Schemas of the tables the Saijou preparers write.

Three preparers emit these tables: the canonical nine-gene TSS scan, the Mdk
specificity background, and the focused Mdk targeted loci. They differ in locus
vocabulary and in how their edits are chosen, but the mutation manifest, the
excluded-window table and the readout table have one schema each, defined here.
Sequence geometry and replacement determinism belong to
`grelu.interpret.ism.mutations`, not here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from grelu.interpret.ism.mutations import AnchoredScan, ScanEdit, ScanExclusion

try:  # package import when the preparer runs as part of the tools package
    from .genomics import centered_interval
except ImportError:  # direct script execution
    from genomics import centered_interval


@dataclass(frozen=True)
class ManifestLocus:
    """The locus and gene vocabulary one manifest block is written against.

    Anchor geometry is delegated to ``scan`` rather than copied, so a locus and
    the scan that produced its edits cannot disagree about chrom, strand or TSS.
    """

    scan: AnchoredScan
    locus_role: str
    gene: str
    tes: int
    control_type: str
    source: str


def scan_mutation_id(
    locus: ManifestLocus, *, tx_offset: int, edit_start: int, edit_end: int, replicate: int
) -> str:
    """Mutation id for anchored scans, keyed by transcription offset."""

    return (
        f"{locus.scan.locus_id}__tx{tx_offset:+d}__{edit_start}_{edit_end}"
        f"__shuffle_rep{replicate:02d}"
    )


def interval_mutation_id(
    locus: ManifestLocus, *, edit_start: int, edit_end: int, replicate: int
) -> str:
    """Mutation id for templates whose edits are explicit genomic intervals."""

    return f"{locus.scan.locus_id}__{edit_start}_{edit_end}__shuffle_rep{replicate:02d}"


def strict_shuffle_mutation_row(
    locus: ManifestLocus,
    *,
    mutation_id: str,
    edit_start: int,
    edit_end: int,
    ref_sequence: str,
    alt_sequence: str,
    replicate: int,
) -> dict[str, object]:
    """Project one strict-shuffle replicate onto the shared manifest schema.

    Column order is part of the contract: prepared TSVs are keyed by it. Both
    offsets are recorded because genomic and transcriptional offsets disagree on
    the minus strand, and biological summaries must use the transcriptional one.
    """

    edit_start = int(edit_start)
    edit_end = int(edit_end)
    scan = locus.scan
    genomic_center = edit_start + (edit_end - edit_start) // 2
    return {
        "mutation_id": mutation_id,
        "locus_id": scan.locus_id,
        "locus_role": locus.locus_role,
        "gene": locus.gene,
        "chrom": scan.chrom,
        "gene_strand": scan.strand,
        "gene_tss": int(scan.anchor),
        "gene_tes": int(locus.tes),
        "edit_start": edit_start,
        "edit_end": edit_end,
        "edit_center_position": genomic_center,
        "edit_length_bp": edit_end - edit_start,
        "ref_sequence": ref_sequence,
        "alt_sequence": alt_sequence,
        "mutation_kind": "strict_mononucleotide_shuffle",
        "replacement_mode": "strict_shuffle",
        "replacement_replicate": replicate,
        "variant_position": genomic_center,
        "variant_offset_from_tss_genomic_bp": genomic_center - int(scan.anchor),
        "variant_offset_from_tss_transcription_bp": scan.tx_offset(genomic_center),
        "ref_base": ref_sequence,
        "alt_base": alt_sequence,
        "control_type": locus.control_type,
        "source": locus.source,
    }


def scan_manifest_rows(
    locus: ManifestLocus, edits: Iterable[ScanEdit]
) -> list[dict[str, object]]:
    """Project every replicate of an anchored scan onto the manifest schema."""

    return [
        strict_shuffle_mutation_row(
            locus,
            mutation_id=scan_mutation_id(
                locus,
                tx_offset=edit.tx_offset,
                edit_start=edit.edit_start,
                edit_end=edit.edit_end,
                replicate=replicate,
            ),
            edit_start=edit.edit_start,
            edit_end=edit.edit_end,
            ref_sequence=edit.ref_sequence,
            alt_sequence=alternate,
            replicate=replicate,
        )
        for edit in edits
        for replicate, alternate in enumerate(edit.alt_sequences)
    ]


def readout_row(
    *,
    readout_id: str,
    gene: str,
    chrom: str,
    start: int,
    end: int,
    anchor: int,
    role: str,
    locus_id: str = "*",
) -> dict[str, object]:
    """Project one readout interval onto the readouts table schema.

    ``locus_id`` is ``*`` for gene-level readouts and a locus id for readouts
    local to one edited locus.
    """

    return {
        "readout_id": readout_id,
        "gene": gene,
        "locus_id": locus_id,
        "chrom": chrom,
        "start": int(start),
        "end": int(end),
        "anchor": int(anchor),
        "role": role,
    }


def standard_readout_row(
    *, gene: str, chrom: str, role: str, center: int, width_bp: int
) -> dict[str, object]:
    """Gene-level readout of ``width_bp`` centred on one anchor.

    Keyed as ``{gene}__{role}_{width_bp}bp``. Anchors are computed rather than
    written down: a pasted interval silently stops tracking its anchor.
    """

    start, end = centered_interval(center, width_bp)
    return readout_row(
        readout_id=f"{gene}__{role}_{width_bp}bp",
        gene=gene,
        chrom=chrom,
        start=start,
        end=end,
        anchor=center,
        role=role,
    )


def scan_exclusion_rows(
    exclusions: Iterable[ScanExclusion], *, gene: str = ""
) -> list[dict[str, object]]:
    """Project unshuffleable windows onto the excluded-window table.

    ``gene`` is omitted when empty: the single-gene background table has no gene
    column and its prepared artifacts pin that schema.
    """

    rows = []
    for exclusion in exclusions:
        row: dict[str, object] = {"gene": gene} if gene else {}
        row.update(
            {
                "variant_offset_from_tss_transcription_bp": exclusion.tx_offset,
                "edit_start": exclusion.edit_start,
                "edit_end": exclusion.edit_end,
                "ref_sequence": exclusion.ref_sequence,
                "reason": exclusion.reason,
            }
        )
        rows.append(row)
    return rows
