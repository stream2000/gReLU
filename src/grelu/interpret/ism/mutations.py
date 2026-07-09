"""Mutation manifest builders for saturation ISM."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pandas as pd

BASES = "ACGT"


@dataclass(frozen=True)
class SaturationWindow:
    """A genomic interval to expand into all non-reference SNVs."""

    window_id: str
    chrom: str
    start: int
    end: int
    anchor: int
    gene: str = ""
    source: str = ""
    label: str = ""


def _slug(value: str) -> str:
    text = str(value).strip().lower()
    return "".join(char if char.isalnum() else "_" for char in text).strip("_")


def enumerate_snv_sites(*, fasta, window: SaturationWindow) -> pd.DataFrame:
    """Return one manifest row per non-reference SNV in ``window``."""

    start = int(window.start)
    end = int(window.end)
    if end <= start:
        raise ValueError(f"Invalid saturation window: {window.chrom}:{start}-{end}")
    sequence = fasta.extract(window.chrom, start, end).upper()
    rows: list[dict] = []
    token = _slug(window.window_id)
    for offset, ref_base in enumerate(sequence):
        if ref_base not in BASES:
            continue
        pos0 = start + offset
        for alt_base in BASES:
            if alt_base == ref_base:
                continue
            mutation_id = f"{token}__snv_{pos0}_{ref_base}_{alt_base}"
            rows.append(
                {
                    "mutation_id": mutation_id,
                    "window_id": window.window_id,
                    "gene": window.gene,
                    "chrom": window.chrom,
                    "anchor": int(window.anchor),
                    "edit_start": pos0,
                    "edit_end": pos0 + 1,
                    "ref_sequence": ref_base,
                    "alt_sequence": alt_base,
                    "mutation_kind": "snv",
                    "control_type": "experimental",
                    "matched_target_id": mutation_id,
                    "variant_position": pos0,
                    "ref_base": ref_base,
                    "alt_base": alt_base,
                    "variant_id": f"{window.chrom}:{pos0}:{ref_base}>{alt_base}",
                    "saturation_start": start,
                    "saturation_end": end,
                    "source": window.source,
                    "label": window.label or window.window_id,
                }
            )
    return pd.DataFrame.from_records(rows)


def enumerate_snv_site_table(*, fasta, windows: Iterable[SaturationWindow]) -> pd.DataFrame:
    """Expand all windows into one mutation manifest."""

    tables = [enumerate_snv_sites(fasta=fasta, window=window) for window in windows]
    if not tables:
        return pd.DataFrame()
    table = pd.concat(tables, ignore_index=True)
    if table["mutation_id"].duplicated().any():
        examples = table.loc[table["mutation_id"].duplicated(), "mutation_id"].head().tolist()
        raise ValueError(f"Duplicate mutation_id values: {examples}")
    return table


def apply_equal_length_edit(sequence: str, sequence_start: int, row: pd.Series) -> str:
    """Apply a manifest edit to a reference sequence after REF validation."""

    edit_start = int(row["edit_start"])
    edit_end = int(row["edit_end"])
    ref = str(row["ref_sequence"]).upper()
    alt = str(row["alt_sequence"]).upper()
    if len(ref) != edit_end - edit_start or len(alt) != len(ref):
        raise ValueError(f"Edit length mismatch for {row['mutation_id']}")
    local_start = edit_start - int(sequence_start)
    local_end = edit_end - int(sequence_start)
    if local_start < 0 or local_end > len(sequence):
        raise ValueError(f"Edit outside sequence context for {row['mutation_id']}")
    observed = sequence[local_start:local_end].upper()
    if observed != ref:
        raise ValueError(
            f"REF mismatch for {row['mutation_id']}: expected {ref}, observed {observed}"
        )
    return sequence[:local_start] + alt + sequence[local_end:]
