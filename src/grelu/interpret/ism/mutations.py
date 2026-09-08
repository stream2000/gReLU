"""Mutation manifest builders for saturation ISM."""

from __future__ import annotations

import hashlib
import random
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


def _rng_for(*, seed: int, mutation_key: str) -> random.Random:
    digest = hashlib.sha256(f"{seed}:{mutation_key}".encode()).digest()
    return random.Random(int.from_bytes(digest[:8], byteorder="big", signed=False))


def _random_sequence(length: int, rng: random.Random) -> str:
    return "".join(rng.choice(BASES) for _ in range(length))


def _force_difference(ref: str, alt: str, rng: random.Random) -> str:
    if alt != ref:
        return alt
    if not ref:
        return alt
    idx = rng.randrange(len(ref))
    choices = [base for base in BASES if base != ref[idx]]
    return alt[:idx] + rng.choice(choices) + alt[idx + 1 :]


def replacement_sequence(ref: str, *, mode: str, seed: int, mutation_key: str) -> str:
    """Create a deterministic same-length background replacement sequence."""

    ref = ref.upper()
    if any(base not in BASES for base in ref):
        raise ValueError(f"Replacement windows must be ACGT-only, got {ref}")
    rng = _rng_for(seed=seed, mutation_key=mutation_key)
    if mode == "random":
        alt = _random_sequence(len(ref), rng)
    elif mode == "shuffle":
        chars = list(ref)
        rng.shuffle(chars)
        alt = "".join(chars)
    else:
        raise ValueError(f"Unknown replacement mode: {mode}")
    return _force_difference(ref, alt, rng)


def strict_unique_shuffles(
    ref: str, *, n: int, seed: int, mutation_key: str
) -> list[str]:
    """Return deterministic unique shuffles that preserve base composition."""

    ref = ref.upper()
    if len(set(ref)) < 2:
        raise ValueError(
            "Cannot create a different composition-preserving shuffle for "
            f"{mutation_key}: {ref}"
        )
    rng = _rng_for(seed=seed, mutation_key=mutation_key)
    seen = {ref}
    values = []
    for _ in range(100_000):
        chars = list(ref)
        rng.shuffle(chars)
        alternate = "".join(chars)
        if alternate in seen:
            continue
        seen.add(alternate)
        values.append(alternate)
        if len(values) == int(n):
            return values
    raise RuntimeError(
        f"Could not generate {n} unique shuffles for {mutation_key}: {ref}"
    )


@dataclass(frozen=True)
class AnchoredScan:
    """A strand-aware scan of same-length windows around one anchor.

    Offsets are transcription-relative: positive is downstream of ``anchor``
    on either strand. Genomic coordinates always increase left to right, so on
    the minus strand a positive offset maps to a smaller coordinate.
    """

    locus_id: str
    chrom: str
    anchor: int
    strand: str = "+"

    def __post_init__(self) -> None:
        if self.strand not in ("+", "-"):
            raise ValueError(f"Strand must be '+' or '-', got {self.strand!r}")

    @property
    def direction(self) -> int:
        return 1 if self.strand == "+" else -1

    def genomic_center(self, tx_offset: int) -> int:
        """Map a transcription-relative offset onto a genomic coordinate."""

        return int(self.anchor) + self.direction * int(tx_offset)

    def tx_offset(self, position: int) -> int:
        """Map a genomic coordinate onto a transcription-relative offset.

        Inverse of :meth:`genomic_center` on both strands.
        """

        return self.direction * (int(position) - int(self.anchor))


@dataclass(frozen=True)
class ScanEdit:
    """One scanned window and its composition-preserving replacements."""

    tx_offset: int
    genomic_center: int
    edit_start: int
    edit_end: int
    ref_sequence: str
    alt_sequences: tuple[str, ...]


@dataclass(frozen=True)
class ScanExclusion:
    """One scanned window that admits no composition-preserving replacement."""

    tx_offset: int
    genomic_center: int
    edit_start: int
    edit_end: int
    ref_sequence: str
    reason: str


def anchored_scan_centers(
    *, half_window_bp: int, span_bp: int, stride_bp: int
) -> list[int]:
    """Return transcription-relative edit centers tiling one anchored window.

    Centers are inset by half a span so every edit stays within
    ``half_window_bp`` of the anchor.
    """

    half_window_bp = int(half_window_bp)
    span_bp = int(span_bp)
    stride_bp = int(stride_bp)
    if span_bp <= 0:
        raise ValueError(f"span_bp must be positive, got {span_bp}")
    if span_bp % 2:
        raise ValueError(f"span_bp must be even so centers are unambiguous, got {span_bp}")
    if stride_bp <= 0:
        raise ValueError(f"stride_bp must be positive, got {stride_bp}")
    if span_bp > 2 * half_window_bp:
        raise ValueError(
            f"span_bp {span_bp} exceeds the {2 * half_window_bp} bp scan window"
        )
    inset = span_bp // 2
    return list(range(-half_window_bp + inset, half_window_bp - inset + 1, stride_bp))


def scan_anchored_strict_shuffles(
    *,
    fasta,
    scan: AnchoredScan,
    centers: Iterable[int],
    span_bp: int,
    replicates: int,
    seed: int,
) -> tuple[list[ScanEdit], list[ScanExclusion]]:
    """Strict-shuffle every centered window of a strand-aware anchored scan.

    A window whose reference has a single distinct base cannot be shuffled into
    anything else, so it is returned as an exclusion instead of raising. This
    lets a scan report its own coverage rather than abort on a homopolymer.

    Callers project the returned records onto their own manifest schema; the
    scan owns only sequence geometry and replacement determinism.
    """

    span_bp = int(span_bp)
    if span_bp <= 0:
        raise ValueError(f"span_bp must be positive, got {span_bp}")
    edits: list[ScanEdit] = []
    exclusions: list[ScanExclusion] = []
    for tx_offset in centers:
        tx_offset = int(tx_offset)
        genomic_center = scan.genomic_center(tx_offset)
        edit_start = genomic_center - span_bp // 2
        edit_end = edit_start + span_bp
        ref = fasta.extract(scan.chrom, edit_start, edit_end).upper()
        mutation_key = f"{scan.locus_id}:{tx_offset}:{edit_start}:{edit_end}"
        try:
            alternates = strict_unique_shuffles(
                ref, n=replicates, seed=seed, mutation_key=mutation_key
            )
        except ValueError as exc:
            exclusions.append(
                ScanExclusion(
                    tx_offset=tx_offset,
                    genomic_center=genomic_center,
                    edit_start=edit_start,
                    edit_end=edit_end,
                    ref_sequence=ref,
                    reason=str(exc),
                )
            )
            continue
        edits.append(
            ScanEdit(
                tx_offset=tx_offset,
                genomic_center=genomic_center,
                edit_start=edit_start,
                edit_end=edit_end,
                ref_sequence=ref,
                alt_sequences=tuple(alternates),
            )
        )
    return edits, exclusions


def enumerate_sliding_window_replacements(
    *,
    fasta,
    window: SaturationWindow,
    span_bp: int,
    stride_bp: int = 1,
    mode: str = "shuffle",
    replicates: int = 1,
    seed: int = 1,
) -> pd.DataFrame:
    """Return one manifest row per same-length sliding-window replacement."""

    start = int(window.start)
    end = int(window.end)
    span_bp = int(span_bp)
    stride_bp = int(stride_bp)
    replicates = int(replicates)
    if end <= start:
        raise ValueError(f"Invalid saturation window: {window.chrom}:{start}-{end}")
    if span_bp <= 0:
        raise ValueError(f"span_bp must be positive, got {span_bp}")
    if stride_bp <= 0:
        raise ValueError(f"stride_bp must be positive, got {stride_bp}")
    if replicates <= 0:
        raise ValueError(f"replicates must be positive, got {replicates}")
    if span_bp > end - start:
        raise ValueError(f"span_bp {span_bp} is larger than window length {end - start}")

    sequence = fasta.extract(window.chrom, start, end).upper()
    rows: list[dict] = []
    token = _slug(window.window_id)
    for offset in range(0, len(sequence) - span_bp + 1, stride_bp):
        edit_start = start + offset
        edit_end = edit_start + span_bp
        ref = sequence[offset : offset + span_bp]
        if any(base not in BASES for base in ref):
            continue
        center = edit_start + span_bp // 2
        for replicate in range(replicates):
            mutation_key = f"{token}:{mode}:{span_bp}:{edit_start}:{edit_end}:{replicate}"
            alt = replacement_sequence(ref, mode=mode, seed=seed, mutation_key=mutation_key)
            mutation_id = (
                f"{token}__window_{span_bp}bp_{mode}_{edit_start}_{edit_end}"
                f"__rep{replicate}"
            )
            rows.append(
                {
                    "mutation_id": mutation_id,
                    "window_id": window.window_id,
                    "gene": window.gene,
                    "chrom": window.chrom,
                    "anchor": int(window.anchor),
                    "edit_start": edit_start,
                    "edit_end": edit_end,
                    "edit_center_position": center,
                    "edit_length_bp": span_bp,
                    "ref_sequence": ref,
                    "alt_sequence": alt,
                    "mutation_kind": "sliding_window_replacement",
                    "replacement_mode": mode,
                    "replacement_replicate": replicate,
                    "control_type": "experimental",
                    "matched_target_id": mutation_id,
                    "variant_position": center,
                    "ref_base": ref,
                    "alt_base": alt,
                    "variant_id": f"{window.chrom}:{edit_start}-{edit_end}:{ref}>{alt}",
                    "saturation_start": start,
                    "saturation_end": end,
                    "source": window.source,
                    "label": window.label or window.window_id,
                }
            )
    return pd.DataFrame.from_records(rows)


def enumerate_sliding_window_replacement_table(
    *,
    fasta,
    windows: Iterable[SaturationWindow],
    span_bp: int,
    stride_bp: int = 1,
    mode: str = "shuffle",
    replicates: int = 1,
    seed: int = 1,
) -> pd.DataFrame:
    """Expand all windows into one sliding-window replacement manifest."""

    tables = [
        enumerate_sliding_window_replacements(
            fasta=fasta,
            window=window,
            span_bp=span_bp,
            stride_bp=stride_bp,
            mode=mode,
            replicates=replicates,
            seed=seed,
        )
        for window in windows
    ]
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
