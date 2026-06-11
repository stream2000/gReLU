"""Strand-aware CTCF motif disruption and matched-control design."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence, Tuple

import numpy as np
import pandas as pd

from grelu.interpret.ism.config import RunConfig
from grelu.interpret.ism.errors import ISMUserError
from grelu.interpret.ism.labels import validate_paper_labels
from grelu.interpret.ism.provenance import (
    output_dir,
    prepare_stage_outputs,
    record_stage,
    reject_completed_downstream_stages,
    require_completed_stage,
)
from grelu.interpret.ism.schemas import MUTATION_SCHEMA

BASES = "ACGT"
BASE_INDEX = {base: index for index, base in enumerate(BASES)}
COMPLEMENT = str.maketrans("ACGTN", "TGCAN")


class MotifDesignError(ISMUserError, ValueError):
    """Raised when motif inputs or mutation design are invalid."""


@dataclass(frozen=True)
class PWMModel:
    name: str
    probabilities: np.ndarray
    pssm: np.ndarray
    min_score: float
    max_score: float

    @property
    def width(self) -> int:
        return int(self.pssm.shape[0])

    def relative_score(self, score: float) -> float:
        span = self.max_score - self.min_score
        if span <= 0:
            raise MotifDesignError(f"PWM {self.name!r} has no score range")
        return float((score - self.min_score) / span)


@dataclass(frozen=True)
class MotifHit:
    start: int
    end: int
    strand: str
    score: float
    relative_score: float
    genomic_sequence: str

    @property
    def center(self) -> int:
        return self.start + (self.end - self.start) // 2


@dataclass(frozen=True)
class DesignedEdit:
    edit_start: int
    edit_end: int
    ref_sequence: str
    alt_sequence: str
    edited_positions: Tuple[int, ...]
    edited_base_count: int
    gc_delta_count: int
    gc_change: float
    pwm_ref_score: float
    pwm_alt_score: float
    relative_ref_score: float
    relative_alt_score: float
    creates_high_scoring_motif: bool


class FastaReference:
    """Small pyfaidx-backed reference accessor."""

    def __init__(self, path: str | Path):
        try:
            from pyfaidx import Fasta
        except ImportError as exc:
            raise MotifDesignError(
                "pyfaidx is required for the mutate stage"
            ) from exc
        self.path = Path(path)
        self._fasta = Fasta(
            str(self.path),
            as_raw=True,
            sequence_always_upper=True,
        )

    def extract(self, chrom: str, start: int, end: int) -> str:
        if start < 0 or end <= start:
            raise MotifDesignError(
                f"Invalid FASTA interval {chrom}:{start}-{end}"
            )
        if chrom not in self._fasta:
            raise MotifDesignError(f"Chromosome {chrom!r} is absent from FASTA")
        sequence = str(self._fasta[chrom][start:end]).upper()
        if len(sequence) != end - start:
            raise MotifDesignError(
                f"Extracted {len(sequence)} bp for {chrom}:{start}-{end}; "
                f"expected {end - start}"
            )
        return sequence

    def chrom_length(self, chrom: str) -> int:
        if chrom not in self._fasta:
            raise MotifDesignError(f"Chromosome {chrom!r} is absent from FASTA")
        return int(len(self._fasta[chrom]))

    def close(self) -> None:
        close = getattr(self._fasta, "close", None)
        if close is not None:
            close()


def reverse_complement(sequence: str) -> str:
    return sequence.upper().translate(COMPLEMENT)[::-1]


def _default_meme_path() -> Path:
    return (
        Path(__file__).resolve().parents[2]
        / "resources"
        / "meme"
        / "jaspar_2024_consensus.meme"
    )


def _motif_header_matches(header: str, motif_name: str) -> bool:
    tokens = re.split(r"[:\s]+", header.upper())
    return motif_name.upper() in tokens


def load_meme_pwm(
    path: str | Path,
    motif_name: str = "CTCF",
    pseudocount: float = 1e-4,
) -> PWMModel:
    """Load the first exact-name-like MEME PWM and derive a PSSM."""

    motif_path = Path(path)
    if not motif_path.exists():
        raise MotifDesignError(f"MEME motif file does not exist: {motif_path}")

    selected_name = ""
    rows: list[list[float]] = []
    collecting = False
    with motif_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if line.startswith("MOTIF "):
                header = line[len("MOTIF ") :].strip()
                collecting = _motif_header_matches(header, motif_name)
                selected_name = header if collecting else ""
                rows = []
                continue
            if not collecting:
                continue
            if line.startswith("letter-probability matrix"):
                continue
            if not line or line.startswith("URL"):
                if rows:
                    break
                continue
            values = [float(value) for value in line.split()]
            if len(values) != 4:
                if rows:
                    break
                continue
            rows.append(values)

    if not rows:
        raise MotifDesignError(
            f"Could not find motif {motif_name!r} in {motif_path}"
        )
    probabilities = np.asarray(rows, dtype=np.float64)
    if np.any(probabilities < 0):
        raise MotifDesignError("PWM probabilities must be non-negative")
    row_totals = probabilities.sum(axis=1, keepdims=True)
    if np.any(row_totals <= 0):
        raise MotifDesignError("PWM rows must have positive probability mass")
    probabilities = probabilities / row_totals
    adjusted = probabilities + pseudocount
    adjusted = adjusted / adjusted.sum(axis=1, keepdims=True)
    pssm = np.log2(adjusted / 0.25)
    return PWMModel(
        name=selected_name,
        probabilities=probabilities,
        pssm=pssm,
        min_score=float(pssm.min(axis=1).sum()),
        max_score=float(pssm.max(axis=1).sum()),
    )


def score_oriented_sequence(sequence: str, pwm: PWMModel) -> float:
    sequence = sequence.upper()
    if len(sequence) != pwm.width or any(base not in BASE_INDEX for base in sequence):
        return float("-inf")
    return float(
        sum(pwm.pssm[offset, BASE_INDEX[base]] for offset, base in enumerate(sequence))
    )


def scan_motif_hits(
    sequence: str,
    genomic_start: int,
    pwm: PWMModel,
    min_relative_score: float | None = None,
) -> list[MotifHit]:
    """Scan both strands and return deterministic genomic-coordinate hits."""

    sequence = sequence.upper()
    hits: list[MotifHit] = []
    for offset in range(0, len(sequence) - pwm.width + 1):
        window = sequence[offset : offset + pwm.width]
        for strand, oriented in (("+", window), ("-", reverse_complement(window))):
            score = score_oriented_sequence(oriented, pwm)
            if not np.isfinite(score):
                continue
            relative = pwm.relative_score(score)
            if min_relative_score is not None and relative < min_relative_score:
                continue
            hits.append(
                MotifHit(
                    start=genomic_start + offset,
                    end=genomic_start + offset + pwm.width,
                    strand=strand,
                    score=score,
                    relative_score=relative,
                    genomic_sequence=window,
                )
            )
    return sorted(
        hits,
        key=lambda hit: (-hit.score, hit.start, hit.strand),
    )


def _choose_site_hit(
    site_row: Any,
    site_sequence: str,
    pwm: PWMModel,
    min_relative_score: float,
) -> MotifHit | None:
    hits = scan_motif_hits(
        site_sequence,
        int(site_row.start),
        pwm,
        min_relative_score=min_relative_score,
    )
    if not hits:
        return None
    annotated_center = getattr(site_row, "CTCF_motif_center", np.nan)
    annotated = bool(getattr(site_row, "has_CTCF_motif", False))
    if annotated and pd.notna(annotated_center):
        center = float(annotated_center)
        return min(
            hits,
            key=lambda hit: (abs(hit.center - center), -hit.score, hit.start),
        )
    return hits[0]


def _oriented_sequence(hit: MotifHit) -> str:
    if hit.strand == "+":
        return hit.genomic_sequence
    return reverse_complement(hit.genomic_sequence)


def _genomic_alt_sequence(
    oriented_alt: Sequence[str],
    strand: str,
) -> str:
    joined = "".join(oriented_alt)
    return joined if strand == "+" else reverse_complement(joined)


def _edited_genomic_positions(
    hit: MotifHit,
    oriented_offsets: Sequence[int],
) -> Tuple[int, ...]:
    if hit.strand == "+":
        return tuple(sorted(hit.start + offset for offset in oriented_offsets))
    return tuple(
        sorted(hit.end - 1 - offset for offset in oriented_offsets)
    )


def _gc_delta_count(ref_sequence: str, alt_sequence: str) -> int:
    ref_gc = sum(base in {"G", "C"} for base in ref_sequence)
    alt_gc = sum(base in {"G", "C"} for base in alt_sequence)
    return int(alt_gc - ref_gc)


def _apply_equal_length_edit(
    sequence: str,
    sequence_start: int,
    edit_start: int,
    edit_end: int,
    ref_edit: str,
    alt_edit: str,
) -> str:
    if len(ref_edit) != edit_end - edit_start or len(alt_edit) != len(ref_edit):
        raise MotifDesignError(
            f"Edit length mismatch at {edit_start}-{edit_end}"
        )
    relative_start = edit_start - sequence_start
    relative_end = edit_end - sequence_start
    if relative_start < 0 or relative_end > len(sequence):
        raise MotifDesignError(
            f"Edit {edit_start}-{edit_end} is outside extracted context"
        )
    observed = sequence[relative_start:relative_end]
    if observed != ref_edit:
        raise MotifDesignError(
            f"Reference mismatch at {edit_start}-{edit_end}: "
            f"expected {ref_edit}, observed {observed}"
        )
    return sequence[:relative_start] + alt_edit + sequence[relative_end:]


def _hit_keys(
    sequence: str,
    genomic_start: int,
    pwm: PWMModel,
    threshold: float,
) -> set[Tuple[int, int, str]]:
    return {
        (hit.start, hit.end, hit.strand)
        for hit in scan_motif_hits(
            sequence,
            genomic_start,
            pwm,
            min_relative_score=threshold,
        )
    }


def _creates_new_high_scoring_motif(
    ref_context: str,
    alt_context: str,
    context_start: int,
    pwm: PWMModel,
    threshold: float,
) -> bool:
    ref_hits = _hit_keys(ref_context, context_start, pwm, threshold)
    alt_hits = _hit_keys(alt_context, context_start, pwm, threshold)
    return bool(alt_hits - ref_hits)


def _positions_overlap_high_scoring_motif(
    sequence: str,
    genomic_start: int,
    positions: Sequence[int],
    pwm: PWMModel,
    threshold: float,
) -> bool:
    high_scoring_hits = scan_motif_hits(
        sequence,
        genomic_start,
        pwm,
        min_relative_score=threshold,
    )
    return any(
        hit.start <= position < hit.end
        for hit in high_scoring_hits
        for position in positions
    )


def _context_for_edit(
    fasta: FastaReference,
    chrom: str,
    edit_start: int,
    edit_end: int,
    flank_bp: int,
) -> Tuple[int, str]:
    chrom_length = fasta.chrom_length(chrom)
    start = max(0, edit_start - flank_bp)
    end = min(chrom_length, edit_end + flank_bp)
    return start, fasta.extract(chrom, start, end)


def _design_motif_disruption(
    *,
    fasta: FastaReference,
    chrom: str,
    hit: MotifHit,
    pwm: PWMModel,
    config: RunConfig,
) -> DesignedEdit | None:
    oriented_ref = _oriented_sequence(hit)
    contributions: list[Tuple[float, int, str]] = []
    for offset, ref_base in enumerate(oriented_ref):
        ref_index = BASE_INDEX[ref_base]
        alternative_scores = [
            (pwm.pssm[offset, index], base)
            for index, base in enumerate(BASES)
            if index != ref_index
        ]
        min_score, min_base = min(
            alternative_scores,
            key=lambda item: (item[0], item[1]),
        )
        contribution = float(pwm.pssm[offset, ref_index] - min_score)
        contributions.append((contribution, offset, min_base))
    contributions.sort(key=lambda item: (-item[0], item[1]))

    context_start, ref_context = _context_for_edit(
        fasta,
        chrom,
        hit.start,
        hit.end,
        config.mutation.motif_rescan_flank_bp,
    )
    for edit_count in range(
        config.mutation.positions_per_motif,
        config.mutation.max_positions_per_motif + 1,
    ):
        selected = contributions[:edit_count]
        oriented_alt = list(oriented_ref)
        offsets = []
        for _, offset, alt_base in selected:
            oriented_alt[offset] = alt_base
            offsets.append(offset)
        genomic_alt = _genomic_alt_sequence(oriented_alt, hit.strand)
        alt_score = score_oriented_sequence("".join(oriented_alt), pwm)
        alt_relative = pwm.relative_score(alt_score)
        if alt_relative >= config.mutation.min_relative_motif_score:
            continue
        alt_context = _apply_equal_length_edit(
            ref_context,
            context_start,
            hit.start,
            hit.end,
            hit.genomic_sequence,
            genomic_alt,
        )
        creates_new = _creates_new_high_scoring_motif(
            ref_context,
            alt_context,
            context_start,
            pwm,
            config.mutation.min_relative_motif_score,
        )
        if config.mutation.reject_new_high_scoring_motif and creates_new:
            continue
        gc_delta = _gc_delta_count(hit.genomic_sequence, genomic_alt)
        return DesignedEdit(
            edit_start=hit.start,
            edit_end=hit.end,
            ref_sequence=hit.genomic_sequence,
            alt_sequence=genomic_alt,
            edited_positions=_edited_genomic_positions(hit, offsets),
            edited_base_count=edit_count,
            gc_delta_count=gc_delta,
            gc_change=float(gc_delta / edit_count),
            pwm_ref_score=hit.score,
            pwm_alt_score=alt_score,
            relative_ref_score=hit.relative_score,
            relative_alt_score=alt_relative,
            creates_high_scoring_motif=creates_new,
        )
    return None


def _effect_for_substitution(ref_base: str, alt_base: str) -> int:
    return int(alt_base in {"G", "C"}) - int(ref_base in {"G", "C"})


def _allowed_alt_bases(ref_base: str, effect: int) -> list[str]:
    return [
        base
        for base in BASES
        if base != ref_base and _effect_for_substitution(ref_base, base) == effect
    ]


def _stable_order(
    values: Iterable[int],
    *,
    namespace: str,
) -> list[int]:
    def key(value: int) -> Tuple[str, int]:
        digest = hashlib.sha256(f"{namespace}:{value}".encode("utf-8")).hexdigest()
        return digest, value

    return sorted(values, key=key)


def _matched_control_alt(
    ref_sequence: str,
    target_effects: Sequence[int],
    *,
    namespace: str,
) -> Tuple[str, Tuple[int, ...]] | None:
    available_positions = set(range(len(ref_sequence)))
    alt = list(ref_sequence)
    selected: list[int] = []
    requirements = sorted(target_effects, key=lambda effect: (effect == 0, effect))
    for requirement_index, effect in enumerate(requirements):
        candidate_positions = [
            position
            for position in available_positions
            if _allowed_alt_bases(ref_sequence[position], effect)
        ]
        if not candidate_positions:
            return None
        ordered_positions = _stable_order(
            candidate_positions,
            namespace=f"{namespace}:effect{effect}:req{requirement_index}",
        )
        position = ordered_positions[0]
        alternatives = _allowed_alt_bases(ref_sequence[position], effect)
        alternative_order = _stable_order(
            range(len(alternatives)),
            namespace=f"{namespace}:position{position}",
        )
        alt[position] = alternatives[alternative_order[0]]
        selected.append(position)
        available_positions.remove(position)
    return "".join(alt), tuple(sorted(selected))


def _windows_overlap(
    first_start: int,
    first_end: int,
    second_start: int,
    second_end: int,
) -> bool:
    return max(first_start, second_start) < min(first_end, second_end)


def _best_window_score(sequence: str, pwm: PWMModel) -> Tuple[float, float]:
    plus = score_oriented_sequence(sequence, pwm)
    minus = score_oriented_sequence(reverse_complement(sequence), pwm)
    score = max(plus, minus)
    return score, pwm.relative_score(score)


def _design_local_controls(
    *,
    fasta: FastaReference,
    chrom: str,
    site_id: str,
    hit: MotifHit,
    disruption: DesignedEdit,
    pwm: PWMModel,
    config: RunConfig,
) -> list[DesignedEdit]:
    chrom_length = fasta.chrom_length(chrom)
    width = pwm.width
    lower = max(0, hit.start - config.mutation.max_control_distance_bp)
    upper = min(
        chrom_length - width,
        hit.end + config.mutation.max_control_distance_bp,
    )
    motif_center = hit.center
    candidate_starts = sorted(
        range(lower, upper + 1),
        key=lambda start: (
            abs((start + width // 2) - motif_center),
            start,
        ),
    )
    target_effects = [
        _effect_for_substitution(
            disruption.ref_sequence[position - disruption.edit_start],
            disruption.alt_sequence[position - disruption.edit_start],
        )
        for position in disruption.edited_positions
    ]
    controls: list[DesignedEdit] = []
    used_intervals: list[Tuple[int, int]] = []

    for candidate_start in candidate_starts:
        candidate_end = candidate_start + width
        if _windows_overlap(candidate_start, candidate_end, hit.start, hit.end):
            continue
        if any(
            _windows_overlap(candidate_start, candidate_end, used_start, used_end)
            for used_start, used_end in used_intervals
        ):
            continue
        ref_window = fasta.extract(chrom, candidate_start, candidate_end)
        if any(base not in BASE_INDEX for base in ref_window):
            continue
        _, ref_relative = _best_window_score(ref_window, pwm)
        if ref_relative >= config.mutation.min_relative_motif_score:
            continue

        control_rank = len(controls) + 1
        designed = _matched_control_alt(
            ref_window,
            target_effects,
            namespace=f"{site_id}:control{control_rank}:{candidate_start}",
        )
        if designed is None:
            continue
        alt_window, relative_positions = designed
        gc_delta = _gc_delta_count(ref_window, alt_window)
        gc_change = float(gc_delta / len(relative_positions))
        if (
            abs(gc_change - disruption.gc_change)
            > config.mutation.max_gc_change_difference
        ):
            continue

        context_start, ref_context = _context_for_edit(
            fasta,
            chrom,
            candidate_start,
            candidate_end,
            config.mutation.motif_rescan_flank_bp,
        )
        edited_positions = tuple(
            candidate_start + position for position in relative_positions
        )
        if _positions_overlap_high_scoring_motif(
            ref_context,
            context_start,
            edited_positions,
            pwm,
            config.mutation.min_relative_motif_score,
        ):
            continue
        alt_context = _apply_equal_length_edit(
            ref_context,
            context_start,
            candidate_start,
            candidate_end,
            ref_window,
            alt_window,
        )
        creates_new = _creates_new_high_scoring_motif(
            ref_context,
            alt_context,
            context_start,
            pwm,
            config.mutation.min_relative_motif_score,
        )
        if config.mutation.reject_new_high_scoring_motif and creates_new:
            continue
        edit_ref_score, edit_ref_relative = _best_window_score(ref_window, pwm)
        edit_alt_score, edit_alt_relative = _best_window_score(alt_window, pwm)
        controls.append(
            DesignedEdit(
                edit_start=candidate_start,
                edit_end=candidate_end,
                ref_sequence=ref_window,
                alt_sequence=alt_window,
                edited_positions=edited_positions,
                edited_base_count=len(relative_positions),
                gc_delta_count=gc_delta,
                gc_change=gc_change,
                pwm_ref_score=edit_ref_score,
                pwm_alt_score=edit_alt_score,
                relative_ref_score=edit_ref_relative,
                relative_alt_score=edit_alt_relative,
                creates_high_scoring_motif=creates_new,
            )
        )
        used_intervals.append((candidate_start, candidate_end))
        if len(controls) >= config.mutation.local_controls_per_motif:
            break
    return controls


def _design_motif_preserving_control(
    *,
    fasta: FastaReference,
    chrom: str,
    hit: MotifHit,
    edit_count: int,
    pwm: PWMModel,
    config: RunConfig,
) -> DesignedEdit | None:
    oriented_ref = _oriented_sequence(hit)
    candidates: list[Tuple[float, int, str]] = []
    for offset, ref_base in enumerate(oriented_ref):
        ref_index = BASE_INDEX[ref_base]
        alternatives = [
            (abs(pwm.pssm[offset, index] - pwm.pssm[offset, ref_index]), base)
            for index, base in enumerate(BASES)
            if index != ref_index
        ]
        penalty, alt_base = min(alternatives, key=lambda item: (item[0], item[1]))
        candidates.append((float(penalty), offset, alt_base))
    candidates.sort(key=lambda item: (item[0], item[1]))
    selected = candidates[:edit_count]
    oriented_alt = list(oriented_ref)
    offsets = []
    for _, offset, alt_base in selected:
        oriented_alt[offset] = alt_base
        offsets.append(offset)
    genomic_alt = _genomic_alt_sequence(oriented_alt, hit.strand)
    alt_score = score_oriented_sequence("".join(oriented_alt), pwm)
    alt_relative = pwm.relative_score(alt_score)
    if alt_relative < config.mutation.min_relative_motif_score:
        return None

    context_start, ref_context = _context_for_edit(
        fasta,
        chrom,
        hit.start,
        hit.end,
        config.mutation.motif_rescan_flank_bp,
    )
    alt_context = _apply_equal_length_edit(
        ref_context,
        context_start,
        hit.start,
        hit.end,
        hit.genomic_sequence,
        genomic_alt,
    )
    creates_new = _creates_new_high_scoring_motif(
        ref_context,
        alt_context,
        context_start,
        pwm,
        config.mutation.min_relative_motif_score,
    )
    if config.mutation.reject_new_high_scoring_motif and creates_new:
        return None
    gc_delta = _gc_delta_count(hit.genomic_sequence, genomic_alt)
    return DesignedEdit(
        edit_start=hit.start,
        edit_end=hit.end,
        ref_sequence=hit.genomic_sequence,
        alt_sequence=genomic_alt,
        edited_positions=_edited_genomic_positions(hit, offsets),
        edited_base_count=edit_count,
        gc_delta_count=gc_delta,
        gc_change=float(gc_delta / edit_count),
        pwm_ref_score=hit.score,
        pwm_alt_score=alt_score,
        relative_ref_score=hit.relative_score,
        relative_alt_score=alt_relative,
        creates_high_scoring_motif=creates_new,
    )


def _edit_record(
    *,
    site_id: str,
    chrom: str,
    hit: MotifHit,
    edit: DesignedEdit,
    perturbation_type: str,
    control_set_id: str,
    control_rank: int,
) -> dict[str, Any]:
    edit_center = edit.edit_start + (edit.edit_end - edit.edit_start) // 2
    return {
        "perturbation_id": f"{control_set_id}:{perturbation_type}:{control_rank}",
        "site_id": site_id,
        "perturbation_type": perturbation_type,
        "chrom": chrom,
        "edit_start": edit.edit_start,
        "edit_end": edit.edit_end,
        "ref_sequence": edit.ref_sequence,
        "alt_sequence": edit.alt_sequence,
        "motif_strand": hit.strand,
        "motif_start": hit.start,
        "motif_end": hit.end,
        "edited_base_count": edit.edited_base_count,
        "gc_change": edit.gc_change,
        "pwm_ref_score": edit.pwm_ref_score,
        "pwm_alt_score": edit.pwm_alt_score,
        "pwm_score_change": edit.pwm_alt_score - edit.pwm_ref_score,
        "creates_high_scoring_ctcf_motif": edit.creates_high_scoring_motif,
        "control_set_id": control_set_id,
        "control_rank": control_rank,
        "motif_relative_ref_score": edit.relative_ref_score,
        "motif_relative_alt_score": edit.relative_alt_score,
        "edited_positions": ",".join(
            str(position) for position in edit.edited_positions
        ),
        "gc_delta_count": edit.gc_delta_count,
        "distance_to_motif_bp": edit_center - hit.center,
        "mutation_status": "ready",
        "mutation_reason": "",
    }


def design_mutation_table(
    sites: pd.DataFrame,
    *,
    fasta: FastaReference,
    pwm: PWMModel,
    config: RunConfig,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Design motif disruptions and matched controls for sampled sites."""

    if sites.empty:
        raise MotifDesignError(
            "site_metadata.tsv contains no sampled sites; mutation design "
            "requires at least one site"
        )

    mutation_records: list[dict[str, Any]] = []
    annotation_records: list[dict[str, Any]] = []
    failure_counts: dict[str, int] = {}

    def count_failure(reason: str) -> None:
        failure_counts[reason] = failure_counts.get(reason, 0) + 1

    for row in sites.itertuples(index=False):
        site_id = str(row.site_id)
        chrom = str(row.chrom)
        site_start = int(row.start)
        site_end = int(row.end)
        site_sequence = fasta.extract(chrom, site_start, site_end)
        hit = _choose_site_hit(
            row,
            site_sequence,
            pwm,
            config.mutation.min_relative_motif_score,
        )
        base_annotation = {
            "site_id": site_id,
            "chrom": chrom,
            "site_start": site_start,
            "site_end": site_end,
        }
        if hit is None:
            reason = "no_high_confidence_ctcf_motif"
            count_failure(reason)
            annotation_records.append(
                {
                    **base_annotation,
                    "scan_status": "no_high_confidence_motif",
                    "motif_start": pd.NA,
                    "motif_end": pd.NA,
                    "motif_center": pd.NA,
                    "motif_strand": ".",
                    "motif_ref_sequence": "",
                    "pwm_ref_score": np.nan,
                    "motif_relative_score": np.nan,
                    "mutation_available": False,
                    "local_control_count": 0,
                    "annotation_reason": reason,
                }
            )
            continue

        disruption = _design_motif_disruption(
            fasta=fasta,
            chrom=chrom,
            hit=hit,
            pwm=pwm,
            config=config,
        )
        if disruption is None:
            reason = "motif_disruption_failed_threshold_or_rescan"
            count_failure(reason)
            annotation_records.append(
                {
                    **base_annotation,
                    "scan_status": "high_confidence_motif",
                    "motif_start": hit.start,
                    "motif_end": hit.end,
                    "motif_center": hit.center,
                    "motif_strand": hit.strand,
                    "motif_ref_sequence": hit.genomic_sequence,
                    "pwm_ref_score": hit.score,
                    "motif_relative_score": hit.relative_score,
                    "mutation_available": False,
                    "local_control_count": 0,
                    "annotation_reason": reason,
                }
            )
            continue

        controls = _design_local_controls(
            fasta=fasta,
            chrom=chrom,
            site_id=site_id,
            hit=hit,
            disruption=disruption,
            pwm=pwm,
            config=config,
        )
        if len(controls) < config.mutation.min_local_controls_per_motif:
            reason = "insufficient_matched_local_controls"
            count_failure(reason)
            annotation_records.append(
                {
                    **base_annotation,
                    "scan_status": "high_confidence_motif",
                    "motif_start": hit.start,
                    "motif_end": hit.end,
                    "motif_center": hit.center,
                    "motif_strand": hit.strand,
                    "motif_ref_sequence": hit.genomic_sequence,
                    "pwm_ref_score": hit.score,
                    "motif_relative_score": hit.relative_score,
                    "mutation_available": False,
                    "local_control_count": len(controls),
                    "annotation_reason": reason,
                }
            )
            continue

        control_set_id = f"{site_id}:ctcf"
        site_records = [
            _edit_record(
                site_id=site_id,
                chrom=chrom,
                hit=hit,
                edit=disruption,
                perturbation_type="ctcf_motif_disruption",
                control_set_id=control_set_id,
                control_rank=0,
            )
        ]
        site_records.extend(
            _edit_record(
                site_id=site_id,
                chrom=chrom,
                hit=hit,
                edit=control,
                perturbation_type="local_non_motif_control",
                control_set_id=control_set_id,
                control_rank=rank,
            )
            for rank, control in enumerate(controls, start=1)
        )

        preserving_status = ""
        if config.mutation.include_motif_preserving_control:
            preserving = _design_motif_preserving_control(
                fasta=fasta,
                chrom=chrom,
                hit=hit,
                edit_count=disruption.edited_base_count,
                pwm=pwm,
                config=config,
            )
            if preserving is None:
                preserving_status = "motif_preserving_control_unavailable"
                count_failure(preserving_status)
            else:
                site_records.append(
                    _edit_record(
                        site_id=site_id,
                        chrom=chrom,
                        hit=hit,
                        edit=preserving,
                        perturbation_type="motif_preserving_control",
                        control_set_id=control_set_id,
                        control_rank=len(controls) + 1,
                    )
                )

        mutation_records.extend(site_records)
        annotation_reason = (
            "ready_partial_control_set"
            if len(controls) < config.mutation.local_controls_per_motif
            else "ready"
        )
        if preserving_status:
            annotation_reason = f"{annotation_reason};{preserving_status}"
        annotation_records.append(
            {
                **base_annotation,
                "scan_status": "high_confidence_motif",
                "motif_start": hit.start,
                "motif_end": hit.end,
                "motif_center": hit.center,
                "motif_strand": hit.strand,
                "motif_ref_sequence": hit.genomic_sequence,
                "pwm_ref_score": hit.score,
                "motif_relative_score": hit.relative_score,
                "mutation_available": True,
                "local_control_count": len(controls),
                "annotation_reason": annotation_reason,
            }
        )

    mutation_columns = list(MUTATION_SCHEMA.required_columns) + list(
        MUTATION_SCHEMA.optional_columns
    )
    mutations = pd.DataFrame.from_records(mutation_records, columns=mutation_columns)
    annotations = pd.DataFrame.from_records(annotation_records)

    qc_rows = [
        {
            "metric": "sampled_site_count",
            "value": int(len(sites)),
            "details": "Input sampled paper sites",
        },
        {
            "metric": "high_confidence_motif_count",
            "value": int(annotations["scan_status"].eq("high_confidence_motif").sum()),
            "details": (
                "Relative PSSM threshold >= "
                f"{config.mutation.min_relative_motif_score}"
            ),
        },
        {
            "metric": "mutation_available_site_count",
            "value": int(annotations["mutation_available"].sum()),
            "details": "Sites with disruption and at least the minimum controls",
        },
        {
            "metric": "perturbation_row_count",
            "value": int(len(mutations)),
            "details": "Ready motif and control perturbations",
        },
    ]
    if not mutations.empty:
        for perturbation_type, count in (
            mutations["perturbation_type"].value_counts().sort_index().items()
        ):
            qc_rows.append(
                {
                    "metric": f"perturbation_type:{perturbation_type}",
                    "value": int(count),
                    "details": "Ready perturbation rows",
                }
            )
    for reason, count in sorted(failure_counts.items()):
        qc_rows.append(
            {
                "metric": f"failure:{reason}",
                "value": int(count),
                "details": "Site-level or optional-control design failure",
            }
        )
    return mutations, annotations, pd.DataFrame.from_records(qc_rows)


def run(
    *,
    config: RunConfig,
    config_path: str | Path,
    allow_provisional_labels: bool,
    overwrite_stage: bool,
) -> None:
    """Execute the mutate stage."""

    root = output_dir(config)
    site_metadata_path = root / "site_metadata.tsv"
    mutation_path = root / "mutation_table.tsv"
    annotation_path = root / "motif_annotation_table.tsv"
    qc_path = root / "mutation_qc_report.tsv"
    require_completed_stage(config, config_path, "sample")
    if not site_metadata_path.exists():
        raise MotifDesignError(
            f"Run the sample stage first; missing {site_metadata_path}"
        )
    if overwrite_stage:
        reject_completed_downstream_stages(
            config,
            config_path,
            ("infer", "features", "analyze", "report"),
        )
    prepare_stage_outputs(
        [mutation_path, annotation_path, qc_path],
        overwrite_stage,
    )

    sites = pd.read_csv(site_metadata_path, sep="\t", low_memory=False)
    sites, _ = validate_paper_labels(
        sites,
        dic_mvalue_threshold=config.labels.dic_mvalue_threshold,
    )
    statuses = set(sites["label_status"])
    if statuses != {"canonical"} and not allow_provisional_labels:
        raise MotifDesignError(
            "Designing mutations for provisional labels requires "
            "--allow-provisional-labels"
        )

    if not config.paths.fasta:
        raise MotifDesignError("paths.fasta is required for the mutate stage")
    fasta_path = Path(config.paths.fasta)
    if not fasta_path.exists():
        raise MotifDesignError(f"Reference FASTA does not exist: {fasta_path}")
    motif_path = (
        Path(config.paths.motif_meme)
        if config.paths.motif_meme
        else _default_meme_path()
    )
    pwm = load_meme_pwm(
        motif_path,
        motif_name=config.mutation.motif_name,
    )
    fasta = FastaReference(fasta_path)
    try:
        mutations, annotations, qc = design_mutation_table(
            sites,
            fasta=fasta,
            pwm=pwm,
            config=config,
        )
    finally:
        fasta.close()

    mutations.to_csv(mutation_path, sep="\t", index=False)
    annotations.to_csv(annotation_path, sep="\t", index=False)
    qc.to_csv(qc_path, sep="\t", index=False)

    manifest_inputs = [site_metadata_path, motif_path]
    fasta_index = Path(f"{fasta_path}.fai")
    if fasta_index.exists():
        manifest_inputs.append(fasta_index)
    fasta_stat = fasta_path.stat()
    label_status = "canonical" if statuses == {"canonical"} else "provisional"
    record_stage(
        config=config,
        config_path=config_path,
        stage="mutate",
        inputs=manifest_inputs,
        outputs=[mutation_path, annotation_path, qc_path],
        label_status=label_status,
        metadata={
            "fasta_path": str(fasta_path),
            "fasta_size_bytes": fasta_stat.st_size,
            "fasta_mtime_ns": fasta_stat.st_mtime_ns,
            "pwm_name": pwm.name,
            "pwm_width": pwm.width,
            "min_relative_motif_score": (
                config.mutation.min_relative_motif_score
            ),
            "mutation_available_site_count": int(
                annotations["mutation_available"].sum()
            ),
            "perturbation_row_count": int(len(mutations)),
        },
    )
