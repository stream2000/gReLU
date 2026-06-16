"""Small, manifest-driven batch ISM helpers.

This module generalizes the validated MREG three-region workflow without
introducing a workflow framework. Sites, tracks, and optional readouts remain
plain TSV files.
"""

from __future__ import annotations

from dataclasses import dataclass
import importlib
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd

from grelu.interpret.ism.motifs import (
    BASE_INDEX,
    DesignedEdit,
    FastaReference,
    MotifHit,
    PWMModel,
    _apply_equal_length_edit,
    _best_window_score,
    _context_for_edit,
    _creates_new_high_scoring_motif,
    _design_local_controls,
    _design_motif_disruption,
    _effect_for_substitution,
    _matched_control_alt,
    _positions_overlap_high_scoring_motif,
    load_meme_pwm,
    reverse_complement,
    scan_motif_hits,
    score_oriented_sequence,
)

AG_INPUT_LEN = 1_048_576
RESOLUTION = 128
CONTROL_STRATEGIES = {
    "local_non_motif_control",
    "local_sequence_control",
    "local_panel_free_control",
}
SITE_REQUIRED_COLUMNS = {
    "site_id",
    "site_class",
    "chrom",
    "start",
    "end",
    "mutation_strategy",
}
TRACK_REQUIRED_COLUMNS = {"track_id", "output_type", "target_name", "biosample"}
MutationStrategy = Callable[
    [Any, "MutationDesignContext"], "MutationDesign"
]
SummaryStrategy = Callable[
    [np.ndarray, np.ndarray, str, Sequence[int], int], dict[str, float]
]
MUTATION_STRATEGIES: dict[str, MutationStrategy] = {}
SUMMARY_STRATEGIES: dict[str, SummaryStrategy] = {}


@dataclass
class _MutationSettings:
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
class _MutationConfig:
    mutation: _MutationSettings


@dataclass
class MutationDesignContext:
    fasta: FastaReference
    pwm: PWMModel
    config: _MutationConfig
    min_relative_motif_score: float
    sequence_controls: int
    max_control_distance_bp: int


@dataclass
class MutationDesign:
    experimental: dict[str, Any] | None
    controls: list[dict[str, Any]]
    region_updates: dict[str, Any]
    messages: list[str]


def register_mutation_strategy(
    name: str,
) -> Callable[[MutationStrategy], MutationStrategy]:
    """Register a sites.tsv mutation_strategy implementation."""

    normalized = name.strip().lower()

    def decorator(function: MutationStrategy) -> MutationStrategy:
        if normalized in MUTATION_STRATEGIES:
            raise ValueError(f"Mutation strategy already registered: {normalized}")
        MUTATION_STRATEGIES[normalized] = function
        return function

    return decorator


def register_summary_strategy(
    name: str,
) -> Callable[[SummaryStrategy], SummaryStrategy]:
    """Register a population feature summarizer."""

    normalized = name.strip().lower()

    def decorator(function: SummaryStrategy) -> SummaryStrategy:
        if normalized in SUMMARY_STRATEGIES:
            raise ValueError(f"Summary strategy already registered: {normalized}")
        SUMMARY_STRATEGIES[normalized] = function
        return function

    return decorator


def load_strategy_modules(module_names: Sequence[str]) -> None:
    """Import optional modules whose decorators extend the strategy registries."""
    for module_name in module_names:
        importlib.import_module(module_name)


def _require_columns(frame: pd.DataFrame, required: set[str], name: str) -> None:
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{name} is missing columns: {missing}")


def _optional_int(row: Any, name: str, default: int = -1) -> int:
    value = getattr(row, name, default)
    if value is None or pd.isna(value) or str(value).strip() == "":
        return default
    return int(value)


def _optional_text(row: Any, name: str, default: str = "") -> str:
    value = getattr(row, name, default)
    if value is None or pd.isna(value):
        return default
    return str(value).strip()


def _optional_float(row: Any, name: str, default: float) -> float:
    value = getattr(row, name, default)
    if value is None or pd.isna(value) or str(value).strip() == "":
        return float(default)
    return float(value)


def _substitution(sequence: str) -> str:
    mapping = str.maketrans("ACGT", "GTAC")
    return sequence.upper().translate(mapping)


def _gc_delta(ref: str, alt: str) -> int:
    return sum(base in "GC" for base in alt) - sum(base in "GC" for base in ref)


def _local_sequence_controls(
    *,
    fasta: FastaReference,
    chrom: str,
    target_start: int,
    target_end: int,
    target_ref: str,
    target_alt: str,
    excluded_start: int,
    excluded_end: int,
    max_distance_bp: int,
    count: int,
) -> list[dict[str, Any]]:
    edit_length = target_end - target_start
    target_center = (target_start + target_end) // 2
    target_delta = _gc_delta(target_ref, target_alt)
    lower = max(0, target_start - max_distance_bp)
    upper = min(
        fasta.chrom_length(chrom) - edit_length,
        target_end + max_distance_bp,
    )
    candidates = sorted(
        range(lower, upper + 1),
        key=lambda start: (
            abs((start + edit_length // 2) - target_center),
            start,
        ),
    )
    controls = []
    for start in candidates:
        end = start + edit_length
        if max(start, target_start) < min(end, target_end):
            continue
        if max(start, excluded_start) < min(end, excluded_end):
            continue
        ref = fasta.extract(chrom, start, end)
        if any(base not in "ACGT" for base in ref):
            continue
        alt = _substitution(ref)
        if _gc_delta(ref, alt) != target_delta:
            continue
        controls.append(
            {
                "edit_start": start,
                "edit_end": end,
                "ref_sequence": ref,
                "alt_sequence": alt,
            }
        )
        if len(controls) >= count:
            break
    return controls


def _validate_context(fasta: FastaReference, chrom: str, center: int) -> bool:
    half = AG_INPUT_LEN // 2
    return center - half >= 0 and center + half <= fasta.chrom_length(chrom)


def _load_row_pwm(row: Any) -> PWMModel:
    motif_path = _optional_text(row, "motif_path")
    motif_name = _optional_text(row, "motif_name")
    if not motif_path or not motif_name:
        raise ValueError(
            "motif_pwm_disruption requires motif_path and motif_name"
        )
    return load_meme_pwm(motif_path, motif_name=motif_name)


def _row_panel_pwms(row: Any, target_pwm: PWMModel) -> list[PWMModel]:
    motif_path = _optional_text(row, "panel_motif_path")
    motif_names = [
        value.strip()
        for value in _optional_text(row, "panel_motif_names").split(";")
        if value.strip()
    ]
    if not motif_path or not motif_names:
        return [target_pwm]
    pwms = [
        load_meme_pwm(motif_path, motif_name=name)
        for name in motif_names
    ]
    if target_pwm.name not in {pwm.name for pwm in pwms}:
        pwms.append(target_pwm)
    return pwms


def _row_motif_hit(
    row: Any,
    context: MutationDesignContext,
    pwm: PWMModel,
) -> MotifHit:
    chrom = str(row.chrom)
    start = _optional_int(row, "motif_start")
    end = _optional_int(row, "motif_end")
    strand = _optional_text(row, "motif_strand")
    if start < 0 or end <= start or strand not in {"+", "-"}:
        raise ValueError(
            "motif_pwm_disruption requires motif_start/end and motif_strand"
        )
    sequence = context.fasta.extract(chrom, start, end)
    expected = _optional_text(row, "motif_sequence").upper()
    if expected and sequence != expected:
        raise ValueError(
            f"Motif FASTA mismatch for {row.site_id}: "
            f"expected {expected}, observed {sequence}"
        )
    if len(sequence) != pwm.width:
        raise ValueError(
            f"Motif width mismatch for {row.site_id}: "
            f"interval={len(sequence)}, PWM={pwm.width}"
        )
    oriented = sequence if strand == "+" else reverse_complement(sequence)
    score = score_oriented_sequence(oriented, pwm)
    relative_score = pwm.relative_score(score)
    threshold = _optional_float(
        row,
        "min_relative_motif_score",
        context.min_relative_motif_score,
    )
    if relative_score < threshold:
        raise ValueError(
            f"Motif score below threshold for {row.site_id}: "
            f"{relative_score:.4f} < {threshold:.4f}"
        )
    return MotifHit(
        start=start,
        end=end,
        strand=strand,
        score=score,
        relative_score=relative_score,
        genomic_sequence=sequence,
    )


def _panel_creates_new_motif(
    *,
    fasta: FastaReference,
    chrom: str,
    edit: DesignedEdit,
    panel_pwms: Sequence[PWMModel],
    threshold: float,
    flank_bp: int,
) -> bool:
    context_start, ref_context = _context_for_edit(
        fasta,
        chrom,
        edit.edit_start,
        edit.edit_end,
        flank_bp,
    )
    alt_context = _apply_equal_length_edit(
        ref_context,
        context_start,
        edit.edit_start,
        edit.edit_end,
        edit.ref_sequence,
        edit.alt_sequence,
    )
    return any(
        _creates_new_high_scoring_motif(
            ref_context,
            alt_context,
            context_start,
            pwm,
            threshold,
        )
        for pwm in panel_pwms
    )


def _design_panel_matched_controls(
    *,
    row: Any,
    context: MutationDesignContext,
    hit: MotifHit,
    disruption: DesignedEdit,
    target_pwm: PWMModel,
    panel_pwms: Sequence[PWMModel],
    threshold: float,
) -> list[DesignedEdit]:
    chrom = str(row.chrom)
    width = target_pwm.width
    region_start = _optional_int(row, "control_region_start", int(row.start))
    region_end = _optional_int(row, "control_region_end", int(row.end))
    region_start = max(0, region_start)
    region_end = min(context.fasta.chrom_length(chrom), region_end)
    if region_end - region_start < width:
        return []
    candidate_starts = sorted(
        range(region_start, region_end - width + 1),
        key=lambda start: (
            abs((start + width // 2) - hit.center),
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
    region_sequence = context.fasta.extract(chrom, region_start, region_end)
    controls = []
    for candidate_start in candidate_starts:
        candidate_end = candidate_start + width
        if max(candidate_start, hit.start) < min(candidate_end, hit.end):
            continue
        ref_window = context.fasta.extract(
            chrom, candidate_start, candidate_end
        )
        if any(base not in BASE_INDEX for base in ref_window):
            continue
        designed = _matched_control_alt(
            ref_window,
            target_effects,
            namespace=f"{row.site_id}:{candidate_start}",
        )
        if designed is None:
            continue
        alt_window, relative_positions = designed
        edited_positions = tuple(
            candidate_start + position for position in relative_positions
        )
        if any(
            _positions_overlap_high_scoring_motif(
                region_sequence,
                region_start,
                edited_positions,
                pwm,
                threshold,
            )
            for pwm in panel_pwms
        ):
            continue
        ref_score, ref_relative = _best_window_score(
            ref_window, target_pwm
        )
        alt_score, alt_relative = _best_window_score(
            alt_window, target_pwm
        )
        control = DesignedEdit(
            edit_start=candidate_start,
            edit_end=candidate_end,
            ref_sequence=ref_window,
            alt_sequence=alt_window,
            edited_positions=edited_positions,
            edited_base_count=len(relative_positions),
            gc_delta_count=_gc_delta(ref_window, alt_window),
            gc_change=float(
                _gc_delta(ref_window, alt_window) / len(relative_positions)
            ),
            pwm_ref_score=ref_score,
            pwm_alt_score=alt_score,
            relative_ref_score=ref_relative,
            relative_alt_score=alt_relative,
            creates_high_scoring_motif=False,
        )
        if _panel_creates_new_motif(
            fasta=context.fasta,
            chrom=chrom,
            edit=control,
            panel_pwms=panel_pwms,
            threshold=threshold,
            flank_bp=context.config.mutation.motif_rescan_flank_bp,
        ):
            continue
        controls.append(control)
        if len(controls) >= context.sequence_controls:
            break
    return controls


@register_mutation_strategy("ctcf_pwm_disruption")
def _prepare_ctcf_pwm_disruption(
    row: Any,
    context: MutationDesignContext,
) -> MutationDesign:
    chrom = str(row.chrom)
    start = int(row.start)
    end = int(row.end)
    site_id = str(row.site_id)
    sequence = context.fasta.extract(chrom, start, end)
    hits = scan_motif_hits(
        sequence,
        start,
        context.pwm,
        min_relative_score=context.min_relative_motif_score,
    )
    if not hits:
        return MutationDesign(None, [], {}, ["no CTCF motif above threshold"])
    hit = hits[0]
    disruption = _design_motif_disruption(
        fasta=context.fasta,
        chrom=chrom,
        hit=hit,
        pwm=context.pwm,
        config=context.config,
    )
    if disruption is None:
        return MutationDesign(None, [], {}, ["CTCF disruption design failed"])
    experimental = {
        "mutation_strategy": "ctcf_pwm_disruption",
        "edit_start": disruption.edit_start,
        "edit_end": disruption.edit_end,
        "ref_sequence": disruption.ref_sequence,
        "alt_sequence": disruption.alt_sequence,
        "target_feature": "CTCF_motif",
        "ref_pwm_score": disruption.pwm_ref_score,
        "alt_pwm_score": disruption.pwm_alt_score,
    }
    controls = [
        {
            "mutation_strategy": "local_non_motif_control",
            "edit_start": control.edit_start,
            "edit_end": control.edit_end,
            "ref_sequence": control.ref_sequence,
            "alt_sequence": control.alt_sequence,
            "target_feature": "non_motif_control",
            "ref_pwm_score": control.pwm_ref_score,
            "alt_pwm_score": control.pwm_alt_score,
        }
        for control in _design_local_controls(
            fasta=context.fasta,
            chrom=chrom,
            site_id=site_id,
            hit=hit,
            disruption=disruption,
            pwm=context.pwm,
            config=context.config,
        )
    ]
    return MutationDesign(
        experimental,
        controls,
        {
            "anchor": (disruption.edit_start + disruption.edit_end) // 2,
            "motif_start": hit.start,
            "motif_end": hit.end,
            "motif_strand": hit.strand,
            "motif_relative_score": hit.relative_score,
        },
        [],
    )


@register_mutation_strategy("motif_pwm_disruption")
def _prepare_motif_pwm_disruption(
    row: Any,
    context: MutationDesignContext,
) -> MutationDesign:
    chrom = str(row.chrom)
    target_pwm = _load_row_pwm(row)
    panel_pwms = _row_panel_pwms(row, target_pwm)
    hit = _row_motif_hit(row, context, target_pwm)
    threshold = _optional_float(
        row,
        "min_relative_motif_score",
        context.min_relative_motif_score,
    )
    settings = _MutationSettings(
        positions_per_motif=3,
        max_positions_per_motif=3,
        min_relative_motif_score=threshold,
        motif_rescan_flank_bp=context.config.mutation.motif_rescan_flank_bp,
        reject_new_high_scoring_motif=True,
        max_control_distance_bp=context.max_control_distance_bp,
        max_gc_change_difference=0.0,
        local_controls_per_motif=context.sequence_controls,
        min_local_controls_per_motif=context.sequence_controls,
        motif_name=target_pwm.name,
    )
    disruption = _design_motif_disruption(
        fasta=context.fasta,
        chrom=chrom,
        hit=hit,
        pwm=target_pwm,
        config=_MutationConfig(mutation=settings),
    )
    if disruption is None:
        return MutationDesign(
            None,
            [],
            {},
            ["three-base motif disruption design failed"],
        )
    if _panel_creates_new_motif(
        fasta=context.fasta,
        chrom=chrom,
        edit=disruption,
        panel_pwms=panel_pwms,
        threshold=threshold,
        flank_bp=settings.motif_rescan_flank_bp,
    ):
        return MutationDesign(
            None,
            [],
            {},
            ["motif disruption creates a new high-scoring panel motif"],
        )
    controls = _design_panel_matched_controls(
        row=row,
        context=context,
        hit=hit,
        disruption=disruption,
        target_pwm=target_pwm,
        panel_pwms=panel_pwms,
        threshold=threshold,
    )
    if len(controls) < context.sequence_controls:
        return MutationDesign(
            None,
            [],
            {},
            ["no same-peak panel-free matched control"],
        )
    target_feature = _optional_text(
        row, "target_feature", f"{target_pwm.name}_motif"
    )
    experimental = {
        "mutation_strategy": "motif_pwm_disruption",
        "edit_start": disruption.edit_start,
        "edit_end": disruption.edit_end,
        "ref_sequence": disruption.ref_sequence,
        "alt_sequence": disruption.alt_sequence,
        "target_feature": target_feature,
        "ref_pwm_score": disruption.pwm_ref_score,
        "alt_pwm_score": disruption.pwm_alt_score,
        "ref_pwm_relative_score": disruption.relative_ref_score,
        "alt_pwm_relative_score": disruption.relative_alt_score,
        "edited_base_count": disruption.edited_base_count,
        "gc_delta_count": disruption.gc_delta_count,
    }
    control_rows = [
        {
            "mutation_strategy": "local_panel_free_control",
            "edit_start": control.edit_start,
            "edit_end": control.edit_end,
            "ref_sequence": control.ref_sequence,
            "alt_sequence": control.alt_sequence,
            "target_feature": "same_pol2_peak_non_motif_control",
            "ref_pwm_score": control.pwm_ref_score,
            "alt_pwm_score": control.pwm_alt_score,
            "ref_pwm_relative_score": control.relative_ref_score,
            "alt_pwm_relative_score": control.relative_alt_score,
            "edited_base_count": control.edited_base_count,
            "gc_delta_count": control.gc_delta_count,
        }
        for control in controls
    ]
    return MutationDesign(
        experimental,
        control_rows,
        {
            "anchor": hit.center,
            "motif_start": hit.start,
            "motif_end": hit.end,
            "motif_strand": hit.strand,
            "motif_relative_score": hit.relative_score,
        },
        [],
    )


@register_mutation_strategy("center_substitution")
def _prepare_center_substitution(
    row: Any,
    context: MutationDesignContext,
) -> MutationDesign:
    chrom = str(row.chrom)
    start = int(row.start)
    end = int(row.end)
    anchor = _optional_int(row, "anchor", (start + end) // 2)
    edit_length = _optional_int(row, "edit_length", 3)
    edit_start = anchor - edit_length // 2
    edit_end = edit_start + edit_length
    ref = context.fasta.extract(chrom, edit_start, edit_end)
    alt = _substitution(ref)
    experimental = {
        "mutation_strategy": "center_substitution",
        "edit_start": edit_start,
        "edit_end": edit_end,
        "ref_sequence": ref,
        "alt_sequence": alt,
        "target_feature": _optional_text(row, "target_feature", "site_center"),
        "ref_pwm_score": np.nan,
        "alt_pwm_score": np.nan,
    }
    controls = [
        {
            "mutation_strategy": "local_sequence_control",
            **control,
            "target_feature": "local_sequence_control",
            "ref_pwm_score": np.nan,
            "alt_pwm_score": np.nan,
        }
        for control in _local_sequence_controls(
            fasta=context.fasta,
            chrom=chrom,
            target_start=edit_start,
            target_end=edit_end,
            target_ref=ref,
            target_alt=alt,
            excluded_start=_optional_int(row, "control_exclude_start", start),
            excluded_end=_optional_int(row, "control_exclude_end", end),
            max_distance_bp=context.max_control_distance_bp,
            count=context.sequence_controls,
        )
    ]
    return MutationDesign(experimental, controls, {}, [])


@register_mutation_strategy("explicit_edit")
def _prepare_explicit_edit(
    row: Any,
    context: MutationDesignContext,
) -> MutationDesign:
    chrom = str(row.chrom)
    edit_start = _optional_int(row, "edit_start")
    edit_end = _optional_int(row, "edit_end")
    ref = _optional_text(row, "ref_sequence")
    alt = _optional_text(row, "alt_sequence")
    if edit_start < 0 or edit_end <= edit_start or not ref or not alt:
        return MutationDesign(
            None,
            [],
            {},
            ["explicit_edit requires edit_start/end and ref/alt"],
        )
    if context.fasta.extract(chrom, edit_start, edit_end) != ref.upper():
        return MutationDesign(
            None, [], {}, ["explicit_edit REF does not match FASTA"]
        )
    return MutationDesign(
        {
            "mutation_strategy": "explicit_edit",
            "edit_start": edit_start,
            "edit_end": edit_end,
            "ref_sequence": ref.upper(),
            "alt_sequence": alt.upper(),
            "target_feature": _optional_text(row, "target_feature", "explicit"),
            "ref_pwm_score": np.nan,
            "alt_pwm_score": np.nan,
        },
        [],
        {},
        [],
    )


@register_summary_strategy("multiscale_log2fc")
def summarize_multiscale_log2fc(
    ref: np.ndarray,
    alt: np.ndarray,
    feature_prefix: str,
    windows_bp: Sequence[int],
    resolution: int,
) -> dict[str, float]:
    """MREG-compatible centered multiscale signal and log2FC summaries."""
    ref = np.maximum(np.asarray(ref, dtype=float), 0.0)
    alt = np.maximum(np.asarray(alt, dtype=float), 0.0)
    effect = np.log2(1.0 + alt) - np.log2(1.0 + ref)
    center = len(ref) // 2
    record = {}
    for width in windows_bp:
        radius = int(np.ceil(width / resolution / 2))
        left = max(0, center - radius)
        right = min(len(ref), center + radius + 1)
        prefix = f"{feature_prefix}__w{width}"
        record[f"{prefix}__S_ref_mean"] = float(ref[left:right].mean())
        record[f"{prefix}__log2fc_signed_mean"] = float(
            effect[left:right].mean()
        )
        record[f"{prefix}__log2fc_absolute_mean"] = float(
            np.abs(effect[left:right]).mean()
        )
        record[f"{prefix}__log2fc_depletion_mean"] = float(
            np.maximum(-effect[left:right], 0.0).mean()
        )
        record[f"{prefix}__peak_log2fc"] = float(
            np.log2(
                (1.0 + float(alt[left:right].max()))
                / (1.0 + float(ref[left:right].max()))
            )
        )
    return record


def load_sites(path: str | Path) -> pd.DataFrame:
    sites = pd.read_csv(path, sep="\t", low_memory=False)
    _require_columns(sites, SITE_REQUIRED_COLUMNS, "sites.tsv")
    sites = sites.copy()
    if sites["site_id"].duplicated().any():
        examples = sites.loc[sites["site_id"].duplicated(), "site_id"].head().tolist()
        raise ValueError(f"sites.tsv has duplicate site_id values: {examples}")
    for column in ("start", "end"):
        sites[column] = pd.to_numeric(sites[column], errors="raise").astype(int)
    invalid = sites["end"] <= sites["start"]
    if invalid.any():
        raise ValueError(
            "sites.tsv has invalid intervals: "
            f"{sites.loc[invalid, 'site_id'].head().tolist()}"
        )
    sites["mutation_strategy"] = (
        sites["mutation_strategy"].fillna("").astype(str).str.strip().str.lower()
    )
    unsupported = sorted(
        set(sites["mutation_strategy"]) - set(MUTATION_STRATEGIES)
    )
    if unsupported:
        raise ValueError(f"Unsupported mutation strategies: {unsupported}")
    return sites


def load_track_registry(path: str | Path) -> pd.DataFrame:
    tracks = pd.read_csv(path, sep="\t", low_memory=False)
    _require_columns(tracks, TRACK_REQUIRED_COLUMNS, "tracks.tsv")
    if tracks["track_id"].duplicated().any():
        raise ValueError("tracks.tsv track_id values must be unique")
    tracks = tracks.copy()
    tracks["output_type"] = tracks["output_type"].astype(str).str.strip().str.lower()
    if "required" not in tracks.columns:
        tracks["required"] = True
    return tracks


def build_default_readouts(sites: pd.DataFrame) -> pd.DataFrame:
    rows = [
        {
            "readout_id": "mutation_center_1kb",
            "site_id": "*",
            "scope": "mutation",
            "chrom": ".",
            "start": -1,
            "end": -1,
            "anchor": -1,
        },
        {
            "readout_id": "mutation_local_4kb",
            "site_id": "*",
            "scope": "mutation",
            "chrom": ".",
            "start": -1,
            "end": -1,
            "anchor": -1,
        },
    ]
    for row in sites.itertuples(index=False):
        rows.append(
            {
                "readout_id": "site_interval",
                "site_id": row.site_id,
                "scope": "site",
                "chrom": row.chrom,
                "start": int(row.start),
                "end": int(row.end),
                "anchor": (int(row.start) + int(row.end)) // 2,
            }
        )
    return pd.DataFrame.from_records(rows)


def prepare_batch_manifests(
    *,
    sites_path: str | Path,
    tracks_path: str | Path,
    fasta_path: str | Path,
    motif_path: str | Path,
    output_dir: str | Path,
    readouts_path: str | Path | None = None,
    min_relative_motif_score: float = 0.80,
    motif_controls: int = 3,
    sequence_controls: int = 1,
    max_control_distance_bp: int = 5_000,
) -> dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    sites = load_sites(sites_path)
    tracks = load_track_registry(tracks_path)
    fasta = FastaReference(fasta_path)
    pwm = load_meme_pwm(motif_path, motif_name="CTCF")
    settings = _MutationSettings(
        min_relative_motif_score=min_relative_motif_score,
        local_controls_per_motif=motif_controls,
        min_local_controls_per_motif=1,
        max_control_distance_bp=max_control_distance_bp,
    )
    config = _MutationConfig(mutation=settings)
    design_context = MutationDesignContext(
        fasta=fasta,
        pwm=pwm,
        config=config,
        min_relative_motif_score=min_relative_motif_score,
        sequence_controls=sequence_controls,
        max_control_distance_bp=max_control_distance_bp,
    )

    region_rows = []
    mutation_rows = []
    qc_rows = []

    for row in sites.itertuples(index=False):
        site_id = str(row.site_id)
        chrom = str(row.chrom)
        start = int(row.start)
        end = int(row.end)
        strategy = str(row.mutation_strategy)
        site_class = str(row.site_class)
        anchor = _optional_int(row, "anchor", (start + end) // 2)
        excluded_start = _optional_int(row, "control_exclude_start", start)
        excluded_end = _optional_int(row, "control_exclude_end", end)
        source = _optional_text(row, "source", str(sites_path))
        label = _optional_text(row, "label", site_id)

        region_record = {
            **row._asdict(),
            "site_id": site_id,
            "site_class": site_class,
            "chrom": chrom,
            "start": start,
            "end": end,
            "anchor": anchor,
            "mutation_strategy": strategy,
            "source": source,
            "label": label,
            "motif_start": -1,
            "motif_end": -1,
            "motif_strand": ".",
            "motif_relative_score": np.nan,
            "preparation_status": "pending",
        }
        messages = []
        if not _validate_context(fasta, chrom, anchor):
            messages.append("anchor lacks a complete 1 Mb AlphaGenome context")

        experimental: dict[str, Any] | None = None
        controls: list[dict[str, Any]] = []
        if not messages:
            design = MUTATION_STRATEGIES[strategy](row, design_context)
            experimental = design.experimental
            controls = design.controls
            region_record.update(design.region_updates)
            messages.extend(design.messages)

        if experimental is not None:
            mutation_id = f"{site_id}__target"
            mutation_rows.append(
                {
                    "mutation_id": mutation_id,
                    "site_id": site_id,
                    "site_class": site_class,
                    "chrom": chrom,
                    "control_type": "experimental",
                    "matched_target_id": mutation_id,
                    **experimental,
                    "fasta_ref_match": True,
                    "qc_status": "ready",
                }
            )
            for index, control in enumerate(controls):
                mutation_rows.append(
                    {
                        "mutation_id": f"{mutation_id}__ctrl_{index:02d}",
                        "site_id": site_id,
                        "site_class": site_class,
                        "chrom": chrom,
                        "control_type": "matched_control",
                        "matched_target_id": mutation_id,
                        **control,
                        "fasta_ref_match": True,
                        "qc_status": "ready",
                    }
                )
            if not controls:
                messages.append("no matched controls generated")
            region_record["preparation_status"] = "ready"
        else:
            region_record["preparation_status"] = "skipped"

        region_rows.append(region_record)
        qc_rows.append(
            {
                "site_id": site_id,
                "status": region_record["preparation_status"],
                "n_controls": len(controls),
                "messages": "; ".join(messages),
            }
        )

    regions = pd.DataFrame.from_records(region_rows)
    mutations = pd.DataFrame.from_records(mutation_rows)
    qc = pd.DataFrame.from_records(qc_rows)
    if readouts_path:
        readouts = pd.read_csv(readouts_path, sep="\t", low_memory=False)
    else:
        readouts = build_default_readouts(sites)

    regions.to_csv(output / "site_manifest.tsv", sep="\t", index=False)
    regions.loc[
        regions["preparation_status"].eq("ready")
    ].to_csv(output / "ready_sites.tsv", sep="\t", index=False)
    mutations.to_csv(output / "mutation_manifest.tsv", sep="\t", index=False)
    tracks.to_csv(output / "track_registry.tsv", sep="\t", index=False)
    readouts.to_csv(output / "readout_manifest.tsv", sep="\t", index=False)
    qc.to_csv(output / "preparation_qc.tsv", sep="\t", index=False)
    fasta.close()
    return {
        "sites": len(sites),
        "ready_sites": int(regions["preparation_status"].eq("ready").sum()),
        "mutations": len(mutations),
        "experimental_mutations": int(
            mutations["control_type"].eq("experimental").sum()
        )
        if not mutations.empty
        else 0,
        "controls": int(mutations["control_type"].eq("matched_control").sum())
        if not mutations.empty
        else 0,
        "output_dir": str(output),
    }


def resolve_track_registry(
    metadata: pd.DataFrame,
    registry: pd.DataFrame,
    *,
    allow_biosample_fallback: bool = True,
) -> pd.DataFrame:
    """Resolve exact target names and preferred biosamples to head-local indices."""
    resolved = []
    for row in registry.itertuples(index=False):
        output_type = str(row.output_type).lower()
        target = str(row.target_name).upper()
        biosample = str(row.biosample).strip().upper()
        subset = metadata[metadata["output_type"].eq(output_type)].copy()
        target_column = (
            "transcription_factor"
            if output_type == "chip_tf"
            else "histone_mark"
            if output_type == "chip_histone"
            else "target"
        )
        if target_column in subset.columns and target not in {"", "*"}:
            subset = subset[
                subset[target_column]
                .fillna("")
                .astype(str)
                .str.upper()
                .eq(target)
            ]
        biosample_column = (
            "biosample_name"
            if "biosample_name" in subset.columns
            else "biosample"
        )
        exact = subset[
            subset[biosample_column]
            .fillna("")
            .astype(str)
            .str.strip()
            .str.upper()
            .eq(biosample)
        ]
        fallback = ""
        if not exact.empty:
            selected = exact.iloc[0]
        elif allow_biosample_fallback and not subset.empty:
            selected = subset.iloc[0]
            fallback = f"no exact {biosample}; used {selected[biosample_column]}"
        else:
            required = str(getattr(row, "required", True)).lower() in {
                "true",
                "1",
                "yes",
            }
            if required:
                raise ValueError(
                    f"Could not resolve required track {row.track_id}: "
                    f"{output_type}/{target}/{biosample}"
                )
            continue
        resolved.append(
            {
                **row._asdict(),
                "track_index": int(selected["track_index"]),
                "biosample_resolved": str(selected.get(biosample_column, "")),
                "track_name": str(selected.get("track_name", "")),
                "fallback_reason": fallback,
            }
        )
    if not resolved:
        raise ValueError("No tracks were resolved")
    return pd.DataFrame.from_records(resolved)


def compute_batch_metrics(
    profiles: pd.DataFrame,
    pseudocount: float = 1.0,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Compute per-mutation metrics, matched-control effects, and class summaries."""
    group_columns = [
        "site_id",
        "site_class",
        "mutation_id",
        "matched_target_id",
        "control_type",
        "readout_id",
        "track_id",
        "target_name",
        "biosample",
    ]
    rows = []
    for keys, group in profiles.groupby(group_columns, dropna=False):
        ref = group["ref_value"].to_numpy(dtype=float)
        alt = group["alt_value"].to_numpy(dtype=float)
        delta = alt - ref
        log2fc = np.log2(
            (np.maximum(alt, 0) + pseudocount)
            / (np.maximum(ref, 0) + pseudocount)
        )
        rows.append(
            {
                **dict(zip(group_columns, keys)),
                "n_bins": len(group),
                "ref_mean": float(ref.mean()),
                "alt_mean": float(alt.mean()),
                "signed_delta_mean": float(delta.mean()),
                "absolute_delta_mean": float(np.abs(delta).mean()),
                "delta_peak": float(np.abs(delta).max()),
                "log2fc_mean": float(log2fc.mean()),
                "absolute_log2fc_mean": float(np.abs(log2fc).mean()),
            }
        )
    metrics = pd.DataFrame.from_records(rows)
    effect_columns = [
        "signed_delta_mean",
        "absolute_delta_mean",
        "delta_peak",
        "log2fc_mean",
        "absolute_log2fc_mean",
    ]
    adjusted_rows = []
    experimental = metrics[metrics["control_type"].eq("experimental")]
    controls = metrics[metrics["control_type"].eq("matched_control")]
    for row in experimental.itertuples(index=False):
        matched = controls[
            controls["matched_target_id"].eq(row.mutation_id)
            & controls["readout_id"].eq(row.readout_id)
            & controls["track_id"].eq(row.track_id)
        ]
        if matched.empty:
            continue
        record = {
            "site_id": row.site_id,
            "site_class": row.site_class,
            "mutation_id": row.mutation_id,
            "readout_id": row.readout_id,
            "track_id": row.track_id,
            "target_name": row.target_name,
            "biosample": row.biosample,
            "n_controls": len(matched),
        }
        for column in effect_columns:
            target_value = float(getattr(row, column))
            control_value = float(matched[column].mean())
            record[column] = target_value
            record[f"control_{column}"] = control_value
            record[f"adjusted_{column}"] = target_value - control_value
        adjusted_rows.append(record)
    adjusted = pd.DataFrame.from_records(adjusted_rows)
    if adjusted.empty:
        class_summary = pd.DataFrame()
    else:
        class_summary = (
            adjusted.groupby(
                ["site_class", "readout_id", "track_id", "target_name"],
                dropna=False,
            )
            .agg(
                n_sites=("site_id", "nunique"),
                adjusted_log2fc_mean=("adjusted_log2fc_mean", "mean"),
                adjusted_log2fc_median=("adjusted_log2fc_mean", "median"),
                adjusted_absolute_log2fc_mean=(
                    "adjusted_absolute_log2fc_mean",
                    "mean",
                ),
                adjusted_delta_mean=("adjusted_signed_delta_mean", "mean"),
            )
            .reset_index()
        )
    return metrics, adjusted, class_summary


def adjust_population_features(
    mutation_features: pd.DataFrame,
) -> pd.DataFrame:
    """Subtract matched-control effects and return one row per target site.

    Reference-signal columns are retained from the experimental mutation.
    Mutation-effect columns are calibrated against the mean of all matched
    local controls.
    """
    required = {
        "site_id",
        "mutation_id",
        "matched_target_id",
        "control_type",
    }
    _require_columns(mutation_features, required, "mutation features")
    identity = [
        column
        for column in (
            "site_id",
            "site_class",
            "mutation_id",
            "matched_target_id",
            "control_type",
        )
        if column in mutation_features.columns
    ]
    feature_columns = [
        column for column in mutation_features.columns if column not in identity
    ]
    effect_columns = [
        column
        for column in feature_columns
        if "__log2fc_" in column or column.endswith("__peak_log2fc")
    ]
    reference_columns = [
        column for column in feature_columns if column.endswith("__S_ref_mean")
    ]
    experimental = mutation_features[
        mutation_features["control_type"].eq("experimental")
    ].copy()
    controls = mutation_features[
        mutation_features["control_type"].eq("matched_control")
    ].copy()
    rows = []
    for row in experimental.itertuples(index=False):
        record = {
            "site_id": str(row.site_id),
            "site_class": str(getattr(row, "site_class", "")),
        }
        for column in reference_columns:
            record[column] = float(getattr(row, column))
        matched = controls[
            controls["matched_target_id"].eq(row.mutation_id)
        ]
        for column in effect_columns:
            target_value = float(getattr(row, column))
            if matched.empty:
                record[column] = target_value
            else:
                record[column] = target_value - float(matched[column].mean())
        record["n_matched_controls"] = int(len(matched))
        rows.append(record)
    result = pd.DataFrame.from_records(rows)
    if result["site_id"].duplicated().any():
        raise ValueError("More than one experimental mutation exists per site")
    return result
