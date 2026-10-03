"""Per-gene targeted-ISM prediction and feature generation."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .fasta import FastaReference
from .model_adapters.base import SequenceToProfileModel, TrackSpec
from .mutations import apply_equal_length_edit
from .profiles import MutationProfileWriter, save_reference_profiles
from .readouts import map_readouts_to_bins, summarize_profile_pair


@dataclass(frozen=True)
class TargetedRunConfig:
    """Runtime and artifact options independent of a model backend."""

    model_backend: str
    batch_size: int = 1
    pseudocount: float = 1.0
    progress_every: int = 10
    resume: bool = False
    save_ref_profiles: bool = False
    save_log2fc_profiles: bool = False
    profile_resolution_bp: int = 128
    profile_dtype: str = "float16"


@dataclass(frozen=True)
class GeneRunResult:
    """Artifacts and deterministic-reference result for one gene."""

    feature_path: Path
    sequence_context: dict[str, object]
    deterministic_diff: float


@dataclass(frozen=True)
class GeneInputs:
    """Prepared per-gene tables and output paths."""

    gene: str
    feature_path: Path
    mutations: pd.DataFrame
    readouts: pd.DataFrame
    expected_rows: int
    stored_profile_path: Path


@dataclass(frozen=True)
class ReferencePrediction:
    """Reference sequence, profile, and genomic output geometry for one gene."""

    seq_start: int
    seq_end: int
    output_start: int
    readout_bins: dict[str, tuple[int, int]]
    sequence: str
    profiles: np.ndarray
    deterministic_diff: float


def build_batch_feature_rows(
    *,
    config: TargetedRunConfig,
    adapter: SequenceToProfileModel,
    gene: str,
    batch: pd.DataFrame,
    alternate_profiles: np.ndarray,
    reference_profiles: np.ndarray,
    gene_readouts: pd.DataFrame,
    readout_bins: dict[str, tuple[int, int]],
    requested_tracks: list[str],
    track_specs: dict[str, TrackSpec],
    output_start: int,
) -> list[dict[str, object]]:
    """Summarize one prediction batch into long-form feature rows."""

    rows: list[dict[str, object]] = []
    resolution_bp = int(adapter.output_resolution_bp)
    for alt_index, (_, mutation) in enumerate(batch.iterrows()):
        for track_index, track_id in enumerate(requested_tracks):
            spec = track_specs[track_id]
            for readout in gene_readouts.itertuples(index=False):
                left, right = readout_bins[str(readout.readout_id)]
                summary = summarize_profile_pair(
                    reference_profiles[track_index, left:right],
                    alternate_profiles[alt_index, track_index, left:right],
                    pseudocount=config.pseudocount,
                )
                summary["peak_shift_bp"] = (
                    summary["alt_peak_bin"] - summary["ref_peak_bin"]
                ) * resolution_bp
                rows.append(
                    {
                        "model_backend": config.model_backend,
                        "model_id": adapter.model_id,
                        "gene": gene,
                        "locus_id": mutation["locus_id"],
                        "locus_role": mutation["locus_role"],
                        "mutation_id": mutation["mutation_id"],
                        "mutation_kind": mutation["mutation_kind"],
                        "edit_start": int(mutation["edit_start"]),
                        "edit_end": int(mutation["edit_end"]),
                        "edit_center_position": int(mutation["edit_center_position"]),
                        "variant_offset_from_tss_transcription_bp": int(
                            mutation["variant_offset_from_tss_transcription_bp"]
                        ),
                        "ref_sequence": mutation["ref_sequence"],
                        "alt_sequence": mutation["alt_sequence"],
                        "replacement_replicate": int(mutation["replacement_replicate"]),
                        "track_id": track_id,
                        "track_group": spec.group or "hsc_finetuned_10x",
                        "track_task_name": spec.task_name,
                        "track_cell_type": spec.cell_type,
                        "track_modality": spec.modality,
                        "track_strand": spec.strand,
                        "readout_id": str(readout.readout_id),
                        "readout_role": str(readout.role),
                        "readout_locus_id": str(readout.locus_id),
                        "output_start": output_start,
                        "output_resolution_bp": resolution_bp,
                        "readout_bin_start": left,
                        "readout_bin_end": right,
                        "effective_readout_start": output_start + left * resolution_bp,
                        "effective_readout_end": output_start + right * resolution_bp,
                        **summary,
                    }
                )
    return rows


def _resume_result(
    *,
    config: TargetedRunConfig,
    adapter: SequenceToProfileModel,
    gene_row,
    gene_path: Path,
    stored_profile_path: Path,
    existing: pd.DataFrame,
    mutations: int,
    readouts: int,
    tracks: int,
    output_span: int,
) -> GeneRunResult:
    """Reconstruct the normal sequence-context row for a resumed gene."""

    gene = str(gene_row.gene)
    seq_start = int(gene_row.analysis_tss) - int(adapter.input_length_bp) // 2
    output_start = int(existing["output_start"].iloc[0])
    context = {
        "gene": gene,
        "chrom": str(gene_row.chrom),
        "seq_start": seq_start,
        "seq_end": seq_start + int(adapter.input_length_bp),
        "output_start": output_start,
        "output_end": output_start + output_span,
        "mutations": mutations,
        "readouts": readouts,
        "tracks": tracks,
        "feature_rows": len(existing),
        "deterministic_ref_max_abs_diff": 0.0,
        "stored_profile_resolution_bp": (
            config.profile_resolution_bp if config.save_log2fc_profiles else ""
        ),
        "stored_profile_path": (
            str(stored_profile_path) if config.save_log2fc_profiles else ""
        ),
    }
    return GeneRunResult(gene_path, context, 0.0)


def _prepare_gene_inputs(
    *,
    config: TargetedRunConfig,
    gene: str,
    manifest: pd.DataFrame,
    readouts: pd.DataFrame,
    requested_tracks: list[str],
    feature_dir: Path,
    profile_dir: Path,
) -> GeneInputs:
    mutations = manifest.loc[manifest["gene"].eq(gene)].copy()
    gene_readouts = readouts.loc[readouts["gene"].eq(gene)].copy()
    return GeneInputs(
        gene=gene,
        feature_path=feature_dir / f"{gene}.tsv",
        mutations=mutations,
        readouts=gene_readouts,
        expected_rows=len(mutations) * len(gene_readouts) * len(requested_tracks),
        stored_profile_path=(
            profile_dir
            / f"{gene}.alt_log2fc_{config.profile_resolution_bp}bp.npy"
        ),
    )


def _predict_reference(
    *,
    adapter: SequenceToProfileModel,
    fasta: FastaReference,
    gene_row,
    gene_inputs: GeneInputs,
    requested_tracks: list[str],
    crop_bp: int,
) -> ReferencePrediction:
    seq_start = int(gene_row.analysis_tss) - int(adapter.input_length_bp) // 2
    seq_end = seq_start + int(adapter.input_length_bp)
    chrom = str(gene_row.chrom)
    if seq_start < 0 or seq_end > fasta.chrom_length(chrom):
        raise ValueError(f"{gene_inputs.gene} model sequence is outside chromosome bounds")
    output_start = seq_start + crop_bp
    readout_bins = map_readouts_to_bins(
        gene_inputs.readouts,
        output_start=output_start,
        n_bins=int(adapter.output_length_bins),
        bin_size=int(adapter.output_resolution_bp),
        chrom=chrom,
    )
    sequence = fasta.extract(chrom, seq_start, seq_end)
    profiles = adapter.predict_profiles([sequence], tracks=requested_tracks)[0]
    expected_shape = (len(requested_tracks), int(adapter.output_length_bins))
    if profiles.shape != expected_shape:
        raise RuntimeError(
            f"Unexpected selected profile shape for {gene_inputs.gene}: {profiles.shape}"
        )
    repeated = adapter.predict_profiles([sequence], tracks=requested_tracks)[0]
    return ReferencePrediction(
        seq_start=seq_start,
        seq_end=seq_end,
        output_start=output_start,
        readout_bins=readout_bins,
        sequence=sequence,
        profiles=profiles,
        deterministic_diff=float(np.max(np.abs(profiles - repeated))),
    )


def _start_profile_storage(
    *,
    config: TargetedRunConfig,
    adapter: SequenceToProfileModel,
    gene_inputs: GeneInputs,
    reference: ReferencePrediction,
    requested_tracks: list[str],
    profile_dir: Path,
) -> tuple[MutationProfileWriter | None, Path | None]:
    if config.save_log2fc_profiles:
        writer = MutationProfileWriter(
            profile_dir,
            gene=gene_inputs.gene,
            mutations=gene_inputs.mutations,
            track_order=requested_tracks,
            reference_profiles=reference.profiles,
            native_resolution_bp=int(adapter.output_resolution_bp),
            stored_resolution_bp=config.profile_resolution_bp,
            dtype=config.profile_dtype,
            pseudocount=config.pseudocount,
            output_start=reference.output_start,
        )
        return writer, writer.final_path
    if config.save_ref_profiles:
        save_reference_profiles(
            profile_dir,
            gene=gene_inputs.gene,
            reference_profiles=reference.profiles,
            track_order=requested_tracks,
        )
    return None, None


def _predict_mutation_batches(
    *,
    config: TargetedRunConfig,
    adapter: SequenceToProfileModel,
    gene_inputs: GeneInputs,
    reference: ReferencePrediction,
    requested_tracks: list[str],
    track_specs: dict[str, TrackSpec],
    profile_writer: MutationProfileWriter | None,
) -> list[dict[str, object]]:
    rows = []
    n_batches = math.ceil(len(gene_inputs.mutations) / config.batch_size)
    starts = range(0, len(gene_inputs.mutations), config.batch_size)
    for batch_index, batch_start in enumerate(starts, start=1):
        batch = gene_inputs.mutations.iloc[
            batch_start : batch_start + config.batch_size
        ]
        sequences = [
            apply_equal_length_edit(reference.sequence, reference.seq_start, row)
            for _, row in batch.iterrows()
        ]
        alternate_profiles = adapter.predict_profiles(
            sequences, tracks=requested_tracks
        )
        if profile_writer is not None:
            profile_writer.write_batch(batch_start, alternate_profiles)
        rows.extend(
            build_batch_feature_rows(
                config=config,
                adapter=adapter,
                gene=gene_inputs.gene,
                batch=batch,
                alternate_profiles=alternate_profiles,
                reference_profiles=reference.profiles,
                gene_readouts=gene_inputs.readouts,
                readout_bins=reference.readout_bins,
                requested_tracks=requested_tracks,
                track_specs=track_specs,
                output_start=reference.output_start,
            )
        )
        if config.progress_every > 0 and (
            batch_index % config.progress_every == 0 or batch_index == n_batches
        ):
            print(
                f"[targeted-ism] gene={gene_inputs.gene} "
                f"batch={batch_index}/{n_batches}",
                flush=True,
            )
    return rows


def _write_gene_features(gene_inputs: GeneInputs, rows: list[dict]) -> pd.DataFrame:
    features = pd.DataFrame.from_records(rows)
    if len(features) != gene_inputs.expected_rows:
        raise RuntimeError(
            f"Feature completeness failure for {gene_inputs.gene}: "
            f"{len(features)} != {gene_inputs.expected_rows}"
        )
    temporary = gene_inputs.feature_path.with_suffix(".tsv.tmp")
    features.to_csv(temporary, sep="\t", index=False)
    temporary.replace(gene_inputs.feature_path)
    return features


def _sequence_context(
    *,
    config: TargetedRunConfig,
    gene_row,
    gene_inputs: GeneInputs,
    reference: ReferencePrediction,
    output_span: int,
    tracks: int,
    feature_rows: int,
    stored_profile_path: Path | None,
) -> dict[str, object]:
    return {
        "gene": gene_inputs.gene,
        "chrom": str(gene_row.chrom),
        "seq_start": reference.seq_start,
        "seq_end": reference.seq_end,
        "output_start": reference.output_start,
        "output_end": reference.output_start + output_span,
        "mutations": len(gene_inputs.mutations),
        "readouts": len(gene_inputs.readouts),
        "tracks": tracks,
        "feature_rows": feature_rows,
        "deterministic_ref_max_abs_diff": reference.deterministic_diff,
        "stored_profile_resolution_bp": (
            config.profile_resolution_bp if config.save_log2fc_profiles else ""
        ),
        "stored_profile_path": str(stored_profile_path) if stored_profile_path else "",
    }


def run_gene(
    *,
    config: TargetedRunConfig,
    adapter: SequenceToProfileModel,
    fasta: FastaReference,
    gene_row,
    manifest: pd.DataFrame,
    readouts: pd.DataFrame,
    requested_tracks: list[str],
    track_specs: dict[str, TrackSpec],
    feature_dir: Path,
    profile_dir: Path,
    output_span: int,
    crop_bp: int,
) -> GeneRunResult:
    """Run all prepared mutations and readouts for one gene."""

    gene_inputs = _prepare_gene_inputs(
        config=config,
        gene=str(gene_row.gene),
        manifest=manifest,
        readouts=readouts,
        requested_tracks=requested_tracks,
        feature_dir=feature_dir,
        profile_dir=profile_dir,
    )
    if config.resume and gene_inputs.feature_path.exists():
        existing = pd.read_csv(gene_inputs.feature_path, sep="\t")
        profile_ready = (
            not config.save_log2fc_profiles
            or gene_inputs.stored_profile_path.exists()
        )
        if len(existing) == gene_inputs.expected_rows and profile_ready:
            print(
                f"[targeted-ism] resume skip gene={gene_inputs.gene} "
                f"rows={len(existing)}",
                flush=True,
            )
            return _resume_result(
                config=config,
                adapter=adapter,
                gene_row=gene_row,
                gene_path=gene_inputs.feature_path,
                stored_profile_path=gene_inputs.stored_profile_path,
                existing=existing,
                mutations=len(gene_inputs.mutations),
                readouts=len(gene_inputs.readouts),
                tracks=len(requested_tracks),
                output_span=output_span,
            )

    reference = _predict_reference(
        adapter=adapter,
        fasta=fasta,
        gene_row=gene_row,
        gene_inputs=gene_inputs,
        requested_tracks=requested_tracks,
        crop_bp=crop_bp,
    )
    writer, final_profile_path = _start_profile_storage(
        config=config,
        adapter=adapter,
        gene_inputs=gene_inputs,
        reference=reference,
        requested_tracks=requested_tracks,
        profile_dir=profile_dir,
    )
    print(
        f"[targeted-ism] gene={gene_inputs.gene} "
        f"mutations={len(gene_inputs.mutations)} "
        f"tracks={len(requested_tracks)} readouts={len(gene_inputs.readouts)} "
        f"det={reference.deterministic_diff:g}",
        flush=True,
    )
    rows = _predict_mutation_batches(
        config=config,
        adapter=adapter,
        gene_inputs=gene_inputs,
        reference=reference,
        requested_tracks=requested_tracks,
        track_specs=track_specs,
        profile_writer=writer,
    )
    if writer is not None:
        final_profile_path = writer.finalize()
    features = _write_gene_features(gene_inputs, rows)
    context = _sequence_context(
        config=config,
        gene_row=gene_row,
        gene_inputs=gene_inputs,
        reference=reference,
        output_span=output_span,
        tracks=len(requested_tracks),
        feature_rows=len(features),
        stored_profile_path=final_profile_path,
    )
    return GeneRunResult(
        gene_inputs.feature_path,
        context,
        reference.deterministic_diff,
    )
