"""Streamed AlphaGenome inference for paper-aligned CTCF ISM."""

from __future__ import annotations

import gc
import math
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from grelu.interpret.ism.config import RunConfig
from grelu.interpret.ism.errors import ISMUserError
from grelu.interpret.ism.labels import validate_paper_labels
from grelu.interpret.ism.motifs import FastaReference, reverse_complement
from grelu.interpret.ism.provenance import (
    output_dir,
    prepare_stage_outputs,
    record_stage,
    reject_completed_downstream_stages,
    require_completed_stage,
    sha256_file,
)
from grelu.interpret.ism.tracks import (
    ONE_BP_HEADS,
    build_track_selection_table,
    load_track_metadata,
    selected_track_count,
    selected_track_rows,
)


class InferenceError(ISMUserError, RuntimeError):
    """Raised when streamed inference cannot satisfy its contract."""


@dataclass(frozen=True)
class SequenceRecord:
    site_id: str
    perturbation_id: str
    allele: str
    chrom: str
    sequence_start: int
    sequence_end: int
    sequence: str
    anchor: int
    nearest_tss_offset_bp: int | None
    nearest_gene_start_offset_bp: int | None
    nearest_gene_end_offset_bp: int | None
    host_tss_offset_bp: int | None
    host_gene_start_offset_bp: int | None
    host_gene_end_offset_bp: int | None
    promoter_offsets_bp: tuple[int, ...]


@dataclass(frozen=True)
class GeneRecord:
    chrom: str
    start: int
    end: int
    strand: str
    gene_id: str
    gene_name: str

    @property
    def tss(self) -> int:
        return self.start if self.strand != "-" else self.end - 1


class GeneAnnotationIndex:
    """Small in-memory index for inference-time gene masks."""

    def __init__(self, genes: Sequence[GeneRecord]):
        self.by_chrom: dict[str, list[GeneRecord]] = {}
        self.by_name: dict[tuple[str, str], list[GeneRecord]] = {}
        for gene in genes:
            self.by_chrom.setdefault(gene.chrom, []).append(gene)
            for name in {gene.gene_id, gene.gene_name} - {""}:
                self.by_name.setdefault(
                    (gene.chrom, name.casefold()), []
                ).append(gene)
        for chrom in self.by_chrom:
            self.by_chrom[chrom].sort(key=lambda gene: (gene.tss, gene.start))

    def context(
        self,
        *,
        chrom: str,
        anchor: int,
        host_gene: str,
        promoter_radius_bp: int = 100_000,
    ) -> dict[str, Any]:
        genes = self.by_chrom.get(chrom, [])
        nearest = (
            min(genes, key=lambda gene: (abs(gene.tss - anchor), gene.tss))
            if genes
            else None
        )
        host_candidates = self.by_name.get(
            (chrom, host_gene.casefold()), []
        )
        host = None
        if host_candidates:
            host = min(
                host_candidates,
                key=lambda gene: (
                    not (gene.start <= anchor < gene.end),
                    -(gene.end - gene.start),
                    gene.start,
                ),
            )
        promoter_offsets = tuple(
            gene.tss - anchor
            for gene in genes
            if abs(gene.tss - anchor) <= promoter_radius_bp
        )
        return {
            "nearest": nearest,
            "host": host,
            "promoter_offsets": promoter_offsets,
        }


def _gtf_attribute(attributes: str, key: str) -> str:
    match = re.search(
        rf"(?:^|;\s*){re.escape(key)}\s+[\"']([^\"']+)[\"']",
        attributes,
    )
    return match.group(1).strip() if match else ""


def load_gene_annotation(path: str | Path) -> GeneAnnotationIndex:
    """Load gene rows from a local GTF without triggering downloads."""

    annotation_path = Path(path)
    if not annotation_path.exists():
        raise InferenceError(f"GTF annotation does not exist: {annotation_path}")
    columns = (
        "chrom",
        "source",
        "feature",
        "start",
        "end",
        "score",
        "strand",
        "frame",
        "attributes",
    )
    table = pd.read_csv(
        annotation_path,
        sep="\t",
        comment="#",
        header=None,
        names=columns,
        usecols=range(9),
        low_memory=False,
    )
    genes = table.loc[table["feature"].eq("gene")].copy()
    if genes.empty:
        transcripts = table.loc[table["feature"].eq("transcript")].copy()
        if transcripts.empty:
            raise InferenceError(
                f"GTF contains neither gene nor transcript features: "
                f"{annotation_path}"
            )
        transcripts["gene_id"] = transcripts["attributes"].map(
            lambda value: _gtf_attribute(str(value), "gene_id")
        )
        transcripts["gene_name"] = transcripts["attributes"].map(
            lambda value: _gtf_attribute(str(value), "gene_name")
        )
        transcripts["start"] = pd.to_numeric(
            transcripts["start"], errors="raise"
        ).astype(int)
        transcripts["end"] = pd.to_numeric(
            transcripts["end"], errors="raise"
        ).astype(int)
        genes = (
            transcripts.groupby(
                ["chrom", "strand", "gene_id", "gene_name"],
                dropna=False,
                sort=False,
            )
            .agg(start=("start", "min"), end=("end", "max"))
            .reset_index()
        )
        genes["attributes"] = [
            f'gene_id "{gene_id}"; gene_name "{gene_name}";'
            for gene_id, gene_name in zip(
                genes["gene_id"], genes["gene_name"]
            )
        ]
    genes["start"] = pd.to_numeric(
        genes["start"], errors="raise"
    ).astype(int) - 1
    genes["end"] = pd.to_numeric(genes["end"], errors="raise").astype(int)
    records = []
    for row in genes.itertuples(index=False):
        gene_id = _gtf_attribute(str(row.attributes), "gene_id")
        gene_name = _gtf_attribute(str(row.attributes), "gene_name")
        records.append(
            GeneRecord(
                chrom=str(row.chrom),
                start=int(row.start),
                end=int(row.end),
                strand=str(row.strand),
                gene_id=gene_id,
                gene_name=gene_name,
            )
        )
    return GeneAnnotationIndex(records)


def _build_model(
    *,
    output_type: str,
    resolution: int,
    config: RunConfig,
):
    if not config.paths.weights_path:
        raise InferenceError(
            "paths.weights_path is required; CP3 refuses uninitialized "
            "AlphaGenome inference"
        )
    weights_path = Path(config.paths.weights_path)
    if not weights_path.exists():
        raise InferenceError(
            f"AlphaGenome weights path does not exist: {weights_path}"
        )
    try:
        import torch
        from alphagenome_pytorch.config import DtypePolicy
        from grelu.lightning import LightningModel
    except ImportError as exc:
        raise InferenceError(
            "AlphaGenome inference dependencies are unavailable"
        ) from exc

    model_params: dict[str, Any] = {
        "model_type": "AlphaGenomeModel",
        "output_key": output_type,
        "dtype_policy": DtypePolicy.mixed_precision(),
        "weights_path": str(weights_path),
        "resolution": resolution,
    }
    model = LightningModel(
        model_params=model_params,
        train_params={"task": "regression", "loss": "mse"},
    )
    model.data_params["train"] = {
        "seq_len": config.model.input_length_bp,
        "bin_size": resolution,
    }
    model.model_params["crop_len"] = 0
    if config.model.compile:
        model.model = torch.compile(model.model, mode="max-autotune")
    return model


def _normalize_devices(devices: Sequence[int]) -> list[int]:
    if not devices:
        raise InferenceError("At least one CUDA device is required")
    return [int(device) for device in devices]


def _predict_sequences(
    sequences: Sequence[str],
    *,
    model,
    config: RunConfig,
) -> np.ndarray:
    """Predict one bounded sequence batch and gather each batch on CPU."""

    if not sequences:
        raise InferenceError("Cannot predict an empty sequence batch")
    try:
        import pytorch_lightning as pl
        import torch
        from torch.utils.data import Dataset
    except ImportError as exc:
        raise InferenceError("PyTorch Lightning is required for inference") from exc

    class RawSequenceDataset(Dataset):
        def __init__(self, values: Sequence[str]):
            self.values = [str(value).upper() for value in values]
            self.n_seqs = len(self.values)
            self.n_augmented = 1
            self.n_alleles = 1
            self.rc = False

        def __len__(self) -> int:
            return len(self.values)

        def __getitem__(self, index: int):
            sequence = self.values[index]
            result = torch.zeros((4, len(sequence)), dtype=torch.float32)
            for position, base in enumerate(sequence):
                base_index = {"A": 0, "C": 1, "G": 2, "T": 3}.get(base)
                if base_index is not None:
                    result[base_index, position] = 1.0
            return result

    devices = _normalize_devices(config.model.devices)
    if len(sequences) < len(devices):
        devices = devices[: len(sequences)]
    dataset = RawSequenceDataset(sequences)
    dataloader = model.make_predict_loader(
        dataset,
        num_workers=config.model.num_workers,
        batch_size=config.model.batch_size,
    )
    accelerator, parsed_devices = model.parse_devices(devices)
    trainer = pl.Trainer(
        accelerator=accelerator,
        devices=parsed_devices,
        logger=None,
        precision=config.model.precision,
    )

    cpu_batches = []
    for batch in trainer.predict(model, dataloader):
        cpu_batches.append(batch.detach().cpu())
    if not cpu_batches:
        raise InferenceError("AlphaGenome returned no prediction batches")
    local_predictions = torch.cat(cpu_batches, dim=0)

    world = trainer.world_size
    if world > 1 and torch.distributed.is_initialized():
        group = torch.distributed.new_group(backend="gloo")
        local_size = torch.tensor([local_predictions.shape[0]], dtype=torch.int)
        sizes = [torch.zeros(1, dtype=torch.int) for _ in range(world)]
        torch.distributed.all_gather(sizes, local_size, group=group)
        max_size = max(int(size.item()) for size in sizes)
        if local_predictions.shape[0] < max_size:
            padding = torch.zeros(
                max_size - local_predictions.shape[0],
                *local_predictions.shape[1:],
                dtype=local_predictions.dtype,
            )
            local_predictions = torch.cat(
                [local_predictions, padding], dim=0
            )
        gathered = [torch.zeros_like(local_predictions) for _ in range(world)]
        torch.distributed.all_gather(
            gathered, local_predictions, group=group
        )
        predictions = torch.stack(gathered, dim=1).view(
            -1, *local_predictions.shape[1:]
        )
        predictions = predictions[: sum(int(size.item()) for size in sizes)]
        torch.distributed.destroy_process_group(group)
    else:
        predictions = local_predictions
    return predictions.numpy()


def _apply_edit(
    reference: str,
    *,
    sequence_start: int,
    row: Any,
) -> str:
    edit_start = int(row.edit_start)
    edit_end = int(row.edit_end)
    ref_edit = str(row.ref_sequence).upper()
    alt_edit = str(row.alt_sequence).upper()
    relative_start = edit_start - sequence_start
    relative_end = edit_end - sequence_start
    if relative_start < 0 or relative_end > len(reference):
        raise InferenceError(
            f"Perturbation {row.perturbation_id} lies outside its input window"
        )
    observed = reference[relative_start:relative_end]
    if observed != ref_edit:
        raise InferenceError(
            f"Reference mismatch for {row.perturbation_id}: "
            f"table={ref_edit}, FASTA={observed}"
        )
    if len(ref_edit) != len(alt_edit):
        raise InferenceError(
            f"Perturbation {row.perturbation_id} is not an equal-length edit"
        )
    return reference[:relative_start] + alt_edit + reference[relative_end:]


def _sequence_records_for_sites(
    sites: pd.DataFrame,
    mutations: pd.DataFrame,
    *,
    fasta: FastaReference,
    genes: GeneAnnotationIndex,
    config: RunConfig,
) -> tuple[list[SequenceRecord], list[dict[str, Any]]]:
    records: list[SequenceRecord] = []
    exclusions: list[dict[str, Any]] = []
    half = config.model.input_length_bp // 2
    mutations_by_site = {
        site_id: group
        for site_id, group in mutations.groupby("site_id", sort=False)
    }
    for site in sites.itertuples(index=False):
        site_id = str(site.site_id)
        anchor = getattr(site, "inference_anchor", site.summit)
        center = int(anchor)
        host_gene_value = getattr(site, "host_gene", "")
        host_gene = (
            ""
            if pd.isna(host_gene_value)
            else str(host_gene_value).strip()
        )
        gene_context = genes.context(
            chrom=str(site.chrom),
            anchor=center,
            host_gene=host_gene,
        )
        nearest = gene_context["nearest"]
        host = gene_context["host"]
        sequence_start = center - half
        sequence_end = sequence_start + config.model.input_length_bp
        try:
            chrom_length = fasta.chrom_length(str(site.chrom))
        except Exception as exc:
            exclusions.append(
                {
                    "site_id": site_id,
                    "reason": "chromosome_missing",
                    "details": str(exc),
                }
            )
            continue
        if sequence_start < 0 or sequence_end > chrom_length:
            exclusions.append(
                {
                    "site_id": site_id,
                    "reason": "input_window_outside_chromosome",
                    "details": (
                        f"{site.chrom}:{sequence_start}-{sequence_end}; "
                        f"chrom_length={chrom_length}"
                    ),
                }
            )
            continue
        reference = fasta.extract(
            str(site.chrom), sequence_start, sequence_end
        )
        records.append(
            SequenceRecord(
                site_id=site_id,
                perturbation_id=f"{site_id}:REF",
                allele="REF",
                chrom=str(site.chrom),
                sequence_start=sequence_start,
                sequence_end=sequence_end,
                sequence=reference,
                anchor=center,
                nearest_tss_offset_bp=(
                    nearest.tss - center if nearest else None
                ),
                nearest_gene_start_offset_bp=(
                    nearest.start - center if nearest else None
                ),
                nearest_gene_end_offset_bp=(
                    nearest.end - center if nearest else None
                ),
                host_tss_offset_bp=host.tss - center if host else None,
                host_gene_start_offset_bp=(
                    host.start - center if host else None
                ),
                host_gene_end_offset_bp=(
                    host.end - center if host else None
                ),
                promoter_offsets_bp=gene_context["promoter_offsets"],
            )
        )
        for mutation in mutations_by_site.get(
            site_id, mutations.iloc[0:0]
        ).itertuples(index=False):
            alternate = _apply_edit(
                reference,
                sequence_start=sequence_start,
                row=mutation,
            )
            records.append(
                SequenceRecord(
                    site_id=site_id,
                    perturbation_id=str(mutation.perturbation_id),
                    allele="ALT",
                    chrom=str(site.chrom),
                    sequence_start=sequence_start,
                    sequence_end=sequence_end,
                    sequence=alternate,
                    anchor=center,
                    nearest_tss_offset_bp=(
                        nearest.tss - center if nearest else None
                    ),
                    nearest_gene_start_offset_bp=(
                        nearest.start - center if nearest else None
                    ),
                    nearest_gene_end_offset_bp=(
                        nearest.end - center if nearest else None
                    ),
                    host_tss_offset_bp=(
                        host.tss - center if host else None
                    ),
                    host_gene_start_offset_bp=(
                        host.start - center if host else None
                    ),
                    host_gene_end_offset_bp=(
                        host.end - center if host else None
                    ),
                    promoter_offsets_bp=gene_context["promoter_offsets"],
                )
            )
    return records, exclusions


def _site_chunks(
    sites: pd.DataFrame,
    chunk_size: int,
) -> list[tuple[int, int, pd.DataFrame]]:
    chunks = []
    for start in range(0, len(sites), chunk_size):
        end = min(start + chunk_size, len(sites))
        chunks.append((start, end, sites.iloc[start:end]))
    return chunks


def _record_chunks(
    records: Sequence[SequenceRecord],
    sequences: Sequence[str],
    chunk_size: int,
) -> list[tuple[Sequence[SequenceRecord], Sequence[str]]]:
    return [
        (
            records[start : start + chunk_size],
            sequences[start : start + chunk_size],
        )
        for start in range(0, len(records), chunk_size)
    ]


def _selected_track_projection(
    predictions: np.ndarray,
    track_rows: pd.DataFrame,
    *,
    reverse_orientation: bool,
    contact: bool,
) -> np.ndarray:
    if predictions.ndim not in ({4} if contact else {3}):
        raise InferenceError(
            f"Unexpected prediction rank for selected output: {predictions.shape}"
        )
    if reverse_orientation and not contact:
        partner_by_id = track_rows.set_index("track_id")[
            "rc_partner_track_id"
        ].to_dict()
        index_by_id = track_rows.set_index("track_id")["track_index"].to_dict()
        partner_indices = [
            index_by_id.get(partner_by_id[track_id], -1)
            for track_id in track_rows["track_id"]
        ]
        if any(index < 0 for index in partner_indices):
            raise InferenceError(
                "Fold RC projection is missing a selected stranded-track partner"
            )
        if partner_indices and max(partner_indices) >= predictions.shape[1]:
            raise InferenceError(
                "RC partner index exceeds the model output channel count: "
                f"max_index={max(partner_indices)}, "
                f"channels={predictions.shape[1]}"
            )
        selected = predictions[:, partner_indices, ::-1]
    else:
        indices = track_rows["track_index"].to_numpy(dtype=int)
        if len(indices) and int(indices.max()) >= predictions.shape[1]:
            raise InferenceError(
                "Track metadata index exceeds the model output channel count: "
                f"max_index={int(indices.max())}, channels={predictions.shape[1]}"
            )
        selected = predictions[:, indices]
        if reverse_orientation and contact:
            selected = selected[:, :, ::-1, ::-1]
    return selected.astype(np.float32, copy=False)


def _summary_masks(
    n_bins: int,
    input_length_bp: int,
    windows_bp: Sequence[int],
) -> list[tuple[str, int, int, np.ndarray]]:
    bin_width = input_length_bp / n_bins
    centers = (np.arange(n_bins, dtype=np.float64) + 0.5) * bin_width
    centers -= input_length_bp / 2
    masks = []
    for window_bp in windows_bp:
        half = window_bp / 2
        mask = (centers >= -half) & (centers < half)
        if not mask.any():
            nearest = int(np.argmin(np.abs(centers)))
            mask[nearest] = True
        masks.append(
            (
                f"center_{window_bp}bp",
                -window_bp // 2,
                window_bp - window_bp // 2,
                mask,
            )
        )
    return masks


def _interval_mask(
    *,
    n_bins: int,
    input_length_bp: int,
    start_offset_bp: int,
    end_offset_bp: int,
) -> np.ndarray:
    bin_width = input_length_bp / n_bins
    centers = (np.arange(n_bins, dtype=np.float64) + 0.5) * bin_width
    centers -= input_length_bp / 2
    mask = (centers >= start_offset_bp) & (centers < end_offset_bp)
    if (
        not mask.any()
        and end_offset_bp > -input_length_bp / 2
        and start_offset_bp < input_length_bp / 2
    ):
        target = 0.5 * (start_offset_bp + end_offset_bp)
        mask[int(np.argmin(np.abs(centers - target)))] = True
    return mask


def _record_masks(
    record: SequenceRecord,
    *,
    n_bins: int,
    config: RunConfig,
) -> list[tuple[str, int, int, np.ndarray]]:
    masks = _summary_masks(
        n_bins,
        config.model.input_length_bp,
        config.features.centered_windows_bp,
    )
    intervals: list[tuple[str, int, int]] = []
    if record.nearest_tss_offset_bp is not None:
        for window_bp in (501, 1001):
            start = record.nearest_tss_offset_bp - window_bp // 2
            intervals.append(
                (
                    f"nearest_tss_{window_bp}bp",
                    start,
                    start + window_bp,
                )
            )
    if record.host_tss_offset_bp is not None:
        start = record.host_tss_offset_bp - 500
        intervals.append(("host_gene_tss_1001bp", start, start + 1001))
    if (
        record.nearest_gene_start_offset_bp is not None
        and record.nearest_gene_end_offset_bp is not None
        and record.nearest_gene_start_offset_bp
        >= -config.model.input_length_bp // 2
        and record.nearest_gene_end_offset_bp
        <= config.model.input_length_bp // 2
    ):
        intervals.append(
            (
                "nearest_gene_body",
                record.nearest_gene_start_offset_bp,
                record.nearest_gene_end_offset_bp,
            )
        )
    if (
        record.host_gene_start_offset_bp is not None
        and record.host_gene_end_offset_bp is not None
        and record.host_gene_start_offset_bp
        >= -config.model.input_length_bp // 2
        and record.host_gene_end_offset_bp
        <= config.model.input_length_bp // 2
    ):
        intervals.append(
            (
                "host_gene_body",
                record.host_gene_start_offset_bp,
                record.host_gene_end_offset_bp,
            )
        )
    for name, start, end in intervals:
        mask = _interval_mask(
            n_bins=n_bins,
            input_length_bp=config.model.input_length_bp,
            start_offset_bp=start,
            end_offset_bp=end,
        )
        if mask.any():
            masks.append((name, start, end, mask))
    return masks


def _summary_row(
    *,
    sequence_record: SequenceRecord,
    orientation: str,
    output_type: str,
    resolution: int,
    track_id: str,
    summary_name: str,
    statistic: str,
    value: float,
    offset_start_bp: int,
    offset_end_bp: int,
    bin_count: int,
) -> dict[str, Any]:
    return {
        "site_id": sequence_record.site_id,
        "perturbation_id": sequence_record.perturbation_id,
        "allele": sequence_record.allele,
        "orientation": orientation,
        "output_type": output_type,
        "resolution": resolution,
        "track_id": track_id,
        "summary_name": summary_name,
        "statistic": statistic,
        "value": value,
        "offset_start_bp": offset_start_bp,
        "offset_end_bp": offset_end_bp,
        "bin_count": bin_count,
    }


def _summarize_one_d(
    predictions: np.ndarray,
    sequence_records: Sequence[SequenceRecord],
    track_rows: pd.DataFrame,
    *,
    output_type: str,
    resolution: int,
    orientation: str,
    config: RunConfig,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for sequence_index, sequence_record in enumerate(sequence_records):
        values = predictions[sequence_index]
        masks = _record_masks(
            sequence_record,
            n_bins=values.shape[-1],
            config=config,
        )
        for summary_name, start_bp, end_bp, mask in masks:
            masked = values[:, mask]
            for track_offset, track in enumerate(
                track_rows.itertuples(index=False)
            ):
                track_values = masked[track_offset]
                for statistic, value in (
                    ("mean", np.mean(track_values)),
                    ("sum", np.sum(track_values)),
                ):
                    rows.append(
                        _summary_row(
                            sequence_record=sequence_record,
                            orientation=orientation,
                            output_type=output_type,
                            resolution=resolution,
                            track_id=track.track_id,
                            summary_name=summary_name,
                            statistic=statistic,
                            value=float(value),
                            offset_start_bp=start_bp,
                            offset_end_bp=end_bp,
                            bin_count=int(mask.sum()),
                        )
                    )
        promoter_values = []
        for promoter_offset in sequence_record.promoter_offsets_bp:
            mask = _interval_mask(
                n_bins=values.shape[-1],
                input_length_bp=config.model.input_length_bp,
                start_offset_bp=promoter_offset - 250,
                end_offset_bp=promoter_offset + 251,
            )
            if mask.any():
                promoter_values.append(values[:, mask].mean(axis=1))
        if promoter_values:
            promoter_matrix = np.stack(promoter_values)
            for track_offset, track in enumerate(
                track_rows.itertuples(index=False)
            ):
                track_promoters = promoter_matrix[:, track_offset]
                max_abs_index = int(np.argmax(np.abs(track_promoters)))
                for statistic, value in (
                    ("max", np.max(track_promoters)),
                    ("min", np.min(track_promoters)),
                    ("max_abs_signed", track_promoters[max_abs_index]),
                ):
                    rows.append(
                        _summary_row(
                            sequence_record=sequence_record,
                            orientation=orientation,
                            output_type=output_type,
                            resolution=resolution,
                            track_id=track.track_id,
                            summary_name="promoters_within_100kb",
                            statistic=statistic,
                            value=float(value),
                            offset_start_bp=-100_000,
                            offset_end_bp=100_000,
                            bin_count=len(promoter_values),
                        )
                    )
    return pd.DataFrame.from_records(rows)


def _mean_or_nan(values: np.ndarray) -> float:
    return float(np.mean(values)) if values.size else float("nan")


def _contact_primitives(
    contact_map: np.ndarray,
    *,
    sequence_record: SequenceRecord,
    input_length_bp: int,
    config: RunConfig,
) -> list[tuple[str, float, int]]:
    n_bins = contact_map.shape[-1]
    if contact_map.shape != (n_bins, n_bins):
        raise InferenceError(
            f"Contact map must be square, got {contact_map.shape}"
        )
    bin_bp = input_length_bp / n_bins
    center = n_bins // 2
    offsets = (np.arange(n_bins) - center) * bin_bp
    left = offsets < 0
    right = offsets >= 0
    local = np.abs(offsets) <= config.features.dlr_distal_max_bp
    left_local = left & local
    right_local = right & local
    cross = contact_map[np.ix_(left_local, right_local)]
    same_left = contact_map[np.ix_(left_local, left_local)]
    same_right = contact_map[np.ix_(right_local, right_local)]
    local_total = contact_map[np.ix_(local, local)]
    primitives: list[tuple[str, float, int]] = [
        ("contact_cross_anchor", _mean_or_nan(cross), int(cross.size)),
        (
            "contact_same_side_left",
            _mean_or_nan(same_left),
            int(same_left.size),
        ),
        (
            "contact_same_side_right",
            _mean_or_nan(same_right),
            int(same_right.size),
        ),
        (
            "contact_local_total",
            _mean_or_nan(local_total),
            int(local_total.size),
        ),
    ]
    bin_indices = np.arange(n_bins)
    pair_distance_bp = np.abs(
        bin_indices[:, None] - bin_indices[None, :]
    ) * bin_bp
    cross_side = (offsets[:, None] < 0) & (offsets[None, :] >= 0)
    same_side = (
        ((offsets[:, None] < 0) & (offsets[None, :] < 0))
        | ((offsets[:, None] >= 0) & (offsets[None, :] >= 0))
    )
    upper_triangle = bin_indices[:, None] < bin_indices[None, :]
    for lower_bp, upper_bp in (
        (10_000, 50_000),
        (50_000, 100_000),
        (100_000, 500_000),
    ):
        distance_band = (
            (pair_distance_bp >= lower_bp)
            & (pair_distance_bp < upper_bp)
        )
        local_window = (
            (np.abs(offsets[:, None]) <= upper_bp)
            & (np.abs(offsets[None, :]) <= upper_bp)
        )
        band_name = f"{lower_bp // 1000}_{upper_bp // 1000}kb"
        for name, mask in (
            ("cross", cross_side & distance_band),
            (
                "same_side",
                same_side & distance_band & upper_triangle,
            ),
            (
                "local_total",
                local_window & distance_band & upper_triangle,
            ),
        ):
            values = contact_map[mask]
            primitives.append(
                (
                    f"contact_{name}_{band_name}",
                    _mean_or_nan(values),
                    int(values.size),
                )
            )
    for radius in config.features.apa_radii_bins:
        start = max(0, center - radius)
        end = min(n_bins, center + radius + 1)
        block = contact_map[start:end, start:end]
        primitives.append(
            (
                f"contact_apa_radius_{radius}_bins",
                _mean_or_nan(block),
                int(block.size),
            )
        )
    center_profile = 0.5 * (
        contact_map[center, :] + contact_map[:, center]
    )
    absolute_offsets = np.abs(offsets)
    local_mask = (
        (absolute_offsets > 0)
        & (absolute_offsets <= config.features.dlr_local_max_bp)
    )
    distal_mask = (
        (absolute_offsets >= config.features.dlr_distal_min_bp)
        & (absolute_offsets <= config.features.dlr_distal_max_bp)
    )
    primitives.extend(
        [
            (
                "contact_anchor_local_0_20kb",
                _mean_or_nan(center_profile[local_mask]),
                int(local_mask.sum()),
            ),
            (
                "contact_anchor_distal_50_500kb",
                _mean_or_nan(center_profile[distal_mask]),
                int(distal_mask.sum()),
            ),
        ]
    )
    for name, offset in (
        ("contact_nearest_tss", sequence_record.nearest_tss_offset_bp),
        ("contact_host_gene_tss", sequence_record.host_tss_offset_bp),
    ):
        if offset is None:
            continue
        target = center + int(round(offset / bin_bp))
        if not 0 <= target < n_bins:
            continue
        anchor_start = max(0, center - 1)
        anchor_end = min(n_bins, center + 2)
        target_start = max(0, target - 1)
        target_end = min(n_bins, target + 2)
        block = contact_map[
            anchor_start:anchor_end,
            target_start:target_end,
        ]
        primitives.append((name, _mean_or_nan(block), int(block.size)))
    promoter_contacts = []
    for offset in sequence_record.promoter_offsets_bp:
        target = center + int(round(offset / bin_bp))
        if 0 <= target < n_bins:
            promoter_contacts.append(
                0.5
                * (
                    contact_map[center, target]
                    + contact_map[target, center]
                )
            )
    if promoter_contacts:
        promoter_array = np.asarray(promoter_contacts)
        primitives.extend(
            [
                (
                    "contact_promoter_proxy_mean",
                    float(np.mean(promoter_array)),
                    len(promoter_array),
                ),
                (
                    "contact_promoter_proxy_max",
                    float(np.max(promoter_array)),
                    len(promoter_array),
                ),
            ]
        )
    return primitives


def _summarize_contacts(
    predictions: np.ndarray,
    sequence_records: Sequence[SequenceRecord],
    track_rows: pd.DataFrame,
    *,
    orientation: str,
    config: RunConfig,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    n_bins = predictions.shape[-1]
    effective_resolution = int(round(config.model.input_length_bp / n_bins))
    if effective_resolution != config.features.contact_bin_bp:
        raise InferenceError(
            "Contact-map resolution disagrees with features.contact_bin_bp: "
            f"model={effective_resolution}, configured="
            f"{config.features.contact_bin_bp}"
        )
    for sequence_index, sequence_record in enumerate(sequence_records):
        for track_offset, track in enumerate(
            track_rows.itertuples(index=False)
        ):
            for summary_name, value, count in _contact_primitives(
                predictions[sequence_index, track_offset],
                sequence_record=sequence_record,
                input_length_bp=config.model.input_length_bp,
                config=config,
            ):
                rows.append(
                    {
                        "site_id": sequence_record.site_id,
                        "perturbation_id": sequence_record.perturbation_id,
                        "allele": sequence_record.allele,
                        "orientation": orientation,
                        "output_type": "contact_maps",
                        "resolution": effective_resolution,
                        "track_id": track.track_id,
                        "summary_name": summary_name,
                        "statistic": "mean",
                        "value": value,
                        "offset_start_bp": (
                            -config.features.dlr_distal_max_bp
                        ),
                        "offset_end_bp": (
                            config.features.dlr_distal_max_bp
                        ),
                        "bin_count": count,
                    }
                )
    return pd.DataFrame.from_records(rows)


def _write_shard(
    table: pd.DataFrame,
    *,
    shard_dir: Path,
    shard_id: str,
) -> tuple[Path, str]:
    shard_dir.mkdir(parents=True, exist_ok=True)
    path = shard_dir / f"{shard_id}.parquet"
    temporary = path.with_suffix(".parquet.tmp")
    table.to_parquet(temporary, index=False)
    temporary.replace(path)
    return path, sha256_file(path)


def _orientations(config: RunConfig) -> tuple[str, ...]:
    if config.model.model_kind == "distilled":
        return ("forward",)
    return ("forward", "reverse_complement")


def _release_model(model: Any) -> None:
    try:
        model.cpu()
    except (AttributeError, RuntimeError):
        pass
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def _run_output(
    *,
    output_type: str,
    resolution: int,
    sites: pd.DataFrame,
    mutations: pd.DataFrame,
    track_rows: pd.DataFrame,
    fasta: FastaReference,
    genes: GeneAnnotationIndex,
    shard_dir: Path,
    config: RunConfig,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    model = _build_model(
        output_type=output_type,
        resolution=resolution,
        config=config,
    )
    index_rows: list[dict[str, Any]] = []
    exclusions: list[dict[str, Any]] = []
    try:
        for chunk_number, (start, end, site_chunk) in enumerate(
            _site_chunks(sites, config.model.site_chunk_size)
        ):
            sequence_records, chunk_exclusions = _sequence_records_for_sites(
                site_chunk,
                mutations,
                fasta=fasta,
                genes=genes,
                config=config,
            )
            exclusions.extend(chunk_exclusions)
            if not sequence_records:
                continue
            forward_sequences = [
                record.sequence for record in sequence_records
            ]
            for orientation in _orientations(config):
                sequences = (
                    forward_sequences
                    if orientation == "forward"
                    else [
                        reverse_complement(sequence)
                        for sequence in forward_sequences
                    ]
                )
                prediction_chunk_size = (
                    1
                    if resolution == 1
                    else config.model.prediction_sequence_chunk_size
                )
                summary_parts = []
                for record_chunk, sequence_chunk in _record_chunks(
                    sequence_records,
                    sequences,
                    prediction_chunk_size,
                ):
                    raw_predictions = _predict_sequences(
                        sequence_chunk,
                        model=model,
                        config=config,
                    )
                    selected = _selected_track_projection(
                        raw_predictions,
                        track_rows,
                        reverse_orientation=(
                            orientation == "reverse_complement"
                        ),
                        contact=output_type == "contact_maps",
                    )
                    if output_type == "contact_maps":
                        summary_parts.append(
                            _summarize_contacts(
                                selected,
                                record_chunk,
                                track_rows,
                                orientation=orientation,
                                config=config,
                            )
                        )
                    else:
                        summary_parts.append(
                            _summarize_one_d(
                                selected,
                                record_chunk,
                                track_rows,
                                output_type=output_type,
                                resolution=resolution,
                                orientation=orientation,
                                config=config,
                            )
                        )
                    del raw_predictions, selected
                    gc.collect()
                summary = pd.concat(summary_parts, ignore_index=True)
                if summary.empty:
                    raise InferenceError(
                        f"No summaries produced for {output_type} "
                        f"resolution {resolution}"
                    )
                shard_id = (
                    f"{output_type}__r{resolution}__{orientation}"
                    f"__part-{chunk_number:05d}"
                )
                path, digest = _write_shard(
                    summary,
                    shard_dir=shard_dir,
                    shard_id=shard_id,
                )
                index_rows.append(
                    {
                        "shard_id": shard_id,
                        "output_type": output_type,
                        "resolution": resolution,
                        "orientation": orientation,
                        "site_start_index": start,
                        "site_end_index": end,
                        "row_count": len(summary),
                        "relative_path": str(path.relative_to(shard_dir.parent)),
                        "sha256": digest,
                    }
                )
                del summary
                gc.collect()
    finally:
        _release_model(model)
    return index_rows, exclusions


def _prepare_inference_outputs(
    *,
    table_paths: Sequence[Path],
    shard_dir: Path,
    overwrite_stage: bool,
) -> None:
    prepare_stage_outputs(table_paths, overwrite_stage)
    if shard_dir.exists():
        if not overwrite_stage:
            raise InferenceError(
                f"Inference shard directory already exists: {shard_dir}"
            )
        shutil.rmtree(shard_dir)


def write_representative_outputs(
    *,
    config: RunConfig,
    site_ids: Sequence[str],
    destination: str | Path,
) -> pd.DataFrame:
    """Run a bounded second pass for selected representative sites.

    This API is intentionally separate from the population inference stage.
    It writes only a local 1D profile or local contact-map crop for explicitly
    selected site IDs.
    """

    if not site_ids:
        raise InferenceError("Representative site_ids must not be empty")
    root = output_dir(config)
    sites = pd.read_csv(
        root / "site_metadata.tsv", sep="\t", low_memory=False
    )
    annotations = pd.read_csv(
        root / "motif_annotation_table.tsv", sep="\t", low_memory=False
    )
    mutations = pd.read_csv(
        root / "mutation_table.tsv", sep="\t", low_memory=False
    )
    track_table = pd.read_csv(
        root / "track_selection_table.tsv", sep="\t", low_memory=False
    )
    requested = list(dict.fromkeys(str(site_id) for site_id in site_ids))
    sites = sites.loc[sites["site_id"].astype(str).isin(requested)].copy()
    missing = sorted(set(requested) - set(sites["site_id"].astype(str)))
    if missing:
        raise InferenceError(
            f"Representative site IDs are absent from site metadata: {missing}"
        )
    anchor_table = annotations.loc[:, ["site_id", "motif_center"]].copy()
    anchor_table["motif_center"] = pd.to_numeric(
        anchor_table["motif_center"], errors="coerce"
    )
    sites = sites.merge(
        anchor_table,
        on="site_id",
        how="left",
        validate="one_to_one",
    )
    sites["inference_anchor"] = sites["motif_center"].fillna(sites["summit"])

    if not config.paths.fasta:
        raise InferenceError("paths.fasta is required for representative output")
    if not config.paths.gtf:
        raise InferenceError(
            "paths.gtf is required for representative gene-aware output"
        )
    genes = load_gene_annotation(config.paths.gtf)
    destination = Path(destination)
    if destination.exists():
        raise InferenceError(
            f"Representative output directory already exists: {destination}"
        )
    destination.mkdir(parents=True)
    fasta = FastaReference(config.paths.fasta)
    index_rows: list[dict[str, Any]] = []
    try:
        sequence_records, exclusions = _sequence_records_for_sites(
            sites,
            mutations,
            fasta=fasta,
            genes=genes,
            config=config,
        )
        if exclusions:
            raise InferenceError(
                "Representative sites cannot be outside the model input "
                f"boundary: {exclusions}"
            )
        records_by_site = {
            site_id: [
                record
                for record in sequence_records
                if record.site_id == site_id
            ]
            for site_id in requested
        }
        for output_type in config.model.output_heads:
            resolutions = (
                (config.features.contact_bin_bp,)
                if output_type == "contact_maps"
                else tuple(
                    resolution
                    for resolution in config.model.one_d_resolutions
                    if resolution == 128 or output_type in ONE_BP_HEADS
                )
            )
            for resolution in resolutions:
                track_rows = selected_track_rows(
                    track_table, output_type, resolution
                )
                if track_rows.empty:
                    continue
                model = _build_model(
                    output_type=output_type,
                    resolution=resolution,
                    config=config,
                )
                try:
                    for site_id in requested:
                        site_records = records_by_site[site_id]
                        arrays: list[np.ndarray] = []
                        labels: list[str] = []
                        for orientation in _orientations(config):
                            for record in site_records:
                                sequence = (
                                    record.sequence
                                    if orientation == "forward"
                                    else reverse_complement(record.sequence)
                                )
                                raw = _predict_sequences(
                                    [sequence],
                                    model=model,
                                    config=config,
                                )
                                selected = _selected_track_projection(
                                    raw,
                                    track_rows,
                                    reverse_orientation=(
                                        orientation == "reverse_complement"
                                    ),
                                    contact=output_type == "contact_maps",
                                )[0]
                                window_radius_bp = (
                                    config.model.representative_window_bp / 2
                                )
                                if output_type == "contact_maps":
                                    n_bins = selected.shape[-1]
                                    bin_bp = (
                                        config.model.input_length_bp / n_bins
                                    )
                                    radius = max(
                                        1,
                                        int(
                                            math.ceil(
                                                window_radius_bp / bin_bp
                                            )
                                        ),
                                    )
                                    center = n_bins // 2
                                    selected = selected[
                                        :,
                                        max(0, center - radius) : min(
                                            n_bins, center + radius + 1
                                        ),
                                        max(0, center - radius) : min(
                                            n_bins, center + radius + 1
                                        ),
                                    ]
                                else:
                                    n_bins = selected.shape[-1]
                                    bin_bp = (
                                        config.model.input_length_bp / n_bins
                                    )
                                    radius = max(
                                        1,
                                        int(
                                            math.ceil(
                                                window_radius_bp / bin_bp
                                            )
                                        ),
                                    )
                                    center = n_bins // 2
                                    selected = selected[
                                        :,
                                        max(0, center - radius) : min(
                                            n_bins, center + radius + 1
                                        ),
                                    ]
                                arrays.append(selected)
                                labels.append(
                                    f"{orientation}:{record.perturbation_id}"
                                )
                                del raw, selected
                        safe_site_id = "".join(
                            character
                            if character.isalnum() or character in {"-", "_"}
                            else "_"
                            for character in site_id
                        )
                        filename = (
                            f"{safe_site_id}__{output_type}"
                            f"__r{resolution}.npz"
                        )
                        path = destination / filename
                        np.savez_compressed(
                            path,
                            predictions=np.stack(arrays),
                            prediction_labels=np.asarray(labels),
                            track_ids=track_rows["track_id"].to_numpy(),
                            track_names=track_rows["track_name"].to_numpy(),
                            resolution_bp=resolution,
                            window_bp=config.model.representative_window_bp,
                        )
                        index_rows.append(
                            {
                                "site_id": site_id,
                                "output_type": output_type,
                                "resolution": resolution,
                                "record_count": len(labels),
                                "track_count": len(track_rows),
                                "relative_path": filename,
                                "sha256": sha256_file(path),
                            }
                        )
                finally:
                    _release_model(model)
    finally:
        fasta.close()
    index = pd.DataFrame.from_records(index_rows)
    index.to_csv(
        destination / "representative_prediction_index.tsv",
        sep="\t",
        index=False,
    )
    return index


def run(
    *,
    config: RunConfig,
    config_path: str | Path,
    allow_provisional_labels: bool,
    overwrite_stage: bool,
) -> None:
    """Execute track selection and streamed REF/ALT inference."""

    root = output_dir(config)
    site_path = root / "site_metadata.tsv"
    mutation_path = root / "mutation_table.tsv"
    motif_annotation_path = root / "motif_annotation_table.tsv"
    track_path = root / "track_selection_table.tsv"
    index_path = root / "inference_index.tsv"
    qc_path = root / "inference_qc_report.tsv"
    shard_dir = root / "inference_summaries"

    require_completed_stage(config, config_path, "mutate")
    for required in (site_path, mutation_path, motif_annotation_path):
        if not required.exists():
            raise InferenceError(f"Required CP2 artifact is missing: {required}")
    if overwrite_stage:
        reject_completed_downstream_stages(
            config,
            config_path,
            ("features", "analyze", "report"),
        )
    _prepare_inference_outputs(
        table_paths=(track_path, index_path, qc_path),
        shard_dir=shard_dir,
        overwrite_stage=overwrite_stage,
    )

    sites = pd.read_csv(site_path, sep="\t", low_memory=False)
    sites, _ = validate_paper_labels(
        sites,
        dic_mvalue_threshold=config.labels.dic_mvalue_threshold,
    )
    statuses = set(sites["label_status"])
    if statuses != {"canonical"} and not allow_provisional_labels:
        raise InferenceError(
            "Inference for provisional labels requires "
            "--allow-provisional-labels"
        )
    mutations = pd.read_csv(mutation_path, sep="\t", low_memory=False)
    motif_annotations = pd.read_csv(
        motif_annotation_path, sep="\t", low_memory=False
    )
    anchor_table = motif_annotations.loc[
        :, ["site_id", "motif_center"]
    ].copy()
    anchor_table["motif_center"] = pd.to_numeric(
        anchor_table["motif_center"], errors="coerce"
    )
    sites = sites.merge(
        anchor_table,
        on="site_id",
        how="left",
        validate="one_to_one",
    )
    sites["inference_anchor"] = sites["motif_center"].fillna(sites["summit"])
    metadata, metadata_path = load_track_metadata(
        config.paths.track_metadata
    )
    track_table = build_track_selection_table(metadata, config)
    if selected_track_count(track_table) == 0:
        raise InferenceError("Track registry selected no AlphaGenome outputs")
    track_table.to_csv(track_path, sep="\t", index=False)

    if not config.paths.fasta:
        raise InferenceError("paths.fasta is required for inference")
    fasta_path = Path(config.paths.fasta)
    if not fasta_path.exists():
        raise InferenceError(f"Reference FASTA does not exist: {fasta_path}")
    if not config.paths.gtf:
        raise InferenceError(
            "paths.gtf is required because population raw tensors are not saved"
        )
    gtf_path = Path(config.paths.gtf)
    genes = load_gene_annotation(gtf_path)
    fasta = FastaReference(fasta_path)
    index_rows: list[dict[str, Any]] = []
    exclusion_rows: list[dict[str, Any]] = []
    try:
        for output_type in config.model.output_heads:
            if output_type == "contact_maps":
                resolutions = (config.features.contact_bin_bp,)
            else:
                resolutions = tuple(
                    resolution
                    for resolution in config.model.one_d_resolutions
                    if resolution == 128 or output_type in ONE_BP_HEADS
                )
            for resolution in resolutions:
                track_rows = selected_track_rows(
                    track_table, output_type, resolution
                )
                if track_rows.empty:
                    continue
                output_index, output_exclusions = _run_output(
                    output_type=output_type,
                    resolution=resolution,
                    sites=sites,
                    mutations=mutations,
                    track_rows=track_rows,
                    fasta=fasta,
                    genes=genes,
                    shard_dir=shard_dir,
                    config=config,
                )
                index_rows.extend(output_index)
                exclusion_rows.extend(output_exclusions)
    finally:
        fasta.close()

    index = pd.DataFrame.from_records(index_rows)
    if index.empty:
        raise InferenceError("Inference produced no summary shards")
    index.to_csv(index_path, sep="\t", index=False)
    unique_exclusions = {
        (row["site_id"], row["reason"], row["details"])
        for row in exclusion_rows
    }
    qc_rows = [
        {
            "metric": "sampled_site_count",
            "value": len(sites),
            "details": "Sites entering the inference boundary",
        },
        {
            "metric": "mutation_row_count",
            "value": len(mutations),
            "details": "ALT sequences requested across eligible sites",
        },
        {
            "metric": "selected_track_count",
            "value": selected_track_count(track_table),
            "details": "Selected non-padding track rows",
        },
        {
            "metric": "summary_shard_count",
            "value": len(index),
            "details": "Bounded Parquet summary shards",
        },
        {
            "metric": "excluded_site_count",
            "value": len({row[0] for row in unique_exclusions}),
            "details": "Sites outside the model input boundary or FASTA",
        },
    ]
    for reason in sorted({row[1] for row in unique_exclusions}):
        qc_rows.append(
            {
                "metric": f"exclusion:{reason}",
                "value": sum(row[1] == reason for row in unique_exclusions),
                "details": "Unique site-level inference exclusion",
            }
        )
    pd.DataFrame.from_records(qc_rows).to_csv(
        qc_path, sep="\t", index=False
    )

    label_status = "canonical" if statuses == {"canonical"} else "provisional"
    weights_path = Path(config.paths.weights_path)
    manifest_inputs = [
        site_path,
        mutation_path,
        motif_annotation_path,
        metadata_path,
        fasta_path,
        gtf_path,
    ]
    if weights_path.is_file():
        manifest_inputs.append(weights_path)
    fasta_index = Path(f"{fasta_path}.fai")
    if fasta_index.exists():
        manifest_inputs.append(fasta_index)
    record_stage(
        config=config,
        config_path=config_path,
        stage="infer",
        inputs=manifest_inputs,
        outputs=[track_path, index_path, qc_path],
        label_status=label_status,
        metadata={
            "model_kind": config.model.model_kind,
            "rc_policy": (
                "forward_only"
                if config.model.model_kind == "distilled"
                else "forward_and_reverse_complement_separate_summaries"
            ),
            "selected_track_count": selected_track_count(track_table),
            "summary_shard_count": len(index),
            "summary_shard_directory": str(shard_dir),
            "population_raw_tensor_saved": False,
            "site_chunk_size": config.model.site_chunk_size,
            "prediction_sequence_chunk_size": (
                config.model.prediction_sequence_chunk_size
            ),
            "inference_anchor_policy": "motif_center_else_rad21_summit",
            "weights_path": str(weights_path),
            "weights_size_bytes": (
                weights_path.stat().st_size if weights_path.is_file() else None
            ),
        },
    )
