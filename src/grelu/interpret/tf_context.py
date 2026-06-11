"""Utilities for TF motif-disruption context classification.

This module implements a small AlphaGenome-oriented pilot pipeline for turning
TF binding/motif sites into motif-disrupting SNVs, grouped delta features, and
coarse response-context cluster annotations.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


DEFAULT_OUTPUT_KEYS = (
    "chip_tf",
    "chip_histone",
    "atac",
    "dnase",
    "cage",
    "rna_seq",
)
DEFAULT_WINDOW_BINS = {
    "local": 1,
    "neighborhood": 4,
    "regional": 16,
    "broad": 64,
}
DEFAULT_TRACK_METADATA_PATH = (
    Path(__file__).resolve().parents[3]
    / "src/alphagenome_pytorch/src/alphagenome_pytorch/data/track_metadata_human.parquet"
)
AG_INPUT_LEN = 1_048_576
AG_BIN_SIZE = 128


@dataclass(frozen=True)
class MotifVariant:
    """A one-base motif-disruption variant associated with one input site."""

    site_id: str
    chrom: str
    position: int
    ref: str
    alt: str
    site_start: int
    site_end: int
    motif_start: int
    motif_end: int
    variant_offset: int
    strand: str = "."

    @property
    def pos0(self) -> int:
        return self.position - 1


class FastaSequenceExtractor:
    """Small FASTA extractor backed by pyfaidx."""

    def __init__(self, fasta_path: str | Path):
        try:
            from pyfaidx import Fasta
        except ImportError as exc:
            raise ImportError("pyfaidx is required when --fasta is used") from exc

        self.fasta = Fasta(str(fasta_path), as_raw=True, sequence_always_upper=True)

    def extract(self, chrom: str, start: int, end: int) -> str:
        if start < 0:
            raise ValueError(f"Cannot extract negative interval start: {chrom}:{start}-{end}")
        seq = str(self.fasta[chrom][start:end]).upper()
        if len(seq) != end - start:
            raise ValueError(f"Extracted {len(seq)} bp for {chrom}:{start}-{end}")
        return seq

    def chrom_length(self, chrom: str) -> int:
        return len(self.fasta[chrom])


class RawSeqDataset(Dataset):
    """Minimal raw-sequence dataset for Lightning prediction."""

    def __init__(self, seqs: Sequence[str]):
        self.seqs = [str(seq).upper() for seq in seqs]
        self.n_seqs = len(self.seqs)
        self.n_augmented = 1
        self.n_alleles = 1
        self.rc = False

    def __len__(self) -> int:
        return self.n_seqs

    def __getitem__(self, idx: int) -> torch.Tensor:
        seq = self.seqs[idx]
        arr = torch.zeros((4, len(seq)), dtype=torch.float32)
        for pos, base in enumerate(seq):
            base_idx = {"A": 0, "C": 1, "G": 2, "T": 3}.get(base)
            if base_idx is not None:
                arr[base_idx, pos] = 1.0
        return arr


def load_tf_sites(path: str | Path, genome: str = "hg38", max_sites: int | None = None) -> pd.DataFrame:
    """Load BED-like TF sites with normalized columns.

    Required columns are ``chrom``, ``start``, and ``end``. If the file has no
    header, the first three BED columns are used and common optional BED columns
    are inferred.
    """

    path = Path(path)
    try:
        sites = pd.read_csv(path, sep="\t", comment="#")
        has_header = {"chrom", "start", "end"}.issubset(sites.columns)
    except pd.errors.EmptyDataError:
        raise ValueError(f"No sites found in {path}")

    if not has_header:
        names = [
            "chrom",
            "start",
            "end",
            "name",
            "peak_score",
            "strand",
            "motif_start",
            "motif_end",
            "matched_seq",
        ]
        sites = pd.read_csv(path, sep="\t", comment="#", header=None)
        sites.columns = names[: len(sites.columns)]

    missing = {"chrom", "start", "end"} - set(sites.columns)
    if missing:
        raise ValueError(f"Missing required site columns: {sorted(missing)}")

    sites = sites.copy()
    sites["chrom"] = sites["chrom"].astype(str)
    sites["start"] = sites["start"].astype(int)
    sites["end"] = sites["end"].astype(int)
    sites = sites[(sites["start"] >= 0) & (sites["end"] > sites["start"])].reset_index(drop=True)
    if sites.empty:
        raise ValueError("No valid sites remain after coordinate filtering")

    if "site_id" not in sites.columns:
        if "name" in sites.columns:
            sites["site_id"] = sites["name"].fillna("").astype(str)
            empty = sites["site_id"].str.len() == 0
            sites.loc[empty, "site_id"] = [f"site_{i}" for i in sites.index[empty]]
        else:
            sites["site_id"] = [f"site_{i}" for i in sites.index]

    if "strand" not in sites.columns:
        sites["strand"] = "."
    sites["strand"] = sites["strand"].fillna(".").replace({"": "."})

    if "motif_start" not in sites.columns:
        sites["motif_start"] = sites["start"]
    if "motif_end" not in sites.columns:
        sites["motif_end"] = sites["end"]

    sites["motif_start"] = sites["motif_start"].astype(int)
    sites["motif_end"] = sites["motif_end"].astype(int)
    sites["motif_start"] = sites[["start", "motif_start"]].max(axis=1)
    sites["motif_end"] = sites[["end", "motif_end"]].min(axis=1)
    sites = sites[sites["motif_end"] > sites["motif_start"]].reset_index(drop=True)

    sites.attrs["genome"] = genome
    if max_sites is not None:
        sites = sites.head(max_sites).reset_index(drop=True)
    return sites


def _alternate_base(ref: str) -> str:
    return {"A": "C", "C": "A", "G": "T", "T": "G"}.get(ref.upper(), "A")


def _choose_disruption_offset(seq: str) -> int:
    bases = [base.upper() for base in seq]
    center = len(bases) // 2
    order = sorted(range(len(bases)), key=lambda i: (abs(i - center), i))
    for idx in order:
        if bases[idx] in {"A", "C", "G", "T"}:
            return idx
    return center


def make_motif_disruption_variants(
    sites: pd.DataFrame,
    strategy: str = "max_ic_snp",
) -> pd.DataFrame:
    """Create one synthetic variant per TF site.

    ``max_ic_snp`` is interpreted conservatively in v0: without a PWM, mutate the
    motif-center base or the nearest concrete A/C/G/T base in ``matched_seq``.
    If the input table already contains explicit ``variant_position``,
    ``variant_ref``, and ``variant_alt`` columns, those values are used instead.
    If explicit ``variant_start``, ``variant_end``, and ``variant_alt_seq``
    columns are present, an equal-length multi-base replacement is used.
    """

    if strategy != "max_ic_snp":
        raise ValueError(f"Unsupported disruption strategy: {strategy}")

    records: list[dict] = []
    for row in sites.itertuples(index=False):
        motif_start = int(row.motif_start)
        motif_end = int(row.motif_end)
        explicit_position = getattr(row, "variant_position", None)
        explicit_ref = getattr(row, "variant_ref", None)
        explicit_alt = getattr(row, "variant_alt", None)
        explicit_start = getattr(row, "variant_start", None)
        explicit_end = getattr(row, "variant_end", None)
        explicit_ref_seq = getattr(row, "variant_ref_seq", None)
        explicit_alt_seq = getattr(row, "variant_alt_seq", None)
        if pd.notna(explicit_start) and pd.notna(explicit_end) and pd.notna(explicit_alt_seq):
            start0 = int(explicit_start)
            end0 = int(explicit_end)
            if end0 <= start0:
                raise ValueError(f"Invalid replacement interval for {row.site_id}: {start0}-{end0}")
            ref_seq = (
                str(explicit_ref_seq).upper()
                if pd.notna(explicit_ref_seq)
                else str(getattr(row, "matched_seq", "")).upper()
            )
            alt_seq = str(explicit_alt_seq).upper()
            if len(ref_seq) != end0 - start0 or len(alt_seq) != end0 - start0:
                raise ValueError(
                    f"Replacement length mismatch for {row.site_id}: "
                    f"{start0}-{end0}, ref={ref_seq}, alt={alt_seq}"
                )
            position = start0 + (end0 - start0) // 2 + 1
            ref = ref_seq
            alt = alt_seq
            offset = position - motif_start - 1
        elif pd.notna(explicit_position) and pd.notna(explicit_ref) and pd.notna(explicit_alt):
            position = int(explicit_position)
            ref = str(explicit_ref).upper()
            alt = str(explicit_alt).upper()
            offset = position - motif_start - 1
            start0 = position - 1
            end0 = position
            ref_seq = ref
            alt_seq = alt
        else:
            matched_seq = getattr(row, "matched_seq", None)
            if isinstance(matched_seq, str) and matched_seq:
                offset = min(_choose_disruption_offset(matched_seq), motif_end - motif_start - 1)
                ref = matched_seq[offset].upper()
            else:
                offset = (motif_end - motif_start) // 2
                ref = "N"
            alt = _alternate_base(ref)
            position = motif_start + offset + 1
            start0 = position - 1
            end0 = position
            ref_seq = ref
            alt_seq = alt
        variant = MotifVariant(
            site_id=str(row.site_id),
            chrom=str(row.chrom),
            position=position,
            ref=ref if set(ref) <= {"A", "C", "G", "T"} else "N",
            alt=alt,
            site_start=int(row.start),
            site_end=int(row.end),
            motif_start=motif_start,
            motif_end=motif_end,
            variant_offset=offset,
            strand=str(getattr(row, "strand", ".")),
        )
        rec = asdict(variant)
        rec.update(
            {
                "variant_start": start0,
                "variant_end": end0,
                "variant_ref_seq": ref_seq,
                "variant_alt_seq": alt_seq,
            }
        )
        records.append(rec)
    return pd.DataFrame.from_records(records)


def filter_sites_for_context_window(
    sites: pd.DataFrame,
    fasta_extractor: FastaSequenceExtractor,
    input_len: int = AG_INPUT_LEN,
) -> pd.DataFrame:
    """Keep sites whose motif center can support a full model input window."""

    half = input_len // 2
    keep: list[bool] = []
    for row in sites.itertuples(index=False):
        center = (int(row.motif_start) + int(row.motif_end)) // 2
        try:
            chrom_len = fasta_extractor.chrom_length(str(row.chrom))
        except KeyError:
            keep.append(False)
            continue
        keep.append(center - half >= 0 and center + half <= chrom_len)
    return sites.loc[keep].reset_index(drop=True)


def load_track_metadata(path: str | Path = DEFAULT_TRACK_METADATA_PATH) -> pd.DataFrame:
    """Load AlphaGenome track metadata parquet."""

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Track metadata not found: {path}")
    return pd.read_parquet(path)


def build_track_groups(
    metadata: pd.DataFrame,
    output_keys: Sequence[str] = DEFAULT_OUTPUT_KEYS,
    tf: str = "CTCF",
) -> dict[str, list[int]]:
    """Build grouped global track indices for concatenated AlphaGenome heads."""

    required = {"output_type", "track_index"}
    missing = required - set(metadata.columns)
    if missing:
        raise ValueError(f"Track metadata missing columns: {sorted(missing)}")

    meta = metadata.copy()
    offsets: dict[str, int] = {}
    cursor = 0
    for key in output_keys:
        offsets[key] = cursor
        cursor += int((meta["output_type"] == key).sum())

    tf_upper = tf.upper()
    tx = meta.get("transcription_factor", pd.Series(index=meta.index, dtype=object)).fillna("").astype(str).str.upper()
    hist = meta.get("histone_mark", pd.Series(index=meta.index, dtype=object)).fillna("").astype(str).str.upper()
    assay = meta.get("assay_title", pd.Series(index=meta.index, dtype=object)).fillna("").astype(str).str.lower()

    masks = {
        "ctcf_binding": (meta["output_type"] == "chip_tf") & (tx == tf_upper),
        "cohesin_binding": (meta["output_type"] == "chip_tf") & tx.isin(["RAD21", "SMC3"]),
        "transcription_machinery": (meta["output_type"] == "chip_tf") & tx.isin(["POLR2A", "AFF4", "BRD4", "MED1"]),
        "activity_marks": (
            meta["output_type"].isin(["atac", "dnase", "cage", "rna_seq"])
            | ((meta["output_type"] == "chip_histone") & hist.isin(["H3K27AC", "H3K4ME3", "H3K36ME3"]))
            | assay.isin(["atac-seq", "dnase-seq"])
        ),
        "repressive_or_inactive_context": (meta["output_type"] == "chip_histone")
        & hist.isin(["H3K27ME3", "H3K9ME3"]),
        "global_expression_outputs": meta["output_type"].isin(["cage", "rna_seq"]),
    }

    groups: dict[str, list[int]] = {}
    selected = set(output_keys)
    for name, mask in masks.items():
        indices: list[int] = []
        for row in meta[mask & meta["output_type"].isin(selected)].itertuples(index=False):
            indices.append(offsets[row.output_type] + int(row.track_index))
        groups[name] = sorted(indices)
    return groups


def build_output_track_table(
    metadata: pd.DataFrame,
    output_keys: Sequence[str] = DEFAULT_OUTPUT_KEYS,
) -> pd.DataFrame:
    """Return track metadata with global indices for concatenated outputs."""

    required = {"output_type", "track_index"}
    missing = required - set(metadata.columns)
    if missing:
        raise ValueError(f"Track metadata missing columns: {sorted(missing)}")

    records: list[dict] = []
    cursor = 0
    for output_type in output_keys:
        sub = metadata[metadata["output_type"] == output_type].copy()
        sub = sub.sort_values("track_index")
        for row in sub.to_dict(orient="records"):
            rec = dict(row)
            rec["global_track_index"] = cursor + int(rec["track_index"])
            records.append(rec)
        cursor += len(sub)
    return pd.DataFrame.from_records(records)


def build_alphagenome_model(
    output_key: str | Sequence[str] = DEFAULT_OUTPUT_KEYS,
    weights_path: str | Path | None = None,
    resolution: int = AG_BIN_SIZE,
):
    """Build a gReLU LightningModel wrapping AlphaGenome selected heads."""

    from alphagenome_pytorch.config import DtypePolicy
    from grelu.lightning import LightningModel

    if resolution not in (1, AG_BIN_SIZE):
        raise ValueError(f"AlphaGenome output resolution must be 1 or {AG_BIN_SIZE}, got {resolution}")
    parsed_output_key = output_key if isinstance(output_key, str) else tuple(output_key)
    model_params = {
        "model_type": "AlphaGenomeModel",
        "output_key": parsed_output_key,
        "dtype_policy": DtypePolicy.mixed_precision(),
        "resolution": resolution,
    }
    if weights_path is not None:
        model_params["weights_path"] = str(weights_path)

    model = LightningModel(
        model_params=model_params,
        train_params={"task": "regression", "loss": "mse"},
    )
    model.data_params["train"] = {"seq_len": AG_INPUT_LEN, "bin_size": resolution}
    model.model_params["crop_len"] = 0
    return model


def normalize_contact_maps(preds: np.ndarray) -> np.ndarray:
    """Return contact maps as ``variants x tracks x bins x bins``.

    The gReLU AlphaGenome wrapper returns contact maps as B x T x S x S when
    ``channels_last=False``. Lower-level AlphaGenome APIs may return
    B x S x S x T, so this helper accepts both layouts.
    """

    arr = np.asarray(preds)
    if arr.ndim != 4:
        raise ValueError(f"Expected 4D contact-map predictions, got {arr.shape}")
    if arr.shape[2] == arr.shape[3] and arr.shape[1] <= min(arr.shape[2], 128):
        return arr.astype(np.float32, copy=False)
    if arr.shape[1] == arr.shape[2] and arr.shape[3] <= min(arr.shape[1], 128):
        return np.transpose(arr, (0, 3, 1, 2)).astype(np.float32, copy=False)
    raise ValueError(f"Could not infer contact-map axes for shape {arr.shape}")


def boundary_center_for_row(row: pd.Series) -> int:
    """Return Nakato boundary center when available, otherwise variant center."""

    if pd.notna(row.get("boundary_start")) and pd.notna(row.get("boundary_end")):
        return (int(row["boundary_start"]) + int(row["boundary_end"])) // 2
    return int(row["position"]) - 1


def cross_boundary_track_means(
    contacts: np.ndarray,
    boundary_bin: int,
    window_bins: int,
    min_distance_bins: int,
) -> tuple[np.ndarray, int]:
    """Mean cross-boundary contact per contact-map track.

    ``contacts`` must be tracks x bins x bins. The left side is
    ``[boundary-window, boundary)`` and the right side is
    ``[boundary, boundary+window)``. Pair distances outside the requested band
    are excluded.
    """

    if contacts.ndim != 3:
        raise ValueError(f"Expected tracks x bins x bins contacts, got {contacts.shape}")
    n_tracks, n_bins, _ = contacts.shape
    center = int(np.clip(boundary_bin, 0, n_bins - 1))
    left = np.arange(max(0, center - window_bins), center)
    right = np.arange(center, min(n_bins, center + window_bins))
    if len(left) == 0 or len(right) == 0:
        return np.full(n_tracks, np.nan, dtype=np.float32), 0

    distance = right[None, :] - left[:, None]
    mask = (distance >= min_distance_bins) & (distance <= window_bins)
    if not np.any(mask):
        return np.full(n_tracks, np.nan, dtype=np.float32), 0

    block = contacts[:, left[:, None], right[None, :]]
    return block[:, mask].mean(axis=1), int(mask.sum())


def summarize_contact_boundary_strength(
    ref_contacts: np.ndarray,
    alt_contacts: np.ndarray,
    variants_with_sites: pd.DataFrame,
    intervals: pd.DataFrame,
    input_len: int = AG_INPUT_LEN,
    window_bp: int = 500_000,
    min_distance_bp: int = 100_000,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Summarize contact-map ref/alt changes at boundary centers.

    Higher cross-boundary contact is interpreted as weaker insulation. Therefore
    ``delta_boundary_strength_proxy = ref_cross_contact - alt_cross_contact``:
    positive values mean stronger predicted insulation after mutation, and
    negative values mean boundary weakening.
    """

    ref_maps = normalize_contact_maps(ref_contacts)
    alt_maps = normalize_contact_maps(alt_contacts)
    if ref_maps.shape != alt_maps.shape:
        raise ValueError(f"Prediction shape mismatch: {ref_maps.shape} vs {alt_maps.shape}")

    n_variants, n_tracks, n_bins, _ = ref_maps.shape
    contact_bin_bp = input_len / n_bins
    window_bins = max(1, int(round(window_bp / contact_bin_bp)))
    min_distance_bins = max(1, int(math.ceil(min_distance_bp / contact_bin_bp)))

    site_records: list[dict] = []
    track_records: list[dict] = []
    for idx in range(n_variants):
        row = variants_with_sites.iloc[idx]
        interval = intervals.iloc[idx]
        boundary_center = boundary_center_for_row(row)
        boundary_offset = boundary_center - int(interval["seq_start"])
        boundary_bin = int(math.floor(boundary_offset / contact_bin_bp))
        ref_track, n_pairs = cross_boundary_track_means(
            ref_maps[idx], boundary_bin, window_bins, min_distance_bins
        )
        alt_track, _ = cross_boundary_track_means(
            alt_maps[idx], boundary_bin, window_bins, min_distance_bins
        )
        delta_track = alt_track - ref_track
        strength_track = ref_track - alt_track

        rec = {
            "site_id": row["site_id"],
            "paired_site_id": row.get("paired_site_id", ""),
            "dic_class": row.get("dic_class", ""),
            "control_type": row.get("control_type", ""),
            "chrom": row["chrom"],
            "position": int(row["position"]),
            "ref": row["ref"],
            "alt": row["alt"],
            "boundary_start": row.get("boundary_start", np.nan),
            "boundary_end": row.get("boundary_end", np.nan),
            "boundary_center": boundary_center,
            "seq_start": int(interval["seq_start"]),
            "seq_end": int(interval["seq_end"]),
            "contact_bin_bp": float(contact_bin_bp),
            "boundary_bin": boundary_bin,
            "window_bp": window_bp,
            "min_distance_bp": min_distance_bp,
            "n_cross_boundary_pairs": n_pairs,
            "ref_cross_contact_mean": float(np.nanmean(ref_track)),
            "alt_cross_contact_mean": float(np.nanmean(alt_track)),
            "delta_cross_contact_mean": float(np.nanmean(delta_track)),
            "delta_abs_cross_contact_mean": float(np.nanmean(np.abs(delta_track))),
            "delta_boundary_strength_proxy": float(np.nanmean(strength_track)),
        }
        for col in ["peak_score", "motif_score", "boundary_rank_score"]:
            if col in row:
                rec[col] = row.get(col)
        site_records.append(rec)

        for track_idx in range(n_tracks):
            track_records.append(
                {
                    "site_id": row["site_id"],
                    "paired_site_id": row.get("paired_site_id", ""),
                    "dic_class": row.get("dic_class", ""),
                    "control_type": row.get("control_type", ""),
                    "contact_track_index": track_idx,
                    "ref_cross_contact": float(ref_track[track_idx]),
                    "alt_cross_contact": float(alt_track[track_idx]),
                    "delta_cross_contact": float(delta_track[track_idx]),
                    "delta_boundary_strength_proxy": float(strength_track[track_idx]),
                }
            )

    return pd.DataFrame.from_records(site_records), pd.DataFrame.from_records(track_records)


def contact_motif_vs_control_delta(site_summary: pd.DataFrame) -> pd.DataFrame:
    """Compute paired motif-minus-control contact-boundary summary rows."""

    if "paired_site_id" not in site_summary.columns or "control_type" not in site_summary.columns:
        return pd.DataFrame()
    records: list[dict] = []
    for pair_id, group in site_summary.groupby("paired_site_id", dropna=True):
        motif = group[group["control_type"] == "ctcf_motif_max_ic_disruption"]
        control = group[group["control_type"] == "same_peak_non_motif_nearby_base"]
        if motif.empty or control.empty:
            continue
        m = motif.iloc[0]
        c = control.iloc[0]
        records.append(
            {
                "paired_site_id": pair_id,
                "dic_class": m.get("dic_class", ""),
                "motif_site_id": m["site_id"],
                "control_site_id": c["site_id"],
                "delta_cross_contact_mean__motif_minus_control": (
                    float(m["delta_cross_contact_mean"]) - float(c["delta_cross_contact_mean"])
                ),
                "delta_boundary_strength_proxy__motif_minus_control": (
                    float(m["delta_boundary_strength_proxy"])
                    - float(c["delta_boundary_strength_proxy"])
                ),
                "delta_abs_cross_contact_mean__motif_minus_control": (
                    float(m["delta_abs_cross_contact_mean"]) - float(c["delta_abs_cross_contact_mean"])
                ),
            }
        )
    return pd.DataFrame.from_records(records)


def contact_perturbations_vs_control_delta(
    site_summary: pd.DataFrame,
    baseline_control_type: str = "same_peak_non_motif_nearby_base",
) -> pd.DataFrame:
    """Compute paired perturbation-minus-control contact summaries for all non-control rows."""

    if "paired_site_id" not in site_summary.columns or "control_type" not in site_summary.columns:
        return pd.DataFrame()
    metrics = [
        "delta_cross_contact_mean",
        "delta_boundary_strength_proxy",
        "delta_abs_cross_contact_mean",
    ]
    records: list[dict] = []
    for pair_id, group in site_summary.groupby("paired_site_id", dropna=True):
        control = group[group["control_type"] == baseline_control_type]
        if control.empty:
            continue
        c = control.iloc[0]
        for _, p in group[group["control_type"] != baseline_control_type].iterrows():
            rec = {
                "paired_site_id": pair_id,
                "dic_class": p.get("dic_class", ""),
                "perturbation_type": p["control_type"],
                "perturbation_site_id": p["site_id"],
                "control_site_id": c["site_id"],
                "baseline_control_type": baseline_control_type,
            }
            for metric in metrics:
                rec[f"{metric}__perturbation_minus_control"] = float(p[metric]) - float(c[metric])
            records.append(rec)
    return pd.DataFrame.from_records(records)


def _mutate_sequence(seq: str, offset: int, alt: str) -> str:
    if offset < 0 or offset >= len(seq):
        raise ValueError(f"Variant offset {offset} outside sequence length {len(seq)}")
    return seq[:offset] + alt + seq[offset + 1 :]


def _has_value(value) -> bool:
    if value is None:
        return False
    try:
        return bool(pd.notna(value))
    except ValueError:
        return True


def _apply_variant_edit(ref_seq: str, seq_start: int, row) -> tuple[str, dict]:
    """Apply a SNV or equal-length replacement variant to a reference sequence."""

    start = getattr(row, "variant_start", None)
    end = getattr(row, "variant_end", None)
    alt_seq = getattr(row, "variant_alt_seq", None)
    if _has_value(start) and _has_value(end) and _has_value(alt_seq):
        var_start = int(start)
        var_end = int(end)
        ref_edit = str(getattr(row, "variant_ref_seq", "")).upper()
        alt_edit = str(alt_seq).upper()
        if var_end <= var_start:
            raise ValueError(f"Invalid variant interval for {row.site_id}: {var_start}-{var_end}")
        if len(alt_edit) != var_end - var_start:
            raise ValueError(
                f"Alternate sequence length mismatch for {row.site_id}: "
                f"{var_start}-{var_end}, alt={alt_edit}"
            )
        rel_start = var_start - seq_start
        rel_end = var_end - seq_start
        if rel_start < 0 or rel_end > len(ref_seq):
            raise ValueError(
                f"Variant interval outside extracted sequence for {row.site_id}: "
                f"{var_start}-{var_end}"
            )
        observed = ref_seq[rel_start:rel_end]
        if ref_edit and set(ref_edit) <= {"A", "C", "G", "T"} and observed != ref_edit:
            raise ValueError(
                f"Reference mismatch at {row.chrom}:{var_start}-{var_end}: "
                f"variant has {ref_edit}, FASTA has {observed}"
            )
        center_offset = rel_start + (rel_end - rel_start) // 2
        return (
            ref_seq[:rel_start] + alt_edit + ref_seq[rel_end:],
            {
                "variant_offset": center_offset,
                "variant_start_offset": rel_start,
                "variant_end_offset": rel_end,
                "variant_ref_seq": observed,
                "variant_alt_seq": alt_edit,
            },
        )

    center = int(row.position) - 1
    rel = center - seq_start
    observed_ref = ref_seq[rel]
    if row.ref in {"A", "C", "G", "T"} and observed_ref != row.ref:
        raise ValueError(
            f"Reference mismatch at {row.chrom}:{row.position}: "
            f"variant has {row.ref}, FASTA has {observed_ref}"
        )
    return (
        _mutate_sequence(ref_seq, rel, row.alt),
        {
            "variant_offset": rel,
            "variant_start_offset": rel,
            "variant_end_offset": rel + 1,
            "variant_ref_seq": observed_ref,
            "variant_alt_seq": row.alt,
        },
    )


def _single_predict_device(devices: str | int | Sequence[int]) -> str | int:
    if isinstance(devices, int):
        return devices
    if isinstance(devices, str):
        value = devices.strip()
        if value.lower() == "cpu":
            return "cpu"
        first = value.split(",")[0].strip()
        if first.isdigit():
            return int(first)
        return first
    if isinstance(devices, Sequence) and devices:
        return int(devices[0])
    return "cpu"


def _normalize_devices(devices: str | int | Sequence[int]) -> str | int | list[int]:
    if isinstance(devices, str):
        value = devices.strip()
        if value.lower() == "cpu":
            return "cpu"
        return [int(part.strip()) for part in value.split(",") if part.strip()]
    if isinstance(devices, int):
        return [devices]
    return [int(device) for device in devices]


def _num_predict_devices(devices: str | int | Sequence[int]) -> int:
    normalized = _normalize_devices(devices)
    if normalized == "cpu":
        return 1
    if isinstance(normalized, int):
        return 1
    return max(1, len(normalized))


def _chunk_slices(n_items: int, chunk_size: int, min_chunk_size: int = 1) -> list[slice]:
    """Create chunks while avoiding a too-small final distributed chunk."""

    chunk_size = max(chunk_size, min_chunk_size)
    slices = [slice(start, min(start + chunk_size, n_items)) for start in range(0, n_items, chunk_size)]
    if len(slices) > 1:
        last = slices[-1]
        if last.stop - last.start < min_chunk_size:
            prev = slices[-2]
            slices[-2] = slice(prev.start, last.stop)
            slices.pop()
    return slices


def _predict_sequence_batch(
    seqs: Sequence[str],
    model,
    devices: str | int | Sequence[int] = "cpu",
    batch_size: int = 1,
    num_workers: int = 1,
    precision: str | None = None,
) -> np.ndarray:
    """Predict raw DNA strings with DDP, gathering results on CPU to avoid GPU OOM."""

    if not hasattr(model, "predict_on_dataset"):
        return model.predict_on_seqs(seqs, device=_single_predict_device(devices))

    # Fallback for simple models (e.g. mocks in tests) that don't provide
    # the full LightningModel predict-infrastructure methods.
    if not hasattr(model, "make_predict_loader"):
        dataset = RawSeqDataset(seqs)
        preds = model.predict_on_dataset(
            dataset,
            devices=_normalize_devices(devices),
            batch_size=batch_size,
            num_workers=num_workers,
            precision=precision,
        )
        if isinstance(preds, torch.Tensor):
            return preds.detach().cpu().numpy()
        return preds

    import pytorch_lightning as pl

    dataset = RawSeqDataset(seqs)
    norm_devices = _normalize_devices(devices)
    accelerator, _ = model.parse_devices(norm_devices)
    dataloader = model.make_predict_loader(
        dataset, num_workers=num_workers, batch_size=batch_size
    )
    trainer = pl.Trainer(
        accelerator=accelerator,
        devices=norm_devices,
        logger=None,
        precision=precision,
    )

    # Collect predictions to CPU immediately — no GPU accumulation
    cpu_batches: list[torch.Tensor] = []
    for batch in trainer.predict(model, dataloader):
        cpu_batches.append(batch.detach().cpu())
    local_preds = torch.cat(cpu_batches, dim=0)

    world = trainer.world_size
    if world > 1 and torch.distributed.is_initialized():
        # Create a gloo sub-group for CPU tensor communication.
        # NCCL is GPU-only; gloo handles CPU tensors via shared memory.
        gloo_group = torch.distributed.new_group(backend="gloo")

        local_size = torch.tensor([local_preds.shape[0]], dtype=torch.int)
        all_sizes = [torch.zeros(1, dtype=torch.int) for _ in range(world)]
        torch.distributed.all_gather(all_sizes, local_size, group=gloo_group)
        max_sz = max(int(s.item()) for s in all_sizes)

        if local_preds.shape[0] < max_sz:
            pad = torch.zeros(
                max_sz - local_preds.shape[0], *local_preds.shape[1:],
                dtype=local_preds.dtype,
            )
            local_preds = torch.cat([local_preds, pad], dim=0)

        gathered = [torch.zeros_like(local_preds) for _ in range(world)]
        torch.distributed.all_gather(gathered, local_preds, group=gloo_group)

        # Interleave to restore DistributedSampler order:
        #   rank r got indices [r, r+W, r+2W, ...]
        preds = torch.stack(gathered, dim=1).view(-1, *local_preds.shape[1:])
        total_real = sum(int(s.item()) for s in all_sizes)
        preds = preds[:total_real]

        torch.distributed.destroy_process_group(gloo_group)
    else:
        preds = local_preds

    return preds.numpy()


def score_variants_chunked(
    variants: pd.DataFrame,
    model,
    fasta_extractor,
    devices: str | int | Sequence[int] = "cpu",
    chunk_size: int = 32,
    batch_size: int = 1,
    num_workers: int = 1,
    precision: str | None = None,
    input_len: int = AG_INPUT_LEN,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Score variants and return reference/alternate predictions.

    Returns ``(ref_preds, alt_preds, intervals)`` where prediction arrays are
    ``sites x tracks x bins``.
    """

    ref_chunks: list[np.ndarray] = []
    alt_chunks: list[np.ndarray] = []
    interval_records: list[dict] = []
    n_devices = _num_predict_devices(devices)
    if len(variants) < n_devices:
        raise ValueError(
            f"Need at least {n_devices} variants for distributed prediction; got {len(variants)}"
        )

    for chunk_slice in _chunk_slices(len(variants), chunk_size, min_chunk_size=n_devices):
        chunk = variants.iloc[chunk_slice]
        ref_seqs: list[str] = []
        alt_seqs: list[str] = []
        for row in chunk.itertuples(index=False):
            center = int(row.position) - 1
            seq_start = center - input_len // 2
            seq_end = seq_start + input_len
            ref_seq = fasta_extractor.extract(row.chrom, seq_start, seq_end).upper()
            alt_seq, edit = _apply_variant_edit(ref_seq, seq_start, row)
            ref_seqs.append(ref_seq)
            alt_seqs.append(alt_seq)
            interval_records.append(
                {
                    "site_id": row.site_id,
                    "chrom": row.chrom,
                    "seq_start": seq_start,
                    "seq_end": seq_end,
                    **edit,
                }
            )

        ref_chunks.append(
            _predict_sequence_batch(
                ref_seqs,
                model=model,
                devices=devices,
                batch_size=batch_size,
                num_workers=num_workers,
                precision=precision,
            )
        )
        alt_chunks.append(
            _predict_sequence_batch(
                alt_seqs,
                model=model,
                devices=devices,
                batch_size=batch_size,
                num_workers=num_workers,
                precision=precision,
            )
        )

    return (
        np.concatenate(ref_chunks, axis=0),
        np.concatenate(alt_chunks, axis=0),
        pd.DataFrame.from_records(interval_records),
    )


def score_variant_features_chunked(
    variants: pd.DataFrame,
    model,
    fasta_extractor,
    track_groups: Mapping[str, Sequence[int]],
    devices: str | int | Sequence[int] = "cpu",
    batch_size: int = 1,
    num_workers: int = 1,
    precision: str | None = None,
    input_len: int = AG_INPUT_LEN,
    window_bins: Mapping[str, int] = DEFAULT_WINDOW_BINS,
    timing_records: list[dict] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Score variants and aggregate features in a single paired predict pass.

    All ref and alt sequences are extracted first, then predicted together in
    one call. DDP prediction results are gathered on CPU (via gloo sub-group)
    to avoid GPU-side all_gather memory spikes.
    """

    n_devices = _num_predict_devices(devices)
    n_variants = len(variants)
    if n_variants < n_devices:
        raise ValueError(
            f"Need at least {n_devices} variants for distributed prediction; got {n_variants}"
        )

    # Phase 1: extract all sequences (CPU, fast)
    extract_t0 = time.perf_counter()
    all_ref_seqs: list[str] = []
    all_alt_seqs: list[str] = []
    interval_records: list[dict] = []
    for row in variants.itertuples(index=False):
        center = int(row.position) - 1
        seq_start = center - input_len // 2
        seq_end = seq_start + input_len
        ref_seq = fasta_extractor.extract(row.chrom, seq_start, seq_end).upper()
        alt_seq, edit = _apply_variant_edit(ref_seq, seq_start, row)
        all_ref_seqs.append(ref_seq)
        all_alt_seqs.append(alt_seq)
        interval_records.append(
            {
                "site_id": row.site_id,
                "chrom": row.chrom,
                "seq_start": seq_start,
                "seq_end": seq_end,
                **edit,
            }
        )
    extract_seconds = time.perf_counter() - extract_t0

    # Phase 2: single paired predict — ref + alt in one call
    predict_t0 = time.perf_counter()
    preds = _predict_sequence_batch(
        [*all_ref_seqs, *all_alt_seqs],
        model=model,
        devices=devices,
        batch_size=batch_size,
        num_workers=num_workers,
        precision=precision,
    )
    predict_seconds = time.perf_counter() - predict_t0

    ref_preds = preds[:n_variants]
    alt_preds = preds[n_variants:]

    # Phase 3: aggregate features (CPU)
    aggregate_t0 = time.perf_counter()
    features = aggregate_delta_features(
        ref_preds,
        alt_preds,
        variants,
        track_groups,
        window_bins=window_bins,
    )
    aggregate_seconds = time.perf_counter() - aggregate_t0
    intervals = pd.DataFrame.from_records(interval_records)

    if timing_records is not None:
        timing_records.append(
            {
                "chunk": 0,
                "start": 0,
                "stop": n_variants,
                "n_variants": n_variants,
                "extract_seconds": extract_seconds,
                "predict_seconds": predict_seconds,
                "aggregate_seconds": aggregate_seconds,
                "total_seconds": extract_seconds + predict_seconds + aggregate_seconds,
            }
        )

    return features, intervals


def score_variant_track_window(
    variants: pd.DataFrame,
    model,
    fasta_extractor,
    output_dir: str | Path,
    devices: str | int | Sequence[int] = "cpu",
    batch_size: int = 1,
    num_workers: int = 1,
    precision: str | None = None,
    input_len: int = AG_INPUT_LEN,
    bin_size: int = AG_BIN_SIZE,
    window_bp: int = 100_000,
    save_delta_window: bool = True,
    predict_chunk_size: int | None = None,
    timing_records: list[dict] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Score variants and save full-track ref/alt/delta windows around center.

    The saved arrays have shape ``sites x tracks x window_bins`` and contain all
    predicted tracks from the selected AlphaGenome heads. This avoids lossy
    group aggregation while keeping output size bounded by slicing to a local
    window around the mutated base.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    n_devices = _num_predict_devices(devices)
    n_variants = len(variants)
    if n_variants < n_devices:
        raise ValueError(
            f"Need at least {n_devices} variants for distributed prediction; got {n_variants}"
        )

    radius_bins = int(math.ceil(window_bp / bin_size))
    interval_records: list[dict] = []
    summary_records: list[dict] = []
    ref_window_mm = None
    alt_window_mm = None
    delta_window_mm = None
    left = right = None
    extract_seconds = 0.0
    predict_seconds = 0.0
    write_seconds = 0.0
    # Full AG 1D predictions are much larger than the local window we keep.
    # Stream large runs through bounded predict calls and write .npy memmaps.
    if predict_chunk_size is None:
        predict_chunk_size = 72 if n_variants > 72 else n_variants
    else:
        predict_chunk_size = max(1, int(predict_chunk_size))

    for chunk_idx, chunk_slice in enumerate(
        _chunk_slices(n_variants, predict_chunk_size, min_chunk_size=n_devices)
    ):
        chunk_t0 = time.perf_counter()
        extract_t0 = time.perf_counter()
        chunk = variants.iloc[chunk_slice]
        ref_seqs: list[str] = []
        alt_seqs: list[str] = []
        chunk_interval_records: list[dict] = []
        for row in chunk.itertuples(index=False):
            center = int(row.position) - 1
            seq_start = center - input_len // 2
            seq_end = seq_start + input_len
            ref_seq = fasta_extractor.extract(row.chrom, seq_start, seq_end).upper()
            alt_seq, edit = _apply_variant_edit(ref_seq, seq_start, row)
            ref_seqs.append(ref_seq)
            alt_seqs.append(alt_seq)
            chunk_interval_records.append(
                {
                    "site_id": row.site_id,
                    "chrom": row.chrom,
                    "seq_start": seq_start,
                    "seq_end": seq_end,
                    **edit,
                }
            )
        chunk_extract_seconds = time.perf_counter() - extract_t0
        extract_seconds += chunk_extract_seconds

        predict_t0 = time.perf_counter()
        preds = _predict_sequence_batch(
            [*ref_seqs, *alt_seqs],
            model=model,
            devices=devices,
            batch_size=batch_size,
            num_workers=num_workers,
            precision=precision,
        )
        chunk_predict_seconds = time.perf_counter() - predict_t0
        predict_seconds += chunk_predict_seconds

        write_t0 = time.perf_counter()
        n_chunk = len(chunk)
        ref_preds = preds[:n_chunk]
        alt_preds = preds[n_chunk:]
        if ref_preds.shape != alt_preds.shape or ref_preds.ndim != 3:
            raise ValueError(f"Unexpected prediction shapes: {ref_preds.shape}, {alt_preds.shape}")

        _, n_tracks, n_bins = ref_preds.shape
        center_bin = n_bins // 2
        chunk_left = max(0, center_bin - radius_bins)
        chunk_right = min(n_bins, center_bin + radius_bins + 1)
        if left is None:
            left, right = chunk_left, chunk_right
            window_width = right - left
            ref_window_mm = np.lib.format.open_memmap(
                output_dir / "all_track_ref_window.npy",
                mode="w+",
                dtype=np.float32,
                shape=(n_variants, n_tracks, window_width),
            )
            alt_window_mm = np.lib.format.open_memmap(
                output_dir / "all_track_alt_window.npy",
                mode="w+",
                dtype=np.float32,
                shape=(n_variants, n_tracks, window_width),
            )
            if save_delta_window:
                delta_window_mm = np.lib.format.open_memmap(
                    output_dir / "all_track_delta_window.npy",
                    mode="w+",
                    dtype=np.float32,
                    shape=(n_variants, n_tracks, window_width),
                )
            window_bins = pd.DataFrame(
                {
                    "window_bin_index": np.arange(window_width, dtype=int),
                    "model_bin_index": np.arange(left, right, dtype=int),
                    "offset_bp": (np.arange(left, right, dtype=int) - center_bin) * bin_size,
                }
            )
            window_bins.to_csv(output_dir / "all_track_window_bins.tsv", sep="\t", index=False)
        elif left != chunk_left or right != chunk_right:
            raise ValueError(f"Inconsistent window bins: {(left, right)} vs {(chunk_left, chunk_right)}")

        assert ref_window_mm is not None
        assert alt_window_mm is not None
        ref_window = ref_preds[:, :, left:right].astype(np.float32, copy=False)
        alt_window = alt_preds[:, :, left:right].astype(np.float32, copy=False)
        delta_window = (alt_window - ref_window).astype(np.float32, copy=False)
        ref_window_mm[chunk_slice] = ref_window
        alt_window_mm[chunk_slice] = alt_window
        if delta_window_mm is not None:
            delta_window_mm[chunk_slice] = delta_window

        for local_idx, row in enumerate(chunk.itertuples(index=False)):
            d = delta_window[local_idx]
            summary_records.append(
                {
                    "site_id": row.site_id,
                    "chrom": row.chrom,
                    "position": row.position,
                    "ref": row.ref,
                    "alt": row.alt,
                    "window_bp": window_bp,
                    "window_bins": right - left,
                    "delta_abs_mean": float(np.mean(np.abs(d))),
                    "delta_abs_max": float(np.max(np.abs(d))),
                    "delta_l2": float(np.linalg.norm(d)),
                }
            )
        interval_records.extend(chunk_interval_records)
        chunk_write_seconds = time.perf_counter() - write_t0
        write_seconds += chunk_write_seconds

        if timing_records is not None:
            timing_records.append(
                {
                    "chunk": chunk_idx,
                    "start": chunk_slice.start,
                    "stop": chunk_slice.stop,
                    "n_variants": n_chunk,
                    "extract_seconds": chunk_extract_seconds,
                    "predict_seconds": chunk_predict_seconds,
                    "write_seconds": chunk_write_seconds,
                    "window_bp": window_bp,
                    "window_bins": right - left,
                    "total_seconds": time.perf_counter() - chunk_t0,
                }
            )

        del preds, ref_preds, alt_preds, ref_window, alt_window, delta_window

    if ref_window_mm is not None:
        ref_window_mm.flush()
    if alt_window_mm is not None:
        alt_window_mm.flush()
    if delta_window_mm is not None:
        delta_window_mm.flush()

    intervals = pd.DataFrame.from_records(interval_records)
    site_summary = pd.DataFrame.from_records(summary_records)
    site_summary.to_csv(output_dir / "all_track_site_delta_summary.tsv", sep="\t", index=False)

    return site_summary, intervals


def score_variant_contact_boundary_strength(
    variants: pd.DataFrame,
    sites: pd.DataFrame,
    model,
    fasta_extractor,
    devices: str | int | Sequence[int] = "cpu",
    batch_size: int = 1,
    num_workers: int = 1,
    precision: str | None = None,
    input_len: int = AG_INPUT_LEN,
    window_bp: int = 500_000,
    min_distance_bp: int = 100_000,
    timing_records: list[dict] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Score contact-map boundary-strength proxies for variants.

    Returns site summary, per-track summary, and scored intervals. Ref and alt
    sequences are predicted together in one paired call, matching the existing
    cache-friendly inference path used by the track-window mode.
    """

    n_devices = _num_predict_devices(devices)
    n_variants = len(variants)
    if n_variants < n_devices:
        raise ValueError(
            f"Need at least {n_devices} variants for distributed prediction; got {n_variants}"
        )

    site_cols = [
        c
        for c in [
            "site_id",
            "paired_site_id",
            "dic_class",
            "control_type",
            "peak_score",
            "motif_score",
            "boundary_start",
            "boundary_end",
            "boundary_rank_score",
        ]
        if c in sites.columns
    ]
    variants_with_sites = variants.merge(sites[site_cols], on="site_id", how="left")

    extract_t0 = time.perf_counter()
    ref_seqs: list[str] = []
    alt_seqs: list[str] = []
    interval_records: list[dict] = []
    for row in variants_with_sites.itertuples(index=False):
        center = int(row.position) - 1
        seq_start = center - input_len // 2
        seq_end = seq_start + input_len
        ref_seq = fasta_extractor.extract(row.chrom, seq_start, seq_end).upper()
        alt_seq, edit = _apply_variant_edit(ref_seq, seq_start, row)
        ref_seqs.append(ref_seq)
        alt_seqs.append(alt_seq)
        interval_records.append(
            {
                "site_id": row.site_id,
                "chrom": row.chrom,
                "seq_start": seq_start,
                "seq_end": seq_end,
                **edit,
            }
        )
    extract_seconds = time.perf_counter() - extract_t0

    predict_t0 = time.perf_counter()
    preds = _predict_sequence_batch(
        [*ref_seqs, *alt_seqs],
        model=model,
        devices=devices,
        batch_size=batch_size,
        num_workers=num_workers,
        precision=precision,
    )
    predict_seconds = time.perf_counter() - predict_t0

    aggregate_t0 = time.perf_counter()
    intervals = pd.DataFrame.from_records(interval_records)
    site_summary, track_summary = summarize_contact_boundary_strength(
        preds[:n_variants],
        preds[n_variants:],
        variants_with_sites,
        intervals,
        input_len=input_len,
        window_bp=window_bp,
        min_distance_bp=min_distance_bp,
    )
    aggregate_seconds = time.perf_counter() - aggregate_t0

    if timing_records is not None:
        timing_records.append(
            {
                "chunk": 0,
                "start": 0,
                "stop": n_variants,
                "n_variants": n_variants,
                "extract_seconds": extract_seconds,
                "predict_seconds": predict_seconds,
                "aggregate_seconds": aggregate_seconds,
                "total_seconds": extract_seconds + predict_seconds + aggregate_seconds,
            }
        )

    return site_summary, track_summary, intervals


def aggregate_delta_features(
    ref_preds: np.ndarray,
    alt_preds: np.ndarray,
    variants: pd.DataFrame,
    track_groups: Mapping[str, Sequence[int]],
    window_bins: Mapping[str, int] = DEFAULT_WINDOW_BINS,
) -> pd.DataFrame:
    """Aggregate baseline and delta predictions into grouped feature columns."""

    if ref_preds.shape != alt_preds.shape:
        raise ValueError(f"Prediction shape mismatch: {ref_preds.shape} vs {alt_preds.shape}")
    if ref_preds.ndim != 3:
        raise ValueError(f"Expected predictions with shape sites x tracks x bins, got {ref_preds.shape}")

    delta = alt_preds - ref_preds
    n_sites, n_tracks, n_bins = delta.shape
    center_bin = n_bins // 2
    records: list[dict] = []

    for site_idx in range(n_sites):
        record = {"site_id": variants.iloc[site_idx]["site_id"]}
        for group_name, indices in track_groups.items():
            valid_indices = [idx for idx in indices if 0 <= idx < n_tracks]
            if not valid_indices:
                continue
            for window_name, radius in window_bins.items():
                left = max(0, center_bin - int(radius))
                right = min(n_bins, center_bin + int(radius) + 1)
                d = delta[site_idx, valid_indices, left:right]
                r = ref_preds[site_idx, valid_indices, left:right]
                record[f"{group_name}__{window_name}__delta_mean"] = float(np.mean(d))
                record[f"{group_name}__{window_name}__delta_abs_mean"] = float(np.mean(np.abs(d)))
                record[f"{group_name}__{window_name}__delta_abs_max"] = float(np.max(np.abs(d)))
                record[f"{group_name}__{window_name}__delta_l2"] = float(np.linalg.norm(d))
                record[f"{group_name}__{window_name}__ref_mean"] = float(np.mean(r))
        records.append(record)

    features = pd.DataFrame.from_records(records)
    return variants[["site_id", "chrom", "position", "site_start", "site_end"]].merge(
        features, on="site_id", how="left"
    )


def fit_context_embedding(
    features: pd.DataFrame,
    n_components: int = 8,
    random_state: int = 0,
) -> tuple[pd.DataFrame, object]:
    """Fit PCA on numeric feature columns and return per-site coordinates."""

    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    numeric = features.select_dtypes(include=[np.number]).drop(
        columns=["position", "site_start", "site_end"], errors="ignore"
    )
    if numeric.empty:
        raise ValueError("No numeric feature columns available for PCA")
    n_components = max(1, min(n_components, numeric.shape[0], numeric.shape[1]))
    scaler = StandardScaler()
    matrix = scaler.fit_transform(numeric.fillna(0.0))
    pca = PCA(n_components=n_components, random_state=random_state)
    coords = pca.fit_transform(matrix)
    embedding = pd.DataFrame(coords, columns=[f"PC{i + 1}" for i in range(n_components)])
    embedding.insert(0, "site_id", features["site_id"].values)
    return embedding, {"scaler": scaler, "pca": pca, "feature_columns": list(numeric.columns)}


def cluster_contexts(
    embedding: pd.DataFrame,
    n_clusters: int = 6,
    random_state: int = 0,
) -> tuple[pd.DataFrame, object]:
    """Cluster PCA coordinates with KMeans."""

    from sklearn.cluster import KMeans

    cols = [c for c in embedding.columns if c.startswith("PC")]
    if not cols:
        raise ValueError("Embedding does not contain PC columns")
    n_clusters = max(1, min(n_clusters, len(embedding)))
    km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = km.fit_predict(embedding[cols].fillna(0.0))
    clusters = embedding[["site_id"]].copy()
    clusters["cluster"] = labels
    return clusters, km


def _mean_col(df: pd.DataFrame, needle: str) -> float:
    cols = [c for c in df.columns if needle in c]
    if not cols:
        return 0.0
    return float(df[cols].abs().mean(axis=1).mean())


def annotate_context_clusters(
    features: pd.DataFrame,
    site_clusters: pd.DataFrame,
) -> pd.DataFrame:
    """Assign conservative descriptive labels to each cluster."""

    merged = features.merge(site_clusters, on="site_id", how="inner")
    records: list[dict] = []
    for cluster, group in merged.groupby("cluster"):
        ctcf = _mean_col(group, "ctcf_binding__local__delta")
        cohesin = _mean_col(group, "cohesin_binding__local__delta")
        active = _mean_col(group, "activity_marks__regional__delta")
        expression = _mean_col(group, "global_expression_outputs__regional__delta")
        repressive_ref = _mean_col(group, "repressive_or_inactive_context__broad__ref")
        total = _mean_col(group, "__delta_abs_mean")

        if total < 1e-6:
            label = "Robust"
        elif ctcf > 0 and cohesin >= 0.5 * ctcf:
            label = "CTCF-cohesin dependent"
        elif ctcf > 0 and cohesin < 0.25 * ctcf and max(active, expression) < 0.25 * ctcf:
            label = "CTCF-only / weak-cohesin"
        elif max(active, expression) >= max(ctcf, cohesin):
            label = "Enhancer-associated regulatory"
        elif repressive_ref > max(active, expression, ctcf, cohesin):
            label = "Repressive / compartment-B-like"
        else:
            label = "Active promoter / CTCF-independent cohesin-like"

        records.append(
            {
                "cluster": cluster,
                "n_sites": len(group),
                "context_label": label,
                "ctcf_delta": ctcf,
                "cohesin_delta": cohesin,
                "activity_delta": active,
                "expression_delta": expression,
                "repressive_ref": repressive_ref,
            }
        )
    return pd.DataFrame.from_records(records).sort_values("cluster").reset_index(drop=True)


def save_run_config(output_dir: str | Path, config: Mapping) -> None:
    """Write a JSON run config."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "run_config.json").open("w") as handle:
        json.dump(dict(config), handle, indent=2, sort_keys=True, default=str)
