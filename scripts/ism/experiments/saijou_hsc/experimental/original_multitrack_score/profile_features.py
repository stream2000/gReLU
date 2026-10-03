"""Validated profile and track-feature readers for multitrack scoring."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Collection

import numpy as np
import pandas as pd

from score_calculations import CENTER_KEYS


LOCAL_BIN_OFFSETS = np.array([-1, 0, 1], dtype=int)


def replacement_mad(values: pd.Series) -> float:
    """Median absolute deviation across a center's shuffle replacements.

    Mirrors the fine-tuned pipeline so that shuffle-to-shuffle spread means the
    same thing on both sides of the workbench.
    """
    array = values.to_numpy(dtype=float)
    return float(np.median(np.abs(array - np.median(array))))


def load_validated_track_manifest(
    root: Path, run_name: str
) -> pd.DataFrame:
    """Load a run's channel manifest and validate its scoring schema."""
    tracks = pd.read_csv(
        root / "runs" / run_name / "track_manifest.tsv", sep="\t"
    ).sort_values("channel_index")
    required = {
        "channel_index",
        "track_id",
        "track_group",
        "modality",
    }
    missing = sorted(required - set(tracks.columns))
    if missing:
        raise ValueError(f"Track manifest is missing columns: {missing}")
    return tracks


def summarize_local_profile_effects(
    root: Path,
    *,
    gene: str,
    run_name: str,
    output_modalities: Collection[str],
) -> pd.DataFrame:
    """Collapse mutation-local saved profiles into center-by-track effects."""
    profile_dir = root / "runs" / run_name / "profiles"
    metadata = json.loads(
        (profile_dir / f"{gene}.profile_metadata.json").read_text()
    )
    profile_index = pd.read_csv(
        profile_dir / metadata["profile_index"], sep="\t"
    )
    tracks = load_validated_track_manifest(root, run_name)
    track_order = json.loads(
        (profile_dir / f"{gene}.track_order.json").read_text()
    )
    if tracks.track_id.tolist() != track_order:
        raise ValueError(f"{gene}: profile track order does not match manifest")

    local_tracks = tracks.loc[
        ~tracks.modality.isin(output_modalities)
    ].copy()
    array = np.load(profile_dir / metadata["array"], mmap_mode="r")
    if tuple(array.shape) != tuple(metadata["shape"]):
        raise ValueError(f"{gene}: stored profile shape mismatch")
    if not profile_index.profile_row.eq(np.arange(len(profile_index))).all():
        raise ValueError(f"{gene}: profile rows are not in saved array order")

    resolution = int(metadata["stored_resolution_bp"])
    output_start = int(metadata["output_start"])
    center_bins = (
        profile_index.edit_center_position.to_numpy(dtype=int) - output_start
    ) // resolution
    channel_indices = local_tracks.channel_index.to_numpy(dtype=int)
    selected_bins = (
        center_bins[:, None, None] + LOCAL_BIN_OFFSETS[None, None, :]
    )
    if selected_bins.min() < 0 or selected_bins.max() >= array.shape[2]:
        raise ValueError(f"{gene}: local profile window is outside output")

    values = np.asarray(
        array[
            np.arange(len(profile_index))[:, None, None],
            channel_indices[None, :, None],
            selected_bins,
        ],
        dtype=np.float32,
    )
    strongest_bin = np.abs(values).argmax(axis=2)
    mutation_absolute = np.max(np.abs(values), axis=2)
    mutation_signed = np.take_along_axis(
        values, strongest_bin[:, :, None], axis=2
    )[:, :, 0]

    # Reference level is read at the edit's own bin rather than the strongest
    # bin: it must not depend on which mutation is being scored, otherwise it
    # could not serve as a baseline for that position.
    reference = np.asarray(
        np.load(profile_dir / metadata["reference_profile"], mmap_mode="r"),
        dtype=np.float32,
    )
    if reference.shape[0] != array.shape[1]:
        raise ValueError(f"{gene}: reference profile track count mismatch")
    reference_level = reference[
        channel_indices[None, :], center_bins[:, None]
    ]

    mutation_tracks = pd.DataFrame(
        {
            "gene": gene,
            "variant_offset_from_tss_transcription_bp": np.repeat(
                profile_index.variant_offset_from_tss_transcription_bp.to_numpy(),
                len(channel_indices),
            ),
            "replacement_replicate": np.repeat(
                profile_index.replacement_replicate.to_numpy(),
                len(channel_indices),
            ),
            "channel_index": np.tile(channel_indices, len(profile_index)),
            "view_absolute_log2fc": mutation_absolute.ravel(),
            "view_signed_log2fc": mutation_signed.ravel(),
            "view_reference_level": reference_level.ravel(),
        }
    ).merge(
        local_tracks[
            ["channel_index", "track_id", "track_group", "modality"]
        ],
        on="channel_index",
        how="left",
        validate="many_to_one",
    )
    if not np.isfinite(
        mutation_tracks[
            [
                "view_absolute_log2fc",
                "view_signed_log2fc",
                "view_reference_level",
            ]
        ].to_numpy()
    ).all():
        raise ValueError(f"{gene}: local profile extraction is non-finite")

    centers = (
        mutation_tracks.groupby(
            [*CENTER_KEYS, "track_id", "track_group", "modality"],
            sort=False,
        )
        .agg(
            replacements=("replacement_replicate", "nunique"),
            median_absolute_log2fc=("view_absolute_log2fc", "median"),
            median_signed_log2fc=("view_signed_log2fc", "median"),
            replacement_mad=("view_signed_log2fc", replacement_mad),
            reference_level=("view_reference_level", "median"),
        )
        .reset_index()
    )
    centers["score_view"] = "local_regulatory"
    centers["readout_definition"] = (
        f"max_abs_log2fc_in_edit_bin_plus_neighbors_{resolution}bp"
    )
    return centers
