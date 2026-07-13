"""Shared sequence and track-selection helpers for profile adapters."""

from __future__ import annotations

import re
from typing import Sequence

import numpy as np
import torch

from .base import TrackSpec


def slug_identifier(value: str) -> str:
    """Normalize a track label for stable artifact identifiers."""

    return re.sub(r"[^a-z0-9]+", "_", str(value).lower()).strip("_")


def sequence_to_tensor(sequence: str) -> torch.Tensor:
    """Encode a DNA string as a channel-first one-hot tensor."""

    from grelu.sequence.format import BASE_TO_INDEX_HASH, indices_to_one_hot

    sequence = str(sequence).upper()
    n_index = BASE_TO_INDEX_HASH["N"]
    indices = np.fromiter(
        (BASE_TO_INDEX_HASH.get(base, n_index) for base in sequence),
        dtype=np.int8,
        count=len(sequence),
    )
    one_hot = indices_to_one_hot(indices)
    if one_hot.shape[0] != 4 and one_hot.shape[-1] == 4:
        one_hot = one_hot.T
    if one_hot.shape[0] != 4:
        raise ValueError(
            f"Expected channel-first one-hot sequence, got {tuple(one_hot.shape)}"
        )
    return one_hot.contiguous()


def sequences_to_tensor(
    sequences: Sequence[str],
    *,
    expected_length: int,
) -> torch.Tensor:
    """Validate sequence lengths and stack channel-first tensors."""

    tensors = []
    for sequence in sequences:
        if len(sequence) != expected_length:
            raise ValueError(
                f"Expected {expected_length} bp sequence, got {len(sequence)}"
            )
        tensors.append(sequence_to_tensor(sequence))
    if not tensors:
        raise ValueError("No sequences provided for prediction")
    return torch.stack(tensors, dim=0)


def track_indices(
    track_specs: Sequence[TrackSpec],
    tracks: Sequence[str] | None,
) -> list[int]:
    """Map public track IDs to adapter output channel indices."""

    if tracks is None:
        return list(range(len(track_specs)))
    by_name = {spec.track_id: spec.channel_index for spec in track_specs}
    missing = [track for track in tracks if track not in by_name]
    if missing:
        raise ValueError(f"Unknown tracks requested: {missing}")
    return [by_name[track] for track in tracks]
