"""Common model-adapter contracts for ISM."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence

import numpy as np


@dataclass(frozen=True)
class TrackSpec:
    track_id: str
    channel_index: int
    task_name: str
    cell_type: str
    modality: str
    resolution_bp: int
    source: str


class SequenceToProfileModel(Protocol):
    model_id: str
    input_length_bp: int
    output_resolution_bp: int
    output_length_bins: int
    track_specs: list[TrackSpec]

    def predict_profiles(
        self,
        sequences: Sequence[str],
        tracks: Sequence[str] | None = None,
    ) -> np.ndarray:
        """Return predictions shaped [n_sequence, n_track, n_bin]."""
