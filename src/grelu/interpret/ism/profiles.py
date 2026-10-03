"""Storage helpers for mutation-by-track ISM profiles."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd


PROFILE_INDEX_COLUMNS = (
    "mutation_id",
    "edit_start",
    "edit_end",
    "edit_center_position",
    "variant_offset_from_tss_transcription_bp",
    "ref_sequence",
    "alt_sequence",
    "replacement_replicate",
)


def aggregate_count_profiles(
    profiles: np.ndarray,
    *,
    native_bp: int,
    target_bp: int,
) -> np.ndarray:
    """Sum adjacent count-profile bins to a coarser resolution."""

    values = np.asarray(profiles)
    native_bp = int(native_bp)
    target_bp = int(target_bp)
    if target_bp < native_bp or target_bp % native_bp:
        raise ValueError(
            "target resolution must be a multiple of native resolution: "
            f"{target_bp} vs {native_bp}"
        )
    factor = target_bp // native_bp
    if values.shape[-1] % factor:
        raise ValueError(
            f"Profile bins {values.shape[-1]} are not divisible by aggregation factor {factor}"
        )
    if factor == 1:
        return values
    shape = (*values.shape[:-1], values.shape[-1] // factor, factor)
    return values.reshape(shape).sum(axis=-1)


def profile_log2_fold_change(
    alternate: np.ndarray,
    reference: np.ndarray,
    *,
    pseudocount: float,
) -> np.ndarray:
    """Return bin-level ``log2((ALT+p)/(REF+p))`` profile values."""

    return np.log2(
        (np.maximum(alternate, 0.0) + float(pseudocount))
        / (np.maximum(reference, 0.0) + float(pseudocount))
    )


def reconstruct_alternate_profiles(
    reference: np.ndarray,
    log2_fold_change: np.ndarray,
    *,
    pseudocount: float,
) -> np.ndarray:
    """Reconstruct ALT profiles from a reference and stored log2 fold changes."""

    return (
        (np.asarray(reference) + float(pseudocount))
        * np.exp2(np.asarray(log2_fold_change, dtype=np.float32))
        - float(pseudocount)
    )


def save_reference_profiles(
    profile_dir: Path,
    *,
    gene: str,
    reference_profiles: np.ndarray,
    track_order: Sequence[str],
) -> None:
    """Save native reference profiles and their explicit track order."""

    profile_dir.mkdir(parents=True, exist_ok=True)
    np.save(profile_dir / f"{gene}.ref_profiles.npy", reference_profiles)
    (profile_dir / f"{gene}.track_order.json").write_text(
        json.dumps(list(track_order), indent=2) + "\n"
    )


@dataclass(frozen=True)
class StoredMutationProfiles:
    """Loaded profile artifacts for one gene."""

    values: np.ndarray
    reference: np.ndarray
    index: pd.DataFrame
    track_order: list[str]
    metadata: dict[str, object]
    values_path: Path


def load_stored_mutation_profiles(
    profile_dir: Path,
    *,
    gene: str,
    resolution_bp: int,
) -> StoredMutationProfiles:
    """Load one gene's stored mutation profiles and companion metadata."""

    values_path = profile_dir / f"{gene}.alt_log2fc_{resolution_bp}bp.npy"
    return StoredMutationProfiles(
        values=np.load(values_path, mmap_mode="r"),
        reference=np.load(profile_dir / f"{gene}.ref_profiles_{resolution_bp}bp.npy"),
        index=pd.read_csv(profile_dir / f"{gene}.profile_index.tsv", sep="\t"),
        track_order=json.loads((profile_dir / f"{gene}.track_order.json").read_text()),
        metadata=json.loads((profile_dir / f"{gene}.profile_metadata.json").read_text()),
        values_path=values_path,
    )


class MutationProfileWriter:
    """Stream mutation log2FC profiles and write their companion artifacts."""

    def __init__(
        self,
        profile_dir: Path,
        *,
        gene: str,
        mutations: pd.DataFrame,
        track_order: Sequence[str],
        reference_profiles: np.ndarray,
        native_resolution_bp: int,
        stored_resolution_bp: int,
        dtype: str,
        pseudocount: float,
        output_start: int,
    ) -> None:
        self.profile_dir = profile_dir
        self.gene = gene
        self.track_order = list(track_order)
        self.native_resolution_bp = int(native_resolution_bp)
        self.stored_resolution_bp = int(stored_resolution_bp)
        self.pseudocount = float(pseudocount)
        self.profile_dir.mkdir(parents=True, exist_ok=True)

        save_reference_profiles(
            profile_dir,
            gene=gene,
            reference_profiles=reference_profiles,
            track_order=self.track_order,
        )
        self.reference = aggregate_count_profiles(
            np.maximum(reference_profiles, 0.0),
            native_bp=self.native_resolution_bp,
            target_bp=self.stored_resolution_bp,
        ).astype(np.float32, copy=False)
        np.save(
            profile_dir / f"{gene}.ref_profiles_{self.stored_resolution_bp}bp.npy",
            self.reference,
        )

        self.final_path = profile_dir / f"{gene}.alt_log2fc_{self.stored_resolution_bp}bp.npy"
        self.partial_path = self.final_path.with_suffix(".partial.npy")
        self._array = np.lib.format.open_memmap(
            self.partial_path,
            mode="w+",
            dtype=np.dtype(dtype),
            shape=(len(mutations), len(self.track_order), self.reference.shape[-1]),
        )
        self._write_index(mutations)
        self._write_metadata(output_start)

    @property
    def dtype(self) -> np.dtype:
        return self._array.dtype

    def _write_index(self, mutations: pd.DataFrame) -> None:
        index = mutations.loc[:, PROFILE_INDEX_COLUMNS].copy()
        index.insert(0, "profile_row", np.arange(len(index)))
        index.to_csv(
            self.profile_dir / f"{self.gene}.profile_index.tsv",
            sep="\t",
            index=False,
        )

    def _write_metadata(self, output_start: int) -> None:
        metadata = {
            "gene": self.gene,
            "array": self.final_path.name,
            "shape": list(self._array.shape),
            "dtype": str(self._array.dtype),
            "native_resolution_bp": self.native_resolution_bp,
            "stored_resolution_bp": self.stored_resolution_bp,
            "aggregation": (
                "sum adjacent nonnegative count bins, then "
                f"log2((ALT+{self.pseudocount:g})/(REF+{self.pseudocount:g}))"
            ),
            "pseudocount_per_stored_bin": self.pseudocount,
            "track_order": self.track_order,
            "profile_index": f"{self.gene}.profile_index.tsv",
            "reference_profile": (
                f"{self.gene}.ref_profiles_{self.stored_resolution_bp}bp.npy"
            ),
            "output_start": int(output_start),
        }
        (self.profile_dir / f"{self.gene}.profile_metadata.json").write_text(
            json.dumps(metadata, indent=2) + "\n"
        )

    def write_batch(self, row_start: int, alternate_profiles: np.ndarray) -> None:
        """Aggregate and write one contiguous ALT-profile batch."""

        alternate = aggregate_count_profiles(
            np.maximum(alternate_profiles, 0.0),
            native_bp=self.native_resolution_bp,
            target_bp=self.stored_resolution_bp,
        )
        row_end = int(row_start) + len(alternate)
        self._array[row_start:row_end] = profile_log2_fold_change(
            alternate,
            self.reference[None, ...],
            pseudocount=self.pseudocount,
        ).astype(self._array.dtype, copy=False)

    def finalize(self) -> Path:
        """Flush the memmap and atomically expose the completed profile array."""

        self._array.flush()
        del self._array
        os.replace(self.partial_path, self.final_path)
        return self.final_path
