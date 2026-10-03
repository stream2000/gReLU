"""Validation and provenance helpers shared by ISM runners."""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd


CORE_FEATURE_NUMERIC_COLUMNS = (
    "ref_mean",
    "alt_mean",
    "ref_sum",
    "alt_sum",
    "signed_delta_mean",
    "signed_delta_sum",
    "log2fc_mean",
    "log2fc_ratio_of_sums",
)
FEATURE_KEY_COLUMNS = ("model_backend", "mutation_id", "track_id", "readout_id")


@dataclass(frozen=True)
class FeatureTableDiagnostics:
    """Completeness diagnostics for a mutation-feature table."""

    feature_rows: int
    expected_rows: int
    nonfinite_core_values: int
    duplicate_feature_keys: int

    @property
    def is_valid(self) -> bool:
        return (
            self.feature_rows == self.expected_rows
            and self.nonfinite_core_values == 0
            and self.duplicate_feature_keys == 0
        )

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


def sha256_file(path: Path, chunk_size: int = 8 << 20) -> str:
    """Return a streaming SHA-256 digest for one file."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def directory_bytes(path: Path) -> int:
    """Return the total size of regular files below ``path``."""

    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def expected_feature_rows(
    genes: pd.DataFrame,
    readouts: pd.DataFrame,
    mutations: pd.DataFrame,
    *,
    n_tracks: int,
) -> int:
    """Calculate expected mutation × readout × track rows across genes."""

    return sum(
        int(mutations["gene"].eq(gene).sum())
        * int(readouts["gene"].eq(gene).sum())
        * int(n_tracks)
        for gene in genes["gene"]
    )


def feature_table_diagnostics(
    features: pd.DataFrame,
    *,
    expected_rows: int,
) -> FeatureTableDiagnostics:
    """Measure completeness, finite values, and key uniqueness."""

    nonfinite = int(
        (~np.isfinite(features.loc[:, CORE_FEATURE_NUMERIC_COLUMNS].to_numpy(dtype=float))).sum()
    )
    duplicates = int(features.duplicated(list(FEATURE_KEY_COLUMNS)).sum())
    return FeatureTableDiagnostics(
        feature_rows=len(features),
        expected_rows=int(expected_rows),
        nonfinite_core_values=nonfinite,
        duplicate_feature_keys=duplicates,
    )


def validate_feature_table(
    features: pd.DataFrame,
    *,
    expected_rows: int,
) -> FeatureTableDiagnostics:
    """Return diagnostics or raise when a feature table is incomplete or invalid."""

    diagnostics = feature_table_diagnostics(features, expected_rows=expected_rows)
    if not diagnostics.is_valid:
        raise RuntimeError(diagnostics.to_dict())
    return diagnostics
