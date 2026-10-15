"""Run-level orchestration for prepared targeted-ISM manifests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .gene_runner import GeneRunResult, TargetedRunConfig, run_gene
from .model_adapters.base import SequenceToProfileModel
from .profiles import load_stored_mutation_profiles
from .validation import expected_feature_rows, validate_feature_table


def select_prepared_genes(
    genes: pd.DataFrame,
    loci: pd.DataFrame,
    readouts: pd.DataFrame,
    manifest: pd.DataFrame,
    requested: str | None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Filter all prepared tables to one optional comma-separated gene set."""

    if not requested:
        return genes, loci, readouts, manifest
    requested_genes = [item.strip() for item in requested.split(",") if item.strip()]
    unknown = sorted(set(requested_genes) - set(genes["gene"]))
    if unknown:
        raise ValueError(f"Unknown prepared genes requested: {unknown}")
    filtered = [
        table.loc[table["gene"].isin(requested_genes)].copy()
        for table in (genes, loci, readouts, manifest)
    ]
    return filtered[0], filtered[1], filtered[2], filtered[3]


def model_output_geometry(adapter: SequenceToProfileModel) -> tuple[int, int]:
    """Return model output span and symmetric input crop in base pairs."""

    output_span = int(adapter.output_length_bins) * int(adapter.output_resolution_bp)
    crop_delta = int(adapter.input_length_bp) - output_span
    if crop_delta < 0 or crop_delta % 2:
        raise ValueError(
            f"Unsupported asymmetric output geometry: "
            f"input={adapter.input_length_bp}, output={output_span}"
        )
    return output_span, crop_delta // 2


def combine_run_artifacts(
    *,
    config: TargetedRunConfig,
    genes: pd.DataFrame,
    readouts: pd.DataFrame,
    manifest: pd.DataFrame,
    results: list[GeneRunResult],
    n_tracks: int,
    feature_dir: Path,
    profile_dir: Path,
    out_dir: Path,
):
    """Combine per-gene outputs and return validation diagnostics."""

    combined = pd.concat(
        [pd.read_csv(result.feature_path, sep="\t") for result in results],
        ignore_index=True,
    )
    expected_rows = expected_feature_rows(
        genes,
        readouts,
        manifest,
        n_tracks=n_tracks,
    )
    diagnostics = validate_feature_table(combined, expected_rows=expected_rows)
    deterministic_diffs = [result.deterministic_diff for result in results]
    if deterministic_diffs and max(deterministic_diffs) > 1e-6:
        raise RuntimeError(f"Determinism check failed: {deterministic_diffs}")

    profile_nonfinite = 0
    stored_profile_bytes = 0
    if config.save_log2fc_profiles:
        for gene in genes["gene"]:
            stored = load_stored_mutation_profiles(
                profile_dir,
                gene=gene,
                resolution_bp=config.profile_resolution_bp,
            )
            profile_nonfinite += int((~np.isfinite(stored.values)).sum())
            stored_profile_bytes += int(stored.values_path.stat().st_size)
        if profile_nonfinite:
            raise RuntimeError(
                f"Stored log2FC profiles contain {profile_nonfinite} nonfinite values"
            )

    combined.to_csv(feature_dir / "combined_mutation_features.tsv", sep="\t", index=False)
    try:
        combined.to_parquet(feature_dir / "combined_mutation_features.parquet", index=False)
    except Exception as exc:
        (feature_dir / "parquet_write_error.txt").write_text(str(exc) + "\n")
    pd.DataFrame.from_records([result.sequence_context for result in results]).to_csv(
        out_dir / "sequence_contexts.tsv", sep="\t", index=False
    )
    return diagnostics, profile_nonfinite, stored_profile_bytes
