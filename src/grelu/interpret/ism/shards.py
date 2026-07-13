"""Merge complete per-gene targeted-ISM shards into one validated run."""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .profiles import load_stored_mutation_profiles
from .validation import expected_feature_rows, sha256_file, validate_feature_table


PROVENANCE_KEYS = (
    "model_id",
    "checkpoint_path",
    "weights_path",
    "checkpoint_sha256",
    "prepared_manifest_sha256",
    "input_length_bp",
    "output_resolution_bp",
    "output_length_bins",
)


@dataclass(frozen=True)
class GeneShardResult:
    """Copied feature data and audit metadata for one completed gene shard."""

    features: pd.DataFrame
    context: dict[str, object]
    source: Path
    profile_bytes: int
    profile_nonfinite_values: int


def atomic_copy(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".merge-tmp")
    shutil.copy2(source, temporary)
    os.replace(temporary, target)


def _load_prepared_tables(prepared: Path) -> dict[str, pd.DataFrame]:
    return {
        name: pd.read_csv(prepared / f"{name}.tsv", sep="\t")
        for name in ("genes", "loci", "readouts", "mutation_manifest")
    }


def _validate_shard_provenance(
    sources: list[Path],
) -> tuple[pd.DataFrame, dict[str, object]]:
    track_manifests = [
        pd.read_csv(source / "track_manifest.tsv", sep="\t") for source in sources
    ]
    metadata = [
        json.loads((source / "model_metadata.json").read_text())
        for source in sources
    ]
    baseline_tracks = track_manifests[0]
    for source, tracks in zip(sources[1:], track_manifests[1:]):
        try:
            pd.testing.assert_frame_equal(baseline_tracks, tracks, check_dtype=False)
        except AssertionError as exc:
            raise RuntimeError(
                f"Track manifest differs in shard {source}: {exc}"
            ) from exc

    baseline_metadata = metadata[0]
    for source, item in zip(sources[1:], metadata[1:]):
        differences = {
            key: (baseline_metadata.get(key), item.get(key))
            for key in PROVENANCE_KEYS
            if baseline_metadata.get(key) != item.get(key)
        }
        if differences:
            raise RuntimeError(
                f"Model provenance differs in shard {source}: {differences}"
            )
    return baseline_tracks, baseline_metadata


def _complete_gene_source(
    sources: list[Path],
    gene: str,
    expected_rows: int,
    profile_resolution_bp: int,
) -> Path:
    for source in sources:
        feature_path = source / f"features/{gene}.tsv"
        profile_path = (
            source
            / f"profiles/{gene}.alt_log2fc_{profile_resolution_bp}bp.npy"
        )
        if not feature_path.exists() or not profile_path.exists():
            continue
        if sum(1 for _ in feature_path.open()) - 1 == expected_rows:
            return source
    raise RuntimeError(
        f"No complete shard found for {gene}; expected {expected_rows} feature rows"
    )


def _copy_gene_outputs(
    source: Path,
    target: Path,
    gene: str,
    profile_resolution_bp: int,
) -> Path:
    source_feature = source / f"features/{gene}.tsv"
    target_feature = target / f"features/{gene}.tsv"
    if source_feature != target_feature:
        atomic_copy(source_feature, target_feature)
    for item in (source / "profiles").glob(f"{gene}.*"):
        destination = target / "profiles" / item.name
        if item != destination:
            atomic_copy(item, destination)
    partial = (
        target
        / "profiles"
        / f"{gene}.alt_log2fc_{profile_resolution_bp}bp.partial.npy"
    )
    if partial.exists():
        partial.unlink()
    return target_feature


def _gene_context(
    *,
    gene_row,
    features: pd.DataFrame,
    mutation_count: int,
    readout_count: int,
    track_count: int,
    profile_resolution_bp: int,
    profile_path: Path,
    model_metadata: dict[str, object],
) -> dict[str, object]:
    input_length = int(model_metadata["input_length_bp"])
    output_span = int(model_metadata["output_resolution_bp"]) * int(
        model_metadata["output_length_bins"]
    )
    seq_start = int(gene_row.analysis_tss) - input_length // 2
    output_start = int(features.output_start.iloc[0])
    return {
        "gene": str(gene_row.gene),
        "chrom": str(gene_row.chrom),
        "seq_start": seq_start,
        "seq_end": seq_start + input_length,
        "output_start": output_start,
        "output_end": output_start + output_span,
        "mutations": mutation_count,
        "readouts": readout_count,
        "tracks": track_count,
        "feature_rows": len(features),
        "deterministic_ref_max_abs_diff": 0.0,
        "stored_profile_resolution_bp": profile_resolution_bp,
        "stored_profile_path": str(profile_path),
    }


def _collect_gene_shard(
    *,
    gene_row,
    sources: list[Path],
    target: Path,
    manifest: pd.DataFrame,
    readouts: pd.DataFrame,
    track_count: int,
    profile_resolution_bp: int,
    model_metadata: dict[str, object],
) -> GeneShardResult:
    gene = str(gene_row.gene)
    mutation_ids = manifest.loc[manifest.gene.eq(gene), "mutation_id"].tolist()
    readout_count = int(readouts.gene.eq(gene).sum())
    expected_rows = len(mutation_ids) * readout_count * track_count
    source = _complete_gene_source(
        sources, gene, expected_rows, profile_resolution_bp
    )
    feature_path = _copy_gene_outputs(
        source, target, gene, profile_resolution_bp
    )
    features = pd.read_csv(feature_path, sep="\t")
    stored = load_stored_mutation_profiles(
        target / "profiles",
        gene=gene,
        resolution_bp=profile_resolution_bp,
    )
    if stored.index.mutation_id.tolist() != mutation_ids:
        raise RuntimeError(f"Profile index differs from prepared manifest for {gene}")
    expected_shape = (len(mutation_ids), track_count, stored.reference.shape[-1])
    if stored.values.shape != expected_shape or len(stored.track_order) != track_count:
        raise RuntimeError(
            f"Profile shape/track mismatch for {gene}: "
            f"{stored.values.shape}, {expected_shape}"
        )
    nonfinite = int((~np.isfinite(stored.values)).sum())
    return GeneShardResult(
        features=features,
        context=_gene_context(
            gene_row=gene_row,
            features=features,
            mutation_count=len(mutation_ids),
            readout_count=readout_count,
            track_count=track_count,
            profile_resolution_bp=profile_resolution_bp,
            profile_path=stored.values_path,
            model_metadata=model_metadata,
        ),
        source=source,
        profile_bytes=int(stored.values_path.stat().st_size),
        profile_nonfinite_values=nonfinite,
    )


def _write_combined_outputs(
    target: Path,
    combined: pd.DataFrame,
    contexts: list[dict[str, object]],
) -> None:
    combined.to_csv(
        target / "features/combined_mutation_features.tsv", sep="\t", index=False
    )
    combined.to_parquet(
        target / "features/combined_mutation_features.parquet", index=False
    )
    pd.DataFrame.from_records(contexts).to_csv(
        target / "sequence_contexts.tsv", sep="\t", index=False
    )


def finalize_sharded_run(
    *,
    prepared: Path,
    target: Path,
    sources: list[Path],
    profile_resolution_bp: int,
) -> dict[str, object]:
    """Merge compatible gene shards and return the final validation summary."""

    tables = _load_prepared_tables(prepared)
    genes = tables["genes"]
    loci = tables["loci"]
    readouts = tables["readouts"]
    manifest = tables["mutation_manifest"]
    tracks, model_metadata = _validate_shard_provenance(sources)
    track_count = len(tracks)

    (target / "features").mkdir(parents=True, exist_ok=True)
    (target / "profiles").mkdir(parents=True, exist_ok=True)
    atomic_copy(sources[0] / "track_manifest.tsv", target / "track_manifest.tsv")
    atomic_copy(sources[0] / "model_metadata.json", target / "model_metadata.json")

    results = [
        _collect_gene_shard(
            gene_row=gene_row,
            sources=sources,
            target=target,
            manifest=manifest,
            readouts=readouts,
            track_count=track_count,
            profile_resolution_bp=profile_resolution_bp,
            model_metadata=model_metadata,
        )
        for gene_row in genes.itertuples(index=False)
    ]
    combined = pd.concat([result.features for result in results], ignore_index=True)
    expected_total = expected_feature_rows(
        genes, readouts, manifest, n_tracks=track_count
    )
    diagnostics = validate_feature_table(combined, expected_rows=expected_total)
    profile_nonfinite = sum(
        result.profile_nonfinite_values for result in results
    )
    if profile_nonfinite:
        raise RuntimeError({"profile_nonfinite_values": profile_nonfinite})
    _write_combined_outputs(
        target, combined, [result.context for result in results]
    )

    validation = {
        "status": "ok",
        "model_backend": str(combined.model_backend.iloc[0]),
        "model_id": str(combined.model_id.iloc[0]),
        "genes": genes.gene.tolist(),
        "loci": loci.locus_id.tolist(),
        "mutations": int(len(manifest)),
        "selected_tracks": track_count,
        "track_groups": sorted(tracks.track_group.unique().tolist()),
        "readouts": int(len(readouts)),
        "expected_feature_rows": int(expected_total),
        "feature_rows": int(len(combined)),
        "nonfinite_core_values": diagnostics.nonfinite_core_values,
        "duplicate_feature_keys": diagnostics.duplicate_feature_keys,
        "deterministic_ref_max_abs_diff": 0.0,
        "saved_log2fc_profiles": True,
        "stored_profile_resolution_bp": profile_resolution_bp,
        "stored_profile_dtype": str(
            np.load(
                target
                / f"profiles/{genes.gene.iloc[0]}.alt_log2fc_{profile_resolution_bp}bp.npy",
                mmap_mode="r",
            ).dtype
        ),
        "stored_profile_bytes": sum(result.profile_bytes for result in results),
        "stored_profile_nonfinite_values": profile_nonfinite,
        "finalized_from_shards": [str(source) for source in sources],
        "gene_sources": {
            str(gene): str(result.source)
            for gene, result in zip(genes.gene, results)
        },
        "prepared_manifest_sha256": sha256_file(
            prepared / "mutation_manifest.tsv"
        ),
    }
    (target / "validation_summary.json").write_text(
        json.dumps(validation, indent=2) + "\n"
    )
    return validation
