#!/usr/bin/env python3
"""E0 source and comparability audit for the frozen Mdk motif--ISM plan.

This entry point deliberately reads existing artifacts only.  It creates an
append-only, versioned audit package and does not prepare edits, run inference,
score regions, or annotate motifs.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import pyarrow


REPO_ROOT = Path(__file__).resolve().parents[6]
DEFAULT_PRIMARY_ROOT = (
    REPO_ROOT / "experiments/ism/20260903_nine_gene_tss6kb_strict_shuffle"
)
DEFAULT_AG_SNV_ROOT = REPO_ROOT / "experiments/ism/saijou_hsc_ag_tss1kb_full/Mdk"
DEFAULT_BZ_SNV_ROOT = REPO_ROOT / "experiments/ism/saijou_hsc_borzoi_tss1kb_full/Mdk"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "experiments/ism/20260909_mdk_motif_ism_v1"
BACKENDS = (
    "alphagenome_finetuned",
    "alphagenome_original",
    "borzoi_finetuned",
    "borzoi_original",
)
MANIFEST_COLUMNS = (
    "mutation_id",
    "gene",
    "chrom",
    "gene_strand",
    "gene_tss",
    "gene_tes",
    "edit_start",
    "edit_end",
    "edit_center_position",
    "variant_offset_from_tss_transcription_bp",
    "ref_sequence",
    "alt_sequence",
    "replacement_replicate",
)
FEATURE_STATIC_COLUMNS = (
    "mutation_id",
    "gene",
    "edit_start",
    "edit_end",
    "edit_center_position",
    "variant_offset_from_tss_transcription_bp",
    "ref_sequence",
    "alt_sequence",
    "replacement_replicate",
)
FEATURE_COLUMNS = FEATURE_STATIC_COLUMNS + (
    "track_id",
    "readout_id",
    "log2fc_ratio_of_sums",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary-root", type=Path, default=DEFAULT_PRIMARY_ROOT)
    parser.add_argument("--ag-snv-root", type=Path, default=DEFAULT_AG_SNV_ROOT)
    parser.add_argument("--bz-snv-root", type=Path, default=DEFAULT_BZ_SNV_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser.parse_args()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def path_record(
    source_id: str,
    role: str,
    path: Path,
    *,
    full_hash: bool,
    note: str,
) -> dict[str, object]:
    exists = path.exists()
    stat = path.stat() if exists else None
    return {
        "source_id": source_id,
        "role": role,
        "path": str(path.resolve()),
        "exists": exists,
        "size_bytes": stat.st_size if stat else None,
        "mtime_utc": (
            datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat()
            if stat
            else None
        ),
        "sha256": sha256(path) if exists and full_hash else None,
        "hash_scope": "full_file" if full_hash else "path_size_mtime_only",
        "note": note,
    }


def read_json(path: Path) -> dict[str, Any]:
    with path.open() as handle:
        return json.load(handle)


def write_tsv(path: Path, rows: list[dict[str, object]]) -> None:
    require(rows, f"Refusing to write an empty required table: {path.name}")
    columns = list(rows[0])
    require(
        all(list(row) == columns for row in rows),
        f"Inconsistent columns while writing {path.name}",
    )
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def load_mdk_manifest(primary_root: Path) -> pd.DataFrame:
    manifest_path = primary_root / "prepared/mutation_manifest.tsv"
    manifest = pd.read_csv(manifest_path, sep="\t", usecols=MANIFEST_COLUMNS)
    manifest = manifest.loc[manifest["gene"].eq("Mdk")].copy()
    require(len(manifest) == 8922, f"Expected 8,922 Mdk edits, found {len(manifest)}")
    require(
        not manifest.duplicated("mutation_id").any(),
        "Mdk manifest has duplicate mutation_id values",
    )
    require(manifest["gene_strand"].eq("-").all(), "Mdk strand is not uniformly '-' ")
    require(manifest["chrom"].eq("chr2").all(), "Mdk chrom is not uniformly chr2")
    require(manifest["gene_tss"].eq(91932297).all(), "Mdk TSS differs from contract")
    require(manifest["gene_tes"].eq(91929827).all(), "Mdk TES differs from contract")
    require(
        manifest["replacement_replicate"].isin((0, 1, 2)).all(),
        "Mdk manifest does not have the expected replacement indices",
    )
    per_center = manifest.groupby("edit_center_position", sort=False)["mutation_id"].size()
    require(
        len(per_center) == 2974 and per_center.eq(3).all(),
        "Mdk manifest does not contain three edits at each of 2,974 centers",
    )
    return manifest.sort_values("mutation_id").reset_index(drop=True)


def mdk_readouts(primary_root: Path) -> list[str]:
    readouts = pd.read_csv(primary_root / "prepared/readouts.tsv", sep="\t")
    result = readouts.loc[readouts["gene"].eq("Mdk"), "readout_id"].tolist()
    require(len(result) == 5 and len(set(result)) == 5, "Expected five unique Mdk readouts")
    return sorted(result)


def audit_backend(
    primary_root: Path,
    backend: str,
    manifest: pd.DataFrame,
    readouts: list[str],
    manifest_sha256: str,
) -> dict[str, object]:
    run_root = primary_root / "runs" / backend
    metadata = read_json(run_root / "model_metadata.json")
    validation = read_json(run_root / "validation_summary.json")
    tracks = pd.read_csv(run_root / "track_manifest.tsv", sep="\t")
    require(not tracks["track_id"].duplicated().any(), f"{backend} has duplicate track IDs")
    track_ids = sorted(tracks["track_id"].tolist())
    track_index = {track_id: index for index, track_id in enumerate(track_ids)}
    readout_index = {readout_id: index for index, readout_id in enumerate(readouts)}
    mutation_ids = manifest["mutation_id"].tolist()
    mutation_index = {mutation_id: index for index, mutation_id in enumerate(mutation_ids)}
    expected_static = {
        column: dict(zip(manifest["mutation_id"], manifest[column]))
        for column in FEATURE_STATIC_COLUMNS
        if column != "mutation_id"
    }
    expected_rows = len(mutation_ids) * len(track_ids) * len(readouts)
    seen = np.zeros(expected_rows, dtype=bool)
    unknown_mutation_rows = 0
    unknown_track_rows = 0
    unknown_readout_rows = 0
    duplicate_feature_keys = 0
    nonfinite_log2fc = 0
    static_mismatch = Counter()
    feature_rows = 0

    feature_path = run_root / "features/combined_mutation_features.parquet"
    scanner = ds.dataset(feature_path, format="parquet").scanner(
        columns=list(FEATURE_COLUMNS),
        filter=ds.field("gene") == "Mdk",
        batch_size=131072,
    )
    for batch in scanner.to_batches():
        frame = batch.to_pandas()
        feature_rows += len(frame)
        nonfinite_log2fc += int(
            (~np.isfinite(frame["log2fc_ratio_of_sums"].to_numpy(dtype=float))).sum()
        )
        mutation_positions = frame["mutation_id"].map(mutation_index)
        unknown_mutation_rows += int(mutation_positions.isna().sum())
        for column, expected in expected_static.items():
            values = frame[column]
            static_mismatch[column] += int(
                values.ne(frame["mutation_id"].map(expected)).sum()
            )
        track_positions = frame["track_id"].map(track_index)
        readout_positions = frame["readout_id"].map(readout_index)
        unknown_track_rows += int(track_positions.isna().sum())
        unknown_readout_rows += int(readout_positions.isna().sum())
        valid = (
            mutation_positions.notna()
            & track_positions.notna()
            & readout_positions.notna()
        )
        key_codes = (
            mutation_positions.loc[valid].astype(np.int64).to_numpy()
            * (len(track_ids) * len(readouts))
            + track_positions.loc[valid].astype(np.int64).to_numpy() * len(readouts)
            + readout_positions.loc[valid].astype(np.int64).to_numpy()
        )
        duplicate_feature_keys += int(seen[key_codes].sum())
        seen[key_codes] = True

    missing_feature_keys = int((~seen).sum())
    existing_validation_ok = validation.get("status") == "ok"
    metadata_manifest_match = metadata.get("prepared_manifest_sha256") == manifest_sha256
    status = "ok"
    if any(
        (
            feature_rows != expected_rows,
            unknown_mutation_rows,
            unknown_track_rows,
            unknown_readout_rows,
            duplicate_feature_keys,
            missing_feature_keys,
            nonfinite_log2fc,
            sum(static_mismatch.values()),
            not existing_validation_ok,
            not metadata_manifest_match,
        )
    ):
        status = "validation_failed"
    return {
        "backend": backend,
        "status": status,
        "model_id": metadata.get("model_id"),
        "existing_validation_status": validation.get("status"),
        "metadata_manifest_sha256": metadata.get("prepared_manifest_sha256"),
        "actual_manifest_sha256": manifest_sha256,
        "metadata_manifest_match": metadata_manifest_match,
        "mismatch_mutation_id_rows": unknown_mutation_rows,
        "mismatch_static_field_rows": sum(static_mismatch.values()),
        "static_mismatch_by_field": dict(sorted(static_mismatch.items())),
        "unknown_track_rows": unknown_track_rows,
        "unknown_readout_rows": unknown_readout_rows,
        "duplicate_feature_keys": duplicate_feature_keys,
        "missing_feature_keys": missing_feature_keys,
        "nonfinite_log2fc_ratio_of_sums": nonfinite_log2fc,
        "actual_feature_rows": feature_rows,
        "expected_feature_rows": expected_rows,
        "track_count": len(track_ids),
        "readout_count": len(readouts),
    }


def hsc_cage_audit(primary_root: Path) -> dict[str, object]:
    tracks = pd.read_csv(
        primary_root / "runs/borzoi_original/track_manifest.tsv", sep="\t"
    )
    hsc = tracks.loc[
        tracks["track_group"].eq("hsc_cage")
        & tracks["cell_type"].str.contains("hepatic stellate", case=False, na=False)
    ].copy()
    expected_negative = hsc.loc[hsc["strand"].eq("-")]
    return {
        "gene": "Mdk",
        "gene_strand": "-",
        "backend": "borzoi_original",
        "assay": "CAGE",
        "cell_name": "; ".join(sorted(hsc["cell_type"].dropna().unique())),
        "candidate_track_ids": "; ".join(sorted(hsc["track_id"].tolist())),
        "selected_transcribed_strand_track_id": (
            expected_negative["track_id"].iloc[0] if len(expected_negative) == 1 else None
        ),
        "strand_status": "ok" if len(hsc) == 2 and len(expected_negative) == 1 else "validation_failed",
        "interpretation": (
            "Mdk is negative-strand; use the negative genomic-strand CAGE track "
            "for any pre-registered BZ-original HSC CAGE display."
        ),
    }


def comparability_rows(primary_root: Path, ag_snv_root: Path, bz_snv_root: Path) -> list[dict[str, object]]:
    metadata = {
        backend: read_json(primary_root / "runs" / backend / "model_metadata.json")
        for backend in BACKENDS
    }
    ag_original_tracks = pd.read_csv(
        primary_root / "runs/alphagenome_original/track_manifest.tsv", sep="\t"
    )
    ag_has_hepatic_stellate = ag_original_tracks.astype(str).apply(
        lambda column: column.str.contains("hepatic stellate", case=False, na=False)
    ).any(axis=1).any()
    rows = [
        {
            "comparison_id": "ag_ft_vs_ag_original",
            "left_source": "alphagenome_finetuned",
            "right_source": "alphagenome_original",
            "mutation_identity": "ok_same_frozen_manifest",
            "input_context": "incomparable_524288bp_vs_1048576bp",
            "output_grid": "matched_128bp",
            "readout": "incomparable_readout",
            "assessment": "descriptive_same-edit comparison only",
            "evidence": (
                "Both metadata files name the same prepared-manifest SHA; "
                "FT is four-cell Saijou 10x pseudobulk while original is selected "
                "mouse RNA/CAGE/chromatin tracks, not a matched HSC RNA head. "
                f"AlphaGenome-original hepatic-stellate track present: {ag_has_hepatic_stellate}."
            ),
        },
        {
            "comparison_id": "bz_ft_vs_bz_original",
            "left_source": "borzoi_finetuned",
            "right_source": "borzoi_original",
            "mutation_identity": "ok_same_frozen_manifest",
            "input_context": "matched_524288bp",
            "output_grid": "matched_32bp",
            "readout": "incomparable_assay_10x_3prime_rna_vs_cage",
            "assessment": "descriptive_same-edit comparison only",
            "evidence": (
                "Both metadata files name the same prepared-manifest SHA. "
                "Original contains the verified Mouse hepatic Stellate Cells CAGE +/- pair; "
                "for negative-strand Mdk the pre-registered transcribed track is "
                "borzoi_mouse_631_cnhs13196 (-)."
            ),
        },
        {
            "comparison_id": "ft_alphagenome_vs_borzoi",
            "left_source": "alphagenome_finetuned",
            "right_source": "borzoi_finetuned",
            "mutation_identity": "ok_same_frozen_manifest",
            "input_context": "matched_524288bp",
            "output_grid": "incomparable_128bp_vs_32bp",
            "readout": "same_four_saijou_cell_labels_but_different_output_geometry",
            "assessment": "do_not_compare_effect_magnitudes_or_combine_scores",
            "evidence": "Both use the four Saijou 10x pseudobulk labels and pseudocount 1.0, but their stored/readout bin geometry differs.",
        },
        {
            "comparison_id": "original_alphagenome_vs_borzoi",
            "left_source": "alphagenome_original",
            "right_source": "borzoi_original",
            "mutation_identity": "ok_same_frozen_manifest",
            "input_context": "incomparable_1048576bp_vs_524288bp",
            "output_grid": "incomparable_128bp_vs_32bp",
            "readout": "incomparable_selected_track_panels",
            "assessment": "do_not_combine_scores; retain model-specific views",
            "evidence": "The original-model track panels, context lengths, and output grids differ.",
        },
        {
            "comparison_id": "ag_old_snv_vs_current_shuffle",
            "left_source": str(ag_snv_root.resolve()),
            "right_source": "alphagenome_finetuned current 6kb shuffle",
            "mutation_identity": "not_applicable_snv_vs_shuffle",
            "input_context": "requires_E5_metadata_check",
            "output_grid": "requires_E5_readout_check",
            "readout": "potential_TSS_1024bp_intersection_only",
            "assessment": "not a direct effect-size comparison; optional scale-response bridge only",
            "evidence": "Both sources record TSS anchor chr2:91932297; old source has 3,072 one-base SNVs while current source has 8,922 three-replacement 10-bp shuffles.",
        },
        {
            "comparison_id": "bz_old_snv_vs_current_shuffle",
            "left_source": str(bz_snv_root.resolve()),
            "right_source": "borzoi_finetuned current 6kb shuffle",
            "mutation_identity": "not_applicable_snv_vs_shuffle",
            "input_context": "requires_E5_metadata_check",
            "output_grid": "requires_E5_readout_check",
            "readout": "potential_TSS_1024bp_intersection_only",
            "assessment": "not a direct effect-size comparison; optional scale-response bridge only",
            "evidence": "Both sources record TSS anchor chr2:91932297. The old BZ manifest records TES 91929804, 23 bp from the current 91929827 TES; do not use TES as an alignment surrogate.",
        },
    ]
    for backend in BACKENDS:
        require(metadata[backend].get("pseudocount") == 1.0, f"{backend} pseudocount is not 1.0")
    return rows


def source_records(primary_root: Path, ag_snv_root: Path, bz_snv_root: Path) -> list[dict[str, object]]:
    records = [
        path_record("plan", "frozen_execution_plan", REPO_ROOT / "docs/mdk_motif_ism_execution_analysis_plan_20260909_zh.md", full_hash=True, note="E0 plan authority"),
        path_record("transcript_authority", "coordinate_authority", REPO_ROOT / "scripts/ism/experiments/saijou_hsc/configs/provided_nine_gene_transcripts.tsv", full_hash=True, note="Mdk-201/mm10 coordinate authority"),
        path_record("prepared_manifest", "primary_mutation_manifest", primary_root / "prepared/mutation_manifest.tsv", full_hash=True, note="Full hash: frozen mutation identity source"),
        path_record("prepared_readouts", "primary_readout_definition", primary_root / "prepared/readouts.tsv", full_hash=True, note="Full hash: readout windows and roles"),
        path_record("prepared_validation", "existing_validation", primary_root / "prepared/validation_summary.json", full_hash=True, note="Existing prepared-artifact validation"),
        path_record("analysis_validation", "existing_validation", primary_root / "analysis/validation_summary.json", full_hash=True, note="Existing cross-model analysis validation; not reused as a new score"),
    ]
    for backend in BACKENDS:
        run_root = primary_root / "runs" / backend
        records.extend(
            [
                path_record(f"{backend}_metadata", "model_metadata", run_root / "model_metadata.json", full_hash=True, note="Full hash: checkpoint/context/grid/pseudocount metadata"),
                path_record(f"{backend}_validation", "existing_validation", run_root / "validation_summary.json", full_hash=True, note="Full hash: existing run validation"),
                path_record(f"{backend}_tracks", "track_metadata", run_root / "track_manifest.tsv", full_hash=True, note="Full hash: selected tracks and strand/assay metadata"),
                path_record(f"{backend}_combined_features", "large_raw_feature_artifact", run_root / "features/combined_mutation_features.parquet", full_hash=False, note="Large raw parquet; E0 records path/size/mtime only and performs selected-column validation"),
            ]
        )
    for label, root in (("ag", ag_snv_root), ("bz", bz_snv_root)):
        for relative, role in (("model_metadata.json", "old_snv_metadata"), ("validation_summary.json", "old_snv_validation"), ("prepared/mutation_manifest.tsv", "old_snv_manifest"), ("inputs/readouts.tsv", "old_snv_readout")):
            records.append(path_record(f"old_{label}_{relative.replace('/', '_').replace('.', '_')}", role, root / relative, full_hash=True, note="Optional E5 bridge source only; no bridge calculated in E0"))
    return records


def schema() -> dict[str, object]:
    return {
        "version": "mdk-motif-ism-v1-e0",
        "files": {
            "source_inventory.tsv": {
                "primary_key": ["source_id"],
                "columns": {"source_id": "string", "role": "string", "path": "absolute path", "exists": "boolean", "size_bytes": "integer nullable", "mtime_utc": "ISO-8601 UTC nullable", "sha256": "hex nullable", "hash_scope": "enum(full_file,path_size_mtime_only)", "note": "string"},
            },
            "comparability_audit.tsv": {
                "primary_key": ["comparison_id"],
                "columns": {"comparison_id": "string", "left_source": "string", "right_source": "string", "mutation_identity": "status", "input_context": "status", "output_grid": "status", "readout": "status", "assessment": "string", "evidence": "string"},
            },
            "execution_manifest.json": {
                "primary_key": ["execution_id"],
                "contents": "command, workspace state, environment, coordinate contract, and independent Mdk backend checks",
            },
        },
        "null_convention": "TSV uses an empty field only for a true unavailable value; status fields carry an explicit reason rather than encoding it as false or zero.",
    }


def repository_state() -> dict[str, str]:
    """Read HEAD without invoking Git's worktree traversal.

    E0 preserves a dirty worktree and never needs Git's index.  This direct
    read keeps the audit usable when an unrelated nested worktree makes
    ``git status`` block on a network filesystem.
    """

    head_path = REPO_ROOT / ".git/HEAD"
    head = head_path.read_text().strip()
    if head.startswith("ref: "):
        reference = head.removeprefix("ref: ")
        commit = (REPO_ROOT / ".git" / reference).read_text().strip()
        branch = reference.removeprefix("refs/heads/")
    else:
        branch = "detached"
        commit = head
    return {
        "branch": branch,
        "commit": commit,
        "dirty_status": "not_collected",
        "dirty_status_scope": (
            "A normal git status/diff call blocked while traversing an unrelated "
            "nested worktree on the shared filesystem. E0 did not modify any "
            "pre-existing path; the user-visible preflight status remains the "
            "authoritative dirty-worktree record for this run."
        ),
    }


def main() -> None:
    args = parse_args()
    primary_root = args.primary_root.resolve()
    ag_snv_root = args.ag_snv_root.resolve()
    bz_snv_root = args.bz_snv_root.resolve()
    out_dir = args.out_dir.resolve()
    require(not out_dir.exists(), f"Output directory already exists; will not overwrite: {out_dir}")
    require(primary_root.is_dir(), f"Missing primary root: {primary_root}")
    for root in (ag_snv_root, bz_snv_root):
        require(root.is_dir(), f"Missing old SNV root: {root}")

    manifest = load_mdk_manifest(primary_root)
    readouts = mdk_readouts(primary_root)
    manifest_sha256 = sha256(primary_root / "prepared/mutation_manifest.tsv")
    backend_checks = [
        audit_backend(primary_root, backend, manifest, readouts, manifest_sha256)
        for backend in BACKENDS
    ]
    require(all(item["status"] == "ok" for item in backend_checks), "One or more independent backend checks failed")
    hsc_check = hsc_cage_audit(primary_root)
    require(hsc_check["strand_status"] == "ok", "Could not verify BZ-original HSC CAGE strand")
    sources = source_records(primary_root, ag_snv_root, bz_snv_root)
    require(all(record["exists"] for record in sources), "Required E0 source is missing")
    comparisons = comparability_rows(primary_root, ag_snv_root, bz_snv_root)

    out_dir.mkdir(parents=True)
    write_tsv(out_dir / "source_inventory.tsv", sources)
    write_tsv(out_dir / "comparability_audit.tsv", comparisons)
    write_json(out_dir / "schema.json", schema())
    execution_manifest = {
        "execution_id": "mdk-motif-ism-v1-e0",
        "stage": "E0_input_inventory_and_comparability_audit",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": [sys.executable, *sys.argv],
        "repository": {"root": str(REPO_ROOT), **repository_state()},
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "pyarrow": pyarrow.__version__,
        },
        "coordinate_contract": {
            "species_assembly": "Mus musculus / mm10",
            "gene_transcript": "Mdk / Mdk-201 / ENSMUST00000028672",
            "strand_reporting_orientation": "negative strand; transcription-direction offsets increase with transcription",
            "tss_anchor_0_based": "chr2:91932297",
            "window_genomic_half_open": "chr2:[91929297,91935297)",
            "authority": str((REPO_ROOT / "scripts/ism/experiments/saijou_hsc/configs/provided_nine_gene_transcripts.tsv").resolve()),
            "operation": "read-only audit of existing artifacts; no coordinate transformation, mutation generation, inference, or motif annotation",
        },
        "primary_source": str(primary_root),
        "old_snv_sources": {"alphagenome": str(ag_snv_root), "borzoi": str(bz_snv_root)},
        "mdk_manifest": {
            "sha256": manifest_sha256,
            "edits": len(manifest),
            "centers": int(manifest["edit_center_position"].nunique()),
            "readouts": readouts,
        },
        "backend_checks": backend_checks,
        "borzoi_original_hsc_cage_check": hsc_check,
        "scope_status": "ok_e0_only",
        "not_run": ["E1 candidate registration", "E2 standardization", "E3 scoring", "E4 motif evidence", "E5 SNV bridge", "E6 acceptance", "new model inference"],
    }
    write_json(out_dir / "execution_manifest.json", execution_manifest)
    print(f"E0 audit completed: {out_dir}")


if __name__ == "__main__":
    main()
