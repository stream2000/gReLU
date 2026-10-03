#!/usr/bin/env python
"""Validate completeness, profile stores, reconstruction, and storage budget."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from grelu.interpret.ism.profiles import (  # noqa: E402
    load_stored_mutation_profiles,
    reconstruct_alternate_profiles,
)
from grelu.interpret.ism.validation import directory_bytes  # noqa: E402


DEFAULT_ROOT = REPO_ROOT / "experiments/ism/saijou_all_genes_10bp_scan"
RUNS = [
    "alphagenome_finetuned",
    "borzoi_finetuned",
    "alphagenome_original",
    "borzoi_original",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--require-original", action="store_true")
    parser.add_argument("--max-run-bytes", type=int, default=3_000_000_000)
    parser.add_argument(
        "--reconstruction-rtol",
        type=float,
        default=2e-3,
        help=(
            "Maximum relative ALT-sum reconstruction error after float16 log2FC "
            "storage. The threshold is 0.2%%; actual maxima are reported."
        ),
    )
    return parser.parse_args()


@dataclass(frozen=True)
class GeneProfileAudit:
    track_order: list[str]
    profile_bytes: int
    nonfinite_values: int
    index_mismatch: int
    reconstruction_rows: int
    max_reconstruction_relative_error: float


def _validate_gene_profiles(
    *,
    run: Path,
    run_name: str,
    gene: str,
    gene_manifest: pd.DataFrame,
) -> GeneProfileAudit:
    stored = load_stored_mutation_profiles(
        run / "profiles",
        gene=gene,
        resolution_bp=128,
    )
    values = stored.values
    reference = stored.reference
    index = stored.index
    metadata = stored.metadata
    track_order = stored.track_order
    expected_bins = int(metadata["shape"][-1])
    if values.shape != (len(gene_manifest), len(track_order), expected_bins):
        raise RuntimeError(
            f"Unexpected profile shape for {run_name}/{gene}: {values.shape}"
        )
    if reference.shape != (len(track_order), expected_bins):
        raise RuntimeError(
            f"Unexpected reference shape for {run_name}/{gene}: {reference.shape}"
        )

    features = pd.read_csv(run / f"features/{gene}.tsv", sep="\t")
    tss = features.loc[features.readout_role.eq("tss_1024bp")].copy()
    if tss.empty:
        raise RuntimeError(f"No TSS readout rows for {run_name}/{gene}")
    output_start = int(tss.output_start.iloc[0])
    start = int(tss.effective_readout_start.iloc[0])
    end = int(tss.effective_readout_end.iloc[0])
    resolution_bp = int(metadata["stored_resolution_bp"])
    if (start - output_start) % resolution_bp or (end - output_start) % resolution_bp:
        raise RuntimeError(
            f"TSS readout is not {resolution_bp}-bp aligned for {run_name}/{gene}"
        )
    left = (start - output_start) // resolution_bp
    right = (end - output_start) // resolution_bp
    reconstructed = reconstruct_alternate_profiles(
        reference[None, :, left:right],
        values[:, :, left:right],
        pseudocount=float(metadata["pseudocount_per_stored_bin"]),
    ).sum(axis=-1)
    observed = (
        tss.pivot(index="mutation_id", columns="track_id", values="alt_sum")
        .loc[index.mutation_id, track_order]
        .to_numpy(dtype=float)
    )
    relative = np.abs(reconstructed - observed) / np.maximum(np.abs(observed), 1.0)
    return GeneProfileAudit(
        track_order=track_order,
        profile_bytes=int(stored.values_path.stat().st_size),
        nonfinite_values=int((~np.isfinite(values)).sum()),
        index_mismatch=int(index.mutation_id.tolist() != gene_manifest.mutation_id.tolist()),
        reconstruction_rows=int(relative.size),
        max_reconstruction_relative_error=float(relative.max()),
    )


def validate_run(
    root: Path,
    run_name: str,
    expected_genes: list[str],
    max_run_bytes: int,
    reconstruction_rtol: float,
) -> dict:
    run = root / "runs" / run_name
    validation_path = run / "validation_summary.json"
    if not validation_path.exists():
        return {"run": run_name, "status": "missing"}
    reported = json.loads(validation_path.read_text())
    if reported.get("status") != "ok":
        raise RuntimeError(f"Run validation is not ok: {validation_path}")
    prepared_dir = (
        root / "prepared_original_candidates"
        if run_name.endswith("original")
        else root / "prepared"
    )
    manifest = pd.read_csv(prepared_dir / "mutation_manifest.tsv", sep="\t")
    track_order = None
    profile_nonfinite = 0
    index_mismatches = 0
    max_reconstruction_relative_error = 0.0
    profile_bytes = 0
    checked_rows = 0
    for gene in expected_genes:
        gene_manifest = manifest.loc[manifest.gene.eq(gene)]
        if gene_manifest.empty:
            continue
        audit = _validate_gene_profiles(
            run=run,
            run_name=run_name,
            gene=gene,
            gene_manifest=gene_manifest,
        )
        if track_order is None:
            track_order = audit.track_order
        profile_nonfinite += audit.nonfinite_values
        profile_bytes += audit.profile_bytes
        index_mismatches += audit.index_mismatch
        max_reconstruction_relative_error = max(
            max_reconstruction_relative_error,
            audit.max_reconstruction_relative_error,
        )
        checked_rows += audit.reconstruction_rows

    run_bytes = directory_bytes(run)
    result = {
        "run": run_name,
        "status": "ok",
        "genes": reported.get("genes"),
        "tracks": len(track_order or []),
        "profile_bytes": profile_bytes,
        "run_bytes": run_bytes,
        "under_3gb": run_bytes < max_run_bytes,
        "profile_nonfinite_values": profile_nonfinite,
        "profile_index_mismatches": index_mismatches,
        "reconstruction_rows": checked_rows,
        "max_reconstruction_relative_error": max_reconstruction_relative_error,
    }
    if (
        not result["under_3gb"]
        or profile_nonfinite
        or index_mismatches
        or max_reconstruction_relative_error > reconstruction_rtol
    ):
        raise RuntimeError(result)
    return result


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    genes = pd.read_csv(root / "prepared/genes.tsv", sep="\t").gene.tolist()
    results = []
    for run_name in RUNS:
        result = validate_run(
            root,
            run_name,
            genes,
            args.max_run_bytes,
            args.reconstruction_rtol,
        )
        if args.require_original and run_name.endswith("original") and result["status"] == "missing":
            raise RuntimeError(f"Required original run is missing: {run_name}")
        results.append(result)
    complete = [row for row in results if row["status"] == "ok"]
    output = {
        "status": "ok",
        "expected_genes": genes,
        "validated_runs": [row["run"] for row in complete],
        "missing_runs": [row["run"] for row in results if row["status"] == "missing"],
        "runs": results,
    }
    out = root / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    (out / "pipeline_validation_summary.json").write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
