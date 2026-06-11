#!/usr/bin/env python
"""Analyze MREG three-region chromatin profiles.

Computes per-track metrics (REF/ALT/DELTA/log2fc means, peaks, areas, position shifts),
control-adjusted effects, and the 3×3 response matrix. Outputs:

- mreg_chromatin_metrics.tsv
- mreg_control_adjusted_metrics.tsv
- mreg_response_matrix.tsv
- result_summary.md
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

RESPONSE_READOUT_ROLES = {
    "mreg_tss_4kb": "tss",
    "hc_dic_4kb": "hc_dic",
    "lc_dic_4kb": "lc_dic",
}


# ---------------------------------------------------------------------------
# Metrics computation
# ---------------------------------------------------------------------------


def _safe_log2fc(alt: float, ref: float, pseudocount: float = 1.0) -> float:
    denom = ref + pseudocount
    if denom <= 0:
        return 0.0
    return float(np.log2((alt + pseudocount) / denom))


def _peak_info(values: np.ndarray, genomic_starts: np.ndarray) -> dict:
    """Extract peak value and position from a 1-D profile."""
    if len(values) == 0:
        return {"peak_value": np.nan, "peak_position_bp": np.nan}
    idx = int(np.argmax(values))
    return {
        "peak_value": float(values[idx]),
        "peak_position_bp": float(genomic_starts[idx] + 64),  # bin center
    }


def compute_metrics(
    profiles: pd.DataFrame,
    pseudocount: float = 1.0,
) -> pd.DataFrame:
    """Compute per-(mutation, readout, track) metrics.

    Parameters
    ----------
    profiles : pd.DataFrame
        Long-form profiles from the runner (ref_value, alt_value, delta, log2fc columns).
    pseudocount : float
        Pseudocount for log2fc calculations.

    Returns
    -------
    pd.DataFrame with one row per (mutation_id, readout_id, track_id).
    """
    records = []

    for (mut_id, readout_id, track_id, target, biosample, control_type), group in profiles.groupby(
        ["mutation_id", "readout_id", "track_id", "target_name", "biosample", "control_type"],
        dropna=False,
    ):
        group = group.sort_values("genomic_start")
        ref_vals = group["ref_value"].to_numpy(dtype=np.float64)
        alt_vals = group["alt_value"].to_numpy(dtype=np.float64)
        delta_vals = group["delta"].to_numpy(dtype=np.float64)
        genomic_starts = group["genomic_start"].to_numpy(dtype=np.float64)

        # Basic statistics
        ref_mean = float(np.mean(ref_vals))
        alt_mean = float(np.mean(alt_vals))
        signed_delta_mean = float(np.mean(delta_vals))
        absolute_delta_mean = float(np.mean(np.abs(delta_vals)))
        delta_area = float(np.sum(delta_vals))

        # Peak info
        ref_peak = _peak_info(ref_vals, genomic_starts)
        alt_peak = _peak_info(alt_vals, genomic_starts)
        delta_peak = _peak_info(np.abs(delta_vals), genomic_starts)
        peak_shift = (
            alt_peak["peak_position_bp"] - ref_peak["peak_position_bp"]
            if not np.isnan(ref_peak["peak_position_bp"]) and not np.isnan(alt_peak["peak_position_bp"])
            else np.nan
        )

        # log2fc metrics
        log2fc_vals = np.array([_safe_log2fc(a, r, pseudocount) for a, r in zip(alt_vals, ref_vals)])
        log2fc_mean = float(np.mean(log2fc_vals))
        log2fc_peak_idx = int(np.argmax(np.abs(log2fc_vals)))
        log2fc_peak = float(log2fc_vals[log2fc_peak_idx])

        records.append({
            "mutation_id": mut_id,
            "readout_id": readout_id,
            "track_id": track_id,
            "target_name": target,
            "biosample": biosample,
            "control_type": control_type,
            "n_bins": len(group),
            "ref_mean": ref_mean,
            "alt_mean": alt_mean,
            "signed_delta_mean": signed_delta_mean,
            "absolute_delta_mean": absolute_delta_mean,
            "ref_peak": ref_peak["peak_value"],
            "alt_peak": alt_peak["peak_value"],
            "delta_peak": delta_peak["peak_value"],
            "ref_area": float(np.sum(ref_vals)),
            "alt_area": float(np.sum(alt_vals)),
            "delta_area": delta_area,
            "peak_position_ref_bp": ref_peak["peak_position_bp"],
            "peak_position_alt_bp": alt_peak["peak_position_bp"],
            "peak_position_shift_bp": peak_shift,
            "log2fc_mean": log2fc_mean,
            "log2fc_peak": log2fc_peak,
        })

    return pd.DataFrame.from_records(records)


# ---------------------------------------------------------------------------
# Control-adjusted effects
# ---------------------------------------------------------------------------


def compute_control_adjusted(
    metrics: pd.DataFrame,
    mutation_manifest: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Compute control-adjusted effects: target_mutation_effect - matched_control_effect.

    Matches each experimental mutation to its matched controls by region_id.
    """
    experimental = metrics[metrics["control_type"] == "experimental"].copy()
    controls = metrics[metrics["control_type"] == "matched_control"].copy()

    records = []
    effect_cols = [
        "signed_delta_mean", "absolute_delta_mean", "delta_area",
        "log2fc_mean", "log2fc_peak", "delta_peak",
    ]
    output_columns = [
        "mutation_id", "region_id", "readout_id", "target_name", "biosample",
        "n_controls",
    ]
    for col in effect_cols:
        output_columns.extend([col, f"control_{col}", f"adjusted_{col}"])

    if experimental.empty or controls.empty:
        return pd.DataFrame(columns=output_columns)

    for _, exp_row in experimental.iterrows():
        exp_id = str(exp_row["mutation_id"])
        readout_id = str(exp_row["readout_id"])
        target = str(exp_row["target_name"])
        biosample = str(exp_row["biosample"])

        # Find matching controls (same region_id prefix, same readout, same target)
        region_prefix = exp_id.rsplit("_", 1)[0]  # e.g., mreg_tss_00_ctcf_snv → mreg_tss_00_ctcf
        ctrl_subset = controls[
            (controls["mutation_id"].str.startswith(exp_id + "_ctrl"))
            & (controls["readout_id"] == readout_id)
            & (controls["target_name"] == target)
        ]
        if ctrl_subset.empty:
            continue

        # Average across control replicates
        ctrl_avg = ctrl_subset[effect_cols].mean()

        rec = {
            "mutation_id": exp_id,
            "region_id": region_prefix,
            "readout_id": readout_id,
            "target_name": target,
            "biosample": biosample,
            "n_controls": len(ctrl_subset),
        }
        for col in effect_cols:
            rec[f"{col}"] = float(exp_row[col])
            rec[f"control_{col}"] = float(ctrl_avg[col])
            rec[f"adjusted_{col}"] = float(exp_row[col] - ctrl_avg[col])

        records.append(rec)

    return pd.DataFrame.from_records(records, columns=output_columns)


# ---------------------------------------------------------------------------
# 3×3 Response matrix
# ---------------------------------------------------------------------------


def build_response_matrix(
    adjusted: pd.DataFrame,
    target_name: str,
    metric: str = "adjusted_signed_delta_mean",
) -> pd.DataFrame:
    """Build a 3×3 response matrix for a single target and metric.

    Rows: mutation region (tss, hc_dic, lc_dic)
    Columns: readout region (tss, hc_dic, lc_dic)
    """
    sub = adjusted[
        (adjusted["target_name"] == target_name)
        & (adjusted["readout_id"].isin(RESPONSE_READOUT_ROLES))
    ].copy()
    if sub.empty:
        return pd.DataFrame()

    def _region_role(mutation_id: str) -> str:
        mid = str(mutation_id).lower()
        if "hc_dic" in mid:
            return "hc_dic"
        if "lc_dic" in mid:
            return "lc_dic"
        return "tss"

    sub["mut_region"] = sub["mutation_id"].apply(_region_role)
    sub["ro_region"] = sub["readout_id"].map(RESPONSE_READOUT_ROLES)

    pivot = sub.pivot_table(
        values=metric,
        index="mut_region",
        columns="ro_region",
        aggfunc="mean",
    )

    # Ensure all three rows and columns exist
    for region in ["tss", "hc_dic", "lc_dic"]:
        if region not in pivot.index:
            pivot.loc[region] = np.nan
        if region not in pivot.columns:
            pivot[region] = np.nan

    pivot = pivot.reindex(index=["tss", "hc_dic", "lc_dic"], columns=["tss", "hc_dic", "lc_dic"])
    pivot.index.name = "mutation_region"
    pivot.columns.name = "readout_region"
    return pivot


def build_all_response_matrices(adjusted: pd.DataFrame) -> dict:
    """Build response matrices for all primary targets and key metrics."""
    primary_targets = ["CTCF", "RAD21", "POLR2A", "POLR2B", "SMC3", "H3K4me3"]
    metrics = [
        "adjusted_signed_delta_mean",
        "adjusted_absolute_delta_mean",
        "adjusted_log2fc_mean",
        "adjusted_delta_peak",
    ]
    matrices = {}
    for target in primary_targets:
        for metric in metrics:
            mat = build_response_matrix(adjusted, target, metric)
            if not mat.empty:
                matrices[f"{target}__{metric}"] = mat
    return matrices


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------


def generate_summary(
    metrics: pd.DataFrame,
    adjusted: pd.DataFrame,
    response_matrices: dict,
    run_config: dict | None = None,
    analysis_status: dict | None = None,
) -> str:
    """Generate a Markdown result summary."""
    lines = [
        "# MREG 128 bp ChIP ISM — Validated Result Summary",
        "",
        f"Generated: {pd.Timestamp.now().isoformat()}",
        "",
        "## Overview",
        "",
        f"- Mutations analyzed: {metrics['mutation_id'].nunique()}",
        f"- Readout regions: {metrics['readout_id'].nunique()}",
        f"- Tracks: {metrics['track_id'].nunique()}",
        f"- Targets: {sorted(metrics['target_name'].dropna().unique())}",
        "",
    ]

    if analysis_status:
        lines.extend([
            "## Validation Status",
            "",
            f"- Status: **{analysis_status['status']}**",
            f"- Cross-region analysis valid: **{analysis_status['cross_region_valid']}**",
            f"- Included mutations: {analysis_status['included_mutation_ids']}",
            f"- Excluded mutations: {analysis_status['excluded_mutation_ids']}",
            f"- Reason: {analysis_status['reason']}",
            "",
        ])

    if run_config:
        lines.extend([
            "## Run Configuration",
            "",
            "```json",
            json.dumps(run_config, indent=2, default=str),
            "```",
            "",
        ])

    # Use one explicit window; averaging nested windows obscures the magnitude.
    lines.append("## TSS Positive-Control Effect (mutation_center_1kb)")
    lines.append("")
    lines.append("| Target | REF mean | ALT mean | Δ mean | log2FC mean |")
    lines.append("|---|---:|---:|---:|---:|")

    center_metrics = metrics[
        (metrics["mutation_id"] == "mreg_tss_00_ctcf_snv")
        & (metrics["readout_id"] == "mutation_center_1kb")
    ]
    for target, grp in center_metrics.groupby("target_name", dropna=False):
        exp = grp[grp["control_type"] == "experimental"]
        if exp.empty:
            continue
        row = exp.iloc[0]
        lines.append(
            f"| {target} | {row['ref_mean']:.4f} | {row['alt_mean']:.4f} | "
            f"{row['signed_delta_mean']:.4f} | {row['log2fc_mean']:.4f} |"
        )

    # Local vs distant TSS response
    lines.append("")
    lines.append("## Distant TSS Response (DIC mutations → TSS readout)")
    lines.append("")
    if analysis_status and analysis_status.get("cross_region_valid") and not adjusted.empty:
        tss_readout = adjusted[
            adjusted["readout_id"] == "mreg_tss_4kb"
        ]
        dic_mutations = tss_readout[
            tss_readout["mutation_id"].str.contains("dic", case=False)
        ]
        if not dic_mutations.empty:
            lines.append("| Mutation | Target | Adjusted Δ Mean | Adjusted log2FC |")
            lines.append("|---|---:|---:|---:|")
            for _, row in dic_mutations.iterrows():
                lines.append(
                    f"| {row['mutation_id']} | {row['target_name']} | "
                    f"{row['adjusted_signed_delta_mean']:.4f} | "
                    f"{row['adjusted_log2fc_mean']:.4f} |"
                )
    else:
        lines.append(
            "Not evaluated: the current run is not coordinate-valid for DIC mutations."
        )

    # Response matrices
    lines.append("")
    lines.append("## 3×3 Response Matrices")
    lines.append("")
    if not response_matrices:
        lines.append(
            "Not generated: the current run is not valid for cross-region comparison."
        )
        lines.append("")
    for name, mat in response_matrices.items():
        lines.append(f"### {name}")
        lines.append("")
        lines.append(mat.to_markdown())
        lines.append("")

    return "\n".join(lines)


def assess_coordinate_frame(
    profiles: pd.DataFrame,
    run_dir: Path,
    run_config: dict | None,
) -> tuple[pd.DataFrame, dict]:
    """Restrict legacy outputs to mutations whose stored coordinate frame is valid."""
    coordinate_frame_version = int((run_config or {}).get("coordinate_frame_version", 1))
    all_mutation_ids = profiles["mutation_id"].drop_duplicates().astype(str).tolist()

    if coordinate_frame_version >= 2:
        shared_context = coordinate_frame_version >= 3
        return profiles, {
            "status": "ready",
            "cross_region_valid": True,
            "included_mutation_ids": all_mutation_ids,
            "excluded_mutation_ids": [],
            "reason": (
                "Profiles use coordinate-valid edit-specific frames; experimental "
                "and control edits also share region-level bin boundaries."
                if shared_context
                else "Profiles were extracted with a per-mutation coordinate frame."
            ),
        }

    intervals_path = run_dir / "scored_intervals.tsv"
    if not intervals_path.exists():
        raise ValueError(
            "Legacy run lacks scored_intervals.tsv; no profile coordinates can be validated"
        )
    intervals = pd.read_csv(intervals_path, sep="\t")
    if intervals.empty:
        raise ValueError("Legacy run has no scored intervals")

    first_mutation_id = str(intervals.iloc[0]["mutation_id"])
    included = [first_mutation_id]
    excluded = [mid for mid in all_mutation_ids if mid not in included]
    valid_profiles = profiles[profiles["mutation_id"].isin(included)].copy()
    return valid_profiles, {
        "status": "partial_tss_positive_control_only",
        "cross_region_valid": False,
        "included_mutation_ids": included,
        "excluded_mutation_ids": excluded,
        "reason": (
            "This legacy run extracted every mutation with the first mutation's "
            "coordinate frame. Only the first TSS mutation is coordinate-valid; "
            "controls, DIC readouts, distal TSS effects, and response matrices are excluded."
        ),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, help="Directory containing mreg_chromatin_profiles.parquet")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--pseudocount", type=float, default=1.0)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    profiles_path = run_dir / "mreg_chromatin_profiles.parquet"
    if not profiles_path.exists():
        raise SystemExit(f"Profiles not found: {profiles_path}")

    profiles = pd.read_parquet(profiles_path)
    print(f"[analyze] Loaded {len(profiles)} profile rows")

    # Optional inputs
    mutation_manifest = None
    manifest_path = run_dir.parent / "prepared" / "mreg_mutation_manifest.tsv"
    if manifest_path.exists():
        mutation_manifest = pd.read_csv(manifest_path, sep="\t")

    run_config = None
    config_path = run_dir / "run_config.json"
    if config_path.exists():
        with config_path.open() as f:
            run_config = json.load(f)

    profiles, analysis_status = assess_coordinate_frame(profiles, run_dir, run_config)
    with (output_dir / "validation_status.json").open("w") as handle:
        json.dump(analysis_status, handle, indent=2)
    print(
        f"[analyze] Validation status: {analysis_status['status']}; "
        f"included {analysis_status['included_mutation_ids']}"
    )

    # Compute metrics only from coordinate-valid profiles.
    metrics = compute_metrics(profiles, pseudocount=args.pseudocount)
    metrics_path = output_dir / "mreg_chromatin_metrics.tsv"
    metrics.to_csv(metrics_path, sep="\t", index=False)
    print(f"[analyze] Wrote {len(metrics)} metric rows to {metrics_path}")

    # Control-adjusted effects
    adjusted = compute_control_adjusted(metrics, mutation_manifest)
    adjusted_path = output_dir / "mreg_control_adjusted_metrics.tsv"
    adjusted.to_csv(adjusted_path, sep="\t", index=False)
    print(f"[analyze] Wrote {len(adjusted)} control-adjusted rows to {adjusted_path}")

    # Response matrices
    matrices = (
        build_all_response_matrices(adjusted)
        if analysis_status["cross_region_valid"]
        else {}
    )
    matrix_records = []
    for name, mat in matrices.items():
        target, metric = name.split("__", 1)
        for mut_region, row in mat.iterrows():
            for ro_region, value in row.items():
                matrix_records.append({
                    "target_name": target,
                    "metric": metric,
                    "mutation_region": mut_region,
                    "readout_region": ro_region,
                    "value": float(value) if not np.isnan(value) else np.nan,
                })
    matrix_df = pd.DataFrame.from_records(
        matrix_records,
        columns=[
            "target_name", "metric", "mutation_region",
            "readout_region", "value",
        ],
    )
    matrix_path = output_dir / "mreg_response_matrix.tsv"
    matrix_df.to_csv(matrix_path, sep="\t", index=False)
    print(f"[analyze] Wrote {len(matrix_df)} response matrix entries to {matrix_path}")

    # Summary
    summary = generate_summary(
        metrics,
        adjusted,
        matrices,
        run_config,
        analysis_status,
    )
    summary_path = output_dir / "result_summary.md"
    with summary_path.open("w") as f:
        f.write(summary)
    print(f"[analyze] Wrote summary to {summary_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
