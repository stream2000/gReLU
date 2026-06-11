#!/usr/bin/env python
"""Plot MREG three-region chromatin ISM results.

Generates the five figures specified in the exploration plan:

1. mreg_locus_schematic.png — MREG gene model with TSS, HC-DIC, LC-DIC, mutations
2. mreg_three_region_ref_alt_delta.png — Per-track REF/ALT overlay + DELTA
3. mreg_dic_to_tss_effects.png — DIC mutation effects on distal TSS readout
4. mreg_response_matrices.png — 3×3 response matrix heatmaps per target
5. mreg_control_adjusted_summary.png — Control-adjusted effect summary

All plots faithfully render 128 bp bins as step plots (no pseudo-1bp interpolation).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns


# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------

plt.rcParams.update({
    "figure.dpi": 150,
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "legend.fontsize": 7,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
})

PRIMARY_COLORS = {
    "CTCF": "#1f77b4",
    "RAD21": "#ff7f0e",
    "POLR2A": "#2ca02c",
    "POLR2B": "#2ca02c",
    "SMC3": "#d62728",
    "H3K4me3": "#9467bd",
    "H3K27ac": "#8c564b",
    "H3K4me2": "#e377c2",
}
REF_COLOR = "#3182bd"
ALT_COLOR = "#de2d26"
DELTA_COLOR = "#636363"
REGION_COLORS = {"tss": "#e41a1c", "hc_dic": "#377eb8", "lc_dic": "#4daf4a"}


# ---------------------------------------------------------------------------
# Figure 1: Locus schematic
# ---------------------------------------------------------------------------


def plot_locus_schematic(
    region_manifest: pd.DataFrame,
    output_path: Path,
    gene_name: str = "MREG",
    chrom: str = "chr2",
):
    """Draw a simplified MREG locus schematic with regions and mutations."""
    fig, ax = plt.subplots(figsize=(12, 3))

    if region_manifest.empty:
        ax.text(0.5, 0.5, "No region data available", ha="center", va="center")
        fig.savefig(output_path, bbox_inches="tight", dpi=150)
        plt.close(fig)
        return

    # Determine coordinate range
    all_starts = region_manifest["start"].tolist()
    all_ends = region_manifest["end"].tolist()
    min_pos = min(all_starts) - 5000
    max_pos = max(all_ends) + 5000

    # Gene body line
    gene_start = min(all_starts)
    gene_end = max(all_ends)
    ax.plot([gene_start, gene_end], [0.5, 0.5], "k-", linewidth=3, label=f"{gene_name} gene body")

    # Draw each region
    y_positions = {"tss": 0.8, "hc_dic": 0.35, "lc_dic": 0.15}
    for _, row in region_manifest.iterrows():
        role = row["region_role"]
        y = y_positions.get(role, 0.5)
        color = REGION_COLORS.get(role, "#999999")
        rect = mpatches.Rectangle(
            (row["start"], y - 0.05), row["end"] - row["start"], 0.1,
            facecolor=color, edgecolor="black", linewidth=0.5, alpha=0.8,
            label=role.upper() if role not in [p.get_label() for p in ax.patches] else "",
        )
        ax.add_patch(rect)
        # Label
        summit = row.get("summit", (row["start"] + row["end"]) // 2)
        ax.annotate(
            role.upper().replace("_", "-"),
            (summit, y + 0.08),
            ha="center", va="bottom", fontsize=7, fontweight="bold",
        )

    # TSS marker
    tss_rows = region_manifest[region_manifest["region_role"] == "tss"]
    if not tss_rows.empty:
        tss_pos = int(tss_rows.iloc[0]["summit"])
        ax.axvline(tss_pos, color="red", linestyle="--", linewidth=1, alpha=0.7)
        ax.annotate("TSS", (tss_pos, 0.95), ha="center", fontsize=8, color="red")

    # Gene direction arrow
    strand = str(region_manifest.iloc[0].get("strand", "."))
    if strand == "-":
        ax.annotate("←", (gene_start + 2000, 0.55), fontsize=14, ha="center")
    elif strand == "+":
        ax.annotate("→", (gene_end - 2000, 0.55), fontsize=14, ha="center")

    ax.set_xlim(min_pos, max_pos)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel(f"Genomic position ({chrom})")
    ax.set_yticks([])
    ax.set_title(f"{gene_name} Locus — TSS, HC-DIC, LC-DIC")
    ax.legend(loc="upper right", fontsize=7, ncol=2)

    fig.savefig(output_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"[plot] Locus schematic → {output_path}")


# ---------------------------------------------------------------------------
# Figure 2: REF/ALT/DELTA overlay per track
# ---------------------------------------------------------------------------


def _plot_ref_alt_delta_panel(
    ax_ref_alt,
    ax_delta,
    genomic_starts: np.ndarray,
    ref_vals: np.ndarray,
    alt_vals: np.ndarray,
    delta_vals: np.ndarray,
    target_name: str,
    biosample: str,
    edit_center: int | None = None,
):
    """Draw one row of the REF/ALT/DELTA grid: ref+alt overlay (top), delta (bottom)."""
    # Top: REF + ALT overlay
    ax_ref_alt.step(genomic_starts, ref_vals, where="post", color=REF_COLOR, linewidth=1.2, label="REF")
    ax_ref_alt.step(genomic_starts, alt_vals, where="post", color=ALT_COLOR, linewidth=1.2, linestyle="--", label="ALT")
    if edit_center is not None:
        ax_ref_alt.axvline(edit_center, color="black", linestyle=":", linewidth=0.5, alpha=0.5)

    # Bottom: DELTA
    ax_delta.step(genomic_starts, delta_vals, where="post", color=DELTA_COLOR, linewidth=1.2)
    ax_delta.axhline(0, color="black", linewidth=0.5, linestyle="-")
    if edit_center is not None:
        ax_delta.axvline(edit_center, color="black", linestyle=":", linewidth=0.5, alpha=0.5)

    # Labels
    ax_ref_alt.set_ylabel(f"{target_name}\n{biosample}", fontsize=6)
    ax_delta.set_ylabel("Δ", fontsize=6)


def plot_three_region_ref_alt_delta(
    profiles: pd.DataFrame,
    mutation_manifest: pd.DataFrame,
    output_path: Path,
):
    """Figure 2: Grid of REF/ALT/DELTA profiles for all experimental mutations and primary tracks."""
    primary_targets = ["CTCF", "RAD21", "POLR2A", "H3K4me3"]
    exp_profiles = profiles[profiles["control_type"] == "experimental"]

    if exp_profiles.empty:
        # Fallback: use all profiles
        exp_profiles = profiles

    mutation_ids = sorted(exp_profiles["mutation_id"].unique())
    targets_present = sorted(set(exp_profiles["target_name"].unique()) & set(primary_targets))
    if not targets_present:
        targets_present = sorted(exp_profiles["target_name"].unique())[:4]

    n_rows = len(mutation_ids)
    n_cols = len(targets_present)

    fig, axes = plt.subplots(
        n_rows * 2, n_cols,
        figsize=(3.5 * n_cols, 1.8 * n_rows * 2),
        gridspec_kw={"height_ratios": [1, 0.6] * n_rows},
    )
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes], [axes]])

    for row_idx, mut_id in enumerate(mutation_ids):
        mut_profiles = exp_profiles[exp_profiles["mutation_id"] == mut_id]
        # Get edit center for this mutation
        edit_center = None
        if mutation_manifest is not None:
            mut_row = mutation_manifest[mutation_manifest["mutation_id"] == mut_id]
            if not mut_row.empty:
                edit_center = (int(mut_row.iloc[0]["edit_start"]) + int(mut_row.iloc[0]["edit_end"])) // 2

        for col_idx, target in enumerate(targets_present):
            track_profiles = mut_profiles[mut_profiles["target_name"] == target]
            # Plot exactly one window. Combining the nested 1 kb and 4 kb
            # readouts duplicates genomic x coordinates and draws invalid lines.
            local = track_profiles[
                track_profiles["readout_id"] == "mutation_center_1kb"
            ]
            if local.empty:
                local = track_profiles  # fallback to any readout

            local = local.sort_values("genomic_start")
            gs = local["genomic_start"].to_numpy(dtype=float)
            ref = local["ref_value"].to_numpy(dtype=float)
            alt = local["alt_value"].to_numpy(dtype=float)
            delta = local["delta"].to_numpy(dtype=float)

            ax_ref_alt = axes[row_idx * 2, col_idx]
            ax_delta = axes[row_idx * 2 + 1, col_idx]

            if len(gs) > 0:
                biosample = str(local.iloc[0].get("biosample", ""))
                _plot_ref_alt_delta_panel(ax_ref_alt, ax_delta, gs, ref, alt, delta, target, biosample, edit_center)
            else:
                ax_ref_alt.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax_ref_alt.transAxes)
                ax_delta.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax_delta.transAxes)

            if row_idx == 0:
                ax_ref_alt.set_title(target, fontsize=9)
            if row_idx == n_rows - 1:
                ax_delta.set_xlabel("Genomic position (bp)", fontsize=7)

    if n_rows > 1:
        row_labels = {
            "mreg_tss_00_ctcf_snv": "TSS\nCTCF edit",
            "mreg_hc_dic_00_ctcf_snv": "HC-DIC\nCTCF edit",
            "mreg_lc_dic_00_pol2_perturbation": "LC-DIC\nPol2 edit",
        }
        for row_idx, mut_id in enumerate(mutation_ids):
            short_id = row_labels.get(mut_id, mut_id.replace("mreg_", ""))
            axes[row_idx * 2, 0].annotate(
                short_id, (-0.28, 0.5), xycoords="axes fraction",
                ha="right", va="center", fontsize=7, fontweight="bold",
                rotation=0,
            )

    title = (
        "MREG TSS CTCF-Motif Positive Control: REF / ALT / DELTA"
        if n_rows == 1 and mutation_ids[0] == "mreg_tss_00_ctcf_snv"
        else "Three-Region Mutation Effects: REF / ALT / DELTA"
    )
    axes[0, 0].legend(fontsize=7, loc="upper right")
    fig.suptitle(title, fontsize=11, y=1.01)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"[plot] REF/ALT/DELTA grid → {output_path}")


# ---------------------------------------------------------------------------
# Figure 3: DIC mutation → distal TSS effects
# ---------------------------------------------------------------------------


def plot_dic_to_tss_effects(
    profiles: pd.DataFrame,
    mutation_manifest: pd.DataFrame,
    output_path: Path,
):
    """Figure 3: DIC mutation effects on TSS readout for Pol2, H3K4me3, CTCF, RAD21."""
    dic_profiles = profiles[
        profiles["mutation_id"].str.contains("dic", case=False)
        & (profiles["control_type"] == "experimental")
    ]
    tss_profiles = dic_profiles[
        dic_profiles["readout_id"] == "mreg_tss_4kb"
    ]

    if tss_profiles.empty:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, "No DIC→TSS data available", ha="center", va="center")
        fig.savefig(output_path, bbox_inches="tight", dpi=150)
        plt.close(fig)
        print(f"[plot] DIC→TSS effects (empty) → {output_path}")
        return

    targets = sorted(set(tss_profiles["target_name"].unique()) & {"POLR2A", "H3K4me3", "CTCF", "RAD21"})
    if not targets:
        targets = sorted(tss_profiles["target_name"].unique())[:4]

    mutation_ids = sorted(tss_profiles["mutation_id"].unique())
    n_cols = len(targets)
    n_rows = len(mutation_ids)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.5 * n_cols, 2.5 * n_rows))
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)

    for row_idx, mut_id in enumerate(mutation_ids):
        mut_data = tss_profiles[tss_profiles["mutation_id"] == mut_id]
        for col_idx, target in enumerate(targets):
            ax = axes[row_idx, col_idx]
            tdata = mut_data[mut_data["target_name"] == target].sort_values("genomic_start")
            if not tdata.empty:
                gs = tdata["genomic_start"].to_numpy(dtype=float)
                ref = tdata["ref_value"].to_numpy(dtype=float)
                alt = tdata["alt_value"].to_numpy(dtype=float)
                delta = tdata["delta"].to_numpy(dtype=float)
                ax.step(gs, ref, where="post", color=REF_COLOR, linewidth=1.2, label="REF")
                ax.step(gs, alt, where="post", color=ALT_COLOR, linewidth=1.2, linestyle="--", label="ALT")
                ax_twin = ax.twinx()
                ax_twin.step(gs, delta, where="post", color=DELTA_COLOR, linewidth=0.8, alpha=0.6)
                ax_twin.set_ylabel("Δ", fontsize=6)
            else:
                ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)

            if row_idx == 0:
                ax.set_title(target, fontsize=9)
            if col_idx == 0:
                short_id = mut_id.replace("mreg_", "")
                ax.set_ylabel(short_id, fontsize=7)

    fig.suptitle("DIC Mutation Effects on Distal TSS", fontsize=11, y=1.01)
    axes[0, 0].legend(fontsize=7, loc="upper left")
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"[plot] DIC→TSS effects → {output_path}")


# ---------------------------------------------------------------------------
# Figure 4: 3×3 Response matrices
# ---------------------------------------------------------------------------


def plot_response_matrices(
    matrix_df: pd.DataFrame,
    output_path: Path,
):
    """Figure 4: 3×3 response matrix heatmaps, one per target per metric."""
    if matrix_df.empty:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, "No response matrix data available", ha="center", va="center")
        fig.savefig(output_path, bbox_inches="tight", dpi=150)
        plt.close(fig)
        return

    targets = sorted(matrix_df["target_name"].dropna().unique())
    metrics = sorted(matrix_df["metric"].dropna().unique())
    if not metrics:
        metrics = ["adjusted_signed_delta_mean"]

    n_targets = len(targets)
    n_metrics = len(metrics)

    fig, axes = plt.subplots(
        n_metrics, n_targets,
        figsize=(3 * n_targets, 2.8 * n_metrics),
        squeeze=False,
    )

    region_order = ["tss", "hc_dic", "lc_dic"]

    for met_idx, metric in enumerate(metrics):
        vmin = matrix_df[matrix_df["metric"] == metric]["value"].min()
        vmax = matrix_df[matrix_df["metric"] == metric]["value"].max()
        vabs = max(abs(vmin), abs(vmax), 1e-6)

        for tgt_idx, target in enumerate(targets):
            ax = axes[met_idx, tgt_idx]
            sub = matrix_df[
                (matrix_df["target_name"] == target)
                & (matrix_df["metric"] == metric)
            ]
            if sub.empty:
                ax.text(0.5, 0.5, "N/A", ha="center", va="center", transform=ax.transAxes)
                continue

            # Build 3×3 matrix
            mat = np.full((3, 3), np.nan)
            for _, row in sub.iterrows():
                mr = str(row["mutation_region"])
                rr = str(row["readout_region"])
                if mr in region_order and rr in region_order:
                    mat[region_order.index(mr), region_order.index(rr)] = row["value"]

            im = ax.imshow(mat, cmap="RdBu_r", aspect="equal", vmin=-vabs, vmax=vabs)
            plt.colorbar(im, ax=ax, shrink=0.8)

            # Annotate cells
            for i in range(3):
                for j in range(3):
                    val = mat[i, j]
                    if not np.isnan(val):
                        ax.text(j, i, f"{val:.3f}", ha="center", va="center", fontsize=7)

            ax.set_xticks(range(3))
            ax.set_xticklabels(["TSS", "HC-DIC", "LC-DIC"], fontsize=6, rotation=30)
            ax.set_yticks(range(3))
            ax.set_yticklabels(["TSS", "HC-DIC", "LC-DIC"], fontsize=6)
            ax.set_xlabel("Readout region", fontsize=7)
            if tgt_idx == 0:
                ax.set_ylabel("Mutation region", fontsize=7)
            if met_idx == 0:
                ax.set_title(target, fontsize=9)

    fig.suptitle("3×3 Response Matrices", fontsize=11, y=1.01)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"[plot] Response matrices → {output_path}")


# ---------------------------------------------------------------------------
# Figure 5: Control-adjusted summary
# ---------------------------------------------------------------------------


def plot_control_adjusted_summary(
    adjusted: pd.DataFrame,
    output_path: Path,
):
    """Figure 5: Bar chart comparing target effect, control effect, and adjusted effect."""
    if adjusted.empty:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, "No control-adjusted data available", ha="center", va="center")
        fig.savefig(output_path, bbox_inches="tight", dpi=150)
        plt.close(fig)
        return

    # Keep mutation identities separate; averaging TSS, HC-DIC, and LC-DIC
    # mutations together removes the comparison the plot is meant to show.
    agg = adjusted.groupby(["mutation_id", "target_name", "readout_id"]).agg(
        target_delta=("signed_delta_mean", "mean"),
        control_delta=("control_signed_delta_mean", "mean"),
        adjusted_delta=("adjusted_signed_delta_mean", "mean"),
        target_log2fc=("log2fc_mean", "mean"),
        adjusted_log2fc=("adjusted_log2fc_mean", "mean"),
    ).reset_index()

    primary_targets = ["CTCF", "RAD21", "POLR2A", "H3K4me3", "SMC3"]
    agg = agg[agg["target_name"].isin(primary_targets)]

    if agg.empty:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, "No primary target data available", ha="center", va="center")
        fig.savefig(output_path, bbox_inches="tight", dpi=150)
        plt.close(fig)
        return

    local_readouts = {
        "mreg_tss_00_ctcf_snv": "mreg_tss_4kb",
        "mreg_hc_dic_00_ctcf_snv": "hc_dic_4kb",
        "mreg_lc_dic_00_pol2_perturbation": "lc_dic_4kb",
    }
    local_parts = []
    for mutation_id, readout_id in local_readouts.items():
        local_parts.append(agg[
            (agg["mutation_id"] == mutation_id)
            & (agg["readout_id"] == readout_id)
        ])
    local = pd.concat(local_parts, ignore_index=True)
    local["panel"] = "Local 4 kb readouts"
    distal_tss = agg[
        (agg["readout_id"] == "mreg_tss_4kb")
        & agg["mutation_id"].str.contains("_dic_", case=False)
    ].copy()
    distal_tss["panel"] = "DIC to TSS 4 kb"
    plot_data = pd.concat([local, distal_tss], ignore_index=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax_idx, label in enumerate(["Local 4 kb readouts", "DIC to TSS 4 kb"]):
        ax = axes[ax_idx]
        sub = plot_data[plot_data["panel"] == label]

        if sub.empty:
            ax.text(0.5, 0.5, "No data", ha="center", va="center")
            ax.set_title(label)
            continue

        x = np.arange(len(sub))
        width = 0.25

        ax.bar(x - width, sub["target_delta"], width, color=ALT_COLOR, alpha=0.7, label="Target Δ")
        ax.bar(x, sub["control_delta"], width, color="#999999", alpha=0.7, label="Control Δ")
        ax.bar(x + width, sub["adjusted_delta"], width, color=DELTA_COLOR, alpha=0.9, label="Adjusted Δ")

        ax.set_xticks(x)
        region_labels = {
            "mreg_tss_00_ctcf_snv": "TSS",
            "mreg_hc_dic_00_ctcf_snv": "HC-DIC",
            "mreg_lc_dic_00_pol2_perturbation": "LC-DIC",
        }
        labels = [
            f"{region_labels.get(r['mutation_id'], r['mutation_id'])}\n"
            f"{r['target_name']}"
            for _, r in sub.iterrows()
        ]
        ax.set_xticklabels(labels, fontsize=5, rotation=45, ha="right")
        ax.set_ylabel("Signed Δ mean")
        ax.set_title(label)
        ax.legend(fontsize=7)
        ax.axhline(0, color="black", linewidth=0.5)

    fig.suptitle("Control-Adjusted Effects Summary", fontsize=11)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"[plot] Control-adjusted summary → {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-dir", required=True, help="Directory with analysis output TSVs")
    parser.add_argument("--run-dir", default=None, help="Directory with profiles parquet (fallback)")
    parser.add_argument("--prepared-dir", default=None, help="Directory with prepared manifests")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    analysis_dir = Path(args.analysis_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    profiles = None
    if args.run_dir:
        profiles_path = Path(args.run_dir) / "mreg_chromatin_profiles.parquet"
        if profiles_path.exists():
            profiles = pd.read_parquet(profiles_path)

    validation_status = {
        "status": "unknown",
        "cross_region_valid": False,
        "included_mutation_ids": [],
    }
    validation_path = analysis_dir / "validation_status.json"
    if validation_path.exists():
        with validation_path.open() as handle:
            validation_status = json.load(handle)
    if profiles is not None and validation_status["included_mutation_ids"]:
        profiles = profiles[
            profiles["mutation_id"].isin(
                validation_status["included_mutation_ids"]
            )
        ].copy()

    metrics_path = analysis_dir / "mreg_chromatin_metrics.tsv"
    metrics = pd.read_csv(metrics_path, sep="\t") if metrics_path.exists() else pd.DataFrame()

    adjusted_path = analysis_dir / "mreg_control_adjusted_metrics.tsv"
    adjusted = pd.read_csv(adjusted_path, sep="\t") if adjusted_path.exists() else pd.DataFrame()

    matrix_path = analysis_dir / "mreg_response_matrix.tsv"
    matrix_df = pd.read_csv(matrix_path, sep="\t") if matrix_path.exists() else pd.DataFrame()

    region_manifest = pd.DataFrame()
    mutation_manifest = pd.DataFrame()
    if args.prepared_dir:
        rm_path = Path(args.prepared_dir) / "mreg_region_manifest.tsv"
        if rm_path.exists():
            region_manifest = pd.read_csv(rm_path, sep="\t")
        mm_path = Path(args.prepared_dir) / "mreg_mutation_manifest.tsv"
        if mm_path.exists():
            mutation_manifest = pd.read_csv(mm_path, sep="\t")

    # A legacy coordinate-frame run can only support its first TSS mutation.
    if not validation_status.get("cross_region_valid", False):
        if profiles is not None and not profiles.empty:
            plot_three_region_ref_alt_delta(
                profiles,
                mutation_manifest,
                output_dir / "mreg_tss_positive_control_ref_alt_delta.png",
            )
        print(
            "[plot] Skipping locus, DIC→TSS, response matrix, and control-adjusted "
            f"figures because validation status is {validation_status['status']}"
        )
        return

    # Figure 1: Locus schematic
    plot_locus_schematic(region_manifest, output_dir / "mreg_locus_schematic.png")

    # Figure 2: REF/ALT/DELTA profiles
    if profiles is not None and not profiles.empty:
        plot_three_region_ref_alt_delta(profiles, mutation_manifest, output_dir / "mreg_three_region_ref_alt_delta.png")
    else:
        print("[plot] Skipping Figure 2: no profile data available")

    # Figure 3: DIC → TSS effects
    if profiles is not None and not profiles.empty:
        plot_dic_to_tss_effects(profiles, mutation_manifest, output_dir / "mreg_dic_to_tss_effects.png")
    else:
        print("[plot] Skipping Figure 3: no profile data available")

    # Figure 4: Response matrices
    plot_response_matrices(matrix_df, output_dir / "mreg_response_matrices.png")

    # Figure 5: Control-adjusted summary
    plot_control_adjusted_summary(adjusted, output_dir / "mreg_control_adjusted_summary.png")

    print(f"\n[plot] All figures written to {output_dir}")


if __name__ == "__main__":
    main()
