#!/usr/bin/env python
"""Compare focused Mdk-control ISM across original and fine-tuned models."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr


REPO_ROOT = Path(__file__).resolve().parents[6]
DEFAULT_ROOT = REPO_ROOT / "experiments/ism/saijou_targeted_original_comparison"
BACKENDS = [
    "alphagenome_original",
    "borzoi_original",
    "alphagenome_finetuned",
    "borzoi_finetuned",
]


SERIES = [
    ("alphagenome_finetuned", "hsc_finetuned_10x", "gene_body", "AG-FT HSC RNA"),
    ("borzoi_finetuned", "hsc_finetuned_10x", "gene_body", "BZ-FT HSC RNA"),
    ("borzoi_original", "hsc_cage", "tss", "BZ-original HSC CAGE"),
    ("alphagenome_original", "liver_rna", "gene_body", "AG-original liver RNA"),
    ("borzoi_original", "liver_rna", "gene_body", "BZ-original liver RNA"),
    ("alphagenome_original", "fibroblast_rna", "gene_body", "AG-original fibroblast RNA"),
    ("borzoi_original", "fibroblast_rna", "gene_body", "BZ-original fibroblast RNA"),
    (
        "alphagenome_original",
        "liver_active_chromatin",
        "local_edit",
        "AG-original liver active chromatin",
    ),
    (
        "borzoi_original",
        "liver_active_chromatin",
        "local_edit",
        "BZ-original liver active chromatin",
    ),
]


LOCUS_LABELS = {
    "mdk_intron_hub_441_507": "Mdk intron hub +441..+507",
    "mdk_splice_donor_369": "Mdk splice donor +369",
    "acta2_202_promoter_carg": "Acta2 promoter CArG",
    "acta2_202_intron1_carg": "Acta2 intron-1 CArG",
    "col1a1_promoter_inverted_ccaat": "Col1a1 CCAAT",
    "timp1_promoter_ets_like_43": "Timp1 ETS-like +43",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    return parser.parse_args()


def read_features(root: Path) -> pd.DataFrame:
    tables = []
    for backend in BACKENDS:
        path = root / "runs" / backend / "features/combined_mutation_features.tsv"
        table = pd.read_csv(path, sep="\t")
        if set(table["model_backend"]) != {backend}:
            raise ValueError(f"Backend mismatch in {path}")
        tables.append(table)
    data = pd.concat(tables, ignore_index=True)
    genes = pd.read_csv(root / "prepared/genes.tsv", sep="\t")
    strand = genes.set_index("gene")["strand"].to_dict()
    data["gene_strand"] = data["gene"].map(strand)
    data["track_strand"] = data["track_strand"].fillna(".").astype(str)
    data = data.loc[
        data["track_strand"].eq(".") | data["track_strand"].eq(data["gene_strand"])
    ].copy()
    data = data.loc[
        data["readout_role"].ne("local_edit")
        | data["readout_locus_id"].eq(data["locus_id"])
    ].copy()
    return data


def group_mutation_effects(data: pd.DataFrame) -> pd.DataFrame:
    keys = [
        "model_backend",
        "model_id",
        "gene",
        "locus_id",
        "locus_role",
        "mutation_id",
        "variant_offset_from_tss_transcription_bp",
        "track_group",
        "readout_role",
    ]
    return (
        data.groupby(keys, sort=False)
        .agg(
            track_count=("track_id", "nunique"),
            effect=("log2fc_ratio_of_sums", "median"),
            mean_bin_log2fc=("log2fc_mean", "median"),
            delta_sum=("signed_delta_sum", "median"),
            ref_sum=("ref_sum", "median"),
            negative_tracks=("log2fc_ratio_of_sums", lambda x: float((x < 0).mean())),
        )
        .reset_index()
    )


def locus_summary(grouped: pd.DataFrame) -> pd.DataFrame:
    keys = [
        "model_backend",
        "model_id",
        "gene",
        "locus_id",
        "locus_role",
        "track_group",
        "readout_role",
    ]
    out = (
        grouped.groupby(keys, sort=False)
        .agg(
            mutations=("mutation_id", "nunique"),
            median_tracks=("track_count", "median"),
            mean_effect=("effect", "mean"),
            median_effect=("effect", "median"),
            mean_abs_effect=("effect", lambda x: float(np.mean(np.abs(x)))),
            median_abs_effect=("effect", lambda x: float(np.median(np.abs(x)))),
            max_abs_effect=("effect", lambda x: float(np.max(np.abs(x)))),
            fraction_negative=("effect", lambda x: float((x < 0).mean())),
            p10_effect=("effect", lambda x: float(np.quantile(x, 0.10))),
            p90_effect=("effect", lambda x: float(np.quantile(x, 0.90))),
            median_ref_sum=("ref_sum", "median"),
        )
        .reset_index()
    )
    out["effect_rank_within_series"] = out.groupby(
        ["model_backend", "track_group", "readout_role"]
    )["median_abs_effect"].rank(method="min", ascending=False)
    out["effect_percentile_within_series"] = out.groupby(
        ["model_backend", "track_group", "readout_role"]
    )["median_abs_effect"].rank(method="average", pct=True)
    return out


def selected_series(table: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for backend, group, readout, label in SERIES:
        subset = table.loc[
            table["model_backend"].eq(backend)
            & table["track_group"].eq(group)
            & table["readout_role"].eq(readout)
        ].copy()
        subset["series_label"] = label
        rows.append(subset)
    return pd.concat(rows, ignore_index=True)


def mdk_position_profiles(grouped: pd.DataFrame) -> pd.DataFrame:
    hub = grouped.loc[grouped["locus_id"].eq("mdk_intron_hub_441_507")].copy()
    profiles = (
        hub.groupby(
            [
                "model_backend",
                "track_group",
                "readout_role",
                "variant_offset_from_tss_transcription_bp",
            ],
            sort=False,
        )["effect"]
        .agg(effect="median", replicate_mean="mean", replicate_sd="std", replicates="size")
        .reset_index()
    )
    profiles["abs_effect_percentile"] = profiles.groupby(
        ["model_backend", "track_group", "readout_role"]
    )["effect"].transform(lambda x: x.abs().rank(method="average", pct=True))
    profiles["signed_effect_percentile"] = np.sign(profiles["effect"]) * profiles[
        "abs_effect_percentile"
    ]
    return profiles


def profile_concordance(profiles: pd.DataFrame) -> pd.DataFrame:
    series_rows = []
    for backend, group, readout, label in SERIES:
        q = profiles.loc[
            profiles["model_backend"].eq(backend)
            & profiles["track_group"].eq(group)
            & profiles["readout_role"].eq(readout),
            ["variant_offset_from_tss_transcription_bp", "effect"],
        ].copy()
        if q.empty:
            continue
        q = q.rename(columns={"effect": label}).set_index(
            "variant_offset_from_tss_transcription_bp"
        )
        series_rows.append(q)
    wide = pd.concat(series_rows, axis=1, join="inner").sort_index()
    rows = []
    for i, left in enumerate(wide.columns):
        for right in wide.columns[i + 1 :]:
            l = wide[left]
            r = wide[right]
            rows.append(
                {
                    "left": left,
                    "right": right,
                    "positions": len(wide),
                    "spearman_signed": float(spearmanr(l, r).statistic),
                    "spearman_absolute": float(spearmanr(l.abs(), r.abs()).statistic),
                    "sign_concordance": float((np.sign(l) == np.sign(r)).mean()),
                    "left_best_offset": int(l.abs().idxmax()),
                    "right_best_offset": int(r.abs().idxmax()),
                    "best_offset_distance": abs(int(l.abs().idxmax()) - int(r.abs().idxmax())),
                }
            )
    return pd.DataFrame.from_records(rows)


def finetuned_specificity(summary: pd.DataFrame) -> pd.DataFrame:
    fine = summary.loc[
        summary["model_backend"].isin(["alphagenome_finetuned", "borzoi_finetuned"])
        & summary["readout_role"].eq("gene_body")
    ].copy()
    fine["cell_type"] = fine["track_group"].str.replace("_finetuned_10x", "", regex=False)
    fine["cell_rank"] = fine.groupby(["model_backend", "locus_id"])[
        "median_abs_effect"
    ].rank(method="min", ascending=False)
    rows = []
    for (backend, locus), group in fine.groupby(["model_backend", "locus_id"]):
        hsc = group.loc[group["cell_type"].eq("hsc")]
        other = group.loc[~group["cell_type"].eq("hsc")]
        if hsc.empty:
            continue
        hsc_row = hsc.iloc[0]
        rows.append(
            {
                "model_backend": backend,
                "locus_id": locus,
                "hsc_median_abs_effect": float(hsc_row["median_abs_effect"]),
                "hsc_median_effect": float(hsc_row["median_effect"]),
                "hsc_fraction_negative": float(hsc_row["fraction_negative"]),
                "hsc_cell_rank": int(hsc_row["cell_rank"]),
                "max_other_abs_effect": float(other["median_abs_effect"].max()),
                "median_other_abs_effect": float(other["median_abs_effect"].median()),
                "hsc_vs_max_other_ratio": float(
                    hsc_row["median_abs_effect"] / max(float(other["median_abs_effect"].max()), 1e-12)
                ),
                "top_cell_type": str(group.nlargest(1, "median_abs_effect")["cell_type"].iloc[0]),
            }
        )
    return pd.DataFrame.from_records(rows)


def write_position_figure(profiles: pd.DataFrame, path: Path) -> None:
    expression_series = SERIES[:7]
    chromatin_series = [
        ("alphagenome_original", "liver_active_chromatin", "local_edit", "AG liver active chromatin"),
        ("borzoi_original", "liver_active_chromatin", "local_edit", "BZ liver active chromatin"),
        ("alphagenome_original", "fibroblast_active_chromatin", "local_edit", "AG fibroblast active chromatin"),
        ("borzoi_original", "fibroblast_active_chromatin", "local_edit", "BZ fibroblast active chromatin"),
        ("alphagenome_original", "liver_accessibility", "local_edit", "AG liver accessibility"),
        ("borzoi_original", "liver_accessibility", "local_edit", "BZ liver accessibility"),
    ]
    colors = ["#0169cc", "#e25507", "#5b3f95", "#3d8b68", "#b07c16", "#3f6f9f", "#9c4f65"]
    fig, axes = plt.subplots(2, 1, figsize=(10.5, 7.5), dpi=180, sharex=True)
    for ax, selected, title in [
        (axes[0], expression_series, "Expression / CAGE reference tracks"),
        (axes[1], chromatin_series, "Local chromatin and accessibility tracks"),
    ]:
        for index, (backend, group, readout, label) in enumerate(selected):
            q = profiles.loc[
                profiles["model_backend"].eq(backend)
                & profiles["track_group"].eq(group)
                & profiles["readout_role"].eq(readout)
            ].sort_values("variant_offset_from_tss_transcription_bp")
            if q.empty:
                continue
            ax.plot(
                q["variant_offset_from_tss_transcription_bp"],
                q["signed_effect_percentile"],
                marker="o",
                ms=2.5,
                lw=1.2,
                color=colors[index % len(colors)],
                label=label,
            )
        ax.axhline(0, color="#888888", lw=0.7)
        ax.set_ylim(-1.05, 1.05)
        ax.set_ylabel("signed within-series |effect| percentile")
        ax.set_title(title, loc="left", fontsize=11)
        ax.grid(axis="y", color="#e2e2e2", lw=0.6)
        ax.legend(frameon=False, fontsize=7, ncol=2)
    axes[-1].set_xlabel("Mdk transcriptional offset from TSS (bp)")
    fig.suptitle("Mdk +441..+507 targeted 10 bp strict-shuffle ISM", y=0.995)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def write_heatmap(summary: pd.DataFrame, path: Path) -> None:
    selected = selected_series(summary)
    pivot = selected.pivot(index="locus_id", columns="series_label", values="median_abs_effect")
    pivot = pivot.reindex(list(LOCUS_LABELS))
    signed = selected.pivot(index="locus_id", columns="series_label", values="median_effect").reindex(pivot.index)
    ranked = pivot.rank(axis=0, method="average", pct=True) * np.sign(signed)
    fig, ax = plt.subplots(figsize=(12.5, 5.2), dpi=180)
    image = ax.imshow(ranked.to_numpy(dtype=float), cmap="coolwarm", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(np.arange(len(ranked.columns)), ranked.columns, rotation=35, ha="right", fontsize=8)
    ax.set_yticks(np.arange(len(ranked.index)), [LOCUS_LABELS.get(x, x) for x in ranked.index], fontsize=8)
    for i in range(len(ranked.index)):
        for j in range(len(ranked.columns)):
            value = pivot.iloc[i, j]
            if np.isfinite(value):
                ax.text(j, i, f"{value:.3g}", ha="center", va="center", fontsize=6)
    ax.set_title("Targeted locus effect scores (annotation: median |ratio-of-sums log2FC|)")
    fig.colorbar(image, ax=ax, label="signed within-series locus percentile")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def write_finetuned_celltype_figure(summary: pd.DataFrame, path: Path) -> None:
    fine = summary.loc[
        summary["model_backend"].isin(["alphagenome_finetuned", "borzoi_finetuned"])
        & summary["readout_role"].eq("gene_body")
    ].copy()
    fine["cell_type"] = fine["track_group"].str.replace("_finetuned_10x", "", regex=False)
    loci = list(LOCUS_LABELS)
    cells = ["hsc", "mac", "lsec", "chol"]
    fig, axes = plt.subplots(2, 1, figsize=(11.5, 7.5), dpi=180, sharex=True)
    colors = {"hsc": "#0169cc", "mac": "#e25507", "lsec": "#808000", "chol": "#8046d9"}
    width = 0.19
    x = np.arange(len(loci))
    for ax, backend, title in [
        (axes[0], "alphagenome_finetuned", "Fine-tuned AlphaGenome"),
        (axes[1], "borzoi_finetuned", "Fine-tuned Borzoi"),
    ]:
        for idx, cell in enumerate(cells):
            values = []
            for locus in loci:
                q = fine.loc[
                    fine["model_backend"].eq(backend)
                    & fine["locus_id"].eq(locus)
                    & fine["cell_type"].eq(cell),
                    "median_abs_effect",
                ]
                values.append(float(q.iloc[0]) if len(q) else np.nan)
            ax.bar(x + (idx - 1.5) * width, values, width=width, label=cell, color=colors[cell])
        ax.set_ylabel("median |gene-body ratio-of-sums log2FC|")
        ax.set_title(title, loc="left")
        ax.legend(frameon=False, ncol=4)
        ax.grid(axis="y", color="#e5e5e5", lw=0.6)
    axes[-1].set_xticks(x, [LOCUS_LABELS[x] for x in loci], rotation=25, ha="right", fontsize=8)
    fig.suptitle("Cell-type specificity learned by Saijou fine-tuning", y=0.995)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def write_mdk_hub_donor_figure(summary: pd.DataFrame, path: Path) -> None:
    selected = selected_series(summary)
    selected = selected.loc[
        selected["locus_id"].isin(["mdk_intron_hub_441_507", "mdk_splice_donor_369"])
    ].copy()
    series = list(dict.fromkeys(selected["series_label"]))
    fig, ax = plt.subplots(figsize=(10.5, 5.5), dpi=180)
    x = np.arange(len(series))
    width = 0.36
    for idx, locus in enumerate(["mdk_intron_hub_441_507", "mdk_splice_donor_369"]):
        values = []
        for label in series:
            q = selected.loc[
                selected["locus_id"].eq(locus) & selected["series_label"].eq(label),
                "median_abs_effect",
            ]
            values.append(float(q.iloc[0]) if len(q) else np.nan)
        ax.bar(
            x + (idx - 0.5) * width,
            values,
            width=width,
            label=LOCUS_LABELS[locus],
            color=["#0169cc", "#e25507"][idx],
        )
    ax.set_xticks(x, series, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("median |ratio-of-sums log2FC|")
    ax.set_title("Mdk intronic hub versus first splice donor")
    ax.legend(frameon=False)
    ax.grid(axis="y", color="#e5e5e5", lw=0.6)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    analysis_dir = args.root / "analysis"
    figure_dir = args.root / "figures"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    data = read_features(args.root)
    grouped = group_mutation_effects(data)
    summary = locus_summary(grouped)
    profiles = mdk_position_profiles(grouped)
    concordance = profile_concordance(profiles)
    specificity = finetuned_specificity(summary)
    key = selected_series(summary)
    grouped.to_csv(analysis_dir / "group_mutation_effects.tsv", sep="\t", index=False)
    summary.to_csv(analysis_dir / "locus_summary.tsv", sep="\t", index=False)
    profiles.to_csv(analysis_dir / "mdk_hub_position_profiles.tsv", sep="\t", index=False)
    concordance.to_csv(analysis_dir / "mdk_hub_model_concordance.tsv", sep="\t", index=False)
    specificity.to_csv(analysis_dir / "finetuned_celltype_specificity.tsv", sep="\t", index=False)
    key.to_csv(analysis_dir / "key_comparison_table.tsv", sep="\t", index=False)
    write_position_figure(profiles, figure_dir / "mdk_hub_position_profiles.png")
    write_heatmap(summary, figure_dir / "targeted_locus_effect_heatmap.png")
    write_finetuned_celltype_figure(summary, figure_dir / "finetuned_celltype_locus_effects.png")
    write_mdk_hub_donor_figure(summary, figure_dir / "mdk_hub_vs_splice_donor.png")
    validation = {
        "status": "ok",
        "input_feature_rows": int(len(data)),
        "group_mutation_rows": int(len(grouped)),
        "locus_summary_rows": int(len(summary)),
        "mdk_position_rows": int(len(profiles)),
        "concordance_rows": int(len(concordance)),
        "specificity_rows": int(len(specificity)),
        "nonfinite_group_effects": int((~np.isfinite(grouped["effect"])).sum()),
        "figures": sorted(path.name for path in figure_dir.glob("*.png")),
    }
    (analysis_dir / "validation_summary.json").write_text(json.dumps(validation, indent=2) + "\n")
    print(json.dumps(validation, indent=2))


if __name__ == "__main__":
    main()
