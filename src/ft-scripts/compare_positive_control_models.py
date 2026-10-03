#!/usr/bin/env python
"""Compare positive-control metrics across Borzoi and AlphaGenome runs."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "experiments/validation/ag_active_aligned128_epoch00_vs_borzoi_epoch39"
DEFAULT_MODELS = [
    (
        "borzoi_e39",
        "Borzoi e39",
        ROOT
        / "experiments/validation/borzoi_lora_poisson_multinomial_epoch39_controls_raw_cpm_right_shared_y/per_track_metrics.csv",
    ),
    (
        "ag_old_e13",
        "AG old e13",
        ROOT
        / "experiments/validation/alphagenome_lora_poisson_multinomial_epoch13_controls_raw_cpm_right_shared_y/per_track_metrics.csv",
    ),
    (
        "ag_old_e15",
        "AG old e15",
        ROOT
        / "experiments/validation/alphagenome_lora_poisson_multinomial_epoch15_controls_raw_cpm_right_shared_y/per_track_metrics.csv",
    ),
    (
        "ag_active_aligned128_e00",
        "AG active 524k e00",
        ROOT
        / "experiments/validation/alphagenome_lora_active_aligned128_epoch00_controls_raw_cpm_right_shared_y/per_track_metrics.csv",
    ),
]

GENE_ORDER = ["Cd68", "Col1a1", "Epcam", "Fabp4", "Mmp2", "Myc", "Nipbl", "Pcdh17", "Trem2"]
TRACK_ORDER = ["hsc", "mac", "lsec", "chol"]

FONT_FAMILY = ["Aptos", "Inter", "Segoe UI", "DejaVu Sans", "Arial", "sans-serif"]
MONO_FONT_FAMILY = ["SF Mono", "Menlo", "Consolas", "DejaVu Sans Mono", "monospace"]
TOKENS = {
    "surface": "#FCFCFD",
    "panel": "#FFFFFF",
    "ink": "#1F2430",
    "muted": "#6F768A",
    "grid": "#E6E8F0",
    "axis": "#D7DBE7",
}
COLORS = {
    "blue": {"xlight": "#EAF1FE", "light": "#CEDFFE", "base": "#A3BEFA", "mid": "#5477C4", "dark": "#2E4780"},
    "gold": {"xlight": "#FFF4C2", "light": "#FFEA8F", "base": "#FFE15B", "mid": "#B8A037", "dark": "#736422"},
    "orange": {"xlight": "#FFEDDE", "light": "#FFBDA1", "base": "#F0986E", "mid": "#CC6F47", "dark": "#804126"},
    "olive": {"xlight": "#D8ECBD", "light": "#BEEB96", "base": "#A3D576", "mid": "#71B436", "dark": "#386411"},
    "pink": {"xlight": "#FCDAD6", "light": "#F5BACC", "base": "#F390CA", "mid": "#BD569B", "dark": "#8A3A6F"},
}
MODEL_COLORS = {
    "Borzoi e39": COLORS["blue"]["mid"],
    "AG old e13": COLORS["gold"]["mid"],
    "AG old e15": COLORS["olive"]["mid"],
    "AG active 524k e00": COLORS["orange"]["mid"],
}
TRACK_COLORS = {
    "hsc": COLORS["blue"]["base"],
    "mac": COLORS["orange"]["base"],
    "lsec": COLORS["olive"]["base"],
    "chol": COLORS["pink"]["base"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument(
        "--model",
        nargs=3,
        action="append",
        metavar=("KEY", "LABEL", "PER_TRACK_METRICS_CSV"),
        help=(
            "Model metrics to compare. May be repeated. Defaults to the "
            "historical Borzoi/AlphaGenome comparison set when omitted."
        ),
    )
    return parser.parse_args()


def model_specs_from_args(args: argparse.Namespace) -> list[tuple[str, str, Path]]:
    if not args.model:
        return DEFAULT_MODELS
    return [(key, label, Path(path)) for key, label, path in args.model]


def palette_for_models(model_specs: list[tuple[str, str, Path]]) -> dict[str, str]:
    colors = [
        COLORS["blue"]["mid"],
        COLORS["orange"]["mid"],
        COLORS["olive"]["mid"],
        COLORS["pink"]["mid"],
        COLORS["gold"]["mid"],
    ]
    return {label: colors[idx % len(colors)] for idx, (_key, label, _path) in enumerate(model_specs)}


def use_theme() -> None:
    sns.set_theme(
        style="whitegrid",
        rc={
            "figure.facecolor": TOKENS["surface"],
            "savefig.facecolor": TOKENS["surface"],
            "axes.facecolor": TOKENS["panel"],
            "axes.edgecolor": TOKENS["axis"],
            "axes.labelcolor": TOKENS["ink"],
            "axes.labelsize": 10,
            "xtick.color": TOKENS["muted"],
            "ytick.color": TOKENS["muted"],
            "grid.color": TOKENS["grid"],
            "grid.linewidth": 0.8,
            "font.family": "sans-serif",
            "font.sans-serif": FONT_FAMILY,
            "font.monospace": MONO_FONT_FAMILY,
        },
    )


def add_chart_header(fig, ax, title: str, subtitle: str) -> None:
    ax.set_title("")
    fig.subplots_adjust(top=0.84)
    left = ax.get_position().x0
    fig.text(left, 0.985, title, ha="left", va="top", fontsize=13, fontweight="semibold", color=TOKENS["ink"])
    fig.text(left, 0.94, subtitle, ha="left", va="top", fontsize=9, color=TOKENS["muted"])


def load_model_metrics(model_specs: list[tuple[str, str, Path]]) -> pd.DataFrame:
    frames = []
    for model_key, model_label, path in model_specs:
        if not path.exists():
            raise FileNotFoundError(path)
        frame = pd.read_csv(path)
        frame["model_key"] = model_key
        frame["model"] = model_label
        frames.append(frame)
    df = pd.concat(frames, ignore_index=True)
    df["is_expected_track"] = df["track"].eq(df["expected_track"])
    df["peak_ratio"] = df["predicted_max_full_label_window"] / df["observed_max_full_label_window"].replace(0, np.nan)
    df["log2_peak_ratio"] = np.log2(df["peak_ratio"].replace(0, np.nan))
    df["abs_log2_peak_ratio"] = df["log2_peak_ratio"].abs()
    df["peak_shift_abs_bp"] = (df["predicted_peak_pos"] - df["observed_peak_pos"]).abs()
    df["mean_ratio"] = df["predicted_mean_full_label_window"] / df["observed_mean_full_label_window"].replace(0, np.nan)
    row_order = {
        (gene, track): i * len(TRACK_ORDER) + j
        for i, gene in enumerate(GENE_ORDER)
        for j, track in enumerate(TRACK_ORDER)
    }
    df["row_order"] = df.apply(lambda r: row_order[(r["gene"], r["track"])], axis=1)
    df["row_label"] = df.apply(
        lambda r: f"{r['gene']} {r['track']}" + (" *" if r["is_expected_track"] else ""),
        axis=1,
    )
    return df.sort_values(["row_order", "model_key"]).reset_index(drop=True)


def summarize(long_df: pd.DataFrame, model_specs: list[tuple[str, str, Path]]) -> pd.DataFrame:
    slices = [
        ("all_tracks", long_df),
        ("expected_tracks_only", long_df[long_df["is_expected_track"]]),
        ("test_loci_all_tracks", long_df[long_df["split_role"].eq("test")]),
        ("train_chrom_loci_all_tracks", long_df[~long_df["split_role"].eq("test")]),
    ]
    rows = []
    for slice_name, subset in slices:
        for model_key, model_label, _path in model_specs:
            part = subset[subset["model_key"].eq(model_key)]
            rows.append(
                {
                    "slice": slice_name,
                    "model_key": model_key,
                    "model": model_label,
                    "n_rows": len(part),
                    "pearson_mean": part["pearson_full_label_window"].mean(),
                    "pearson_median": part["pearson_full_label_window"].median(),
                    "mse_median": part["mse_full_label_window"].median(),
                    "peak_ratio_median": part["peak_ratio"].median(),
                    "abs_log2_peak_ratio_median": part["abs_log2_peak_ratio"].median(),
                    "peak_shift_abs_bp_median": part["peak_shift_abs_bp"].median(),
                    "mean_ratio_mean": part["predicted_mean_full_label_window"].mean()
                    / part["observed_mean_full_label_window"].mean(),
                }
            )
    return pd.DataFrame(rows)


def export_both(fig: plt.Figure, out_path: Path) -> None:
    fig.savefig(out_path.with_suffix(".png"), dpi=180, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def plot_summary(summary: pd.DataFrame, out_dir: Path, model_specs: list[tuple[str, str, Path]]) -> None:
    part = summary[summary["slice"].eq("all_tracks")].copy()
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.6), sharey=True)
    model_order = [label for _key, label, _path in model_specs]
    model_palette = palette_for_models(model_specs)
    metrics = [
        ("pearson_mean", "Mean Pearson", "%.2f"),
        ("abs_log2_peak_ratio_median", "Median |log2 peak ratio|", "%.2f"),
        ("peak_shift_abs_bp_median", "Median peak shift (bp)", "%.0f"),
    ]
    for idx, (ax, (metric, label, value_fmt)) in enumerate(zip(axes, metrics)):
        sns.barplot(
            data=part,
            y="model",
            x=metric,
            order=model_order,
            hue="model",
            palette=model_palette,
            ax=ax,
            legend=False,
            edgecolor=TOKENS["ink"],
            linewidth=0.7,
        )
        ax.set_xlabel(label)
        ax.set_ylabel("")
        ax.grid(True, axis="x")
        xmax = float(part[metric].max())
        ax.set_xlim(0, xmax * 1.22 if xmax > 0 else 1)
        if idx > 0:
            ax.tick_params(axis="y", left=False, labelleft=False)
        for container in ax.containers:
            ax.bar_label(container, fmt=value_fmt, fontsize=8, padding=3)
    add_chart_header(
        fig,
        axes[0],
        "Positive-control summary across Borzoi and AlphaGenome runs",
        "All 36 gene-track rows; Pearson higher is better, peak-ratio error and peak shift lower are better.",
    )
    export_both(fig, out_dir / "summary_all_tracks")


def plot_pearson_heatmap(long_df: pd.DataFrame, out_dir: Path, model_specs: list[tuple[str, str, Path]]) -> None:
    model_order = [label for _key, label, _path in model_specs]
    matrix = (
        long_df.pivot_table(index="row_label", columns="model", values="pearson_full_label_window", aggfunc="first")
        .loc[[label for label in long_df.sort_values("row_order")["row_label"].drop_duplicates()]]
        .loc[:, model_order]
    )
    cmap = LinearSegmentedColormap.from_list("pearson", [TOKENS["panel"], COLORS["blue"]["light"], COLORS["blue"]["dark"]])
    fig, ax = plt.subplots(figsize=(8.4, 10.4))
    sns.heatmap(
        matrix,
        ax=ax,
        cmap=cmap,
        vmin=0,
        vmax=1,
        linewidths=0.65,
        linecolor=TOKENS["panel"],
        annot=True,
        fmt=".2f",
        cbar_kws={"label": "Pearson"},
    )
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(axis="x", labelrotation=25, labelsize=8)
    ax.tick_params(axis="y", labelsize=8)
    add_chart_header(
        fig,
        ax,
        "Per-locus Pearson heatmap",
        "Rows marked with * are expected cell-type tracks; each cell uses the full supervised label window.",
    )
    export_both(fig, out_dir / "pearson_heatmap")


def plot_peak_scatter(long_df: pd.DataFrame, out_dir: Path, model_specs: list[tuple[str, str, Path]]) -> None:
    models = [label for _key, label, _path in model_specs]
    n_models = len(models)
    n_cols = 2 if n_models > 1 else 1
    n_rows = int(math.ceil(n_models / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.9 * n_cols, 5.0 * n_rows), sharex=True, sharey=True)
    axes = np.atleast_1d(axes).flatten()
    finite = long_df[["observed_max_full_label_window", "predicted_max_full_label_window"]].replace([np.inf, -np.inf], np.nan)
    lim_min = max(0.05, float(np.nanmin(finite.to_numpy())) * 0.7)
    lim_max = float(np.nanmax(finite.to_numpy())) * 1.5
    for ax, model in zip(axes, models):
        part = long_df[long_df["model"].eq(model)]
        for track in TRACK_ORDER:
            sub = part[part["track"].eq(track)]
            ax.scatter(
                sub["observed_max_full_label_window"],
                sub["predicted_max_full_label_window"],
                s=np.where(sub["is_expected_track"], 58, 34),
                color=TRACK_COLORS[track],
                edgecolor=TOKENS["ink"],
                linewidth=np.where(sub["is_expected_track"], 1.2, 0.45),
                alpha=0.86,
                label=track,
            )
        ax.plot([lim_min, lim_max], [lim_min, lim_max], color=TOKENS["ink"], linestyle=":", linewidth=1.0)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(lim_min, lim_max)
        ax.set_ylim(lim_min, lim_max)
        ax.set_title(model, fontsize=10, color=TOKENS["ink"])
        ax.grid(True, which="major", color=TOKENS["grid"])
    for ax in axes[n_cols * (n_rows - 1) :]:
        ax.set_xlabel("Observed peak CPM")
    for ax in axes[::n_cols]:
        ax.set_ylabel("Predicted peak CPM")
    for ax in axes[n_models:]:
        ax.axis("off")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles[:4], labels[:4], loc="upper right", bbox_to_anchor=(0.96, 0.93), frameon=False, ncol=4)
    add_chart_header(
        fig,
        axes[0],
        "Predicted versus observed peak height",
        "Log-log scale across 36 gene-track rows; outlined larger points are expected tracks.",
    )
    axes[0].set_title(models[0], fontsize=10, color=TOKENS["ink"])
    export_both(fig, out_dir / "peak_height_scatter")


def write_wide(long_df: pd.DataFrame, out_dir: Path, model_specs: list[tuple[str, str, Path]]) -> None:
    base_cols = ["gene", "chrom", "start", "end", "split_role", "expected_track", "track", "is_expected_track"]
    value_cols = [
        "pearson_full_label_window",
        "mse_full_label_window",
        "observed_mean_full_label_window",
        "predicted_mean_full_label_window",
        "observed_max_full_label_window",
        "predicted_max_full_label_window",
        "peak_ratio",
        "log2_peak_ratio",
        "abs_log2_peak_ratio",
        "peak_shift_abs_bp",
        "mean_ratio",
    ]
    pieces = []
    for model_key, _model_label, _path in model_specs:
        part = long_df[long_df["model_key"].eq(model_key)][base_cols + value_cols].copy()
        part = part.rename(columns={col: f"{model_key}_{col}" for col in value_cols})
        pieces.append(part)
    wide = pieces[0]
    for part in pieces[1:]:
        wide = wide.merge(part, on=base_cols, how="inner")
    wide.to_csv(out_dir / "per_track_model_comparison_wide.csv", index=False)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    use_theme()
    model_specs = model_specs_from_args(args)
    long_df = load_model_metrics(model_specs)
    summary = summarize(long_df, model_specs)
    long_df.to_csv(out_dir / "per_track_model_comparison_long.csv", index=False)
    summary.to_csv(out_dir / "summary.csv", index=False)
    write_wide(long_df, out_dir, model_specs)
    plot_summary(summary, out_dir, model_specs)
    plot_pearson_heatmap(long_df, out_dir, model_specs)
    plot_peak_scatter(long_df, out_dir, model_specs)
    print("wrote", out_dir)


if __name__ == "__main__":
    main()
