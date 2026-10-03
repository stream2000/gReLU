#!/usr/bin/env python
"""Build a compact visual report for positive-control peak comparisons."""

from __future__ import annotations

import html
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.patches import Patch


ROOT = Path(__file__).resolve().parents[2]
INPUT_CSV = ROOT / "experiments/validation/positive_control_peak_model_comparison.csv"
OUT_DIR = ROOT / "experiments/validation/peak_display"

MODEL_SPECS = [
    ("mse_lora_e19", "MSE LoRA e19", "gene-window", "observed_peak_cpm_mse_source", "observed_peak_pos_mse_source"),
    (
        "poisson_lora_e19",
        "Poisson LoRA e19",
        "full-label",
        "observed_peak_cpm_poisson_lora_e19_source",
        "observed_peak_pos_poisson_lora_e19_source",
    ),
    (
        "poisson_lora_e39",
        "Poisson LoRA e39",
        "full-label",
        "observed_peak_cpm_poisson_lora_e39_source",
        "observed_peak_pos_poisson_lora_e39_source",
    ),
]

GENE_ORDER = ["Cd68", "Col1a1", "Epcam", "Fabp4", "Mmp2", "Myc", "Nipbl", "Pcdh17", "Trem2"]
TRACK_ORDER = ["hsc", "mac", "lsec", "chol"]

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

TRACK_COLORS = {
    "hsc": COLORS["blue"]["base"],
    "mac": COLORS["orange"]["base"],
    "lsec": COLORS["olive"]["base"],
    "chol": COLORS["pink"]["base"],
}


def use_theme() -> None:
    sns.set_theme(
        style="whitegrid",
        rc={
            "figure.facecolor": TOKENS["surface"],
            "savefig.facecolor": TOKENS["surface"],
            "axes.facecolor": TOKENS["panel"],
            "axes.edgecolor": TOKENS["axis"],
            "axes.labelcolor": TOKENS["ink"],
            "grid.color": TOKENS["grid"],
            "grid.linewidth": 0.8,
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Arial", "sans-serif"],
        },
    )


def add_header(fig, ax, title: str, subtitle: str) -> None:
    ax.set_title("")
    fig.subplots_adjust(top=0.86)
    left = ax.get_position().x0
    fig.text(left, 0.985, title, ha="left", va="top", fontsize=13, fontweight="semibold", color=TOKENS["ink"])
    fig.text(left, 0.94, subtitle, ha="left", va="top", fontsize=9, color=TOKENS["muted"])
    sns.despine(ax=ax)


def load_long() -> tuple[pd.DataFrame, pd.DataFrame]:
    wide = pd.read_csv(INPUT_CSV)
    order = {(gene, track): i * len(TRACK_ORDER) + j for i, gene in enumerate(GENE_ORDER) for j, track in enumerate(TRACK_ORDER)}
    wide["row_order"] = wide.apply(lambda r: order[(r["gene"], r["track"])], axis=1)
    wide["expected_track_flag"] = wide["expected_track_flag"].fillna("")
    wide["row_label"] = wide.apply(
        lambda r: f"{r['gene']} {r['track']}" + (" [expected]" if r["expected_track_flag"] == "expected" else ""),
        axis=1,
    )
    wide = wide.sort_values("row_order").reset_index(drop=True)

    rows = []
    for _, source in wide.iterrows():
        for key, label, window, obs_peak_col, _obs_pos_col in MODEL_SPECS:
            pred_over_obs = float(source[f"{key}_pred_over_obs"])
            rows.append(
                {
                    "gene": source["gene"],
                    "track": source["track"],
                    "row_label": source["row_label"],
                    "row_order": source["row_order"],
                    "model": label,
                    "model_key": key,
                    "peak_source_window": window,
                    "split_role": source["split_role"],
                    "is_expected_track": source["expected_track_flag"] == "expected",
                    "observed_peak_cpm": float(source[obs_peak_col]),
                    "pred_peak_cpm": float(source[f"{key}_pred_peak_cpm"]),
                    "pred_over_obs": pred_over_obs,
                    "delta_vs_obs": float(source[f"{key}_delta_vs_obs"]),
                    "log2_pred_over_obs": math.log2(pred_over_obs),
                    "abs_delta_vs_obs": abs(float(source[f"{key}_delta_vs_obs"])),
                    "peak_shift_abs_bp": abs(float(source[f"{key}_peak_shift_bp"])),
                    "peak_shift_bp": float(source[f"{key}_peak_shift_bp"]),
                    "pearson": float(source[f"{key}_pearson"]),
                    "mse": float(source[f"{key}_mse"]),
                }
            )
    return wide, pd.DataFrame(rows)


def chart_heatmap(long: pd.DataFrame, out_path: Path) -> None:
    matrix = (
        long.pivot_table(index="row_label", columns="model", values="log2_pred_over_obs", aggfunc="first")
        .loc[[label for label in long.sort_values("row_order")["row_label"].drop_duplicates()]]
        .loc[:, [spec[1] for spec in MODEL_SPECS]]
    )
    def ratio_label(log2_ratio: float) -> str:
        ratio = 2 ** float(log2_ratio)
        if ratio < 0.1:
            return "<0.1x"
        if ratio >= 10:
            return f"{ratio:.0f}x"
        return f"{ratio:.1f}x"

    annot = matrix.applymap(ratio_label)
    cmap = LinearSegmentedColormap.from_list(
        "under_over",
        [COLORS["orange"]["dark"], COLORS["orange"]["xlight"], TOKENS["panel"], COLORS["blue"]["xlight"], COLORS["blue"]["dark"]],
    )

    fig, ax = plt.subplots(figsize=(8.8, 11.0))
    sns.heatmap(
        matrix,
        ax=ax,
        cmap=cmap,
        norm=TwoSlopeNorm(vmin=-2.5, vcenter=0, vmax=2.5),
        linewidths=0.8,
        linecolor=TOKENS["panel"],
        annot=annot,
        fmt="",
        cbar_kws={"label": "log2(predicted / observed)"},
    )
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(axis="y", labelsize=8)
    ax.tick_params(axis="x", labelrotation=0, labelsize=9)
    add_header(
        fig,
        ax,
        "Peak-height error pattern across loci and tracks",
        "Cell labels show predicted/observed peak height; orange is underprediction and blue is overprediction.",
    )
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def chart_scatter(long: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.4), sharex=True, sharey=True)
    for ax, (_, label, *_rest) in zip(axes, MODEL_SPECS):
        part = long[long["model"] == label]
        for track in TRACK_ORDER:
            sub = part[part["track"] == track]
            ax.scatter(
                sub["observed_peak_cpm"],
                sub["pred_peak_cpm"],
                s=np.where(sub["is_expected_track"], 60, 34),
                facecolor=TRACK_COLORS[track],
                edgecolor=COLORS["gold"]["dark"] if track else TOKENS["ink"],
                linewidth=np.where(sub["is_expected_track"], 1.4, 0.6),
                alpha=0.82,
                label=track,
            )
        lim_min = 0.08
        lim_max = max(part["observed_peak_cpm"].max(), part["pred_peak_cpm"].max()) * 1.35
        ax.plot([lim_min, lim_max], [lim_min, lim_max], color=TOKENS["ink"], linestyle=":", linewidth=1.0)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(lim_min, lim_max)
        ax.set_ylim(lim_min, lim_max)
        ax.set_title(label, fontsize=10, color=TOKENS["ink"])
        ax.grid(True, which="major", color=TOKENS["grid"])
    axes[0].set_ylabel("Predicted peak CPM")
    for ax in axes:
        ax.set_xlabel("Observed peak CPM")
    handles = [Patch(facecolor=TRACK_COLORS[t], edgecolor=TOKENS["ink"], label=t) for t in TRACK_ORDER]
    axes[-1].legend(handles=handles, loc="lower right", frameon=False, fontsize=8)
    add_header(
        fig,
        axes[0],
        "Predicted peak heights move closer to the identity line in later Poisson LoRA",
        "Log-log scatter; larger outlined points are expected tracks where a canonical cell type was specified.",
    )
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def chart_summary(long: pd.DataFrame, out_path: Path) -> pd.DataFrame:
    summary_rows = []
    for subset_label, subset in [("All 36 peaks", long), ("Expected tracks only", long[long["is_expected_track"]])]:
        for _, model_label, *_rest in MODEL_SPECS:
            part = subset[subset["model"] == model_label]
            summary_rows.append(
                {
                    "subset": subset_label,
                    "model": model_label,
                    "median_abs_float": part["abs_delta_vs_obs"].median(),
                    "within_2x": part["pred_over_obs"].between(0.5, 2.0).mean(),
                    "median_abs_shift_bp": part["peak_shift_abs_bp"].median(),
                    "median_pearson": part["pearson"].median(),
                    "n": len(part),
                }
            )
    summary = pd.DataFrame(summary_rows)

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))
    palette = {
        "MSE LoRA e19": COLORS["orange"]["base"],
        "Poisson LoRA e19": COLORS["gold"]["base"],
        "Poisson LoRA e39": COLORS["olive"]["base"],
    }
    sns.barplot(
        data=summary,
        x="median_abs_float",
        y="model",
        hue="subset",
        ax=axes[0],
        palette=[COLORS["blue"]["light"], COLORS["blue"]["base"]],
        edgecolor=TOKENS["ink"],
        linewidth=0.8,
    )
    axes[0].set_xlabel("Median absolute float")
    axes[0].set_ylabel("")
    axes[0].xaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    axes[0].legend(frameon=False, loc="lower right", fontsize=8)

    wide = summary[summary["subset"] == "All 36 peaks"].copy()
    bars = axes[1].barh(
        wide["model"],
        wide["within_2x"],
        color=[palette[m] for m in wide["model"]],
        edgecolor=[COLORS["orange"]["dark"], COLORS["gold"]["dark"], COLORS["olive"]["dark"]],
        linewidth=0.8,
    )
    for bar, value in zip(bars, wide["within_2x"]):
        axes[1].text(value + 0.015, bar.get_y() + bar.get_height() / 2, f"{value:.0%}", va="center", fontsize=8)
    axes[1].set_xlim(0, 1)
    axes[1].set_xlabel("Share within 0.5x-2x of observed")
    axes[1].set_ylabel("")
    axes[1].xaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    add_header(
        fig,
        axes[0],
        "Poisson LoRA e39 has the most compact peak-height error",
        "Summary uses pred/observed peak height float across the 36 locus-track rows and the expected-track subset.",
    )
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return summary


def write_html(wide: pd.DataFrame, summary: pd.DataFrame, out_path: Path) -> None:
    display_cols = [
        "gene",
        "split_role",
        "track",
        "expected_track_flag",
        "observed_peak_cpm_mse_source",
        "mse_lora_e19_pred_over_obs",
        "poisson_lora_e19_pred_over_obs",
        "poisson_lora_e39_pred_over_obs",
        "mse_lora_e19_peak_shift_bp",
        "poisson_lora_e19_peak_shift_bp",
        "poisson_lora_e39_peak_shift_bp",
    ]
    table = wide[display_cols].copy()
    table = table.rename(
        columns={
            "split_role": "split",
            "expected_track_flag": "expected",
            "observed_peak_cpm_mse_source": "obs_peak_cpm",
            "mse_lora_e19_pred_over_obs": "mse_e19_x",
            "poisson_lora_e19_pred_over_obs": "pois_e19_x",
            "poisson_lora_e39_pred_over_obs": "pois_e39_x",
            "mse_lora_e19_peak_shift_bp": "mse_shift_bp",
            "poisson_lora_e19_peak_shift_bp": "pois_e19_shift_bp",
            "poisson_lora_e39_peak_shift_bp": "pois_e39_shift_bp",
        }
    )
    table["expected"] = table["expected"].fillna("")
    for col in ["obs_peak_cpm", "mse_e19_x", "pois_e19_x", "pois_e39_x"]:
        table[col] = table[col].map(lambda v: f"{v:.2f}")
    for col in ["mse_shift_bp", "pois_e19_shift_bp", "pois_e39_shift_bp"]:
        table[col] = table[col].map(lambda v: f"{int(v):+d}")

    cards = []
    for _, row in summary[summary["subset"] == "All 36 peaks"].iterrows():
        cards.append(
            f"""
            <div class="card">
              <div class="kpi-label">{html.escape(row['model'])}</div>
              <div class="kpi-value">{row['median_abs_float']:.0%}</div>
              <div class="kpi-note">median abs float; {row['within_2x']:.0%} within 0.5x-2x</div>
            </div>
            """
        )

    html_text = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Positive-control peak comparison</title>
  <style>
    :root {{
      --surface: #FCFCFD;
      --panel: #FFFFFF;
      --ink: #1F2430;
      --muted: #6F768A;
      --grid: #E6E8F0;
      --axis: #D7DBE7;
      --blue: #5477C4;
    }}
    body {{
      margin: 0;
      background: var(--surface);
      color: var(--ink);
      font-family: Inter, Segoe UI, Arial, sans-serif;
      line-height: 1.45;
    }}
    main {{
      max-width: 1180px;
      margin: 0 auto;
      padding: 28px 24px 42px;
    }}
    h1 {{
      margin: 0 0 6px;
      font-size: 26px;
      letter-spacing: 0;
    }}
    h2 {{
      margin: 28px 0 8px;
      font-size: 17px;
    }}
    p {{
      max-width: 920px;
      margin: 0 0 12px;
      color: var(--muted);
    }}
    .cards {{
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 12px;
      margin: 20px 0 22px;
    }}
    .card {{
      background: var(--panel);
      border: 1px solid var(--grid);
      border-radius: 8px;
      padding: 14px 16px;
    }}
    .kpi-label {{
      font-size: 13px;
      color: var(--muted);
    }}
    .kpi-value {{
      margin-top: 4px;
      font-size: 28px;
      font-weight: 700;
    }}
    .kpi-note {{
      margin-top: 2px;
      font-size: 12px;
      color: var(--muted);
    }}
    img {{
      width: 100%;
      height: auto;
      display: block;
      background: white;
      border: 1px solid var(--grid);
      border-radius: 8px;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      background: var(--panel);
      border: 1px solid var(--grid);
      font-size: 12px;
    }}
    th, td {{
      border-bottom: 1px solid var(--grid);
      padding: 6px 8px;
      text-align: right;
      white-space: nowrap;
    }}
    th:first-child, td:first-child,
    th:nth-child(2), td:nth-child(2),
    th:nth-child(3), td:nth-child(3),
    th:nth-child(4), td:nth-child(4) {{
      text-align: left;
    }}
    th {{
      color: var(--muted);
      font-weight: 600;
      position: sticky;
      top: 0;
      background: var(--panel);
    }}
    .table-wrap {{
      overflow-x: auto;
      border-radius: 8px;
    }}
    .note {{
      border-left: 3px solid var(--blue);
      padding-left: 12px;
    }}
    @media (max-width: 760px) {{
      .cards {{ grid-template-columns: 1fr; }}
      main {{ padding: 20px 14px 30px; }}
    }}
  </style>
</head>
<body>
<main>
  <h1>Positive-control peak comparison</h1>
  <p>Readable view of the 36 locus-track rows. The main encoding is predicted/observed peak height, with peak-position shift kept as a secondary check.</p>
  <div class="cards">
    {''.join(cards)}
  </div>
  <h2>Best first view: signed peak-height error</h2>
  <p>Use this heatmap to spot systematic underprediction and overprediction. A value near 1.0x is calibrated; orange cells are low, blue cells are high.</p>
  <img src="peak_height_error_heatmap.png" alt="Peak-height error heatmap">
  <h2>Calibration view</h2>
  <p>Points on the dotted line have correct peak height. Later Poisson LoRA brings more points close to the identity line, but expected tracks still contain large underpredictions.</p>
  <img src="peak_height_scatter.png" alt="Predicted versus observed peak heights">
  <h2>Summary view</h2>
  <p>This compresses the comparison into two operational metrics: median absolute float and share of peaks predicted within 0.5x to 2x of observed.</p>
  <img src="peak_summary_bars.png" alt="Summary bars">
  <h2>Source table</h2>
  <p class="note">Caveat: current MSE-LoRA rows use saved gene-window diagnostics; current Poisson-LoRA rows use saved full-label-window diagnostics. For a final paper figure, regenerate all three models from the same exact peak-window definition.</p>
  <div class="table-wrap">
    {table.to_html(index=False, escape=True)}
  </div>
</main>
</body>
</html>
"""
    out_path.write_text(html_text)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    use_theme()
    wide, long = load_long()
    chart_heatmap(long, OUT_DIR / "peak_height_error_heatmap.png")
    chart_scatter(long, OUT_DIR / "peak_height_scatter.png")
    summary = chart_summary(long, OUT_DIR / "peak_summary_bars.png")
    summary.to_csv(OUT_DIR / "peak_summary.csv", index=False)
    long.to_csv(OUT_DIR / "peak_comparison_long.csv", index=False)
    write_html(wide, summary, OUT_DIR / "index.html")
    print(OUT_DIR / "index.html")


if __name__ == "__main__":
    main()
