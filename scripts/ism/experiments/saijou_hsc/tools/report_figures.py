"""Print-stable figures for the canonical Saijou report."""

from __future__ import annotations

import io
from pathlib import Path

import pandas as pd


MODEL_COLORS = ("#3274a1", "#e1812c")
MDK_SERIES_COLORS = ("#3274a1", "#8ab4d2", "#e1812c", "#f0b27a")


def save_svg(fig, path: Path, *, height_px: int = 292) -> str:
    """Save a reusable SVG and return a sandbox-safe inline HTML body."""

    import matplotlib.pyplot as plt

    stream = io.StringIO()
    fig.savefig(stream, format="svg", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    svg = stream.getvalue()
    svg = svg[svg.index("<svg") :]
    svg = svg.replace(
        "<svg ",
        f'<svg style="display:block;width:100%;height:{height_px}px" ',
        1,
    )
    path.write_text(svg)
    return f'<div style="padding:4px;background:#fff;color:#111">{svg}</div>'


def grouped_horizontal_bars(
    table: pd.DataFrame,
    *,
    labels: list[str],
    value: str,
    model_order: list[str],
    title: str,
    xlabel: str,
):
    """Build a compact two-model horizontal bar chart."""

    import matplotlib.pyplot as plt
    import numpy as np

    pivot = table.pivot(index="segment_label", columns="model", values=value).reindex(
        labels
    )
    fig, ax = plt.subplots(figsize=(8.8, 3.0))
    y = np.arange(len(labels))
    height = 0.36
    for index, model in enumerate(model_order):
        ax.barh(
            y + (index - 0.5) * height,
            pivot[model].to_numpy(),
            height=height,
            label=model.replace(" fine-tuned", "-FT").replace(" original", "-original"),
            color=MODEL_COLORS[index],
        )
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlabel(xlabel)
    ax.set_title(title, loc="left", fontweight="bold")
    ax.grid(axis="x", color="#d9d9d9", linewidth=0.6)
    ax.set_axisbelow(True)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.legend(frameon=False, ncol=2, fontsize=7, loc="lower right")
    fig.tight_layout(pad=0.8)
    return fig


def four_cell_heatmap(
    fine_matrix: pd.DataFrame,
    *,
    model: str,
    ordered_segments: list[str],
    title: str,
):
    """Build one model's candidate-by-cell effect heatmap."""

    import matplotlib.pyplot as plt

    cell_order = ["HSC", "MAC", "LSEC", "CHOL"]
    subset = fine_matrix.loc[fine_matrix.model_label.eq(model)]
    pivot = subset.pivot(
        index="segment_label", columns="cell_label", values="median_abs_effect"
    ).reindex(index=ordered_segments, columns=cell_order)
    fig, ax = plt.subplots(figsize=(8.8, 3.25))
    image = ax.imshow(pivot.to_numpy(), aspect="auto", cmap="Blues")
    ax.set_xticks(range(len(cell_order)), cell_order)
    ax.set_yticks(range(len(ordered_segments)), ordered_segments, fontsize=6.2)
    ax.set_title(title, loc="left", fontweight="bold")
    colorbar = fig.colorbar(image, ax=ax, fraction=0.025, pad=0.02)
    colorbar.set_label("Median |effect|", fontsize=7)
    fig.tight_layout(pad=0.7)
    return fig


def mdk_robustness_figure(mdk_centers: pd.DataFrame):
    """Build the cross-manifest Mdk HSC-margin figure."""

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8.8, 2.9))
    for index, (series, table) in enumerate(mdk_centers.groupby("series", sort=True)):
        table = table.sort_values("variant_offset_from_tss_transcription_bp")
        ax.plot(
            table.variant_offset_from_tss_transcription_bp,
            table.hsc_margin,
            label=series,
            linewidth=1.5,
            color=MDK_SERIES_COLORS[index],
        )
    ax.axhline(0, color="#555", linewidth=0.8, linestyle="--")
    ax.set_title(
        "Mdk +441..+507 HSC-selectivity margin across shuffle manifests",
        loc="left",
        fontweight="bold",
    )
    ax.set_xlabel("TSS offset (bp, transcription direction)")
    ax.set_ylabel("HSC margin")
    ax.grid(color="#e5e5e5", linewidth=0.5)
    ax.legend(frameon=False, fontsize=6.5, ncol=2, loc="lower right")
    fig.tight_layout(pad=0.8)
    return fig


def nine_gene_signed_browser(browser: pd.DataFrame):
    """Build a compact nine-gene view of direct signed HSC effect sizes."""

    import matplotlib.pyplot as plt
    import numpy as np

    genes = sorted(browser.gene.unique())
    fig, axes = plt.subplots(3, 3, figsize=(8.8, 7.1), sharex=True)
    style = {
        "AlphaGenome fine-tuned": (MODEL_COLORS[0], "-"),
        "Borzoi fine-tuned": (MODEL_COLORS[1], "--"),
    }
    for axis, gene in zip(axes.flat, genes):
        gene_data = browser.loc[browser.gene.eq(gene)]
        limit = max(0.05, float(gene_data.median_signed_effect.abs().max()) * 1.08)
        for model, model_data in gene_data.groupby("model_label", sort=False):
            model_data = model_data.sort_values(
                "variant_offset_from_tss_transcription_bp"
            )
            color, linestyle = style[model]
            axis.plot(
                model_data.variant_offset_from_tss_transcription_bp,
                model_data.signed_effect_smoothed,
                color=color,
                linestyle=linestyle,
                linewidth=1.15,
                label=model.replace(" fine-tuned", "-FT"),
            )
        axis.axhline(0, color="#555", linewidth=0.55)
        axis.axvline(0, color="#777", linewidth=0.5, linestyle=":")
        axis.set_ylim(-limit, limit)
        axis.set_title(gene, loc="left", fontweight="bold", fontsize=8.5)
        axis.grid(axis="y", color="#e8e8e8", linewidth=0.4)
        axis.tick_params(labelsize=6.5)
    for axis in axes[-1]:
        axis.set_xlabel("TSS offset (bp)")
    for axis in axes[:, 0]:
        axis.set_ylabel("signed log2FC")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, ncol=2, loc="upper center")
    fig.suptitle(
        "Nine-gene HSC Browser: direct signed effect (gene-specific symmetric scales)",
        x=0.02,
        y=0.995,
        ha="left",
        fontsize=10,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.965), pad=0.8)
    return fig


def build_all_genes_static_figures(
    report_dir: Path,
    overview: pd.DataFrame,
    hsc_ratio: pd.DataFrame,
    fine_matrix: pd.DataFrame,
    mdk_centers: pd.DataFrame,
    original_top: pd.DataFrame,
    fine_signed_browser: pd.DataFrame,
) -> dict[str, str]:
    """Build all print-stable figures for the nine-gene report."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.titlesize": 10,
            "axes.labelsize": 8,
            "svg.fonttype": "none",
        }
    )
    figure_dir = report_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    ordered = overview.sort_values(["gene", "segment_rank"]).segment_label.tolist()
    panels = [ordered[:10], ordered[10:]]
    static: dict[str, str] = {}

    for index, labels in enumerate(panels, start=1):
        figure = grouped_horizontal_bars(
            hsc_ratio,
            labels=labels,
            value="hsc_ratio",
            model_order=["AlphaGenome fine-tuned", "Borzoi fine-tuned"],
            title=f"HSC / strongest non-HSC effect ratio (panel {index}/2)",
            xlabel="HSC / max(mac, LSEC, chol)",
        )
        static[f"hsc_ratio_{index}"] = save_svg(
            figure, figure_dir / f"hsc_ratio_{index}.svg"
        )

    for model, key, title in [
        (
            "AlphaGenome fine-tuned",
            "fine_heatmap_ag",
            "AlphaGenome-FT four-cell effect matrix",
        ),
        (
            "Borzoi fine-tuned",
            "fine_heatmap_bz",
            "Borzoi-FT four-cell effect matrix",
        ),
    ]:
        static[key] = save_svg(
            four_cell_heatmap(
                fine_matrix,
                model=model,
                ordered_segments=ordered,
                title=title,
            ),
            figure_dir / f"{key}.svg",
        )

    static["mdk_robustness"] = save_svg(
        mdk_robustness_figure(mdk_centers), figure_dir / "mdk_robustness.svg"
    )
    static["fine_signed_browser"] = save_svg(
        nine_gene_signed_browser(fine_signed_browser),
        figure_dir / "fine_signed_browser.svg",
        height_px=610,
    )
    for index, labels in enumerate(panels, start=1):
        figure = grouped_horizontal_bars(
            original_top,
            labels=labels,
            value="abs_effect",
            model_order=["AlphaGenome original", "Borzoi original"],
            title=f"Strongest original-model track-group effect (panel {index}/2)",
            xlabel="Median |effect|",
        )
        static[f"original_top_{index}"] = save_svg(
            figure, figure_dir / f"original_top_{index}.svg"
        )
    return static
