#!/usr/bin/env python
"""Render the focused Mdk report from audited TSV/JSON outputs only."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from jinja2 import Environment, FileSystemLoader, StrictUndefined  # noqa: E402

if __package__:
    from ..harness import SAIJOU_ROOT, read_json, require_columns
else:
    from sys import path as sys_path

    sys_path.insert(0, str(Path(__file__).resolve().parents[2]))
    from tools.harness import SAIJOU_ROOT, read_json, require_columns


DEFAULT_ANALYSIS = SAIJOU_ROOT / "analysis/mdk_audited_effects"
DEFAULT_OUT = SAIJOU_ROOT / "report/mdk_audited"
TEMPLATE_DIR = Path(__file__).resolve().parent / "templates"
PRIMARY_LOCUS = "mdk_candidate_02_tx+463_+475"
SECONDARY_LOCUS = "mdk_candidate_01_tx+499_+507"
MODEL_STYLE = {
    "alphagenome_effect": ("AlphaGenome-FT", "#3274a1", "-"),
    "borzoi_effect": ("Borzoi-FT", "#e1812c", "--"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-dir", type=Path, default=DEFAULT_ANALYSIS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    return parser.parse_args()


def load_tables(analysis_dir: Path) -> tuple[dict[str, pd.DataFrame], dict]:
    """Load and schema-check the immutable analysis/report boundary."""

    names = (
        "four_cell_browser",
        "four_cell_window_summary",
        "borzoi_original_track_group_summary",
        "borzoi_original_track_group_chart",
        "borzoi_original_track_inventory",
        "borzoi_original_liver_rna_tracks",
        "motif_family_summary",
    )
    tables = {
        name: pd.read_csv(analysis_dir / f"{name}.tsv", sep="\t") for name in names
    }
    require_columns(
        tables["four_cell_browser"],
        ["cell", "offset_bp", "alphagenome_effect", "borzoi_effect"],
        name="four_cell_browser",
    )
    require_columns(
        tables["borzoi_original_track_group_summary"],
        ["locus_id", "track_group", "median_signed_log2fc"],
        name="borzoi_original_track_group_summary",
    )
    validation = read_json(analysis_dir / "validation_summary.json")
    if validation.get("status") != "ok":
        raise ValueError("Mdk analysis validation is not ok")
    return tables, validation


def plot_browser(table: pd.DataFrame, path: Path, y_limit: float) -> None:
    """Render four cell panels on one shared signed-effect scale."""

    cells = ("HSC", "Macrophage", "LSEC", "Cholangiocyte")
    fig, axes = plt.subplots(2, 2, figsize=(8.6, 5.2), sharex=True, sharey=True)
    for axis, cell in zip(axes.flat, cells):
        subset = table.loc[table.cell.eq(cell)].sort_values("offset_bp")
        for column, (label, color, linestyle) in MODEL_STYLE.items():
            axis.plot(
                subset.offset_bp,
                subset[column],
                label=label,
                color=color,
                linestyle=linestyle,
                linewidth=1.05,
            )
        axis.axhline(0, color="#555", linewidth=0.6)
        axis.axvline(0, color="#777", linewidth=0.55, linestyle=":")
        axis.axvspan(463, 475, color="#3274a1", alpha=0.11)
        axis.axvspan(499, 507, color="#e1812c", alpha=0.10)
        axis.set_title(cell, loc="left", fontweight="bold")
        axis.set_ylim(-y_limit, y_limit)
        axis.grid(axis="y", color="#e8e8e8", linewidth=0.45)
    for axis in axes[-1]:
        axis.set_xlabel("TSS offset (bp, transcription direction)")
    for axis in axes[:, 0]:
        axis.set_ylabel("median signed log2FC")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, ncol=2, loc="upper center")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, format="svg", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_original(table: pd.DataFrame, path: Path) -> None:
    """Render the two original-Borzoi window effects by track group."""

    table = table.sort_values("maximum_abs_effect").reset_index(drop=True)
    labels = table.track_group.str.replace("_", " ", regex=False)
    y = range(len(table))
    fig, axis = plt.subplots(figsize=(8.6, 4.6))
    axis.hlines(y, table.primary_effect, table.secondary_effect, color="#c8cdd3")
    axis.scatter(table.primary_effect, y, color="#3274a1", label="+463..+475", s=24)
    axis.scatter(table.secondary_effect, y, color="#e1812c", label="+499..+507", s=24)
    axis.axvline(0, color="#555", linewidth=0.7)
    axis.set_yticks(list(y), labels)
    axis.set_xlabel("median signed log2(mutant/reference)")
    axis.grid(axis="x", color="#e8e8e8", linewidth=0.45)
    axis.legend(frameon=False, ncol=2, loc="lower right")
    axis.spines[["top", "right", "left"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, format="svg", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _window_table(table: pd.DataFrame) -> str:
    display = table[
        [
            "window",
            "cell",
            "alphagenome_median_signed_log2fc",
            "alphagenome_cell_rank",
            "borzoi_median_signed_log2fc",
            "borzoi_cell_rank",
        ]
    ].copy()
    display.columns = ("Window", "Cell", "AG signed", "AG rank", "BZ signed", "BZ rank")
    return display.to_html(
        index=False, float_format=lambda value: f"{value:.3f}", border=0
    )


def _original_table(table: pd.DataFrame) -> str:
    display = table[
        ["window", "track_group", "median_signed_log2fc", "fraction_mutations_negative"]
    ].copy()
    display.columns = ("Window", "Track group", "Median signed", "Fraction negative")
    display = display.sort_values(["Window", "Median signed"])
    return display.to_html(
        index=False, float_format=lambda value: f"{value:.3f}", border=0
    )


def main() -> None:
    args = parse_args()
    analysis_dir = args.analysis_dir.resolve()
    out = args.out_dir.resolve()
    figures = out / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    tables, validation = load_tables(analysis_dir)

    browser_path = figures / "four_cell_browser.svg"
    original_path = figures / "original_borzoi_track_groups.svg"
    plot_browser(
        tables["four_cell_browser"], browser_path, validation["common_y_limit"]
    )
    plot_original(tables["borzoi_original_track_group_chart"], original_path)

    liver = tables["borzoi_original_liver_rna_tracks"].copy()
    liver["negative"] = liver.track_median_signed_log2fc.lt(0)
    direction = liver.groupby("locus_id").agg(
        tracks=("track_id", "nunique"), negative=("negative", "sum")
    )
    environment = Environment(
        loader=FileSystemLoader(TEMPLATE_DIR),
        undefined=StrictUndefined,
        autoescape=True,
    )
    template = environment.get_template("mdk_audited_report.html.j2")
    rendered = template.render(
        title="Mdk TSS-proximal 10-bp ISM",
        css=(TEMPLATE_DIR / "mdk_audited_report.css").read_text(),
        generated_at=datetime.now(timezone.utc).date().isoformat(),
        browser_figure=browser_path.relative_to(out).as_posix(),
        original_figure=original_path.relative_to(out).as_posix(),
        window_table=_window_table(tables["four_cell_window_summary"]),
        original_table=_original_table(tables["borzoi_original_track_group_summary"]),
        selected_tracks=validation["original_selected_tracks"],
        compatible_tracks=validation["original_strand_compatible_tracks"],
        secondary_negative=int(direction.loc[SECONDARY_LOCUS, "negative"]),
        liver_track_count=int(direction.loc[SECONDARY_LOCUS, "tracks"]),
        contract_version=validation["analysis_contract_version"],
        max_difference=f"{validation['original_summary_max_abs_difference_vs_saved']:.3g}",
    )
    html_path = out / "mdk_audited_report.html"
    html_path.write_text(rendered)
    print(html_path)


if __name__ == "__main__":
    main()
