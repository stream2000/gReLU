#!/usr/bin/env python
"""Render MREG REF/ALT predictions as genome-browser style tracks.

This draws REF and ALT as separate filled signal tracks on the same genomic
scale, with shared y-limits per target and a vertical marker for the edit or
TSS anchor. The default output is a single combined PDF:

1. Three local mutation-window pages: TSS, HC-DIC, LC-DIC edits.
2. DIC-to-MREG-TSS pages split into binding/TF tracks and active-mark tracks.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd


MUTATION_LABELS = {
    "mreg_tss_00_ctcf_snv": "MREG TSS CTCF edit",
    "mreg_hc_dic_00_ctcf_snv": "HC-DIC CTCF edit",
    "mreg_lc_dic_00_pol2_perturbation": "LC-DIC Pol2-center edit",
}

LOCAL_READOUT = "mutation_local_4kb"
TSS_READOUT = "mreg_tss_4kb"
TSS_ANCHOR = 216_013_551

REF_COLOR = "#2B8CFF"
ALT_COLOR = "#F59E0B"
REF_EDGE = "#0B4F8A"
ALT_EDGE = "#C2410C"
MARKER_COLOR = "#DC2626"

TSS_TARGET_PAGE_GROUPS = [
    (
        "binding and TF tracks",
        ["CTCF", "RAD21", "POLR2A", "EP300", "FOXA1", "GATA3", "ESR1"],
    ),
    (
        "active chromatin marks",
        ["H3K27ac", "H3K4me1", "H3K4me2", "H3K4me3"],
    ),
]


def _load_inputs(
    profiles_path: Path,
    resolved_tracks_path: Path,
    intervals_path: Path,
    biosample: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    all_profiles = pd.read_parquet(profiles_path)
    all_profiles = all_profiles[
        all_profiles["biosample"].astype(str) == biosample
    ].copy()
    tracks = pd.read_csv(resolved_tracks_path, sep="\t")
    tracks = tracks[tracks["biosample_resolved"].astype(str) == biosample].copy()
    intervals = pd.read_csv(intervals_path, sep="\t")
    all_profiles = all_profiles[
        all_profiles["target_name"].isin(tracks["target_name"].astype(str))
    ].copy()
    profiles = all_profiles[
        all_profiles["control_type"].astype(str) == "experimental"
    ].copy()
    if profiles.empty:
        raise SystemExit(f"No experimental profiles found for biosample={biosample!r}")
    return profiles, tracks, intervals, all_profiles


def _target_order(tracks: pd.DataFrame) -> list[str]:
    preferred = [
        "CTCF",
        "RAD21",
        "POLR2A",
        "EP300",
        "FOXA1",
        "GATA3",
        "ESR1",
        "H3K27ac",
        "H3K4me1",
        "H3K4me2",
        "H3K4me3",
    ]
    present = set(tracks["target_name"].astype(str))
    ordered = [t for t in preferred if t in present]
    ordered.extend(sorted(present - set(ordered)))
    return ordered


def _plot_track_pair(
    ax,
    data: pd.DataFrame,
    *,
    target: str,
    marker_x: int | None,
    marker_label: str,
    mode: str,
    label_marker: bool,
) -> None:
    if data.empty:
        ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return

    data = data.sort_values("genomic_start")
    x = data["genomic_start"].astype(float)
    x2 = data["genomic_end"].astype(float)
    ref = data["ref_value"].astype(float)
    alt = data["alt_value"].astype(float)
    ymax = max(float(ref.max()), float(alt.max()), 1.0)

    if mode == "stacked":
        gap = ymax * 0.18
        alt_offset = -(ymax + gap)
        ax.fill_between(
            x,
            0,
            ref,
            step="post",
            facecolor=REF_COLOR,
            edgecolor=REF_EDGE,
            linewidth=0.85,
            alpha=0.9,
        )
        ax.fill_between(
            x,
            alt_offset,
            alt_offset + alt,
            step="post",
            facecolor=ALT_COLOR,
            edgecolor=ALT_EDGE,
            linewidth=0.85,
            alpha=0.9,
        )
        ax.hlines(0, x.min(), x2.max(), color=REF_EDGE, linewidth=0.75)
        ax.hlines(alt_offset, x.min(), x2.max(), color=ALT_EDGE, linewidth=0.75)
        ax.text(
            x.min(),
            ymax * 0.82,
            "REF",
            ha="left",
            va="center",
            fontsize=6.5,
            color=REF_EDGE,
        )
        ax.text(
            x.min(),
            alt_offset + ymax * 0.82,
            "ALT",
            ha="left",
            va="center",
            fontsize=6.5,
            color=ALT_EDGE,
        )
        ax.set_ylim(alt_offset - ymax * 0.08, ymax * 1.22)
    elif mode == "overlay":
        ax.fill_between(
            x,
            0,
            ref,
            step="post",
            facecolor=REF_COLOR,
            edgecolor=REF_EDGE,
            linewidth=0.95,
            alpha=0.32,
        )
        ax.step(x, ref, where="post", color=REF_EDGE, linewidth=1.1, label="REF")
        ax.fill_between(
            x,
            0,
            alt,
            step="post",
            facecolor=ALT_COLOR,
            edgecolor=ALT_EDGE,
            linewidth=0.85,
            alpha=0.24,
        )
        ax.step(x, alt, where="post", color=ALT_EDGE, linewidth=1.1, label="ALT")
        ax.hlines(0, x.min(), x2.max(), color="#333333", linewidth=0.65)
        delta_mean = float(alt.mean() - ref.mean())
        ax.text(
            0.995,
            0.82,
            f"mean Δ {delta_mean:+.3g}",
            ha="right",
            va="center",
            fontsize=6.2,
            color="#555555",
            transform=ax.transAxes,
        )
        ax.set_ylim(-ymax * 0.06, ymax * 1.22)
    else:
        raise ValueError(f"unknown plotting mode: {mode}")

    if marker_x is not None:
        ax.axvline(marker_x, color=MARKER_COLOR, linestyle="--", linewidth=0.85, alpha=0.75)
        if label_marker:
            ax.text(
                marker_x,
                0.97,
                marker_label,
                ha="center",
                va="top",
                fontsize=7,
                color=MARKER_COLOR,
                rotation=90,
                transform=ax.get_xaxis_transform(),
            )

    ax.set_xlim(x.min(), x2.max())
    ax.set_yticks([])
    ax.tick_params(axis="x", labelsize=6)
    ax.spines[["left", "right", "top"]].set_visible(False)
    ax.set_ylabel(target, fontsize=9, rotation=0, ha="right", va="center", labelpad=50)


def _render_page(
    pdf: PdfPages,
    profiles: pd.DataFrame,
    tracks: pd.DataFrame,
    intervals: pd.DataFrame,
    *,
    mutation_id: str,
    readout_id: str,
    title: str,
    marker_kind: str,
    targets: list[str],
) -> None:
    mode = "stacked"
    fig_h = (
        max(6.2, 0.92 * len(targets) + 2.1)
        if marker_kind == "tss"
        else max(8.0, 0.62 * len(targets) + 1.8)
    )
    fig, axes = plt.subplots(len(targets), 1, figsize=(12.8, fig_h), sharex=True)
    if len(targets) == 1:
        axes = [axes]

    mut_profiles = profiles[
        (profiles["mutation_id"].astype(str) == mutation_id)
        & (profiles["readout_id"].astype(str) == readout_id)
    ]
    interval = intervals[intervals["mutation_id"].astype(str) == mutation_id]
    edit_center = int(interval.iloc[0]["edit_center"]) if not interval.empty else None
    marker_x = edit_center if marker_kind == "edit" else TSS_ANCHOR
    marker_label = "edit" if marker_kind == "edit" else "MREG TSS"

    for row_idx, (ax, target) in enumerate(zip(axes, targets)):
        panel = mut_profiles[mut_profiles["target_name"].astype(str) == target]
        _plot_track_pair(
            ax,
            panel,
            target=target,
            marker_x=marker_x,
            marker_label=marker_label,
            mode=mode,
            label_marker=row_idx == 0,
        )

    axes[-1].set_xlabel("Genomic position (hg38)", fontsize=8)
    fig.suptitle(title, fontsize=13, y=0.97)
    fig.text(
        0.14,
        0.935,
        "MCF-7 exact AlphaGenome 128 bp ChIP predictions; filled tracks show raw model signal",
        ha="left",
        va="top",
        fontsize=8,
        color="#555555",
    )
    fig.subplots_adjust(left=0.15, right=0.99, top=0.88, bottom=0.085, hspace=0.42)
    pdf.savefig(fig)
    plt.close(fig)



def render_browser_style_pdf(
    *,
    profiles_path: Path,
    resolved_tracks_path: Path,
    intervals_path: Path,
    output_dir: Path,
    biosample: str,
) -> dict[str, Path]:
    profiles, tracks, intervals, all_profiles = _load_inputs(
        profiles_path, resolved_tracks_path, intervals_path, biosample
    )
    targets = _target_order(tracks)
    output_dir.mkdir(parents=True, exist_ok=True)

    combined_pdf = output_dir / "mreg_mcf7_browser_style_ref_alt_combined.pdf"
    with PdfPages(combined_pdf) as pdf:
        for mutation_id, label in MUTATION_LABELS.items():
            if mutation_id not in set(profiles["mutation_id"].astype(str)):
                continue
            _render_page(
                pdf,
                profiles,
                tracks,
                intervals,
                mutation_id=mutation_id,
                readout_id=LOCAL_READOUT,
                title=f"{label}: local 4 kb REF vs ALT tracks",
                marker_kind="edit",
                targets=targets,
            )

        for mutation_id in [
            "mreg_hc_dic_00_ctcf_snv",
            "mreg_lc_dic_00_pol2_perturbation",
        ]:
            if mutation_id not in set(profiles["mutation_id"].astype(str)):
                continue
            for page_label, page_targets in TSS_TARGET_PAGE_GROUPS:
                present_targets = [target for target in page_targets if target in targets]
                if not present_targets:
                    continue
                _render_page(
                    pdf,
                    profiles,
                    tracks,
                    intervals,
                    mutation_id=mutation_id,
                    readout_id=TSS_READOUT,
                    title=(
                        f"{MUTATION_LABELS.get(mutation_id, mutation_id)}: "
                        f"MREG TSS 4 kb REF vs ALT, {page_label}"
                    ),
                    marker_kind="tss",
                    targets=present_targets,
                )

    return {"combined_pdf": combined_pdf}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profiles", required=True)
    parser.add_argument("--resolved-tracks", required=True)
    parser.add_argument("--intervals", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--biosample", default="MCF-7")
    args = parser.parse_args()

    paths = render_browser_style_pdf(
        profiles_path=Path(args.profiles),
        resolved_tracks_path=Path(args.resolved_tracks),
        intervals_path=Path(args.intervals),
        output_dir=Path(args.output_dir),
        biosample=args.biosample,
    )
    for label, path in paths.items():
        print(f"[plot] {label}: {path}")


if __name__ == "__main__":
    main()
