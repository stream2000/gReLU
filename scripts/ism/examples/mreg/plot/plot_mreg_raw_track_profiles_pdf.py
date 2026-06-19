#!/usr/bin/env python
"""Render MREG raw REF/ALT track profiles into a multi-page PDF.

The input is the long-form parquet emitted by
run_mreg_three_region_chromatin.py. This plotter intentionally shows raw
128 bp model-track values before matched-control adjustment. It is intended for
cell-line-clean inspection, especially MCF-7 exact-track runs.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd


READOUT_COLUMNS = [
    ("mutation_center_1kb", "mutation center 1 kb"),
    ("mutation_local_4kb", "mutation local 4 kb"),
    ("mreg_tss_4kb", "MREG TSS 4 kb"),
]

MUTATION_LABELS = {
    "mreg_tss_00_ctcf_snv": "MREG TSS CTCF edit",
    "mreg_hc_dic_00_ctcf_snv": "HC-DIC CTCF edit",
    "mreg_lc_dic_00_pol2_perturbation": "LC-DIC Pol2-center edit",
}

REF_COLOR = "#2b6cb0"
ALT_COLOR = "#c53030"
DELTA_COLOR = "#4a5568"


def _experimental_profiles(profiles: pd.DataFrame) -> pd.DataFrame:
    exp = profiles[profiles["control_type"] == "experimental"].copy()
    if exp.empty:
        return profiles.copy()
    return exp


def _plot_panel(ax, group: pd.DataFrame, title: str) -> None:
    if group.empty:
        ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title, fontsize=8)
        ax.set_axis_off()
        return

    group = group.sort_values("genomic_start")
    x = group["genomic_start"].astype(float).to_numpy()
    ref = group["ref_value"].astype(float).to_numpy()
    alt = group["alt_value"].astype(float).to_numpy()
    delta = group["delta"].astype(float).to_numpy()

    ax.step(x, ref, where="post", color=REF_COLOR, linewidth=1.0, label="REF")
    ax.step(x, alt, where="post", color=ALT_COLOR, linewidth=1.0, linestyle="--", label="ALT")
    ax.set_title(title, fontsize=8)
    ax.tick_params(axis="both", labelsize=6)
    ax.set_ylabel("raw value", fontsize=7)

    ax2 = ax.twinx()
    ax2.step(x, delta, where="post", color=DELTA_COLOR, linewidth=0.7, alpha=0.55, label="ALT-REF")
    ax2.axhline(0, color="#999999", linewidth=0.5)
    ax2.tick_params(axis="y", labelsize=6)
    ax2.set_ylabel("delta", fontsize=7)


def render_pdf(
    *,
    profiles_path: Path,
    resolved_tracks_path: Path | None,
    output_pdf: Path,
    biosample: str = "MCF-7",
) -> pd.DataFrame:
    profiles = pd.read_parquet(profiles_path)
    profiles = _experimental_profiles(profiles)
    profiles = profiles[profiles["biosample"].astype(str) == biosample].copy()
    if profiles.empty:
        raise SystemExit(f"No profiles found for biosample={biosample!r}")

    track_meta = pd.DataFrame()
    if resolved_tracks_path and resolved_tracks_path.exists():
        track_meta = pd.read_csv(resolved_tracks_path, sep="\t")
        keep = track_meta[track_meta["biosample_resolved"].astype(str) == biosample]
        profiles = profiles[profiles["target_name"].isin(keep["target_name"].astype(str))].copy()

    targets = (
        profiles[["target_name", "track_id"]]
        .drop_duplicates()
        .sort_values(["target_name", "track_id"])
    )
    mutation_ids = [m for m in MUTATION_LABELS if m in set(profiles["mutation_id"])]
    if not mutation_ids:
        mutation_ids = sorted(profiles["mutation_id"].dropna().astype(str).unique())

    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(output_pdf) as pdf:
        for _, target_row in targets.iterrows():
            target = str(target_row["target_name"])
            track_id = str(target_row["track_id"])
            target_profiles = profiles[
                (profiles["target_name"].astype(str) == target)
                & (profiles["track_id"].astype(str) == track_id)
            ]

            fig, axes = plt.subplots(
                len(mutation_ids),
                len(READOUT_COLUMNS),
                figsize=(12, max(3.0, 2.4 * len(mutation_ids))),
                squeeze=False,
            )

            for row_idx, mutation_id in enumerate(mutation_ids):
                mut_profiles = target_profiles[target_profiles["mutation_id"].astype(str) == mutation_id]
                for col_idx, (readout_id, readout_label) in enumerate(READOUT_COLUMNS):
                    ax = axes[row_idx, col_idx]
                    panel = mut_profiles[mut_profiles["readout_id"].astype(str) == readout_id]
                    title = readout_label
                    _plot_panel(ax, panel, title)
                    if col_idx == 0:
                        ax.text(
                            -0.18,
                            0.5,
                            MUTATION_LABELS.get(mutation_id, mutation_id),
                            transform=ax.transAxes,
                            ha="right",
                            va="center",
                            fontsize=8,
                            fontweight="bold",
                        )
                    if row_idx == len(mutation_ids) - 1:
                        ax.set_xlabel("genomic position (bp)", fontsize=7)

            fig.suptitle(
                f"{target} raw 128 bp profile ({biosample})\n{track_id}",
                fontsize=11,
                y=0.995,
            )
            handles, labels = axes[0, 0].get_legend_handles_labels()
            if handles:
                fig.legend(handles, labels, loc="upper right", fontsize=7)
            fig.tight_layout(rect=(0.06, 0.02, 0.98, 0.94))
            pdf.savefig(fig)
            plt.close(fig)

    summary = (
        profiles.groupby(["target_name", "track_id", "biosample", "mutation_id", "readout_id"])
        .agg(
            n_bins=("ref_value", "size"),
            ref_mean=("ref_value", "mean"),
            alt_mean=("alt_value", "mean"),
            delta_mean=("delta", "mean"),
            delta_min=("delta", "min"),
            delta_max=("delta", "max"),
        )
        .reset_index()
        .sort_values(["target_name", "mutation_id", "readout_id"])
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profiles", required=True, help="mreg_chromatin_profiles.parquet")
    parser.add_argument("--resolved-tracks", default=None, help="resolved_tracks.tsv")
    parser.add_argument("--output-pdf", required=True)
    parser.add_argument("--summary-tsv", default=None)
    parser.add_argument("--biosample", default="MCF-7")
    args = parser.parse_args()

    summary = render_pdf(
        profiles_path=Path(args.profiles),
        resolved_tracks_path=Path(args.resolved_tracks) if args.resolved_tracks else None,
        output_pdf=Path(args.output_pdf),
        biosample=args.biosample,
    )
    if args.summary_tsv:
        out = Path(args.summary_tsv)
        out.parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(out, sep="\t", index=False)
    print(f"[plot] wrote {args.output_pdf}")
    if args.summary_tsv:
        print(f"[plot] wrote {args.summary_tsv}")


if __name__ == "__main__":
    main()
