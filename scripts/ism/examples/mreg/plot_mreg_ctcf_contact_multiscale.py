#!/usr/bin/env python
"""Plot MREG CTCF contact-map multiscale summaries."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _mean_delta_window(delta_maps: np.ndarray, bins: pd.DataFrame, half_window_bp: int) -> tuple[np.ndarray, np.ndarray]:
    keep = bins["offset_bp"].abs().to_numpy() <= half_window_bp
    offsets = bins.loc[keep, "offset_bp"].to_numpy() / 1000.0
    mat = delta_maps[:, keep, :][:, :, keep].mean(axis=0)
    return offsets, mat


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_dir", required=True)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    fig_dir = run_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    delta_maps = np.load(run_dir / "contact_delta_maps.npy")
    bins = pd.read_csv(run_dir / "contact_bins.tsv", sep="\t")
    site = pd.read_csv(run_dir / "contact_multiscale_site_summary.tsv", sep="\t")

    fig, axes = plt.subplots(2, 2, figsize=(13, 10), constrained_layout=True)

    offsets_100, mat_100 = _mean_delta_window(delta_maps, bins, 100_000)
    vmax = float(np.nanpercentile(np.abs(mat_100), 98))
    im = axes[0, 0].imshow(
        mat_100,
        origin="lower",
        cmap="coolwarm",
        vmin=-vmax,
        vmax=vmax,
        extent=[offsets_100.min(), offsets_100.max(), offsets_100.min(), offsets_100.max()],
        interpolation="nearest",
    )
    axes[0, 0].axhline(0, color="black", lw=0.7)
    axes[0, 0].axvline(0, color="black", lw=0.7)
    axes[0, 0].set_title("Mean ALT-REF contact map, +/-100 kb")
    axes[0, 0].set_xlabel("Offset from motif (kb)")
    axes[0, 0].set_ylabel("Offset from motif (kb)")
    fig.colorbar(im, ax=axes[0, 0], label="ALT-REF")

    multiscale = site[site["metric"] == "motif_cross_boundary_multiscale"].copy()
    for min_dist, group in multiscale.groupby("min_distance_bp"):
        group = group.sort_values("window_bp")
        axes[0, 1].plot(
            group["window_bp"] / 1000.0,
            group["delta_mean"],
            marker="o",
            label=f"min dist {int(min_dist/1000)} kb",
        )
    axes[0, 1].axhline(0, color="black", lw=0.8)
    axes[0, 1].set_xscale("log")
    axes[0, 1].set_title("Cross-motif contact effect by mask size")
    axes[0, 1].set_xlabel("Half-window around motif (kb)")
    axes[0, 1].set_ylabel("Mean ALT-REF contact")
    axes[0, 1].legend(frameon=False)

    gene = site[site["metric"].str.startswith(("gene_body", "motif_anchor"))].copy()
    labels = [
        "left vs right\nwithin gene",
        "motif to\ngene body",
        "motif to\nleft gene",
        "motif to\nright gene",
    ]
    label_map = {
        "gene_body_left_vs_right_around_motif": labels[0],
        "motif_anchor_to_gene_body": labels[1],
        "motif_anchor_to_left_gene_body": labels[2],
        "motif_anchor_to_right_gene_body": labels[3],
    }
    gene["plot_label"] = gene["metric"].map(label_map)
    gene = gene.set_index("plot_label").loc[labels].reset_index()
    colors = ["#3b6ea8" if x >= 0 else "#b24a4a" for x in gene["delta_mean"]]
    axes[1, 0].bar(gene["plot_label"], gene["delta_mean"], color=colors)
    axes[1, 0].axhline(0, color="black", lw=0.8)
    axes[1, 0].set_title("Gene-aware MREG contact masks")
    axes[1, 0].set_ylabel("Mean ALT-REF contact")

    offsets_500, mat_500 = _mean_delta_window(delta_maps, bins, 500_000)
    center_idx = int(np.argmin(np.abs(offsets_500)))
    stripe = mat_500[center_idx, :]
    axes[1, 1].plot(offsets_500, stripe, color="#2f5f8f", lw=1.5)
    axes[1, 1].axhline(0, color="black", lw=0.8)
    axes[1, 1].axvline(0, color="black", lw=0.8)
    axes[1, 1].set_title("Motif-anchor stripe across full 1 Mb context")
    axes[1, 1].set_xlabel("Offset from motif (kb)")
    axes[1, 1].set_ylabel("Mean ALT-REF contact")

    fig.suptitle("MREG intragenic CTCF SNV: 2D contact-map effect", fontsize=15)
    fig.savefig(fig_dir / "mreg_ctcf_contact_multiscale_overview.png", dpi=220)
    fig.savefig(fig_dir / "mreg_ctcf_contact_multiscale_overview.svg")

    with (fig_dir / "README.md").open("w") as handle:
        handle.write(
            "Figure source: AlphaGenome contact maps from the multiscale MREG CTCF run.\n"
            "Contact-map resolution is 2048 bp/bin; 1 kb center masking is below 2D output resolution.\n"
        )

    print(f"wrote: {fig_dir / 'mreg_ctcf_contact_multiscale_overview.png'}")
    print(f"wrote: {fig_dir / 'mreg_ctcf_contact_multiscale_overview.svg'}")


if __name__ == "__main__":
    main()
