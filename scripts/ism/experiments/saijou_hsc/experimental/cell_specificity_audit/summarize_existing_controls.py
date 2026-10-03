#!/usr/bin/env python
"""Nine-locus window-mean sensitivity analysis from saved control metrics.

This is explicitly not a bin-level or full-validation substitute. Coordinate
columns in the source CSVs are neither retained nor used.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[5]
sys.path.insert(0, str(HERE))

from metrics import (  # noqa: E402
    centered_l2_over_total,
    cluster_bootstrap_mean_ci,
    normalized_entropy_specificity,
    top_identity_metrics,
    top_second_log2_margin,
)


DEFAULT_INPUTS = {
    "borzoi_lora_e39": REPO_ROOT / "experiments/validation/borzoi_lora_poisson_multinomial_epoch39_controls_raw_cpm_right_shared_y/per_track_metrics.csv",
    "alphagenome_lora_e19": REPO_ROOT / "experiments/validation/alphagenome_lora_active_aligned128_epoch19_controls_raw_cpm_right_shared_y/per_track_metrics.csv",
}
TASKS = ["hsc", "mac", "lsec", "chol"]


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "experiments/validation/cell_specificity_audit/existing_controls_sensitivity")
    parser.add_argument("--seed", type=int, default=20260824)
    parser.add_argument("--bootstrap", type=int, default=10000)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows, summaries, sources = [], [], {}
    for model, path in DEFAULT_INPUTS.items():
        frame = pd.read_csv(path)
        if frame.duplicated(["gene", "track"]).any():
            raise RuntimeError(f"duplicate gene/track keys: {path}")
        if sorted(frame.track.unique()) != sorted(TASKS):
            raise RuntimeError(f"task mismatch: {path}")
        observed = frame.pivot(index="gene", columns="track", values="observed_mean_full_label_window").reindex(columns=TASKS)
        predicted = frame.pivot(index="gene", columns="track", values="predicted_mean_full_label_window").reindex(columns=TASKS)
        if not observed.index.equals(predicted.index) or not np.isfinite(observed.to_numpy()).all() or not np.isfinite(predicted.to_numpy()).all():
            raise RuntimeError(f"shape/key/finiteness failure: {path}")
        obs, pred = observed.to_numpy(), predicted.to_numpy()
        positive = obs[obs > 0]
        eps = float(np.quantile(positive, 0.01))
        metrics = {
            "entropy": (normalized_entropy_specificity(obs), normalized_entropy_specificity(pred)),
            "centered_l2": (centered_l2_over_total(obs), centered_l2_over_total(pred)),
            "log2_margin": (top_second_log2_margin(obs, eps), top_second_log2_margin(pred, eps)),
        }
        identity = top_identity_metrics(obs, pred)
        for gene_idx, gene in enumerate(observed.index):
            row = {"model": model, "gene": gene, "observed_top": TASKS[int(np.argmax(obs[gene_idx]))], "predicted_top": TASKS[int(np.argmax(pred[gene_idx]))]}
            for metric, (obs_m, pred_m) in metrics.items():
                row[f"observed_{metric}"] = obs_m[gene_idx]
                row[f"predicted_{metric}"] = pred_m[gene_idx]
                row[f"gap_{metric}"] = pred_m[gene_idx] - obs_m[gene_idx]
            rows.append(row)
        for metric, (obs_m, pred_m) in metrics.items():
            gap_ci = cluster_bootstrap_mean_ci(
                pred_m - obs_m,
                np.arange(len(obs_m)),
                seed=args.seed,
                n_boot=args.bootstrap,
            )
            summaries.append({
                "model": model, "metric": metric, "n_genes": len(obs),
                "observed_mean": float(obs_m.mean()), "predicted_mean": float(pred_m.mean()),
                "gap_mean": gap_ci["mean"], "gap_ci_low": gap_ci["ci_low"],
                "gap_ci_high": gap_ci["ci_high"],
                "prediction_observation_ratio": float(pred_m.mean() / obs_m.mean()) if obs_m.mean() > 0 else np.nan,
                "top_accuracy": identity["accuracy"], "top_macro_recall": identity["macro_recall"],
            })
        sources[model] = {"path": str(path.relative_to(REPO_ROOT)), "sha256": sha256_file(path), "n_rows": len(frame), "n_genes": len(observed), "margin_eps_source": "1% quantile of positive observed window means in this saved sensitivity dataset", "margin_eps": eps}
    metrics_frame = pd.DataFrame(rows)
    summary_frame = pd.DataFrame(summaries)
    metrics_frame.to_csv(args.out_dir / "control_window_mean_metrics.tsv", sep="\t", index=False)
    summary_frame.to_csv(args.out_dir / "control_window_mean_summary.tsv", sep="\t", index=False)

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    positions = np.arange(len(summary_frame))
    errors = np.vstack(
        [summary_frame.gap_mean - summary_frame.gap_ci_low,
         summary_frame.gap_ci_high - summary_frame.gap_mean]
    )
    colors = np.where(summary_frame.model.eq("borzoi_lora_e39"), "#3B6FB6", "#D18F00")
    for position, row, error, color in zip(positions, summary_frame.itertuples(), errors.T, colors):
        ax.errorbar(position, row.gap_mean, yerr=error[:, None], fmt="o", color=color, ecolor="#4A4A4A", capsize=4)
    ax.axhline(0, color="#333333", linewidth=1)
    ax.set_xticks(positions, [f"{row.model.replace('_lora_', ' ')}\n{row.metric}" for row in summary_frame.itertuples()], rotation=25, ha="right")
    ax.set_ylabel("Predicted - observed specificity")
    ax.set_title("Nine-control window-mean specificity gaps", pad=32)
    ax.text(0, 1.025, "N=9 selected genes; paired-gene bootstrap 95% CIs", transform=ax.transAxes, color="#555555")
    ax.grid(axis="y", color="#E5E5E5")
    fig.tight_layout()
    fig.savefig(args.out_dir / "control_window_mean_specificity_gaps.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharex=True, sharey=True)
    for ax, (model, group) in zip(axes, metrics_frame.groupby("model", sort=False)):
        ax.scatter(group.observed_entropy, group.predicted_entropy, color="#7A5195", edgecolor="#333333", s=45)
        limit = float(max(group.observed_entropy.max(), group.predicted_entropy.max()))
        ax.plot([0, limit], [0, limit], linestyle="--", color="#444444")
        for row in group.itertuples():
            ax.annotate(row.gene, (row.observed_entropy, row.predicted_entropy), xytext=(3, 3), textcoords="offset points", fontsize=7)
        ax.set_title(model.replace("_lora_", " "))
        ax.set_xlabel("Observed entropy specificity")
        ax.grid(color="#E5E5E5")
    axes[0].set_ylabel("Predicted entropy specificity")
    fig.suptitle("Nine-control window-mean specificity calibration\nN=9 selected genes per model; dashed line is ideal")
    fig.tight_layout()
    fig.savefig(args.out_dir / "control_window_mean_entropy_calibration.png", dpi=180)
    plt.close(fig)
    validation = {
        "status": "ok",
        "scope": "nine-locus full-label-window mean sensitivity analysis only; not bin-level and not a full-validation substitute",
        "scientific_boundary": "output specificity only; cannot prove catastrophic forgetting or embedding feature collapse",
        "task_order": TASKS,
        "uncertainty": {
            "method": "paired gene bootstrap of predicted-minus-observed metric",
            "seed": args.seed,
            "replicates": args.bootstrap,
            "warning": "Only nine selected control genes; intervals are sensitivity descriptors, not population-level generalization evidence.",
        },
        "sources": sources,
        "outputs_exclude_genomic_coordinates": True,
        "outputs": [
            "control_window_mean_metrics.tsv",
            "control_window_mean_summary.tsv",
            "control_window_mean_specificity_gaps.png",
            "control_window_mean_entropy_calibration.png",
        ],
        "command": " ".join(sys.argv),
    }
    (args.out_dir / "validation_summary.json").write_text(json.dumps(validation, indent=2) + "\n")
    print(json.dumps({"out_dir": str(args.out_dir), "models": list(DEFAULT_INPUTS)}, indent=2))


if __name__ == "__main__":
    main()
