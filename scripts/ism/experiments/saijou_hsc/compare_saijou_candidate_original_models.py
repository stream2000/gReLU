#!/usr/bin/env python
"""Compare selected fine-tuned hotspots with original model track panels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_ROOT = REPO_ROOT / "experiments/ism/saijou_all_genes_10bp_scan"
RUNS = {
    "AlphaGenome fine-tuned": "runs/alphagenome_finetuned",
    "Borzoi fine-tuned": "runs/borzoi_finetuned",
    "AlphaGenome original": "runs/alphagenome_original",
    "Borzoi original": "runs/borzoi_original",
}
FINE_MODELS = {"AlphaGenome fine-tuned", "Borzoi fine-tuned"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    return parser.parse_args()


def read_selected_features(root: Path, label: str, relpath: str) -> pd.DataFrame:
    data = pd.read_csv(root / relpath / "features/combined_mutation_features.tsv", sep="\t")
    selected = pd.read_csv(root / "prepared_original_candidates/mutation_manifest.tsv", sep="\t")
    mapping = selected[["mutation_id", "locus_id", "locus_role"]].drop_duplicates()
    if label in FINE_MODELS:
        data = data.loc[data.mutation_id.isin(mapping.mutation_id)].drop(
            columns=["locus_id", "locus_role"]
        )
        data = data.merge(mapping, on="mutation_id", how="inner", validate="many_to_one")
    data["model_label"] = label
    genes = pd.read_csv(root / "prepared/genes.tsv", sep="\t")
    strand = genes.set_index("gene").strand.to_dict()
    data["gene_strand"] = data.gene.map(strand)
    data["track_strand"] = data.track_strand.fillna(".").astype(str)
    data = data.loc[
        data.track_strand.eq(".") | data.track_strand.eq(data.gene_strand)
    ].copy()
    return data


def mutation_group_effects(data: pd.DataFrame) -> pd.DataFrame:
    keys = [
        "model_label",
        "model_backend",
        "gene",
        "locus_id",
        "mutation_id",
        "variant_offset_from_tss_transcription_bp",
        "track_group",
        "readout_role",
    ]
    return (
        data.groupby(keys, sort=False)
        .agg(
            track_count=("track_id", "nunique"),
            median_effect=("log2fc_ratio_of_sums", "median"),
            median_abs_bin_effect=("absolute_log2fc_mean", "median"),
            median_delta_sum=("signed_delta_sum", "median"),
            median_ref_sum=("ref_sum", "median"),
        )
        .reset_index()
    )


def segment_summary(grouped: pd.DataFrame) -> pd.DataFrame:
    keys = [
        "model_label",
        "model_backend",
        "gene",
        "locus_id",
        "track_group",
        "readout_role",
    ]
    return (
        grouped.groupby(keys, sort=False)
        .agg(
            mutations=("mutation_id", "nunique"),
            centers=("variant_offset_from_tss_transcription_bp", "nunique"),
            median_tracks=("track_count", "median"),
            median_effect=("median_effect", "median"),
            median_abs_effect=("median_effect", lambda x: float(np.median(np.abs(x)))),
            mean_abs_effect=("median_effect", lambda x: float(np.mean(np.abs(x)))),
            max_abs_effect=("median_effect", lambda x: float(np.max(np.abs(x)))),
            fraction_negative=("median_effect", lambda x: float((x < 0).mean())),
            median_ref_sum=("median_ref_sum", "median"),
        )
        .reset_index()
    )


def fine_specificity(summary: pd.DataFrame) -> pd.DataFrame:
    fine = summary.loc[summary.model_label.isin(FINE_MODELS)].copy()
    fine["cell_type"] = fine.track_group.str.replace("_finetuned_10x", "", regex=False)
    fine["cell_rank"] = fine.groupby(
        ["model_label", "locus_id", "readout_role"]
    ).median_abs_effect.rank(method="min", ascending=False)
    rows = []
    for key, group in fine.groupby(["model_label", "gene", "locus_id", "readout_role"]):
        hsc = group.loc[group.cell_type.eq("hsc")]
        if hsc.empty:
            continue
        row = hsc.iloc[0]
        other = group.loc[~group.cell_type.eq("hsc")]
        rows.append(
            {
                "model_label": key[0],
                "gene": key[1],
                "locus_id": key[2],
                "readout_role": key[3],
                "hsc_median_abs_effect": row.median_abs_effect,
                "hsc_median_effect": row.median_effect,
                "hsc_rank": int(row.cell_rank),
                "top_cell": group.nlargest(1, "median_abs_effect").cell_type.iloc[0],
                "max_other_abs_effect": other.median_abs_effect.max(),
                "hsc_vs_max_other_ratio": row.median_abs_effect
                / max(float(other.median_abs_effect.max()), 1e-12),
            }
        )
    return pd.DataFrame.from_records(rows)


def original_evidence(summary: pd.DataFrame) -> pd.DataFrame:
    original = summary.loc[~summary.model_label.isin(FINE_MODELS)].copy()
    priority_groups = [
        "liver_rna",
        "fibroblast_rna",
        "liver_active_chromatin",
        "fibroblast_active_chromatin",
        "liver_accessibility",
        "fibroblast_accessibility",
        "liver_cage",
        "hsc_cage",
        "mesenchymal_cage",
        "smooth_muscle_cage",
        "liver_tf",
        "fibroblast_tf",
    ]
    original = original.loc[original.track_group.isin(priority_groups)]
    original["rank_within_model_locus_readout"] = original.groupby(
        ["model_label", "locus_id", "readout_role"]
    ).median_abs_effect.rank(method="min", ascending=False)
    return original.sort_values(
        ["gene", "locus_id", "model_label", "readout_role", "median_abs_effect"],
        ascending=[True, True, True, True, False],
    )


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    out = root / "analysis/original_comparison"
    out.mkdir(parents=True, exist_ok=True)
    data = pd.concat(
        [read_selected_features(root, label, relpath) for label, relpath in RUNS.items()],
        ignore_index=True,
    )
    grouped = mutation_group_effects(data)
    summary = segment_summary(grouped)
    fine = fine_specificity(summary)
    original = original_evidence(summary)
    grouped.to_csv(out / "mutation_group_effects.tsv", sep="\t", index=False)
    summary.to_csv(out / "segment_track_group_summary.tsv", sep="\t", index=False)
    fine.to_csv(out / "fine_tuned_cell_specificity.tsv", sep="\t", index=False)
    original.to_csv(out / "original_track_evidence.tsv", sep="\t", index=False)
    validation = {
        "status": "ok",
        "models": sorted(data.model_label.unique().tolist()),
        "genes": sorted(data.gene.unique().tolist()),
        "segments": int(data.locus_id.nunique()),
        "feature_rows": int(len(data)),
        "mutation_group_rows": int(len(grouped)),
        "summary_rows": int(len(summary)),
        "nonfinite_summary_values": int(
            (~np.isfinite(summary[["median_effect", "median_abs_effect"]].to_numpy())).sum()
        ),
    }
    if validation["nonfinite_summary_values"]:
        raise RuntimeError(validation)
    (out / "validation_summary.json").write_text(json.dumps(validation, indent=2) + "\n")
    print(json.dumps(validation, indent=2))


if __name__ == "__main__":
    main()
