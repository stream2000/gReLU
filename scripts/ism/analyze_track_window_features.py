#!/usr/bin/env python
"""Analyze full-track local-window AlphaGenome delta outputs.

This script reads ``all_track_delta_window.npy`` and track metadata produced by
``run_tf_context.py --feature_mode track_window``. It preserves the all-track
output and derives biological feature summaries only as downstream views.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def _upper_series(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series([""] * len(df), index=df.index)
    return df[col].fillna("").astype(str).str.upper()


def build_feature_groups(metadata: pd.DataFrame) -> dict[str, np.ndarray]:
    tx = _upper_series(metadata, "transcription_factor")
    hist = _upper_series(metadata, "histone_mark")
    output = metadata["output_type"].fillna("").astype(str) if "output_type" in metadata else pd.Series("")
    assay = metadata["assay_title"].fillna("").astype(str).str.lower() if "assay_title" in metadata else pd.Series("")

    groups = {
        "ctcf_binding": (output == "chip_tf") & (tx == "CTCF"),
        "cohesin_rad21_smc3": (output == "chip_tf") & tx.isin(["RAD21", "SMC3"]),
        "pol2_transcription_machinery": (output == "chip_tf")
        & tx.isin(["POLR2A", "POL2", "POL2SER2", "AFF4", "BRD4", "MED1"]),
        "enhancer_histone": (output == "chip_histone") & hist.isin(["H3K27AC", "H3K4ME1"]),
        "promoter_histone": (output == "chip_histone") & hist.isin(["H3K4ME3"]),
        "accessibility": output.isin(["atac", "dnase"]) | assay.isin(["atac-seq", "dnase-seq"]),
        "expression": output.isin(["cage", "rna_seq"]),
        "repressive_histone": (output == "chip_histone") & hist.isin(["H3K27ME3", "H3K9ME3"]),
    }
    return {name: np.flatnonzero(mask.to_numpy()) for name, mask in groups.items()}


def summarize_features(
    delta: np.ndarray,
    sites: pd.DataFrame,
    groups: dict[str, np.ndarray],
) -> pd.DataFrame:
    records: list[dict] = []
    for site_idx, row in sites.reset_index(drop=True).iterrows():
        rec = {
            "site_id": row["site_id"],
            "dic_class": row.get("dic_class", ""),
            "control_type": row.get("control_type", ""),
            "paired_site_id": row.get("paired_site_id", ""),
            "peak_score": row.get("peak_score", np.nan),
            "motif_score": row.get("motif_score", np.nan),
        }
        site_delta = np.asarray(delta[site_idx])
        for group_name, track_indices in groups.items():
            if len(track_indices) == 0:
                continue
            group_delta = site_delta[track_indices]
            rec[f"{group_name}__delta_abs_mean"] = float(np.mean(np.abs(group_delta)))
            rec[f"{group_name}__delta_abs_max"] = float(np.max(np.abs(group_delta)))
            rec[f"{group_name}__delta_l2"] = float(np.linalg.norm(group_delta))
        records.append(rec)
    return pd.DataFrame.from_records(records)


def top_tracks(delta: np.ndarray, sites: pd.DataFrame, metadata: pd.DataFrame, n: int) -> pd.DataFrame:
    rows: list[dict] = []
    for col in [
        "output_type",
        "track_index",
        "global_track_index",
        "transcription_factor",
        "histone_mark",
        "biosample_name",
        "assay_title",
        "track_name",
    ]:
        if col not in metadata.columns:
            metadata[col] = ""

    for site_idx, row in sites.reset_index(drop=True).iterrows():
        score = np.mean(np.abs(np.asarray(delta[site_idx])), axis=1)
        top = np.argsort(score)[-n:][::-1]
        for rank, track_idx in enumerate(top, start=1):
            meta = metadata.iloc[int(track_idx)]
            target = meta.get("transcription_factor")
            if pd.isna(target) or target == "":
                target = meta.get("histone_mark")
            if pd.isna(target) or target == "":
                target = meta.get("track_name")
            rows.append(
                {
                    "site_id": row["site_id"],
                    "dic_class": row.get("dic_class", ""),
                    "control_type": row.get("control_type", ""),
                    "rank": rank,
                    "global_track_index": int(track_idx),
                    "score_abs_mean": float(score[track_idx]),
                    "output_type": meta.get("output_type", ""),
                    "target": target,
                    "biosample_name": meta.get("biosample_name", ""),
                    "assay_title": meta.get("assay_title", ""),
                    "track_name": meta.get("track_name", ""),
                }
            )
    return pd.DataFrame.from_records(rows)


def motif_vs_control_delta(features: pd.DataFrame) -> pd.DataFrame:
    if "paired_site_id" not in features.columns or "control_type" not in features.columns:
        return pd.DataFrame()
    feature_cols = [c for c in features.columns if c.endswith("__delta_abs_mean")]
    records: list[dict] = []
    for pair_id, group in features.groupby("paired_site_id"):
        motif = group[group["control_type"] == "ctcf_motif_max_ic_disruption"]
        control = group[group["control_type"] == "same_peak_non_motif_nearby_base"]
        if motif.empty or control.empty:
            continue
        m = motif.iloc[0]
        c = control.iloc[0]
        rec = {
            "paired_site_id": pair_id,
            "dic_class": m.get("dic_class", ""),
            "motif_site_id": m["site_id"],
            "control_site_id": c["site_id"],
        }
        for col in feature_cols:
            rec[col + "__motif_minus_control"] = float(m[col] - c[col])
        records.append(rec)
    return pd.DataFrame.from_records(records)


def perturbations_vs_control_delta(
    features: pd.DataFrame,
    baseline_control_type: str = "same_peak_non_motif_nearby_base",
) -> pd.DataFrame:
    if "paired_site_id" not in features.columns or "control_type" not in features.columns:
        return pd.DataFrame()
    feature_cols = [c for c in features.columns if c.endswith("__delta_abs_mean")]
    records: list[dict] = []
    for pair_id, group in features.groupby("paired_site_id"):
        control = group[group["control_type"] == baseline_control_type]
        if control.empty:
            continue
        c = control.iloc[0]
        for _, p in group[group["control_type"] != baseline_control_type].iterrows():
            rec = {
                "paired_site_id": pair_id,
                "dic_class": p.get("dic_class", ""),
                "perturbation_type": p["control_type"],
                "perturbation_site_id": p["site_id"],
                "control_site_id": c["site_id"],
                "baseline_control_type": baseline_control_type,
            }
            for col in feature_cols:
                rec[col + "__perturbation_minus_control"] = float(p[col] - c[col])
            records.append(rec)
    return pd.DataFrame.from_records(records)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--sites", required=True, help="Input site/variant table used for the run")
    parser.add_argument("--top_n", type=int, default=20)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    delta = np.load(run_dir / "all_track_delta_window.npy", mmap_mode="r")
    metadata = pd.read_csv(run_dir / "all_track_metadata.tsv", sep="\t", low_memory=False)
    sites = pd.read_csv(args.sites, sep="\t")
    if len(sites) != delta.shape[0]:
        variants_path = run_dir / "variants.tsv"
        if variants_path.exists():
            variants = pd.read_csv(variants_path, sep="\t")
            sites = variants[["site_id"]].merge(sites, on="site_id", how="left")
        if len(sites) != delta.shape[0]:
            raise ValueError(f"Site rows {len(sites)} do not match delta rows {delta.shape[0]}")

    groups = build_feature_groups(metadata)
    features = summarize_features(delta, sites, groups)
    features.to_csv(run_dir / "analysis_biological_feature_summary.tsv", sep="\t", index=False)

    top = top_tracks(delta, sites, metadata, n=args.top_n)
    top.to_csv(run_dir / f"analysis_top{args.top_n}_tracks_by_site.tsv", sep="\t", index=False)

    pair_delta = motif_vs_control_delta(features)
    if not pair_delta.empty:
        pair_delta.to_csv(run_dir / "analysis_motif_minus_control_feature_delta.tsv", sep="\t", index=False)
    all_pair_delta = perturbations_vs_control_delta(features)
    if not all_pair_delta.empty:
        all_pair_delta.to_csv(
            run_dir / "analysis_perturbation_minus_control_feature_delta.tsv",
            sep="\t",
            index=False,
        )

    class_summary = features.groupby(["dic_class", "control_type"], dropna=False).mean(numeric_only=True)
    class_summary.to_csv(run_dir / "analysis_biological_feature_class_summary.tsv", sep="\t")

    print(f"delta shape: {delta.shape}")
    print(f"wrote: {run_dir / 'analysis_biological_feature_summary.tsv'}")
    print(f"wrote: {run_dir / f'analysis_top{args.top_n}_tracks_by_site.tsv'}")
    if not pair_delta.empty:
        print(f"wrote: {run_dir / 'analysis_motif_minus_control_feature_delta.tsv'}")
    if not all_pair_delta.empty:
        print(f"wrote: {run_dir / 'analysis_perturbation_minus_control_feature_delta.tsv'}")
    print("\nClass/control summary:")
    print(class_summary[[c for c in class_summary.columns if c.endswith('__delta_abs_mean')]].to_string())


if __name__ == "__main__":
    main()
