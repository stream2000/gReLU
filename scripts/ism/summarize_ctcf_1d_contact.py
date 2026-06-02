#!/usr/bin/env python
"""Join 1D CTCF/cohesin deltas with contact-map boundary deltas."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


FEATURE_COLUMNS = [
    "ctcf_binding__delta_abs_mean__motif_minus_control",
    "cohesin_rad21_smc3__delta_abs_mean__motif_minus_control",
    "pol2_transcription_machinery__delta_abs_mean__motif_minus_control",
    "enhancer_histone__delta_abs_mean__motif_minus_control",
    "promoter_histone__delta_abs_mean__motif_minus_control",
    "accessibility__delta_abs_mean__motif_minus_control",
    "expression__delta_abs_mean__motif_minus_control",
    "repressive_histone__delta_abs_mean__motif_minus_control",
]


def _zscore(values: pd.Series) -> pd.Series:
    values = values.astype(float)
    std = values.std(ddof=0)
    if not np.isfinite(std) or std == 0:
        return pd.Series(np.zeros(len(values)), index=values.index)
    return (values - values.mean()) / std


def _load_pair_tables(one_d_dir: Path, contact_dir: Path) -> pd.DataFrame:
    one_d = pd.read_csv(one_d_dir / "analysis_motif_minus_control_feature_delta.tsv", sep="\t")
    contact = pd.read_csv(contact_dir / "contact_boundary_strength_motif_minus_control.tsv", sep="\t")

    contact = contact.rename(
        columns={
            "delta_cross_contact_mean__motif_minus_control": "contact_boundary_weakening__motif_minus_control",
            "delta_boundary_strength_proxy__motif_minus_control": "boundary_strength_proxy__motif_minus_control",
            "delta_abs_cross_contact_mean__motif_minus_control": "contact_abs_change__motif_minus_control",
        }
    )
    keep_contact = [
        "paired_site_id",
        "contact_boundary_weakening__motif_minus_control",
        "boundary_strength_proxy__motif_minus_control",
        "contact_abs_change__motif_minus_control",
    ]
    merged = one_d.merge(contact[keep_contact], on="paired_site_id", how="inner")

    for col in FEATURE_COLUMNS:
        if col not in merged.columns:
            merged[col] = np.nan

    merged["ctcf_z"] = _zscore(merged["ctcf_binding__delta_abs_mean__motif_minus_control"])
    merged["cohesin_z"] = _zscore(merged["cohesin_rad21_smc3__delta_abs_mean__motif_minus_control"])
    merged["contact_z"] = _zscore(merged["contact_boundary_weakening__motif_minus_control"])
    merged["paper_aligned_combined_score"] = merged["contact_z"] + merged["ctcf_z"] + merged["cohesin_z"]
    merged["contact_rank"] = merged["contact_boundary_weakening__motif_minus_control"].rank(
        ascending=False, method="min"
    )
    merged["ctcf_rank"] = merged["ctcf_binding__delta_abs_mean__motif_minus_control"].rank(
        ascending=False, method="min"
    )
    merged["cohesin_rank"] = merged["cohesin_rad21_smc3__delta_abs_mean__motif_minus_control"].rank(
        ascending=False, method="min"
    )
    merged["combined_rank"] = merged["paper_aligned_combined_score"].rank(ascending=False, method="min")

    contact_median = merged["contact_boundary_weakening__motif_minus_control"].median()
    ctcf_median = merged["ctcf_binding__delta_abs_mean__motif_minus_control"].median()
    cohesin_median = merged["cohesin_rad21_smc3__delta_abs_mean__motif_minus_control"].median()
    merged["exploratory_signature"] = np.select(
        [
            (merged["contact_boundary_weakening__motif_minus_control"] >= contact_median)
            & (merged["ctcf_binding__delta_abs_mean__motif_minus_control"] >= ctcf_median)
            & (merged["cohesin_rad21_smc3__delta_abs_mean__motif_minus_control"] >= cohesin_median),
            (merged["contact_boundary_weakening__motif_minus_control"] >= contact_median)
            & (merged["ctcf_binding__delta_abs_mean__motif_minus_control"] < ctcf_median),
            (merged["contact_boundary_weakening__motif_minus_control"] < contact_median)
            & (merged["ctcf_binding__delta_abs_mean__motif_minus_control"] >= ctcf_median),
        ],
        [
            "high_contact_high_ctcf_high_cohesin",
            "contact_sensitive_without_high_1d_ctcf",
            "1d_ctcf_sensitive_without_high_contact",
        ],
        default="lower_or_mixed_response",
    )
    return merged.sort_values("combined_rank").reset_index(drop=True)


def _write_report(combined: pd.DataFrame, out_path: Path) -> None:
    class_summary = (
        combined.groupby("dic_class", dropna=False)
        .agg(
            n_sites=("paired_site_id", "count"),
            contact_boundary_weakening_mean=(
                "contact_boundary_weakening__motif_minus_control",
                "mean",
            ),
            ctcf_1d_delta_mean=("ctcf_binding__delta_abs_mean__motif_minus_control", "mean"),
            cohesin_1d_delta_mean=("cohesin_rad21_smc3__delta_abs_mean__motif_minus_control", "mean"),
            combined_score_mean=("paper_aligned_combined_score", "mean"),
        )
        .sort_values("contact_boundary_weakening_mean", ascending=False)
        .reset_index()
    )

    top_cols = [
        "paired_site_id",
        "dic_class",
        "contact_boundary_weakening__motif_minus_control",
        "ctcf_binding__delta_abs_mean__motif_minus_control",
        "cohesin_rad21_smc3__delta_abs_mean__motif_minus_control",
        "paper_aligned_combined_score",
        "exploratory_signature",
    ]
    top = combined.sort_values("contact_boundary_weakening__motif_minus_control", ascending=False).head(12)
    combined_top = combined.sort_values("paper_aligned_combined_score", ascending=False).head(12)

    def as_markdown_table(df: pd.DataFrame) -> str:
        view = df.copy()
        for col in view.columns:
            if pd.api.types.is_float_dtype(view[col]):
                view[col] = view[col].map(lambda x: "" if pd.isna(x) else f"{x:.6g}")
        headers = [str(col) for col in view.columns]
        rows = view.astype(str).values.tolist()
        lines = [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join(["---"] * len(headers)) + " |",
        ]
        lines.extend("| " + " | ".join(row) + " |" for row in rows)
        return "\n".join(lines)

    lines = [
        "# CTCF 1D + Contact Boundary Exploratory Summary",
        "",
        "Interpretation: `contact_boundary_weakening__motif_minus_control` is alt-minus-ref cross-boundary contact change after subtracting the nearby-control change. Larger positive values mean the CTCF motif mutation increases cross-boundary contact, consistent with weaker insulation. This is a local mutation proxy, not a global siCTCF/siRAD21/siNIPBL perturbation.",
        "",
        "## Class Means",
        "",
        as_markdown_table(class_summary),
        "",
        "## Top Contact-Weakening Sites",
        "",
        as_markdown_table(top[top_cols]),
        "",
        "## Top Combined 1D+Contact Sites",
        "",
        as_markdown_table(combined_top[top_cols]),
        "",
    ]
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--one_d_dir", required=True)
    parser.add_argument("--contact_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    one_d_dir = Path(args.one_d_dir)
    contact_dir = Path(args.contact_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    combined = _load_pair_tables(one_d_dir, contact_dir)
    combined.to_csv(output_dir / "ctcf_1d_contact_pair_summary.tsv", sep="\t", index=False)
    (
        combined.groupby(["dic_class", "exploratory_signature"], dropna=False)
        .size()
        .rename("n_sites")
        .reset_index()
        .to_csv(output_dir / "ctcf_1d_contact_signature_counts.tsv", sep="\t", index=False)
    )
    _write_report(combined, output_dir / "ctcf_1d_contact_exploratory_summary.md")
    print(f"wrote: {output_dir / 'ctcf_1d_contact_pair_summary.tsv'}")
    print(f"wrote: {output_dir / 'ctcf_1d_contact_signature_counts.tsv'}")
    print(f"wrote: {output_dir / 'ctcf_1d_contact_exploratory_summary.md'}")


if __name__ == "__main__":
    main()
