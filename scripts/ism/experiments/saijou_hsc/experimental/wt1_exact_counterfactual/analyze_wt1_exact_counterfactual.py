#!/usr/bin/env python
"""Analyze exact-site WT1 counterfactuals across model tracks and readouts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import statsmodels.api as sm
from scipy.stats import spearmanr


REPO = Path(__file__).resolve().parents[6]
DEFAULT_ROOT = REPO / "experiments/ism/20260918_1349_wt1_exact_counterfactual"
BACKENDS = (
    "alphagenome_original",
    "alphagenome_finetuned",
    "borzoi_original",
    "borzoi_finetuned",
)
OUTPUT_MODALITIES = {"rna", "rna_seq", "cage", "10x_scRNAseq_3prime_pseudobulk"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    return parser.parse_args()


def read_backend(root: Path, backend: str) -> pd.DataFrame:
    run = root / "runs" / backend
    validation = json.loads((run / "validation_summary.json").read_text())
    if validation.get("status") != "ok":
        raise ValueError(f"{backend}: run validation is not ok")
    path = run / "features" / "combined_mutation_features.parquet"
    columns = [
        "model_backend", "mutation_id", "locus_id", "track_id", "track_group",
        "track_task_name", "track_cell_type", "track_modality", "track_strand",
        "readout_id", "readout_role", "readout_locus_id", "n_bins", "ref_sum",
        "alt_sum", "log2fc_ratio_of_sums", "absolute_delta_mean",
    ]
    table = pq.read_table(path, columns=columns).to_pandas()
    table["backend"] = backend

    manifest = pd.read_csv(run / "track_manifest.tsv", sep="\t")
    optional = [
        column for column in (
            "track_id", "biosample_name", "biosample_life_stage", "description",
            "sample", "assay", "histone_mark", "transcription_factor"
        ) if column in manifest.columns
    ]
    metadata = manifest[optional].drop_duplicates("track_id")
    table = table.merge(metadata, on="track_id", how="left", validate="many_to_one")
    return table


def classify_stage(row: pd.Series) -> str:
    explicit = str(row.get("biosample_life_stage", "")).strip().lower()
    if explicit and explicit != "nan":
        return explicit
    text = " ".join(
        str(row.get(column, ""))
        for column in ("track_task_name", "track_cell_type", "description", "sample")
    ).lower()
    if "embry" in text or "fetal" in text:
        return "embryonic"
    if "postnatal" in text:
        return "postnatal"
    if "adult" in text or "week" in text or "pregnant" in text:
        return "adult"
    return "unspecified"


def eligible_rows(table: pd.DataFrame) -> pd.DataFrame:
    local_ok = table.readout_role.ne("local_edit") | table.readout_locus_id.eq(table.locus_id)
    table = table[local_ok].copy()
    table["sample_stage"] = table.apply(classify_stage, axis=1)
    table["signed_effect"] = table.log2fc_ratio_of_sums.astype(float)
    table["absolute_effect"] = table.signed_effect.abs()
    return table


def build_panels(table: pd.DataFrame) -> pd.DataFrame:
    original = table.backend.str.endswith("_original")
    fine = ~original
    modality = table.track_modality.fillna("")
    output = modality.isin(OUTPUT_MODALITIES)
    output_geometry = (
        modality.isin({"rna", "rna_seq", "10x_scRNAseq_3prime_pseudobulk"})
        & table.readout_role.isin(
            {"gene_body", "gene_body_output_clipped", "tes_3prime", "tes_3prime_1024bp"}
        )
    ) | (modality.eq("cage") & table.readout_role.isin({"tss", "tss_1024bp"}))
    regulatory_geometry = ~output & table.readout_role.eq("local_edit")
    selected = table[(original & (output_geometry | regulatory_geometry)) | (fine & output_geometry)].copy()

    selected["strand_class"] = selected.track_strand.fillna(".").replace({".": "unstranded", "-": "minus", "+": "plus"})
    selected.loc[fine.reindex(selected.index, fill_value=False), "strand_class"] = "model_head"
    selected.loc[fine.reindex(selected.index, fill_value=False), "sample_stage"] = "saijou"
    selected["panel_id"] = (
        selected.track_group.fillna(selected.track_id)
        + "__" + selected.track_modality.fillna("unknown")
        + "__" + selected.sample_stage
        + "__" + selected.strand_class
        + "__" + selected.readout_role
    )
    keys = ["backend", "mutation_id", "locus_id", "panel_id", "track_group", "track_modality", "sample_stage", "strand_class", "readout_role"]
    panels = (
        selected.groupby(keys, sort=False, dropna=False)
        .agg(
            tracks=("track_id", "nunique"),
            signed_effect=("signed_effect", "median"),
            absolute_effect=("absolute_effect", "median"),
            median_ref_sum=("ref_sum", "median"),
        )
        .reset_index()
    )
    return panels


def add_finetuned_cell_contrasts(panels: pd.DataFrame) -> pd.DataFrame:
    fine = panels[
        panels.backend.str.endswith("_finetuned")
        & panels.track_group.isin({"hsc_finetuned_10x", "mac_finetuned_10x", "lsec_finetuned_10x", "chol_finetuned_10x"})
    ].copy()
    if fine.empty:
        return panels
    fine["cell"] = fine.track_group.str.replace("_finetuned_10x", "", regex=False)
    wide = fine.pivot_table(
        index=["backend", "mutation_id", "locus_id", "readout_role"],
        columns="cell", values="signed_effect", aggfunc="first"
    ).reset_index()
    required = {"hsc", "mac", "lsec", "chol"}
    if not required <= set(wide.columns):
        raise ValueError("Fine-tuned cell panel is incomplete")
    wide["signed_effect"] = wide[["mac", "lsec", "chol"]].median(axis=1) - wide.hsc
    wide["absolute_effect"] = wide.signed_effect.abs()
    wide["panel_id"] = "hsc_loss_specificity__" + wide.readout_role
    wide["track_group"] = "hsc_vs_other_signed"
    wide["track_modality"] = "derived_cell_contrast"
    wide["sample_stage"] = "saijou"
    wide["strand_class"] = "derived"
    wide["tracks"] = 4
    wide["median_ref_sum"] = np.nan
    return pd.concat([panels, wide[panels.columns]], ignore_index=True)


def original_track_case_rows(table: pd.DataFrame) -> pd.DataFrame:
    """Summarize each original-model track under its biologically relevant readout."""

    original = table[table.backend.str.endswith("_original")].copy()
    modality = original.track_modality.fillna("")
    output_geometry = (
        modality.isin({"rna", "rna_seq", "10x_scRNAseq_3prime_pseudobulk"})
        & original.readout_role.isin(
            {"gene_body", "gene_body_output_clipped", "tes_3prime", "tes_3prime_1024bp"}
        )
    ) | (modality.eq("cage") & original.readout_role.isin({"tss", "tss_1024bp"}))
    regulatory_geometry = ~modality.isin(OUTPUT_MODALITIES) & original.readout_role.eq("local_edit")
    selected = original[output_geometry | regulatory_geometry].copy()
    selected["design_set"] = np.where(
        selected.design_class.str.startswith("single_base_substitution"),
        "single_base_substitution",
        selected.design_class,
    )

    keys = [
        "backend", "locus_id", "track_id", "track_group", "track_task_name",
        "track_cell_type", "track_modality", "sample_stage", "track_strand",
        "readout_role", "design_set",
    ]
    rows: list[dict[str, object]] = []
    for key, group in selected.groupby(keys, sort=False, dropna=False):
        if len(group) < 10:
            continue
        wt1_rho, wt1_pvalue = safe_spearman(
            group.wt1_score_delta, group.signed_effect
        )
        confounder_rho, confounder_pvalue = safe_spearman(
            group.sp_klf_egr_score_delta, group.signed_effect
        )
        rows.append(
            {
                **dict(zip(keys, key)),
                "n": int(len(group)),
                "median_signed_effect": float(group.signed_effect.median()),
                "median_absolute_effect": float(group.absolute_effect.median()),
                "q90_absolute_effect": float(group.absolute_effect.quantile(0.9)),
                "spearman_wt1_delta_vs_signed": wt1_rho,
                "spearman_wt1_pvalue": wt1_pvalue,
                "spearman_confounder_delta_vs_signed": confounder_rho,
                "spearman_confounder_pvalue": confounder_pvalue,
            }
        )
    result = pd.DataFrame.from_records(rows)
    for source, target in (
        ("spearman_wt1_pvalue", "spearman_wt1_fdr"),
        ("spearman_confounder_pvalue", "spearman_confounder_fdr"),
    ):
        result[target] = result.groupby(
            ["backend", "locus_id", "design_set"]
        )[source].transform(bh_adjust)
    return result


def bh_adjust(values: pd.Series) -> pd.Series:
    result = pd.Series(np.nan, index=values.index, dtype=float)
    valid = values.dropna().sort_values()
    if valid.empty:
        return result
    n = len(valid)
    adjusted = valid.to_numpy() * n / np.arange(1, n + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    result.loc[valid.index] = np.clip(adjusted, 0, 1)
    return result


def safe_spearman(x: pd.Series, y: pd.Series) -> tuple[float, float]:
    if len(x) < 5 or x.nunique() < 2 or y.nunique() < 2:
        return float("nan"), float("nan")
    result = spearmanr(x, y)
    return float(result.statistic), float(result.pvalue)


def standardized_ols(frame: pd.DataFrame) -> tuple[float, float, float]:
    predictors = [
        "wt1_score_delta", "sp_klf_egr_score_delta", "changed_bases",
        "dinucleotide_l1_distance",
    ]
    x = frame[predictors].astype(float)
    x = x.loc[:, x.std(ddof=0).gt(0)]
    if "wt1_score_delta" not in x or len(frame) <= len(x.columns) + 3:
        return float("nan"), float("nan"), float("nan")
    x = (x - x.mean()) / x.std(ddof=0)
    y = frame.signed_effect.astype(float)
    if y.std(ddof=0) == 0:
        return float("nan"), float("nan"), float("nan")
    y = (y - y.mean()) / y.std(ddof=0)
    fit = sm.OLS(y, sm.add_constant(x)).fit(cov_type="HC3")
    return (
        float(fit.params["wt1_score_delta"]),
        float(fit.pvalues["wt1_score_delta"]),
        float(fit.rsquared),
    )


def association_rows(panels: pd.DataFrame) -> pd.DataFrame:
    rows = []
    groups = ["backend", "locus_id", "panel_id", "track_group", "track_modality", "sample_stage", "strand_class", "readout_role"]
    for key, group in panels.groupby(groups, sort=False, dropna=False):
        base = dict(zip(groups, key))
        for design_set, subset in (
            ("composition_permutation", group[group.design_class.eq("composition_permutation")]),
            ("single_base_substitution", group[group.design_class.str.startswith("single_base_substitution")]),
        ):
            if len(subset) < 10:
                continue
            rho, pvalue = safe_spearman(subset.wt1_score_delta, subset.signed_effect)
            conf_rho, conf_pvalue = safe_spearman(subset.sp_klf_egr_score_delta, subset.signed_effect)
            abs_rho, abs_pvalue = safe_spearman(-subset.wt1_score_delta, subset.absolute_effect)
            coefficient, coefficient_pvalue, r_squared = standardized_ols(subset)
            low = subset.wt1_score_alt <= subset.wt1_score_alt.quantile(0.25)
            high = subset.wt1_score_alt >= subset.wt1_score_alt.quantile(0.75)
            rows.append(
                {
                    **base,
                    "design_set": design_set,
                    "n": int(len(subset)),
                    "spearman_wt1_delta_vs_signed": rho,
                    "spearman_wt1_pvalue": pvalue,
                    "spearman_confounder_delta_vs_signed": conf_rho,
                    "spearman_confounder_pvalue": conf_pvalue,
                    "spearman_wt1_loss_vs_absolute": abs_rho,
                    "spearman_absolute_pvalue": abs_pvalue,
                    "ols_standardized_wt1_coefficient": coefficient,
                    "ols_wt1_pvalue": coefficient_pvalue,
                    "ols_r_squared": r_squared,
                    "low_wt1_score_median_signed": float(subset.loc[low, "signed_effect"].median()),
                    "high_wt1_score_median_signed": float(subset.loc[high, "signed_effect"].median()),
                    "high_minus_low_wt1_quartile_effect": float(
                        subset.loc[high, "signed_effect"].median()
                        - subset.loc[low, "signed_effect"].median()
                    ),
                }
            )
    result = pd.DataFrame.from_records(rows)
    for source, target in (
        ("spearman_wt1_pvalue", "spearman_wt1_fdr"),
        ("spearman_confounder_pvalue", "spearman_confounder_fdr"),
        ("spearman_absolute_pvalue", "spearman_absolute_fdr"),
        ("ols_wt1_pvalue", "ols_wt1_fdr"),
    ):
        result[target] = result.groupby(["backend", "locus_id", "design_set"])[source].transform(bh_adjust)
    return result


def main() -> None:
    args = parse_args()
    design = pd.read_csv(args.root / "prepared" / "variant_design.tsv", sep="\t")
    available = [backend for backend in BACKENDS if (args.root / "runs" / backend / "validation_summary.json").exists()]
    if not available:
        raise ValueError("No completed backend runs")
    tracks = eligible_rows(pd.concat([read_backend(args.root, backend) for backend in available], ignore_index=True))
    tracks = tracks.merge(
        design, on=["mutation_id", "locus_id"], how="left", validate="many_to_one"
    )
    if tracks.wt1_score_delta.isna().any():
        raise ValueError("Track rows did not map to variant design")
    original_track_cases = original_track_case_rows(tracks)
    panels = add_finetuned_cell_contrasts(build_panels(tracks))
    panels = panels.merge(
        design, on=["mutation_id", "locus_id"], how="left", validate="many_to_one"
    )
    if panels.wt1_score_delta.isna().any():
        raise ValueError("Panel rows did not map to variant design")
    associations = association_rows(panels)

    analysis = args.root / "analysis"
    analysis.mkdir(parents=True, exist_ok=True)
    panels.to_csv(analysis / "panel_effects.tsv", sep="\t", index=False)
    associations.to_csv(analysis / "association_summary.tsv", sep="\t", index=False)
    original_track_cases.to_csv(
        analysis / "original_track_case_summary.tsv", sep="\t", index=False
    )
    validation = {
        "status": "ok",
        "backends": available,
        "mutations": int(design.mutation_id.nunique()),
        "panel_rows": int(len(panels)),
        "panel_keys_unique": bool(
            ~panels.duplicated(["backend", "mutation_id", "locus_id", "panel_id"]).any()
        ),
        "association_rows": int(len(associations)),
        "original_track_case_rows": int(len(original_track_cases)),
        "original_track_case_keys_unique": bool(
            ~original_track_cases.duplicated(
                ["backend", "locus_id", "track_id", "readout_role", "design_set"]
            ).any()
        ),
        "nonfinite_signed_effects": int((~np.isfinite(panels.signed_effect)).sum()),
        "statistical_scope": (
            "Variant-space associations are deterministic model diagnostics, not "
            "biological replicates or causal p-values."
        ),
    }
    if (
        not validation["panel_keys_unique"]
        or not validation["original_track_case_keys_unique"]
        or validation["nonfinite_signed_effects"]
    ):
        raise ValueError(f"Analysis validation failed: {validation}")
    (analysis / "validation_summary.json").write_text(
        json.dumps(validation, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(validation, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
