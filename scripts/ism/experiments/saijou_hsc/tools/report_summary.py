"""Derive compact tables from canonical Saijou ISM outputs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


FINE_RUNS = {
    "AlphaGenome fine-tuned": "alphagenome_finetuned",
    "Borzoi fine-tuned": "borzoi_finetuned",
}
CONTROL_SPECS = {
    "Acta2": {
        "target": -61,
        "family": "SRF_CARG",
        "label": "promoter CArG / SRF",
    },
    "Col1a1": {
        "target": -71,
        "family": "NFY_CCAAT",
        "label": "inverted CCAAT / NF-Y",
    },
}
CANONICAL_FAMILIES = {
    "CTCF_CTCFL",
    "NFY_CCAAT",
    "SRF_CARG",
    "KLF_SP_GC",
    "EGR_ZBTB",
    "ETS",
    "AP1",
    "STAT",
    "HNF",
    "CEBP",
    "SMAD",
    "TEAD",
    "HIF",
    "NFKB",
    "RREB1",
}


@dataclass(frozen=True)
class SummaryTables:
    frames: dict[str, pd.DataFrame]
    validation: dict[str, object]


def _load_inputs(root: Path) -> dict[str, pd.DataFrame]:
    analysis = root / "analysis"
    comparison = analysis / "original_comparison"
    return {
        "segments": pd.read_csv(analysis / "candidate_segments.tsv", sep="\t"),
        "selected_manifest": pd.read_csv(
            root / "prepared_original_candidates/mutation_manifest.tsv", sep="\t"
        ),
        "segment_summary": pd.read_csv(
            comparison / "segment_track_group_summary.tsv", sep="\t"
        ),
        "fine_specificity": pd.read_csv(
            comparison / "fine_tuned_cell_specificity.tsv", sep="\t"
        ),
        "motif_summary": pd.read_csv(
            analysis / "candidate_motif_disruption_summary.tsv", sep="\t"
        ),
        "transcripts": pd.read_csv(
            analysis / "representative_transcripts.tsv", sep="\t"
        ),
        "priors": pd.read_csv(root / "literature_regulatory_priors.tsv", sep="\t"),
    }


def _selected_readout_rows(
    table: pd.DataFrame, segment_readout: dict[str, str]
) -> pd.DataFrame:
    return table.loc[
        table.apply(
            lambda row: row.readout_role == segment_readout.get(row.locus_id),
            axis=1,
        )
    ].copy()


def _effect_matrices(relevant: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    relevant = relevant.copy()
    relevant["cell_type"] = relevant.track_group.str.replace(
        "_finetuned_10x", "", regex=False
    )
    fine = relevant.loc[relevant.model_label.isin(FINE_RUNS)].copy()
    fine["cell_rank"] = fine.groupby(
        ["model_label", "locus_id"]
    ).median_abs_effect.rank(method="min", ascending=False)
    original = relevant.loc[~relevant.model_label.isin(FINE_RUNS)].copy()
    original["group_rank"] = original.groupby(
        ["model_label", "locus_id"]
    ).median_abs_effect.rank(method="min", ascending=False)
    return fine, original


def _fine_specificity_wide(
    fine_specificity: pd.DataFrame, segment_readout: dict[str, str]
) -> pd.DataFrame:
    fields = [
        "hsc_median_abs_effect",
        "hsc_median_effect",
        "hsc_rank",
        "top_cell",
        "hsc_vs_max_other_ratio",
    ]
    selected = _selected_readout_rows(fine_specificity, segment_readout)
    wide = selected.pivot(index="locus_id", columns="model_label", values=fields)
    wide.columns = [
        f"{metric}_{'ag' if model.startswith('AlphaGenome') else 'bz'}"
        for metric, model in wide.columns
    ]
    return wide.reset_index().rename(columns={"locus_id": "candidate_segment_id"})


def _motif_family_tables(
    motif_summary: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    canonical = motif_summary.loc[
        motif_summary.family.isin(CANONICAL_FAMILIES)
    ].copy()
    summary = (
        canonical.groupby(["candidate_segment_id", "gene", "family"])
        .agg(
            affected_centers=("affected_centers", "max"),
            max_abs_score_change=("max_abs_score_change", "max"),
            median_abs_score_change=("median_abs_score_change", "max"),
        )
        .reset_index()
        .sort_values(
            ["candidate_segment_id", "affected_centers", "max_abs_score_change"],
            ascending=[True, False, False],
        )
    )
    summary["family_rank"] = summary.groupby("candidate_segment_id").cumcount() + 1
    top = summary.loc[
        summary.family_rank.eq(1),
        [
            "candidate_segment_id",
            "family",
            "affected_centers",
            "max_abs_score_change",
        ],
    ].rename(
        columns={
            "family": "top_motif_family",
            "affected_centers": "top_motif_affected_centers",
            "max_abs_score_change": "top_motif_max_score_change",
        }
    )
    return canonical, summary, top


def _top_original_groups(original_matrix: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model, locus), group in original_matrix.groupby(
        ["model_label", "locus_id"]
    ):
        row = group.nlargest(1, "median_abs_effect").iloc[0]
        rows.append(
            {
                "candidate_segment_id": locus,
                "original_model": "ag" if model.startswith("AlphaGenome") else "bz",
                "top_original_group": row.track_group,
                "top_original_abs_effect": row.median_abs_effect,
                "top_original_signed_effect": row.median_effect,
                "top_original_ref_sum": row.median_ref_sum,
            }
        )
    wide = pd.DataFrame(rows).pivot(
        index="candidate_segment_id", columns="original_model"
    )
    wide.columns = [f"{field}_{model}" for field, model in wide.columns]
    return wide.reset_index()


def _candidate_overview(
    segments: pd.DataFrame,
    selected_manifest: pd.DataFrame,
    fine_wide: pd.DataFrame,
    top_family: pd.DataFrame,
    original_top: pd.DataFrame,
) -> pd.DataFrame:
    actual_counts = (
        selected_manifest.groupby("locus_id")
        .agg(
            tested_centers=("variant_offset_from_tss_transcription_bp", "nunique"),
            tested_mutations=("mutation_id", "nunique"),
        )
        .reset_index()
        .rename(columns={"locus_id": "candidate_segment_id"})
    )
    overview = (
        segments.merge(actual_counts, on="candidate_segment_id", validate="one_to_one")
        .merge(fine_wide, on="candidate_segment_id", validate="one_to_one")
        .merge(top_family, on="candidate_segment_id", how="left", validate="one_to_one")
        .merge(original_top, on="candidate_segment_id", validate="one_to_one")
    )
    numeric = [
        "hsc_median_abs_effect_ag",
        "hsc_median_abs_effect_bz",
        "hsc_median_effect_ag",
        "hsc_median_effect_bz",
        "hsc_vs_max_other_ratio_ag",
        "hsc_vs_max_other_ratio_bz",
        "top_original_abs_effect_ag",
        "top_original_abs_effect_bz",
    ]
    overview[numeric] = overview[numeric].apply(pd.to_numeric, errors="raise")
    overview["both_models_hsc_top"] = overview.top_cell_ag.eq(
        "hsc"
    ) & overview.top_cell_bz.eq("hsc")
    overview["minimum_hsc_ratio"] = overview[
        ["hsc_vs_max_other_ratio_ag", "hsc_vs_max_other_ratio_bz"]
    ].min(axis=1)
    overview["fine_signed_direction_agreement"] = np.sign(
        overview.hsc_median_effect_ag
    ).eq(np.sign(overview.hsc_median_effect_bz))
    return overview


def _best_control_readout(data: pd.DataFrame, target: int) -> dict[str, object]:
    rows = []
    for role, group in data.groupby("readout_role"):
        effects = group.groupby(
            "variant_offset_from_tss_transcription_bp"
        ).abs_effect.median()
        center = min(effects.index, key=lambda value: abs(value - target))
        rows.append(
            {
                "readout_role": role,
                "center": int(center),
                "median_abs_effect": float(effects.loc[center]),
                "rank": int(effects.rank(method="min", ascending=False).loc[center]),
                "centers_total": int(len(effects)),
                "percentile": float(effects.rank(pct=True).loc[center]),
            }
        )
    return max(rows, key=lambda row: row["percentile"])


def _control_recovery(root: Path, canonical: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for gene, spec in CONTROL_SPECS.items():
        motif = canonical.loc[
            canonical.gene.eq(gene) & canonical.family.eq(spec["family"])
        ].nlargest(1, "max_abs_score_change")
        motif_row = motif.iloc[0] if not motif.empty else None
        for model_label, run_name in FINE_RUNS.items():
            data = pd.read_csv(
                root / f"runs/{run_name}/features/{gene}.tsv", sep="\t"
            )
            data = data.loc[data.track_id.eq("hsc")].copy()
            data["abs_effect"] = data.log2fc_ratio_of_sums.abs()
            rows.append(
                {
                    "gene": gene,
                    "control": spec["label"],
                    "model": model_label,
                    **_best_control_readout(data, spec["target"]),
                    "motif_family": spec["family"],
                    "motif_affected_centers": (
                        int(motif_row.affected_centers) if motif_row is not None else 0
                    ),
                    "motif_max_abs_score_change": (
                        float(motif_row.max_abs_score_change)
                        if motif_row is not None
                        else np.nan
                    ),
                }
            )
    return pd.DataFrame(rows)


def _gene_summary(
    overview: pd.DataFrame,
    transcripts: pd.DataFrame,
    priors: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for gene, group in overview.groupby("gene", sort=False):
        best = group.nlargest(1, "minimum_hsc_ratio").iloc[0]
        prior = priors.loc[priors.gene.eq(gene)].iloc[0]
        transcript = transcripts.loc[transcripts.gene.eq(gene)].iloc[0]
        rows.append(
            {
                "gene": gene,
                "best_segment": best.candidate_segment_id,
                "best_segment_region": best.peak_region_type,
                "best_transcript_feature": best.peak_transcript_feature,
                "both_models_hsc_top": bool(best.both_models_hsc_top),
                "minimum_hsc_ratio": float(best.minimum_hsc_ratio),
                "top_motif_family": best.top_motif_family,
                "top_original_group_ag": best.top_original_group_ag,
                "top_original_group_bz": best.top_original_group_bz,
                "representative_transcript_id": transcript.representative_transcript_id,
                "transcript_name": transcript.transcript_name,
                "transcript_biotype": transcript.transcript_biotype,
                "transcript_support_level": transcript.transcript_support_level,
                "is_basic": bool(transcript.is_basic),
                "biological_role": prior.biological_role_in_this_project,
                "regulatory_prior": prior.regulatory_prior_for_hotspot_interpretation,
                "scope_caveat": prior.scope_caveat,
            }
        )
    return pd.DataFrame(rows)


def _summary_validation(
    overview: pd.DataFrame,
    selected_manifest: pd.DataFrame,
    controls: pd.DataFrame,
) -> dict[str, object]:
    numeric = [
        "hsc_median_abs_effect_ag",
        "hsc_median_abs_effect_bz",
        "top_original_abs_effect_ag",
        "top_original_abs_effect_bz",
    ]
    tested_centers = int(
        selected_manifest[["gene", "variant_offset_from_tss_transcription_bp"]]
        .drop_duplicates()
        .shape[0]
    )
    if selected_manifest.gene.nunique() == 1:
        tested_centers = int(
            selected_manifest.variant_offset_from_tss_transcription_bp.nunique()
        )
    return {
        "status": "ok",
        "genes": int(overview.gene.nunique()),
        "segments": int(len(overview)),
        "tested_centers": tested_centers,
        "mutations": int(selected_manifest.mutation_id.nunique()),
        "both_models_hsc_top_segments": int(overview.both_models_hsc_top.sum()),
        "known_control_rows": int(len(controls)),
        "missing_overview_values": int(overview[numeric].isna().sum().sum()),
        "nonfinite_overview_values": int(
            (~np.isfinite(overview[numeric].to_numpy(dtype=float))).sum()
        ),
    }


def build_summary_tables(root: Path) -> SummaryTables:
    """Build all report tables without writing files."""

    inputs = _load_inputs(root)
    segments = inputs["segments"]
    segment_readout = (
        segments.set_index("candidate_segment_id").peak_readout_role.to_dict()
    )
    relevant = _selected_readout_rows(inputs["segment_summary"], segment_readout)
    fine_matrix, original_matrix = _effect_matrices(relevant)
    fine_wide = _fine_specificity_wide(inputs["fine_specificity"], segment_readout)
    canonical, motif_families, top_family = _motif_family_tables(
        inputs["motif_summary"]
    )
    overview = _candidate_overview(
        segments,
        inputs["selected_manifest"],
        fine_wide,
        top_family,
        _top_original_groups(original_matrix),
    )
    controls = _control_recovery(root, canonical)
    genes = _gene_summary(overview, inputs["transcripts"], inputs["priors"])
    validation = _summary_validation(
        overview, inputs["selected_manifest"], controls
    )
    if validation["missing_overview_values"] or validation["nonfinite_overview_values"]:
        raise RuntimeError(validation)
    return SummaryTables(
        frames={
            "fine_cell_matrix": fine_matrix,
            "original_group_matrix": original_matrix,
            "motif_family_summary": motif_families,
            "candidate_overview": overview,
            "known_control_recovery": controls,
            "gene_summary": genes,
        },
        validation=validation,
    )
