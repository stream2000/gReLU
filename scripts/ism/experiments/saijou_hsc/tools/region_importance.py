#!/usr/bin/env python
"""Reusable cross-model track-calibrated importance and region calling.

The score is used only to select regions for highlighting.  Plotted effect
curves remain signed log2FC.  The minimal score is the empirical percentile of
the weaker model for the same cell track on one fixed gene-body expression
readout, calibrated within reference-signal strata.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


from .genomics import (
    add_genomic_annotation,
    load_gtf_annotations,
    load_transcript_overrides,
)


MODELS = {
    "AlphaGenome": "runs/alphagenome_finetuned/features/combined_mutation_features.parquet",
    "Borzoi": "runs/borzoi_finetuned/features/combined_mutation_features.parquet",
}
TRACKS = ("hsc", "mac", "lsec", "chol")
DEFAULT_READOUT_ROLE = "gene_body_output_clipped"
KEYS = (
    "gene",
    "variant_offset_from_tss_transcription_bp",
    "track_id",
    "readout_role",
)
CENTER_KEYS = ("gene", "variant_offset_from_tss_transcription_bp")


def reference_bins(values: pd.Series, requested: int) -> pd.Series:
    """Make deterministic near-equal bins while tolerating tied reference sums."""
    count = max(1, min(int(requested), max(1, len(values) // 100)))
    if count == 1:
        return pd.Series(np.zeros(len(values), dtype=int), index=values.index)
    ranked = np.log1p(values.clip(lower=0)).rank(method="first")
    return pd.qcut(
        ranked, q=count, labels=False, duplicates="drop"
    ).astype(int)


def summarize_model(
    path: Path,
    model: str,
    requested_bins: int,
    readout_role: str,
) -> pd.DataFrame:
    data = pd.read_parquet(path)
    data = data.loc[
        data.track_id.isin(TRACKS)
        & data.readout_role.eq(readout_role)
    ].copy()
    data["abs_log2fc"] = data.log2fc_ratio_of_sums.abs()
    grouped = (
        data.groupby(list(KEYS), sort=False)
        .agg(
            replacement_replicates=("mutation_id", "nunique"),
            median_signed_log2fc=("log2fc_ratio_of_sums", "median"),
            median_abs_log2fc=("abs_log2fc", "median"),
            median_ref_sum=("ref_sum", "median"),
            replacement_mad=(
                "log2fc_ratio_of_sums",
                lambda values: float(
                    np.median(np.abs(values - np.median(values)))
                ),
            ),
        )
        .reset_index()
    )
    grouped["model"] = model
    grouped["reference_bin"] = -1
    calibration = ["model", "track_id", "readout_role"]
    for _, indices in grouped.groupby(calibration, sort=False).groups.items():
        grouped.loc[indices, "reference_bin"] = reference_bins(
            grouped.loc[indices, "median_ref_sum"], requested_bins
        ).to_numpy()
    grouped["model_tail_percentile"] = grouped.groupby(
        calibration + ["reference_bin"], sort=False
    )["median_abs_log2fc"].rank(method="average", pct=True)
    return grouped


def combine_models(summaries: pd.DataFrame) -> pd.DataFrame:
    wide = summaries.pivot(index=list(KEYS), columns="model")
    wide.columns = [
        f"{field}_{model.lower()}" for field, model in wide.columns
    ]
    wide = wide.reset_index()
    required = [
        "model_tail_percentile_alphagenome",
        "model_tail_percentile_borzoi",
        "median_signed_log2fc_alphagenome",
        "median_signed_log2fc_borzoi",
    ]
    wide = wide.dropna(subset=required).copy()
    wide["cross_model_conjunction"] = wide[
        [
            "model_tail_percentile_alphagenome",
            "model_tail_percentile_borzoi",
        ]
    ].min(axis=1)
    wide["model_direction_agreement"] = np.sign(
        wide.median_signed_log2fc_alphagenome
    ).eq(np.sign(wide.median_signed_log2fc_borzoi))
    return wide


def select_centers(track_scores: pd.DataFrame) -> pd.DataFrame:
    opportunities = (
        track_scores.groupby(list(CENTER_KEYS), sort=False)
        .size()
        .rename("track_readout_opportunities")
        .reset_index()
    )
    centers = (
        track_scores.sort_values(
            [
                "cross_model_conjunction",
                "model_direction_agreement",
                "track_id",
                "readout_role",
            ],
            ascending=[False, False, True, True],
        )
        .drop_duplicates(list(CENTER_KEYS), keep="first")
        .merge(
            opportunities,
            on=list(CENTER_KEYS),
            how="left",
            validate="one_to_one",
        )
    )
    centers["importance_score"] = (
        100.0
        * centers["cross_model_conjunction"].rank(
            method="average", pct=True
        )
    )
    return centers


def select_cell_logfc(track_scores: pd.DataFrame) -> pd.DataFrame:
    keys = [*CENTER_KEYS, "track_id"]
    selected = (
        track_scores.sort_values(
            [
                "cross_model_conjunction",
                "model_direction_agreement",
                "readout_role",
            ],
            ascending=[False, False, True],
        )
        .drop_duplicates(keys, keep="first")
        .rename(columns={"track_id": "cell"})
        .copy()
    )
    for column in (
        "median_signed_log2fc_alphagenome",
        "median_signed_log2fc_borzoi",
    ):
        selected[f"{column}_smoothed"] = selected.groupby(
            ["gene", "cell"], sort=False
        )[column].transform(
            lambda values: values.rolling(
                5, center=True, min_periods=1
            ).median()
        )
    return selected


def call_regions(
    centers: pd.DataFrame,
    threshold: float,
    max_gap: int,
    top_per_gene: int,
) -> pd.DataFrame:
    eligible = centers.loc[
        centers.importance_score.ge(threshold)
        & ~centers.overlaps_splice_site.fillna(False)
    ].copy()
    rows: list[dict[str, object]] = []
    for gene, group in eligible.groupby("gene", sort=False):
        group = group.sort_values(
            "variant_offset_from_tss_transcription_bp"
        )
        block = (
            group.variant_offset_from_tss_transcription_bp.diff()
            .fillna(max_gap + 1)
            .gt(max_gap)
            .cumsum()
        )
        for _, region in group.groupby(block, sort=False):
            if len(region) < 2:
                continue
            peak = region.nlargest(1, "importance_score").iloc[0]
            strongest = region.importance_score.nlargest(min(3, len(region)))
            rows.append(
                {
                    "gene": gene,
                    "region_start": int(
                        region.variant_offset_from_tss_transcription_bp.min()
                    ),
                    "region_end": int(
                        region.variant_offset_from_tss_transcription_bp.max()
                    ),
                    "centers": int(len(region)),
                    "region_score": float(strongest.mean()),
                    "peak_score": float(peak.importance_score),
                    "peak_offset": int(
                        peak.variant_offset_from_tss_transcription_bp
                    ),
                    "driving_track": str(peak.track_id),
                    "driving_readout": str(peak.readout_role),
                    "alphagenome_signed_log2fc": float(
                        peak.median_signed_log2fc_alphagenome
                    ),
                    "borzoi_signed_log2fc": float(
                        peak.median_signed_log2fc_borzoi
                    ),
                    "direction_agreement_fraction": float(
                        region.model_direction_agreement.mean()
                    ),
                    "region_type": str(peak.region_type),
                    "transcript_feature": str(peak.transcript_feature),
                }
            )
    regions = pd.DataFrame.from_records(rows)
    if regions.empty:
        return regions
    regions = regions.sort_values(
        ["gene", "region_score", "peak_score"],
        ascending=[True, False, False],
    )
    regions["gene_rank"] = regions.groupby("gene").cumcount() + 1
    regions = regions.loc[regions.gene_rank.le(top_per_gene)].copy()
    regions = regions.sort_values(
        ["region_score", "peak_score"], ascending=False
    ).reset_index(drop=True)
    regions.insert(0, "screen_rank", np.arange(1, len(regions) + 1))
    regions["region_id"] = regions.screen_rank.map(
        lambda value: f"region_{int(value):02d}"
    )
    regions["region_label"] = regions.apply(
        lambda row: (
            f"{row.gene} {int(row.region_start):+d}.."
            f"{int(row.region_end):+d}"
        ),
        axis=1,
    )
    return regions


def assess_controls(
    centers: pd.DataFrame,
    controls: pd.DataFrame,
    threshold: float,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for control_row in controls.itertuples(index=False):
        control = control_row._asdict()
        # A 10-bp edit is centered at the saved offset and spans center +/-5.
        subset = centers.loc[
            centers.gene.eq(control["gene"])
            & centers.variant_offset_from_tss_transcription_bp.between(
                int(control["start"]) - 5,
                int(control["end"]) + 5,
            )
        ].copy()
        if subset.empty:
            rows.append(
                {
                    **control,
                    "covered_centers": 0,
                    "best_center": np.nan,
                    "best_score": np.nan,
                    "recalled_at_threshold": False,
                }
            )
            continue
        best = subset.nlargest(1, "importance_score").iloc[0]
        rows.append(
            {
                **control,
                "covered_centers": int(len(subset)),
                "best_center": int(
                    best.variant_offset_from_tss_transcription_bp
                ),
                "best_score": float(best.importance_score),
                "recalled_at_threshold": bool(
                    best.importance_score >= threshold
                ),
                "driving_track": str(best.track_id),
                "driving_readout": str(best.readout_role),
                "alphagenome_signed_log2fc": float(
                    best.median_signed_log2fc_alphagenome
                ),
                "borzoi_signed_log2fc": float(
                    best.median_signed_log2fc_borzoi
                ),
                "model_direction_agreement": bool(
                    best.model_direction_agreement
                ),
            }
        )
    return pd.DataFrame.from_records(rows)


def run_cross_model_track_analysis(
    *,
    root: Path,
    gtf: Path,
    positive_control_registry: Path,
    reference_bins_count: int = 10,
    readout_role: str = DEFAULT_READOUT_ROLE,
    score_threshold: float = 95.0,
    max_center_gap_bp: int = 4,
    top_regions_per_gene: int = 3,
    splice_buffer_bp: int = 2,
    transcript_authority: Path | None = None,
) -> dict[str, object]:
    """Write the generic importance interface for any prepared Saijou scan."""

    root = root.resolve()
    analysis = root / "analysis"
    analysis.mkdir(parents=True, exist_ok=True)
    genes = pd.read_csv(root / "prepared/genes.tsv", sep="\t")
    transcript_overrides = (
        load_transcript_overrides(transcript_authority)
        if transcript_authority is not None
        else None
    )
    (
        transcript_ids,
        exon_intervals,
        splice_sites,
        transcript_features,
        transcript_metadata,
    ) = load_gtf_annotations(
        gtf,
        genes,
        transcript_overrides=transcript_overrides,
    )

    summaries = pd.concat(
        [
            summarize_model(
                root / relpath,
                model,
                reference_bins_count,
                readout_role,
            )
            for model, relpath in MODELS.items()
        ],
        ignore_index=True,
    )
    track_scores = combine_models(summaries)
    centers = select_centers(track_scores)
    centers = add_genomic_annotation(
        centers,
        genes,
        exon_intervals,
        splice_sites,
        splice_buffer_bp,
        transcript_features,
    )
    cell_logfc = select_cell_logfc(track_scores)
    regions = call_regions(
        centers,
        score_threshold,
        max_center_gap_bp,
        top_regions_per_gene,
    )
    control_registry = pd.read_csv(
        positive_control_registry, sep="\t"
    )
    controls = assess_controls(
        centers, control_registry, score_threshold
    )

    summaries.to_csv(
        analysis / "model_track_center_effects.tsv", sep="\t", index=False
    )
    track_scores.to_parquet(
        analysis / "cross_model_track_readout_scores.parquet", index=False
    )
    centers.to_csv(
        analysis / "center_importance_scores.tsv", sep="\t", index=False
    )
    cell_logfc.to_csv(
        analysis / "cell_specific_log2fc.tsv", sep="\t", index=False
    )
    regions.to_csv(
        analysis / "top_important_regions.tsv", sep="\t", index=False
    )
    controls.to_csv(
        analysis / "positive_control_recall.tsv", sep="\t", index=False
    )
    pd.DataFrame(
        [
            {
                "score_threshold": threshold,
                "positive_controls_recalled": int(
                    controls.best_score.ge(threshold).sum()
                ),
                "positive_controls_total": int(len(controls)),
                "recall_fraction": float(
                    controls.best_score.ge(threshold).mean()
                ),
            }
            for threshold in (50.0, 60.0, 70.0, 80.0, 90.0, 95.0, 97.5)
        ]
    ).to_csv(
        analysis / "positive_control_threshold_sensitivity.tsv",
        sep="\t",
        index=False,
    )
    pd.DataFrame(
        [
            {
                "gene": gene,
                "transcript_id": transcript_ids[gene],
                **transcript_metadata[gene],
            }
            for gene in genes.gene
        ]
    ).to_csv(
        analysis / "representative_transcript_audit.tsv",
        sep="\t",
        index=False,
    )

    expected_centers = (
        pd.read_csv(root / "prepared/mutation_manifest.tsv", sep="\t")
        .groupby("gene")[
            "variant_offset_from_tss_transcription_bp"
        ]
        .nunique()
        .to_dict()
    )
    actual_centers = centers.groupby("gene").size().to_dict()
    numeric = summaries.select_dtypes(include=[np.number]).to_numpy()
    validation = {
        "status": "ok",
        "scope": "Saijou mouse mm10 TSS strict-shuffle",
        "score_role": "region selection only; plotted y-axis is signed log2FC",
        "score_definition": (
            "100 x empirical percentile of the maximum across four cells of "
            "min(AlphaGenome percentile, Borzoi percentile) on the fixed "
            "gene-body expression readout, with model percentiles calibrated "
            "by cell/reference decile"
        ),
        "fixed_readout_role": str(readout_role),
        "score_threshold": float(score_threshold),
        "top_regions_per_gene": int(top_regions_per_gene),
        "models": sorted(summaries.model.unique().tolist()),
        "tracks": sorted(summaries.track_id.unique().tolist()),
        "genes": genes.gene.tolist(),
        "expected_centers_by_gene": expected_centers,
        "actual_centers_by_gene": actual_centers,
        "replacement_replicates": sorted(
            summaries.replacement_replicates.unique().astype(int).tolist()
        ),
        "center_rows": int(len(centers)),
        "cell_logfc_rows": int(len(cell_logfc)),
        "selected_regions": int(len(regions)),
        "positive_controls": int(len(controls)),
        "positive_controls_recalled": int(
            controls.recalled_at_threshold.sum()
        ),
        "nonfinite_summary_values": int((~np.isfinite(numeric)).sum()),
        "transcripts_match_authority": True,
    }
    if transcript_authority is not None:
        authority = pd.read_csv(transcript_authority, sep="\t")
        validation["transcripts_match_authority"] = bool(
            all(
                str(transcript_ids[row.gene]) == str(row.transcript_id)
                for row in authority.itertuples(index=False)
            )
        )
    if (
        validation["replacement_replicates"] != [3]
        or validation["expected_centers_by_gene"]
        != validation["actual_centers_by_gene"]
        or validation["nonfinite_summary_values"] != 0
        or not validation["transcripts_match_authority"]
    ):
        validation["status"] = "failed"
    (analysis / "validation_summary.json").write_text(
        json.dumps(validation, indent=2) + "\n"
    )
    print(json.dumps(validation, indent=2))
    if validation["status"] != "ok":
        raise RuntimeError(validation)
    return validation
