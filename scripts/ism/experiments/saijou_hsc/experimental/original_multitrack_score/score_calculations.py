"""Pure score calculations for the original-model multitrack experiment.

This module has no CLI, path defaults or file I/O. Experiment entry points own
orchestration, validation summaries and plots; this module owns calculations
that must remain identical across AlphaGenome methods and Borzoi transfer.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


CENTER_KEYS = ["gene", "variant_offset_from_tss_transcription_bp"]
THRESHOLD = 95.0


@dataclass(frozen=True)
class SameWidthWindowScore:
    """A candidate-window statistic calibrated against same-width windows."""

    statistic: float
    background_windows: int
    percentile_score: float


def compute_spatial_median_support(
    frame: pd.DataFrame,
    column: str,
    radius_bp: int = 4,
) -> pd.DataFrame:
    """Aggregate a center statistic across one edit-width neighborhood."""
    rows: list[dict[str, object]] = []
    for gene, group in frame.groupby("gene", sort=False):
        group = group.sort_values(
            "variant_offset_from_tss_transcription_bp"
        )
        positions = group.variant_offset_from_tss_transcription_bp.to_numpy()
        values = group[column].to_numpy()
        for position in positions:
            mask = (positions >= position - radius_bp) & (
                positions <= position + radius_bp
            )
            rows.append(
                {
                    "gene": gene,
                    "variant_offset_from_tss_transcription_bp": int(position),
                    "spatial_support": float(np.median(values[mask])),
                    "spatial_support_centers": int(mask.sum()),
                    "spatial_support_span_bp": int(
                        positions[mask].max() - positions[mask].min()
                    ),
                }
            )
    return pd.DataFrame.from_records(rows)


def score_same_width_window(
    gene_centers: pd.DataFrame,
    subset: pd.DataFrame,
    column: str,
) -> SameWidthWindowScore:
    """Summarize and rank a candidate against all same-width gene windows."""
    width = int(len(subset))
    background = (
        gene_centers.sort_values(
            "variant_offset_from_tss_transcription_bp"
        )[column]
        .rolling(width, min_periods=width)
        .median()
        .dropna()
    )
    statistic = float(subset[column].median())
    less = int((background < statistic).sum())
    equal = int(
        np.isclose(
            background.to_numpy(), statistic, rtol=0, atol=1e-12
        ).sum()
    )
    return SameWidthWindowScore(
        statistic=statistic,
        background_windows=int(len(background)),
        percentile_score=float(
            100 * (less + (equal + 1) / 2) / len(background)
        ),
    )


def evaluate_interval_recall(
    centers: pd.DataFrame,
    intervals: pd.DataFrame,
    threshold: float,
    *,
    interval_kind: str,
    window_statistic_column: str = "consensus_percentile",
) -> pd.DataFrame:
    """Evaluate any-center and same-width-window recall for intervals."""
    rows: list[dict[str, object]] = []
    for interval in intervals.itertuples(index=False):
        record = interval._asdict()
        subset = centers.loc[
            centers.gene.eq(record["gene"])
            & centers.variant_offset_from_tss_transcription_bp.between(
                int(record["start"]) - 5,
                int(record["end"]) + 5,
            )
        ]
        if subset.empty:
            rows.append(
                {
                    **record,
                    "interval_kind": interval_kind,
                    "covered_centers": 0,
                    "best_center": np.nan,
                    "best_score": np.nan,
                    "recalled_at_threshold": False,
                }
            )
            continue

        best = subset.nlargest(1, "importance_score").iloc[0]
        gene_centers = centers.loc[
            centers.gene.eq(record["gene"])
        ].sort_values("variant_offset_from_tss_transcription_bp")
        window = score_same_width_window(
            gene_centers, subset, window_statistic_column
        )
        rows.append(
            {
                **record,
                "interval_kind": interval_kind,
                "covered_centers": int(len(subset)),
                "best_center": int(
                    best.variant_offset_from_tss_transcription_bp
                ),
                "best_score": float(best.importance_score),
                "recalled_at_threshold": bool(
                    best.importance_score > threshold
                ),
                "window_statistic": window.statistic,
                "same_width_background_windows": window.background_windows,
                "window_percentile_score": window.percentile_score,
                "window_recalled_at_threshold": bool(
                    window.percentile_score > threshold
                ),
                "best_score_statistic": float(
                    best[window_statistic_column]
                ),
                "best_median_group_signed_log2fc": float(
                    best.median_group_signed_log2fc
                ),
                "best_loss_direction_group_fraction": float(
                    best.loss_direction_group_fraction
                ),
            }
        )
    return pd.DataFrame.from_records(rows)


def call_threshold_regions(
    centers: pd.DataFrame,
    threshold: float,
    *,
    score_statistic_column: str = "consensus_percentile",
) -> pd.DataFrame:
    """Join adjacent score-selected centers into regions."""
    rows: list[dict[str, object]] = []
    selected = centers.loc[centers.importance_score > threshold].copy()
    for gene, group in selected.groupby("gene", sort=False):
        group = group.sort_values(
            "variant_offset_from_tss_transcription_bp"
        )
        block = (
            group.variant_offset_from_tss_transcription_bp.diff()
            .fillna(5)
            .gt(4)
            .cumsum()
        )
        for _, region in group.groupby(block, sort=False):
            peak = region.nlargest(1, "importance_score").iloc[0]
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
                    "peak_center": int(
                        peak.variant_offset_from_tss_transcription_bp
                    ),
                    "peak_score": float(peak.importance_score),
                    "peak_score_statistic": float(
                        peak[score_statistic_column]
                    ),
                    "peak_loss_direction_group_fraction": float(
                        peak.loss_direction_group_fraction
                    ),
                }
            )
    result = pd.DataFrame.from_records(rows)
    if result.empty:
        return result
    return result.sort_values(
        ["gene", "peak_score", "region_start"],
        ascending=[True, False, True],
    ).reset_index(drop=True)


def _score_track_groups(tracks: pd.DataFrame) -> pd.DataFrame:
    """Collapse tracks and calibrate effects within gene, view and group."""
    groups = (
        tracks.groupby(
            [*CENTER_KEYS, "score_view", "track_group"], sort=False
        )
        .agg(
            tracks=("track_id", "nunique"),
            group_absolute_log2fc=("median_absolute_log2fc", "median"),
            group_signed_log2fc=("median_signed_log2fc", "median"),
        )
        .reset_index()
    )
    groups["group_effect_percentile"] = groups.groupby(
        ["gene", "score_view", "track_group"], sort=False
    ).group_absolute_log2fc.rank(method="average", pct=True)
    groups["group_loss_direction"] = groups.group_signed_log2fc < 0
    return groups


def _score_view_centers(groups: pd.DataFrame) -> pd.DataFrame:
    """Require three groups and add one-edit spatial support per view."""
    view_centers = (
        groups.groupby([*CENTER_KEYS, "score_view"], sort=False)
        .agg(
            track_groups=("track_group", "nunique"),
            third_highest_group_percentile=(
                "group_effect_percentile",
                lambda values: float(values.nlargest(3).iloc[-1]),
            ),
            loss_direction_group_fraction=("group_loss_direction", "mean"),
            median_group_signed_log2fc=("group_signed_log2fc", "median"),
        )
        .reset_index()
    )
    spatial_parts: list[pd.DataFrame] = []
    for view, part in view_centers.groupby("score_view", sort=False):
        spatial = compute_spatial_median_support(
            part, "third_highest_group_percentile"
        )
        spatial["score_view"] = view
        spatial_parts.append(spatial)
    view_centers = view_centers.merge(
        pd.concat(spatial_parts, ignore_index=True),
        on=[*CENTER_KEYS, "score_view"],
        how="left",
        validate="one_to_one",
    )
    view_centers["view_score"] = (
        100.0
        * view_centers.groupby(
            ["gene", "score_view"], sort=False
        ).spatial_support.rank(method="average", pct=True)
    )
    return view_centers


def _combine_view_scores(
    view_centers: pd.DataFrame, threshold: float
) -> pd.DataFrame:
    """Combine independently ranked output and local-regulatory views."""
    values = [
        "spatial_support",
        "view_score",
        "loss_direction_group_fraction",
        "median_group_signed_log2fc",
    ]
    centers = view_centers.pivot(
        index=CENTER_KEYS,
        columns="score_view",
        values=values,
    )
    centers.columns = [
        f"{field}_{view}" for field, view in centers.columns
    ]
    centers = centers.reset_index()
    centers["max_view_score_raw"] = centers[
        ["view_score_local_regulatory", "view_score_output"]
    ].max(axis=1)
    centers["driving_view"] = np.where(
        centers.view_score_local_regulatory
        >= centers.view_score_output,
        "local_regulatory",
        "output",
    )
    centers["importance_score"] = (
        100.0
        * centers.groupby("gene", sort=False).max_view_score_raw.rank(
            method="average", pct=True
        )
    )
    centers["top_five_percent"] = centers.importance_score > threshold
    local_drives = centers.driving_view.eq("local_regulatory")
    centers["loss_direction_group_fraction"] = np.where(
        local_drives,
        centers.loss_direction_group_fraction_local_regulatory,
        centers.loss_direction_group_fraction_output,
    )
    centers["median_group_signed_log2fc"] = np.where(
        local_drives,
        centers.median_group_signed_log2fc_local_regulatory,
        centers.median_group_signed_log2fc_output,
    )
    return centers


def score_modality_views(
    tracks: pd.DataFrame,
    threshold: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build the frozen two-view score from center-by-track effects."""
    groups = _score_track_groups(tracks)
    view_centers = _score_view_centers(groups)
    centers = _combine_view_scores(view_centers, threshold)
    return groups, view_centers, centers


def evaluate_modality_view_recall(
    centers: pd.DataFrame,
    intervals: pd.DataFrame,
    threshold: float,
    *,
    interval_kind: str,
) -> pd.DataFrame:
    """Audit output and local-regulatory views separately."""
    rows: list[dict[str, object]] = []
    for interval in intervals.itertuples(index=False):
        record = interval._asdict()
        gene_centers = centers.loc[
            centers.gene.eq(record["gene"])
        ].copy()
        subset = gene_centers.loc[
            gene_centers.variant_offset_from_tss_transcription_bp.between(
                int(record["start"]) - 5,
                int(record["end"]) + 5,
            )
        ]
        row: dict[str, object] = {
            **record,
            "interval_kind": interval_kind,
            "covered_centers": int(len(subset)),
        }
        for view in ["local_regulatory", "output"]:
            score_column = f"view_score_{view}"
            statistic_column = f"spatial_support_{view}"
            best = subset.nlargest(1, score_column).iloc[0]
            row[f"{view}_best_center"] = int(
                best.variant_offset_from_tss_transcription_bp
            )
            row[f"{view}_best_score"] = float(best[score_column])
            row[f"{view}_recalled_any_center"] = bool(
                best[score_column] > threshold
            )
            window = score_same_width_window(
                gene_centers, subset, statistic_column
            )
            row[f"{view}_window_percentile_score"] = (
                window.percentile_score
            )
        row["either_view_recalled_any_center"] = bool(
            row["local_regulatory_recalled_any_center"]
            or row["output_recalled_any_center"]
        )
        rows.append(row)
    return pd.DataFrame.from_records(rows)
