"""Reusable effect summaries for the canonical nine-gene and Mdk analyses."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from .harness import require_columns, require_finite, require_unique


DEFAULT_HSC_READOUT_ROLES = (
    "tss_1024bp",
    "tes_3prime_1024bp",
    "gene_body_output_clipped",
    "hsc_observed_peak_1024bp",
)


def strongest_signed_profile(
    path: Path,
    *,
    model_label: str,
    genes: Sequence[str],
    readout_roles: Sequence[str] = DEFAULT_HSC_READOUT_ROLES,
) -> pd.DataFrame:
    """Select the strongest median HSC readout at every eligible scan center.

    Strength is the median absolute effect across the three deterministic
    composition-preserving replacements. The returned sign comes from that
    same readout. This is an effect-size summary, not a significance estimate.
    """

    data = pd.read_parquet(path, filters=[("gene", "in", list(genes))])
    require_columns(
        data,
        [
            "gene",
            "track_id",
            "readout_role",
            "variant_offset_from_tss_transcription_bp",
            "replacement_replicate",
            "log2fc_ratio_of_sums",
        ],
        name=str(path),
    )
    data = data.loc[
        data.track_id.eq("hsc") & data.readout_role.isin(readout_roles)
    ].copy()
    data["absolute_effect"] = data.log2fc_ratio_of_sums.abs()
    by_readout = (
        data.groupby(
            ["gene", "variant_offset_from_tss_transcription_bp", "readout_role"],
            sort=False,
        )
        .agg(
            median_abs_effect=("absolute_effect", "median"),
            median_signed_effect=("log2fc_ratio_of_sums", "median"),
            replacement_replicates=("replacement_replicate", "nunique"),
        )
        .reset_index()
    )
    strongest = (
        by_readout.sort_values(
            [
                "gene",
                "variant_offset_from_tss_transcription_bp",
                "median_abs_effect",
                "readout_role",
            ]
        )
        .groupby(["gene", "variant_offset_from_tss_transcription_bp"], as_index=False)
        .tail(1)
        .copy()
    )
    strongest["model_label"] = model_label
    strongest["signed_effect_smoothed"] = (
        strongest.sort_values(["gene", "variant_offset_from_tss_transcription_bp"])
        .groupby("gene")["median_signed_effect"]
        .transform(
            lambda values: values.rolling(5, center=True, min_periods=1).median()
        )
    )
    require_unique(
        strongest,
        ["model_label", "gene", "variant_offset_from_tss_transcription_bp"],
        name="strongest HSC profile",
    )
    require_finite(
        strongest,
        ["median_abs_effect", "median_signed_effect", "signed_effect_smoothed"],
        name="strongest HSC profile",
    )
    if not strongest.replacement_replicates.eq(3).all():
        raise ValueError(
            "Every retained browser center must contain three replacements"
        )
    return strongest.sort_values(
        ["gene", "model_label", "variant_offset_from_tss_transcription_bp"]
    ).reset_index(drop=True)


def fixed_readout_cell_profile(
    path: Path,
    *,
    model_label: str,
    gene: str,
    readout_role: str,
    cell_groups: Mapping[str, str],
    gene_strand: str,
) -> pd.DataFrame:
    """Aggregate one fixed readout for several fine-tuned cell heads."""

    data = pd.read_parquet(path, filters=[("gene", "==", gene)])
    require_columns(
        data,
        [
            "track_group",
            "track_strand",
            "readout_role",
            "variant_offset_from_tss_transcription_bp",
            "replacement_replicate",
            "log2fc_ratio_of_sums",
        ],
        name=str(path),
    )
    compatible = data.track_strand.fillna(".").astype(str).isin([".", gene_strand])
    data = data.loc[
        data.readout_role.eq(readout_role)
        & data.track_group.isin(cell_groups)
        & compatible
    ].copy()
    profile = (
        data.groupby(
            ["track_group", "variant_offset_from_tss_transcription_bp"], sort=True
        )
        .agg(
            effect=("log2fc_ratio_of_sums", "median"),
            replacements=("replacement_replicate", "nunique"),
        )
        .reset_index()
    )
    profile["model"] = model_label
    profile["cell"] = profile.track_group.map(cell_groups)
    profile["readout_role"] = readout_role
    require_unique(
        profile,
        ["model", "cell", "variant_offset_from_tss_transcription_bp"],
        name=f"{gene} fixed-readout profile",
    )
    require_finite(profile, ["effect"], name=f"{gene} fixed-readout profile")
    if not profile.replacements.eq(3).all():
        raise ValueError(
            "Every retained browser center must contain three replacements"
        )
    return profile


def summarize_cell_windows(
    profile: pd.DataFrame,
    windows: Mapping[str, tuple[int, int]],
) -> pd.DataFrame:
    """Summarize signed and absolute effects for named positional windows."""

    rows: list[pd.DataFrame] = []
    for label, (start, end) in windows.items():
        subset = profile.loc[
            profile.variant_offset_from_tss_transcription_bp.between(start, end)
        ]
        summary = (
            subset.groupby(["model", "cell"], sort=False)
            .agg(
                median_signed_log2fc=("effect", "median"),
                median_abs_log2fc=(
                    "effect",
                    lambda values: float(np.median(np.abs(values))),
                ),
                fraction_centers_negative=(
                    "effect",
                    lambda values: float((values < 0).mean()),
                ),
                centers=("effect", "size"),
            )
            .reset_index()
        )
        summary["cell_rank"] = summary.groupby("model").median_abs_log2fc.rank(
            method="min", ascending=False
        )
        summary.insert(0, "window", label)
        summary.insert(1, "window_start_bp", start)
        summary.insert(2, "window_end_bp", end)
        rows.append(summary)
    result = pd.concat(rows, ignore_index=True)
    require_finite(
        result,
        ["median_signed_log2fc", "median_abs_log2fc", "fraction_centers_negative"],
        name="window summary",
    )
    return result
