"""Contact-map feature derivation and response interpretation."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from grelu.interpret.ism.config import RunConfig


def _derived_name(prefix: str, summary: str) -> str:
    return f"{prefix}__{summary}__mean__r2048"


def _ratio_rows(
    table: pd.DataFrame,
    *,
    numerator: str,
    denominator: str,
    output_summary: str,
    config: RunConfig,
    logarithmic: bool,
) -> pd.DataFrame:
    identity = [
        "site_id",
        "perturbation_id",
        "perturbation_type",
        "feature_level",
        "aggregation_group",
        "paper_feature_family",
        "track_id",
    ]
    subset = table.loc[
        table["summary_name"].isin([numerator, denominator])
        & table["output_type"].eq("contact_maps")
    ].copy()
    if subset.empty:
        return subset
    numerator_rows = subset.loc[
        subset["summary_name"].eq(numerator),
        identity + ["S_ref", "S_alt"],
    ].rename(
        columns={
            "S_ref": "numerator_ref",
            "S_alt": "numerator_alt",
        }
    )
    denominator_rows = subset.loc[
        subset["summary_name"].eq(denominator),
        identity + ["S_ref", "S_alt"],
    ].rename(
        columns={
            "S_ref": "denominator_ref",
            "S_alt": "denominator_alt",
        }
    )
    if numerator_rows.empty or denominator_rows.empty:
        return pd.DataFrame()
    result = numerator_rows.merge(
        denominator_rows,
        on=identity,
        how="inner",
        validate="one_to_one",
    )
    pseudocount = config.features.pseudocount
    if logarithmic:
        result["S_ref"] = np.log2(
            (result["numerator_ref"] + pseudocount)
            / (result["denominator_ref"] + pseudocount)
        )
        result["S_alt"] = np.log2(
            (result["numerator_alt"] + pseudocount)
            / (result["denominator_alt"] + pseudocount)
        )
    else:
        result["S_ref"] = (
            result["numerator_ref"]
            / (result["denominator_ref"] + pseudocount)
        )
        result["S_alt"] = (
            result["numerator_alt"]
            / (result["denominator_alt"] + pseudocount)
        )
    result["output_type"] = "contact_maps"
    result["resolution"] = config.features.contact_bin_bp
    result["summary_name"] = output_summary
    result["statistic"] = "mean"
    result["feature_name"] = [
        _derived_name(
            f"{level}__{track_id if level == 'track' else group}",
            output_summary,
        )
        for level, track_id, group in zip(
            result["feature_level"],
            result["track_id"],
            result["aggregation_group"],
        )
    ]
    return result.loc[:, table.columns]


def add_contact_derived_features(
    feature_long: pd.DataFrame,
    config: RunConfig,
) -> pd.DataFrame:
    """Add relative cross-boundary and distal/local ratio features."""

    additions = [
        _ratio_rows(
            feature_long,
            numerator="contact_cross_10_50kb",
            denominator="contact_local_total_10_50kb",
            output_summary="contact_relative_cross_10_50kb",
            config=config,
            logarithmic=False,
        ),
        _ratio_rows(
            feature_long,
            numerator="contact_anchor_distal_50_500kb",
            denominator="contact_anchor_local_0_20kb",
            output_summary="contact_dlr_50_500kb_vs_0_20kb",
            config=config,
            logarithmic=True,
        ),
    ]
    additions = [addition for addition in additions if not addition.empty]
    if not additions:
        return feature_long
    return pd.concat([feature_long, *additions], ignore_index=True)


def add_contact_effect_fractions(feature_long: pd.DataFrame) -> pd.DataFrame:
    """Add per-summary positive/negative fractions across contact channels."""

    tracks = feature_long.loc[
        feature_long["output_type"].eq("contact_maps")
        & feature_long["feature_level"].eq("track")
    ].copy()
    if tracks.empty:
        return feature_long
    keys = [
        "site_id",
        "perturbation_id",
        "perturbation_type",
        "output_type",
        "resolution",
        "summary_name",
        "statistic",
    ]
    rows = []
    for key, group in tracks.groupby(keys, dropna=False, sort=False):
        base = dict(zip(keys, key))
        effects = group["log2fc"].to_numpy(dtype=float)
        finite = np.isfinite(effects)
        for direction, predicate in (
            ("positive", effects > 0),
            ("negative", effects < 0),
        ):
            fraction = (
                float(np.mean(predicate[finite])) if finite.any() else np.nan
            )
            identity = f"contact_{direction}_fraction"
            rows.append(
                {
                    **base,
                    "feature_name": _derived_name(
                        f"derived__{identity}", base["summary_name"]
                    ),
                    "feature_level": "derived",
                    "paper_feature_family": "chromatin_architecture_contact",
                    "aggregation_group": identity,
                    "track_id": identity,
                    "S_ref": np.nan,
                    "S_alt": fraction,
                    "absolute_delta": fraction,
                    "delta_magnitude": abs(fraction),
                    "log2fc": fraction,
                    "effect_available": np.isfinite(fraction),
                }
            )
    return pd.concat(
        [feature_long, pd.DataFrame.from_records(rows)],
        ignore_index=True,
    )


def _first_effect(
    rows: pd.DataFrame,
    summaries: Iterable[str],
) -> float:
    for summary in summaries:
        values = rows.loc[
            rows["summary_name"].eq(summary), "control_adjusted_score"
        ].dropna()
        if not values.empty:
            return float(values.iloc[0])
    return float("nan")


def classify_contact_responses(feature_long: pd.DataFrame) -> pd.DataFrame:
    """Classify motif responses without exposing the label to clustering."""

    motif = feature_long.loc[
        feature_long["perturbation_type"].eq("ctcf_motif_disruption")
        & feature_long["feature_level"].eq("group")
        & feature_long["aggregation_group"].eq("contact_architecture")
    ].copy()
    columns = [
        "site_id",
        "contact_response_type",
        "contact_cross_adjusted",
        "contact_local_adjusted",
        "contact_dlr_adjusted",
        "contact_apa_adjusted",
        "included_in_primary_clustering",
    ]
    rows = []
    for site_id, group in motif.groupby("site_id", sort=True):
        cross = _first_effect(
            group,
            ("contact_relative_cross_10_50kb", "contact_cross_10_50kb"),
        )
        local = _first_effect(
            group,
            ("contact_local_total_10_50kb", "contact_anchor_local_0_20kb"),
        )
        dlr = _first_effect(
            group, ("contact_dlr_50_500kb_vs_0_20kb",)
        )
        apa = _first_effect(
            group,
            ("contact_apa_radius_5_bins", "contact_apa_radius_12_bins"),
        )
        finite = np.asarray([cross, local, dlr, apa], dtype=float)
        if not np.isfinite(finite).any():
            response = "unavailable"
        elif np.nanmax(np.abs(finite)) <= np.finfo(float).eps:
            response = "no_clear_contact_response"
        elif np.isfinite(cross) and cross > 0 and (
            not np.isfinite(local) or local < 0
        ):
            response = "cross_boundary_increase_boundary_loss_like"
        elif np.isfinite(cross) and cross < 0:
            response = "cross_boundary_decrease_anchor_loss_like"
        elif np.isfinite(dlr) and dlr > 0:
            response = "distal_relative_gain_decompaction_like"
        elif np.isfinite(apa) and apa < 0:
            response = "local_anchor_contact_loss_like"
        else:
            response = "mixed_contact_response"
        rows.append(
            {
                "site_id": site_id,
                "contact_response_type": response,
                "contact_cross_adjusted": cross,
                "contact_local_adjusted": local,
                "contact_dlr_adjusted": dlr,
                "contact_apa_adjusted": apa,
                "included_in_primary_clustering": False,
            }
        )
    return pd.DataFrame.from_records(rows, columns=columns)
