"""Build paper-aligned raw and control-adjusted ISM feature matrices."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from grelu.interpret.ism.config import RunConfig
from grelu.interpret.ism.contact import (
    add_contact_derived_features,
    add_contact_effect_fractions,
    classify_contact_responses,
)
from grelu.interpret.ism.errors import ISMUserError
from grelu.interpret.ism.provenance import (
    output_dir,
    prepare_stage_outputs,
    record_stage,
    reject_completed_downstream_stages,
    require_completed_stage,
    sha256_file,
)
from grelu.interpret.ism.schemas import INFERENCE_SUMMARY_SCHEMA


class FeatureError(ISMUserError, RuntimeError):
    """Raised when inference summaries cannot satisfy the feature contract."""


LONG_COLUMNS = [
    "site_id",
    "perturbation_id",
    "perturbation_type",
    "feature_name",
    "feature_level",
    "paper_feature_family",
    "aggregation_group",
    "output_type",
    "resolution",
    "track_id",
    "summary_name",
    "statistic",
    "S_ref",
    "S_alt",
]


def _slug(value: Any) -> str:
    text = re.sub(r"[^A-Za-z0-9]+", "_", str(value).strip()).strip("_")
    return text.lower() or "unknown"


def _feature_name(
    *,
    level: str,
    identity: str,
    summary_name: str,
    statistic: str,
    resolution: int,
) -> str:
    return "__".join(
        (
            level,
            _slug(identity),
            _slug(summary_name),
            _slug(statistic),
            f"r{int(resolution)}",
        )
    )


def _read_summaries(root: Path) -> pd.DataFrame:
    index_path = root / "inference_index.tsv"
    if not index_path.exists():
        raise FeatureError(f"Missing inference index: {index_path}")
    index = pd.read_csv(index_path, sep="\t", low_memory=False)
    required_index = {"relative_path", "sha256", "row_count"}
    missing_index = required_index - set(index.columns)
    if missing_index:
        raise FeatureError(
            f"Inference index is missing columns: {sorted(missing_index)}"
        )
    shards = []
    for row in index.itertuples(index=False):
        path = root / str(row.relative_path)
        if not path.exists():
            raise FeatureError(f"Missing inference shard: {path}")
        if sha256_file(path) != str(row.sha256):
            raise FeatureError(f"Inference shard checksum mismatch: {path}")
        shard = pd.read_parquet(path)
        if len(shard) != int(row.row_count):
            raise FeatureError(
                f"Inference shard row-count mismatch: {path}"
            )
        shards.append(shard)
    if not shards:
        raise FeatureError("Inference index contains no summary shards")
    summaries = pd.concat(shards, ignore_index=True)
    missing = (
        set(INFERENCE_SUMMARY_SCHEMA.required_columns)
        - set(summaries.columns)
    )
    if missing:
        raise FeatureError(
            f"Inference summaries are missing columns: {sorted(missing)}"
        )
    summaries["value"] = pd.to_numeric(
        summaries["value"], errors="coerce"
    )
    return summaries


def _average_orientations(summaries: pd.DataFrame) -> pd.DataFrame:
    keys = [
        column
        for column in INFERENCE_SUMMARY_SCHEMA.required_columns
        if column not in {"orientation", "value"}
    ]
    optional = [
        column
        for column in INFERENCE_SUMMARY_SCHEMA.optional_columns
        if column in summaries.columns
    ]
    grouped = (
        summaries.groupby(keys + optional, dropna=False, sort=False)["value"]
        .mean()
        .reset_index()
    )
    return grouped


def _selected_tracks(track_table: pd.DataFrame) -> pd.DataFrame:
    selected = track_table["used_in_v1_2"]
    if not pd.api.types.is_bool_dtype(selected):
        selected = (
            selected.fillna("")
            .astype(str)
            .str.lower()
            .isin({"true", "1"})
        )
    result = track_table.loc[selected].copy()
    if result.empty:
        raise FeatureError("Track registry contains no selected tracks")
    return result


def _weighted_mean(group: pd.DataFrame) -> float:
    values = group["value"].to_numpy(dtype=float)
    weights = group["aggregation_weight"].to_numpy(dtype=float)
    finite = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not finite.any():
        return float("nan")
    return float(np.average(values[finite], weights=weights[finite]))


def _signal_rows(
    summaries: pd.DataFrame,
    tracks: pd.DataFrame,
) -> pd.DataFrame:
    metadata_columns = [
        "track_id",
        "paper_feature_family",
        "aggregation_group",
        "aggregation_weight",
    ]
    signals = summaries.merge(
        tracks.loc[:, metadata_columns],
        on="track_id",
        how="inner",
        validate="many_to_one",
    )
    signals["feature_level"] = "track"
    signals["feature_name"] = [
        _feature_name(
            level="track",
            identity=track_id,
            summary_name=summary,
            statistic=statistic,
            resolution=resolution,
        )
        for track_id, summary, statistic, resolution in zip(
            signals["track_id"],
            signals["summary_name"],
            signals["statistic"],
            signals["resolution"],
        )
    ]
    group_keys = [
        "site_id",
        "perturbation_id",
        "allele",
        "resolution",
        "summary_name",
        "statistic",
        "paper_feature_family",
        "aggregation_group",
    ]
    output_types_by_group = (
        signals.groupby("aggregation_group", sort=False)["output_type"]
        .agg(lambda values: tuple(sorted(set(values.astype(str)))))
        .to_dict()
    )
    aggregated = (
        signals.groupby(group_keys, dropna=False, sort=False)
        .apply(_weighted_mean, include_groups=False)
        .rename("value")
        .reset_index()
    )
    aggregated["track_id"] = aggregated["aggregation_group"]
    aggregated["aggregation_weight"] = 1.0
    aggregated["feature_level"] = "group"
    aggregated["output_type"] = aggregated["aggregation_group"].map(
        lambda group: (
            output_types_by_group[group][0]
            if len(output_types_by_group[group]) == 1
            else "multi_head"
        )
    )
    aggregated["feature_name"] = [
        _feature_name(
            level="group",
            identity=group,
            summary_name=summary,
            statistic=statistic,
            resolution=resolution,
        )
        for group, summary, statistic, resolution in zip(
            aggregated["aggregation_group"],
            aggregated["summary_name"],
            aggregated["statistic"],
            aggregated["resolution"],
        )
    ]
    composite_definitions = {
        "cohesin_mean": {
            "groups": {"rad21", "smc3"},
            "family": "cohesin_CTCF_binding",
        },
        "enhancer_mean": {
            "groups": {
                "h3k27ac",
                "h3k4me1",
                "h3k4me2",
                "p300",
                "ep300",
                "cbp",
                "crebbp",
            },
            "family": "enhancer_like",
        },
    }
    composite_parts = []
    composite_keys = [
        "site_id",
        "perturbation_id",
        "allele",
        "resolution",
        "summary_name",
        "statistic",
    ]
    for composite_name, definition in composite_definitions.items():
        candidates = signals.loc[
            signals["aggregation_group"].isin(definition["groups"])
        ]
        if candidates.empty:
            continue
        composite = (
            candidates.groupby(
                composite_keys, dropna=False, sort=False
            )
            .apply(_weighted_mean, include_groups=False)
            .rename("value")
            .reset_index()
        )
        composite["paper_feature_family"] = definition["family"]
        composite["aggregation_group"] = composite_name
        composite["track_id"] = composite_name
        composite["aggregation_weight"] = 1.0
        composite["feature_level"] = "group"
        composite["output_type"] = "multi_head"
        composite["feature_name"] = [
            _feature_name(
                level="group",
                identity=composite_name,
                summary_name=summary,
                statistic=statistic,
                resolution=resolution,
            )
            for summary, statistic, resolution in zip(
                composite["summary_name"],
                composite["statistic"],
                composite["resolution"],
            )
        ]
        composite_parts.append(composite)

    columns = list(signals.columns)
    for column in columns:
        if column not in aggregated.columns:
            aggregated[column] = np.nan
    for composite in composite_parts:
        for column in columns:
            if column not in composite.columns:
                composite[column] = np.nan
    combined = pd.concat(
        [
            signals.loc[:, columns],
            aggregated.loc[:, columns],
            *[composite.loc[:, columns] for composite in composite_parts],
        ],
        ignore_index=True,
    )
    duplicated = combined.duplicated(
        ["site_id", "perturbation_id", "feature_name"], keep=False
    )
    if duplicated.any():
        examples = combined.loc[
            duplicated, ["site_id", "perturbation_id", "feature_name"]
        ].head(5)
        raise FeatureError(
            "Feature-name collision after signal aggregation: "
            f"{examples.to_dict(orient='records')}"
        )
    return combined


def _pair_ref_alt(
    signals: pd.DataFrame,
    mutations: pd.DataFrame,
) -> pd.DataFrame:
    identity = [
        "site_id",
        "feature_name",
        "feature_level",
        "paper_feature_family",
        "aggregation_group",
        "output_type",
        "resolution",
        "track_id",
        "summary_name",
        "statistic",
    ]
    ref = signals.loc[signals["allele"].eq("REF"), identity + ["value"]].copy()
    ref = ref.rename(columns={"value": "S_ref"})
    if ref.duplicated(identity).any():
        raise FeatureError("REF summaries are not unique by site and feature")
    alt = signals.loc[
        signals["allele"].eq("ALT"),
        identity + ["perturbation_id", "value"],
    ].copy()
    alt = alt.rename(columns={"value": "S_alt"})
    mutation_types = mutations.loc[
        :, ["perturbation_id", "site_id", "perturbation_type"]
    ].copy()
    if mutation_types["perturbation_id"].duplicated().any():
        raise FeatureError("Mutation perturbation_id values must be unique")
    alt = alt.merge(
        mutation_types,
        on=["perturbation_id", "site_id"],
        how="left",
        validate="many_to_one",
    )
    if alt["perturbation_type"].isna().any():
        raise FeatureError("ALT summaries contain unknown perturbation IDs")
    paired = alt.merge(ref, on=identity, how="left", validate="many_to_one")
    return paired.loc[:, LONG_COLUMNS]


def _compute_effects(table: pd.DataFrame, config: RunConfig) -> pd.DataFrame:
    result = table.copy()
    result["absolute_delta"] = result["S_alt"] - result["S_ref"]
    result["delta_magnitude"] = result["absolute_delta"].abs()
    pseudocount = config.features.pseudocount
    result["log2fc"] = np.log2(
        (result["S_alt"] + pseudocount)
        / (result["S_ref"] + pseudocount)
    )
    already_transformed = result["summary_name"].isin(
        {"contact_dlr_50_500kb_vs_0_20kb"}
    )
    result.loc[already_transformed, "log2fc"] = result.loc[
        already_transformed, "absolute_delta"
    ]
    result["effect_available"] = (
        np.isfinite(result["S_ref"]) & np.isfinite(result["S_alt"])
    )
    return result


def _add_ctcf_effect_aggregates(
    table: pd.DataFrame,
) -> pd.DataFrame:
    ctcf = table.loc[
        table["feature_level"].eq("track")
        & table["aggregation_group"].eq("ctcf")
    ].copy()
    if ctcf.empty:
        return table
    keys = [
        "site_id",
        "perturbation_id",
        "perturbation_type",
        "output_type",
        "resolution",
        "summary_name",
        "statistic",
    ]
    records = []
    for key, group in ctcf.groupby(keys, dropna=False, sort=False):
        base = dict(zip(keys, key))
        effects = group["log2fc"].to_numpy(dtype=float)
        finite = effects[np.isfinite(effects)]
        for label, reducer in (
            ("ctcf_min", np.nanmin),
            ("ctcf_mean", np.nanmean),
        ):
            score = float(reducer(finite)) if finite.size else np.nan
            record = {
                **base,
                "feature_name": _feature_name(
                    level="derived",
                    identity=label,
                    summary_name=base["summary_name"],
                    statistic=base["statistic"],
                    resolution=base["resolution"],
                ),
                "feature_level": "derived",
                "paper_feature_family": "cohesin_CTCF_binding",
                "aggregation_group": label,
                "track_id": label,
                "S_ref": np.nan,
                "S_alt": score,
                "absolute_delta": score,
                "delta_magnitude": abs(score),
                "log2fc": score,
                "effect_available": np.isfinite(score),
            }
            records.append(record)
        fraction = float(np.mean(finite < 0)) if finite.size else np.nan
        records.append(
            {
                **base,
                "feature_name": _feature_name(
                    level="derived",
                    identity="ctcf_negative_fraction",
                    summary_name=base["summary_name"],
                    statistic=base["statistic"],
                    resolution=base["resolution"],
                ),
                "feature_level": "derived",
                "paper_feature_family": "cohesin_CTCF_binding",
                "aggregation_group": "ctcf_negative_fraction",
                "track_id": "ctcf_negative_fraction",
                "S_ref": np.nan,
                "S_alt": fraction,
                "absolute_delta": fraction,
                "delta_magnitude": abs(fraction),
                "log2fc": fraction,
                "effect_available": np.isfinite(fraction),
            }
        )
    additions = pd.DataFrame.from_records(records)
    return pd.concat([table, additions], ignore_index=True)


def _primary_mutations(mutations: pd.DataFrame) -> pd.DataFrame:
    primary = mutations.loc[
        mutations["perturbation_type"].eq("ctcf_motif_disruption"),
        ["site_id", "perturbation_id"],
    ].copy()
    if primary["site_id"].duplicated().any():
        raise FeatureError(
            "Each site may have at most one CTCF motif disruption"
        )
    return primary


def _control_calibration(
    table: pd.DataFrame,
    config: RunConfig,
) -> pd.DataFrame:
    controls = table.loc[
        table["perturbation_type"].eq("local_non_motif_control")
    ]
    summary = (
        controls.groupby(["site_id", "feature_name"], sort=False)["log2fc"]
        .agg(
            control_median_score="median",
            control_mad=lambda values: float(
                np.nanmedian(
                    np.abs(
                        values.to_numpy(dtype=float)
                        - np.nanmedian(values.to_numpy(dtype=float))
                    )
                )
            ),
        )
        .reset_index()
    )
    result = table.merge(
        summary,
        on=["site_id", "feature_name"],
        how="left",
        validate="many_to_one",
    )
    result["control_adjusted_score"] = (
        result["log2fc"] - result["control_median_score"]
    )
    result["motif_vs_control_z"] = (
        result["control_adjusted_score"]
        / (result["control_mad"] + np.finfo(float).eps)
    )
    return result


def _ref_feature_table(
    signals: pd.DataFrame,
    sites: pd.DataFrame,
) -> pd.DataFrame:
    ref = signals.loc[
        signals["allele"].eq("REF"), ["site_id", "feature_name", "value"]
    ].rename(columns={"value": "S_ref"})
    return (
        sites.loc[:, ["site_id"]]
        .merge(ref, on="site_id", how="left")
        .drop_duplicates(["site_id", "feature_name"])
    )


def _wide(
    values: pd.DataFrame,
    *,
    value_columns: Iterable[str],
    sites: pd.DataFrame,
) -> pd.DataFrame:
    pieces = [sites.loc[:, ["site_id"]].drop_duplicates().set_index("site_id")]
    for value_column in value_columns:
        pivot = values.pivot(
            index="site_id",
            columns="feature_name",
            values=value_column,
        )
        pivot.columns = [
            f"{feature}__{value_column}" for feature in pivot.columns
        ]
        pieces.append(pivot)
    return pd.concat(pieces, axis=1).reset_index()


def _registry(table: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "feature_name",
        "feature_level",
        "paper_feature_family",
        "aggregation_group",
        "output_type",
        "resolution",
        "summary_name",
        "statistic",
    ]
    result = table.loc[:, columns].drop_duplicates().copy()
    result["biological_fingerprint_feature"] = ~result[
        "paper_feature_family"
    ].eq("QC_metadata")
    result["leakage_control_feature"] = (
        result["biological_fingerprint_feature"]
        & ~result["aggregation_group"].isin(
            {"ctcf", "ctcf_min", "ctcf_mean", "ctcf_negative_fraction"}
        )
    )
    result["primary_clustering_feature"] = True
    result["interpretation_only"] = False
    return result.sort_values("feature_name").reset_index(drop=True)


def run(
    *,
    config: RunConfig,
    config_path: str | Path,
    allow_provisional_labels: bool,
    overwrite_stage: bool,
) -> None:
    """Create raw, contact-derived, and control-adjusted feature matrices."""

    root = output_dir(config)
    paths = {
        "sites": root / "site_metadata.tsv",
        "mutations": root / "mutation_table.tsv",
        "tracks": root / "track_selection_table.tsv",
        "index": root / "inference_index.tsv",
        "raw_signal": root / "raw_signal_feature_matrix.tsv",
        "raw_delta": root / "raw_delta_feature_matrix.tsv",
        "control_summary": root / "control_summary_matrix.tsv",
        "control_adjusted": root / "control_adjusted_feature_matrix.tsv",
        "long": root / "feature_long.parquet",
        "registry": root / "feature_registry.tsv",
        "contact_response": root / "contact_response_type.tsv",
    }
    require_completed_stage(config, config_path, "infer")
    for key in ("sites", "mutations", "tracks", "index"):
        if not paths[key].exists():
            raise FeatureError(f"Required inference artifact is missing: {paths[key]}")
    if overwrite_stage:
        reject_completed_downstream_stages(
            config, config_path, ("analyze", "report")
        )
    output_paths = list(paths.values())[4:]
    prepare_stage_outputs(output_paths, overwrite_stage)

    sites = pd.read_csv(paths["sites"], sep="\t", low_memory=False)
    statuses = set(sites["label_status"].astype(str))
    if statuses != {"canonical"} and not allow_provisional_labels:
        raise FeatureError(
            "Features for provisional labels require --allow-provisional-labels"
        )
    mutations = pd.read_csv(paths["mutations"], sep="\t", low_memory=False)
    tracks = _selected_tracks(
        pd.read_csv(paths["tracks"], sep="\t", low_memory=False)
    )
    summaries = _average_orientations(_read_summaries(root))
    signals = _signal_rows(summaries, tracks)
    paired = _pair_ref_alt(signals, mutations)
    paired = add_contact_derived_features(paired, config)
    effects = _compute_effects(paired, config)
    effects = _add_ctcf_effect_aggregates(effects)
    effects = add_contact_effect_fractions(effects)
    effects = _control_calibration(effects, config)

    primary = _primary_mutations(mutations)
    motif = effects.merge(
        primary,
        on=["site_id", "perturbation_id"],
        how="inner",
        validate="many_to_one",
    )
    ref = _ref_feature_table(signals, sites)
    thresholds = (
        ref.groupby("feature_name", sort=False)["S_ref"]
        .quantile(config.features.low_ref_quantile)
        .rename("low_ref_threshold")
        .reset_index()
    )
    ref = ref.merge(thresholds, on="feature_name", how="left")
    ref["low_ref_signal_flag"] = ref["S_ref"] <= ref["low_ref_threshold"]
    primary_signal = ref.merge(
        motif.loc[:, ["site_id", "feature_name", "S_alt"]],
        on=["site_id", "feature_name"],
        how="left",
        validate="one_to_one",
    )
    raw_signal = _wide(
        primary_signal,
        value_columns=("S_ref", "S_alt", "low_ref_signal_flag"),
        sites=sites,
    )
    raw_delta = _wide(
        motif,
        value_columns=(
            "absolute_delta",
            "delta_magnitude",
            "log2fc",
            "effect_available",
        ),
        sites=sites,
    )
    control_summary = _wide(
        motif,
        value_columns=("control_median_score", "control_mad"),
        sites=sites,
    )
    control_adjusted = _wide(
        motif,
        value_columns=("control_adjusted_score", "motif_vs_control_z"),
        sites=sites,
    )
    raw_signal.to_csv(paths["raw_signal"], sep="\t", index=False)
    raw_delta.to_csv(paths["raw_delta"], sep="\t", index=False)
    control_summary.to_csv(
        paths["control_summary"], sep="\t", index=False
    )
    control_adjusted.to_csv(
        paths["control_adjusted"], sep="\t", index=False
    )
    effects.to_parquet(paths["long"], index=False)
    registry = _registry(
        pd.concat([signals, effects], ignore_index=True, sort=False)
    )
    registry.to_csv(paths["registry"], sep="\t", index=False)
    contact_response = classify_contact_responses(effects)
    contact_response.to_csv(
        paths["contact_response"], sep="\t", index=False
    )

    label_status = "canonical" if statuses == {"canonical"} else "provisional"
    record_stage(
        config=config,
        config_path=config_path,
        stage="features",
        inputs=[
            paths["sites"],
            paths["mutations"],
            paths["tracks"],
            paths["index"],
        ],
        outputs=output_paths,
        label_status=label_status,
        metadata={
            "orientation_policy": "aligned_mean_before_ref_alt_effect",
            "feature_count": int(registry["feature_name"].nunique()),
            "primary_mutation_site_count": int(primary["site_id"].nunique()),
            "control_effect": "log2fc",
            "contact_response_excluded_from_clustering": True,
            "population_raw_tensor_loaded": False,
        },
    )
