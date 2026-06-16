#!/usr/bin/env python
"""Cluster a real AlphaGenome CTCF perturbation population without full outputs.

The script consumes an existing ref/alt population memmap run, streams only
centered slices, and compares three clustering representations:

1. all non-padding tracks in a centered 4 kb window;
2. output-type-balanced responsive tracks across multiple windows;
3. biologically interpretable track groups across multiple windows.

Optional contact summaries add a fourth biological-plus-contact
representation. No AlphaGenome prediction tensor is written by this script.
"""

from __future__ import annotations

import argparse
import html
import json
import math
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd


WINDOW_WIDTHS_BP = (1_024, 4_096, 20_000, 100_000)
EFFECT_METRICS = ("signed_mean", "absolute_mean", "peak_log2fc")
SENTINEL_TARGETS = (
    "CTCF",
    "RAD21",
    "SMC3",
    "POLR2A",
    "POLR2B",
    "AFF4",
    "BRD4",
    "MED1",
    "EP300",
    "H3K27AC",
    "H3K4ME1",
    "H3K4ME3",
    "H3K27ME3",
    "H3K9ME3",
)
CONTACT_FEATURES = (
    "delta_cross_contact_mean",
    "delta_abs_cross_contact_mean",
    "delta_boundary_strength_proxy",
)


@dataclass(frozen=True)
class RepresentationResult:
    name: str
    selected_k: int
    silhouette: float
    stability_ari: float
    min_cluster_size: int
    n_features_before_qc: int
    n_features_after_qc: int
    n_pcs: int


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--sensitivity-dir", required=True)
    parser.add_argument("--source-sites", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--contact-summary", default=None)
    parser.add_argument("--top-per-output-type", type=int, default=32)
    parser.add_argument("--sentinel-per-target", type=int, default=8)
    parser.add_argument("--site-chunk-size", type=int, default=1)
    parser.add_argument("--stability-iterations", type=int, default=30)
    parser.add_argument("--random-seed", type=int, default=20260610)
    parser.add_argument("--max-sites", type=int, default=None)
    return parser.parse_args()


def _target(metadata: pd.DataFrame) -> pd.Series:
    tf = metadata["transcription_factor"].fillna("").astype(str)
    histone = metadata["histone_mark"].fillna("").astype(str)
    fallback = metadata["track_name"].fillna("").astype(str)
    return tf.where(tf.ne(""), histone.where(histone.ne(""), fallback))


def _is_padding(metadata: pd.DataFrame) -> pd.Series:
    names = metadata["track_name"].fillna("").astype(str).str.casefold()
    return names.str.contains("padding", regex=False)


def _load_site_metadata(
    run_dir: Path,
    source_sites_path: Path,
) -> pd.DataFrame:
    variants = pd.read_csv(run_dir / "variants.tsv", sep="\t", low_memory=False)
    source = pd.read_csv(source_sites_path, sep="\t", low_memory=False)
    keep = [
        column
        for column in (
            "site_id",
            "name",
            "peak_score",
            "motif_score",
            "rpeak_ubiquity",
            "cCRE",
            "matched_seq",
            "strand",
        )
        if column in source.columns
    ]
    if "site_id" not in keep:
        raise ValueError("source sites must contain site_id")
    result = variants.merge(
        source[keep],
        on="site_id",
        how="left",
        validate="one_to_one",
        suffixes=("", "_source"),
    )
    if result["site_id"].duplicated().any():
        raise ValueError("site_id is not unique after source annotation")
    if "rpeak_ubiquity" in result.columns:
        parsed = (
            result["rpeak_ubiquity"]
            .fillna("")
            .astype(str)
            .str.extract(r"^\s*(\d+)\s*/\s*(\d+)\s*$")
        )
        result["rpeak_count"] = pd.to_numeric(parsed[0], errors="coerce")
        result["rpeak_fraction"] = (
            pd.to_numeric(parsed[0], errors="coerce")
            / pd.to_numeric(parsed[1], errors="coerce")
        )
    result["has_ccre"] = (
        result.get("cCRE", pd.Series("", index=result.index))
        .fillna("")
        .astype(str)
        .str.strip()
        .ne("")
    )
    return result


def _load_sensitivity_table(path: Path) -> pd.DataFrame:
    table = pd.read_csv(path, sep="\t", low_memory=False)
    required = {
        "global_track_index",
        "output_type",
        "full_100kb__recommended_no_padding_rank",
        "full_100kb__rank_within_output_type",
    }
    missing = required - set(table.columns)
    if missing:
        raise ValueError(f"sensitivity table missing columns: {sorted(missing)}")
    table["global_track_index"] = pd.to_numeric(
        table["global_track_index"], errors="raise"
    ).astype(int)
    return table


def _select_tracks(
    metadata: pd.DataFrame,
    sensitivity: pd.DataFrame,
    *,
    top_per_output_type: int,
    sentinel_per_target: int,
) -> pd.DataFrame:
    ranked = sensitivity.copy()
    ranked["target"] = _target(ranked)
    ranked["selection_reason"] = ""
    balanced = (
        ranked["full_100kb__rank_within_output_type"]
        .astype(int)
        .le(top_per_output_type)
    )
    ranked.loc[balanced, "selection_reason"] = "output_type_balanced"

    target_upper = ranked["target"].str.upper()
    for target in SENTINEL_TARGETS:
        candidates = ranked.loc[target_upper.eq(target)].sort_values(
            "full_100kb__recommended_no_padding_rank", kind="stable"
        )
        selected = candidates.head(sentinel_per_target).index
        current = ranked.loc[selected, "selection_reason"]
        ranked.loc[selected, "selection_reason"] = current.map(
            lambda value: (
                f"{value},sentinel:{target}"
                if value
                else f"sentinel:{target}"
            )
        )

    selected = ranked.loc[ranked["selection_reason"].ne("")].copy()
    selected = selected.sort_values(
        ["output_type", "full_100kb__rank_within_output_type"],
        kind="stable",
    )
    selected["selected_for_balanced_representation"] = True
    all_meta = metadata.copy()
    all_meta["target"] = _target(all_meta)
    all_meta["is_padding"] = _is_padding(all_meta)
    result = all_meta.merge(
        selected[
            [
                "global_track_index",
                "selection_reason",
                "full_100kb__recommended_no_padding_rank",
                "full_100kb__rank_within_output_type",
                "selected_for_balanced_representation",
            ]
        ],
        on="global_track_index",
        how="left",
        validate="one_to_one",
    )
    result["selected_for_balanced_representation"] = result[
        "selected_for_balanced_representation"
    ].fillna(False)
    result["selection_reason"] = result["selection_reason"].fillna("")
    return result


def _biological_groups(metadata: pd.DataFrame) -> dict[str, np.ndarray]:
    output = metadata["output_type"].fillna("").astype(str)
    target = metadata["target"].fillna("").astype(str).str.upper()
    nonpadding = ~metadata["is_padding"]
    masks = {
        "CTCF": output.eq("chip_tf") & target.eq("CTCF"),
        "RAD21": output.eq("chip_tf") & target.eq("RAD21"),
        "SMC3": output.eq("chip_tf") & target.eq("SMC3"),
        "transcription_machinery": output.eq("chip_tf")
        & target.isin(["POLR2A", "POLR2B", "AFF4", "BRD4", "MED1"]),
        "enhancer_like": (
            output.eq("chip_histone")
            & target.isin(["H3K27AC", "H3K4ME1", "H3K4ME2"])
        )
        | (output.eq("chip_tf") & target.isin(["EP300", "P300", "CREBBP"])),
        "promoter_like": output.eq("chip_histone") & target.eq("H3K4ME3"),
        "repressive": output.eq("chip_histone")
        & target.isin(["H3K27ME3", "H3K9ME3"]),
        "ATAC": output.eq("atac"),
        "DNase": output.eq("dnase"),
        "CAGE": output.eq("cage"),
        "RNA_seq": output.eq("rna_seq"),
    }
    return {
        name: metadata.loc[mask & nonpadding, "global_track_index"]
        .astype(int)
        .to_numpy()
        for name, mask in masks.items()
        if (mask & nonpadding).any()
    }


def _window_masks(offsets: np.ndarray) -> dict[int, np.ndarray]:
    return {
        width: np.abs(offsets) <= width / 2
        for width in WINDOW_WIDTHS_BP
    }


def _track_effect_metrics(
    ref_values: np.ndarray,
    alt_values: np.ndarray,
) -> dict[str, np.ndarray]:
    ref_nonnegative = np.maximum(ref_values, 0.0)
    alt_nonnegative = np.maximum(alt_values, 0.0)
    log_delta = np.log2(1.0 + alt_nonnegative) - np.log2(
        1.0 + ref_nonnegative
    )
    return {
        "signed_mean": log_delta.mean(axis=1),
        "absolute_mean": np.abs(log_delta).mean(axis=1),
        "peak_log2fc": np.log2(
            (1.0 + alt_nonnegative.max(axis=1))
            / (1.0 + ref_nonnegative.max(axis=1))
        ),
    }


def _feature_names_for_tracks(
    tracks: pd.DataFrame,
    windows: Sequence[int],
    metrics: Sequence[str],
) -> list[str]:
    names = []
    indexed = tracks.set_index("global_track_index")
    for width in windows:
        for track_index in tracks["global_track_index"].astype(int):
            row = indexed.loc[track_index]
            target = str(row["target"]).replace("\t", " ").replace("|", "/")
            biosample = (
                str(row.get("biosample_name", ""))
                .replace("\t", " ")
                .replace("|", "/")
            )
            prefix = (
                f"track:{track_index}:{row['output_type']}:{target}:"
                f"{biosample}:w{width}"
            )
            names.extend(f"{prefix}:{metric}" for metric in metrics)
    return names


def _feature_names_for_groups(
    groups: dict[str, np.ndarray],
    windows: Sequence[int],
    metrics: Sequence[str],
) -> list[str]:
    return [
        f"group:{group}:w{width}:{metric}"
        for width in windows
        for group in groups
        for metric in metrics
    ]


def _extract_feature_matrices(
    run_dir: Path,
    track_selection: pd.DataFrame,
    site_metadata: pd.DataFrame,
    *,
    site_chunk_size: int,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    ref = np.load(run_dir / "all_track_ref_window.npy", mmap_mode="r")
    alt = np.load(run_dir / "all_track_alt_window.npy", mmap_mode="r")
    if ref.shape != alt.shape or ref.ndim != 3:
        raise ValueError(f"invalid ref/alt shapes: {ref.shape} and {alt.shape}")
    if len(site_metadata) > ref.shape[0]:
        raise ValueError(
            f"site rows {len(site_metadata)} exceed arrays {ref.shape[0]}"
        )
    bins = pd.read_csv(run_dir / "all_track_window_bins.tsv", sep="\t")
    offsets = bins["offset_bp"].to_numpy(dtype=float)
    if len(offsets) != ref.shape[2]:
        raise ValueError("bin metadata does not match prediction arrays")
    masks = _window_masks(offsets)
    largest_width = max(WINDOW_WIDTHS_BP)
    largest_indices = np.flatnonzero(masks[largest_width])
    largest_start = int(largest_indices[0])
    largest_stop = int(largest_indices[-1]) + 1
    local_offsets = offsets[largest_start:largest_stop]
    local_masks = {
        width: np.abs(local_offsets) <= width / 2
        for width in WINDOW_WIDTHS_BP
    }

    balanced_tracks = track_selection.loc[
        track_selection["selected_for_balanced_representation"]
    ].copy()
    balanced_indices = balanced_tracks["global_track_index"].astype(int).to_numpy()
    nonpadding_tracks = track_selection.loc[~track_selection["is_padding"]].copy()
    nonpadding_indices = nonpadding_tracks["global_track_index"].astype(int).to_numpy()
    groups = _biological_groups(track_selection)

    balanced_names = _feature_names_for_tracks(
        balanced_tracks, WINDOW_WIDTHS_BP, EFFECT_METRICS
    )
    group_names = _feature_names_for_groups(
        groups, WINDOW_WIDTHS_BP, EFFECT_METRICS
    )
    all_names = [
        f"all_track:{track_index}:w4096:{metric}"
        for track_index in nonpadding_indices
        for metric in ("signed_mean", "absolute_mean")
    ]

    n_sites = len(site_metadata)
    balanced_matrix = np.empty(
        (n_sites, len(balanced_names)), dtype=np.float32
    )
    group_matrix = np.empty((n_sites, len(group_names)), dtype=np.float32)
    all_matrix = np.empty((n_sites, len(all_names)), dtype=np.float32)
    site_effect_rows: list[dict[str, float | str]] = []

    chunk_size = max(1, int(site_chunk_size))
    for chunk_start in range(0, n_sites, chunk_size):
        chunk_stop = min(chunk_start + chunk_size, n_sites)
        for site_index in range(chunk_start, chunk_stop):
            ref_site = np.asarray(
                ref[site_index, :, largest_start:largest_stop],
                dtype=np.float32,
            )
            alt_site = np.asarray(
                alt[site_index, :, largest_start:largest_stop],
                dtype=np.float32,
            )

            balanced_values: list[np.ndarray] = []
            group_values: list[float] = []
            for width in WINDOW_WIDTHS_BP:
                mask = local_masks[width]
                selected_metrics = _track_effect_metrics(
                    ref_site[balanced_indices][:, mask],
                    alt_site[balanced_indices][:, mask],
                )
                for track_position in range(len(balanced_indices)):
                    for metric in EFFECT_METRICS:
                        balanced_values.append(
                            np.asarray(
                                selected_metrics[metric][track_position],
                                dtype=np.float32,
                            )
                        )

                all_track_metrics = _track_effect_metrics(
                    ref_site[:, mask], alt_site[:, mask]
                )
                for group_indices in groups.values():
                    for metric in EFFECT_METRICS:
                        group_values.append(
                            float(
                                np.nanmedian(
                                    all_track_metrics[metric][group_indices]
                                )
                            )
                        )

                if width == 4_096:
                    for track_position, track_index in enumerate(
                        nonpadding_indices
                    ):
                        all_matrix[
                            site_index,
                            2 * track_position,
                        ] = all_track_metrics["signed_mean"][track_index]
                        all_matrix[
                            site_index,
                            2 * track_position + 1,
                        ] = all_track_metrics["absolute_mean"][track_index]

            balanced_matrix[site_index] = np.asarray(
                balanced_values, dtype=np.float32
            )
            group_matrix[site_index] = np.asarray(
                group_values, dtype=np.float32
            )
            local_effect = _track_effect_metrics(
                ref_site[:, local_masks[4_096]],
                alt_site[:, local_masks[4_096]],
            )
            site_effect_rows.append(
                {
                    "site_id": str(site_metadata.iloc[site_index]["site_id"]),
                    "all_track_4kb_absolute_mean": float(
                        np.nanmean(local_effect["absolute_mean"][nonpadding_indices])
                    ),
                    "all_track_4kb_signed_mean": float(
                        np.nanmean(local_effect["signed_mean"][nonpadding_indices])
                    ),
                    "ctcf_4kb_signed_median": float(
                        np.nanmedian(
                            local_effect["signed_mean"][groups["CTCF"]]
                        )
                    ),
                    "ctcf_4kb_absolute_median": float(
                        np.nanmedian(
                            local_effect["absolute_mean"][groups["CTCF"]]
                        )
                    ),
                }
            )
        print(f"extracted sites {chunk_start}:{chunk_stop} / {n_sites}", flush=True)

    site_ids = site_metadata["site_id"].astype(str).to_numpy()
    matrices = {
        "all_tracks_4kb": pd.DataFrame(all_matrix, columns=all_names).assign(
            site_id=site_ids
        ),
        "balanced_tracks": pd.DataFrame(
            balanced_matrix, columns=balanced_names
        ).assign(site_id=site_ids),
        "biological_groups": pd.DataFrame(
            group_matrix, columns=group_names
        ).assign(site_id=site_ids),
    }
    for key, table in matrices.items():
        matrices[key] = table[["site_id", *[c for c in table if c != "site_id"]]]
    return matrices, pd.DataFrame.from_records(site_effect_rows)


def _load_contact_features(path: Path, site_ids: Sequence[str]) -> pd.DataFrame:
    contact = pd.read_csv(path, sep="\t", low_memory=False)
    missing = {"site_id", *CONTACT_FEATURES} - set(contact.columns)
    if missing:
        raise ValueError(f"contact summary missing columns: {sorted(missing)}")
    if contact["site_id"].duplicated().any():
        raise ValueError("contact summary has duplicate site_id rows")
    result = pd.DataFrame({"site_id": list(site_ids)}).merge(
        contact[["site_id", *CONTACT_FEATURES]],
        on="site_id",
        how="left",
        validate="one_to_one",
    )
    if result[list(CONTACT_FEATURES)].isna().any().any():
        missing_ids = result.loc[
            result[list(CONTACT_FEATURES)].isna().any(axis=1), "site_id"
        ].head(10)
        raise ValueError(
            f"contact summary lacks complete rows; examples={missing_ids.tolist()}"
        )
    return result


def _qc_and_scale(
    matrix: pd.DataFrame,
) -> tuple[np.ndarray, list[str], pd.DataFrame]:
    from sklearn.preprocessing import RobustScaler

    columns = [column for column in matrix.columns if column != "site_id"]
    rows = []
    retained = []
    for column in columns:
        values = pd.to_numeric(matrix[column], errors="coerce").replace(
            [np.inf, -np.inf], np.nan
        )
        missing_rate = float(values.isna().mean())
        q1 = float(values.quantile(0.25))
        q3 = float(values.quantile(0.75))
        iqr = q3 - q1
        include = missing_rate <= 0.05 and np.isfinite(iqr) and iqr > 1e-12
        rows.append(
            {
                "feature_name": column,
                "missing_rate": missing_rate,
                "iqr": iqr,
                "included": include,
                "exclusion_reason": (
                    "" if include else "missingness_or_zero_iqr"
                ),
            }
        )
        if include:
            retained.append(column)
    if len(retained) < 2:
        raise ValueError("fewer than two features remain after QC")
    numeric = (
        matrix[retained]
        .apply(pd.to_numeric, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
    )
    numeric = numeric.fillna(numeric.median())
    scaled = RobustScaler().fit_transform(numeric)
    scaled = np.clip(scaled, -8.0, 8.0)
    return scaled, retained, pd.DataFrame.from_records(rows)


def _pca_coordinates(
    values: np.ndarray,
    *,
    random_seed: int,
) -> tuple[np.ndarray, pd.DataFrame]:
    from sklearn.decomposition import PCA

    max_components = min(50, values.shape[0] - 1, values.shape[1])
    pca = PCA(
        n_components=max_components,
        svd_solver="randomized" if values.shape[1] > 500 else "auto",
        random_state=random_seed,
    )
    coordinates = pca.fit_transform(values)
    cumulative = np.cumsum(pca.explained_variance_ratio_)
    selected = int(np.searchsorted(cumulative, 0.85) + 1)
    selected = min(max(selected, min(5, max_components)), 30, max_components)
    variance = pd.DataFrame(
        {
            "component": np.arange(1, max_components + 1),
            "explained_variance_ratio": pca.explained_variance_ratio_,
            "cumulative_explained_variance_ratio": cumulative,
            "used_for_clustering": np.arange(1, max_components + 1)
            <= selected,
        }
    )
    return coordinates[:, :selected], variance


def _evaluate_k(
    coordinates: np.ndarray,
    *,
    random_seed: int,
    stability_iterations: int,
) -> tuple[pd.DataFrame, dict[int, np.ndarray]]:
    from sklearn.cluster import KMeans
    from sklearn.metrics import adjusted_rand_score, silhouette_score

    rng = np.random.default_rng(random_seed)
    rows = []
    labels_by_k: dict[int, np.ndarray] = {}
    n_sites = len(coordinates)
    minimum_allowed = max(10, int(math.ceil(n_sites * 0.01)))
    for k in range(2, min(12, n_sites - 1) + 1):
        model = KMeans(n_clusters=k, n_init=50, random_state=random_seed + k)
        labels = model.fit_predict(coordinates)
        labels_by_k[k] = labels
        silhouette = float(silhouette_score(coordinates, labels))
        min_cluster_size = int(np.bincount(labels, minlength=k).min())
        stability = []
        for iteration in range(stability_iterations):
            sample = rng.choice(
                n_sites,
                size=min(
                    n_sites,
                    max(k * 3, int(round(n_sites * 0.80))),
                ),
                replace=False,
            )
            subsample_model = KMeans(
                n_clusters=k,
                n_init=20,
                random_state=random_seed + k * 100 + iteration,
            ).fit(coordinates[sample])
            predicted = subsample_model.predict(coordinates)
            stability.append(adjusted_rand_score(labels, predicted))
        rows.append(
            {
                "k": k,
                "silhouette": silhouette,
                "stability_ari_mean": float(np.mean(stability)),
                "stability_ari_sd": float(np.std(stability, ddof=1)),
                "min_cluster_size": min_cluster_size,
                "eligible_min_cluster_size": min_cluster_size
                >= minimum_allowed,
            }
        )
    table = pd.DataFrame.from_records(rows)
    eligible = table.loc[table["eligible_min_cluster_size"]].copy()
    if eligible.empty:
        eligible = table.copy()
    eligible["silhouette_rank"] = eligible["silhouette"].rank(
        ascending=False, method="min"
    )
    eligible["stability_rank"] = eligible["stability_ari_mean"].rank(
        ascending=False, method="min"
    )
    eligible["selection_rank_sum"] = (
        eligible["silhouette_rank"] + eligible["stability_rank"]
    )
    selected_k = int(
        eligible.sort_values(
            ["selection_rank_sum", "stability_ari_mean", "silhouette"],
            ascending=[True, False, False],
        ).iloc[0]["k"]
    )
    table["selected_k"] = table["k"].eq(selected_k)
    return table, labels_by_k


def _cluster_feature_profiles(
    scaled_values: np.ndarray,
    feature_names: Sequence[str],
    labels: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    centroids = []
    top_rows = []
    for cluster in sorted(np.unique(labels)):
        center = scaled_values[labels == cluster].mean(axis=0)
        centroids.append(
            {
                "cluster": int(cluster),
                **{
                    feature: float(value)
                    for feature, value in zip(feature_names, center)
                },
            }
        )
        order = np.argsort(np.abs(center))[::-1][:20]
        for rank, index in enumerate(order, start=1):
            top_rows.append(
                {
                    "cluster": int(cluster),
                    "rank": rank,
                    "feature_name": feature_names[index],
                    "scaled_centroid": float(center[index]),
                }
            )
    return (
        pd.DataFrame.from_records(centroids),
        pd.DataFrame.from_records(top_rows),
    )


def _numeric_covariates(site_metadata: pd.DataFrame) -> list[str]:
    candidates = (
        "peak_score",
        "motif_score",
        "rpeak_count",
        "rpeak_fraction",
        "variant_offset",
        "rad21_ctrl_signal",
        "ctcf_ctrl_signal",
        "pol2_ctrl_signal",
    )
    return [
        column
        for column in candidates
        if column in site_metadata.columns
        and pd.to_numeric(site_metadata[column], errors="coerce").notna().sum()
        >= 20
    ]


def _covariate_tests(
    assignments: pd.DataFrame,
    site_metadata: pd.DataFrame,
    representation: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    from scipy.stats import chi2_contingency, kruskal

    joined = assignments.merge(
        site_metadata, on="site_id", how="left", validate="one_to_one"
    )
    tests = []
    summaries = []
    for column in _numeric_covariates(joined):
        values = pd.to_numeric(joined[column], errors="coerce")
        groups = [
            values[joined["cluster"].eq(cluster)].dropna().to_numpy()
            for cluster in sorted(joined["cluster"].unique())
        ]
        groups = [group for group in groups if len(group)]
        stat, pvalue = kruskal(*groups) if len(groups) >= 2 else (np.nan, np.nan)
        grand = float(values.mean())
        total_ss = float(np.nansum((values - grand) ** 2))
        between_ss = 0.0
        for cluster, rows in joined.groupby("cluster"):
            cluster_values = pd.to_numeric(rows[column], errors="coerce").dropna()
            if len(cluster_values):
                between_ss += len(cluster_values) * (
                    float(cluster_values.mean()) - grand
                ) ** 2
        eta_squared = between_ss / total_ss if total_ss > 0 else np.nan
        tests.append(
            {
                "representation": representation,
                "covariate": column,
                "test_type": "kruskal",
                "kruskal_h": float(stat),
                "kruskal_p": float(pvalue),
                "eta_squared": float(eta_squared),
            }
        )
    if "site_class" in joined.columns and joined["site_class"].nunique() > 1:
        contingency = pd.crosstab(joined["cluster"], joined["site_class"])
        chi2, pvalue, _, _ = chi2_contingency(contingency)
        denominator = len(joined) * max(
            1, min(contingency.shape[0] - 1, contingency.shape[1] - 1)
        )
        tests.append(
            {
                "representation": representation,
                "covariate": "site_class",
                "test_type": "chi_squared",
                "chi_squared": float(chi2),
                "chi_squared_p": float(pvalue),
                "cramers_v": float(np.sqrt(chi2 / denominator)),
            }
        )
    for cluster, rows in joined.groupby("cluster", sort=True):
        record: dict[str, object] = {
            "representation": representation,
            "cluster": int(cluster),
            "n_sites": int(len(rows)),
        }
        for column in _numeric_covariates(joined):
            values = pd.to_numeric(rows[column], errors="coerce")
            record[f"{column}__median"] = float(values.median())
            record[f"{column}__q25"] = float(values.quantile(0.25))
            record[f"{column}__q75"] = float(values.quantile(0.75))
        if "site_class" in rows.columns:
            for site_class, count in rows["site_class"].value_counts().items():
                record[f"site_class__{site_class}__fraction"] = float(
                    count / len(rows)
                )
        record["has_ccre_fraction"] = float(rows["has_ccre"].mean())
        summaries.append(record)
    return pd.DataFrame.from_records(tests), pd.DataFrame.from_records(summaries)


def _run_representation(
    name: str,
    matrix: pd.DataFrame,
    site_metadata: pd.DataFrame,
    output_dir: Path,
    *,
    random_seed: int,
    stability_iterations: int,
) -> tuple[RepresentationResult, pd.DataFrame, pd.DataFrame]:
    rep_dir = output_dir / "representations" / name
    rep_dir.mkdir(parents=True, exist_ok=True)
    scaled, retained, qc = _qc_and_scale(matrix)
    coordinates, variance = _pca_coordinates(
        scaled, random_seed=random_seed
    )
    k_metrics, labels_by_k = _evaluate_k(
        coordinates,
        random_seed=random_seed,
        stability_iterations=stability_iterations,
    )
    selected_row = k_metrics.loc[k_metrics["selected_k"]].iloc[0]
    selected_k = int(selected_row["k"])
    labels = labels_by_k[selected_k]
    assignments = pd.DataFrame(
        {
            "site_id": matrix["site_id"].astype(str),
            "cluster": labels.astype(int),
        }
    )
    pca_scores = pd.DataFrame(
        coordinates,
        columns=[f"PC{index + 1}" for index in range(coordinates.shape[1])],
    )
    pca_scores.insert(0, "site_id", matrix["site_id"].astype(str).to_numpy())
    centroids, top_features = _cluster_feature_profiles(
        scaled, retained, labels
    )
    covariate_tests, cluster_summary = _covariate_tests(
        assignments, site_metadata, name
    )

    matrix.to_parquet(rep_dir / "feature_matrix.parquet", index=False)
    qc.to_csv(rep_dir / "feature_qc.tsv", sep="\t", index=False)
    variance.to_csv(rep_dir / "pca_variance.tsv", sep="\t", index=False)
    pca_scores.to_csv(rep_dir / "pca_scores.tsv", sep="\t", index=False)
    k_metrics.to_csv(rep_dir / "k_selection.tsv", sep="\t", index=False)
    assignments.to_csv(rep_dir / "cluster_assignments.tsv", sep="\t", index=False)
    centroids.to_csv(rep_dir / "cluster_centroids.tsv", sep="\t", index=False)
    top_features.to_csv(
        rep_dir / "cluster_top_features.tsv", sep="\t", index=False
    )
    covariate_tests.to_csv(
        rep_dir / "cluster_covariate_tests.tsv", sep="\t", index=False
    )
    cluster_summary.to_csv(
        rep_dir / "cluster_summary.tsv", sep="\t", index=False
    )
    return (
        RepresentationResult(
            name=name,
            selected_k=selected_k,
            silhouette=float(selected_row["silhouette"]),
            stability_ari=float(selected_row["stability_ari_mean"]),
            min_cluster_size=int(selected_row["min_cluster_size"]),
            n_features_before_qc=len(matrix.columns) - 1,
            n_features_after_qc=len(retained),
            n_pcs=coordinates.shape[1],
        ),
        assignments,
        top_features,
    )


def _cross_representation_ari(
    assignments: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    from sklearn.metrics import adjusted_rand_score

    rows = []
    names = list(assignments)
    for left_index, left in enumerate(names):
        for right in names[left_index + 1 :]:
            joined = assignments[left].merge(
                assignments[right],
                on="site_id",
                suffixes=("_left", "_right"),
                validate="one_to_one",
            )
            rows.append(
                {
                    "left_representation": left,
                    "right_representation": right,
                    "adjusted_rand_index": float(
                        adjusted_rand_score(
                            joined["cluster_left"], joined["cluster_right"]
                        )
                    ),
                }
            )
    return pd.DataFrame.from_records(rows)


def _plot_outputs(
    output_dir: Path,
    results: pd.DataFrame,
    assignments: dict[str, pd.DataFrame],
    track_selection: pd.DataFrame,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    figures = output_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")
    ink = "#1F2430"
    blue = "#5477C4"
    gold = "#B8A037"
    orange = "#CC6F47"

    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    for axis, metric, color in zip(
        axes,
        ("silhouette", "stability_ari_mean", "min_cluster_size"),
        (blue, gold, orange),
    ):
        for name in results["name"]:
            table = pd.read_csv(
                output_dir
                / "representations"
                / name
                / "k_selection.tsv",
                sep="\t",
            )
            axis.plot(
                table["k"],
                table[metric],
                marker="o",
                label=name,
                linewidth=1.5,
            )
        axis.set_xlabel("Number of clusters (k)")
        axis.set_ylabel(metric.replace("_", " "))
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Cluster-number diagnostics across feature representations")
    fig.text(
        0.5,
        0.01,
        "Silhouette, 80% subsample stability, and minimum cluster size; n=1,000 sites",
        ha="center",
        color="#6F768A",
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.94))
    fig.savefig(figures / "k_selection.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    names = list(assignments)
    fig, axes = plt.subplots(
        1, len(names), figsize=(6 * len(names), 5), squeeze=False
    )
    for axis, name in zip(axes[0], names):
        scores = pd.read_csv(
            output_dir / "representations" / name / "pca_scores.tsv",
            sep="\t",
        )
        joined = scores.merge(assignments[name], on="site_id")
        sns.scatterplot(
            data=joined,
            x="PC1",
            y="PC2",
            hue="cluster",
            palette="tab10",
            s=18,
            linewidth=0,
            alpha=0.75,
            legend=False,
            ax=axis,
        )
        axis.set_title(name.replace("_", " "))
        axis.spines[["top", "right"]].set_visible(False)
    fig.suptitle("PCA projections of CTCF mutation-response representations")
    fig.text(
        0.5,
        0.01,
        "Colors are independently selected k-means clusters within each representation",
        ha="center",
        color="#6F768A",
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.94))
    fig.savefig(figures / "pca_clusters.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    composition = (
        track_selection.loc[
            track_selection["selected_for_balanced_representation"]
        ]
        .groupby("output_type")
        .size()
        .sort_values()
    )
    fig, axis = plt.subplots(figsize=(9, 5))
    axis.barh(composition.index, composition.values, color="#A3BEFA", edgecolor=ink)
    for y, value in enumerate(composition.values):
        axis.text(value + 1, y, str(value), va="center", fontsize=9)
    axis.set_xlabel("Selected tracks")
    axis.set_ylabel("AlphaGenome output head")
    axis.set_title("Balanced track representation retains every 1D output head")
    axis.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(
        figures / "track_selection_composition.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(fig)

    biological_name = (
        "biological_groups_contact"
        if "biological_groups_contact" in names
        else "biological_groups"
    )
    centroids = pd.read_csv(
        output_dir
        / "representations"
        / biological_name
        / "cluster_centroids.tsv",
        sep="\t",
    ).set_index("cluster")
    feature_order = (
        centroids.abs().max(axis=0).sort_values(ascending=False).head(30).index
    )
    fig, axis = plt.subplots(figsize=(13, max(5, 0.35 * len(centroids))))
    sns.heatmap(
        centroids.loc[:, feature_order],
        cmap="vlag",
        center=0,
        robust=True,
        ax=axis,
        cbar_kws={"label": "Mean robust-scaled feature"},
    )
    axis.set_title("Biological-group cluster profiles")
    axis.set_xlabel("Top discriminating group features")
    axis.set_ylabel("Cluster")
    axis.tick_params(axis="x", labelrotation=75, labelsize=7)
    fig.tight_layout()
    fig.savefig(
        figures / "biological_cluster_profiles.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(fig)


def _write_html_report(
    output_dir: Path,
    results: pd.DataFrame,
    cross_ari: pd.DataFrame,
    top_features: dict[str, pd.DataFrame],
    contact_included: bool,
) -> None:
    best = results.sort_values(
        ["stability_ari", "silhouette"], ascending=False
    ).iloc[0]
    ari_text = (
        "No pairwise representation comparison was available."
        if cross_ari.empty
        else (
            f"Cross-representation ARI ranged from "
            f"{cross_ari['adjusted_rand_index'].min():.3f} to "
            f"{cross_ari['adjusted_rand_index'].max():.3f}."
        )
    )
    result_rows = "\n".join(
        "<tr>"
        + "".join(
            f"<td>{html.escape(str(value))}</td>"
            for value in (
                row["name"],
                int(row["selected_k"]),
                f"{row['silhouette']:.3f}",
                f"{row['stability_ari']:.3f}",
                int(row["min_cluster_size"]),
                int(row["n_features_after_qc"]),
                int(row["n_pcs"]),
            )
        )
        + "</tr>"
        for _, row in results.iterrows()
    )
    best_features = top_features[str(best["name"])]
    feature_items = "\n".join(
        f"<li><code>{html.escape(str(row.feature_name))}</code> "
        f"(cluster {int(row.cluster)}, scaled centroid "
        f"{float(row.scaled_centroid):+.2f})</li>"
        for row in best_features.head(12).itertuples(index=False)
    )
    contact_note = (
        "Contact metrics were included as compact site-level features; no "
        "contact map tensor was persisted."
        if contact_included
        else "Contact metrics were not yet included in this report."
    )
    document = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>1000-site CTCF AlphaGenome mutation-response clustering</title>
  <style>
    body {{ font-family: Inter, Arial, sans-serif; margin: 0; color: #1F2430; background: #FCFCFD; }}
    main {{ max-width: 1120px; margin: 0 auto; padding: 40px 28px 80px; }}
    h1 {{ font-size: 32px; margin-bottom: 8px; }}
    h2 {{ margin-top: 38px; border-top: 1px solid #E6E8F0; padding-top: 24px; }}
    p, li {{ line-height: 1.65; }}
    .summary {{ background: #EAF1FE; border-left: 5px solid #5477C4; padding: 18px 22px; }}
    .caveat {{ background: #FFF4C2; border-left: 5px solid #B8A037; padding: 16px 20px; }}
    table {{ width: 100%; border-collapse: collapse; background: white; }}
    th, td {{ padding: 10px; border-bottom: 1px solid #E6E8F0; text-align: left; }}
    th {{ background: #F4F5F7; }}
    img {{ width: 100%; height: auto; background: white; margin: 12px 0; }}
    code {{ font-family: "SFMono-Regular", Consolas, monospace; font-size: 0.92em; }}
  </style>
</head>
<body><main>
  <h1>1000-site CTCF AlphaGenome mutation-response clustering</h1>

  <section>
    <h2>Technical summary</h2>
    <div class="summary">
      <p><strong>The most stable representation was
      {html.escape(str(best['name']))}</strong>, selecting k={int(best['selected_k'])}
      with silhouette={float(best['silhouette']):.3f} and 80% subsample
      stability ARI={float(best['stability_ari']):.3f}. {ari_text}</p>
      <p>The result supports using assay-balanced or biological-group features
      as the primary clustering input. A plain global top-track list is not
      defensible because the response ranking is strongly output-head
      dependent.</p>
    </div>
  </section>

  <section>
    <h2>Cluster-number evidence is representation dependent</h2>
    <p>The figure compares separation, resampling stability, and minimum
    cluster size for k=2..12. The selected k minimizes the combined rank of
    silhouette and stability while rejecting very small clusters where
    possible.</p>
    <img src="figures/k_selection.png" alt="Cluster-number diagnostics">
    <table>
      <thead><tr><th>Representation</th><th>k</th><th>Silhouette</th>
      <th>Stability ARI</th><th>Min cluster</th><th>QC features</th>
      <th>PCs</th></tr></thead>
      <tbody>{result_rows}</tbody>
    </table>
  </section>

  <section>
    <h2>Global track ranking is not a suitable clustering policy</h2>
    <p>The balanced representation takes the same number of high-response
    tracks from each AlphaGenome output head, then force-retains CTCF/cohesin,
    transcription machinery, and canonical histone sentinels. This prevents a
    single assay family from winning only because it has higher response scale
    or more redundant tracks.</p>
    <img src="figures/track_selection_composition.png"
         alt="Selected tracks by output head">
  </section>

  <section>
    <h2>The site population contains several response regimes</h2>
    <p>PCA is used only as a denoising step before k-means. These projections
    show whether the selected clusters are visibly separated or mainly
    partition a continuum; the latter should be interpreted as response
    regimes rather than discrete biological classes.</p>
    <img src="figures/pca_clusters.png" alt="PCA cluster projections">
  </section>

  <section>
    <h2>Biological-group profiles make the clusters interpretable</h2>
    <p>The heatmap shows the most discriminating robust-scaled group features
    across 1, 4, 20, and 100 kb centered windows. Signs are mutation minus
    reference in log2(1+signal) space. {html.escape(contact_note)}</p>
    <img src="figures/biological_cluster_profiles.png"
         alt="Biological group cluster profiles">
  </section>

  <section>
    <h2>Leading features in the most stable representation</h2>
    <ul>{feature_items}</ul>
  </section>

  <section>
    <h2>Scope, data, and metric definitions</h2>
    <p>The cohort contains 1,000 ENCODE4 CTCF sites. Each motif was replaced
    by an equal-length random sequence. AlphaGenome used its full 1,048,576 bp
    input context, but this analysis reads only centered output slices and
    writes compact feature tables. The 1D output resolution is 128 bp.</p>
    <p>Per-track effects are the mean signed log2(1+ALT)-log2(1+REF), mean
    absolute log effect, and peak log2 fold-change inside each centered
    window. Biological groups aggregate per-track effects by the median.</p>
  </section>

  <section>
    <h2>Methodology</h2>
    <ol>
      <li>Remove padding tracks and stream centered slices from the real
      AlphaGenome REF/ALT memmaps.</li>
      <li>Build all-track, output-head-balanced, and biological-group feature
      representations.</li>
      <li>Remove missing or zero-IQR features, robust-scale, clip extreme
      scaled values, and retain up to 30 PCs explaining at least 85% variance.</li>
      <li>Evaluate k=2..12 using silhouette and 30 repeated 80% subsample
      refits; select k by their combined rank.</li>
      <li>Compare cluster assignments across representations with adjusted
      Rand index and test known site covariates across clusters.</li>
    </ol>
  </section>

  <section>
    <h2>Limitations and robustness boundary</h2>
    <div class="caveat">
      <p>This is descriptive EDA, not evidence that clusters are discrete
      biological classes. The perturbation is whole-motif random replacement,
      not a subtle SNV, and this 1000-site run has no matched local controls.
      Track sensitivity and clustering use the same site population, so the
      balanced and biological representations are more trustworthy than a
      pure data-driven global top-track selection.</p>
      <p>The sites are genome-wide CTCF examples rather than a single-gene
      cohort or paper-labeled DIC classes. Results identify response regimes
      to investigate, not causal mechanisms.</p>
    </div>
  </section>

  <section>
    <h2>Recommended next steps</h2>
    <ol>
      <li>Use biological groups as the primary population representation and
      balanced tracks as a sensitivity analysis.</li>
      <li>Inspect representative sites from stable clusters at 1 bp or with
      local 1 kb plots; do not rerun all 1,000 sites at 1 bp.</li>
      <li>Validate the cluster-defining effects with matched non-motif controls
      and strand-aware motif SNVs on a smaller stratified subset.</li>
    </ol>
  </section>

  <section>
    <h2>Further questions</h2>
    <ul>
      <li>Do the stable response regimes persist within one gene or one
      chromatin state?</li>
      <li>Are contact-response features aligned with local CTCF/cohesin loss,
      or do they define an orthogonal axis?</li>
      <li>How much of the structure remains after matching motif score, peak
      score, and CTCF ubiquity?</li>
    </ul>
  </section>
</main></body></html>
"""
    (output_dir / "report.html").write_text(document, encoding="utf-8")


def main() -> None:
    args = _parse_args()
    run_dir = Path(args.run_dir)
    sensitivity_dir = Path(args.sensitivity_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata = pd.read_csv(
        run_dir / "all_track_metadata.tsv", sep="\t", low_memory=False
    )
    if "global_track_index" not in metadata.columns:
        metadata["global_track_index"] = np.arange(len(metadata), dtype=int)
    metadata["global_track_index"] = pd.to_numeric(
        metadata["global_track_index"], errors="raise"
    ).astype(int)
    site_metadata = _load_site_metadata(run_dir, Path(args.source_sites))
    if args.max_sites is not None:
        site_metadata = site_metadata.head(args.max_sites).copy()
    sensitivity = _load_sensitivity_table(
        sensitivity_dir / "track_sensitivity_rank_full_100kb_no_padding.tsv"
    )
    track_selection = _select_tracks(
        metadata,
        sensitivity,
        top_per_output_type=args.top_per_output_type,
        sentinel_per_target=args.sentinel_per_target,
    )
    track_selection.to_csv(
        output_dir / "track_selection.tsv", sep="\t", index=False
    )
    site_metadata.to_csv(
        output_dir / "site_metadata.tsv", sep="\t", index=False
    )

    matrices, site_effects = _extract_feature_matrices(
        run_dir,
        track_selection,
        site_metadata,
        site_chunk_size=args.site_chunk_size,
    )
    site_effects.to_csv(
        output_dir / "site_effect_summary.tsv", sep="\t", index=False
    )

    contact_included = False
    if args.contact_summary:
        contact = _load_contact_features(
            Path(args.contact_summary), site_metadata["site_id"].astype(str)
        )
        matrices["biological_groups_contact"] = matrices[
            "biological_groups"
        ].merge(contact, on="site_id", validate="one_to_one")
        contact_included = True

    results = []
    assignments: dict[str, pd.DataFrame] = {}
    top_features: dict[str, pd.DataFrame] = {}
    for offset, (name, matrix) in enumerate(matrices.items()):
        result, assignment, top = _run_representation(
            name,
            matrix,
            site_metadata,
            output_dir,
            random_seed=args.random_seed + offset * 1000,
            stability_iterations=args.stability_iterations,
        )
        results.append(result.__dict__)
        assignments[name] = assignment
        top_features[name] = top
        print(
            f"{name}: k={result.selected_k}, "
            f"silhouette={result.silhouette:.4f}, "
            f"stability={result.stability_ari:.4f}",
            flush=True,
        )

    result_table = pd.DataFrame.from_records(results)
    result_table.to_csv(
        output_dir / "representation_comparison.tsv", sep="\t", index=False
    )
    cross_ari = _cross_representation_ari(assignments)
    cross_ari.to_csv(
        output_dir / "cross_representation_ari.tsv", sep="\t", index=False
    )
    _plot_outputs(output_dir, result_table, assignments, track_selection)
    _write_html_report(
        output_dir,
        result_table,
        cross_ari,
        top_features,
        contact_included,
    )

    provenance = {
        "source_run_dir": str(run_dir),
        "source_sensitivity_dir": str(sensitivity_dir),
        "source_sites": str(Path(args.source_sites)),
        "ref_array": str(run_dir / "all_track_ref_window.npy"),
        "alt_array": str(run_dir / "all_track_alt_window.npy"),
        "array_shape": list(
            np.load(run_dir / "all_track_ref_window.npy", mmap_mode="r").shape
        ),
        "window_widths_bp": list(WINDOW_WIDTHS_BP),
        "output_resolution_bp": 128,
        "top_per_output_type": args.top_per_output_type,
        "sentinel_per_target": args.sentinel_per_target,
        "stability_iterations": args.stability_iterations,
        "random_seed": args.random_seed,
        "contact_summary": args.contact_summary,
        "population_tensor_storage": "none; source arrays read by memmap",
    }
    (output_dir / "provenance.json").write_text(
        json.dumps(provenance, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=FutureWarning)
        main()
