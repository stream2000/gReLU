#!/usr/bin/env python
"""EDA scoring for full-track AlphaGenome local-window outputs.

This script reuses ``all_track_{ref,alt,delta}_window.npy`` from
``run_tf_context.py --feature_mode track_window``. It computes paired
perturbation-minus-control effects track by track, ranks responsive tracks
without using class labels, and reports class tests across top-K track sets.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


PERTURBATION_TYPES = (
    "ctcf_motif_max_ic_disruption",
    "whole_motif_random_replacement",
)
CONTROL_TYPE = "same_peak_non_motif_nearby_base"
TOP_K_VALUES = (50, 100, 200, 500)


def _load_ordered_sites(run_dir: Path, sites_path: Path) -> pd.DataFrame:
    variants = pd.read_csv(run_dir / "variants.tsv", sep="\t")
    sites = pd.read_csv(sites_path, sep="\t")
    ordered = variants[["site_id"]].merge(sites, on="site_id", how="left", validate="one_to_one")
    if ordered["paired_site_id"].isna().any():
        missing = ordered.loc[ordered["paired_site_id"].isna(), "site_id"].head().tolist()
        raise ValueError(f"Could not annotate variants from sites table; examples: {missing}")
    return ordered


def _track_label(row: pd.Series) -> str:
    target = row.get("transcription_factor")
    if pd.isna(target) or target == "":
        target = row.get("histone_mark")
    if pd.isna(target) or target == "":
        target = row.get("track_name")
    return "" if pd.isna(target) else str(target)


def _paired_indices(sites: pd.DataFrame) -> list[dict]:
    records: list[dict] = []
    for pair_id, group in sites.reset_index().groupby("paired_site_id", sort=False):
        control = group[group["control_type"] == CONTROL_TYPE]
        if control.empty:
            continue
        c = control.iloc[0]
        for perturbation_type in PERTURBATION_TYPES:
            perturb = group[group["control_type"] == perturbation_type]
            if perturb.empty:
                continue
            p = perturb.iloc[0]
            records.append(
                {
                    "paired_site_id": pair_id,
                    "dic_class": p["dic_class"],
                    "perturbation_type": perturbation_type,
                    "perturbation_index": int(p["index"]),
                    "control_index": int(c["index"]),
                    "perturbation_site_id": p["site_id"],
                    "control_site_id": c["site_id"],
                    "peak_score": p.get("peak_score", np.nan),
                    "motif_score": p.get("motif_score", np.nan),
                    "motif_to_boundary_center_bp": p.get("motif_to_boundary_center_bp", np.nan),
                    "boundary_width_bp": p.get("boundary_width_bp", np.nan),
                }
            )
    return records


def _log_effect(ref: np.ndarray, alt: np.ndarray) -> np.ndarray:
    """Return log2(1 + alt) - log2(1 + ref), clamping tiny negatives."""

    return np.log2(1.0 + np.maximum(alt, 0.0)) - np.log2(1.0 + np.maximum(ref, 0.0))


def _mean_by_track(arr: np.ndarray) -> np.ndarray:
    return np.asarray(arr.mean(axis=1), dtype=np.float64)


def _mean_abs_by_track(arr: np.ndarray) -> np.ndarray:
    return np.asarray(np.abs(arr).mean(axis=1), dtype=np.float64)


def _permutation_anova_p(values: np.ndarray, groups: np.ndarray, rng: np.random.Generator, n_perm: int) -> tuple[float, float]:
    valid = np.isfinite(values)
    values = values[valid]
    groups = groups[valid]
    labels = np.unique(groups)
    if len(labels) < 2:
        return np.nan, np.nan

    def f_stat(vals: np.ndarray, labs: np.ndarray) -> float:
        grand = vals.mean()
        ss_between = 0.0
        ss_within = 0.0
        for label in labels:
            x = vals[labs == label]
            if len(x) == 0:
                continue
            ss_between += len(x) * float((x.mean() - grand) ** 2)
            ss_within += float(((x - x.mean()) ** 2).sum())
        df_between = len(labels) - 1
        df_within = len(vals) - len(labels)
        if df_between <= 0 or df_within <= 0 or ss_within == 0:
            return np.nan
        return (ss_between / df_between) / (ss_within / df_within)

    observed = f_stat(values, groups)
    if not np.isfinite(observed):
        return observed, np.nan
    hits = 0
    for _ in range(n_perm):
        permuted = rng.permutation(groups)
        stat = f_stat(values, permuted)
        if np.isfinite(stat) and stat >= observed:
            hits += 1
    return observed, (hits + 1) / (n_perm + 1)


def _kruskal_p(values: np.ndarray, groups: np.ndarray) -> tuple[float, float]:
    try:
        from scipy import stats
    except ImportError:
        return np.nan, np.nan

    arrays = [values[(groups == label) & np.isfinite(values)] for label in np.unique(groups)]
    arrays = [arr for arr in arrays if len(arr) > 0]
    if len(arrays) < 2:
        return np.nan, np.nan
    stat, p = stats.kruskal(*arrays)
    return float(stat), float(p)


def _top_indices(rank: np.ndarray, k: int | str) -> np.ndarray:
    if k == "all":
        return np.arange(len(rank), dtype=int)
    return rank[: int(k)]


def _write_markdown_summary(
    out_path: Path,
    class_tests: pd.DataFrame,
    track_rank: pd.DataFrame,
    n_pairs: int,
    n_tracks: int,
) -> None:
    focus = class_tests[
        (class_tests["track_set"] == "top200")
        & (class_tests["metric"].isin(["log1p_abs_mean", "log1p_depletion_mean", "raw_delta_abs_mean"]))
    ].copy()
    focus = focus.sort_values(["perturbation_type", "metric"])

    def table(df: pd.DataFrame, max_rows: int = 20) -> str:
        view = df.head(max_rows).copy()
        for col in view.columns:
            if pd.api.types.is_float_dtype(view[col]):
                view[col] = view[col].map(lambda x: "" if pd.isna(x) else f"{x:.5g}")
        lines = [
            "| " + " | ".join(map(str, view.columns)) + " |",
            "| " + " | ".join(["---"] * len(view.columns)) + " |",
        ]
        lines.extend("| " + " | ".join(map(str, row)) + " |" for row in view.to_numpy())
        return "\n".join(lines)

    top_cols = [
        "response_rank",
        "global_track_index",
        "output_type",
        "target",
        "biosample_name",
        "log1p_abs_response_mean",
        "raw_abs_response_mean",
    ]
    lines = [
        "# CTCF EDA power scoring: top tracks + log1p paired effect",
        "",
        f"Input contained {n_pairs} paired perturbation-control comparisons across {n_tracks} tracks.",
        "",
        "Track ranking did not use `dic_class`; tracks were ranked by the mean absolute paired log1p effect across all paired sites and both perturbation types.",
        "",
        "Primary EDA score set is `top200`. Top 50, 100, 500, and all tracks are included as sensitivity checks in `class_tests_by_score.tsv`.",
        "",
        "## Top 200 class-test focus",
        "",
        table(
            focus[
                [
                    "perturbation_type",
                    "metric",
                    "kruskal_p",
                    "perm_anova_p",
                    "CTCF-dependent__mean",
                    "Cohesin-dependent__mean",
                    "Cohesin-separated__mean",
                    "Robust__mean",
                ]
            ]
        ),
        "",
        "## Top responsive tracks",
        "",
        table(track_rank[top_cols], max_rows=20),
        "",
        "Interpretation note: this is exploratory. A lower p-value after top-track trimming should be treated as evidence of reduced noise only if the direction is stable across top-K values and remains compatible with the paired perturbation-minus-control design.",
        "",
    ]
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--sites", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--n_perm", type=int, default=999)
    parser.add_argument("--seed", type=int, default=20260529)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sites = _load_ordered_sites(run_dir, Path(args.sites))
    pairs = _paired_indices(sites)
    if not pairs:
        raise ValueError("No perturbation-control pairs found")

    ref = np.load(run_dir / "all_track_ref_window.npy", mmap_mode="r")
    alt = np.load(run_dir / "all_track_alt_window.npy", mmap_mode="r")
    delta = np.load(run_dir / "all_track_delta_window.npy", mmap_mode="r")
    metadata = pd.read_csv(run_dir / "all_track_metadata.tsv", sep="\t", low_memory=False)
    if ref.shape != alt.shape or ref.shape != delta.shape:
        raise ValueError(f"Shape mismatch: ref={ref.shape}, alt={alt.shape}, delta={delta.shape}")
    if len(sites) != ref.shape[0]:
        raise ValueError(f"Site rows {len(sites)} do not match arrays {ref.shape[0]}")

    n_tracks = ref.shape[1]
    log_abs_sum = np.zeros(n_tracks, dtype=np.float64)
    raw_abs_sum = np.zeros(n_tracks, dtype=np.float64)
    per_track_rows: list[dict] = []

    pair_rows: list[dict] = []
    for item in pairs:
        p_idx = item["perturbation_index"]
        c_idx = item["control_index"]

        control_log = _log_effect(ref[c_idx], alt[c_idx])
        perturb_log = _log_effect(ref[p_idx], alt[p_idx])
        paired_log = perturb_log - control_log

        raw_abs_track = _mean_abs_by_track(delta[p_idx]) - _mean_abs_by_track(delta[c_idx])
        log_signed_track = _mean_by_track(paired_log)
        log_abs_track = _mean_abs_by_track(paired_log)
        log_depletion_track = _mean_by_track(np.maximum(-paired_log, 0.0))

        log_abs_sum += log_abs_track
        raw_abs_sum += np.abs(raw_abs_track)

        track_record_base = {
            "paired_site_id": item["paired_site_id"],
            "dic_class": item["dic_class"],
            "perturbation_type": item["perturbation_type"],
        }
        for track_idx in range(n_tracks):
            per_track_rows.append(
                {
                    **track_record_base,
                    "global_track_index": track_idx,
                    "raw_delta_abs_mean": raw_abs_track[track_idx],
                    "log1p_signed_mean": log_signed_track[track_idx],
                    "log1p_abs_mean": log_abs_track[track_idx],
                    "log1p_depletion_mean": log_depletion_track[track_idx],
                }
            )
        pair_rows.append(
            {
                **item,
                "all_raw_delta_abs_mean": float(raw_abs_track.mean()),
                "all_log1p_signed_mean": float(log_signed_track.mean()),
                "all_log1p_abs_mean": float(log_abs_track.mean()),
                "all_log1p_depletion_mean": float(log_depletion_track.mean()),
            }
        )

    n_comparisons = len(pairs)
    response = log_abs_sum / n_comparisons
    raw_response = raw_abs_sum / n_comparisons
    rank = np.argsort(response)[::-1]

    track_rank = metadata.copy()
    if "global_track_index" not in track_rank.columns:
        track_rank["global_track_index"] = np.arange(len(track_rank))
    track_rank = track_rank.reset_index(drop=True)
    track_rank["target"] = track_rank.apply(_track_label, axis=1)
    track_rank["log1p_abs_response_mean"] = response
    track_rank["raw_abs_response_mean"] = raw_response
    rank_pos = np.empty(n_tracks, dtype=int)
    rank_pos[rank] = np.arange(1, n_tracks + 1)
    track_rank["response_rank"] = rank_pos
    track_rank = track_rank.sort_values("response_rank").reset_index(drop=True)
    track_rank.to_csv(output_dir / "track_responsiveness_rank.tsv", sep="\t", index=False)

    top_rows = []
    for k in TOP_K_VALUES:
        for track_idx in rank[:k]:
            top_rows.append({"track_set": f"top{k}", "global_track_index": int(track_idx)})
    for track_idx in range(n_tracks):
        top_rows.append({"track_set": "all", "global_track_index": int(track_idx)})
    pd.DataFrame(top_rows).to_csv(output_dir / "top_track_sets.tsv", sep="\t", index=False)

    per_track = pd.DataFrame.from_records(per_track_rows)
    per_track.to_csv(output_dir / "paired_log1p_effects_by_track.tsv", sep="\t", index=False)

    metrics = ["raw_delta_abs_mean", "log1p_signed_mean", "log1p_abs_mean", "log1p_depletion_mean"]
    score_rows: list[dict] = []
    for (pair_id, perturbation_type), group in per_track.groupby(["paired_site_id", "perturbation_type"], sort=False):
        meta = next(
            item for item in pairs if item["paired_site_id"] == pair_id and item["perturbation_type"] == perturbation_type
        )
        values_by_track = group.sort_values("global_track_index")
        for k in (*TOP_K_VALUES, "all"):
            idx = _top_indices(rank, k)
            rec = {
                **meta,
                "track_set": f"top{k}" if isinstance(k, int) else "all",
                "n_tracks": len(idx),
            }
            for metric in metrics:
                rec[metric] = float(values_by_track.iloc[idx][metric].mean())
            score_rows.append(rec)
    scores = pd.DataFrame.from_records(score_rows)
    scores.to_csv(output_dir / "paired_eda_scores.tsv", sep="\t", index=False)

    rng = np.random.default_rng(args.seed)
    test_rows: list[dict] = []
    for (perturbation_type, track_set), group in scores.groupby(["perturbation_type", "track_set"], sort=False):
        groups = group["dic_class"].to_numpy()
        for metric in metrics:
            vals = group[metric].astype(float).to_numpy()
            h, kp = _kruskal_p(vals, groups)
            f, pp = _permutation_anova_p(vals, groups, rng, args.n_perm)
            rec = {
                "perturbation_type": perturbation_type,
                "track_set": track_set,
                "metric": metric,
                "kruskal_h": h,
                "kruskal_p": kp,
                "perm_anova_f": f,
                "perm_anova_p": pp,
            }
            for label in sorted(np.unique(groups)):
                x = vals[groups == label]
                rec[f"{label}__n"] = int(np.isfinite(x).sum())
                rec[f"{label}__mean"] = float(np.nanmean(x))
                rec[f"{label}__median"] = float(np.nanmedian(x))
            test_rows.append(rec)
    class_tests = pd.DataFrame.from_records(test_rows)
    class_tests.to_csv(output_dir / "class_tests_by_score.tsv", sep="\t", index=False)

    pd.DataFrame.from_records(pair_rows).to_csv(output_dir / "paired_eda_alltrack_baseline.tsv", sep="\t", index=False)
    _write_markdown_summary(
        output_dir / "eda_power_summary_zh.md",
        class_tests=class_tests,
        track_rank=track_rank,
        n_pairs=n_comparisons,
        n_tracks=n_tracks,
    )

    print(f"wrote: {output_dir / 'track_responsiveness_rank.tsv'}")
    print(f"wrote: {output_dir / 'paired_log1p_effects_by_track.tsv'}")
    print(f"wrote: {output_dir / 'paired_eda_scores.tsv'}")
    print(f"wrote: {output_dir / 'class_tests_by_score.tsv'}")
    print(f"wrote: {output_dir / 'eda_power_summary_zh.md'}")


if __name__ == "__main__":
    main()
