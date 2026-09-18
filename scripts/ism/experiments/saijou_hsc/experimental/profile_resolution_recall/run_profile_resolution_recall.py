#!/usr/bin/env python3
"""Recalculate single-AlphaGenome profile recall for res128 and res1.

This is an experimental, read-only re-analysis of completed 6-kb strict-
shuffle inference.  It deliberately does not use browser JSON, Borzoi, or
original-model Method 3 scores.  For each resolution it calculates:

* per-cell empirical tails of median absolute profile effect, calibrated in
  model x cell x reference-signal-decile strata across all nine genes;
* a single-model Canonical score: rank(maximum of the four cell tails) across
  all centres; and
* interval-level actual recall for the seven registered controls.  The two
  frozen Mdk WT1 intervals are reported, but excluded from that denominator.

The result is a controlled profile-resolution comparison.  Raw effect values
are descriptive because bin count and pseudocount geometry differ; ranked
scores and recall are the primary comparison.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset as ds


REPO = Path(__file__).resolve().parents[6]
SAIJOU = Path(__file__).resolve().parents[2]
if str(SAIJOU) not in sys.path:
    sys.path.insert(0, str(SAIJOU))
from tools.region_importance import reference_bins  # noqa: E402


OLD = REPO / "experiments/ism/20260903_nine_gene_tss6kb_strict_shuffle"
RES1 = REPO / "experiments/ism/20260911_nine_gene_tss6kb_strict_shuffle_ag_res1"
R5 = REPO / "experiments/ism/20260909_mdk_motif_ism_v1_e1e6_r5"
REGISTRY = SAIJOU / "configs/positive_control_registry.tsv"
OUT = REPO / "experiments/ism/20260913_profile_resolution_recall"

TRACKS = ("hsc", "mac", "lsec", "chol")
READOUT = "gene_body_output_clipped"
THRESHOLDS = (50.0, 60.0, 70.0, 80.0, 90.0, 95.0, 97.5)
FEATURE_COLUMNS = [
    "gene",
    "variant_offset_from_tss_transcription_bp",
    "track_id",
    "readout_role",
    "mutation_id",
    "log2fc_ratio_of_sums",
    "ref_sum",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_paths() -> dict[str, list[Path]]:
    shards: dict[str, Path] = {}
    for shard in sorted((RES1 / "shards").glob("gpu*")):
        summary = json.loads((shard / "validation_summary.json").read_text())
        for gene in summary["genes"]:
            if gene in shards:
                raise ValueError(f"duplicate res1 shard ownership for {gene}")
            shards[gene] = shard / "features/combined_mutation_features.parquet"
    expected = set(pd.read_csv(OLD / "prepared/genes.tsv", sep="\t").gene)
    if set(shards) != expected:
        raise ValueError(f"res1 gene coverage mismatch: {sorted(set(shards) ^ expected)}")
    unique_paths = sorted(set(shards.values()))
    expected_manifest = sha256(OLD / "prepared/mutation_manifest.tsv")
    observed_manifests = {
        json.loads((path.parent.parent / "model_metadata.json").read_text())["prepared_manifest_sha256"]
        for path in unique_paths
    }
    if observed_manifests != {expected_manifest}:
        raise ValueError(
            "res1 shard manifest hash does not match the res128 prepared manifest"
        )
    return {
        "res128": [OLD / "runs/alphagenome_finetuned/features/combined_mutation_features.parquet"],
        "res1": unique_paths,
    }


def summarize_profile(paths: list[Path], resolution: str) -> pd.DataFrame:
    parts = []
    for path in paths:
        table = ds.dataset(path, format="parquet").to_table(
            columns=FEATURE_COLUMNS,
            filter=(ds.field("readout_role") == READOUT)
            & ds.field("track_id").isin(TRACKS),
        )
        parts.append(table.to_pandas())
    data = pd.concat(parts, ignore_index=True)
    data["absolute_effect"] = data.log2fc_ratio_of_sums.abs()
    keys = ["gene", "variant_offset_from_tss_transcription_bp", "track_id"]
    grouped = (
        data.groupby(keys, sort=False)
        .agg(
            replacement_replicates=("mutation_id", "nunique"),
            median_signed_effect=("log2fc_ratio_of_sums", "median"),
            median_absolute_effect=("absolute_effect", "median"),
            median_ref_sum=("ref_sum", "median"),
        )
        .reset_index()
    )
    if not grouped.replacement_replicates.eq(3).all():
        raise ValueError(f"{resolution}: expected exactly three replacements")
    grouped["resolution"] = resolution
    grouped["reference_bin"] = -1
    for _, indices in grouped.groupby("track_id", sort=False).groups.items():
        grouped.loc[indices, "reference_bin"] = reference_bins(
            grouped.loc[indices, "median_ref_sum"], 10
        ).to_numpy()
    grouped["cell_tail_percentile"] = grouped.groupby(
        ["track_id", "reference_bin"], sort=False
    )["median_absolute_effect"].rank(method="average", pct=True)
    return grouped


def canonical_scores(summary: pd.DataFrame) -> pd.DataFrame:
    index = ["gene", "variant_offset_from_tss_transcription_bp"]
    tails = summary.pivot(index=index, columns="track_id", values="cell_tail_percentile")
    if set(tails.columns) != set(TRACKS):
        raise ValueError(f"missing profile tracks: {set(TRACKS) - set(tails.columns)}")
    tails = tails.loc[:, TRACKS]
    result = tails.reset_index()
    result["max_cell_tail_percentile"] = tails.max(axis=1).to_numpy()
    result["canonical_score"] = 100.0 * result.max_cell_tail_percentile.rank(
        method="average", pct=True
    )
    result["canonical_driving_cell"] = tails.idxmax(axis=1).to_numpy()
    return result


def intervals() -> pd.DataFrame:
    controls = pd.read_csv(REGISTRY, sep="\t")
    controls["item_type"] = "positive_control"
    candidates = pd.read_csv(R5 / "candidate_instances.tsv", sep="\t")
    wt1 = candidates.loc[
        candidates.instance_id.isin(["wt1_0001", "wt1_0002"]),
        ["instance_id", "tx_start", "tx_end", "category"],
    ].rename(
        columns={
            "instance_id": "control_id",
            "tx_start": "start",
            "tx_end": "end",
            "category": "label",
        }
    )
    wt1["gene"] = "Mdk"
    if set(wt1.control_id) != {"wt1_0001", "wt1_0002"}:
        raise ValueError("frozen WT1 candidates are incomplete")
    wt1 = wt1.rename(columns={"instance_id": "control_id"})
    wt1["family"] = "WT1"
    wt1["evidence"] = "frozen_r5_candidate_not_positive_control"
    wt1["coordinate_rule"] = "frozen r5 transcript-oriented candidate interval"
    wt1["item_type"] = "wt1_context"
    return pd.concat([controls, wt1[controls.columns]], ignore_index=True)


def score_intervals(
    items: pd.DataFrame, summary: pd.DataFrame, canonical: pd.DataFrame
) -> pd.DataFrame:
    rows = []
    for item in items.itertuples(index=False):
        # A strict 10-bp shuffle centred at x spans x +/- 5 bp.
        lower, upper = int(item.start) - 5, int(item.end) + 5
        centers = canonical.loc[
            canonical.gene.eq(item.gene)
            & canonical.variant_offset_from_tss_transcription_bp.between(lower, upper)
        ].copy()
        if centers.empty:
            raise ValueError(f"{item.control_id}: no covered centres")
        best = centers.nlargest(1, "canonical_score").iloc[0]
        row = {
            "item_id": item.control_id,
            "item_type": item.item_type,
            "gene": item.gene,
            "family": item.family,
            "label": item.label,
            "tx_start": int(item.start),
            "tx_end": int(item.end),
            "covered_centers": len(centers),
            "best_center": int(best.variant_offset_from_tss_transcription_bp),
            "canonical_score": float(best.canonical_score),
            "canonical_driving_cell": str(best.canonical_driving_cell),
        }
        for cell in TRACKS:
            source = summary.loc[
                summary.gene.eq(item.gene)
                & summary.track_id.eq(cell)
                & summary.variant_offset_from_tss_transcription_bp.between(lower, upper)
            ]
            if len(source) != len(centers):
                raise ValueError(f"{item.control_id}/{cell}: profile centre mismatch")
            gene_source = summary.loc[
                summary.gene.eq(item.gene) & summary.track_id.eq(cell)
            ]
            median_abs = float(source.median_absolute_effect.median())
            baseline = float(gene_source.median_absolute_effect.median())
            row.update(
                {
                    f"{cell}_tail_peak": float(source.cell_tail_percentile.max() * 100),
                    f"{cell}_tail_median": float(source.cell_tail_percentile.median() * 100),
                    f"{cell}_absolute_median": median_abs,
                    f"{cell}_signed_median": float(source.median_signed_effect.median()),
                    f"{cell}_gene_baseline": baseline,
                    f"{cell}_enrichment": median_abs / baseline,
                }
            )
        rows.append(row)
    return pd.DataFrame(rows)


def compare_intervals(long: pd.DataFrame) -> pd.DataFrame:
    key = ["item_id", "item_type", "gene", "family", "label", "tx_start", "tx_end"]
    base = long.loc[long.resolution.eq("res128")].set_index(key)
    fine = long.loc[long.resolution.eq("res1")].set_index(key)
    if not base.index.equals(fine.index):
        raise ValueError("interval identities differ between resolutions")
    result = base[key[:0]].reset_index()
    result["covered_centers"] = base.covered_centers.to_numpy()
    for name in ["canonical_score", *[f"{cell}_{suffix}" for cell in TRACKS for suffix in ("tail_peak", "enrichment")]]:
        result[f"{name}_res128"] = base[name].to_numpy()
        result[f"{name}_res1"] = fine[name].to_numpy()
        result[f"{name}_delta"] = fine[name].to_numpy() - base[name].to_numpy()
    result["best_center_res128"] = base.best_center.to_numpy()
    result["best_center_res1"] = fine.best_center.to_numpy()
    result["canonical_driving_cell_res128"] = base.canonical_driving_cell.to_numpy()
    result["canonical_driving_cell_res1"] = fine.canonical_driving_cell.to_numpy()
    return result


def recall_table(long: pd.DataFrame) -> pd.DataFrame:
    positives = long.loc[long.item_type.eq("positive_control")].copy()
    rows = []
    for resolution, part in positives.groupby("resolution", sort=False):
        metrics = {"canonical": "canonical_score"}
        metrics.update({cell: f"{cell}_tail_peak" for cell in TRACKS})
        for metric, column in metrics.items():
            for threshold in THRESHOLDS:
                recalled = int(part[column].ge(threshold).sum())
                rows.append(
                    {
                        "resolution": resolution,
                        "metric": metric,
                        "threshold": threshold,
                        "recalled_controls": recalled,
                        "total_controls": len(part),
                        "recall_fraction": recalled / len(part),
                    }
                )
    return pd.DataFrame(rows)


def write_readme(comparison: pd.DataFrame, recall: pd.DataFrame) -> str:
    canonical = recall.loc[
        (recall.metric.eq("canonical")) & recall.threshold.eq(95.0)
    ].set_index("resolution")
    up = int((comparison.loc[comparison.item_type.eq("positive_control"), "canonical_score_delta"] > 0).sum())
    return f"""# res128 versus res1: actual single-AlphaGenome profile recall

This re-analysis reads completed 6-kb strict-shuffle feature parquet only. It
does **not** use the browser snapshot, Borzoi, original-model Method 3, or new
inference. Both resolutions use the same prepared mutation manifest.

## Score contract

For each resolution and each of HSC, macrophage, LSEC, and cholangiocyte,
absolute median profile effects are ranked within that model × cell ×
reference-signal decile over all nine genes. `canonical_score` is the global
0–100 rank of the maximum of those four one-model cell percentiles at each
edit centre. Positive-control recall is an interval-level maximum of that
score; a 10-bp shuffle overlaps centres from `start−5` through `end+5`.

The seven registered controls define recall. `wt1_0001` (Mdk −1300..−1290)
and `wt1_0002` (+2260..+2270; splice-near) are context rows only and are not
added to the denominator.

## Primary result at threshold 95

| resolution | canonical recall |
|---|---:|
| res128 | {int(canonical.loc['res128', 'recalled_controls'])}/7 |
| res1 | {int(canonical.loc['res1', 'recalled_controls'])}/7 |

At the interval level, canonical score rises for {up}/7 registered controls.
Raw effect magnitudes are descriptive rather than directly comparable because
the output bins and pseudocount geometry differ; use percentile/rank and
baseline-normalized enrichment for the resolution comparison.

## Files

- `profile_center_scores.tsv`: per-centre, per-resolution four-cell profile
  percentiles and one-model canonical score.
- `interval_scores.tsv`: exact scores and effects for all seven controls plus
  both WT1 context intervals.
- `interval_comparison.tsv`: paired res128/res1 deltas.
- `threshold_recall.tsv`: actual recall curves for Canonical and each cell.
- `validation_summary.json`: input hashes, coverage and finite/key checks.
"""


def main() -> None:
    if OUT.exists():
        raise SystemExit(f"refusing to overwrite existing output: {OUT}")
    manifest = OLD / "prepared/mutation_manifest.tsv"
    paths = source_paths()
    items = intervals()
    score_rows, interval_rows, canonical_rows = [], [], []
    validation = {"input_manifest_sha256": sha256(manifest), "readout": READOUT, "tracks": list(TRACKS)}
    for resolution, resolution_paths in paths.items():
        profile = summarize_profile(resolution_paths, resolution)
        canonical = canonical_scores(profile)
        interval = score_intervals(items, profile, canonical)
        profile = profile.merge(canonical, on=["gene", "variant_offset_from_tss_transcription_bp"], validate="many_to_one")
        profile["resolution"] = resolution
        interval["resolution"] = resolution
        score_rows.append(profile)
        interval_rows.append(interval)
        canonical_rows.append(canonical.assign(resolution=resolution))
        validation[resolution] = {
            "source_paths": [str(path.relative_to(REPO)) for path in resolution_paths],
            "profile_rows": int(len(profile)),
            "centers": int(len(canonical)),
            "finite_numeric": bool(np.isfinite(profile.select_dtypes(include="number").to_numpy()).all()),
            "unique_profile_keys": bool(not profile.duplicated(["gene", "variant_offset_from_tss_transcription_bp", "track_id"]).any()),
        }
    profiles = pd.concat(score_rows, ignore_index=True)
    interval_long = pd.concat(interval_rows, ignore_index=True)
    comparison = compare_intervals(interval_long)
    recall = recall_table(interval_long)
    validation["interval_rows"] = int(len(interval_long))
    validation["registered_controls"] = int(items.item_type.eq("positive_control").sum())
    validation["wt1_context_rows"] = int(items.item_type.eq("wt1_context").sum())
    validation["paired_interval_keys"] = bool(len(comparison) == len(items))
    validation["status"] = "ok" if all(value["finite_numeric"] and value["unique_profile_keys"] for key, value in validation.items() if key in {"res128", "res1"}) else "validation_failed"
    OUT.mkdir(parents=True)
    profiles.to_csv(OUT / "profile_center_scores.tsv", sep="\t", index=False)
    interval_long.to_csv(OUT / "interval_scores.tsv", sep="\t", index=False)
    comparison.to_csv(OUT / "interval_comparison.tsv", sep="\t", index=False)
    recall.to_csv(OUT / "threshold_recall.tsv", sep="\t", index=False)
    (OUT / "validation_summary.json").write_text(json.dumps(validation, indent=2, sort_keys=True) + "\n")
    (OUT / "README.md").write_text(write_readme(comparison, recall))
    print(comparison[["item_id", "item_type", "canonical_score_res128", "canonical_score_res1", "canonical_score_delta", "canonical_driving_cell_res128", "canonical_driving_cell_res1"]].to_string(index=False))
    print(recall.loc[recall.threshold.eq(95.0)].to_string(index=False))
    print(OUT)


if __name__ == "__main__":
    main()
