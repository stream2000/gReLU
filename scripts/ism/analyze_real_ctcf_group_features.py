#!/usr/bin/env python
"""Cluster real-checkpoint grouped ISM features with optional shard controls."""

from __future__ import annotations

import argparse
import html
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from analyze_ctcf_population_clustering import (  # noqa: E402
    CONTACT_FEATURES,
    _cross_representation_ari,
    _load_contact_features,
    _run_representation,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-dir", action="append", required=True)
    parser.add_argument("--source-sites", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--contact-summary", default=None)
    parser.add_argument(
        "--site-class",
        action="append",
        default=[],
        help="Analyze only these site_class values; repeat as needed",
    )
    parser.add_argument("--stability-iterations", type=int, default=30)
    parser.add_argument("--random-seed", type=int, default=20260610)
    return parser.parse_args()


def _load_real_features(
    feature_dirs: list[Path],
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict]]:
    frames = []
    provenance = []
    selections = []
    expected_weights = None
    expected_selection = None
    for shard_index, feature_dir in enumerate(feature_dirs):
        metadata_path = feature_dir / "run_metadata.json"
        if not metadata_path.exists():
            raise ValueError(f"missing real-run metadata: {metadata_path}")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        weights_path = metadata.get("weights_path")
        if not metadata.get("weights_required") or not weights_path:
            raise ValueError(
                f"feature shard lacks required real checkpoint: {feature_dir}"
            )
        if expected_weights is None:
            expected_weights = weights_path
        elif weights_path != expected_weights:
            raise ValueError("feature shards used different checkpoint paths")
        features = pd.read_parquet(feature_dir / "group_features.parquet")
        if len(features) != int(metadata["n_sites"]):
            raise ValueError(
                f"row count mismatch in {feature_dir}: "
                f"{len(features)} vs {metadata['n_sites']}"
            )
        features["inference_shard"] = shard_index
        frames.append(features)
        selection = pd.read_csv(
            feature_dir / "track_selection.tsv", sep="\t", low_memory=False
        )
        comparable_selection = selection.sort_values(
            ["output_type", "track_index"], kind="stable"
        ).reset_index(drop=True)
        if expected_selection is None:
            expected_selection = comparable_selection
        elif not comparable_selection.equals(expected_selection):
            raise ValueError(
                "feature shards used different track-selection tables"
            )
        selection["inference_shard"] = shard_index
        selections.append(selection)
        provenance.append(metadata)
    merged = pd.concat(frames, ignore_index=True)
    if merged["site_id"].duplicated().any():
        examples = merged.loc[
            merged["site_id"].duplicated(keep=False), "site_id"
        ].head(10)
        raise ValueError(f"duplicate site ids: {examples.tolist()}")
    metadata_columns = {"site_id", "site_class", "inference_shard"}
    feature_columns = [
        column
        for column in merged.columns
        if column not in metadata_columns
    ]
    if merged[feature_columns].isna().any().any():
        raise ValueError("real grouped feature matrix contains missing values")
    return merged, pd.concat(selections, ignore_index=True), provenance


def _site_metadata(source_path: Path) -> pd.DataFrame:
    sites = pd.read_csv(source_path, sep="\t", low_memory=False)
    if sites["site_id"].duplicated().any():
        raise ValueError("source sites contain duplicate site_id")
    if "rpeak_ubiquity" in sites.columns:
        parsed = (
            sites["rpeak_ubiquity"]
            .fillna("")
            .astype(str)
            .str.extract(r"^\s*(\d+)\s*/\s*(\d+)\s*$")
        )
        sites["rpeak_count"] = pd.to_numeric(parsed[0], errors="coerce")
        sites["rpeak_fraction"] = (
            pd.to_numeric(parsed[0], errors="coerce")
            / pd.to_numeric(parsed[1], errors="coerce")
        )
    sites["has_ccre"] = (
        sites.get("cCRE", pd.Series("", index=sites.index))
        .fillna("")
        .astype(str)
        .str.strip()
        .ne("")
    )
    return sites


def _representations(
    features: pd.DataFrame,
    sites: pd.DataFrame,
    contact_summary: str | None,
) -> dict[str, pd.DataFrame]:
    identity = features[["site_id"]].copy()
    effect_columns = [
        column
        for column in features.columns
        if "__log2fc_" in column or column.endswith("__peak_log2fc")
    ]
    signed_depletion = [
        column
        for column in effect_columns
        if column.endswith(
            ("__log2fc_signed_mean", "__log2fc_depletion_mean")
        )
    ]
    local = [
        column
        for column in effect_columns
        if "__w1024__" in column or "__w4096__" in column
    ]
    matrices = {
        "real_biological_all_scales": pd.concat(
            [identity, features[effect_columns]], axis=1
        ),
        "real_signed_depletion": pd.concat(
            [identity, features[signed_depletion]], axis=1
        ),
        "real_local_1_4kb": pd.concat(
            [identity, features[local]], axis=1
        ),
    }
    candidate_covariates = (
        "peak_score",
        "motif_score",
        "rpeak_fraction",
        "rad21_ctrl_signal",
        "ctcf_ctrl_signal",
        "pol2_ctrl_signal",
    )
    covariate_columns = [
        column for column in candidate_covariates if column in sites.columns
    ]
    if covariate_columns:
        covariates = sites.set_index("site_id").loc[
            features["site_id"].astype(str), covariate_columns
        ]
        covariates = covariates.apply(pd.to_numeric, errors="coerce")
        covariates = covariates.loc[
            :, covariates.notna().any(axis=0) & covariates.nunique().gt(1)
        ]
        if not covariates.empty:
            covariates = covariates.fillna(covariates.median())
            covariate_values = covariates.to_numpy(dtype=float)
            covariate_values = (
                covariate_values
                - covariate_values.mean(axis=0, keepdims=True)
            ) / np.where(
                covariate_values.std(axis=0, keepdims=True) > 0,
                covariate_values.std(axis=0, keepdims=True),
                1.0,
            )
            design = np.column_stack(
                [np.ones(len(covariate_values)), covariate_values]
            )
            effect_values = features[effect_columns].to_numpy(dtype=float)
            coefficients, _, _, _ = np.linalg.lstsq(
                design, effect_values, rcond=None
            )
            residuals = effect_values - design @ coefficients
            residual_table = pd.DataFrame(
                residuals, columns=effect_columns
            )
            matrices["real_strength_residual"] = pd.concat(
                [identity, residual_table], axis=1
            )
    if contact_summary:
        contact = _load_contact_features(
            Path(contact_summary), features["site_id"].astype(str)
        )
        matrices["real_biological_contact"] = matrices[
            "real_biological_all_scales"
        ].merge(contact, on="site_id", validate="one_to_one")
    return matrices


def _shard_confound(
    assignments: dict[str, pd.DataFrame],
    features: pd.DataFrame,
) -> pd.DataFrame:
    from scipy.stats import chi2_contingency
    from sklearn.metrics import adjusted_rand_score

    shard_labels = features.set_index("site_id")["inference_shard"]
    rows = []
    if features["inference_shard"].nunique() < 2:
        return pd.DataFrame.from_records(
            [
                {
                    "representation": name,
                    "cluster_vs_inference_shard_ari": np.nan,
                    "cluster_vs_inference_shard_chi2": np.nan,
                    "cluster_vs_inference_shard_p": np.nan,
                    "max_shard_fraction_within_cluster": 1.0,
                }
                for name in assignments
            ]
        )
    for name, assignment in assignments.items():
        joined = assignment.copy()
        joined["inference_shard"] = joined["site_id"].map(shard_labels)
        contingency = pd.crosstab(
            joined["inference_shard"], joined["cluster"]
        )
        chi2, pvalue, _, _ = chi2_contingency(contingency)
        rows.append(
            {
                "representation": name,
                "cluster_vs_inference_shard_ari": float(
                    adjusted_rand_score(
                        joined["inference_shard"], joined["cluster"]
                    )
                ),
                "cluster_vs_inference_shard_chi2": float(chi2),
                "cluster_vs_inference_shard_p": float(pvalue),
                "max_shard_fraction_within_cluster": float(
                    contingency.div(contingency.sum(axis=0), axis=1)
                    .max(axis=0)
                    .max()
                ),
            }
        )
    return pd.DataFrame.from_records(rows)


def _plot_final(
    output_dir: Path,
    result_table: pd.DataFrame,
    assignments: dict[str, pd.DataFrame],
    shard_check: pd.DataFrame,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    figures = output_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    for axis, metric in zip(
        axes,
        ("silhouette", "stability_ari_mean", "min_cluster_size"),
    ):
        for name in result_table["name"]:
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
                linewidth=1.5,
                label=name,
            )
        axis.set_xlabel("Number of clusters (k)")
        axis.set_ylabel(metric.replace("_", " "))
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False, fontsize=7)
    fig.suptitle("Real-checkpoint cluster-number diagnostics")
    fig.text(
        0.5,
        0.01,
        "Matched-control-adjusted grouped 128 bp AlphaGenome effects",
        ha="center",
        color="#6F768A",
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.94))
    fig.savefig(figures / "k_selection.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    names = list(assignments)
    fig, axes = plt.subplots(
        1, len(names), figsize=(5.5 * len(names), 5), squeeze=False
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
    fig.suptitle("AlphaGenome mutation-response regimes")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(figures / "pca_clusters.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(9, 5))
    bars = shard_check.sort_values("cluster_vs_inference_shard_ari")
    axis.barh(
        bars["representation"],
        bars["cluster_vs_inference_shard_ari"],
        color="#A3BEFA",
        edgecolor="#1F2430",
    )
    axis.axvline(0, color="#464C55", linewidth=1)
    axis.set_xlabel("Adjusted Rand index")
    axis.set_ylabel("Feature representation")
    axis.set_title("Cluster assignments should not reproduce inference shards")
    axis.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(
        figures / "inference_shard_check.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(fig)


def _write_report(
    output_dir: Path,
    results: pd.DataFrame,
    cross_ari: pd.DataFrame,
    shard_check: pd.DataFrame,
    top_features: dict[str, pd.DataFrame],
    contact_included: bool,
) -> None:
    best = results.sort_values(
        ["stability_ari", "silhouette"], ascending=False
    ).iloc[0]
    best_name = str(best["name"])
    best_shard = shard_check.loc[
        shard_check["representation"].eq(best_name)
    ].iloc[0]
    cross_range = (
        "not available"
        if cross_ari.empty
        else (
            f"{cross_ari['adjusted_rand_index'].min():.3f} to "
            f"{cross_ari['adjusted_rand_index'].max():.3f}"
        )
    )
    rows = "\n".join(
        "<tr>"
        f"<td>{html.escape(str(row['name']))}</td>"
        f"<td>{int(row['selected_k'])}</td>"
        f"<td>{float(row['silhouette']):.3f}</td>"
        f"<td>{float(row['stability_ari']):.3f}</td>"
        f"<td>{int(row['min_cluster_size'])}</td>"
        f"<td>{int(row['n_features_after_qc'])}</td>"
        "</tr>"
        for _, row in results.iterrows()
    )
    top = top_features[best_name].head(15)
    top_items = "\n".join(
        f"<li><code>{html.escape(str(row.feature_name))}</code>: "
        f"cluster {int(row.cluster)}, centroid "
        f"{float(row.scaled_centroid):+.2f}</li>"
        for row in top.itertuples(index=False)
    )
    contact_text = (
        "Real contact-map summaries were included without persisting contact "
        "matrices."
        if contact_included
        else "Contact-map summaries were not included."
    )
    report = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>AlphaGenome batch ISM clustering</title>
<style>
body{{font-family:Inter,Arial,sans-serif;margin:0;background:#FCFCFD;color:#1F2430}}
main{{max-width:1120px;margin:auto;padding:40px 28px 80px}}
h1{{font-size:32px}}h2{{margin-top:38px;padding-top:24px;border-top:1px solid #E6E8F0}}
p,li{{line-height:1.65}}img{{width:100%;height:auto;background:white}}
table{{width:100%;border-collapse:collapse;background:white}}th,td{{padding:10px;border-bottom:1px solid #E6E8F0;text-align:left}}
th{{background:#F4F5F7}}.summary{{background:#EAF1FE;border-left:5px solid #5477C4;padding:18px 22px}}
.warn{{background:#FFF4C2;border-left:5px solid #B8A037;padding:16px 20px}}
code{{font-family:"SFMono-Regular",Consolas,monospace;font-size:.92em}}
</style></head><body><main>
<h1>AlphaGenome batch ISM clustering</h1>
<h2>Technical summary</h2>
<div class="summary"><p><strong>{html.escape(best_name)}</strong> was the most
stable representation: k={int(best['selected_k'])}, silhouette
{float(best['silhouette']):.3f}, stability ARI
{float(best['stability_ari']):.3f}, minimum cluster size
{int(best['min_cluster_size'])}. Cross-representation ARI was {cross_range}.</p>
<p>The inference-shard ARI for this representation was
{float(best_shard['cluster_vs_inference_shard_ari']):.3f}. A value near zero
supports that the structure is not a repeat of the three GPU shards.
{html.escape(contact_text)}</p></div>

<h2>Cluster number and robustness</h2>
<p>k=2..12 was evaluated with silhouette and 30 repeated 80% subsample
refits. Higher k is accepted only when it improves separation without
creating small unstable groups.</p>
<img src="figures/k_selection.png" alt="k selection">
<table><thead><tr><th>Representation</th><th>k</th><th>Silhouette</th>
<th>Stability ARI</th><th>Min cluster</th><th>Features</th></tr></thead>
<tbody>{rows}</tbody></table>

<h2>The response structure is compared across feature definitions</h2>
<p>The PCA views compare all multiscale biological effects, signed/depletion
effects, local 1-4 kb effects, and contact-enhanced effects when available.
Agreement across these views is stronger evidence than one high silhouette.</p>
<img src="figures/pca_clusters.png" alt="PCA clusters">

<h2>Inference shard is an explicit negative control</h2>
<p>Every shard must use the same explicit checkpoint and track-selection table.
The report includes shard ARI so inference batches cannot silently become the
dominant clustering signal.</p>
<img src="figures/inference_shard_check.png" alt="shard confound check">

<h2>Leading features</h2><ul>{top_items}</ul>

<h2>Scope and metric definitions</h2>
<p>AlphaGenome receives 1,048,576 bp sequence context. Selected biological
tracks are retained at 128 bp resolution and summarized over centered 1, 4,
20, and 100 kb windows. Effects use
<code>log2(1+ALT)-log2(1+REF)</code>.</p>

<h2>What this analysis establishes</h2>
<p>It identifies reproducible mutation-response regimes and the assay families
that distinguish them. It does not establish that the regimes are discrete
cell types, causal mechanisms, or paper-defined DIC classes.</p>

<h2>Limitations and robustness boundary</h2>
<div class="warn"><p>Model predictions are hypotheses, not experimental
validation. Track selection must be fixed before comparing site classes, and
class differences should be checked against signal-strength, distance, and
inference-shard confounds.</p></div>

<h2>Recommended next steps</h2>
<ol><li>Use the most stable biological representation for discovery and treat
other representations as sensitivity analyses.</li>
<li>Select representative sites from each stable regime for local profile and
distant-readout inspection rather than persisting full outputs for all sites.</li>
<li>Validate the main axes on matched motif-SNV and non-motif-control subsets,
stratified by motif score and CTCF ubiquity.</li></ol>

<h2>Further questions</h2>
<ul><li>Does contact response add an orthogonal axis beyond local
CTCF/cohesin depletion?</li><li>Do the regimes persist within one gene or one
chromatin state?</li><li>Which regimes survive matched-control calibration?</li></ul>
</main></body></html>"""
    (output_dir / "report.html").write_text(report, encoding="utf-8")


def main() -> None:
    args = _parse_args()
    feature_dirs = [Path(value) for value in args.feature_dir]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    features, selections, provenance = _load_real_features(feature_dirs)
    sites = _site_metadata(Path(args.source_sites))
    if args.site_class:
        selected_classes = set(args.site_class)
        features = features[
            features["site_class"].isin(selected_classes)
        ].reset_index(drop=True)
        sites = sites[
            sites["site_class"].isin(selected_classes)
        ].reset_index(drop=True)
        if features.empty:
            raise ValueError(
                f"No features remain for site classes {sorted(selected_classes)}"
            )
    feature_ids = set(features["site_id"].astype(str))
    source_ids = set(sites["site_id"].astype(str))
    if feature_ids != source_ids:
        raise ValueError(
            "real features do not exactly cover source sites: "
            f"missing={len(source_ids - feature_ids)}, "
            f"unexpected={len(feature_ids - source_ids)}"
        )
    sites = features[["site_id", "inference_shard"]].merge(
        sites, on="site_id", how="left", validate="one_to_one"
    )
    matrices = _representations(features, sites, args.contact_summary)

    results = []
    assignments = {}
    top_features = {}
    for offset, (name, matrix) in enumerate(matrices.items()):
        result, assignment, top = _run_representation(
            name,
            matrix,
            sites,
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
    shard_check = _shard_confound(assignments, features)
    shard_check.to_csv(
        output_dir / "inference_shard_confound.tsv", sep="\t", index=False
    )
    sites.to_csv(output_dir / "site_metadata.tsv", sep="\t", index=False)
    selections.to_csv(
        output_dir / "track_selection_all_shards.tsv", sep="\t", index=False
    )
    _plot_final(output_dir, result_table, assignments, shard_check)
    _write_report(
        output_dir,
        result_table,
        cross_ari,
        shard_check,
        top_features,
        args.contact_summary is not None,
    )
    (output_dir / "provenance.json").write_text(
        json.dumps(
            {
                "feature_dirs": [str(path) for path in feature_dirs],
                "source_sites": args.source_sites,
                "contact_summary": args.contact_summary,
                "site_classes": args.site_class,
                "real_inference_runs": provenance,
                "n_sites": len(features),
                "inference_shard_negative_control": True,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
