#!/usr/bin/env python
"""Compare the final Mdk background scan with the independent all-gene manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_ROOT = REPO_ROOT / "experiments/ism/saijou_all_genes_10bp_scan"
RUNS = {
    ("final_mdk_background", "AlphaGenome e19"): (
        REPO_ROOT
        / "experiments/ism/saijou_targeted_original_comparison/specificity_background/alphagenome_e19/features/Mdk.tsv",
        "gene_body",
    ),
    ("final_mdk_background", "Borzoi e39"): (
        REPO_ROOT
        / "experiments/ism/saijou_targeted_original_comparison/specificity_background/borzoi_e39/features/Mdk.tsv",
        "gene_body",
    ),
    ("all_gene_manifest", "AlphaGenome e19"): (
        DEFAULT_ROOT / "runs/alphagenome_finetuned/features/Mdk.tsv",
        "gene_body_output_clipped",
    ),
    ("all_gene_manifest", "Borzoi e39"): (
        DEFAULT_ROOT / "runs/borzoi_finetuned/features/Mdk.tsv",
        "gene_body_output_clipped",
    ),
}
CELLS = ["hsc", "mac", "lsec", "chol"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--hub-start", type=int, default=441)
    parser.add_argument("--hub-end", type=int, default=507)
    return parser.parse_args()


def center_table(path: Path, role: str) -> pd.DataFrame:
    data = pd.read_csv(path, sep="\t")
    data = data.loc[data.readout_role.eq(role) & data.track_id.isin(CELLS)].copy()
    data["abs_effect"] = data.log2fc_ratio_of_sums.abs()
    grouped = (
        data.groupby(["variant_offset_from_tss_transcription_bp", "track_id"], sort=False)
        .agg(
            median_abs_effect=("abs_effect", "median"),
            median_signed_effect=("log2fc_ratio_of_sums", "median"),
            shuffles=("mutation_id", "nunique"),
        )
        .reset_index()
    )
    absolute = grouped.pivot(
        index="variant_offset_from_tss_transcription_bp",
        columns="track_id",
        values="median_abs_effect",
    )
    signed = grouped.pivot(
        index="variant_offset_from_tss_transcription_bp",
        columns="track_id",
        values="median_signed_effect",
    ).add_suffix("_signed")
    table = absolute.join(signed).reset_index()
    table["max_other"] = table[["mac", "lsec", "chol"]].max(axis=1)
    table["hsc_ratio"] = table.hsc / table.max_other.clip(lower=1e-12)
    table["hsc_margin"] = table.hsc - table.max_other
    table["top_cell"] = table[CELLS].idxmax(axis=1)
    return table


def _run_path(root: Path, manifest_label: str, model: str, path: Path) -> Path:
    if manifest_label != "all_gene_manifest":
        return path
    backend = (
        "alphagenome_finetuned"
        if model.startswith("AlphaGenome")
        else "borzoi_finetuned"
    )
    return root / f"runs/{backend}/features/Mdk.tsv"


def _hub(table: pd.DataFrame, start: int, end: int) -> pd.DataFrame:
    return table.loc[
        table.variant_offset_from_tss_transcription_bp.between(start, end)
    ]


def _hub_summary(
    table: pd.DataFrame,
    *,
    manifest_label: str,
    model: str,
    start: int,
    end: int,
) -> dict[str, object]:
    hub = _hub(table, start, end)
    cell_medians = hub[CELLS].median()
    max_other = float(cell_medians[["mac", "lsec", "chol"]].max())
    return {
        "manifest_label": manifest_label,
        "model": model,
        "hub_start": start,
        "hub_end": end,
        "centers": int(len(hub)),
        "hsc_median_abs_effect": float(cell_medians.hsc),
        "max_other_median_abs_effect": max_other,
        "hsc_vs_max_other_ratio": float(cell_medians.hsc / max(max_other, 1e-12)),
        "hsc_margin": float(cell_medians.hsc - max_other),
        "hsc_top_center_fraction": float(hub.top_cell.eq("hsc").mean()),
        "hsc_median_signed_effect": float(hub.hsc_signed.median()),
    }


def _load_runs(
    root: Path, start: int, end: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    tables = []
    summaries = []
    for (manifest_label, model), (path, role) in RUNS.items():
        table = center_table(_run_path(root, manifest_label, model, path), role)
        table["manifest_label"] = manifest_label
        table["model"] = model
        tables.append(table)
        summaries.append(
            _hub_summary(
                table,
                manifest_label=manifest_label,
                model=model,
                start=start,
                end=end,
            )
        )
    return pd.concat(tables, ignore_index=True), pd.DataFrame.from_records(summaries)


def _concordance_record(
    hub: pd.DataFrame,
    *,
    comparison: str,
    model: str,
    left_suffix: str,
    right_suffix: str,
) -> dict[str, object]:
    left_margin = hub[f"hsc_margin_{left_suffix}"]
    right_margin = hub[f"hsc_margin_{right_suffix}"]
    left_abs = hub[f"hsc_{left_suffix}"]
    right_abs = hub[f"hsc_{right_suffix}"]
    left_signed = hub[f"hsc_signed_{left_suffix}"]
    right_signed = hub[f"hsc_signed_{right_suffix}"]
    return {
        "comparison": comparison,
        "model": model,
        "centers": len(hub),
        "hsc_margin_spearman": float(spearmanr(left_margin, right_margin).statistic),
        "hsc_abs_effect_spearman": float(spearmanr(left_abs, right_abs).statistic),
        "hsc_signed_effect_spearman": float(
            spearmanr(left_signed, right_signed).statistic
        ),
        "hsc_signed_direction_agreement": float(
            (np.sign(left_signed) == np.sign(right_signed)).mean()
        ),
    }


def _join_pair(
    left: pd.DataFrame,
    right: pd.DataFrame,
    *,
    suffixes: tuple[str, str],
    start: int,
    end: int,
) -> pd.DataFrame:
    joined = left.merge(
        right,
        on="variant_offset_from_tss_transcription_bp",
        suffixes=suffixes,
        validate="one_to_one",
    )
    return _hub(joined, start, end)


def _manifest_concordance(
    centers: pd.DataFrame, models: list[str], start: int, end: int
) -> list[dict]:
    rows = []
    for model in models:
        left = centers.loc[
            centers.model.eq(model)
            & centers.manifest_label.eq("final_mdk_background")
        ]
        right = centers.loc[
            centers.model.eq(model) & centers.manifest_label.eq("all_gene_manifest")
        ]
        hub = _join_pair(
            left, right, suffixes=("_old", "_new"), start=start, end=end
        )
        rows.append(
            _concordance_record(
                hub,
                comparison="old_vs_all_gene_manifest",
                model=model,
                left_suffix="old",
                right_suffix="new",
            )
        )
    return rows


def _model_concordance(
    centers: pd.DataFrame, manifests: list[str], start: int, end: int
) -> list[dict]:
    rows = []
    for manifest_label in manifests:
        ag = centers.loc[
            centers.manifest_label.eq(manifest_label)
            & centers.model.eq("AlphaGenome e19")
        ]
        bz = centers.loc[
            centers.manifest_label.eq(manifest_label)
            & centers.model.eq("Borzoi e39")
        ]
        hub = _join_pair(ag, bz, suffixes=("_ag", "_bz"), start=start, end=end)
        rows.append(
            _concordance_record(
                hub,
                comparison="alphagenome_vs_borzoi",
                model=manifest_label,
                left_suffix="ag",
                right_suffix="bz",
            )
        )
    return rows


def _validation_summary(summary: pd.DataFrame, concordance: pd.DataFrame) -> dict:
    validation = {
        "status": "ok",
        "hub_centers_per_run": summary.centers.unique().tolist(),
        "runs": int(len(summary)),
        "concordance_rows": int(len(concordance)),
        "nonfinite_values": int(
            (~np.isfinite(concordance.select_dtypes(include=["number"]).to_numpy())).sum()
        ),
    }
    if validation["nonfinite_values"]:
        raise RuntimeError(validation)
    return validation


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    out = root / "analysis/mdk_manifest_robustness"
    out.mkdir(parents=True, exist_ok=True)
    centers, summary = _load_runs(root, args.hub_start, args.hub_end)
    concordance = pd.DataFrame.from_records(
        _manifest_concordance(
            centers, summary.model.unique().tolist(), args.hub_start, args.hub_end
        )
        + _model_concordance(
            centers,
            summary.manifest_label.unique().tolist(),
            args.hub_start,
            args.hub_end,
        )
    )
    centers.to_csv(out / "center_metrics.tsv", sep="\t", index=False)
    summary.to_csv(out / "hub_summary.tsv", sep="\t", index=False)
    concordance.to_csv(out / "concordance.tsv", sep="\t", index=False)
    validation = _validation_summary(summary, concordance)
    (out / "validation_summary.json").write_text(json.dumps(validation, indent=2) + "\n")
    print(json.dumps(validation, indent=2))


if __name__ == "__main__":
    main()
