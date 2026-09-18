#!/usr/bin/env python3
"""Re-score the seven registered positive controls on res128 vs res1 AlphaGenome FT.

Same scoring contract as the Mdk re-score (imported from run_mdk_motif_ism.py):
region statistic = median over covered centers of the median absolute
ratio-of-sums effect on the HSC gene_body_output_clipped readout, ranked
against same-geometry background windows inside that gene's own scan.

Controls are read from the frozen 20260729 audit and are NOT re-selected.
Only AlphaGenome FT is compared; no other backend is touched.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset as ds

REPO = Path(__file__).resolve().parents[6]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_mdk_motif_ism import FEATURE_COLUMNS, percentile  # noqa: E402

OLD = REPO / "experiments/ism/20260903_nine_gene_tss6kb_strict_shuffle"
RES1 = REPO / "experiments/ism/20260911_nine_gene_tss6kb_strict_shuffle_ag_res1"
AUDIT = REPO / "experiments/ism/positive_control_global_audit_20260729/positive_control_evidence.tsv"
OUT = REPO / "experiments/ism/20260913_positive_control_res1_rescore"
READOUT = "gene_body_output_clipped"


def tx_to_genomic(start: int, end: int, tss: int, strand: str) -> tuple[int, int]:
    """Half-open transcript-direction interval -> half-open genomic interval."""
    if strand == "+":
        return tss + start, tss + end
    return tss - end, tss - start


def gene_centers(path: Path, gene: str) -> pd.DataFrame:
    d = ds.dataset(path, format="parquet").to_table(
        columns=FEATURE_COLUMNS,
        filter=(ds.field("gene") == gene)
        & (ds.field("readout_role") == READOUT)
        & (ds.field("track_id") == "hsc"),
    ).to_pandas()
    d["a"] = d.log2fc_ratio_of_sums.abs()
    return d.groupby("edit_center_position").a.median()


def same_geometry_background(manifest: pd.DataFrame, gs: int, ge: int) -> tuple[list[int], list[list[int]]]:
    """Reproduce the r5 background rule: identical relative center geometry,
    slid across every center of this gene's scan, dropping windows that fall on
    a coverage gap or touch the candidate's own support."""
    centers = sorted(manifest.edit_center_position.unique())
    cset = set(centers)
    support = sorted(manifest[(manifest.edit_start < ge) & (manifest.edit_end > gs)].edit_center_position.unique())
    if not support:
        return [], []
    rel = [c - support[0] for c in support]
    excluded = set(support)
    windows = []
    for start in centers:
        w = [start + d for d in rel]
        if all(x in cset for x in w) and not set(w) & excluded:
            windows.append(w)
    return support, windows


def main() -> None:
    if OUT.exists():
        raise SystemExit(f"refusing overwrite: {OUT}")

    controls = pd.read_csv(AUDIT, sep="\t")
    genes = pd.read_csv(OLD / "prepared/genes.tsv", sep="\t").set_index("gene")
    manifest = pd.read_csv(OLD / "prepared/mutation_manifest.tsv", sep="\t")

    shard_for = {}
    for shard in sorted((RES1 / "shards").glob("gpu*")):
        for g in json.loads((shard / "validation_summary.json").read_text())["genes"]:
            shard_for[g] = shard / "features/combined_mutation_features.parquet"
    old_parquet = OLD / "runs/alphagenome_finetuned/features/combined_mutation_features.parquet"

    cache: dict[tuple[str, str], pd.Series] = {}
    rows = []
    for c in controls.itertuples(index=False):
        g = genes.loc[c.gene]
        gs, ge = tx_to_genomic(int(c.start), int(c.end), int(g.analysis_tss), str(g.strand))
        sub = manifest[manifest.gene.eq(c.gene)]
        support, windows = same_geometry_background(sub, gs, ge)
        rec = dict(control_id=c.control_id, gene=c.gene, family=c.family, label=c.label,
                   tx_start=int(c.start), tx_end=int(c.end), genomic_start=gs, genomic_end=ge,
                   covered_centers=len(support), background_n=len(windows),
                   audit_2026_07_29_finetuned_best_score=float(c.finetuned_best_score))
        for tag, path in (("res128", old_parquet), ("res1", shard_for[c.gene])):
            key = (tag, c.gene)
            if key not in cache:
                cache[key] = gene_centers(path, c.gene)
            ser = cache[key]
            if not support or set(support) - set(ser.index):
                rec[f"raw_{tag}"] = np.nan
                rec[f"score_{tag}"] = np.nan
                rec[f"status_{tag}"] = "missing_center_coverage"
                continue
            stat = float(ser.reindex(support).median())
            bg = np.asarray([ser.reindex(w).median() for w in windows], float)
            pct, status = percentile(stat, bg)
            rec[f"raw_{tag}"] = stat
            rec[f"score_{tag}"] = pct
            rec[f"status_{tag}"] = status
        rows.append(rec)

    out = pd.DataFrame(rows)
    out["score_delta"] = out.score_res1 - out.score_res128
    out["raw_ratio_res1_over_res128"] = out.raw_res1 / out.raw_res128
    out["pass_res128"] = out.score_res128 > 95
    out["pass_res1"] = out.score_res1 > 95
    OUT.mkdir(parents=True)
    out.to_csv(OUT / "positive_control_score_comparison.tsv", sep="\t", index=False)
    print(out[["control_id", "gene", "family", "covered_centers", "background_n",
               "raw_res128", "raw_res1", "score_res128", "score_res1", "score_delta",
               "pass_res128", "pass_res1"]].to_string(index=False))
    print()
    print(f"pass>95  res128: {int(out.pass_res128.sum())}/7   res1: {int(out.pass_res1.sum())}/7")
    print(f"score improved in res1: {int((out.score_delta > 0).sum())}/7   median delta {out.score_delta.median():+.2f}")
    print(OUT)


if __name__ == "__main__":
    main()
