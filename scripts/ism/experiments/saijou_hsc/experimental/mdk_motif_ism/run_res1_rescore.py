#!/usr/bin/env python3
"""Re-score the frozen r5 Mdk candidate instances with the res1/bin32 AlphaGenome FT model.

The 20260911 res1 run reuses the SAME prepared/ manifest (identical
prepared_manifest_sha256) and the SAME readout windows as the 20260903 run that
r5 scored, so the only thing that changes is the model: output resolution
128bp -> 32bp, training labels res128 -> raw 1bp.

Scoring functions are imported from run_mdk_motif_ism.py so the contract
(effect formula, same-geometry background, percentile rule, HSC joint rule) is
bit-identical to r5. Candidate instances are read from r5 and never re-selected.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset as ds

REPO = Path(__file__).resolve().parents[6]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_mdk_motif_ism import (  # noqa: E402  frozen r5 scoring contract
    FEATURE_COLUMNS,
    backgrounds,
    bg_records,
    hsc_priority_pass,
    score_region,
    support,
)

R5 = REPO / "experiments/ism/20260909_mdk_motif_ism_v1_e1e6_r5"
OLD_PRIMARY = REPO / "experiments/ism/20260903_nine_gene_tss6kb_strict_shuffle"
RES1 = REPO / "experiments/ism/20260911_nine_gene_tss6kb_strict_shuffle_ag_res1"
OUT = REPO / "experiments/ism/20260913_mdk_motif_ism_res1_rescore"
READOUT = "gene_body_output_clipped"
CELLS = ("hsc", "mac", "lsec", "chol")


def load_res1_mdk() -> pd.DataFrame:
    """Mdk lives entirely in the gpu0 shard; every shard holds disjoint genes."""
    parts = []
    for shard in sorted((RES1 / "shards").glob("gpu*")):
        p = shard / "features/combined_mutation_features.parquet"
        t = ds.dataset(p, format="parquet").to_table(
            columns=FEATURE_COLUMNS, filter=ds.field("gene") == "Mdk"
        ).to_pandas()
        if len(t):
            parts.append(t)
    if not parts:
        raise SystemExit("no Mdk rows found in res1 shards")
    return pd.concat(parts, ignore_index=True)


def to_centers(x: pd.DataFrame) -> pd.DataFrame:
    x = x.copy()
    x["signed_effect"] = x.log2fc_ratio_of_sums
    x["absolute_effect"] = x.signed_effect.abs()
    keys = [
        "model_backend", "model_id", "gene", "track_id", "track_group",
        "track_modality", "track_strand", "readout_id", "readout_role",
        "edit_center_position", "variant_offset_from_tss_transcription_bp",
    ]
    return x.groupby(keys, dropna=False, sort=False).agg(
        median_absolute_effect=("absolute_effect", "median"),
        median_signed_effect=("signed_effect", "median"),
        replacement_mad=("signed_effect", lambda v: float(np.median(np.abs(v - np.median(v))))),
        replacements=("replacement_replicate", "nunique"),
        ref_sum=("ref_sum", "median"),
        alt_sum=("alt_sum", "median"),
        n_bins=("n_bins", "median"),
    ).reset_index()


def score(selected: pd.DataFrame, centers: pd.DataFrame, manifest: pd.DataFrame, label: str):
    rows, brows = [], []
    bgs = backgrounds(selected, manifest, "edit_center_position")
    z = centers[centers.readout_role.eq(READOUT)]
    piv = z.pivot(index="edit_center_position", columns="track_id", values="median_absolute_effect")
    if not set(CELLS) <= set(piv):
        raise SystemExit(f"{label} lacks the four cell tracks")
    values = {"S_RNA": piv.hsc, "S_HSC": piv.hsc - piv[["mac", "lsec", "chol"]].max(axis=1)}
    for r in selected.itertuples(index=False):
        s = support(manifest, r, "edit_center_position")
        calc = {}
        for sid, ser in values.items():
            stat, bg, pct, status = score_region(ser, s, bgs[r.instance_id])
            calc[sid] = (stat, pct, status)
            track = "hsc" if sid == "S_RNA" else "hsc_vs_max_other"
            brows += bg_records(r.instance_id, label, sid, track, READOUT,
                                "genomic_0_based_edit_center_position", bgs[r.instance_id], bg)
        rs, rp, rst = calc["S_RNA"]
        hc, hp, hst = calc["S_HSC"]
        rows.append(dict(instance_id=r.instance_id, backend=label, score_id="S_RNA", track_id="hsc",
                         readout_role=READOUT, raw_statistic=rs, percentile_score=rp,
                         background_n=len(bgs[r.instance_id]),
                         rank_pass=rp > 95 if rst == "ok" else np.nan, status=rst, reason=""))
        rows.append(dict(instance_id=r.instance_id, backend=label, score_id="S_HSC",
                         track_id="hsc_vs_max_other", readout_role=READOUT, raw_statistic=hc,
                         percentile_score=hp, background_n=len(bgs[r.instance_id]),
                         rank_pass=hsc_priority_pass(hc, hp, rp) if hst == rst == "ok" else np.nan,
                         status=hst, reason="requires C>0, S_HSC>95, and HSC S_RNA>95"))
        for cell in CELLS:
            rows.append(dict(instance_id=r.instance_id, backend=label, score_id="cell_descriptive",
                             track_id=cell, readout_role=READOUT,
                             raw_statistic=float(piv[cell].reindex(s).median()),
                             percentile_score=np.nan, background_n=np.nan, rank_pass=np.nan,
                             status="not_ranked", reason="descriptive_cell_effect"))
    return pd.DataFrame(rows), pd.DataFrame(brows)


def main() -> None:
    if OUT.exists():
        raise SystemExit(f"refusing overwrite: {OUT}")

    old_meta = json.loads((OLD_PRIMARY / "runs/alphagenome_finetuned/model_metadata.json").read_text())
    new_meta = json.loads((RES1 / "shards/gpu0/model_metadata.json").read_text())
    if old_meta["prepared_manifest_sha256"] != new_meta["prepared_manifest_sha256"]:
        raise SystemExit("prepared manifest differs; the two runs are not a controlled comparison")

    manifest = pd.read_csv(OLD_PRIMARY / "prepared/mutation_manifest.tsv", sep="\t")
    manifest = manifest[manifest.gene.eq("Mdk")]
    candidates = pd.read_csv(R5 / "candidate_instances.tsv", sep="\t")
    selected = candidates[candidates.selected_for_scoring.fillna(False)].copy()

    res1 = load_res1_mdk()
    centers_new = to_centers(res1)

    old = ds.dataset(OLD_PRIMARY / "runs/alphagenome_finetuned/features/combined_mutation_features.parquet",
                     format="parquet").to_table(columns=FEATURE_COLUMNS,
                                                filter=ds.field("gene") == "Mdk").to_pandas()
    centers_old = to_centers(old)

    OUT.mkdir(parents=True)
    new_scores, new_bg = score(selected, centers_new, manifest, "alphagenome_finetuned_res1")
    new_scores.to_csv(OUT / "region_scores_res1.tsv", sep="\t", index=False)
    new_bg.to_csv(OUT / "background_windows_res1.tsv", sep="\t", index=False)

    # side-by-side against the frozen r5 numbers
    r5_scores = pd.read_csv(R5 / "region_scores.tsv", sep="\t")
    r5_ag = r5_scores[r5_scores.backend.eq("alphagenome_finetuned")]
    merged = r5_ag.merge(
        new_scores, on=["instance_id", "score_id", "track_id", "readout_role"],
        suffixes=("_res128", "_res1"), validate="one_to_one",
    )
    merged = merged[["instance_id", "score_id", "track_id",
                     "raw_statistic_res128", "raw_statistic_res1",
                     "percentile_score_res128", "percentile_score_res1",
                     "rank_pass_res128", "rank_pass_res1"]]
    merged.to_csv(OUT / "score_comparison_res128_vs_res1.tsv", sep="\t", index=False)

    # per-center agreement on the primary readout / HSC track
    def hsc_series(centers):
        z = centers[centers.readout_role.eq(READOUT) & centers.track_id.eq("hsc")]
        return z.set_index("edit_center_position")[["median_absolute_effect", "median_signed_effect"]]

    a, b = hsc_series(centers_old), hsc_series(centers_new)
    common = a.index.intersection(b.index)
    agree = pd.DataFrame([dict(
        n_common_centers=len(common),
        spearman_absolute=float(a.loc[common].median_absolute_effect.corr(
            b.loc[common].median_absolute_effect, method="spearman")),
        pearson_absolute=float(a.loc[common].median_absolute_effect.corr(
            b.loc[common].median_absolute_effect, method="pearson")),
        spearman_signed=float(a.loc[common].median_signed_effect.corr(
            b.loc[common].median_signed_effect, method="spearman")),
        median_absolute_effect_res128=float(a.loc[common].median_absolute_effect.median()),
        median_absolute_effect_res1=float(b.loc[common].median_absolute_effect.median()),
        ratio_res1_over_res128=float(b.loc[common].median_absolute_effect.median()
                                     / a.loc[common].median_absolute_effect.median()),
    )])
    agree.to_csv(OUT / "center_agreement_res128_vs_res1.tsv", sep="\t", index=False)

    json.dump({
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "scoring_contract": "imported verbatim from run_mdk_motif_ism.py (r5)",
        "candidates": "frozen r5 candidate_instances.tsv; not re-selected",
        "res128_model": {k: old_meta[k] for k in ("checkpoint_path", "checkpoint_epoch", "output_resolution_bp", "checkpoint_sha256")},
        "res1_model": {k: new_meta[k] for k in ("checkpoint_path", "checkpoint_epoch", "output_resolution_bp", "checkpoint_sha256")},
        "shared_prepared_manifest_sha256": new_meta["prepared_manifest_sha256"],
        "readout_bins": {
            "res128_gene_body_bins": int(centers_old[centers_old.readout_role.eq(READOUT)].n_bins.iloc[0]),
            "res1_gene_body_bins": int(centers_new[centers_new.readout_role.eq(READOUT)].n_bins.iloc[0]),
        },
        "limitation": "same physical readout window, different bin granularity; per-bin pseudocount alpha=1 therefore normalises differently between the two runs",
    }, open(OUT / "res1_rescore_status.json", "w"), indent=2, sort_keys=True)
    print(OUT)


if __name__ == "__main__":
    main()
