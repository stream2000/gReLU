"""Summarize WT1 counterfactual effects on every original AlphaGenome RNA/CAGE track.

Experimental. Effect = log2(ALT_sum / REF_sum) per variant x track x readout.
Strand: Mdk is on the minus strand, so stranded tracks are restricted to "-";
unstranded RNA tracks are kept. RNA uses the exon readouts, CAGE uses tss512.
A track is "readable" when its REF Mdk signal is at least the 25th percentile
of that assay's tracks (low-signal tracks give unstable ratios).
Track-specific effect = effect minus the median effect over readable tracks of
the same assay for the same variant (removes the shared sequence effect).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("experiments/ism/20260918_1349_wt1_exact_counterfactual")
RUN = ROOT / "runs/alphagenome_original_all_tracks"
META = Path("src/alphagenome_pytorch/src/alphagenome_pytorch/data/track_metadata_mouse.parquet")
DZ = 0.005  # |log2FC| <= DZ counts as unchanged (~0.35%)


def load() -> pd.DataFrame:
    shards = [np.load(p, allow_pickle=True) for p in sorted(RUN.glob("sums_shard*.npz"))]
    alt = np.concatenate([s["alt"] for s in shards])
    ref = np.concatenate([s["ref"] for s in shards])
    mids = np.concatenate([s["mutation_id"] for s in shards])
    readouts = list(shards[0]["readouts"])
    ch = pd.read_csv(RUN / "channels.tsv", sep="\t")
    meta = pd.read_parquet(META)
    ch = ch.merge(meta, on=["output_type", "track_index"], how="left")
    keep = (ch.ontology_curie.notna() & (ch.ontology_curie != "")
            & ch.strand.isin(["-", "."])).to_numpy()
    rows = []
    for r_name, r_idx in [("exon", 0), ("exon_noedit", 1), ("gene_window", 2), ("tss512", 3)]:
        for assay in ("rna_seq", "cage"):
            if (assay == "cage") != (r_name == "tss512"):
                continue
            sel = keep & (ch.output_type == assay).to_numpy()
            a, r = alt[:, sel, r_idx], ref[:, sel, r_idx]
            lfc = np.log2(np.maximum(a, 1e-6) / np.maximum(r, 1e-6))
            cs = ch[sel].reset_index(drop=True)
            for j, c in cs.iterrows():
                rows.append(pd.DataFrame(dict(
                    mutation_id=mids, readout=r_name, output_type=assay,
                    track_index=c.track_index, biosample=c.biosample_name,
                    biosample_type=c.biosample_type, life_stage=c.biosample_life_stage,
                    assay_title=c.assay_title, strand=c.strand, ref=r[:, j], lfc=lfc[:, j])))
    df = pd.concat(rows, ignore_index=True)
    man = pd.read_csv(ROOT / "prepared/variant_design.tsv", sep="\t")
    cols = [c for c in ["mutation_id", "locus_id", "wt1_score_delta", "design_class"] if c in man]
    if "wt1_score_delta" not in man:
        pe = pd.read_csv(ROOT / "analysis/panel_effects.tsv", sep="\t",
                         usecols=["mutation_id", "locus_id", "wt1_score_delta", "design_class"])
        man, cols = pe.drop_duplicates("mutation_id"), ["mutation_id", "locus_id", "wt1_score_delta", "design_class"]
    df = df.merge(man[cols].drop_duplicates("mutation_id"), on="mutation_id", how="left")
    df["motif"] = np.select([df.wt1_score_delta <= -8, df.wt1_score_delta >= -2], ["destroyed", "kept"], "partial")
    return df


def main() -> None:
    df = load()
    thr = df.drop_duplicates(["readout", "output_type", "track_index"]).groupby(["readout", "output_type"]).ref.quantile(0.25)
    df["readable"] = df.ref >= df.set_index(["readout", "output_type"]).index.map(thr)
    med = df[df.readable].groupby(["mutation_id", "readout", "output_type"]).lfc.median().rename("lfc_median_all")
    df = df.join(med, on=["mutation_id", "readout", "output_type"])
    df["lfc_specific"] = df.lfc - df.lfc_median_all

    grp = ["locus_id", "motif", "readout", "output_type", "track_index", "biosample", "life_stage", "assay_title"]
    s = df.groupby(grp, dropna=False).agg(
        n=("lfc", "size"), ref=("ref", "first"), readable=("readable", "first"),
        median_pct=("lfc", lambda x: 100 * (2 ** x.median() - 1)),
        up_pct=("lfc", lambda x: 100 * (x > DZ).mean()),
        flat_pct=("lfc", lambda x: 100 * (x.abs() <= DZ).mean()),
        down_pct=("lfc", lambda x: 100 * (x < -DZ).mean()),
        specific_median_pct=("lfc_specific", lambda x: 100 * (2 ** x.median() - 1)),
    ).reset_index()
    s.to_csv(RUN / "track_summary.tsv", sep="\t", index=False)
    df.to_parquet(RUN / "variant_track_effects.parquet")
    print(s.groupby(["locus_id", "motif", "readout", "output_type"]).size())


if __name__ == "__main__":
    main()
