"""Build gene-symbol x sample expression tables for the track-similarity prototype.

Experimental (see ANALYSIS_HARNESS.md). Inputs are downloaded into
``<out>/raw`` beforehand (ENCODE gene quantifications, FANTOM5 gene counts).

Units written to ``prepared/expr_cpm.parquet``: counts-per-million-like
molecule fractions, so every column sums to 1e6.
  * ENCODE RNA-seq: RSEM TPM (length-corrected, closest to UMI molecule counts),
    averaged over the selected files of one ontology x assay sample.
  * FANTOM5 CAGE: gene-level tag counts, summed over libraries of one
    description group, then CPM.
  * Saijou: pseudobulk CPM bigWig summed over the union of exons per gene
    (mm10 GTF, 1-based closed converted to 0-based half-open), then CPM.
"""

from __future__ import annotations

import argparse
import re
import urllib.parse
from pathlib import Path

import numpy as np
import pandas as pd
import pyBigWig

REPO = Path(__file__).resolve().parents[6]
AG_META = REPO / "src/alphagenome_pytorch/src/alphagenome_pytorch/data/track_metadata_mouse.parquet"
GTF = Path("/work/Database/Database_fromDocker/Referencedata_mm10/gtf_chrUCSC/chr.gtf")
BIGWIG_DIR = Path("/work2/Projects/Project_DL_pillar/Finetune_Borzoi_Saijou_scRNAseq/bigwig")
SAIJOU_GROUPS = ("hsc", "mac", "lsec", "chol")


def read_gtf_exons() -> pd.DataFrame:
    rows = []
    pat = re.compile(r'gene_id "([^"]+)".*?gene_name "([^"]+)".*?gene_biotype "([^"]+)"')
    with open(GTF) as fh:
        for line in fh:
            f = line.split("\t", 8)
            if len(f) < 9 or f[2] != "exon":
                continue
            mt = pat.search(f[8])
            if mt:
                rows.append((f[0], int(f[3]) - 1, int(f[4]), mt.group(1), mt.group(2), mt.group(3)))
    return pd.DataFrame(rows, columns=["chrom", "start", "end", "gene_id", "gene_name", "biotype"])


def merged_exons(ex: pd.DataFrame):
    for (gid, chrom), g in ex.groupby(["gene_id", "chrom"], sort=False):
        iv = sorted(zip(g.start, g.end))
        cur_s, cur_e = iv[0]
        for s, e in iv[1:]:
            if s <= cur_e:
                cur_e = max(cur_e, e)
            else:
                yield gid, chrom, cur_s, cur_e
                cur_s, cur_e = s, e
        yield gid, chrom, cur_s, cur_e


def saijou_from_bigwig(ex: pd.DataFrame) -> pd.DataFrame:
    ivs = pd.DataFrame(list(merged_exons(ex)), columns=["gene_id", "chrom", "start", "end"])
    out = {}
    for grp in SAIJOU_GROUPS:
        bw = pyBigWig.open(str(BIGWIG_DIR / f"{grp}.CPM.mapq10.bw"))
        chroms = bw.chroms()
        vals = np.zeros(len(ivs))
        for i, (c, s, e) in enumerate(zip(ivs.chrom, ivs.start, ivs.end)):
            if c in chroms and e <= chroms[c]:
                v = bw.stats(c, s, e, type="sum", exact=True)[0]
                vals[i] = v or 0.0
        bw.close()
        out[f"saijou_{grp}"] = pd.Series(vals).groupby(ivs.gene_id.values).sum()
    return pd.DataFrame(out)


def encode_table(raw: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    sel = pd.read_csv(raw / "encode_selected_files.tsv", sep="\t")
    cols, meta = {}, []
    for (curie, assay), g in sel.groupby(["ontology_curie", "assay_title"]):
        tpms, raw_counts = [], []
        for acc in g.accession:
            path = raw / "encode" / f"{acc}.tsv"
            if open(path).readline().startswith("gene_id"):
                t = pd.read_csv(path, sep="\t", usecols=["gene_id", "TPM"])
            else:  # headerless gene-count table (no lengths): only used if no RSEM file exists
                t = pd.read_csv(path, sep="\t", header=None, usecols=[0, 1], names=["gene_id", "TPM"])
            t = t[t.gene_id.astype(str).str.startswith("ENSMUSG")]
            t["gene_id"] = t.gene_id.str.split(".").str[0]
            s = t.groupby("gene_id").TPM.sum()
            (tpms if open(path).readline().startswith("gene_id") else raw_counts).append(s / s.sum() * 1e6)
        tpms = tpms or raw_counts
        sid = f"encode|{assay}|{curie}"
        cols[sid] = pd.concat(tpms, axis=1).fillna(0).mean(axis=1)
        meta.append(dict(sample_id=sid, source="ENCODE", assay=assay, ontology_curie=curie,
                         biosample_name=g.biosample_name.iloc[0], n_files=len(g),
                         accessions=",".join(g.accession)))
    return pd.DataFrame(cols).fillna(0), pd.DataFrame(meta)


def fantom_group(col: str) -> str:
    desc = urllib.parse.unquote(col.split(".CNhs")[0].removeprefix("counts."))
    desc = re.sub(r",?\s*(donor|pool|biol_rep|rep|tech_rep)\s*\d+.*$", "", desc, flags=re.I)
    return desc.strip(" ,")


def fantom_table(raw: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    t = pd.read_csv(raw / "fantom_genes_counts.osc.txt.gz", sep="\t", comment="#", index_col=0)
    t = t[[c for c in t.columns if c.startswith("counts.")]]
    groups = pd.Series({c: fantom_group(c) for c in t.columns})
    agg = t.T.groupby(groups).sum().T
    agg.columns = [f"fantom|{g}" for g in agg.columns]
    meta = pd.DataFrame(dict(sample_id=agg.columns, source="FANTOM5", assay="CAGE",
                             biosample_name=[c.split("|", 1)[1] for c in agg.columns],
                             n_files=groups.value_counts().reindex([c.split("|", 1)[1] for c in agg.columns]).values))
    return agg, meta


def to_cpm(df: pd.DataFrame) -> pd.DataFrame:
    return df / df.sum(axis=0) * 1e6


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    raw, prep = args.out / "raw", args.out / "prepared"
    prep.mkdir(parents=True, exist_ok=True)

    ex = read_gtf_exons()
    id2sym = ex.drop_duplicates("gene_id").set_index("gene_id")
    keep = id2sym.index[id2sym.biotype == "protein_coding"]

    sj = saijou_from_bigwig(ex[ex.gene_id.isin(keep)])
    enc, enc_meta = encode_table(raw)
    fan, fan_meta = fantom_table(raw)

    def by_symbol(df):
        df = df.loc[df.index.intersection(keep)]
        return df.groupby(id2sym.loc[df.index, "gene_name"].values).sum()

    sj_s, enc_s = by_symbol(sj), by_symbol(enc)
    fan_s = fan.loc[fan.index.intersection(set(id2sym.loc[keep, "gene_name"]))]
    genes = sj_s.index.intersection(enc_s.index).intersection(fan_s.index)
    expr = pd.concat([sj_s.loc[genes], enc_s.loc[genes], fan_s.loc[genes]], axis=1)
    expr = to_cpm(expr.astype(float))
    expr.to_parquet(prep / "expr_cpm.parquet")

    sj_meta = pd.DataFrame(dict(sample_id=sj.columns, source="Saijou", assay="10x3p_pseudobulk",
                                biosample_name=[c.split("_", 1)[1] for c in sj.columns], n_files=1))
    meta = pd.concat([sj_meta, enc_meta, fan_meta], ignore_index=True)

    # Flag FANTOM groups whose description matches an AlphaGenome CAGE biosample name (rough).
    ag = pd.read_parquet(AG_META)
    ag_cage = ag[(ag.output_type == "cage") & ag.ontology_curie.notna()]
    names = {n.lower() for n in ag_cage.biosample_name.dropna()}
    meta["ag_cage_name_hit"] = [
        next((n for n in names if n in str(b).lower()), "") if s == "FANTOM5" else "" for s, b in
        zip(meta.source, meta.biosample_name)
    ]
    meta.to_csv(prep / "samples.tsv", sep="\t", index=False)
    print(f"genes={len(genes)} samples={expr.shape[1]} "
          f"(saijou={sj.shape[1]}, encode={enc.shape[1]}, fantom={fan.shape[1]})")


if __name__ == "__main__":
    main()
