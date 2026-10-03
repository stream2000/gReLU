"""Score Saijou groups against AlphaGenome-training bulk samples (M0 + UCE).

Readout selection is frozen before looking at results (plan v2, section 4):
a bulk sample is "HSC-like" when it is in the top 5% of HSC similarity within its
source (ENCODE RNA / FANTOM5 CAGE are ranked separately) and spec[HSC] > 1, in
both M0 and UCE, and the Mac control passes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import anndata as ad
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

QUERIES = ["saijou_hsc", "saijou_mac", "saijou_lsec", "saijou_chol"]
HSC_MARKERS = ["Lrat", "Des", "Pdgfrb", "Reln", "Dcn", "Colec11"]
NEG_MARKERS = ["Ptprc", "Pecam1"]
MAC_PATTERN = "macrophage|microglia|kupffer"


def cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a / np.linalg.norm(a, axis=-1, keepdims=True)
    b = b / np.linalg.norm(b, axis=-1, keepdims=True)
    return a @ b.T


def score_table(S: pd.DataFrame, method: str, meta: pd.DataFrame) -> pd.DataFrame:
    """S: queries x bulk samples similarity. Adds within-source rank pct and spec."""
    spec = (S - S.mean(axis=0)) / S.std(axis=0, ddof=0)
    rows = []
    src = meta.set_index("sample_id").source.reindex(S.columns)
    for q in S.index:
        for source in src.unique():
            cols = src.index[src == source]
            pct = S.loc[q, cols].rank(pct=True, ascending=True)
            for s in cols:
                rows.append(dict(method=method, query_group=q, sample_id=s, source=source,
                                 S=S.loc[q, s], rank_pct=pct[s], spec=spec.loc[q, s]))
    return pd.DataFrame(rows)


def m0_similarity(expr: pd.DataFrame, bulk: list[str]) -> pd.DataFrame:
    lg = np.log1p(expr)
    sj = lg[QUERIES]
    genes = sj[(expr[QUERIES].max(axis=1) > 10)].var(axis=1).nlargest(2000).index
    genes = genes.union(pd.Index([g for g in HSC_MARKERS + NEG_MARKERS if g in lg.index]))
    out = {}
    for source_cols in (bulk,):
        z = lg.loc[genes, QUERIES + source_cols]
        z = z.sub(z.mean(axis=1), axis=0).div(z.std(axis=1) + 1e-6, axis=0)
        rho = spearmanr(z.to_numpy()).statistic
        rho = pd.DataFrame(rho, index=z.columns, columns=z.columns)
        out = rho.loc[QUERIES, source_cols]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n-boot", type=int, default=200)
    args = ap.parse_args()
    an = args.out / "analysis"
    fig = args.out / "figures"
    an.mkdir(exist_ok=True)
    fig.mkdir(exist_ok=True)

    expr = pd.read_parquet(args.out / "prepared/expr_cpm.parquet")
    meta = pd.read_csv(args.out / "prepared/samples.tsv", sep="\t")
    bulk = [s for s in expr.columns if s not in QUERIES]

    # ---- UCE ----
    shards = sorted((args.out / "uce_output").glob("*_uce_adata.h5ad"))
    emb = ad.concat([ad.read_h5ad(p) for p in shards])
    E = pd.DataFrame(emb.obsm["X_uce"], index=emb.obs_names)
    sid = emb.obs.sample_id.astype(str)
    centroids = E.groupby(sid.values).mean()
    S_uce = pd.DataFrame(cosine(centroids.loc[QUERIES].to_numpy(), centroids.loc[bulk].to_numpy()),
                         index=QUERIES, columns=bulk)

    # Bootstrap over pseudocells -> rank interval for each (query, sample).
    rng = np.random.default_rng(0)
    groups = {s: E.index[sid.values == s] for s in centroids.index}
    src = meta.set_index("sample_id").source.reindex(bulk)
    boot_ranks = []
    for _ in range(args.n_boot):
        c = np.stack([E.loc[rng.choice(groups[s], len(groups[s]))].mean().to_numpy() for s in QUERIES + bulk])
        Sb = pd.DataFrame(cosine(c[:4], c[4:]), index=QUERIES, columns=bulk)
        boot_ranks.append(Sb.T.groupby(src).rank(pct=True).T)
    br = np.stack([b.to_numpy() for b in boot_ranks])
    lo = pd.DataFrame(np.percentile(br, 2.5, axis=0), index=QUERIES, columns=bulk)
    hi = pd.DataFrame(np.percentile(br, 97.5, axis=0), index=QUERIES, columns=bulk)

    # ---- M0 ----
    S_m0 = m0_similarity(expr, bulk)

    long = pd.concat([score_table(S_m0, "M0_centered_spearman", meta),
                      score_table(S_uce, "UCE33_cosine", meta)], ignore_index=True)
    long["rank_pct_lo95"] = np.nan
    long["rank_pct_hi95"] = np.nan
    m = long.method == "UCE33_cosine"
    long.loc[m, "rank_pct_lo95"] = [lo.loc[q, s] for q, s in zip(long[m].query_group, long[m].sample_id)]
    long.loc[m, "rank_pct_hi95"] = [hi.loc[q, s] for q, s in zip(long[m].query_group, long[m].sample_id)]
    long = long.merge(meta[["sample_id", "biosample_name", "assay", "ontology_curie", "ag_cage_name_hit"]],
                      on="sample_id", how="left")
    assert not long.duplicated(["method", "query_group", "sample_id"]).any()
    assert np.isfinite(long[["S", "rank_pct", "spec"]]).all().all()
    long.to_csv(an / "similarity_long.tsv", sep="\t", index=False)

    top = (long.sort_values("S", ascending=False)
           .groupby(["method", "query_group", "source"]).head(20))
    top["rank"] = top.groupby(["method", "query_group", "source"]).cumcount() + 1
    top.to_csv(an / "top_samples_by_query.tsv", sep="\t", index=False)

    # Combined verdict: both methods top 5% and spec > 1.
    w = long.pivot_table(index=["query_group", "sample_id", "source", "biosample_name"],
                         columns="method", values=["rank_pct", "spec"]).reset_index()
    w.columns = ["_".join(c).strip("_") for c in w.columns]
    w["pass_both"] = ((w.rank_pct_M0_centered_spearman >= 0.95) & (w.rank_pct_UCE33_cosine >= 0.95)
                      & (w.spec_M0_centered_spearman > 1) & (w.spec_UCE33_cosine > 1))
    w.to_csv(an / "consensus_wide.tsv", sep="\t", index=False)

    # ---- Controls ----
    ctrl = []
    for method in long.method.unique():
        mac = long[(long.method == method) & (long.query_group == "saijou_mac")]
        is_mac = mac.biosample_name.str.contains(MAC_PATTERN, case=False, na=False)
        for source in mac.source.unique():
            sub = mac[mac.source == source]
            msub = sub[is_mac.loc[sub.index]]
            ctrl.append(dict(control_id=f"mac_top5pct|{method}|{source}",
                             value=float(msub.rank_pct.max()) if len(msub) else np.nan,
                             n_mac_samples=len(msub),
                             best_mac_sample=msub.sort_values("S").sample_id.iloc[-1] if len(msub) else "",
                             passed=bool(len(msub) and msub.rank_pct.max() >= 0.95 and msub.spec.max() > 1)))
    lg = np.log1p(expr)
    for method in long.method.unique():
        hs = long[(long.method == method) & (long.query_group == "saijou_hsc")].nlargest(5, "spec")
        for s in hs.sample_id:
            pos = lg.loc[[g for g in HSC_MARKERS if g in lg.index], s].mean()
            neg = lg.loc[[g for g in NEG_MARKERS if g in lg.index], s].mean()
            ctrl.append(dict(control_id=f"hsc_markers|{method}|{s}", value=float(pos - neg),
                             passed=bool(pos > neg)))
    pd.DataFrame(ctrl).to_csv(an / "controls.tsv", sep="\t", index=False)

    # ---- UMAP ----
    import umap

    xy = umap.UMAP(random_state=0, n_neighbors=30, min_dist=0.3).fit_transform(E.to_numpy())
    srcmap = meta.set_index("sample_id").source
    cs = srcmap.reindex(sid.values).values
    f, ax = plt.subplots(figsize=(9, 8))
    for name, col in [("ENCODE", "#9aa5b1"), ("FANTOM5", "#c9b38a")]:
        k = cs == name
        ax.scatter(xy[k, 0], xy[k, 1], s=2, c=col, label=name, alpha=0.5, rasterized=True)
    for q, col in zip(QUERIES, ["#d62728", "#1f77b4", "#2ca02c", "#9467bd"]):
        k = sid.values == q
        ax.scatter(xy[k, 0], xy[k, 1], s=14, c=col, label=q, edgecolor="k", linewidth=0.3)
    for q in QUERIES:
        hs = long[(long.method == "UCE33_cosine") & (long.query_group == q)].nlargest(3, "S")
        for s in hs.sample_id:
            k = sid.values == s
            ax.annotate(s.split("|")[-1][:28] if "fantom" in s else hs.set_index("sample_id").biosample_name[s][:28],
                        xy[k].mean(axis=0), fontsize=6)
    ax.legend(markerscale=3, fontsize=8)
    ax.set_title("UCE-33 pseudocell embedding: Saijou pseudobulk vs AlphaGenome training sources")
    f.tight_layout()
    f.savefig(fig / "umap_joint.png", dpi=180)

    json.dump(dict(n_genes=int(expr.shape[0]), n_bulk=len(bulk),
                   n_encode=int((src == "ENCODE").sum()), n_fantom=int((src == "FANTOM5").sum()),
                   n_cells=int(emb.n_obs), uce_model="33l_8ep_1024t_1280 (HF lza1/uce_hf mirror)",
                   saijou_input="pseudobulk CPM bigWig, exon-union sum (no single-cell matrix access)",
                   n_boot=args.n_boot),
              open(an / "validation_summary.json", "w"), indent=2)
    print("done")


if __name__ == "__main__":
    main()
