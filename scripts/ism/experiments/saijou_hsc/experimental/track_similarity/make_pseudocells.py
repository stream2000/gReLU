"""Draw multinomial pseudocells from every sample and write UCE input shards.

Both Saijou (pseudobulk bigWig) and bulk samples go through the identical
sampling step, so pseudocell artefacts are shared by both sides.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--cells-per-sample", type=int, default=30)
    ap.add_argument("--umi", type=int, default=5000)
    ap.add_argument("--shards", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    expr = pd.read_parquet(args.out / "prepared/expr_cpm.parquet")
    rng = np.random.default_rng(args.seed)
    p = (expr / expr.sum(axis=0)).to_numpy().T  # samples x genes
    blocks, obs = [], []
    for i, sid in enumerate(expr.columns):
        blocks.append(sp.csr_matrix(rng.multinomial(args.umi, p[i], size=args.cells_per_sample)))
        obs += [(f"{sid}#{k}", sid) for k in range(args.cells_per_sample)]
    X = sp.vstack(blocks).tocsr().astype(np.float32)
    obs = pd.DataFrame(obs, columns=["cell_id", "sample_id"]).set_index("cell_id")
    adata = ad.AnnData(X=X, obs=obs, var=pd.DataFrame(index=expr.index))

    d = args.out / "uce_input"
    d.mkdir(parents=True, exist_ok=True)
    order = rng.permutation(adata.n_obs)
    for s, idx in enumerate(np.array_split(order, args.shards)):
        adata[np.sort(idx)].copy().write_h5ad(d / f"shard{s}.h5ad")
    print(f"cells={adata.n_obs} genes={adata.n_vars} shards={args.shards}")


if __name__ == "__main__":
    main()
