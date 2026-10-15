"""Compare WT1 counterfactual effects between two AlphaGenome checkpoints.

Experimental. Both runs must come from run_all_tracks_readout.py on the same
mutation manifest and be analysed with analyze_all_tracks_readout.py --run.
Default pair: fold_0 (pretrained only) vs model_all_folds (distilled release).
Reports, per locus x motif class x readout:
  - Pearson/Spearman of per-variant x track log2FC between checkpoints
  - sign agreement on entries where either checkpoint has |log2FC| > DZ
  - median effect (%) under each checkpoint
and puts the sample-level up/flat/down counts side by side.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

ROOT = Path("experiments/ism/20260918_1349_wt1_exact_counterfactual/runs")
DZ = 0.005


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", type=Path, default=ROOT / "alphagenome_original_all_tracks")
    ap.add_argument("--b", type=Path, default=ROOT / "alphagenome_distilled_all_tracks")
    ap.add_argument("--out", type=Path, default=ROOT / "checkpoint_comparison_fold0_vs_distilled")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    key = ["mutation_id", "readout", "output_type", "track_index"]
    cols = key + ["locus_id", "motif", "readable", "lfc"]
    a = pd.read_parquet(args.a / "variant_track_effects.parquet", columns=cols)
    b = pd.read_parquet(args.b / "variant_track_effects.parquet", columns=key + ["readable", "lfc"])
    m = a.merge(b, on=key, suffixes=("_a", "_b"))
    m = m[m.readable_a & m.readable_b]

    rows = []
    for (loc, motif, ro, assay), g in m.groupby(["locus_id", "motif", "readout", "output_type"]):
        moved = (g.lfc_a.abs() > DZ) | (g.lfc_b.abs() > DZ)
        rows.append(dict(
            locus=loc, motif=motif, readout=ro, assay=assay, n=len(g),
            pearson=pearsonr(g.lfc_a, g.lfc_b)[0], spearman=spearmanr(g.lfc_a, g.lfc_b)[0],
            sign_agree_pct=100 * (np.sign(g.lfc_a[moved]) == np.sign(g.lfc_b[moved])).mean(),
            median_pct_a=100 * (2 ** g.lfc_a.median() - 1),
            median_pct_b=100 * (2 ** g.lfc_b.median() - 1),
            slope_b_on_a=np.polyfit(g.lfc_a, g.lfc_b, 1)[0],
        ))
    effects = pd.DataFrame(rows)
    effects.to_csv(args.out / "effect_agreement.tsv", sep="\t", index=False)

    ca = pd.read_csv(args.a / "sample_level_counts.tsv", sep="\t")
    cb = pd.read_csv(args.b / "sample_level_counts.tsv", sep="\t")
    counts = ca.merge(cb, on=["tpm_min", "locus", "readout", "samples"], suffixes=("_a", "_b"))
    counts.to_csv(args.out / "sample_level_counts_side_by_side.tsv", sep="\t", index=False)

    pd.set_option("display.width", 200)
    print(f"A = {args.a.name}\nB = {args.b.name}\n")
    print(effects.round(3).to_string(index=False))
    print()
    print(counts.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
