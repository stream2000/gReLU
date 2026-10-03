"""Re-run the WT1 exact counterfactuals on ALL original AlphaGenome mouse RNA/CAGE tracks.

Experimental. The 2026-09-18 run kept only 28 hand-picked tracks and read RNA as
the sum of a 2.5 kb gene-body window (introns included). This script keeps every
rna_seq and cage channel and stores, per variant x channel, the raw sums of
REF and ALT for these readouts (mm10, 0-based half-open, 128 bp bins):

  exon        Mdk-201 exon union, each bin weighted by its exon overlap / 128
  exon_noedit same, with any bin overlapping the edited 10 bp set to weight 0
  gene_window bins 4076-4096 (identical to the old gene_body readout)
  tss512      CAGE-style window TSS +/- 256 bp (Mdk TSS 91932297, minus strand)
  cage_peak   bins 4090-4092 around the predicted dominant Mdk CAGE peak
              (chr2:91931657-91931785), ~600 bp downstream of the annotated TSS

Strand selection and effect computation happen in the analysis step.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyfaidx
import torch

from grelu.interpret.ism.model_adapters.alphagenome_original import DEFAULT_WEIGHTS
from grelu.interpret.ism.model_adapters.utils import sequences_to_tensor

ROOT = Path("experiments/ism/20260918_1349_wt1_exact_counterfactual")
FASTA = "/work/Database/Database_fromDocker/Referencedata_mm10/genome.fa"
CHROM, SEQ_START, SEQ_END = "chr2", 91_408_009, 92_456_585
RES = 128
MDK_EXONS = [(91929827, 91930115), (91930817, 91930979), (91931095, 91931254),
             (91931371, 91931448), (91931929, 91932297)]
MDK_TSS = 91932297
HEADS = ("cage", "rna_seq")
READOUTS = ("exon", "exon_noedit", "gene_window", "tss512", "cage_peak")


def bin_weights(intervals: list[tuple[int, int]], n_bins: int) -> np.ndarray:
    w = np.zeros(n_bins, dtype=np.float32)
    for s, e in intervals:
        for b in range((s - SEQ_START) // RES, (e - 1 - SEQ_START) // RES + 1):
            bs = SEQ_START + b * RES
            w[b] += max(0, min(e, bs + RES) - max(s, bs)) / RES
    return w


def readout_matrix(edit: tuple[int, int] | None, n_bins: int) -> np.ndarray:
    exon = bin_weights(MDK_EXONS, n_bins)
    noedit = exon.copy()
    if edit is not None:
        noedit[(edit[0] - SEQ_START) // RES:(edit[1] - 1 - SEQ_START) // RES + 1] = 0
    gene = np.zeros(n_bins, dtype=np.float32)
    gene[4076:4096] = 1
    tss = bin_weights([(MDK_TSS - 256, MDK_TSS + 256)], n_bins)
    peak = np.zeros(n_bins, dtype=np.float32)
    peak[4090:4093] = 1  # predicted dominant Mdk CAGE peak (bin 4091) +/- 1 bin
    return np.stack([exon, noedit, gene, tss, peak], axis=1)  # bins x readouts


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n-shards", type=int, default=1)
    ap.add_argument("--out", type=Path, default=ROOT / "runs/alphagenome_original_all_tracks")
    ap.add_argument("--mutation-ids", type=Path, default=None, help="optional subset, one id per line")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    from alphagenome_pytorch.config import DtypePolicy
    from grelu.lightning import LightningModel

    model = LightningModel(
        model_params={"model_type": "AlphaGenomeModel", "output_key": HEADS,
                      "weights_path": str(DEFAULT_WEIGHTS),
                      "dtype_policy": DtypePolicy.mixed_precision(), "resolution": RES,
                      "organism_index": 1, "num_organisms": 2},
        train_params={"task": "regression", "loss": "mse"},
    ).eval().to(args.device)
    heads = model.model.embedding.model.heads
    channels = []
    for h in HEADS:
        channels += [(h, i) for i in range(int(heads[h].num_tracks))]

    ref_seq = str(pyfaidx.Fasta(FASTA)[CHROM][SEQ_START:SEQ_END]).upper()
    man = pd.read_csv(ROOT / "prepared/mutation_manifest.tsv", sep="\t")
    if args.mutation_ids is not None:
        man = man[man.mutation_id.isin(args.mutation_ids.read_text().split())]
    man = man.iloc[args.shard::args.n_shards].reset_index(drop=True)

    def predict(seq: str) -> np.ndarray:
        with torch.inference_mode():
            y = model.forward(sequences_to_tensor([seq], expected_length=len(seq)).to(args.device))
        return y[0].float().cpu().numpy()  # channels x bins

    ref = predict(ref_seq)
    n_bins = ref.shape[-1]
    assert ref.shape[0] == len(channels)
    ref_ro = {}
    alt_sums = np.zeros((len(man), len(channels), len(READOUTS)), dtype=np.float32)
    ref_sums = np.zeros_like(alt_sums)
    for k, row in enumerate(man.itertuples(index=False)):
        a, b = int(row.edit_start) - SEQ_START, int(row.edit_end) - SEQ_START
        assert ref_seq[a:b] == row.ref_sequence, row.mutation_id
        alt_seq = ref_seq[:a] + row.alt_sequence + ref_seq[b:]
        key = (int(row.edit_start), int(row.edit_end))
        if key not in ref_ro:
            W = readout_matrix(key, n_bins)
            ref_ro[key] = (W, ref @ W)
        W, rs = ref_ro[key]
        alt_sums[k] = predict(alt_seq) @ W
        ref_sums[k] = rs
        if k % 25 == 0:
            print(f"shard {args.shard}: {k}/{len(man)}", flush=True)

    np.savez(args.out / f"sums_shard{args.shard}.npz", alt=alt_sums, ref=ref_sums,
             mutation_id=man.mutation_id.to_numpy(), readouts=np.array(READOUTS))
    pd.DataFrame(channels, columns=["output_type", "track_index"]).to_csv(
        args.out / "channels.tsv", sep="\t", index=False)
    json.dump({"weights": str(DEFAULT_WEIGHTS), "heads": HEADS, "exons": MDK_EXONS,
               "tss": MDK_TSS, "resolution_bp": RES, "seq": [CHROM, SEQ_START, SEQ_END]},
              open(args.out / "run_metadata.json", "w"), indent=2)
    print("done", flush=True)


if __name__ == "__main__":
    main()
