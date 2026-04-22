"""eQTL AUROC benchmark (Linder et al. 2025, Nature Genetics Fig. 5b standard).

Scores pos/neg VCF pairs with |SUM score| (all tracks, count space) and
computes per-tissue AUROC across 49 GTEx tissues.

SUM score = |Σ_t Σ_l (untransform(y_alt) - untransform(y_ref))|
  - untransform: y^(4/3)  (inverse of Borzoi's training squash y^(3/4))
  - sum over all 7,611 Borzoi tracks and all 16,384 length bins

Usage:
    source activate.sh
    python scripts/run_eqtl_auroc.py \\
        --model borzoi \\
        --manifest ~/.cache/eqtl_finemapping/benchmark/manifest.tsv \\
        --devices 0,1,2,3 \\
        --output eqtl_auroc_results.tsv

    # Quick validation on one tissue:
    python scripts/run_eqtl_auroc.py --model borzoi --tissue brain_cortex
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.distributed as dist
from sklearn.metrics import roc_auc_score

import grelu.resources
from grelu.lightning import LightningModel
from grelu.variant import predict_variant_effects

from scripts.smoke_tests.config import (
    WEIGHTS_PATH, AG_META_PATH, BORZOI_INPUT_LEN, AG_INPUT_LEN, AG_BIN_SIZE,
)

# hg38 chromosome sizes (canonical autosomes + X)
HG38_CHROM_SIZES = {
    "chr1": 248956422, "chr2": 242193529, "chr3": 198295559,
    "chr4": 190214555, "chr5": 181538259, "chr6": 170805979,
    "chr7": 159345973, "chr8": 145138636, "chr9": 138394717,
    "chr10": 133797422, "chr11": 135086622, "chr12": 133275309,
    "chr13": 114364328, "chr14": 107043718, "chr15": 101991189,
    "chr16": 90338345, "chr17": 83257441, "chr18": 80373285,
    "chr19": 58617616, "chr20": 64444167, "chr21": 46709983,
    "chr22": 50818468, "chrX": 156040895,
}


def filter_edge_variants(df: pd.DataFrame, seq_len: int) -> pd.Series:
    """Return boolean mask of variants that fit within chromosome bounds.

    The paper VCFs use Borzoi's smaller chromosome edge margin. AlphaGenome
    has a narrower window (131kb vs 524kb), so it keeps all Borzoi-filtered
    variants. We filter independently per model to avoid silent errors.
    """
    half = seq_len // 2
    chrom_sz = df["chrom"].map(HG38_CHROM_SIZES)
    ok = (
        df["chrom"].isin(HG38_CHROM_SIZES) &
        (df["pos"] - half >= 0) &
        (df["pos"] + half <= chrom_sz)
    )
    return ok

MANIFEST_DEFAULT = Path("~/.cache/eqtl_finemapping/paper_vcfs/manifest.tsv").expanduser()


# ── Score transforms ──────────────────────────────────────────────────────────

class BorzoiSumTransform(nn.Module):
    """Inverse Borzoi squash (y^(4/3)) then sum across length bins.

    1. Inverse Squash: Borzoi was trained on y^(3/4) compressed targets to handle the 
       high dynamic range of RNA-seq. To compare absolute physical counts (Count Space), 
       we apply y^(4/3) to revert the prediction.
    
    2. Spatial Summation: The model outputs a spatial profile across 16,384 bins (each 32bp). 
       In eQTL analysis, we care about the TOTAL expression change of a gene across the 
       entire 524kb window. Summing across all L bins collapses the 'spatial shape' into 
       a single 'total abundance' scalar (per track).
    """
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Revert training squash: y_count = y_pred ^ (4/3)
        x = x.clamp(min=0) ** (4.0 / 3.0)
        # Sum over Length dimension (N, T, L) -> (N, T, 1) to get total region expression
        return x.sum(dim=-1, keepdim=True)


class AGSumTransform(nn.Module):
    """AlphaGenome: sum across length bins only.

    1. No Explicit Inverse Squash: AlphaGenome's internal GenomeTracksHead already 
       applies the inverse squash (x^(1/0.75)) and track-mean scaling. The tensor 
       reaching this transform is already in experimental (count) space.

    2. Spatial Summation: Similar to Borzoi, we sum over the 1,024 bins (each 128bp) 
       to integrate the total signal across the 131kb receptive field. This makes 
       the score robust to minor spatial shifts in transcription start sites.
    """
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Sum over Length dimension (N, T, L) -> (N, T, 1)
        return x.sum(dim=-1, keepdim=True)


# ── VCF loading ───────────────────────────────────────────────────────────────

def load_vcf(path: str | Path) -> pd.DataFrame:
    """Load a minimal VCF into a DataFrame with chrom/pos/ref/alt columns."""
    rows = []
    with open(path) as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            parts = line.rstrip().split("\t")
            rows.append({
                "chrom": parts[0],
                "pos": int(parts[1]),
                "ref": parts[3],
                "alt": parts[4],
            })
    return pd.DataFrame(rows)


# ── AUROC helpers ─────────────────────────────────────────────────────────────

def bootstrap_auroc(scores: np.ndarray, labels: np.ndarray,
                    n: int = 1000, seed: int = 42) -> tuple[float, float]:
    """Compute 95% confidence interval for AUROC using bootstrap resampling.

    Args:
        scores: Model predictions (SUM scores).
        labels: Ground truth binary labels (1 for eQTL, 0 for negative).
        n: Number of bootstrap iterations.
        seed: Random seed for reproducibility.

    Returns:
        (lower_ci, upper_ci): The 2.5th and 97.5th percentiles of the bootstrap distribution.
    """
    rng = np.random.default_rng(seed)
    aucs = []
    for _ in range(n):
        # Sample with replacement to create a new dataset of the same size
        idx = rng.choice(len(labels), len(labels), replace=True)
        # Ensure the bootstrap sample contains both classes
        if labels[idx].nunique() < 2 if hasattr(labels[idx], "nunique") else len(np.unique(labels[idx])) < 2:
            continue
        aucs.append(roc_auc_score(labels[idx], scores[idx]))

    aucs = np.array(aucs)
    return float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))


# ── Scoring ───────────────────────────────────────────────────────────────────

def score_variants(vdf: pd.DataFrame, model, transform: nn.Module,
                   devices: list[int], batch_size: int, num_workers: int,
                   seq_len: int) -> np.ndarray:
    """Predict effects for all variants and compute the absolute SUM score.

    This function utilizes grelu.variant.predict_variant_effects to perform 
    (alt - ref) comparison in count space.

    Args:
        vdf: DataFrame with variants (chrom, pos, ref, alt).
        model: The LightningModel instance.
        transform: The PredictionTransform (Sum over length).
        devices: List of GPU indices.
        batch_size: Inference batch size.
        num_workers: DataLoader workers.
        seq_len: Model receptive field (input sequence length).

    Returns:
        1D array of |SUM scores| for each variant.
    """
    # predict_variant_effects handles the full pipeline:
    # 1. Sequence extraction from hg38 genome
    # 2. One-hot encoding
    # 3. Model forward pass (distributed if devices > 1)
    # 4. Applying the 'transform' to each output
    # 5. Comparing alt and ref via 'subtract'
    odds = predict_variant_effects(
        variants=vdf,
        model=model,
        devices=devices,
        batch_size=batch_size,
        num_workers=num_workers,
        genome="hg38",
        compare_func="subtract",
        return_ad=False,
        prediction_transform=transform,
        seq_len=seq_len,
    )

    # odds shape: (Num_Variants, Num_Tracks, 1)
    # 1. Remove the dummy 1 dimension
    # 2. Sum over all tracks to get the aggregate effect
    # 3. Take absolute value as we don't care about direction for AUROC
    sum_score = odds.squeeze(-1).sum(axis=1)
    return np.abs(sum_score)


# ── Per-tissue evaluation ────────────────────────────────────────────────────

def evaluate_tissue(tissue: str, pos_vcf: str, neg_vcf: str,
                    model, transform: nn.Module, seq_len: int,
                    devices: list[int], batch_size: int, num_workers: int
                    ) -> dict:
    """Full evaluation pipeline for a single GTEx tissue.

    1. Load eQTL (pos) and Negative (neg) VCFs.
    2. Filter variants that are too close to chromosome ends for the model's window.
    3. Run model inference to get scores.
    4. Compute AUROC and 95% bootstrap CI.
    """
    pos_df = load_vcf(pos_vcf)
    neg_df = load_vcf(neg_vcf)

    # Filter variants that extend beyond chromosome ends for this model's window
    pos_ok = filter_edge_variants(pos_df, seq_len)
    neg_ok = filter_edge_variants(neg_df, seq_len)
    n_filtered = (~pos_ok).sum() + (~neg_ok).sum()

    if n_filtered:
        print(f"  [{tissue}] edge-filtered {n_filtered} variants (seq_len={seq_len})")

    pos_df = pos_df[pos_ok].reset_index(drop=True)
    neg_df = neg_df[neg_ok].reset_index(drop=True)

    n_pos, n_neg = len(pos_df), len(neg_df)
    if n_pos == 0 or n_neg == 0:
        print(f"  [{tissue}] SKIP after filter: pos={n_pos}, neg={n_neg}")
        return {}

    # Combine positive and negative variants for a single inference pass
    all_variants = pd.concat([pos_df, neg_df], ignore_index=True)
    labels = np.array([1] * n_pos + [0] * n_neg)

    print(f"  [{tissue}] scoring {n_pos} pos + {n_neg} neg variants ...")
    scores = score_variants(all_variants, model, transform, devices, batch_size,
                            num_workers, seq_len)

    # DDP Handling:
    # If using multiple GPUs, predict_variant_effects will synchronize and gather 
    # results on Rank 0. Workers on other ranks will return partial/padded data.
    if dist.is_available() and dist.is_initialized() and dist.get_rank() != 0:
        return {}

    # Trim DDP-padding: Lightning ensures all ranks have equal chunks, so the 
    # gathered array might be slightly longer than the original input.
    scores = scores[: len(labels)]

    # Metrics
    auroc = roc_auc_score(labels, scores)
    ci_lo, ci_hi = bootstrap_auroc(scores, labels, n=1000)
    print(f"  [{tissue}] AUROC = {auroc:.4f}  95% CI [{ci_lo:.4f}, {ci_hi:.4f}]")

    return {
        "tissue": tissue,
        "n_pos": n_pos,
        "n_neg": n_neg,
        "auroc": auroc,
        "ci_lo": ci_lo,
        "ci_hi": ci_hi,
    }


# ── Model loading ─────────────────────────────────────────────────────────────

def load_model_and_transform(model_name: str) -> tuple:
    """Instantiate the model and corresponding score transformation logic.

    Returns:
        (model, transform, batch_size, seq_len)
    """
    if model_name == "borzoi":
        # Borzoi: 524kb receptive field, 32bp resolution
        print("Loading Borzoi (human_rep0) ...")
        model = grelu.resources.load_model(
            repo_id="Genentech/borzoi-model", filename="human_rep0.ckpt"
        )
        transform = BorzoiSumTransform()
        batch_size = 2 # Borzoi is heavy; uses small batch to avoid OOM
        seq_len = BORZOI_INPUT_LEN

    elif model_name == "alphagenome":
        # AlphaGenome: 131kb receptive field, 128bp resolution
        print("Loading AlphaGenome ...")
        model = LightningModel(
            model_params={
                "model_type": "AlphaGenomeModel",
                "output_key": "rna_seq", # Defaulting to RNA-seq head for eQTL task
                "weights_path": WEIGHTS_PATH,
                "resolution": 128,
            },
            train_params={"task": "regression", "loss": "mse"},
        )
        # Ensure crop_len is 0 for SUM score to use the full receptive field
        model.data_params["train"] = {"seq_len": AG_INPUT_LEN, "bin_size": AG_BIN_SIZE}
        model.model_params["crop_len"] = 0
        transform = AGSumTransform()
        batch_size = 4 # AlphaGenome is lighter; uses larger batch
        seq_len = AG_INPUT_LEN
    else:
        raise ValueError(f"Unknown model: {model_name}. Choose 'borzoi' or 'alphagenome'.")

    return model, transform, batch_size, seq_len



# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="eQTL AUROC benchmark (paper standard).")
    parser.add_argument("--model", default="borzoi", choices=["borzoi", "alphagenome"],
                        help="Model to evaluate (default: borzoi).")
    parser.add_argument("--manifest", default=str(MANIFEST_DEFAULT),
                        help="Path to manifest.tsv from build_eqtl_data.py.")
    parser.add_argument("--tissue", default=None,
                        help="Evaluate only this tissue (substring match). Default: all 49.")
    parser.add_argument("--devices", default="0,1,2,3",
                        help="Comma-separated GPU indices or 'cpu' (default: 0,1,2,3).")
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--output", default=None,
                        help="Save per-tissue results TSV to this path.")
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        sys.exit(f"ERROR: manifest not found at {manifest_path}")

    manifest = pd.read_csv(manifest_path, sep="\t")

    if args.tissue:
        mask = manifest["tissue"].str.contains(args.tissue, case=False)
        manifest = manifest[mask]
        if manifest.empty:
            sys.exit(f"No tissues matching '{args.tissue}' in manifest.")
    print(f"Evaluating {len(manifest)} tissue(s) ...")

    devices = [int(x) for x in args.devices.split(",")] if args.devices != "cpu" else "cpu"
    model, transform, batch_size, seq_len = load_model_and_transform(args.model)

    results = []
    for _, row in manifest.iterrows():
        res = evaluate_tissue(
            tissue=row["tissue"],
            pos_vcf=row["pos_vcf"],
            neg_vcf=row["neg_vcf"],
            model=model,
            transform=transform,
            seq_len=seq_len,
            devices=devices,
            batch_size=batch_size,
            num_workers=args.num_workers,
        )
        if res:
            results.append(res)

    if not results:
        print("No tissues evaluated.")
        return

    df = pd.DataFrame(results)

    # Summary
    mean_auroc = df["auroc"].mean()
    std_auroc = df["auroc"].std()
    print(f"\n{'─'*60}")
    print(f"Model: {args.model}   Tissues: {len(df)}")
    print(f"Mean AUROC = {mean_auroc:.4f} ± {std_auroc:.4f} (SD)")
    print(f"Reference  : Borzoi ensemble = 0.7943 (Linder et al. 2025)")
    print(f"{'─'*60}")
    print(df[["tissue", "n_pos", "n_neg", "auroc", "ci_lo", "ci_hi"]].to_string(index=False))

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, sep="\t", index=False)
        print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()
