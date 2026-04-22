"""eQTL AUROC benchmark (Linder et al. 2025, Nature Genetics Fig. 5b standard).

Scores pos/neg VCF pairs with L2 score (default) or |SUM score| and computes
per-tissue AUROC across 49 GTEx tissues.

Scoring modes (--score argument):
  l2  [default]  Per-track L1 of log-space sums — the memory-efficient approximation
                 of the paper's L2 score (Methods section).
                 u_t = Σ_l log2(1 + y_count_l)  per track (in transform)
                 score = Σ_t |u_t^alt - u_t^ref|   (always ≥ 0, no cross-track cancellation)
                 The paper's exact L2 is sqrt(Σ_l diff_l^2); using |Σ_l diff_l| (logSUM)
                 avoids materializing (N × 7611 × 16384) ≈ 640 GB RAM for all variants.
                 In practice, eQTL bins shift in one direction per gene, so |logSUM| ≈ L2.

  sum            |Σ_t Σ_l (inv_squash(y_alt) - inv_squash(y_ref))|
                 Full piecewise inverse of Borzoi squash before summing.
                 u_t = Σ_l (y_alt^count - y_ref^count) per track
                 score = |Σ_t u_t|  (can cancel when tracks move in opposite directions)

Relationship to paper AUROC numbers:
  - 0.7943 (reported in paper Fig. 5b) = 4-rep ensemble + L2 + supervised RF
  - 0.7880 = single model + L2 + supervised RF
  - 0.7720 = ensemble + SUM + supervised RF
  - Our zero-shot L2 (Σ_t u_t, no RF) is comparable but lower than paper numbers
    because the RF learns tissue-specific weights the |Σ_t u_t| heuristic ignores.

Usage:
    source activate.sh
    python scripts/run_eqtl_auroc.py \\
        --model borzoi \\
        --manifest ~/.cache/eqtl_finemapping/paper_vcfs/manifest.tsv \\
        --score l2 \\
        --devices 0,1,2,3 \\
        --output eqtl_auroc_results.tsv

    # Quick validation on one tissue:
    python scripts/run_eqtl_auroc.py --model borzoi --tissue brain_cortex --score l2
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

def _inverse_squash_borzoi(x: torch.Tensor) -> torch.Tensor:
    """Full piecewise inverse of Borzoi's training squash transform.

    Forward squash (training):
      y_sq = y^(3/4)              if y^(3/4) <= 384
             384 + sqrt(y^(3/4) - 384)   otherwise

    Inverse (used at inference to recover count space):
      y = y_sq^(4/3)              if y_sq <= 384
          (384 + (y_sq-384)^2)^(4/3)    otherwise

    The piecewise branch matters for highly-expressed bins (y_sq > 384).
    Without it, the model systematically underestimates changes in high-expression regions.
    """
    x = x.clamp(min=0)
    z = (x - 384.0).clamp(min=0)
    # torch.where avoids modifying x in-place; both branches computed for all elements
    return torch.where(x > 384.0, (384.0 + z ** 2) ** (4.0 / 3.0), x ** (4.0 / 3.0))


class BorzoiSumTransform(nn.Module):
    """Full piecewise inverse Borzoi squash → sum across length bins.

    Output shape: (T, 1) per sample — the summed count-space expression per track.
    Used for SUM score: score = |Σ_t (sum_alt - sum_ref)|.

    Note: signed sum can cancel when tracks move in opposite directions
    (e.g., brain up, liver down). Use BorzoiL2Transform for cancellation-free scoring.
    """
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = _inverse_squash_borzoi(x)
        return x.sum(dim=-1, keepdim=True)  # (T, L) → (T, 1)


class BorzoiL2Transform(nn.Module):
    """Inverse Borzoi squash → log2(1+x), then sum over L bins. Output: (T, 1).

    Computes the log-space sum per track: u_t = Σ_l log2(1 + y_count_l).
    After predict_variant_effects subtracts alt-ref, the caller computes:
      score = Σ_t |u_t^alt - u_t^ref|   (L1 of per-track log sums)

    This approximates the paper's L2 score in spirit:
    - Log space: focuses on fold change, not absolute magnitude.
    - Per-track absolute value: eliminates cross-track cancellation
      (brain up, liver down no longer cancel each other).
    - Memory efficient: (T, 1) output, not (T, L) — storing (N, T, 16384) would
      require ~640 GB RAM for Borzoi's 7611 tracks and N~1000 variants.

    The paper's exact L2 is u_t = sqrt(Σ_l diff_l^2), which additionally avoids
    within-track spatial cancellation. For eQTL variants affecting a gene, bins
    typically shift in the same direction, so |logSUM| ≈ L2 in practice.
    """
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        log_x = torch.log2(1.0 + _inverse_squash_borzoi(x))
        return log_x.sum(dim=-1, keepdim=True)  # (T, L) → (T, 1)


class AGSumTransform(nn.Module):
    """AlphaGenome sum score: sum across length bins. Output: (T, 1).

    AlphaGenome's GenomeTracksHead already applies inverse squash and track-mean
    scaling, so the output is already in count space.
    """
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.sum(dim=-1, keepdim=True)  # (T, 1)


class AGL2Transform(nn.Module):
    """AlphaGenome log-space sum per track. Output: (T, 1).

    AG output is already in count space, so we apply log2(1+x) then sum over L.
    Score = Σ_t |u_t^alt - u_t^ref| (L1 of per-track log sums, no cancellation).
    """
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.log2(1.0 + x.clamp(min=0)).sum(dim=-1, keepdim=True)  # (T, 1)


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
                   seq_len: int, score_mode: str = "l2") -> np.ndarray:
    """Predict effects for all variants and compute the eQTL classification score.

    Args:
        vdf: DataFrame with variants (chrom, pos, ref, alt).
        model: The LightningModel instance.
        transform: BorzoiSumTransform / BorzoiL2Transform / AGSumTransform / AGL2Transform.
        devices: List of GPU indices.
        batch_size: Inference batch size.
        num_workers: DataLoader workers.
        seq_len: Model receptive field (input sequence length).
        score_mode: "l2" (default) or "sum". Both modes receive (N, T, 1) diffs.
            l2:  transform outputs log-space sums per track.
                 score = Σ_t |u_t^alt - u_t^ref|  — always ≥ 0, no cancellation.
            sum: transform outputs count-space sums per track.
                 score = |Σ_t u_t|  — signed sum, can cancel across tissues.

    Returns:
        1D array of scores (always non-negative) for each variant.
    """
    diffs = predict_variant_effects(
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

    if score_mode == "l2":
        # diffs: (N, T, 1) — per-track log-space sums (from BorzoiL2Transform / AGL2Transform)
        # score = Σ_t |u_t^alt - u_t^ref|   (L1 of per-track log sums, no cross-track cancellation)
        per_track_logdiff = diffs.squeeze(-1)  # (N, T)
        return np.abs(per_track_logdiff).sum(axis=-1)  # (N,)  — always ≥ 0
    else:
        # diffs: (N, T, 1) count-space per-track sums
        # score = |Σ_t u_t|  (signed sum, can cancel across tracks)
        sum_score = diffs.squeeze(-1).sum(axis=-1)  # (N,)
        return np.abs(sum_score)


# ── Per-tissue evaluation ────────────────────────────────────────────────────

def evaluate_tissue(tissue: str, pos_vcf: str, neg_vcf: str,
                    model, transform: nn.Module, seq_len: int,
                    devices: list[int], batch_size: int, num_workers: int,
                    score_mode: str = "l2") -> dict:
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
                            num_workers, seq_len, score_mode=score_mode)

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

def load_model_and_transform(model_name: str, score_mode: str) -> tuple:
    """Instantiate the model and select the transform for the chosen score mode.

    Args:
        model_name: "borzoi" or "alphagenome".
        score_mode: "l2" or "sum".

    Returns:
        (model, transform, batch_size, seq_len)
    """
    if model_name == "borzoi":
        print("Loading Borzoi (human_rep0) ...")
        model = grelu.resources.load_model(
            repo_id="Genentech/borzoi-model", filename="human_rep0.ckpt"
        )
        # l2: log2(1+inv_squash(x)), output (T, L) — L2 norm computed in score_variants
        # sum: inv_squash(x) summed over L, output (T, 1) — signed sum per track
        transform = BorzoiL2Transform() if score_mode == "l2" else BorzoiSumTransform()
        batch_size = 2
        seq_len = BORZOI_INPUT_LEN

    elif model_name == "alphagenome":
        print("Loading AlphaGenome ...")
        model = LightningModel(
            model_params={
                "model_type": "AlphaGenomeModel",
                "output_key": "rna_seq",
                "weights_path": WEIGHTS_PATH,
                "resolution": 128,
            },
            train_params={"task": "regression", "loss": "mse"},
        )
        model.data_params["train"] = {"seq_len": AG_INPUT_LEN, "bin_size": AG_BIN_SIZE}
        model.model_params["crop_len"] = 0
        transform = AGL2Transform() if score_mode == "l2" else AGSumTransform()
        batch_size = 4
        seq_len = AG_INPUT_LEN
    else:
        raise ValueError(f"Unknown model: {model_name}. Choose 'borzoi' or 'alphagenome'.")

    return model, transform, batch_size, seq_len



# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="eQTL AUROC benchmark (paper standard).")
    parser.add_argument("--model", default="borzoi", choices=["borzoi", "alphagenome"],
                        help="Model to evaluate (default: borzoi).")
    parser.add_argument("--score", default="l2", choices=["l2", "sum"],
                        help="Scoring statistic: 'l2' (default, paper best) or 'sum'.")
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
    model, transform, batch_size, seq_len = load_model_and_transform(args.model, args.score)

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
            score_mode=args.score,
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
    print(f"Model: {args.model}   Score: {args.score}   Tissues: {len(df)}")
    print(f"Mean AUROC = {mean_auroc:.4f} ± {std_auroc:.4f} (SD)")
    print(f"Paper refs : Borzoi ensemble+L2+RF=0.7943, single+L2+RF=0.7880, ensemble+SUM+RF=0.7720")
    print(f"             (zero-shot heuristic is lower than RF-supervised baselines)")
    print(f"{'─'*60}")
    print(df[["tissue", "n_pos", "n_neg", "auroc", "ci_lo", "ci_hi"]].to_string(index=False))

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, sep="\t", index=False)
        print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()
