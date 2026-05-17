"""eQTL AUROC benchmark (Linder et al. 2025, Nature Genetics Fig. 5b standard).

Scores pos/neg VCF pairs with SAD score (default, paper standard) and computes
per-tissue AUROC across 49 GTEx tissues.

Scoring modes (--score argument):
  sum [default]  Paper SAD: per-track count-space sum after full piecewise inverse
                 squash. u_t = Σ_l (y_alt - y_ref) per track.
                 zero-shot: |Σ_t u_t|   RF mode: signed (N, T) feature matrix.

  l2             Log-space approximation: u_t = Σ_l log2(1 + y_count_l) per track.
                 zero-shot: Σ_t |u_t^alt - u_t^ref|
                 RF mode: signed per-track logSUM diff (N, T).

Paper standard: --score sum --rc --rf  →  matches single-model SAD+RF baseline.
Add --rc for forward+reverse complement averaging (~1-2% AUROC gain).

Paper AUROC refs:
  - 0.7943 = 4-rep ensemble + SAD + RF + RC
  - 0.7880 = single model + SAD + RF + RC
  - 0.7720 = ensemble + SAD + RF (no RC?)

Usage:
    source activate.sh

    # Paper standard (SAD + RC + RF):
    python scripts/eqtl/run_eqtl_auroc.py --model borzoi --rc --rf --devices 0

    # Single tissue:
    python scripts/eqtl/run_eqtl_auroc.py --model borzoi --tissue brain_cortex --rc --rf --devices 0

    # AlphaGenome + torch.compile (~2.7x inference speedup):
    python scripts/eqtl/run_eqtl_auroc.py --model alphagenome --rc --rf --devices 0 --compile
"""
import argparse
import sys
import warnings
from pathlib import Path

# Suppress PyTorch's weights_only=False warning from Lightning
warnings.filterwarnings("ignore", message=".*weights_only=False.*", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*weights_only=False.*", category=UserWarning)

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.distributed as dist
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold

import grelu.resources
from grelu.lightning import LightningModel
from grelu.variant import predict_variant_effects
from alphagenome_pytorch.config import DtypePolicy

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
    """Return boolean mask of variants that fit within chromosome bounds."""
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

    Inverse:
      y = y_sq^(4/3)              if y_sq <= 384
          (384 + (y_sq-384)^2)^(4/3)    otherwise
    """
    x = x.clamp(min=0)
    z = (x - 384.0).clamp(min=0)
    return torch.where(x > 384.0, (384.0 + z ** 2) ** (4.0 / 3.0), x ** (4.0 / 3.0))


class BorzoiSumTransform(nn.Module):
    """Full piecewise inverse Borzoi squash → sum across length bins. Output: (T, 1)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = _inverse_squash_borzoi(x)
        return x.sum(dim=-1, keepdim=True)


class BorzoiL2Transform(nn.Module):
    """Inverse Borzoi squash → log2(1+x), then sum over L bins. Output: (T, 1).

    Computes logSUM per track: u_t = Σ_l log2(1 + y_count_l).
    The paper's exact L2 per track is sqrt(Σ_l diff_l^2); using |logSUM| is a
    memory-efficient approximation (avoids 640 GB for N×7611×16384).
    For RF features, we use the signed per-track logSUM diff (N, T), giving the
    RF the same information as a T-dim feature vector.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        log_x = torch.log2(1.0 + _inverse_squash_borzoi(x))
        return log_x.sum(dim=-1, keepdim=True)  # (T, L) → (T, 1)


class AGSumTransform(nn.Module):
    """AlphaGenome sum score: sum across length bins. Output: (T, 1)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.sum(dim=-1, keepdim=True)


class AGL2Transform(nn.Module):
    """AlphaGenome log-space sum per track. Output: (T, 1)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.log2(1.0 + x.clamp(min=0)).sum(dim=-1, keepdim=True)


# ── VCF(Variant Call Format) loading ───────────────────────────────────────────────────────────────

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
    """Compute 95% CI for AUROC using bootstrap resampling."""
    rng = np.random.default_rng(seed)
    aucs = []
    for _ in range(n):
        idx = rng.choice(len(labels), len(labels), replace=True)
        if len(np.unique(labels[idx])) < 2:
            continue
        aucs.append(roc_auc_score(labels[idx], scores[idx]))
    aucs = np.array(aucs)
    return float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))


# ── Scoring ───────────────────────────────────────────────────────────────────

def score_variants(vdf: pd.DataFrame, model, transform: nn.Module,
                   devices: list[int], batch_size: int, num_workers: int,
                   seq_len: int, score_mode: str = "sum",
                   return_per_track: bool = False,
                   rc: bool = False) -> np.ndarray:
    """Predict variant effects and return scores or per-track feature matrix.

    Args:
        return_per_track: If True, return (N, T) feature matrix for RF training.
                          If False (default), return (N,) scalar scores.
        rc: Average forward and reverse complement predictions.

    Returns:
        (N, T) feature matrix when return_per_track=True, else (N,) scores.
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
        rc=rc,
    )

    if score_mode == "l2":
        # diffs: (N, T, 1) — per-track log-space sums (from BorzoiL2Transform / AGL2Transform)
        per_track = diffs.squeeze(-1)  # (N, T) signed per-track logSUM diff
        if return_per_track:
            # RF features: signed per-track logSUM diff.
            # Paper says: "For RF models, we provide the signed difference to allow the RF to learn the direction"
            return per_track  # (N, T)
        return np.abs(per_track).sum(axis=-1)  # (N,) zero-shot heuristic
    else:
        # diffs: (N, T, 1) count-space per-track sums
        per_track = diffs.squeeze(-1)  # (N, T)
        if return_per_track:
            return per_track  # (N, T) signed — RF can learn sign direction
        return np.abs(per_track.sum(axis=-1))  # (N,)


# ── Random Forest evaluation ──────────────────────────────────────────────────

def auroc_rf_cv(X: np.ndarray, y: np.ndarray,
                n_folds: int = 8, n_iterations: int = 100,
                n_estimators: int = 100, seed: int = 44) -> tuple[float, float, float]:
    """Random Forest CV AUROC — strictly matches Borzoi's `randfor_roc` (paper standard).

    Reference: borzoi/src/scripts/borzoi_bench_classify.py::randfor_roc
    Pipeline:  borzoi/src/scripts/borzoi_bench_gtex_folds_sad.py
               → calls borzoi_bench_classify.py with `-i 100 -p 2 -r 44 -s --msl 1`

    Method:
      - KFold (NOT stratified) with shuffle=True
      - n_iterations rounds, each with random_state = seed + i
      - Per-fold RF random_state = (seed + i) + test_index[0]
      - Collect per-fold AUROC across all iterations × folds
      - Report mean ± SE (std / sqrt(N))

    Args:
        X: Feature matrix (N, T) — per-track signed differences.
        y: Binary labels (N,) — 1=eQTL, 0=negative.
        n_folds: Borzoi default = 8.
        n_iterations: Borzoi GTEx pipeline default = 100.
        n_estimators: Trees in the forest (Borzoi = 100).
        seed: Borzoi GTEx pipeline default = 44.

    Returns:
        (mean_auroc, mean - SE, mean + SE)
    """
    aurocs = []
    for i in range(n_iterations):
        rs_iter = seed + i
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=rs_iter)

        for train_idx, test_idx in kf.split(X):
            rs_rf = rs_iter + int(test_idx[0])
            clf = RandomForestClassifier(
                n_estimators=n_estimators,
                max_features='log2',
                max_depth=64,
                min_samples_split=2,
                min_samples_leaf=1,
                n_jobs=-1,
                random_state=rs_rf,
            )
            clf.fit(X[train_idx], y[train_idx])
            preds = clf.predict_proba(X[test_idx])[:, 1]
            # KFold can produce a fold with only one class label when n is small;
            # roc_auc_score requires both classes — skip degenerate folds.
            if len(np.unique(y[test_idx])) < 2:
                continue
            aurocs.append(roc_auc_score(y[test_idx], preds))

    aurocs = np.array(aurocs)
    mean_auroc = float(aurocs.mean())
    se = float(aurocs.std() / np.sqrt(len(aurocs)))
    return mean_auroc, mean_auroc - se, mean_auroc + se


# ── Per-tissue evaluation ────────────────────────────────────────────────────

def evaluate_tissue(tissue: str, pos_vcf: str, neg_vcf: str,
                    model, transform: nn.Module, seq_len: int,
                    devices: list[int], batch_size: int, num_workers: int,
                    score_mode: str = "sum", use_rf: bool = False,
                    rc: bool = False,
                    save_features_dir: str = None) -> dict:
    """Full evaluation pipeline for a single GTEx tissue.

    1. Load eQTL (pos) and negative (neg) VCFs.
    2. Filter edge variants outside model's receptive field.
    3. Run model inference.
    4. Compute AUROC — either zero-shot heuristic or RF with 10-fold CV.
    """
    pos_df = load_vcf(pos_vcf)
    neg_df = load_vcf(neg_vcf)

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

    all_variants = pd.concat([pos_df, neg_df], ignore_index=True)
    labels = np.array([1] * n_pos + [0] * n_neg)

    mode_str = "RF 8-fold CV ×100" if use_rf else "zero-shot"

    # DDP rank guard for setup (VCF loading, filtering, printing)
    # Only rank 0 does this work. Other ranks will participate via score_variants -> PL
    is_dist = torch.distributed.is_available() and torch.distributed.is_initialized()
    if is_dist and torch.distributed.get_rank() != 0:
        result_data = score_variants(
            all_variants, model, transform, devices, batch_size,
            num_workers, seq_len, score_mode=score_mode,
            return_per_track=use_rf, rc=rc,
        )
        return {}

    print(f"  [{tissue}] scoring {n_pos} pos + {n_neg} neg ({mode_str}) ...")

    result_data = score_variants(
        all_variants, model, transform, devices, batch_size,
        num_workers, seq_len, score_mode=score_mode,
        return_per_track=use_rf, rc=rc,
    )

    # DDP rank guard: only rank 0 should run RF evaluation and save files
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        if torch.distributed.get_rank() != 0:
            return {}

    # Trim DDP padding
    if len(result_data) < len(labels):
        labels = labels[: len(result_data)]
    else:
        result_data = result_data[: len(labels)]

    if use_rf:
        # result_data is (N, T) feature matrix
        X = result_data  # (N, T)
        # Borzoi default: 8 folds × 100 iterations
        n_folds = min(8, n_pos, n_neg)
        if n_folds < 2:
            print(f"  [{tissue}] SKIP RF: too few samples for CV (n_pos={n_pos}, n_neg={n_neg})")
            return {}
        if n_folds < 8:
            print(f"  [{tissue}] WARNING: using {n_folds}-fold CV (too few samples for 8)")

        if save_features_dir is not None:
            import pathlib
            fdir = pathlib.Path(save_features_dir)
            fdir.mkdir(parents=True, exist_ok=True)
            np.save(fdir / f"{tissue}_X.npy", X)
            np.save(fdir / f"{tissue}_y.npy", labels)
            print(f"  [{tissue}] features saved → {fdir}/{tissue}_X.npy  shape={X.shape}")

        auroc, ci_lo, ci_hi = auroc_rf_cv(X, labels, n_folds=n_folds, n_iterations=100)
    else:
        scores = result_data  # (N,)
        auroc = roc_auc_score(labels, scores)
        ci_lo, ci_hi = bootstrap_auroc(scores, labels, n=1000)

    band = "±SE" if use_rf else "95% CI"
    print(f"  [{tissue}] AUROC = {auroc:.4f}  {band} [{ci_lo:.4f}, {ci_hi:.4f}]")

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

    Returns:
        (model, transform, batch_size, seq_len)
    """
    if model_name == "borzoi":
        print("Loading Borzoi (human_rep0) ...")
        model = grelu.resources.load_model(
            repo_id="Genentech/borzoi-model", filename="human_rep0.ckpt"
        )
        transform = BorzoiL2Transform() if score_mode == "l2" else BorzoiSumTransform()
        batch_size = 2
        seq_len = BORZOI_INPUT_LEN

    elif model_name == "alphagenome":
        print("Loading AlphaGenome (all 7 standard track heads, ~4992 tracks) ...")
        # output_key="all" concatenates atac+dnase+procap+cage+rna_seq+chip_tf+chip_histone
        # along the channel dim. This makes the AG variant score comparable to Borzoi's,
        # which mixes all assays into its 7611-track output.
        model = LightningModel(
            model_params={
                "model_type": "AlphaGenomeModel",
                "output_key": "all",
                "weights_path": WEIGHTS_PATH,
                "resolution": 128,
                "dtype_policy": DtypePolicy.mixed_precision(),
            },
            train_params={"task": "regression", "loss": "mse"},
        )
        model.data_params["train"] = {"seq_len": AG_INPUT_LEN, "bin_size": AG_BIN_SIZE}
        model.model_params["crop_len"] = 0
        transform = AGL2Transform() if score_mode == "l2" else AGSumTransform()
        batch_size = 1
        seq_len = AG_INPUT_LEN
    else:
        raise ValueError(f"Unknown model: {model_name}. Choose 'borzoi' or 'alphagenome'.")

    return model, transform, batch_size, seq_len


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="eQTL AUROC benchmark (paper standard).")
    parser.add_argument("--model", default="borzoi", choices=["borzoi", "alphagenome"],
                        help="Model to evaluate (default: borzoi).")
    parser.add_argument("--score", default="sum", choices=["sum", "l2"],
                        help="Scoring statistic: 'sum' (default, paper SAD) or 'l2' (log-space approx).")
    parser.add_argument("--rc", action="store_true",
                        help="Average forward and reverse complement predictions (paper standard).")
    parser.add_argument("--rf", action="store_true",
                        help="Use Random Forest with 8-fold CV × 100 iterations "
                             "(Borzoi paper standard, randfor_roc). "
                             "Default: zero-shot heuristic scalar score.")
    parser.add_argument("--manifest", default=str(MANIFEST_DEFAULT),
                        help="Path to manifest.tsv from build_eqtl_data.py.")
    parser.add_argument("--tissue", default=None,
                        help="Evaluate only this tissue (substring match). Default: all 49.")
    parser.add_argument("--devices", default="0,1,2,3",
                        help="Comma-separated GPU indices or 'cpu' (default: 0,1,2,3).")
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=None,
                        help="Override default batch size (borzoi=2, alphagenome=4).")
    parser.add_argument("--compile", action="store_true",
                        help="Enable torch.compile(max-autotune) for ~2.7x inference speedup.")
    parser.add_argument("--n_estimators", type=int, default=100,
                        help="Number of trees in the Random Forest (default: 100).")
    parser.add_argument("--output", default=None,
                        help="Save per-tissue results TSV to this path.")
    parser.add_argument("--save_features", default=None,
                        help="Directory to save per-tissue feature matrices (X.npy, y.npy). "
                             "Only used with --rf.")
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
    if args.batch_size is not None:
        batch_size = args.batch_size

    if args.compile:
        print("Applying torch.compile(max-autotune) ...")
        model.model = torch.compile(model.model, mode="max-autotune")
        print("  (first forward pass will be slow due to autotuning)")

    eval_mode = "RF 8-fold CV ×100 iters" if args.rf else "zero-shot heuristic"
    compile_tag = "+compile" if args.compile else ""
    print(f"Mode: {args.model}{compile_tag} | score={args.score} | eval={eval_mode}")

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
            use_rf=args.rf,
            rc=args.rc,
            save_features_dir=args.save_features,
        )
        if res:
            results.append(res)

    if not results:
        print("No tissues evaluated.")
        return

    df = pd.DataFrame(results)

    mean_auroc = df["auroc"].mean()
    std_auroc = df["auroc"].std()
    print(f"\n{'─' * 60}")
    print(f"Model: {args.model}   Score: {args.score}   Mode: {eval_mode}   Tissues: {len(df)}")
    print(f"Mean AUROC = {mean_auroc:.4f} ± {std_auroc:.4f} (SD)")
    print(f"Paper refs : Borzoi ensemble+L2+RF=0.7943, single+L2+RF=0.7880, ensemble+SUM+RF=0.7720")
    print(f"{'─' * 60}")
    print(df[["tissue", "n_pos", "n_neg", "auroc", "ci_lo", "ci_hi"]].to_string(index=False))

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, sep="\t", index=False)
        print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()
