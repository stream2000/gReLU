"""Experiment: Genome-wide Pred-vs-Obs Pearson R evaluation.

Slides a window across held-out chromosomes, runs model inference, reads the
experimental BigWig signal, bins both tracks to the same resolution, and
computes per-window Pearson R.  Writes a JSON summary and an optional
per-window CSV.

Usage:
    python -m scripts.run_eval \
        --eval alphagenome \
        --eval_bigwig /path/to/signal.bw \
        --eval_track_idx 0 \
        --eval_chroms chr1 chr8 chr21 \
        --eval_output results/eval_ag.json
"""
import argparse
import json
import os
from dataclasses import dataclass
from typing import Any, List, Optional

import numpy as np
import pandas as pd
import torch
from alphagenome_pytorch.metrics import pearson_r as _pearson_r

import grelu.io
from grelu.data.dataset import BigWigSeqDataset
from scripts.smoke_tests.setup import GreluTutorialApp
from scripts.smoke_tests.utils import _stats, _bin_obs


# ── Experiment logic ──────────────────────────────────────────────────────────

@dataclass
class EvalConfig:
    """Configuration for the genome-wide evaluation experiment."""
    genome: str
    chroms: List[str]
    seq_len: int
    bin_size: int
    stride: int
    max_windows: int
    batch_size: int
    seed: int
    min_obs_mean: float
    devices: Any
    num_workers: int


class GenomeWideEvaluator:
    """Orchestrates genome-wide Pred-vs-Obs evaluation.

    Encapsulates the experimental lifecycle into clear preprocessing,
    inference, and postprocessing stages.
    """

    def __init__(self, config: EvalConfig):
        self.config = config

    def preprocess(self, bigwig_path: str) -> (pd.DataFrame, BigWigSeqDataset):
        """Step 1: Preprocessing — Generate windows and build the dataset."""
        print(f"\n[Preprocessing] Generating windows: chroms={self.config.chroms}, "
              f"seq_len={self.config.seq_len:,}, stride={self.config.stride:,}")

        max_w = self.config.max_windows if self.config.max_windows > 0 else None
        windows = self._make_windows(
            self.config.genome,
            self.config.chroms,
            self.config.seq_len,
            self.config.stride,
            max_w,
            self.config.seed
        )

        if len(windows) == 0:
            raise ValueError("No windows generated — check chromosome names.")

        print(f"[Preprocessing] Building BigWigSeqDataset (seq={self.config.seq_len}bp, bin={self.config.bin_size}bp) ...")
        bw_files = [bigwig_path] if isinstance(bigwig_path, str) else bigwig_path
        dataset = BigWigSeqDataset(
            intervals=windows,
            bw_files=bw_files,
            seq_len=self.config.seq_len,
            genome=self.config.genome,
            bin_size=self.config.bin_size,
            label_aggfunc="mean",
        )
        return windows, dataset

    def inference(self, model: Any, dataset: BigWigSeqDataset) -> np.ndarray:
        """Step 2: Inference — Run model predictions on the dataset."""
        print(f"[Inference] Running model on {len(dataset)} windows (batch={self.config.batch_size}) ...")
        preds_all = model.predict_on_dataset(
            dataset,
            devices=self.config.devices,
            num_workers=self.config.num_workers,
            batch_size=self.config.batch_size,
        )
        return preds_all

    def postprocess(
        self,
        preds_all: np.ndarray,
        dataset: BigWigSeqDataset,
        windows: pd.DataFrame,
        track_idx: int,
        bigwig_path: str,
        model_name: str,
        output_key: str = "rna_seq",
        save_per_window: Optional[str] = None,
    ) -> dict:
        """Step 3: Postprocessing — Compute metrics and format results."""
        print("[Postprocessing] Computing metrics ...")

        # Extract predictions for the specific track
        preds_track = preds_all[:, track_idx, :]  # (N, n_pred_bins)

        # dataset.labels is base-resolution (N, 1, seq_len)
        obs_raw = dataset.labels[:, 0, :]  # (N, seq_len)

        # bin + center crop observed signal to match model output resolution
        obs = _bin_obs(obs_raw, bin_size=self.config.bin_size, n_pred_bins=preds_track.shape[1])

        # Pearson R per window
        pred_t = torch.as_tensor(preds_track, dtype=torch.float32)
        obs_t = torch.as_tensor(obs, dtype=torch.float32)
        per_window_r = _pearson_r(pred_t, obs_t, dim=-1).numpy()  # (N,)

        windows_r = windows.copy()
        windows_r["pearson_r"] = per_window_r
        windows_r["obs_mean"] = obs.mean(axis=-1)  # Average observed signal per window

        all_valid = per_window_r[np.isfinite(per_window_r)]
        # Only keep windows with signal (obs_mean > min_obs_mean)
        signal_mask = np.isfinite(per_window_r) & (windows_r["obs_mean"].values > self.config.min_obs_mean)
        signal_r = per_window_r[signal_mask]

        summary = {
            "model": model_name,
            "output_key": output_key,
            "track_idx": track_idx,
            "bigwig": bigwig_path,
            "min_obs_mean_filter": self.config.min_obs_mean,
            # All windows (including blank areas)
            "all_windows": _stats(per_window_r),
            # Only signal windows
            "signal_windows": _stats(signal_r),
            "n_nan": int(len(per_window_r) - len(all_valid)),
        }

        per_chrom = {}
        for chrom, grp in windows_r.groupby("chrom"):
            per_chrom[str(chrom)] = {
                "all": _stats(grp["pearson_r"].values),
                "signal": _stats(grp.loc[grp["obs_mean"] > self.config.min_obs_mean, "pearson_r"].values),
            }
        summary["per_chrom"] = per_chrom

        # Report to console
        self._print_report(summary, model_name, track_idx)

        if save_per_window:
            windows_r.to_csv(save_per_window, index=False)
            print(f"Per-window table \u2192 {save_per_window}")

        return summary

    def _make_windows(
        self,
        genome: str,
        chroms: list,
        seq_len: int,
        stride: int,
        max_windows: int | None,
        seed: int = 42,
    ) -> pd.DataFrame:
        """Sliding-window intervals on the requested chromosomes."""
        sizes = grelu.io.genome.read_sizes(genome).set_index("chrom")["size"].to_dict()
        rows = []
        for chrom in chroms:
            chrom_len = sizes.get(chrom)
            if chrom_len is None:
                print(f"  [warn] {chrom} not found in chrom sizes \u2014 skipping")
                continue
            start = 0
            while start + seq_len <= chrom_len:
                rows.append({"chrom": chrom, "start": start,
                             "end": start + seq_len, "strand": "+"})
                start += stride
        df = pd.DataFrame(rows)
        print(f"  Generated {len(df)} windows across {chroms}")
        if max_windows is not None and len(df) > max_windows:
            rng = np.random.default_rng(seed)
            idx = sorted(rng.choice(len(df), max_windows, replace=False))
            df = df.iloc[idx].reset_index(drop=True)
            print(f"  Sub-sampled to {len(df)} windows (seed={seed})")
        return df

    def _print_report(self, summary: dict, model_name: str, track_idx: int):
        """Standard console report format."""
        aw = summary["all_windows"]
        sw = summary["signal_windows"]
        print(f"\n{'=' * 60}")
        print(f"  Model      : {model_name}  track_idx={track_idx}")
        print(f"  [All Windows]  mean_r={aw['mean_r']:.4f}  median_r={aw['median_r']:.4f}  n={aw['n_windows']}")
        print(f"  [Signal Windows obs_mean>{summary['min_obs_mean_filter']}]  "
              f"mean_r={sw['mean_r']:.4f}  median_r={sw['median_r']:.4f}  n={sw['n_windows']}")
        for chrom, st in sorted(summary["per_chrom"].items()):
            print(f"    {chrom:10s}  all mean_r={st['all']['mean_r']:.4f}  "
                  f"signal mean_r={st['signal']['mean_r']:.4f}  "
                  f"signal_n={st['signal']['n_windows']}")
        print(f"{'=' * 60}")


def run_genome_wide_eval(
        app: GreluTutorialApp,
        model_name: str,
        bigwig_path: str,
        track_idx: int,
        output_key: str = "rna_seq",
        chroms: list | None = None,
        stride: int | None = None,
        max_windows: int = 300,
        batch_size: int = 4,
        seed: int = 42,
        min_obs_mean: float = 0.05,
        save_per_window: str | None = None,
) -> dict:
    """Pred-vs-Obs profile Pearson R on held-out chromosomes.

    Uses dependency injection: the app instance provides model initialization
    and configuration, which are then passed to the `GenomeWideEvaluator`.
    """
    app._cleanup()

    if chroms is None:
        chroms = ["chr1", "chr8", "chr21"]

    # Initialize model via the app
    if model_name == "borzoi":
        app.setup_borzoi()
        model_obj = app.borzoi
    else:
        app.ag_rna = app._setup_alpha_genome(output_key)
        model_obj = app.ag_rna

    # Prepare evaluation configuration
    config = EvalConfig(
        genome=app.genome,
        chroms=chroms,
        seq_len=app._model_cfg.seq_len,
        bin_size=app._model_cfg.bin_size,
        stride=stride or app._model_cfg.seq_len,
        max_windows=max_windows,
        batch_size=batch_size,
        seed=seed,
        min_obs_mean=min_obs_mean,
        devices=app.devices,
        num_workers=app.num_workers
    )

    evaluator = GenomeWideEvaluator(config)

    # 1. Preprocessing
    windows, dataset = evaluator.preprocess(bigwig_path)

    # 2. Inference
    preds_all = evaluator.inference(model_obj, dataset)

    # Cleanup app state before heavy postprocessing if needed
    app._cleanup()

    # 3. Postprocessing
    summary = evaluator.postprocess(
        preds_all=preds_all,
        dataset=dataset,
        windows=windows,
        track_idx=track_idx,
        bigwig_path=bigwig_path,
        model_name=model_name,
        output_key=output_key,
        save_per_window=save_per_window
    )

    return summary


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Genome-wide Pred-vs-Obs Pearson R evaluation."
    )
    parser.add_argument("--eval", choices=["borzoi", "alphagenome"], required=True,
                        help="Model to evaluate.")
    parser.add_argument("--eval_bigwig", metavar="PATH", required=True,
                        help="Path to observed-signal BigWig.")
    parser.add_argument("--eval_track_idx", type=int, default=0, metavar="INT",
                        help="Zero-based output track index to compare against BigWig (default: 0).")
    parser.add_argument("--eval_output_key", default="rna_seq",
                        help="AlphaGenome head key (rna_seq, cage, \u2026). Default: rna_seq.")
    parser.add_argument("--eval_chroms", nargs="+", default=["chr1", "chr8", "chr21"],
                        metavar="CHROM",
                        help="Chromosomes to evaluate (default: chr1 chr8 chr21).")
    parser.add_argument("--eval_max_windows", type=int, default=300,
                        help="Max windows to evaluate (0=all). Default: 300.")
    parser.add_argument("--eval_batch_size", type=int, default=4,
                        help="Inference batch size (default: 4).")
    parser.add_argument("--eval_output", default="eval_results.json",
                        help="JSON output path (default: eval_results.json).")
    parser.add_argument("--eval_min_obs_mean", type=float, default=0.05,
                        help="Only report stats for signal windows exceeding this obs_mean. Default: 0.05.")
    parser.add_argument("--eval_save_per_window", metavar="CSV",
                        help="Optional CSV path for per-window Pearson R table.")
    parser.add_argument("--devices", default="0,1,2,3",
                        help="Comma-separated GPU indices, or 'cpu' (default: 0,1,2,3).")
    parser.add_argument("--num_workers", type=int, default=1,
                        help="DataLoader worker count (default: 1).")
    args = parser.parse_args()

    app = GreluTutorialApp(devices=args.devices, num_workers=args.num_workers)

    summary = run_genome_wide_eval(
        app=app,
        model_name=args.eval,
        bigwig_path=args.eval_bigwig,
        track_idx=args.eval_track_idx,
        output_key=args.eval_output_key,
        chroms=args.eval_chroms,
        max_windows=args.eval_max_windows,
        batch_size=args.eval_batch_size,
        min_obs_mean=args.eval_min_obs_mean,
        save_per_window=args.eval_save_per_window,
    )

    os.makedirs(os.path.dirname(args.eval_output) or ".", exist_ok=True)
    with open(args.eval_output, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary JSON \u2192 {args.eval_output}")
