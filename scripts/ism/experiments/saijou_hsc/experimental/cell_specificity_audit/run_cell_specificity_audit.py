#!/usr/bin/env python
"""Audit observed versus predicted four-track specificity on a saved split.

This experimental CLI records only ``window_idx`` and ``bin_idx``. It never
reads, writes, transforms, annotates, or plots genomic coordinates.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import subprocess
import sys
import time
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[5]
FT_DIR = REPO_ROOT / "src/ft-scripts"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(FT_DIR))

import grelu.lightning  # noqa: E402
from borzoi_mmap_dataset import BorzoiMmapSeqDataset, read_intervals  # noqa: E402
from metrics import (  # noqa: E402
    build_masks,
    build_stratified_masks,
    aggregate_bin_axis,
    centered_l2_over_total,
    cluster_bootstrap_mean_ci,
    derive_train_thresholds_from_cache,
    normalized_entropy_specificity,
    safe_correlation,
    specificity_deciles,
    top_identity_metrics,
    top_second_log2_margin,
    unique_top,
)
from train_borzoi import (  # noqa: E402
    BORZOI_LORA_TARGET_MODULES,
    apply_lora_to_borzoi_embedding,
)


DEFAULT_CHECKPOINTS = {
    "borzoi_lora_e39": REPO_ROOT
    / "runs/borzoi_split_chr10_chr11_poisson_multinomial_lora/checkpoints/epochepoch=39.ckpt",
    "borzoi_headonly_e14": REPO_ROOT
    / "runs/borzoi_split_chr10_chr11_poisson_multinomial_headonly/checkpoints/epochepoch=14.ckpt",
}
TASKS = ["hsc", "mac", "lsec", "chol"]
PALETTE = {"observed": "#3B6FB6", "predicted": "#D18F00", "gap": "#7A5195"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=sorted(DEFAULT_CHECKPOINTS), default="borzoi_lora_e39")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--max-windows", type=int, default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=20260824)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument(
        "--analysis-bin-size",
        type=int,
        default=None,
        help="coarsen model bins for analysis, e.g. 128 from a 32-bp checkpoint",
    )
    parser.add_argument(
        "--prediction-cache-dir",
        type=Path,
        help="reuse observed.npy/predicted.npy from a compatible source-bin audit",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=REPO_ROOT / "experiments/validation/cell_specificity_audit",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
    ).strip()


def load_checkpoint_metadata(path: Path) -> tuple[dict, dict]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    metadata = {
        "epoch": int(checkpoint["epoch"]),
        "global_step": int(checkpoint["global_step"]),
        "model_params": checkpoint["hyper_parameters"]["model_params"],
        "train_params": checkpoint["hyper_parameters"]["train_params"],
        "data": {
            split: {
                key: checkpoint["data_params"][split][key]
                for key in (
                    "genome_file",
                    "seq_len",
                    "label_len",
                    "bin_size",
                    "binned_label_len",
                    "n_seqs",
                    "n_tasks",
                    "rc",
                )
            }
            for split in ("train", "val")
        },
        "task_names": list(checkpoint["data_params"]["tasks"]["name"]),
    }
    return checkpoint, metadata


def matching_cache(metadata: dict, bin_size: int | None = None) -> tuple[Path, dict]:
    train = metadata["data"]["train"]
    requested_bin_size = int(bin_size or train["bin_size"])
    expected_train_bins = train["label_len"] // requested_bin_size
    expected_val_bins = metadata["data"]["val"]["label_len"] // requested_bin_size
    candidates = sorted((REPO_ROOT / "dataset_store/mmap_caches").glob("*/manifest.json"))
    matches = []
    for path in candidates:
        manifest = json.loads(path.read_text())
        if (
            manifest.get("split_name") == "split_chr10_chr11"
            and manifest.get("target_mode") == metadata["train_params"]["loss"]
            and manifest.get("seq_len") == train["seq_len"]
            and manifest.get("label_len") == train["label_len"]
            and manifest.get("bin_size") == requested_bin_size
            and manifest.get("task_names") == metadata["task_names"]
            and manifest.get("train_shape")
            == [train["n_seqs"], train["n_tasks"], expected_train_bins]
            and manifest.get("val_shape")
            == [metadata["data"]["val"]["n_seqs"], train["n_tasks"], expected_val_bins]
        ):
            matches.append((path.parent, manifest))
    if len(matches) != 1:
        raise RuntimeError(f"expected exactly one matching label cache, found {len(matches)}")
    return matches[0]


def load_model(checkpoint: dict, model_name: str, device: torch.device):
    hparams = checkpoint["hyper_parameters"]
    model = grelu.lightning.LightningModel(
        model_params=hparams["model_params"], train_params=hparams["train_params"]
    )
    loader_details = {"finetune_mode": "headonly"}
    if "lora" in model_name:
        # The old Borzoi checkpoint did not serialize adapter settings. Reuse
        # the canonical training helper and verify the saved state structurally.
        lora_keys = [k for k in checkpoint["state_dict"] if "lora_A" in k]
        if not lora_keys:
            raise RuntimeError("LoRA model selected but checkpoint has no lora_A tensors")
        rank = int(checkpoint["state_dict"][lora_keys[0]].shape[0])
        targets = apply_lora_to_borzoi_embedding(
            model,
            target_modules=BORZOI_LORA_TARGET_MODULES,
            rank=rank,
            alpha=16,
            dropout=0.0,
        )
        loader_details = {
            "finetune_mode": "lora",
            "lora_rank_from_state": rank,
            "lora_alpha_from_canonical_training_source": 16,
            "lora_targets": targets,
        }
    missing, unexpected = model.load_state_dict(checkpoint["state_dict"], strict=False)
    if missing or unexpected:
        raise RuntimeError(f"checkpoint mismatch: missing={missing}, unexpected={unexpected}")
    model.eval().to(device)
    return model, loader_details


def make_val_dataset(cache: Path, metadata: dict) -> BorzoiMmapSeqDataset:
    val = metadata["data"]["val"]
    intervals = read_intervals(
        REPO_ROOT / "dataset_store/splits/split_chr10_chr11/val_intervals.bed"
    )
    if len(intervals) != val["n_seqs"]:
        raise RuntimeError("val interval count differs from checkpoint")
    return BorzoiMmapSeqDataset(
        intervals=intervals,
        labels_path=cache / "val_labels.npy",
        genome=val["genome_file"],
        tasks=metadata["task_names"],
        seq_len=val["seq_len"],
        label_len=val["label_len"],
        bin_size=val["bin_size"],
        rc=False,
    )


def infer(model, dataset, n_windows: int, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    observed, predicted = [], []
    with torch.inference_mode():
        for window_idx in range(n_windows):
            sequence, label = dataset[window_idx]
            output = model(sequence.unsqueeze(0).to(device), logits=False)[0]
            observed.append(label.numpy())
            predicted.append(output.float().cpu().numpy())
            print(f"inference {window_idx + 1}/{n_windows}", flush=True)
    obs = np.stack(observed)
    pred = np.stack(predicted)
    if obs.shape != pred.shape:
        raise RuntimeError(f"observed/predicted shape mismatch: {obs.shape} vs {pred.shape}")
    return obs, pred


def write_analysis(
    out_dir: Path,
    model_name: str,
    observed: np.ndarray,
    predicted: np.ndarray,
    thresholds: dict,
    seed: int,
    n_boot: int,
) -> tuple[pd.DataFrame, dict]:
    n_windows, n_tasks, n_bins = observed.shape
    obs_rows = observed.transpose(0, 2, 1).reshape(-1, n_tasks)
    pred_rows = predicted.transpose(0, 2, 1).reshape(-1, n_tasks)
    window_ids = np.repeat(np.arange(n_windows), n_bins)
    bin_ids = np.tile(np.arange(n_bins), n_windows)
    masks = {
        k: v.reshape(-1)
        for k, v in build_stratified_masks(
            observed.transpose(0, 2, 1), thresholds, seed=seed + 1
        ).items()
    }
    obs_metrics = {
        "entropy": normalized_entropy_specificity(obs_rows),
        "centered_l2": centered_l2_over_total(obs_rows),
        "log2_margin": top_second_log2_margin(obs_rows, thresholds["margin_eps"]),
    }
    pred_metrics = {
        "entropy": normalized_entropy_specificity(pred_rows),
        "centered_l2": centered_l2_over_total(pred_rows),
        "log2_margin": top_second_log2_margin(pred_rows, thresholds["margin_eps"]),
    }
    long_data = {
        "model": model_name,
        "window_idx": window_ids,
        "bin_idx": bin_ids,
        "observed_top": np.argmax(obs_rows, axis=1),
        "predicted_top": np.argmax(pred_rows, axis=1),
    }
    for i, task in enumerate(TASKS):
        long_data[f"observed_{task}"] = obs_rows[:, i]
        long_data[f"predicted_{task}"] = pred_rows[:, i]
    for metric in obs_metrics:
        long_data[f"observed_{metric}"] = obs_metrics[metric]
        long_data[f"predicted_{metric}"] = pred_metrics[metric]
        long_data[f"gap_{metric}"] = pred_metrics[metric] - obs_metrics[metric]
    for name, mask in masks.items():
        long_data[f"mask_{name}"] = mask
    long_df = pd.DataFrame(long_data)
    long_df.to_csv(out_dir / "bin_metrics.tsv.gz", sep="\t", index=False, compression="gzip")
    selected = long_df[
        long_df[["mask_shared_peak", "mask_cell_specific_peak", "mask_silent_negative"]].any(axis=1)
    ].copy()
    selected["stratum"] = np.select(
        [selected["mask_shared_peak"], selected["mask_cell_specific_peak"], selected["mask_silent_negative"]],
        ["shared_peak", "cell_specific_peak", "silent_negative"],
        default="unclassified",
    )
    selected.to_csv(out_dir / "stratified_bins.tsv.gz", sep="\t", index=False, compression="gzip")

    summary_rows, decile_rows, correlation_rows, confusion_rows = [], [], [], []
    mask_details = {}
    for mask_name, mask in masks.items():
        selected = np.flatnonzero(mask)
        class_counts = (
            np.bincount(np.argmax(obs_rows[selected], axis=1), minlength=n_tasks)
            if len(selected)
            else np.zeros(n_tasks, dtype=int)
        )
        mask_details[mask_name] = {
            "n_bins": int(len(selected)),
            "observed_top_class_counts": dict(zip(TASKS, class_counts.astype(int).tolist())),
            "observed_top_tie_count": int((~unique_top(obs_rows[selected])).sum()) if len(selected) else 0,
        }
        if not len(selected):
            continue
        identity = top_identity_metrics(obs_rows[selected], pred_rows[selected])
        for i, true_task in enumerate(TASKS):
            for j, pred_task in enumerate(TASKS):
                confusion_rows.append(
                    {"model": model_name, "mask": mask_name, "observed_top": true_task,
                     "predicted_top": pred_task, "n": identity["confusion_matrix"][i][j]}
                )
        for metric in obs_metrics:
            obs_v, pred_v = obs_metrics[metric][selected], pred_metrics[metric][selected]
            boot = cluster_bootstrap_mean_ci(
                pred_v - obs_v, window_ids[selected], seed=seed, n_boot=n_boot
            )
            slope = (
                float(np.polyfit(obs_v, pred_v, 1)[0])
                if np.ptp(obs_v) > 0
                else float("nan")
            )
            obs_mean, pred_mean = float(obs_v.mean()), float(pred_v.mean())
            summary_rows.append({
                "model": model_name, "mask": mask_name, "metric": metric,
                "n_bins": len(selected), "n_windows": boot["n_clusters"],
                "observed_mean": obs_mean, "predicted_mean": pred_mean,
                "gap_mean": boot["mean"], "gap_ci_low": boot["ci_low"],
                "gap_ci_high": boot["ci_high"], "calibration_slope": slope,
                "prediction_observation_ratio": pred_mean / obs_mean if obs_mean > 0 else float("nan"),
                "top_accuracy": identity["accuracy"], "top_macro_recall": identity["macro_recall"],
                "observed_top_ties": identity["observed_tie_count"],
                "predicted_top_ties": identity["predicted_tie_count"],
            })
            for row in specificity_deciles(obs_v, pred_v):
                decile_rows.append({"model": model_name, "mask": mask_name, "metric": metric, **row})
        for source, values in (("observed", obs_rows[selected]), ("predicted", pred_rows[selected])):
            for i, j in combinations(range(n_tasks), 2):
                for method in ("pearson", "spearman"):
                    result = safe_correlation(values[:, i], values[:, j], method)
                    correlation_rows.append({
                        "model": model_name, "mask": mask_name, "source": source,
                        "task_a": TASKS[i], "task_b": TASKS[j], "method": method,
                        "correlation": result.value, "status": result.status, "n_bins": len(selected),
                    })
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(out_dir / "summary.tsv", sep="\t", index=False)
    pd.DataFrame(decile_rows).to_csv(out_dir / "calibration_deciles.tsv", sep="\t", index=False)
    pd.DataFrame(correlation_rows).to_csv(out_dir / "pairwise_correlations.tsv", sep="\t", index=False)
    pd.DataFrame(confusion_rows).to_csv(out_dir / "top_cell_confusion.tsv", sep="\t", index=False)
    return summary, mask_details


def make_plots(out_dir: Path, summary: pd.DataFrame) -> None:
    entropy = summary[summary.metric == "entropy"].copy()
    order = [m for m in [
        "all_finite", "observed_active", "train_total_p90", "train_total_p95", "train_total_p99",
        "train_union_track_p95", "peak_candidate", "shared_peak", "cell_specific_peak", "other_peak_0or3of4", "silent_negative",
    ] if m in set(entropy["mask"])]
    entropy["mask"] = pd.Categorical(entropy["mask"], order, ordered=True)
    entropy = entropy.sort_values("mask")
    fig, ax = plt.subplots(figsize=(10, 5.5))
    x = np.arange(len(entropy))
    lower_error = np.maximum(
        0.0,
        entropy.gap_mean.to_numpy(dtype=float)
        - entropy.gap_ci_low.to_numpy(dtype=float),
    )
    upper_error = np.maximum(
        0.0,
        entropy.gap_ci_high.to_numpy(dtype=float)
        - entropy.gap_mean.to_numpy(dtype=float),
    )
    ax.errorbar(
        x, entropy.gap_mean,
        yerr=np.vstack([lower_error, upper_error]),
        fmt="o", color=PALETTE["gap"], ecolor="#4A4A4A", capsize=4,
    )
    ax.axhline(0, color="#333333", linewidth=1)
    ax.set_xticks(x, entropy["mask"], rotation=25, ha="right")
    ax.set_ylabel("Predicted - observed entropy specificity")
    ax.set_title("Specificity gap by train-defined signal mask", pad=46)
    ax.text(0, 1.025, "Points are means; bars are 95% window-cluster bootstrap CIs", transform=ax.transAxes, color="#555555")
    for xi, row in enumerate(entropy.itertuples()):
        ax.annotate(f"N={row.n_bins:,}", (xi, row.gap_ci_high), xytext=(0, 6), textcoords="offset points", ha="center", fontsize=8)
    ax.margins(y=0.15)
    ax.grid(axis="y", color="#E5E5E5")
    fig.tight_layout()
    fig.savefig(out_dir / "specificity_gap_by_mask.png", dpi=180)
    plt.close(fig)

    deciles = pd.read_csv(out_dir / "calibration_deciles.tsv", sep="\t")
    deciles = deciles[(deciles.metric == "entropy") & (deciles["mask"].isin(["observed_active", "train_total_p95"]))]
    fig, ax = plt.subplots(figsize=(7.5, 6))
    if deciles.empty:
        ax.text(
            0.5,
            0.5,
            "No observed-active or train-P95 bins\nin this deterministic subset",
            transform=ax.transAxes,
            ha="center",
            va="center",
            color="#555555",
        )
    else:
        maximum = max(deciles.observed_mean.max(), deciles.predicted_mean.max())
        ax.plot([0, maximum], [0, maximum], linestyle="--", color="#444444", label="ideal")
        styles = {"observed_active": ("o", "-"), "train_total_p95": ("s", "--")}
        for name, group in deciles.groupby("mask", observed=True):
            marker, linestyle = styles[str(name)]
            ax.plot(group.observed_mean, group.predicted_mean, marker=marker, linestyle=linestyle, color=PALETTE["predicted"], label=f"{name} (N={group.n.sum():,})")
    ax.set_xlabel("Observed entropy specificity (decile mean)")
    ax.set_ylabel("Predicted entropy specificity (decile mean)")
    ax.set_title("Observed versus predicted specificity calibration")
    if not deciles.empty:
        ax.legend(frameon=False)
    ax.grid(color="#E5E5E5")
    fig.tight_layout()
    fig.savefig(out_dir / "specificity_calibration_deciles.png", dpi=180)
    plt.close(fig)


def json_safe(value):
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def main() -> None:
    args = parse_args()
    started = time.time()
    checkpoint_path = (args.checkpoint or DEFAULT_CHECKPOINTS[args.model]).resolve()
    checkpoint, metadata = load_checkpoint_metadata(checkpoint_path)
    if metadata["task_names"] != TASKS:
        raise RuntimeError(f"unexpected task order: {metadata['task_names']}")
    model_bin_size = int(metadata["data"]["train"]["bin_size"])
    analysis_bin_size = int(args.analysis_bin_size or model_bin_size)
    inference_cache, inference_manifest = matching_cache(metadata, model_bin_size)
    analysis_cache, analysis_manifest = matching_cache(metadata, analysis_bin_size)
    suffix = f"subset_{args.max_windows}" if args.max_windows else "full_val"
    if analysis_bin_size != model_bin_size:
        suffix += f"_bin{analysis_bin_size}"
    out_dir = args.out_dir / args.model / suffix
    out_dir.mkdir(parents=True, exist_ok=True)
    train_labels = np.load(analysis_cache / "train_labels.npy", mmap_mode="r")
    thresholds = derive_train_thresholds_from_cache(train_labels)
    n_windows = metadata["data"]["val"]["n_seqs"]
    if args.max_windows is not None:
        n_windows = min(n_windows, args.max_windows)
    expected_bins = int(analysis_manifest["val_shape"][-1])
    analysis_val_labels = np.load(analysis_cache / "val_labels.npy", mmap_mode="r")
    expected_analysis_shape = (metadata["data"]["val"]["n_seqs"], len(TASKS), expected_bins)
    if analysis_val_labels.shape != expected_analysis_shape:
        raise RuntimeError(
            f"analysis validation label shape mismatch: {analysis_val_labels.shape} vs {expected_analysis_shape}"
        )
    observed_path = out_dir / "observed.npy"
    predicted_path = out_dir / "predicted.npy"
    source_dir = args.prediction_cache_dir.resolve() if args.prediction_cache_dir else None
    if source_dir is not None:
        source_observed_path = source_dir / "observed.npy"
        source_predicted_path = source_dir / "predicted.npy"
        if not source_observed_path.exists() or not source_predicted_path.exists():
            raise FileNotFoundError(f"prediction cache is incomplete: {source_dir}")
        source_observed = np.load(source_observed_path)
        source_predicted = np.load(source_predicted_path)
        source_bins = int(metadata["data"]["val"]["binned_label_len"])
        if (
            source_observed.shape[1:] != (len(TASKS), source_bins)
            or source_predicted.shape != source_observed.shape
            or source_observed.shape[0] < n_windows
        ):
            raise RuntimeError("source prediction cache does not match checkpoint output shape")
        observed = source_observed[:n_windows]
        predicted = source_predicted[:n_windows]
        if analysis_bin_size != model_bin_size:
            predicted = aggregate_bin_axis(
                predicted,
                source_bin_size=model_bin_size,
                target_bin_size=analysis_bin_size,
            )
        # Use the canonical validation labels for the requested analysis bin
        # size. Model-output aggregation is only for predictions; independently
        # cached 128-bp labels may differ at bin boundaries from summing 32-bp
        # labels because they were built directly from the source tracks.
        observed = np.asarray(analysis_val_labels[:n_windows]).copy()
        loader_details = {
            "finetune_mode": "source_prediction_cache",
            "prediction_cache_reused": True,
            "prediction_cache_dir": str(source_dir),
            "analysis_aggregation": {
                "source_bin_size": model_bin_size,
                "target_bin_size": analysis_bin_size,
                "operation": "sum_contiguous_source_bins",
            },
        }
        print(f"reused source predictions: {source_dir}", flush=True)
        np.save(observed_path, observed)
        np.save(predicted_path, predicted)
    elif observed_path.exists() and predicted_path.exists():
        observed = np.load(observed_path)
        predicted = np.load(predicted_path)
        if (
            observed.shape != (n_windows, len(TASKS), expected_bins)
            or predicted.shape != observed.shape
        ):
            raise RuntimeError("saved prediction cache does not match requested window count")
        lora_keys = [key for key in checkpoint["state_dict"] if "lora_A" in key]
        loader_details = {
            "finetune_mode": "lora" if lora_keys else "headonly",
            "prediction_cache_reused": True,
        }
        if lora_keys:
            loader_details["lora_rank_from_state"] = int(
                checkpoint["state_dict"][lora_keys[0]].shape[0]
            )
        print(f"reused predictions: {observed_path}", flush=True)
    else:
        device = torch.device(args.device if torch.cuda.is_available() else "cpu")
        model, loader_details = load_model(checkpoint, args.model, device)
        dataset = make_val_dataset(inference_cache, metadata)
        observed, predicted = infer(model, dataset, n_windows, device)
        dataset.close_handles()
        if analysis_bin_size != model_bin_size:
            predicted = aggregate_bin_axis(
                predicted,
                source_bin_size=model_bin_size,
                target_bin_size=analysis_bin_size,
            )
            loader_details["analysis_aggregation"] = {
                "source_bin_size": model_bin_size,
                "target_bin_size": analysis_bin_size,
                "operation": "sum_contiguous_source_bins",
            }
        observed = np.asarray(analysis_val_labels[:n_windows]).copy()
        np.save(observed_path, observed)
        np.save(predicted_path, predicted)
    del checkpoint
    gc.collect()
    if not np.isfinite(observed).all() or not np.isfinite(predicted).all():
        raise RuntimeError("non-finite observed or predicted values")
    if (observed < 0).any() or (predicted < 0).any():
        raise RuntimeError("negative observed or predicted values")
    summary, mask_details = write_analysis(
        out_dir, args.model, observed, predicted, thresholds, args.seed, args.bootstrap
    )
    make_plots(out_dir, summary)
    split_manifest_path = REPO_ROOT / "dataset_store/splits/split_chr10_chr11/manifest.json"
    validation = {
        "status": "ok",
        "scope": "output specificity only; does not establish catastrophic forgetting or embedding feature collapse",
        "background_warning": "all_finite is background-sensitive; primary interpretation uses active and train-derived high-signal masks",
        "stratification": {
            "peak_candidate": "validation bins with total signal >= train-derived P90",
            "shared_peak": "peak_candidate with all four tracks >= their train-derived P95",
            "cell_specific_peak": "peak_candidate with one or two tracks >= their train-derived P95",
            "other_peak_0or3of4": "peak_candidate with zero or three tracks >= their train-derived P95; retained as an explicit boundary stratum",
            "silent_negative": "global seeded sample of validation bins with total signal == 0, matched to positive peak-stratum count when available",
            "negative_sampling_seed": args.seed + 1,
            "prediction_free_selection": True,
        },
        "subset_interpretation": (
            "background-only smoke test with no active validation bins; not scientifically interpretable"
            if args.max_windows is not None
            and not mask_details.get("observed_active", {}).get("n_bins", 0)
            else "contains active validation bins"
        ),
        "model": args.model,
        "model_bin_size": model_bin_size,
        "analysis_bin_size": analysis_bin_size,
        "checkpoint": {
            "path": str(checkpoint_path.relative_to(REPO_ROOT)),
            "sha256": sha256_file(checkpoint_path),
            "epoch": metadata["epoch"],
            "global_step": metadata["global_step"],
            **loader_details,
        },
        "checkpoint_hparams": {"model_params": metadata["model_params"], "train_params": metadata["train_params"]},
        "task_order": TASKS,
        "cache_manifest": {"path": str((analysis_cache / "manifest.json").relative_to(REPO_ROOT)), "sha256": sha256_file(analysis_cache / "manifest.json"), "content": analysis_manifest},
        "inference_cache_manifest": {"path": str((inference_cache / "manifest.json").relative_to(REPO_ROOT)), "sha256": sha256_file(inference_cache / "manifest.json"), "content": inference_manifest},
        "split_manifest": {"path": str(split_manifest_path.relative_to(REPO_ROOT)), "sha256": sha256_file(split_manifest_path)},
        "thresholds_train_only": thresholds,
        "seed": args.seed,
        "bootstrap_replicates": args.bootstrap,
        "processed_windows": n_windows,
        "expected_val_windows": metadata["data"]["val"]["n_seqs"],
        "processed_bins": int(observed.shape[0] * observed.shape[2]),
        "shape": list(observed.shape),
        "unique_window_bin_keys": int(observed.shape[0] * observed.shape[2]),
        "finiteness": True,
        "nonnegativity": True,
        "masks": mask_details,
        "command": " ".join(sys.argv),
        "git_commit": git_commit(),
        "runtime_seconds": time.time() - started,
        "outputs": [
            "bin_metrics.tsv.gz", "stratified_bins.tsv.gz", "summary.tsv", "calibration_deciles.tsv",
            "pairwise_correlations.tsv", "top_cell_confusion.tsv",
            "specificity_gap_by_mask.png", "specificity_calibration_deciles.png",
            "observed.npy", "predicted.npy",
        ],
    }
    (out_dir / "validation_summary.json").write_text(json.dumps(json_safe(validation), indent=2) + "\n")
    print(json.dumps({"out_dir": str(out_dir), "runtime_seconds": validation["runtime_seconds"], "processed_windows": n_windows}, indent=2))


if __name__ == "__main__":
    main()
