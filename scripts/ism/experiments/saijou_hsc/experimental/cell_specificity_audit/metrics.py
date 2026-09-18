"""Pure metrics for four-track output-specificity auditing.

All arrays use ``(..., task)`` layout and non-negative values. A zero vector
has specificity zero. These functions deliberately know nothing about genomic
coordinates or model implementations.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats


def _validated(values: np.ndarray) -> np.ndarray:
    x = np.asarray(values, dtype=np.float64)
    if x.ndim < 2:
        raise ValueError("values must have at least two dimensions (..., task)")
    if x.shape[-1] < 2:
        raise ValueError("at least two tasks are required")
    if not np.isfinite(x).all():
        raise ValueError("values must be finite")
    if (x < 0).any():
        raise ValueError("values must be non-negative")
    return x


def normalized_entropy_specificity(values: np.ndarray) -> np.ndarray:
    """Return ``1 - H(x/sum(x))/log(K)``; zero vectors map to zero."""
    x = _validated(values)
    total = x.sum(axis=-1)
    p = np.divide(x, total[..., None], out=np.zeros_like(x), where=total[..., None] > 0)
    terms = np.zeros_like(p)
    positive = p > 0
    terms[positive] = p[positive] * np.log(p[positive])
    out = 1.0 + terms.sum(axis=-1) / np.log(x.shape[-1])
    return np.where(total > 0, np.clip(out, 0.0, 1.0), 0.0)


def centered_l2_over_total(values: np.ndarray) -> np.ndarray:
    """Return ``sqrt(sum_i (x_i - mean(x))^2) / sum_i x_i``.

    The ratio is scale invariant. It is zero for zero/uniform vectors and
    ``sqrt((K-1)/K)`` for a one-hot vector. Zero vectors map to zero.
    """
    x = _validated(values)
    total = x.sum(axis=-1)
    contrast = np.linalg.norm(x - x.mean(axis=-1, keepdims=True), axis=-1)
    return np.divide(contrast, total, out=np.zeros_like(total), where=total > 0)


def top_second_log2_margin(values: np.ndarray, eps: float | np.ndarray) -> np.ndarray:
    """Return ``log2((largest + eps)/(second_largest + eps))``."""
    x = _validated(values)
    e = np.asarray(eps, dtype=np.float64)
    if np.any(~np.isfinite(e)) or np.any(e <= 0):
        raise ValueError("eps must be finite and positive")
    ordered = np.sort(x, axis=-1)
    return np.log2((ordered[..., -1] + e) / (ordered[..., -2] + e))


def unique_top(values: np.ndarray) -> np.ndarray:
    """Return whether each vector has one unique maximum."""
    x = _validated(values)
    top = x.max(axis=-1, keepdims=True)
    return (x == top).sum(axis=-1) == 1


def derive_train_thresholds(
    train_values: np.ndarray,
    quantiles: tuple[float, ...] = (0.90, 0.95, 0.99),
    eps_quantile: float = 0.01,
) -> dict[str, object]:
    """Derive every selection threshold and margin epsilon from train labels."""
    x = _validated(train_values).reshape(-1, train_values.shape[-1])
    totals = x.sum(axis=1)
    positive = x[x > 0]
    if positive.size == 0:
        raise ValueError("train labels contain no positive values")
    if not 0 < eps_quantile < 1:
        raise ValueError("eps_quantile must be in (0, 1)")
    return {
        "total_quantiles": {f"p{int(q * 100):02d}": float(np.quantile(totals, q)) for q in quantiles},
        "per_track_p95": np.quantile(x, 0.95, axis=0).astype(float).tolist(),
        "margin_eps": float(np.quantile(positive, eps_quantile)),
        "margin_eps_quantile": eps_quantile,
        "train_bin_count": int(len(x)),
    }


def derive_train_thresholds_from_cache(
    train_cache: np.ndarray,
    quantiles: tuple[float, ...] = (0.90, 0.95, 0.99),
    eps_quantile: float = 0.01,
) -> dict[str, object]:
    """Derive thresholds from the repository cache layout ``(window, task, bin)``.

    Keeping this layout conversion explicit prevents the genomic-bin axis from
    being silently interpreted as the task axis.
    """
    x = np.asarray(train_cache)
    if x.ndim != 3:
        raise ValueError("train cache must have (window, task, bin) layout")
    return derive_train_thresholds(
        x.transpose(0, 2, 1),
        quantiles=quantiles,
        eps_quantile=eps_quantile,
    )


def aggregate_bin_axis(
    values: np.ndarray,
    *,
    source_bin_size: int,
    target_bin_size: int,
) -> np.ndarray:
    """Sum contiguous ``(window, task, bin)`` values into a coarser bin size."""
    x = np.asarray(values)
    if x.ndim != 3:
        raise ValueError("values must have (window, task, bin) layout")
    if source_bin_size <= 0 or target_bin_size <= 0 or target_bin_size < source_bin_size:
        raise ValueError("target bin size must be a positive coarsening of source bin size")
    if target_bin_size % source_bin_size:
        raise ValueError("target bin size must be divisible by source bin size")
    factor = target_bin_size // source_bin_size
    if x.shape[-1] % factor:
        raise ValueError("bin axis is not divisible by the requested aggregation factor")
    return x.reshape(x.shape[0], x.shape[1], x.shape[2] // factor, factor).sum(axis=-1)


def build_masks(values: np.ndarray, thresholds: dict[str, object]) -> dict[str, np.ndarray]:
    """Apply train-derived thresholds to another split without re-estimation."""
    x = _validated(values)
    totals = x.sum(axis=-1)
    masks = {
        "all_finite": np.isfinite(x).all(axis=-1),
        "observed_active": totals > 0,
    }
    for name, threshold in dict(thresholds["total_quantiles"]).items():
        masks[f"train_total_{name}"] = totals >= float(threshold)
    per_track = np.asarray(thresholds["per_track_p95"], dtype=np.float64)
    if per_track.shape != (x.shape[-1],):
        raise ValueError("per_track_p95 does not match task dimension")
    masks["train_union_track_p95"] = (x >= per_track).any(axis=-1)
    return masks


def build_stratified_masks(
    values: np.ndarray,
    thresholds: dict[str, object],
    *,
    seed: int = 20260824,
) -> dict[str, np.ndarray]:
    """Build train-defined peak strata and matched silent controls.

    ``values`` is the held-out observed label array in ``(..., task)`` layout.
    Peak membership uses only thresholds derived from the training labels:
    total signal must reach train P90, then the number of tracks reaching their
    own train P95 defines shared (4/4) versus cell-specific (1-2/4) peaks.
    Three-of-four candidates are retained as an explicit boundary stratum.
    Silent controls are sampled globally from held-out all-zero bins, matched
    to the total number of positive peak strata when enough controls exist.
    They are an evaluation control and never affect peak selection.
    """
    x = _validated(values)
    if x.ndim < 3:
        raise ValueError("stratified masks require a leading window dimension")
    base = build_masks(x, thresholds)
    total_p90 = float(thresholds["total_quantiles"]["p90"])
    per_track = np.asarray(thresholds["per_track_p95"], dtype=np.float64)
    if per_track.shape != (x.shape[-1],):
        raise ValueError("per_track_p95 does not match task dimension")

    totals = x.sum(axis=-1)
    high_track_count = (x >= per_track).sum(axis=-1)
    peak_candidate = totals >= total_p90
    base["peak_candidate"] = peak_candidate
    base["shared_peak"] = peak_candidate & (high_track_count == x.shape[-1])
    base["cell_specific_peak"] = peak_candidate & (high_track_count >= 1) & (high_track_count <= 2)
    base["other_peak_0or3of4"] = peak_candidate & ((high_track_count == 0) | (high_track_count == 3))

    # Match controls globally so the negative set has the same total sample
    # size as the positive strata whenever the held-out split has enough
    # silent bins. The sampler is deterministic and never inspects predictions.
    negative = np.zeros(peak_candidate.shape, dtype=bool)
    rng = np.random.default_rng(seed)
    peak_any = base["shared_peak"] | base["cell_specific_peak"]
    peak_count = int(peak_any.sum())
    silent_flat = np.flatnonzero((totals == 0).reshape(-1))
    n_sample = min(peak_count, len(silent_flat))
    if n_sample:
        chosen = rng.choice(silent_flat, size=n_sample, replace=False)
        negative.reshape(-1)[chosen] = True
    base["silent_negative"] = negative
    return base


@dataclass(frozen=True)
class CorrelationResult:
    value: float
    status: str


def safe_correlation(a: np.ndarray, b: np.ndarray, method: str) -> CorrelationResult:
    """Compute Pearson/Spearman, returning an explicit constant-input status."""
    x = np.asarray(a, dtype=np.float64).reshape(-1)
    y = np.asarray(b, dtype=np.float64).reshape(-1)
    if x.shape != y.shape or x.size < 2:
        return CorrelationResult(float("nan"), "insufficient")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        return CorrelationResult(float("nan"), "nonfinite")
    if np.ptp(x) == 0 or np.ptp(y) == 0:
        return CorrelationResult(float("nan"), "constant")
    if method == "pearson":
        value = stats.pearsonr(x, y).statistic
    elif method == "spearman":
        value = stats.spearmanr(x, y).statistic
    else:
        raise ValueError(f"unknown correlation method: {method}")
    return CorrelationResult(float(value), "ok")


def top_identity_metrics(
    observed: np.ndarray, predicted: np.ndarray, n_tasks: int | None = None
) -> dict[str, object]:
    """Top-task metrics on bins with a unique observed top.

    Predicted ties use stable ``argmax`` but are reported separately, so their
    direction-dependent bias is visible rather than silently discarded.
    """
    obs = _validated(observed).reshape(-1, observed.shape[-1])
    pred = _validated(predicted).reshape(-1, predicted.shape[-1])
    if obs.shape != pred.shape:
        raise ValueError("observed and predicted shapes differ")
    k = n_tasks or obs.shape[1]
    keep = unique_top(obs)
    true = np.argmax(obs[keep], axis=1)
    guess = np.argmax(pred[keep], axis=1)
    matrix = np.zeros((k, k), dtype=np.int64)
    np.add.at(matrix, (true, guess), 1)
    recalls = []
    for i in range(k):
        denom = matrix[i].sum()
        recalls.append(float(matrix[i, i] / denom) if denom else float("nan"))
    return {
        "n": int(keep.sum()),
        "observed_tie_count": int((~keep).sum()),
        "predicted_tie_count": int((~unique_top(pred[keep])).sum()),
        "accuracy": float(np.mean(true == guess)) if len(true) else float("nan"),
        "macro_recall": float(np.nanmean(recalls)) if np.isfinite(recalls).any() else float("nan"),
        "recall_by_class": recalls,
        "confusion_matrix": matrix.tolist(),
    }


def cluster_bootstrap_mean_ci(
    values: np.ndarray,
    cluster_ids: np.ndarray,
    *,
    seed: int = 20260824,
    n_boot: int = 2000,
    confidence: float = 0.95,
) -> dict[str, float | int]:
    """Bootstrap a bin-weighted mean by resampling whole window clusters."""
    x = np.asarray(values, dtype=np.float64).reshape(-1)
    clusters = np.asarray(cluster_ids).reshape(-1)
    if x.shape != clusters.shape or not np.isfinite(x).all():
        raise ValueError("values and cluster_ids must align and values must be finite")
    unique, inverse = np.unique(clusters, return_inverse=True)
    sums = np.bincount(inverse, weights=x)
    counts = np.bincount(inverse)
    if len(unique) == 0:
        raise ValueError("no clusters")
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(unique), size=(n_boot, len(unique)))
    means = sums[draws].sum(axis=1) / counts[draws].sum(axis=1)
    alpha = (1 - confidence) / 2
    return {
        "mean": float(x.mean()),
        "ci_low": float(np.quantile(means, alpha)),
        "ci_high": float(np.quantile(means, 1 - alpha)),
        "n_clusters": int(len(unique)),
        "n_values": int(len(x)),
        "n_boot": int(n_boot),
        "seed": int(seed),
    }


def specificity_deciles(observed: np.ndarray, predicted: np.ndarray) -> list[dict[str, float | int]]:
    """Return equal-frequency observed-specificity calibration strata."""
    obs = np.asarray(observed, dtype=np.float64).reshape(-1)
    pred = np.asarray(predicted, dtype=np.float64).reshape(-1)
    if obs.shape != pred.shape or not np.isfinite(obs).all() or not np.isfinite(pred).all():
        raise ValueError("observed and predicted must be aligned and finite")
    order = np.argsort(obs, kind="stable")
    rows = []
    for decile, idx in enumerate(np.array_split(order, 10), start=1):
        if not len(idx):
            continue
        obs_mean = float(obs[idx].mean())
        pred_mean = float(pred[idx].mean())
        rows.append({
            "decile": decile,
            "n": int(len(idx)),
            "observed_mean": obs_mean,
            "predicted_mean": pred_mean,
            "gap_mean": pred_mean - obs_mean,
            "prediction_observation_ratio": pred_mean / obs_mean if obs_mean > 0 else float("nan"),
        })
    return rows
