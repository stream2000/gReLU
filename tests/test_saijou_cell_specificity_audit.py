from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


MODULE_DIR = (
    Path(__file__).parents[1]
    / "scripts/ism/experiments/saijou_hsc/experimental/cell_specificity_audit"
)
sys.path.insert(0, str(MODULE_DIR))

from metrics import (  # noqa: E402
    build_masks,
    centered_l2_over_total,
    cluster_bootstrap_mean_ci,
    derive_train_thresholds,
    derive_train_thresholds_from_cache,
    build_stratified_masks,
    aggregate_bin_axis,
    normalized_entropy_specificity,
    safe_correlation,
    top_identity_metrics,
    top_second_log2_margin,
)


def test_uniform_one_hot_zero_tie_and_scale_invariance():
    x = np.array([[1, 1, 1, 1], [1, 0, 0, 0], [0, 0, 0, 0], [2, 2, 1, 0]], dtype=float)
    entropy = normalized_entropy_specificity(x)
    l2 = centered_l2_over_total(x)
    margin = top_second_log2_margin(x, eps=0.1)
    assert np.allclose(entropy[:3], [0, 1, 0])
    assert np.allclose(l2[:3], [0, np.sqrt(3 / 4), 0])
    assert margin[0] == 0 and margin[2] == 0 and margin[3] == 0
    assert np.allclose(normalized_entropy_specificity(x), normalized_entropy_specificity(7 * x))
    assert np.allclose(centered_l2_over_total(x), centered_l2_over_total(7 * x))


def test_mask_thresholds_come_from_train_only():
    train = np.array([[[0, 0, 0, 0], [1, 2, 3, 4]], [[10, 0, 0, 0], [0, 8, 0, 0]]], dtype=float)
    val = np.array([[[1000, 0, 0, 0], [0, 0, 0, 0]]], dtype=float)
    thresholds = derive_train_thresholds(train)
    masks = build_masks(val, thresholds)
    assert thresholds["total_quantiles"]["p99"] < 1000
    assert masks["train_total_p99"].tolist() == [[True, False]]
    changed = build_masks(val * 1_000_000, thresholds)
    assert thresholds == thresholds
    assert changed["train_total_p99"].tolist() == [[True, False]]


def test_repository_cache_layout_keeps_four_tasks_as_last_metric_axis():
    cache = np.arange(2 * 4 * 3, dtype=float).reshape(2, 4, 3)
    thresholds = derive_train_thresholds_from_cache(cache)
    expected = derive_train_thresholds(cache.transpose(0, 2, 1))
    assert thresholds == expected
    assert len(thresholds["per_track_p95"]) == 4
    assert thresholds["train_bin_count"] == 6


def test_bin_aggregation_sums_contiguous_source_bins():
    values = np.arange(2 * 1 * 8, dtype=float).reshape(2, 1, 8)
    result = aggregate_bin_axis(values, source_bin_size=32, target_bin_size=128)
    assert result.tolist() == [[[6.0, 22.0]], [[38.0, 54.0]]]


def test_stratified_masks_use_train_thresholds_and_match_silent_controls_per_window():
    thresholds = {
        "total_quantiles": {"p90": 10.0},
        "per_track_p95": [5.0, 5.0, 5.0, 5.0],
        "margin_eps": 0.1,
    }
    val = np.array(
        [
            [[10, 0, 0, 0], [0, 10, 0, 0], [10, 10, 10, 10], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
            [[10, 10, 0, 0], [10, 10, 10, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
        ],
        dtype=float,
    )
    masks = build_stratified_masks(val, thresholds, seed=7)
    assert masks["shared_peak"].sum() == 1
    assert masks["cell_specific_peak"].sum() == 3
    assert masks["silent_negative"].sum() == 4
    assert np.all(masks["silent_negative"] <= (val.sum(axis=-1) == 0))
    repeat = build_stratified_masks(val, thresholds, seed=7)
    for name in ("shared_peak", "cell_specific_peak", "silent_negative"):
        assert np.array_equal(masks[name], repeat[name])


def test_cluster_bootstrap_is_reproducible_and_clustered():
    values = np.array([0.0, 0.0, 10.0, 10.0])
    clusters = np.array([0, 0, 1, 1])
    a = cluster_bootstrap_mean_ci(values, clusters, seed=7, n_boot=500)
    b = cluster_bootstrap_mean_ci(values, clusters, seed=7, n_boot=500)
    assert a == b
    assert a["mean"] == 5.0 and a["n_clusters"] == 2


def test_constant_correlation_is_explicit():
    out = safe_correlation(np.ones(5), np.arange(5), "pearson")
    assert np.isnan(out.value)
    assert out.status == "constant"


def test_synthetic_identical_track_collapse_degrades_specificity_and_identity():
    observed = np.array([[8, 1, 1, 1], [1, 8, 1, 1], [1, 1, 8, 1], [1, 1, 1, 8]], dtype=float)
    predicted = np.full_like(observed, 2.0)
    assert normalized_entropy_specificity(predicted).mean() < normalized_entropy_specificity(observed).mean()
    assert centered_l2_over_total(predicted).mean() < centered_l2_over_total(observed).mean()
    identity = top_identity_metrics(observed, predicted)
    assert identity["accuracy"] == 0.25
    assert identity["predicted_tie_count"] == 4
