from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pandas as pd


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts/ism/experiments/saijou_hsc/experimental"
    / "original_multitrack_score/score_calculations.py"
)
SPEC = importlib.util.spec_from_file_location(
    "original_multitrack_score_calculations", MODULE_PATH
)
assert SPEC is not None and SPEC.loader is not None
scoring = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = scoring
SPEC.loader.exec_module(scoring)


def test_spatial_support_uses_coordinate_radius() -> None:
    frame = pd.DataFrame(
        {
            "gene": ["G"] * 5,
            "variant_offset_from_tss_transcription_bp": [-4, -2, 0, 2, 4],
            "value": [1.0, 2.0, 9.0, 4.0, 5.0],
        }
    )

    result = scoring.compute_spatial_median_support(
        frame, "value", radius_bp=2
    )

    assert result.spatial_support.tolist() == [1.5, 2.0, 4.0, 5.0, 4.5]
    assert result.spatial_support_centers.tolist() == [2, 3, 3, 3, 2]


def test_interval_recall_separates_center_and_window_metrics() -> None:
    centers = pd.DataFrame(
        {
            "gene": ["G"] * 7,
            "variant_offset_from_tss_transcription_bp": [
                -6,
                -4,
                -2,
                0,
                2,
                4,
                6,
            ],
            "importance_score": [10.0, 20.0, 30.0, 99.0, 40.0, 50.0, 60.0],
            "statistic": [1.0, 2.0, 3.0, 9.0, 4.0, 5.0, 6.0],
            "median_group_signed_log2fc": [-0.1] * 7,
            "loss_direction_group_fraction": [1.0] * 7,
        }
    )
    intervals = pd.DataFrame(
        [{"control_id": "c", "gene": "G", "start": 0, "end": 0}]
    )

    result = scoring.evaluate_interval_recall(
        centers,
        intervals,
        95.0,
        interval_kind="test",
        window_statistic_column="statistic",
    ).iloc[0]

    assert result.best_center == 0
    assert bool(result.recalled_at_threshold)
    assert np.isfinite(result.window_percentile_score)
    assert result.covered_centers == 5


def test_modality_view_score_requires_three_groups_per_view() -> None:
    rows: list[dict[str, object]] = []
    positions = [-4, -2, 0, 2, 4]
    effects = [0.1, 0.2, 1.0, 0.4, 0.3]
    for view in ["output", "local_regulatory"]:
        for group in ["g1", "g2", "g3"]:
            for position, effect in zip(positions, effects, strict=True):
                rows.append(
                    {
                        "gene": "G",
                        "variant_offset_from_tss_transcription_bp": position,
                        "score_view": view,
                        "track_group": group,
                        "track_id": f"{view}_{group}",
                        "median_absolute_log2fc": effect,
                        "median_signed_log2fc": -effect,
                    }
                )
    tracks = pd.DataFrame.from_records(rows)

    groups, view_centers, centers = scoring.score_modality_views(
        tracks, 95.0
    )

    assert groups.track_group.nunique() == 3
    assert set(view_centers.score_view) == {"output", "local_regulatory"}
    assert len(centers) == len(positions)
    assert not centers.duplicated(scoring.CENTER_KEYS).any()
    peak = centers.loc[
        centers.variant_offset_from_tss_transcription_bp.eq(0)
    ].iloc[0]
    assert centers.importance_score.max() == 100.0
    assert centers.importance_score.between(0.0, 100.0).all()
    assert peak.loss_direction_group_fraction == 1.0
