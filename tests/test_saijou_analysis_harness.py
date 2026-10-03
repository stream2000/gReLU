import numpy as np
import pandas as pd
import pytest

from scripts.ism.experiments.saijou_hsc.tools.effect_summary import (
    summarize_cell_windows,
)
from scripts.ism.experiments.saijou_hsc.tools.harness import (
    require_columns,
    require_finite,
    require_unique,
)
from scripts.ism.experiments.saijou_hsc.tools.region_importance import (
    combine_models,
    reference_bins,
)


def test_harness_schema_key_and_finite_contracts():
    frame = pd.DataFrame({"id": [1, 2], "effect": [0.1, -0.2]})
    require_columns(frame, ["id", "effect"], name="synthetic")
    require_unique(frame, ["id"], name="synthetic")
    require_finite(frame, ["effect"], name="synthetic")

    with pytest.raises(ValueError, match="missing required columns"):
        require_columns(frame, ["unknown"], name="synthetic")
    with pytest.raises(ValueError, match="duplicate rows"):
        require_unique(pd.concat([frame, frame.iloc[[0]]]), ["id"], name="synthetic")
    with pytest.raises(ValueError, match="non-finite"):
        require_finite(pd.DataFrame({"effect": [np.nan]}), ["effect"], name="synthetic")


def test_window_summary_preserves_signed_effect_and_ranks_cells():
    rows = []
    for model in ("AlphaGenome fine-tuned", "Borzoi fine-tuned"):
        for cell, magnitude in (("HSC", 0.4), ("Macrophage", 0.1)):
            for offset, scale in ((0, 1.0), (2, 0.5)):
                rows.append(
                    {
                        "model": model,
                        "cell": cell,
                        "variant_offset_from_tss_transcription_bp": offset,
                        "effect": -magnitude * scale,
                    }
                )
    summary = summarize_cell_windows(pd.DataFrame(rows), {"test": (0, 2)})
    hsc = summary.loc[summary.cell.eq("HSC")]
    assert hsc.cell_rank.eq(1).all()
    assert hsc.median_signed_log2fc.lt(0).all()
    assert hsc.fraction_centers_negative.eq(1).all()
    assert hsc.centers.eq(2).all()


def test_cross_model_track_score_uses_the_weaker_same_track_percentile():
    rows = []
    for model, percentile, effect in (
        ("AlphaGenome", 0.95, -0.4),
        ("Borzoi", 0.80, -0.2),
    ):
        rows.append(
            {
                "gene": "GeneA",
                "variant_offset_from_tss_transcription_bp": 11,
                "track_id": "hsc",
                "readout_role": "gene_body_output_clipped",
                "model": model,
                "model_tail_percentile": percentile,
                "median_signed_log2fc": effect,
            }
        )
    combined = combine_models(pd.DataFrame(rows))

    assert combined.cross_model_conjunction.tolist() == [0.80]
    assert combined.model_direction_agreement.tolist() == [True]


def test_reference_bins_are_deterministic_with_tied_values():
    values = pd.Series([0.0] * 100 + [1.0] * 100)

    first = reference_bins(values, requested=10)
    second = reference_bins(values, requested=10)

    assert first.equals(second)
    assert first.nunique() == 2
