from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]


def load_module(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


run = load_module(
    "mdk_motif_run",
    "scripts/ism/experiments/saijou_hsc/experimental/mdk_motif_ism/run_mdk_motif_ism.py",
)
complete = load_module(
    "mdk_motif_complete",
    "scripts/ism/experiments/saijou_hsc/experimental/mdk_motif_ism/complete_e4_e6.py",
)


def test_percentile_ties_insufficient_and_nonfinite_background():
    score, status = run.percentile(2.0, np.full(100, 2.0))
    assert status == "ok"
    assert score == 50.0
    score, status = run.percentile(2.0, np.ones(99))
    assert status == "insufficient_background"
    assert np.isnan(score)
    with pytest.raises(ValueError, match="non-finite"):
        run.percentile(2.0, np.r_[np.ones(99), np.nan])


def test_original_background_uses_requested_coordinate_basis():
    manifest = pd.DataFrame(
        {
            "edit_start": [100, 102, 104, 106, 108],
            "edit_end": [102, 104, 106, 108, 110],
            "edit_center_position": [101, 103, 105, 107, 109],
            "variant_offset_from_tss_transcription_bp": [9, 7, 5, 3, 1],
        }
    )
    selected = pd.DataFrame(
        [{"instance_id": "x", "genomic_start": 103, "genomic_end": 105}]
    )
    backgrounds = run.backgrounds(
        selected, manifest, "variant_offset_from_tss_transcription_bp"
    )
    assert backgrounds["x"]
    assert all(max(window) < 20 for window in backgrounds["x"])


def test_rar_rxr_rule_excludes_ppar_rxr():
    assert run.motif_matches_rule("MA0159.1_RARA::RXRA", "rar_rxr_heterodimer")
    assert not run.motif_matches_rule("MA0065.3_Pparg::Rxra", "rar_rxr_heterodimer")


def test_hsc_rank_requires_all_three_conditions():
    assert run.hsc_priority_pass(0.2, 96, 96)
    assert not run.hsc_priority_pass(0.0, 99, 99)
    assert not run.hsc_priority_pass(0.2, 95, 99)
    assert not run.hsc_priority_pass(0.2, 99, 95)


def test_same_center_same_change_count_control_pair_is_found():
    evidence = pd.DataFrame(
        [
            {
                "instance_id": "motif_1",
                "edit_center_position": 91930956,
                "actual_changed_bases": 6,
                "mutation_id": "loss",
                "continuous_delta": 5.7,
                "loss_class": "loss",
            },
            {
                "instance_id": "motif_1",
                "edit_center_position": 91930956,
                "actual_changed_bases": 6,
                "mutation_id": "preserved",
                "continuous_delta": 0.0,
                "loss_class": "preserved",
            },
        ]
    )
    pairs = complete.build_pair_definitions(evidence)
    assert len(pairs) == 1
    assert pairs.iloc[0].loss_mutation_id == "loss"
    assert pairs.iloc[0].preserved_mutation_id == "preserved"
