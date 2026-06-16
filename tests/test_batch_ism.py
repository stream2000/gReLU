from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from grelu.interpret.ism.batch import (
    MUTATION_STRATEGIES,
    SUMMARY_STRATEGIES,
    MutationDesignContext,
    _MutationConfig,
    _MutationSettings,
    adjust_population_features,
)
from grelu.interpret.ism.motifs import load_meme_pwm


def test_mreg_batch_strategies_are_registered():
    assert "ctcf_pwm_disruption" in MUTATION_STRATEGIES
    assert "center_substitution" in MUTATION_STRATEGIES
    assert "motif_pwm_disruption" in MUTATION_STRATEGIES
    assert "multiscale_log2fc" in SUMMARY_STRATEGIES


def test_generic_motif_strategy_designs_three_base_edit_and_control(tmp_path):
    motif_path = tmp_path / "test.meme"
    motif_path.write_text(
        "\n".join(
            [
                "MEME version 4",
                "",
                "ALPHABET= ACGT",
                "",
                "MOTIF TEST",
                "letter-probability matrix: alength= 4 w= 5",
                "0.97 0.01 0.01 0.01",
                "0.01 0.97 0.01 0.01",
                "0.01 0.01 0.97 0.01",
                "0.01 0.01 0.01 0.97",
                "0.97 0.01 0.01 0.01",
                "",
            ]
        ),
        encoding="utf-8",
    )
    chromosome = "C" * 90 + "GGGTTCCCCC" + "ACGTA" + "G" * 195

    class FakeFasta:
        def extract(self, chrom, start, end):
            assert chrom == "chr1"
            return chromosome[start:end]

        def chrom_length(self, chrom):
            assert chrom == "chr1"
            return len(chromosome)

    pwm = load_meme_pwm(motif_path, motif_name="TEST")
    settings = _MutationSettings(
        min_relative_motif_score=0.80,
        motif_rescan_flank_bp=20,
    )
    context = MutationDesignContext(
        fasta=FakeFasta(),
        pwm=pwm,
        config=_MutationConfig(mutation=settings),
        min_relative_motif_score=0.80,
        sequence_controls=1,
        max_control_distance_bp=30,
    )
    row = SimpleNamespace(
        site_id="candidate",
        chrom="chr1",
        start=90,
        end=120,
        motif_path=str(motif_path),
        motif_name="TEST",
        motif_start=100,
        motif_end=105,
        motif_strand="+",
        motif_sequence="ACGTA",
        min_relative_motif_score=0.80,
        panel_motif_path=str(motif_path),
        panel_motif_names="TEST",
        control_region_start=90,
        control_region_end=120,
        target_feature="TEST_motif",
    )
    design = MUTATION_STRATEGIES["motif_pwm_disruption"](row, context)
    assert design.experimental is not None
    target = design.experimental
    assert sum(
        ref != alt
        for ref, alt in zip(target["ref_sequence"], target["alt_sequence"])
    ) == 3
    assert target["alt_pwm_relative_score"] < 0.80
    assert target["edited_base_count"] == 3
    assert len(design.controls) == 1
    control = design.controls[0]
    assert 90 <= control["edit_start"] < control["edit_end"] <= 120
    assert control["edited_base_count"] == 3
    assert control["gc_delta_count"] == target["gc_delta_count"]


def test_multiscale_summary_has_stable_feature_names():
    summary = SUMMARY_STRATEGIES["multiscale_log2fc"](
        np.ones(16),
        np.full(16, 3.0),
        "ctcf",
        (1024,),
        128,
    )
    assert summary["ctcf__w1024__S_ref_mean"] == 1.0
    assert summary["ctcf__w1024__log2fc_signed_mean"] == 1.0


def test_population_features_subtract_matched_control_mean():
    table = pd.DataFrame(
        [
            {
                "site_id": "site",
                "site_class": "lc_dic",
                "mutation_id": "target",
                "matched_target_id": "target",
                "control_type": "experimental",
                "pol2__w1024__S_ref_mean": 4.0,
                "pol2__w1024__log2fc_signed_mean": -2.0,
            },
            {
                "site_id": "site",
                "site_class": "lc_dic",
                "mutation_id": "control_a",
                "matched_target_id": "target",
                "control_type": "matched_control",
                "pol2__w1024__S_ref_mean": 5.0,
                "pol2__w1024__log2fc_signed_mean": -0.5,
            },
            {
                "site_id": "site",
                "site_class": "lc_dic",
                "mutation_id": "control_b",
                "matched_target_id": "target",
                "control_type": "matched_control",
                "pol2__w1024__S_ref_mean": 6.0,
                "pol2__w1024__log2fc_signed_mean": 0.5,
            },
        ]
    )
    result = adjust_population_features(table)
    assert len(result) == 1
    assert result.iloc[0]["pol2__w1024__S_ref_mean"] == 4.0
    assert result.iloc[0]["pol2__w1024__log2fc_signed_mean"] == -2.0
    assert result.iloc[0]["n_matched_controls"] == 2


def test_dic_population_representation_uses_available_signal_covariates():
    script = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "ism"
        / "analyze_real_ctcf_group_features.py"
    )
    spec = importlib.util.spec_from_file_location("batch_analyzer", script)
    analyzer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(analyzer)
    features = pd.DataFrame(
        {
            "site_id": [f"site_{index}" for index in range(4)],
            "ctcf__w1024__log2fc_signed_mean": [-1.0, -0.5, 0.5, 1.0],
            "ctcf__w4096__log2fc_depletion_mean": [1.0, 0.5, 0.0, 0.0],
        }
    )
    sites = pd.DataFrame(
        {
            "site_id": features["site_id"],
            "rad21_ctrl_signal": [1.0, 2.0, 3.0, 4.0],
            "ctcf_ctrl_signal": [4.0, 3.0, 2.0, 1.0],
        }
    )
    matrices = analyzer._representations(features, sites, None)
    assert "real_biological_all_scales" in matrices
    assert "real_strength_residual" in matrices


def test_single_inference_shard_is_reported_as_not_applicable():
    script = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "ism"
        / "analyze_real_ctcf_group_features.py"
    )
    spec = importlib.util.spec_from_file_location("batch_analyzer_shard", script)
    analyzer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(analyzer)
    assignments = {
        "representation": pd.DataFrame(
            {"site_id": ["a", "b"], "cluster": [0, 1]}
        )
    }
    features = pd.DataFrame(
        {"site_id": ["a", "b"], "inference_shard": [0, 0]}
    )
    result = analyzer._shard_confound(assignments, features)
    assert np.isnan(result.iloc[0]["cluster_vs_inference_shard_ari"])
