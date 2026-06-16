import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_script(name):
    path = REPO_ROOT / "scripts" / "ism" / "examples" / "mreg" / name
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


runner = _load_script("run_mreg_three_region_chromatin.py")
analyzer = _load_script("analyze_mreg_three_region_chromatin.py")
preparer = _load_script("prepare_mreg_three_region_pilot.py")


def test_track_resolution_prefers_exact_mcf7_over_mcf10a():
    metadata = pd.DataFrame(
        [
            {
                "output_type": "chip_tf",
                "transcription_factor": "CTCF",
                "track_index": 0,
                "track_name": "TF ChIP-seq CTCF MCF 10A",
                "biosample_name": "MCF 10A",
            },
            {
                "output_type": "chip_tf",
                "transcription_factor": "CTCF",
                "track_index": 1,
                "track_name": "TF ChIP-seq CTCF MCF-7",
                "biosample_name": "MCF-7",
            },
        ]
    )
    registry = pd.DataFrame(
        [
            {
                "track_id": "ctcf",
                "output_type": "chip_tf",
                "target_name": "CTCF",
                "biosample": "MCF-7",
                "selection_priority": 1,
            }
        ]
    )

    resolved, _ = runner._resolve_tracks(
        metadata,
        registry,
        output_keys=("chip_tf",),
    )

    assert resolved.iloc[0]["track_index"] == 1
    assert resolved.iloc[0]["biosample_resolved"] == "MCF-7"


def test_readout_bins_reject_partial_context_overlap():
    readouts = pd.DataFrame(
        [{"readout_id": "mreg_tss_4kb", "start": 900, "end": 1200}]
    )

    with pytest.raises(ValueError, match="not fully contained"):
        runner._compute_readout_bins(readouts, seq_start=1000, n_bins=10)


def test_profile_extraction_uses_each_mutations_coordinate_frame():
    mutations = pd.DataFrame(
        [
            {
                "mutation_id": "mut_a",
                "mutation_strategy": "disruption",
                "edit_start": 1499,
                "edit_end": 1501,
            },
            {
                "mutation_id": "mut_b",
                "mutation_strategy": "perturbation",
                "edit_start": 2499,
                "edit_end": 2501,
            },
        ]
    )
    tracks = pd.DataFrame(
        [
            {
                "track_id": "ctcf",
                "track_index": 0,
                "target_name": "CTCF",
                "biosample_resolved": "MCF-7",
            }
        ]
    )
    readouts = pd.DataFrame(
        [
            {
                "readout_id": "mreg_tss_4kb",
                "start": 2100,
                "end": 2228,
                "anchor_coordinate": 2164,
            },
            {
                "readout_id": "mutation_center_1kb",
                "start": -1,
                "end": -1,
                "anchor_coordinate": -1,
            },
            {
                "readout_id": "mutation_local_4kb",
                "start": -1,
                "end": -1,
                "anchor_coordinate": -1,
            },
        ]
    )
    ref = np.zeros((2, 1, 32), dtype=np.float32)
    alt = np.ones((2, 1, 32), dtype=np.float32)

    profiles = runner._extract_profiles(
        ref,
        alt,
        mutations,
        tracks,
        readouts,
        {"mut_a": 1000, "mut_b": 2000},
    )

    starts = (
        profiles[profiles["readout_id"] == "mreg_tss_4kb"]
        .groupby("mutation_id")["genomic_start"]
        .min()
        .to_dict()
    )
    assert starts == {"mut_a": 2024, "mut_b": 2000}


def test_controls_share_experimental_context_center():
    mutations = pd.DataFrame(
        [
            {
                "mutation_id": "exp",
                "region_id": "site",
                "mutation_strategy": "strand_aware_ctcf_pwm_disruption",
                "edit_start": 100,
                "edit_end": 112,
            },
            {
                "mutation_id": "ctrl",
                "region_id": "site",
                "mutation_strategy": "local_non_motif_control",
                "edit_start": 124,
                "edit_end": 136,
            },
        ]
    )

    centers = runner._shared_context_centers(mutations)

    assert centers == {"exp": 106, "ctrl": 106}


def test_response_matrix_does_not_average_nested_tss_windows():
    adjusted = pd.DataFrame(
        [
            {
                "mutation_id": "mreg_tss_00_ctcf_snv",
                "readout_id": "mreg_tss_4kb",
                "target_name": "CTCF",
                "adjusted_signed_delta_mean": 1.0,
            },
            {
                "mutation_id": "mreg_tss_00_ctcf_snv",
                "readout_id": "mreg_tss_20kb",
                "target_name": "CTCF",
                "adjusted_signed_delta_mean": 100.0,
            },
            {
                "mutation_id": "mreg_tss_00_ctcf_snv",
                "readout_id": "mutation_center_1kb",
                "target_name": "CTCF",
                "adjusted_signed_delta_mean": 200.0,
            },
        ]
    )

    matrix = analyzer.build_response_matrix(
        adjusted,
        "CTCF",
        "adjusted_signed_delta_mean",
    )

    assert matrix.loc["tss", "tss"] == 1.0


def test_legacy_run_keeps_only_first_mutation(tmp_path):
    profiles = pd.DataFrame(
        {
            "mutation_id": ["tss_mut", "dic_mut"],
            "ref_value": [1.0, 1.0],
        }
    )
    pd.DataFrame(
        {
            "mutation_id": ["tss_mut", "dic_mut"],
            "seq_start": [1000, 2000],
        }
    ).to_csv(tmp_path / "scored_intervals.tsv", sep="\t", index=False)

    valid, status = analyzer.assess_coordinate_frame(
        profiles,
        tmp_path,
        run_config={},
    )

    assert valid["mutation_id"].tolist() == ["tss_mut"]
    assert status["status"] == "partial_tss_positive_control_only"
    assert status["cross_region_valid"] is False


class _AllAFasta:
    def chrom_length(self, chrom):
        return 1000

    def extract(self, chrom, start, end):
        return "A" * (end - start)


def test_non_peak_control_is_outside_target_peak():
    control = preparer._design_local_sequence_control(
        fasta=_AllAFasta(),
        chrom="chr1",
        target_edit_start=100,
        target_edit_end=103,
        target_ref="AAA",
        target_alt="GGG",
        target_gc_delta=3,
        target_edit_length=3,
        peak_start=90,
        peak_end=110,
        max_distance_bp=30,
        pwm=None,
    )

    assert control is not None
    assert control["edit_end"] <= 90 or control["edit_start"] >= 110


def test_curated_pol2_peak_coordinates_are_preserved_in_region_manifest(
    tmp_path, monkeypatch
):
    class _Fasta:
        def extract(self, chrom, start, end):
            return "A" * (end - start)

        def close(self):
            pass

    monkeypatch.setattr(preparer, "FastaReference", lambda path: _Fasta())
    monkeypatch.setattr(preparer, "load_meme_pwm", lambda *args, **kwargs: object())
    monkeypatch.setattr(
        preparer,
        "_parse_gtf_gene_region",
        lambda *args: ("chr2", 100, 1000, "-"),
    )
    monkeypatch.setattr(
        preparer,
        "_parse_gtf_transcripts",
        lambda *args: pd.DataFrame(
            [{"transcript_id": "tx", "chrom": "chr2", "tss": 900, "strand": "-"}]
        ),
    )
    monkeypatch.setattr(preparer, "_scan_region_for_ctcf", lambda *args, **kwargs: [])

    curated = pd.DataFrame(
        [
            {
                "region_role": "tss",
                "chrom": "chr2",
                "start": 850,
                "end": 950,
                "summit": 900,
                "strand": "-",
                "review_status": "confirmed",
            },
            {
                "region_role": "lc_dic",
                "chrom": "chr2",
                "start": 400,
                "end": 500,
                "summit": 450,
                "strand": ".",
                "review_status": "needs_review",
                "pol2_peak_start": 410,
                "pol2_peak_end": 490,
                "pol2_peak_summit": 470,
            },
        ]
    )

    preparer.prepare_manifests(
        curated_regions=curated,
        fasta_path=tmp_path / "fake.fa",
        gtf_path=tmp_path / "fake.gtf",
        meme_path=tmp_path / "fake.meme",
        output_dir=tmp_path / "out",
    )

    regions = pd.read_csv(
        tmp_path / "out" / "mreg_region_manifest.tsv", sep="\t"
    )
    lc = regions.loc[regions["region_role"] == "lc_dic"].iloc[0]
    assert lc["pol2_peak_start"] == 410
    assert lc["pol2_peak_end"] == 490
    assert lc["pol2_peak_summit"] == 470
