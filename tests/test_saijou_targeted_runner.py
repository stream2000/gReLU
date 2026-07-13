from types import SimpleNamespace

import numpy as np
import pandas as pd

from grelu.interpret.ism.model_adapters.base import TrackSpec
from grelu.interpret.ism.runner import (
    TargetedRunConfig,
    combine_run_artifacts,
    run_gene,
)


class FakeFasta:
    sequence = "A" * 64

    def chrom_length(self, chrom):
        return len(self.sequence)

    def extract(self, chrom, start, end):
        return self.sequence[start:end]


class FakeAdapter:
    model_id = "fake"
    input_length_bp = 16
    output_resolution_bp = 4
    output_length_bins = 4
    track_specs = [
        TrackSpec("hsc", 0, "HSC", "hsc", "rna", 4, "test"),
        TrackSpec("mac", 1, "MAC", "mac", "rna", 4, "test"),
    ]

    def predict_profiles(self, sequences, tracks=None):
        rows = []
        for sequence in sequences:
            bins = np.array(
                [sequence[i : i + 4].count("C") for i in range(0, 16, 4)],
                dtype=np.float32,
            )
            rows.append(np.stack([bins + 1, bins + 2]))
        return np.stack(rows)


def mutation_manifest():
    rows = []
    for index, position in enumerate([14, 18]):
        rows.append(
            {
                "gene": "GeneA",
                "locus_id": "locus",
                "locus_role": "test",
                "mutation_id": f"m{index}",
                "mutation_kind": "snv",
                "edit_start": position,
                "edit_end": position + 1,
                "edit_center_position": position,
                "variant_offset_from_tss_transcription_bp": position - 20,
                "ref_sequence": "A",
                "alt_sequence": "C",
                "replacement_replicate": index,
            }
        )
    return pd.DataFrame(rows)


def test_targeted_runner_gene_and_combination(tmp_path):
    config = TargetedRunConfig(
        model_backend="fake_backend",
        profile_resolution_bp=8,
        resume=False,
        save_log2fc_profiles=True,
        save_ref_profiles=True,
        profile_dtype="float16",
        pseudocount=1.0,
        batch_size=1,
        progress_every=0,
    )
    feature_dir = tmp_path / "features"
    profile_dir = tmp_path / "profiles"
    feature_dir.mkdir()
    result = run_gene(
        config=config,
        adapter=FakeAdapter(),
        fasta=FakeFasta(),
        gene_row=SimpleNamespace(gene="GeneA", chrom="chr1", analysis_tss=20),
        manifest=mutation_manifest(),
        readouts=pd.DataFrame(
            [
                {
                    "gene": "GeneA",
                    "readout_id": "readout",
                    "locus_id": "locus",
                    "role": "tss_1024bp",
                    "chrom": "chr1",
                    "start": 12,
                    "end": 28,
                }
            ]
        ),
        requested_tracks=["hsc", "mac"],
        track_specs={spec.track_id: spec for spec in FakeAdapter.track_specs},
        feature_dir=feature_dir,
        profile_dir=profile_dir,
        output_span=16,
        crop_bp=0,
    )
    features = pd.read_csv(result.feature_path, sep="\t")
    assert len(features) == 4
    assert result.deterministic_diff == 0

    diagnostics, nonfinite, profile_bytes = combine_run_artifacts(
        config=config,
        genes=pd.DataFrame([{"gene": "GeneA"}]),
        readouts=pd.DataFrame([{"gene": "GeneA"}]),
        manifest=mutation_manifest(),
        results=[result],
        n_tracks=2,
        feature_dir=feature_dir,
        profile_dir=profile_dir,
        out_dir=tmp_path,
    )
    assert diagnostics.is_valid
    assert nonfinite == 0
    assert profile_bytes > 0
