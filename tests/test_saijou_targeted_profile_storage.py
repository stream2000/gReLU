import numpy as np
import pandas as pd
import pytest

from grelu.interpret.ism.profiles import (
    MutationProfileWriter,
    aggregate_count_profiles,
    load_stored_mutation_profiles,
    reconstruct_alternate_profiles,
)


def test_aggregate_count_profiles_sums_adjacent_bins():
    profiles = np.arange(2 * 8, dtype=np.float32).reshape(2, 8)
    observed = aggregate_count_profiles(profiles, native_bp=32, target_bp=128)
    expected = profiles.reshape(2, 2, 4).sum(axis=-1)
    np.testing.assert_array_equal(observed, expected)


def test_aggregate_count_profiles_keeps_native_resolution():
    profiles = np.arange(8, dtype=np.float32).reshape(1, 8)
    observed = aggregate_count_profiles(profiles, native_bp=128, target_bp=128)
    assert observed is profiles


def test_aggregate_count_profiles_rejects_finer_resolution():
    with pytest.raises(ValueError, match="multiple of native resolution"):
        aggregate_count_profiles(np.zeros((1, 8)), native_bp=32, target_bp=16)


def test_aggregate_count_profiles_rejects_incompatible_bin_count():
    with pytest.raises(ValueError, match="not divisible by aggregation factor"):
        aggregate_count_profiles(np.zeros((1, 8)), native_bp=32, target_bp=96)


def test_mutation_profile_writer_round_trip(tmp_path):
    mutations = pd.DataFrame(
        [
            {
                "mutation_id": f"m{index}",
                "edit_start": index,
                "edit_end": index + 1,
                "edit_center_position": index,
                "variant_offset_from_tss_transcription_bp": index,
                "ref_sequence": "A",
                "alt_sequence": "C",
                "replacement_replicate": index,
            }
            for index in range(2)
        ]
    )
    reference = np.arange(16, dtype=np.float32).reshape(2, 8)
    alternate = np.stack([reference + 1, reference + 2])
    writer = MutationProfileWriter(
        tmp_path,
        gene="GeneA",
        mutations=mutations,
        track_order=["track_a", "track_b"],
        reference_profiles=reference,
        native_resolution_bp=32,
        stored_resolution_bp=128,
        dtype="float16",
        pseudocount=1.0,
        output_start=100,
    )
    writer.write_batch(0, alternate)
    writer.finalize()

    stored = load_stored_mutation_profiles(tmp_path, gene="GeneA", resolution_bp=128)
    expected_alt = aggregate_count_profiles(alternate, native_bp=32, target_bp=128)
    reconstructed = reconstruct_alternate_profiles(
        stored.reference[None, ...], stored.values, pseudocount=1.0
    )
    np.testing.assert_allclose(reconstructed, expected_alt, rtol=2e-3, atol=2e-2)
    assert stored.index.mutation_id.tolist() == ["m0", "m1"]
    assert stored.track_order == ["track_a", "track_b"]
    assert stored.metadata["shape"] == [2, 2, 2]
