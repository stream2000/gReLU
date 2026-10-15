import pytest

from grelu.interpret.ism.model_adapters.base import TrackSpec
from grelu.interpret.ism.model_adapters.utils import (
    sequences_to_tensor,
    slug_identifier,
    track_indices,
)


def test_sequences_to_tensor_is_channel_first_and_maps_unknown_to_n():
    encoded = sequences_to_tensor(["ACGTN"], expected_length=5)
    assert tuple(encoded.shape) == (1, 4, 5)
    assert encoded[0, :, :4].sum().item() == 4
    assert encoded[0, :, 4].sum().item() == 0


def test_sequences_to_tensor_rejects_wrong_length():
    with pytest.raises(ValueError, match="Expected 5 bp"):
        sequences_to_tensor(["ACGT"], expected_length=5)


def test_track_indices_preserve_requested_order():
    specs = [
        TrackSpec("hsc", 0, "HSC", "hsc", "rna", 128, "test"),
        TrackSpec("mac", 1, "MAC", "mac", "rna", 128, "test"),
    ]
    assert track_indices(specs, ["mac", "hsc"]) == [1, 0]
    with pytest.raises(ValueError, match="Unknown tracks"):
        track_indices(specs, ["chol"])


def test_slug_identifier_is_stable_for_track_labels():
    assert slug_identifier("H3K27ac / Liver (adult)") == "h3k27ac_liver_adult"
