import pandas as pd
import pytest

from grelu.interpret.ism.mutations import (
    SaturationWindow,
    apply_equal_length_edit,
    enumerate_snv_site_table,
)
from grelu.interpret.ism.readouts import map_readouts_to_bins, summarize_profile_pair


class FakeFasta:
    def __init__(self, chroms):
        self.chroms = chroms

    def extract(self, chrom, start, end):
        return self.chroms[chrom][start:end]


def test_enumerate_snv_site_table_counts_three_non_ref_alts_per_base():
    fasta = FakeFasta({"chr1": "AACGTN"})
    windows = [
        SaturationWindow(
            window_id="demo",
            chrom="chr1",
            start=1,
            end=6,
            anchor=3,
            gene="GeneA",
        )
    ]

    table = enumerate_snv_site_table(fasta=fasta, windows=windows)

    assert len(table) == 12
    assert table["mutation_id"].is_unique
    assert set(table["ref_base"]) == {"A", "C", "G", "T"}
    assert not (table["ref_base"] == table["alt_base"]).any()


def test_apply_equal_length_edit_validates_reference_base():
    row = pd.Series(
        {
            "mutation_id": "m1",
            "edit_start": 2,
            "edit_end": 3,
            "ref_sequence": "C",
            "alt_sequence": "T",
        }
    )
    assert apply_equal_length_edit("AACG", 0, row) == "AATG"

    bad = row.copy()
    bad["ref_sequence"] = "A"
    with pytest.raises(ValueError, match="REF mismatch"):
        apply_equal_length_edit("AACG", 0, bad)


def test_map_readouts_to_bins_and_summarize_profile_pair():
    readouts = pd.DataFrame(
        [
            {
                "readout_id": "center",
                "chrom": "chr1",
                "start": 128,
                "end": 384,
            }
        ]
    )

    mapping = map_readouts_to_bins(
        readouts,
        output_start=0,
        n_bins=4,
        bin_size=128,
        chrom="chr1",
    )
    assert mapping == {"center": (1, 3)}

    summary = summarize_profile_pair([1.0, 3.0], [2.0, 1.0])
    assert summary["n_bins"] == 2
    assert summary["signed_delta_mean"] == pytest.approx(-0.5)
    assert summary["absolute_delta_mean"] == pytest.approx(1.5)
