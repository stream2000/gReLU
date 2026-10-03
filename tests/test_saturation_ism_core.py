import pandas as pd
import pytest

from grelu.interpret.ism.mutations import (
    AnchoredScan,
    SaturationWindow,
    anchored_scan_centers,
    apply_equal_length_edit,
    enumerate_sliding_window_replacement_table,
    enumerate_snv_site_table,
    scan_anchored_strict_shuffles,
    strict_unique_shuffles,
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


def test_strict_unique_shuffles_are_deterministic_and_composition_preserving():
    first = strict_unique_shuffles(
        "AACCGGTT", n=3, seed=23, mutation_key="candidate"
    )
    repeat = strict_unique_shuffles(
        "AACCGGTT", n=3, seed=23, mutation_key="candidate"
    )
    assert first == repeat
    assert len(set(first)) == 3
    assert "AACCGGTT" not in first
    assert all(sorted(value) == sorted("AACCGGTT") for value in first)

    with pytest.raises(ValueError, match="composition-preserving"):
        strict_unique_shuffles("AAAA", n=1, seed=23, mutation_key="homopolymer")


def _scan_fasta():
    return FakeFasta({"chr1": "ACGTACGTACGTACGT" + "AAAA" + "CGTA" + "ACGTACGTACGTACGT"})


def test_anchored_scan_centers_inset_by_half_a_span():
    centers = anchored_scan_centers(half_window_bp=512, span_bp=10, stride_bp=2)

    assert len(centers) == 508
    assert (centers[0], centers[-1]) == (-507, 507)
    assert centers[0] - 10 // 2 == -512
    assert centers[-1] + 10 // 2 == 512

    with pytest.raises(ValueError, match="even"):
        anchored_scan_centers(half_window_bp=512, span_bp=9, stride_bp=2)
    with pytest.raises(ValueError, match="exceeds"):
        anchored_scan_centers(half_window_bp=4, span_bp=10, stride_bp=2)


def test_anchored_scan_maps_transcription_offsets_by_strand():
    plus = AnchoredScan(locus_id="demo", chrom="chr1", anchor=20, strand="+")
    minus = AnchoredScan(locus_id="demo", chrom="chr1", anchor=20, strand="-")

    assert plus.genomic_center(3) == 23
    assert minus.genomic_center(3) == 17

    # tx_offset inverts genomic_center on both strands.
    assert plus.tx_offset(23) == 3
    assert minus.tx_offset(17) == 3
    for scan in (plus, minus):
        assert all(scan.tx_offset(scan.genomic_center(k)) == k for k in range(-5, 6))

    with pytest.raises(ValueError, match="Strand"):
        AnchoredScan(locus_id="demo", chrom="chr1", anchor=20, strand="?")


def test_scan_anchored_strict_shuffles_excludes_homopolymers_and_is_deterministic():
    scan = AnchoredScan(locus_id="demo", chrom="chr1", anchor=20, strand="+")
    kwargs = dict(centers=[-2, 0, 2], span_bp=4, replicates=2, seed=23)

    edits, exclusions = scan_anchored_strict_shuffles(fasta=_scan_fasta(), scan=scan, **kwargs)
    repeat, _ = scan_anchored_strict_shuffles(fasta=_scan_fasta(), scan=scan, **kwargs)

    assert [edit.alt_sequences for edit in edits] == [edit.alt_sequences for edit in repeat]
    assert [edit.tx_offset for edit in edits] == [0, 2]
    assert all(len(set(edit.alt_sequences)) == 2 for edit in edits)
    assert all(edit.ref_sequence not in edit.alt_sequences for edit in edits)
    assert all(
        sorted(alt) == sorted(edit.ref_sequence)
        for edit in edits
        for alt in edit.alt_sequences
    )

    assert [exclusion.tx_offset for exclusion in exclusions] == [-2]
    excluded = exclusions[0]
    assert excluded.ref_sequence == "AAAA"
    assert (excluded.edit_start, excluded.edit_end) == (16, 20)
    # The mutation key seeds the replacement RNG and is quoted verbatim in the
    # exclusion reason that prepared artifacts record, so its shape is a contract.
    assert "demo:-2:16:20" in excluded.reason


def test_scan_anchored_strict_shuffles_mirrors_windows_across_strands():
    centers = [-2, 0, 2]
    geometry = dict(centers=centers, span_bp=4, replicates=1, seed=23)

    plus, plus_excluded = scan_anchored_strict_shuffles(
        fasta=_scan_fasta(),
        scan=AnchoredScan(locus_id="demo", chrom="chr1", anchor=20, strand="+"),
        **geometry,
    )
    minus, minus_excluded = scan_anchored_strict_shuffles(
        fasta=_scan_fasta(),
        scan=AnchoredScan(locus_id="demo", chrom="chr1", anchor=20, strand="-"),
        **geometry,
    )

    windows = {(edit.edit_start, edit.edit_end) for edit in plus}
    assert windows == {(edit.edit_start, edit.edit_end) for edit in minus}
    assert [edit.tx_offset for edit in plus] == [0, 2]
    assert [edit.tx_offset for edit in minus] == [-2, 0]
    assert plus_excluded[0].tx_offset == -2
    assert minus_excluded[0].tx_offset == 2


def test_enumerate_sliding_window_replacements_are_deterministic_equal_length_edits():
    fasta = FakeFasta({"chr1": "AACGTN"})
    windows = [
        SaturationWindow(
            window_id="demo",
            chrom="chr1",
            start=0,
            end=5,
            anchor=2,
            gene="GeneA",
        )
    ]

    table = enumerate_sliding_window_replacement_table(
        fasta=fasta,
        windows=windows,
        span_bp=3,
        stride_bp=1,
        mode="random",
        replicates=2,
        seed=7,
    )
    repeat = enumerate_sliding_window_replacement_table(
        fasta=fasta,
        windows=windows,
        span_bp=3,
        stride_bp=1,
        mode="random",
        replicates=2,
        seed=7,
    )

    assert len(table) == 6
    assert table["mutation_id"].is_unique
    assert table["alt_sequence"].tolist() == repeat["alt_sequence"].tolist()
    assert (table["edit_end"] - table["edit_start"]).eq(3).all()
    assert table["ref_sequence"].str.len().eq(table["alt_sequence"].str.len()).all()
    assert not (table["ref_sequence"] == table["alt_sequence"]).any()

    edited = apply_equal_length_edit("AACGTN", 0, table.iloc[0])
    assert len(edited) == len("AACGTN")
    assert edited[table.iloc[0]["edit_start"] : table.iloc[0]["edit_end"]] == table.iloc[0][
        "alt_sequence"
    ]


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
    assert summary["signed_delta_sum"] == pytest.approx(-1.0)
    assert summary["ref_sum"] == pytest.approx(4.0)
    assert summary["alt_sum"] == pytest.approx(3.0)
    assert summary["log2fc_ratio_of_sums"] == pytest.approx(
        __import__("math").log2((3.0 + 2.0) / (4.0 + 2.0))
    )
    assert summary["absolute_delta_mean"] == pytest.approx(1.5)
