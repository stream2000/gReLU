import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SAIJOU_DIR = REPO_ROOT / "scripts/ism/experiments/saijou_hsc"
if str(SAIJOU_DIR) not in sys.path:
    sys.path.insert(0, str(SAIJOU_DIR))

from grelu.interpret.ism.mutations import AnchoredScan, ScanEdit, ScanExclusion  # noqa: E402
from tools.genomics import TRANSCRIPT_OVERRIDES  # noqa: E402
from tools.manifests import (  # noqa: E402
    ManifestLocus,
    interval_mutation_id,
    readout_row,
    scan_exclusion_rows,
    scan_manifest_rows,
    scan_mutation_id,
    standard_readout_row,
    strict_shuffle_mutation_row,
)


MANIFEST_COLUMNS = [
    "mutation_id",
    "locus_id",
    "locus_role",
    "gene",
    "chrom",
    "gene_strand",
    "gene_tss",
    "gene_tes",
    "edit_start",
    "edit_end",
    "edit_center_position",
    "edit_length_bp",
    "ref_sequence",
    "alt_sequence",
    "mutation_kind",
    "replacement_mode",
    "replacement_replicate",
    "variant_position",
    "variant_offset_from_tss_genomic_bp",
    "variant_offset_from_tss_transcription_bp",
    "ref_base",
    "alt_base",
    "control_type",
    "source",
]


def _locus(strand="-"):
    return ManifestLocus(
        scan=AnchoredScan(locus_id="mdk_demo", chrom="chr2", anchor=91932297, strand=strand),
        locus_role="tss_10bp_strict_scan",
        gene="Mdk",
        tes=91929804,
        control_type="standard_scan",
        source="demo",
    )


def _edit(tx_offset=-507):
    scan = _locus().scan
    center = scan.genomic_center(tx_offset)
    return ScanEdit(
        tx_offset=tx_offset,
        genomic_center=center,
        edit_start=center - 5,
        edit_end=center + 5,
        ref_sequence="CAACAAGAAA",
        alt_sequences=("AAACGCAAAA", "AACAAAGACA"),
    )


def test_manifest_column_order_is_the_prepared_artifact_contract():
    rows = scan_manifest_rows(_locus(), [_edit()])

    assert list(rows[0]) == MANIFEST_COLUMNS


def test_scan_manifest_rows_emit_one_row_per_replicate():
    edit = _edit()
    rows = scan_manifest_rows(_locus(), [edit])

    assert len(rows) == 2
    assert [row["replacement_replicate"] for row in rows] == [0, 1]
    assert [row["alt_sequence"] for row in rows] == list(edit.alt_sequences)
    assert all(row["ref_base"] == row["ref_sequence"] for row in rows)
    assert all(row["alt_base"] == row["alt_sequence"] for row in rows)
    assert all(row["edit_length_bp"] == 10 for row in rows)


def test_manifest_rows_record_both_offsets_which_differ_on_the_minus_strand():
    row = scan_manifest_rows(_locus(strand="-"), [_edit(tx_offset=-507)])[0]

    # Mdk is minus strand: upstream in transcription order is a larger coordinate.
    assert row["variant_offset_from_tss_transcription_bp"] == -507
    assert row["variant_offset_from_tss_genomic_bp"] == 507
    assert row["variant_position"] == row["edit_center_position"]


def test_mutation_id_shapes():
    locus = _locus()

    assert (
        scan_mutation_id(locus, tx_offset=-507, edit_start=91932799, edit_end=91932809, replicate=0)
        == "mdk_demo__tx-507__91932799_91932809__shuffle_rep00"
    )
    assert (
        interval_mutation_id(locus, edit_start=91932799, edit_end=91932809, replicate=11)
        == "mdk_demo__91932799_91932809__shuffle_rep11"
    )


def test_strict_shuffle_mutation_row_derives_centre_and_offset_from_the_interval():
    locus = _locus()
    row = strict_shuffle_mutation_row(
        locus,
        mutation_id="m1",
        edit_start=91932799,
        edit_end=91932809,
        ref_sequence="CAACAAGAAA",
        alt_sequence="AAACGCAAAA",
        replicate=0,
    )

    assert row["edit_center_position"] == 91932804
    assert row["variant_offset_from_tss_transcription_bp"] == locus.scan.tx_offset(91932804)
    assert row["mutation_kind"] == "strict_mononucleotide_shuffle"
    assert row["replacement_mode"] == "strict_shuffle"


def test_readout_row_column_order_is_the_prepared_artifact_contract():
    row = readout_row(
        readout_id="Mdk__tss_1024bp",
        gene="Mdk",
        chrom="chr2",
        start=91931785,
        end=91932809,
        anchor=91932297,
        role="tss",
    )

    assert list(row) == [
        "readout_id",
        "gene",
        "locus_id",
        "chrom",
        "start",
        "end",
        "anchor",
        "role",
    ]
    assert row["locus_id"] == "*"


def test_standard_readout_row_centres_on_its_anchor():
    row = standard_readout_row(
        gene="Mdk", chrom="chr2", role="tss", center=91932297, width_bp=1024
    )

    assert row["readout_id"] == "Mdk__tss_1024bp"
    assert (row["start"], row["end"]) == (91931785, 91932809)
    assert row["anchor"] == 91932297
    assert row["end"] - row["start"] == 1024


def test_acta2_transcript_override_has_a_single_definition():
    # The preparers scan around the TSS this picks and the analysis annotates
    # against the transcript this picks; a second copy would let them diverge.
    assert TRANSCRIPT_OVERRIDES == {"Acta2": "ENSMUST00000238147"}

    preparer = (SAIJOU_DIR / "prepare_saijou_all_genes_10bp_scan.py").read_text()
    assert "ENSMUST" not in preparer


@pytest.mark.parametrize(
    "gene, expected",
    [
        ("Mdk", ["gene", "variant_offset_from_tss_transcription_bp", "edit_start", "edit_end", "ref_sequence", "reason"]),
        ("", ["variant_offset_from_tss_transcription_bp", "edit_start", "edit_end", "ref_sequence", "reason"]),
    ],
)
def test_scan_exclusion_rows_omit_the_gene_column_for_single_gene_tables(gene, expected):
    exclusion = ScanExclusion(
        tx_offset=-223,
        genomic_center=91932520,
        edit_start=91932515,
        edit_end=91932525,
        ref_sequence="CCCCCCCCCC",
        reason="Cannot create a different composition-preserving shuffle",
    )

    rows = scan_exclusion_rows([exclusion], gene=gene)

    assert list(rows[0]) == expected
