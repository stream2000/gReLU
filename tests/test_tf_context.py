import numpy as np
import pandas as pd
import pytest

from grelu.interpret.tf_context import (
    aggregate_delta_features,
    annotate_context_clusters,
    build_output_track_table,
    build_track_groups,
    cluster_contexts,
    contact_motif_vs_control_delta,
    contact_perturbations_vs_control_delta,
    cross_boundary_track_means,
    fit_context_embedding,
    filter_sites_for_context_window,
    load_tf_sites,
    make_motif_disruption_variants,
    normalize_contact_maps,
    score_variant_features_chunked,
    score_variant_contact_boundary_strength,
    score_variant_track_window,
    score_variants_chunked,
    summarize_contact_boundary_strength,
)
from scripts.ism.summarize_center_track_effects import (
    center_mask,
    expand_center_bp_signals,
    summarize_center_track_effects,
)


def test_load_tf_sites_headerless_bed(tmp_path):
    bed = tmp_path / "sites.bed"
    bed.write_text("chr1\t10\t30\tpeak1\t42\t+\t12\t27\tACGTACGTACGTACG\n")

    sites = load_tf_sites(bed)

    assert list(sites[["chrom", "start", "end", "site_id"]].iloc[0]) == [
        "chr1",
        10,
        30,
        "peak1",
    ]
    assert sites.iloc[0]["motif_start"] == 12
    assert sites.iloc[0]["motif_end"] == 27


def test_make_motif_disruption_variants_uses_matched_seq_center():
    sites = pd.DataFrame(
        {
            "chrom": ["chr1"],
            "start": [100],
            "end": [120],
            "motif_start": [104],
            "motif_end": [114],
            "matched_seq": ["NNNNACGTNN"],
            "site_id": ["s1"],
            "strand": ["+"],
        }
    )

    variants = make_motif_disruption_variants(sites)

    assert variants.loc[0, "position"] == 110
    assert variants.loc[0, "ref"] == "C"
    assert variants.loc[0, "alt"] == "A"


def test_make_motif_disruption_variants_honors_explicit_variant():
    sites = pd.DataFrame(
        {
            "chrom": ["chr1"],
            "start": [100],
            "end": [120],
            "motif_start": [104],
            "motif_end": [114],
            "matched_seq": ["NNNNACGTNN"],
            "site_id": ["s1"],
            "strand": ["+"],
            "variant_position": [107],
            "variant_ref": ["G"],
            "variant_alt": ["T"],
        }
    )

    variants = make_motif_disruption_variants(sites)

    assert variants.loc[0, "position"] == 107
    assert variants.loc[0, "ref"] == "G"
    assert variants.loc[0, "alt"] == "T"
    assert variants.loc[0, "variant_offset"] == 2


def test_make_motif_disruption_variants_honors_motif_replacement():
    sites = pd.DataFrame(
        {
            "chrom": ["chr1"],
            "start": [100],
            "end": [120],
            "motif_start": [104],
            "motif_end": [114],
            "matched_seq": ["AAAACCCCGG"],
            "site_id": ["s1"],
            "strand": ["+"],
            "variant_start": [104],
            "variant_end": [114],
            "variant_ref_seq": ["AAAACCCCGG"],
            "variant_alt_seq": ["TTTTGGGGAA"],
        }
    )

    variants = make_motif_disruption_variants(sites)

    assert variants.loc[0, "position"] == 110
    assert variants.loc[0, "ref"] == "AAAACCCCGG"
    assert variants.loc[0, "alt"] == "TTTTGGGGAA"
    assert variants.loc[0, "variant_start"] == 104
    assert variants.loc[0, "variant_end"] == 114


def test_build_track_groups_global_indices():
    metadata = pd.DataFrame(
        {
            "output_type": ["chip_tf", "chip_tf", "chip_tf", "chip_histone", "atac", "rna_seq"],
            "track_index": [0, 1, 2, 0, 0, 0],
            "transcription_factor": ["CTCF", "RAD21", "POLR2A", None, None, None],
            "histone_mark": [None, None, None, "H3K27ac", None, None],
            "assay_title": ["ChIP-seq", "ChIP-seq", "ChIP-seq", "ChIP-seq", "ATAC-seq", "RNA-seq"],
        }
    )

    groups = build_track_groups(metadata, output_keys=("chip_tf", "chip_histone", "atac", "rna_seq"))

    assert groups["ctcf_binding"] == [0]
    assert groups["cohesin_binding"] == [1]
    assert groups["transcription_machinery"] == [2]
    assert 3 in groups["activity_marks"]
    assert 4 in groups["activity_marks"]
    assert 5 in groups["global_expression_outputs"]


def test_build_output_track_table_global_indices():
    metadata = pd.DataFrame(
        {
            "output_type": ["chip_tf", "chip_tf", "rna_seq"],
            "track_index": [0, 1, 0],
            "name": ["ctcf", "rad21", "rna"],
        }
    )

    tracks = build_output_track_table(metadata, output_keys=("chip_tf", "rna_seq"))

    assert tracks["global_track_index"].tolist() == [0, 1, 2]
    assert tracks["name"].tolist() == ["ctcf", "rad21", "rna"]


class ToyFasta:
    def extract(self, chrom, start, end):
        seq = ("ACGT" * ((end - start) // 4 + 1))[: end - start]
        return seq

    def chrom_length(self, chrom):
        return 100


class ToyModel:
    def predict_on_seqs(self, seqs, device="cpu"):
        out = np.zeros((len(seqs), 3, 5), dtype=float)
        for i, seq in enumerate(seqs):
            out[i, :, :] = seq.count("A")
        return out


class ToyDatasetModel:
    def __init__(self):
        self.calls = []

    def predict_on_dataset(self, dataset, devices="cpu", batch_size=1, num_workers=1, precision=None):
        self.calls.append(
            {
                "devices": devices,
                "batch_size": batch_size,
                "num_workers": num_workers,
                "precision": precision,
                "n": dataset.n_seqs,
            }
        )
        out = np.zeros((dataset.n_seqs, 3, 5), dtype=float)
        for i, seq in enumerate(dataset.seqs):
            out[i, :, :] = seq.count("A")
        return out


class ToyContactModel:
    def __init__(self):
        self.calls = []

    def predict_on_dataset(self, dataset, devices="cpu", batch_size=1, num_workers=1, precision=None):
        self.calls.append({"devices": devices, "n": dataset.n_seqs})
        out = np.zeros((dataset.n_seqs, 2, 8, 8), dtype=np.float32)
        for i, seq in enumerate(dataset.seqs):
            out[i, :, :, :] = seq.count("A")
        return out


def test_score_variants_chunked_scores_ref_and_alt():
    variants = pd.DataFrame(
        {
            "site_id": ["s1"],
            "chrom": ["chr1"],
            "position": [5],
            "ref": ["A"],
            "alt": ["C"],
            "site_start": [0],
            "site_end": [10],
        }
    )

    ref, alt, intervals = score_variants_chunked(
        variants,
        model=ToyModel(),
        fasta_extractor=ToyFasta(),
        input_len=8,
    )

    assert ref.shape == (1, 3, 5)
    assert alt.shape == (1, 3, 5)
    assert alt[0, 0, 0] == ref[0, 0, 0] - 1
    assert intervals.loc[0, "variant_offset"] == 4


def test_score_variants_chunked_uses_dataset_multigpu_path():
    variants = pd.DataFrame(
        {
            "site_id": ["s1", "s2"],
            "chrom": ["chr1", "chr1"],
            "position": [5, 9],
            "ref": ["A", "A"],
            "alt": ["C", "C"],
            "site_start": [0, 0],
            "site_end": [10, 10],
        }
    )
    model = ToyDatasetModel()

    score_variants_chunked(
        variants,
        model=model,
        fasta_extractor=ToyFasta(),
        devices="0,1",
        chunk_size=1,
        batch_size=2,
        num_workers=3,
        precision="bf16-mixed",
        input_len=8,
    )

    assert model.calls[0]["devices"] == [0, 1]
    assert model.calls[0]["batch_size"] == 2
    assert model.calls[0]["num_workers"] == 3
    assert model.calls[0]["precision"] == "bf16-mixed"


def test_score_variants_chunked_merges_small_distributed_tail():
    variants = pd.DataFrame(
        {
            "site_id": ["s1", "s2", "s3"],
            "chrom": ["chr1", "chr1", "chr1"],
            "position": [5, 9, 13],
            "ref": ["A", "A", "A"],
            "alt": ["C", "C", "C"],
            "site_start": [0, 0, 0],
            "site_end": [20, 20, 20],
        }
    )
    model = ToyDatasetModel()

    score_variants_chunked(
        variants,
        model=model,
        fasta_extractor=ToyFasta(),
        devices="0,1",
        chunk_size=2,
        input_len=8,
    )

    assert model.calls[0]["n"] == 3


def test_score_variant_features_chunked_aggregates_without_returning_predictions():
    variants = pd.DataFrame(
        {
            "site_id": ["s1", "s2", "s3"],
            "chrom": ["chr1", "chr1", "chr1"],
            "position": [5, 9, 13],
            "ref": ["A", "A", "A"],
            "alt": ["C", "C", "C"],
            "site_start": [0, 0, 0],
            "site_end": [20, 20, 20],
        }
    )
    model = ToyDatasetModel()

    features, intervals = score_variant_features_chunked(
        variants,
        model=model,
        fasta_extractor=ToyFasta(),
        track_groups={"ctcf_binding": [0]},
        devices="0,1",
        input_len=8,
    )

    assert features.shape[0] == 3
    assert intervals.shape[0] == 3
    assert "ctcf_binding__local__delta_abs_mean" in features.columns
    assert len(model.calls) == 1
    assert model.calls[0]["n"] == 6  # 3 ref + 3 alt in one paired predict


def test_score_variant_track_window_writes_full_track_arrays(tmp_path):
    variants = pd.DataFrame(
        {
            "site_id": ["s1", "s2"],
            "chrom": ["chr1", "chr1"],
            "position": [5, 9],
            "ref": ["A", "A"],
            "alt": ["C", "C"],
            "site_start": [0, 0],
            "site_end": [20, 20],
        }
    )
    model = ToyDatasetModel()

    summary, intervals = score_variant_track_window(
        variants,
        model=model,
        fasta_extractor=ToyFasta(),
        output_dir=tmp_path,
        devices="0,1",
        input_len=8,
        bin_size=1,
        window_bp=1,
    )

    delta = np.load(tmp_path / "all_track_delta_window.npy")
    ref = np.load(tmp_path / "all_track_ref_window.npy")
    alt = np.load(tmp_path / "all_track_alt_window.npy")

    assert delta.shape == (2, 3, 3)
    assert ref.shape == alt.shape == delta.shape
    assert intervals.shape[0] == 2
    assert summary["window_bins"].tolist() == [3, 3]
    assert (tmp_path / "all_track_window_bins.tsv").exists()
    assert len(model.calls) == 1
    assert model.calls[0]["n"] == 4  # 2 ref + 2 alt in one paired predict


def test_score_variant_track_window_applies_multibase_replacement(tmp_path):
    variants = pd.DataFrame(
        {
            "site_id": ["s1", "s2"],
            "chrom": ["chr1", "chr1"],
            "position": [5, 9],
            "ref": ["AC", "AC"],
            "alt": ["TG", "TG"],
            "variant_start": [0, 4],
            "variant_end": [2, 6],
            "variant_ref_seq": ["AC", "AC"],
            "variant_alt_seq": ["TG", "TG"],
            "site_start": [0, 4],
            "site_end": [10, 14],
        }
    )
    model = ToyDatasetModel()

    _, intervals = score_variant_track_window(
        variants,
        model=model,
        fasta_extractor=ToyFasta(),
        output_dir=tmp_path,
        devices="0,1",
        input_len=8,
        bin_size=1,
        window_bp=1,
    )

    assert intervals.loc[0, "variant_ref_seq"] == "AC"
    assert intervals.loc[0, "variant_alt_seq"] == "TG"
    assert intervals.loc[0, "variant_end_offset"] - intervals.loc[0, "variant_start_offset"] == 2


def test_normalize_contact_maps_accepts_both_axis_orders():
    track_first = np.zeros((2, 3, 8, 8), dtype=np.float32)
    assert normalize_contact_maps(track_first).shape == (2, 3, 8, 8)

    track_last = np.zeros((2, 8, 8, 3), dtype=np.float32)
    assert normalize_contact_maps(track_last).shape == (2, 3, 8, 8)


def test_cross_boundary_track_means_uses_distance_band():
    contacts = np.zeros((2, 8, 8), dtype=np.float32)
    contacts[0] = np.arange(64, dtype=np.float32).reshape(8, 8)
    contacts[1] = 2 * contacts[0]

    means, n_pairs = cross_boundary_track_means(
        contacts,
        boundary_bin=4,
        window_bins=3,
        min_distance_bins=2,
    )

    left = np.arange(1, 4)
    right = np.arange(4, 7)
    distance = right[None, :] - left[:, None]
    mask = (distance >= 2) & (distance <= 3)
    expected = contacts[:, left[:, None], right[None, :]][:, mask].mean(axis=1)
    np.testing.assert_allclose(means, expected)
    assert n_pairs == int(mask.sum())


def test_summarize_contact_boundary_strength_reports_strength_proxy():
    ref = np.ones((1, 2, 8, 8), dtype=np.float32)
    alt = ref.copy()
    alt[:, :, 2:4, 4:6] += 3
    alt[:, :, 4:6, 2:4] += 3

    variants = pd.DataFrame(
        {
            "site_id": ["s1"],
            "paired_site_id": ["p1"],
            "dic_class": ["CTCF-dependent"],
            "control_type": ["ctcf_motif_max_ic_disruption"],
            "chrom": ["chr1"],
            "position": [5],
            "ref": ["A"],
            "alt": ["C"],
            "boundary_start": [4],
            "boundary_end": [4],
        }
    )
    intervals = pd.DataFrame({"site_id": ["s1"], "chrom": ["chr1"], "seq_start": [0], "seq_end": [8]})

    summary, tracks = summarize_contact_boundary_strength(
        ref,
        alt,
        variants,
        intervals,
        input_len=8,
        window_bp=2,
        min_distance_bp=1,
    )

    assert summary.loc[0, "delta_cross_contact_mean"] > 0
    assert summary.loc[0, "delta_boundary_strength_proxy"] < 0
    assert tracks.shape[0] == 2


def test_contact_motif_vs_control_delta_pairs_rows():
    site_summary = pd.DataFrame(
        {
            "site_id": ["motif", "control"],
            "paired_site_id": ["p1", "p1"],
            "dic_class": ["CTCF-dependent", "CTCF-dependent"],
            "control_type": ["ctcf_motif_max_ic_disruption", "same_peak_non_motif_nearby_base"],
            "delta_cross_contact_mean": [3.0, 1.0],
            "delta_boundary_strength_proxy": [-3.0, -1.0],
            "delta_abs_cross_contact_mean": [3.0, 1.0],
        }
    )

    paired = contact_motif_vs_control_delta(site_summary)

    assert paired.loc[0, "delta_cross_contact_mean__motif_minus_control"] == 2.0
    assert paired.loc[0, "delta_boundary_strength_proxy__motif_minus_control"] == -2.0


def test_contact_perturbations_vs_control_delta_includes_random_replacement():
    site_summary = pd.DataFrame(
        {
            "site_id": ["motif", "random", "control"],
            "paired_site_id": ["p1", "p1", "p1"],
            "dic_class": ["CTCF-dependent", "CTCF-dependent", "CTCF-dependent"],
            "control_type": [
                "ctcf_motif_max_ic_disruption",
                "whole_motif_random_replacement",
                "same_peak_non_motif_nearby_base",
            ],
            "delta_cross_contact_mean": [3.0, 5.0, 1.0],
            "delta_boundary_strength_proxy": [-3.0, -5.0, -1.0],
            "delta_abs_cross_contact_mean": [3.0, 5.0, 1.0],
        }
    )

    paired = contact_perturbations_vs_control_delta(site_summary)

    assert set(paired["perturbation_type"]) == {
        "ctcf_motif_max_ic_disruption",
        "whole_motif_random_replacement",
    }
    random_row = paired[paired["perturbation_type"] == "whole_motif_random_replacement"].iloc[0]
    assert random_row["delta_cross_contact_mean__perturbation_minus_control"] == 4.0


def test_score_variant_contact_boundary_strength_uses_toy_contact_model():
    variants = pd.DataFrame(
        {
            "site_id": ["s1", "s2", "s3", "s4"],
            "chrom": ["chr1", "chr1", "chr1", "chr1"],
            "position": [5, 9, 13, 17],
            "ref": ["A", "A", "A", "A"],
            "alt": ["C", "C", "C", "C"],
            "site_start": [0, 4, 8, 12],
            "site_end": [10, 14, 18, 22],
        }
    )
    sites = variants[["site_id"]].copy()
    sites["dic_class"] = "Robust"

    summary, tracks, intervals = score_variant_contact_boundary_strength(
        variants,
        sites=sites,
        model=ToyContactModel(),
        fasta_extractor=ToyFasta(),
        devices=[0, 1],
        input_len=8,
        window_bp=2,
        min_distance_bp=1,
    )

    assert summary.shape[0] == 4
    assert tracks["contact_track_index"].nunique() == 2
    assert intervals.shape[0] == 4


def test_filter_sites_for_context_window_drops_edges():
    sites = pd.DataFrame(
        {
            "chrom": ["chr1", "chr1"],
            "start": [0, 40],
            "end": [10, 60],
            "motif_start": [2, 48],
            "motif_end": [6, 52],
            "site_id": ["edge", "ok"],
            "strand": [".", "."],
        }
    )

    filtered = filter_sites_for_context_window(sites, ToyFasta(), input_len=20)

    assert filtered["site_id"].tolist() == ["ok"]


def test_aggregate_embedding_cluster_and_annotation():
    variants = pd.DataFrame(
        {
            "site_id": ["s1", "s2"],
            "chrom": ["chr1", "chr1"],
            "position": [10, 20],
            "site_start": [0, 10],
            "site_end": [20, 30],
        }
    )
    ref = np.ones((2, 4, 9))
    alt = ref.copy()
    alt[0, 0, 4] -= 2
    alt[0, 1, 4] -= 1
    track_groups = {
        "ctcf_binding": [0],
        "cohesin_binding": [1],
        "activity_marks": [2],
        "global_expression_outputs": [3],
    }

    features = aggregate_delta_features(ref, alt, variants, track_groups)
    embedding, state = fit_context_embedding(features, n_components=1)
    clusters, model = cluster_contexts(embedding, n_clusters=2)
    summary = annotate_context_clusters(features, clusters)

    assert "ctcf_binding__local__delta_abs_mean" in features.columns
    assert embedding.shape == (2, 2)
    assert state["feature_columns"]
    assert set(clusters.columns) == {"site_id", "cluster"}
    assert summary["n_sites"].sum() == 2


def test_aggregate_delta_features_rejects_shape_mismatch():
    with pytest.raises(ValueError, match="Prediction shape mismatch"):
        aggregate_delta_features(
            np.zeros((1, 2, 3)),
            np.zeros((1, 2, 4)),
            pd.DataFrame({"site_id": ["s1"]}),
            {},
        )


def test_summarize_center_track_effects_keeps_tracks_separate():
    ref = np.ones((1, 2, 5), dtype=np.float32)
    alt = ref.copy()
    alt[0, 0, 2] = 0.0
    alt[0, 1, 2] = 3.0
    metadata = pd.DataFrame(
        {
            "global_track_index": [0, 1],
            "output_type": ["chip_tf", "atac"],
            "track_index": [0, 0],
            "transcription_factor": ["CTCF", ""],
            "assay_title": ["ChIP-seq", "ATAC-seq"],
        }
    )
    variants = pd.DataFrame(
        {
            "site_id": ["mreg_ctcf_example_01_motif_snv"],
            "gene": ["MREG"],
            "control_type": ["ctcf_motif_max_ic_disruption"],
            "chrom": ["chr2"],
            "position": [10],
            "ref": ["A"],
            "alt": ["C"],
        }
    )
    bins = pd.DataFrame({"offset_bp": [-512, -256, 0, 256, 512]})

    mask = center_mask(bins, center_bp=512)
    effects = summarize_center_track_effects(
        ref,
        alt,
        metadata=metadata,
        variants=variants,
        window_bins=bins,
        center_bp=512,
    ).sort_values("global_track_index")

    assert mask.tolist() == [False, True, True, True, False]
    assert effects.shape[0] == 2
    assert effects["site_id"].nunique() == 1
    assert effects["target"].tolist() == ["CTCF", "ATAC-seq"]
    assert effects["raw_signed_change"].tolist() == pytest.approx([-1 / 3, 2 / 3])
    assert effects["raw_change_magnitude"].tolist() == pytest.approx([1 / 3, 2 / 3])
    assert effects["loss_log2_fold_change_magnitude"].iloc[0] > 0
    assert effects["gain_log2_fold_change_magnitude"].iloc[1] > 0


def test_expand_center_bp_signals_uses_dense_bp_axis():
    ref = np.array([[[10.0, 20.0, 30.0]]], dtype=np.float32)
    alt = np.array([[[11.0, 18.0, 35.0]]], dtype=np.float32)
    bins = pd.DataFrame(
        {
            "window_bin_index": [0, 1, 2],
            "model_bin_index": [100, 101, 102],
            "offset_bp": [-2, 0, 2],
        }
    )

    signals, coords = expand_center_bp_signals(
        ref,
        alt,
        window_bins=bins,
        center_bp=4,
        bin_size_bp=2,
    )

    assert coords["bp_offset"].tolist() == [-2, -1, 0, 1]
    assert coords["source_window_bin_index"].tolist() == [0, 0, 1, 1]
    assert signals["ref"].shape == (1, 1, 4)
    assert signals["alt"].shape == (1, 1, 4)
    assert signals["ref"][0, 0].tolist() == pytest.approx([10.0, 10.0, 20.0, 20.0])
    assert signals["raw_change"][0, 0].tolist() == pytest.approx([1.0, 1.0, -2.0, -2.0])
    assert signals["log2fc"].shape == (1, 1, 4)
