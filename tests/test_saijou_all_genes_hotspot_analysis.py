import pandas as pd

from scripts.ism.experiments.saijou_hsc.analyze_saijou_all_genes_10bp_scan import (
    add_genomic_annotation,
    center_scores,
    cluster_candidates,
    combined_center_scores,
)


def synthetic_features():
    rows = []
    for model in ["AlphaGenome e19", "Borzoi e39"]:
        for offset in [0, 2, 4, 6, 20]:
            for cell in ["hsc", "mac", "lsec", "chol"]:
                for replicate in range(3):
                    magnitude = (offset + 2) / 100
                    if cell == "hsc":
                        magnitude *= 2
                    rows.append(
                        {
                            "model": model,
                            "gene": "GeneA",
                            "readout_role": "tss_1024bp",
                            "variant_offset_from_tss_transcription_bp": offset,
                            "cell_type": cell,
                            "mutation_id": f"{model}:{offset}:{cell}:{replicate}",
                            "abs_effect": magnitude,
                            "log2fc_ratio_of_sums": -magnitude,
                        }
                    )
    return pd.DataFrame(rows)


def test_hotspot_scoring_and_splice_exclusion():
    scores = center_scores(synthetic_features())
    combined = combined_center_scores(scores)
    genes = pd.DataFrame(
        [
            {
                "gene": "GeneA",
                "chrom": "chr1",
                "strand": "+",
                "analysis_tss": 1000,
                "length": 100,
            }
        ]
    )
    annotated = add_genomic_annotation(
        combined,
        genes,
        exon_intervals={"GeneA": [(1000, 1004), (1004, 1100)]},
        splice_sites={"GeneA": [(1004, "splice_donor")]},
        splice_buffer_bp=2,
    )
    assert annotated.loc[
        annotated.variant_offset_from_tss_transcription_bp.eq(4),
        "overlaps_splice_site",
    ].all()
    segments, candidates = cluster_candidates(
        annotated,
        segments_per_gene=1,
        max_centers_per_segment=20,
        candidate_quantile=0,
    )
    assert len(segments) == 1
    assert not candidates.overlaps_splice_site.any()
    assert set(candidates.gene) == {"GeneA"}


def test_requested_acta2_positive_control_is_retained():
    rows = []
    for offset, priority in [(-63, 0.2), (-61, 0.1), (-59, 0.3), (20, 0.9), (22, 0.8)]:
        rows.append(
            {
                "gene": "Acta2",
                "readout_role": "tss_1024bp",
                "variant_offset_from_tss_transcription_bp": offset,
                "selection_priority": priority,
                "discovery_score": priority,
                "discovery_class": "background",
                "region_type": "promoter_tss",
                "ag_hsc_ratio": 1.0,
                "bz_hsc_ratio": 1.0,
                "both_hsc_top": False,
                "nearest_splice_distance_bp": 100.0,
                "overlaps_splice_site": False,
            }
        )
    segments, candidates = cluster_candidates(
        pd.DataFrame(rows),
        segments_per_gene=2,
        max_centers_per_segment=20,
        candidate_quantile=0.8,
    )
    control = segments.loc[
        segments.peak_biological_prior.eq("promoter_CArG_SRF_positive_control")
    ]
    assert len(control) == 1
    assert int(control.iloc[0].peak_tx_offset) == -61
    assert -61 in set(candidates.variant_offset_from_tss_transcription_bp)


def test_final_mdk_submodules_are_preregistered():
    rows = []
    for offset in range(455, 511, 2):
        rows.append(
            {
                "gene": "Mdk",
                "readout_role": "gene_body_output_clipped",
                "variant_offset_from_tss_transcription_bp": offset,
                "selection_priority": offset / 1000,
                "discovery_score": offset / 1000,
                "discovery_class": "background",
                "region_type": "intron",
                "ag_hsc_ratio": 1.0,
                "bz_hsc_ratio": 1.0,
                "both_hsc_top": False,
                "nearest_splice_distance_bp": 90.0,
                "overlaps_splice_site": False,
            }
        )
    segments, candidates = cluster_candidates(
        pd.DataFrame(rows),
        segments_per_gene=2,
        max_centers_per_segment=20,
        candidate_quantile=0.95,
    )
    assert segments[["tx_start", "tx_end"]].values.tolist() == [[499, 507], [463, 475]]
    assert set(segments.peak_biological_prior) == {
        "Mdk_intron1_SP_KLF_submodule",
        "Mdk_intron1_secondary_peak",
    }
    assert set(candidates.variant_offset_from_tss_transcription_bp) == set(
        list(range(463, 476, 2)) + list(range(499, 508, 2))
    )
