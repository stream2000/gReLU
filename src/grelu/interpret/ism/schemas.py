"""Stable table and artifact contracts for CTCF ISM EDA v1.2."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple


@dataclass(frozen=True)
class TableSchema:
    """A lightweight public contract for one tabular artifact."""

    filename: str
    row_grain: str
    required_columns: Tuple[str, ...]
    optional_columns: Tuple[str, ...] = ()
    dynamic_column_suffixes: Tuple[str, ...] = ()


PAPER_LABEL_SCHEMA = TableSchema(
    filename="paper_label_table.tsv",
    row_grain="one row per cohesin/RAD21 peak",
    required_columns=(
        "site_id",
        "chrom",
        "start",
        "end",
        "summit",
        "paper_class",
        "DIC_label",
        "HC_DIC_label",
        "LC_DIC_label",
        "Rad21_Mvalue_E2_vs_Ctrl",
        "CTCF_density",
        "host_gene",
        "gene_length",
        "intragenic_flag",
        "distance_to_TSS",
        "distance_to_TES",
        "genomic_context",
        "has_CTCF_motif",
        "CTCF_motif_score",
        "CTCF_motif_center",
        "label_status",
        "provenance_source",
        "provenance_version",
    ),
    optional_columns=("paper_cluster_if_available", "strand"),
)

SITE_METADATA_SCHEMA = TableSchema(
    filename="site_metadata.tsv",
    row_grain="one row per sampled paper site",
    required_columns=(
        "site_id",
        "chrom",
        "start",
        "end",
        "summit",
        "paper_class",
        "sampling_stratum",
        "sampling_rank",
        "has_CTCF_motif",
        "mutation_available",
    ),
)

MUTATION_SCHEMA = TableSchema(
    filename="mutation_table.tsv",
    row_grain="one row per motif or control perturbation",
    required_columns=(
        "perturbation_id",
        "site_id",
        "perturbation_type",
        "chrom",
        "edit_start",
        "edit_end",
        "ref_sequence",
        "alt_sequence",
        "motif_strand",
        "motif_start",
        "motif_end",
        "edited_base_count",
        "gc_change",
        "pwm_ref_score",
        "pwm_alt_score",
        "pwm_score_change",
        "creates_high_scoring_ctcf_motif",
        "control_set_id",
        "control_rank",
    ),
    optional_columns=(
        "motif_relative_ref_score",
        "motif_relative_alt_score",
        "edited_positions",
        "gc_delta_count",
        "distance_to_motif_bp",
        "mutation_status",
        "mutation_reason",
    ),
)

MOTIF_ANNOTATION_SCHEMA = TableSchema(
    filename="motif_annotation_table.tsv",
    row_grain="one row per sampled paper site",
    required_columns=(
        "site_id",
        "chrom",
        "site_start",
        "site_end",
        "scan_status",
        "motif_start",
        "motif_end",
        "motif_center",
        "motif_strand",
        "motif_ref_sequence",
        "pwm_ref_score",
        "motif_relative_score",
        "mutation_available",
        "local_control_count",
        "annotation_reason",
    ),
)

MUTATION_QC_SCHEMA = TableSchema(
    filename="mutation_qc_report.tsv",
    row_grain="one row per mutation-design metric or failure reason",
    required_columns=("metric", "value", "details"),
)

TRACK_SELECTION_SCHEMA = TableSchema(
    filename="track_selection_table.tsv",
    row_grain="one row per selected or rejected AlphaGenome track",
    required_columns=(
        "track_id",
        "modality",
        "assay_type",
        "target_name",
        "cell_type",
        "tissue",
        "biosample",
        "paper_feature_family",
        "ag_feature_proxy",
        "used_in_v1_2",
        "aggregation_group",
        "aggregation_weight",
        "notes",
    ),
    optional_columns=(
        "output_type",
        "track_index",
        "track_name",
        "track_strand",
        "supported_resolutions",
        "selection_tier",
        "rc_partner_track_id",
    ),
)

INFERENCE_SUMMARY_SCHEMA = TableSchema(
    filename="inference_summaries/*.parquet",
    row_grain=(
        "one row per site, perturbation, orientation, track, mask, and statistic"
    ),
    required_columns=(
        "site_id",
        "perturbation_id",
        "allele",
        "orientation",
        "output_type",
        "resolution",
        "track_id",
        "summary_name",
        "statistic",
        "value",
    ),
    optional_columns=(
        "offset_start_bp",
        "offset_end_bp",
        "bin_count",
    ),
)

INFERENCE_INDEX_SCHEMA = TableSchema(
    filename="inference_index.tsv",
    row_grain="one row per streamed inference shard",
    required_columns=(
        "shard_id",
        "output_type",
        "resolution",
        "orientation",
        "site_start_index",
        "site_end_index",
        "row_count",
        "relative_path",
        "sha256",
    ),
)

INFERENCE_QC_SCHEMA = TableSchema(
    filename="inference_qc_report.tsv",
    row_grain="one row per inference metric or exclusion reason",
    required_columns=("metric", "value", "details"),
)

RAW_SIGNAL_SCHEMA = TableSchema(
    filename="raw_signal_feature_matrix.tsv",
    row_grain="one row per paper site",
    required_columns=("site_id",),
    dynamic_column_suffixes=("__S_ref", "__S_alt", "__low_ref_signal_flag"),
)

RAW_DELTA_SCHEMA = TableSchema(
    filename="raw_delta_feature_matrix.tsv",
    row_grain="one row per paper site",
    required_columns=("site_id",),
    dynamic_column_suffixes=(
        "__absolute_delta",
        "__delta_magnitude",
        "__log2fc",
        "__effect_available",
    ),
)

CONTROL_SUMMARY_SCHEMA = TableSchema(
    filename="control_summary_matrix.tsv",
    row_grain="one row per paper site",
    required_columns=("site_id",),
    dynamic_column_suffixes=("__control_median_score", "__control_mad"),
)

CONTROL_ADJUSTED_SCHEMA = TableSchema(
    filename="control_adjusted_feature_matrix.tsv",
    row_grain="one row per paper site",
    required_columns=("site_id",),
    dynamic_column_suffixes=(
        "__control_adjusted_score",
        "__motif_vs_control_z",
    ),
)

EDA_SCALED_SCHEMA = TableSchema(
    filename="eda_scaled_feature_matrix.tsv",
    row_grain="one row per paper site",
    required_columns=("site_id",),
)

FEATURE_QC_SCHEMA = TableSchema(
    filename="feature_qc_report.tsv",
    row_grain="one row per candidate EDA feature",
    required_columns=(
        "feature_name",
        "feature_family",
        "missing_rate",
        "low_signal_rate",
        "iqr",
        "included_in_eda",
        "exclusion_reason",
    ),
)

LABEL_VALIDATION_SCHEMA = TableSchema(
    filename="label_validation_report.tsv",
    row_grain="one row per validation rule or label count",
    required_columns=("check", "status", "count", "details"),
)

LABEL_RECONSTRUCTION_SCHEMA = TableSchema(
    filename="label_reconstruction_report.tsv",
    row_grain="one row per reconstruction rule or class",
    required_columns=("metric", "value", "details"),
)

SAMPLING_SUMMARY_SCHEMA = TableSchema(
    filename="sampling_summary.tsv",
    row_grain="one row per sampling stratum",
    required_columns=(
        "sampling_stratum",
        "requested_count",
        "available_count",
        "selected_count",
        "shortage_count",
        "label_status",
    ),
)

TABLE_SCHEMAS: Dict[str, TableSchema] = {
    schema.filename: schema
    for schema in (
        PAPER_LABEL_SCHEMA,
        SITE_METADATA_SCHEMA,
        MUTATION_SCHEMA,
        MOTIF_ANNOTATION_SCHEMA,
        MUTATION_QC_SCHEMA,
        TRACK_SELECTION_SCHEMA,
        INFERENCE_SUMMARY_SCHEMA,
        INFERENCE_INDEX_SCHEMA,
        INFERENCE_QC_SCHEMA,
        RAW_SIGNAL_SCHEMA,
        RAW_DELTA_SCHEMA,
        CONTROL_SUMMARY_SCHEMA,
        CONTROL_ADJUSTED_SCHEMA,
        EDA_SCALED_SCHEMA,
        FEATURE_QC_SCHEMA,
        LABEL_VALIDATION_SCHEMA,
        LABEL_RECONSTRUCTION_SCHEMA,
        SAMPLING_SUMMARY_SCHEMA,
    )
}


@dataclass(frozen=True)
class ArtifactSpec:
    index: int
    path: str
    stage: str
    description: str


_ARTIFACT_ROWS = (
    (1, "paper_label_table.tsv", "labels", "Paper-aligned site labels"),
    (2, "site_metadata.tsv", "sample", "Sampled site metadata"),
    (3, "mutation_table.tsv", "mutate", "Motif and matched-control edits"),
    (4, "track_selection_table.tsv", "infer", "Track-family mapping and use decisions"),
    (5, "raw_signal_feature_matrix.tsv", "features", "REF and ALT signal summaries"),
    (6, "raw_delta_feature_matrix.tsv", "features", "Signed and fold-change effects"),
    (7, "control_summary_matrix.tsv", "features", "Control median and MAD"),
    (
        8,
        "control_adjusted_feature_matrix.tsv",
        "features",
        "Adjusted effects and z-scores",
    ),
    (9, "eda_scaled_feature_matrix.tsv", "analyze", "QC-filtered robust-scaled matrix"),
    (10, "feature_qc_report.tsv", "analyze", "Feature inclusion and exclusion audit"),
    (11, "low_signal_rate_by_feature.tsv", "analyze", "Low-signal prevalence"),
    (12, "feature_correlation_heatmap.png", "analyze", "Paper-style feature modules"),
    (13, "pca_variance_explained.tsv", "analyze", "PCA variance summary"),
    (14, "pca_umap_by_paper_labels.png", "analyze", "Embedding views"),
    (15, "kmeans_k10_cluster_assignment.tsv", "analyze", "Paper-style clusters"),
    (16, "cluster_label_enrichment.tsv", "analyze", "Cluster label enrichment"),
    (17, "cluster_centroid.tsv", "analyze", "Cluster feature centroids"),
    (18, "cluster_stability.tsv", "analyze", "Bootstrap stability"),
    (19, "feature_family_ablation_report.tsv", "analyze", "Feature-family ablations"),
    (20, "supervised_reproduction_metrics.tsv", "analyze", "Classification metrics"),
    (21, "supervised_feature_importance.tsv", "analyze", "Interpretable model weights"),
    (22, "elastic_net_mvalue_driver_report.tsv", "analyze", "M-value regression"),
    (23, "representative_hc_dic_examples.tsv", "report", "HC-DIC examples"),
    (24, "representative_lc_dic_examples.tsv", "report", "LC-DIC examples"),
    (25, "representative_stable_control_examples.tsv", "report", "Stable controls"),
    (26, "representative_local_1d_delta_plots/", "report", "Local 1D plots"),
    (27, "representative_contact_delta_plots/", "report", "Contact plots"),
)

ARTIFACTS: Tuple[ArtifactSpec, ...] = tuple(
    ArtifactSpec(*row) for row in _ARTIFACT_ROWS
)
