"""Versioned configuration contract for the CTCF ISM v1.2 workflow."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Type, TypeVar

from grelu.interpret.ism.errors import ISMUserError

CONFIG_SCHEMA_VERSION = "ctcf-ism-v1.2"


class ConfigError(ISMUserError, ValueError):
    """Raised when an ISM configuration violates the public contract."""


ConfigSection = TypeVar("ConfigSection")


def _build_section(
    section_name: str,
    section_type: Type[ConfigSection],
    value: Any,
) -> ConfigSection:
    if not isinstance(value, Mapping):
        raise ConfigError(f"Configuration section {section_name!r} must be an object")
    try:
        return section_type(**value)
    except TypeError as exc:
        raise ConfigError(
            f"Invalid keys or values in configuration section {section_name!r}: {exc}"
        ) from exc


@dataclass(frozen=True)
class PathConfig:
    canonical_labels: Optional[str] = None
    provisional_inputs_dir: Optional[str] = None
    fasta: Optional[str] = None
    gtf: Optional[str] = None
    motif_meme: Optional[str] = None
    track_metadata: Optional[str] = None
    weights_path: Optional[str] = None
    output_dir: str = "agent-doc/ism_context/ctcf_ism_v1_2"


@dataclass(frozen=True)
class LabelConfig:
    dic_mvalue_threshold: float = -0.5
    min_gene_length_bp: int = 20_000
    gene_end_exclusion_bp: int = 10_000
    pol2ser2_ratio_threshold: float = 1.2
    provisional_hc_lc_method: str = "kmeans"


@dataclass(frozen=True)
class SamplingConfig:
    hc_dic: int = 141
    lc_dic: int = 417
    stable_intragenic: int = 175
    upregulated_intragenic: int = 117
    intergenic: int = 100
    promoter_proximal: int = 50
    require_full_quota: bool = True
    random_seed: int = 20260609

    @property
    def total(self) -> int:
        return (
            self.hc_dic
            + self.lc_dic
            + self.stable_intragenic
            + self.upregulated_intragenic
            + self.intergenic
            + self.promoter_proximal
        )


@dataclass(frozen=True)
class MutationConfig:
    positions_per_motif: int = 3
    max_positions_per_motif: int = 5
    local_controls_per_motif: int = 5
    min_local_controls_per_motif: int = 3
    include_motif_preserving_control: bool = False
    max_control_distance_bp: int = 500
    max_gc_change_difference: float = 0.0
    reject_new_high_scoring_motif: bool = True
    motif_name: str = "CTCF"
    min_relative_motif_score: float = 0.80
    motif_rescan_flank_bp: int = 100


@dataclass(frozen=True)
class ModelConfig:
    model_kind: str = "distilled"
    devices: List[int] = field(default_factory=lambda: [0, 1, 2, 3])
    batch_size: int = 1
    num_workers: int = 1
    site_chunk_size: int = 4
    prediction_sequence_chunk_size: int = 4
    precision: Optional[str] = None
    compile: bool = False
    input_length_bp: int = 1_048_576
    output_heads: List[str] = field(
        default_factory=lambda: [
            "chip_tf",
            "chip_histone",
            "atac",
            "dnase",
            "procap",
            "cage",
            "rna_seq",
            "contact_maps",
        ]
    )
    one_d_resolutions: List[int] = field(default_factory=lambda: [1, 128])
    preferred_biosamples: List[str] = field(
        default_factory=lambda: ["MCF-7", "MCF7"]
    )
    preferred_tissues: List[str] = field(
        default_factory=lambda: ["breast", "mammary"]
    )
    allow_biosample_fallback: bool = True
    max_tracks_per_aggregation_group: int = 16
    tissue_specific_tf_targets: List[str] = field(
        default_factory=lambda: ["ESR1", "FOXA1", "GATA3", "PGR", "NR2F2"]
    )
    representative_site_count_per_class: int = 5
    representative_window_bp: int = 100_000


@dataclass(frozen=True)
class FeatureConfig:
    pseudocount: float = 1.0
    low_ref_quantile: float = 0.10
    centered_windows_bp: List[int] = field(default_factory=lambda: [501, 1001, 2001])
    contact_bin_bp: int = 2048
    apa_radii_bins: List[int] = field(default_factory=lambda: [5, 12])
    dlr_local_max_bp: int = 20_000
    dlr_distal_min_bp: int = 50_000
    dlr_distal_max_bp: int = 500_000


@dataclass(frozen=True)
class AnalysisConfig:
    missingness_threshold: float = 0.25
    low_signal_rate_threshold: float = 0.80
    kmeans_k: int = 10
    cluster_bootstrap_iterations: int = 100
    elastic_net_l1_ratio: float = 0.5
    chromosome_holdout_start: int = 16
    chromosome_holdout_end: int = 22
    grouped_cv_folds: int = 5
    enable_umap: bool = False
    enable_smote: bool = False
    random_seed: int = 20260609


@dataclass(frozen=True)
class RunConfig:
    schema_version: str = CONFIG_SCHEMA_VERSION
    genome: str = "hg38"
    label_mode: str = "canonical"
    paths: PathConfig = field(default_factory=PathConfig)
    labels: LabelConfig = field(default_factory=LabelConfig)
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    mutation: MutationConfig = field(default_factory=MutationConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    analysis: AnalysisConfig = field(default_factory=AnalysisConfig)

    def validate(self) -> None:
        if self.schema_version != CONFIG_SCHEMA_VERSION:
            raise ConfigError(
                f"Unsupported schema_version={self.schema_version!r}; "
                f"expected {CONFIG_SCHEMA_VERSION!r}"
            )
        if self.label_mode not in {"canonical", "experimental_reconstruction"}:
            raise ConfigError(
                "label_mode must be 'canonical' or 'experimental_reconstruction'"
            )
        if self.label_mode == "canonical" and not self.paths.canonical_labels:
            raise ConfigError("paths.canonical_labels is required in canonical mode")
        if (
            self.label_mode == "experimental_reconstruction"
            and not self.paths.provisional_inputs_dir
        ):
            raise ConfigError(
                "paths.provisional_inputs_dir is required in "
                "experimental_reconstruction mode"
            )
        if self.labels.min_gene_length_bp <= 0:
            raise ConfigError("labels.min_gene_length_bp must be positive")
        if self.labels.gene_end_exclusion_bp < 0:
            raise ConfigError("labels.gene_end_exclusion_bp must be non-negative")
        if self.labels.pol2ser2_ratio_threshold <= 0:
            raise ConfigError("labels.pol2ser2_ratio_threshold must be positive")
        if self.labels.provisional_hc_lc_method != "kmeans":
            raise ConfigError(
                "labels.provisional_hc_lc_method currently only supports 'kmeans'"
            )
        sampling_values = (
            self.sampling.hc_dic,
            self.sampling.lc_dic,
            self.sampling.stable_intragenic,
            self.sampling.upregulated_intragenic,
            self.sampling.intergenic,
            self.sampling.promoter_proximal,
        )
        if any(value < 0 for value in sampling_values):
            raise ConfigError("sampling quotas must be non-negative")
        if self.sampling.total <= 0:
            raise ConfigError("sampling quotas must select at least one site")
        if not 3 <= self.mutation.positions_per_motif <= 5:
            raise ConfigError("mutation.positions_per_motif must be between 3 and 5")
        if not (
            self.mutation.positions_per_motif
            <= self.mutation.max_positions_per_motif
            <= 5
        ):
            raise ConfigError(
                "mutation.max_positions_per_motif must be between "
                "positions_per_motif and 5"
            )
        if not 3 <= self.mutation.local_controls_per_motif <= 5:
            raise ConfigError(
                "mutation.local_controls_per_motif must be between 3 and 5"
            )
        if not (
            3
            <= self.mutation.min_local_controls_per_motif
            <= self.mutation.local_controls_per_motif
        ):
            raise ConfigError(
                "mutation.min_local_controls_per_motif must be between 3 and "
                "local_controls_per_motif"
            )
        if self.mutation.max_control_distance_bp <= 0:
            raise ConfigError("mutation.max_control_distance_bp must be positive")
        if self.mutation.max_gc_change_difference < 0:
            raise ConfigError(
                "mutation.max_gc_change_difference must be non-negative"
            )
        if not self.mutation.motif_name.strip():
            raise ConfigError("mutation.motif_name must not be empty")
        if not 0 < self.mutation.min_relative_motif_score <= 1:
            raise ConfigError(
                "mutation.min_relative_motif_score must be in (0, 1]"
            )
        if self.mutation.motif_rescan_flank_bp < 0:
            raise ConfigError(
                "mutation.motif_rescan_flank_bp must be non-negative"
            )
        if self.model.model_kind not in {"distilled", "fold"}:
            raise ConfigError("model.model_kind must be 'distilled' or 'fold'")
        if not self.model.devices:
            raise ConfigError("model.devices must contain at least one device")
        if any(device < 0 for device in self.model.devices):
            raise ConfigError("model.devices must be non-negative")
        if len(set(self.model.devices)) != len(self.model.devices):
            raise ConfigError("model.devices must not contain duplicates")
        if self.model.batch_size <= 0:
            raise ConfigError("model.batch_size must be positive")
        if self.model.num_workers < 0:
            raise ConfigError("model.num_workers must be non-negative")
        if self.model.site_chunk_size <= 0:
            raise ConfigError("model.site_chunk_size must be positive")
        if self.model.prediction_sequence_chunk_size <= 0:
            raise ConfigError(
                "model.prediction_sequence_chunk_size must be positive"
            )
        if self.model.input_length_bp <= 0 or self.model.input_length_bp % 2:
            raise ConfigError(
                "model.input_length_bp must be a positive even integer"
            )
        allowed_heads = {
            "atac",
            "dnase",
            "procap",
            "cage",
            "rna_seq",
            "chip_tf",
            "chip_histone",
            "contact_maps",
        }
        unknown_heads = sorted(set(self.model.output_heads) - allowed_heads)
        if unknown_heads:
            raise ConfigError(
                f"model.output_heads contains unsupported heads: {unknown_heads}"
            )
        if not self.model.output_heads:
            raise ConfigError("model.output_heads must not be empty")
        if len(set(self.model.output_heads)) != len(self.model.output_heads):
            raise ConfigError("model.output_heads must not contain duplicates")
        if sorted(set(self.model.one_d_resolutions) - {1, 128}):
            raise ConfigError("model.one_d_resolutions may only contain 1 and 128")
        if not self.model.one_d_resolutions:
            raise ConfigError("model.one_d_resolutions must not be empty")
        if len(set(self.model.one_d_resolutions)) != len(
            self.model.one_d_resolutions
        ):
            raise ConfigError(
                "model.one_d_resolutions must not contain duplicates"
            )
        if self.model.max_tracks_per_aggregation_group <= 0:
            raise ConfigError(
                "model.max_tracks_per_aggregation_group must be positive"
            )
        if not self.model.tissue_specific_tf_targets:
            raise ConfigError(
                "model.tissue_specific_tf_targets must not be empty"
            )
        if self.model.representative_site_count_per_class < 0:
            raise ConfigError(
                "model.representative_site_count_per_class must be non-negative"
            )
        if self.model.representative_window_bp <= 0:
            raise ConfigError("model.representative_window_bp must be positive")
        if not self.features.centered_windows_bp or any(
            window <= 0 or window % 2 == 0
            for window in self.features.centered_windows_bp
        ):
            raise ConfigError(
                "features.centered_windows_bp must contain positive odd windows"
            )
        if self.features.pseudocount <= 0:
            raise ConfigError("features.pseudocount must be positive")
        if not 0 < self.features.low_ref_quantile < 1:
            raise ConfigError("features.low_ref_quantile must be in (0, 1)")
        if self.features.contact_bin_bp != 2048:
            raise ConfigError("v1.2 contact_bin_bp is fixed at 2048")
        if not self.features.apa_radii_bins or any(
            radius <= 0 for radius in self.features.apa_radii_bins
        ):
            raise ConfigError(
                "features.apa_radii_bins must contain positive radii"
            )
        if not (
            0
            < self.features.dlr_local_max_bp
            < self.features.dlr_distal_min_bp
            < self.features.dlr_distal_max_bp
            <= self.model.input_length_bp // 2
        ):
            raise ConfigError(
                "DLR distances must satisfy 0 < local_max < distal_min < "
                "distal_max <= half the model input"
            )
        if self.analysis.kmeans_k != 10:
            raise ConfigError("v1.2 paper-style k-means requires kmeans_k=10")
        if not 0 <= self.analysis.missingness_threshold < 1:
            raise ConfigError(
                "analysis.missingness_threshold must be in [0, 1)"
            )
        if not 0 <= self.analysis.low_signal_rate_threshold <= 1:
            raise ConfigError(
                "analysis.low_signal_rate_threshold must be in [0, 1]"
            )
        if self.analysis.cluster_bootstrap_iterations != 100:
            raise ConfigError(
                "v1.2 cluster_bootstrap_iterations is fixed at 100"
            )
        if self.analysis.elastic_net_l1_ratio != 0.5:
            raise ConfigError("v1.2 Elastic Net requires elastic_net_l1_ratio=0.5")
        if self.analysis.grouped_cv_folds < 2:
            raise ConfigError("analysis.grouped_cv_folds must be at least 2")
        if not (
            1
            <= self.analysis.chromosome_holdout_start
            <= self.analysis.chromosome_holdout_end
            <= 22
        ):
            raise ConfigError(
                "analysis chromosome holdout must be within autosomes 1-22"
            )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "RunConfig":
        known = {
            "schema_version",
            "genome",
            "label_mode",
            "paths",
            "labels",
            "sampling",
            "mutation",
            "model",
            "features",
            "analysis",
        }
        unknown = sorted(set(value) - known)
        if unknown:
            raise ConfigError(f"Unknown top-level configuration keys: {unknown}")

        config = cls(
            schema_version=value.get("schema_version", CONFIG_SCHEMA_VERSION),
            genome=value.get("genome", "hg38"),
            label_mode=value.get("label_mode", "canonical"),
            paths=_build_section("paths", PathConfig, value.get("paths", {})),
            labels=_build_section("labels", LabelConfig, value.get("labels", {})),
            sampling=_build_section(
                "sampling", SamplingConfig, value.get("sampling", {})
            ),
            mutation=_build_section(
                "mutation", MutationConfig, value.get("mutation", {})
            ),
            model=_build_section("model", ModelConfig, value.get("model", {})),
            features=_build_section(
                "features", FeatureConfig, value.get("features", {})
            ),
            analysis=_build_section(
                "analysis", AnalysisConfig, value.get("analysis", {})
            ),
        )
        config.validate()
        return config

    @classmethod
    def load(cls, path: str | Path) -> "RunConfig":
        config_path = Path(path)
        with config_path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
        if not isinstance(value, dict):
            raise ConfigError("Configuration root must be a JSON object")
        return cls.from_mapping(value)
