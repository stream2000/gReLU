"""Paper-label import, validation, and experimental reconstruction."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from grelu.interpret.ism.config import RunConfig
from grelu.interpret.ism.errors import ISMUserError
from grelu.interpret.ism.provenance import (
    ensure_manifest_config,
    output_dir,
    prepare_stage_outputs,
    record_stage,
    reject_completed_downstream_stages,
    sha256_file,
)
from grelu.interpret.ism.schemas import PAPER_LABEL_SCHEMA

PAPER_CLASSES = {
    "hc_dic",
    "lc_dic",
    "dic_unclassified",
    "stable_intragenic",
    "upregulated_intragenic",
    "intergenic",
    "promoter_proximal",
    "other",
}
LABEL_STATUSES = {"canonical", "provisional"}
GENOMIC_CONTEXTS = {"intragenic", "promoter_proximal", "intergenic", "other"}
BOOLEAN_COLUMNS = (
    "DIC_label",
    "HC_DIC_label",
    "LC_DIC_label",
    "intragenic_flag",
    "has_CTCF_motif",
)
INTEGER_COLUMNS = ("start", "end", "summit")
FLOAT_COLUMNS = (
    "Rad21_Mvalue_E2_vs_Ctrl",
    "CTCF_density",
    "gene_length",
    "distance_to_TSS",
    "distance_to_TES",
    "CTCF_motif_score",
    "CTCF_motif_center",
)

EXPERIMENTAL_REQUIRED_COLUMNS = (
    "site_id",
    "chrom",
    "start",
    "end",
    "summit",
    "Rad21_Mvalue_E2_vs_Ctrl",
    "CTCF_density",
    "host_gene",
    "gene_length",
    "distance_to_TSS",
    "distance_to_TES",
    "Pol2ser2_E2_Ctrl_ratio",
    "alternative_promoter_flag",
)


class LabelValidationError(ISMUserError, ValueError):
    """Raised when a paper-label table violates the v1.2 contract."""


def _require_columns(
    table: pd.DataFrame,
    required: Sequence[str],
    table_name: str,
) -> None:
    missing = [column for column in required if column not in table.columns]
    if missing:
        raise LabelValidationError(
            f"{table_name} is missing required columns: {missing}"
        )


def _coerce_boolean(series: pd.Series, column: str) -> pd.Series:
    mapping = {
        True: True,
        False: False,
        1: True,
        0: False,
        "1": True,
        "0": False,
        "true": True,
        "false": False,
        "yes": True,
        "no": False,
        "y": True,
        "n": False,
    }

    def convert(value: Any) -> Any:
        if pd.isna(value):
            return pd.NA
        key = (
            value
            if isinstance(value, (bool, int, np.integer))
            else str(value).strip().lower()
        )
        return mapping.get(key, pd.NA)

    converted = series.map(convert).astype("boolean")
    if converted.isna().any():
        examples = series.loc[converted.isna()].head(5).tolist()
        raise LabelValidationError(
            f"Column {column!r} contains invalid or missing boolean values: {examples}"
        )
    return converted.astype(bool)


def _coerce_numeric(
    series: pd.Series,
    column: str,
    *,
    allow_missing: bool,
) -> pd.Series:
    converted = pd.to_numeric(series, errors="coerce")
    invalid = converted.isna() & series.notna()
    if invalid.any():
        examples = series.loc[invalid].head(5).tolist()
        raise LabelValidationError(
            f"Column {column!r} contains non-numeric values: {examples}"
        )
    if not allow_missing and converted.isna().any():
        raise LabelValidationError(f"Column {column!r} contains missing values")
    non_finite = converted.notna() & ~np.isfinite(converted)
    if non_finite.any():
        examples = series.loc[non_finite].head(5).tolist()
        raise LabelValidationError(
            f"Column {column!r} contains non-finite values: {examples}"
        )
    return converted


def _coerce_integer(series: pd.Series, column: str) -> pd.Series:
    converted = _coerce_numeric(series, column, allow_missing=False)
    non_integer = converted.mod(1).ne(0)
    if non_integer.any():
        examples = series.loc[non_integer].head(5).tolist()
        raise LabelValidationError(
            f"Column {column!r} must contain integer coordinates: {examples}"
        )
    return converted.astype(np.int64)


def _coerce_required_text(series: pd.Series, column: str) -> pd.Series:
    missing = series.isna()
    converted = series.fillna("").astype(str).str.strip()
    missing |= converted.eq("")
    if missing.any():
        raise LabelValidationError(f"Column {column!r} contains missing values")
    return converted


def validate_paper_labels(
    table: pd.DataFrame,
    *,
    required_status: str | None = None,
    dic_mvalue_threshold: float = -0.5,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Normalize and strictly validate a paper-label table."""

    _require_columns(
        table,
        PAPER_LABEL_SCHEMA.required_columns,
        PAPER_LABEL_SCHEMA.filename,
    )
    labels = table.copy()
    for column in (
        "site_id",
        "chrom",
        "paper_class",
        "genomic_context",
        "label_status",
        "provenance_source",
        "provenance_version",
    ):
        labels[column] = _coerce_required_text(labels[column], column)
    labels["paper_class"] = labels["paper_class"].str.lower()
    labels["genomic_context"] = labels["genomic_context"].str.lower()
    labels["label_status"] = labels["label_status"].str.lower()

    if labels["site_id"].duplicated().any():
        examples = labels.loc[labels["site_id"].duplicated(), "site_id"].head(5).tolist()
        raise LabelValidationError(f"site_id must be unique; duplicates: {examples}")

    for column in BOOLEAN_COLUMNS:
        labels[column] = _coerce_boolean(labels[column], column)
    for column in INTEGER_COLUMNS:
        labels[column] = _coerce_integer(labels[column], column)
    for column in FLOAT_COLUMNS:
        labels[column] = _coerce_numeric(
            labels[column],
            column,
            allow_missing=column
            in {
                "gene_length",
                "distance_to_TSS",
                "distance_to_TES",
                "CTCF_motif_score",
                "CTCF_motif_center",
            },
        )

    invalid_interval = labels["start"] >= labels["end"]
    if invalid_interval.any():
        examples = labels.loc[invalid_interval, "site_id"].head(5).tolist()
        raise LabelValidationError(f"start must be less than end: {examples}")
    summit_outside = (labels["summit"] < labels["start"]) | (
        labels["summit"] >= labels["end"]
    )
    if summit_outside.any():
        examples = labels.loc[summit_outside, "site_id"].head(5).tolist()
        raise LabelValidationError(f"summit must lie within [start, end): {examples}")
    negative_coordinate = (
        labels["start"].lt(0) | labels["end"].lt(0) | labels["summit"].lt(0)
    )
    if negative_coordinate.any():
        examples = labels.loc[negative_coordinate, "site_id"].head(5).tolist()
        raise LabelValidationError(f"Genomic coordinates must be non-negative: {examples}")
    negative_density = labels["CTCF_density"].lt(0)
    if negative_density.any():
        examples = labels.loc[negative_density, "site_id"].head(5).tolist()
        raise LabelValidationError(f"CTCF_density must be non-negative: {examples}")
    negative_gene_length = labels["gene_length"].notna() & labels["gene_length"].lt(0)
    if negative_gene_length.any():
        examples = labels.loc[negative_gene_length, "site_id"].head(5).tolist()
        raise LabelValidationError(f"gene_length must be non-negative: {examples}")

    invalid_classes = sorted(set(labels["paper_class"]) - PAPER_CLASSES)
    if invalid_classes:
        raise LabelValidationError(f"Unsupported paper_class values: {invalid_classes}")
    invalid_statuses = sorted(set(labels["label_status"]) - LABEL_STATUSES)
    if invalid_statuses:
        raise LabelValidationError(
            f"Unsupported label_status values: {invalid_statuses}"
        )
    invalid_contexts = sorted(set(labels["genomic_context"]) - GENOMIC_CONTEXTS)
    if invalid_contexts:
        raise LabelValidationError(
            f"Unsupported genomic_context values: {invalid_contexts}"
        )
    if "strand" in labels.columns:
        labels["strand"] = labels["strand"].fillna(".").astype(str).str.strip()
        invalid_strands = sorted(set(labels["strand"]) - {"+", "-", "."})
        if invalid_strands:
            raise LabelValidationError(
                f"Unsupported strand values: {invalid_strands}"
            )
    if required_status and not labels["label_status"].eq(required_status).all():
        counts = labels["label_status"].value_counts(dropna=False).to_dict()
        raise LabelValidationError(
            f"Expected every label_status to be {required_status!r}; got {counts}"
        )
    hc_lc_overlap = labels["HC_DIC_label"] & labels["LC_DIC_label"]
    if hc_lc_overlap.any():
        examples = labels.loc[hc_lc_overlap, "site_id"].head(5).tolist()
        raise LabelValidationError(f"HC-DIC and LC-DIC must be exclusive: {examples}")
    subtype_without_dic = (
        labels["HC_DIC_label"] | labels["LC_DIC_label"]
    ) & ~labels["DIC_label"]
    if subtype_without_dic.any():
        examples = labels.loc[subtype_without_dic, "site_id"].head(5).tolist()
        raise LabelValidationError(f"HC/LC subtype must imply DIC: {examples}")

    class_invariants = (
        (labels["paper_class"].eq("hc_dic") & ~labels["HC_DIC_label"])
        | (labels["paper_class"].eq("lc_dic") & ~labels["LC_DIC_label"])
        | (labels["paper_class"].eq("dic_unclassified") & ~labels["DIC_label"])
        | (
            labels["HC_DIC_label"]
            & ~labels["paper_class"].eq("hc_dic")
        )
        | (
            labels["LC_DIC_label"]
            & ~labels["paper_class"].eq("lc_dic")
        )
        | (
            labels["DIC_label"]
            & ~labels["paper_class"].isin(
                {"hc_dic", "lc_dic", "dic_unclassified"}
            )
        )
        | (
            ~labels["DIC_label"]
            & labels["paper_class"].isin(
                {"hc_dic", "lc_dic", "dic_unclassified"}
            )
        )
    )
    if class_invariants.any():
        examples = labels.loc[
            class_invariants,
            ["site_id", "paper_class", "DIC_label", "HC_DIC_label", "LC_DIC_label"],
        ].head(5)
        raise LabelValidationError(
            "paper_class and DIC subtype flags disagree: "
            f"{examples.to_dict(orient='records')}"
        )
    intragenic_control_mismatch = labels["paper_class"].isin(
        {"stable_intragenic", "upregulated_intragenic"}
    ) & ~labels["intragenic_flag"]
    if intragenic_control_mismatch.any():
        examples = labels.loc[
            intragenic_control_mismatch, "site_id"
        ].head(5).tolist()
        raise LabelValidationError(
            f"Intragenic control classes require intragenic_flag=true: {examples}"
        )
    excluded_context_mismatch = labels["paper_class"].isin(
        {"intergenic", "promoter_proximal"}
    ) & labels["intragenic_flag"]
    if excluded_context_mismatch.any():
        examples = labels.loc[
            excluded_context_mismatch, "site_id"
        ].head(5).tolist()
        raise LabelValidationError(
            "Intergenic/promoter-proximal classes must not be marked intragenic: "
            f"{examples}"
        )

    dic_above_threshold = labels["DIC_label"] & (
        labels["Rad21_Mvalue_E2_vs_Ctrl"] >= dic_mvalue_threshold
    )
    if dic_above_threshold.any():
        examples = labels.loc[dic_above_threshold, "site_id"].head(5).tolist()
        raise LabelValidationError(
            "DIC rows must have Rad21_Mvalue_E2_vs_Ctrl < "
            f"{dic_mvalue_threshold}; "
            f"examples: {examples}"
        )
    dic_not_intragenic = labels["DIC_label"] & ~labels["intragenic_flag"]
    if dic_not_intragenic.any():
        examples = labels.loc[dic_not_intragenic, "site_id"].head(5).tolist()
        raise LabelValidationError(f"DIC rows must be intragenic: {examples}")
    normalized_host_gene = (
        labels["host_gene"].fillna("").astype(str).str.strip().str.lower()
    )
    intragenic_missing_gene = labels["intragenic_flag"] & normalized_host_gene.isin(
        {"", ".", "nan", "none"}
    )
    if intragenic_missing_gene.any():
        examples = labels.loc[intragenic_missing_gene, "site_id"].head(5).tolist()
        raise LabelValidationError(f"Intragenic rows require host_gene: {examples}")
    intragenic_missing_geometry = labels["intragenic_flag"] & (
        labels["gene_length"].isna()
        | labels["distance_to_TSS"].isna()
        | labels["distance_to_TES"].isna()
    )
    if intragenic_missing_geometry.any():
        examples = labels.loc[intragenic_missing_geometry, "site_id"].head(5).tolist()
        raise LabelValidationError(
            f"Intragenic rows require gene length and endpoint distances: {examples}"
        )

    motif_missing = labels["has_CTCF_motif"] & (
        labels["CTCF_motif_score"].isna() | labels["CTCF_motif_center"].isna()
    )
    if motif_missing.any():
        examples = labels.loc[motif_missing, "site_id"].head(5).tolist()
        raise LabelValidationError(
            "Rows with has_CTCF_motif=true require motif score and center: "
            f"{examples}"
        )
    motif_outside = labels["has_CTCF_motif"] & (
        (labels["CTCF_motif_center"] < labels["start"])
        | (labels["CTCF_motif_center"] >= labels["end"])
    )
    if motif_outside.any():
        examples = labels.loc[motif_outside, "site_id"].head(5).tolist()
        raise LabelValidationError(
            f"CTCF_motif_center must lie within the site interval: {examples}"
        )

    report_rows = [
        {
            "check": "row_count",
            "status": "pass",
            "count": int(len(labels)),
            "details": "Validated paper-label rows",
        },
        {
            "check": "unique_site_id",
            "status": "pass",
            "count": int(labels["site_id"].nunique()),
            "details": "site_id is unique",
        },
    ]
    for paper_class in sorted(PAPER_CLASSES):
        report_rows.append(
            {
                "check": f"paper_class:{paper_class}",
                "status": "count",
                "count": int(labels["paper_class"].eq(paper_class).sum()),
                "details": "Rows assigned to paper class",
            }
        )
    for status, count in labels["label_status"].value_counts().sort_index().items():
        report_rows.append(
            {
                "check": f"label_status:{status}",
                "status": "count",
                "count": int(count),
                "details": "Rows by provenance status",
            }
        )
    return labels, pd.DataFrame.from_records(report_rows)


def load_canonical_labels(
    path: str | Path,
    *,
    dic_mvalue_threshold: float = -0.5,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    labels = pd.read_csv(path, sep="\t", low_memory=False)
    return validate_paper_labels(
        labels,
        required_status="canonical",
        dic_mvalue_threshold=dic_mvalue_threshold,
    )


def _optional_table(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    return pd.read_csv(path, sep="\t", low_memory=False)


def _normalize_hc_lc_assignment(value: Any) -> str:
    normalized = str(value).strip().lower().replace("-", "_")
    mapping = {
        "hc": "hc_dic",
        "hc_dic": "hc_dic",
        "high": "hc_dic",
        "high_ctcf": "hc_dic",
        "lc": "lc_dic",
        "lc_dic": "lc_dic",
        "low": "lc_dic",
        "low_ctcf": "lc_dic",
    }
    if normalized not in mapping:
        raise LabelValidationError(f"Unsupported HC/LC assignment: {value!r}")
    return mapping[normalized]


def _assign_provisional_hc_lc(
    labels: pd.DataFrame,
    assignments: pd.DataFrame | None,
    random_seed: int,
) -> Tuple[pd.Series, str]:
    result = pd.Series("dic_unclassified", index=labels.index, dtype="object")
    result.loc[~labels["DIC_label"]] = ""
    dic_indices = labels.index[labels["DIC_label"]]
    if not len(dic_indices):
        return result, "no_dic_rows"

    if assignments is not None:
        _require_columns(
            assignments,
            ("site_id", "HC_LC_label"),
            "hc_lc_assignments.tsv",
        )
        normalized = assignments.copy()
        normalized["site_id"] = _coerce_required_text(
            normalized["site_id"],
            "hc_lc_assignments.site_id",
        )
        if normalized["site_id"].duplicated().any():
            raise LabelValidationError(
                "hc_lc_assignments.tsv contains duplicate site_id values"
            )
        unknown_ids = sorted(set(normalized["site_id"]) - set(labels["site_id"]))
        if unknown_ids:
            raise LabelValidationError(
                "hc_lc_assignments.tsv contains unknown site_id values: "
                f"{unknown_ids[:5]}"
            )
        non_dic_ids = sorted(
            set(normalized["site_id"])
            - set(labels.loc[labels["DIC_label"], "site_id"])
        )
        if non_dic_ids:
            raise LabelValidationError(
                "hc_lc_assignments.tsv may only assign reconstructed DIC rows: "
                f"{non_dic_ids[:5]}"
            )
        normalized["assignment"] = normalized["HC_LC_label"].map(
            _normalize_hc_lc_assignment
        )
        assignment_map = normalized.set_index("site_id")["assignment"]
        mapped = labels.loc[dic_indices, "site_id"].map(assignment_map)
        result.loc[dic_indices[mapped.notna()]] = mapped.loc[mapped.notna()].to_numpy()
        return result, "imported_hc_lc_assignments"

    density = labels.loc[dic_indices, "CTCF_density"]
    finite = density.notna() & np.isfinite(density) & density.ge(0)
    usable_indices = dic_indices[finite.to_numpy()]
    if (
        len(usable_indices) < 2
        or labels.loc[usable_indices, "CTCF_density"].nunique() < 2
    ):
        return result, "insufficient_ctcf_density_for_kmeans"
    values = np.log1p(labels.loc[usable_indices, "CTCF_density"].to_numpy()).reshape(-1, 1)
    model = KMeans(n_clusters=2, random_state=random_seed, n_init=20)
    cluster = model.fit_predict(values)
    centers = model.cluster_centers_.ravel()
    high_cluster = int(np.argmax(centers))
    result.loc[usable_indices] = np.where(
        cluster == high_cluster,
        "hc_dic",
        "lc_dic",
    )
    return result, "kmeans_log1p_ctcf_density"


def reconstruct_experimental_labels(
    input_dir: str | Path,
    config: RunConfig,
) -> Tuple[pd.DataFrame, pd.DataFrame, list[Path]]:
    """Reconstruct provisional labels from locally prepared GEO-derived tables."""

    root = Path(input_dir)
    sites_path = root / "cohesin_sites.tsv"
    if not sites_path.exists():
        raise LabelValidationError(
            f"Experimental reconstruction requires {sites_path}"
        )
    sites = pd.read_csv(sites_path, sep="\t", low_memory=False)
    _require_columns(sites, EXPERIMENTAL_REQUIRED_COLUMNS, "cohesin_sites.tsv")

    labels = sites.copy()
    labels["site_id"] = _coerce_required_text(labels["site_id"], "site_id")
    for column in ("start", "end", "summit"):
        labels[column] = _coerce_integer(labels[column], column)
    for column in (
        "Rad21_Mvalue_E2_vs_Ctrl",
        "CTCF_density",
        "gene_length",
        "distance_to_TSS",
        "distance_to_TES",
        "Pol2ser2_E2_Ctrl_ratio",
    ):
        labels[column] = _coerce_numeric(
            labels[column],
            column,
            allow_missing=column
            in {"gene_length", "distance_to_TSS", "distance_to_TES"},
        )
    labels["alternative_promoter_flag"] = _coerce_boolean(
        labels["alternative_promoter_flag"],
        "alternative_promoter_flag",
    )

    host_gene = labels["host_gene"].fillna("").astype(str).str.strip()
    has_host_gene = host_gene.ne("") & ~host_gene.str.lower().isin({"nan", "none", "."})
    distance_to_tss = labels["distance_to_TSS"].abs()
    distance_to_tes = labels["distance_to_TES"].abs()
    intragenic = (
        has_host_gene
        & labels["gene_length"].gt(config.labels.min_gene_length_bp)
        & distance_to_tss.gt(config.labels.gene_end_exclusion_bp)
        & distance_to_tes.gt(config.labels.gene_end_exclusion_bp)
        & ~labels["alternative_promoter_flag"]
    )
    e2_responsive = labels["Pol2ser2_E2_Ctrl_ratio"].gt(
        config.labels.pol2ser2_ratio_threshold
    )
    dic = (
        labels["Rad21_Mvalue_E2_vs_Ctrl"].lt(
            config.labels.dic_mvalue_threshold
        )
        & intragenic
        & e2_responsive
    )

    labels["intragenic_flag"] = intragenic
    labels["DIC_label"] = dic
    assignments_path = root / "hc_lc_assignments.tsv"
    assignments = _optional_table(assignments_path)
    subtype, subtype_method = _assign_provisional_hc_lc(
        labels,
        assignments,
        config.sampling.random_seed,
    )
    labels["HC_DIC_label"] = subtype.eq("hc_dic")
    labels["LC_DIC_label"] = subtype.eq("lc_dic")

    promoter = (
        ~dic
        & has_host_gene
        & distance_to_tss.le(config.labels.gene_end_exclusion_bp)
    )
    intergenic = ~dic & ~has_host_gene
    stable_intragenic = ~dic & intragenic & ~e2_responsive
    upregulated_intragenic = ~dic & intragenic & e2_responsive

    labels["paper_class"] = "other"
    labels.loc[promoter, "paper_class"] = "promoter_proximal"
    labels.loc[intergenic, "paper_class"] = "intergenic"
    labels.loc[stable_intragenic, "paper_class"] = "stable_intragenic"
    labels.loc[upregulated_intragenic, "paper_class"] = "upregulated_intragenic"
    labels.loc[dic, "paper_class"] = subtype.loc[dic]

    labels["genomic_context"] = "other"
    labels.loc[promoter, "genomic_context"] = "promoter_proximal"
    labels.loc[intergenic, "genomic_context"] = "intergenic"
    labels.loc[intragenic, "genomic_context"] = "intragenic"
    labels["E2_responsive_gene_flag"] = e2_responsive
    labels["HC_LC_assignment_method"] = ""
    labels.loc[dic, "HC_LC_assignment_method"] = subtype_method

    motif_path = root / "motif_annotations.tsv"
    motifs = _optional_table(motif_path)
    if motifs is not None:
        _require_columns(
            motifs,
            (
                "site_id",
                "has_CTCF_motif",
                "CTCF_motif_score",
                "CTCF_motif_center",
            ),
            "motif_annotations.tsv",
        )
        motifs["site_id"] = _coerce_required_text(
            motifs["site_id"],
            "motif_annotations.site_id",
        )
        if motifs["site_id"].duplicated().any():
            raise LabelValidationError(
                "motif_annotations.tsv contains duplicate site_id values"
            )
        unknown_motif_ids = sorted(set(motifs["site_id"]) - set(labels["site_id"]))
        if unknown_motif_ids:
            raise LabelValidationError(
                "motif_annotations.tsv contains unknown site_id values: "
                f"{unknown_motif_ids[:5]}"
            )
        motif_columns = [
            "site_id",
            "has_CTCF_motif",
            "CTCF_motif_score",
            "CTCF_motif_center",
        ]
        if "strand" in motifs.columns:
            motif_columns.append("strand")
        labels = labels.drop(
            columns=[
                column
                for column in motif_columns
                if column != "site_id" and column in labels.columns
            ]
        ).merge(
            motifs[motif_columns],
            on="site_id",
            how="left",
            validate="one_to_one",
        )
        labels["has_CTCF_motif"] = labels["has_CTCF_motif"].fillna(False)
    else:
        labels["has_CTCF_motif"] = False
        labels["CTCF_motif_score"] = np.nan
        labels["CTCF_motif_center"] = np.nan

    source_hash = sha256_file(sites_path)
    labels["label_status"] = "provisional"
    labels["provenance_source"] = f"GSE177045-derived:{sites_path}"
    labels["provenance_version"] = f"sha256:{source_hash[:16]}"
    if "strand" not in labels.columns:
        labels["strand"] = "."

    ordered_columns = list(PAPER_LABEL_SCHEMA.required_columns)
    ordered_columns.extend(
        column
        for column in (
            "strand",
            "E2_responsive_gene_flag",
            "Pol2ser2_E2_Ctrl_ratio",
            "alternative_promoter_flag",
            "HC_LC_assignment_method",
        )
        if column in labels.columns
    )
    labels, validation_report = validate_paper_labels(
        labels[ordered_columns],
        required_status="provisional",
        dic_mvalue_threshold=config.labels.dic_mvalue_threshold,
    )

    report_rows = [
        {
            "metric": "source_rows",
            "value": int(len(sites)),
            "details": str(sites_path),
        },
        {
            "metric": "intragenic_rows",
            "value": int(intragenic.sum()),
            "details": (
                f"gene_length>{config.labels.min_gene_length_bp}; "
                f"|TSS| and |TES|>{config.labels.gene_end_exclusion_bp}; "
                "alternative promoters excluded"
            ),
        },
        {
            "metric": "e2_responsive_rows",
            "value": int(e2_responsive.sum()),
            "details": (
                "Pol2ser2_E2_Ctrl_ratio>"
                f"{config.labels.pol2ser2_ratio_threshold}"
            ),
        },
        {
            "metric": "dic_rows",
            "value": int(dic.sum()),
            "details": (
                f"Rad21 M<{config.labels.dic_mvalue_threshold}, intragenic, "
                "and E2-responsive"
            ),
        },
        {
            "metric": "hc_lc_assignment_method",
            "value": subtype_method,
            "details": "All reconstructed labels remain provisional",
        },
    ]
    for paper_class, count in labels["paper_class"].value_counts().sort_index().items():
        report_rows.append(
            {
                "metric": f"paper_class:{paper_class}",
                "value": int(count),
                "details": "Reconstructed row count",
            }
        )

    inputs = [sites_path]
    if assignments is not None:
        inputs.append(assignments_path)
    if motifs is not None:
        inputs.append(motif_path)
    reconstruction_report = pd.DataFrame.from_records(report_rows)
    reconstruction_report.attrs["validation_report"] = validation_report
    return labels, reconstruction_report, inputs


def run(
    *,
    config: RunConfig,
    config_path: str | Path,
    allow_provisional_labels: bool,
    overwrite_stage: bool,
) -> None:
    """Execute the labels stage."""

    del allow_provisional_labels
    root = output_dir(config)
    labels_path = root / "paper_label_table.tsv"
    validation_path = root / "label_validation_report.tsv"
    reconstruction_path = root / "label_reconstruction_report.tsv"
    ensure_manifest_config(config, config_path)
    if overwrite_stage:
        reject_completed_downstream_stages(
            config,
            config_path,
            ("sample", "mutate", "infer", "features", "analyze", "report"),
        )
    stage_outputs = [labels_path, validation_path, reconstruction_path]
    prepare_stage_outputs(stage_outputs, overwrite_stage)
    root.mkdir(parents=True, exist_ok=True)

    if config.label_mode == "canonical":
        source_path = Path(config.paths.canonical_labels or "")
        if not source_path.exists():
            raise LabelValidationError(
                f"Canonical label table does not exist: {source_path}"
            )
        labels, validation_report = load_canonical_labels(
            source_path,
            dic_mvalue_threshold=config.labels.dic_mvalue_threshold,
        )
        inputs = [source_path]
        label_status = "canonical"
        reconstruction_report = None
    else:
        labels, reconstruction_report, inputs = reconstruct_experimental_labels(
            config.paths.provisional_inputs_dir or "",
            config,
        )
        validation_report = reconstruction_report.attrs["validation_report"]
        label_status = "provisional"

    labels.to_csv(labels_path, sep="\t", index=False)
    validation_report.to_csv(validation_path, sep="\t", index=False)
    outputs = [labels_path, validation_path]
    if reconstruction_report is not None:
        reconstruction_report.to_csv(reconstruction_path, sep="\t", index=False)
        outputs.append(reconstruction_path)
    elif reconstruction_path.exists():
        reconstruction_path.unlink()

    record_stage(
        config=config,
        config_path=config_path,
        stage="labels",
        inputs=inputs,
        outputs=outputs,
        label_status=label_status,
        metadata={
            "label_mode": config.label_mode,
            "row_count": int(len(labels)),
            "paper_class_counts": labels["paper_class"].value_counts().to_dict(),
        },
    )
