"""Paper-aligned AlphaGenome track selection."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

from grelu.interpret.ism.config import RunConfig
from grelu.interpret.ism.errors import ISMUserError
from grelu.interpret.ism.schemas import TRACK_SELECTION_SCHEMA


class TrackSelectionError(ISMUserError, ValueError):
    """Raised when AlphaGenome track metadata cannot satisfy the contract."""


ONE_BP_HEADS = {"atac", "dnase", "procap", "cage", "rna_seq"}


def _clean_text(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def _boolean_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    normalized = series.fillna("").astype(str).str.strip().str.lower()
    invalid = ~normalized.isin({"true", "false", "1", "0"})
    if invalid.any():
        examples = sorted(set(normalized[invalid]))[:5]
        raise TrackSelectionError(
            f"Invalid boolean values in track table: {examples}"
        )
    return normalized.isin({"true", "1"})


def default_track_metadata_path() -> Path:
    return (
        Path(__file__).resolve().parents[3]
        / "alphagenome_pytorch"
        / "src"
        / "alphagenome_pytorch"
        / "data"
        / "track_metadata_human.parquet"
    )


def load_track_metadata(path: str | Path | None) -> tuple[pd.DataFrame, Path]:
    metadata_path = Path(path) if path else default_track_metadata_path()
    if not metadata_path.exists():
        raise TrackSelectionError(
            f"AlphaGenome track metadata does not exist: {metadata_path}"
        )
    suffix = metadata_path.suffix.lower()
    if suffix == ".parquet":
        metadata = pd.read_parquet(metadata_path)
    elif suffix in {".tsv", ".txt"}:
        metadata = pd.read_csv(metadata_path, sep="\t", low_memory=False)
    elif suffix == ".csv":
        metadata = pd.read_csv(metadata_path, low_memory=False)
    else:
        raise TrackSelectionError(
            "Track metadata must be parquet, TSV, TXT, or CSV"
        )
    required = {"output_type"}
    missing = required - set(metadata.columns)
    if missing:
        raise TrackSelectionError(
            f"Track metadata is missing columns: {sorted(missing)}"
        )
    metadata = metadata.copy()
    metadata["output_type"] = (
        metadata["output_type"].fillna("").astype(str).str.lower()
    )
    if "track_index" not in metadata.columns:
        metadata["track_index"] = metadata.groupby(
            "output_type", sort=False
        ).cumcount()
    metadata["track_index"] = pd.to_numeric(
        metadata["track_index"], errors="raise"
    ).astype(int)
    if metadata.duplicated(["output_type", "track_index"]).any():
        examples = metadata.loc[
            metadata.duplicated(["output_type", "track_index"], keep=False),
            ["output_type", "track_index"],
        ].head(5)
        raise TrackSelectionError(
            "Track metadata contains duplicate head-local indices: "
            f"{examples.to_dict(orient='records')}"
        )
    return metadata, metadata_path


def _contains_any(value: str, candidates: Sequence[str]) -> bool:
    normalized = value.casefold()
    return any(candidate.casefold() in normalized for candidate in candidates)


def _target_name(row: Any) -> str:
    tf = _clean_text(getattr(row, "transcription_factor", ""))
    histone = _clean_text(getattr(row, "histone_mark", ""))
    return tf or histone


def _paper_family(
    output_type: str,
    target_name: str,
    tissue_specific_tf_targets: Sequence[str],
) -> tuple[str, str, str]:
    target = target_name.upper()
    if output_type == "contact_maps":
        return (
            "chromatin_architecture_contact",
            "AlphaGenome contact-map architecture proxy",
            "contact_architecture",
        )
    if output_type == "chip_tf" and target in {"CTCF", "RAD21", "SMC3"}:
        return (
            "cohesin_CTCF_binding",
            f"{target} ChIP-like binding",
            target.lower(),
        )
    if output_type in {"atac", "dnase"}:
        return (
            "open_chromatin",
            f"{output_type.upper()} accessibility proxy",
            "accessibility",
        )
    if (
        output_type == "chip_histone"
        and target in {"H3K27AC", "H3K4ME1", "H3K4ME2"}
    ) or (
        output_type == "chip_tf"
        and target in {"P300", "EP300", "CBP", "CREBBP"}
    ):
        return (
            "enhancer_like",
            f"{target} enhancer proxy",
            target.lower(),
        )
    if output_type == "chip_histone" and target == "H3K4ME3":
        return (
            "promoter_like",
            "H3K4me3 promoter proxy",
            "h3k4me3",
        )
    if output_type in {"cage", "procap"}:
        return (
            "Pol2_or_transcription_proxy",
            f"{output_type.upper()} promoter activity proxy",
            output_type,
        )
    if output_type == "chip_tf" and target in {
        "POLR2A",
        "POLR2B",
        "AFF4",
        "BRD4",
        "MED1",
    }:
        return (
            "Pol2_or_transcription_proxy",
            f"{target} transcription-machinery proxy",
            target.lower(),
        )
    if output_type == "rna_seq":
        return (
            "RNA_or_gene_output",
            "RNA-seq gene-output proxy",
            "rna_seq",
        )
    if output_type == "chip_tf" and target in {
        value.upper() for value in tissue_specific_tf_targets
    }:
        return (
            "tissue_specific_TF_like",
            f"{target or 'TF'} ChIP-like regulatory proxy",
            target.lower() or "other_tf",
        )
    return ("QC_metadata", "Not used by the v1.2 biological registry", "qc")


def _selection_match(
    row: Any,
    preferred_biosamples: Sequence[str],
    preferred_tissues: Sequence[str],
) -> bool:
    values = (
        _clean_text(getattr(row, "biosample_name", "")),
        _clean_text(getattr(row, "gtex_tissue", "")),
        _clean_text(getattr(row, "track_name", "")),
    )
    return any(
        _contains_any(value, preferred_biosamples)
        or _contains_any(value, preferred_tissues)
        for value in values
    )


def _rc_signature(record: dict[str, Any]) -> tuple[str, ...]:
    return tuple(
        str(record.get(column, "") or "").casefold()
        for column in (
            "output_type",
            "assay_type",
            "target_name",
            "cell_type",
            "tissue",
            "biosample",
            "paper_feature_family",
            "aggregation_group",
        )
    )


def _assign_rc_partners(table: pd.DataFrame) -> pd.DataFrame:
    result = table.copy()
    result["rc_partner_track_id"] = ""
    records = result.to_dict(orient="records")
    by_signature: dict[tuple[str, ...], dict[str, list[dict[str, Any]]]] = {}
    for record in records:
        strand = str(record["track_strand"])
        if strand not in {"+", "-"}:
            continue
        by_signature.setdefault(_rc_signature(record), {}).setdefault(
            strand, []
        ).append(record)
    partner_by_id: dict[str, str] = {}
    for strands in by_signature.values():
        plus = sorted(strands.get("+", []), key=lambda row: row["track_id"])
        minus = sorted(strands.get("-", []), key=lambda row: row["track_id"])
        if len(plus) != len(minus):
            continue
        for plus_row, minus_row in zip(plus, minus):
            partner_by_id[plus_row["track_id"]] = minus_row["track_id"]
            partner_by_id[minus_row["track_id"]] = plus_row["track_id"]
    strand_by_id = result.set_index("track_id")["track_strand"].to_dict()
    unstranded = ~result["track_strand"].isin(["+", "-"])
    result.loc[unstranded, "rc_partner_track_id"] = result.loc[
        unstranded, "track_id"
    ]
    result["rc_partner_track_id"] = result["track_id"].map(
        lambda track_id: partner_by_id.get(
            track_id,
            track_id if strand_by_id[track_id] not in {"+", "-"} else "",
        )
    )
    return result


def _contact_records() -> list[dict[str, Any]]:
    records = []
    for track_index in range(28):
        records.append(
            {
                "track_id": f"contact_maps:{track_index}",
                "modality": "contact_maps",
                "assay_type": "contact_maps",
                "target_name": "",
                "cell_type": "",
                "tissue": "",
                "biosample": "",
                "paper_feature_family": "chromatin_architecture_contact",
                "ag_feature_proxy": "AlphaGenome contact-map architecture proxy",
                "used_in_v1_2": True,
                "aggregation_group": "contact_architecture",
                "aggregation_weight": 1.0 / 28.0,
                "notes": "Built-in AlphaGenome contact-map output",
                "output_type": "contact_maps",
                "track_index": track_index,
                "track_name": f"contact_track_{track_index}",
                "track_strand": ".",
                "supported_resolutions": "2048",
                "selection_tier": "built_in_contact",
                "rc_partner_track_id": f"contact_maps:{track_index}",
            }
        )
    return records


def build_track_selection_table(
    metadata: pd.DataFrame,
    config: RunConfig,
) -> pd.DataFrame:
    """Map AlphaGenome tracks onto paper feature families."""

    records: list[dict[str, Any]] = []
    requested_heads = set(config.model.output_heads)
    for row in metadata.sort_values(
        ["output_type", "track_index"], kind="stable"
    ).itertuples(index=False):
        output_type = str(row.output_type).lower()
        if output_type == "contact_maps":
            continue
        track_index = int(row.track_index)
        track_name = _clean_text(getattr(row, "track_name", ""))
        track_strand = _clean_text(getattr(row, "track_strand", ".")) or "."
        is_padding = track_name.casefold() == "padding"
        target_name = _target_name(row)
        family, proxy, aggregation_group = _paper_family(
            output_type,
            target_name,
            config.model.tissue_specific_tf_targets,
        )
        requested = output_type in requested_heads
        biologically_registered = family != "QC_metadata"
        if is_padding:
            initial_tier = "padding_excluded"
            initial_note = "Padding track excluded"
        elif not requested:
            initial_tier = "head_not_requested"
            initial_note = "Output head is not requested by model.output_heads"
        elif not biologically_registered:
            initial_tier = "no_paper_family"
            initial_note = "No v1.2 paper-aligned feature family"
        else:
            initial_tier = "candidate"
            initial_note = "Candidate pending biosample preference"
        records.append(
            {
                "track_id": f"{output_type}:{track_index}",
                "modality": output_type,
                "assay_type": (
                    _clean_text(getattr(row, "assay_title", ""))
                    or output_type
                ),
                "target_name": target_name,
                "cell_type": _clean_text(
                    getattr(row, "ontology_curie", "")
                ),
                "tissue": _clean_text(getattr(row, "gtex_tissue", "")),
                "biosample": _clean_text(
                    getattr(row, "biosample_name", "")
                ),
                "paper_feature_family": family,
                "ag_feature_proxy": proxy,
                "used_in_v1_2": bool(
                    requested and biologically_registered and not is_padding
                ),
                "aggregation_group": aggregation_group,
                "aggregation_weight": 0.0,
                "notes": initial_note,
                "output_type": output_type,
                "track_index": track_index,
                "track_name": track_name or f"{output_type}_{track_index}",
                "track_strand": (
                    track_strand if track_strand in {"+", "-", "."} else "."
                ),
                "supported_resolutions": (
                    "1,128" if output_type in ONE_BP_HEADS else "128"
                ),
                "selection_tier": initial_tier,
                "_preferred_match": _selection_match(
                    row,
                    config.model.preferred_biosamples,
                    config.model.preferred_tissues,
                ),
            }
        )

    if "contact_maps" in requested_heads:
        records.extend(_contact_records())
    if not records:
        raise TrackSelectionError("Track metadata contains no usable rows")

    table = pd.DataFrame.from_records(records)
    biological = _boolean_series(table["used_in_v1_2"]) & ~table[
        "output_type"
    ].eq("contact_maps")
    for _, group in table.loc[biological].groupby(
        ["output_type", "aggregation_group"], sort=True
    ):
        preferred = group["_preferred_match"].fillna(False).astype(bool)
        if preferred.any():
            keep = group.index[preferred]
            reject = group.index[~preferred]
            table.loc[keep, "selection_tier"] = "preferred_biosample"
            table.loc[keep, "notes"] = "Preferred biosample/tissue match"
            table.loc[reject, "used_in_v1_2"] = False
            table.loc[reject, "selection_tier"] = "nonpreferred_excluded"
            table.loc[reject, "notes"] = (
                "Excluded because preferred biosample/tissue tracks exist"
            )
        elif config.model.allow_biosample_fallback:
            table.loc[group.index, "selection_tier"] = "family_fallback"
            table.loc[group.index, "notes"] = (
                "No preferred biosample/tissue match; family fallback retained"
            )
        else:
            table.loc[group.index, "used_in_v1_2"] = False
            table.loc[group.index, "selection_tier"] = "missing_preferred"
            table.loc[group.index, "notes"] = (
                "No preferred biosample/tissue match and fallback is disabled"
            )

    selected = _boolean_series(table["used_in_v1_2"])
    capped_groups = table.loc[
        selected & ~table["output_type"].eq("contact_maps")
    ].groupby(["output_type", "aggregation_group"], sort=True)
    for _, group in capped_groups:
        limit = config.model.max_tracks_per_aggregation_group
        if len(group) <= limit:
            continue
        ordered = group.sort_values(["track_index", "track_id"], kind="stable")
        reject = ordered.index[limit:]
        table.loc[reject, "used_in_v1_2"] = False
        table.loc[reject, "selection_tier"] = "group_cap_excluded"
        table.loc[reject, "notes"] = (
            f"Excluded by deterministic per-group cap of {limit}"
        )
    selected = _boolean_series(table["used_in_v1_2"])
    for _, group in table.loc[selected].groupby(
        ["output_type", "aggregation_group"], sort=True
    ):
        table.loc[group.index, "aggregation_weight"] = 1.0 / len(group)

    table = _assign_rc_partners(table)
    if config.model.model_kind == "fold":
        selected_ids = set(
            table.loc[
                _boolean_series(table["used_in_v1_2"]), "track_id"
            ]
        )
        missing_partner = (
            _boolean_series(table["used_in_v1_2"])
            & table["track_strand"].isin(["+", "-"])
            & ~table["rc_partner_track_id"].isin(selected_ids)
        )
        table.loc[missing_partner, "used_in_v1_2"] = False
        table.loc[missing_partner, "selection_tier"] = "missing_rc_partner"
        table.loc[missing_partner, "aggregation_weight"] = 0.0
        table.loc[missing_partner, "notes"] = (
            "Excluded from fold RC averaging because no strand partner exists"
        )
        selected = _boolean_series(table["used_in_v1_2"])
        table.loc[selected, "aggregation_weight"] = 0.0
        for _, group in table.loc[selected].groupby(
            ["output_type", "aggregation_group"], sort=True
        ):
            table.loc[group.index, "aggregation_weight"] = 1.0 / len(group)

    table = table.drop(columns="_preferred_match", errors="ignore")
    columns = list(TRACK_SELECTION_SCHEMA.required_columns) + list(
        TRACK_SELECTION_SCHEMA.optional_columns
    )
    return table.loc[:, columns].sort_values(
        ["output_type", "track_index"], kind="stable"
    ).reset_index(drop=True)


def selected_track_indices(
    table: pd.DataFrame,
    output_type: str,
    resolution: int,
) -> np.ndarray:
    selected = table.loc[
        _boolean_series(table["used_in_v1_2"])
        & table["output_type"].eq(output_type)
        & table["supported_resolutions"].map(
            lambda value: str(resolution) in str(value).split(",")
        )
    ].sort_values("track_index")
    return selected["track_index"].to_numpy(dtype=int)


def selected_track_rows(
    table: pd.DataFrame,
    output_type: str,
    resolution: int,
) -> pd.DataFrame:
    indices = set(selected_track_indices(table, output_type, resolution))
    return table.loc[
        table["output_type"].eq(output_type)
        & table["track_index"].isin(indices)
    ].sort_values("track_index").reset_index(drop=True)


def selected_heads(table: pd.DataFrame) -> Iterable[str]:
    selected = table.loc[_boolean_series(table["used_in_v1_2"]), "output_type"]
    return tuple(sorted(set(selected)))


def selected_track_count(table: pd.DataFrame) -> int:
    return int(_boolean_series(table["used_in_v1_2"]).sum())
