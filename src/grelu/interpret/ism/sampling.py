"""Deterministic paper-class sampling for the v1.2 pilot."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd

from grelu.interpret.ism.config import RunConfig
from grelu.interpret.ism.labels import LabelValidationError, validate_paper_labels
from grelu.interpret.ism.provenance import (
    output_dir,
    prepare_stage_outputs,
    record_stage,
    reject_completed_downstream_stages,
    require_completed_stage,
)

SAMPLING_STRATA: Tuple[Tuple[str, str], ...] = (
    ("hc_dic", "hc_dic"),
    ("lc_dic", "lc_dic"),
    ("stable_intragenic", "stable_intragenic"),
    ("upregulated_intragenic", "upregulated_intragenic"),
    ("intergenic", "intergenic"),
    ("promoter_proximal", "promoter_proximal"),
)


def _sampling_quotas(config: RunConfig) -> Dict[str, int]:
    return {
        "hc_dic": config.sampling.hc_dic,
        "lc_dic": config.sampling.lc_dic,
        "stable_intragenic": config.sampling.stable_intragenic,
        "upregulated_intragenic": config.sampling.upregulated_intragenic,
        "intergenic": config.sampling.intergenic,
        "promoter_proximal": config.sampling.promoter_proximal,
    }


def _stable_sampling_key(site_id: str, seed: int) -> str:
    value = f"{seed}:{site_id}".encode("utf-8")
    return hashlib.sha256(value).hexdigest()


def sample_paper_sites(
    labels: pd.DataFrame,
    config: RunConfig,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Select deterministic rows from each paper-aligned stratum."""

    quotas = _sampling_quotas(config)
    selected_frames = []
    summary_rows = []
    statuses = ",".join(sorted(labels["label_status"].unique()))

    for offset, (stratum, paper_class) in enumerate(SAMPLING_STRATA):
        requested = quotas[stratum]
        candidates = labels.loc[labels["paper_class"].eq(paper_class)].copy()
        candidates = candidates.sort_values("site_id").reset_index(drop=True)
        available = len(candidates)
        if config.sampling.require_full_quota and available < requested:
            raise LabelValidationError(
                f"Sampling stratum {stratum!r} requires {requested} rows but "
                f"only {available} are available"
            )
        selected_count = min(requested, available)
        if selected_count:
            stratum_seed = config.sampling.random_seed + offset
            candidates["_sampling_key"] = candidates["site_id"].map(
                lambda site_id: _stable_sampling_key(site_id, stratum_seed)
            )
            selected = (
                candidates.sort_values(["_sampling_key", "site_id"])
                .head(selected_count)
                .drop(columns="_sampling_key")
                .reset_index(drop=True)
            )
            selected["sampling_stratum"] = stratum
            selected["sampling_rank"] = range(1, selected_count + 1)
            selected["sampling_requested_count"] = requested
            selected["sampling_seed"] = stratum_seed
            selected["mutation_available"] = selected["has_CTCF_motif"]
            selected_frames.append(selected)

        summary_rows.append(
            {
                "sampling_stratum": stratum,
                "requested_count": requested,
                "available_count": available,
                "selected_count": selected_count,
                "shortage_count": max(0, requested - available),
                "label_status": statuses,
            }
        )

    if selected_frames:
        sampled = pd.concat(selected_frames, ignore_index=True)
    else:
        sampled = labels.iloc[0:0].copy()
        sampled["sampling_stratum"] = pd.Series(dtype="object")
        sampled["sampling_rank"] = pd.Series(dtype="int64")
        sampled["sampling_requested_count"] = pd.Series(dtype="int64")
        sampled["sampling_seed"] = pd.Series(dtype="int64")
        sampled["mutation_available"] = pd.Series(dtype="bool")

    order = {stratum: index for index, (stratum, _) in enumerate(SAMPLING_STRATA)}
    sampled["_stratum_order"] = sampled["sampling_stratum"].map(order)
    sampled = (
        sampled.sort_values(["_stratum_order", "sampling_rank"])
        .drop(columns="_stratum_order")
        .reset_index(drop=True)
    )
    summary = pd.DataFrame.from_records(summary_rows)
    return sampled, summary


def run(
    *,
    config: RunConfig,
    config_path: str | Path,
    allow_provisional_labels: bool,
    overwrite_stage: bool,
) -> None:
    """Execute the sample stage."""

    root = output_dir(config)
    labels_path = root / "paper_label_table.tsv"
    site_metadata_path = root / "site_metadata.tsv"
    summary_path = root / "sampling_summary.tsv"
    if not labels_path.exists():
        raise LabelValidationError(
            f"Run the labels stage first; missing {labels_path}"
        )
    require_completed_stage(config, config_path, "labels")
    if overwrite_stage:
        reject_completed_downstream_stages(
            config,
            config_path,
            ("mutate", "infer", "features", "analyze", "report"),
        )
    prepare_stage_outputs(
        [site_metadata_path, summary_path],
        overwrite_stage,
    )

    labels = pd.read_csv(labels_path, sep="\t", low_memory=False)
    labels, _ = validate_paper_labels(
        labels,
        dic_mvalue_threshold=config.labels.dic_mvalue_threshold,
    )
    statuses = set(labels["label_status"])
    if statuses != {"canonical"} and not allow_provisional_labels:
        raise LabelValidationError(
            "Sampling provisional labels requires --allow-provisional-labels"
        )

    sampled, summary = sample_paper_sites(labels, config)
    sampled.to_csv(site_metadata_path, sep="\t", index=False)
    summary.to_csv(summary_path, sep="\t", index=False)
    label_status = "canonical" if statuses == {"canonical"} else "provisional"
    record_stage(
        config=config,
        config_path=config_path,
        stage="sample",
        inputs=[labels_path],
        outputs=[site_metadata_path, summary_path],
        label_status=label_status,
        metadata={
            "selected_count": int(len(sampled)),
            "sampling_seed": config.sampling.random_seed,
            "require_full_quota": config.sampling.require_full_quota,
            "selected_by_stratum": summary.set_index("sampling_stratum")[
                "selected_count"
            ].to_dict(),
        },
    )
