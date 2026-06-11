"""Representative-site selection and bounded second-pass plots."""

from __future__ import annotations

import re
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from grelu.interpret.ism.config import RunConfig
from grelu.interpret.ism.errors import ISMUserError
from grelu.interpret.ism.inference import write_representative_outputs
from grelu.interpret.ism.provenance import (
    output_dir,
    prepare_stage_outputs,
    record_stage,
    require_completed_stage,
)


class ArtifactError(ISMUserError, RuntimeError):
    """Raised when representative report artifacts cannot be produced."""


def _as_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    return (
        series.fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
        .isin({"true", "1", "yes"})
    )


def _safe_name(value: Any) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_")


def _representative_scores(root: Path, sites: pd.DataFrame) -> pd.DataFrame:
    adjusted = pd.read_csv(
        root / "control_adjusted_feature_matrix.tsv",
        sep="\t",
        low_memory=False,
    )
    score_columns = [
        column
        for column in adjusted.columns
        if column.endswith("__control_adjusted_score")
    ]
    if score_columns:
        numeric = adjusted.loc[:, score_columns].apply(
            pd.to_numeric, errors="coerce"
        )
        adjusted["representative_effect_score"] = numeric.abs().max(
            axis=1, skipna=True
        )
    else:
        adjusted["representative_effect_score"] = np.nan
    result = sites.merge(
        adjusted.loc[:, ["site_id", "representative_effect_score"]],
        on="site_id",
        how="left",
        validate="one_to_one",
    )
    assignment_path = root / "kmeans_k10_cluster_assignment.tsv"
    if assignment_path.exists():
        assignments = pd.read_csv(
            assignment_path, sep="\t", low_memory=False
        )
        result = result.merge(
            assignments.loc[:, ["site_id", "cluster", "PC1", "PC2"]],
            on="site_id",
            how="left",
            validate="one_to_one",
        )
    response_path = root / "contact_response_type.tsv"
    if response_path.exists():
        response = pd.read_csv(response_path, sep="\t", low_memory=False)
        result = result.merge(
            response, on="site_id", how="left", validate="one_to_one"
        )
    return result


def _select_rows(
    table: pd.DataFrame,
    mask: pd.Series,
    count: int,
) -> pd.DataFrame:
    return (
        table.loc[mask]
        .sort_values(
            ["representative_effect_score", "site_id"],
            ascending=[False, True],
            na_position="last",
            kind="stable",
        )
        .head(count)
        .copy()
    )


def _averaged_predictions(
    archive: np.lib.npyio.NpzFile,
) -> dict[str, np.ndarray]:
    labels = archive["prediction_labels"].astype(str)
    arrays = archive["predictions"]
    grouped: dict[str, list[np.ndarray]] = {}
    for label, array in zip(labels, arrays):
        _, perturbation_id = label.split(":", 1)
        grouped.setdefault(perturbation_id, []).append(array)
    return {
        perturbation_id: np.mean(np.stack(values), axis=0)
        for perturbation_id, values in grouped.items()
    }


def _plot_one_d(
    *,
    site_id: str,
    archive: np.lib.npyio.NpzFile,
    predictions: dict[str, np.ndarray],
    motif_perturbation_id: str,
    destination: Path,
) -> None:
    import matplotlib.pyplot as plt

    ref_id = f"{site_id}:REF"
    if ref_id not in predictions or motif_perturbation_id not in predictions:
        return
    delta = predictions[motif_perturbation_id] - predictions[ref_id]
    names = archive["track_names"].astype(str)
    track_order = np.argsort(np.nanmax(np.abs(delta), axis=1))[::-1][:8]
    window_bp = int(archive["window_bp"])
    x = np.linspace(-window_bp / 2, window_bp / 2, delta.shape[-1])
    figure, axis = plt.subplots(figsize=(11, 6))
    for index in track_order:
        axis.plot(x, delta[index], linewidth=1.1, label=names[index])
    axis.axvline(0, color="black", linewidth=0.8, linestyle="--")
    axis.axhline(0, color="black", linewidth=0.6)
    axis.set_xlabel("Position relative to ISM anchor (bp)")
    axis.set_ylabel("ALT - REF prediction")
    axis.set_title(f"{site_id}: local 1D motif-disruption delta")
    axis.legend(fontsize=7, loc="best")
    figure.savefig(destination, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _plot_contact(
    *,
    site_id: str,
    archive: np.lib.npyio.NpzFile,
    predictions: dict[str, np.ndarray],
    motif_perturbation_id: str,
    destination: Path,
) -> None:
    import matplotlib.pyplot as plt

    ref_id = f"{site_id}:REF"
    if ref_id not in predictions or motif_perturbation_id not in predictions:
        return
    delta = np.nanmean(
        predictions[motif_perturbation_id] - predictions[ref_id], axis=0
    )
    limit = float(np.nanpercentile(np.abs(delta), 99))
    if not np.isfinite(limit) or limit <= 0:
        limit = 1.0
    window_bp = int(archive["window_bp"])
    extent = (
        -window_bp / 2000,
        window_bp / 2000,
        window_bp / 2000,
        -window_bp / 2000,
    )
    figure, axis = plt.subplots(figsize=(7, 6))
    image = axis.imshow(
        delta,
        cmap="coolwarm",
        vmin=-limit,
        vmax=limit,
        extent=extent,
        interpolation="nearest",
    )
    axis.set_xlabel("Position relative to anchor (kb)")
    axis.set_ylabel("Position relative to anchor (kb)")
    axis.set_title(f"{site_id}: mean contact ALT - REF")
    figure.colorbar(image, ax=axis, label="Contact prediction delta")
    figure.savefig(destination, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _render_representative_plots(
    *,
    prediction_dir: Path,
    mutation_table: pd.DataFrame,
    one_d_dir: Path,
    contact_dir: Path,
) -> pd.DataFrame:
    index = pd.read_csv(
        prediction_dir / "representative_prediction_index.tsv",
        sep="\t",
        low_memory=False,
    )
    motif_ids = (
        mutation_table.loc[
            mutation_table["perturbation_type"].eq("ctcf_motif_disruption"),
            ["site_id", "perturbation_id"],
        ]
        .drop_duplicates("site_id")
        .set_index("site_id")["perturbation_id"]
        .astype(str)
        .to_dict()
    )
    one_d_dir.mkdir(parents=True, exist_ok=True)
    contact_dir.mkdir(parents=True, exist_ok=True)
    columns = [
        "site_id",
        "output_type",
        "resolution",
        "plot_type",
        "relative_path",
        "motif_perturbation_id",
    ]
    rows = []
    for row in index.itertuples(index=False):
        site_id = str(row.site_id)
        motif_id = motif_ids.get(site_id)
        if motif_id is None:
            continue
        path = prediction_dir / str(row.relative_path)
        with np.load(path, allow_pickle=False) as archive:
            predictions = _averaged_predictions(archive)
            stem = _safe_name(
                f"{site_id}__{row.output_type}__r{row.resolution}"
            )
            if str(row.output_type) == "contact_maps":
                destination = contact_dir / f"{stem}.png"
                _plot_contact(
                    site_id=site_id,
                    archive=archive,
                    predictions=predictions,
                    motif_perturbation_id=motif_id,
                    destination=destination,
                )
                plot_type = "contact_delta"
            else:
                destination = one_d_dir / f"{stem}.png"
                _plot_one_d(
                    site_id=site_id,
                    archive=archive,
                    predictions=predictions,
                    motif_perturbation_id=motif_id,
                    destination=destination,
                )
                plot_type = "local_1d_delta"
        if destination.exists():
            rows.append(
                {
                    "site_id": site_id,
                    "output_type": row.output_type,
                    "resolution": row.resolution,
                    "plot_type": plot_type,
                    "relative_path": str(destination.relative_to(one_d_dir.parent)),
                    "motif_perturbation_id": motif_id,
                }
            )
    return pd.DataFrame.from_records(rows, columns=columns)


def run_report(
    *,
    config: RunConfig,
    config_path: str | Path,
    allow_provisional_labels: bool,
    overwrite_stage: bool,
) -> None:
    """Select representatives, run the bounded second pass, and render plots."""

    root = output_dir(config)
    table_paths = {
        "hc": root / "representative_hc_dic_examples.tsv",
        "lc": root / "representative_lc_dic_examples.tsv",
        "stable": root / "representative_stable_control_examples.tsv",
        "spec": root / "representative_plot_spec.tsv",
    }
    one_d_dir = root / "representative_local_1d_delta_plots"
    contact_dir = root / "representative_contact_delta_plots"
    prediction_dir = root / "representative_predictions"
    require_completed_stage(config, config_path, "analyze")
    sites_path = root / "site_metadata.tsv"
    mutation_path = root / "mutation_table.tsv"
    for path in (
        sites_path,
        mutation_path,
        root / "control_adjusted_feature_matrix.tsv",
        root / "kmeans_k10_cluster_assignment.tsv",
    ):
        if not path.exists():
            raise ArtifactError(f"Required analysis artifact is missing: {path}")
    prepare_stage_outputs(table_paths.values(), overwrite_stage)
    for directory in (one_d_dir, contact_dir, prediction_dir):
        if directory.exists():
            if not overwrite_stage:
                raise ArtifactError(
                    f"Representative output already exists: {directory}"
                )
            shutil.rmtree(directory)

    sites = pd.read_csv(sites_path, sep="\t", low_memory=False)
    statuses = set(sites["label_status"].astype(str))
    if statuses != {"canonical"} and not allow_provisional_labels:
        raise ArtifactError(
            "Reporting for provisional labels requires "
            "--allow-provisional-labels"
        )
    scored = _representative_scores(root, sites)
    count = config.model.representative_site_count_per_class
    selected = {
        "hc": _select_rows(scored, _as_bool(scored["HC_DIC_label"]), count),
        "lc": _select_rows(scored, _as_bool(scored["LC_DIC_label"]), count),
        "stable": _select_rows(
            scored,
            scored["sampling_stratum"].astype(str).eq("stable_intragenic"),
            count,
        ),
    }
    for key, table in selected.items():
        table.to_csv(table_paths[key], sep="\t", index=False)
    representative_ids = list(
        dict.fromkeys(
            site_id
            for table in selected.values()
            for site_id in table["site_id"].astype(str)
        )
    )
    mutation_table = pd.read_csv(
        mutation_path, sep="\t", low_memory=False
    )
    eligible_ids = set(
        mutation_table.loc[
            mutation_table["perturbation_type"].eq(
                "ctcf_motif_disruption"
            ),
            "site_id",
        ].astype(str)
    )
    representative_ids = [
        site_id for site_id in representative_ids if site_id in eligible_ids
    ]
    if representative_ids:
        write_representative_outputs(
            config=config,
            site_ids=representative_ids,
            destination=prediction_dir,
        )
        plot_spec = _render_representative_plots(
            prediction_dir=prediction_dir,
            mutation_table=mutation_table,
            one_d_dir=one_d_dir,
            contact_dir=contact_dir,
        )
    else:
        prediction_dir.mkdir(parents=True)
        one_d_dir.mkdir(parents=True)
        contact_dir.mkdir(parents=True)
        plot_spec = pd.DataFrame(
            columns=[
                "site_id",
                "output_type",
                "resolution",
                "plot_type",
                "relative_path",
                "motif_perturbation_id",
            ]
        )
    plot_spec.to_csv(table_paths["spec"], sep="\t", index=False)

    label_status = "canonical" if statuses == {"canonical"} else "provisional"
    record_stage(
        config=config,
        config_path=config_path,
        stage="report",
        inputs=[
            sites_path,
            mutation_path,
            root / "control_adjusted_feature_matrix.tsv",
            root / "kmeans_k10_cluster_assignment.tsv",
        ],
        outputs=[
            *table_paths.values(),
            one_d_dir,
            contact_dir,
            prediction_dir,
        ],
        label_status=label_status,
        metadata={
            "selected_representative_count": len(representative_ids),
            "plot_count": len(plot_spec),
            "representative_second_pass": True,
            "population_raw_tensor_reused": False,
        },
    )
