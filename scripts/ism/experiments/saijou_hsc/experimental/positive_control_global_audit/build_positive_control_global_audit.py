#!/usr/bin/env python
"""Build an English positive-control-only global audit PDF.

The report reuses the validated nine-gene fine-tuned score tables and the
frozen original-model modality-view calculation. Non-control high-score
regions are shown only as genomic background and are not annotated or
interpreted as candidates.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Patch
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[6]
ORIGINAL_CODE = (
    REPO_ROOT
    / "scripts/ism/experiments/saijou_hsc/experimental"
    / "original_multitrack_score"
)
if str(ORIGINAL_CODE) not in sys.path:
    sys.path.insert(0, str(ORIGINAL_CODE))

import profile_features  # noqa: E402
import score_calculations as scoring  # noqa: E402


DEFAULT_FINE_ROOT = (
    REPO_ROOT
    / "experiments/ism"
    / "saijou_nine_gene_tss_3kb_strict_shuffle_pdf_transcripts"
)
DEFAULT_ORIGINAL_ROOT = (
    REPO_ROOT / "experiments/ism/original_multitrack_score_20260728"
)
DEFAULT_OUTPUT = (
    REPO_ROOT / "experiments/ism/positive_control_global_audit_20260729"
)
CONFIG_ROOT = REPO_ROOT / "scripts/ism/experiments/saijou_hsc/configs"
CONTROL_REGISTRY = CONFIG_ROOT / "positive_control_registry.tsv"
TRANSCRIPT_AUTHORITY = CONFIG_ROOT / "provided_nine_gene_transcripts.tsv"

CONTROL_ORDER = [
    "acta2_srf_carg",
    "col1a1_ap1",
    "col1a1_sp1_klf6",
    "col1a1_smad3_smad4",
    "col1a2_smad3",
    "col1a2_ap1",
    "timp1_ap1",
]
ORIGINAL_RUNS = {
    "AlphaGenome": {
        "Acta2": "alphagenome_original_acta2",
        "Col1a1": "alphagenome_original_col1a1",
        "Col1a2": "alphagenome_original_col1a2",
        "Timp1": "alphagenome_original_timp1",
    },
    "Borzoi": {
        "Acta2": "borzoi_original_acta2",
        "Col1a1": "borzoi_original_col1a1",
        "Col1a2": "borzoi_original_col1a2",
        "Timp1": "borzoi_original_timp1",
    },
}
OUTPUT_MODALITIES = {
    "AlphaGenome": {"rna_seq", "cage"},
    "Borzoi": {"rna", "cage"},
}
EXPECTED_TRACKS = {"AlphaGenome": 28, "Borzoi": 65}
EXPECTED_CENTERS = {
    "Acta2": 1495,
    "Col1a1": 1494,
    "Col1a2": 1496,
    "Timp1": 1496,
}
EXPECTED_SOURCE_READOUTS = {
    "Acta2": 1,
    "Col1a1": 1,
    "Col1a2": 5,
    "Timp1": 5,
}
COLORS = {
    "AlphaGenome": "#2364AA",
    "Borzoi": "#D97706",
    "Fine-tuned": "#111827",
}
THRESHOLD = 95.0
REVIEW_THRESHOLD = 90.0
LOCAL_HALF_WIDTH_BP = 150


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fine-root", type=Path, default=DEFAULT_FINE_ROOT)
    parser.add_argument(
        "--original-root", type=Path, default=DEFAULT_ORIGINAL_ROOT
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def require_columns(
    frame: pd.DataFrame, required: set[str], label: str
) -> None:
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{label} missing columns: {missing}")


def load_json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text())
    if payload.get("status") != "ok":
        raise ValueError(f"{path}: validation status is not ok")
    return payload


def load_controls() -> tuple[pd.DataFrame, pd.DataFrame]:
    controls = pd.read_csv(CONTROL_REGISTRY, sep="\t")
    transcripts = pd.read_csv(TRANSCRIPT_AUTHORITY, sep="\t")
    if set(controls.control_id) != set(CONTROL_ORDER):
        raise ValueError(
            "The live control registry does not match the frozen seven-control "
            f"report scope: {sorted(controls.control_id)}"
        )
    controls["control_id"] = pd.Categorical(
        controls.control_id, categories=CONTROL_ORDER, ordered=True
    )
    controls = controls.sort_values("control_id").reset_index(drop=True)
    controls["control_id"] = controls.control_id.astype(str)
    controls = controls.merge(
        transcripts[
            [
                "gene",
                "transcript_name",
                "transcript_id",
                "chrom",
                "strand",
                "analysis_tss_0based",
            ]
        ],
        on="gene",
        how="left",
        validate="many_to_one",
    )
    if controls.transcript_name.isna().any():
        raise ValueError("Missing transcript authority for a registered control")
    return controls, transcripts


def validate_original_run(
    root: Path, model: str, gene: str, run_name: str
) -> dict[str, object]:
    run = root / "runs" / run_name
    validation = load_json(run / "validation_summary.json")
    expected_backend = (
        "alphagenome_original" if model == "AlphaGenome" else "borzoi_original"
    )
    checks = {
        "model_backend": validation.get("model_backend") == expected_backend,
        "gene": validation.get("genes") == [gene],
        "mutations": validation.get("mutations") == 3 * EXPECTED_CENTERS[gene],
        "tracks": validation.get("selected_tracks") == EXPECTED_TRACKS[model],
        "source_readouts": (
            validation.get("readouts") == EXPECTED_SOURCE_READOUTS[gene]
        ),
        "profiles": validation.get("saved_log2fc_profiles") is True,
        "nonfinite": validation.get("stored_profile_nonfinite_values") == 0,
    }
    if not all(checks.values()):
        raise ValueError(f"{run}: validation mismatch {checks}")
    return validation


def build_output_view(
    *,
    root: Path,
    gene: str,
    run_name: str,
    output_modalities: set[str],
) -> pd.DataFrame:
    run = root / "runs" / run_name
    manifest = profile_features.load_validated_track_manifest(root, run_name)
    track_metadata = manifest[
        ["track_id", "track_group", "modality"]
    ].drop_duplicates("track_id")
    output_tracks = track_metadata.loc[
        track_metadata.modality.isin(output_modalities)
    ]
    features = pd.read_parquet(
        run / "features" / "combined_mutation_features.parquet"
    )
    required = {
        "gene",
        "variant_offset_from_tss_transcription_bp",
        "replacement_replicate",
        "track_id",
        "track_group",
        "readout_role",
        "log2fc_ratio_of_sums",
    }
    require_columns(features, required, f"{run_name} features")
    output = features.loc[
        features.gene.eq(gene)
        & features.readout_role.eq("tss_1024bp")
        & features.track_id.isin(output_tracks.track_id)
    ].copy()
    output["absolute_log2fc"] = output.log2fc_ratio_of_sums.abs()
    output = (
        output.groupby(
            [*scoring.CENTER_KEYS, "track_id", "track_group"], sort=False
        )
        .agg(
            replacements=("replacement_replicate", "nunique"),
            median_absolute_log2fc=("absolute_log2fc", "median"),
            median_signed_log2fc=("log2fc_ratio_of_sums", "median"),
        )
        .reset_index()
        .merge(
            output_tracks,
            on=["track_id", "track_group"],
            how="left",
            validate="many_to_one",
        )
    )
    output["score_view"] = "output"
    output["readout_definition"] = "tss_1024bp_ratio_of_sums"
    return output


def score_original_model(
    root: Path, model: str
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, object]]]:
    track_parts: list[pd.DataFrame] = []
    validations: list[dict[str, object]] = []
    for gene, run_name in ORIGINAL_RUNS[model].items():
        validations.append(
            validate_original_run(root, model, gene, run_name)
        )
        output = build_output_view(
            root=root,
            gene=gene,
            run_name=run_name,
            output_modalities=OUTPUT_MODALITIES[model],
        )
        local = profile_features.summarize_local_profile_effects(
            root,
            gene=gene,
            run_name=run_name,
            output_modalities=OUTPUT_MODALITIES[model],
        )
        track_parts.extend([output, local])
    tracks = pd.concat(track_parts, ignore_index=True).sort_values(
        [*scoring.CENTER_KEYS, "score_view", "track_group", "track_id"]
    )
    groups, views, centers = scoring.score_modality_views(tracks, THRESHOLD)
    observed_centers = centers.groupby("gene").size().astype(int).to_dict()
    if observed_centers != EXPECTED_CENTERS:
        raise ValueError(
            f"{model}: center-count mismatch {observed_centers}"
        )
    if not np.isfinite(
        centers[
            [
                "importance_score",
                "max_view_score_raw",
                "median_group_signed_log2fc",
            ]
        ].to_numpy(float)
    ).all():
        raise ValueError(f"{model}: non-finite center score")
    if centers.duplicated(scoring.CENTER_KEYS).any():
        raise ValueError(f"{model}: duplicate center keys")
    return centers, views, validations


def fine_tuned_window_recall(
    fine_centers: pd.DataFrame, controls: pd.DataFrame
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for control in controls.itertuples(index=False):
        gene_centers = fine_centers.loc[
            fine_centers.gene.eq(control.gene)
        ].sort_values("variant_offset_from_tss_transcription_bp")
        subset = gene_centers.loc[
            gene_centers.variant_offset_from_tss_transcription_bp.between(
                int(control.start) - 5, int(control.end) + 5
            )
        ]
        if subset.empty:
            raise ValueError(f"No fine-tuned centers for {control.control_id}")
        best = subset.nlargest(1, "importance_score").iloc[0]
        window = scoring.score_same_width_window(
            gene_centers, subset, "importance_score"
        )
        rows.append(
            {
                **control._asdict(),
                "covered_centers": int(len(subset)),
                "best_center": int(
                    best.variant_offset_from_tss_transcription_bp
                ),
                "best_score": float(best.importance_score),
                "window_statistic": window.statistic,
                "same_width_background_windows": window.background_windows,
                "window_percentile_score": window.percentile_score,
                "window_recalled_at_threshold": bool(
                    window.percentile_score > THRESHOLD
                ),
            }
        )
    return pd.DataFrame.from_records(rows)


def connected_high_region(
    centers: pd.DataFrame,
    *,
    gene: str,
    start: int,
    end: int,
    threshold: float,
) -> tuple[int, int | None, int | None]:
    group = centers.loc[centers.gene.eq(gene)].sort_values(
        "variant_offset_from_tss_transcription_bp"
    )
    selected = group.loc[group.importance_score >= threshold].copy()
    if selected.empty:
        return 0, None, None
    selected["block"] = (
        selected.variant_offset_from_tss_transcription_bp.diff()
        .fillna(5)
        .gt(4)
        .cumsum()
    )
    motif_lo, motif_hi = start - 5, end + 5
    for _, block in selected.groupby("block", sort=False):
        lo = int(block.variant_offset_from_tss_transcription_bp.min())
        hi = int(block.variant_offset_from_tss_transcription_bp.max())
        if hi >= motif_lo and lo <= motif_hi:
            return hi - lo, lo, hi
    return 0, None, None


def local_contrast(
    centers: pd.DataFrame, *, gene: str, start: int, end: int
) -> float:
    group = centers.loc[centers.gene.eq(gene)]
    motif = group.loc[
        group.variant_offset_from_tss_transcription_bp.between(
            start - 5, end + 5
        )
    ]
    flank = group.loc[
        group.variant_offset_from_tss_transcription_bp.between(
            start - 100, end + 100
        )
        & ~group.index.isin(motif.index)
    ]
    if motif.empty or flank.empty:
        return float("nan")
    return float(
        motif.importance_score.median() - flank.importance_score.median()
    )


def original_interval_recall(
    centers: pd.DataFrame, controls: pd.DataFrame
) -> pd.DataFrame:
    return scoring.evaluate_interval_recall(
        centers,
        controls,
        THRESHOLD,
        interval_kind="registered_positive_control",
        window_statistic_column="max_view_score_raw",
    )


def build_evidence_table(
    *,
    controls: pd.DataFrame,
    fine_centers: pd.DataFrame,
    fine_recall: pd.DataFrame,
    ag_centers: pd.DataFrame,
    bz_centers: pd.DataFrame,
    ag_recall: pd.DataFrame,
    bz_recall: pd.DataFrame,
    motif_annotations: pd.DataFrame,
) -> pd.DataFrame:
    evidence = controls[
        [
            "control_id",
            "gene",
            "label",
            "family",
            "start",
            "end",
            "evidence",
            "transcript_name",
            "transcript_id",
            "chrom",
            "strand",
        ]
    ].copy()
    for prefix, recall in (
        ("finetuned", fine_recall),
        ("original_alphagenome", ag_recall),
        ("original_borzoi", bz_recall),
    ):
        keep = recall[
            [
                "control_id",
                "covered_centers",
                "best_center",
                "best_score",
                "window_percentile_score",
                "window_recalled_at_threshold",
            ]
        ].rename(
            columns={
                column: f"{prefix}_{column}"
                for column in recall.columns
                if column != "control_id"
            }
        )
        evidence = evidence.merge(
            keep, on="control_id", how="left", validate="one_to_one"
        )

    motif = motif_annotations.loc[
        motif_annotations.scan_kind.eq("registered_positive_control"),
        [
            "gene",
            "region_start",
            "region_end",
            "annotation_status",
            "matches_expected_family",
            "top_motif",
        ],
    ].rename(columns={"region_start": "start", "region_end": "end"})
    evidence = evidence.merge(
        motif,
        on=["gene", "start", "end"],
        how="left",
        validate="one_to_one",
    )

    center_sets = {
        "finetuned": fine_centers,
        "original_alphagenome": ag_centers,
        "original_borzoi": bz_centers,
    }
    shape_rows: list[dict[str, object]] = []
    for control in controls.itertuples(index=False):
        row: dict[str, object] = {"control_id": control.control_id}
        midpoint = (int(control.start) + int(control.end)) / 2
        for prefix, centers in center_sets.items():
            recall = (
                fine_recall
                if prefix == "finetuned"
                else ag_recall
                if prefix == "original_alphagenome"
                else bz_recall
            )
            hit = recall.loc[
                recall.control_id.eq(control.control_id)
            ].iloc[0]
            width, region_start, region_end = connected_high_region(
                centers,
                gene=control.gene,
                start=int(control.start),
                end=int(control.end),
                threshold=REVIEW_THRESHOLD,
            )
            row[f"{prefix}_peak_distance_bp"] = float(
                abs(float(hit.best_center) - midpoint)
            )
            row[f"{prefix}_score90_region_width_bp"] = int(width)
            row[f"{prefix}_score90_region_start"] = region_start
            row[f"{prefix}_score90_region_end"] = region_end
            row[f"{prefix}_local_contrast"] = local_contrast(
                centers,
                gene=control.gene,
                start=int(control.start),
                end=int(control.end),
            )
        shape_rows.append(row)
    evidence = evidence.merge(
        pd.DataFrame.from_records(shape_rows),
        on="control_id",
        how="left",
        validate="one_to_one",
    )
    return evidence


def set_report_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.edgecolor": "#9CA3AF",
            "axes.labelcolor": "#374151",
            "axes.titlecolor": "#111827",
            "xtick.color": "#4B5563",
            "ytick.color": "#4B5563",
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def save_page(
    fig: plt.Figure,
    *,
    pdf: PdfPages,
    output_dir: Path,
    stem: str,
) -> Path:
    path = output_dir / f"{stem}.png"
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    pdf.savefig(fig, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def add_wrapped_text(
    fig: plt.Figure,
    text: str,
    *,
    x: float,
    y: float,
    width: int,
    fontsize: float = 10,
    color: str = "#374151",
    line_spacing: float = 1.35,
    weight: str = "normal",
) -> float:
    wrapped = textwrap.fill(text, width=width)
    lines = wrapped.count("\n") + 1
    fig.text(
        x,
        y,
        wrapped,
        ha="left",
        va="top",
        fontsize=fontsize,
        color=color,
        linespacing=line_spacing,
        fontweight=weight,
    )
    return y - lines * fontsize / 780 * line_spacing


def render_summary_page(
    *,
    evidence: pd.DataFrame,
    pdf: PdfPages,
    output_dir: Path,
) -> Path:
    fig = plt.figure(figsize=(11.7, 8.3))
    fig.text(
        0.07,
        0.93,
        "Positive-control global sensitivity audit",
        fontsize=22,
        fontweight="bold",
        color="#111827",
        ha="left",
    )
    fig.text(
        0.07,
        0.885,
        (
            "Seven registered motifs across four genes; original and "
            "fine-tuned AlphaGenome/Borzoi; mouse mm10"
        ),
        fontsize=11,
        color="#4B5563",
        ha="left",
    )
    metrics = [
        (
            "Fine-tuned windows >95",
            int(evidence.finetuned_window_recalled_at_threshold.sum()),
            "of 7",
        ),
        (
            "Original AG windows >95",
            int(
                evidence.original_alphagenome_window_recalled_at_threshold.sum()
            ),
            "of 7",
        ),
        (
            "Original BZ windows >95",
            int(evidence.original_borzoi_window_recalled_at_threshold.sum()),
            "of 7",
        ),
        (
            "Expected motif loss",
            int(evidence.matches_expected_family.fillna(0).sum()),
            "of 7",
        ),
    ]
    for index, (label, value, denominator) in enumerate(metrics):
        x = 0.07 + index * 0.225
        fig.text(
            x,
            0.79,
            str(value),
            fontsize=28,
            fontweight="bold",
            color="#111827",
        )
        fig.text(x + 0.035, 0.79, denominator, fontsize=10, color="#6B7280")
        fig.text(x, 0.745, label, fontsize=8.5, color="#4B5563")
    y = 0.66
    sections = [
        (
            "Technical summary",
            (
                "This report asks only whether registered positive-control "
                "intervals are globally prominent within their own 3-kb scan, "
                "whether the local high-score shape is centered on the motif, "
                "and whether original-versus-fine-tuned evidence is stable. "
                "Other high-score regions are visible as background but are "
                "not annotated, ranked as candidates, or interpreted."
            ),
        ),
        (
            "How to read the score",
            (
                "Each score is an empirical within-gene rank. Original "
                "AlphaGenome, original Borzoi, and the fine-tuned combined "
                "score are separately calibrated and must not be numerically "
                "subtracted. Formal original-model recall uses a same-width "
                "window percentile; the report recomputes an analogous "
                "descriptive window rank from the fine-tuned importance "
                "profile."
            ),
        ),
        (
            "What a broad high-score neighborhood means",
            (
                "Adjacent centers are correlated because 10-bp edits with a "
                "2-bp stride can disrupt the same motif repeatedly. A high "
                "plateau is therefore not independent replication. The atlas "
                "pairs window rank with peak offset, score-90 region width, "
                "and signed effects to distinguish a centered footprint from "
                "a motif embedded in broad regional sensitivity."
            ),
        ),
    ]
    for heading, body in sections:
        fig.text(
            0.07,
            y,
            heading,
            fontsize=13,
            fontweight="bold",
            color="#111827",
            va="top",
        )
        y = add_wrapped_text(
            fig, body, x=0.07, y=y - 0.038, width=112, fontsize=10
        ) - 0.045
    fig.text(
        0.07,
        0.045,
        (
            "Thresholds 90 and 95 are review and discovery ranks, not "
            "genome-wide significance thresholds, p-values, or FDR."
        ),
        fontsize=8.5,
        color="#6B7280",
    )
    return save_page(
        fig,
        pdf=pdf,
        output_dir=output_dir,
        stem="01_technical_summary",
    )


def profile_for_control(
    centers: pd.DataFrame, control: pd.Series
) -> pd.DataFrame:
    return centers.loc[centers.gene.eq(control.gene)].sort_values(
        "variant_offset_from_tss_transcription_bp"
    )


def render_global_atlas(
    *,
    controls: pd.DataFrame,
    profiles: dict[str, pd.DataFrame],
    pdf: PdfPages,
    output_dir: Path,
) -> Path:
    score_cmap = LinearSegmentedColormap.from_list(
        "score",
        ["#F9FAFB", "#DBEAFE", "#93C5FD", "#2364AA", "#111827"],
        N=256,
    )
    fig, axes = plt.subplots(
        len(controls),
        3,
        figsize=(15.0, 10.2),
        sharex=True,
        gridspec_kw={"hspace": 0.22, "wspace": 0.08},
    )
    columns = ["Original AlphaGenome", "Original Borzoi", "Fine-tuned"]
    profile_keys = [
        "original_alphagenome",
        "original_borzoi",
        "finetuned",
    ]
    for column, title in enumerate(columns):
        axes[0, column].set_title(title, fontsize=11, fontweight="bold")
    for row_index, control in controls.iterrows():
        for column_index, key in enumerate(profile_keys):
            axis = axes[row_index, column_index]
            profile = profile_for_control(profiles[key], control)
            x = profile.variant_offset_from_tss_transcription_bp.to_numpy(float)
            score = profile.importance_score.to_numpy(float)
            edges = np.concatenate(
                [[x[0] - 1], (x[:-1] + x[1:]) / 2, [x[-1] + 1]]
            )
            axis.pcolormesh(
                edges,
                [0, 1],
                score[np.newaxis, :],
                cmap=score_cmap,
                norm=Normalize(0, 100),
                shading="flat",
            )
            axis.axvspan(
                int(control.start),
                int(control.end),
                facecolor="none",
                edgecolor="#111827",
                linewidth=1.3,
                hatch="////",
            )
            axis.set_ylim(0, 1)
            axis.set_xlim(-1500, 1500)
            axis.set_yticks([])
            axis.spines[["top", "right", "left"]].set_visible(False)
            axis.tick_params(axis="x", labelsize=7)
            if column_index == 0:
                axis.set_ylabel(
                    f"{control.gene}\n{control.label}",
                    rotation=0,
                    ha="right",
                    va="center",
                    labelpad=54,
                    fontsize=8.3,
                )
            if row_index != len(controls) - 1:
                axis.tick_params(labelbottom=False)
    for axis in axes[-1]:
        axis.set_xlabel(
            "Position relative to TSS (bp, transcription direction)",
            fontsize=8,
        )
    fig.suptitle(
        "Full 3-kb importance-rank landscape around registered controls",
        x=0.06,
        y=0.985,
        ha="left",
        fontsize=16,
        fontweight="bold",
    )
    fig.text(
        0.06,
        0.948,
        (
            "Dark regions are high within-gene ranks. Hatched boxes mark the "
            "registered motif intervals; non-control peaks are context only."
        ),
        ha="left",
        fontsize=9.3,
        color="#4B5563",
    )
    color_axis = fig.add_axes([0.91, 0.20, 0.012, 0.58])
    scalar = plt.cm.ScalarMappable(
        norm=Normalize(0, 100), cmap=score_cmap
    )
    colorbar = fig.colorbar(scalar, cax=color_axis)
    colorbar.set_label("Within-gene importance rank", fontsize=8)
    colorbar.ax.tick_params(labelsize=7)
    fig.text(
        0.06,
        0.015,
        (
            "Original columns use the frozen modality-view Method 3; the "
            "fine-tuned column uses the nine-gene combined rank. The three "
            "columns share a 0–100 display scale but not a subtractable metric."
        ),
        fontsize=8,
        color="#6B7280",
    )
    fig.subplots_adjust(
        left=0.14, right=0.89, top=0.91, bottom=0.08
    )
    return save_page(
        fig,
        pdf=pdf,
        output_dir=output_dir,
        stem="02_global_rank_atlas",
    )


def short_control_label(row: pd.Series) -> str:
    return f"{row.gene} {row.label}"


def render_evidence_matrix(
    *,
    evidence: pd.DataFrame,
    pdf: PdfPages,
    output_dir: Path,
) -> Path:
    fig = plt.figure(figsize=(14.2, 9.2))
    grid = fig.add_gridspec(
        2,
        1,
        height_ratios=[1.15, 1.0],
        left=0.24,
        right=0.96,
        top=0.86,
        bottom=0.08,
        hspace=0.34,
    )
    axis = fig.add_subplot(grid[0])
    y = np.arange(len(evidence))
    series = [
        ("Fine-tuned", "finetuned_window_percentile_score", "o", "#111827"),
        (
            "Original AlphaGenome",
            "original_alphagenome_window_percentile_score",
            "s",
            COLORS["AlphaGenome"],
        ),
        (
            "Original Borzoi",
            "original_borzoi_window_percentile_score",
            "^",
            COLORS["Borzoi"],
        ),
    ]
    offsets = [-0.16, 0.0, 0.16]
    for offset, (label, column, marker, color) in zip(
        offsets, series, strict=True
    ):
        axis.scatter(
            evidence[column],
            y + offset,
            marker=marker,
            s=52,
            color=color,
            edgecolor="white",
            linewidth=0.6,
            label=label,
            zorder=3,
        )
    axis.axvline(
        THRESHOLD,
        color="#111827",
        linestyle=(0, (5, 3)),
        linewidth=1.0,
    )
    axis.axvline(
        REVIEW_THRESHOLD,
        color="#6B7280",
        linestyle=(0, (2, 3)),
        linewidth=1.0,
    )
    axis.set_xlim(0, 101)
    axis.set_yticks(y)
    axis.set_yticklabels(
        [short_control_label(row) for _, row in evidence.iterrows()],
        fontsize=8.7,
    )
    axis.invert_yaxis()
    axis.set_xlabel("Same-width window percentile within the gene")
    axis.grid(axis="x", color="#E5E7EB", linewidth=0.7)
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=3,
        frameon=False,
        fontsize=8.5,
    )

    table_axis = fig.add_subplot(grid[1])
    table_axis.axis("off")
    table_rows: list[list[str]] = []
    for _, row in evidence.iterrows():
        loss = (
            "Confirmed"
            if bool(row.matches_expected_family)
            else "Not confirmed"
        )
        table_rows.append(
            [
                short_control_label(row),
                f"{int(row.finetuned_score90_region_width_bp)}",
                f"{int(row.original_alphagenome_score90_region_width_bp)}",
                f"{int(row.original_borzoi_score90_region_width_bp)}",
                f"{row.finetuned_peak_distance_bp:.1f}",
                loss,
            ]
        )
    table = table_axis.table(
        cellText=table_rows,
        colLabels=[
            "Control",
            "FT width ≥90\n(bp)",
            "Original AG\nwidth ≥90 (bp)",
            "Original BZ\nwidth ≥90 (bp)",
            "FT peak offset\n(bp)",
            "Expected motif loss",
        ],
        cellLoc="center",
        colLoc="center",
        loc="center",
        colWidths=[0.27, 0.13, 0.16, 0.16, 0.13, 0.15],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(7.8)
    table.scale(1, 1.55)
    for (row, column), cell in table.get_celld().items():
        cell.set_edgecolor("#D1D5DB")
        cell.set_linewidth(0.6)
        if row == 0:
            cell.set_facecolor("#F3F4F6")
            cell.set_text_props(fontweight="bold", color="#111827")
        elif column == 0:
            cell.set_text_props(ha="left")
    fig.suptitle(
        "Window-level recall and local high-score shape",
        x=0.055,
        y=0.96,
        ha="left",
        fontsize=16,
        fontweight="bold",
    )
    fig.text(
        0.055,
        0.915,
        (
            "Window ranks test global prominence; score-90 region width and "
            "peak offset describe whether the motif is centered within a "
            "compact footprint or embedded in a broad plateau."
        ),
        fontsize=9.2,
        color="#4B5563",
    )
    fig.text(
        0.055,
        0.025,
        (
            "The fine-tuned window percentile is descriptive and is computed "
            "from the fine-tuned rank profile. Original window percentiles use "
            "the frozen raw support statistic. Neither is an FDR."
        ),
        fontsize=8,
        color="#6B7280",
    )
    return save_page(
        fig,
        pdf=pdf,
        output_dir=output_dir,
        stem="03_evidence_matrix",
    )


def window_subset(
    frame: pd.DataFrame,
    control: pd.Series,
    half_width: int = LOCAL_HALF_WIDTH_BP,
) -> pd.DataFrame:
    midpoint = (int(control.start) + int(control.end)) / 2
    return frame.loc[
        frame.gene.eq(control.gene)
        & frame.variant_offset_from_tss_transcription_bp.between(
            midpoint - half_width, midpoint + half_width
        )
    ].sort_values("variant_offset_from_tss_transcription_bp")


def view_at_control(
    centers: pd.DataFrame, control: pd.Series
) -> str:
    subset = centers.loc[
        centers.gene.eq(control.gene)
        & centers.variant_offset_from_tss_transcription_bp.between(
            int(control.start) - 5, int(control.end) + 5
        )
    ]
    if subset.empty:
        raise ValueError(f"No original center for {control.control_id}")
    return str(subset.nlargest(1, "importance_score").iloc[0].driving_view)


def shared_effect_limit(
    *,
    controls: pd.DataFrame,
    ag_centers: pd.DataFrame,
    bz_centers: pd.DataFrame,
    ag_views: pd.DataFrame,
    bz_views: pd.DataFrame,
    fine_effects: pd.DataFrame,
) -> float:
    values: list[np.ndarray] = []
    fine_hsc = fine_effects.loc[
        fine_effects.cell.eq("hsc")
        & fine_effects.readout_role.eq("gene_body_output_clipped")
    ]
    for _, control in controls.iterrows():
        for centers, views in (
            (ag_centers, ag_views),
            (bz_centers, bz_views),
        ):
            view = view_at_control(centers, control)
            part = window_subset(
                views.loc[views.score_view.eq(view)], control
            )
            values.append(part.median_group_signed_log2fc.to_numpy(float))
        part = window_subset(fine_hsc, control)
        values.extend(
            [
                part.median_signed_log2fc_alphagenome.to_numpy(float),
                part.median_signed_log2fc_borzoi.to_numpy(float),
            ]
        )
    maximum = max(float(np.nanmax(np.abs(value))) for value in values)
    return max(0.5, math.ceil(maximum * 10) / 10)


def style_local_axis(axis: plt.Axes, control: pd.Series) -> None:
    axis.axvspan(
        int(control.start),
        int(control.end),
        facecolor="#D1D5DB",
        edgecolor="#4B5563",
        linewidth=0.8,
        hatch="////",
        alpha=0.55,
        zorder=0,
    )
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(axis="y", color="#E5E7EB", linewidth=0.7)


def render_local_control_page(
    *,
    control: pd.Series,
    evidence: pd.Series,
    ag_centers: pd.DataFrame,
    bz_centers: pd.DataFrame,
    ag_views: pd.DataFrame,
    bz_views: pd.DataFrame,
    fine_centers: pd.DataFrame,
    fine_effects: pd.DataFrame,
    effect_limit: float,
    pdf: PdfPages,
    output_dir: Path,
    page_number: int,
) -> Path:
    ag_score = window_subset(ag_centers, control)
    bz_score = window_subset(bz_centers, control)
    ft_score = window_subset(fine_centers, control)
    ag_view = view_at_control(ag_centers, control)
    bz_view = view_at_control(bz_centers, control)
    ag_effect = window_subset(
        ag_views.loc[ag_views.score_view.eq(ag_view)], control
    )
    bz_effect = window_subset(
        bz_views.loc[bz_views.score_view.eq(bz_view)], control
    )
    ft_effect = window_subset(
        fine_effects.loc[
            fine_effects.cell.eq("hsc")
            & fine_effects.readout_role.eq("gene_body_output_clipped")
        ],
        control,
    )
    frames = [
        ag_score,
        bz_score,
        ft_score,
        ag_effect,
        bz_effect,
        ft_effect,
    ]
    if any(frame.empty for frame in frames):
        raise ValueError(f"Missing local data for {control.control_id}")
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(14.4, 8.5),
        sharex=True,
        gridspec_kw={
            "height_ratios": [1.0, 1.3],
            "hspace": 0.16,
            "wspace": 0.13,
        },
    )
    original_score, fine_score = axes[0]
    original_effect, fine_effect = axes[1]
    for label, frame, color in (
        ("AlphaGenome", ag_score, COLORS["AlphaGenome"]),
        ("Borzoi", bz_score, COLORS["Borzoi"]),
    ):
        original_score.plot(
            frame.variant_offset_from_tss_transcription_bp,
            frame.importance_score,
            color=color,
            linewidth=1.45,
            label=label,
        )
    fine_score.plot(
        ft_score.variant_offset_from_tss_transcription_bp,
        ft_score.importance_score,
        color=COLORS["Fine-tuned"],
        linewidth=1.55,
        label="Fine-tuned combined rank",
    )
    for axis, title, ylabel in (
        (
            original_score,
            "Original models: frozen modality-view rank",
            "Original importance rank",
        ),
        (
            fine_score,
            "Fine-tuned models: nine-gene combined rank",
            "Fine-tuned importance rank",
        ),
    ):
        axis.axhline(
            THRESHOLD,
            color="#111827",
            linewidth=1.0,
            linestyle=(0, (5, 3)),
        )
        axis.axhline(
            REVIEW_THRESHOLD,
            color="#6B7280",
            linewidth=1.0,
            linestyle=(0, (2, 3)),
        )
        axis.set_ylim(0, 102)
        axis.set_title(title, loc="left", fontsize=10.5, fontweight="bold")
        axis.set_ylabel(ylabel)
        axis.legend(loc="lower left", frameon=False, fontsize=8.3)

    original_effect.plot(
        ag_effect.variant_offset_from_tss_transcription_bp,
        ag_effect.median_group_signed_log2fc,
        color=COLORS["AlphaGenome"],
        linewidth=1.55,
        label=f"AlphaGenome ({ag_view})",
    )
    original_effect.plot(
        bz_effect.variant_offset_from_tss_transcription_bp,
        bz_effect.median_group_signed_log2fc,
        color=COLORS["Borzoi"],
        linewidth=1.55,
        label=f"Borzoi ({bz_view})",
    )
    fine_effect.plot(
        ft_effect.variant_offset_from_tss_transcription_bp,
        ft_effect.median_signed_log2fc_alphagenome,
        color=COLORS["AlphaGenome"],
        linewidth=1.55,
        label="AlphaGenome",
    )
    fine_effect.plot(
        ft_effect.variant_offset_from_tss_transcription_bp,
        ft_effect.median_signed_log2fc_borzoi,
        color=COLORS["Borzoi"],
        linewidth=1.55,
        label="Borzoi",
    )
    for axis, title, ylabel in (
        (
            original_effect,
            "Original signed effect at each model's motif-driving view",
            "Signed group-median log2FC",
        ),
        (
            fine_effect,
            "Fine-tuned signed effect: fixed HSC gene-body output",
            "Signed log2FC",
        ),
    ):
        axis.axhline(0, color="#4B5563", linewidth=0.9)
        axis.set_ylim(-effect_limit, effect_limit)
        axis.set_title(title, loc="left", fontsize=10.2)
        axis.set_ylabel(ylabel)
        axis.set_xlabel(
            "Position relative to TSS (bp, transcription direction)"
        )
        axis.legend(loc="lower left", frameon=False, fontsize=8.2)
    midpoint = (int(control.start) + int(control.end)) / 2
    for axis in axes.flat:
        style_local_axis(axis, control)
        axis.set_xlim(
            midpoint - LOCAL_HALF_WIDTH_BP,
            midpoint + LOCAL_HALF_WIDTH_BP,
        )
    fig.suptitle(
        f"{control.gene} {control.label}: local positive-control evidence",
        x=0.055,
        y=0.98,
        ha="left",
        fontsize=16,
        fontweight="bold",
        color="#111827",
    )
    fig.text(
        0.055,
        0.94,
        (
            f"Interval {int(control.start):+d}..{int(control.end):+d} bp; "
            f"{control.transcript_name}; mm10. Window percentiles — "
            f"fine-tuned {evidence.finetuned_window_percentile_score:.2f}, "
            f"original AG "
            f"{evidence.original_alphagenome_window_percentile_score:.2f}, "
            f"original BZ "
            f"{evidence.original_borzoi_window_percentile_score:.2f}."
        ),
        fontsize=9.2,
        color="#374151",
    )
    fig.text(
        0.055,
        0.018,
        (
            "The bottom panels share one symmetric y-axis across all seven "
            "controls. Original and fine-tuned readouts are not cell-matched; "
            "the comparison tests rank retention and effect direction, not "
            "numerical equivalence."
        ),
        fontsize=8,
        color="#6B7280",
    )
    fig.subplots_adjust(left=0.075, right=0.985, top=0.88, bottom=0.11)
    return save_page(
        fig,
        pdf=pdf,
        output_dir=output_dir,
        stem=f"{page_number:02d}_{control.control_id}_local",
    )


def render_methods_page(
    *,
    pdf: PdfPages,
    output_dir: Path,
) -> Path:
    fig = plt.figure(figsize=(11.7, 8.3))
    fig.text(
        0.07,
        0.93,
        "Scope, metric definitions, and limitations",
        fontsize=18,
        fontweight="bold",
        color="#111827",
    )
    sections = [
        (
            "Scope and coordinate authority",
            (
                "Seven registered controls are evaluated on mouse mm10 using "
                "the boss-supplied transcript authority. Coordinates are "
                "relative to the transcript TSS and increase in transcription "
                "direction. Each gene uses the complete requested 3-kb scan; "
                "the actual 10-bp edit centers span -1495 to +1495 bp."
            ),
        ),
        (
            "Perturbation and dependence",
            (
                "The mutation manifest uses 10-bp strict mononucleotide-"
                "preserving shuffles, a 2-bp stride, and three deterministic "
                "replacement sequences per center. Neighboring centers are "
                "correlated because overlapping edits can disrupt the same "
                "motif. Region width is descriptive and is not counted as "
                "independent replication."
            ),
        ),
        (
            "Original-model score",
            (
                "The frozen Method 3 separates output and local-regulatory "
                "views. Each view requires support from at least three track "
                "groups, applies +/-4-bp spatial support, and is ranked within "
                "gene. The stronger view is ranked again to produce the final "
                "0-100 importance score. Formal control recall is the "
                "percentile of the control window's median raw support against "
                "all same-width windows in the same gene. The newly completed "
                "Col1a2 and Timp1 source runs contain five diagnostic readouts, "
                "but the frozen output score consumes only tss_1024bp; the "
                "other readouts do not enter the score."
            ),
        ),
        (
            "Fine-tuned score and signed effects",
            (
                "The fine-tuned importance profile is the existing nine-gene "
                "combined rank. Its displayed same-width window percentile is "
                "a descriptive rank of the median importance profile and is "
                "not identical to Method 3's raw-support calibration. Local "
                "fine-tuned effects use a fixed HSC gene-body readout. Original "
                "effects use the score-driving modality view and therefore are "
                "not cell-matched to HSC."
            ),
        ),
        (
            "Interpretive boundary",
            (
                "A high control window supports model sensitivity at the "
                "registered interval; it does not establish a specific TF, "
                "HSC specificity, experimental causality, or genome-wide "
                "significance. Non-control high-score regions remain outside "
                "this report's analytical scope."
            ),
        ),
    ]
    y = 0.86
    for heading, body in sections:
        fig.text(
            0.07,
            y,
            heading,
            fontsize=12.5,
            fontweight="bold",
            color="#111827",
            va="top",
        )
        y = add_wrapped_text(
            fig,
            body,
            x=0.07,
            y=y - 0.036,
            width=112,
            fontsize=9.5,
        ) - 0.035
    fig.text(
        0.07,
        0.055,
        "Recommended next question",
        fontsize=11,
        fontweight="bold",
        color="#111827",
    )
    fig.text(
        0.07,
        0.025,
        (
            "After the positive-control audit is frozen, a separate analysis "
            "can evaluate non-control high-score regions without changing the "
            "control calibration or using discoveries as additional controls."
        ),
        fontsize=8.5,
        color="#4B5563",
    )
    return save_page(
        fig,
        pdf=pdf,
        output_dir=output_dir,
        stem="11_methods_and_limitations",
    )


def source_inventory(
    *,
    fine_root: Path,
    original_root: Path,
) -> dict[str, object]:
    return {
        "coordinate_authority": str(TRANSCRIPT_AUTHORITY),
        "control_registry": str(CONTROL_REGISTRY),
        "fine_tuned_center_scores": str(
            fine_root / "analysis/center_importance_scores.tsv"
        ),
        "fine_tuned_signed_effects": str(
            fine_root / "analysis/cell_specific_log2fc.tsv"
        ),
        "fine_tuned_motif_annotations": str(
            fine_root / "analysis/region_motif_annotations.tsv"
        ),
        "original_run_root": str(original_root / "runs"),
        "score_implementation": str(ORIGINAL_CODE / "score_calculations.py"),
        "profile_implementation": str(ORIGINAL_CODE / "profile_features.py"),
    }


def chart_map() -> list[dict[str, object]]:
    return [
        {
            "section": "Global rank landscape",
            "question": (
                "Is each registered interval globally prominent within the "
                "complete 3-kb scan?"
            ),
            "family": "matrix",
            "type": "faceted one-dimensional heat strips",
            "fields": [
                "control_id",
                "variant_offset_from_tss_transcription_bp",
                "importance_score",
                "model_score_system",
            ],
            "claim": (
                "Shows whether the motif lies in a compact peak, broad "
                "plateau, or low-ranked background."
            ),
        },
        {
            "section": "Window-level evidence",
            "question": (
                "Do the control windows pass gene-matched same-width "
                "calibration, and how broad is the local high-score region?"
            ),
            "family": "comparison and matrix",
            "type": "faceted dot plot plus audit table",
            "fields": [
                "window_percentile_score",
                "score90_region_width_bp",
                "peak_distance_bp",
                "matches_expected_family",
            ],
            "claim": (
                "Separates global window prominence from local footprint "
                "shape and motif-loss evidence."
            ),
        },
        {
            "section": "Local positive-control evidence",
            "question": (
                "How do original and fine-tuned rank and signed-effect "
                "profiles behave around each registered interval?"
            ),
            "family": "trend",
            "type": "four-panel highlighted multi-series line",
            "fields": [
                "importance_score",
                "median_group_signed_log2fc",
                "median_signed_log2fc_alphagenome",
                "median_signed_log2fc_borzoi",
            ],
            "claim": (
                "Tests rank retention, peak alignment, and effect-direction "
                "agreement without subtracting separately calibrated scores."
            ),
        },
    ]


def main() -> None:
    args = parse_args()
    fine_root = args.fine_root.resolve()
    original_root = args.original_root.resolve()
    output_dir = args.out_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    set_report_style()

    controls, _ = load_controls()
    fine_validation = load_json(
        fine_root / "analysis/validation_summary.json"
    )
    fine_centers = pd.read_csv(
        fine_root / "analysis/center_importance_scores.tsv", sep="\t"
    )
    fine_effects = pd.read_csv(
        fine_root / "analysis/cell_specific_log2fc.tsv", sep="\t"
    )
    motif_annotations = pd.read_csv(
        fine_root / "analysis/region_motif_annotations.tsv", sep="\t"
    )
    require_columns(
        fine_centers,
        {
            "gene",
            "variant_offset_from_tss_transcription_bp",
            "importance_score",
        },
        "fine-tuned center scores",
    )
    require_columns(
        fine_effects,
        {
            "gene",
            "variant_offset_from_tss_transcription_bp",
            "cell",
            "readout_role",
            "median_signed_log2fc_alphagenome",
            "median_signed_log2fc_borzoi",
        },
        "fine-tuned signed effects",
    )
    ag_centers, ag_views, ag_validations = score_original_model(
        original_root, "AlphaGenome"
    )
    bz_centers, bz_views, bz_validations = score_original_model(
        original_root, "Borzoi"
    )
    fine_recall = fine_tuned_window_recall(fine_centers, controls)
    ag_recall = original_interval_recall(ag_centers, controls)
    bz_recall = original_interval_recall(bz_centers, controls)
    evidence = build_evidence_table(
        controls=controls,
        fine_centers=fine_centers,
        fine_recall=fine_recall,
        ag_centers=ag_centers,
        bz_centers=bz_centers,
        ag_recall=ag_recall,
        bz_recall=bz_recall,
        motif_annotations=motif_annotations,
    )
    evidence["control_id"] = pd.Categorical(
        evidence.control_id, CONTROL_ORDER, ordered=True
    )
    evidence = evidence.sort_values("control_id").reset_index(drop=True)
    evidence["control_id"] = evidence.control_id.astype(str)

    ag_centers.to_csv(
        output_dir / "original_alphagenome_center_scores.tsv",
        sep="\t",
        index=False,
    )
    ag_views.to_csv(
        output_dir / "original_alphagenome_view_scores.tsv",
        sep="\t",
        index=False,
    )
    bz_centers.to_csv(
        output_dir / "original_borzoi_center_scores.tsv",
        sep="\t",
        index=False,
    )
    bz_views.to_csv(
        output_dir / "original_borzoi_view_scores.tsv",
        sep="\t",
        index=False,
    )
    evidence.to_csv(
        output_dir / "positive_control_evidence.tsv",
        sep="\t",
        index=False,
    )
    (output_dir / "source_inventory.json").write_text(
        json.dumps(
            source_inventory(
                fine_root=fine_root, original_root=original_root
            ),
            indent=2,
        )
        + "\n"
    )
    (output_dir / "chart_map.json").write_text(
        json.dumps(chart_map(), indent=2) + "\n"
    )

    report_path = output_dir / "positive_control_global_audit_en.pdf"
    page_paths: list[str] = []
    effect_limit = shared_effect_limit(
        controls=controls,
        ag_centers=ag_centers,
        bz_centers=bz_centers,
        ag_views=ag_views,
        bz_views=bz_views,
        fine_effects=fine_effects,
    )
    with PdfPages(
        report_path,
        metadata={
            "Title": "Positive-control global sensitivity audit",
            "Author": "gReLU replication workflow",
            "Subject": (
                "Original-versus-fine-tuned positive-control motif audit"
            ),
        },
    ) as pdf:
        page_paths.append(
            str(
                render_summary_page(
                    evidence=evidence,
                    pdf=pdf,
                    output_dir=output_dir,
                )
            )
        )
        page_paths.append(
            str(
                render_global_atlas(
                    controls=controls,
                    profiles={
                        "original_alphagenome": ag_centers,
                        "original_borzoi": bz_centers,
                        "finetuned": fine_centers,
                    },
                    pdf=pdf,
                    output_dir=output_dir,
                )
            )
        )
        page_paths.append(
            str(
                render_evidence_matrix(
                    evidence=evidence,
                    pdf=pdf,
                    output_dir=output_dir,
                )
            )
        )
        for page_number, (_, control) in enumerate(
            controls.iterrows(), start=4
        ):
            evidence_row = evidence.loc[
                evidence.control_id.eq(control.control_id)
            ].iloc[0]
            page_paths.append(
                str(
                    render_local_control_page(
                        control=control,
                        evidence=evidence_row,
                        ag_centers=ag_centers,
                        bz_centers=bz_centers,
                        ag_views=ag_views,
                        bz_views=bz_views,
                        fine_centers=fine_centers,
                        fine_effects=fine_effects,
                        effect_limit=effect_limit,
                        pdf=pdf,
                        output_dir=output_dir,
                        page_number=page_number,
                    )
                )
            )
        page_paths.append(
            str(
                render_methods_page(
                    pdf=pdf,
                    output_dir=output_dir,
                )
            )
        )

    validation = {
        "status": "ok",
        "scope": "registered_positive_controls_only",
        "assembly": "mm10",
        "coordinate_convention": (
            "TSS-relative bp increasing in transcription direction"
        ),
        "controls": int(len(evidence)),
        "control_ids": evidence.control_id.tolist(),
        "original_models_complete_for_all_controls": True,
        "fine_tuned_validation_status": fine_validation.get("status"),
        "original_alphagenome_run_validations": len(ag_validations),
        "original_borzoi_run_validations": len(bz_validations),
        "original_alphagenome_centers": int(len(ag_centers)),
        "original_borzoi_centers": int(len(bz_centers)),
        "fine_tuned_centers_in_scope": int(
            fine_centers.gene.isin(controls.gene.unique()).sum()
        ),
        "effect_axis_symmetric_limit": effect_limit,
        "report_pages": len(page_paths),
        "report_path": str(report_path),
        "report_exists": report_path.exists(),
        "report_bytes": report_path.stat().st_size,
        "page_pngs": page_paths,
        "all_page_pngs_exist": all(
            Path(path).exists() for path in page_paths
        ),
        "duplicate_evidence_controls": int(
            evidence.control_id.duplicated().sum()
        ),
        "nonfinite_window_scores": int(
            (
                ~np.isfinite(
                    evidence[
                        [
                            "finetuned_window_percentile_score",
                            "original_alphagenome_window_percentile_score",
                            "original_borzoi_window_percentile_score",
                        ]
                    ].to_numpy(float)
                )
            ).sum()
        ),
        "non_control_regions_interpreted": False,
    }
    if validation["controls"] != 7:
        raise RuntimeError(validation)
    if validation["report_pages"] != 11:
        raise RuntimeError(validation)
    if validation["report_bytes"] < 100_000:
        raise RuntimeError(validation)
    if not validation["all_page_pngs_exist"]:
        raise RuntimeError(validation)
    if validation["duplicate_evidence_controls"]:
        raise RuntimeError(validation)
    if validation["nonfinite_window_scores"]:
        raise RuntimeError(validation)
    (output_dir / "validation_summary.json").write_text(
        json.dumps(validation, indent=2) + "\n"
    )
    print(json.dumps(validation, indent=2))
    print(evidence.to_string(index=False))


if __name__ == "__main__":
    main()
