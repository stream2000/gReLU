"""Render four two-panel nine-gene reports from a validated Saijou analysis."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pyBigWig  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from matplotlib.ticker import FuncFormatter, MaxNLocator  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parents[5]
BIGWIG_ROOT = Path(
    "/work2/Projects/Project_DL_pillar/Finetune_Borzoi_Saijou_scRNAseq/bigwig"
)
CELLS = {
    "hsc": {
        "label": "HSC",
        "slug": "hsc",
        "bigwig": BIGWIG_ROOT / "hsc.CPM.mapq10.bw",
    },
    "mac": {
        "label": "Macrophage",
        "slug": "macrophage",
        "bigwig": BIGWIG_ROOT / "mac.CPM.mapq10.bw",
    },
    "lsec": {
        "label": "LSEC",
        "slug": "lsec",
        "bigwig": BIGWIG_ROOT / "lsec.CPM.mapq10.bw",
    },
    "chol": {
        "label": "Cholangiocyte",
        "slug": "cholangiocyte",
        "bigwig": BIGWIG_ROOT / "chol.CPM.mapq10.bw",
    },
}
GENES = (
    "Mdk",
    "Acta2",
    "Col1a1",
    "Col1a2",
    "Timp1",
    "Vegfc",
    "Hgf",
    "Igf1",
    "Ngf",
)
MODELS = {
    "alphagenome": {
        "label": "AlphaGenome-FT",
        "raw": "median_signed_log2fc_alphagenome",
        "smooth": "median_signed_log2fc_alphagenome_smoothed",
        "color": "#2364AA",
        "style": "-",
    },
    "borzoi": {
        "label": "Borzoi-FT",
        "raw": "median_signed_log2fc_borzoi",
        "smooth": "median_signed_log2fc_borzoi_smoothed",
        "color": "#D97706",
        "style": "--",
    },
}
FAMILY_LABELS = {
    "AP1": "AP-1",
    "CEBP": "CEBP",
    "CTCF_CTCFL": "CTCF",
    "EGR_ZBTB": "EGR/ZBTB",
    "ETS": "ETS",
    "HIF": "HIF",
    "HNF": "HNF/FOXA",
    "KLF_SP_GC": "SP/KLF",
    "NFKB": "NF-kB",
    "NFY_CCAAT": "NF-Y",
    "RREB1": "RREB1",
    "SMAD": "SMAD",
    "SRF_CARG": "SRF/CArG",
    "STAT": "STAT",
    "TEAD": "TEAD",
}
CANDIDATE_COLOR = "#E7B83A"
CONTROL_COLOR = "#C62F72"


def load_observed(
    genes: pd.DataFrame, *, half_window_bp: int, observed_bin_bp: int
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    handles = {
        cell: pyBigWig.open(str(spec["bigwig"]))
        for cell, spec in CELLS.items()
    }
    try:
        for gene in genes.itertuples(index=False):
            half = int(half_window_bp)
            bins = (2 * half) // int(observed_bin_bp)
            start = int(gene.analysis_tss) - half
            end = int(gene.analysis_tss) + half
            offsets = np.linspace(
                -half + observed_bin_bp / 2,
                half - observed_bin_bp / 2,
                bins,
            )
            for cell, handle in handles.items():
                values = handle.stats(
                    str(gene.chrom),
                    start,
                    end,
                    nBins=bins,
                    type="mean",
                    exact=True,
                )
                signal = np.nan_to_num(
                    np.asarray(
                        [
                            np.nan if value is None else value
                            for value in values
                        ],
                        dtype=float,
                    ),
                    nan=0.0,
                )
                if str(gene.strand) == "-":
                    signal = signal[::-1]
                rows.extend(
                    {
                        "gene": gene.gene,
                        "cell": cell,
                        "offset": float(offset),
                        "observed_cpm": float(value),
                    }
                    for offset, value in zip(offsets, signal, strict=True)
                )
    finally:
        for handle in handles.values():
            handle.close()
    observed = pd.DataFrame.from_records(rows)
    if not np.isfinite(observed.observed_cpm.to_numpy()).all():
        raise ValueError("Observed CPM contains non-finite values")
    return observed


def load_reference_predictions(
    genes: pd.DataFrame, *, root: Path, half_window_bp: int
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for model, spec in MODELS.items():
        run = root / "runs" / f"{model}_finetuned"
        profile_dir = run / "profiles"
        for gene in genes.itertuples(index=False):
            profile_meta = json.loads(
                (profile_dir / f"{gene.gene}.profile_metadata.json").read_text()
            )
            resolution = int(profile_meta["stored_resolution_bp"])
            output_start = int(profile_meta["output_start"])
            profiles = np.load(
                profile_dir / f"{gene.gene}.ref_profiles_{resolution}bp.npy",
                mmap_mode="r",
            )
            track_order = json.loads(
                (profile_dir / f"{gene.gene}.track_order.json").read_text()
            )
            if profiles.shape[0] != len(track_order):
                profiles = profiles.T
            if profiles.shape[0] != len(track_order):
                raise ValueError(
                    f"{model}/{gene.gene}: profile and track order disagree"
                )
            midpoints = (
                output_start
                + np.arange(profiles.shape[1], dtype=float) * resolution
                + resolution / 2
            )
            direction = 1.0 if str(gene.strand) == "+" else -1.0
            offsets = direction * (midpoints - int(gene.analysis_tss))
            keep = np.abs(offsets) <= int(half_window_bp) + resolution
            for cell in CELLS:
                track_index = track_order.index(cell)
                # Fine-tuning uses 128-bp bin sums. Dividing by the bin width
                # returns the per-base CPM height used by the source BigWigs.
                cpm = (
                    np.asarray(profiles[track_index], dtype=float) / resolution
                )
                rows.extend(
                    {
                        "gene": gene.gene,
                        "cell": cell,
                        "model": model,
                        "model_label": spec["label"],
                        "offset": float(offset),
                        "predicted_cpm": float(value),
                        "resolution_bp": resolution,
                    }
                    for offset, value in zip(
                        offsets[keep], cpm[keep], strict=True
                    )
                )
    predictions = pd.DataFrame.from_records(rows)
    if (predictions.predicted_cpm < 0).any():
        raise ValueError("Reference predictions contain negative CPM")
    if not np.isfinite(predictions.predicted_cpm.to_numpy()).all():
        raise ValueError("Reference predictions contain non-finite values")
    return predictions


def motif_labels(
    regions: pd.DataFrame, annotations: pd.DataFrame
) -> pd.DataFrame:
    ranked = annotations.loc[
        annotations.scan_kind.eq("ranked_candidate")
    ].copy()
    ranked = ranked.sort_values(
        ["screen_rank", "family_rank"]
    ).drop_duplicates("screen_rank")
    ranked["motif_label"] = ranked.family.map(FAMILY_LABELS).fillna("")
    ranked.loc[ranked.family.eq("UNRESOLVED"), "motif_label"] = ""
    return regions.merge(
        ranked[["screen_rank", "motif_label", "top_motif"]],
        on="screen_rank",
        how="left",
        validate="one_to_one",
    )


def _style_axis(axis: plt.Axes) -> None:
    axis.spines[["top", "right"]].set_visible(False)
    axis.spines[["left", "bottom"]].set_color("#9CA3AF")
    axis.grid(axis="y", color="#E5E7EB", linewidth=0.6)
    axis.tick_params(labelsize=8, colors="#374151")
    axis.axvline(0, color="#4B5563", linewidth=0.85, linestyle=":")


def _nice_positive_limit(values: np.ndarray) -> float:
    maximum = max(0.01, float(np.nanmax(values)))
    if maximum < 1:
        return float(np.ceil(maximum * 10) / 10)
    if maximum <= 10:
        return float(np.ceil(maximum))
    if maximum <= 100:
        return float(np.ceil(maximum / 5) * 5)
    return float(np.ceil(maximum / 50) * 50)


def _add_spans(
    axis: plt.Axes,
    regions: pd.DataFrame,
    controls: pd.DataFrame,
) -> None:
    for region in regions.itertuples(index=False):
        axis.axvspan(
            int(region.region_start) - 5,
            int(region.region_end) + 5,
            color=CANDIDATE_COLOR,
            alpha=0.20,
            linewidth=0,
            zorder=0,
        )
    for control in controls.itertuples(index=False):
        axis.axvspan(
            int(control.start),
            int(control.end),
            color=CONTROL_COLOR,
            alpha=0.30,
            linewidth=0,
            zorder=1,
        )


def draw_expression(
    axis: plt.Axes,
    observed: pd.DataFrame,
    predictions: pd.DataFrame,
    regions: pd.DataFrame,
    controls: pd.DataFrame,
    cell_label: str,
) -> None:
    observed = observed.sort_values("offset")
    x = observed.offset.to_numpy()
    y = observed.observed_cpm.to_numpy()
    _add_spans(axis, regions, controls)
    axis.fill_between(x, 0, y, color="#AFC3D2", alpha=0.62, linewidth=0)
    axis.plot(
        x,
        y,
        color="#405F78",
        linewidth=1.0,
        label="Observed CPM",
        zorder=3,
    )
    combined = [y]
    for model, spec in MODELS.items():
        part = predictions.loc[predictions.model.eq(model)].sort_values(
            "offset"
        )
        axis.plot(
            part.offset,
            part.predicted_cpm,
            color=spec["color"],
            linestyle=spec["style"],
            linewidth=1.15,
            drawstyle="steps-mid",
            label=f"{spec['label']} reference",
            zorder=4,
        )
        combined.append(part.predicted_cpm.to_numpy())
    limit = _nice_positive_limit(np.concatenate(combined))
    axis.set_ylim(0, limit)
    axis.set_yticks([0, limit])
    axis.set_ylabel(
        f"{cell_label} expression\nCPM",
        fontsize=9,
    )
    axis.legend(
        loc="upper left",
        frameon=False,
        ncol=3,
        fontsize=7.6,
        handlelength=2.5,
    )
    axis.tick_params(axis="x", labelbottom=False)
    _style_axis(axis)


def _candidate_label_positions(regions: pd.DataFrame) -> list[tuple[object, float]]:
    items = list(regions.sort_values("peak_offset").itertuples(index=False))
    if not items:
        return []
    centers = np.asarray([float(item.peak_offset) for item in items])
    placed = centers.copy()
    minimum_gap = 220.0
    for index in range(1, len(placed)):
        placed[index] = max(placed[index], placed[index - 1] + minimum_gap)
    overflow = placed[-1] - 1370
    if overflow > 0:
        placed -= overflow
    underflow = -1370 - placed[0]
    if underflow > 0:
        placed += underflow
    return list(zip(items, placed, strict=True))


def draw_logfc(
    axis: plt.Axes,
    data: pd.DataFrame,
    regions: pd.DataFrame,
    controls: pd.DataFrame,
) -> None:
    _add_spans(axis, regions, controls)
    maximum = 0.01
    for spec in MODELS.values():
        maximum = max(
            maximum,
            float(data[spec["raw"]].abs().max()),
            float(data[spec["smooth"]].abs().max()),
        )
        axis.plot(
            data.variant_offset_from_tss_transcription_bp,
            data[spec["raw"]],
            color=spec["color"],
            linewidth=0.55,
            alpha=0.28,
        )
        axis.plot(
            data.variant_offset_from_tss_transcription_bp,
            data[spec["smooth"]],
            color=spec["color"],
            linestyle=spec["style"],
            linewidth=1.45,
            label=spec["label"],
        )
    limit = float(np.ceil(maximum * 1.15 * 10) / 10)
    axis.set_ylim(-limit, limit)
    axis.axhline(0, color="#374151", linewidth=0.8)
    axis.set_ylabel("Median signed log2FC\n(shuffle / reference)", fontsize=9)
    axis.set_xlabel(
        "Position relative to supplied transcript TSS "
        "(bp, transcription direction)",
        fontsize=9,
    )
    axis.xaxis.set_major_locator(MaxNLocator(nbins=7, integer=True))
    axis.xaxis.set_major_formatter(
        FuncFormatter(lambda value, _: f"{value:+.0f}")
    )
    axis.yaxis.set_major_locator(MaxNLocator(nbins=6))
    _style_axis(axis)

    for lane, (region, label_x) in enumerate(
        _candidate_label_positions(regions)
    ):
        motif = str(region.motif_label) if pd.notna(region.motif_label) else ""
        label = (
            f"R{int(region.gene_rank)}"
            + (f" {motif}" if motif else "")
            + f" | {float(region.region_score):.1f}"
        )
        y_text = limit * (0.84 - 0.13 * (lane % 2))
        axis.annotate(
            label,
            xy=(float(region.peak_offset), limit * 0.96),
            xytext=(label_x, y_text),
            ha="center",
            va="top",
            fontsize=7.4,
            color="#5E4300",
            fontweight="bold",
            arrowprops={
                "arrowstyle": "-",
                "color": "#9A7410",
                "linewidth": 0.6,
            },
            bbox={
                "boxstyle": "round,pad=0.16",
                "facecolor": "#FFF8DD",
                "edgecolor": "#C79B25",
                "linewidth": 0.55,
                "alpha": 0.94,
            },
        )
    for lane, control in enumerate(controls.itertuples(index=False)):
        center = (float(control.start) + float(control.end)) / 2
        axis.annotate(
            str(control.label),
            xy=(center, -limit * 0.98),
            xytext=(center, -limit * (0.78 - 0.13 * (lane % 2))),
            ha="center",
            va="bottom",
            fontsize=7.2,
            color="#8B124D",
            fontweight="bold",
            arrowprops={
                "arrowstyle": "-",
                "color": CONTROL_COLOR,
                "linewidth": 0.7,
            },
            bbox={
                "boxstyle": "round,pad=0.16",
                "facecolor": "#FDEAF3",
                "edgecolor": CONTROL_COLOR,
                "linewidth": 0.6,
                "alpha": 0.94,
            },
        )


def plot_page(
    gene: pd.Series,
    cell: str,
    observed: pd.DataFrame,
    predictions: pd.DataFrame,
    logfc: pd.DataFrame,
    regions: pd.DataFrame,
    controls: pd.DataFrame,
    half: int,
) -> plt.Figure:
    cell_spec = CELLS[cell]
    gene_observed = observed.loc[
        observed.gene.eq(gene.gene) & observed.cell.eq(cell)
    ]
    gene_predictions = predictions.loc[
        predictions.gene.eq(gene.gene) & predictions.cell.eq(cell)
    ]
    gene_logfc = logfc.loc[
        logfc.gene.eq(gene.gene) & logfc.cell.eq(cell)
    ].sort_values("variant_offset_from_tss_transcription_bp")
    gene_regions = regions.loc[regions.gene.eq(gene.gene)]
    gene_controls = controls.loc[controls.gene.eq(gene.gene)]

    figure = plt.figure(figsize=(13.6, 7.8), facecolor="white")
    grid = figure.add_gridspec(
        2, 1, height_ratios=[1.15, 2.55], hspace=0.10
    )
    observed_axis = figure.add_subplot(grid[0])
    logfc_axis = figure.add_subplot(grid[1], sharex=observed_axis)
    draw_expression(
        observed_axis,
        gene_observed,
        gene_predictions,
        gene_regions,
        gene_controls,
        str(cell_spec["label"]),
    )
    draw_logfc(logfc_axis, gene_logfc, gene_regions, gene_controls)
    observed_axis.set_xlim(-half, half)
    logfc_axis.set_xlim(-half, half)

    handles = [
        Line2D(
            [0],
            [0],
            color=spec["color"],
            linestyle=spec["style"],
            linewidth=1.6,
            label=spec["label"],
        )
        for spec in MODELS.values()
    ] + [
        Patch(
            facecolor=CANDIDATE_COLOR,
            alpha=0.35,
            label="score-selected important region",
        ),
        Patch(
            facecolor=CONTROL_COLOR,
            alpha=0.35,
            label="registered positive control",
        ),
    ]
    logfc_axis.legend(
        handles=handles,
        loc="upper left",
        frameon=False,
        ncol=4,
        fontsize=8,
        handlelength=2.7,
    )
    figure.suptitle(
        (
            f"{cell_spec['label']} | {gene.gene} "
            f"({gene.transcript_name}, {gene.strand}) | "
            f"mm10 {gene.chrom}:{int(gene.start):,}-{int(gene.end):,}"
        ),
        x=0.075,
        y=0.975,
        ha="left",
        fontsize=14,
        fontweight="bold",
        color="#111827",
    )
    figure.text(
        0.075,
        0.936,
        (
            f"TSS {int(gene.analysis_tss):,}; strict shuffle "
            f"-{half:,}..+{half:,} bp. Top: observed + FT reference CPM "
            "(128-bp sum / 128), shared scale. Bottom: signed log2FC; "
            "bold = five-center median; fixed gene-body readout."
        ),
        ha="left",
        va="top",
        fontsize=8.2,
        color="#4B5563",
    )
    figure.text(
        0.985,
        0.018,
        (
            "Gold labels: within-gene rank, motif family (if supported), "
            "selection score. Magenta: registered wet/reported control."
        ),
        ha="right",
        va="bottom",
        fontsize=7.2,
        color="#6B7280",
    )
    figure.subplots_adjust(
        left=0.09, right=0.985, top=0.90, bottom=0.095
    )
    return figure


def build_cell_browser_reports(
    *, root: Path, half_window_bp: int = 1500, observed_bin_bp: int = 4
) -> dict[str, object]:
    """Build the four cell PDFs and return their validation summary."""

    root = root.resolve()
    span_bp = 2 * int(half_window_bp)
    span_tag = f"{span_bp // 1000}kb" if span_bp % 1000 == 0 else f"{span_bp}bp"
    analysis = root / "analysis"
    figures = root / "report"
    figures.mkdir(parents=True, exist_ok=True)
    genes = pd.read_csv(root / "prepared/genes.tsv", sep="\t")
    authority = pd.read_csv(
        root / "prepared/coordinate_authority_audit.tsv", sep="\t"
    )
    genes = genes.merge(
        authority[["gene", "transcript_name", "transcript_id"]],
        on="gene",
        how="left",
        validate="one_to_one",
    )
    genes = genes.set_index("gene").loc[list(GENES)].reset_index()
    logfc = pd.read_csv(analysis / "cell_specific_log2fc.tsv", sep="\t")
    regions = pd.read_csv(
        analysis / "top_important_regions.tsv", sep="\t"
    )
    annotations = pd.read_csv(
        analysis / "region_motif_annotations.tsv", sep="\t"
    )
    controls = pd.read_csv(
        analysis / "positive_control_recall.tsv", sep="\t"
    )
    regions = motif_labels(regions, annotations)
    observed = load_observed(
        genes,
        half_window_bp=half_window_bp,
        observed_bin_bp=observed_bin_bp,
    )
    predictions = load_reference_predictions(
        genes, root=root, half_window_bp=half_window_bp
    )
    observed.to_csv(
        analysis / "observed_browser_signal.tsv",
        sep="\t",
        index=False,
    )
    predictions.to_csv(
        analysis / "reference_prediction_browser_signal.tsv",
        sep="\t",
        index=False,
    )
    regions.to_csv(
        analysis / "plot_important_regions.tsv", sep="\t", index=False
    )
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.unicode_minus": False,
            "savefig.facecolor": "white",
        }
    )

    pdfs: list[Path] = []
    pngs: list[Path] = []
    for cell, cell_spec in CELLS.items():
        pdf_path = (
            figures / f"{cell_spec['slug']}_nine_gene_{span_tag}_ism.pdf"
        )
        pdfs.append(pdf_path)
        with PdfPages(pdf_path) as pdf:
            for gene in genes.itertuples(index=False):
                figure = plot_page(
                    pd.Series(gene._asdict()),
                    cell,
                    observed,
                    predictions,
                    logfc,
                    regions,
                    controls,
                    int(half_window_bp),
                )
                png_path = figures / (
                    f"{cell_spec['slug']}_{gene.gene.lower()}_{span_tag}_ism.png"
                )
                figure.savefig(png_path, dpi=190)
                pdf.savefig(figure)
                pngs.append(png_path)
                plt.close(figure)

    expected_logfc = (
        len(CELLS)
        * sum(
            pd.read_csv(
                root / "prepared/mutation_manifest.tsv", sep="\t"
            )
            .groupby("gene")[
                "variant_offset_from_tss_transcription_bp"
            ]
            .nunique()
        )
    )
    validation = {
        "status": "ok",
        "pdfs": [str(path) for path in pdfs],
        "pdf_count": int(len(pdfs)),
        "pages_per_pdf": int(len(GENES)),
        "png_count": int(len(pngs)),
        "plots_per_page": 2,
        "x_axis": (
            f"TSS -{int(half_window_bp)}..+{int(half_window_bp)} bp "
            "in transcription direction"
        ),
        "logfc_y_axis": "median signed log2FC (strict shuffle / reference)",
        "score_drawn_as_y_axis": False,
        "observed_rows": int(len(observed)),
        "reference_prediction_rows": int(len(predictions)),
        "logfc_rows": int(len(logfc)),
        "expected_logfc_rows": int(expected_logfc),
        "candidate_regions": int(len(regions)),
        "registered_positive_controls": int(len(controls)),
        "observed_finite": bool(
            np.isfinite(observed.observed_cpm.to_numpy()).all()
        ),
        "reference_predictions_finite": bool(
            np.isfinite(predictions.predicted_cpm.to_numpy()).all()
        ),
        "logfc_finite": bool(
            np.isfinite(
                logfc[
                    [
                        spec["raw"]
                        for spec in MODELS.values()
                    ]
                    + [
                        spec["smooth"]
                        for spec in MODELS.values()
                    ]
                ].to_numpy()
            ).all()
        ),
    }
    if (
        validation["pdf_count"] != len(CELLS)
        or validation["png_count"] != len(CELLS) * len(GENES)
        or validation["logfc_rows"] != validation["expected_logfc_rows"]
        or not validation["observed_finite"]
        or not validation["reference_predictions_finite"]
        or not validation["logfc_finite"]
    ):
        validation["status"] = "failed"
    (analysis / "report_validation_summary.json").write_text(
        json.dumps(validation, indent=2) + "\n"
    )
    print(json.dumps(validation, indent=2))
    if validation["status"] != "ok":
        raise RuntimeError(validation)
    return validation
