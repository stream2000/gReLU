#!/usr/bin/env python
"""Build the source-backed HTML-report artifact for the frozen score study."""

from __future__ import annotations

import argparse
import base64
import io
import json
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO = Path(__file__).resolve().parents[6]
DEFAULT_ROOT = REPO / "experiments/ism/original_multitrack_score_20260728"
GENES = ["Mdk", "Col1a1", "Acta2"]
COLORS = {
    "alphagenome": "#326891",
    "borzoi": "#d17a22",
    "hsc": "#7b3294",
    "mac": "#008837",
    "lsec": "#c51b7d",
    "chol": "#2c7fb8",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    return parser.parse_args()


def dataframe_records(frame: pd.DataFrame) -> list[dict[str, object]]:
    clean = frame.replace({np.nan: None})
    return json.loads(clean.to_json(orient="records"))


def figure_data_uri(fig: plt.Figure) -> str:
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(
        buffer.getvalue()
    ).decode("ascii")


def build_sensitivity_figure(
    gene: str,
    ag_views: pd.DataFrame,
    bz_views: pd.DataFrame,
    controls: pd.DataFrame,
    motif_regions: pd.DataFrame,
) -> tuple[str, pd.DataFrame]:
    fig, axes = plt.subplots(
        2, 1, figsize=(10.2, 2.2), sharex=True, constrained_layout=True
    )
    chart_rows: list[pd.DataFrame] = []
    for ax, view in zip(
        axes, ["output", "local_regulatory"], strict=True
    ):
        for model, source in [
            ("alphagenome", ag_views),
            ("borzoi", bz_views),
        ]:
            group = source.loc[
                source.gene.eq(gene) & source.score_view.eq(view)
            ].sort_values("variant_offset_from_tss_transcription_bp")
            ax.plot(
                group.variant_offset_from_tss_transcription_bp,
                group.median_group_signed_log2fc,
                color=COLORS[model],
                linewidth=0.8,
                label=model,
            )
            selected = group[
                [
                    "gene",
                    "variant_offset_from_tss_transcription_bp",
                    "median_group_signed_log2fc",
                ]
            ].copy()
            selected["model"] = model
            selected["view"] = view
            chart_rows.append(selected)
        ax.axhline(0, color="#444444", linewidth=0.6)
        for row in controls.loc[controls.gene.eq(gene)].itertuples():
            ax.axvspan(
                int(row.start),
                int(row.end),
                color="#f0c75e",
                alpha=0.18,
                linewidth=0,
            )
        for row in motif_regions.loc[
            motif_regions.gene.eq(gene)
            & motif_regions.annotation_status.eq(
                "consistent_native_motif_loss"
            )
        ].itertuples():
            ax.axvspan(
                int(row.region_start),
                int(row.region_end),
                color="#54a24b",
                alpha=0.08,
                linewidth=0,
            )
        ax.set_ylabel(
            "output\nlog2FC"
            if view == "output"
            else "local\nlog2FC",
            fontsize=8,
        )
        ax.grid(axis="y", color="#dddddd", linewidth=0.4)
        ax.tick_params(labelsize=7)
    axes[0].legend(
        loc="upper right", frameon=False, ncol=2, fontsize=7
    )
    axes[-1].set_xlabel(
        "TSS offset in transcription direction (bp)", fontsize=8
    )
    fig.suptitle(
        f"{gene}: frozen-score signed 3 kb sensitivity", fontsize=10
    )
    return figure_data_uri(fig), pd.concat(chart_rows, ignore_index=True)


def build_observed_expression_figure(
    gene: str, observed: pd.DataFrame
) -> tuple[str, pd.DataFrame]:
    subset = observed.loc[observed.gene.eq(gene)].copy()
    fig, axes = plt.subplots(
        1, 4, figsize=(10.2, 1.95), sharex=True, constrained_layout=True
    )
    for ax, cell in zip(axes, ["hsc", "mac", "lsec", "chol"], strict=True):
        group = subset.loc[subset.cell.eq(cell)].sort_values("offset")
        ax.plot(
            group.offset,
            group.observed_cpm,
            color=COLORS[cell],
            linewidth=0.9,
        )
        ax.fill_between(
            group.offset,
            group.observed_cpm,
            color=COLORS[cell],
            alpha=0.12,
        )
        ax.set_title(cell.upper(), fontsize=8)
        ax.set_ylabel("CPM", fontsize=7)
        ax.tick_params(labelsize=6)
        ax.grid(axis="y", color="#e5e5e5", linewidth=0.35)
    for ax in axes:
        ax.set_xlabel("TSS offset", fontsize=7)
    fig.suptitle(
        f"{gene}: observed fine-tuning source signal", fontsize=10
    )
    return figure_data_uri(fig), subset


def build_source_spec(
    source_id: str, label: str, path: Path, root: Path
) -> dict[str, object]:
    relative = path.relative_to(REPO)
    return {
        "id": source_id,
        "label": label,
        "path": str(relative),
        "query": {
            "engine": "duckdb",
            "language": "sql",
            "sql": (
                f"SELECT * FROM read_csv_auto('{relative}', "
                "delim='\\t', header=true)"
            ),
            "description": label,
            "tables_used": [str(relative)],
        },
    }


def table_column_spec(
    field: str, label: str, fmt: str = "text"
) -> dict[str, object]:
    return {"field": field, "label": label, "format": fmt}


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    report_dir = root / "report"
    figures_dir = report_dir / "figures"
    data_dir = report_dir / "data"
    figures_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    ag_dir = root / "analysis" / "method3_modality_views"
    bz_dir = root / "analysis" / "borzoi_frozen_score"
    summary_dir = root / "analysis" / "exploration_summary"
    ag_motif_dir = (
        root / "motif_annotation_method3_alphagenome" / "analysis"
    )
    bz_motif_dir = (
        root / "motif_annotation_method3_borzoi" / "analysis"
    )
    required = [
        bz_dir / "validation_summary.json",
        bz_motif_dir / "region_motif_annotations.tsv",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Borzoi scoring/motif annotation is incomplete: " + ", ".join(missing)
        )

    method_comparison = pd.read_csv(
        summary_dir / "method_comparison.tsv", sep="\t"
    )
    ag_controls = pd.read_csv(
        summary_dir / "positive_control_evidence.tsv", sep="\t"
    )
    bz_controls = pd.read_csv(
        bz_dir / "positive_control_recall.tsv", sep="\t"
    )
    bz_view_controls = pd.read_csv(
        bz_dir / "positive_control_view_recall.tsv", sep="\t"
    )
    concordance = pd.read_csv(
        bz_dir / "alphagenome_borzoi_concordance.tsv", sep="\t"
    )
    ag_views = pd.read_csv(ag_dir / "view_center_scores.tsv", sep="\t")
    bz_views = pd.read_csv(bz_dir / "view_center_scores.tsv", sep="\t")
    controls = pd.read_csv(
        root
        / "prepared_full3kb_three_gene"
        / "positive_control_registry.tsv",
        sep="\t",
    )
    observed_path = (
        REPO
        / "experiments/ism/"
        "saijou_nine_gene_tss_3kb_strict_shuffle_pdf_transcripts/"
        "analysis/observed_3kb_browser_signal.tsv"
    )
    observed = pd.read_csv(observed_path, sep="\t")
    ag_regions = pd.read_csv(
        ag_motif_dir / "region_motif_annotations.tsv", sep="\t"
    )
    bz_regions = pd.read_csv(
        bz_motif_dir / "region_motif_annotations.tsv", sep="\t"
    )
    ag_candidate_regions = ag_regions.loc[
        ag_regions.scan_kind.eq("ranked_candidate")
    ].copy()
    bz_candidate_regions = bz_regions.loc[
        bz_regions.scan_kind.eq("ranked_candidate")
    ].copy()

    sensitivity_rows: list[pd.DataFrame] = []
    observed_rows: list[pd.DataFrame] = []
    image_blocks: list[dict[str, object]] = []
    for gene in GENES:
        combined_motifs = pd.concat(
            [
                ag_candidate_regions.assign(model="AlphaGenome"),
                bz_candidate_regions.assign(model="Borzoi"),
            ],
            ignore_index=True,
        )
        sensitivity_uri, sensitivity_data = build_sensitivity_figure(
            gene, ag_views, bz_views, controls, combined_motifs
        )
        expression_uri, expression_data = build_observed_expression_figure(
            gene, observed
        )
        sensitivity_rows.append(sensitivity_data)
        observed_rows.append(expression_data)
        image_blocks.extend(
            [
                {
                    "id": f"{gene.lower()}_sensitivity",
                    "type": "html",
                    "body": (
                        f'<img alt="{gene} signed 3 kb sensitivity" '
                        'style="display:block;width:100%;max-height:220px;'
                        f'object-fit:contain" src="{sensitivity_uri}">'
                    ),
                },
                {
                    "id": f"{gene.lower()}_expression",
                    "type": "html",
                    "body": (
                        f'<img alt="{gene} observed expression source" '
                        'style="display:block;width:100%;max-height:220px;'
                        f'object-fit:contain" src="{expression_uri}">'
                    ),
                },
            ]
        )
        # Keep stand-alone images next to the report as auditable artifacts.
        for name, uri in [
            ("signed_sensitivity", sensitivity_uri),
            ("observed_expression", expression_uri),
        ]:
            (figures_dir / f"{gene}_{name}.png").write_bytes(
                base64.b64decode(uri.split(",", 1)[1])
            )

    sensitivity_data = pd.concat(sensitivity_rows, ignore_index=True)
    observed_data = pd.concat(observed_rows, ignore_index=True)
    sensitivity_path = data_dir / "three_gene_signed_sensitivity.tsv"
    observed_subset_path = data_dir / "three_gene_observed_expression.tsv"
    sensitivity_data.to_csv(sensitivity_path, sep="\t", index=False)
    observed_data.to_csv(observed_subset_path, sep="\t", index=False)

    control_rows: list[dict[str, object]] = []
    for row in ag_controls.itertuples(index=False):
        control_rows.append(
            {
                "model": "AlphaGenome",
                "gene": row.gene,
                "motif": row.label,
                "motif_loss": bool(row.native_motif_loss_confirmed),
                "best_score": round(float(row.m3_best_score), 2),
                "window_score": round(
                    float(row.m3_window_percentile_score), 2
                ),
                "formal_window_recovery": bool(
                    row.m3_window_recalled_at_threshold
                ),
                "local_score": round(
                    float(row.local_regulatory_best_score), 2
                ),
                "output_score": round(float(row.output_best_score), 2),
                "signed_log2fc": round(
                    float(row.m3_best_median_group_signed_log2fc), 4
                ),
                "negative_group_fraction": round(
                    float(row.m3_best_loss_direction_group_fraction), 3
                ),
            }
        )
    bz_join = bz_controls.merge(
        bz_view_controls,
        on=["control_id", "gene"],
        suffixes=("", "_view"),
    )
    labels = controls.set_index("control_id")
    for row in bz_join.itertuples(index=False):
        label = labels.loc[row.control_id]
        control_rows.append(
            {
                "model": "Borzoi",
                "gene": row.gene,
                "motif": label.label,
                "motif_loss": True,
                "best_score": round(float(row.best_score), 2),
                "window_score": round(
                    float(row.window_percentile_score), 2
                ),
                "formal_window_recovery": bool(
                    row.window_recalled_at_threshold
                ),
                "local_score": round(
                    float(row.local_regulatory_best_score), 2
                ),
                "output_score": round(float(row.output_best_score), 2),
                "signed_log2fc": round(
                    float(row.best_median_group_signed_log2fc), 4
                ),
                "negative_group_fraction": round(
                    float(row.best_loss_direction_group_fraction), 3
                ),
            }
        )
    control_table = pd.DataFrame.from_records(control_rows)
    control_path = data_dir / "positive_control_model_evidence.tsv"
    control_table.to_csv(control_path, sep="\t", index=False)

    region_frames: list[pd.DataFrame] = []
    for model, frame in [
        ("AlphaGenome", ag_candidate_regions),
        ("Borzoi", bz_candidate_regions),
    ]:
        selected = frame[
            [
                "screen_rank",
                "gene",
                "region_label",
                "region_score",
                "peak_score",
                "driving_track",
                "family",
                "top_motif",
                "annotation_status",
                "matches_expected_family",
            ]
        ].copy()
        selected.insert(0, "model", model)
        selected["region_score"] = selected.region_score.round(2)
        selected["peak_score"] = selected.peak_score.round(2)
        region_frames.append(selected)
    region_table = pd.concat(region_frames, ignore_index=True)
    region_path = data_dir / "score95_region_motif_evidence.tsv"
    region_table.to_csv(region_path, sep="\t", index=False)

    bz_validation = json.loads(
        (bz_dir / "validation_summary.json").read_text()
    )
    ag_formal = int(
        control_table.loc[
            control_table.model.eq("AlphaGenome"),
            "formal_window_recovery",
        ].sum()
    )
    bz_formal = int(
        control_table.loc[
            control_table.model.eq("Borzoi"),
            "formal_window_recovery",
        ].sum()
    )
    ag_motif_count = int(
        ag_candidate_regions.annotation_status.eq(
            "consistent_native_motif_loss"
        ).sum()
    )
    bz_motif_count = int(
        bz_candidate_regions.annotation_status.eq(
            "consistent_native_motif_loss"
        ).sum()
    )
    summary = pd.DataFrame(
        [
            {
                "centers": 4474,
                "positive_controls": 4,
                "ag_formal_recovery": ag_formal,
                "bz_formal_recovery": bz_formal,
                "ag_motif_annotated_regions": ag_motif_count,
                "bz_motif_annotated_regions": bz_motif_count,
            }
        ]
    )

    method_path = summary_dir / "method_comparison.tsv"
    concordance_path = (
        bz_dir / "alphagenome_borzoi_concordance.tsv"
    )
    sources = [
        build_source_spec(
            "methods",
            "Progressive AlphaGenome score comparison",
            method_path,
            root,
        ),
        build_source_spec(
            "controls",
            "Positive-control model and motif evidence",
            control_path,
            root,
        ),
        build_source_spec(
            "regions",
            "Score >95 regions and motif annotations",
            region_path,
            root,
        ),
        build_source_spec(
            "concordance",
            "AlphaGenome-Borzoi frozen-score concordance",
            concordance_path,
            root,
        ),
        build_source_spec(
            "sensitivity",
            "Three-gene signed sensitivity curves",
            sensitivity_path,
            root,
        ),
        build_source_spec(
            "observed",
            "Observed fine-tuning expression source",
            observed_subset_path,
            root,
        ),
    ]

    tables: list[dict[str, object]] = [
        {
            "id": "method_table",
            "title": "AlphaGenome 渐进评分比较",
            "dataset": "methods",
            "sourceId": "methods",
            "columns": [
                table_column_spec("method", "Method"),
                table_column_spec(
                    "positive_control_windows_recalled",
                    "Formal windows",
                    "number",
                ),
                table_column_spec(
                    "positive_controls_recalled_either_view_any_center",
                    "Either-view centers",
                    "number",
                ),
                table_column_spec(
                    "score95_regions", "Regions >95", "number"
                ),
                table_column_spec(
                    "score95_singleton_regions", "Singletons", "number"
                ),
                table_column_spec("score_definition", "Definition"),
            ],
        },
        {
            "id": "control_table",
            "title": "正向对照：motif 是否被破坏与模型是否正式恢复",
            "dataset": "controls",
            "sourceId": "controls",
            "columns": [
                table_column_spec("model", "Model"),
                table_column_spec("gene", "Gene"),
                table_column_spec("motif", "Motif"),
                table_column_spec("motif_loss", "Motif loss"),
                table_column_spec("best_score", "Best score", "number"),
                table_column_spec("window_score", "Window score", "number"),
                table_column_spec(
                    "formal_window_recovery", "Window >95"
                ),
                table_column_spec("local_score", "Local", "number"),
                table_column_spec("output_score", "Output", "number"),
                table_column_spec(
                    "signed_log2fc", "Signed log2FC", "number"
                ),
            ],
        },
        {
            "id": "concordance_table",
            "title": "同一冻结分数的跨模型一致性",
            "dataset": "concordance",
            "sourceId": "concordance",
            "columns": [
                table_column_spec("gene", "Gene"),
                table_column_spec(
                    "overall_score_spearman", "Overall ρ", "number"
                ),
                table_column_spec(
                    "output_view_score_spearman", "Output ρ", "number"
                ),
                table_column_spec(
                    "local_view_score_spearman", "Local ρ", "number"
                ),
                table_column_spec(
                    "score95_center_jaccard", ">95 Jaccard", "number"
                ),
                table_column_spec(
                    "driving_view_agreement_fraction",
                    "View agreement",
                    "number",
                ),
            ],
        },
    ]
    charts = [
        {
            "id": "method_singletons",
            "title": "渐进评分如何消除孤立高分区域",
            "subtitle": (
                "Method 3 的选择重点是跨视图语义与空间稳健性，"
                "不是最大化四个对照的召回。"
            ),
            "type": "bar",
            "dataset": "methods",
            "sourceId": "methods",
            "layout": "full",
            "encodings": {
                "x": {
                    "field": "method",
                    "type": "nominal",
                    "label": "Method",
                },
                "y": {
                    "field": "score95_singleton_regions",
                    "type": "quantitative",
                    "label": "Singleton score >95 regions",
                },
            },
            "yAxisTitle": "Singleton regions",
            "valueFormat": "number",
        }
    ]
    region_blocks: list[dict[str, object]] = []
    for model in ["AlphaGenome", "Borzoi"]:
        for gene in GENES:
            count = int(
                (
                    region_table.model.eq(model)
                    & region_table.gene.eq(gene)
                ).sum()
            )
            parts = max(1, int(np.ceil(count / 15)))
            for part in range(parts):
                dataset = (
                    f"regions_{model.lower()}_{gene.lower()}_part{part + 1}"
                )
                table_id = f"table_{dataset}"
                suffix = (
                    f" · part {part + 1}/{parts}" if parts > 1 else ""
                )
                tables.append(
                    {
                        "id": table_id,
                        "title": (
                            f"{model} · {gene} · score >95 regions{suffix}"
                        ),
                        "dataset": dataset,
                        "sourceId": "regions",
                        "columns": [
                            table_column_spec("screen_rank", "Rank", "number"),
                            table_column_spec("region_label", "Region"),
                            table_column_spec(
                                "region_score", "Region score", "number"
                            ),
                            table_column_spec("driving_track", "View"),
                            table_column_spec("family", "Motif family"),
                            table_column_spec("top_motif", "Top motif"),
                            table_column_spec(
                                "annotation_status", "Motif-loss status"
                            ),
                        ],
                    }
                )
                region_blocks.append(
                    {
                        "id": f"block_{table_id}",
                        "type": "table",
                        "tableId": table_id,
                    }
                )

    cards = [
        {
            "id": "centers",
            "dataset": "summary",
            "sourceId": "methods",
            "metrics": [
                {"label": "Centers", "field": "centers", "format": "number"}
            ],
        },
        {
            "id": "ag_recall",
            "dataset": "summary",
            "sourceId": "controls",
            "metrics": [
                {
                    "label": "AG formal controls",
                    "field": "ag_formal_recovery",
                    "format": "number",
                }
            ],
        },
        {
            "id": "bz_recall",
            "dataset": "summary",
            "sourceId": "controls",
            "metrics": [
                {
                    "label": "Borzoi formal controls",
                    "field": "bz_formal_recovery",
                    "format": "number",
                }
            ],
        },
        {
            "id": "motif_regions",
            "dataset": "summary",
            "sourceId": "regions",
            "metrics": [
                {
                    "label": "AG motif-backed regions",
                    "field": "ag_motif_annotated_regions",
                    "format": "number",
                }
            ],
        },
    ]

    method3 = method_comparison.loc[
        method_comparison.method.eq("method3_modality_views")
    ].iloc[0]
    mean_rho = concordance.overall_score_spearman.mean()
    blocks: list[dict[str, object]] = [
        {
            "id": "title",
            "type": "markdown",
            "body": (
                "# Mdk、Col1a1、Acta2：原始 AlphaGenome 到原始 Borzoi 的"
                "全 3 kb 多轨道敏感度\n\n"
                "**冻结评分契约、motif-loss 证据与正向对照审计 · "
                f"{datetime.now().astimezone().date().isoformat()}**"
            ),
        },
        {
            "id": "technical_summary",
            "type": "markdown",
            "body": (
                "## 技术结论\n\n"
                f"在不使用 HSC specificity、也不按正向对照调参的前提下，"
                f"最终冻结的 Method 3 在 AlphaGenome 对 4 个正式 motif 窗口恢复 "
                f"**{ag_formal}/4**，在 Borzoi 恢复 **{bz_formal}/4**。"
                "四个正向对照的原生 motif 均被突变实际破坏，因此未恢复反映的是"
                "模型响应或评分层面的缺失，而不是突变没有打中 motif。"
                f"三基因跨模型总体分数 Spearman 均值为 **{mean_rho:.3f}**。"
                "Mdk 没有参与评分选择的正式正向对照；+463..+531 只作为独立"
                "motif-backed holdout。"
            ),
        },
        {
            "id": "metrics",
            "type": "metric-strip",
            "cardIds": ["centers", "ag_recall", "bz_recall", "motif_regions"],
        },
        {
            "id": "key_findings",
            "type": "markdown",
            "body": (
                "## 关键发现与为何冻结 Method 3\n\n"
                "Method 1 的单点高分太多，Method 2 加入空间和至少三个轨道组支持后"
                "大幅减少孤立点；Method 3 再把 output 与 local-regulatory 两种"
                "生物学语义分开计算，最后取两视图较强者并在基因内重排。"
                f"它在 AlphaGenome 产生 {int(method3.score95_regions)} 个区域且"
                f"孤立区域为 {int(method3.score95_singleton_regions)}。选择依据是"
                "语义、可移植性和空间稳健性，不是最大化 4 个对照的召回。"
            ),
        },
        {
            "id": "method_block",
            "type": "table",
            "tableId": "method_table",
        },
        {
            "id": "method_chart_block",
            "type": "chart",
            "chartId": "method_singletons",
        },
    ]
    for gene in GENES:
        if gene == "Mdk":
            gene_text = (
                "Mdk 没有正式正向对照；独立 holdout +463..+531 在 AlphaGenome "
                "保持高分，且该区可对应 KLF/SP 与 EGR/ZBTB motif。另一个最强区"
                " −553..−539 对应 AP-1/FOSL2。"
            )
        elif gene == "Col1a1":
            gene_text = (
                "Col1a1 的正式对照分别针对 SMAD3/4、AP-1 和 SP1/KLF6。"
                "三者的 motif 均被有效破坏，但是否达到 >95 必须以整段同宽窗口"
                "校准分数判断，不能用窗口内任一点碰巧超过 95 代替。"
            )
        else:
            gene_text = (
                "Acta2 的正式对照是 +1294..+1303 的 SRF/CArG。"
                "AlphaGenome 中该区以 local-regulatory 视图为主，且 signed "
                "log2FC 为负，符合 motif 破坏导致预测信号下降的方向。"
            )
        blocks.extend(
            [
                {
                    "id": f"{gene.lower()}_heading",
                    "type": "markdown",
                    "body": f"## {gene}\n\n{gene_text}",
                },
                next(
                    block
                    for block in image_blocks
                    if block["id"] == f"{gene.lower()}_sensitivity"
                ),
                next(
                    block
                    for block in image_blocks
                    if block["id"] == f"{gene.lower()}_expression"
                ),
            ]
        )
    blocks.extend(
        [
            {
                "id": "controls_heading",
                "type": "markdown",
                "body": (
                    "## 正向对照与突变表现\n\n"
                    "`best_score` 是区间内最优中心，容易受多次搜索影响；"
                    "`window_score` 把候选区与基因内所有同宽滑窗比较，是本报告"
                    "的正式恢复判据。`>95` 只表示在本基因约 3 kb 扫描中位于"
                    "前 5%，不是 95% 概率、p 值、FDR 或功能恢复率。"
                ),
            },
            {
                "id": "controls_block",
                "type": "table",
                "tableId": "control_table",
            },
            {
                "id": "concordance_heading",
                "type": "markdown",
                "body": (
                    "## 原始 Borzoi 的精简适配\n\n"
                    "没有重新拟合权重：仍要求每个视图至少三个轨道组支持，"
                    "仍做 ±4 bp 空间中位数，仍在视图内和基因内做经验百分位。"
                    "唯一适配是按 Borzoi 实际轨道 manifest 映射 output 视图"
                    "（RNA/CAGE）与 local-regulatory 视图（accessibility、"
                    "histone、TF）；最终包含 7 个 output 组和 6 个 local 组。"
                ),
            },
            {
                "id": "concordance_block",
                "type": "table",
                "tableId": "concordance_table",
            },
            {
                "id": "regions_heading",
                "type": "markdown",
                "body": (
                    "## >95 区域与对应 motif\n\n"
                    "以下为全部区域级调用。`consistent_native_motif_loss` 表示"
                    "至少两种替换重复地削弱同一原生 motif；`UNRESOLVED` 表示"
                    "当前 JASPAR family 扫描未得到稳定 motif-loss 归因，"
                    "不等于该区域无功能。"
                ),
            },
            *region_blocks,
            {
                "id": "methodology",
                "type": "markdown",
                "body": (
                    "## 分数如何计算\n\n"
                    "1. 每个中心、轨道对 3 次 10-bp composition-preserving "
                    "shuffle 取 `median signed log2FC` 与 `median |log2FC|`。"
                    "\n2. output 视图使用 TSS ±512 bp 的 ratio-of-sums；"
                    "local 视图使用突变所在 128-bp bin 及左右各一 bin 中最强的"
                    " signed 响应。\n3. 各 track group 在同一基因内转经验百分位，"
                    "取第三高 group，要求信号不是单一轨道偶然值。\n4. 在 ±4 bp "
                    "中心邻域取空间中位数，再分别得到 output/local 视图百分位。"
                    "\n5. 两视图取较强者，并在同一基因内再次重排为 0–100。"
                    "严格 `score > 95` 就是该基因扫描中心的前 5%。"
                ),
            },
            {
                "id": "limitations",
                "type": "markdown",
                "body": (
                    "## 局限与稳健性边界\n\n"
                    "- 41 个“命中中心”实际只对应 4 个相互重叠的 motif 窗口，"
                    "不能当作 41 个独立正例。\n"
                    "- 一个含 10–11 个中心的窗口即使随机抽样，也较容易出现某个"
                    "中心 >95；因此正式结论使用同宽窗口百分位。\n"
                    "- 正向对照分布不均：Acta2 1 个、Col1a1 3 个、Mdk 0 个，"
                    "三基因平均召回会掩盖这种不平衡。\n"
                    "- 分数是本基因、本 3 kb 扫描内的相对重要性；它不估计"
                    " genome-wide 错误率，也不能把正 signed log2FC 自动解释为"
                    " motif gain。\n"
                    "- observed CPM 图是微调数据源的原始聚合轨道，不能作为"
                    "突变因果证据。"
                ),
            },
            {
                "id": "next_steps",
                "type": "markdown",
                "body": (
                    "## 下一步\n\n"
                    "1. 保持 score contract 不变，扩展到更多基因以估计正式"
                    " precision–recall，而不是继续围绕这 4 个对照调分。\n"
                    "2. 对 AlphaGenome/Borzoi 同时 >95 且 motif-loss 一致的区域"
                    "优先做 MPRA/CRISPR 或更小窗口 saturation。\n"
                    "3. 对高分但 UNRESOLVED 的区域补充 motif grammar、"
                    "非 JASPAR motif 与 splice/sequence-complexity 审计。"
                ),
            },
            {
                "id": "questions",
                "type": "markdown",
                "body": (
                    "## 尚未回答的问题\n\n"
                    "- 对跨模型方向相反、但两边都高分的区域，实验优先级应降低"
                    "还是只标记为机制不确定？\n"
                    "- output 与 local-regulatory 视图是否应在未来的实验标签上"
                    "分别校准阈值？\n"
                    "- Mdk 的 +463..+531 holdout 是否需要拆成 KLF/SP 与 "
                    "EGR/ZBTB 两个更窄的独立验证元件？"
                ),
            },
        ]
    )

    datasets: dict[str, list[dict[str, object]]] = {
        "summary": dataframe_records(summary),
        "methods": dataframe_records(method_comparison),
        "controls": dataframe_records(control_table),
        "concordance": dataframe_records(concordance.round(4)),
    }
    for model in ["AlphaGenome", "Borzoi"]:
        for gene in GENES:
            selected_regions = region_table.loc[
                region_table.model.eq(model)
                & region_table.gene.eq(gene)
            ].sort_values("screen_rank")
            for part, start in enumerate(
                range(0, len(selected_regions), 15), start=1
            ):
                key = (
                    f"regions_{model.lower()}_{gene.lower()}_part{part}"
                )
                datasets[key] = dataframe_records(
                    selected_regions.iloc[start : start + 15]
                )

    generated = datetime.now().astimezone().isoformat(timespec="seconds")
    artifact = {
        "surface": "report",
        "manifest": {
            "version": 1,
            "surface": "report",
            "title": "三基因原始模型多轨道敏感度与 motif 召回",
            "description": (
                "Progressive frozen-score analysis for Mdk, Col1a1 and Acta2."
            ),
            "generatedAt": generated,
            "sources": sources,
            "cards": cards,
            "charts": charts,
            "tables": tables,
            "blocks": blocks,
        },
        "snapshot": {
            "version": 1,
            "status": "ready",
            "generatedAt": generated,
            "datasets": datasets,
        },
        "sources": sources,
    }
    artifact_path = report_dir / "artifact.json"
    artifact_path.write_text(
        json.dumps(artifact, indent=2, ensure_ascii=False) + "\n"
    )

    chart_map = pd.DataFrame(
        [
            {
                "report_segment": f"{gene} sensitivity",
                "analytical_question": (
                    "Where are signed output/local effects across 3 kb?"
                ),
                "visual": "two-panel line profile",
                "fields": "offset, model, view, median_group_signed_log2fc",
                "supported_claim": (
                    "location, direction and cross-model replication"
                ),
            }
            for gene in GENES
        ]
        + [
            {
                "report_segment": f"{gene} observed source",
                "analytical_question": (
                    "What expression signal entered fine-tuning?"
                ),
                "visual": "four-panel line profile",
                "fields": "offset, cell, observed_cpm",
                "supported_claim": "observed source signal, not causal effect",
            }
            for gene in GENES
        ]
    )
    chart_map.to_csv(
        report_dir / "chart_map.tsv", sep="\t", index=False
    )
    validation = {
        "status": "ok",
        "artifact": str(artifact_path),
        "genes": GENES,
        "sensitivity_rows": int(len(sensitivity_data)),
        "observed_rows": int(len(observed_data)),
        "control_rows": int(len(control_table)),
        "region_rows": int(len(region_table)),
        "ag_formal_window_recovery": ag_formal,
        "bz_formal_window_recovery": bz_formal,
        "borzoi_score_rule_changed": bool(
            bz_validation["score_rule_changed_after_alphagenome"]
        ),
    }
    if validation["sensitivity_rows"] != 17896:
        raise RuntimeError(validation)
    if validation["observed_rows"] != 9000:
        raise RuntimeError(validation)
    if validation["control_rows"] != 8:
        raise RuntimeError(validation)
    if validation["borzoi_score_rule_changed"]:
        raise RuntimeError(validation)
    (report_dir / "report_input_validation.json").write_text(
        json.dumps(validation, indent=2) + "\n"
    )
    print(json.dumps(validation, indent=2))


if __name__ == "__main__":
    main()
