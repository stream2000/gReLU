#!/usr/bin/env python
"""Create the portable-report artifact for the Mdk specificity audit."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[6]
DEFAULT_ROOT = REPO_ROOT / "experiments/ism/saijou_targeted_original_comparison"
REPORT_TITLE = "Saijou HSC ISM：靶向位点比较与 Mdk 细胞特异性审计"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    return parser.parse_args()


def records(table: pd.DataFrame) -> list[dict]:
    return json.loads(table.to_json(orient="records"))


def source(source_id: str, label: str, path: str, command: str) -> dict:
    if path.endswith(".json"):
        sql = f"SELECT * FROM read_json_auto('{path}')"
    elif path.endswith(".csv"):
        sql = f"SELECT * FROM read_csv_auto('{path}', header=true)"
    else:
        target = path if path.endswith(".tsv") else f"{path}/*.tsv"
        sql = (
            f"SELECT * FROM read_csv_auto('{target}', delim='\\t', header=true, "
            "union_by_name=true, filename=true)"
        )
    return {
        "id": source_id,
        "label": label,
        "path": path,
        "query": {
            "engine": "duckdb",
            "language": "sql",
            "sql": sql,
            "description": f"{label}. Transformation command: {command}",
            "tables_used": [path],
        },
    }


def _load_report_tables(root: Path) -> dict[str, pd.DataFrame]:
    analysis = root / "analysis"
    deep = analysis / "cell_specificity_deep_dive"
    names = {
        "decomposition": "cell_metric_decomposition.tsv",
        "centers": "mdk_hub_center_specificity.tsv",
        "bootstrap": "bootstrap_specificity.tsv",
        "checkpoint": "checkpoint_stability.tsv",
        "cross_model": "cross_model_selectivity_concordance.tsv",
        "observed": "observed_signal_by_readout.tsv",
        "training": "training_split_overlap.tsv",
        "heads": "head_geometry.tsv",
        "background_windows": "background_window_scan.tsv",
        "background_hub": "background_hub_enrichment.tsv",
        "seeds": "shuffle_seed_replication.tsv",
        "ref_fit": "reference_profile_fit.tsv",
        "motifs": "motif_disruption_associations.tsv",
    }
    tables = {key: pd.read_csv(deep / filename, sep="\t") for key, filename in names.items()}
    tables["candidate"] = pd.read_csv(analysis / "key_comparison_table.tsv", sep="\t")
    tables["validation"] = pd.read_csv(
        REPO_ROOT
        / "experiments/validation/ag_active_aligned128_epoch19_vs_borzoi_epoch39/summary.csv"
    )
    return tables


def _candidate_loci_table(candidate: pd.DataFrame) -> pd.DataFrame:
    selected_series = [
        "AG-FT HSC RNA",
        "BZ-FT HSC RNA",
        "BZ-original HSC CAGE",
        "AG-original liver RNA",
        "BZ-original liver RNA",
    ]
    table = candidate.loc[
        candidate.series_label.isin(selected_series),
        ["gene", "locus_id", "series_label", "median_effect", "median_abs_effect", "mutations"],
    ].copy()
    table["locus"] = table.locus_id.map(
        {
            "mdk_intron_hub_441_507": "Mdk hub +441..+507",
            "mdk_splice_donor_369": "Mdk splice donor +369",
            "acta2_202_promoter_carg": "Acta2 promoter CArG",
            "acta2_202_intron1_carg": "Acta2 intron-1 CArG",
            "col1a1_promoter_inverted_ccaat": "Col1a1 inverted CCAAT",
            "timp1_promoter_ets_like_43": "Timp1 ETS-like +43",
        }
    )
    return table[
        ["locus", "series_label", "median_effect", "median_abs_effect", "mutations"]
    ]


def _robustness_tables(
    bootstrap: pd.DataFrame,
    background_hub: pd.DataFrame,
    seeds: pd.DataFrame,
    motifs: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    robustness = bootstrap.merge(background_hub, on="model").merge(seeds, on="model")
    robustness = robustness[
        [
            "model", "probability_hsc_top", "ratio_ci_low", "ratio_ci_high",
            "ratio_percentile", "margin_percentile", "hsc_margin_spearman",
        ]
    ]
    motif_summary = pd.concat(
        [
            motifs.loc[motifs.level.eq("family")],
            motifs.loc[motifs.level.eq("motif")]
            .sort_values(["model", "qvalue", "pvalue"])
            .groupby("model")
            .head(5),
        ],
        ignore_index=True,
    )
    return robustness, motif_summary[
        ["model", "feature", "centers", "spearman", "pvalue", "qvalue"]
    ]


def _prepare_report_data(
    root: Path,
) -> tuple[dict[str, pd.DataFrame], pd.Series, pd.Series]:
    tables = _load_report_tables(root)
    decomposition = tables["decomposition"]
    centers = tables["centers"]
    bootstrap = tables["bootstrap"]
    checkpoint = tables["checkpoint"]
    cross_model = tables["cross_model"]
    observed = tables["observed"]
    background_windows = tables["background_windows"]
    background_hub = tables["background_hub"]
    ref_fit = tables["ref_fit"]

    ag = decomposition.loc[decomposition.model.eq("AlphaGenome e19")].iloc[0]
    bz = decomposition.loc[decomposition.model.eq("Borzoi e39")].iloc[0]
    ag_boot = bootstrap.loc[bootstrap.model.eq("AlphaGenome e19")].iloc[0]
    bz_boot = bootstrap.loc[bootstrap.model.eq("Borzoi e39")].iloc[0]
    ag_bg = background_hub.loc[background_hub.model.eq("AlphaGenome e19")].iloc[0]
    bz_bg = background_hub.loc[background_hub.model.eq("Borzoi e39")].iloc[0]
    selectivity_rho = cross_model.spearman_hsc_margin.iloc[0]

    summary = pd.DataFrame(
        [{
            "borzoi_hsc_ratio": bz.relative_hsc_ratio,
            "borzoi_bootstrap_top_probability": bz_boot.probability_hsc_top,
            "alphagenome_hsc_ratio": ag.relative_hsc_ratio,
            "alphagenome_bootstrap_top_probability": ag_boot.probability_hsc_top,
            "borzoi_background_percentile": bz_bg.ratio_percentile,
            "alphagenome_background_percentile": ag_bg.ratio_percentile,
            "cross_model_selectivity_rho": selectivity_rho,
        }]
    )

    metric_long = decomposition.melt(
        id_vars=["model"],
        value_vars=["baseline_hsc_ratio", "raw_delta_hsc_ratio", "relative_hsc_ratio"],
        var_name="metric",
        value_name="hsc_ratio",
    )
    metric_long["metric"] = metric_long.metric.map(
        {
            "baseline_hsc_ratio": "Reference baseline",
            "raw_delta_hsc_ratio": "Absolute delta",
            "relative_hsc_ratio": "Relative log2 effect",
        }
    )
    centers = centers.sort_values(
        ["variant_offset_from_tss_transcription_bp", "model"]
    ).reset_index(drop=True)
    background_plot = pd.concat(
        [
            part
            for _, group in background_windows.groupby("model", sort=False)
            for part in (group.iloc[::10], group.loc[group.is_target_hub])
        ]
    ).drop_duplicates(["model", "start_offset", "end_offset"])
    background_plot = background_plot.sort_values(["start_offset", "model"]).reset_index(drop=True)
    checkpoint["family"] = checkpoint.model.str.split().str[0]
    checkpoint["checkpoint"] = checkpoint.model.str.replace("AlphaGenome ", "AG ", regex=False).str.replace("Borzoi ", "BZ ", regex=False)
    observed_gene = observed.loc[observed.readout_role.eq("gene_body")].copy()
    ref_gene = ref_fit.loc[ref_fit.scope.eq("gene_body")].copy()
    ref_gene["model_cell"] = ref_gene.model.str.replace("AlphaGenome", "AG", regex=False).str.replace("Borzoi", "BZ", regex=False) + " / " + ref_gene.cell_type.str.upper()

    candidate_table = _candidate_loci_table(tables["candidate"])
    robustness, motif_summary = _robustness_tables(
        bootstrap, background_hub, tables["seeds"], tables["motifs"]
    )

    frames = {
        "summary": summary,
        "metric_decomposition": metric_long,
        "center_specificity": centers,
        "checkpoint": checkpoint,
        "background_windows": background_plot,
        "observed_gene_body": observed_gene,
        "reference_fit": ref_gene,
        "candidate_loci": candidate_table,
        "robustness": robustness,
        "motif_summary": motif_summary,
        "training_overlap": tables["training"],
        "head_geometry": tables["heads"],
        "positive_control_validation": tables["validation"],
    }
    return frames, ag_bg, bz_bg


def _build_sources() -> list[dict]:
    sources = [
        source(
            "deep_dive",
            "Mdk cell-specificity audit tables",
            "experiments/ism/saijou_targeted_original_comparison/analysis/cell_specificity_deep_dive",
            "python scripts/ism/experiments/saijou_hsc/tools/mdk/analyze_mdk_cell_specificity_deep_dive.py",
        ),
        source(
            "motif_scan",
            "JASPAR2024 vertebrate motif disruption scan",
            "experiments/ism/saijou_targeted_original_comparison/analysis/cell_specificity_deep_dive/motif_disruption_associations.tsv",
            "python scripts/ism/experiments/saijou_hsc/tools/mdk/analyze_mdk_motif_disruption.py",
        ),
        source(
            "targeted_comparison",
            "Same-manifest original and fine-tuned targeted ISM",
            "experiments/ism/saijou_targeted_original_comparison/analysis/key_comparison_table.tsv",
            "python scripts/ism/experiments/saijou_hsc/tools/mdk/analyze_saijou_targeted_original_comparison.py",
        ),
        source(
            "split_manifest",
            "Saijou chromosome split manifest",
            "dataset_store/splits/split_chr10_chr11/manifest.json",
            "python scripts/ism/experiments/saijou_hsc/tools/mdk/analyze_mdk_cell_specificity_deep_dive.py",
        ),
        source(
            "positive_controls",
            "Held-out positive-control profile validation",
            "experiments/validation/ag_active_aligned128_epoch19_vs_borzoi_epoch39/summary.csv",
            "python src/ft-scripts/plot_borzoi_positive_controls.py",
        ),
    ]
    return sources


def _build_report_spec(
    ag_bg: pd.Series, bz_bg: pd.Series
) -> tuple[list[dict], list[dict], list[dict], list[dict]]:
    cards = [
        {"id": "bz_ratio", "description": "Borzoi e39: HSC relative ISM effect divided by the largest non-HSC head.", "dataset": "summary", "sourceId": "deep_dive", "metrics": [{"label": "Borzoi HSC / max other", "field": "borzoi_hsc_ratio", "format": "number"}]},
        {"id": "bz_boot", "description": "Center-cluster bootstrap probability that HSC remains the top head.", "dataset": "summary", "sourceId": "deep_dive", "metrics": [{"label": "Borzoi P(HSC top)", "field": "borzoi_bootstrap_top_probability", "format": "percent"}]},
        {"id": "ag_ratio", "description": "AlphaGenome e19: HSC relative ISM effect divided by the largest non-HSC head.", "dataset": "summary", "sourceId": "deep_dive", "metrics": [{"label": "AlphaGenome HSC / max other", "field": "alphagenome_hsc_ratio", "format": "number"}]},
        {"id": "ag_boot", "description": "Center-cluster bootstrap probability that HSC remains the top head.", "dataset": "summary", "sourceId": "deep_dive", "metrics": [{"label": "AlphaGenome P(HSC top)", "field": "alphagenome_bootstrap_top_probability", "format": "percent"}]},
        {"id": "cross_rho", "description": "Spearman agreement of per-center HSC selectivity margins across the two final models.", "dataset": "summary", "sourceId": "deep_dive", "metrics": [{"label": "Cross-model selectivity rho", "field": "cross_model_selectivity_rho", "format": "number", "signed": True}]},
    ]

    charts = [
        {
            "id": "metric_decomposition", "title": "Mdk hub HSC specificity by effect definition",
            "subtitle": "HSC value divided by the largest mac/LSEC/chol value; 102 edits, 34 centers, 3 shuffles per center.",
            "type": "bar", "dataset": "metric_decomposition", "sourceId": "deep_dive", "layout": "full",
            "encodings": {"x": {"field": "metric", "type": "nominal", "label": "Effect definition"}, "y": {"field": "hsc_ratio", "type": "quantitative", "label": "HSC / max other"}, "color": {"field": "model", "type": "nominal", "label": "Model"}},
            "yAxisTitle": "HSC / max other", "valueFormat": "number",
        },
        {
            "id": "center_margin", "title": "Center-level HSC selectivity across the Mdk +441..+507 hub",
            "subtitle": "Margin = median |log2 ratio-of-sums| in HSC minus the largest non-HSC head; 3 shuffle edits per center.",
            "type": "line", "dataset": "center_specificity", "sourceId": "deep_dive", "layout": "full",
            "encodings": {"x": {"field": "variant_offset_from_tss_transcription_bp", "type": "quantitative", "label": "TSS offset, transcription direction (bp)"}, "y": {"field": "hsc_margin", "type": "quantitative", "label": "HSC selectivity margin"}, "color": {"field": "model", "type": "nominal", "label": "Model"}},
            "yAxisTitle": "HSC selectivity margin", "valueFormat": "number",
        },
        {
            "id": "checkpoint_emergence", "title": "Mdk HSC relative effect across fine-tuning checkpoints",
            "subtitle": "HSC median absolute relative effect divided by the largest non-HSC head; three checkpoints per model family.",
            "type": "bar", "dataset": "checkpoint", "sourceId": "deep_dive", "layout": "full",
            "encodings": {"x": {"field": "checkpoint", "type": "ordinal", "label": "Checkpoint"}, "y": {"field": "relative_hsc_ratio", "type": "quantitative", "label": "HSC / max other"}, "color": {"field": "family", "type": "nominal", "label": "Model family"}},
            "yAxisTitle": "HSC / max other", "valueFormat": "number",
        },
        {
            "id": "background_scan", "title": "Mdk ±0.5 kb background-window HSC specificity",
            "subtitle": "Every eligible 66-bp / 34-center window, strict 10-bp composition-preserving shuffles; target hub starts at +441 bp.",
            "type": "line", "dataset": "background_windows", "sourceId": "deep_dive", "layout": "full",
            "encodings": {"x": {"field": "start_offset", "type": "quantitative", "label": "Window start offset from TSS (bp)"}, "y": {"field": "hsc_vs_max_other_ratio", "type": "quantitative", "label": "HSC / max other"}, "color": {"field": "model", "type": "nominal", "label": "Model"}},
            "yAxisTitle": "HSC / max other", "valueFormat": "number",
        },
        {
            "id": "observed_gene_body", "title": "Observed Saijou Mdk gene-body 10X pseudobulk signal",
            "subtitle": "CPM bigWig sum across the 2,493-bp annotated Mdk gene body; four pooled cell types.",
            "type": "bar", "dataset": "observed_gene_body", "sourceId": "deep_dive", "layout": "full",
            "encodings": {"x": {"field": "cell_type", "type": "nominal", "label": "Cell type"}, "y": {"field": "observed_cpm_sum", "type": "quantitative", "label": "Observed CPM sum"}},
            "yAxisTitle": "Observed CPM sum", "valueFormat": "compact",
        },
        {
            "id": "reference_fit", "title": "Predicted reference-profile fit at the Mdk gene body",
            "subtitle": "Pearson correlation between saved reference predictions and the corresponding observed bigWig bins; resolution differs by model (AG 128 bp, BZ 32 bp).",
            "type": "bar", "dataset": "reference_fit", "sourceId": "deep_dive", "layout": "full",
            "encodings": {"x": {"field": "model_cell", "type": "nominal", "label": "Model / cell"}, "y": {"field": "pearson", "type": "quantitative", "label": "Pearson r"}, "color": {"field": "model", "type": "nominal", "label": "Model"}},
            "yAxisTitle": "Pearson r", "valueFormat": "number",
        },
    ]

    tables = [
        {
            "id": "candidate_loci", "title": "Targeted loci across fine-tuned and original readouts",
            "subtitle": "Signed and absolute median log2 ratio-of-sums; exact lookup table for the six tested elements.",
            "dataset": "candidate_loci", "sourceId": "targeted_comparison", "layout": "full",
            "defaultSort": {"field": "median_abs_effect", "direction": "desc"}, "density": "dense",
            "columns": [
                {"field": "locus", "label": "Locus", "type": "text"}, {"field": "series_label", "label": "Readout", "type": "text"},
                {"field": "median_effect", "label": "Signed median effect", "format": "number", "movement": True},
                {"field": "median_abs_effect", "label": "Median |effect|", "format": "number"}, {"field": "mutations", "label": "Edits", "format": "number"},
            ],
        },
        {
            "id": "robustness", "title": "Mdk specificity robustness checks",
            "subtitle": "Bootstrap uncertainty, broad-background percentile, and independent shuffle-seed concordance.",
            "dataset": "robustness", "sourceId": "deep_dive", "layout": "full",
            "defaultSort": {"field": "model", "direction": "asc"}, "density": "dense",
            "columns": [
                {"field": "model", "label": "Model", "type": "text"}, {"field": "probability_hsc_top", "label": "P(HSC top)", "format": "percent"},
                {"field": "ratio_ci_low", "label": "Ratio CI low", "format": "number"}, {"field": "ratio_ci_high", "label": "Ratio CI high", "format": "number"},
                {"field": "ratio_percentile", "label": "Background percentile", "format": "percent"}, {"field": "margin_percentile", "label": "Margin percentile", "format": "percent"},
                {"field": "hsc_margin_spearman", "label": "Shuffle-seed rho", "format": "number"},
            ],
        },
        {
            "id": "motif_summary", "title": "Canonical motif-family disruption associations",
            "subtitle": "Center-level Spearman tests across all 34 centers; JASPAR2024 vertebrate CORE, FIMO p<1e-3; FDR within model.",
            "dataset": "motif_summary", "sourceId": "motif_scan", "layout": "full",
            "defaultSort": {"field": "qvalue", "direction": "asc"}, "density": "dense",
            "columns": [
                {"field": "model", "label": "Model", "type": "text"}, {"field": "feature", "label": "Motif family", "type": "text"},
                {"field": "centers", "label": "Centers", "format": "number"}, {"field": "spearman", "label": "Spearman rho", "format": "number", "movement": True},
                {"field": "pvalue", "label": "P value", "format": "number"}, {"field": "qvalue", "label": "FDR q", "format": "number"},
            ],
        },
        {
            "id": "training_overlap", "title": "Training-split status of the candidate genes",
            "subtitle": "Input-window and supervised label-window overlap under split_chr10_chr11.",
            "dataset": "training_overlap", "sourceId": "split_manifest", "layout": "full",
            "defaultSort": {"field": "gene", "direction": "asc"}, "density": "dense",
            "columns": [
                {"field": "gene", "label": "Gene", "type": "text"}, {"field": "chrom", "label": "Chromosome", "type": "text"},
                {"field": "split", "label": "Split", "type": "text"}, {"field": "input_window_overlaps", "label": "Input overlaps", "format": "number"},
                {"field": "supervised_label_window_overlaps", "label": "Label overlaps", "format": "number"},
            ],
        },
        {
            "id": "reference_calibration", "title": "Mdk reference-profile shape and abundance calibration",
            "subtitle": "Gene-body bins; Pearson measures shape, while predicted/observed sum ratio measures abundance calibration.",
            "dataset": "reference_fit", "sourceId": "deep_dive", "layout": "full",
            "defaultSort": {"field": "model_cell", "direction": "asc"}, "density": "dense",
            "columns": [
                {"field": "model_cell", "label": "Model / cell", "type": "text"}, {"field": "resolution_bp", "label": "Resolution (bp)", "format": "number"},
                {"field": "bins", "label": "Bins", "format": "number"}, {"field": "pearson", "label": "Pearson r", "format": "number"},
                {"field": "observed_sum", "label": "Observed sum", "format": "compact"}, {"field": "predicted_sum", "label": "Predicted sum", "format": "compact"},
                {"field": "predicted_to_observed_ratio", "label": "Predicted / observed", "format": "number"},
            ],
        },
    ]

    blocks = [
        {"id": "title", "type": "markdown", "body": f"# {REPORT_TITLE}\n\n**技术报告 · 2026-07-11**"},
        {"id": "summary_text", "type": "markdown", "body": "## 技术结论：Borzoi 学到了稳健的 HSC 相对敏感性；AlphaGenome 只提供边际支持\n\nMdk +441..+507 的结果不能被概括为‘两个模型都明确复现了同一 HSC enhancer’。更准确的结论是：**Borzoi e39 的 HSC 相对 ISM 效应稳健，并且是在微调过程中出现的；AlphaGenome e19 虽仍将 HSC 排第一，但优势小、对中心抽样敏感。** 两个模型在‘哪些碱基窗口最 HSC-selective’上几乎不相关，因此目前证据支持的是 task-head-dependent sequence sensitivity，而不是已经定位到一个跨架构一致的单一 motif 机制。"},
        {"id": "headline_metrics", "type": "metric-strip", "cardIds": ["bz_ratio", "bz_boot", "ag_ratio", "ag_boot", "cross_rho"]},
        {"id": "candidate_heading", "type": "markdown", "body": "## 六个靶向元件给出了正控与反例\n\nCol1a1 promoter CCAAT 和 Acta2 promoter CArG 在原始组织轨道及部分微调读出中产生清晰效应；Acta2 intron-1 CArG 基本接近零。这个反例说明高表达或已知 motif 并不会自动变成 HSC-first ISM。Mdk hub 与 splice donor 的聚合效应都可见，但只有 hub 在两个最终模型的相对效应排序中为 HSC 第一。"},
        {"id": "candidate_table_block", "type": "table", "tableId": "candidate_loci", "layout": "full"},
        {"id": "metric_heading", "type": "markdown", "body": "## Borzoi 的 HSC 优势不是高基线或大 head norm 的产物\n\nAlphaGenome 的原始绝对变化随 HSC 高 reference baseline 放大，而相对效应只比最佳非 HSC 头高约 8%。Borzoi 则相反：HSC reference baseline 低于 LSEC/mac，原始绝对变化也不是最大，但归一化后的相对 ISM 效应高约 51%。最终 HSC head 的权重范数在两个模型中都不是最大，因此不能用一个全局输出增益解释。"},
        {"id": "metric_chart_block", "type": "chart", "chartId": "metric_decomposition", "layout": "full"},
        {"id": "center_heading", "type": "markdown", "body": "## 两个模型同意‘这里有序列效应’，但不同意‘哪里最 HSC-specific’\n\n中心级分析保留了 34 个位置作为独立抽样单位。Borzoi 在多数中心保持正 HSC margin；AlphaGenome 的整体排序更依赖少数极端窗口。两个最终模型的中心级 HSC margin Spearman 接近零，这会限制‘同一细胞特异机制被两种架构复现’的表述。"},
        {"id": "center_chart_block", "type": "chart", "chartId": "center_margin", "layout": "full"},
        {"id": "checkpoint_heading", "type": "markdown", "body": "## Borzoi 的 HSC specificity 在微调中出现并在晚期稳定\n\nBorzoi e00 时 HSC 排第四，晚期 e37/e39 转为第一且中心选择性轮廓高度一致；这是当前最直接的‘由 Saijou 微调学得’证据。AlphaGenome e12/e16/e19 一直是弱 HSC-first，说明它的边际优势可复现，但并未随晚期训练明显增强。"},
        {"id": "checkpoint_chart_block", "type": "chart", "chartId": "checkpoint_emergence", "layout": "full"},
        {"id": "background_heading", "type": "markdown", "body": f"## 广域背景支持 Borzoi hub 异常，但削弱 AlphaGenome 结论\n\n背景扫描覆盖 TSS 两侧约 0.5 kb，保持每个 10-bp 窗口的单核苷酸组成，并用与目标扫描不同的 shuffle seed 重复。在 398 个合格同长度窗口中，**Borzoi hub 的 HSC ratio 为 {bz_bg.hub_hsc_ratio:.3f}，位于第 {100*bz_bg.ratio_percentile:.1f} 百分位；HSC margin 位于第 {100*bz_bg.margin_percentile:.1f} 百分位。** 相反，**AlphaGenome hub ratio 为 {ag_bg.hub_hsc_ratio:.3f}，仅第 {100*ag_bg.ratio_percentile:.1f} 百分位，margin 仅第 {100*ag_bg.margin_percentile:.1f} 百分位。** 因此不同 shuffle seed 与广域背景共同支持 Borzoi 的局部异常性，但表明 AlphaGenome 的 HSC-first 聚合排序并不稳健。"},
        {"id": "background_chart_block", "type": "chart", "chartId": "background_scan", "layout": "full"},
        {"id": "robustness_table_block", "type": "table", "tableId": "robustness", "layout": "full"},
        {"id": "observed_heading", "type": "markdown", "body": "## 原始 10X 信号支持 Mdk 为 HSC-rich locus，但不能单独证明 enhancer specificity\n\nMdk gene body 与 3′/TES 区在 HSC pseudobulk 中显著高于其他细胞；TSS 附近却由 mac 更高。因为数据来自 10X 3′ 捕获，gene-body/TES readout 同时混合了表达量、转录本长度与 3′ 覆盖偏倚。ISM 的相对变化比 raw delta 更适合跨 cell head 比较，但仍不是直接的染色质可及性或 TF binding 测量。"},
        {"id": "observed_chart_block", "type": "chart", "chartId": "observed_gene_body", "layout": "full"},
        {"id": "reference_heading", "type": "markdown", "body": "## Reference profile 形状拟合较好，但非 HSC abundance 校准很差\n\nMdk gene-body 内各模型/细胞的 Pearson shape correlation 较高；然而 predicted/observed sum 暴露出明显校准偏差，尤其 Borzoi 对 mac 与 LSEC 分别高估约 17.7 倍与 41.0 倍。这个结果进一步说明跨 cell head 比较应使用相对 edit effect，而不能把 reference prediction 的绝对高度直接解释为细胞表达量。良好 shape fit 也不会把训练染色体上的拟合转化为因果或外推证据。"},
        {"id": "reference_chart_block", "type": "chart", "chartId": "reference_fit", "layout": "full"},
        {"id": "reference_table_block", "type": "table", "tableId": "reference_calibration", "layout": "full"},
        {"id": "motif_heading", "type": "markdown", "body": "## GC-rich 序列包含 SP/KLF/EGR grammar，但还没有单一 motif 解释 HSC margin\n\nde-novo 序列评分显示 hub 中存在高分 SP/KLF GC-box；但在 34 个中心上，把 JASPAR motif disruption 与 HSC-selective margin 做关联后，没有 motif 在任一模型中通过 5% FDR，也没有 canonical family 在两个模型中给出一致方向。因此‘GC-box enhancer’目前是合理假设，不是已恢复的机制。"},
        {"id": "motif_table_block", "type": "table", "tableId": "motif_summary", "layout": "full"},
        {"id": "scope_heading", "type": "markdown", "body": "## 证据边界：Mdk 位于训练染色体，两个模型也不是独立生物学复现\n\n两个微调模型使用相同的四条 Saijou pseudobulk 轨道。Mdk 位于 chr2，并与一个 supervised label window 重叠，因此真实 HSC enhancer grammar 与 locus-specific memorization 尚未分离。Col1a1 位于 held-out chr11，是更强的外推正控；Timp1 位于 chrX，本轮常染色体 split 中属于 OOD。"},
        {"id": "training_table_block", "type": "table", "tableId": "training_overlap", "layout": "full"},
        {"id": "validation_heading", "type": "markdown", "body": "## 模型整体具备可用的 profile 预测能力，但 Mdk 需要专门的 held-out 验证\n\n既有 positive-control 集中，AlphaGenome e19 的 all-track 平均 Pearson 为 0.831，Borzoi e39 为 0.781；held-out test loci 分别为 0.791 与 0.748。这支持继续做定位分析，但不能替代把 chr2/Mdk 从训练监督中移出的实验。"},
        {"id": "method_heading", "type": "markdown", "body": "## 数据、模型与指标定义\n\n- 数据：Saijou mouse liver fibrosis/recovery 10X scRNA-seq 四细胞类型 pseudobulk bigWig（HSC、mac、LSEC、chol）。\n- 模型：AlphaGenome 与 Borzoi，均为 LoRA + 新四轨输出 head，Poisson–multinomial loss。\n- 主指标：每个 edit 在 Mdk gene body 上的 `|log2((alt_sum+1)/(ref_sum+1))|`；先在 3 个 composition-preserving shuffles 内取中心中位数，再比较 HSC 与最佳非 HSC 头。\n- HSC margin：HSC 中位绝对相对效应减去 mac/LSEC/chol 中最大值。\n- 不确定性：以中心为 cluster 做 10,000 次 bootstrap；背景使用 34-center/66-bp 滑窗。\n- motif：80-bp context、JASPAR2024 vertebrate CORE、FIMO p<1e-3，关联检验以全部 34 个中心为分母并做模型内 BH-FDR。"},
        {"id": "next_steps", "type": "markdown", "body": "## 推荐的下一步实验\n\n1. **最高优先级：重训一个 chr2-held-out split。** 如果 Borzoi 的 HSC margin 与背景高分位仍保留，才可显著降低 memorization 解释。\n2. **对 +441..+507 做单碱基 SNV 与 motif-preserving matched edits。** 重点比较 Borzoi 的分散型正中心与 AlphaGenome 的少数极端中心，不预设二者共享同一 motif。\n3. **加入 HSC ATAC-seq、H3K27ac 或 TF occupancy。** 这能把 RNA/TES sensitivity 与真正 enhancer/chromatin-state 证据分开。\n4. **按四个时间点建立独立或条件化输出头。** 若 HSC specificity 随 fibrosis/recovery 轨迹变化，将比四时点合并 pseudobulk 更有生物学解释力。\n5. **把 Col1a1 chr11 作为 held-out motif recovery 标尺。** 它比训练染色体上的 Mdk 更适合校准‘机制恢复’的措辞。"},
        {"id": "questions", "type": "markdown", "body": "## 尚待回答的问题\n\n- Mdk HSC sensitivity 能否在不同 chromosome split、LoRA seed 与 pseudobulk replicate 中复现？\n- hub 的高背景分位（若存在）来自局部 GC grammar、剪接耦合，还是更长程 enhancer-promoter context？\n- 四个时间点的 HSC readout 是否共享同一序列 grammar，还是 fibrosis 与 recovery 具有不同调控状态？\n- Borzoi 与 AlphaGenome 的选择性中心不一致，是分辨率/head 架构差异，还是各自对训练 locus 的不同记忆方式？"},
    ]
    return cards, charts, tables, blocks


def _build_artifact(
    frames: dict[str, pd.DataFrame],
    sources: list[dict],
    cards: list[dict],
    charts: list[dict],
    tables: list[dict],
    blocks: list[dict],
    generated_at: str,
) -> dict[str, object]:
    artifact = {
        "surface": "report",
        "manifest": {
            "version": 1, "surface": "report", "title": REPORT_TITLE,
            "description": "Complete targeted ISM comparison and Mdk HSC cell-specificity robustness audit.",
            "generatedAt": generated_at, "cards": cards, "charts": charts, "tables": tables,
            "sources": [{"id": s["id"], "label": s["label"], "path": s["path"]} for s in sources],
            "blocks": blocks,
        },
        "snapshot": {
            "version": 1, "generatedAt": generated_at, "status": "ready",
            "datasets": {
                name: records(table) for name, table in frames.items()
            },
        },
        "sources": sources,
    }
    return artifact


def _write_supporting_files(report: Path, generated_at: str) -> None:
    chart_map = """# Chart map

| Section | Question | Family / type | Fields | Supported takeaway | Palette | Surface |
|---|---|---|---|---|---|---|
| Metric decomposition | Is HSC-first caused by baseline scale? | comparison / grouped bar | metric, ratio, model | Borzoi relative specificity is not baseline-driven | hard two-root | portable report |
| Center specificity | Do models agree on selective centers? | ordered-axis / line | offset, margin, model | center-level selectivity differs across architectures | hard two-root plus zero line | portable report |
| Checkpoint emergence | Was specificity learned during tuning? | progression / grouped bar | checkpoint, ratio, family | Borzoi flips from HSC-last to HSC-first | hard two-root | portable report |
| Broad background | Is the hub unusual within Mdk? | ordered-axis / line | window start, ratio, model | reports target-window context rather than a cherry-picked score | hard two-root | portable report |
| Observed signal | Is Mdk HSC-rich in the raw pseudobulk? | comparison / bar | cell, CPM sum | gene body and TES are HSC-rich | single-root | portable report |
| Reference fit | Does the model reproduce local profile shape? | comparison / grouped bar | model-cell, Pearson, model | bounds trust in Mdk-local predictions | hard two-root | portable report |
"""
    (report / "chart_map.md").write_text(chart_map)
    notes = f"""# Report source notes

- Audience: technical; primary decision is how strongly the Mdk result can be described as an HSC-specific enhancer mechanism.
- Scope: same-manifest targeted ISM, checkpoint stability, broad Mdk background, raw bigWig signal, reference-profile QA, head geometry, training-split audit, and JASPAR motif-disruption association.
- Main comparison: HSC versus the maximum of mac/LSEC/chol using absolute ratio-of-sums log2 effect.
- Main caveat: Mdk is on train chromosome chr2 and overlaps one supervised label window; both fine-tuned models share the same four Saijou tracks.
- Quantitative omission: no causal effect size or independent biological replicate is available; report uses diagnostic and robustness language only.
- Generated: {generated_at}
"""
    (report / "source_notes.md").write_text(notes)


def _utc_timestamp() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def main() -> None:
    root = parse_args().root.resolve()
    report = root / "report/mdk_cell_specificity_deep_dive"
    report.mkdir(parents=True, exist_ok=True)
    frames, ag_bg, bz_bg = _prepare_report_data(root)
    sources = _build_sources()
    cards, charts, tables, blocks = _build_report_spec(ag_bg, bz_bg)
    generated_at = _utc_timestamp()
    artifact = _build_artifact(
        frames, sources, cards, charts, tables, blocks, generated_at
    )
    artifact_path = report / "artifact.json"
    artifact_path.write_text(
        json.dumps(artifact, indent=2, ensure_ascii=False) + "\n"
    )
    _write_supporting_files(report, generated_at)
    print(artifact_path)


if __name__ == "__main__":
    main()
