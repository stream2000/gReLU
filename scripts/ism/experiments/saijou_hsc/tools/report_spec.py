"""Declarative specification for the canonical Saijou report."""

from __future__ import annotations

from collections.abc import Mapping


TITLE = "Saijou HSC 九基因 10-bp ISM：微调、原始模型与 motif 证据"


def _source(source_id: str, label: str, path: str, command: str) -> dict:
    if path.endswith(".json"):
        sql = f"SELECT * FROM read_json_auto('{path}')"
    else:
        target = path if path.endswith((".tsv", ".csv")) else f"{path}/*.tsv"
        delim = "\\t" if not target.endswith(".csv") else ","
        sql = (
            f"SELECT * FROM read_csv_auto('{target}', delim='{delim}', header=true, "
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


def build_report_spec(
    static_figures: Mapping[str, str],
) -> tuple[list[dict], list[dict], list[dict], list[dict], list[dict]]:
    """Build sources, cards, charts, tables, and ordered report blocks."""

    sources = [
        _source(
            "fine_scan",
            "Nine-gene fine-tuned 10-bp ISM center scores",
            "experiments/ism/saijou_all_genes_10bp_scan/analysis/combined_center_scores_annotated.tsv",
            "python scripts/ism/experiments/saijou_hsc/analyze_saijou_all_genes_10bp_scan.py",
        ),
        _source(
            "candidate_summary",
            "Selected candidate report tables",
            "experiments/ism/saijou_all_genes_10bp_scan/analysis/report_data",
            "python scripts/ism/experiments/saijou_hsc/build_saijou_all_genes_10bp_report.py",
        ),
        _source(
            "original_comparison",
            "Original and fine-tuned same-edit comparison",
            "experiments/ism/saijou_all_genes_10bp_scan/analysis/original_comparison",
            "python scripts/ism/experiments/saijou_hsc/compare_saijou_candidate_original_models.py",
        ),
        _source(
            "motif_scan",
            "JASPAR2024 candidate motif disruptions",
            "experiments/ism/saijou_all_genes_10bp_scan/analysis/candidate_motif_disruption_summary.tsv",
            "python scripts/ism/experiments/saijou_hsc/annotate_saijou_candidate_motifs.py",
        ),
        _source(
            "pipeline_validation",
            "Four-run profile and storage validation",
            "experiments/ism/saijou_all_genes_10bp_scan/analysis/pipeline_validation_summary.json",
            "python scripts/ism/experiments/saijou_hsc/validate_saijou_all_genes_10bp_pipeline.py --require-original",
        ),
        _source(
            "mdk_robustness",
            "Mdk cross-manifest shuffle robustness",
            "experiments/ism/saijou_all_genes_10bp_scan/analysis/mdk_manifest_robustness",
            "python scripts/ism/experiments/saijou_hsc/analyze_mdk_shuffle_manifest_robustness.py",
        ),
        _source(
            "transcripts",
            "Representative-transcript and splice annotations",
            "experiments/ism/saijou_all_genes_10bp_scan/analysis/representative_transcripts.tsv",
            "python scripts/ism/experiments/saijou_hsc/analyze_saijou_all_genes_10bp_scan.py",
        ),
        _source(
            "literature_priors",
            "Primary-source regulatory priors",
            "experiments/ism/saijou_all_genes_10bp_scan/literature_regulatory_priors.tsv",
            "curated primary literature links and scope caveats",
        ),
    ]

    cards = [
        {"id": "genes", "description": "Genes from the supervisor candidate list with complete fine-tuned scans.", "dataset": "summary", "sourceId": "candidate_summary", "metrics": [{"label": "Genes completed", "field": "genes", "format": "number"}]},
        {"id": "segments", "description": "Two non-splice candidate regions retained per gene.", "dataset": "summary", "sourceId": "candidate_summary", "metrics": [{"label": "Candidate segments", "field": "segments", "format": "number"}]},
        {"id": "hsc_segments", "description": "Selection-conditioned candidates whose HSC head ranks first in both fine-tuned models at the chosen readout.", "dataset": "summary", "sourceId": "candidate_summary", "metrics": [{"label": "HSC-top in both", "field": "both_hsc_segments", "format": "number"}]},
        {"id": "mutations", "description": "Strict composition-preserving 10-bp shuffles across the 18 original-model candidate segments.", "dataset": "summary", "sourceId": "candidate_summary", "metrics": [{"label": "Candidate edits", "field": "candidate_mutations", "format": "number"}]},
        {"id": "storage", "description": "Largest actual run directory; the user-defined limit is 3 GB per study.", "dataset": "summary", "sourceId": "pipeline_validation", "metrics": [{"label": "Largest run", "field": "largest_run_gb", "format": "number", "unit": "GB"}]},
    ]

    charts = [
        {
            "id": "hsc_ratio", "title": "HSC effect relative to the strongest non-HSC head", "subtitle": "Selected readout per segment; ratio uses median absolute log2 ratio-of-sums across the retained edits.", "type": "bar", "dataset": "hsc_ratio", "sourceId": "candidate_summary", "layout": "full",
            "encodings": {"x": {"field": "segment_label", "type": "nominal", "label": "Candidate segment"}, "y": {"field": "hsc_ratio", "type": "quantitative", "label": "HSC / max(mac, LSEC, chol)"}, "color": {"field": "model", "type": "nominal", "label": "Fine-tuned model"}},
            "settings": {"orientation": "horizontal", "groupMode": "grouped"}, "yAxisTitle": "HSC / max other", "valueFormat": "number",
        },
        {
            "id": "fine_heatmap", "title": "Four-cell fine-tuned effect matrix", "subtitle": "Median absolute edit effect at each segment's selected readout; rows separate model architecture.", "type": "heatmap", "dataset": "fine_cell_matrix", "sourceId": "candidate_summary", "layout": "full",
            "encodings": {"x": {"field": "cell_label", "type": "nominal", "label": "Cell head"}, "y": {"field": "median_abs_effect", "type": "quantitative", "label": "Median |effect|"}, "color": {"field": "row_label", "type": "nominal", "label": "Segment / model"}},
            "valueFormat": "number",
        },
        {
            "id": "mdk_robustness", "title": "Mdk +441..+507 HSC-selectivity margin across shuffle manifests", "subtitle": "34 aligned centers; positive values indicate HSC effect exceeds all three non-HSC heads.", "type": "line", "dataset": "mdk_centers", "sourceId": "mdk_robustness", "layout": "full",
            "encodings": {"x": {"field": "variant_offset_from_tss_transcription_bp", "type": "quantitative", "label": "TSS offset (bp, transcription direction)"}, "y": {"field": "hsc_margin", "type": "quantitative", "label": "HSC selectivity margin"}, "color": {"field": "series", "type": "nominal", "label": "Model / manifest"}},
            "yAxisTitle": "HSC margin", "valueFormat": "number",
        },
        {
            "id": "original_top", "title": "Strongest original-model track-group effect per candidate", "subtitle": "Maximum median absolute effect among curated RNA, CAGE, accessibility, active-chromatin and TF groups; original tracks are cell-type proxies, not matched Saijou heads.", "type": "bar", "dataset": "original_top", "sourceId": "original_comparison", "layout": "full",
            "encodings": {"x": {"field": "segment_label", "type": "nominal", "label": "Candidate segment"}, "y": {"field": "abs_effect", "type": "quantitative", "label": "Top original median |effect|"}, "color": {"field": "model", "type": "nominal", "label": "Original model"}},
            "settings": {"orientation": "horizontal", "groupMode": "grouped"}, "yAxisTitle": "Median |effect|", "valueFormat": "number",
        },
        {
            "id": "storage_runs", "title": "Actual storage per canonical run", "subtitle": "Four validated run directories; the hard study budget is 3 GB per run.", "type": "bar", "dataset": "validation", "sourceId": "pipeline_validation", "layout": "full",
            "encodings": {"x": {"field": "run", "type": "nominal", "label": "Run"}, "y": {"field": "run_megabytes", "type": "quantitative", "label": "Run size (MB)"}, "color": {"field": "run", "type": "nominal", "label": "Run"}},
            "yAxisTitle": "Run size (MB)", "valueFormat": "number",
        },
    ]

    tables = [
        {
            "id": "controls", "title": "Acta2 and Col1a1 motif positive controls", "subtitle": "Best HSC-readout percentile for the exact 10-bp control window; motif disruption is computed from JASPAR2024 hits overlapping the edit.", "dataset": "controls", "sourceId": "candidate_summary", "layout": "full", "density": "spacious", "defaultSort": {"field": "percentile", "direction": "desc"},
            "columns": [
                {"field": "gene", "label": "Gene", "type": "text"}, {"field": "control", "label": "Control", "type": "text"}, {"field": "model", "label": "Fine-tuned model", "type": "text"}, {"field": "readout_role", "label": "Best readout", "type": "text"}, {"field": "center", "label": "TSS offset", "format": "number"}, {"field": "rank", "label": "Rank", "format": "number"}, {"field": "centers_total", "label": "Centers", "format": "number"}, {"field": "percentile", "label": "Percentile", "format": "percent"}, {"field": "motif_family", "label": "Motif family", "type": "text"}, {"field": "motif_max_abs_score_change", "label": "Max motif score change", "format": "number"},
            ],
        },
        {
            "id": "candidate_table", "title": "Eighteen retained candidate segments", "subtitle": "Exact candidate location, transcript context, fine-tuned HSC ratios, top original track groups, motif family and splice distance.", "dataset": "overview", "sourceId": "candidate_summary", "layout": "full", "density": "dense", "defaultSort": {"field": "hsc_vs_max_other_ratio_bz", "direction": "desc"},
            "columns": [
                {"field": "segment_label", "label": "Segment", "type": "text"}, {"field": "peak_region_type", "label": "Region", "type": "text"}, {"field": "peak_transcript_feature", "label": "Transcript feature", "type": "text"}, {"field": "tested_centers", "label": "Centers", "format": "number"}, {"field": "minimum_splice_distance_bp", "label": "Nearest splice (bp)", "format": "number"}, {"field": "hsc_vs_max_other_ratio_ag", "label": "AG HSC ratio", "format": "number"}, {"field": "hsc_vs_max_other_ratio_bz", "label": "BZ HSC ratio", "format": "number"}, {"field": "top_cell_ag", "label": "AG top", "type": "text"}, {"field": "top_cell_bz", "label": "BZ top", "type": "text"}, {"field": "top_motif_family", "label": "Top motif family", "type": "text"}, {"field": "top_original_group_ag", "label": "AG-original top", "type": "text"}, {"field": "top_original_group_bz", "label": "BZ-original top", "type": "text"},
            ],
        },
        {
            "id": "gene_interpretation", "title": "Gene-by-gene interpretation and action", "subtitle": "Biological interpretation combines the selected edit evidence, original-model proxies, motif annotation and transcript caveats.", "dataset": "gene_interpretation", "sourceId": "candidate_summary", "layout": "full", "density": "spacious", "defaultSort": {"field": "gene", "direction": "asc"},
            "columns": [{"field": "gene", "label": "Gene", "type": "text"}, {"field": "candidate", "label": "Focus", "type": "text"}, {"field": "interpretation", "label": "Interpretation", "type": "text"}],
        },
        {
            "id": "original_evidence", "title": "Top original-model evidence per candidate", "subtitle": "Exact strongest group, signed effect and reference activity at the selected readout; values are model-specific and should not be compared as calibrated assay units.", "dataset": "original_top", "sourceId": "original_comparison", "layout": "full", "density": "dense", "defaultSort": {"field": "abs_effect", "direction": "desc"},
            "columns": [{"field": "segment_label", "label": "Segment", "type": "text"}, {"field": "model", "label": "Original model", "type": "text"}, {"field": "track_group", "label": "Top track group", "type": "text"}, {"field": "abs_effect", "label": "Median |effect|", "format": "number"}, {"field": "signed_effect", "label": "Signed effect", "format": "number", "movement": True}, {"field": "reference_sum", "label": "Reference sum", "format": "compact"}],
        },
        {
            "id": "motifs", "title": "Top three canonical motif families per candidate", "subtitle": "Family labels are sequence hypotheses; shared PWMs prevent single-factor assignment without orthogonal occupancy data.", "dataset": "motif_top", "sourceId": "motif_scan", "layout": "full", "density": "dense", "defaultSort": {"field": "max_abs_score_change", "direction": "desc"},
            "columns": [{"field": "segment_label", "label": "Segment", "type": "text"}, {"field": "family", "label": "Family", "type": "text"}, {"field": "family_rank", "label": "Rank", "format": "number"}, {"field": "affected_centers", "label": "Affected centers", "format": "number"}, {"field": "max_abs_score_change", "label": "Max score change", "format": "number"}],
        },
        {
            "id": "transcripts", "title": "Representative transcript audit", "subtitle": "Transcript used for splice exclusion and feature annotation; non-basic/retained-intron entries require coordinate re-check before wet-lab design.", "dataset": "transcripts", "sourceId": "transcripts", "layout": "full", "density": "dense", "defaultSort": {"field": "gene", "direction": "asc"},
            "columns": [{"field": "gene", "label": "Gene", "type": "text"}, {"field": "representative_transcript_id", "label": "Transcript", "type": "text"}, {"field": "transcript_name", "label": "Name", "type": "text"}, {"field": "transcript_biotype", "label": "Biotype", "type": "text"}, {"field": "transcript_support_level", "label": "TSL", "type": "text"}, {"field": "is_basic", "label": "Basic", "type": "boolean"}, {"field": "has_ccds_tag", "label": "CCDS", "type": "boolean"}, {"field": "tss_matches_analysis", "label": "TSS matched", "type": "boolean"}],
        },
        {
            "id": "validation", "title": "Run completeness, storage and profile reconstruction", "subtitle": "All runs use 128-bp float16 log2FC storage; the acceptance threshold is 0.2% relative ALT-sum reconstruction error and 3 GB per run.", "dataset": "validation", "sourceId": "pipeline_validation", "layout": "full", "density": "dense", "defaultSort": {"field": "run_megabytes", "direction": "desc"},
            "columns": [{"field": "run", "label": "Run", "type": "text"}, {"field": "genes", "label": "Genes", "format": "number"}, {"field": "tracks", "label": "Tracks", "format": "number"}, {"field": "run_megabytes", "label": "Run MB", "format": "number"}, {"field": "profile_megabytes", "label": "Profile MB", "format": "number"}, {"field": "max_reconstruction_relative_error", "label": "Max reconstruction error", "format": "percent"}, {"field": "profile_nonfinite_values", "label": "Nonfinite", "format": "number"}, {"field": "profile_index_mismatches", "label": "Index mismatches", "format": "number"}, {"field": "under_3gb", "label": "Under 3 GB", "type": "boolean"}],
        },
        {
            "id": "mdk_hub", "title": "Mdk broad-hub robustness summary", "subtitle": "+441..+507 aggregated over 34 aligned centers in the final Mdk audit and the independent nine-gene manifest.", "dataset": "mdk_hub", "sourceId": "mdk_robustness", "layout": "full", "density": "dense", "defaultSort": {"field": "hsc_vs_max_other_ratio", "direction": "desc"},
            "columns": [{"field": "manifest_label", "label": "Manifest", "type": "text"}, {"field": "model", "label": "Model", "type": "text"}, {"field": "centers", "label": "Centers", "format": "number"}, {"field": "hsc_vs_max_other_ratio", "label": "HSC ratio", "format": "number"}, {"field": "hsc_margin", "label": "HSC margin", "format": "number", "movement": True}, {"field": "hsc_top_center_fraction", "label": "HSC-top fraction", "format": "percent"}],
        },
    ]

    blocks = [
        {"id": "title", "type": "markdown", "body": f"# {TITLE}\n\n**完整技术报告 · 2026-07-13**"},
        {"id": "summary", "type": "markdown", "sourceId": "candidate_summary", "body": "## 技术结论：九基因流程可运行，但‘HSC-specific’与‘已知调控元件’不是同一件事\n\n九个基因均完成 fine-tuned Borzoi/AlphaGenome 的 TSS-centered 1,024-bp、10-bp strict-shuffle 扫描。每个基因保留两段非剪接候选，再以完全相同的 234 个 edits 运行原始 AlphaGenome 与原始 Borzoi。**18 段中 13 段在所选 readout 上由两个微调模型同时把 HSC 排第一；这是 selection-conditioned 候选比例，不是无偏的基因组发生率。** Acta2 CArG 与 Col1a1 CCAAT 被准确恢复，却通常不是 HSC-first，说明管线能区分‘经典通用 promoter grammar’与‘HSC head 相对敏感性’。原始模型在多数候选上已经显示 RNA、CAGE、可及性或活性染色质效应，因此 fine-tuning 通常是在既有 regulatory context 上重分配细胞类型读出，而不是凭空学习新的 DNA grammar。"},
        {"id": "metrics", "type": "metric-strip", "cardIds": ["genes", "segments", "hsc_segments", "mutations", "storage"]},
        {"id": "candidate_heading", "type": "markdown", "body": "## HSC 排名最高的候选集中在 promoter、近端内含子和 alternative 5′ exon\n\n横条图把强度与 specificity 分开：比值大不等于绝对效应大，反之亦然。Col1a2 的 alternative 5′ exon 候选具有最高的双模型 HSC ratio，但其转录本支持度低；Ngf -51..-37 的绝对效应很强却仅轻度 HSC enrichment；Mdk +463..+475 的 ratio 较温和，但有先前独立 Mdk 审计和明确 SP/KLF grammar 支撑。优先级应综合效应、跨模型一致、原始轨道、isoform 与 motif，而不是只按一个分数排序。"},
        {"id": "hsc_ratio_chart_1", "type": "html", "body": static_figures["hsc_ratio_1"]},
        {"id": "hsc_ratio_chart_2", "type": "html", "body": static_figures["hsc_ratio_2"]},
        {"id": "fine_matrix_heading", "type": "markdown", "body": "## 四细胞矩阵揭示五个重要反例\n\nActa2 CArG、Col1a1 CCAAT、Col1a1 +93..+99、Mdk +499..+507 和 Vegfc +447..+455 至少在一个模型中不是 HSC top。前两个是故意保留的机制正控；后三个提醒我们，‘某段总体敏感’不能自动写成‘HSC 特异 enhancer’。"},
        {"id": "fine_heatmap_ag", "type": "html", "body": static_figures["fine_heatmap_ag"]},
        {"id": "fine_heatmap_bz", "type": "html", "body": static_figures["fine_heatmap_bz"]},
        {"id": "controls_heading", "type": "markdown", "body": "## 正控恢复成功，但效应依赖模型与 readout\n\nActa2 -61 的 SRF/CArG 窗在 AlphaGenome HSC/TSS 排 7/508，在 Borzoi 最佳 readout 排 40/508；JASPAR SRF score 在 3/3 centers 被扰动。Col1a1 -71 的 NF-Y/CCAAT 窗在 AlphaGenome 与 Borzoi 分别排 8/508 和 14/508，NFYA/B/C score change 为 11.53–14.48。正控定位成立，但它们在四细胞头中并不 HSC-first，这正好校准了主问题。"},
        {"id": "controls_table", "type": "table", "tableId": "controls", "layout": "full"},
        {"id": "mdk_heading", "type": "markdown", "sourceId": "mdk_robustness", "body": "## Mdk：Borzoi 的区段级 HSC 偏好复现，精确中心仍对 shuffle 敏感\n\n在 +441..+507 的 34 个对齐中心上，最终 Mdk audit 与本轮九基因 manifest 都保持 Borzoi HSC-first（ratio 1.39 与 1.20），AlphaGenome 则均不是（0.89 与 0.77）。旧/新 manifest 的 HSC-margin Spearman 为 AG 0.62、BZ 0.31，说明区段结论比具体碱基排序稳健。预注册的 +463..+475 SP/KLF 子模块在当前所选 readout 上两模型均 HSC top，但下一步必须用 motif-preserving / motif-breaking matched edits，而不是继续依赖随机 shuffle。"},
        {"id": "mdk_chart", "type": "html", "body": static_figures["mdk_robustness"]},
        {"id": "mdk_table", "type": "table", "tableId": "mdk_hub", "layout": "full"},
        {"id": "original_heading", "type": "markdown", "body": "## 原始模型说明多数候选早已处于可响应的 RNA/染色质环境\n\nActa2 CArG、Col1a1 CCAAT、Col1a2 5′ exon、Hgf 内含子、Ngf 与 Timp1 promoter 均在原始 fibroblast/smooth-muscle/liver RNA、CAGE、accessibility 或 active-chromatin 组中产生清楚效应。Vegfc +447..+455 是最典型反例：fine-tuned 模型不 HSC-specific，但两个原始模型都显示强 liver regulatory sensitivity。原始轨道与 Saijou 四细胞并不匹配，因此这些结果只能说明 pre-existing context，不可当作同细胞类型因果复现。"},
        {"id": "original_chart_1", "type": "html", "body": static_figures["original_top_1"]},
        {"id": "original_chart_2", "type": "html", "body": static_figures["original_top_2"]},
        {"id": "original_table", "type": "table", "tableId": "original_evidence", "layout": "full"},
        {"id": "gene_heading", "type": "markdown", "body": "## 九个基因的可辩护解释与下一步\n\n下面的表把每个基因压缩为一个主候选和一个行动判断。它刻意保留 negative/ambiguous cases：Col1a1 没有双模型 HSC-first 新位点；Col1a2 与 Hgf 需要先修正 transcript/TSS 语义；Igf1 与 Vegfc 的主要已知调控元件在本扫描窗口之外。"},
        {"id": "gene_table", "type": "table", "tableId": "gene_interpretation", "layout": "full"},
        {"id": "candidate_exact", "type": "table", "tableId": "candidate_table", "layout": "full"},
        {"id": "motif_heading", "type": "markdown", "body": "## Motif 结果只支持 family-level hypotheses\n\nMdk +463..+475 由 KLF/SP GC-box family 主导；Col1a1 CCAAT 恢复 NF-Y；Acta2 +321 倾向 AP-1/ETS；Hgf +339 倾向 AP-1；Timp1 涉及 AP-1/ETS/SP；Vegfc +447 倾向 HNF/FOXA；Igf1 与 Ngf 候选出现 STAT、CTCF/KLF/NF-κB。多个 PWM 共享序列 grammar，且本轮没有 HSC occupancy 数据，所以不能从表中指定 SP1、KLF5 或某个 STAT 因子。"},
        {"id": "motif_table", "type": "table", "tableId": "motifs", "layout": "full"},
        {"id": "transcript_heading", "type": "markdown", "body": "## Transcript 语义是当前最大的生物学解释风险\n\n所有 edit 均排除了代表转录本 splice boundary ±6 bp 的扩展核心；561 个 center/readout rows 因此被标记为 splice-overlapping 并从候选中排除。但 Col1a2 使用低支持度、非-basic 的 Col1a2-205 TSS，Hgf 使用 retained-intron Hgf-202；这两者的高分不能直接投射到 canonical isoform。Ngf +55..+57 距最近边界 13 bp，虽未触及保护区，仍应做 splice-aware matched controls。"},
        {"id": "transcript_table", "type": "table", "tableId": "transcripts", "layout": "full"},
        {"id": "methods", "type": "markdown", "body": "## 数据、指标与实验设计\n\n- 数据：Saijou mouse liver fibrosis/recovery 10X 3′ scRNA-seq 四细胞类型 pseudobulk（HSC、mac、LSEC、chol）；四时间点在模型 head 中未分开。\n- 扫描：每个 TSS 周围 1,024 bp；10-bp strict mononucleotide-preserving shuffle；2-bp stride；每个可变 center 3 个 deterministic replacements。\n- 读出：TSS 1,024 bp、proximal gene 10 kb、output-clipped gene body，以及在输出范围内的 TES/HSC observed peak。\n- 主效应：`log2((ALT sum + 1)/(REF sum + 1))`；candidate discovery 使用 3 shuffles 的 median absolute effect，同时保留 signed effect。HSC ratio 为 HSC median absolute effect / 最大非-HSC head。\n- 候选：每基因两段；允许 HSC-specific 或 generic strong evidence；Acta2 CArG、Col1a1 CCAAT 与两个 final-Mdk 子模块预注册；排除代表转录本 splice boundary ±6 bp 扩展核心。\n- 原始模型：同一 234 edits；AlphaGenome 28 条 curated mouse tracks，Borzoi 65 条；包括 RNA/CAGE/accessibility/H3K27ac-H3K4me1-H3K4me3/TF proxies。\n- 存储：完整 mutation×track×output-bin log2FC 以 128-bp、float16 保存；native REF profile 单独保存，可重建 ALT。"},
        {"id": "validation_heading", "type": "markdown", "sourceId": "pipeline_validation", "body": "## 完整性、重建与存储预算全部通过\n\n两个 fine-tuned 全扫描各有 261,288 feature rows 与 13,674 edits；两个 original candidate runs 分别有 31,584（28 tracks）和 73,320（65 tracks）rows。四个运行均无 nonfinite、无 profile-index mismatch，重复 REF 差异为 0。实际目录 168.5–534.6 MB，远低于 3 GB。float16 ALT 重建最大相对误差为 0.109%（绝对误差 0.00113，原始信号 1.036），低于 0.2% 闸门。"},
        {"id": "storage_chart", "type": "chart", "chartId": "storage_runs", "layout": "full"},
        {"id": "validation_table", "type": "table", "tableId": "validation", "layout": "full"},
        {"id": "limitations", "type": "markdown", "body": "## 限制与不确定性\n\n1. **10X 3′ bias：** 长基因的监督信号主要来自 TES；TSS edits 通过模型长程映射影响 3′ coverage，可信度低于短基因或 promoter/CAGE 直接读出。\n2. **选择偏倚：** 13/18 HSC-top 是从每基因全扫描中挑出的候选，不是独立测试集命中率。\n3. **非独立模型：** AlphaGenome 与 Borzoi 架构不同，但共享 Saijou 四条 fine-tuning tracks；跨模型一致不是独立生物学 replicate。\n4. **随机 edit：** 三个 mononucleotide shuffles 不能保证每次都同等破坏 motif；Mdk 中心排序已显示 replacement sensitivity。\n5. **轨道 proxy：** 原始 liver/fibroblast/smooth-muscle tracks 与 Saijou HSC/mac/LSEC/chol 不同，且 assay 尺度不可直接校准比较。\n6. **无时间 head：** 当前结果不能解释 fibrosis 与 recovery 各时间点的差异，也不能证明恢复期 feedback loop。\n7. **因果边界：** RNA profile ISM 只说明模型预测敏感性，不证明 enhancer、剪接产物或具体 TF occupancy。"},
        {"id": "recommendations", "type": "markdown", "body": "## 推荐的下一步优先级\n\n1. **Mdk +463..+475：** 做 motif-breaking、motif-preserving、dinucleotide-matched edits，并以 HSC ATAC/H3K27ac/footprint 验证 SP/KLF family；+499..+507 作为次模块。\n2. **先修坐标再验证 Col1a2/Hgf：** 用 canonical basic Col1a2-201 与 canonical Hgf promoter 重新定义 TSS，复跑 ±0.5 kb；若信号仍在再进入 MPRA/CRISPR。\n3. **Timp1 与 Ngf promoter：** 保留 AP-1/ETS/SP/STAT paired edits；Ngf +55..+57 加入 splice-neutral control。\n4. **Acta2/Col1a1 正控继续作为每轮 calibration：** 位置应恢复，但不要要求 HSC-first。\n5. **Igf1/Vegfc 扩窗：** Igf1 加入已知 distal/intronic STAT5 sites；Vegfc 单独测试约 -152 kb SOX7 element，不能用本次近端阴性推翻远端调控。\n6. **时间维度：** 若可取得四时间点分别的 pseudobulk tracks，建立 time-conditioned heads 或逐时间点评估，才能回答 fibrosis/recovery grammar 是否变化。"},
        {"id": "questions", "type": "markdown", "body": "## 尚待回答的问题\n\n- Mdk 的 Borzoi HSC ratio 能否在新的 LoRA seed、chr2-held-out split 与真正 HSC chromatin 数据中保留？\n- Col1a2/Hgf 的信号来自 alternative promoter/isoform grammar，还是 gene-level 3′ label 对错误 TSS 的长程归因？\n- Ngf +55 与 Hgf +339 的 motif effect 在 splice-neutral edits 中是否仍存在？\n- 原始模型中强 liver/fibroblast regulatory sensitivity 与 fine-tuned HSC specificity 是加性重加权，还是 head-specific nonlinear interaction？\n- 四时间点是否共享同一候选排序？"},
    ]
    return sources, cards, charts, tables, blocks
