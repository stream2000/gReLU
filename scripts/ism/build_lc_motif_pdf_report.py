#!/usr/bin/env python
"""Build the complete LC-DIC motif-ISM HTML report and static chart assets."""

from __future__ import annotations

import argparse
import html
import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import spearmanr


REPO_ROOT = Path(__file__).resolve().parents[2]

TOKENS = {
    "surface": "#FCFCFD",
    "panel": "#FFFFFF",
    "ink": "#1F2430",
    "muted": "#6F768A",
    "grid": "#E6E8F0",
    "axis": "#D7DBE7",
}
FAMILIES = ["AP1", "Forkhead", "GATA3", "TEAD4"]
FAMILY_COLORS = {
    "AP1": "#F0986E",
    "Forkhead": "#5477C4",
    "GATA3": "#A3D576",
    "TEAD4": "#F390CA",
}
TRACK_LABELS = {
    "foxa1": "FOXA1",
    "gata3": "GATA3",
    "polr2a": "POLR2A",
    "rad21": "RAD21",
    "smc3": "SMC3",
    "ep300": "EP300",
    "h3k27ac": "H3K27ac",
    "h3k4me1": "H3K4me1",
    "h3k4me2": "H3K4me2",
    "h3k4me3": "H3K4me3",
}
KEY_TRACKS = [
    "foxa1",
    "gata3",
    "polr2a",
    "rad21",
    "smc3",
    "ep300",
    "h3k27ac",
    "h3k4me1",
]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def _theme() -> None:
    sns.set_theme(
        style="whitegrid",
        rc={
            "figure.facecolor": TOKENS["surface"],
            "savefig.facecolor": TOKENS["surface"],
            "axes.facecolor": TOKENS["panel"],
            "axes.edgecolor": TOKENS["axis"],
            "axes.labelcolor": TOKENS["ink"],
            "axes.grid": True,
            "grid.color": TOKENS["grid"],
            "grid.linewidth": 0.8,
            "font.family": "sans-serif",
            "font.sans-serif": [
                "Noto Sans CJK JP",
                "Noto Sans CJK SC",
                "DejaVu Sans",
                "sans-serif",
            ],
            "axes.unicode_minus": False,
        },
    )


def _header(
    fig: plt.Figure,
    ax: plt.Axes,
    title: str,
    subtitle: str,
) -> None:
    ax.set_title("")
    fig.subplots_adjust(top=0.78)
    left = ax.get_position().x0
    fig.text(
        left,
        0.97,
        title,
        ha="left",
        va="top",
        fontsize=14,
        fontweight="semibold",
        color=TOKENS["ink"],
    )
    fig.text(
        left,
        0.91,
        subtitle,
        ha="left",
        va="top",
        fontsize=9.5,
        color=TOKENS["muted"],
    )
    sns.despine(ax=ax)


def _save(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor=TOKENS["surface"])
    plt.close(fig)


def _cluster_bootstrap_ci(
    table: pd.DataFrame,
    value_column: str,
    *,
    iterations: int = 5000,
    seed: int = 20260613,
) -> tuple[float, float]:
    source_ids = table["source_site_id"].drop_duplicates().to_numpy()
    if len(source_ids) == 0:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    medians = np.empty(iterations, dtype=float)
    groups = {
        source_id: group[value_column].dropna().to_numpy(dtype=float)
        for source_id, group in table.groupby("source_site_id")
    }
    for index in range(iterations):
        sampled = rng.choice(source_ids, size=len(source_ids), replace=True)
        values = np.concatenate([groups[source_id] for source_id in sampled])
        medians[index] = np.median(values)
    return tuple(np.quantile(medians, [0.025, 0.975]))


def _plot_cohort_flow(output: Path) -> None:
    labels = [
        "重建后可分析 LC-DIC",
        "具有真实 Pol2 peak/summit",
        "含合格 panel motif",
        "site × motif 候选",
        "完成正式编辑与对照",
    ]
    counts = [414, 168, 108, 144, 143]
    colors = ["#E2E5EA", "#CEDFFE", "#A3BEFA", "#5477C4", "#2E4780"]
    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    bars = ax.barh(labels[::-1], counts[::-1], color=colors[::-1], edgecolor="#464C55")
    for bar, count in zip(bars, counts[::-1]):
        ax.text(
            bar.get_width() + 7,
            bar.get_y() + bar.get_height() / 2,
            f"{count}",
            va="center",
            fontsize=10,
            color=TOKENS["ink"],
        )
    ax.set_xlim(0, 455)
    ax.set_xlabel("位点或 site × motif 候选数")
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)
    _header(
        fig,
        ax,
        "正式队列通过真实 Pol2 锚定和 motif 设计逐步收敛",
        "起点为本项目 414 个计算就绪 LC-DIC；论文报告的 LC-DIC 总数为 417。",
    )
    _save(fig, output / "cohort_flow.png")


def _plot_local_heatmap(local: pd.DataFrame, output: Path) -> None:
    subset = local[
        local["window_bp"].eq(4096)
        & local["group"].isin(FAMILIES)
        & local["track"].isin(KEY_TRACKS)
    ].copy()
    matrix = subset.pivot(index="group", columns="track", values="median")
    matrix = matrix.reindex(index=FAMILIES, columns=KEY_TRACKS)
    matrix.columns = [TRACK_LABELS[column] for column in matrix.columns]
    fig, ax = plt.subplots(figsize=(10.8, 4.8))
    limit = float(np.nanmax(np.abs(matrix.to_numpy())))
    cmap = sns.diverging_palette(25, 240, s=70, l=65, as_cmap=True)
    sns.heatmap(
        matrix,
        ax=ax,
        cmap=cmap,
        center=0,
        vmin=-limit,
        vmax=limit,
        annot=True,
        fmt=".3f",
        linewidths=1,
        linecolor="#FFFFFF",
        cbar_kws={"label": "matched-control-adjusted log2FC"},
    )
    ax.set_xlabel("")
    ax.set_ylabel("motif 家族")
    _header(
        fig,
        ax,
        "Forkhead 编辑产生最一致的局部转录与活性染色质下降",
        "4 kb 窗口中位效应；负值表示 motif 编辑比同 Pol2 peak 匹配对照下降更多。",
    )
    _save(fig, output / "local_family_heatmap.png")


def _plot_response_fraction(enrichment: pd.DataFrame, output: Path) -> None:
    plot = enrichment.set_index("motif_family").loc[FAMILIES].reset_index()
    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    bars = ax.bar(
        plot["motif_family"],
        plot["family_response_fraction"],
        color=[FAMILY_COLORS[value] for value in plot["motif_family"]],
        edgecolor="#464C55",
    )
    overall = 27 / 143
    ax.axhline(overall, color=TOKENS["ink"], linestyle=":", linewidth=1.2)
    ax.text(
        3.48,
        overall + 0.012,
        f"总体 {overall:.1%}",
        ha="right",
        fontsize=9,
        color=TOKENS["ink"],
    )
    for bar, row in zip(bars, plot.itertuples(index=False)):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.012,
            f"{row.family_response_n}/{row.family_n}\nq={row.fisher_q:.3g}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    ax.set_ylim(0, 0.43)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.set_xlabel("")
    ax.set_ylabel("高耗竭响应群比例")
    ax.grid(axis="y")
    ax.grid(axis="x", visible=False)
    _header(
        fig,
        ax,
        "响应群富集支持 motif 家族特异性，而非统一 LC-DIC 敏感性",
        "最稳定 signed-depletion 表示的 k=2 聚类；Fisher 检验经四个家族 BH 校正。",
    )
    _save(fig, output / "response_fraction.png")


def _plot_tss_distance(distance: pd.DataFrame, output: Path) -> None:
    plot = distance[
        distance["track"].eq("polr2a")
        & distance["group"].isin(FAMILIES)
        & distance["minimum_distance_bp"].isin([0, 10_000, 50_000, 100_000])
    ].copy()
    x_order = [0, 10_000, 50_000, 100_000]
    labels = ["全部", ">10 kb", ">50 kb", ">100 kb"]
    fig, ax = plt.subplots(figsize=(9.4, 5.1))
    for family in FAMILIES:
        part = plot[plot["group"].eq(family)].set_index("minimum_distance_bp")
        values = part.loc[x_order, "median"].to_numpy()
        ax.plot(
            np.arange(len(x_order)),
            values,
            marker="o",
            color=FAMILY_COLORS[family],
            linewidth=1.4,
            label=family,
        )
    ax.axhline(0, color=TOKENS["ink"], linewidth=1, linestyle=":")
    ax.set_xticks(np.arange(len(x_order)), labels)
    ax.set_ylabel("TSS POLR2A matched-control-adjusted log2FC")
    ax.set_xlabel("LC-DIC motif 到代理 TSS 的最小距离")
    ax.legend(
        loc="lower left",
        bbox_to_anchor=(0, 1.02),
        ncol=4,
        frameon=False,
        borderaxespad=0,
    )
    _header(
        fig,
        ax,
        "远端 TSS 效应量远小于局部效应，Forkhead 方向最稳定",
        "每条线为各 motif 家族的 TSS 4 kb POLR2A 中位效应；TSS 为宿主基因代理映射。",
    )
    _save(fig, output / "tss_polr2a_by_distance.png")


def _plot_local_tss_scatter(site_effects: pd.DataFrame, output: Path) -> None:
    plot = site_effects[
        site_effects["distance_tss_from_edit"].abs().gt(10_000)
        & site_effects[
            "polr2a__host_tss_4kb__log2fc_signed_mean"
        ].notna()
    ].copy()
    specs = [
        ("polr2a", "POLR2A"),
        ("h3k27ac", "H3K27ac"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.9))
    for ax, (track, label) in zip(axes, specs):
        xcol = f"{track}__w4096__log2fc_signed_mean"
        ycol = f"{track}__host_tss_4kb__log2fc_signed_mean"
        sns.scatterplot(
            data=plot,
            x=xcol,
            y=ycol,
            hue="motif_family",
            hue_order=FAMILIES,
            palette=FAMILY_COLORS,
            edgecolor="#464C55",
            linewidth=0.35,
            alpha=0.72,
            s=38,
            ax=ax,
            legend=ax is axes[0],
        )
        rho, pvalue = spearmanr(plot[xcol], plot[ycol])
        ax.axhline(0, color=TOKENS["ink"], linewidth=0.8, linestyle=":")
        ax.axvline(0, color=TOKENS["ink"], linewidth=0.8, linestyle=":")
        ax.text(
            0.03,
            0.96,
            f"Spearman ρ={rho:.3f}\np={pvalue:.2g}, n={len(plot)}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9,
            bbox={
                "boxstyle": "round,pad=0.3",
                "facecolor": "#FFFFFF",
                "edgecolor": "#D7DBE7",
            },
        )
        ax.set_xlabel(f"局部 4 kb {label} 效应")
        ax.set_ylabel(f"代理 TSS 4 kb {label} 效应")
    handles, labels = axes[0].get_legend_handles_labels()
    axes[0].legend(
        handles,
        labels,
        loc="lower left",
        bbox_to_anchor=(0, 1.02),
        ncol=4,
        frameon=False,
        borderaxespad=0,
    )
    axes[1].legend([], [], frameon=False)
    _header(
        fig,
        axes[0],
        "局部下降与远端 TSS 方向相关，但远端幅度被显著压缩",
        "仅显示距离代理 TSS 超过 10 kb 的 124 个候选；两个面板独立缩放，不能按斜率比较效应大小。",
    )
    fig.subplots_adjust(wspace=0.28, top=0.78)
    _save(fig, output / "local_tss_scatter.png")


def _format_number(value: float, digits: int = 4) -> str:
    return f"{value:+.{digits}f}"


def _table_rows(frame: pd.DataFrame, columns: list[tuple[str, str]]) -> str:
    rows = []
    for row in frame.itertuples(index=False):
        cells = []
        values = row._asdict()
        for key, _ in columns:
            value = values[key]
            if isinstance(value, (float, np.floating)):
                text = "" if not np.isfinite(value) else f"{value:.4g}"
            else:
                text = str(value)
            cells.append(f"<td>{html.escape(text)}</td>")
        rows.append("<tr>" + "".join(cells) + "</tr>")
    headers = "".join(f"<th>{html.escape(label)}</th>" for _, label in columns)
    return f"<table><thead><tr>{headers}</tr></thead><tbody>{''.join(rows)}</tbody></table>"


def _build_report(
    output_dir: Path,
    sites: pd.DataFrame,
    mutations: pd.DataFrame,
    qc: pd.DataFrame,
    local: pd.DataFrame,
    tss: pd.DataFrame,
    distance: pd.DataFrame,
    enrichment: pd.DataFrame,
    site_effects: pd.DataFrame,
    representations: pd.DataFrame,
    shard: pd.DataFrame,
) -> None:
    assets = output_dir / "assets"
    local_all = local[
        local["group"].eq("all") & local["window_bp"].eq(4096)
    ].set_index("track")
    forkhead = local[
        local["group"].eq("Forkhead") & local["window_bp"].eq(4096)
    ].set_index("track")
    tss_all = tss[tss["group"].eq("all")].set_index("track")
    distant = distance[
        distance["minimum_distance_bp"].eq(50_000)
        & distance["track"].eq("polr2a")
    ].set_index("group")
    best_rep = representations.sort_values(
        ["stability_ari", "silhouette"], ascending=False
    ).iloc[0]
    best_shard = shard[
        shard["representation"].eq(best_rep["name"])
    ].iloc[0]

    site_effects_ready = site_effects[
        site_effects["polr2a__host_tss_4kb__log2fc_signed_mean"].notna()
    ].copy()
    forkhead_sites = site_effects[
        site_effects["motif_family"].eq("Forkhead")
    ]
    bootstrap_rows = []
    for group, table in (
        ("all", site_effects),
        ("Forkhead", forkhead_sites),
    ):
        for track in ("polr2a", "h3k27ac", "h3k4me1", "foxa1"):
            column = f"{track}__w4096__log2fc_signed_mean"
            low, high = _cluster_bootstrap_ci(table, column)
            bootstrap_rows.append(
                {
                    "group": group,
                    "readout": f"local_4kb_{track}",
                    "median": table[column].median(),
                    "ci95_low": low,
                    "ci95_high": high,
                    "n_candidates": len(table),
                    "n_source_sites": table["source_site_id"].nunique(),
                }
            )
    distant_forkhead = site_effects_ready[
        site_effects_ready["motif_family"].eq("Forkhead")
        & site_effects_ready["distance_tss_from_edit"].abs().gt(50_000)
    ]
    low, high = _cluster_bootstrap_ci(
        distant_forkhead,
        "polr2a__host_tss_4kb__log2fc_signed_mean",
    )
    bootstrap_rows.append(
        {
            "group": "Forkhead",
            "readout": "host_tss_4kb_polr2a_gt50kb",
            "median": distant_forkhead[
                "polr2a__host_tss_4kb__log2fc_signed_mean"
            ].median(),
            "ci95_low": low,
            "ci95_high": high,
            "n_candidates": len(distant_forkhead),
            "n_source_sites": distant_forkhead["source_site_id"].nunique(),
        }
    )
    bootstrap = pd.DataFrame.from_records(bootstrap_rows)
    bootstrap.to_csv(
        output_dir / "bootstrap_median_ci.tsv", sep="\t", index=False
    )
    fork_pol2_ci = bootstrap[
        bootstrap["readout"].eq("local_4kb_polr2a")
        & bootstrap["group"].eq("Forkhead")
    ].iloc[0]
    fork_tss_ci = bootstrap[
        bootstrap["readout"].eq("host_tss_4kb_polr2a_gt50kb")
    ].iloc[0]

    candidate = site_effects_ready[
        site_effects_ready["distance_tss_from_edit"].abs().gt(10_000)
    ].copy()
    candidate["candidate_score"] = (
        candidate["polr2a__w4096__log2fc_signed_mean"]
        + candidate["h3k27ac__w4096__log2fc_signed_mean"]
        + 10
        * (
            candidate["polr2a__host_tss_4kb__log2fc_signed_mean"]
            + candidate["h3k27ac__host_tss_4kb__log2fc_signed_mean"]
        )
    )
    candidate = candidate.sort_values("candidate_score").head(10).copy()
    candidate["distance_kb"] = candidate["distance_tss_from_edit"] / 1000
    candidate_table = _table_rows(
        candidate[
            [
                "site_id",
                "motif_family",
                "gene_name",
                "distance_kb",
                "polr2a__w4096__log2fc_signed_mean",
                "h3k27ac__w4096__log2fc_signed_mean",
                "polr2a__host_tss_4kb__log2fc_signed_mean",
                "h3k27ac__host_tss_4kb__log2fc_signed_mean",
            ]
        ],
        [
            ("site_id", "候选"),
            ("motif_family", "家族"),
            ("gene_name", "代理基因"),
            ("distance_kb", "距离 kb"),
            ("polr2a__w4096__log2fc_signed_mean", "局部 POLR2A"),
            ("h3k27ac__w4096__log2fc_signed_mean", "局部 H3K27ac"),
            (
                "polr2a__host_tss_4kb__log2fc_signed_mean",
                "TSS POLR2A",
            ),
            (
                "h3k27ac__host_tss_4kb__log2fc_signed_mean",
                "TSS H3K27ac",
            ),
        ],
    )

    representation_table = representations.copy()
    representation_table["representation"] = representation_table["name"].map(
        {
            "real_biological_all_scales": "全部尺度",
            "real_signed_depletion": "signed depletion",
            "real_local_1_4kb": "局部 1/4 kb",
            "real_strength_residual": "基线强度残差",
        }
    )
    representation_html = _table_rows(
        representation_table[
            [
                "representation",
                "selected_k",
                "silhouette",
                "stability_ari",
                "min_cluster_size",
                "n_features_after_qc",
            ]
        ],
        [
            ("representation", "表示"),
            ("selected_k", "k"),
            ("silhouette", "silhouette"),
            ("stability_ari", "稳定性 ARI"),
            ("min_cluster_size", "最小群"),
            ("n_features_after_qc", "特征数"),
        ],
    )

    qc_ready = int(qc["status"].eq("ready").sum())
    qc_skipped = int(qc["status"].ne("ready").sum())
    alt_max = mutations.loc[
        mutations["control_type"].eq("experimental"),
        "alt_pwm_relative_score",
    ].max()
    tss_ambiguous = int(
        site_effects_ready["mapping_ambiguous"].fillna(False).sum()
    )
    fork_enrichment = enrichment[
        enrichment["motif_family"].eq("Forkhead")
    ].iloc[0]
    gata_enrichment = enrichment[
        enrichment["motif_family"].eq("GATA3")
    ].iloc[0]

    title = "LC-DIC Pol2 峰内 motif 三碱基扰动：AlphaGenome 批量 ISM 实验报告"
    report = f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{title}</title>
<style>
@page {{ size: A4; margin: 16mm 15mm 17mm; }}
* {{ box-sizing: border-box; }}
body {{
  margin: 0; background: #FCFCFD; color: #1F2430;
  font-family: "Noto Sans CJK JP","Noto Sans CJK SC","Source Han Sans SC",sans-serif;
  font-size: 10.4pt;
}}
main {{ max-width: 980px; margin: auto; padding: 34px 30px 60px; }}
header {{ border-bottom: 3px solid #5477C4; padding-bottom: 20px; margin-bottom: 28px; }}
h1 {{ font-size: 27px; line-height: 1.25; margin: 0 0 10px; }}
.subtitle {{ color: #6F768A; font-size: 12px; }}
h2 {{ font-size: 19px; margin: 36px 0 12px; padding-top: 4px; }}
h3 {{ font-size: 14px; margin: 22px 0 8px; }}
p, li {{ line-height: 1.68; }}
strong {{ color: #17213A; }}
.summary {{ background: #EAF1FE; border-left: 5px solid #5477C4; padding: 16px 19px; }}
.summary li {{ margin: 7px 0; }}
.warning {{ background: #FFF4C2; border-left: 5px solid #B8A037; padding: 14px 18px; }}
.result {{ background: #F4F5F7; border-left: 5px solid #7A828F; padding: 14px 18px; }}
.cards {{ display: grid; grid-template-columns: repeat(4,1fr); gap: 10px; margin: 17px 0; }}
.card {{ background: white; border: 1px solid #E6E8F0; padding: 12px; border-radius: 8px; }}
.card .value {{ font-size: 22px; font-weight: 700; color: #2E4780; }}
.card .label {{ color: #6F768A; font-size: 9px; margin-top: 4px; }}
figure {{ margin: 20px 0 24px; page-break-inside: avoid; }}
figure img {{ width: 100%; height: auto; display: block; }}
figcaption {{ color: #6F768A; font-size: 9px; margin-top: 7px; line-height: 1.5; }}
table {{ width: 100%; border-collapse: collapse; font-size: 8.6px; margin: 12px 0 20px; page-break-inside: auto; }}
th {{ background: #F4F5F7; text-align: left; }}
th, td {{ border-bottom: 1px solid #E6E8F0; padding: 7px 6px; vertical-align: top; }}
tr {{ page-break-inside: avoid; }}
code {{ font-family: "DejaVu Sans Mono",monospace; font-size: 0.88em; }}
.formula {{ background: #F4F5F7; padding: 12px 15px; border-radius: 6px; font-family: "DejaVu Sans Mono",monospace; }}
.small {{ color: #6F768A; font-size: 9px; }}
.page-break {{ break-before: page; }}
@media print {{
  main {{ padding: 0; }}
  h2, h3 {{ break-after: avoid; }}
  a {{ color: inherit; text-decoration: none; }}
}}
</style>
</head>
<body>
<main data-report-audience="technical">
<header data-contract-section="title">
  <h1>{title}</h1>
  <div class="subtitle">正式 Pol2 锚定队列 · 128 bp ChIP-seq readouts · 2026-06-13</div>
</header>

<section data-contract-section="technical-summary">
<h2>技术摘要</h2>
<div class="summary">
<ul>
  <li><strong>严格实验队列包含 143 个 site × motif 编辑，来自 108 个 LC-DIC。</strong>
  每个实验编辑均改变 PWM 贡献最大的 3 个碱基，并配有同一 Pol2 peak 内、编辑长度与 GC 变化匹配的非 motif 对照。</li>
  <li><strong>总体局部效应较弱但显著偏负，主要由 Forkhead 亚群驱动。</strong>
  全队列 4 kb POLR2A 中位效应为 {_format_number(local_all.loc['polr2a','median'])}；
  Forkhead 为 {_format_number(forkhead.loc['polr2a','median'])}
  （source-site cluster bootstrap 95% CI
  [{_format_number(fork_pol2_ci.ci95_low)}, {_format_number(fork_pol2_ci.ci95_high)}]）。</li>
  <li><strong>响应不是普遍 LC-DIC 性质。</strong>
  高耗竭群为 27/143；Forkhead 11/32 被富集（OR={fork_enrichment.odds_ratio:.2f},
  q={fork_enrichment.fisher_q:.3g}），GATA3 仅 3/59（OR={gata_enrichment.odds_ratio:.2f},
  q={gata_enrichment.fisher_q:.3g}）。</li>
  <li><strong>远端 TSS 信号方向一致但效应量非常小。</strong>
  137 个候选有代理 TSS readout；距离 &gt;50 kb 时 Forkhead POLR2A 中位效应为
  {_format_number(distant.loc['Forkhead','median'])}
  （95% CI [{_format_number(fork_tss_ci.ci95_low)}, {_format_number(fork_tss_ci.ci95_high)}]）。
  本轮未运行 contact map，因此不能把该信号解释为 enhancer-promoter 接触变化。</li>
</ul>
</div>
</section>

<section data-contract-section="key-findings">
<h2>1. 实验队列经真实 Pol2 锚定后收敛为 143 个编辑</h2>
<p>早期 414 位点扫描沿用了一个会在 Pol2 缺失时回退到 RAD21 summit 的旧适配器。
正式分析重新读取 DIC audit，只接受 Ctrl、E2-30 或 E2-45 条件中存在真实 peak 与 summit 的 Pol2 位点，
并要求 motif 完全位于该 Pol2 peak 内且距离 summit 不超过 100 bp。</p>
<div class="cards">
  <div class="card"><div class="value">143</div><div class="label">正式 motif 编辑</div></div>
  <div class="card"><div class="value">108</div><div class="label">独立 LC-DIC</div></div>
  <div class="card"><div class="value">286</div><div class="label">target + matched control</div></div>
  <div class="card"><div class="value">15</div><div class="label">128 bp ChIP tracks</div></div>
</div>
<p>图 1 展示了从项目内 414 个计算就绪 LC-DIC 到正式实验队列的筛选过程。
这里的 414 与论文报告的 417 个 LC-DIC 不同：它是本项目完成前置坐标与模型上下文过滤后的集合。</p>
<figure>
  <img src="assets/cohort_flow.png" alt="cohort flow">
  <figcaption>图 1｜正式队列筛选。144 个可设计候选中，<code>lc_dic_108__gata3</code>
  无法通过 3 bp 规则将 motif 降至阈值以下，因此最终为 143 个。</figcaption>
</figure>
</section>

<section>
<h2>2. 三碱基 motif 破坏与同峰匹配对照通过全部核心 QC</h2>
<p>编辑策略不是随机替换。对每个 motif，算法选择参考序列中 PWM 贡献最大的三个位置，
将其改为贡献最低的替代碱基；编辑后相对 PWM 分数必须低于 0.90。
随后在同一 Pol2 peak 中寻找一个不覆盖 panel motif 的 3 bp 对照，并严格匹配 GC 变化。</p>
<div class="formula">effect = [log2(1 + ALT) - log2(1 + REF)]target
       - [log2(1 + ALT) - log2(1 + REF)]matched control</div>
<ul>
  <li>{qc_ready}/144 个候选完成准备，{qc_skipped} 个因 motif 破坏规则失败被跳过。</li>
  <li>所有 286 个 mutation manifest 条目均为 FASTA REF 匹配且 QC 状态通过。</li>
  <li>所有 target/control 均改变 3 个碱基，且每一对的 GC delta 完全一致。</li>
  <li>target 编辑后的最高相对 PWM 分数为 {alt_max:.3f}，低于 0.90 阈值。</li>
  <li>panel 包含 FOXA1、ESR1、GATA3、FOS、JUN 和 TEAD4；AP-1 将 FOS/JUN 合并为一个家族。</li>
</ul>
<div class="warning"><strong>解释边界：</strong>“PWM 降至阈值以下”只证明设计在序列评分层面成功，
不等同于真实细胞中 TF 一定失去结合。模型输出仍是计算预测。</div>
</section>

<section class="page-break">
<h2>3. Forkhead 编辑引发最强的局部 Pol2 与活性染色质下降</h2>
<p>全队列的局部效应总体较小：4 kb 中位效应为 POLR2A
{_format_number(local_all.loc['polr2a','median'])}、RAD21
{_format_number(local_all.loc['rad21','median'])}、H3K27ac
{_format_number(local_all.loc['h3k27ac','median'])}、H3K4me1
{_format_number(local_all.loc['h3k4me1','median'])}。但是按 motif 家族分层后，响应结构非常清楚。</p>
<figure>
  <img src="assets/local_family_heatmap.png" alt="local family heatmap">
  <figcaption>图 2｜各 motif 家族在 4 kb 窗口的中位 matched-control-adjusted log2FC。
  每行是独立的 site × motif 候选集合；负值代表 motif 编辑比匹配对照产生更大下降。</figcaption>
</figure>
<div class="result"><strong>核心发现：</strong>Forkhead 组的 FOXA1、POLR2A、H3K27ac 和 H3K4me1
中位效应分别为 {_format_number(forkhead.loc['foxa1','median'])}、
{_format_number(forkhead.loc['polr2a','median'])}、
{_format_number(forkhead.loc['h3k27ac','median'])} 和
{_format_number(forkhead.loc['h3k4me1','median'])}。
相反，GATA3 组 POLR2A 中位效应接近零。这与“部分 LC-DIC 依赖组织特异性 enhancer motif”
一致，但不支持所有 LC-DIC 或所有高分 motif 都有相同功能。</div>
</section>

<section>
<h2>4. 无监督分析识别出 27 个高耗竭响应候选</h2>
<p>群体分析比较了全部尺度、signed/depletion、局部 1/4 kb 和基线强度残差四种特征表示。
最稳定的是 signed-depletion 表示：k={int(best_rep.selected_k)}，
silhouette={best_rep.silhouette:.3f}，重复子采样稳定性 ARI={best_rep.stability_ari:.3f}。
该聚类与 GPU shard 的 ARI 为 {best_shard.cluster_vs_inference_shard_ari:.4f}，
说明响应群不是分片批次效应。</p>
{representation_html}
<figure>
  <img src="assets/pca_clusters.png" alt="PCA cluster views">
  <figcaption>图 3｜四种特征表示的 PCA 投影。橙色为各表示识别的高响应群；
  signed-depletion 表示包含 27 个高耗竭候选。</figcaption>
</figure>
<p>响应群对 motif 家族并非随机分布。Forkhead 进入响应群的比例最高，而 GATA3 显著偏低。</p>
<figure>
  <img src="assets/response_fraction.png" alt="response fraction by motif family">
  <figcaption>图 4｜motif 家族与高耗竭响应群的关系。q 值为四个家族 Fisher 检验的 BH 校正结果。</figcaption>
</figure>
</section>

<section class="page-break">
<h2>5. 代理 TSS 效应方向偏负，但数量级远小于局部效应</h2>
<p>143 个候选中，137 个具有模型上下文内的宿主基因代理 TSS，{tss_ambiguous} 个映射存在多基因歧义。
全体 TSS POLR2A 中位效应为 {_format_number(tss_all.loc['polr2a','median'])}，
仅约为局部 4 kb POLR2A 效应的十分之一。即使限制到距离大于 50 kb，
全体中位效应仍只有 {_format_number(distant.loc['all','median'])}。</p>
<figure>
  <img src="assets/tss_polr2a_by_distance.png" alt="TSS POLR2A by distance">
  <figcaption>图 5｜按最小距离阈值分层的代理 TSS POLR2A 中位效应。
  &gt;50 kb 时 Forkhead 为 {distant.loc['Forkhead','n']:.0f} 个候选，
  {distant.loc['Forkhead','fraction_negative']:.1%} 为负。</figcaption>
</figure>
<p>局部效应和 TSS 效应在方向上相关，即使限制到 &gt;50 kb，POLR2A Spearman ρ 仍约 0.42。
但这不能单独证明信息通过三维接触传播：模型可同时利用长程序列上下文，
而且宿主 TSS 不一定是论文 Hi-C loop 的真实 partner。</p>
<figure>
  <img src="assets/local_tss_scatter.png" alt="local versus TSS effects">
  <figcaption>图 6｜距离 &gt;10 kb 候选的局部与代理 TSS 效应。
  两个坐标轴的尺度不同，图中相关性表示方向排序一致，而不是远端效应与局部效应等幅。</figcaption>
</figure>
</section>

<section>
<h2>6. 候选优先级用于后续验证，不构成新的生物学分类</h2>
<p>下表按局部 POLR2A/H3K27ac 与代理 TSS POLR2A/H3K27ac 的联合负向得分列出前十名，
并限制 motif 到 TSS 距离超过 10 kb。它用于选择 contact-map 和湿实验候选，
不能被解释为已验证 enhancer-promoter 对。</p>
{candidate_table}
<p class="small">联合得分仅用于排序：局部两项直接相加，TSS 两项乘以 10 后相加，
以避免远端小数量级完全被局部效应淹没。该权重没有生物物理含义。</p>
</section>

<section data-contract-section="scope-data-and-metric-definitions">
<h2>7. 范围、数据与指标定义</h2>
<ul>
  <li><strong>研究对象：</strong>Wang et al. 2022 定义框架下重建的 LC-DIC；正式队列为 108 个独立 LC-DIC 上的 143 个 motif 家族候选。</li>
  <li><strong>模型：</strong>本地 AlphaGenome fold-0 checkpoint，1,048,576 bp 序列上下文。</li>
  <li><strong>readout：</strong>15 条 MCF-7 优先的 128 bp ChIP-seq tracks，包括 CTCF、RAD21、SMC3、POLR2A、FOXA1、ESR1、GATA3、EP300、BRD4 与活性组蛋白标记。</li>
  <li><strong>局部窗口：</strong>以编辑位点为中心的 1 kb 与 4 kb；本报告主要使用 4 kb。</li>
  <li><strong>TSS 窗口：</strong>代理宿主基因 TSS 周围 4 kb。映射来自本地 GTF，不是作者发布的 DIC-to-gene 表。</li>
  <li><strong>效应值：</strong>target 的 <code>log2(1+ALT)-log2(1+REF)</code> 减去匹配对照的同一指标。</li>
  <li><strong>统计：</strong>组内方向使用双侧 Wilcoxon 并在分析组内 BH 校正；motif 家族富集使用 Fisher exact 并对四家族 BH 校正。</li>
</ul>
</section>

<section data-contract-section="methodology">
<h2>8. 方法与可复现流程</h2>
<ol>
  <li>从 DIC audit 中筛选具有真实 Pol2 peak 与 summit 的 LC-DIC。</li>
  <li>在 summit ±100 bp 且 peak 内扫描 HOCOMOCO H13 的 FOXA1、ESR1、GATA3、FOS、JUN、TEAD4 PWM。</li>
  <li>对每个 site × motif 家族保留最高分 hit，设计 3 bp 最大 PWM 贡献破坏。</li>
  <li>在同一 Pol2 peak 中设计 panel-motif-free、编辑长度与 GC delta 匹配对照。</li>
  <li>使用四张 GPU 分片运行局部 1D readout，并对 137 个有效 TSS 候选运行独立 TSS pass。</li>
  <li>先计算 target 与 control 的 REF/ALT 效应，再进行 matched-control adjustment。</li>
  <li>在不使用 motif 家族标签的情况下完成群体聚类，最后才检验 motif 家族富集。</li>
</ol>
<p>本报告另用 source-site cluster bootstrap 计算中位数 95% CI，以降低同一 LC-DIC 上多个 motif 候选造成的重复计数影响。
原始群体统计的 Wilcoxon q 值仍以 site × motif 候选为分析单位，因此二者回答的问题不同。</p>
</section>

<section data-contract-section="limitations-uncertainty-and-robustness-checks">
<h2>9. 局限性、负结果与稳健性检查</h2>
<ul>
  <li><strong>尚无 contact map：</strong>本轮所有正式 run 仅包含 1D ChIP heads。任何“远端传播”措辞都应视为候选信号，而非三维接触证据。</li>
  <li><strong>TSS 是代理映射：</strong>论文指出只有约 19.2% LC-DIC loops 接触宿主基因 promoter；最近或包含基因 TSS 可能不是功能靶点。</li>
  <li><strong>效应量与显著性需分开：</strong>TSS POLR2A 虽有方向检验 q&lt;0.05，但中位效应约 -4×10<sup>-4</sup>，生物学幅度很弱。</li>
  <li><strong>GATA3 是重要负结果：</strong>高分 GATA3 motif 并不自动产生局部响应，说明 motif 存在性不足以定义功能亚型。</li>
  <li><strong>AP-1/TEAD4 缺少 cognate TF track：</strong>当前 15-track panel 无 FOS/JUN/TEAD4 readout，因此只能通过 POLR2A、cohesin 与活性染色质间接解释。</li>
  <li><strong>批次检查通过：</strong>四种表示的 cluster-vs-shard ARI 均接近 0；正式四分片使用同一 checkpoint 和相同 track selection。</li>
  <li><strong>模型预测不等同于湿实验：</strong>AlphaGenome 可用于候选优先级和机制假说生成，不能替代 TF binding、CRISPR editing 或 3C 类验证。</li>
</ul>
</section>

<section data-contract-section="recommended-next-steps">
<h2>10. 下一步：用 contact map 区分“局部 enhancer 崩解”与“远端接触改变”</h2>
<p>下一阶段应复用同一 143 个 target/control manifest，仅增加 AlphaGenome
<code>contact_maps</code> head，不重新设计突变。主指标应是 motif anchor 到代理/loop-supported promoter
的接触变化，而不是 HC-DIC 使用的跨边界绝缘指标。</p>
<ol>
  <li><strong>第一优先：</strong>对 27 个高耗竭响应候选、家族匹配非响应候选和重点远端候选运行 contact pass。</li>
  <li><strong>主比较：</strong>anchor-to-TSS contact、anchor-local contact、50–500 kb distal/local ratio，全部进行 target-minus-matched-control 校正。</li>
  <li><strong>关键判别：</strong>若 Forkhead 局部 FOXA1/POLR2A/H3K27ac 下降同时伴随 anchor-to-promoter contact 下降，才更符合远端 enhancer communication 假说。</li>
  <li><strong>负对照：</strong>GATA3 近零组应同时作为 contact 阴性背景，避免只看强响应候选。</li>
</ol>
</section>

<section data-contract-section="further-questions">
<h2>11. 尚未解决的问题</h2>
<ul>
  <li>Forkhead 响应是 FOXA1 motif 本身的因果作用，还是该 motif 标记了更强的 enhancer 上下文？</li>
  <li>AP-1 的局部活性染色质下降能否在加入 FOS/JUN track 后得到 cognate TF 支持？</li>
  <li>哪些远端候选在作者 Hi-C/ChIA-PET 或外部 promoter-capture 数据中有真实 loop partner？</li>
  <li>同一 source LC-DIC 上多个 motif 的效应是否加性、冗余或存在 motif grammar？</li>
</ul>
</section>

<section>
<h2>参考与审计材料</h2>
<p><strong>论文：</strong>Wang J, Bando M, Shirahige K, Nakato R.
“Large-scale multi-omics analysis suggests specific roles for intragenic cohesin in transcriptional regulation.”
Nature Communications 13, 3218 (2022). DOI: 10.1038/s41467-022-30792-9.</p>
<p><strong>正式数据入口：</strong>
<code>data/DICs/lc_motif_pol2_batch_prepared/</code>、
<code>outputs/lc_motif_pol2_effect_analysis/</code>、
<code>outputs/lc_motif_pol2_population_analysis/</code>。</p>
<p><strong>实现：</strong>
<code>src/grelu/interpret/ism/batch.py</code>、
<code>scripts/ism/scan_lc_dic_motifs.py</code>、
<code>scripts/ism/make_lc_motif_batch_inputs.py</code>、
<code>scripts/ism/analyze_lc_motif_effects.py</code>。</p>
</section>
</main>
</body>
</html>
"""
    (output_dir / "report.html").write_text(report, encoding="utf-8")

    source_notes = """# Report source notes

## Report contract

- Audience: technical.
- Delivery source: static HTML; PDF is printed from this HTML.
- Scope: formal Pol2-anchored LC-DIC motif experiment only.
- Contact maps: explicitly excluded from current results and reserved for next phase.

## Chart map

| Section | Question | Chart | Source |
|---|---|---|---|
| Cohort | How did the formal cohort narrow? | Horizontal stage bars | prepared manifests and audit counts |
| Local effects | Which motif family changes which local readout? | Diverging annotated heatmap | local_track_summary.tsv |
| Clustering | Which family enters the response group? | Categorical bars + overall reference | cluster_motif_enrichment.tsv |
| Distal TSS | Does POLR2A change persist with distance? | Ordered line-dot | tss_distance_summary.tsv |
| Local-to-TSS | Are local and TSS effects directionally linked? | Two-panel scatter | site_effects.tsv |
| Population structure | Is k=2 reproducible across representations? | Existing PCA panels | population analysis artifact |

## Primary source files

- data/DICs/lc_motif_pol2_batch_prepared/ready_sites.tsv
- data/DICs/lc_motif_pol2_batch_prepared/mutation_manifest.tsv
- data/DICs/lc_motif_pol2_batch_prepared/preparation_qc.tsv
- outputs/lc_motif_pol2_effect_analysis/*.tsv
- outputs/lc_motif_pol2_population_analysis/*.tsv
- outputs/lc_motif_pol2_{1d,tss}_shard*/run_metadata.json
- agent-doc/paper/nc-1.txt
"""
    (output_dir / "source_notes.md").write_text(
        source_notes, encoding="utf-8"
    )


def main() -> None:
    args = _parse_args()
    output_dir = Path(args.output_dir).resolve()
    assets = output_dir / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    _theme()

    sites = pd.read_csv(
        REPO_ROOT / "data/DICs/lc_motif_pol2_batch_prepared/ready_sites.tsv",
        sep="\t",
        low_memory=False,
    )
    mutations = pd.read_csv(
        REPO_ROOT
        / "data/DICs/lc_motif_pol2_batch_prepared/mutation_manifest.tsv",
        sep="\t",
        low_memory=False,
    )
    qc = pd.read_csv(
        REPO_ROOT
        / "data/DICs/lc_motif_pol2_batch_prepared/preparation_qc.tsv",
        sep="\t",
        low_memory=False,
    )
    analysis = REPO_ROOT / "outputs/lc_motif_pol2_effect_analysis"
    population = REPO_ROOT / "outputs/lc_motif_pol2_population_analysis"
    local = pd.read_csv(analysis / "local_track_summary.tsv", sep="\t")
    tss = pd.read_csv(analysis / "tss_track_summary.tsv", sep="\t")
    distance = pd.read_csv(analysis / "tss_distance_summary.tsv", sep="\t")
    enrichment = pd.read_csv(
        analysis / "cluster_motif_enrichment.tsv", sep="\t"
    )
    site_effects = pd.read_csv(
        analysis / "site_effects.tsv", sep="\t", low_memory=False
    )
    representations = pd.read_csv(
        population / "representation_comparison.tsv", sep="\t"
    )
    shard = pd.read_csv(
        population / "inference_shard_confound.tsv", sep="\t"
    )

    _plot_cohort_flow(assets)
    _plot_local_heatmap(local, assets)
    _plot_response_fraction(enrichment, assets)
    _plot_tss_distance(distance, assets)
    _plot_local_tss_scatter(site_effects, assets)
    shutil.copy2(
        population / "figures/pca_clusters.png",
        assets / "pca_clusters.png",
    )
    _build_report(
        output_dir,
        sites,
        mutations,
        qc,
        local,
        tss,
        distance,
        enrichment,
        site_effects,
        representations,
        shard,
    )
    metadata = {
        "report_title": (
            "LC-DIC Pol2 peak motif 3-bp perturbation AlphaGenome batch ISM"
        ),
        "generated_at": "2026-06-13T18:44:00+09:00",
        "n_ready_candidates": len(sites),
        "n_source_lc_dics": int(sites["source_site_id"].nunique()),
        "n_mutations": len(mutations),
        "contact_maps_included": False,
        "source_html": str(output_dir / "report.html"),
    }
    (output_dir / "report_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(output_dir / "report.html")


if __name__ == "__main__":
    main()
