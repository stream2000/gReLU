#!/usr/bin/env python
"""Build a distribution-first report for the MREG TSS/HC-DIC/LC-DIC ISM run."""

from __future__ import annotations

import argparse
import html
import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib import font_manager


REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_RUN = REPO_ROOT / "agent-doc/ism_context/20260611_1145_mreg_tss_hc_lc_dic_corrected"

TRACKS = ["CTCF", "RAD21", "POLR2A"]
REGION_LABELS = {
    "mreg_tss_00_ctcf_snv": "TSS CTCF edit",
    "mreg_hc_dic_00_ctcf_snv": "HC-DIC CTCF edit",
    "mreg_lc_dic_00_pol2_perturbation": "LC-DIC Pol2-center edit",
}
LOCAL_READOUT = {
    "mreg_tss_00_ctcf_snv": "mreg_tss_4kb",
    "mreg_hc_dic_00_ctcf_snv": "hc_dic_4kb",
    "mreg_lc_dic_00_pol2_perturbation": "lc_dic_4kb",
}
READOUT_LABELS = {
    "mreg_tss_4kb": "TSS +/-2 kb",
    "hc_dic_4kb": "HC-DIC +/-2 kb",
    "lc_dic_4kb": "LC-DIC +/-2 kb",
}
COLORS = {
    "REF": "#5B8DB8",
    "ALT": "#D78A59",
    "Target edit": "#C65A5A",
    "Matched controls": "#7A8794",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", default=str(DEFAULT_RUN))
    parser.add_argument(
        "--output-dir",
        default=str(REPO_ROOT / "agent-doc/ism_context/20260617_mreg_single_point_distribution_report"),
    )
    return parser.parse_args()


def _set_theme() -> None:
    for font_path in (
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/fonts-ukij-uyghur/UKIJCJK.ttf",
    ):
        if Path(font_path).exists():
            font_manager.fontManager.addfont(font_path)
            plt.rcParams["font.family"] = "Noto Sans CJK JP"
            break
    sns.set_theme(
        style="whitegrid",
        rc={
            "figure.facecolor": "#FCFCFD",
            "savefig.facecolor": "#FCFCFD",
            "axes.facecolor": "#FFFFFF",
            "axes.edgecolor": "#D8DCE7",
            "grid.color": "#E8EAF1",
            "axes.labelcolor": "#202633",
            "xtick.color": "#202633",
            "ytick.color": "#202633",
            "font.size": 9,
            "axes.titlesize": 10,
        },
    )


def _load(run_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    profiles = pd.read_parquet(run_dir / "run/mreg_chromatin_profiles.parquet")
    metrics = pd.read_csv(run_dir / "analysis/mreg_chromatin_metrics.tsv", sep="\t")
    adjusted = pd.read_csv(run_dir / "analysis/mreg_control_adjusted_metrics.tsv", sep="\t")
    mutations = pd.read_csv(run_dir / "prepared/mreg_mutation_manifest.tsv", sep="\t")
    return profiles, metrics, adjusted, mutations


def _controls_for(mutations: pd.DataFrame, mutation_id: str) -> list[str]:
    prefix = str(mutations.loc[mutations["mutation_id"].eq(mutation_id), "matched_control_id"].iloc[0])
    return sorted(mutations[mutations["mutation_id"].str.startswith(prefix)]["mutation_id"].tolist())


def _adj_value(adjusted: pd.DataFrame, mutation_id: str, readout_id: str, target: str, column: str) -> float:
    row = adjusted[
        adjusted["mutation_id"].eq(mutation_id)
        & adjusted["readout_id"].eq(readout_id)
        & adjusted["target_name"].eq(target)
    ]
    if row.empty:
        return float("nan")
    return float(row[column].iloc[0])


def _raw_mean(metrics: pd.DataFrame, mutation_id: str, readout_id: str, target: str, column: str) -> float:
    row = metrics[
        metrics["mutation_id"].eq(mutation_id)
        & metrics["readout_id"].eq(readout_id)
        & metrics["target_name"].eq(target)
    ]
    if row.empty:
        return float("nan")
    return float(row[column].iloc[0])


def _plot_ref_alt_distribution(
    profiles: pd.DataFrame,
    adjusted: pd.DataFrame,
    cases: list[tuple[str, str]],
    output_path: Path,
    title: str,
) -> None:
    records: list[dict] = []
    for mutation_id, readout_id in cases:
        subset = profiles[
            profiles["mutation_id"].eq(mutation_id)
            & profiles["readout_id"].eq(readout_id)
            & profiles["target_name"].isin(TRACKS)
        ]
        for _, row in subset.iterrows():
            label = REGION_LABELS[mutation_id]
            for state, value in (("REF", row["ref_value"]), ("ALT", row["alt_value"])):
                records.append(
                    {
                        "case": label,
                        "track": row["target_name"],
                        "state": state,
                        "value": float(value),
                    }
                )
    df = pd.DataFrame(records)
    fig, axes = plt.subplots(len(TRACKS), len(cases), figsize=(11.2, 7.0), sharey=False)
    for i, track in enumerate(TRACKS):
        for j, (mutation_id, readout_id) in enumerate(cases):
            ax = axes[i, j]
            label = REGION_LABELS[mutation_id]
            panel = df[df["case"].eq(label) & df["track"].eq(track)]
            sns.boxplot(
                data=panel,
                x="state",
                y="value",
                order=["REF", "ALT"],
                palette=COLORS,
                width=0.5,
                fliersize=2,
                ax=ax,
            )
            sns.stripplot(
                data=panel,
                x="state",
                y="value",
                order=["REF", "ALT"],
                color="#202633",
                alpha=0.35,
                size=2.4,
                jitter=0.16,
                ax=ax,
            )
            adj = _adj_value(adjusted, mutation_id, readout_id, track, "adjusted_signed_delta_mean")
            ax.set_title(f"{label}\n{READOUT_LABELS[readout_id]}\nadj mean={adj:.3g}")
            ax.set_xlabel("")
            ax.set_ylabel(track if j == 0 else "")
    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _plot_delta_distribution(
    profiles: pd.DataFrame,
    adjusted: pd.DataFrame,
    mutations: pd.DataFrame,
    cases: list[tuple[str, str]],
    output_path: Path,
    title: str,
) -> None:
    records: list[dict] = []
    for mutation_id, readout_id in cases:
        control_ids = set(_controls_for(mutations, mutation_id))
        relevant = [mutation_id, *sorted(control_ids)]
        subset = profiles[
            profiles["mutation_id"].isin(relevant)
            & profiles["readout_id"].eq(readout_id)
            & profiles["target_name"].isin(TRACKS)
        ]
        for _, row in subset.iterrows():
            records.append(
                {
                    "case": REGION_LABELS[mutation_id],
                    "track": row["target_name"],
                    "type": "Target edit" if row["mutation_id"] == mutation_id else "Matched controls",
                    "delta": float(row["delta"]),
                }
            )
    df = pd.DataFrame(records)
    fig, axes = plt.subplots(len(TRACKS), len(cases), figsize=(11.2, 7.0), sharey=False)
    for i, track in enumerate(TRACKS):
        for j, (mutation_id, readout_id) in enumerate(cases):
            ax = axes[i, j]
            label = REGION_LABELS[mutation_id]
            panel = df[df["case"].eq(label) & df["track"].eq(track)]
            sns.violinplot(
                data=panel,
                x="type",
                y="delta",
                order=["Target edit", "Matched controls"],
                palette=COLORS,
                cut=0,
                inner="quartile",
                linewidth=0.8,
                ax=ax,
            )
            sns.stripplot(
                data=panel,
                x="type",
                y="delta",
                order=["Target edit", "Matched controls"],
                color="#202633",
                alpha=0.35,
                size=2.0,
                jitter=0.18,
                ax=ax,
            )
            ax.axhline(0, color="#202633", linewidth=0.8, alpha=0.6)
            adj = _adj_value(adjusted, mutation_id, readout_id, track, "adjusted_signed_delta_mean")
            ax.set_title(f"{label}\n{READOUT_LABELS[readout_id]}\ntarget-control={adj:.3g}")
            ax.set_xlabel("")
            ax.tick_params(axis="x", rotation=18)
            ax.set_ylabel(f"{track} ALT-REF" if j == 0 else "")
    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _plot_adjusted_heatmap(adjusted: pd.DataFrame, output_path: Path) -> None:
    cases = [
        ("mreg_tss_00_ctcf_snv", "mreg_tss_4kb", "TSS edit -> TSS"),
        ("mreg_hc_dic_00_ctcf_snv", "hc_dic_4kb", "HC edit -> HC"),
        ("mreg_lc_dic_00_pol2_perturbation", "lc_dic_4kb", "LC edit -> LC"),
        ("mreg_hc_dic_00_ctcf_snv", "mreg_tss_4kb", "HC edit -> TSS"),
        ("mreg_lc_dic_00_pol2_perturbation", "mreg_tss_4kb", "LC edit -> TSS"),
    ]
    rows = []
    for mutation_id, readout_id, label in cases:
        for track in TRACKS:
            rows.append(
                {
                    "case": label,
                    "track": track,
                    "adjusted_log2fc": _adj_value(adjusted, mutation_id, readout_id, track, "adjusted_log2fc_mean"),
                }
            )
    mat = pd.DataFrame(rows).pivot(index="case", columns="track", values="adjusted_log2fc")
    fig, ax = plt.subplots(figsize=(8.3, 3.8))
    sns.heatmap(
        mat[TRACKS],
        cmap="vlag",
        center=0,
        annot=True,
        fmt=".3f",
        linewidths=0.5,
        linecolor="#FFFFFF",
        cbar_kws={"label": "matched-control adjusted mean log2FC"},
        ax=ax,
    )
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title("Control-adjusted ISM effects across local and distal readouts", fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _format_num(value: float) -> str:
    if pd.isna(value):
        return "NA"
    if abs(value) >= 10:
        return f"{value:.2f}"
    return f"{value:.4f}"


def _build_tables(metrics: pd.DataFrame, adjusted: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    local_cases = [
        ("TSS edit", "mreg_tss_00_ctcf_snv", "mreg_tss_4kb"),
        ("HC-DIC edit", "mreg_hc_dic_00_ctcf_snv", "hc_dic_4kb"),
        ("LC-DIC edit", "mreg_lc_dic_00_pol2_perturbation", "lc_dic_4kb"),
    ]
    distal_cases = [
        ("HC-DIC edit -> TSS", "mreg_hc_dic_00_ctcf_snv", "mreg_tss_4kb"),
        ("LC-DIC edit -> TSS", "mreg_lc_dic_00_pol2_perturbation", "mreg_tss_4kb"),
    ]

    def build(cases: list[tuple[str, str, str]]) -> pd.DataFrame:
        rows = []
        for label, mutation_id, readout_id in cases:
            for track in TRACKS:
                rows.append(
                    {
                        "comparison": label,
                        "readout": READOUT_LABELS[readout_id],
                        "track": track,
                        "REF mean": _raw_mean(metrics, mutation_id, readout_id, track, "ref_mean"),
                        "ALT mean": _raw_mean(metrics, mutation_id, readout_id, track, "alt_mean"),
                        "target ALT-REF": _raw_mean(metrics, mutation_id, readout_id, track, "signed_delta_mean"),
                        "control ALT-REF mean": _adj_value(adjusted, mutation_id, readout_id, track, "control_signed_delta_mean"),
                        "adjusted ALT-REF": _adj_value(adjusted, mutation_id, readout_id, track, "adjusted_signed_delta_mean"),
                        "adjusted mean log2FC": _adj_value(adjusted, mutation_id, readout_id, track, "adjusted_log2fc_mean"),
                    }
                )
        df = pd.DataFrame(rows)
        for col in df.columns:
            if col not in {"comparison", "readout", "track"}:
                df[col] = df[col].map(_format_num)
        return df

    return build(local_cases), build(distal_cases)


def _table_html(df: pd.DataFrame) -> str:
    return df.to_html(index=False, escape=False, border=0, classes="data-table")


def _write_html(
    run_dir: Path,
    output_dir: Path,
    figures: dict[str, Path],
    local_table: pd.DataFrame,
    distal_table: pd.DataFrame,
) -> None:
    title = "MREG 单点 TSS / HC-DIC / LC-DIC ISM distribution 报告"
    fig = {key: html.escape(str(path.relative_to(output_dir))) for key, path in figures.items()}
    source = html.escape(str(run_dir.relative_to(REPO_ROOT)))
    report = f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>{title}</title>
<style>
@page {{ size: A4; margin: 15mm 14mm 16mm; }}
* {{ box-sizing: border-box; }}
body {{
  margin: 0;
  background: #FCFCFD;
  color: #202633;
  font-family: "Noto Sans CJK JP","Noto Sans CJK SC","Source Han Sans SC",Arial,sans-serif;
  font-size: 10.3pt;
}}
main {{ max-width: 980px; margin: auto; padding: 30px 28px 58px; }}
header {{ border-bottom: 3px solid #3E6F9E; margin-bottom: 24px; padding-bottom: 16px; }}
h1 {{ margin: 0 0 8px; font-size: 25px; line-height: 1.25; }}
h2 {{ margin: 30px 0 10px; font-size: 18px; }}
h3 {{ margin: 18px 0 8px; font-size: 13px; }}
p, li {{ line-height: 1.68; }}
.subtitle {{ color: #687286; font-size: 10px; }}
.summary {{ background: #EEF5FB; border-left: 5px solid #3E6F9E; padding: 14px 17px; }}
.note {{ background: #FFF6D6; border-left: 5px solid #B99B2E; padding: 13px 16px; }}
figure {{ margin: 18px 0 24px; break-inside: avoid; }}
figure img {{ width: 100%; display: block; }}
figcaption {{ color: #687286; font-size: 8.8px; line-height: 1.5; margin-top: 6px; }}
table {{ width: 100%; border-collapse: collapse; font-size: 8.3px; margin: 10px 0 18px; }}
th {{ background: #F0F2F6; text-align: left; }}
th, td {{ border-bottom: 1px solid #E0E4EC; padding: 6px 5px; vertical-align: top; }}
code {{ font-family: "DejaVu Sans Mono",monospace; font-size: 0.88em; }}
.small {{ color: #687286; font-size: 8.8px; }}
.page-break {{ break-before: page; }}
@media print {{ main {{ padding: 0; }} h2, h3 {{ break-after: avoid; }} }}
</style>
</head>
<body>
<main>
<header>
<h1>{title}</h1>
<div class="subtitle">来源：<code>{source}</code>；128 bp ChIP tracks；展示每个 128 bp bin 的 REF/ALT/ALT-REF 分布。</div>
</header>

<section>
<h2>结论先行</h2>
<div class="summary">
<ul>
<li><strong>TSS 正对照成立。</strong>在 TSS 破坏 CTCF motif 后，TSS readout 内 CTCF、RAD21 和 POLR2A 都出现负向变化，其中 CTCF 下降最明显。</li>
<li><strong>HC-DIC 是局部 CTCF/cohesin 响应。</strong>HC-DIC CTCF motif 编辑在本地 readout 中强烈降低 CTCF 和 RAD21，但到 MREG TSS 的 adjusted effect 接近零。</li>
<li><strong>LC-DIC 是不同的局部响应。</strong>LC-DIC 没有采用 CTCF motif knockout，而是 Pol2 peak-center 3 bp 替换；本地 POLR2A/RAD21/CTCF 方向为负，但构成和 HC-DIC 不同。</li>
<li><strong>远端 TSS 只看到弱预测信号。</strong>LC-DIC 到 TSS 的 POLR2A/RAD21/CTCF adjusted effect 方向一致偏负，但效应远小于局部响应；不能单独解释为 enhancer-promoter contact 变化。</li>
</ul>
</div>
</section>

<section>
<h2>1. 对应老师问题的实验设计</h2>
<p>这版报告按三个问题组织：先比较 TSS、HC-DIC、LC-DIC 三个区域各自被编辑后的本地 ISM distribution；
再看 DIC 编辑是否传播到 MREG TSS；最后用 matched-control-adjusted 值比较三个区域的响应模式。</p>
<ul>
<li><strong>TSS：</strong>strand-aware CTCF PWM disruption，作为正对照。</li>
<li><strong>HC-DIC：</strong>strand-aware CTCF PWM disruption，测试高 CTCF/cohesin DIC 的局部和远端影响。</li>
<li><strong>LC-DIC：</strong>因为该位点没有选中的 CTCF motif，使用 Pol2 peak-center 3 bp substitution，而不是 Pol2 motif knockout。</li>
</ul>
<p class="small">分布图每个点是一个 128 bp bin；box/violin 显示 readout window 内的 bin-level read distribution。</p>
</section>

<section>
<h2>2. 本地 read distribution：REF/ALT 原始分布</h2>
<figure>
<img src="{fig['local_ref_alt']}" alt="local REF ALT distribution">
<figcaption>每列是一个编辑区域在自己的本地 4 kb readout 中的 REF/ALT 分布。标题中的 adj mean 是 target ALT-REF 扣除 matched-control ALT-REF 均值后的值。</figcaption>
</figure>
</section>

<section>
<h2>3. 本地 subtracted value：target delta 与 matched controls</h2>
<figure>
<img src="{fig['local_delta']}" alt="local delta distribution">
<figcaption>Target edit 的 ALT-REF distribution 与 matched controls 的 ALT-REF distribution 并列。HC-DIC 的 CTCF/RAD21 分布整体明显负移；LC-DIC 的分布更弱且不是 CTCF/cohesin 单轴。</figcaption>
</figure>
<h3>本地 adjusted 数值表</h3>
{_table_html(local_table)}
</section>

<section class="page-break">
<h2>4. DIC 编辑是否影响 MREG TSS</h2>
<figure>
<img src="{fig['tss_ref_alt']}" alt="TSS REF ALT distribution after DIC edits">
<figcaption>这里 readout 固定为 MREG TSS +/-2 kb，只比较 HC-DIC 和 LC-DIC 编辑后的 TSS 端 REF/ALT distribution。</figcaption>
</figure>
<figure>
<img src="{fig['tss_delta']}" alt="TSS delta distribution after DIC edits">
<figcaption>HC-DIC 到 TSS 的 target-control adjusted value 接近零；LC-DIC 到 TSS 有一致负向但弱的 POLR2A/RAD21/CTCF distribution shift。</figcaption>
</figure>
<h3>DIC 到 TSS adjusted 数值表</h3>
{_table_html(distal_table)}
</section>

<section>
<h2>5. 三个区域的 ISM 模式差异</h2>
<figure>
<img src="{fig['heatmap']}" alt="adjusted effect heatmap">
<figcaption>matched-control-adjusted mean log2FC。local rows 展示各区域本地响应；distal rows 展示 DIC 编辑到 TSS 的传播信号。</figcaption>
</figure>
<div class="note">
<p><strong>解释边界：</strong>这些是 AlphaGenome model prediction，不是湿实验验证。LC-DIC 编辑是 peak-center sequence perturbation，不应描述为 Pol2 motif knockout。
远端 TSS readout 是一维 ChIP prediction；若要解释为三维接触变化，需要另跑 contact map。</p>
</div>
</section>

<section>
<h2>6. 输出和复现</h2>
<ul>
<li>原始 profile：<code>{source}/run/mreg_chromatin_profiles.parquet</code></li>
<li>汇总表：<code>{source}/analysis/mreg_chromatin_metrics.tsv</code></li>
<li>control-adjusted 表：<code>{source}/analysis/mreg_control_adjusted_metrics.tsv</code></li>
<li>本报告目录：<code>{html.escape(str(output_dir.relative_to(REPO_ROOT)))}</code></li>
</ul>
</section>
</main>
</body>
</html>
"""
    (output_dir / "report.html").write_text(report, encoding="utf-8")


def main() -> None:
    args = _parse_args()
    _set_theme()
    run_dir = Path(args.run_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = output_dir / "figures"
    if fig_dir.exists():
        shutil.rmtree(fig_dir)
    fig_dir.mkdir(parents=True)

    profiles, metrics, adjusted, mutations = _load(run_dir)
    local_cases = [(mutation_id, readout_id) for mutation_id, readout_id in LOCAL_READOUT.items()]
    tss_cases = [
        ("mreg_hc_dic_00_ctcf_snv", "mreg_tss_4kb"),
        ("mreg_lc_dic_00_pol2_perturbation", "mreg_tss_4kb"),
    ]
    figures = {
        "local_ref_alt": fig_dir / "local_ref_alt_distribution.png",
        "local_delta": fig_dir / "local_delta_distribution.png",
        "tss_ref_alt": fig_dir / "dic_to_tss_ref_alt_distribution.png",
        "tss_delta": fig_dir / "dic_to_tss_delta_distribution.png",
        "heatmap": fig_dir / "adjusted_effect_heatmap.png",
    }
    _plot_ref_alt_distribution(
        profiles,
        adjusted,
        local_cases,
        figures["local_ref_alt"],
        "Local read distributions for the three MREG ISM edits",
    )
    _plot_delta_distribution(
        profiles,
        adjusted,
        mutations,
        local_cases,
        figures["local_delta"],
        "Local ALT-REF distributions before and after matched-control comparison",
    )
    _plot_ref_alt_distribution(
        profiles,
        adjusted,
        tss_cases,
        figures["tss_ref_alt"],
        "MREG TSS read distributions after DIC edits",
    )
    _plot_delta_distribution(
        profiles,
        adjusted,
        mutations,
        tss_cases,
        figures["tss_delta"],
        "MREG TSS ALT-REF distributions after DIC edits",
    )
    _plot_adjusted_heatmap(adjusted, figures["heatmap"])
    local_table, distal_table = _build_tables(metrics, adjusted)
    local_table.to_csv(output_dir / "local_distribution_summary.tsv", sep="\t", index=False)
    distal_table.to_csv(output_dir / "dic_to_tss_distribution_summary.tsv", sep="\t", index=False)
    _write_html(run_dir, output_dir, figures, local_table, distal_table)
    (output_dir / "report_metadata.json").write_text(
        json.dumps(
            {
                "source_run": str(run_dir),
                "tracks": TRACKS,
                "figures": {key: str(path) for key, path in figures.items()},
                "html": str(output_dir / "report.html"),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(output_dir / "report.html")


if __name__ == "__main__":
    main()
