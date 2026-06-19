#!/usr/bin/env python
"""Build a concise English report for MREG, LC-DIC, and HC-DIC ISM results."""

from __future__ import annotations

import html
import json
import shutil
from datetime import datetime
from io import StringIO
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "agent-doc/ism_context/20260617_dic_ism_english_report"
ASSET_DIR = OUT_DIR / "assets"

MREG_DIST = ROOT / "agent-doc/ism_context/20260617_mreg_single_point_distribution_report"
MREG_CONTACT = ROOT / "agent-doc/ism_context/20260617_mreg_three_region_contact"
LC_REPORT = ROOT / "agent-doc/ism_context/20260613_1844_lc_dic_motif_batch_report"
LC_EFFECT = ROOT / "outputs/lc_motif_pol2_effect_analysis"
HC_SUMMARY = ROOT / "experiment/hc_dic_population_result_summary.md"

TOKENS = {
    "surface": "#FCFCFD",
    "panel": "#FFFFFF",
    "ink": "#1F2430",
    "muted": "#6F768A",
    "grid": "#E6E8F0",
    "axis": "#D7DBE7",
}
COLORS = {
    "blue": "#A3BEFA",
    "blue_dark": "#2E4780",
    "orange": "#F0986E",
    "orange_dark": "#804126",
    "olive": "#A3D576",
    "pink": "#F390CA",
    "neutral": "#C5CAD3",
    "neutral_dark": "#464C55",
}


def apply_style() -> None:
    sns.set_theme(style="whitegrid")
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Aptos", "Inter", "Segoe UI", "DejaVu Sans", "Arial"],
            "figure.facecolor": TOKENS["surface"],
            "axes.facecolor": TOKENS["panel"],
            "axes.edgecolor": TOKENS["axis"],
            "axes.labelcolor": TOKENS["ink"],
            "xtick.color": TOKENS["muted"],
            "ytick.color": TOKENS["muted"],
            "text.color": TOKENS["ink"],
            "grid.color": TOKENS["grid"],
            "grid.linestyle": "-",
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
        }
    )


def add_header(fig, title: str, subtitle: str) -> None:
    fig.text(0.08, 0.955, title, ha="left", va="top", fontsize=15, fontweight="bold")
    fig.text(0.08, 0.915, subtitle, ha="left", va="top", fontsize=10.5, color=TOKENS["muted"])


def copy_asset(src: Path, dest_name: str) -> str:
    dest = ASSET_DIR / dest_name
    shutil.copy2(src, dest)
    return f"assets/{dest.name}"


def read_markdown_table(path: Path, section_marker: str) -> pd.DataFrame:
    lines = path.read_text().splitlines()
    start = next(i for i, line in enumerate(lines) if section_marker in line)
    table_start = next(i for i in range(start, len(lines)) if lines[i].startswith("|"))
    table_lines = []
    for line in lines[table_start:]:
        if not line.startswith("|"):
            break
        table_lines.append(line)
    df = pd.read_csv(StringIO("\n".join(table_lines)), sep="|", engine="python")
    df = df.drop(columns=[df.columns[0], df.columns[-1]])
    df.columns = [c.strip() for c in df.columns]
    df = df.iloc[1:].copy()
    for col in df.columns:
        df[col] = df[col].astype(str).str.strip()
    return df


def make_contact_schematic(contact_path: Path) -> str:
    contact = pd.read_csv(contact_path, sep="\t")
    anchor = contact[contact["metric"] == "edit_anchor_to_readout"].copy()
    tss4 = anchor[anchor["target_region"] == "mreg_tss_4kb"].set_index("mutation_id")
    local4 = anchor[anchor["target_region"].str.endswith("_4kb")].copy()

    coords = {
        "HC-DIC": 215950160,
        "LC-DIC": 215979780,
        "MREG TSS": 216013551,
    }
    mutation_ids = {
        "TSS edit": "mreg_tss_00_ctcf_snv",
        "HC-DIC edit": "mreg_hc_dic_00_ctcf_snv",
        "LC-DIC edit": "mreg_lc_dic_00_pol2_perturbation",
    }

    fig, ax = plt.subplots(figsize=(10.5, 4.0))
    add_header(
        fig,
        "Contact-map summary at the MREG locus",
        "Adjusted contact-map change, target edit minus matched controls; schematic uses bounded contact metrics, not raw heatmap tensors.",
    )
    ax.set_xlim(215940000, 216020000)
    ax.set_ylim(-0.8, 1.55)
    ax.axhline(0, color=TOKENS["axis"], lw=2)

    intervals = {
        "HC-DIC": (215949640, 215951021, COLORS["blue"]),
        "LC-DIC": (215979550, 215980412, COLORS["olive"]),
    }
    for label, (start, end, color) in intervals.items():
        ax.add_patch(plt.Rectangle((start, -0.12), end - start, 0.24, color=color, alpha=0.9))
        ax.text((start + end) / 2, -0.34, label, ha="center", va="top", fontsize=10)
    ax.scatter([coords["MREG TSS"]], [0], s=130, marker="^", color=COLORS["orange"], zorder=3)
    ax.text(coords["MREG TSS"], -0.34, "MREG TSS", ha="center", va="top", fontsize=10)

    arrow_specs = [
        ("HC-DIC edit", coords["HC-DIC"], coords["MREG TSS"], 1.05),
        ("LC-DIC edit", coords["LC-DIC"], coords["MREG TSS"], 0.58),
    ]
    for label, x0, x1, y in arrow_specs:
        mid = (x0 + x1) / 2
        val = float(tss4.loc[mutation_ids[label], "adjusted_delta_contact_mean"])
        color = COLORS["orange_dark"] if val < -0.02 else COLORS["neutral_dark"]
        ax.annotate(
            "",
            xy=(x1, y),
            xytext=(x0, y),
            arrowprops=dict(arrowstyle="-|>", lw=2.0, color=color, shrinkA=0, shrinkB=6),
        )
        ax.text(mid, y + 0.08, f"{label} -> TSS: {val:+.3f}", ha="center", va="bottom", fontsize=10, color=color)

    local_labels = [
        ("TSS edit", coords["MREG TSS"], "mreg_tss_4kb", 1.34),
        ("HC-DIC edit", coords["HC-DIC"], "hc_dic_4kb", 0.32),
        ("LC-DIC edit", coords["LC-DIC"], "lc_dic_4kb", 0.32),
    ]
    for label, x, target, y in local_labels:
        row = local4[(local4["mutation_id"] == mutation_ids[label]) & (local4["target_region"] == target)]
        val = float(row["adjusted_delta_contact_mean"].iloc[0])
        ax.text(x, y, f"local {val:+.3f}", ha="center", va="center", fontsize=9, color=COLORS["blue_dark"])

    ax.set_xlabel("chr2 coordinate")
    ax.set_yticks([])
    ax.set_xticks([215950000, 215980000, 216010000])
    ax.set_xticklabels(["215.950 Mb", "215.980 Mb", "216.010 Mb"])
    for spine in ["left", "right", "top"]:
        ax.spines[spine].set_visible(False)
    ax.grid(False)
    fig.subplots_adjust(left=0.08, right=0.98, top=0.82, bottom=0.22)
    out = ASSET_DIR / "mreg_contact_schematic.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return f"assets/{out.name}"


def make_hc_bar(summary_path: Path) -> tuple[str, pd.DataFrame]:
    effects = read_markdown_table(summary_path, "Overall 4 kb local peak effects")
    selected = ["CTCF", "RAD21", "SMC3", "POLR2A", "H3K27ac", "H3K4me1", "H3K4me2", "H3K4me3"]
    effects = effects[effects["Track"].isin(selected)].copy()
    effects["Median"] = effects["Median"].astype(float)
    effects["Sites < -0.1"] = effects["Sites < -0.1"].astype(int)
    effects["Track"] = pd.Categorical(effects["Track"], categories=selected, ordered=True)
    effects = effects.sort_values("Track")

    fig, ax = plt.subplots(figsize=(9.5, 4.4))
    add_header(
        fig,
        "HC-DIC CTCF disruption primarily removes local CTCF and cohesin",
        "Median 4 kb peak log2FC after matched-control adjustment, n=134 HC-DIC sites.",
    )
    palette = [
        COLORS["blue_dark"],
        COLORS["blue_dark"],
        COLORS["blue_dark"],
        COLORS["neutral"],
        COLORS["orange"],
        COLORS["orange"],
        COLORS["orange"],
        COLORS["orange"],
    ]
    sns.barplot(
        data=effects,
        x="Median",
        y="Track",
        hue="Track",
        ax=ax,
        palette=dict(zip(selected, palette)),
        edgecolor=COLORS["neutral_dark"],
        legend=False,
    )
    ax.axvline(0, color=TOKENS["ink"], lw=1)
    ax.set_xlabel("Median adjusted peak log2FC")
    ax.set_ylabel("")
    for i, row in enumerate(effects.itertuples(index=False)):
        val = row.Median
        if val < -0.7:
            ax.text(val + 0.08, i, f"{val:.3g}", va="center", ha="left", fontsize=9, color=TOKENS["panel"])
        elif val < 0:
            ax.text(val - 0.07, i, f"{val:.3g}", va="center", ha="right", fontsize=9, color=TOKENS["ink"])
        else:
            ax.text(val + 0.04, i, f"{val:.3g}", va="center", ha="left", fontsize=9, color=TOKENS["ink"])
    sns.despine(ax=ax, left=True)
    fig.subplots_adjust(left=0.16, right=0.98, top=0.80, bottom=0.18)
    out = ASSET_DIR / "hc_local_effects_bar.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return f"assets/{out.name}", effects


def make_lc_family_heatmap(local_summary: pd.DataFrame) -> str:
    families = ["AP1", "Forkhead", "GATA3", "TEAD4"]
    tracks = ["foxa1", "polr2a", "h3k27ac", "h3k4me1", "rad21", "smc3", "gata3"]
    labels = {
        "foxa1": "FOXA1",
        "polr2a": "POLR2A",
        "h3k27ac": "H3K27ac",
        "h3k4me1": "H3K4me1",
        "rad21": "RAD21",
        "smc3": "SMC3",
        "gata3": "GATA3",
    }
    data = (
        local_summary[
            (local_summary["window_bp"] == 4096)
            & (local_summary["group"].isin(families))
            & (local_summary["track"].isin(tracks))
        ]
        .pivot(index="group", columns="track", values="median")
        .reindex(index=families, columns=tracks)
    )
    data.columns = [labels[c] for c in data.columns]

    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    add_header(
        fig,
        "LC-DIC motif effects are family-specific",
        "Median local 4 kb effect, target motif edit minus same-peak matched control.",
    )
    cmap = sns.diverging_palette(24, 220, s=80, l=55, center="light", as_cmap=True)
    sns.heatmap(
        data,
        ax=ax,
        cmap=cmap,
        center=0,
        vmin=-0.09,
        vmax=0.04,
        annot=True,
        fmt=".3f",
        linewidths=0.7,
        linecolor=TOKENS["grid"],
        cbar_kws={"label": "Median adjusted log2FC"},
    )
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(axis="x", rotation=0)
    ax.tick_params(axis="y", rotation=0)
    fig.subplots_adjust(left=0.13, right=0.98, top=0.78, bottom=0.16)
    out = ASSET_DIR / "lc_family_effect_heatmap.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return f"assets/{out.name}"


def make_lc_response_fraction(enrichment: pd.DataFrame) -> str:
    df = enrichment.copy()
    order = ["Forkhead", "AP1", "TEAD4", "GATA3"]
    df["motif_family"] = pd.Categorical(df["motif_family"], categories=order, ordered=True)
    df = df.sort_values("motif_family")
    df["response_pct"] = 100 * df["family_response_fraction"]

    fig, ax = plt.subplots(figsize=(8.6, 4.2))
    add_header(
        fig,
        "Forkhead is enriched in the LC-DIC high-depletion cluster",
        "Response fraction by motif family; cluster is exploratory and model-derived.",
    )
    palette = {
        "Forkhead": COLORS["blue_dark"],
        "AP1": COLORS["blue"],
        "TEAD4": COLORS["neutral"],
        "GATA3": COLORS["orange"],
    }
    sns.barplot(data=df, x="response_pct", y="motif_family", hue="motif_family", palette=palette, legend=False, ax=ax)
    ax.set_xlabel("High-depletion cluster fraction (%)")
    ax.set_ylabel("")
    ax.set_xlim(0, max(40, df["response_pct"].max() + 7))
    for i, row in enumerate(df.itertuples(index=False)):
        ax.text(
            row.response_pct + 1.0,
            i,
            f"{int(row.family_response_n)}/{int(row.family_n)}; FDR {row.fisher_q:.3g}",
            va="center",
            ha="left",
            fontsize=9,
        )
    sns.despine(ax=ax, left=True)
    fig.subplots_adjust(left=0.16, right=0.98, top=0.78, bottom=0.18)
    out = ASSET_DIR / "lc_response_fraction.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return f"assets/{out.name}"


def make_lc_distal_polr2a(tss_summary: pd.DataFrame) -> str:
    order = ["all", "Forkhead", "AP1", "GATA3", "TEAD4"]
    df = tss_summary[
        (tss_summary["minimum_distance_bp"] == 50000)
        & (tss_summary["track"] == "polr2a")
        & (tss_summary["group"].isin(order))
    ].copy()
    df["group"] = pd.Categorical(df["group"], categories=order, ordered=True)
    df = df.sort_values("group")

    fig, ax = plt.subplots(figsize=(8.8, 4.3))
    add_header(
        fig,
        "Distal TSS POLR2A effects are directionally negative but tiny",
        "Host-TSS proxy, sites at least 50 kb from the edited motif.",
    )
    colors = [COLORS["neutral_dark"] if g == "all" else COLORS["blue_dark"] if g == "Forkhead" else COLORS["neutral"] for g in df["group"]]
    sns.barplot(data=df, x="median", y="group", hue="group", palette=dict(zip(df["group"], colors)), legend=False, ax=ax)
    ax.axvline(0, color=TOKENS["ink"], lw=1)
    ax.set_xlabel("Median adjusted POLR2A log2FC at host TSS")
    ax.set_ylabel("")
    for i, row in enumerate(df.itertuples(index=False)):
        pct = 100 * row.fraction_negative
        ax.text(0.00072, i, f"{row.median:+.4f}; {pct:.0f}% negative; n={int(row.n)}", va="center", ha="right", fontsize=9)
    ax.set_xlim(-0.00155, 0.00078)
    sns.despine(ax=ax, left=True)
    fig.subplots_adjust(left=0.16, right=0.98, top=0.78, bottom=0.18)
    out = ASSET_DIR / "lc_distal_polr2a.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return f"assets/{out.name}"


def build_html(asset_paths: dict[str, str], metrics: dict[str, object]) -> str:
    def img(path: str, alt: str, caption: str) -> str:
        return (
            f'<figure><img src="{path}" alt="{html.escape(alt)}">'
            f"<figcaption>{html.escape(caption)}</figcaption></figure>"
        )

    title = "AlphaGenome ISM analysis of MREG DIC perturbations and DIC population cohorts"
    generated = datetime.now().strftime("%Y-%m-%d %H:%M")
    mreg = metrics["mreg"]
    lc = metrics["lc"]
    hc = metrics["hc"]

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(title)}</title>
  <style>
    :root {{
      --surface: #FCFCFD;
      --panel: #FFFFFF;
      --ink: #1F2430;
      --muted: #6F768A;
      --grid: #E6E8F0;
      --axis: #D7DBE7;
      --blue: #2E4780;
      --orange: #804126;
    }}
    body {{ margin: 0; background: var(--surface); color: var(--ink); font-family: Aptos, Inter, "Segoe UI", Arial, sans-serif; }}
    main {{ max-width: 980px; margin: 0 auto; padding: 46px 24px 72px; }}
    header, section {{ margin-bottom: 38px; }}
    h1 {{ font-size: 33px; line-height: 1.12; margin: 0 0 12px; letter-spacing: 0; }}
    h2 {{ font-size: 23px; line-height: 1.22; margin: 0 0 13px; letter-spacing: 0; }}
    h3 {{ font-size: 17px; margin: 20px 0 8px; }}
    p, li {{ font-size: 16px; line-height: 1.62; }}
    .lede {{ color: var(--muted); max-width: 850px; }}
    .summary {{ background: var(--panel); border: 1px solid var(--grid); border-radius: 8px; padding: 18px 22px; }}
    .summary ul {{ margin: 0; padding-left: 20px; }}
    .note {{ color: var(--muted); font-size: 14px; }}
    figure {{ margin: 20px 0 24px; }}
    img {{ max-width: 100%; display: block; border: 1px solid var(--grid); border-radius: 8px; background: white; }}
    figcaption {{ color: var(--muted); font-size: 13px; line-height: 1.45; margin-top: 8px; }}
    table {{ width: 100%; border-collapse: collapse; margin: 16px 0; font-size: 14px; }}
    th, td {{ border-bottom: 1px solid var(--grid); padding: 8px 7px; text-align: left; vertical-align: top; }}
    th {{ color: var(--muted); font-weight: 700; }}
    .metric-grid {{ display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 12px; margin: 18px 0; }}
    .metric {{ background: var(--panel); border: 1px solid var(--grid); border-radius: 8px; padding: 13px 14px; }}
    .metric strong {{ display: block; font-size: 21px; margin-bottom: 4px; }}
    .metric span {{ color: var(--muted); font-size: 13px; line-height: 1.35; display: block; }}
    code {{ background: #F4F5F7; padding: 1px 4px; border-radius: 4px; }}
    @media (max-width: 760px) {{ .metric-grid {{ grid-template-columns: 1fr; }} main {{ padding: 30px 16px 58px; }} }}
  </style>
</head>
<body>
<main data-report-audience="technical">
  <header data-contract-section="title">
    <h1>{html.escape(title)}</h1>
    <p class="lede">A concise report of what was perturbed, what was measured, and how the predicted 1D ChIP-seq and contact-map signals changed. Generated {generated} from local artifacts in <code>/home/fqijun/python/gReLU</code>.</p>
  </header>

  <section data-contract-section="technical-summary">
    <h2>Summary of analyses and results</h2>
    <div class="summary">
      <ul>
        <li><strong>MREG single-locus ISM.</strong> We perturbed three nearby regions: the MREG TSS CTCF motif, the MREG HC-DIC CTCF motif, and the MREG LC-DIC Pol2 peak center. The TSS edit mainly reduced local promoter CTCF/RAD21/POLR2A. The HC-DIC edit strongly reduced local CTCF/RAD21. The LC-DIC edit produced a weaker local reduction across CTCF/RAD21/POLR2A.</li>
        <li><strong>MREG contact-map readout.</strong> On the same edited sequences, the HC-DIC edit reduced the predicted HC-DIC-to-MREG-TSS contact metric ({mreg["hc_to_tss_contact"]:+.3f}). The LC-DIC edit had almost no contact effect to the TSS ({mreg["lc_to_tss_contact"]:+.3f}). This is the clearest single-locus difference between the HC-DIC and LC-DIC perturbations.</li>
        <li><strong>Population ISM.</strong> In the HC-DIC cohort, CTCF motif disruption produced large local CTCF/cohesin depletion with near-zero POLR2A change. In the LC-DIC motif cohort, effects were modest overall and varied by motif family; Forkhead edits showed the clearest local response.</li>
        <li><strong>Limited inference.</strong> The distal TSS proxy signals are small, especially in the LC-DIC population. These results support candidate-level follow-up, not a broad claim that all LC-DICs regulate their host promoters.</li>
      </ul>
    </div>
  </section>

  <section data-contract-section="key-findings">
    <h2>Background and analysis goal</h2>
    <p><strong>The biological question is whether DIC sequence perturbations affect only the local chromatin state or also distal promoter/contact readouts.</strong> HC-DICs and LC-DICs are defined by different chromatin and looping features, so we analyzed them with different perturbation strategies rather than forcing them into one combined mutation design.</p>
    <p><strong>The analysis has two parts.</strong> First, we used the MREG locus as a controlled single-locus example because the region contains a TSS, an HC-DIC, and an LC-DIC close enough to compare in one model context. Second, we summarized existing HC-DIC and LC-DIC population runs to ask whether the single-locus behavior is consistent with broader cohorts. All results are AlphaGenome predictions, not experimental validation.</p>
  </section>

  <section data-contract-section="key-findings">
    <h2>MREG single-locus experiment: local 1D and contact-map signals</h2>
    <p><strong>We tested three perturbations near MREG.</strong> The TSS and HC-DIC edits disrupt local CTCF motifs. The LC-DIC lacks a CTCF motif, so this single-locus LC-DIC test used an exploratory 3 bp substitution at the Pol2 peak center. For each perturbation, we measured local 128 bp ChIP-seq predictions, the MREG TSS readout for DIC edits, and contact-map summary metrics.</p>
    <table>
      <thead><tr><th>Edit</th><th>Local 1D effect, adjusted log2FC</th><th>MREG TSS 1D effect for DIC edits</th><th>Contact-map metric</th></tr></thead>
      <tbody>
        <tr><td>TSS CTCF motif</td><td>CTCF {mreg["tss_local_ctcf"]:+.3f}, RAD21 {mreg["tss_local_rad21"]:+.3f}, POLR2A {mreg["tss_local_polr2a"]:+.3f}</td><td>Not applicable; edit is at TSS</td><td>Local TSS contact metric {mreg["tss_local_contact"]:+.3f}</td></tr>
        <tr><td>HC-DIC CTCF motif</td><td>CTCF {mreg["hc_local_ctcf"]:+.3f}, RAD21 {mreg["hc_local_rad21"]:+.3f}, POLR2A {mreg["hc_local_polr2a"]:+.3f}</td><td>CTCF {mreg["hc_tss_ctcf"]:+.3f}, RAD21 {mreg["hc_tss_rad21"]:+.3f}, POLR2A {mreg["hc_tss_polr2a"]:+.3f}</td><td>HC-DIC to TSS {mreg["hc_to_tss_contact"]:+.3f}</td></tr>
        <tr><td>LC-DIC Pol2 center</td><td>CTCF {mreg["lc_local_ctcf"]:+.3f}, RAD21 {mreg["lc_local_rad21"]:+.3f}, POLR2A {mreg["lc_local_polr2a"]:+.3f}</td><td>CTCF {mreg["lc_tss_ctcf"]:+.3f}, RAD21 {mreg["lc_tss_rad21"]:+.3f}, POLR2A {mreg["lc_tss_polr2a"]:+.3f}</td><td>LC-DIC to TSS {mreg["lc_to_tss_contact"]:+.3f}</td></tr>
      </tbody>
    </table>
    <p><strong>The local 1D effects show that the edits changed the intended regions.</strong> The strongest local losses are TSS CTCF, HC-DIC CTCF/RAD21, and weaker LC-DIC CTCF/RAD21/POLR2A. At the MREG TSS, the HC-DIC edit gives almost no 1D ChIP-seq change, while the LC-DIC edit gives a small negative POLR2A/RAD21 shift.</p>
    {img(asset_paths["mreg_local"], "MREG local REF ALT delta distributions", "Local adjusted ALT-REF distributions for CTCF, RAD21 and POLR2A at the edited TSS, HC-DIC and LC-DIC regions.")}
    <p><strong>The contact-map summary adds information not visible from the 1D TSS tracks alone.</strong> HC-DIC editing reduces the predicted HC-DIC-to-TSS contact metric ({mreg["hc_to_tss_contact"]:+.3f}). LC-DIC editing has almost no contact effect to the TSS ({mreg["lc_to_tss_contact"]:+.3f}). This is a single-locus result, but it is the strongest evidence in the report that the HC-DIC and LC-DIC edits have different distal-contact behavior at MREG.</p>
    {img(asset_paths["mreg_distal"], "MREG DIC to TSS delta distributions", "DIC-edit effects measured at the MREG TSS. This is a named distal readout, not a genome-wide promoter screen.")}
    {img(asset_paths["mreg_contact"], "MREG contact schematic", "Schematic contact-map summary. Values are adjusted contact-map means, target edit minus matched controls. Raw full contact tensors were not retained in the final run; summary metrics were retained.")}
  </section>

  <section data-contract-section="key-findings">
    <h2>HC-DIC population run: strong local CTCF/cohesin depletion</h2>
    <p><strong>The HC-DIC population run used CTCF motif disruption at 134 ready HC-DICs, with three local non-motif controls per site.</strong> The main readout here is the matched-control-adjusted local 4 kb peak log2FC. The median effects are CTCF {hc["ctcf_median"]:+.3f}, RAD21 {hc["rad21_median"]:+.3f}, SMC3 {hc["smc3_median"]:+.3f}, and POLR2A {hc["polr2a_median"]:+.3f}. The observed population signal is therefore dominated by local CTCF/cohesin depletion, not by a Pol2 response.</p>
    {img(asset_paths["hc_bar"], "HC-DIC local effects", "Median 4 kb peak log2FC after matched-control adjustment. The dominant population effect is structural CTCF/cohesin depletion.")}
    <p><strong>A limited interpretation is appropriate.</strong> This result says that the CTCF motif perturbation is sufficient to strongly change local HC-DIC CTCF/cohesin predictions. It does not by itself establish distal promoter regulation for the whole HC-DIC class.</p>
  </section>

  <section data-contract-section="key-findings">
    <h2>LC-DIC population run: motif effects are modest and family-dependent</h2>
    <p><strong>The formal LC-DIC population run used motif edits at Pol2-anchored LC-DICs, not the older arbitrary peak-center edit.</strong> The final cohort contains {lc["n_candidates"]} ready site-by-family motif edits across {lc["n_sites"]} LC-DICs, each compared to a same-Pol2-peak matched control. Overall local POLR2A is negative but small (median {lc["local_polr2a_median"]:+.4f}).</p>
    <div class="metric-grid">
      <div class="metric"><strong>{lc["n_candidates"]}</strong><span>ready LC-DIC motif edits after matched control design</span></div>
      <div class="metric"><strong>{lc["local_polr2a_median"]:+.4f}</strong><span>median local 4 kb POLR2A effect, target minus control</span></div>
      <div class="metric"><strong>{lc["distal_polr2a_median"]:+.4f}</strong><span>median host-TSS POLR2A effect among {lc["tss_n"]} candidates with TSS readout</span></div>
    </div>
    <p><strong>The strongest LC-DIC population pattern is family dependence.</strong> Forkhead edits show the clearest local depletion pattern. In the high-depletion cluster, Forkhead contributes {lc["forkhead_resp"]}/{lc["forkhead_n"]} candidates (FDR {lc["forkhead_q"]:.3g}), while GATA3 contributes only {lc["gata3_resp"]}/{lc["gata3_n"]} candidates (FDR {lc["gata3_q"]:.3g}).</p>
    {img(asset_paths["lc_heatmap"], "LC-DIC local family heatmap", "Family-level local effects show that Forkhead is the clearest responsive family; GATA3 remains near zero despite being the largest family.")}
    {img(asset_paths["lc_response"], "LC-DIC response fraction", "Population response cluster by motif family. The high-depletion cluster is an exploratory model-derived group, not a paper-defined subtype.")}
    <p><strong>The distal TSS proxy remains small.</strong> Across all TSS-mapped LC-DIC motif candidates the median POLR2A effect is {lc["distal_polr2a_median"]:+.4f}. Beyond 50 kb, Forkhead remains directionally negative ({lc["forkhead_distal_polr2a_median"]:+.4f}; {lc["forkhead_distal_neg_pct"]:.1f}% negative), but the magnitude is far smaller than local chromatin effects. This is a weak distal signal and should be used to prioritize candidates, not to claim a general LC-DIC promoter effect.</p>
    {img(asset_paths["lc_tss"], "LC-DIC TSS POLR2A by distance", "Host-TSS proxy effects by distance. The distal signal is directionally consistent in selected families but very small in absolute magnitude.")}
  </section>

  <section data-contract-section="scope-data-and-metric-definitions">
    <h2>What infrastructure was used</h2>
    <p><strong>The batch ISM infrastructure separates the expensive inference step from downstream summaries.</strong> Site manifests define coordinates and anchors. Mutation strategies define target edits and matched controls. AlphaGenome inference writes reusable REF/ALT feature tables. Local, TSS, population, and contact-map summaries are then computed from those outputs. This is why the same run can be summarized by local tracks, named TSS readouts, motif-family summaries, and contact metrics without redesigning the whole experiment.</p>
    <table>
      <thead><tr><th>Layer</th><th>What it owns</th><th>Why it matters</th></tr></thead>
      <tbody>
        <tr><td>Site and track manifests</td><td>Coordinates, anchors, tracks, cohort labels</td><td>Makes HC, LC and single-locus comparisons auditable.</td></tr>
        <tr><td>Mutation preparation</td><td>Target edits and matched controls</td><td>Keeps biological perturbation separate from downstream summaries.</td></tr>
        <tr><td>1D / contact inference</td><td>REF/ALT model predictions</td><td>Expensive step; saved once and reused.</td></tr>
        <tr><td>Feature and readout analysis</td><td>Local, TSS, population, and contact summaries</td><td>Allows new questions without rerunning the model when raw summaries are sufficient.</td></tr>
      </tbody>
    </table>
  </section>

  <section data-contract-section="methodology">
    <h2>Appendix: experimental operations</h2>
    <h3>MREG single-locus experiment</h3>
    <p>Three regions were evaluated in a shared 1,048,576 bp AlphaGenome context: MREG TSS at chr2:216013551, HC-DIC Rad21 summit at chr2:215950160, and LC-DIC Rad21/Pol2 region near chr2:215979931/215979780. TSS and HC-DIC edits used strand-aware CTCF motif disruption. LC-DIC used the exploratory deterministic 3 bp Pol2-center substitution because this LC-DIC site does not have a CTCF motif.</p>
    <h3>LC-DIC population experiment</h3>
    <p>The formal LC-DIC run restricted to audited Pol2-anchored sites, scanned for high-confidence GATA3, AP-1, Forkhead and TEAD4 motifs within the selected Pol2 peak, and paired each valid motif edit with a same-peak matched control. Effects reported here are target motif edit minus matched control.</p>
    <h3>HC-DIC population experiment</h3>
    <p>The HC-DIC run used CTCF PWM disruption at ready HC-DICs, with three local non-motif matched controls per site. The main population readout is the local 4 kb peak log2FC after matched-control adjustment.</p>
  </section>

  <section data-contract-section="limitations-uncertainty-and-robustness-checks">
    <h2>Interpretation limits</h2>
    <p><strong>The main conclusion should remain limited.</strong> The MREG contact-map result distinguishes the HC-DIC and LC-DIC edits at one locus. The HC-DIC population result shows strong local CTCF/cohesin depletion. The LC-DIC population result shows modest, motif-family-dependent local effects and very small distal TSS proxy effects. These results motivate focused follow-up rather than broad claims about all DICs.</p>
    <ul>
      <li>All results are AlphaGenome predictions, not wet-lab perturbation measurements.</li>
      <li>The MREG contact-map report retained summary contact metrics, not full raw contact tensors, after the raw-map rerun was cancelled.</li>
      <li>The LC-DIC host-TSS assignment is a proxy from local gene annotation unless loop-supported promoter evidence is added.</li>
      <li>The old LC-DIC Pol2-center edit is not a Pol2 motif knockout; the formal motif experiment is the stronger LC-DIC population analysis.</li>
      <li>HC and LC population clusters should not be interpreted from a combined HC/LC run because the mutation strategies differ.</li>
    </ul>
  </section>

  <section data-contract-section="recommended-next-steps">
    <h2>Next direction</h2>
    <p><strong>The next useful experiment is focused follow-up on selected LC-DICs.</strong> Forkhead-positive candidates with loop-supported promoters are the most reasonable subset to test next. This directly asks whether a selected LC-DIC subset shows stronger distal propagation than the broad host-TSS proxy cohort.</p>
  </section>

  <section data-contract-section="further-questions">
    <h2>Further questions</h2>
    <ul>
      <li>Do Forkhead-positive LC-DICs with loop-supported promoters show stronger distal propagation than the host-TSS proxy cohort?</li>
      <li>Does the HC-DIC contact decrease to TSS at MREG generalize to other promoter-proximal HC-DICs, or is it locus-specific?</li>
      <li>Which LC-DIC motif families remain active after adding contact-map readouts, not only 1D ChIP-seq summaries?</li>
    </ul>
    <p class="note">Source inventory and report QA are saved in <code>source_notes.md</code> next to this report.</p>
  </section>
</main>
</body>
</html>
"""


def main() -> None:
    apply_style()
    ASSET_DIR.mkdir(parents=True, exist_ok=True)

    asset_paths = {
        "mreg_local": copy_asset(MREG_DIST / "figures/local_delta_distribution.png", "mreg_local_delta_distribution.png"),
        "mreg_distal": copy_asset(MREG_DIST / "figures/dic_to_tss_delta_distribution.png", "mreg_dic_to_tss_delta_distribution.png"),
    }
    asset_paths["mreg_contact"] = make_contact_schematic(MREG_CONTACT / "contact_metric_adjusted.tsv")
    hc_bar, hc_effects = make_hc_bar(HC_SUMMARY)
    asset_paths["hc_bar"] = hc_bar

    local_summary = pd.read_csv(LC_EFFECT / "local_track_summary.tsv", sep="\t")
    tss_summary = pd.read_csv(LC_EFFECT / "tss_distance_summary.tsv", sep="\t")
    cluster_enrich = pd.read_csv(LC_EFFECT / "cluster_motif_enrichment.tsv", sep="\t")
    asset_paths["lc_heatmap"] = make_lc_family_heatmap(local_summary)
    asset_paths["lc_response"] = make_lc_response_fraction(cluster_enrich)
    asset_paths["lc_tss"] = make_lc_distal_polr2a(tss_summary)
    site_effects = pd.read_csv(LC_EFFECT / "site_effects.tsv", sep="\t")
    contact = pd.read_csv(MREG_CONTACT / "contact_metric_adjusted.tsv", sep="\t")
    mreg_local = pd.read_csv(MREG_DIST / "local_distribution_summary.tsv", sep="\t")
    mreg_distal = pd.read_csv(MREG_DIST / "dic_to_tss_distribution_summary.tsv", sep="\t")

    local_polr2a = local_summary[
        (local_summary["group"] == "all") & (local_summary["track"] == "polr2a") & (local_summary["window_bp"] == 4096)
    ].iloc[0]
    distal_polr2a = tss_summary[
        (tss_summary["minimum_distance_bp"] == 0) & (tss_summary["group"] == "all") & (tss_summary["track"] == "polr2a")
    ].iloc[0]
    fork_distal = tss_summary[
        (tss_summary["minimum_distance_bp"] == 50000)
        & (tss_summary["group"] == "Forkhead")
        & (tss_summary["track"] == "polr2a")
    ].iloc[0]
    fork = cluster_enrich[cluster_enrich["motif_family"] == "Forkhead"].iloc[0]
    gata3 = cluster_enrich[cluster_enrich["motif_family"] == "GATA3"].iloc[0]
    contact_tss = contact[(contact["metric"] == "edit_anchor_to_readout") & (contact["target_region"] == "mreg_tss_4kb")].set_index(
        "mutation_id"
    )
    mreg_local_lookup = mreg_local.set_index(["comparison", "track"])["adjusted mean log2FC"]
    mreg_distal_lookup = mreg_distal.set_index(["comparison", "track"])["adjusted mean log2FC"]

    hc_lookup = hc_effects.set_index("Track")
    cluster_summary = read_markdown_table(HC_SUMMARY, "Final cluster sizes")
    cluster0_n = int(cluster_summary[cluster_summary["Cluster"] == "0"]["Sites"].iloc[0])
    cluster1_n = int(cluster_summary[cluster_summary["Cluster"] == "1"]["Sites"].iloc[0])

    metrics = {
        "mreg": {
            "tss_local_contact": float(
                contact_tss.loc["mreg_tss_00_ctcf_snv", "adjusted_delta_contact_mean"]
            ),
            "hc_to_tss_contact": float(
                contact_tss.loc["mreg_hc_dic_00_ctcf_snv", "adjusted_delta_contact_mean"]
            ),
            "lc_to_tss_contact": float(
                contact_tss.loc["mreg_lc_dic_00_pol2_perturbation", "adjusted_delta_contact_mean"]
            ),
            "tss_local_ctcf": float(mreg_local_lookup.loc[("TSS edit", "CTCF")]),
            "tss_local_rad21": float(mreg_local_lookup.loc[("TSS edit", "RAD21")]),
            "tss_local_polr2a": float(mreg_local_lookup.loc[("TSS edit", "POLR2A")]),
            "hc_local_ctcf": float(mreg_local_lookup.loc[("HC-DIC edit", "CTCF")]),
            "hc_local_rad21": float(mreg_local_lookup.loc[("HC-DIC edit", "RAD21")]),
            "hc_local_polr2a": float(mreg_local_lookup.loc[("HC-DIC edit", "POLR2A")]),
            "lc_local_ctcf": float(mreg_local_lookup.loc[("LC-DIC edit", "CTCF")]),
            "lc_local_rad21": float(mreg_local_lookup.loc[("LC-DIC edit", "RAD21")]),
            "lc_local_polr2a": float(mreg_local_lookup.loc[("LC-DIC edit", "POLR2A")]),
            "hc_tss_ctcf": float(mreg_distal_lookup.loc[("HC-DIC edit -> TSS", "CTCF")]),
            "hc_tss_rad21": float(mreg_distal_lookup.loc[("HC-DIC edit -> TSS", "RAD21")]),
            "hc_tss_polr2a": float(mreg_distal_lookup.loc[("HC-DIC edit -> TSS", "POLR2A")]),
            "lc_tss_ctcf": float(mreg_distal_lookup.loc[("LC-DIC edit -> TSS", "CTCF")]),
            "lc_tss_rad21": float(mreg_distal_lookup.loc[("LC-DIC edit -> TSS", "RAD21")]),
            "lc_tss_polr2a": float(mreg_distal_lookup.loc[("LC-DIC edit -> TSS", "POLR2A")]),
            "distal_polr2a_lc": float(
                mreg_distal[
                    (mreg_distal["comparison"] == "LC-DIC edit -> TSS") & (mreg_distal["track"] == "POLR2A")
                ]["adjusted mean log2FC"].iloc[0]
            ),
        },
        "lc": {
            "n_candidates": int(site_effects["site_id"].nunique()),
            "n_sites": int(site_effects["source_site_id"].nunique()) if "source_site_id" in site_effects.columns else 108,
            "local_polr2a_median": float(local_polr2a["median"]),
            "distal_polr2a_median": float(distal_polr2a["median"]),
            "tss_n": int(distal_polr2a["n"]),
            "cluster_n": int(cluster_enrich["family_response_n"].sum()),
            "forkhead_resp": int(fork["family_response_n"]),
            "forkhead_n": int(fork["family_n"]),
            "forkhead_q": float(fork["fisher_q"]),
            "gata3_resp": int(gata3["family_response_n"]),
            "gata3_n": int(gata3["family_n"]),
            "gata3_q": float(gata3["fisher_q"]),
            "forkhead_distal_polr2a_median": float(fork_distal["median"]),
            "forkhead_distal_neg_pct": float(fork_distal["fraction_negative"]) * 100,
        },
        "hc": {
            "n_sites": 134,
            "ctcf_median": float(hc_lookup.loc["CTCF", "Median"]),
            "rad21_median": float(hc_lookup.loc["RAD21", "Median"]),
            "smc3_median": float(hc_lookup.loc["SMC3", "Median"]),
            "polr2a_median": float(hc_lookup.loc["POLR2A", "Median"]),
            "cluster0_n": cluster0_n,
            "cluster1_n": cluster1_n,
        },
    }

    report_html = build_html(asset_paths, metrics)
    (OUT_DIR / "report.html").write_text(report_html)

    source_notes = f"""# Source notes

## Report contract

- Delivery mode: static HTML.
- Audience specification: technical.
- Required structure mapping:
  - Title: report header.
  - Technical summary: top summary section.
  - Key findings with visual evidence: MREG, LC-DIC, and HC-DIC result sections.
  - Scope, data, and metric definitions: infrastructure section plus appendix methods.
  - Methodology: appendix "what was done".
  - Limitations, uncertainty, and robustness checks: appendix limitations.
  - Recommended next steps: recommended next steps section.
  - Further questions: further questions section.

## Local sources used

- MREG distribution report: `{MREG_DIST}`
- MREG contact summaries: `{MREG_CONTACT}`
- LC-DIC formal motif effect analysis: `{LC_EFFECT}`
- Previous LC-DIC motif PDF/report assets retained for provenance: `{LC_REPORT}`
- HC-DIC population summary: `{HC_SUMMARY}`
- HC-DIC report and raw representation tables: `{ROOT / 'outputs/dic_batch_analysis_hc'}`

## Chart map

- `mreg_local_delta_distribution.png`: copied from MREG single-locus distribution report; supports local REF/ALT/delta interpretation.
- `mreg_dic_to_tss_delta_distribution.png`: copied from MREG single-locus distribution report; supports distal TSS 1D readout.
- `mreg_contact_schematic.png`: generated from `contact_metric_adjusted.tsv`; supports contact-map interpretation.
- `lc_family_effect_heatmap.png`: generated from `local_track_summary.tsv`; supports motif-family local effects.
- `lc_response_fraction.png`: generated from `cluster_motif_enrichment.tsv`; supports response-cluster enrichment by family.
- `lc_distal_polr2a.png`: generated from `tss_distance_summary.tsv`; supports distal proxy magnitude.
- `hc_local_effects_bar.png`: generated from `experiment/hc_dic_population_result_summary.md`; supports HC-DIC local population conclusion.

## Important omissions

- Full raw contact-map tensors are not included. The complete summary contact metrics are retained in `contact_metric_adjusted.tsv`; the raw-map rerun was cancelled at the user's request.
- RNA-seq is not included; current primary readout is 128 bp ChIP-seq.
"""
    (OUT_DIR / "source_notes.md").write_text(source_notes)
    (OUT_DIR / "report_metadata.json").write_text(
        json.dumps(
            {
                "generated_at": datetime.now().isoformat(),
                "report": str(OUT_DIR / "report.html"),
                "assets": asset_paths,
                "metrics": metrics,
            },
            indent=2,
        )
    )
    print(OUT_DIR / "report.html")


if __name__ == "__main__":
    main()
