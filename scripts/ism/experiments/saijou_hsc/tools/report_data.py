"""Prepare data frames consumed by the canonical Saijou report."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


ORIGINAL_GROUP_LABELS = {
    "liver_rna": "liver RNA",
    "fibroblast_rna": "fibroblast RNA",
    "liver_cage": "liver CAGE",
    "fibroblast_cage": "fibroblast CAGE",
    "hsc_cage": "HSC CAGE",
    "smooth_muscle_cage": "smooth muscle CAGE",
    "mesenchymal_cage": "mesenchymal CAGE",
    "liver_accessibility": "liver accessibility",
    "fibroblast_accessibility": "fibroblast accessibility",
    "liver_active_chromatin": "liver active chromatin",
    "fibroblast_active_chromatin": "fibroblast active chromatin",
    "liver_tf": "liver TF",
    "fibroblast_tf": "fibroblast TF",
}


@dataclass(frozen=True)
class ReportData:
    """Prepared report datasets plus the validations printed by the CLI."""

    frames: dict[str, pd.DataFrame]
    candidate_validation: dict[str, object]
    motif_validation: dict[str, object]


def segment_label(row: pd.Series) -> str:
    """Return the compact transcription-oriented label used in report figures."""

    return f"{row.gene} {int(row.tx_start):+d}..{int(row.tx_end):+d}"


def _read_report_inputs(root: Path) -> tuple[dict[str, pd.DataFrame], dict[str, object]]:
    analysis = root / "analysis"
    data_dir = analysis / "report_data"
    frames = {
        "overview": pd.read_csv(data_dir / "candidate_overview.tsv", sep="\t"),
        "fine_cell_matrix": pd.read_csv(data_dir / "fine_cell_matrix.tsv", sep="\t"),
        "original_group_matrix": pd.read_csv(
            data_dir / "original_group_matrix.tsv", sep="\t"
        ),
        "motif_families": pd.read_csv(
            data_dir / "motif_family_summary.tsv", sep="\t"
        ),
        "controls": pd.read_csv(data_dir / "known_control_recovery.tsv", sep="\t"),
        "genes": pd.read_csv(data_dir / "gene_summary.tsv", sep="\t"),
        "transcripts": pd.read_csv(
            analysis / "representative_transcripts.tsv", sep="\t"
        ),
        "mdk_hub": pd.read_csv(
            analysis / "mdk_manifest_robustness/hub_summary.tsv", sep="\t"
        ),
        "mdk_centers": pd.read_csv(
            analysis / "mdk_manifest_robustness/center_metrics.tsv", sep="\t"
        ),
        "mdk_concordance": pd.read_csv(
            analysis / "mdk_manifest_robustness/concordance.tsv", sep="\t"
        ),
    }
    validations = {
        "pipeline": json.loads(
            (analysis / "pipeline_validation_summary.json").read_text()
        ),
        "candidate": json.loads((analysis / "validation_summary.json").read_text()),
        "motif": json.loads((analysis / "motif_validation_summary.json").read_text()),
    }
    return frames, validations


def _prepare_overview(overview: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, str]]:
    overview = overview.copy()
    overview["segment_label"] = overview.apply(segment_label, axis=1)
    overview["hsc_status"] = overview.both_models_hsc_top.map(
        {True: "HSC top in both", False: "not HSC top in both"}
    )
    overview["fine_direction"] = overview.fine_signed_direction_agreement.map(
        {True: "same signed direction", False: "different signed direction"}
    )
    labels = overview.set_index("candidate_segment_id").segment_label.to_dict()
    return overview, labels


def _prepare_hsc_ratios(overview: pd.DataFrame) -> pd.DataFrame:
    shared = ["candidate_segment_id", "segment_label", "gene"]
    model_columns = {
        "AlphaGenome fine-tuned": {
            "hsc_vs_max_other_ratio_ag": "hsc_ratio",
            "hsc_median_abs_effect_ag": "hsc_abs_effect",
            "hsc_median_effect_ag": "hsc_signed_effect",
            "top_cell_ag": "top_cell",
        },
        "Borzoi fine-tuned": {
            "hsc_vs_max_other_ratio_bz": "hsc_ratio",
            "hsc_median_abs_effect_bz": "hsc_abs_effect",
            "hsc_median_effect_bz": "hsc_signed_effect",
            "top_cell_bz": "top_cell",
        },
    }
    parts = []
    for model, columns in model_columns.items():
        parts.append(
            overview[shared + list(columns)]
            .rename(columns=columns)
            .assign(model=model)
        )
    ratios = pd.concat(parts, ignore_index=True)
    segment_order = (
        overview.sort_values(["gene", "segment_rank"])
        .reset_index(drop=True)
        .reset_index()
        .set_index("candidate_segment_id")["index"]
    )
    ratios["segment_order"] = ratios.candidate_segment_id.map(segment_order)
    return ratios.sort_values(["segment_order", "model"])


def _prepare_fine_matrix(
    fine_matrix: pd.DataFrame, labels: dict[str, str]
) -> pd.DataFrame:
    fine_matrix = fine_matrix.copy()
    fine_matrix["segment_label"] = fine_matrix.locus_id.map(labels)
    fine_matrix["model_short"] = fine_matrix.model_label.map(
        {"AlphaGenome fine-tuned": "AG-FT", "Borzoi fine-tuned": "BZ-FT"}
    )
    fine_matrix["row_label"] = (
        fine_matrix.segment_label + " / " + fine_matrix.model_short
    )
    fine_matrix["cell_label"] = fine_matrix.cell_type.str.upper()
    return fine_matrix


def _prepare_original_top(overview: pd.DataFrame) -> pd.DataFrame:
    shared = ["candidate_segment_id", "segment_label", "gene"]
    model_columns = {
        "AlphaGenome original": {
            "top_original_group_ag": "track_group",
            "top_original_abs_effect_ag": "abs_effect",
            "top_original_signed_effect_ag": "signed_effect",
            "top_original_ref_sum_ag": "reference_sum",
        },
        "Borzoi original": {
            "top_original_group_bz": "track_group",
            "top_original_abs_effect_bz": "abs_effect",
            "top_original_signed_effect_bz": "signed_effect",
            "top_original_ref_sum_bz": "reference_sum",
        },
    }
    parts = []
    for model, columns in model_columns.items():
        parts.append(
            overview[shared + list(columns)]
            .rename(columns=columns)
            .assign(model=model)
        )
    return pd.concat(parts, ignore_index=True)


def _prepare_original_matrix(
    original_matrix: pd.DataFrame, labels: dict[str, str]
) -> pd.DataFrame:
    original_matrix = original_matrix.copy()
    original_matrix["segment_label"] = original_matrix.locus_id.map(labels)
    original_matrix["track_group_label"] = original_matrix.track_group.map(
        ORIGINAL_GROUP_LABELS
    )
    original_matrix["model_short"] = original_matrix.model_label.map(
        {"AlphaGenome original": "AG-orig", "Borzoi original": "BZ-orig"}
    )
    original_matrix["row_label"] = (
        original_matrix.segment_label + " / " + original_matrix.model_short
    )
    return original_matrix


def _prepare_motif_top(
    motif_families: pd.DataFrame, labels: dict[str, str]
) -> pd.DataFrame:
    motif_top = motif_families.loc[motif_families.family_rank.le(3)].copy()
    motif_top["segment_label"] = motif_top.candidate_segment_id.map(labels)
    return motif_top


def _prepare_mdk_centers(mdk_centers: pd.DataFrame) -> pd.DataFrame:
    centers = mdk_centers.loc[
        mdk_centers.variant_offset_from_tss_transcription_bp.between(441, 507)
    ].copy()
    model = centers.model.str.replace("AlphaGenome", "AG", regex=False).str.replace(
        "Borzoi", "BZ", regex=False
    )
    manifest = centers.manifest_label.map(
        {
            "final_mdk_background": "final Mdk audit",
            "all_gene_manifest": "nine-gene manifest",
        }
    )
    centers["series"] = model + " / " + manifest
    return centers


def _prepare_validation_table(pipeline_validation: dict[str, object]) -> pd.DataFrame:
    rows = []
    for run in pipeline_validation["runs"]:
        rows.append(
            {
                "run": run["run"],
                "genes": len(run.get("genes") or []),
                "tracks": run.get("tracks"),
                "run_megabytes": run.get("run_bytes", 0) / 1_000_000,
                "profile_megabytes": run.get("profile_bytes", 0) / 1_000_000,
                "max_reconstruction_relative_error": run.get(
                    "max_reconstruction_relative_error"
                ),
                "profile_nonfinite_values": run.get("profile_nonfinite_values"),
                "profile_index_mismatches": run.get("profile_index_mismatches"),
                "under_3gb": run.get("under_3gb"),
            }
        )
    return pd.DataFrame(rows)


def _overview_segment(overview: pd.DataFrame, gene: str, rank: int) -> pd.Series:
    return overview.loc[
        overview.gene.eq(gene) & overview.segment_rank.eq(rank)
    ].iloc[0]


def _prepare_gene_interpretation(overview: pd.DataFrame) -> pd.DataFrame:
    focus = {
        "Mdk": overview.loc[
            overview.candidate_segment_id.eq("mdk_candidate_02_tx+463_+475")
        ].iloc[0],
        "Acta2": _overview_segment(overview, "Acta2", 2),
        "Col1a1": _overview_segment(overview, "Col1a1", 2),
        "Col1a2": _overview_segment(overview, "Col1a2", 1),
        "Hgf": _overview_segment(overview, "Hgf", 2),
        "Igf1": _overview_segment(overview, "Igf1", 1),
        "Ngf": _overview_segment(overview, "Ngf", 2),
        "Timp1": _overview_segment(overview, "Timp1", 1),
        "Vegfc": _overview_segment(overview, "Vegfc", 1),
    }
    mdk = focus["Mdk"]
    text = {
        "Mdk": f"+463..+475 的 SP/KLF-rich 内含子模块在本轮两个模型均 HSC-first（AG {mdk.hsc_vs_max_other_ratio_ag:.2f}×；BZ {mdk.hsc_vs_max_other_ratio_bz:.2f}×）。整段 +441..+507 的 Borzoi 区段级偏好可复现，但中心级排序受 shuffle realization 影响；原始模型已见 fibroblast RNA / liver RNA 与活性染色质敏感性，因此微调更像叠加 HSC RNA specificity，而不是创造全新 grammar。",
        "Acta2": f"已知 -61 CArG 被 motif 与位置排名恢复，但不是 HSC-specific；新发现 +321..+323 在两模型均 HSC-first（最低比值 {focus['Acta2'].minimum_hsc_ratio:.2f}×），原始 smooth-muscle/fibroblast 轨道也有响应，适合作为 HSC 状态依赖候选而非替代 CArG。",
        "Col1a1": f"-71 inverted CCAAT / NF-Y 是强位置正控并在原始 RNA/染色质轨道显著；+93..+99 只有 Borzoi HSC-first，AlphaGenome 不是（跨模型最低比值 {focus['Col1a1'].minimum_hsc_ratio:.2f}×），不应升级为稳健 HSC 元件。",
        "Col1a2": f"+447..+459 是本轮最强跨模型 HSC ratio 之一（最低 {focus['Col1a2'].minimum_hsc_ratio:.2f}×），但位于低支持度、非-basic 的 Col1a2-205 alternative 5′ exon，且原始 smooth-muscle/fibroblast 轨道强响应。湿实验前必须以 canonical Col1a2-201 TSS 复扫。",
        "Hgf": f"+339..+341 在两模型 HSC-first（最低 {focus['Hgf'].minimum_hsc_ratio:.2f}×）并扰动 AP-1 family；原始 smooth-muscle CAGE / liver chromatin 已敏感。但分析 TSS 对应 retained-intron Hgf-202，不能直接等同 canonical Hgf core-promoter C/EBP 元件。",
        "Igf1": f"-81..-79 为温和但双模型 HSC-first（最低 {focus['Igf1'].minimum_hsc_ratio:.2f}×），原始 accessibility/chromatin 有响应；本次 ±0.5 kb 不能检验远端/内含子 STAT5 sites，因此阴性或弱近端结果不能否定经典 GH–STAT5 regulation。",
        "Ngf": f"-51..-37 是强但较泛的 promoter sensitivity；+55..+57 更 HSC-selective（最低 {focus['Ngf'].minimum_hsc_ratio:.2f}×）并扰动 STAT/KLF motifs，但距代表转录本 splice boundary 13 bp，虽通过 ±6 bp core 保护，仍需 splice-aware secondary edits。原始 smooth-muscle/chromatin 响应明显。",
        "Timp1": "-85..-75 与 -35..-29 均为负向 promoter effects 且 HSC-first；AP-1/ETS/SP motifs 与原始 accessibility/RNA/chromatin 同时响应，支持 pre-existing active promoter grammar。已知 +159 donor 未进入候选。",
        "Vegfc": f"-505..-503 是温和 HSC-first（最低 {focus['Vegfc'].minimum_hsc_ratio:.2f}×）；+447..+455 在 fine-tuned 模型并不 HSC-specific，却在原始 liver RNA/CAGE/accessibility/chromatin 强响应，更像通用 liver regulatory sensitivity。已验证的远端 SOX7 element 约 -152 kb，完全超出本扫描。",
    }
    return pd.DataFrame(
        [
            {
                "gene": gene,
                "candidate": focus[gene].segment_label,
                "interpretation": text[gene],
            }
            for gene in focus
        ]
    )


def prepare_report_data(root: Path) -> ReportData:
    """Load and derive all datasets used by report figures and the artifact."""

    raw, validations = _read_report_inputs(root)
    overview, labels = _prepare_overview(raw["overview"])
    validation = _prepare_validation_table(validations["pipeline"])
    frames = {
        "overview": overview,
        "hsc_ratio": _prepare_hsc_ratios(overview),
        "fine_cell_matrix": _prepare_fine_matrix(raw["fine_cell_matrix"], labels),
        "original_top": _prepare_original_top(overview),
        "original_group_matrix": _prepare_original_matrix(
            raw["original_group_matrix"], labels
        ),
        "controls": raw["controls"],
        "motif_top": _prepare_motif_top(raw["motif_families"], labels),
        "gene_interpretation": _prepare_gene_interpretation(overview),
        "genes": raw["genes"],
        "transcripts": raw["transcripts"],
        "mdk_hub": raw["mdk_hub"],
        "mdk_centers": _prepare_mdk_centers(raw["mdk_centers"]),
        "mdk_concordance": raw["mdk_concordance"],
        "validation": validation,
    }
    frames["summary"] = pd.DataFrame(
        [
            {
                "genes": int(overview.gene.nunique()),
                "segments": int(len(overview)),
                "both_hsc_segments": int(overview.both_models_hsc_top.sum()),
                "candidate_mutations": int(overview.tested_mutations.sum()),
                "largest_run_gb": float(validation.run_megabytes.max() / 1000),
            }
        ]
    )
    return ReportData(
        frames=frames,
        candidate_validation=validations["candidate"],
        motif_validation=validations["motif"],
    )
