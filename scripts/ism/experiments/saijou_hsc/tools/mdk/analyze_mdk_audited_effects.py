#!/usr/bin/env python
"""Build report-independent audited effect tables for the focused Mdk study."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

if __package__:
    from ..effect_summary import fixed_readout_cell_profile, summarize_cell_windows
    from ..harness import SAIJOU_ROOT, require_finite, require_unique, write_json
else:
    from sys import path as sys_path

    sys_path.insert(0, str(Path(__file__).resolve().parents[2]))
    from tools.effect_summary import fixed_readout_cell_profile, summarize_cell_windows
    from tools.harness import SAIJOU_ROOT, require_finite, require_unique, write_json


DEFAULT_OUT = SAIJOU_ROOT / "analysis/mdk_audited_effects"
READOUT_ROLE = "tes_3prime_1024bp"
GENE_STRAND = "-"
CELLS = {
    "hsc_finetuned_10x": "HSC",
    "mac_finetuned_10x": "Macrophage",
    "lsec_finetuned_10x": "LSEC",
    "chol_finetuned_10x": "Cholangiocyte",
}
WINDOWS = {
    "Primary SP/KLF window (+463..+475)": (463, 475),
    "Secondary window (+499..+507)": (499, 507),
}
PRIMARY_LOCUS = "mdk_candidate_02_tx+463_+475"
SECONDARY_LOCUS = "mdk_candidate_01_tx+499_+507"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=SAIJOU_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    return parser.parse_args()


def build_four_cell_browser(root: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Build the fixed-readout four-cell Browser and two-window summary."""

    profiles = []
    for model, run in (
        ("AlphaGenome fine-tuned", "alphagenome_finetuned"),
        ("Borzoi fine-tuned", "borzoi_finetuned"),
    ):
        profiles.append(
            fixed_readout_cell_profile(
                root / "runs" / run / "features/combined_mutation_features.parquet",
                model_label=model,
                gene="Mdk",
                readout_role=READOUT_ROLE,
                cell_groups=CELLS,
                gene_strand=GENE_STRAND,
            )
        )
    long = pd.concat(profiles, ignore_index=True)
    center_sets = [
        set(part.variant_offset_from_tss_transcription_bp) for part in profiles
    ]
    if center_sets[0] != center_sets[1]:
        raise ValueError("Fine-tuned model center sets differ for Mdk")

    wide = long.pivot(
        index=["cell", "track_group", "variant_offset_from_tss_transcription_bp"],
        columns="model",
        values="effect",
    ).reset_index()
    wide.columns.name = None
    wide = wide.rename(
        columns={
            "variant_offset_from_tss_transcription_bp": "offset_bp",
            "AlphaGenome fine-tuned": "alphagenome_effect",
            "Borzoi fine-tuned": "borzoi_effect",
        }
    )
    wide["readout"] = "TES/3-prime 1,024-bp RNA output"
    wide["in_primary_window"] = wide.offset_bp.between(463, 475)
    wide["in_secondary_window"] = wide.offset_bp.between(499, 507)
    wide = wide.sort_values(["cell", "offset_bp"]).reset_index(drop=True)
    require_unique(wide, ["cell", "offset_bp"], name="Mdk four-cell Browser")
    require_finite(
        wide,
        ["alphagenome_effect", "borzoi_effect"],
        name="Mdk four-cell Browser",
    )

    summary_long = summarize_cell_windows(long, WINDOWS)
    rows = []
    for (window, start, end, cell), group in summary_long.groupby(
        ["window", "window_start_bp", "window_end_bp", "cell"], sort=False
    ):
        by_model = group.set_index("model")
        ag = by_model.loc["AlphaGenome fine-tuned"]
        bz = by_model.loc["Borzoi fine-tuned"]
        rows.append(
            {
                "window": window,
                "window_start_bp": start,
                "window_end_bp": end,
                "cell": cell,
                "centers": int(ag.centers),
                "alphagenome_median_signed_log2fc": ag.median_signed_log2fc,
                "alphagenome_median_abs_log2fc": ag.median_abs_log2fc,
                "alphagenome_fraction_negative": ag.fraction_centers_negative,
                "alphagenome_cell_rank": int(ag.cell_rank),
                "borzoi_median_signed_log2fc": bz.median_signed_log2fc,
                "borzoi_median_abs_log2fc": bz.median_abs_log2fc,
                "borzoi_fraction_negative": bz.fraction_centers_negative,
                "borzoi_cell_rank": int(bz.cell_rank),
            }
        )
    summary = pd.DataFrame(rows)
    metrics = {
        "centers_per_cell": len(center_sets[0]),
        "common_y_limit": float(
            np.ceil(
                10
                * 1.08
                * max(
                    wide.alphagenome_effect.abs().max(),
                    wide.borzoi_effect.abs().max(),
                )
            )
            / 10
        ),
    }
    return wide, summary, metrics


def _assay_class(manifest: pd.DataFrame) -> pd.Series:
    result = manifest.modality.map(
        {
            "rna": "RNA-seq",
            "cage": "CAGE",
            "atac": "Chromatin accessibility",
            "dnase": "Chromatin accessibility",
        }
    )
    result.loc[manifest.track_group.str.contains("active_chromatin")] = (
        "Histone ChIP-seq"
    )
    result.loc[manifest.track_group.str.endswith("_tf")] = "TF/Pol II ChIP-seq"
    return result


def summarize_original_borzoi(
    root: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Summarize the 65 selected original-Borzoi tracks for the two Mdk edits."""

    run = root / "runs/borzoi_original"
    data = pd.read_parquet(
        run / "features/combined_mutation_features.parquet",
        filters=[("gene", "==", "Mdk")],
    )
    compatible = data.track_strand.fillna(".").astype(str).isin([".", GENE_STRAND])
    data = data.loc[data.readout_role.eq(READOUT_ROLE) & compatible].copy()
    by_mutation = (
        data.groupby(
            [
                "locus_id",
                "mutation_id",
                "variant_offset_from_tss_transcription_bp",
                "track_group",
            ],
            sort=False,
        )
        .agg(
            compatible_tracks=("track_id", "nunique"),
            signed_log2fc=("log2fc_ratio_of_sums", "median"),
            ref_sum=("ref_sum", "median"),
        )
        .reset_index()
    )
    summary = (
        by_mutation.groupby(["locus_id", "track_group"], sort=False)
        .agg(
            mutations=("mutation_id", "nunique"),
            centers=("variant_offset_from_tss_transcription_bp", "nunique"),
            compatible_tracks=("compatible_tracks", "median"),
            median_signed_log2fc=("signed_log2fc", "median"),
            median_abs_log2fc=(
                "signed_log2fc",
                lambda values: float(np.median(np.abs(values))),
            ),
            fraction_mutations_negative=(
                "signed_log2fc",
                lambda values: float((values < 0).mean()),
            ),
            median_ref_sum=("ref_sum", "median"),
        )
        .reset_index()
    )
    summary["window"] = summary.locus_id.map(
        {PRIMARY_LOCUS: "Primary +463..+475", SECONDARY_LOCUS: "Secondary +499..+507"}
    )
    if summary.window.isna().any():
        raise ValueError("Unexpected Mdk original-model locus")
    require_finite(
        summary,
        ["median_signed_log2fc", "median_abs_log2fc", "median_ref_sum"],
        name="Mdk original Borzoi summary",
    )

    chart = summary.pivot(
        index="track_group", columns="locus_id", values="median_signed_log2fc"
    ).reset_index()
    chart.columns.name = None
    chart = chart.rename(
        columns={PRIMARY_LOCUS: "primary_effect", SECONDARY_LOCUS: "secondary_effect"}
    )
    chart["maximum_abs_effect"] = (
        chart[["primary_effect", "secondary_effect"]].abs().max(axis=1)
    )
    chart = chart.sort_values("maximum_abs_effect", ascending=False)

    manifest = pd.read_csv(run / "track_manifest.tsv", sep="\t")
    manifest["assay_class"] = _assay_class(manifest)
    compatible_manifest = manifest.loc[
        manifest.strand.fillna(".").astype(str).isin([".", GENE_STRAND])
    ]
    inventory = (
        manifest.groupby("assay_class", sort=False)
        .agg(
            selected_tracks=("track_id", "nunique"),
            track_groups=("track_group", lambda values: ", ".join(sorted(set(values)))),
        )
        .reset_index()
    )
    compatible_counts = compatible_manifest.groupby("assay_class").track_id.nunique()
    inventory["strand_compatible_tracks"] = (
        inventory.assay_class.map(compatible_counts).fillna(0).astype(int)
    )

    liver_tracks = (
        data.loc[data.track_group.eq("liver_rna")]
        .groupby(["locus_id", "track_id", "track_task_name"], sort=False)
        .agg(track_median_signed_log2fc=("log2fc_ratio_of_sums", "median"))
        .reset_index()
    )
    return summary, chart, inventory, liver_tracks


def validate_saved_original(root: Path, summary: pd.DataFrame) -> float:
    """Independently compare the recalculation with the canonical saved matrix."""

    saved = pd.read_csv(
        root / "analysis/report_data/original_group_matrix.tsv", sep="\t"
    )
    saved = saved.loc[
        saved.gene.eq("Mdk")
        & saved.model_label.eq("Borzoi original")
        & saved.readout_role.eq(READOUT_ROLE),
        [
            "locus_id",
            "track_group",
            "median_effect",
            "median_abs_effect",
            "fraction_negative",
            "median_ref_sum",
        ],
    ]
    merged = summary.merge(
        saved,
        on=["locus_id", "track_group"],
        how="outer",
        validate="one_to_one",
        indicator=True,
        suffixes=("_recalculated", "_saved"),
    )
    if not merged._merge.eq("both").all():
        raise ValueError("Original Borzoi recalculation does not match saved groups")
    pairs = (
        ("median_signed_log2fc", "median_effect"),
        ("median_abs_log2fc", "median_abs_effect"),
        ("fraction_mutations_negative", "fraction_negative"),
        ("median_ref_sum_recalculated", "median_ref_sum_saved"),
    )
    maximum = max(
        float((merged[left] - merged[right]).abs().max()) for left, right in pairs
    )
    if maximum > 1e-12:
        raise ValueError(f"Original Borzoi summary mismatch: {maximum}")
    return maximum


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    out = args.out_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    browser, windows, metrics = build_four_cell_browser(root)
    original, chart, inventory, liver_tracks = summarize_original_borzoi(root)
    maximum_difference = validate_saved_original(root, original)
    motifs = pd.read_csv(
        root / "analysis/report_data/motif_family_summary.tsv", sep="\t"
    )
    motifs = motifs.loc[
        motifs.candidate_segment_id.isin([PRIMARY_LOCUS, SECONDARY_LOCUS])
    ].copy()

    tables = {
        "four_cell_browser": browser,
        "four_cell_window_summary": windows,
        "borzoi_original_track_group_summary": original,
        "borzoi_original_track_group_chart": chart,
        "borzoi_original_track_inventory": inventory,
        "borzoi_original_liver_rna_tracks": liver_tracks,
        "motif_family_summary": motifs,
    }
    for name, table in tables.items():
        table.to_csv(out / f"{name}.tsv", sep="\t", index=False)

    validation = {
        "status": "ok",
        "analysis_contract_version": 1,
        "readout_role": READOUT_ROLE,
        "browser_rows": len(browser),
        "centers_per_cell": metrics["centers_per_cell"],
        "common_y_limit": metrics["common_y_limit"],
        "original_selected_tracks": int(inventory.selected_tracks.sum()),
        "original_strand_compatible_tracks": int(
            inventory.strand_compatible_tracks.sum()
        ),
        "original_group_rows": len(original),
        "original_summary_max_abs_difference_vs_saved": maximum_difference,
        "tables": {name: len(table) for name, table in tables.items()},
    }
    write_json(out / "validation_summary.json", validation)
    print(out / "validation_summary.json")


if __name__ == "__main__":
    main()
