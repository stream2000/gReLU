#!/usr/bin/env python
"""Run Borzoi REF/ALT profiles for the single-locus MREG experiment.

This is a Borzoi-specific companion to the AlphaGenome MREG runner. It reuses
the validated MREG mutation and readout manifests, but changes model-specific
details:

- Borzoi input length: 524,288 bp.
- Borzoi output resolution: 32 bp.
- Borzoi crop: taken from the loaded model metadata.
- Tracks: exact MCF-7 Borzoi tasks when available; missing AlphaGenome-style
  tracks such as RAD21/EP300/ESR1 are not synthesized.

The script intentionally runs only experimental mutations by default and scores
one REF/ALT pair at a time because the full Borzoi output tensor is large.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

import grelu.resources
from grelu.interpret.tf_context import FastaSequenceExtractor


DEFAULT_FASTA = Path("/home/fqijun/.local/share/genomes/hg38/hg38.fa")
BORZOI_INPUT_LEN = 524_288
PSEUDOCOUNT = 1.0

MUTATION_LABELS = {
    "mreg_tss_00_ctcf_snv": "MREG TSS CTCF edit",
    "mreg_hc_dic_00_ctcf_snv": "HC-DIC CTCF edit",
    "mreg_lc_dic_00_pol2_perturbation": "LC-DIC Pol2-center edit",
}

TRACK_SPECS = [
    ("CAGE_MCF7_PLUS", "CAGE", "CNhs11943+", "CAGE", "breast carcinoma cell line:MCF7", "promoter"),
    ("CAGE_MCF7_MINUS", "CAGE", "CNhs11943-", "CAGE", "breast carcinoma cell line:MCF7", "promoter"),
    ("RNA_MCF7_PLUS", "RNA", "ENCFF795XIC+", "RNA", "MCF-7", "rna"),
    ("RNA_MCF7_MINUS", "RNA", "ENCFF795XIC-", "RNA", "MCF-7", "rna"),
    ("DNASE_MCF7", "DNASE", "ENCFF924FJR", "DNASE", "MCF-7", "accessibility"),
    ("CTCF_MCF7", "CTCF", None, "CHIP", "CTCF:MCF-7", "structural"),
    ("POLR2A_MCF7", "POLR2A", None, "CHIP", "POLR2A:MCF-7", "polymerase"),
    ("FOXA1_MCF7", "FOXA1", None, "CHIP", "FOXA1:MCF-7", "tf"),
    ("GATA3_MCF7", "GATA3", None, "CHIP", "GATA3:MCF-7", "tf"),
    ("H3K27ac_MCF7", "H3K27ac", None, "CHIP", "H3K27ac:MCF-7", "active_mark"),
    ("H3K4me1_MCF7", "H3K4me1", None, "CHIP", "H3K4me1:MCF-7", "active_mark"),
    ("H3K4me2_MCF7", "H3K4me2", None, "CHIP", "H3K4me2:MCF-7", "active_mark"),
    ("H3K4me3_MCF7", "H3K4me3", None, "CHIP", "H3K4me3:MCF-7", "active_mark"),
]


@dataclass(frozen=True)
class ModelGeometry:
    input_len: int
    bin_size: int
    crop_len_bins: int

    @property
    def output_start_offset(self) -> int:
        return self.crop_len_bins * self.bin_size


def _apply_edit(seq: str, seq_start: int, row: pd.Series) -> str:
    start = int(row["edit_start"])
    end = int(row["edit_end"])
    ref = str(row["ref_sequence"]).upper()
    alt = str(row["alt_sequence"]).upper()
    rel_start = start - seq_start
    rel_end = end - seq_start
    observed = seq[rel_start:rel_end]
    if observed != ref:
        raise ValueError(
            f"FASTA REF mismatch for {row['mutation_id']}: expected {ref}, got {observed}"
        )
    return seq[:rel_start] + alt + seq[rel_end:]


def _select_borzoi_tracks(tasks: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for rank, (track_id, target, name, assay, sample_token, group) in enumerate(TRACK_SPECS, start=1):
        if name is not None:
            sub = tasks[tasks["name"].astype(str) == name].copy()
        else:
            exact = f"{assay}:{sample_token}"
            sub = tasks[tasks["description"].astype(str) == exact].copy()
            if sub.empty:
                sub = tasks[
                    (tasks["assay"].astype(str) == assay)
                    & tasks["description"].astype(str).str.contains(sample_token, case=False, na=False)
                ].copy()
        if sub.empty:
            rows.append(
                {
                    "track_id": track_id,
                    "target_name": target,
                    "borzoi_task_index": -1,
                    "task_name": "",
                    "assay": assay,
                    "sample": sample_token,
                    "description": "",
                    "analysis_group": group,
                    "selection_priority": rank,
                    "fallback_reason": f"no Borzoi task matched {sample_token}",
                }
            )
            continue
        best = sub.iloc[0]
        fallback = ""
        if name is None and str(best["description"]) != f"{assay}:{sample_token}":
            fallback = f"used first task containing {sample_token}"
        rows.append(
            {
                "track_id": track_id,
                "target_name": target,
                "borzoi_task_index": int(best.name),
                "task_name": str(best["name"]),
                "assay": str(best["assay"]),
                "sample": str(best["sample"]),
                "description": str(best["description"]),
                "analysis_group": group,
                "selection_priority": rank,
                "fallback_reason": fallback,
            }
        )
    resolved = pd.DataFrame.from_records(rows)
    return resolved[resolved["borzoi_task_index"] >= 0].reset_index(drop=True)


def _readout_bins(
    row: pd.Series,
    seq_start: int,
    output_start: int,
    output_bins: int,
    geometry: ModelGeometry,
) -> tuple[int, int]:
    rel_start = int(row["start"]) - output_start
    rel_end = int(row["end"]) - output_start
    if rel_start < 0 or rel_end > output_bins * geometry.bin_size:
        raise ValueError(
            f"Readout {row['readout_id']} is outside Borzoi output: "
            f"{row['chrom']}:{row['start']}-{row['end']} seq_start={seq_start}"
        )
    bin_start = rel_start // geometry.bin_size
    bin_end = (rel_end + geometry.bin_size - 1) // geometry.bin_size
    return int(bin_start), int(bin_end)


def _mutation_local_readout(row: pd.Series, window_bp: int = 4_000) -> dict:
    center = (int(row["edit_start"]) + int(row["edit_end"])) // 2
    return {
        "readout_id": "mutation_local_4kb",
        "source_region_id": row["region_id"],
        "chrom": row["chrom"],
        "start": center - window_bp // 2,
        "end": center + window_bp // 2,
        "role": "mutation_local",
        "anchor_coordinate": center,
    }


def _extract_rows(
    ref_pred: np.ndarray,
    alt_pred: np.ndarray,
    mutation_row: pd.Series,
    readout_rows: Iterable[dict],
    tracks: pd.DataFrame,
    seq_start: int,
    geometry: ModelGeometry,
) -> list[dict]:
    output_start = seq_start + geometry.output_start_offset
    output_bins = ref_pred.shape[-1]
    records: list[dict] = []
    for readout in readout_rows:
        readout_series = pd.Series(readout)
        bin_start, bin_end = _readout_bins(
            readout_series, seq_start, output_start, output_bins, geometry
        )
        for _, track in tracks.iterrows():
            task_idx = int(track["borzoi_task_index"])
            ref_values = ref_pred[task_idx, bin_start:bin_end]
            alt_values = alt_pred[task_idx, bin_start:bin_end]
            for bin_idx, (ref_value, alt_value) in enumerate(zip(ref_values, alt_values), start=bin_start):
                genomic_start = output_start + bin_idx * geometry.bin_size
                records.append(
                    {
                        "mutation_id": mutation_row["mutation_id"],
                        "region_id": mutation_row["region_id"],
                        "mutation_strategy": mutation_row["mutation_strategy"],
                        "readout_id": readout["readout_id"],
                        "readout_role": readout["role"],
                        "target_name": track["target_name"],
                        "track_id": track["track_id"],
                        "analysis_group": track["analysis_group"],
                        "assay": track["assay"],
                        "task_name": track["task_name"],
                        "borzoi_task_index": task_idx,
                        "genomic_start": genomic_start,
                        "genomic_end": genomic_start + geometry.bin_size,
                        "ref_value": float(ref_value),
                        "alt_value": float(alt_value),
                        "delta": float(alt_value - ref_value),
                        "log2fc": float(np.log2((alt_value + PSEUDOCOUNT) / (ref_value + PSEUDOCOUNT))),
                    }
                )
    return records


def _summarize_profiles(profiles: pd.DataFrame) -> pd.DataFrame:
    return (
        profiles.groupby(
            [
                "mutation_id",
                "readout_id",
                "track_id",
                "target_name",
                "assay",
                "analysis_group",
                "task_name",
            ],
            as_index=False,
        )
        .agg(
            ref_mean=("ref_value", "mean"),
            alt_mean=("alt_value", "mean"),
            delta_mean=("delta", "mean"),
            log2fc_mean=("log2fc", "mean"),
            max_abs_delta=("delta", lambda x: float(np.max(np.abs(x)))),
            n_bins=("delta", "size"),
        )
        .sort_values(["mutation_id", "readout_id", "analysis_group", "target_name"])
    )


def _plot_pair(ax, panel: pd.DataFrame, target: str, marker_x: int | None, marker_label: str) -> None:
    panel = panel.sort_values("genomic_start")
    x = panel["genomic_start"].astype(float)
    x2 = panel["genomic_end"].astype(float)
    ref = panel["ref_value"].astype(float)
    alt = panel["alt_value"].astype(float)
    ymax = max(float(ref.max()), float(alt.max()), 1.0)
    gap = ymax * 0.18
    alt_offset = -(ymax + gap)
    ax.fill_between(x, 0, ref, step="post", color="#2B8CFF", edgecolor="#0B4F8A", linewidth=0.55, alpha=0.9)
    ax.fill_between(x, alt_offset, alt_offset + alt, step="post", color="#F59E0B", edgecolor="#C2410C", linewidth=0.55, alpha=0.9)
    ax.hlines(0, x.min(), x2.max(), color="#0B4F8A", linewidth=0.6)
    ax.hlines(alt_offset, x.min(), x2.max(), color="#C2410C", linewidth=0.6)
    ax.text(x.min(), ymax * 0.78, "REF", color="#0B4F8A", ha="left", va="center", fontsize=6)
    ax.text(x.min(), alt_offset + ymax * 0.78, "ALT", color="#C2410C", ha="left", va="center", fontsize=6)
    if marker_x is not None:
        ax.axvline(marker_x, color="#DC2626", linestyle="--", linewidth=0.75, alpha=0.75)
        ax.text(marker_x, 0.96, marker_label, ha="center", va="top", fontsize=6, color="#DC2626", rotation=90, transform=ax.get_xaxis_transform())
    ax.set_ylim(alt_offset - ymax * 0.08, ymax * 1.18)
    ax.set_xlim(x.min(), x2.max())
    ax.set_yticks([])
    ax.spines[["left", "right", "top"]].set_visible(False)
    ax.set_ylabel(target, fontsize=8, rotation=0, ha="right", va="center", labelpad=50)


def _plot_pdf(profiles: pd.DataFrame, mutation_manifest: pd.DataFrame, output_pdf: Path) -> None:
    page_groups = [
        ("promoter / accessibility / TF", ["CAGE_MCF7_PLUS", "CAGE_MCF7_MINUS", "RNA_MCF7_PLUS", "RNA_MCF7_MINUS", "DNASE", "CTCF", "POLR2A", "FOXA1", "GATA3"]),
        ("active chromatin marks", ["H3K27ac", "H3K4me1", "H3K4me2", "H3K4me3"]),
    ]
    with PdfPages(output_pdf) as pdf:
        for mutation_id, mut_profiles in profiles.groupby("mutation_id", sort=False):
            mut_row = mutation_manifest[mutation_manifest["mutation_id"] == mutation_id].iloc[0]
            marker_x = (int(mut_row["edit_start"]) + int(mut_row["edit_end"])) // 2
            readout_ids = ["mutation_local_4kb"]
            if mutation_id != "mreg_tss_00_ctcf_snv":
                readout_ids.append("mreg_tss_4kb")
            for readout_id in readout_ids:
                readout_profiles = mut_profiles[mut_profiles["readout_id"] == readout_id]
                if readout_profiles.empty:
                    continue
                marker_label = "edit" if readout_id == "mutation_local_4kb" else "MREG TSS"
                marker = marker_x if readout_id == "mutation_local_4kb" else 216_013_551
                for page_label, group_targets in page_groups:
                    group_profiles = readout_profiles[
                        readout_profiles["target_name"].isin(group_targets)
                        | readout_profiles["track_id"].isin(group_targets)
                    ]
                    if group_profiles.empty:
                        continue
                    row_keys = []
                    for key in group_targets:
                        if key in set(group_profiles["track_id"]):
                            row_keys.append(("track_id", key))
                        elif key in set(group_profiles["target_name"]):
                            row_keys.append(("target_name", key))
                    fig_h = max(5.6, 0.9 * len(row_keys) + 1.8)
                    fig, axes = plt.subplots(len(row_keys), 1, figsize=(12.5, fig_h), sharex=True)
                    if len(row_keys) == 1:
                        axes = [axes]
                    for ax, (column, key) in zip(axes, row_keys):
                        panel = group_profiles[group_profiles[column] == key]
                        label = key.replace("_MCF7_PLUS", "+").replace("_MCF7_MINUS", "-")
                        _plot_pair(ax, panel, label, marker, marker_label)
                    axes[-1].set_xlabel("Genomic position (hg38)", fontsize=8)
                    fig.suptitle(
                        f"Borzoi {MUTATION_LABELS.get(mutation_id, mutation_id)}: {readout_id}, {page_label}",
                        fontsize=12,
                        y=0.97,
                    )
                    fig.text(
                        0.15,
                        0.935,
                        "Borzoi raw predictions, 32 bp bins; REF above ALT with shared y-scale per track",
                        ha="left",
                        va="top",
                        fontsize=8,
                        color="#555555",
                    )
                    fig.subplots_adjust(left=0.15, right=0.99, top=0.88, bottom=0.085, hspace=0.42)
                    pdf.savefig(fig)
                    plt.close(fig)


def run(args: argparse.Namespace) -> dict[str, Path]:
    device = int(args.device) if str(args.device).isdigit() else str(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    mutation_manifest = pd.read_csv(args.mutation_manifest, sep="\t")
    region_manifest = pd.read_csv(args.region_manifest, sep="\t")
    readout_regions = pd.read_csv(args.readout_regions, sep="\t")
    readout_regions = readout_regions[
        (readout_regions["start"].astype(int) >= 0)
        & (readout_regions["end"].astype(int) > readout_regions["start"].astype(int))
    ].copy()
    mutation_manifest = mutation_manifest.merge(
        region_manifest[["region_id", "chrom"]],
        on="region_id",
        how="left",
    )
    if args.experimental_only:
        mutation_manifest = mutation_manifest[
            ~mutation_manifest["mutation_strategy"].astype(str).str.contains("control", case=False, na=False)
        ].copy()
    mutation_manifest = mutation_manifest[
        mutation_manifest["mutation_id"].isin(MUTATION_LABELS)
    ].copy()
    if mutation_manifest.empty:
        raise SystemExit("No MREG experimental mutations found")

    fasta = FastaSequenceExtractor(args.fasta)
    model = grelu.resources.load_model(repo_id=args.repo_id, filename=args.filename)
    tasks = pd.DataFrame(model.data_params["tasks"])
    tracks = _select_borzoi_tracks(tasks)
    geometry = ModelGeometry(
        input_len=int(model.data_params["train"]["seq_len"]),
        bin_size=int(model.data_params["train"]["bin_size"]),
        crop_len_bins=int(model.model_params.get("crop_len", 0)),
    )
    if geometry.input_len != BORZOI_INPUT_LEN:
        raise ValueError(f"Unexpected Borzoi input length: {geometry.input_len}")

    records: list[dict] = []
    scored_rows: list[dict] = []
    timing_rows: list[dict] = []
    for _, mut_row in mutation_manifest.iterrows():
        tic = time.perf_counter()
        center = (int(mut_row["edit_start"]) + int(mut_row["edit_end"])) // 2
        seq_start = center - geometry.input_len // 2
        seq_end = seq_start + geometry.input_len
        ref_seq = fasta.extract(mut_row["chrom"], seq_start, seq_end)
        alt_seq = _apply_edit(ref_seq, seq_start, mut_row)
        print(f"[borzoi] {mut_row['mutation_id']} {mut_row['chrom']}:{seq_start}-{seq_end}", flush=True)
        ref_pred = model.predict_on_seqs(ref_seq, device=device)[0]
        alt_pred = model.predict_on_seqs(alt_seq, device=device)[0]
        readouts = [_mutation_local_readout(mut_row)]
        readouts.extend(readout_regions.to_dict("records"))
        records.extend(
            _extract_rows(ref_pred, alt_pred, mut_row, readouts, tracks, seq_start, geometry)
        )
        timing_rows.append({"mutation_id": mut_row["mutation_id"], "seconds": time.perf_counter() - tic})
        scored_rows.append(
            {
                "mutation_id": mut_row["mutation_id"],
                "chrom": mut_row["chrom"],
                "seq_start": seq_start,
                "seq_end": seq_end,
                "output_start": seq_start + geometry.output_start_offset,
                "output_end": seq_end - geometry.output_start_offset,
                "edit_center": center,
                "variant_ref_seq": mut_row["ref_sequence"],
                "variant_alt_seq": mut_row["alt_sequence"],
            }
        )
        del ref_pred, alt_pred
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    profiles = pd.DataFrame.from_records(records)
    summary = _summarize_profiles(profiles)
    profiles_path = output_dir / "borzoi_mreg_profiles.parquet"
    summary_path = output_dir / "borzoi_mreg_effect_summary.tsv"
    tracks_path = output_dir / "resolved_borzoi_tracks.tsv"
    intervals_path = output_dir / "scored_intervals.tsv"
    timing_path = output_dir / "timing.tsv"
    pdf_path = output_dir / "borzoi_mreg_ref_alt_profiles.pdf"
    metadata_path = output_dir / "run_metadata.json"
    profiles.to_parquet(profiles_path, index=False)
    summary.to_csv(summary_path, sep="\t", index=False)
    tracks.to_csv(tracks_path, sep="\t", index=False)
    pd.DataFrame.from_records(scored_rows).to_csv(intervals_path, sep="\t", index=False)
    pd.DataFrame.from_records(timing_rows).to_csv(timing_path, sep="\t", index=False)
    _plot_pdf(profiles, mutation_manifest, pdf_path)
    metadata = {
        "model": "Borzoi",
        "repo_id": args.repo_id,
        "filename": args.filename,
        "input_len": geometry.input_len,
        "bin_size": geometry.bin_size,
        "crop_len_bins": geometry.crop_len_bins,
        "device": device,
        "experimental_only": bool(args.experimental_only),
        "n_mutations": int(len(mutation_manifest)),
        "n_tracks": int(len(tracks)),
        "outputs": {
            "profiles": str(profiles_path),
            "summary": str(summary_path),
            "tracks": str(tracks_path),
            "pdf": str(pdf_path),
        },
        "limitations": [
            "Borzoi MCF-7 RAD21, EP300 and ESR1 tracks were not available in the loaded task metadata.",
            "This run uses experimental MREG perturbations only; matched controls were not scored.",
        ],
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    return {
        "profiles": profiles_path,
        "summary": summary_path,
        "tracks": tracks_path,
        "intervals": intervals_path,
        "timing": timing_path,
        "pdf": pdf_path,
        "metadata": metadata_path,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mutation-manifest", required=True)
    parser.add_argument("--region-manifest", required=True)
    parser.add_argument("--readout-regions", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--fasta", default=str(DEFAULT_FASTA))
    parser.add_argument("--repo-id", default="Genentech/borzoi-model")
    parser.add_argument("--filename", default="human_rep0.ckpt")
    parser.add_argument("--device", default="0")
    parser.add_argument("--experimental-only", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    paths = run(args)
    for name, path in paths.items():
        print(f"[borzoi] {name}: {path}")


if __name__ == "__main__":
    main()
