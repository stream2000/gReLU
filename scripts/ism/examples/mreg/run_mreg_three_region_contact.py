#!/usr/bin/env python
"""Run contact-map summaries for the corrected MREG three-region ISM run."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from grelu.interpret.tf_context import (  # noqa: E402
    AG_INPUT_LEN,
    FastaSequenceExtractor,
    _apply_variant_edit,
    load_track_metadata,
    normalize_contact_maps,
)

DEFAULT_RUN_DIR = REPO_ROOT / "agent-doc/ism_context/20260611_1145_mreg_tss_hc_lc_dic_corrected"
DEFAULT_FASTA = Path("/home/fqijun/.local/share/genomes/hg38/hg38.fa")
DEFAULT_WEIGHTS = Path("/home/fqijun/.cache/alphagenome/model_fold_0.safetensors")
CONTACT_HEAD = "contact_maps"
CONTROL_STRATEGIES = {"local_non_motif_control", "non_peak_local_control"}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", default=str(DEFAULT_RUN_DIR))
    parser.add_argument(
        "--output-dir",
        default=str(REPO_ROOT / "agent-doc/ism_context/20260617_mreg_three_region_contact"),
    )
    parser.add_argument("--fasta", default=str(DEFAULT_FASTA))
    parser.add_argument("--weights-path", default=str(DEFAULT_WEIGHTS))
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument(
        "--save-contact-maps",
        action="store_true",
        help="Persist per-mutation REF, ALT and ALT-REF contact maps as .npy files.",
    )
    return parser.parse_args()


def _one_hot_nlc(seq: str) -> torch.Tensor:
    arr = torch.zeros((len(seq), 4), dtype=torch.float32)
    for pos, base in enumerate(seq.upper()):
        idx = {"A": 0, "C": 1, "G": 2, "T": 3}.get(base)
        if idx is not None:
            arr[pos, idx] = 1.0
    return arr.unsqueeze(0)


def _predict_contacts(model, seq: str, device: torch.device) -> np.ndarray:
    dna = _one_hot_nlc(seq).to(device)
    organism = torch.zeros((1,), dtype=torch.long, device=device)
    with torch.inference_mode():
        out = model.predict(dna, organism, heads=(CONTACT_HEAD,), channels_last=False)
    contacts = normalize_contact_maps(out[CONTACT_HEAD].detach().cpu().numpy())[0]
    del dna, organism, out
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return contacts.astype(np.float32)


def _shared_context_centers(mutations: pd.DataFrame) -> dict[str, int]:
    centers: dict[str, int] = {}
    for _, group in mutations.groupby("region_id", sort=False):
        exp = group[~group["mutation_strategy"].isin(CONTROL_STRATEGIES)]
        anchor_row = exp.iloc[0] if not exp.empty else group.iloc[0]
        center = (int(anchor_row["edit_start"]) + int(anchor_row["edit_end"])) // 2
        for mutation_id in group["mutation_id"].astype(str):
            centers[mutation_id] = center
    return centers


def _region_bins(readouts: pd.DataFrame, seq_start: int, bin_bp: int) -> dict[str, np.ndarray]:
    bins: dict[str, np.ndarray] = {}
    for _, row in readouts.iterrows():
        readout_id = str(row["readout_id"])
        if int(row["start"]) < 0:
            continue
        start = int(math.floor((int(row["start"]) - seq_start) / bin_bp))
        end = int(math.ceil((int(row["end"]) - seq_start) / bin_bp))
        bins[readout_id] = np.arange(max(0, start), min(AG_INPUT_LEN // bin_bp, end), dtype=int)
    return bins


def _mean_pair_delta(delta_maps: np.ndarray, left: np.ndarray, right: np.ndarray) -> tuple[float, float, float, float, int]:
    if len(left) == 0 or len(right) == 0:
        return (float("nan"), float("nan"), float("nan"), float("nan"), 0)
    pairs = delta_maps[:, left[:, None], right[None, :]]
    track_values = pairs.reshape(delta_maps.shape[0], -1).mean(axis=1)
    return (
        float(np.nanmean(track_values)),
        float(np.nanmedian(track_values)),
        float(np.nanmin(track_values)),
        float(np.nanmax(track_values)),
        int(pairs.shape[1] * pairs.shape[2]),
    )


def _local_cross_bins(edit_bin: int, n_bins: int, bin_bp: int, window_bp: int, min_distance_bp: int) -> tuple[np.ndarray, np.ndarray]:
    window_bins = max(1, int(round(window_bp / bin_bp)))
    min_bins = max(1, int(math.ceil(min_distance_bp / bin_bp)))
    left = np.arange(max(0, edit_bin - window_bins), max(0, edit_bin - min_bins + 1), dtype=int)
    right = np.arange(min(n_bins, edit_bin + min_bins), min(n_bins, edit_bin + window_bins + 1), dtype=int)
    return left, right


def _build_variant_row(row: pd.Series, chrom: str, edit_center: int):
    class VariantRow:
        pass

    v = VariantRow()
    v.site_id = str(row["mutation_id"])
    v.chrom = chrom
    v.position = edit_center + 1
    v.ref = str(row["ref_sequence"])[0]
    v.alt = str(row["alt_sequence"])[0]
    v.variant_start = int(row["edit_start"])
    v.variant_end = int(row["edit_end"])
    v.variant_ref_seq = str(row["ref_sequence"])
    v.variant_alt_seq = str(row["alt_sequence"])
    return v


def _metric_records(
    mutation_id: str,
    region_id: str,
    control_type: str,
    delta_maps: np.ndarray,
    edit_bin: int,
    region_bin_map: dict[str, np.ndarray],
    bin_bp: int,
) -> list[dict]:
    records: list[dict] = []
    n_bins = delta_maps.shape[1]
    for window_bp, min_distance_bp in [(20_000, 2_048), (100_000, 10_000), (500_000, 100_000)]:
        left, right = _local_cross_bins(edit_bin, n_bins, bin_bp, window_bp, min_distance_bp)
        mean, median, minv, maxv, n_pairs = _mean_pair_delta(delta_maps, left, right)
        records.append(
            {
                "mutation_id": mutation_id,
                "region_id": region_id,
                "control_type": control_type,
                "metric": "local_cross_boundary",
                "target_region": "edit_left_vs_right",
                "window_bp": window_bp,
                "min_distance_bp": min_distance_bp,
                "delta_contact_mean": mean,
                "delta_contact_median": median,
                "delta_contact_min_track": minv,
                "delta_contact_max_track": maxv,
                "n_bin_pairs": n_pairs,
            }
        )
    anchor = np.array([edit_bin], dtype=int)
    for readout_id, bins in region_bin_map.items():
        if len(bins) == 0:
            continue
        mean, median, minv, maxv, n_pairs = _mean_pair_delta(delta_maps, anchor, bins)
        records.append(
            {
                "mutation_id": mutation_id,
                "region_id": region_id,
                "control_type": control_type,
                "metric": "edit_anchor_to_readout",
                "target_region": readout_id,
                "window_bp": np.nan,
                "min_distance_bp": bin_bp,
                "delta_contact_mean": mean,
                "delta_contact_median": median,
                "delta_contact_min_track": minv,
                "delta_contact_max_track": maxv,
                "n_bin_pairs": n_pairs,
            }
        )
    return records


def _adjusted_summary(metric_df: pd.DataFrame, mutations: pd.DataFrame) -> pd.DataFrame:
    records: list[dict] = []
    exp_ids = mutations[~mutations["mutation_strategy"].isin(CONTROL_STRATEGIES)]["mutation_id"].astype(str)
    for mutation_id in exp_ids:
        prefix = str(mutations.loc[mutations["mutation_id"].eq(mutation_id), "matched_control_id"].iloc[0])
        controls = mutations[mutations["mutation_id"].astype(str).str.startswith(prefix)]["mutation_id"].astype(str).tolist()
        target = metric_df[metric_df["mutation_id"].eq(mutation_id)]
        control = metric_df[metric_df["mutation_id"].isin(controls)]
        for _, row in target.iterrows():
            keys = {
                "metric": row["metric"],
                "target_region": row["target_region"],
                "window_bp": row["window_bp"],
                "min_distance_bp": row["min_distance_bp"],
            }
            mask = control["metric"].eq(keys["metric"]) & control["target_region"].eq(keys["target_region"])
            if pd.notna(keys["window_bp"]):
                mask &= control["window_bp"].eq(keys["window_bp"])
            else:
                mask &= control["window_bp"].isna()
            mask &= control["min_distance_bp"].eq(keys["min_distance_bp"])
            control_mean = float(control.loc[mask, "delta_contact_mean"].mean()) if mask.any() else float("nan")
            records.append(
                {
                    "mutation_id": mutation_id,
                    "region_id": row["region_id"],
                    **keys,
                    "target_delta_contact_mean": float(row["delta_contact_mean"]),
                    "control_delta_contact_mean": control_mean,
                    "adjusted_delta_contact_mean": float(row["delta_contact_mean"]) - control_mean,
                    "n_controls": int(mask.sum()),
                    "n_bin_pairs": int(row["n_bin_pairs"]),
                }
            )
    return pd.DataFrame.from_records(records)


def main() -> None:
    args = _parse_args()
    run_t0 = time.perf_counter()
    run_dir = Path(args.run_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fasta = FastaSequenceExtractor(str(args.fasta))
    mutations = pd.read_csv(run_dir / "prepared/mreg_mutation_manifest.tsv", sep="\t")
    readouts = pd.read_csv(run_dir / "prepared/mreg_readout_regions.tsv", sep="\t")
    region_manifest = pd.read_csv(run_dir / "prepared/mreg_region_manifest.tsv", sep="\t")
    region_chrom = dict(zip(region_manifest["region_id"], region_manifest["chrom"]))
    context_centers = _shared_context_centers(mutations)

    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(args.device)
    from alphagenome_pytorch.config import DtypePolicy
    from alphagenome_pytorch.model import AlphaGenome

    model = AlphaGenome.from_pretrained(
        str(args.weights_path),
        dtype_policy=DtypePolicy.mixed_precision(),
    ).to(device)
    model.eval()

    all_records: list[dict] = []
    interval_records: list[dict] = []
    timing_records: list[dict] = []
    raw_map_dir = output_dir / "contact_maps"
    if args.save_contact_maps:
        raw_map_dir.mkdir(exist_ok=True)

    n_bins = None
    bin_bp = None
    for idx, row in mutations.reset_index(drop=True).iterrows():
        t0 = time.perf_counter()
        mutation_id = str(row["mutation_id"])
        region_id = str(row["region_id"])
        chrom = region_chrom.get(region_id, "chr2")
        edit_center = (int(row["edit_start"]) + int(row["edit_end"])) // 2
        context_center = context_centers[mutation_id]
        seq_start = context_center - AG_INPUT_LEN // 2
        seq_end = seq_start + AG_INPUT_LEN
        ref_seq = fasta.extract(chrom, seq_start, seq_end).upper()
        alt_seq, edit = _apply_variant_edit(ref_seq, seq_start, _build_variant_row(row, chrom, edit_center))
        ref_maps = _predict_contacts(model, ref_seq, device)
        alt_maps = _predict_contacts(model, alt_seq, device)
        delta_maps = alt_maps - ref_maps
        n_bins = int(delta_maps.shape[1])
        bin_bp = int(round(AG_INPUT_LEN / n_bins))
        edit_bin = int(math.floor((edit_center - seq_start) / bin_bp))
        region_bin_map = _region_bins(readouts, seq_start, bin_bp)
        control_type = "matched_control" if str(row["mutation_strategy"]) in CONTROL_STRATEGIES else "experimental"
        all_records.extend(
            _metric_records(mutation_id, region_id, control_type, delta_maps, edit_bin, region_bin_map, bin_bp)
        )
        if args.save_contact_maps:
            np.save(raw_map_dir / f"{mutation_id}.ref_contact_maps.npy", ref_maps.astype(np.float32))
            np.save(raw_map_dir / f"{mutation_id}.alt_contact_maps.npy", alt_maps.astype(np.float32))
            np.save(raw_map_dir / f"{mutation_id}.delta_contact_maps.npy", delta_maps.astype(np.float32))
        interval_records.append(
            {
                "mutation_id": mutation_id,
                "region_id": region_id,
                "chrom": chrom,
                "seq_start": seq_start,
                "seq_end": seq_end,
                "edit_center": edit_center,
                "context_center": context_center,
                "edit_bin": edit_bin,
                "contact_bin_bp": bin_bp,
                **edit,
            }
        )
        elapsed = time.perf_counter() - t0
        timing_records.append({"mutation_id": mutation_id, "seconds": elapsed})
        print(f"[contact] {idx + 1}/{len(mutations)} {mutation_id} {elapsed:.1f}s")

    metric_df = pd.DataFrame.from_records(all_records)
    adjusted = _adjusted_summary(metric_df, mutations)
    metric_df.to_csv(output_dir / "contact_metric_by_mutation.tsv", sep="\t", index=False)
    adjusted.to_csv(output_dir / "contact_metric_adjusted.tsv", sep="\t", index=False)
    pd.DataFrame(interval_records).to_csv(output_dir / "contact_scored_intervals.tsv", sep="\t", index=False)
    pd.DataFrame(timing_records).to_csv(output_dir / "timing.tsv", sep="\t", index=False)
    metadata = load_track_metadata()
    contact_meta = metadata[metadata["output_type"].eq(CONTACT_HEAD)].copy()
    contact_meta.to_csv(output_dir / "contact_track_metadata.tsv", sep="\t", index=False)
    (output_dir / "run_config.json").write_text(
        json.dumps(
            {
                **vars(args),
                "n_mutations": int(len(mutations)),
                "contact_map_shape": [int(len(contact_meta)), int(n_bins), int(n_bins)] if n_bins is not None else None,
                "contact_bin_bp": bin_bp,
                "total_process_seconds": time.perf_counter() - run_t0,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(output_dir / "contact_metric_adjusted.tsv")


if __name__ == "__main__":
    main()
