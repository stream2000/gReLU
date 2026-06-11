#!/usr/bin/env python
"""Run AlphaGenome TF motif perturbation and save centered multi-resolution tracks."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from grelu.interpret.tf_context import (  # noqa: E402
    AG_INPUT_LEN,
    DEFAULT_OUTPUT_KEYS,
    FastaSequenceExtractor,
    _apply_variant_edit,
    build_output_track_table,
    filter_sites_for_context_window,
    load_tf_sites,
    load_track_metadata,
    make_motif_disruption_variants,
    save_run_config,
)


SUPPORTS_1BP = {"atac", "dnase", "procap", "cage", "rna_seq"}


def _parse_output_keys(value: str) -> tuple[str, ...]:
    if value == "default":
        return DEFAULT_OUTPUT_KEYS
    return tuple(part.strip() for part in value.split(",") if part.strip())


def _one_hot_nlc(seq: str) -> torch.Tensor:
    arr = torch.zeros((len(seq), 4), dtype=torch.float32)
    for pos, base in enumerate(seq.upper()):
        idx = {"A": 0, "C": 1, "G": 2, "T": 3}.get(base)
        if idx is not None:
            arr[pos, idx] = 1.0
    return arr.unsqueeze(0)


def _center_window_bins(n_bins: int, resolution: int, center_bp: int) -> pd.DataFrame:
    center_bin = n_bins // 2
    offsets = (np.arange(n_bins, dtype=int) - center_bin) * int(resolution)
    left_bp = -(int(center_bp) // 2)
    right_bp = left_bp + int(center_bp)
    if resolution == 1:
        keep = (offsets >= left_bp) & (offsets < right_bp)
    else:
        half_bin = float(resolution) / 2.0
        keep = (offsets + half_bin > left_bp) & (offsets - half_bin < right_bp)
    model_bins = np.flatnonzero(keep)
    return pd.DataFrame(
        {
            "window_bin_index": np.arange(len(model_bins), dtype=int),
            "model_bin_index": model_bins.astype(int),
            "offset_bp": offsets[model_bins].astype(int),
            "bin_size_bp": int(resolution),
            "center_mask_start_bp": int(left_bp),
            "center_mask_end_bp": int(right_bp),
        }
    )


def _predict_head(
    ag_model,
    seq: str,
    head: str,
    resolutions: Sequence[int],
    center_bp: int,
    device: torch.device,
) -> dict[int, np.ndarray]:
    dna = _one_hot_nlc(seq).to(device)
    organism = torch.zeros((1,), dtype=torch.long, device=device)
    outputs = ag_model.predict(
        dna,
        organism,
        heads=(head,),
        resolutions=tuple(resolutions),
        channels_last=False,
    )
    result: dict[int, np.ndarray] = {}
    for resolution, tensor in outputs[head].items():
        bins = _center_window_bins(tensor.shape[-1], int(resolution), center_bp)
        idx = torch.as_tensor(bins["model_bin_index"].to_numpy(dtype=np.int64), device=tensor.device)
        result[int(resolution)] = tensor.index_select(-1, idx)[0].detach().cpu().numpy().astype(np.float32)
    del dna, organism, outputs
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def _write_metadata(run_dir: Path, metadata: pd.DataFrame, heads: Sequence[str]) -> pd.DataFrame:
    table = build_output_track_table(metadata, output_keys=tuple(heads))
    table.to_csv(run_dir / "all_track_metadata.tsv", sep="\t", index=False)
    return table


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sites", required=True)
    parser.add_argument("--fasta", required=True)
    parser.add_argument("--genome", default="hg38")
    parser.add_argument("--metadata", default=None)
    parser.add_argument("--weights_path", required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--max_sites", type=int, default=None)
    parser.add_argument("--output_keys", default="default")
    parser.add_argument("--center_bp", type=int, default=1000)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    run_t0 = time.perf_counter()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_1bp = output_dir / "resolution_1bp"
    out_128bp = output_dir / "resolution_128bp"
    out_1bp.mkdir(parents=True, exist_ok=True)
    out_128bp.mkdir(parents=True, exist_ok=True)

    selected_heads = _parse_output_keys(args.output_keys)
    highres_heads = tuple(head for head in selected_heads if head in SUPPORTS_1BP)
    lowres_heads = selected_heads
    if not highres_heads:
        raise SystemExit("ERROR: none of the selected output heads support 1bp resolution.")

    fasta = FastaSequenceExtractor(args.fasta)
    sites = load_tf_sites(args.sites, genome=args.genome, max_sites=args.max_sites)
    sites = filter_sites_for_context_window(sites, fasta)
    if sites.empty:
        raise SystemExit("ERROR: no sites remain after full AlphaGenome context-window filtering.")
    variants = make_motif_disruption_variants(sites)
    variants.to_csv(output_dir / "variants.tsv", sep="\t", index=False)

    metadata = load_track_metadata(args.metadata) if args.metadata else load_track_metadata()
    meta_1bp = _write_metadata(out_1bp, metadata, highres_heads)
    meta_128bp = _write_metadata(out_128bp, metadata, lowres_heads)

    bins_1bp = _center_window_bins(AG_INPUT_LEN, 1, args.center_bp)
    bins_128bp = _center_window_bins(AG_INPUT_LEN // 128, 128, args.center_bp)
    bins_1bp.to_csv(out_1bp / "all_track_window_bins.tsv", sep="\t", index=False)
    bins_128bp.to_csv(out_128bp / "all_track_window_bins.tsv", sep="\t", index=False)

    for subdir, meta, bins in [(out_1bp, meta_1bp, bins_1bp), (out_128bp, meta_128bp, bins_128bp)]:
        shape = (len(variants), len(meta), len(bins))
        np.lib.format.open_memmap(subdir / "all_track_ref_window.npy", mode="w+", dtype=np.float32, shape=shape)
        np.lib.format.open_memmap(subdir / "all_track_alt_window.npy", mode="w+", dtype=np.float32, shape=shape)
        np.lib.format.open_memmap(subdir / "all_track_delta_window.npy", mode="w+", dtype=np.float32, shape=shape)
        variants.to_csv(subdir / "variants.tsv", sep="\t", index=False)

    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(args.device)

    from alphagenome_pytorch.config import DtypePolicy
    from alphagenome_pytorch.model import AlphaGenome

    ag_model = AlphaGenome.from_pretrained(
        args.weights_path,
        dtype_policy=DtypePolicy.mixed_precision(),
    ).to(device)
    ag_model.eval()

    ref_1bp = np.load(out_1bp / "all_track_ref_window.npy", mmap_mode="r+")
    alt_1bp = np.load(out_1bp / "all_track_alt_window.npy", mmap_mode="r+")
    delta_1bp = np.load(out_1bp / "all_track_delta_window.npy", mmap_mode="r+")
    ref_128 = np.load(out_128bp / "all_track_ref_window.npy", mmap_mode="r+")
    alt_128 = np.load(out_128bp / "all_track_alt_window.npy", mmap_mode="r+")
    delta_128 = np.load(out_128bp / "all_track_delta_window.npy", mmap_mode="r+")

    intervals: list[dict] = []
    timing_records: list[dict] = []
    for site_idx, row in enumerate(variants.itertuples(index=False)):
        center = int(row.position) - 1
        seq_start = center - AG_INPUT_LEN // 2
        seq_end = seq_start + AG_INPUT_LEN
        ref_seq = fasta.extract(row.chrom, seq_start, seq_end).upper()
        alt_seq, edit = _apply_variant_edit(ref_seq, seq_start, row)
        intervals.append({"site_id": row.site_id, "chrom": row.chrom, "seq_start": seq_start, "seq_end": seq_end, **edit})

        offsets_1bp = {head: i for i, head in enumerate(highres_heads)}
        offsets_128 = {head: i for i, head in enumerate(lowres_heads)}
        cursor = 0
        for head in highres_heads:
            offsets_1bp[head] = cursor
            cursor += int((meta_1bp["output_type"] == head).sum())
        cursor = 0
        for head in lowres_heads:
            offsets_128[head] = cursor
            cursor += int((meta_128bp["output_type"] == head).sum())

        for head in lowres_heads:
            head_t0 = time.perf_counter()
            resolutions = (1, 128) if head in SUPPORTS_1BP else (128,)
            ref_pred = _predict_head(ag_model, ref_seq, head, resolutions, args.center_bp, device)
            alt_pred = _predict_head(ag_model, alt_seq, head, resolutions, args.center_bp, device)

            start = offsets_128[head]
            stop = start + ref_pred[128].shape[0]
            ref_128[site_idx, start:stop, :] = ref_pred[128]
            alt_128[site_idx, start:stop, :] = alt_pred[128]
            delta_128[site_idx, start:stop, :] = alt_pred[128] - ref_pred[128]

            if 1 in ref_pred:
                start = offsets_1bp[head]
                stop = start + ref_pred[1].shape[0]
                ref_1bp[site_idx, start:stop, :] = ref_pred[1]
                alt_1bp[site_idx, start:stop, :] = alt_pred[1]
                delta_1bp[site_idx, start:stop, :] = alt_pred[1] - ref_pred[1]

            timing_records.append(
                {
                    "site_id": row.site_id,
                    "head": head,
                    "resolutions": ",".join(str(r) for r in resolutions),
                    "seconds": time.perf_counter() - head_t0,
                }
            )

    pd.DataFrame.from_records(intervals).to_csv(output_dir / "scored_intervals.tsv", sep="\t", index=False)
    with (output_dir / "timing.json").open("w") as handle:
        json.dump(
            {
                "total_process_seconds": time.perf_counter() - run_t0,
                "head_predictions": timing_records,
            },
            handle,
            indent=2,
        )
    save_run_config(
        output_dir,
        {
            **vars(args),
            "selected_heads": selected_heads,
            "highres_1bp_heads": highres_heads,
            "lowres_128bp_heads": lowres_heads,
            "n_sites": len(sites),
            "n_variants": len(variants),
            "resolution_1bp_shape": list(ref_1bp.shape),
            "resolution_128bp_shape": list(ref_128.shape),
        },
    )
    print(f"wrote 1bp raw centered outputs: {out_1bp}")
    print(f"wrote 128bp raw centered outputs: {out_128bp}")


if __name__ == "__main__":
    main()
