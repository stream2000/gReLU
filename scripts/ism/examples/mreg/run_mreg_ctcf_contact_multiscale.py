#!/usr/bin/env python
"""Run and summarize AlphaGenome contact-map effects for the MREG CTCF MVP site."""

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

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from grelu.interpret.tf_context import (  # noqa: E402
    AG_INPUT_LEN,
    FastaSequenceExtractor,
    _apply_variant_edit,
    filter_sites_for_context_window,
    load_tf_sites,
    load_track_metadata,
    make_motif_disruption_variants,
    normalize_contact_maps,
)


def _one_hot_nlc(seq: str) -> torch.Tensor:
    arr = torch.zeros((len(seq), 4), dtype=torch.float32)
    for pos, base in enumerate(seq.upper()):
        idx = {"A": 0, "C": 1, "G": 2, "T": 3}.get(base)
        if idx is not None:
            arr[pos, idx] = 1.0
    return arr.unsqueeze(0)


def _parse_int_list(value: str) -> list[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def _predict_contacts(ag_model, seq: str, device: torch.device) -> np.ndarray:
    dna = _one_hot_nlc(seq).to(device)
    organism = torch.zeros((1,), dtype=torch.long, device=device)
    with torch.inference_mode():
        outputs = ag_model.predict(
            dna,
            organism,
            heads=("contact_maps",),
            channels_last=False,
        )
    out = outputs["contact_maps"].detach().cpu().numpy()
    del dna, organism, outputs
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return normalize_contact_maps(out)[0]


def _contact_bins(
    seq_start: int,
    n_bins: int,
    bin_bp: int,
    variant_center: int,
    row: pd.Series,
) -> pd.DataFrame:
    starts = seq_start + np.arange(n_bins) * bin_bp
    ends = starts + bin_bp
    centers = starts + bin_bp / 2.0
    gene_start = row.get("gene_start", np.nan)
    gene_end = row.get("gene_end", np.nan)
    in_gene = np.zeros(n_bins, dtype=bool)
    if pd.notna(gene_start) and pd.notna(gene_end):
        in_gene = (starts < int(gene_end)) & (ends > int(gene_start))
    return pd.DataFrame(
        {
            "bin_index": np.arange(n_bins, dtype=int),
            "chrom": row["chrom"],
            "start": starts.astype(int),
            "end": ends.astype(int),
            "center": centers.astype(int),
            "offset_bp": (centers - variant_center).astype(int),
            "in_gene_body": in_gene,
        }
    )


def _summarize_pair_set(
    ref_maps: np.ndarray,
    alt_maps: np.ndarray,
    left_bins: np.ndarray,
    right_bins: np.ndarray,
    max_distance_bins: int | None = None,
    min_distance_bins: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    if len(left_bins) == 0 or len(right_bins) == 0:
        n_tracks = ref_maps.shape[0]
        nan = np.full(n_tracks, np.nan, dtype=np.float32)
        return nan, nan, nan, 0

    distance = np.abs(right_bins[None, :] - left_bins[:, None])
    mask = distance >= min_distance_bins
    if max_distance_bins is not None:
        mask &= distance <= max_distance_bins
    if not np.any(mask):
        n_tracks = ref_maps.shape[0]
        nan = np.full(n_tracks, np.nan, dtype=np.float32)
        return nan, nan, nan, 0

    ref_block = ref_maps[:, left_bins[:, None], right_bins[None, :]]
    alt_block = alt_maps[:, left_bins[:, None], right_bins[None, :]]
    ref_track = ref_block[:, mask].mean(axis=1)
    alt_track = alt_block[:, mask].mean(axis=1)
    return ref_track, alt_track, alt_track - ref_track, int(mask.sum())


def _records_for_track_values(
    site_id: str,
    metric: str,
    track_values: tuple[np.ndarray, np.ndarray, np.ndarray, int],
    extra: dict,
) -> tuple[dict, list[dict]]:
    ref_track, alt_track, delta_track, n_pairs = track_values
    rec = {
        "site_id": site_id,
        "metric": metric,
        "n_pairs": n_pairs,
        "ref_mean": float(np.nanmean(ref_track)),
        "alt_mean": float(np.nanmean(alt_track)),
        "delta_mean": float(np.nanmean(delta_track)),
        "delta_abs_mean": float(np.nanmean(np.abs(delta_track))),
        "delta_min": float(np.nanmin(delta_track)),
        "delta_median": float(np.nanmedian(delta_track)),
        "delta_max": float(np.nanmax(delta_track)),
        "positive_track_fraction": float(np.nanmean(delta_track > 0)),
        **extra,
    }
    track_recs = [
        {
            "site_id": site_id,
            "metric": metric,
            "contact_track_index": int(track_idx),
            "ref_contact": float(ref_track[track_idx]),
            "alt_contact": float(alt_track[track_idx]),
            "delta_contact": float(delta_track[track_idx]),
            **extra,
        }
        for track_idx in range(len(delta_track))
    ]
    return rec, track_recs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sites", required=True)
    parser.add_argument("--fasta", required=True)
    parser.add_argument("--weights_path", required=True)
    parser.add_argument("--genome", default="hg38")
    parser.add_argument("--metadata", default=None)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--windows_bp", default="10000,20000,50000,100000,200000,500000")
    parser.add_argument("--min_distances_bp", default="2048,10000,100000")
    args = parser.parse_args()

    run_t0 = time.perf_counter()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fasta = FastaSequenceExtractor(args.fasta)
    sites = load_tf_sites(args.sites, genome=args.genome)
    sites = filter_sites_for_context_window(sites, fasta)
    if len(sites) != 1:
        raise SystemExit(f"Expected exactly one MREG site, got {len(sites)}")
    variants = make_motif_disruption_variants(sites)
    row = variants.iloc[0]
    site_row = sites.iloc[0]

    center = int(row["position"]) - 1
    seq_start = center - AG_INPUT_LEN // 2
    seq_end = seq_start + AG_INPUT_LEN
    ref_seq = fasta.extract(row["chrom"], seq_start, seq_end).upper()
    alt_seq, edit = _apply_variant_edit(ref_seq, seq_start, row)
    scored_interval = {"site_id": row["site_id"], "chrom": row["chrom"], "seq_start": seq_start, "seq_end": seq_end, **edit}

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

    predict_t0 = time.perf_counter()
    ref_maps = _predict_contacts(ag_model, ref_seq, device)
    alt_maps = _predict_contacts(ag_model, alt_seq, device)
    predict_seconds = time.perf_counter() - predict_t0
    delta_maps = alt_maps - ref_maps

    np.save(output_dir / "contact_ref_maps.npy", ref_maps.astype(np.float32))
    np.save(output_dir / "contact_alt_maps.npy", alt_maps.astype(np.float32))
    np.save(output_dir / "contact_delta_maps.npy", delta_maps.astype(np.float32))

    n_tracks, n_bins, _ = ref_maps.shape
    bin_bp = int(round(AG_INPUT_LEN / n_bins))
    variant_bin = int(math.floor((center - seq_start) / bin_bp))
    bins = _contact_bins(seq_start, n_bins, bin_bp, center, site_row)
    bins.to_csv(output_dir / "contact_bins.tsv", sep="\t", index=False)

    metadata = load_track_metadata(args.metadata) if args.metadata else load_track_metadata()
    contact_meta = metadata[metadata["output_type"] == "contact_maps"].copy()
    if len(contact_meta) == n_tracks:
        contact_meta.to_csv(output_dir / "contact_track_metadata.tsv", sep="\t", index=False)
    else:
        pd.DataFrame({"contact_track_index": np.arange(n_tracks, dtype=int)}).to_csv(
            output_dir / "contact_track_metadata.tsv", sep="\t", index=False
        )

    windows_bp = _parse_int_list(args.windows_bp)
    min_distances_bp = _parse_int_list(args.min_distances_bp)
    site_records: list[dict] = []
    track_records: list[dict] = []
    site_id = str(row["site_id"])

    for window_bp in windows_bp:
        window_bins = max(1, int(round(window_bp / bin_bp)))
        left = np.arange(max(0, variant_bin - window_bins), variant_bin, dtype=int)
        right = np.arange(variant_bin, min(n_bins, variant_bin + window_bins), dtype=int)
        for min_distance_bp in min_distances_bp:
            min_distance_bins = max(1, int(math.ceil(min_distance_bp / bin_bp)))
            if min_distance_bins > window_bins:
                continue
            values = _summarize_pair_set(
                ref_maps,
                alt_maps,
                left,
                right,
                max_distance_bins=window_bins,
                min_distance_bins=min_distance_bins,
            )
            rec, tracks = _records_for_track_values(
                site_id,
                "motif_cross_boundary_multiscale",
                values,
                {
                    "window_bp": window_bp,
                    "min_distance_bp": min_distance_bp,
                    "window_bins": window_bins,
                    "min_distance_bins": min_distance_bins,
                },
            )
            site_records.append(rec)
            track_records.extend(tracks)

    gene_start = site_row.get("gene_start", np.nan)
    gene_end = site_row.get("gene_end", np.nan)
    if pd.notna(gene_start) and pd.notna(gene_end):
        gene_bins = bins.index[bins["in_gene_body"].to_numpy()].to_numpy(dtype=int)
        left_gene = gene_bins[gene_bins < variant_bin]
        right_gene = gene_bins[gene_bins >= variant_bin]
        motif_anchor = np.array([variant_bin], dtype=int)
        gene_without_anchor = gene_bins[gene_bins != variant_bin]
        gene_metrics = [
            ("gene_body_left_vs_right_around_motif", left_gene, right_gene),
            ("motif_anchor_to_gene_body", motif_anchor, gene_without_anchor),
            ("motif_anchor_to_left_gene_body", motif_anchor, left_gene),
            ("motif_anchor_to_right_gene_body", motif_anchor, right_gene[right_gene != variant_bin]),
        ]
        for metric, left, right in gene_metrics:
            values = _summarize_pair_set(ref_maps, alt_maps, left, right, min_distance_bins=1)
            rec, tracks = _records_for_track_values(
                site_id,
                metric,
                values,
                {
                    "window_bp": np.nan,
                    "min_distance_bp": bin_bp,
                    "window_bins": np.nan,
                    "min_distance_bins": 1,
                    "gene_start": int(gene_start),
                    "gene_end": int(gene_end),
                },
            )
            site_records.append(rec)
            track_records.extend(tracks)

    site_summary = pd.DataFrame.from_records(site_records)
    track_summary = pd.DataFrame.from_records(track_records)
    site_summary.to_csv(output_dir / "contact_multiscale_site_summary.tsv", sep="\t", index=False)
    track_summary.to_csv(output_dir / "contact_multiscale_track_summary.tsv", sep="\t", index=False)
    variants.to_csv(output_dir / "variants.tsv", sep="\t", index=False)
    pd.DataFrame([scored_interval]).to_csv(output_dir / "scored_intervals.tsv", sep="\t", index=False)

    with (output_dir / "README.md").open("w") as handle:
        handle.write(
            "\n".join(
                [
                    "# MREG CTCF contact-map multiscale outputs",
                    "",
                    "AlphaGenome contact maps are 2048 bp/bin for the 1,048,576 bp input.",
                    "Therefore a 1 kb center mask is below 2D output resolution; this run saves",
                    "the full 512 x 512 contact maps and summarizes biologically relevant masks.",
                    "",
                    "Positive `delta_mean` means ALT has higher predicted contact than REF for",
                    "the selected bin pairs. For cross-motif masks, this is compatible with",
                    "weaker local insulation.",
                    "",
                    "Files:",
                    "- `contact_ref_maps.npy`, `contact_alt_maps.npy`, `contact_delta_maps.npy`: tracks x bins x bins.",
                    "- `contact_bins.tsv`: genomic coordinate and variant-relative offset per 2048 bp contact bin.",
                    "- `contact_multiscale_site_summary.tsv`: site-level summaries across masks.",
                    "- `contact_multiscale_track_summary.tsv`: per-contact-track summaries across masks.",
                ]
            )
            + "\n"
        )

    with (output_dir / "run_config.json").open("w") as handle:
        json.dump(
            {
                **vars(args),
                "n_sites": int(len(sites)),
                "n_variants": int(len(variants)),
                "contact_map_shape": list(ref_maps.shape),
                "contact_bin_bp": bin_bp,
                "variant_bin": variant_bin,
                "predict_seconds": predict_seconds,
                "total_process_seconds": time.perf_counter() - run_t0,
            },
            handle,
            indent=2,
        )

    print(f"wrote contact multiscale outputs: {output_dir}")
    print(f"contact_map_shape={ref_maps.shape} contact_bin_bp={bin_bp} variant_bin={variant_bin}")


if __name__ == "__main__":
    main()
