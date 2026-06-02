#!/usr/bin/env python
"""Run the TF motif-disruption context pilot pipeline."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from grelu.interpret.tf_context import (  # noqa: E402
    DEFAULT_OUTPUT_KEYS,
    FastaSequenceExtractor,
    annotate_context_clusters,
    build_alphagenome_model,
    build_output_track_table,
    build_track_groups,
    cluster_contexts,
    contact_motif_vs_control_delta,
    contact_perturbations_vs_control_delta,
    fit_context_embedding,
    filter_sites_for_context_window,
    load_tf_sites,
    load_track_metadata,
    make_motif_disruption_variants,
    save_run_config,
    score_variant_contact_boundary_strength,
    score_variant_features_chunked,
    score_variant_track_window,
)


def _parse_output_keys(value: str) -> tuple[str, ...]:
    if value == "default":
        return DEFAULT_OUTPUT_KEYS
    return tuple(part.strip() for part in value.split(",") if part.strip())


def _parse_devices(value: str) -> str | list[int]:
    if value == "cpu":
        return "cpu"
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def _validate_multigpu(devices: str | list[int]) -> None:
    if devices == "cpu":
        raise SystemExit("ERROR: this pipeline is configured for GPU inference; --devices cannot be 'cpu'.")
    if not torch.cuda.is_available():
        raise SystemExit("ERROR: CUDA is not available; cannot run required GPU inference.")
    n_gpu = torch.cuda.device_count()
    missing = [device for device in devices if device >= n_gpu]
    if missing:
        raise SystemExit(f"ERROR: requested GPU IDs {missing}, but only {n_gpu} CUDA device(s) are visible.")


def main() -> None:
    run_t0 = time.perf_counter()
    timings: dict[str, float | list[dict]] = {}
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sites", required=True, help="BED-like TF binding/motif sites")
    parser.add_argument("--fasta", required=True, help="Reference genome FASTA")
    parser.add_argument("--tf", default="CTCF", help="TF name for track grouping")
    parser.add_argument("--genome", default="hg38")
    parser.add_argument("--metadata", default=None, help="AlphaGenome track metadata parquet")
    parser.add_argument("--weights_path", default=None, help="AlphaGenome weights path")
    parser.add_argument("--devices", default="0,1,2,3", help="Comma-separated GPU IDs or 'cpu'")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--precision", default=None, help="Lightning precision, e.g. bf16-mixed")
    parser.add_argument("--compile", action="store_true", help="Enable torch.compile(max-autotune)")
    parser.add_argument("--max_sites", type=int, default=None)
    parser.add_argument("--n_components", type=int, default=8)
    parser.add_argument("--n_clusters", type=int, default=6)
    parser.add_argument("--output_keys", default="default", help="Comma-separated AG heads or 'default'")
    parser.add_argument(
        "--feature_mode",
        choices=["track_window", "grouped", "contact_boundary"],
        default="track_window",
        help=(
            "track_window saves all 1D tracks in a local window; grouped is legacy "
            "lossy aggregation; contact_boundary scores contact-map boundary strength"
        ),
    )
    parser.add_argument("--track_window_bp", type=int, default=100_000, help="Half-window around variant center")
    parser.add_argument(
        "--track_predict_chunk_size",
        type=int,
        default=None,
        help="Number of variants per track-window predict chunk; lower this to reduce memory peak",
    )
    parser.add_argument(
        "--no_save_delta_window",
        action="store_true",
        help="In track_window mode, save ref/alt windows but skip all_track_delta_window.npy",
    )
    parser.add_argument("--contact_window_bp", type=int, default=500_000, help="Cross-boundary contact distance")
    parser.add_argument(
        "--contact_min_distance_bp",
        type=int,
        default=100_000,
        help="Minimum pair distance included in contact-boundary score",
    )
    parser.add_argument(
        "--output_dir",
        default="agent-doc/ism_context/ctcf_pilot",
        help="Directory for variants/features/clusters",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_keys = _parse_output_keys(args.output_keys)
    devices = _parse_devices(args.devices)
    stage_t0 = time.perf_counter()
    _validate_multigpu(devices)
    sites = load_tf_sites(args.sites, genome=args.genome, max_sites=args.max_sites)
    fasta = FastaSequenceExtractor(args.fasta)
    sites = filter_sites_for_context_window(sites, fasta)
    if sites.empty:
        raise SystemExit("ERROR: no sites remain after full AlphaGenome context-window filtering.")
    variants = make_motif_disruption_variants(sites)
    variants.to_csv(output_dir / "variants.tsv", sep="\t", index=False)
    timings["load_sites_and_variants_seconds"] = time.perf_counter() - stage_t0

    stage_t0 = time.perf_counter()
    if args.feature_mode == "contact_boundary":
        metadata = None
        track_groups = {}
        pd.DataFrame({"contact_track_index": list(range(28))}).to_csv(
            output_dir / "contact_track_metadata.tsv", sep="\t", index=False
        )
    else:
        metadata = load_track_metadata(args.metadata) if args.metadata else load_track_metadata()
        track_groups = build_track_groups(metadata, output_keys=output_keys, tf=args.tf)
        track_table = build_output_track_table(metadata, output_keys=output_keys)
        track_table.to_csv(output_dir / "all_track_metadata.tsv", sep="\t", index=False)
    timings["load_metadata_and_tracks_seconds"] = time.perf_counter() - stage_t0

    stage_t0 = time.perf_counter()
    model_output_key = "contact_maps" if args.feature_mode == "contact_boundary" else output_keys
    model = build_alphagenome_model(output_key=model_output_key, weights_path=args.weights_path)
    timings["build_model_seconds"] = time.perf_counter() - stage_t0
    if args.compile:
        stage_t0 = time.perf_counter()
        print("Applying torch.compile(max-autotune) ...")
        model.model = torch.compile(model.model, mode="max-autotune")
        print("  (first forward pass will be slow due to autotuning)")
        timings["compile_wrap_seconds"] = time.perf_counter() - stage_t0

    chunk_timings: list[dict] = []
    stage_t0 = time.perf_counter()
    if args.feature_mode == "track_window":
        features, intervals = score_variant_track_window(
            variants,
            model=model,
            fasta_extractor=fasta,
            output_dir=output_dir,
            devices=devices,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            precision=args.precision,
            window_bp=args.track_window_bp,
            save_delta_window=not args.no_save_delta_window,
            predict_chunk_size=args.track_predict_chunk_size,
            timing_records=chunk_timings,
        )
        track_features = None
    elif args.feature_mode == "contact_boundary":
        features, track_features, intervals = score_variant_contact_boundary_strength(
            variants,
            sites=sites,
            model=model,
            fasta_extractor=fasta,
            devices=devices,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            precision=args.precision,
            window_bp=args.contact_window_bp,
            min_distance_bp=args.contact_min_distance_bp,
            timing_records=chunk_timings,
        )
    else:
        features, intervals = score_variant_features_chunked(
            variants,
            model=model,
            fasta_extractor=fasta,
            track_groups=track_groups,
            devices=devices,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            precision=args.precision,
            timing_records=chunk_timings,
        )
        track_features = None
    timings["score_and_aggregate_seconds"] = time.perf_counter() - stage_t0
    timings["chunks"] = chunk_timings
    intervals.to_csv(output_dir / "scored_intervals.tsv", sep="\t", index=False)

    if args.feature_mode == "track_window":
        features.to_parquet(output_dir / "all_track_site_delta_summary.parquet", index=False)
        features.to_csv(output_dir / "all_track_site_delta_summary.tsv", sep="\t", index=False)
    elif args.feature_mode == "contact_boundary":
        features.to_parquet(output_dir / "contact_boundary_strength_site_summary.parquet", index=False)
        features.to_csv(output_dir / "contact_boundary_strength_site_summary.tsv", sep="\t", index=False)
        if track_features is not None:
            track_features.to_csv(
                output_dir / "contact_boundary_strength_track_summary.tsv",
                sep="\t",
                index=False,
            )
        class_summary = features.groupby(["dic_class", "control_type"], dropna=False).mean(numeric_only=True)
        class_summary.to_csv(output_dir / "contact_boundary_strength_class_summary.tsv", sep="\t")
        pair_delta = contact_motif_vs_control_delta(features)
        if not pair_delta.empty:
            pair_delta.to_csv(
                output_dir / "contact_boundary_strength_motif_minus_control.tsv",
                sep="\t",
                index=False,
            )
        all_pair_delta = contact_perturbations_vs_control_delta(features)
        if not all_pair_delta.empty:
            all_pair_delta.to_csv(
                output_dir / "contact_boundary_strength_perturbation_minus_control.tsv",
                sep="\t",
                index=False,
            )
    else:
        features.to_parquet(output_dir / "delta_features.parquet", index=False)
        features.to_csv(output_dir / "delta_features.tsv", sep="\t", index=False)

        stage_t0 = time.perf_counter()
        embedding, _ = fit_context_embedding(features, n_components=args.n_components)
        embedding.to_csv(output_dir / "site_embedding.tsv", sep="\t", index=False)
        timings["embedding_seconds"] = time.perf_counter() - stage_t0

        stage_t0 = time.perf_counter()
        site_clusters, _ = cluster_contexts(embedding, n_clusters=args.n_clusters)
        site_clusters.to_csv(output_dir / "site_clusters.tsv", sep="\t", index=False)

        summary = annotate_context_clusters(features, site_clusters)
        summary.to_csv(output_dir / "cluster_summary.tsv", sep="\t", index=False)

    if track_groups:
        pd.Series({name: len(indices) for name, indices in track_groups.items()}).to_csv(
            output_dir / "track_group_counts.tsv", sep="\t", header=["n_tracks"]
        )
    timings["cluster_and_summary_seconds"] = time.perf_counter() - stage_t0
    timings["total_process_seconds"] = time.perf_counter() - run_t0
    with (output_dir / "timing.json").open("w") as handle:
        json.dump(timings, handle, indent=2)
    save_run_config(
        output_dir,
        {
            **vars(args),
            "parsed_devices": devices,
            "output_keys": output_keys,
            "model_output_key": model_output_key,
            "n_sites": len(sites),
            "n_variants": len(variants),
        },
    )


if __name__ == "__main__":
    main()
