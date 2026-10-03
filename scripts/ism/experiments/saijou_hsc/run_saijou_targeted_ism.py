#!/usr/bin/env python
"""Run one targeted Saijou manifest on original or fine-tuned sequence models."""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from grelu.interpret.ism.fasta import FastaReference  # noqa: E402
from grelu.interpret.ism.model_adapters import (  # noqa: E402
    AlphaGenomeFinetunedAdapter,
    AlphaGenomeOriginalAdapter,
    BorzoiFinetunedAdapter,
    BorzoiOriginalAdapter,
)
from grelu.interpret.ism.runner import (  # noqa: E402
    TargetedRunConfig,
    combine_run_artifacts,
    model_output_geometry,
    run_gene,
    select_prepared_genes,
)
from grelu.interpret.ism.validation import sha256_file  # noqa: E402


DEFAULT_PREPARED = (
    REPO_ROOT / "experiments/ism/saijou_targeted_original_comparison/prepared"
)
DEFAULT_ROOT = REPO_ROOT / "experiments/ism/saijou_targeted_original_comparison/runs"
DEFAULT_FASTA = "/work/Database/Database_fromDocker/Referencedata_mm10/genome.fa"
DEFAULT_AG_FINETUNED = (
    REPO_ROOT
    / "runs/alphagenome_split_chr10_chr11_seq524288_label196608_bin128_res128_"
    "poisson_multinomial_lora_active_h512x1/checkpoints/epochepoch=19.ckpt"
)
DEFAULT_BORZOI_FINETUNED = (
    REPO_ROOT
    / "runs/borzoi_split_chr10_chr11_poisson_multinomial_lora/checkpoints/"
    "epochepoch=39.ckpt"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-backend",
        required=True,
        choices=[
            "alphagenome_original",
            "borzoi_original",
            "alphagenome_finetuned",
            "borzoi_finetuned",
        ],
    )
    parser.add_argument("--prepared-dir", type=Path, default=DEFAULT_PREPARED)
    parser.add_argument("--fasta", default=DEFAULT_FASTA)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--pseudocount", type=float, default=1.0)
    parser.add_argument(
        "--genes",
        default=None,
        help="Optional comma-separated gene subset from the prepared manifest.",
    )
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--save-ref-profiles",
        action="store_true",
        help="Save one reference prediction array per analyzed gene.",
    )
    parser.add_argument(
        "--save-log2fc-profiles",
        action="store_true",
        help=(
            "Stream mutation x track x bin log2FC arrays. Together with the "
            "aggregated reference profile they reconstruct ALT signal."
        ),
    )
    parser.add_argument(
        "--profile-resolution-bp",
        type=int,
        default=128,
        help="Aggregate count profiles to this resolution before storage.",
    )
    parser.add_argument(
        "--profile-dtype",
        choices=["float16", "float32"],
        default="float16",
    )
    parser.add_argument("--out-dir", type=Path, default=None)
    return parser.parse_args()


def build_adapter(args: argparse.Namespace):
    """Build one of the four supported Saijou comparison adapters."""

    if args.model_backend == "alphagenome_original":
        kwargs = {"device": args.device}
        if args.checkpoint is not None:
            kwargs["weights_path"] = args.checkpoint
        return AlphaGenomeOriginalAdapter(**kwargs)
    if args.model_backend == "borzoi_original":
        kwargs = {"device": args.device}
        if args.checkpoint is not None:
            kwargs["checkpoint_path"] = args.checkpoint
        return BorzoiOriginalAdapter(**kwargs)
    if args.model_backend == "alphagenome_finetuned":
        return AlphaGenomeFinetunedAdapter(
            checkpoint_path=args.checkpoint or DEFAULT_AG_FINETUNED,
            device=args.device,
            input_length_bp=524_288,
        )
    return BorzoiFinetunedAdapter(
        checkpoint_path=args.checkpoint or DEFAULT_BORZOI_FINETUNED,
        device=args.device,
        input_length_bp=524_288,
    )


def track_manifest(adapter) -> pd.DataFrame:
    """Return a stable track table enriched with adapter TrackSpec fields."""

    specs = pd.DataFrame.from_records([asdict(spec) for spec in adapter.track_specs])
    if hasattr(adapter, "track_manifest") and not adapter.track_manifest.empty:
        panel = adapter.track_manifest.copy().reset_index(drop=True)
        specs = specs.reset_index(drop=True)
        for column in specs.columns:
            panel[column] = specs[column]
        return panel
    specs["track_group"] = specs["group"].replace("", "hsc_finetuned_10x")
    return specs


def read_inputs(
    prepared_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Read the four tables that define a prepared targeted run."""

    return tuple(
        pd.read_csv(prepared_dir / filename, sep="\t")
        for filename in (
            "genes.tsv",
            "loci.tsv",
            "readouts.tsv",
            "mutation_manifest.tsv",
        )
    )


def write_model_metadata(adapter, args: argparse.Namespace, out_dir: Path) -> None:
    """Write model and prepared-manifest provenance."""

    metadata = adapter.metadata
    checkpoint = metadata.get("checkpoint_path") or metadata.get("weights_path")
    if checkpoint and Path(str(checkpoint)).exists():
        metadata["checkpoint_sha256"] = sha256_file(Path(str(checkpoint)))
    metadata["prepared_manifest_sha256"] = sha256_file(
        args.prepared_dir / "mutation_manifest.tsv"
    )
    metadata["pseudocount"] = float(args.pseudocount)
    (out_dir / "model_metadata.json").write_text(
        json.dumps(metadata, indent=2, default=str) + "\n"
    )


def _run_config(args: argparse.Namespace) -> TargetedRunConfig:
    if args.batch_size <= 0:
        raise ValueError("batch-size must be positive")
    return TargetedRunConfig(
        model_backend=args.model_backend,
        batch_size=args.batch_size,
        pseudocount=args.pseudocount,
        progress_every=args.progress_every,
        resume=args.resume,
        save_ref_profiles=args.save_ref_profiles,
        save_log2fc_profiles=args.save_log2fc_profiles,
        profile_resolution_bp=args.profile_resolution_bp,
        profile_dtype=args.profile_dtype,
    )


def _output_directories(
    args: argparse.Namespace,
) -> tuple[Path, Path, Path]:
    out_dir = args.out_dir or (DEFAULT_ROOT / args.model_backend)
    feature_dir = out_dir / "features"
    profile_dir = out_dir / "profiles"
    feature_dir.mkdir(parents=True, exist_ok=True)
    if args.save_ref_profiles or args.save_log2fc_profiles:
        profile_dir.mkdir(parents=True, exist_ok=True)
    return out_dir, feature_dir, profile_dir


def _execute_genes(
    *,
    args: argparse.Namespace,
    config: TargetedRunConfig,
    adapter,
    genes: pd.DataFrame,
    manifest: pd.DataFrame,
    readouts: pd.DataFrame,
    feature_dir: Path,
    profile_dir: Path,
    requested_tracks: list[str],
) -> list:
    output_span, crop_bp = model_output_geometry(adapter)
    track_specs = {spec.track_id: spec for spec in adapter.track_specs}
    results = []
    with FastaReference(args.fasta) as fasta:
        for gene_row in genes.itertuples(index=False):
            results.append(
                run_gene(
                    config=config,
                    adapter=adapter,
                    fasta=fasta,
                    gene_row=gene_row,
                    manifest=manifest,
                    readouts=readouts,
                    requested_tracks=requested_tracks,
                    track_specs=track_specs,
                    feature_dir=feature_dir,
                    profile_dir=profile_dir,
                    output_span=output_span,
                    crop_bp=crop_bp,
                )
            )
    return results


def _validation_summary(
    *,
    args: argparse.Namespace,
    adapter,
    genes: pd.DataFrame,
    loci: pd.DataFrame,
    manifest: pd.DataFrame,
    readouts: pd.DataFrame,
    requested_tracks: list[str],
    results: list,
    diagnostics,
    profile_nonfinite: int,
    profile_bytes: int,
    started: float,
) -> dict[str, object]:
    deterministic_diffs = [result.deterministic_diff for result in results]
    return {
        "status": "ok",
        "model_backend": args.model_backend,
        "model_id": adapter.model_id,
        "genes": genes["gene"].tolist(),
        "loci": loci["locus_id"].tolist(),
        "mutations": int(len(manifest)),
        "selected_tracks": len(requested_tracks),
        "track_groups": sorted(
            {spec.group or "hsc_finetuned_10x" for spec in adapter.track_specs}
        ),
        "readouts": int(len(readouts)),
        "expected_feature_rows": diagnostics.expected_rows,
        "feature_rows": diagnostics.feature_rows,
        "nonfinite_core_values": diagnostics.nonfinite_core_values,
        "duplicate_feature_keys": diagnostics.duplicate_feature_keys,
        "deterministic_ref_max_abs_diff": (
            max(deterministic_diffs) if deterministic_diffs else None
        ),
        "saved_log2fc_profiles": args.save_log2fc_profiles,
        "stored_profile_resolution_bp": (
            args.profile_resolution_bp if args.save_log2fc_profiles else None
        ),
        "stored_profile_dtype": (
            args.profile_dtype if args.save_log2fc_profiles else None
        ),
        "stored_profile_bytes": profile_bytes,
        "stored_profile_nonfinite_values": profile_nonfinite,
        "elapsed_seconds": time.perf_counter() - started,
    }


def run(args: argparse.Namespace) -> dict[str, object]:
    """Execute one prepared manifest and write validated run artifacts."""

    started = time.perf_counter()
    config = _run_config(args)
    genes, loci, readouts, manifest = select_prepared_genes(
        *read_inputs(args.prepared_dir), args.genes
    )
    out_dir, feature_dir, profile_dir = _output_directories(args)

    adapter = build_adapter(args)
    print(
        f"[targeted-ism] setup backend={args.model_backend} device={args.device}",
        flush=True,
    )
    adapter.setup()
    tracks = track_manifest(adapter)
    tracks.to_csv(out_dir / "track_manifest.tsv", sep="\t", index=False)
    write_model_metadata(adapter, args, out_dir)

    requested_tracks = [spec.track_id for spec in adapter.track_specs]
    if not requested_tracks:
        raise ValueError("Adapter selected no tracks")
    results = _execute_genes(
        args=args,
        config=config,
        adapter=adapter,
        genes=genes,
        manifest=manifest,
        readouts=readouts,
        feature_dir=feature_dir,
        profile_dir=profile_dir,
        requested_tracks=requested_tracks,
    )
    diagnostics, profile_nonfinite, profile_bytes = combine_run_artifacts(
        config=config,
        genes=genes,
        readouts=readouts,
        manifest=manifest,
        results=results,
        n_tracks=len(requested_tracks),
        feature_dir=feature_dir,
        profile_dir=profile_dir,
        out_dir=out_dir,
    )
    validation = _validation_summary(
        args=args,
        adapter=adapter,
        genes=genes,
        loci=loci,
        manifest=manifest,
        readouts=readouts,
        requested_tracks=requested_tracks,
        results=results,
        diagnostics=diagnostics,
        profile_nonfinite=profile_nonfinite,
        profile_bytes=profile_bytes,
        started=started,
    )
    (out_dir / "validation_summary.json").write_text(
        json.dumps(validation, indent=2) + "\n"
    )
    print(json.dumps(validation, indent=2), flush=True)
    return validation


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
