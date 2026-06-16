#!/usr/bin/env python
"""Run resumable targeted batch ISM and save population-scale 1D features."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from grelu.interpret.ism.batch import (  # noqa: E402
    AG_INPUT_LEN,
    SUMMARY_STRATEGIES,
    adjust_population_features,
    load_strategy_modules,
    resolve_track_registry,
)
from grelu.interpret.ism.tracks import load_track_metadata  # noqa: E402
from grelu.interpret.tf_context import (  # noqa: E402
    FastaSequenceExtractor,
    _apply_variant_edit,
)

RESOLUTION = 128
def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-dir", required=True)
    parser.add_argument("--fasta", required=True)
    parser.add_argument("--weights-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--track-metadata")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--max-sites", type=int)
    parser.add_argument("--mutation-id")
    parser.add_argument(
        "--site-class",
        action="append",
        default=[],
        help="Run only these site_class values; repeat as needed",
    )
    parser.add_argument(
        "--readout-table",
        help="Optional site-specific genomic readouts TSV",
    )
    parser.add_argument("--site-shard-index", type=int, default=0)
    parser.add_argument("--site-shard-count", type=int, default=1)
    parser.add_argument(
        "--windows-bp",
        default="1024,4096,20000,100000",
        help="Comma-separated centered summary windows",
    )
    parser.add_argument(
        "--summary-strategy",
        default="multiscale_log2fc",
    )
    parser.add_argument(
        "--strategy-module",
        action="append",
        default=[],
        help="Import a Python module that registers extra summary strategies",
    )
    return parser.parse_args()


def _one_hot(sequence: str, device: torch.device) -> torch.Tensor:
    encoded = np.frombuffer(sequence.upper().encode("ascii"), dtype=np.uint8)
    lookup = np.full(256, -1, dtype=np.int16)
    for index, base in enumerate(b"ACGT"):
        lookup[base] = index
    indices = lookup[encoded]
    array = np.zeros((len(sequence), 4), dtype=np.float32)
    valid = indices >= 0
    array[np.flatnonzero(valid), indices[valid]] = 1.0
    return torch.from_numpy(array).unsqueeze(0).to(device)


def _context_centers(mutations: pd.DataFrame) -> dict[str, int]:
    centers = {}
    for target_id, group in mutations.groupby("matched_target_id", sort=False):
        target = group[group["mutation_id"].eq(target_id)]
        if len(target) != 1:
            raise ValueError(f"Expected one target mutation for {target_id}")
        row = target.iloc[0]
        center = (int(row["edit_start"]) + int(row["edit_end"])) // 2
        centers.update({str(value): center for value in group["mutation_id"]})
    return centers


def _apply_edit(
    reference: str,
    sequence_start: int,
    row: pd.Series,
) -> str:
    class Variant:
        pass

    variant = Variant()
    variant.site_id = str(row["mutation_id"])
    variant.chrom = str(row["chrom"])
    variant.position = (
        int(row["edit_start"]) + int(row["edit_end"])
    ) // 2 + 1
    variant.ref = str(row["ref_sequence"])[0]
    variant.alt = str(row["alt_sequence"])[0]
    variant.variant_start = int(row["edit_start"])
    variant.variant_end = int(row["edit_end"])
    variant.variant_ref_seq = str(row["ref_sequence"])
    variant.variant_alt_seq = str(row["alt_sequence"])
    alternate, _ = _apply_variant_edit(reference, sequence_start, variant)
    return alternate


def _summarize(
    ref_outputs: dict,
    alt_outputs: dict,
    tracks: pd.DataFrame,
    mutation: pd.Series,
    *,
    summary_strategy: str,
    windows_bp: tuple[int, ...],
    sequence_start: int,
    readouts: pd.DataFrame | None,
) -> pd.DataFrame:
    summarizer = SUMMARY_STRATEGIES[summary_strategy]
    wide = {}
    for track in tracks.itertuples(index=False):
        ref = (
            ref_outputs[track.output_type][RESOLUTION][
                0, int(track.track_index), :
            ]
            .detach()
            .float()
            .cpu()
            .numpy()
        )
        alt = (
            alt_outputs[track.output_type][RESOLUTION][
                0, int(track.track_index), :
            ]
            .detach()
            .float()
            .cpu()
            .numpy()
        )
        wide.update(
            summarizer(
                ref,
                alt,
                str(track.track_id),
                windows_bp,
                RESOLUTION,
            )
        )
        if readouts is not None:
            for readout in readouts.itertuples(index=False):
                if str(readout.chrom) != str(mutation["chrom"]):
                    raise ValueError(
                        f"Readout chromosome mismatch for {mutation['site_id']}"
                    )
                left = max(
                    0, (int(readout.start) - sequence_start) // RESOLUTION
                )
                right = min(
                    len(ref),
                    int(np.ceil((int(readout.end) - sequence_start) / RESOLUTION)),
                )
                if left >= right:
                    raise ValueError(
                        f"Readout {readout.readout_id} is outside the model "
                        f"output for {mutation['site_id']}"
                    )
                prefix = f"{track.track_id}__{readout.readout_id}"
                ref_slice = np.maximum(ref[left:right], 0.0)
                alt_slice = np.maximum(alt[left:right], 0.0)
                effect = np.log2(1.0 + alt_slice) - np.log2(1.0 + ref_slice)
                wide[f"{prefix}__S_ref_mean"] = float(ref_slice.mean())
                wide[f"{prefix}__log2fc_signed_mean"] = float(effect.mean())
                wide[f"{prefix}__log2fc_absolute_mean"] = float(
                    np.abs(effect).mean()
                )
                wide[f"{prefix}__log2fc_depletion_mean"] = float(
                    np.maximum(-effect, 0.0).mean()
                )
                wide[f"{prefix}__peak_log2fc"] = float(
                    np.log2(
                        (1.0 + float(alt_slice.max()))
                        / (1.0 + float(ref_slice.max()))
                    )
                )
    return pd.DataFrame.from_records(
        [
            {
                "site_id": str(mutation["site_id"]),
                "site_class": str(mutation["site_class"]),
                "mutation_id": str(mutation["mutation_id"]),
                "matched_target_id": str(mutation["matched_target_id"]),
                "control_type": str(mutation["control_type"]),
                **wide,
            }
        ]
    )


def main() -> None:
    args = _parse_args()
    load_strategy_modules(args.strategy_module)
    if args.summary_strategy not in SUMMARY_STRATEGIES:
        raise ValueError(
            f"Unknown summary strategy {args.summary_strategy!r}; "
            f"available={sorted(SUMMARY_STRATEGIES)}"
        )
    windows_bp = tuple(
        int(value) for value in args.windows_bp.split(",") if value.strip()
    )
    if not windows_bp or any(value <= 0 for value in windows_bp):
        raise ValueError("--windows-bp must contain positive integers")
    prepared = Path(args.prepared_dir)
    output = Path(args.output_dir)
    chunks = output / "chunks"
    chunks.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    mutations = pd.read_csv(
        prepared / "mutation_manifest.tsv", sep="\t", low_memory=False
    )
    sites = pd.read_csv(
        prepared / "site_manifest.tsv", sep="\t", low_memory=False
    )
    sites = sites[sites["preparation_status"].eq("ready")].copy()
    if args.site_class:
        sites = sites[sites["site_class"].isin(set(args.site_class))].copy()
    readout_table = (
        pd.read_csv(args.readout_table, sep="\t", low_memory=False)
        if args.readout_table
        else None
    )
    if readout_table is not None:
        required = {"site_id", "readout_id", "chrom", "start", "end"}
        missing = required - set(readout_table.columns)
        if missing:
            raise ValueError(
                f"Readout table is missing columns: {sorted(missing)}"
            )
        sites = sites[
            sites["site_id"].isin(readout_table["site_id"])
        ].copy()
    if args.site_shard_count < 1:
        raise ValueError("--site-shard-count must be positive")
    if not 0 <= args.site_shard_index < args.site_shard_count:
        raise ValueError(
            "--site-shard-index must be in [0, site-shard-count)"
        )
    sites = sites.iloc[
        args.site_shard_index :: args.site_shard_count
    ].copy()
    if args.max_sites is not None:
        sites = sites.head(args.max_sites)
    mutations = mutations[mutations["site_id"].isin(sites["site_id"])].copy()
    if args.mutation_id:
        mutations = mutations[mutations["mutation_id"].eq(args.mutation_id)]
    if mutations.empty:
        raise ValueError("No prepared mutations selected")

    registry = pd.read_csv(
        prepared / "track_registry.tsv", sep="\t", low_memory=False
    )
    metadata, metadata_path = load_track_metadata(args.track_metadata)
    tracks = resolve_track_registry(metadata, registry)
    tracks.to_csv(output / "track_selection.tsv", sep="\t", index=False)
    sites.to_csv(output / "source_sites.tsv", sep="\t", index=False)

    device = torch.device(
        f"cuda:{args.device}" if torch.cuda.is_available() else "cpu"
    )
    if device.type == "cuda":
        torch.cuda.set_device(args.device)
    from alphagenome_pytorch.config import DtypePolicy
    from alphagenome_pytorch.model import AlphaGenome

    model = AlphaGenome.from_pretrained(
        args.weights_path,
        dtype_policy=DtypePolicy.mixed_precision(),
    ).to(device)
    model.eval()
    heads = tuple(dict.fromkeys(tracks["output_type"].astype(str)))
    fasta = FastaSequenceExtractor(args.fasta)
    centers = _context_centers(
        pd.read_csv(
            prepared / "mutation_manifest.tsv", sep="\t", low_memory=False
        )
    )
    timing = []

    for index, (_, row) in enumerate(mutations.iterrows(), start=1):
        mutation_id = str(row["mutation_id"])
        chunk_path = chunks / f"{mutation_id}.parquet"
        if chunk_path.exists():
            print(f"[{index}/{len(mutations)}] {mutation_id}: complete", flush=True)
            continue
        mutation_started = time.perf_counter()
        context_center = centers[mutation_id]
        sequence_start = context_center - AG_INPUT_LEN // 2
        sequence_end = sequence_start + AG_INPUT_LEN
        reference = fasta.extract(
            str(row["chrom"]), sequence_start, sequence_end
        ).upper()
        alternate = _apply_edit(reference, sequence_start, row)
        organism = torch.zeros((1,), dtype=torch.long, device=device)
        with torch.no_grad():
            ref_outputs = model.predict(
                _one_hot(reference, device),
                organism,
                heads=heads,
                resolutions=(RESOLUTION,),
                channels_last=False,
            )
            alt_outputs = model.predict(
                _one_hot(alternate, device),
                organism,
                heads=heads,
                resolutions=(RESOLUTION,),
                channels_last=False,
            )
        _summarize(
            ref_outputs,
            alt_outputs,
            tracks,
            row,
            summary_strategy=args.summary_strategy,
            windows_bp=windows_bp,
            sequence_start=sequence_start,
            readouts=(
                readout_table[
                    readout_table["site_id"].eq(row["site_id"])
                ]
                if readout_table is not None
                else None
            ),
        ).to_parquet(
            chunk_path, index=False
        )
        elapsed = time.perf_counter() - mutation_started
        timing.append({"mutation_id": mutation_id, "seconds": elapsed})
        del ref_outputs, alt_outputs
        if device.type == "cuda":
            torch.cuda.empty_cache()
        print(
            f"[{index}/{len(mutations)}] {mutation_id}: {elapsed:.1f}s",
            flush=True,
        )

    expected = [
        chunks / f"{mutation_id}.parquet"
        for mutation_id in mutations["mutation_id"].astype(str)
    ]
    missing = [path for path in expected if not path.exists()]
    if missing:
        raise RuntimeError(f"Incomplete mutation chunks: {len(missing)} missing")
    mutation_features = pd.concat(
        [pd.read_parquet(path) for path in expected], ignore_index=True
    )
    group_features = adjust_population_features(mutation_features)
    mutation_features.to_parquet(
        output / "mutation_features.parquet", index=False
    )
    group_features.to_parquet(output / "group_features.parquet", index=False)
    group_features.to_csv(
        output / "group_features.tsv", sep="\t", index=False
    )
    if timing:
        pd.DataFrame.from_records(timing).to_csv(
            output / "timing.tsv", sep="\t", index=False
        )
    interval_rows = []
    for _, row in mutations.iterrows():
        mutation_id = str(row["mutation_id"])
        context_center = centers[mutation_id]
        interval_rows.append(
            {
                "mutation_id": mutation_id,
                "site_id": row["site_id"],
                "chrom": row["chrom"],
                "seq_start": context_center - AG_INPUT_LEN // 2,
                "seq_end": context_center - AG_INPUT_LEN // 2 + AG_INPUT_LEN,
                "context_center": context_center,
                "edit_start": int(row["edit_start"]),
                "edit_end": int(row["edit_end"]),
            }
        )
    pd.DataFrame.from_records(interval_rows).to_csv(
        output / "scored_intervals.tsv", sep="\t", index=False
    )
    run_metadata = {
        "prepared_dir": str(prepared),
        "fasta": args.fasta,
        "weights_path": args.weights_path,
        "weights_required": True,
        "track_metadata": str(metadata_path),
        "device": args.device,
        "output_heads": list(heads),
        "resolution_bp": RESOLUTION,
        "windows_bp": list(windows_bp),
        "summary_strategy": args.summary_strategy,
        "strategy_modules": args.strategy_module,
        "n_sites": int(group_features["site_id"].nunique()),
        "n_variants": len(mutation_features),
        "n_tracks": len(tracks),
        "site_shard_index": args.site_shard_index,
        "site_shard_count": args.site_shard_count,
        "site_classes": args.site_class,
        "readout_table": args.readout_table,
        "matched_control_adjustment": True,
        "elapsed_seconds": time.perf_counter() - started,
        "population_prediction_storage": "per-mutation summaries only",
    }
    (output / "run_metadata.json").write_text(
        json.dumps(run_metadata, indent=2), encoding="utf-8"
    )
    print(
        f"Wrote {len(group_features)} site rows and "
        f"{len(mutation_features)} mutation rows to {output}"
    )


if __name__ == "__main__":
    main()
