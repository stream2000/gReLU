#!/usr/bin/env python
"""Run AlphaGenome 128 bp ChIP inference for MREG three-region mutations.

This script implements Phase 2–3 of the MREG TSS / HC-DIC / LC-DIC 128 bp ChIP ISM
exploration plan. For each mutation in the manifest it:

1. Extracts the 1 Mb AlphaGenome input window centered on the edit.
2. Runs REF and ALT inference for chip_tf + chip_histone heads at 128 bp resolution.
3. Extracts signals from each named readout region (TSS, HC-DIC, LC-DIC, mutation-local).
4. Resolves the track registry against actual model metadata (MCF-7 preferred).
5. Saves long-form chromatin profiles as parquet.

Key differences from the existing run_tf_context_multires_center.py:
- 128 bp only (no 1 bp output).
- chip_tf + chip_histone only (no ATAC/DNase/CAGE/RNA-seq).
- Named genomic readout regions instead of mutation-centered windows.
- Long-form parquet output instead of memmap .npy arrays.
- No full 1 Mb output persistence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from grelu.interpret.tf_context import (
    AG_BIN_SIZE,
    AG_INPUT_LEN,
    FastaSequenceExtractor,
    _apply_variant_edit,
    load_track_metadata,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_TRACK_METADATA = (
    REPO_ROOT
    / "src/alphagenome_pytorch/src/alphagenome_pytorch/data/track_metadata_human.parquet"
)
DEFAULT_FASTA = Path("/home/fqijun/.local/share/genomes/hg38/hg38.fa")
DEFAULT_WEIGHTS = Path("/home/fqijun/.cache/alphagenome/model_fold_0.safetensors")
OUTPUT_KEYS = ("chip_tf", "chip_histone")
RESOLUTION = 128
CONTROL_STRATEGIES = {"local_non_motif_control", "non_peak_local_control"}

# Pseudoautosomal / unplaced contigs whose chrom lengths may differ across
# FASTA indexes; we always treat them as absent from the model's chrom-sizes
# enum and skip context-window checks for them.
UNSUPPORTED_CHROM_PREFIXES = ("chrUn", "chrM", "chrEBV", "GL", "KI", "JH", "KB")


# ---------------------------------------------------------------------------
# One-hot encoding for AlphaGenome direct predict
# ---------------------------------------------------------------------------


def _one_hot_1mb(seq: str) -> torch.Tensor:
    """One-hot encode a 1 Mb sequence into (1, L, 4) float32 tensor."""
    arr = torch.zeros((len(seq), 4), dtype=torch.float32)
    for pos, base in enumerate(seq.upper()):
        idx = {"A": 0, "C": 1, "G": 2, "T": 3}.get(base)
        if idx is not None:
            arr[pos, idx] = 1.0
    return arr.unsqueeze(0)


# ---------------------------------------------------------------------------
# Track resolution
# ---------------------------------------------------------------------------


def _resolve_tracks(
    metadata: pd.DataFrame,
    track_registry: pd.DataFrame,
    output_keys: Sequence[str] = OUTPUT_KEYS,
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """Match track_registry targets to actual model metadata rows.

    For each target in the registry, find the best MCF-7 track in metadata.
    Falls back to the highest-priority track from any biosample if no MCF-7
    track exists.

    Returns
    -------
    resolved : pd.DataFrame
        Merged track registry with track_index and track_name columns.
    global_offset : dict
        Mapping from output_key to starting global track index.
    """
    meta = metadata.copy()
    meta = meta[meta["output_type"].isin(output_keys)].copy()

    # Build global offset map
    offsets: dict[str, int] = {}
    cursor = 0
    for key in output_keys:
        offsets[key] = cursor
        cursor += int((meta["output_type"] == key).sum())

    resolved_rows = []
    for _, reg_row in track_registry.iterrows():
        target = str(reg_row["target_name"]).upper()
        output_type = str(reg_row["output_type"])
        priority = int(reg_row["selection_priority"])

        sub = meta[meta["output_type"] == output_type].copy()

        # Try to match by target name (case-insensitive)
        tx_col = "transcription_factor" if output_type == "chip_tf" else "histone_mark"
        if tx_col in sub.columns:
            sub["_target"] = sub[tx_col].fillna("").astype(str).str.upper()
            sub = sub[sub["_target"] == target]
        elif "target" in sub.columns:
            sub["_target"] = sub["target"].fillna("").astype(str).str.upper()
            sub = sub[sub["_target"] == target]

        if sub.empty:
            resolved_rows.append({
                **reg_row.to_dict(),
                "track_index": -1,
                "track_name": "",
                "biosample_resolved": "",
                "fallback_reason": f"no track found for target {target}",
            })
            continue

        # Prefer the requested biosample exactly. Substring matching is unsafe:
        # "MCF" matches MCF 10A before MCF-7 in the current metadata.
        biosample_col = "biosample_name" if "biosample_name" in sub.columns else "biosample"
        requested_biosample = str(reg_row.get("biosample", "")).strip().upper()
        exact_biosample = sub[
            sub[biosample_col].fillna("").astype(str).str.strip().str.upper()
            == requested_biosample
        ]
        if not exact_biosample.empty:
            best = exact_biosample.iloc[0]
            fallback = ""
        else:
            best = sub.iloc[0]
            best_biosample = str(best.get(biosample_col, best.get("biosample_name", "unknown")))
            fallback = (
                f"no exact {requested_biosample} track for {target}; "
                f"using {best_biosample}"
            )

        global_idx = offsets[output_type] + int(best["track_index"])
        resolved_rows.append({
            **reg_row.to_dict(),
            "track_index": int(best["track_index"]),
            "track_name": str(best.get("track_name", best.get("target", ""))),
            "biosample_resolved": str(best.get(biosample_col, best.get("biosample_name", ""))),
            "fallback_reason": fallback,
        })

    return pd.DataFrame.from_records(resolved_rows), offsets


# ---------------------------------------------------------------------------
# Readout region bin extraction
# ---------------------------------------------------------------------------

BinMapping = Dict[str, Tuple[int, int]]  # readout_id → (start_bin, end_bin)


def _shared_context_centers(mutations: pd.DataFrame) -> Dict[str, int]:
    """Use one model window per region so experimental and control bins align."""
    centers: Dict[str, int] = {}
    for _, group in mutations.groupby("region_id", sort=False):
        experimental = group[
            ~group["mutation_strategy"].isin(CONTROL_STRATEGIES)
        ]
        anchor_row = experimental.iloc[0] if not experimental.empty else group.iloc[0]
        context_center = (
            int(anchor_row["edit_start"]) + int(anchor_row["edit_end"])
        ) // 2
        for mutation_id in group["mutation_id"].astype(str):
            centers[mutation_id] = context_center
    return centers


def _compute_readout_bins(
    readout_regions: pd.DataFrame,
    seq_start: int,
    n_bins: int,
) -> BinMapping:
    """Map genomic readout regions to model output bin indices.

    Each model output bin covers 128 bp. The i-th bin corresponds to genomic
    interval [seq_start + i*128, seq_start + (i+1)*128).

    Raises ValueError if any readout region falls outside the model context.
    """
    mapping: BinMapping = {}
    for _, row in readout_regions.iterrows():
        readout_id = str(row["readout_id"])
        # Per-mutation regions are handled separately
        if readout_id in ("mutation_center_1kb", "mutation_local_4kb"):
            continue
        r_start = int(row["start"])
        r_end = int(row["end"])
        rel_start = r_start - seq_start
        rel_end = r_end - seq_start
        if rel_start < 0 or rel_end > n_bins * RESOLUTION:
            raise ValueError(
                f"Readout {readout_id} ({r_start}-{r_end}) is not fully contained in "
                f"model context (seq_start={seq_start}, n_bins={n_bins})"
            )
        bin_start = rel_start // RESOLUTION
        bin_end = (rel_end + RESOLUTION - 1) // RESOLUTION
        if bin_start >= bin_end:
            raise ValueError(f"Readout {readout_id} has no output bins")
        mapping[readout_id] = (int(bin_start), int(bin_end))
    return mapping


def _compute_mutation_local_bins(
    edit_center: int,
    seq_start: int,
    n_bins: int,
    window_bp: int,
) -> Tuple[int, int]:
    """Compute bin slice for a mutation-centered window."""
    rel_center = edit_center - seq_start
    half_bp = window_bp // 2
    rel_start = rel_center - half_bp
    rel_end = rel_center + half_bp
    bin_start = max(0, rel_start // RESOLUTION)
    bin_end = min(n_bins, (rel_end + RESOLUTION - 1) // RESOLUTION)
    return int(bin_start), int(bin_end)


# ---------------------------------------------------------------------------
# Long-form profile extraction
# ---------------------------------------------------------------------------


def _extract_profiles(
    ref_preds: np.ndarray,     # (n_mutations, n_tracks, n_bins)
    alt_preds: np.ndarray,
    mutation_manifest: pd.DataFrame,
    track_registry_resolved: pd.DataFrame,
    readout_regions: pd.DataFrame,
    seq_starts: Mapping[str, int],
    pseudocount: float = 1.0,
) -> pd.DataFrame:
    """Extract long-form chromatin profiles for all mutation × readout × track combos.

    Returns a DataFrame with columns:
    mutation_id, control_type, readout_id, track_id, target_name, biosample,
    genomic_start, genomic_end, offset_from_readout_anchor_bp,
    ref_value, alt_value, delta, log2fc
    """
    records = []
    n_mutations, n_tracks, n_bins = ref_preds.shape

    for mut_idx in range(n_mutations):
        mut_row = mutation_manifest.iloc[mut_idx]
        mutation_id = str(mut_row["mutation_id"])
        strategy = str(mut_row["mutation_strategy"])
        control_type = (
            "experimental" if "disruption" in strategy or "perturbation" in strategy
            else "matched_control"
        )
        edit_center = (int(mut_row["edit_start"]) + int(mut_row["edit_end"])) // 2
        if mutation_id not in seq_starts:
            raise ValueError(f"Missing model interval for mutation {mutation_id}")
        seq_start = int(seq_starts[mutation_id])

        # Collect all readout windows for this mutation
        all_windows = _compute_readout_bins(readout_regions, seq_start, n_bins)
        readout_anchors = {
            str(row["readout_id"]): int(row["anchor_coordinate"])
            for _, row in readout_regions.iterrows()
            if str(row["readout_id"])
            not in ("mutation_center_1kb", "mutation_local_4kb")
        }

        # Add mutation-local windows
        for window_name, window_bp in [("mutation_center_1kb", 1024), ("mutation_local_4kb", 4096)]:
            bs, be = _compute_mutation_local_bins(edit_center, seq_start, n_bins, window_bp)
            all_windows[window_name] = (bs, be)
            readout_anchors[window_name] = edit_center

        for readout_id, (bin_start, bin_end) in all_windows.items():
            n_readout_bins = bin_end - bin_start
            if n_readout_bins <= 0:
                continue

            # Compute genomic coordinates for each bin
            bin_genomic_starts = seq_start + np.arange(bin_start, bin_end) * RESOLUTION
            readout_anchor = readout_anchors[readout_id]

            for track_idx in range(n_tracks):
                track_row = track_registry_resolved.iloc[track_idx]
                if int(track_row["track_index"]) < 0:
                    continue

                ref_vals = ref_preds[mut_idx, track_idx, bin_start:bin_end]
                alt_vals = alt_preds[mut_idx, track_idx, bin_start:bin_end]

                for bin_offset in range(n_readout_bins):
                    rv = float(ref_vals[bin_offset])
                    av = float(alt_vals[bin_offset])
                    delta = av - rv
                    pc = pseudocount
                    log2fc = float(np.log2((av + pc) / (rv + pc))) if rv + pc > 0 else 0.0

                    records.append({
                        "mutation_id": mutation_id,
                        "control_type": control_type,
                        "readout_id": readout_id,
                        "track_id": str(track_row["track_id"]),
                        "target_name": str(track_row["target_name"]),
                        "biosample": str(track_row.get("biosample_resolved", "")),
                        "genomic_start": int(bin_genomic_starts[bin_offset]),
                        "genomic_end": int(bin_genomic_starts[bin_offset] + RESOLUTION),
                        "offset_from_readout_anchor_bp": int(
                            bin_genomic_starts[bin_offset] - readout_anchor
                        ),
                        "ref_value": rv,
                        "alt_value": av,
                        "delta": delta,
                        "log2fc": log2fc,
                    })

    return pd.DataFrame.from_records(records)


# ---------------------------------------------------------------------------
# Main prediction loop
# ---------------------------------------------------------------------------


def run_chromatin_inference(
    *,
    mutation_manifest: pd.DataFrame,
    readout_regions: pd.DataFrame,
    track_registry: pd.DataFrame,
    fasta_path: Path,
    weights_path: Path,
    track_metadata_path: Path,
    output_dir: Path,
    device: int = 0,
    pseudocount: float = 1.0,
) -> dict:
    """Run AlphaGenome inference for all mutations and extract named readout profiles.

    Returns a provenance dict with output paths and timing.
    """
    run_t0 = time.perf_counter()
    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load resources
    # ------------------------------------------------------------------
    fasta = FastaSequenceExtractor(str(fasta_path))
    metadata = load_track_metadata(str(track_metadata_path))
    track_resolved, head_offsets = _resolve_tracks(metadata, track_registry)

    # Filter to valid tracks
    valid_tracks = track_resolved[track_resolved["track_index"] >= 0].reset_index(drop=True)
    if valid_tracks.empty:
        raise ValueError("No tracks could be resolved from the track registry")

    n_tracks = len(valid_tracks)
    print(f"[runner] Resolved {n_tracks} tracks from registry")

    # ------------------------------------------------------------------
    # 2. Build AlphaGenome model
    # ------------------------------------------------------------------
    device_obj = torch.device(f"cuda:{device}" if torch.cuda.is_available() else "cpu")
    if device_obj.type == "cuda":
        torch.cuda.set_device(device)

    from alphagenome_pytorch.config import DtypePolicy
    from alphagenome_pytorch.model import AlphaGenome

    ag_model = AlphaGenome.from_pretrained(
        str(weights_path),
        dtype_policy=DtypePolicy.mixed_precision(),
    ).to(device_obj)
    ag_model.eval()

    # ------------------------------------------------------------------
    # 3. Predict each mutation
    # ------------------------------------------------------------------
    mutation_manifest = mutation_manifest.reset_index(drop=True)
    n_mutations = len(mutation_manifest)
    context_centers = _shared_context_centers(mutation_manifest)

    # Pre-compute n_bins
    n_bins = AG_INPUT_LEN // RESOLUTION

    # Accumulate prediction arrays (mutations × tracks × bins)
    ref_all = np.zeros((n_mutations, n_tracks, n_bins), dtype=np.float32)
    alt_all = np.zeros((n_mutations, n_tracks, n_bins), dtype=np.float32)

    interval_records = []
    timing_records = []

    for mut_idx, (_, mut_row) in enumerate(mutation_manifest.iterrows()):
        mut_t0 = time.perf_counter()
        mutation_id = str(mut_row["mutation_id"])
        chrom = str(mut_row.get("chrom", ""))
        if not chrom:
            # Infer from region manifest if available
            chrom = "chr2"  # MREG default

        edit_start = int(mut_row["edit_start"])
        edit_end = int(mut_row["edit_end"])
        ref_edit = str(mut_row["ref_sequence"])
        alt_edit = str(mut_row["alt_sequence"])
        edit_center = (edit_start + edit_end) // 2
        context_center = context_centers[mutation_id]

        # Experimental edits and their controls share an identical 1 Mb window.
        # This keeps 128 bp bin boundaries fixed for control subtraction.
        seq_start = context_center - AG_INPUT_LEN // 2
        seq_end = seq_start + AG_INPUT_LEN

        # Validate chromosome bounds
        try:
            chrom_len = fasta.chrom_length(chrom)
        except (KeyError, ValueError) as exc:
            raise ValueError(f"Cannot get chromosome length for {mutation_id}: {chrom}") from exc
        if seq_start < 0 or seq_end > chrom_len:
            raise ValueError(f"1 Mb input window is out of bounds for {mutation_id}")

        # Extract REF and apply edit
        ref_seq = fasta.extract(chrom, seq_start, seq_end).upper()

        # Build a minimal row for _apply_variant_edit
        class _VariantRow:
            pass
        var_row = _VariantRow()
        var_row.site_id = mutation_id
        var_row.chrom = chrom
        var_row.position = edit_center + 1  # 1-based
        var_row.ref = ref_edit[0]
        var_row.alt = alt_edit[0]
        var_row.variant_start = edit_start
        var_row.variant_end = edit_end
        var_row.variant_ref_seq = ref_edit
        var_row.variant_alt_seq = alt_edit

        alt_seq, edit_info = _apply_variant_edit(ref_seq, seq_start, var_row)
        interval_records.append({
            "mutation_id": mutation_id,
            "chrom": chrom,
            "seq_start": seq_start,
            "seq_end": seq_end,
            "edit_center": edit_center,
            "context_center": context_center,
            **edit_info,
        })

        # One-hot encode
        ref_dna = _one_hot_1mb(ref_seq).to(device_obj)
        alt_dna = _one_hot_1mb(alt_seq).to(device_obj)
        organism = torch.zeros((1,), dtype=torch.long, device=device_obj)

        # Predict each head
        ref_head_preds = []
        alt_head_preds = []
        for head_key in OUTPUT_KEYS:
            with torch.no_grad():
                ref_out = ag_model.predict(
                    ref_dna, organism, heads=(head_key,), resolutions=(RESOLUTION,), channels_last=False,
                )
                alt_out = ag_model.predict(
                    alt_dna, organism, heads=(head_key,), resolutions=(RESOLUTION,), channels_last=False,
                )
            ref_head_preds.append(ref_out[head_key][RESOLUTION][0].detach().cpu().numpy().astype(np.float32))
            alt_head_preds.append(alt_out[head_key][RESOLUTION][0].detach().cpu().numpy().astype(np.float32))
            del ref_out, alt_out

        # Concatenate heads in canonical order
        ref_concat = np.concatenate(ref_head_preds, axis=0)  # (total_tracks, n_bins)
        alt_concat = np.concatenate(alt_head_preds, axis=0)

        # Map from global track index → our resolved track index
        for our_idx, (_, track_row) in enumerate(valid_tracks.iterrows()):
            head = str(track_row["output_type"])
            local_idx = int(track_row["track_index"])
            global_offset = head_offsets[head]
            global_idx = global_offset + local_idx
            if global_idx < ref_concat.shape[0]:
                ref_all[mut_idx, our_idx, :] = ref_concat[global_idx, :]
                alt_all[mut_idx, our_idx, :] = alt_concat[global_idx, :]

        del ref_dna, alt_dna, ref_concat, alt_concat
        if device_obj.type == "cuda":
            torch.cuda.empty_cache()

        elapsed = time.perf_counter() - mut_t0
        timing_records.append({
            "mutation_id": mutation_id,
            "seconds": elapsed,
        })
        print(f"[runner] [{mut_idx+1}/{n_mutations}] {mutation_id} done in {elapsed:.1f}s")

    # ------------------------------------------------------------------
    # 4. Extract long-form profiles
    # ------------------------------------------------------------------
    seq_starts = {
        str(record["mutation_id"]): int(record["seq_start"])
        for record in interval_records
    }
    profiles = _extract_profiles(
        ref_all, alt_all,
        mutation_manifest, valid_tracks,
        readout_regions, seq_starts,
        pseudocount=pseudocount,
    )

    # ------------------------------------------------------------------
    # 5. Save outputs
    # ------------------------------------------------------------------
    profiles_path = output_dir / "mreg_chromatin_profiles.parquet"
    profiles.to_parquet(profiles_path, index=False)

    intervals_df = pd.DataFrame.from_records(interval_records)
    intervals_df.to_csv(output_dir / "scored_intervals.tsv", sep="\t", index=False)

    valid_tracks.to_csv(output_dir / "resolved_tracks.tsv", sep="\t", index=False)

    total_seconds = time.perf_counter() - run_t0
    run_config = {
        "fasta_path": str(fasta_path),
        "weights_path": str(weights_path),
        "output_keys": list(OUTPUT_KEYS),
        "resolution": RESOLUTION,
        "device": device,
        "pseudocount": pseudocount,
        "n_mutations": n_mutations,
        "n_tracks": n_tracks,
        "n_profiles": len(profiles),
        "total_seconds": total_seconds,
        "coordinate_frame_version": 3,
        "shared_region_context": True,
    }
    with (output_dir / "run_config.json").open("w") as handle:
        json.dump(run_config, handle, indent=2, default=str)

    timing_df = pd.DataFrame.from_records(timing_records)
    timing_df.to_csv(output_dir / "timing.tsv", sep="\t", index=False)

    provenance = {
        "output_dir": str(output_dir),
        "profiles": str(profiles_path),
        "intervals": str(output_dir / "scored_intervals.tsv"),
        "resolved_tracks": str(output_dir / "resolved_tracks.tsv"),
        "run_config": str(output_dir / "run_config.json"),
        "timing": str(output_dir / "timing.tsv"),
        "total_seconds": total_seconds,
        "n_mutations": n_mutations,
    }
    with (output_dir / "provenance.json").open("w") as handle:
        json.dump(provenance, handle, indent=2, default=str)

    print(f"\n[runner] Complete in {total_seconds:.1f}s")
    print(f"[runner] Profiles: {profiles_path} ({len(profiles)} rows)")
    return provenance


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mutations", required=True, help="Path to mreg_mutation_manifest.tsv")
    parser.add_argument("--readout-regions", required=True, help="Path to mreg_readout_regions.tsv")
    parser.add_argument("--track-registry", required=True, help="Path to mreg_track_registry.tsv")
    parser.add_argument("--mutation-id", default=None, help="Run only this mutation (for smoke testing)")
    parser.add_argument("--fasta", default=str(DEFAULT_FASTA))
    parser.add_argument("--weights-path", default=str(DEFAULT_WEIGHTS))
    parser.add_argument("--track-metadata", default=str(DEFAULT_TRACK_METADATA))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--pseudocount", type=float, default=1.0)
    args = parser.parse_args()

    for path_attr, desc in [
        (args.mutations, "Mutation manifest"),
        (args.readout_regions, "Readout regions"),
        (args.track_registry, "Track registry"),
        (args.fasta, "FASTA"),
    ]:
        if not Path(path_attr).exists():
            raise SystemExit(f"{desc} not found: {path_attr}")

    mutation_manifest = pd.read_csv(args.mutations, sep="\t")
    if args.mutation_id:
        mutation_manifest = mutation_manifest[
            mutation_manifest["mutation_id"] == args.mutation_id
        ]
        if mutation_manifest.empty:
            raise SystemExit(f"Mutation {args.mutation_id} not found in manifest")

    readout_regions = pd.read_csv(args.readout_regions, sep="\t")
    track_registry = pd.read_csv(args.track_registry, sep="\t")

    result = run_chromatin_inference(
        mutation_manifest=mutation_manifest,
        readout_regions=readout_regions,
        track_registry=track_registry,
        fasta_path=Path(args.fasta),
        weights_path=Path(args.weights_path),
        track_metadata_path=Path(args.track_metadata),
        output_dir=Path(args.output_dir),
        device=args.device,
        pseudocount=args.pseudocount,
    )

    print(f"\nOutput: {result['output_dir']}")
    print(f"Profiles: {result['n_mutations']} mutations → {result['profiles']}")


if __name__ == "__main__":
    main()
