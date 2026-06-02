#!/usr/bin/env python
"""Make random ENCODE4 CTCF motif sites with whole-motif replacement edits."""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import pandas as pd
from pyfaidx import Fasta

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "scripts/ism") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts/ism"))

from prepare_ctcf_sites import (  # noqa: E402
    CANONICAL_HG38,
    DEFAULT_BIGBEDTOBED,
    DEFAULT_FASTA,
    DEFAULT_MEME,
    DEFAULT_RPEAK_BB,
    DEFAULT_UCSC_API,
    best_motif_hit,
    iter_ucsc_api_ctcf_rpeaks,
    iter_ucsc_ctcf_rpeaks,
    load_meme_pwm,
    pwm_to_pssm,
)

BASES = "ACGT"
AG_INPUT_LEN = 1_048_576


def _reservoir_sample(iterable, k: int, rng: random.Random) -> list[dict]:
    sample: list[dict] = []
    for i, item in enumerate(iterable, start=1):
        if len(sample) < k:
            sample.append(item)
            continue
        j = rng.randrange(i)
        if j < k:
            sample[j] = item
    return sample


def _random_dna(length: int, ref: str, rng: random.Random) -> str:
    if length <= 0:
        raise ValueError("Replacement length must be positive")
    for _ in range(100):
        alt = "".join(rng.choice(BASES) for _ in range(length))
        if alt != ref:
            return alt
    return ref[:-1] + next(base for base in BASES if base != ref[-1])


def _peak_iter(args):
    if args.source == "api":
        return iter_ucsc_api_ctcf_rpeaks(
            api_url=args.api_url,
            chunk_bp=args.chunk_bp,
            max_peaks=None,
            chromosomes=list(CANONICAL_HG38),
        )
    if args.source == "bigbed":
        return iter_ucsc_ctcf_rpeaks(args.bigBedToBed, args.rpeak_bb, max_peaks=None)
    raise ValueError(f"Unknown source: {args.source}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num_sites", type=int, default=1000)
    parser.add_argument("--reservoir_size", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260529)
    parser.add_argument("--fasta", default=str(DEFAULT_FASTA))
    parser.add_argument("--output", required=True)
    parser.add_argument("--source", choices=["api", "bigbed"], default="bigbed")
    parser.add_argument("--api_url", default=DEFAULT_UCSC_API)
    parser.add_argument("--rpeak_bb", default=DEFAULT_RPEAK_BB)
    parser.add_argument("--chunk_bp", type=int, default=5_000_000)
    parser.add_argument("--meme", default=str(DEFAULT_MEME))
    parser.add_argument("--bigBedToBed", default=DEFAULT_BIGBEDTOBED)
    parser.add_argument("--min_motif_score", type=float, default=None)
    args = parser.parse_args()

    if args.reservoir_size < args.num_sites:
        raise SystemExit("--reservoir_size must be >= --num_sites")

    rng = random.Random(args.seed)
    peaks = _reservoir_sample(_peak_iter(args), args.reservoir_size, rng)
    rng.shuffle(peaks)

    pwm = load_meme_pwm(args.meme, motif_name="CTCF")
    pssm = pwm_to_pssm(pwm)
    fasta = Fasta(str(args.fasta), as_raw=True, sequence_always_upper=True)

    records: list[dict] = []
    for peak in peaks:
        chrom = peak["chrom"]
        if chrom not in fasta:
            continue
        seq = str(fasta[chrom][peak["start"] : peak["end"]]).upper()
        if len(seq) != peak["end"] - peak["start"]:
            continue
        motif_rel_start, motif_rel_end, strand, motif_score = best_motif_hit(seq, pssm)
        if args.min_motif_score is not None and motif_score < args.min_motif_score:
            continue

        motif_start = int(peak["start"]) + motif_rel_start
        motif_end = int(peak["start"]) + motif_rel_end
        motif_center = (motif_start + motif_end) // 2
        chrom_len = len(fasta[chrom])
        if motif_center - AG_INPUT_LEN // 2 < 0 or motif_center + AG_INPUT_LEN // 2 > chrom_len:
            continue

        ref_seq = seq[motif_rel_start:motif_rel_end]
        alt_seq = _random_dna(len(ref_seq), ref_seq, rng)
        site_id = f"random_ctcf_whole_motif_{len(records) + 1:04d}"
        records.append(
            {
                "site_id": site_id,
                "paired_site_id": site_id,
                "chrom": chrom,
                "start": int(peak["start"]),
                "end": int(peak["end"]),
                "name": peak["name"],
                "peak_score": int(peak["peak_score"]),
                "strand": strand,
                "motif_start": motif_start,
                "motif_end": motif_end,
                "matched_seq": ref_seq,
                "motif_score": motif_score,
                "rpeak_ubiquity": peak.get("ubiquity", ""),
                "cCRE": peak.get("cCRE", ""),
                "control_type": "whole_motif_random_replacement",
                "mutation_strategy": "whole_motif_random_replacement",
                "variant_position": motif_center + 1,
                "variant_ref": ref_seq,
                "variant_alt": alt_seq,
                "variant_start": motif_start,
                "variant_end": motif_end,
                "variant_ref_seq": ref_seq,
                "variant_alt_seq": alt_seq,
                "variant_offset": motif_center - motif_start,
                "random_seed": args.seed,
            }
        )
        if len(records) >= args.num_sites:
            break

    if len(records) < args.num_sites:
        raise SystemExit(
            f"Only produced {len(records)} sites from reservoir_size={args.reservoir_size}; "
            "increase --reservoir_size or relax filters."
        )

    out = pd.DataFrame.from_records(records)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output, sep="\t", index=False)
    print(f"Wrote {len(out)} random CTCF whole-motif replacement sites to {output}")
    print(out[["site_id", "chrom", "motif_start", "motif_end", "strand", "motif_score"]].head().to_string(index=False))


if __name__ == "__main__":
    main()
