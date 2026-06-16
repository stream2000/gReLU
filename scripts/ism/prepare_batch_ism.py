#!/usr/bin/env python
"""Prepare validated mutation, track, and readout manifests for batch ISM."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from grelu.interpret.ism.batch import (  # noqa: E402
    load_strategy_modules,
    prepare_batch_manifests,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sites", required=True)
    parser.add_argument("--tracks", required=True)
    parser.add_argument("--fasta", required=True)
    parser.add_argument("--motif", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--readouts")
    parser.add_argument(
        "--min-relative-motif-score",
        type=float,
        default=0.80,
    )
    parser.add_argument("--motif-controls", type=int, default=3)
    parser.add_argument("--sequence-controls", type=int, default=1)
    parser.add_argument("--max-control-distance-bp", type=int, default=5000)
    parser.add_argument(
        "--strategy-module",
        action="append",
        default=[],
        help="Import a Python module that registers extra mutation strategies",
    )
    args = parser.parse_args()
    load_strategy_modules(args.strategy_module)
    result = prepare_batch_manifests(
        sites_path=args.sites,
        tracks_path=args.tracks,
        fasta_path=args.fasta,
        motif_path=args.motif,
        output_dir=args.output_dir,
        readouts_path=args.readouts,
        min_relative_motif_score=args.min_relative_motif_score,
        motif_controls=args.motif_controls,
        sequence_controls=args.sequence_controls,
        max_control_distance_bp=args.max_control_distance_bp,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
