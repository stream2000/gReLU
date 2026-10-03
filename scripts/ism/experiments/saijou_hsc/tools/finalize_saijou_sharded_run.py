#!/usr/bin/env python
"""Merge complete per-gene targeted-ISM shards and rebuild validation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[5]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from grelu.interpret.ism.shards import finalize_sharded_run  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-dir", type=Path, required=True)
    parser.add_argument("--target-dir", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, action="append", required=True)
    parser.add_argument("--profile-resolution-bp", type=int, default=128)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    validation = finalize_sharded_run(
        prepared=args.prepared_dir.resolve(),
        target=args.target_dir.resolve(),
        sources=[path.resolve() for path in args.source_dir],
        profile_resolution_bp=args.profile_resolution_bp,
    )
    print(json.dumps(validation, indent=2))


if __name__ == "__main__":
    main()
