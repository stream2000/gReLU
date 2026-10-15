#!/usr/bin/env python
"""Build the shared four-cell, two-panel nine-gene Browser PDFs."""

from __future__ import annotations

import argparse
from pathlib import Path

if __package__:
    from .cell_browser_report import build_cell_browser_reports
else:
    from sys import path as sys_path

    sys_path.insert(0, str(Path(__file__).resolve().parents[1]))
    from tools.cell_browser_report import build_cell_browser_reports


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--half-window-bp", type=int, default=512)
    parser.add_argument("--observed-bin-bp", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_cell_browser_reports(
        root=args.root,
        half_window_bp=args.half_window_bp,
        observed_bin_bp=args.observed_bin_bp,
    )


if __name__ == "__main__":
    main()
