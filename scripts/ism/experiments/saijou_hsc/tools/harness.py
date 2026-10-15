"""Small, dependency-light contracts shared by Saijou analysis entry points."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[5]
SAIJOU_ROOT = REPO_ROOT / "experiments/ism/saijou_all_genes_10bp_scan"


def require_columns(frame: pd.DataFrame, columns: Iterable[str], *, name: str) -> None:
    """Fail early when an upstream table no longer satisfies its contract."""

    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def require_finite(frame: pd.DataFrame, columns: Iterable[str], *, name: str) -> None:
    """Require finite numeric values in report-facing metrics."""

    columns = list(columns)
    require_columns(frame, columns, name=name)
    if not np.isfinite(frame[columns].to_numpy(dtype=float)).all():
        raise ValueError(f"{name} contains non-finite values in {columns}")


def require_unique(frame: pd.DataFrame, columns: Iterable[str], *, name: str) -> None:
    """Require a declared compound key to be unique."""

    columns = list(columns)
    require_columns(frame, columns, name=name)
    if frame.duplicated(columns).any():
        raise ValueError(f"{name} has duplicate rows for key {columns}")


def write_json(path: Path, payload: object) -> None:
    """Write deterministic, newline-terminated JSON."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


def read_json(path: Path) -> object:
    """Read JSON using the same path contract as :func:`write_json`."""

    return json.loads(path.read_text())
