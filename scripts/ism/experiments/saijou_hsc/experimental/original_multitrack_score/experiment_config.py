"""Paths, run names and benchmark intervals for the multitrack experiment."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


REPO = Path(__file__).resolve().parents[6]
DEFAULT_ROOT = REPO / "experiments/ism/original_multitrack_score_20260728"
ALPHAGENOME_RUNS = {
    "Mdk": "alphagenome_original_mdk",
    "Col1a1": "alphagenome_original_col1a1",
    "Acta2": "alphagenome_original_acta2",
}


def load_benchmark_intervals(
    root: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load registered controls and the prespecified non-fitting Mdk holdout."""
    controls = pd.read_csv(
        root
        / "prepared_full3kb_three_gene"
        / "positive_control_registry.tsv",
        sep="\t",
    )
    holdout = pd.DataFrame(
        [
            {
                "control_id": "mdk_motif_backed_holdout",
                "gene": "Mdk",
                "family": "KLF_SP_EGR",
                "label": "Mdk +463..+531 motif-backed holdout",
                "start": 463,
                "end": 531,
                "evidence": "prior_mdk_audit_not_score_fitting",
                "coordinate_rule": "transcription-direction offset",
            }
        ]
    )
    return controls, holdout
