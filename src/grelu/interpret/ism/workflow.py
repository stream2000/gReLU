"""Ordered execution for the complete CTCF ISM EDA v1.2 workflow."""

from __future__ import annotations

from pathlib import Path

from grelu.interpret.ism import analysis, artifacts, features, inference
from grelu.interpret.ism import labels, motifs, sampling
from grelu.interpret.ism.config import RunConfig


def run_all(
    *,
    config: RunConfig,
    config_path: str | Path,
    allow_provisional_labels: bool,
    overwrite_stage: bool,
) -> None:
    """Run every stage in dependency order.

    A full rerun should use a new output directory. Stage overwrite guards
    intentionally prevent rewriting an upstream stage beneath completed
    downstream artifacts.
    """

    common = {
        "config": config,
        "config_path": config_path,
        "allow_provisional_labels": allow_provisional_labels,
        "overwrite_stage": overwrite_stage,
    }
    labels.run(**common)
    sampling.run(**common)
    motifs.run(**common)
    inference.run(**common)
    features.run(**common)
    analysis.run(**common)
    artifacts.run_report(**common)
