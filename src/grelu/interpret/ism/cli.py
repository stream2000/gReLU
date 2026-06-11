"""Unified command-line contract for CTCF ISM EDA v1.2."""

from __future__ import annotations

import argparse
import importlib
from pathlib import Path
from typing import Callable, Dict, Sequence, Tuple

from grelu.interpret.ism.config import RunConfig
from grelu.interpret.ism.errors import ISMUserError


class StageNotImplementedError(ISMUserError, RuntimeError):
    """Raised when a declared checkpoint stage is not implemented yet."""


COMMANDS: Tuple[str, ...] = (
    "labels",
    "sample",
    "mutate",
    "infer",
    "features",
    "analyze",
    "report",
    "all",
)

_COMMAND_TARGETS: Dict[str, Tuple[str, str]] = {
    "labels": ("grelu.interpret.ism.labels", "run"),
    "sample": ("grelu.interpret.ism.sampling", "run"),
    "mutate": ("grelu.interpret.ism.motifs", "run"),
    "infer": ("grelu.interpret.ism.inference", "run"),
    "features": ("grelu.interpret.ism.features", "run"),
    "analyze": ("grelu.interpret.ism.analysis", "run"),
    "report": ("grelu.interpret.ism.artifacts", "run_report"),
    "all": ("grelu.interpret.ism.workflow", "run_all"),
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ctcf_ism.py",
        description="Paper-aligned CTCF ISM EDA v1.2 workflow",
    )
    parser.add_argument(
        "command",
        choices=COMMANDS,
        help="Workflow stage to execute",
    )
    parser.add_argument(
        "--config",
        required=True,
        type=Path,
        help="Versioned JSON run configuration",
    )
    parser.add_argument(
        "--allow-provisional-labels",
        action="store_true",
        help=(
            "Allow experimental reconstructed labels to proceed beyond the labels "
            "stage. Outputs remain marked provisional."
        ),
    )
    parser.add_argument(
        "--overwrite-stage",
        action="store_true",
        help="Replace artifacts owned by the requested stage only",
    )
    return parser


def _load_target(command: str) -> Callable[..., None]:
    module_name, function_name = _COMMAND_TARGETS[command]
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        if exc.name == module_name:
            raise StageNotImplementedError(
                f"Command {command!r} is declared by the CP0 interface but its "
                "runtime stage has not been implemented yet"
            ) from exc
        raise
    target = getattr(module, function_name, None)
    if target is None:
        raise StageNotImplementedError(
            f"Command target {module_name}:{function_name} is not implemented"
        )
    return target


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        config = RunConfig.load(args.config)
        target = _load_target(args.command)
        target(
            config=config,
            config_path=args.config,
            allow_provisional_labels=args.allow_provisional_labels,
            overwrite_stage=args.overwrite_stage,
        )
    except ISMUserError as exc:
        parser.error(str(exc))
    return 0
