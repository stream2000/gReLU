"""Run-manifest and stage-output helpers."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence

from grelu.interpret.ism.config import CONFIG_SCHEMA_VERSION, RunConfig
from grelu.interpret.ism.errors import ISMUserError


class StageOutputError(ISMUserError, RuntimeError):
    """Raised when a stage would overwrite an existing artifact."""


def sha256_file(path: str | Path) -> str:
    """Return a streaming SHA-256 digest for one file."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def output_dir(config: RunConfig) -> Path:
    return Path(config.paths.output_dir)


def ensure_manifest_config(
    config: RunConfig,
    config_path: str | Path,
) -> Dict[str, Any] | None:
    """Reject a run directory already owned by a different configuration."""

    manifest_path = output_dir(config) / "run_manifest.json"
    if not manifest_path.exists():
        return None
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    config_digest = sha256_file(config_path)
    if manifest.get("config_sha256") != config_digest:
        raise StageOutputError(
            f"Output directory {output_dir(config)} already belongs to a "
            "different configuration. Use a new paths.output_dir."
        )
    return manifest


def require_completed_stage(
    config: RunConfig,
    config_path: str | Path,
    required_stage: str,
) -> None:
    manifest = ensure_manifest_config(config, config_path)
    if manifest is None or required_stage not in manifest.get("stages", {}):
        raise StageOutputError(
            f"Run stage {required_stage!r} with this configuration before continuing"
        )


def reject_completed_downstream_stages(
    config: RunConfig,
    config_path: str | Path,
    downstream_stages: Sequence[str],
) -> None:
    manifest = ensure_manifest_config(config, config_path)
    if manifest is None:
        return
    completed = sorted(set(downstream_stages) & set(manifest.get("stages", {})))
    if completed:
        raise StageOutputError(
            "Cannot overwrite an upstream stage while downstream stages are "
            f"recorded as complete: {completed}. Use a new paths.output_dir."
        )


def prepare_stage_outputs(
    paths: Iterable[str | Path],
    overwrite_stage: bool,
) -> None:
    existing = [Path(path) for path in paths if Path(path).exists()]
    if existing and not overwrite_stage:
        rendered = ", ".join(str(path) for path in existing)
        raise StageOutputError(
            f"Stage output already exists: {rendered}. "
            "Use --overwrite-stage to replace only this stage's artifacts."
        )
    for path in existing:
        if path.is_dir():
            raise StageOutputError(
                f"Stage helper does not replace directory artifact {path}; "
                "the owning stage must handle it explicitly"
            )


def _file_records(paths: Sequence[str | Path]) -> list[Dict[str, Any]]:
    records: list[Dict[str, Any]] = []
    for value in paths:
        path = Path(value)
        record: Dict[str, Any] = {"path": str(path)}
        if path.is_file():
            record["sha256"] = sha256_file(path)
            record["size_bytes"] = path.stat().st_size
        elif path.is_dir():
            files = sorted(item for item in path.rglob("*") if item.is_file())
            record["directory"] = True
            record["file_count"] = len(files)
            record["size_bytes"] = sum(item.stat().st_size for item in files)
        else:
            record["missing"] = True
        records.append(record)
    return records


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item"):
        try:
            return value.item()
        except (TypeError, ValueError):
            pass
    return value


def record_stage(
    *,
    config: RunConfig,
    config_path: str | Path,
    stage: str,
    inputs: Sequence[str | Path],
    outputs: Sequence[str | Path],
    label_status: str,
    metadata: Mapping[str, Any] | None = None,
) -> None:
    """Create or update `run_manifest.json` after a successful stage."""

    root = output_dir(config)
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "run_manifest.json"
    config_digest = sha256_file(config_path)
    manifest = ensure_manifest_config(config, config_path)
    if manifest is None:
        manifest = {
            "schema_version": CONFIG_SCHEMA_VERSION,
            "config_path": str(Path(config_path)),
            "config_sha256": config_digest,
            "stages": {},
        }

    manifest["label_status"] = label_status
    manifest["stages"][stage] = {
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": _file_records(inputs),
        "outputs": _file_records(outputs),
        "metadata": _json_safe(dict(metadata or {})),
    }

    temporary_path = manifest_path.with_suffix(".json.tmp")
    temporary_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(manifest_path)
