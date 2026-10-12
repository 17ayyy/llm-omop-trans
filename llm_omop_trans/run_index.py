"""Rebuild a small, secret-free index from local run evidence."""

from __future__ import annotations

import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any

import duckdb
from filelock import FileLock

RUN_ID = re.compile(r"run-[0-9a-f]{16}\Z")


def run_directory(
    root: Path, run_id: str, run_type: str, *, resume: bool
) -> Path:
    """Use the isolated Pilot path, with a root fallback for old Pilots."""
    if run_type != "pilot":
        return root / run_id
    preferred = root / "test" / run_id
    legacy = root / run_id
    if resume and not preferred.is_dir() and legacy.is_dir():
        return legacy
    return preferred


def pilot_directories(root: Path) -> list[Path]:
    """Discover migrated and unmigrated Pilot manifests."""
    return sorted([*root.glob("run-*"), *(root / "test").glob("run-*")])


def _run_directories(root: Path) -> list[Path]:
    directories = [*root.glob("run-*"), *(root / "test").glob("run-*")]
    return sorted(
        path
        for path in directories
        if path.is_dir()
        and not path.is_symlink()
        and RUN_ID.fullmatch(path.name)
    )


def _state_summary(
    path: Path, live: duckdb.DuckDBPyConnection | None
) -> tuple[dict[str, int], dict[str, int], list[str], list[str]]:
    connection = live or duckdb.connect(str(path), read_only=True)
    try:
        statuses = dict(
            sorted(
                connection.execute(
                    "SELECT tech_status, count(*) FROM targets "
                    "GROUP BY tech_status"
                ).fetchall()
            )
        )
        columns = {
            row[1]
            for row in connection.execute(
                "PRAGMA table_info('calls')"
            ).fetchall()
        }
        models: dict[str, int] = {}
        if "model_name" in columns:
            models = {
                model: count
                for model, count in connection.execute(
                    "SELECT model_name, count(*) FROM calls "
                    "WHERE model_name IS NOT NULL GROUP BY model_name"
                ).fetchall()
            }
        target_columns = {
            row[1]
            for row in connection.execute(
                "PRAGMA table_info('targets')"
            ).fetchall()
        }
        target_models: list[str] = []
        if "model_name" in target_columns:
            target_models = sorted(
                model
                for (model,) in connection.execute(
                    "SELECT DISTINCT model_name FROM targets "
                    "WHERE model_name IS NOT NULL"
                ).fetchall()
                if model
            )
        requested: list[str] = []
        if "metadata_json" in target_columns:
            requested = sorted(
                model
                for (model,) in connection.execute(
                    "SELECT DISTINCT json_extract_string(metadata_json, "
                    "'$.requested_model') FROM targets "
                    "WHERE metadata_json IS NOT NULL"
                ).fetchall()
                if model
            )
        return statuses, dict(sorted(models.items())), requested, target_models
    finally:
        if live is None:
            connection.close()


def _entry(
    root: Path,
    directory: Path,
    previous: dict[str, Any] | None,
    live: duckdb.DuckDBPyConnection | None,
) -> dict[str, Any]:
    relative = directory.relative_to(root).as_posix()
    try:
        manifest = json.loads(
            (directory / "run_manifest.json").read_text(encoding="utf-8")
        )
    except (OSError, ValueError):
        return {
            "run_type": "unknown",
            "vocabulary_id": None,
            "model_name": None,
            "models_used": [],
            "purpose": None,
            "output_dir": relative,
            "status": "unknown",
            "unresolved_reason": "Missing or invalid run manifest",
        }
    if (
        not isinstance(manifest, dict)
        or manifest.get("run_id") != directory.name
    ):
        raise ValueError(f"Run manifest identity mismatch: {directory.name}")
    semantic = manifest.get("semantic")
    if not isinstance(semantic, dict):
        if "run_fingerprint" not in manifest:
            return {
                "run_type": "unknown",
                "vocabulary_id": None,
                "model_name": None,
                "models_used": [],
                "purpose": None,
                "output_dir": relative,
                "status": "unknown",
                "unresolved_reason": "Unrecognized run manifest format",
            }
        return {
            "run_type": "pipeline",
            "vocabulary_id": None,
            "model_name": None,
            "models_used": [],
            "purpose": None,
            "output_dir": relative,
            "status": (
                "completed"
                if (directory / "07_final.parquet").is_file()
                else "partial"
            ),
        }
    run_type = semantic.get("run_kind")
    reason = None
    if run_type not in {"pilot", "full"}:
        run_type = "unknown"
        reason = "Legacy translator manifest has no run_kind"
    vocabulary = semantic.get("vocabulary_id")
    if not isinstance(vocabulary, str) or not vocabulary:
        vocabulary = None
    label = semantic.get("pilot_label")
    purpose = (
        f"Pilot test: {label}"
        if run_type == "pilot" and isinstance(label, str) and label
        else (
            "Pilot test"
            if run_type == "pilot"
            else "Full vocabulary translation"
            if run_type == "full"
            else None
        )
    )
    entry: dict[str, Any] = {
        "run_type": run_type,
        "vocabulary_id": vocabulary,
        "model_name": None,
        "models_used": [],
        "purpose": purpose,
        "output_dir": relative,
        "status": "unknown",
    }
    if reason:
        entry["unresolved_reason"] = reason
    configured = semantic.get("model") or manifest.get("requested_model")
    if isinstance(configured, str) and configured:
        entry["configured_model"] = configured
    state = directory / "state.duckdb"
    if not state.is_file():
        return entry
    try:
        statuses, models, requested, target_models = _state_summary(state, live)
    except (duckdb.Error, OSError, ValueError):
        if previous and previous.get("output_dir") == relative:
            return previous
        entry["unresolved_reason"] = "State is unavailable for inspection"
        return entry
    entry["status_counts"] = statuses
    models_used = sorted(set(models) | set(target_models))
    entry["models_used"] = models_used
    entry["model_name"] = models_used[0] if len(models_used) == 1 else None
    if models:
        entry["model_call_counts"] = models
    if requested:
        entry["requested_models"] = requested
    target_count = manifest.get("target_row_count")
    if isinstance(target_count, int) and target_count >= 0:
        entry["target_row_count"] = target_count
    total = sum(statuses.values())
    if (
        total == target_count
        and not statuses.get("ready", 0)
        and not statuses.get("failed", 0)
        and (directory / "omop-trans.csv").is_file()
    ):
        entry["status"] = "completed"
    else:
        entry["status"] = "partial"
    return entry


def rebuild_index(
    output_root: Path,
    *,
    live_run_id: str | None = None,
    live_connection: duckdb.DuckDBPyConnection | None = None,
) -> Path:
    """Reconcile disk runs under one lock and atomically replace index."""
    output_root.mkdir(parents=True, exist_ok=True)
    index_path = output_root / "run-index.json"
    with FileLock(str(output_root / ".run-index.lock"), timeout=60):
        old: dict[str, Any] = {}
        if index_path.is_file():
            try:
                parsed = json.loads(index_path.read_text(encoding="utf-8"))
                if parsed.get("schema_version") == 1 and isinstance(
                    parsed.get("runs"), dict
                ):
                    old = parsed["runs"]
            except (OSError, ValueError, AttributeError):
                pass
        runs: dict[str, Any] = {}
        for directory in _run_directories(output_root):
            if directory.name in runs:
                raise ValueError(
                    f"Duplicate run ID in outputs: {directory.name}"
                )
            live = live_connection if directory.name == live_run_id else None
            runs[directory.name] = _entry(
                output_root, directory, old.get(directory.name), live
            )
        data = {"schema_version": 1, "runs": dict(sorted(runs.items()))}
        serialized = json.dumps(data, ensure_ascii=False, indent=2) + "\n"
        temp_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                newline="\n",
                prefix=".run-index-",
                suffix=".tmp",
                dir=output_root,
                delete=False,
            ) as target:
                temp_path = Path(target.name)
                target.write(serialized)
                target.flush()
                os.fsync(target.fileno())
            os.replace(temp_path, index_path)
        finally:
            if temp_path is not None:
                temp_path.unlink(missing_ok=True)
    return index_path


if __name__ == "__main__":
    from llm_omop_trans.config import OUTPUT_ROOT

    print(rebuild_index(OUTPUT_ROOT))
