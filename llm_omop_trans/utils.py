"""Provide deterministic serialization, hashing, and SQL quoting."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import cast


def _canonical_json(value: object) -> str:
    """Serialize semantic inputs with deterministic key ordering."""
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _sha256_bytes(value: bytes) -> str:
    """Return a SHA-256 hex digest."""
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    """Hash a file without loading it entirely into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_strict_json(value: str) -> object:
    """Reject duplicate keys and nonstandard numeric constants."""

    def reject_constant(_value: str) -> None:
        raise ValueError("nonstandard JSON constant")

    def reject_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, item in pairs:
            if key in result:
                raise ValueError("duplicate JSON key")
            result[key] = item
        return result

    return cast(
        object,
        json.loads(
            value,
            parse_constant=reject_constant,
            object_pairs_hook=reject_duplicates,
        ),
    )


def _sql_string(value: str) -> str:
    """Quote a string as a DuckDB SQL literal."""
    return "'" + value.replace("'", "''") + "'"


def _implementation_hash() -> str:
    """Hash every executable source file that defines run semantics."""
    package = Path(__file__).resolve().parent
    entrypoint = package.parent / "llm_trans_main.py"
    paths = [
        package / name
        for name in ("config.py", "utils.py", "validator.py", "llm_trans.py")
    ]
    paths.append(entrypoint)
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.name.encode("utf-8"))
        digest.update(_sha256_file(path).encode("ascii"))
    return digest.hexdigest()
