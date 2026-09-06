"""Protocol-freeze integrity gate for outcome and confirmatory stages."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


def require_protocol_freeze(manifest_path: Path, project_root: Path) -> dict[str, object]:
    """Require a frozen manifest and verify every recorded artifact hash."""

    if not manifest_path.exists():
        raise RuntimeError(
            "protocol freeze manifest is absent; close P0 data gates before outcomes"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "FROZEN":
        raise RuntimeError("protocol freeze manifest status is not FROZEN")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict) or not artifacts:
        raise RuntimeError("protocol freeze manifest has no artifact hashes")
    for relative, expected in artifacts.items():
        path = (project_root / relative).resolve()
        try:
            path.relative_to(project_root.resolve())
        except ValueError as exc:
            raise RuntimeError(f"freeze artifact escapes project root: {relative}") from exc
        if not path.is_file():
            raise RuntimeError(f"frozen artifact is missing: {relative}")
        observed = hashlib.sha256(path.read_bytes()).hexdigest()
        if observed != expected:
            raise RuntimeError(f"frozen artifact changed after freeze: {relative}")
    return manifest
