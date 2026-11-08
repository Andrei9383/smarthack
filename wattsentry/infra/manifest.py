"""`manifest.json`: git sha, data hashes, config, kernel backend, package versions (feeds audit model_version)."""
from __future__ import annotations

import contextlib
import hashlib
import json
import os
import platform
import subprocess
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any, Iterable

PACKAGES = ("numpy", "polars", "pyarrow", "torch", "onnx", "onnxruntime", "lightgbm", "scikit-learn")


def git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return os.environ.get("LAB_COMMIT", "unknown")


def file_hash(p: str | Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def data_hashes(paths: Iterable[str | Path]) -> dict[str, str]:
    """sha256 per file; a directory expands to its Parquet files (recursively)."""
    files = []
    for p in map(Path, paths):
        files += sorted(p.rglob("*.parquet")) if p.is_dir() else [p]
    return {str(f): file_hash(f) for f in files}


def kernel_info() -> dict[str, Any]:
    try:
        from wattsentry import kernels
        return {"backend": kernels.backend(), "rust_available": kernels.rust_available()}
    except Exception as e:  # kernels must never break a manifest
        return {"backend": "unknown", "error": repr(e)}


def versions() -> dict[str, str]:
    out = {"python": platform.python_version()}
    for p in PACKAGES:
        with contextlib.suppress(metadata.PackageNotFoundError):
            out[p] = metadata.version(p)
    return out


def build_manifest(config: dict | None = None, data: Iterable[str | Path] = (), outputs: Iterable[str | Path] = (),
                   **extra: Any) -> dict[str, Any]:
    return {"git_sha": git_sha(), "created_utc": datetime.now(timezone.utc).isoformat(),
            "data": data_hashes(data), "config": config or {}, "kernels": kernel_info(), "versions": versions(),
            "files": {Path(p).name: file_hash(p) for p in outputs}, **extra}


def write_manifest(out_dir: str | Path, **kw: Any) -> Path:
    p = Path(out_dir) / "manifest.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(build_manifest(**kw), indent=1, default=str))
    return p
