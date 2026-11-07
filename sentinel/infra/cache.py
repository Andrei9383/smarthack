"""Memmapped float16 window cache keyed by (data hash, window config), under `$LAB_CACHE_DIR` or `.cache/`."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Callable

import numpy as np


def cache_dir() -> Path:
    return Path(os.environ.get("LAB_CACHE_DIR", ".cache")) / "windows"


def cache_key(data_hash: str, config: dict[str, Any]) -> str:
    blob = json.dumps({"data": data_hash, "config": config}, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:20]


def window_cache(data_hash: str, config: dict[str, Any], build: Callable[[], np.ndarray],
                 root: str | Path | None = None, dtype: Any = np.float16) -> np.memmap:
    """Read-only memmap of `build()` cast to `dtype`; built once per key and written atomically."""
    root = Path(root) if root else cache_dir()
    path = root / f"{cache_key(data_hash, config)}.npy"
    if not path.exists():
        root.mkdir(parents=True, exist_ok=True)
        arr = np.asarray(build())
        tmp = path.with_suffix(f".{os.getpid()}.tmp")
        out = np.lib.format.open_memmap(tmp, mode="w+", dtype=dtype, shape=arr.shape)
        out[:] = arr
        out.flush()
        del out
        os.replace(tmp, path)
        path.with_suffix(".json").write_text(json.dumps({"data": data_hash, "config": config}, default=str))
    return np.load(path, mmap_mode="r")
