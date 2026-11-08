"""`with stage("windows"):` → `LAB_METRIC t_windows_s`, plus peak RSS and peak CUDA memory (MB) for the stage."""
from __future__ import annotations

import contextlib
import resource
import sys
import time

from .metrics import Metrics, emit


def _rss_mb() -> float:  # process peak; ru_maxrss is KiB on Linux, bytes on macOS
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / (1 << 20) if sys.platform == "darwin" else r / 1024


def _cuda():
    torch = sys.modules.get("torch")  # never import torch just for timing
    return torch.cuda if torch is not None and torch.cuda.is_available() else None


@contextlib.contextmanager
def stage(name: str, metrics: Metrics | None = None, timings: dict | None = None):
    out = metrics.emit if metrics else emit
    if cuda := _cuda():
        cuda.reset_peak_memory_stats()
    t = time.perf_counter()
    yield
    dt = time.perf_counter() - t
    if timings is not None:
        timings[f"t_{name}_s"] = round(dt, 3)
    out(f"t_{name}_s", dt)
    out(f"rss_{name}_mb", _rss_mb())
    if cuda := _cuda():
        out(f"cuda_{name}_mb", cuda.max_memory_allocated() / (1 << 20))
