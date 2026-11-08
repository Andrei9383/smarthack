"""`LAB_METRIC name=value` lines (collected by the lab) mirrored into the `$LAB_METRICS` JSON file."""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

_NAME = re.compile(r"[^A-Za-z0-9_]+")


def emit(name: str, value: float, prefix: str = "") -> str:
    name = _NAME.sub("_", f"{prefix}_{name}" if prefix else name).strip("_")
    value = float(value)
    print(f"LAB_METRIC {name}={value:.6g}", flush=True)
    if path := os.environ.get("LAB_METRICS"):
        p = Path(path)
        data = json.loads(p.read_text()) if p.exists() and p.stat().st_size else {}
        data[name] = value
        p.write_text(json.dumps(data))
    return name


class Metrics:
    """Per-model emitter: remembers every value it emitted (for the report / manifest)."""

    def __init__(self, prefix: str = ""):
        self.prefix, self.values = prefix, {}

    def emit(self, name: str, value: float) -> None:
        self.values[emit(name, value, self.prefix)] = float(value)
