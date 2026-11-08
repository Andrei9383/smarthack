"""Eval report: every number is a (block, label, model) row, printed as LAB_METRIC and saved as JSON.

Labels: "synthetic_RO", "real_RO", "non_RO". Metric names on the lab are
`<block>_<model>_<metric>` (block e.g. syn_unseen, pvdaq, pvdaq_inj, ro_load), so the leaderboard and slides can
quote them directly; the JSON keeps the label, the model version and the case count next to each number.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import subprocess
import time
from pathlib import Path
from typing import Any


def _git_sha() -> str:
    for k in ("LAB_COMMIT", "LAB_SHA", "GIT_COMMIT"):
        if os.environ.get(k):
            return os.environ[k]
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return "unknown"


def _clean(v: Any) -> Any:
    if isinstance(v, float) and not math.isfinite(v):
        return None
    if isinstance(v, dict):
        return {k: _clean(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_clean(x) for x in v]
    if hasattr(v, "tolist"):  # numpy arrays and scalars
        return _clean(v.tolist())
    return v


class Report:
    def __init__(self, args: dict[str, Any], out_dir: str | Path | None = None):
        self.out_dir = Path(out_dir or os.environ.get("LAB_OUTPUT_DIR") or ".lab-out")
        self.rows: list[dict[str, Any]] = []
        self.tables: dict[str, Any] = {}
        self.t0 = time.time()
        manifest = Path(__file__).resolve().parents[2] / "data" / "raw" / "MANIFEST.json"
        self.meta = {
            "run_id": os.environ.get("LAB_RUN_ID") or os.environ.get("LAB_RUN"),
            "git_sha": _git_sha(),
            "data_manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest() if manifest.exists() else None,
            "args": args,
        }

    def add(self, block: str, label: str, model: str, metrics: dict[str, float], *, n: int,
            model_version: str = "", note: str = "") -> None:
        self.rows.append(dict(block=block, label=label, model=model, model_version=model_version, n=n,
                              metrics=_clean(metrics), note=note))
        for k, v in metrics.items():
            if isinstance(v, (int, float)) and math.isfinite(v):
                emit(f"{block}_{model}_{k}", v)

    def table(self, name: str, value: Any) -> None:
        """Non-scalar results (confusion matrices, per-class AP, case lists) go only to the JSON."""
        self.tables[name] = _clean(value)

    def write(self) -> Path:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        path = self.out_dir / "eval_report.json"
        body = {**self.meta, "seconds": round(time.time() - self.t0, 1), "results": self.rows, "tables": self.tables}
        path.write_text(json.dumps(body, indent=1, sort_keys=False) + "\n")
        print(f"eval report: {path} ({len(self.rows)} result rows)")
        return path

    def summary(self) -> str:
        """Plain-text table grouped by label, for the run log."""
        lines = []
        for label in sorted({r["label"] for r in self.rows}):
            lines.append(f"== {label}")
            for r in (r for r in self.rows if r["label"] == label):
                ms = "  ".join(f"{k}={v:.3f}" for k, v in r["metrics"].items() if isinstance(v, (int, float)))
                lines.append(f"  {r['block']:<11} {r['model']:<16} n={r['n']:<5} {ms}")
        return "\n".join(lines)


_NAME = re.compile(r"[^A-Za-z0-9_]+")


def emit(name: str, value: float) -> None:
    """`LAB_METRIC name=value` (the lab collects these) and the same pair merged into $LAB_METRICS when set."""
    name = _NAME.sub("_", name).strip("_")
    print(f"LAB_METRIC {name}={value:.6g}", flush=True)
    path = os.environ.get("LAB_METRICS")
    if path:
        p = Path(path)
        data = json.loads(p.read_text()) if p.exists() and p.stat().st_size else {}
        data[name] = float(value)
        p.write_text(json.dumps(data))
