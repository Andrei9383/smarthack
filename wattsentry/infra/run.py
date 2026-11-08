"""RunContext: seeds, device + AMP dtype, `$LAB_OUTPUT_DIR`, config dump; metrics/stage/manifest in one place."""
from __future__ import annotations

import json
import os
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from . import timing
from .manifest import write_manifest
from .metrics import Metrics


def _torch():
    try:
        import torch
        return torch
    except ImportError:
        return None


@dataclass
class RunContext:
    name: str = ""  # metric prefix per model ("" = none)
    seed: int = 0
    config: dict[str, Any] = field(default_factory=dict)
    out: Path | None = None
    device: str = "cpu"
    amp_dtype: Any = None
    timings: dict[str, float] = field(default_factory=dict)

    def __post_init__(self):
        self.t0 = time.perf_counter()
        self.metrics = Metrics(self.name)
        self.out = Path(self.out or os.environ.get("LAB_OUTPUT_DIR", ".lab-out"))
        self.out.mkdir(parents=True, exist_ok=True)
        random.seed(self.seed)
        np.random.seed(self.seed)
        self.rng = np.random.default_rng(self.seed)
        if (torch := _torch()) is not None:
            torch.manual_seed(self.seed)
            if torch.cuda.is_available():
                self.device = "cuda"
                self.amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        (self.out / "config.json").write_text(json.dumps(self.config, indent=1, default=str))

    def emit(self, name: str, value: float) -> None:
        self.metrics.emit(name, value)

    def stage(self, name: str):
        return timing.stage(name, self.metrics, self.timings)

    def manifest(self, data: Iterable[str | Path] = (), outputs: Iterable[str | Path] = (), **extra: Any) -> Path:
        self.emit("t_total_s", time.perf_counter() - self.t0)
        return write_manifest(self.out, config=self.config, data=data, outputs=outputs, model=self.name,
                              seed=self.seed, device=self.device, amp_dtype=str(self.amp_dtype),
                              metrics=self.metrics.values, timings=self.timings, argv=sys.argv, **extra)
