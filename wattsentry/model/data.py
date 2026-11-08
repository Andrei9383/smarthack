"""Window sources and splits for M1 / M2 / M4.

A source is per-site 15-min arrays on one global time axis (memmapped float16 / int8, `wattsentry-windows-v1`
layout: index.json + .npy arrays, see synth.py) plus site metadata and fault episodes: the shared data windows
(wattsentry.data.windows) when available, else the deterministic synthetic RO dataset (synth.py).

A model window is a pure slice: target steps [i, i+96), history steps [i-14*96, i), so history row k step j has the
same time-of-day as target step j for any t0.

Splits: sites 70% train / 30% unseen (fixed in the data); time by the target window's UTC date: train 2022-2023,
val 2024 (ISO weeks with week % 4 != 3), calibration 2024 (week % 4 == 3), test 2025. Train/val/cal use train
sites only; test uses all sites (seen + unseen reported separately).
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from . import physics as ph

HIST_DAYS = 14
H = ph.DAY
FORMAT = "wattsentry-windows-v1"
REQUIRED = ("pv_ac", "load", "batt", "soc", "inv_v", "inv_status", "fault", "pvlib_rec", "pvlib_rec_fc",
            "pvclear_rec", "weather_act", "weather_fc", "geom", "calendar")


@dataclass
class ArraySource:
    """Directory in the wattsentry-windows-v1 layout (synthetic or shared data windows)."""

    root: Path
    index: dict = field(default_factory=dict)
    _cache: dict = field(default_factory=dict, repr=False)

    def __post_init__(self):
        self.root = Path(self.root)
        self.index = json.loads((self.root / "index.json").read_text())
        if self.index.get("format") != FORMAT:
            raise ValueError(f"{self.root}: format {self.index.get('format')!r} != {FORMAT}")
        missing = [k for k in REQUIRED if k not in self.index["arrays"]]
        if missing:
            raise ValueError(f"{self.root}: missing arrays {missing}")
        self.sites = json.loads((self.root / "sites.json").read_text())
        ep = self.root / "episodes.json"
        self.episodes = json.loads(ep.read_text()) if ep.exists() else []
        self.n_steps = int(self.index["n_steps"])
        self.t_start = pd.Timestamp(self.index["t_start"])
        if self.t_start.tzinfo is None:
            self.t_start = self.t_start.tz_localize("UTC")
        self.label = self.index.get("label", "synthetic RO")

    def array(self, name: str) -> np.ndarray:
        if name not in self._cache:
            self._cache[name] = np.load(self.root / self.index["arrays"][name]["file"], mmap_mode="r")
        return self._cache[name]

    @property
    def n_days(self) -> int:
        return self.n_steps // H

    def day_dates(self) -> pd.DatetimeIndex:
        return pd.date_range(self.t_start, periods=self.n_days, freq="D")

    def site_meta(self, key: str, dtype=np.float32) -> np.ndarray:
        """[S] metadata column, built once and cached read-only (batch gathers read it on every step)."""
        ck = ("meta", key, np.dtype(dtype).str)
        if ck not in self._cache:
            a = np.array([(s.get(key) if s.get(key) is not None else 0) for s in self.sites], dtype=dtype)
            a.setflags(write=False)
            self._cache[ck] = a
        return self._cache[ck]

    def has(self, name: str) -> bool:
        return name in self.index["arrays"]

    def loc_idx(self) -> np.ndarray:
        return self.site_meta("loc_idx", np.int64)


def find_lab_windows() -> Path | None:
    """Highest-N $WATTSENTRY_CACHE/data-v{N}/windows (default ~/.cache/sentinel) in our layout, else None."""
    base = Path(os.environ.get("WATTSENTRY_CACHE", Path.home() / ".cache" / "sentinel"))
    cands = []
    for p in base.glob("data-v*/windows/index.json"):
        m = re.match(r"data-v(\d+)", p.parent.parent.name)
        if m:
            cands.append((int(m.group(1)), p.parent))
    for _, d in sorted(cands, reverse=True):
        try:
            if json.loads((d / "index.json").read_text()).get("format") == FORMAT:
                return d
        except Exception:
            continue
    return None


def build_lab_windows(verbose: bool = True) -> Path | None:
    """Build the shared windows (fleet + faults) into the cache via wattsentry.data.windows; None if unavailable."""
    try:
        from wattsentry.data import windows as W
    except ImportError as e:
        if verbose:
            print(f"[data] wattsentry.data.windows not available ({e})", flush=True)
        return None
    return W.build_windows(log=(print if verbose else (lambda *a, **k: None)))


def open_source(kind: str = "auto", synth_sites: int | None = None, verbose: bool = True) -> ArraySource:
    """kind: auto | lab | synth | <path to a layout dir>.

    auto/lab = the shared data windows (built with wattsentry.data.windows when missing). auto falls back to the
    synthetic generator if they are absent or unusable; lab raises instead."""
    if kind not in ("auto", "lab", "synth"):
        return ArraySource(Path(kind))
    if kind in ("auto", "lab"):
        d = find_lab_windows()
        err = None
        if d is None:
            try:
                d = build_lab_windows(verbose)
            except Exception as e:  # pragma: no cover - surfaced in the log
                err = e  # e.g. a concurrent run on the same worker renamed its build into place first
                d = find_lab_windows()
        if d is not None:
            try:
                src = ArraySource(d)
                if verbose:
                    print(f"[data] using shared windows at {d} ({src.index.get('label')})", flush=True)
                return src
            except ValueError as e:
                err = e
        if kind == "lab":
            raise FileNotFoundError(f"shared data windows not available: {err}")
        print(f"[data] shared windows not usable ({err}); FALLING BACK to the synthetic generator", flush=True)
    from .synth import SynthConfig, generate

    cfg = SynthConfig() if synth_sites is None else SynthConfig(n_sites=synth_sites)
    root = generate(cfg, verbose=verbose)
    if verbose:
        print(f"[data] using synthetic RO data at {root}", flush=True)
    return ArraySource(root)


def make_splits(src: ArraySource):
    """The shared split definition (wattsentry.data.splits) for a window source."""
    from wattsentry.data import splits as shared

    return shared.make_splits(src.day_dates(), [s.get("split", "train") for s in src.sites], HIST_DAYS)


def window_labels(fault: np.ndarray, daylight: np.ndarray, min_steps: int = 4) -> np.ndarray:
    """Window label per row: the fault class observable on >= min_steps steps (daylight for PV classes), else healthy.
    fault [B, H] int, daylight [B, H] bool -> [B]."""
    from .synth import FAULT_IDX

    obs_any = (fault == FAULT_IDX["battery_fade"]) | (fault == FAULT_IDX["sensor_fault"])
    vis = (fault > 0) & (daylight | obs_any)
    n_cls = 8
    counts = np.stack([(vis & (fault == c)).sum(1) for c in range(n_cls)], 1)
    counts[:, 0] = 0
    lab = counts.argmax(1)
    lab[counts.max(1) < min_steps] = 0
    return lab
