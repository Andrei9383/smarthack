"""Window sources and splits for M1 / M2 / M4.

`WindowSource` = per-site 15-min arrays on one global time axis, memmapped (float16 / int8), plus site metadata
and fault episodes. Two implementations behind one interface:

(a) #10's memmapped windows at ~/.cache/sentinel/data-v{N}/windows/ when present and in the
    `sentinel-windows-v1` layout (index.json + .npy arrays, documented in synth.py);
(b) else the deterministic synthetic RO dataset (synth.py), generated on first use and cached.

A model window is a pure slice: target steps [i, i+96), history steps [i-14*96, i). Because history rows are whole
days back from i, history row k step j has the same time-of-day as target step j for any t0.

Splits (plan §4): sites 70% train / 30% unseen (fixed in the data); time by the target window's UTC date:
train 2022-2023, val 2024 (ISO weeks with week % 4 != 3), calibration 2024 (week % 4 == 3), test 2025.
Train/val/cal use train sites only; test uses all sites (seen + unseen reported separately).
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import numpy as np
import pandas as pd

from . import physics as ph

HIST_DAYS = 14
REF_DAYS = 7
H = ph.DAY
FORMAT = "sentinel-windows-v1"
REQUIRED = ("pv_ac", "load", "batt", "soc", "inv_v", "inv_status", "fault", "pvlib_rec", "pvlib_rec_fc",
            "pvclear_rec", "weather_act", "weather_fc", "geom", "calendar")


class WindowSource(Protocol):
    root: Path
    label: str
    sites: list[dict]
    episodes: list[dict]
    n_steps: int
    t_start: pd.Timestamp

    def array(self, name: str) -> np.ndarray: ...


@dataclass
class ArraySource:
    """Directory in the sentinel-windows-v1 layout (synthetic or #10)."""

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
        return np.array([(s.get(key) if s.get(key) is not None else 0) for s in self.sites], dtype=dtype)

    def has(self, name: str) -> bool:
        return name in self.index["arrays"]

    def loc_idx(self) -> np.ndarray:
        return self.site_meta("loc_idx", np.int64)


def find_lab_windows() -> Path | None:
    """Highest-N ~/.cache/sentinel/data-v{N}/windows with an index.json in our layout, else None."""
    base = Path(os.environ.get("SENTINEL_CACHE", Path.home() / ".cache" / "sentinel"))
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
    """Build #10's windows (fleet #11 + faults #12) into the shared cache via sentinel.data.windows; None if the
    data cards are not on this branch."""
    try:
        from sentinel.data import windows as W
    except ImportError as e:
        if verbose:
            print(f"[data] sentinel.data.windows (#10) not available ({e})", flush=True)
        return None
    return W.build_windows(log=(print if verbose else (lambda *a, **k: None)))


def open_source(kind: str = "auto", synth_sites: int | None = None, verbose: bool = True) -> ArraySource:
    """kind: auto | lab | synth | <path to a layout dir>.

    auto/lab = #10's windows (shared #11 fleet + #12 faults): read ~/.cache/sentinel/data-v{N}/windows, building it
    with sentinel.data.windows when missing. auto falls back to the #14 synthetic generator only if #10 is absent
    or fails; lab raises instead."""
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
                    print(f"[data] using #10 windows at {d} ({src.index.get('label')})", flush=True)
                return src
            except ValueError as e:
                err = e
        if kind == "lab":
            raise FileNotFoundError(f"#10 windows not available: {err}")
        print(f"[data] #10 windows not usable ({err}); FALLING BACK to the #14 synthetic generator", flush=True)
    from .synth import SynthConfig, generate

    cfg = SynthConfig() if synth_sites is None else SynthConfig(n_sites=synth_sites)
    root = generate(cfg, verbose=verbose)
    if verbose:
        print(f"[data] using synthetic RO data at {root}", flush=True)
    return ArraySource(root)


# --- splits ------------------------------------------------------------------------------------------------
@dataclass
class Splits:
    train_sites: np.ndarray
    unseen_sites: np.ndarray
    days: dict[str, np.ndarray]  # split -> target day indices
    seen_mask: np.ndarray  # [S] bool

    def site_days(self, split: str) -> tuple[np.ndarray, np.ndarray]:
        """(site, day) pairs for a split: train/val/cal over train sites, test over all sites."""
        sites = np.arange(len(self.seen_mask)) if split == "test" else self.train_sites
        d = self.days[split]
        s_, d_ = np.meshgrid(sites, d, indexing="ij")
        return s_.ravel(), d_.ravel()


def make_splits(src: ArraySource):
    """#10's split definition (sentinel.data.splits) when present, else the identical local copy below."""
    try:
        from sentinel.data import splits as S10
    except ImportError:
        return _make_splits_local(src)
    return S10.make_splits(src.day_dates(), [s.get("split", "train") for s in src.sites], HIST_DAYS)


def _make_splits_local(src: ArraySource) -> Splits:
    dates = src.day_dates()
    years = dates.year.to_numpy()
    week = dates.isocalendar().week.to_numpy().astype(int)
    day_idx = np.arange(len(dates))
    ok = day_idx >= HIST_DAYS  # need 14 days of history
    seen = np.array([s.get("split", "train") == "train" for s in src.sites])
    days = {
        "train": day_idx[ok & np.isin(years, [2022, 2023])],
        "val": day_idx[ok & (years == 2024) & (week % 4 != 3)],
        "cal": day_idx[ok & (years == 2024) & (week % 4 == 3)],
        "test": day_idx[ok & (years == 2025)],
    }
    # Small datasets (tests, smoke): fall back to proportional time blocks so every split is non-empty.
    if any(len(v) == 0 for v in days.values()):
        avail = day_idx[ok]
        n = len(avail)
        a, b, c = int(n * 0.55), int(n * 0.7), int(n * 0.8)
        days = {"train": avail[:a], "val": avail[a:b], "cal": avail[b:c], "test": avail[c:]}
    return Splits(np.nonzero(seen)[0], np.nonzero(~seen)[0], days, seen)


def fault_daylight_label(fault_row: np.ndarray, daylight: np.ndarray, min_steps: int = 4) -> int:
    """Window label: the fault class observable on >= min_steps steps (daylight for PV classes), else healthy."""
    from .synth import FAULT_IDX

    obs_any = (fault_row == FAULT_IDX["battery_fade"]) | (fault_row == FAULT_IDX["sensor_fault"])
    vis = (fault_row > 0) & (daylight | obs_any)
    if vis.sum() < min_steps:
        return 0
    vals, cnt = np.unique(fault_row[vis], return_counts=True)
    return int(vals[np.argmax(cnt)])


def window_labels(fault: np.ndarray, daylight: np.ndarray, min_steps: int = 4) -> np.ndarray:
    """Vectorised fault_daylight_label over a batch: fault [B, H] int, daylight [B, H] bool -> [B]."""
    from .synth import FAULT_IDX

    obs_any = (fault == FAULT_IDX["battery_fade"]) | (fault == FAULT_IDX["sensor_fault"])
    vis = (fault > 0) & (daylight | obs_any)
    n_cls = 8
    counts = np.stack([(vis & (fault == c)).sum(1) for c in range(n_cls)], 1)
    counts[:, 0] = 0
    lab = counts.argmax(1)
    lab[counts.max(1) < min_steps] = 0
    return lab


def healthy_day_mask(src: ArraySource, sites: np.ndarray, days: np.ndarray) -> np.ndarray:
    """True where the target day (00-24 UTC) has no fault step at all."""
    f = src.array("fault")
    out = np.empty(len(sites), bool)
    for k in range(0, len(sites), 4096):
        s, d = sites[k:k + 4096], days[k:k + 4096]
        idx = d[:, None] * H + np.arange(H)[None]
        out[k:k + 4096] = (np.asarray(f[s[:, None], idx]) == 0).all(1)
    return out
