"""Failure episodes from `fault_labels` (injector truth) → daily targets per turbine-component.

A *failure* is a `fault_stop` episode whose `params.cause` is `failure_after_overheating[:<component>]`; it starts
exactly where a gearbox / generator-bearing degradation drift (an `overheating` episode, 3–21 days) ends. Trips
(`cause = trip`) are not component failures. The simulator only fails gearbox_t / gen1_t / gen2_t; converter and
transformer rows stay all-negative (reported, not hidden).

Day d covers [00:00, 24:00) UTC; its features use data up to the END of day d. Target for horizon H days:
    y_H[d, j, c] = 1  iff  a failure of (turbine j, component c) starts in (end_d, end_d + H days].
Rows on days that overlap a failure stop of turbine j (the turbine is down / the failure already happened) are
excluded from training and evaluation (`down`).
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np

from wattsentry.model.wind.data import _to_naive_utc

from . import COMPONENTS

DAY = np.timedelta64(1, "D")


@dataclass
class Failure:
    turbine: int  # column index
    turbine_id: str
    component: str
    t_fail: np.datetime64  # failure stop onset (minutes, naive UTC)
    t_end: np.datetime64 | None  # failure stop end
    drift_onset: np.datetime64 | None  # onset of the overheating drift that preceded it
    drift_days: float | None
    amp_c: float | None

    def to_json(self) -> dict:
        return {
            "turbine_id": self.turbine_id, "component": self.component, "t_fail": str(self.t_fail) + "Z",
            "drift_onset": None if self.drift_onset is None else str(self.drift_onset) + "Z",
            "drift_days": self.drift_days, "amp_c": self.amp_c,
        }


def _p(e: dict) -> dict:
    p = e.get("params")
    if isinstance(p, str):
        try:
            return json.loads(p)
        except ValueError:
            return {}
    return dict(p or {})


def failures_from_episodes(episodes: list[dict], turbine_ids: list[str]) -> tuple[list[Failure], list[dict]]:
    """→ (failures, overheating episodes) with times as naive-UTC datetime64[m]."""
    col = {t: i for i, t in enumerate(turbine_ids)}
    drifts = []
    for e in episodes:
        if e["fault"] == "overheating" and e["asset_id"] in col:
            p = _p(e)
            drifts.append({
                "turbine": col[e["asset_id"]], "turbine_id": e["asset_id"], "component": p.get("component"),
                "onset": _to_naive_utc(e["onset"]), "end": _to_naive_utc(e.get("end")),
                "amp_c": p.get("amp_c"), "days": p.get("days"), "ends_in_failure": p.get("ends_in_failure"),
            })
    by_end = {(d["turbine"], d["end"]): d for d in drifts if d["end"] is not None}
    out = []
    for e in episodes:
        if e["fault"] != "fault_stop" or e["asset_id"] not in col:
            continue
        cause = str(_p(e).get("cause", ""))
        if not cause.startswith("failure_after_overheating"):
            continue
        j = col[e["asset_id"]]
        t = _to_naive_utc(e["onset"])
        d = by_end.get((j, t))
        comp = cause.split(":", 1)[1] if ":" in cause else (d or {}).get("component")
        if comp not in COMPONENTS:
            continue
        out.append(Failure(
            turbine=j, turbine_id=e["asset_id"], component=comp, t_fail=t, t_end=_to_naive_utc(e.get("end")),
            drift_onset=None if d is None else d["onset"],
            drift_days=None if d is None or d["days"] is None else float(d["days"]),
            amp_c=None if d is None or d["amp_c"] is None else float(d["amp_c"]),
        ))
    out.sort(key=lambda f: (f.t_fail, f.turbine, f.component))
    return out, drifts


def day_grid(ts: np.ndarray) -> np.ndarray:
    """Day starts (datetime64[m]) for a 15-min grid that starts at 00:00 and covers whole days."""
    d0 = ts[0].astype("datetime64[D]")
    n = len(ts) // 96
    return (d0 + np.arange(n) * DAY).astype("datetime64[m]")


def daily_targets(failures: list[Failure], days: np.ndarray, n_turbines: int, horizons=(7, 14)):
    """→ y {H: uint8 [D, N, C]}, down bool [D, N] (turbine-days overlapping a failure stop)."""
    D, C = len(days), len(COMPONENTS)
    ends = days + DAY.astype("timedelta64[m]")
    y = {h: np.zeros((D, n_turbines, C), np.uint8) for h in horizons}
    down = np.zeros((D, n_turbines), bool)
    for f in failures:
        c = COMPONENTS.index(f.component)
        for h in horizons:
            m = (f.t_fail > ends) & (f.t_fail <= ends + np.timedelta64(h, "D").astype("timedelta64[m]"))
            y[h][m, f.turbine, c] = 1
        t_end = f.t_end if f.t_end is not None else ends[-1]
        down[(days < t_end) & (ends > f.t_fail), f.turbine] = True
    return y, down
