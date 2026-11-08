"""Schedule fault episodes per site, apply them, re-simulate the battery, write labels (data.md fault_labels).

Episodes on one site never overlap (one PvFault class at a time, so a site-day has exactly one label), and an
episode that turns out to have no visible effect (e.g. shading in a sector the sun never reaches that season) is
dropped, not labelled. Physical faults change pv_ac_w; the self-consumption battery rule (fleet/battery.py) is then
re-run from the first onset, so batt_w / soc / grid_w stay physically consistent. Sensor faults are applied last
and only change what is reported.
"""

from __future__ import annotations

import json
import zlib
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ... import kernels
from ..fleet.battery import BatteryParams
from .kinds import APPLY, STEPS_PER_DAY, Episode, overvoltage_threshold

TELEMETRY_COLS = ("pv_ac_w", "inv_voltage_v", "inv_status", "soc", "batt_w", "load_w", "grid_w")
PHYSICAL = ("soiling", "shading", "string_outage", "clipping", "overvoltage_trip")


@dataclass(frozen=True)
class FaultConfig:
    seed: int = 11
    faulty_share: float = 0.15  # target share of faulty site-days
    weights: dict[str, float] = field(default_factory=lambda: {
        "soiling": 0.20, "shading": 0.14, "string_outage": 0.14, "clipping": 0.12,
        "overvoltage_trip": 0.12, "battery_fade": 0.12, "sensor_fault": 0.16})
    # Episode length in days (uniform between lo and hi)
    days: dict[str, tuple[float, float]] = field(default_factory=lambda: {
        "soiling": (10, 45), "shading": (30, 200), "string_outage": (3, 30), "clipping": (7, 60),
        "overvoltage_trip": (14, 90), "battery_fade": (60, 365), "sensor_fault": (0.5, 5)})
    min_effect: float = 0.01  # drop physical episodes that lose less than 1 % of the energy in their window

    def to_json(self) -> dict:
        return {"seed": self.seed, "faulty_share": self.faulty_share, "weights": self.weights,
                "days": {k: list(v) for k, v in self.days.items()}, "min_effect": self.min_effect}


def site_rng(seed: int, site_id: str) -> np.random.Generator:
    return np.random.default_rng([seed, zlib.crc32(str(site_id).encode())])


def available_kinds(cols: dict[str, np.ndarray], site: pd.Series) -> list[str]:
    kinds = ["soiling", "string_outage", "sensor_fault"]
    if "lat" in site and "lon" in site:
        kinds.append("shading")
    if np.isfinite(site.get("inverter_kw", np.nan)):
        kinds.append("clipping")
    if "inv_voltage_v" in cols and np.isfinite(cols["inv_voltage_v"]).any():
        kinds.append("overvoltage_trip")
    if float(site.get("battery_kwh", 0) or 0) > 0 and {"soc", "batt_w", "load_w"} <= cols.keys():
        kinds.append("battery_fade")
    return kinds


def schedule(T: int, kinds: list[str], rng, cfg: FaultConfig) -> list[tuple[str, int, int]]:
    """Non-overlapping (kind, i0, i1) episodes; expected faulty share of time ~ cfg.faulty_share."""
    w = np.array([cfg.weights[k] for k in kinds])
    if w.sum() <= 0:
        return []
    w = w / w.sum()
    mean_days = float(sum(wi * np.mean(cfg.days[k]) for wi, k in zip(w, kinds)))
    # ~1/3 of scheduled episodes get dropped (no visible effect) or start later (sensor faults wait for daylight):
    # schedule for a higher duty so the realised faulty share lands near the target.
    duty = min(cfg.faulty_share * 1.25, 0.6)
    gap_days = mean_days * (1 - duty) / duty
    out, t = [], rng.exponential(gap_days) * STEPS_PER_DAY
    while t < T - STEPS_PER_DAY:
        k = kinds[rng.choice(len(kinds), p=w)]
        dur = rng.uniform(*cfg.days[k]) * STEPS_PER_DAY
        i0 = int(t)
        i1 = int(min(t + dur, T))
        out.append((k, i0, i1))
        t = i1 + rng.exponential(gap_days) * STEPS_PER_DAY
    return out


def _resimulate_battery(arrays, rows, starts, cf, pf, bp: BatteryParams) -> None:
    """Re-run the self-consumption rule for `rows` from their first fault onset (vectorised over rows)."""
    if not rows:
        return
    rows = np.array(rows)
    start = np.array(starts)
    t0 = int(start.min())
    T = arrays["pv_ac_w"].shape[1]
    pv, load = arrays["pv_ac_w"][rows], arrays["load_w"][rows]
    soc0, batt0 = arrays["soc"][rows].astype(float), arrays["batt_w"][rows].astype(float)
    cap = bp.capacity_kwh[rows]
    e = np.nan_to_num(soc0[:, t0 - 1]) * cap if t0 > 0 else 0.5 * cap
    # The per-step loop (capacity/power derating + apply_power, original series kept before each row's onset) is
    # kernel K7: numpy reference wattsentry.kernels.battery_resim_py, Rust copy when built (#134).
    batt, soc = kernels.battery_resim(pv - load, soc0, batt0, cf[rows], pf[rows], start, e, cap, bp.p_max_kw[rows],
                                      bp.eta_charge[rows], bp.eta_discharge[rows], bp.soc_min[rows],
                                      bp.soc_max[rows], t0, 0.25)
    arrays["batt_w"][rows] = batt
    arrays["soc"][rows] = soc
    arrays["grid_w"][rows] = np.where(np.arange(T)[None] >= start[:, None],
                                      pv - load - batt, arrays["grid_w"][rows])


def inject_batch(ts: pd.DatetimeIndex, arrays: dict[str, np.ndarray], sites: pd.DataFrame,
                 cfg: FaultConfig) -> list[dict]:
    """Apply faults to a batch of sites in place. arrays[col] is (n_sites, T); returns label rows."""
    n, T = arrays["pv_ac_w"].shape
    labels: list[dict] = []
    has_batt = {"soc", "batt_w", "load_w", "grid_w"} <= arrays.keys()
    cf = np.ones((n, T))
    pf = np.ones((n, T))
    resim_rows, resim_start, sensor_eps = [], [], []
    for r in range(n):
        site = sites.iloc[r]
        rng = site_rng(cfg.seed, site.site_id)
        cols = {k: v[r] for k, v in arrays.items()}  # views: edits land in arrays
        first_phys = None
        for kind, i0, i1 in schedule(T, available_kinds(cols, site), rng, cfg):
            if kind == "sensor_fault":
                sensor_eps.append((r, i0, i1, rng))
                continue
            snap = {k: v[i0:i1].copy() for k, v in cols.items()}
            ep = Episode(ts, cols, site, i0, i1, rng)
            if kind == "overvoltage_trip":
                ep.extra["threshold_v"] = overvoltage_threshold(cols["inv_voltage_v"][i0:i1], cols["pv_ac_w"][i0:i1], rng)
                if ep.extra["threshold_v"] is None:
                    continue
            params, severity = APPLY[kind](ep)
            if kind in PHYSICAL and (severity < cfg.min_effect or params.get("trips", 1) == 0):
                for k, v in snap.items():
                    cols[k][i0:i1] = v
                continue
            if kind == "battery_fade":
                if ep.capacity_factor is not None:
                    cf[r] = np.minimum(cf[r], ep.capacity_factor)
                if ep.power_factor is not None:
                    pf[r] = np.minimum(pf[r], ep.power_factor)
            first_phys = i0 if first_phys is None else min(first_phys, i0)
            labels.append(_label(site, kind, ts, i0, i1, severity, params))
        if first_phys is not None and has_batt and float(site.get("battery_kwh", 0) or 0) > 0:
            resim_rows.append(r)
            resim_start.append(first_phys)
        elif first_phys is not None and "grid_w" in arrays and "load_w" in arrays:
            sl = slice(first_phys, T)
            b = arrays["batt_w"][r, sl] if "batt_w" in arrays else 0
            arrays["grid_w"][r, sl] = arrays["pv_ac_w"][r, sl] - arrays["load_w"][r, sl] - b
    if has_batt:
        _resimulate_battery(arrays, resim_rows, resim_start, cf, pf, BatteryParams.from_sites(sites))
    for r, i0, i1, rng in sensor_eps:
        site = sites.iloc[r]
        cols = {k: v[r] for k, v in arrays.items()}
        day = np.flatnonzero(cols["pv_ac_w"][i0:i1] > 0)
        if len(day) == 0:
            continue
        i0 = i0 + int(day[0])  # a sensor fault shows from the first daylight reading
        params, severity = APPLY["sensor_fault"](Episode(ts, cols, site, i0, i1, rng))
        labels.append(_label(site, "sensor_fault", ts, i0, i1, severity, params))
    return labels


def _label(site, kind, ts, i0, i1, severity, params) -> dict:
    return {"asset_id": str(site.site_id), "asset_kind": "site", "fault": kind, "onset": ts[i0],
            "end": ts[i1] if i1 < len(ts) else pd.NaT, "severity": float(round(severity, 4)),
            "params": json.dumps(params, default=lambda o: o.item() if hasattr(o, "item") else str(o))}


def labels_frame(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=["asset_id", "asset_kind", "fault", "onset", "end", "severity", "params"])
    df["onset"] = pd.to_datetime(df["onset"], utc=True)
    df["end"] = pd.to_datetime(df["end"], utc=True)
    df["severity"] = df.severity.astype("float32")
    return df.sort_values(["asset_id", "onset"]).reset_index(drop=True)


def inject(telemetry: pd.DataFrame, sites: pd.DataFrame, cfg: FaultConfig = FaultConfig()
           ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """DataFrame API for any long telemetry (synthetic fleet, or PVDAQ with only ts, site_id, pv_ac_w).
    `sites` needs site_id, and lat/lon (shading), inverter_kw (clipping), battery_* (battery_fade) when present.
    Returns (faulty telemetry, fault_labels)."""
    out, labels = [], []
    sites = sites.set_index("site_id", drop=False)
    for sid, g in telemetry.groupby("site_id", sort=False, observed=True):
        g = g.sort_values("ts")
        ts = pd.DatetimeIndex(g.ts)
        arrays = {c: g[c].to_numpy(np.float64 if c != "inv_status" else np.int16)[None].copy()
                  for c in TELEMETRY_COLS if c in g}
        labels += inject_batch(ts, arrays, sites.loc[[sid]].reset_index(drop=True), cfg)
        g = g.copy()
        for c, v in arrays.items():
            g[c] = v[0].astype(g[c].dtype)
        out.append(g)
    return pd.concat(out, ignore_index=True), labels_frame(labels)
