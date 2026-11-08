"""Next-15-min forecasts per home and feeder.

M1 pvnet_fc and M2 LoadNet are reused as in the replay: asked once per home per local day (t0 = local midnight,
history before the day + forecast weather), then sliced per step, so no reading at or after t0 is used (never the
pv_now nowcast, which reads actual weather). Three methods are stored every tick:
- `model`:       the day-ahead envelope's step for the target ts.
- `persistence`: PV = last reading x clear-sky-index ratio (cos zenith), load = last reading.
- `corrected`:   model + ALPHA x last residual (actual - model q50 at the step that just ended), the headline.
A home whose last reading is missing falls back to `model` for the other two methods.

Net (export after the battery rule, + = export) per quantile: the self-consumption rule from the current battery
state on (pv q10, load q90), (pv q50, load q50), (pv q90, load q10). Feeder rows sum the homes: summed q10/q90 is a
conservative band, not a feeder quantile.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Any, Mapping, Optional

from wattsentry.contracts import DAY_STEPS, Window
from wattsentry.contracts.opt import self_consumption_step

from .source import STEP_S, Reading

ALPHA = 0.9  # residual weight of the `corrected` method (not fit on calibration days)
NIGHT_CZ = 0.02  # cos zenith below this: no PV
MAX_CSI_RATIO = 3.0

# one-step version of the notification grades (packages/api/src/notify/forecast.ts THRESHOLDS)
SEVERE_PEAK_RATIO = 1.3
WATCH_RATIO = 0.9


def iso(ts: int) -> str:
    return datetime.fromtimestamp(int(ts), tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def cos_zenith(ts: float, lat: float, lon: float) -> float:
    """NOAA solar position (Spencer series), good to ~0.5 deg: enough for a clear-sky-index ratio."""
    dt = datetime.fromtimestamp(ts, tz=timezone.utc)
    hour = dt.hour + dt.minute / 60 + dt.second / 3600
    g = 2 * math.pi / 365 * (dt.timetuple().tm_yday - 1 + (hour - 12) / 24)
    decl = (0.006918 - 0.399912 * math.cos(g) + 0.070257 * math.sin(g) - 0.006758 * math.cos(2 * g)
            + 0.000907 * math.sin(2 * g) - 0.002697 * math.cos(3 * g) + 0.00148 * math.sin(3 * g))
    eqt = 229.18 * (0.000075 + 0.001868 * math.cos(g) - 0.032077 * math.sin(g) - 0.014615 * math.cos(2 * g)
                    - 0.040849 * math.sin(2 * g))
    ha = math.radians((hour * 60 + eqt + 4 * lon) / 4 - 180)
    la = math.radians(lat)
    return math.sin(la) * math.sin(decl) + math.cos(la) * math.cos(decl) * math.cos(ha)


class DayAhead:
    """The world's own predictors (SimWorld: promoted ONNX M1 pvnet_fc / M2 LoadNet; MockWorld: MockPredictor),
    asked once per home per local day exactly like the replay's loop, then sliced per step."""

    def __init__(self, world: Any, sites: Mapping[str, Any]):
        self.world = world
        self.sites = sites
        preds = world.default_predictors()
        self.pred = {"pv": preds["pv"], "load": preds["load"]}
        self.cache: dict[tuple[str, str, int], Any] = {}

    def versions(self) -> dict[str, str]:
        return {k: str(getattr(p, "model_version", type(p).__name__)) for k, p in self.pred.items()}

    def q(self, quantity: str, sid: str, ts: int) -> Optional[tuple[float, float, float, str, int]]:
        """(q10, q50, q90, model_version, is_fallback) for the step starting at ts, None outside the world."""
        w = self.world
        d = (ts - w.day_ts(0)) // 86400
        if not 0 <= d < w.days:
            return None
        key = (quantity, sid, d)
        p = self.cache.get(key)
        if p is None:
            p = self.pred[quantity].predict(self.sites[sid].row, Window(t0=w.day_t0(d), horizon=DAY_STEPS))
            self.cache[key] = p
            for old in [k for k in self.cache if k[2] < d - 1]:  # keep yesterday for the residual at midnight
                del self.cache[old]
        k = (ts - w.day_ts(d)) // STEP_S
        fb = int((p.extra or {}).get("is_fallback", 0))
        return max(0.0, p.q10[k]), max(0.0, p.q50[k]), max(0.0, p.q90[k]), p.model_version, fb


def flag(q10: float, q50: float, q90: float, limit: float) -> str:
    if limit <= 0:
        return "none"
    if q10 > limit or q50 >= SEVERE_PEAK_RATIO * limit:
        return "severe"
    if q50 > limit:
        return "congested"
    if q90 > limit or q50 >= WATCH_RATIO * limit:
        return "watch"
    return "none"


def err_flag(actual: float, q10: float, q50: float, q90: float, n_sites: int = 1) -> int:
    """Large miss: outside the band and more than max(25% of the level, 1 kW per home) off."""
    err = actual - q50
    outside = actual < q10 - 1e-9 or actual > q90 + 1e-9
    return int(outside and abs(err) > max(0.25 * max(abs(q50), abs(actual)), 1.0 * n_sites))


def _site_pv_load(da: DayAhead, sid: str, site: Any, ts: int, last: Optional[Reading]):
    """{method: {"pv": (q10,q50,q90), "load": (...)}} for the step starting at ts, + (version, fallback)."""
    pv_m, load_m = da.q("pv", sid, ts), da.q("load", sid, ts)
    if pv_m is None or load_m is None:
        return None, {}
    out = {"model": {"pv": pv_m[:3], "load": load_m[:3]}}
    meta = {"pv": pv_m[3:], "load": load_m[3:]}
    have = last is not None and last.pv_kw is not None and last.load_kw is not None
    prev_pv, prev_load = (da.q("pv", sid, ts - STEP_S), da.q("load", sid, ts - STEP_S)) if have else (None, None)
    if not have or prev_pv is None or prev_load is None:
        out["persistence"] = {"pv": (pv_m[1],) * 3, "load": (load_m[1],) * 3}
        out["corrected"] = out["model"]
        return out, meta
    # persistence: clear-sky-index ratio on cos zenith at the step midpoints
    lat, lon = float(site.row.get("lat") or 45.9), float(site.row.get("lon") or 25.0)
    cz_now, cz_next = cos_zenith(ts - STEP_S / 2, lat, lon), cos_zenith(ts + STEP_S / 2, lat, lon)
    if cz_next <= NIGHT_CZ:
        pv_p = 0.0
    elif cz_now <= NIGHT_CZ:
        pv_p = last.pv_kw
    else:
        pv_p = last.pv_kw * min(MAX_CSI_RATIO, cz_next / cz_now)
    out["persistence"] = {"pv": (pv_p,) * 3, "load": (last.load_kw,) * 3}
    # corrected: shift the model's band by ALPHA x the residual of the step that just ended
    r_pv, r_load = last.pv_kw - prev_pv[1], last.load_kw - prev_load[1]
    night = pv_m[2] <= 0.0
    out["corrected"] = {
        "pv": tuple(0.0 if night else max(0.0, q + ALPHA * r_pv) for q in pv_m[:3]),
        "load": tuple(max(0.0, q + ALPHA * r_load) for q in load_m[:3]),
    }
    return out, meta


def site_rows(da: DayAhead, sid: str, site: Any, ts: int, last: Optional[Reading], e_kwh: float) -> list[dict]:
    """live_predictions rows (without run_id) for one home and the step starting at ts."""
    pl, meta = _site_pv_load(da, sid, site, ts, last)
    if pl is None:
        return []
    rows = []
    for method, v in pl.items():
        pv, load = v["pv"], v["load"]
        net = tuple(self_consumption_step(site.battery, e_kwh, pv[i], load[2 - i])[2] for i in range(3))
        lo, hi = min(net), max(net)
        net = (lo, min(max(net[1], lo), hi), hi)
        for qty, q in (("pv", pv), ("load", load), ("net", net)):
            ver, fb = meta["load" if qty == "load" else "pv"] if qty != "net" else (
                f"{meta['pv'][0]}+{meta['load'][0]}", max(meta["pv"][1], meta["load"][1]))
            rows.append({"ts": iso(ts), "t0": iso(ts), "level": "site", "entity_id": sid, "feeder_id": site.feeder_id,
                         "quantity": qty, "method": method, "q10": round(q[0], 4), "q50": round(q[1], 4),
                         "q90": round(q[2], 4), "limit_kw": None, "flag": None, "n_sites": 1,
                         "model_version": ver, "is_fallback": fb})
    return rows


def feeder_rows(site_rows_: list[dict], feeders: Mapping[str, Any], ts: int) -> list[dict]:
    """Sum the homes per feeder / quantity / method; flag the net rows against the feeder limit."""
    acc: dict[tuple[str, str, str], dict] = {}
    for r in site_rows_:
        key = (r["feeder_id"], r["quantity"], r["method"])
        a = acc.setdefault(key, {"q10": 0.0, "q50": 0.0, "q90": 0.0, "n": 0, "fb": 0, "ver": r["model_version"]})
        a["q10"] += r["q10"]
        a["q50"] += r["q50"]
        a["q90"] += r["q90"]
        a["n"] += 1
        a["fb"] = max(a["fb"], r["is_fallback"])
    rows = []
    for (fid, qty, method), a in sorted(acc.items()):
        limit = float(feeders[fid].spec.g_kw)
        rows.append({"ts": iso(ts), "t0": iso(ts), "level": "feeder", "entity_id": fid, "feeder_id": fid,
                     "quantity": qty, "method": method, "q10": round(a["q10"], 4), "q50": round(a["q50"], 4),
                     "q90": round(a["q90"], 4), "limit_kw": round(limit, 3),
                     "flag": flag(a["q10"], a["q50"], a["q90"], limit) if qty == "net" else None,
                     "n_sites": a["n"], "model_version": a["ver"], "is_fallback": a["fb"]})
    return rows
