"""Curve rules: WindMon's rules + binned power curve + park context + per-turbine thermal baseline (no learning).
Rule definitions are in the module-level `RULES_DOC` (copied into the JSON report)."""

from __future__ import annotations

import numpy as np

from sentinel.model.wind.data import AMBIENT, CLASS_INDEX, CLASSES, POWER, TEMP_SENSORS, runs_by_class
from sentinel.model.wind.features import rmean

from .curve import BinnedCurve, park_inputs
from .windmon_rules import windmon_flags

RULES_DOC = """per 15-min step, one park at a time; pf = power / rated, met curve = binned curve on meteo wind:
data_gap: WindMon DATA_GAP or SUSPECT_DATA.
stop: pf < stop_frac while met P50 > min_expected (met wind: a broken anemometer cannot hide it).
low: pf < met P10 - low_margin while met P50 > min_expected.
icing: low, ambient <= icing_t_c, RH >= icing_rh or precipitation.
curtailment: low or stop while >= curtail_park_frac of the park's reporting turbines are low/stopped (park context).
anemometer_fault: |v_nac - v_met| / max(v_met, 3) > anem_rel while pf is inside the met band, on >= 60% of 4 h.
overheating: 1-day mean of (rise over ambient - model's expected rise at the 4-h load - turbine's own baseline
  21 days ending 7 days ago) > overheat_c.
underperformance: 1-day mean of (pf - met P50) on producing steps < under_resid.
priority: data_gap > icing > curtailment > fault_stop > anemometer > overheating > underperformance; runs shorter
than MIN_RUN dropped."""

PARAMS = dict(stop_frac=0.01, min_expected=0.05, low_margin=0.05, curtail_park_frac=0.3, icing_t_c=1.5, icing_rh=85.0,
              anem_rel=0.25, overheat_c=6.5, under_resid=-0.08)
MIN_RUN = {"fault_stop": 2, "curtailment": 2, "icing": 4, "anemometer_fault": 16, "overheating": 16,
           "underperformance": 16}
PRIORITY = ("data_gap", "icing", "curtailment", "fault_stop", "anemometer_fault", "overheating", "underperformance")
DAY, BASE, LAG = 96, 21 * 96, 7 * 96


def detect_park(curve: BinnedCurve, turbines: list[dict], sensors: dict, err: np.ndarray, meteo: dict,
                weather: dict | None = None, params: dict | None = None) -> np.ndarray:
    """WindAnomaly index [T, n] for all turbines of one park (the park context needs all of them)."""
    P = {**PARAMS, **(params or {})}
    weather = weather or {}
    rated = np.array([float(t["rated_kw"]) for t in turbines])
    pf = np.asarray(sensors[POWER], np.float64) / rated[None]
    T, n = pf.shape
    v = park_inputs(sensors, meteo, weather, T, n)
    qm = curve.predict_block(turbines, v, "met")
    wm = windmon_flags({k: np.asarray(x, np.float32) for k, x in sensors.items()}, np.asarray(err), rated)
    gap = wm["DATA_GAP"] | wm["SUSPECT_DATA"]
    valid = np.isfinite(pf) & ~gap
    expect = valid & (qm[..., 1] > P["min_expected"])  # NaN compares False
    stop = expect & (pf < P["stop_frac"])
    low = expect & (pf < qm[..., 0] - P["low_margin"])
    n_rep = valid.sum(1, keepdims=True)
    curtail = (low | stop) & ((low | stop).sum(1, keepdims=True) >= P["curtail_park_frac"] * np.maximum(n_rep, 3))

    def w(k):
        return np.asarray(weather.get(k, np.full(T, np.nan)), np.float64).reshape(T, -1)

    amb = np.asarray(sensors.get(AMBIENT, np.full((T, n), np.nan)), np.float64)
    amb = np.where(np.isfinite(amb), amb, w("temp_c"))
    icing = low & (amb <= P["icing_t_c"]) & ((w("rh_pct") >= P["icing_rh"]) | (w("precip_mm") > 0))

    in_band = valid & (pf >= qm[..., 0] - P["low_margin"]) & (pf <= qm[..., 2] + P["low_margin"])
    dis = np.abs(v["nacelle"] - v["met"]) / np.maximum(v["met"], 3.0) > P["anem_rel"]
    anem = valid & (rmean(np.where(valid, dis & in_band, np.nan), 16) >= 0.6)

    over = np.zeros((T, n), bool)
    load = rmean(np.where(valid, pf, np.nan), 16)  # thermal lag
    for s in TEMP_SENSORS:
        exp = np.stack([curve.expected_rise(curve.key_for(t), s, load[:, j]) for j, t in enumerate(turbines)], 1)
        res = np.where(valid, np.asarray(sensors[s], np.float64) - amb - exp, np.nan)
        base = np.zeros_like(res)
        base[LAG:] = rmean(res, BASE)[: T - LAG]
        over |= valid & (rmean(res - base, DAY) > P["overheat_c"])

    prod = expect & ~stop & ~curtail
    under = prod & (rmean(np.where(prod, pf - qm[..., 1], np.nan), DAY) < P["under_resid"]) \
        & (rmean(prod.astype(float), DAY) > 0.25)
    flags = {"data_gap": gap, "icing": icing, "curtailment": curtail, "fault_stop": stop, "anemometer_fault": anem,
             "overheating": over, "underperformance": under}
    out = np.zeros((T, n), np.uint8)
    for c in reversed(PRIORITY):
        out[flags[c]] = CLASS_INDEX[c]
    for j in range(n):
        for c, s, e in runs_by_class(out[:, j]):
            if e - s < MIN_RUN.get(CLASSES[c], 1):
                out[s:e, j] = 0
    return out


def detect_all(curve: BinnedCurve, data, params: dict | None = None) -> np.ndarray:
    """WindAnomaly index [T, N] for a whole WindData, park by park."""
    out = np.zeros((data.n_steps, len(data.turbines)), np.uint8)
    for p in range(len(data.park_ids)):
        c = np.flatnonzero(data.park_of == p)
        out[:, c] = detect_park(curve, [data.turbines[j] for j in c], {k: x[:, c] for k, x in data.sensors.items()},
                                data.err[:, c], {k: x[:, c] for k, x in data.meteo.items()},
                                {k: x[:, p] for k, x in data.weather.items()}, params)
    return out
