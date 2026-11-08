"""Fault baselines on the pvlib residual (M4 SiteHealth fallback): a threshold rule and a CUSUM.

Daily performance ratio r[s, d] = measured / pvlib(recorded metadata) energy over daylight steps. A day is
- uninformative (skipped, CUSUM state carried) when pvlib expects < 0.3 kWh/kWp or there is snow (> 1 cm),
- missing (NaN, CUSUM reset) when more than half of the daylight readings are null.
Each site's ratio is divided by its own reference level (median r over the calibration = training days), so wrong
recorded metadata does not look like a fault: q = r / ref, x = 1 - q (> 0 = under-producing).

threshold rule: alarm on day d when q < theta on d-2, d-1 and d (theta = 0.8, i.e. "< 80 % for 3 days").
CUSUM: two-sided CUSUM on x (cusum.py); alarm while s_pos >= h.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import pandas as pd

from wattsentry.contracts.envelope import PvFault, Prediction, Window

from .cusum import cusum_two_sided

DAYLIGHT_ZENITH = 85.0
MIN_KWH_PER_KWP = 0.3
SNOW_CM = 1.0
THETA = 0.8
RUN_DAYS = 3


# ------------------------------------------------------------------------------------------------ predictor
INV_OVERVOLTAGE = 37  # inverter code "grid voltage too high" (sim/faults/kinds.py)


class ThresholdCusumPredictor:
    """M4 fallback as a contracts.Predictor (same Window conventions as SiteHealth).

    Today's ratio = window.meta["telemetry"]["pv_ac_w"] / pvlib(window.weather, recorded metadata) over daylight.
    Past ratios come from window.residual_summary[:, 0] (oldest first) when given, otherwise from the ratios this
    predictor computed for the same asset on earlier days (it remembers them, so calling it day by day per site
    reproduces the bulk rule). q = ratio / ratio_ref (asset["ratio_ref"], else the median of the past ratios, else 1).
    Rules: threshold (q < 0.8 on the last 3 informative days), CUSUM on 1 - q, inverter overvoltage code, no data.
    fault_probs: the alarm probability goes to one heuristic class (no data / ~0 output -> sensor_fault, overvoltage
    code -> overvoltage_trip, threshold -> string_outage, CUSUM only -> soiling); extra["p_fault_threshold"] = 0.5.
    """

    model = "threshold_cusum"
    P_THRESHOLD = 0.5

    def __init__(self, k: float = 0.1, h: float = 0.1, theta: float = THETA, model_version: str = "threshold_cusum-1"):
        self.k, self.h, self.theta = k, h, theta
        self.model_version = model_version
        self._memory: dict[str, dict[str, tuple[float, bool]]] = {}

    def _today(self, asset, window: Window, H: int, expected: np.ndarray):
        """(ratio or NaN if missing, informative, inverter overvoltage code seen)."""
        from .lgbm import _to_steps, parse_weather, power_scale, weather_channels
        from .physical import recorded_meta, solar_geometry

        tel = (window.meta or {}).get("telemetry") or {}
        if "pv_ac_w" not in tel or window.weather is None:
            return np.nan, False, False
        meas = _to_steps(np.asarray(tel["pv_ac_w"], np.float64), H) / power_scale(window)
        status = np.nan_to_num(np.asarray(tel.get("inv_status", []), np.float64))
        ov = bool((status == INV_OVERVOLTAGE).any())
        meta = recorded_meta(asset)
        g = solar_geometry(pd.date_range(pd.Timestamp(window.t0), periods=H, freq="15min"), meta["lat"], meta["lon"])
        day = g["zenith"] < DAYLIGHT_ZENITH
        ok = day & np.isfinite(meas)
        snow = parse_weather(window.weather, H, weather_channels(window))["snow_cm"]
        informative = expected[day].sum() * 0.25 > MIN_KWH_PER_KWP * meta["kwp"] and np.nanmean(snow) <= SNOW_CM
        if ok.sum() < 0.5 * max(day.sum(), 1):
            return np.nan, True, ov
        return float(meas[ok].sum() / max(expected[ok].sum(), 1e-9)), bool(informative), ov

    def predict(self, asset: Mapping[str, Any], window: Window) -> Prediction:
        from .persistence import PvlibPredictor

        H = int(window.horizon)
        aid = str(asset.get("site_id", asset.get("asset_id", "?")))
        has_tel = bool((window.meta or {}).get("telemetry"))
        q50 = PvlibPredictor().expected_kw(asset, window) if window.weather is not None else np.zeros(H)
        r0, inf0, ov = self._today(asset, window, H, q50)
        day0 = pd.Timestamp(window.t0).normalize()
        if window.residual_summary is not None:
            rs = np.asarray(window.residual_summary, np.float64)
            past = list(rs[:, 0] if rs.ndim == 2 else rs)
            past_inf = [True] * len(past)
        else:
            mem = self._memory.get(aid, {})
            keys = [(day0 - pd.Timedelta(days=i)).strftime("%Y-%m-%d") for i in range(14, 0, -1)]
            past = [mem[k][0] for k in keys if k in mem]
            past_inf = [mem[k][1] for k in keys if k in mem]
        if has_tel:
            self._memory.setdefault(aid, {})[day0.strftime("%Y-%m-%d")] = (r0, inf0)
        if has_tel or not past:
            r = np.array(past + [r0], np.float64)
            inf = np.array(past_inf + [inf0], bool)
        else:  # only a residual summary: its last row is the assessed day
            r, inf = np.array(past, np.float64), np.array(past_inf, bool)
        hist = r[:-1][inf[:-1]]
        hist = hist[np.isfinite(hist)]
        ref = asset.get("ratio_ref")
        ref = float(ref) if ref is not None else (float(np.median(hist)) if len(hist) >= 3 else 1.0)
        q = r / max(ref, 1e-6)
        qs = q[inf | np.isnan(q)]  # skipped (uninformative) days drop out, missing ones reset the CUSUM
        last = qs[-RUN_DAYS:]
        th = len(last) == RUN_DAYS and bool(np.isfinite(last).all()) and bool(last.max() < self.theta)
        s_last = float(cusum_two_sided(1.0 - qs, self.k, self.h)[0][-1]) if len(qs) else 0.0
        cu = s_last >= self.h
        q0 = q[-1]
        if ov:
            p, cls = 0.8, PvFault.OVERVOLTAGE_TRIP.value
        elif not np.isfinite(q0):
            p, cls = (0.6 if has_tel or window.residual_summary is not None else 0.0), PvFault.SENSOR_FAULT.value
        elif th or cu:
            p = 0.9 if th else 0.5 + 0.25 * min(1.0, s_last / self.h)
            cls = (PvFault.SENSOR_FAULT.value if q0 < 0.1 else
                   PvFault.STRING_OUTAGE.value if th else PvFault.SOILING.value)
        else:
            day_score = float(np.clip((1 - q0) / (1 - self.theta), 0, 1))
            p, cls = 0.45 * max(day_score, s_last / self.h), PvFault.SOILING.value
        probs = {f.value: 0.0 for f in PvFault}
        probs[PvFault.HEALTHY.value] = 1 - p
        probs[cls] += p
        return Prediction(asset_id=aid, model=self.model, model_version=self.model_version, t0=window.t0, horizon=H,
                          q10=q50, q50=q50, q90=q50, fault_probs=probs, ood_score=float(np.isnan(r).mean()),
                          extra={"p_fault_threshold": self.P_THRESHOLD, "ratio_today": None if np.isnan(r0) else r0,
                                 "ratio_ref": ref, "threshold_alarm": bool(th), "cusum_alarm": bool(cu),
                                 "cusum_s": s_last, "overvoltage_code": ov, "rule": "heuristic class split"})
