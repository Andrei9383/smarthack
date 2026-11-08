"""Persistence and pvlib baselines, in bulk (wide [n, d, 96] arrays) and as contract Predictors.

PV:   `pv_d1`  same step yesterday;  `pv_ref7`  per-step 2nd-largest of the 7 reference days scaled by
      today's / the reference days' clear-sky ratio ("smart" reference-window persistence);
      `pvlib` physical model on the window's weather with recorded metadata.
Load: `load_d1` same step yesterday;  `load_w1` same step last week.

Point baselines put q10 = q50 = q90 except pvlib, whose band is q50 x (train-set P10 / P90 of actual / pvlib).
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import pandas as pd

from wattsentry.contracts.envelope import Prediction, Window, parse_utc

from .data import D, FleetArrays
from .features import REF_DAYS, _second_largest
from .lgbm import parse_weather, power_scale, weather_channels
from .physical import clearsky_power_kw, pv_power_kw, recorded_meta, solar_geometry


def shift_days(arr: np.ndarray, k: int) -> np.ndarray:
    """[n, d, 96] -> value k days earlier (NaN for the first k days)."""
    out = np.full_like(arr, np.nan)
    out[:, k:] = arr[:, :-k]
    return out


def pv_ref7_bulk(a: FleetArrays) -> np.ndarray:
    out = np.full_like(a.pv_kw, np.nan)
    for d in range(REF_DAYS, a.pv_kw.shape[1]):
        env = _second_largest(a.pv_kw[:, d - REF_DAYS:d], axis=1)
        cs = np.nanmean(a.pvlib_cs_kw[:, d - REF_DAYS:d], axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            out[:, d] = np.where(cs > 1e-3, env / cs, 0.0) * a.pvlib_cs_kw[:, d]
    return out


def pvlib_band(a: FleetArrays, day_mask: np.ndarray, site_mask: np.ndarray) -> tuple[float, float]:
    """P10 / P90 of actual / pvlib on healthy daylight training steps with meaningful power."""
    sl = (slice(None), day_mask)
    p, e = a.pvlib_kw[sl][site_mask], a.pv_kw[sl][site_mask]
    ok = (p > 0.1 * a.sites.kwp_recorded.to_numpy()[site_mask, None, None]) & np.isfinite(e) & \
        (a.fault[sl][site_mask] == 0)
    r = e[ok] / p[ok]
    lo, hi = np.quantile(r, [0.1, 0.9])
    return float(min(lo, 1.0)), float(max(hi, 1.0))


class PvlibPredictor:
    """Physical baseline as a Predictor: pvlib on window.weather with recorded metadata."""

    model = "pvlib"

    def __init__(self, band: tuple[float, float] = (0.8, 1.1), model_version: str = "pvlib-recorded"):
        self.band = band
        self.model_version = model_version

    def expected_kw(self, asset: Mapping[str, Any], window: Window) -> np.ndarray:
        H = int(window.horizon)
        meta = recorded_meta(asset)
        ts = pd.date_range(pd.Timestamp(parse_utc(window.t0)), periods=H, freq="15min")
        g = solar_geometry(ts, meta["lat"], meta["lon"])
        wx = parse_weather(window.weather, H, weather_channels(window))
        if "_need_split" in wx:
            import pvlib
            sp = pvlib.irradiance.erbs(wx["ghi"], g["zenith"], g["doy"])
            wx["dni"], wx["dhi"] = np.asarray(sp["dni"]), np.asarray(sp["dhi"])
        return np.nan_to_num(pv_power_kw(meta, g, wx["ghi"], wx["dni"], wx["dhi"], wx["temp_c"],
                                         wx["wind10_ms"])["ac_kw"])

    def predict(self, asset: Mapping[str, Any], window: Window) -> Prediction:
        q50 = self.expected_kw(asset, window)
        return Prediction(asset_id=str(asset.get("site_id", "?")), model=self.model, model_version=self.model_version,
                          t0=window.t0, horizon=int(window.horizon), q10=q50 * self.band[0], q50=q50,
                          q90=q50 * self.band[1], extra={"kind": "pv"})


class PersistencePredictor:
    """kind: pv_d1 | pv_ref7 (window.reference, W) | load_d1 | load_w1 (window.history, W). Output kW."""

    model = "persistence"
    model_version = "persistence-1"

    def __init__(self, kind: str = "pv_ref7"):
        if kind not in ("pv_d1", "pv_ref7", "load_d1", "load_w1"):
            raise ValueError(kind)
        self.kind = kind

    def predict(self, asset: Mapping[str, Any], window: Window) -> Prediction:
        H = int(window.horizon)
        src = window.reference if self.kind.startswith("pv") else window.history
        x = np.asarray(src if src is not None else [], np.float64).ravel() / power_scale(window)
        need = REF_DAYS * D
        x = np.concatenate([np.full(max(0, need - len(x)), np.nan), x])[-need:].reshape(REF_DAYS, D)
        if self.kind in ("pv_d1", "load_d1"):
            q = x[-1, :H]
        elif self.kind == "load_w1":
            q = x[0, :H]
        else:
            meta = recorded_meta(asset)
            t0 = pd.Timestamp(parse_utc(window.t0))
            ts = pd.date_range(t0 - pd.Timedelta(days=REF_DAYS), periods=(REF_DAYS + 1) * D, freq="15min")
            cs = clearsky_power_kw(meta, solar_geometry(ts, meta["lat"], meta["lon"])).reshape(REF_DAYS + 1, D)
            env = _second_largest(x, axis=0)
            m = np.nanmean(cs[:-1], axis=0)
            with np.errstate(invalid="ignore", divide="ignore"):
                q = (np.where(m > 1e-3, env / m, 0.0) * cs[-1])[:H]
        q = np.nan_to_num(np.clip(q, 0, None))
        return Prediction(asset_id=str(asset.get("site_id", "?")), model=self.model,
                          model_version=f"{self.model_version}-{self.kind}", t0=window.t0, horizon=H, q10=q, q50=q,
                          q90=q, ood_score=float(np.isnan(x).mean()), extra={"kind": self.kind})
