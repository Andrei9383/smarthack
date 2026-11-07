"""Shared PV physics + Romanian calendar (numpy / pandas / pvlib only, no torch).

Used by the synthetic generator (synth.py), the feature builders (features.py) and the ONNX predictor, so the
pvlib baseline and the model inputs are computed the same way at training and at inference time.
"""

from __future__ import annotations

import datetime as dt
import warnings
from functools import lru_cache

import numpy as np
import pandas as pd

TZ = "Europe/Bucharest"
STEP_MIN = 15
DAY = 96

# Weather channels, in this order, everywhere (synth arrays, Window.weather default order).
WEATHER_CHANNELS = ("ghi", "dni", "dhi", "temp", "cloud", "wind", "snow")
# Window.solar_geometry contract order: zenith (deg), azimuth (deg), clear-sky GHI (W/m2).
GEOM_CHANNELS = ("zenith", "azimuth", "clearsky_ghi")

# Default physics assumptions for the *baseline* pvlib model (what an installer's tool would assume).
DEFAULT_LOSSES = 0.14
DEFAULT_GAMMA = -0.004  # 1/K
NOCT_RISE = 0.033  # K per W/m2 (≈ (NOCT-20)/800 with NOCT 46 °C)
INV_ETA = 0.96


def times_utc(t0: str | pd.Timestamp, n: int) -> pd.DatetimeIndex:
    """n 15-min step starts from t0 (UTC)."""
    start = pd.Timestamp(t0)
    if start.tzinfo is None:
        start = start.tz_localize("UTC")
    return pd.date_range(start.tz_convert("UTC"), periods=n, freq=f"{STEP_MIN}min")


def solar_geometry(times: pd.DatetimeIndex, lat: float, lon: float) -> np.ndarray:
    """[T, 3] = apparent zenith (deg), azimuth (deg), Ineichen clear-sky GHI (W/m2), at the step midpoint."""
    import pvlib

    mid = times + pd.Timedelta(minutes=STEP_MIN / 2)
    sp = pvlib.solarposition.get_solarposition(mid, lat, lon, method="nrel_numpy")
    zen = sp["apparent_zenith"].to_numpy(np.float64)
    az = sp["azimuth"].to_numpy(np.float64)
    cs = clearsky(mid, zen)
    return np.stack([zen, az, cs["ghi"]], axis=1)


def clearsky(times: pd.DatetimeIndex, zenith: np.ndarray, linke: float = 3.0) -> dict[str, np.ndarray]:
    import pvlib

    with warnings.catch_warnings(), np.errstate(divide="ignore", invalid="ignore"):
        warnings.simplefilter("ignore")
        am = pvlib.atmosphere.get_relative_airmass(zenith)
        ama = pvlib.atmosphere.get_absolute_airmass(am)
        dni_extra = np.asarray(pvlib.irradiance.get_extra_radiation(times), dtype=np.float64)
        cs = pvlib.clearsky.ineichen(zenith, ama, linke, dni_extra=dni_extra)
    return {k: np.nan_to_num(np.asarray(cs[k], dtype=np.float64), nan=0.0).clip(min=0.0) for k in ("ghi", "dni", "dhi")}


def decompose(ghi: np.ndarray, zenith: np.ndarray, doy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Erbs GHI -> (DNI, DHI)."""
    import pvlib

    with warnings.catch_warnings(), np.errstate(divide="ignore", invalid="ignore"):
        warnings.simplefilter("ignore")
        out = pvlib.irradiance.erbs(ghi, zenith, doy)
    dni = np.nan_to_num(np.asarray(out["dni"], dtype=np.float64)).clip(min=0.0)
    dhi = np.nan_to_num(np.asarray(out["dhi"], dtype=np.float64)).clip(min=0.0)
    return dni, dhi


def poa(tilt, azimuth, zenith, sun_az, dni, ghi, dhi, albedo: float = 0.2) -> tuple[np.ndarray, np.ndarray]:
    """Isotropic transposition -> (poa_global, poa_direct) W/m2."""
    import pvlib

    with warnings.catch_warnings(), np.errstate(divide="ignore", invalid="ignore"):
        warnings.simplefilter("ignore")
        out = pvlib.irradiance.get_total_irradiance(
            tilt, azimuth, zenith, sun_az, dni, ghi, dhi, albedo=albedo, model="isotropic"
        )
    g = np.nan_to_num(np.asarray(out["poa_global"], dtype=np.float64)).clip(min=0.0)
    b = np.nan_to_num(np.asarray(out["poa_direct"], dtype=np.float64)).clip(min=0.0)
    return g, b


def ac_power_kw(poa_global, temp_air, kwp, inverter_kw, losses=DEFAULT_LOSSES, gamma=DEFAULT_GAMMA,
                wind=None) -> np.ndarray:
    """DC from POA with a NOCT-style cell temperature, then a PVWatts-like inverter that clips at inverter_kw."""
    cool = 1.0 if wind is None else 1.0 / (1.0 + 0.08 * np.clip(np.asarray(wind, dtype=np.float64), 0, 15))
    t_cell = temp_air + NOCT_RISE * poa_global * cool
    dc = kwp * poa_global / 1000.0 * (1.0 + gamma * (t_cell - 25.0)) * (1.0 - losses)
    dc = np.clip(dc, 0.0, None)
    # Part-load efficiency curve (PVWatts shape): lower efficiency at low load.
    pdc0 = inverter_kw / INV_ETA
    zeta = np.clip(dc / np.maximum(pdc0, 1e-6), 1e-4, None)
    eta = INV_ETA / 0.9637 * (-0.0162 * zeta - 0.0059 / zeta + 0.9858)
    ac = np.where(dc > 0, dc * np.clip(eta, 0.0, 1.0), 0.0)
    return np.minimum(ac, inverter_kw)


def pvlib_expected_kw(meta: dict, weather: np.ndarray, geom: np.ndarray) -> np.ndarray:
    """Baseline pvlib AC power (kW) from RECORDED metadata + weather [T, 7] + geometry [T, 3].

    meta keys: kwp_recorded, tilt_recorded, azimuth_recorded, inverter_kw. Arrays may have a leading batch axis
    only if meta values are broadcastable; normally called per site.
    """
    zen, saz = geom[..., 0], geom[..., 1]
    ghi, dni, dhi, temp, wind = weather[..., 0], weather[..., 1], weather[..., 2], weather[..., 3], weather[..., 5]
    g, _ = poa(meta["tilt_recorded"], meta["azimuth_recorded"], zen, saz, dni, ghi, dhi)
    return ac_power_kw(g, temp, meta["kwp_recorded"], meta["inverter_kw"], wind=wind)


def pvlib_clear_kw(meta: dict, times: pd.DatetimeIndex, geom: np.ndarray) -> np.ndarray:
    """Clear-sky pvlib AC power (kW) with recorded metadata at 20 °C: the 'what a perfect day gives' profile."""
    zen, saz = geom[..., 0], geom[..., 1]
    cs = clearsky(times + pd.Timedelta(minutes=STEP_MIN / 2), zen)
    g, _ = poa(meta["tilt_recorded"], meta["azimuth_recorded"], zen, saz, cs["dni"], cs["ghi"], cs["dhi"])
    return ac_power_kw(g, 20.0, meta["kwp_recorded"], meta["inverter_kw"])


# --- Romanian calendar -------------------------------------------------------------------------------------
def orthodox_easter(year: int) -> dt.date:
    """Meeus' Julian algorithm shifted to the Gregorian calendar (+13 days, valid 1900-2099)."""
    a, b, c = year % 4, year % 7, year % 19
    d = (19 * c + 15) % 30
    e = (2 * a + 4 * b - d + 34) % 7
    month = (d + e + 114) // 31
    day = (d + e + 114) % 31 + 1
    return dt.date(year, month, day) + dt.timedelta(days=13)


@lru_cache(maxsize=64)
def ro_holidays(year: int) -> frozenset:
    easter = orthodox_easter(year)
    fixed = [(1, 1), (1, 2), (1, 24), (5, 1), (6, 1), (8, 15), (11, 30), (12, 1), (12, 25), (12, 26)]
    if year >= 2024:  # Bobotează and Sf. Ioan became public holidays in 2024
        fixed += [(1, 6), (1, 7)]
    days = {dt.date(year, m, d) for m, d in fixed}
    for off in (-2, 0, 1, 49, 50):  # Good Friday, Easter Sunday + Monday, Rusalii Sunday + Monday
        days.add(easter + dt.timedelta(days=off))
    return frozenset(days)


CAL_CHANNELS = ("hour_sin", "hour_cos", "dow_sin", "dow_cos", "off_day", "holiday", "doy_sin", "doy_cos")


def calendar_features(times: pd.DatetimeIndex) -> np.ndarray:
    """[T, 8] local-time (Europe/Bucharest) calendar features, DST-aware."""
    loc = times.tz_convert(TZ)
    hour = (loc.hour + loc.minute / 60.0).to_numpy(np.float64)
    dow = loc.dayofweek.to_numpy()
    dates = loc.date
    hol_set = set().union(*(ro_holidays(y) for y in sorted({d.year for d in dates})))
    hol = np.fromiter((d in hol_set for d in dates), dtype=bool, count=len(dates))
    doy = loc.day_of_year.to_numpy(np.float64)
    w = 2 * np.pi
    return np.stack([
        np.sin(w * hour / 24), np.cos(w * hour / 24),
        np.sin(w * dow / 7), np.cos(w * dow / 7),
        ((dow >= 5) | hol).astype(np.float64), hol.astype(np.float64),
        np.sin(w * doy / 365.25), np.cos(w * doy / 365.25),
    ], axis=1)
