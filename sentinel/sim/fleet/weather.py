"""Weather per locality at 15 min.

Source "raw": the committed Open-Meteo / ERA5 Parquet from #9 (see RAW_PATH and RAW_COLUMNS below).
Source "synthetic": a deterministic stand-in with Romanian climatology (spatially correlated cloud days, seasonal
and diurnal temperature, snow), so the simulator runs before #9 lands. The manifest records which one was used.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from . import solar
from .localities import Locality

REPO = Path(__file__).resolve().parents[3]
RAW_PATH = REPO / "data" / "raw" / "weather_localities.parquet"
# Expected long format from #9 (hourly or 15-min, UTC): one row per (time, locality_id).
RAW_COLUMNS = ["time", "locality_id", "ghi", "temp_air", "wind_speed_10m", "snow_depth_cm"]

VARS = ("ghi", "dni", "dhi", "temp_air", "wind_speed", "snow_depth_cm", "zenith", "azimuth")


@dataclass
class Weather:
    ts: pd.DatetimeIndex                 # 15-min UTC, interval start
    locality_ids: list[str]
    data: dict[str, np.ndarray]          # each (n_localities, T) float32
    source: str

    def slice(self, ts: pd.DatetimeIndex) -> "Weather":
        i0 = self.ts.get_loc(ts[0])
        sl = slice(i0, i0 + len(ts))
        return Weather(ts, self.locality_ids, {k: v[:, sl] for k, v in self.data.items()}, self.source)


def load_weather(locs: list[Locality], ts: pd.DatetimeIndex, seed: int, source: str = "auto") -> Weather:
    if source == "raw" or (source == "auto" and RAW_PATH.exists()):
        return _from_raw(locs, ts)
    return _synthetic(locs, ts, seed)


def _geometry(locs: list[Locality], ts: pd.DatetimeIndex) -> tuple[np.ndarray, np.ndarray]:
    zen = np.empty((len(locs), len(ts)), np.float32)
    az = np.empty_like(zen)
    for i, loc in enumerate(locs):
        zen[i], az[i] = solar.solar_position(ts, loc.lat, loc.lon)
    return zen, az


def _finish(locs, ts, zen, az, ghi, temp, wind, snow, source) -> Weather:
    doy = ts.dayofyear.to_numpy()
    dni = np.empty_like(ghi)
    dhi = np.empty_like(ghi)
    for i in range(len(locs)):
        dni[i], dhi[i] = solar.erbs(ghi[i], zen[i], doy)
    data = dict(ghi=ghi, dni=dni, dhi=dhi, temp_air=temp, wind_speed=wind, snow_depth_cm=snow, zenith=zen, azimuth=az)
    return Weather(ts, [l.id for l in locs], {k: np.asarray(v, np.float32) for k, v in data.items()}, source)


def _from_raw(locs: list[Locality], ts: pd.DatetimeIndex) -> Weather:
    df = pd.read_parquet(RAW_PATH, columns=RAW_COLUMNS)
    df["time"] = pd.to_datetime(df["time"], utc=True)
    zen, az = _geometry(locs, ts)
    out = {k: np.empty((len(locs), len(ts)), np.float32) for k in ("ghi", "temp_air", "wind_speed_10m", "snow_depth_cm")}
    for i, loc in enumerate(locs):
        g = df[df.locality_id == loc.id].set_index("time").sort_index()
        if g.empty:
            raise ValueError(f"{RAW_PATH.name} has no rows for locality {loc.id}")
        g = g.reindex(g.index.union(ts)).interpolate("time", limit_direction="both").reindex(ts)
        for k in out:
            out[k][i] = g[k].to_numpy()
    # Hourly ERA5 GHI is an hourly mean: rescale to 15 min with the clear-sky index (k = ghi / clear-sky).
    cs = np.stack([solar.clearsky_ghi(z) for z in zen])
    k = np.clip(np.divide(out["ghi"], cs, out=np.zeros_like(cs), where=cs > 20), 0, 1.1)
    ghi = k * cs
    return _finish(locs, ts, zen, az, ghi, out["temp_air"], out["wind_speed_10m"], out["snow_depth_cm"], "raw")


def _correlated_normals(rng, locs: list[Locality], n: int, corr_km: float) -> np.ndarray:
    """(n_loc, n) standard normals, correlated in space by exp(-distance / corr_km)."""
    lat = np.radians([l.lat for l in locs])
    lon = np.radians([l.lon for l in locs])
    d = 6371 * np.arccos(np.clip(np.sin(lat)[:, None] * np.sin(lat) +
                                 np.cos(lat)[:, None] * np.cos(lat) * np.cos(lon[:, None] - lon), -1, 1))
    chol = np.linalg.cholesky(np.exp(-d / corr_km) + 1e-6 * np.eye(len(locs)))
    return chol @ rng.standard_normal((len(locs), n))


def _ar1_days(eps: np.ndarray, phi: float) -> np.ndarray:
    out = np.empty_like(eps)
    out[:, 0] = eps[:, 0]
    s = np.sqrt(1 - phi**2)
    for t in range(1, eps.shape[1]):
        out[:, t] = phi * out[:, t - 1] + s * eps[:, t]
    return out


def _synthetic(locs: list[Locality], ts: pd.DatetimeIndex, seed: int) -> Weather:
    rng = np.random.default_rng([seed, 101])
    n_loc, T = len(locs), len(ts)
    zen, az = _geometry(locs, ts)
    cs = np.stack([solar.clearsky_ghi(z) for z in zen])

    local = ts.tz_convert("Europe/Bucharest")
    day_idx = (local.normalize() - local[0].normalize()).days.to_numpy()
    n_days = day_idx[-1] + 1
    day_doy = (local[0].normalize() + pd.to_timedelta(np.arange(n_days), "D")).dayofyear.to_numpy()
    season = -np.cos(2 * np.pi * (day_doy - 18) / 365)  # -1 mid-January, +1 mid-July

    # Cloudiness: daily latent field, correlated in space (~400 km) and time (fronts last a few days).
    z = _ar1_days(_correlated_normals(rng, locs, n_days, 400.0), 0.7)
    k_day = 0.08 + 0.92 / (1 + np.exp(-(0.55 + 0.9 * season + 1.3 * z)))
    # Intraday variability, largest on broken-cloud days.
    eps = rng.standard_normal((n_loc, T)).astype(np.float32)
    noise = np.empty_like(eps)
    noise[:, 0] = eps[:, 0]
    for t in range(1, T):
        noise[:, t] = 0.85 * noise[:, t - 1] + 0.527 * eps[:, t]
    kd = k_day[:, day_idx]
    k = np.clip(kd + 0.35 * 4 * kd * (1 - kd) * noise, 0.03, 1.08)
    ghi = (k * cs).astype(np.float32)

    # Temperature: seasonal mean (colder north), synoptic anomaly, coupling to cloud, diurnal cycle.
    lat = np.array([l.lat for l in locs])[:, None]
    anom = 3.5 * _ar1_days(_correlated_normals(rng, locs, n_days, 600.0), 0.8)
    t_day = 10.5 + 12.5 * season - 0.8 * (lat - 45) + anom + 8 * season * (k_day - 0.55)
    frac_day = day_idx + local.hour.to_numpy() / 24 + local.minute.to_numpy() / 1440
    t_mean = np.stack([np.interp(frac_day, np.arange(n_days) + 0.5, t_day[i]) for i in range(n_loc)])
    amp = 2 + 7 * kd
    hour = (local.hour + local.minute / 60).to_numpy()
    temp = t_mean + amp / 2 * np.cos(2 * np.pi * (hour - 15) / 24)

    # Wind (10 m) and snow depth (daily bucket: precipitation on cloudy days, melt above 0 C).
    wind = np.clip(2.8 + 1.2 * _ar1_days(rng.standard_normal((n_loc, n_days)), 0.6)[:, day_idx]
                   + 0.6 * rng.standard_normal((n_loc, T)), 0.2, None)
    precip = np.where((k_day < 0.3) & (rng.random(k_day.shape) < 0.6), rng.exponential(6.0, k_day.shape), 0.0)
    snow_d = np.zeros_like(k_day)
    depth = np.zeros(n_loc)
    for d in range(n_days):
        tm = t_day[:, d]
        depth = depth + np.where(tm < 0.5, precip[:, d], 0.0)          # 1 mm water ~ 1 cm snow
        depth = np.maximum(depth - (0.1 + 1.2 * np.maximum(tm, 0)), 0)  # melt cm/day
        snow_d[:, d] = depth
    snow = snow_d[:, day_idx]
    return _finish(locs, ts, zen, az, ghi, temp, wind, snow, "synthetic")
