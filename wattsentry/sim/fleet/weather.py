"""Weather per locality at 15 min.

Source "raw": the committed Open-Meteo / ERA5 Parquet from #9 (see RAW_CANDIDATES; $WATTSENTRY_WEATHER overrides).
Source "synthetic": a deterministic stand-in with Romanian climatology (spatially correlated cloud days, seasonal
and diurnal temperature, snow), so the simulator runs before #9 lands. The manifest records which one was used.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from . import solar
from .localities import Locality
from .ro_calendar import local_day_index

REPO = Path(__file__).resolve().parents[3]
# #9's files, first match wins: the data.md weather table (point_id "loc:<Locality.id>"), or the locality-only file.
RAW_CANDIDATES = [REPO / "data" / "raw" / "weather.parquet", REPO / "data" / "raw" / "weather_localities.parquet"]
# Old column names -> data.md names (schemas.WEATHER)
_RENAME = {"time": "ts", "locality_id": "point_id", "ghi": "ghi_wm2", "temp_air": "temp_c",
           "wind_speed_10m": "wind10_ms", "snow_depth_cm": "snow_cm"}
_NEED = ("ghi_wm2", "temp_c", "wind10_ms", "snow_cm")


def weather_key(source: str = "auto") -> str:
    """Part of the cache key: which weather the simulator would use (raw file name + size, or synthetic)."""
    path = raw_path()
    if source == "synthetic" or (source == "auto" and path is None):
        return "synthetic"
    return f"{path.stem}-{path.stat().st_size}" if path is not None else "raw-missing"


def raw_path() -> Path | None:
    env = os.environ.get("WATTSENTRY_WEATHER")
    if env:
        return Path(env)
    return next((p for p in RAW_CANDIDATES if p.exists()), None)


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
    path = raw_path()
    if source == "raw" or (source == "auto" and path is not None):
        if path is None:
            raise FileNotFoundError(f"no raw weather file: {RAW_CANDIDATES}")
        return _from_raw(locs, ts, path)
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


def read_raw(path: Path, locality_ids: list[str]) -> pd.DataFrame:
    """#9's weather in data.md names, actuals only, point_id = Locality.id, for the given localities."""
    import pyarrow.parquet as pq

    names = pq.read_schema(path).names
    cols = [c for c in names if c in _RENAME or c in _RENAME.values() or c == "source"]
    df = pd.read_parquet(path, columns=cols).rename(columns=_RENAME)
    if "source" in df:
        df = df[df.source.astype(str) != "forecast"]  # history = actuals, never the noisy planning forecast
    df["point_id"] = df.point_id.astype(str).str.removeprefix("loc:")
    df = df[df.point_id.isin(locality_ids)]
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    missing = [c for c in _NEED if c not in df]
    if missing:
        raise ValueError(f"{path.name}: missing columns {missing}")
    return df


def _from_raw(locs: list[Locality], ts: pd.DatetimeIndex, path: Path) -> Weather:
    df = read_raw(path, [l.id for l in locs])
    zen, az = _geometry(locs, ts)
    out = {k: np.empty((len(locs), len(ts)), np.float32) for k in _NEED}
    for i, loc in enumerate(locs):
        g = df[df.point_id == loc.id].set_index("ts")[list(_NEED)].sort_index()
        g = g[~g.index.duplicated()]
        if g.empty:
            raise ValueError(f"{path.name} has no rows for locality {loc.id}")
        step = g.index.to_series().diff().median()
        if step >= pd.Timedelta("1h"):
            # #9 stamps hourly radiation at the interval START (mean over [t, t+1h)): centre it before interpolating.
            ghi = g.ghi_wm2.copy()
            ghi.index = ghi.index + step / 2
            g = g.drop(columns="ghi_wm2").join(ghi, how="outer")
        g = g.reindex(g.index.union(ts)).interpolate("time", limit_direction="both").reindex(ts)
        for k in out:
            out[k][i] = g[k].to_numpy()
    # Hourly ERA5 GHI is an hourly mean: rescale to 15 min with the clear-sky index (k = ghi / clear-sky).
    cs = np.stack([solar.clearsky_ghi(z) for z in zen])
    out["snow_cm"] = np.nan_to_num(out["snow_cm"])
    k = np.clip(np.divide(out["ghi_wm2"], cs, out=np.zeros_like(cs), where=cs > 20), 0, 1.1)
    ghi = k * cs
    return _finish(locs, ts, zen, az, ghi, out["temp_c"], out["wind10_ms"], out["snow_cm"], "raw")


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
    day_idx = local_day_index(local)
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
