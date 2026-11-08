"""M5 WindPower inputs: per-step, per-turbine features (numpy only).

Hub wind (meteo, else 100 m wind sheared to hub height), density-normalised v_norm, direction sin/cos, turbulence
proxy, season and hour sin/cos, hub height, rated power, the class curve at v_norm (`cf`, the residual prior) and the
layout's Jensen wake deficit for the current direction. `build` works on plain arrays (NWP inputs); `build_data` on
a WindData (training)."""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np

from wattsentry.eval.baselines.wind.curve import model_key
from wattsentry.model.wind.physics import CLASSES, air_density, class_for, density_normalise, shear_to_hub

FEATS = ("v", "cf", "dir_s", "dir_c", "turb", "rho", "wake", "doy_s", "doy_c", "hod_s", "hod_c", "hub", "rated")
N_DIR = 36
TURB_W = 5


def tclass(t: Mapping):
    return CLASSES.get(model_key(dict(t))) or class_for(float(t["rated_kw"]), t.get("rotor_d_m"), t.get("hub_height_m"))


def wake_table(turbines: Sequence[Mapping], n_dir: int = N_DIR, k: float = 0.075, ct: float = 0.8) -> np.ndarray:
    """Jensen deficit [N, n_dir] each turbine sees from upstream turbines of its own park, wind FROM each bin."""
    lat, lon = (np.array([float(t[c]) for t in turbines]) for c in ("lat", "lon"))
    x, y = (lon - lon.mean()) * 111320 * np.cos(np.deg2rad(lat.mean())), (lat - lat.mean()) * 110540
    r0 = np.array([float(t.get("rotor_d_m") or 100.0) for t in turbines]) / 2
    park = np.array([str(t["park_id"]) for t in turbines])
    dx, dy, same = x[None] - x[:, None], y[None] - y[:, None], park[:, None] == park[None]  # [j, k]: j -> k
    a, out = 1 - np.sqrt(1 - ct), np.zeros((len(turbines), n_dir))
    for d, th in enumerate(np.deg2rad(np.arange(n_dir) * 360 / n_dir)):
        along, cross = dx * np.sin(th) + dy * np.cos(th), np.abs(dy * np.sin(th) - dx * np.cos(th))
        r = r0[None] + k * np.clip(along, 0, None)
        out[:, d] = np.sqrt(np.where(same & (along > 0) & (cross < r), (a * (r0[None] / r) ** 2) ** 2, 0).sum(1))
    return out


def _roll(x: np.ndarray, w: int) -> np.ndarray:
    """Centred rolling mean over time of [T, N] (edges padded)."""
    p = np.pad(x, ((w // 2, w // 2), (0, 0)), mode="edge")
    c = np.cumsum(np.vstack([np.zeros((1, x.shape[1])), p]), 0)
    return (c[w:] - c[:-w]) / w


def _field(src: Mapping, k: str, shape: tuple, default: float) -> np.ndarray:
    x = src.get(k)
    if x is None:
        return np.full(shape, default)
    x = np.broadcast_to(np.asarray(x, np.float64).reshape(shape[0], -1), shape)
    return np.where(np.isfinite(x), x, default)


def hub_wind(turbines: Sequence[Mapping], meteo: Mapping, weather: Mapping, T: int) -> np.ndarray:
    """Hub-height wind [T, N] m/s: meteo `wind_ms` (per turbine, already at hub), filled from `wind100_ms` sheared
    to each turbine's hub height (power law, alpha 0.14) where missing."""
    N = len(turbines)
    hub = np.array([float(t.get("hub_height_m") or 100.0) for t in turbines])
    v = np.asarray(meteo["wind_ms"], np.float64).reshape(T, N) if meteo.get("wind_ms") is not None else np.full((T, N), np.nan)
    if weather.get("wind100_ms") is not None:
        v100 = np.broadcast_to(np.asarray(weather["wind100_ms"], np.float64).reshape(T, -1), (T, N))
        v = np.where(np.isfinite(v), v, shear_to_hub(v100, hub[None]))
    return v


def build(turbines: Sequence[Mapping], ts: np.ndarray, meteo: Mapping, weather: Mapping) -> dict[str, np.ndarray]:
    """FEATS -> float32 arrays, [T, N] or broadcastable [T, 1] / [1, N].

    turbines: rows with rated_kw, hub_height_m, rotor_d_m, lat, lon, park_id, model; ts: datetime64 [T] (UTC, 15 min);
    meteo: {"wind_ms": [T, N] hub wind, "wdir_deg"?: [T, N]}; weather: per-turbine [T, N] (or [T, 1]) fields
    temp_c, pressure_hpa, wdir100_deg, wind100_ms (all optional)."""
    ts = np.asarray(ts).astype("datetime64[m]")
    T, N, f32 = len(ts), len(turbines), np.float32
    sh = (T, N)
    rho = air_density(_field(weather, "pressure_hpa", sh, 1013.25), _field(weather, "temp_c", sh, 10.0))
    v_hub = hub_wind(turbines, meteo, weather, T)
    v = np.clip(np.nan_to_num(density_normalise(v_hub, rho)), 0, 35)
    cf = np.stack([tclass(t).power_frac(v[:, j]) for j, t in enumerate(turbines)], 1)
    wd = np.asarray(meteo["wdir_deg"], np.float64).reshape(sh) if meteo.get("wdir_deg") is not None else np.full(sh, np.nan)
    wd = np.where(np.isfinite(wd), wd, _field(weather, "wdir100_deg", sh, 270.0))
    raw = np.nan_to_num(v_hub)
    m = _roll(raw, TURB_W)
    turb = np.sqrt(np.clip(_roll(raw**2, TURB_W) - m**2, 0, None)) / np.maximum(m, 2.0)
    wk = wake_table(turbines)[np.arange(N)[None], np.round(wd / (360 / N_DIR)).astype(int) % N_DIR]
    doy = 2 * np.pi * (ts.astype("datetime64[D]") - ts.astype("datetime64[Y]")).astype(int)[:, None] / 365.25
    hod = 2 * np.pi * (ts.astype(np.int64) % 1440)[:, None] / 1440
    stat = lambda k, s, d: np.array([[float(t.get(k) or d) / s for t in turbines]])  # noqa: E731
    out = {"v": v / 15, "cf": cf, "dir_s": np.sin(np.deg2rad(wd)), "dir_c": np.cos(np.deg2rad(wd)), "turb": turb * 5,
           "rho": (rho - 1.225) / 0.05, "wake": wk * 5, "doy_s": np.sin(doy), "doy_c": np.cos(doy),
           "hod_s": np.sin(hod), "hod_c": np.cos(hod), "hub": stat("hub_height_m", 100, 100.0),
           "rated": stat("rated_kw", 5000, 2500.0)}
    return {k: x.astype(f32) for k, x in out.items()}


def weather_per_turbine(weather: Mapping, park_of: np.ndarray) -> dict[str, np.ndarray]:
    """Park weather [T, P] -> per turbine [T, N]."""
    return {k: np.asarray(v)[:, park_of] for k, v in weather.items()}


def build_data(data) -> dict[str, np.ndarray]:
    """`build` on a wattsentry.model.wind.data.WindData."""
    return build(data.turbines, data.ts, data.meteo, weather_per_turbine(data.weather, data.park_of))


def gather(base: dict[str, np.ndarray], ti: np.ndarray, tj: np.ndarray) -> np.ndarray:
    """Rows (ti, tj) -> [n, len(FEATS)] float32."""
    return np.stack([base[k][ti if base[k].shape[0] > 1 else 0, tj if base[k].shape[1] > 1 else 0] for k in FEATS], 1)


def park_sector(base: dict[str, np.ndarray], park_of: np.ndarray, n_parks: int, n_sec: int = 12) -> np.ndarray:
    """Park-mean wind direction sector [T, P] (30 deg bins for n_sec = 12)."""
    T = max(base["dir_s"].shape[0], 1)
    s = np.stack([base["dir_s"][:, park_of == p].mean(1) if (park_of == p).any() else np.zeros(T)
                  for p in range(n_parks)], 1)
    c = np.stack([base["dir_c"][:, park_of == p].mean(1) if (park_of == p).any() else np.ones(T)
                  for p in range(n_parks)], 1)
    return np.round(np.rad2deg(np.arctan2(s, c)) % 360 / (360 / n_sec)).astype(int) % n_sec
