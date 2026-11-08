"""Power-curve helpers shared by the wind models: density-normalised hub wind per source (`met` = meteo wind,
`nacelle` = anemometer), 0.5 m/s bins and the per-bin P10/P50/P90 fit of power / rated."""

from __future__ import annotations

import numpy as np

from wattsentry.model.wind.data import AMBIENT, WIND, WindData
from wattsentry.model.wind.physics import air_density, density_normalise

EDGES = np.arange(0.0, 30.5, 0.5)


def model_key(t: dict) -> str:
    return str(t.get("model") or f"generic-{float(t['rated_kw']) / 1000:.1f}MW")


def park_inputs(sensors: dict, meteo: dict, weather: dict, T: int, n: int) -> dict[str, np.ndarray]:
    """v_norm [T, n] per source; ρ from the park's pressure and temperature (nacelle: the turbine's ambient)."""
    nan = np.full((T, n), np.nan)

    def w(k, default):
        x = np.asarray(weather.get(k, nan[:, 0]), np.float64).reshape(T, -1)
        return np.where(np.isfinite(x), x, default)

    pres, temp = w("pressure_hpa", 1013.25), w("temp_c", 10.0)
    amb = np.asarray(sensors.get(AMBIENT, nan), np.float64)
    return {"met": density_normalise(np.asarray(meteo.get("wind_ms", nan), np.float64), air_density(pres, temp)),
            "nacelle": density_normalise(np.asarray(sensors.get(WIND, nan), np.float64) / 3.6,
                                         air_density(pres, np.where(np.isfinite(amb), amb, temp)))}


def data_inputs(data: WindData) -> dict[str, np.ndarray]:
    po = data.park_of
    return park_inputs(data.sensors, data.meteo, {k: v[:, po] for k, v in data.weather.items()}, data.n_steps,
                       len(data.turbines))


def _interp(edges, ys, x):
    e = np.asarray(edges)
    return np.interp(np.nan_to_num(x), 0.5 * (e[1:] + e[:-1]), np.asarray(ys))


def _binned(v, y, edges, min_n=30) -> dict:
    """Quantiles per bin (>= min_n samples); empty bins: 0 below the first populated one, linear fill elsewhere."""
    nb = len(edges) - 1
    b = np.clip(np.digitize(v, edges) - 1, 0, nb - 1)
    o = np.argsort(b, kind="stable")
    cuts = np.searchsorted(b[o], np.arange(nb + 1))
    q = np.full((3, nb), np.nan)
    for i in np.flatnonzero(np.diff(cuts) >= min_n):
        q[:, i] = np.quantile(y[o][cuts[i] : cuts[i + 1]], (0.1, 0.5, 0.9))
    good = np.flatnonzero(np.isfinite(q[1]))
    if len(good):
        q[:, : good[0]] = 0.0
        good = np.flatnonzero(np.isfinite(q[1]))
        q = np.stack([np.interp(np.arange(nb), good, x[good]) for x in q])
    q = np.sort(np.nan_to_num(q), axis=0)
    return {"q10": q[0].tolist(), "q50": q[1].tolist(), "q90": q[2].tolist(), "n": np.diff(cuts).tolist()}

