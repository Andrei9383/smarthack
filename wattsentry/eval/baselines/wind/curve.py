"""Binned empirical power curve per turbine model: P10/P50/P90 of power / rated per 0.5 m/s bin of density-normalised
hub wind (plan §3.5), for the meteo wind (`met`, Production-mode forecast) and the nacelle anemometer (`nacelle`).
Fitted on train days x training parks x normal-labelled steps only (the mask #41's NBM trains on)."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from wattsentry.model.wind.data import AMBIENT, POWER, TEMP_SENSORS, WIND, WindData
from wattsentry.model.wind.physics import air_density, density_normalise

EDGES = np.arange(0.0, 30.5, 0.5)
PF_EDGES = np.arange(0.0, 1.15, 0.05)  # load bins for the expected temperature rise (detector)
SOURCES = ("met", "nacelle")


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


@dataclass
class BinnedCurve:
    curves: dict  # source -> model -> {q10, q50, q90, n} per EDGES bin
    rated_kw: dict  # model -> rated kW
    temp_rise: dict  # model -> sensor -> median rise over ambient per PF_EDGES bin
    meta: dict = field(default_factory=dict)  # nowcast_source: chosen on validation
    edges: list = field(default_factory=lambda: EDGES.tolist())
    pf_edges: list = field(default_factory=lambda: PF_EDGES.tolist())

    def key_for(self, t: dict) -> str:
        k = model_key(t)
        return k if k in self.rated_kw else min(self.rated_kw, key=lambda m: abs(self.rated_kw[m] - float(t["rated_kw"])))

    def predict_block(self, turbines: list[dict], v: dict, source: str = "nowcast") -> np.ndarray:
        """[T, n, 3] power fraction; 'nowcast' = meta['nowcast_source'], the other source where its wind is NaN."""
        order = [source] if source in SOURCES else sorted(SOURCES, key=lambda s: s != self.meta.get("nowcast_source"))
        out = np.full((*v["met"].shape, 3), np.nan)
        for j, t in enumerate(turbines):
            c = self.curves
            for s in reversed(order):
                x = v[s][:, j]
                q = np.stack([_interp(self.edges, c[s][self.key_for(t)][k], x) for k in ("q10", "q50", "q90")], -1)
                out[:, j] = np.where(np.isfinite(x)[:, None], q, out[:, j])
        return out

    def expected_rise(self, key: str, sensor: str, pf: np.ndarray) -> np.ndarray:
        return _interp(self.pf_edges, self.temp_rise[key][sensor], pf)

    def save(self, path) -> Path:
        Path(path).write_text(json.dumps(asdict(self)))
        return Path(path)

    @classmethod
    def load(cls, path) -> "BinnedCurve":
        return cls(**json.loads(Path(path).read_text()))


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


def fit_curves(data: WindData, t_mask: np.ndarray, turbines: np.ndarray, min_n: int = 30) -> BinnedCurve:
    v = data_inputs(data)
    rated = data.rated_kw.astype(np.float64)
    pf = data.sensors[POWER] / rated[None]
    ok = np.isfinite(pf) & (data.err == 0) & t_mask[:, None]
    if data.labels is not None:
        ok &= data.labels == 0
    keys = np.array([model_key(t) for t in data.turbines])
    curves, rise, rk = {s: {} for s in SOURCES}, {}, {}
    for k in sorted(set(keys[turbines])):
        c = turbines[keys[turbines] == k]
        rk[k] = float(np.median(rated[c]))
        for s in SOURCES:
            m = ok[:, c] & np.isfinite(v[s][:, c])
            curves[s][k] = _binned(v[s][:, c][m], pf[:, c][m], EDGES, min_n)
        rise[k] = {}
        for s in TEMP_SENSORS:
            r = data.sensors[s][:, c] - data.sensors[AMBIENT][:, c]
            m = ok[:, c] & np.isfinite(r)
            rise[k][s] = _binned(np.clip(pf[:, c][m], 0, 1.1), r[m], PF_EDGES, min_n)["q50"]
    return BinnedCurve(curves, rk, rise, {"data": data.source, "fingerprint": data.fingerprint(),
                                         "n_fit_turbines": len(turbines), "n_fit_steps": int(t_mask.sum())})
