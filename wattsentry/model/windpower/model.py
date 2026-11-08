"""M5 WindPower predict API (card #164). Torch-free: the MLP runs through onnxruntime, the fallback is numpy.

    from wattsentry.model.windpower import WindPowerModel
    m = WindPowerModel.load()                           # models/windpower*.onnx (+ sidecar .json); several = seed ensemble
    m = WindPowerModel.load("out/windpower.s0.onnx")    # one artifact (a lab run's $LAB_OUTPUT_DIR)
    m = WindPowerModel.from_json("out/windpower.s0.json", binned=True)   # binned fallback only (no ONNX needed)

    q = m.predict(turbines, ts, meteo, weather)         # [T, N, 3] kW = P10/P50/P90 per turbine per 15-min step
    q = m.predict_data(data)                            # same on a wattsentry.model.wind.data.WindData
    pk = m.park(q, turbines, ts, meteo, weather, availability=None, horizon_h=24)   # {park_id: [T, 3] kW}

Inputs (one row per turbine, one step per 15 min, UTC): `turbines` = turbines-table rows (turbine_id, park_id, model,
rated_kw, hub_height_m, rotor_d_m, lat, lon); `meteo` = {"wind_ms": [T, N] hub wind (NWP / meteo), "wdir_deg"?:
[T, N]}; `weather` = per-turbine [T, N] (or [T, 1]) temp_c, pressure_hpa, wdir100_deg, wind100_ms (used for hub wind
where wind_ms is missing, sheared to hub height). Unknown turbines (e.g. a park not seen in training) use the
"unseen turbine" embedding; unknown models map to the class with the nearest rated power.

Park = sum over turbines of P50 x availability x rated, x a per-sector wake ratio fitted on training days (pooled over
training parks for an unseen park); P10/P90 from the summed turbine spreads x multipliers fitted on validation park
totals. `availability` [T, N] in [0, 1] (W2: M6/M7 verdicts) defaults to 1. W2 builds scenarios on top of `predict`.
"""

from __future__ import annotations

import glob
import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from .binned import BinnedPower
from .features import FEATS, _field, build, gather, hub_wind, weather_per_turbine
from wattsentry.eval.baselines.wind.curve import model_key
from wattsentry.model.wind.physics import air_density, density_normalise

REPO = Path(__file__).resolve().parents[3]
MODELS = REPO / "models"
N_SEC = 12
CHUNK = 262_144


def _sector(turbines: Sequence[Mapping], meteo: Mapping, weather: Mapping, T: int) -> tuple[list[str], np.ndarray]:
    """Park ids and park-mean direction sector [T, P] (30 deg)."""
    N = len(turbines)
    wd = np.asarray(meteo["wdir_deg"], np.float64).reshape(T, N) if meteo.get("wdir_deg") is not None else np.full((T, N), np.nan)
    if weather.get("wdir100_deg") is not None:
        wd = np.where(np.isfinite(wd), wd, np.broadcast_to(np.asarray(weather["wdir100_deg"], np.float64).reshape(T, -1), (T, N)))
    wd = np.deg2rad(np.nan_to_num(wd, nan=270.0))
    parks = list(dict.fromkeys(str(t["park_id"]) for t in turbines))
    po = np.array([parks.index(str(t["park_id"])) for t in turbines])
    s = np.stack([np.sin(wd[:, po == p]).mean(1) for p in range(len(parks))], 1)
    c = np.stack([np.cos(wd[:, po == p]).mean(1) for p in range(len(parks))], 1)
    return parks, np.round(np.rad2deg(np.arctan2(s, c)) % 360 / (360 / N_SEC)).astype(int) % N_SEC


def park_aggregate(q_kw: np.ndarray, avail: np.ndarray | None, ratio_t: np.ndarray, mult: Sequence[float]) -> np.ndarray:
    """One park: turbine quantiles [T, n, 3] kW -> park [T, 3] kW (P50 x availability x wake ratio; P10/P90 by spread
    multipliers). ratio_t: [T] wake ratio for the park's sector at each step; mult: (a, b)."""
    w = q_kw if avail is None else q_kw * np.asarray(avail, np.float64)[..., None]
    p50 = w[..., 1].sum(1) * ratio_t
    lo, hi = (w[..., 1] - w[..., 0]).sum(1), (w[..., 2] - w[..., 1]).sum(1)
    return np.stack([np.clip(p50 - mult[0] * lo, 0, None), p50, p50 + mult[1] * hi], -1)


class WindPowerModel:
    """M5 per-turbine P10/P50/P90. `sessions` = onnxruntime sessions (seed ensemble; empty = binned fallback)."""

    model = "windpower"

    def __init__(self, meta: dict, sessions: list | None = None, binned: BinnedPower | None = None):
        self.meta, self.sessions = meta, sessions or []
        self.binned = binned if binned is not None else (BinnedPower(**meta["binned"]) if meta.get("binned") else None)
        if not self.sessions and self.binned is None:
            raise ValueError("WindPowerModel needs an ONNX session or a binned curve")
        self.model_version = meta.get("model_version", "windpower") + ("" if self.sessions else "-binned")
        self.tid_of = {str(k): int(v) for k, v in meta.get("tid_of", {}).items()}
        self.models = list(meta.get("models", []))
        self.models_rated = {k: float(v) for k, v in meta.get("models_rated", {}).items()}

    # ---- loading --------------------------------------------------------------------------------------------------
    @classmethod
    def from_json(cls, path, binned: bool = False) -> "WindPowerModel":
        """Sidecar JSON (+ the ONNX next to it unless binned=True)."""
        path = Path(path)
        meta = json.loads(path.read_text())
        if binned:
            return cls(meta, [])
        return cls(meta, [_session(path.with_suffix(".onnx"))])

    @classmethod
    def load(cls, paths=None) -> "WindPowerModel":
        """ONNX path(s) or a directory (default: models/). Several artifacts = seed ensemble (quantiles averaged)."""
        if paths is None or isinstance(paths, (str, Path)) and Path(paths).is_dir():
            d = Path(paths or MODELS)
            paths = sorted(glob.glob(str(d / "windpower*.onnx")))
            if not paths:
                raise FileNotFoundError(f"no windpower*.onnx in {d}; use WindPowerModel.from_json(..., binned=True) "
                                        "or train: python train.py --model windpower")
        paths = [Path(p) for p in ([paths] if isinstance(paths, (str, Path)) else paths)]
        meta = json.loads(paths[0].with_suffix(".json").read_text())
        if len(paths) > 1:
            meta = {**meta, "model_version": "+".join(json.loads(p.with_suffix(".json").read_text())["model_version"]
                                                     for p in paths)}
        return cls(meta, [_session(p) for p in paths])

    # ---- prediction -----------------------------------------------------------------------------------------------
    def _ids(self, turbines: Sequence[Mapping]) -> tuple[np.ndarray, np.ndarray]:
        tid = np.array([self.tid_of.get(str(t.get("turbine_id")), 0) for t in turbines], np.int64)
        mids = []
        for t in turbines:
            k = model_key(dict(t))
            if k not in self.models:
                k = min(self.models_rated, key=lambda m: abs(self.models_rated[m] - float(t["rated_kw"])))
            mids.append(self.models.index(k))
        return tid, np.array(mids, np.int64)

    def predict_frac(self, turbines: Sequence[Mapping], ts, meteo: Mapping, weather: Mapping,
                     members: bool = False) -> np.ndarray:
        """[T, N, 3] power / rated (members=True: [M, T, N, 3], one per ensemble member)."""
        ts = np.asarray(ts).astype("datetime64[m]")
        T, N = len(ts), len(turbines)
        if not self.sessions:
            rho = air_density(_field(weather, "pressure_hpa", (T, N), 1013.25), _field(weather, "temp_c", (T, N), 10.0))
            v = np.clip(np.nan_to_num(density_normalise(hub_wind(turbines, meteo, weather, T), rho)), 0, 35)
            q = self.binned.predict_frac(turbines, v)
            return q[None] if members else q
        base = build(turbines, ts, meteo, weather)
        tid, mid = self._ids(turbines)
        ti, tj = (a.ravel() for a in np.meshgrid(np.arange(T), np.arange(N), indexing="ij"))
        outs = []
        for s in self.sessions:
            q = np.concatenate([s.run(None, {"x": gather(base, ti[i:i + CHUNK], tj[i:i + CHUNK]),
                                             "tid": tid[tj[i:i + CHUNK]], "mid": mid[tj[i:i + CHUNK]]})[0]
                                for i in range(0, T * N, CHUNK)]) if T * N else np.zeros((0, 3), np.float32)
            outs.append(np.sort(np.clip(q.reshape(T, N, 3), 0.0, 1.05), -1))
        q = np.stack(outs)
        return q if members else q.mean(0)

    def predict(self, turbines: Sequence[Mapping], ts, meteo: Mapping, weather: Mapping) -> np.ndarray:
        """[T, N, 3] kW: P10/P50/P90 per turbine per 15-min step."""
        rated = np.array([float(t["rated_kw"]) for t in turbines])
        return self.predict_frac(turbines, ts, meteo, weather) * rated[None, :, None]

    def predict_data(self, data) -> np.ndarray:
        return self.predict(data.turbines, data.ts, data.meteo, weather_per_turbine(data.weather, data.park_of))

    def nowcast_adjust(self, q_kw: np.ndarray, resid0_kw: np.ndarray, horizon_h: int) -> np.ndarray:
        """Adds beta_h x the residual observed at the forecast origin (actual - P50, per turbine; NaN = unknown).
        beta_h was fitted on validation for h in {1, 6, 24}; the nearest one is used (it is ~0 day-ahead)."""
        betas = {int(k): float(v) for k, v in self.meta.get("beta_h", {}).items()} or {24: 0.0}
        b = betas[min(betas, key=lambda h: abs(h - horizon_h))]
        return np.clip(q_kw + b * np.nan_to_num(np.asarray(resid0_kw, np.float64))[..., None], 0, None)

    def park(self, q_kw: np.ndarray, turbines: Sequence[Mapping], ts, meteo: Mapping, weather: Mapping,
             availability: np.ndarray | None = None, horizon_h: int = 24) -> dict[str, np.ndarray]:
        """{park_id: [T, 3] kW} from turbine quantiles q_kw [T, N, 3]."""
        T = q_kw.shape[0]
        parks, sec = _sector(turbines, meteo, weather, T)
        ratios = self.meta.get("wake_ratio", {})
        pooled = np.asarray(self.meta.get("wake_ratio_pooled", np.ones(N_SEC)), np.float64)
        mults = {int(k): v for k, v in self.meta.get("park_mult", {}).items()} or {24: (1.0, 1.0)}
        mult = mults[min(mults, key=lambda h: abs(h - horizon_h))]
        pid = np.array([str(t["park_id"]) for t in turbines])
        out = {}
        for p, park in enumerate(parks):
            c = pid == park
            r = np.asarray(ratios.get(park, pooled), np.float64)
            out[park] = park_aggregate(q_kw[:, c], None if availability is None else availability[:, c], r[sec[:, p]], mult)
        return out


def _session(path: Path):
    import onnxruntime as ort

    return ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])


__all__ = ["FEATS", "WindPowerModel", "park_aggregate"]
