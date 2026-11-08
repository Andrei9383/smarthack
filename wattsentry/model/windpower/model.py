"""M5 WindPower predict API, torch-free (the MLP runs through onnxruntime).

    m = WindPowerModel.load("out/windpower.s0.onnx")    # or a list of artifacts = seed ensemble
    q = m.predict_frac(turbines, ts, meteo, weather)    # [T, N, 3] P10/P50/P90 power / rated per 15-min step

Inputs (one row per turbine, one step per 15 min, UTC): `turbines` = turbines-table rows; `meteo` = {"wind_ms": [T, N]
hub wind, "wdir_deg"?: [T, N]}; `weather` = per-turbine [T, N] (or [T, 1]) temp_c, pressure_hpa, wdir100_deg,
wind100_ms (hub wind where wind_ms is missing). Unknown turbines use the "unseen turbine" embedding; unknown models
map to the class with the nearest rated power. `park_aggregate` sums turbines to a park total.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from .features import FEATS, build, gather
from wattsentry.eval.baselines.wind.curve import model_key

N_SEC = 12
CHUNK = 262_144


def park_aggregate(q_kw: np.ndarray, avail: np.ndarray | None, ratio_t: np.ndarray, mult: Sequence[float]) -> np.ndarray:
    """One park: turbine quantiles [T, n, 3] kW -> park [T, 3] kW (P50 x availability x wake ratio; P10/P90 by spread
    multipliers). ratio_t: [T] wake ratio for the park's sector at each step; mult: (a, b)."""
    w = q_kw if avail is None else q_kw * np.asarray(avail, np.float64)[..., None]
    p50 = w[..., 1].sum(1) * ratio_t
    lo, hi = (w[..., 1] - w[..., 0]).sum(1), (w[..., 2] - w[..., 1]).sum(1)
    return np.stack([np.clip(p50 - mult[0] * lo, 0, None), p50, p50 + mult[1] * hi], -1)


class WindPowerModel:
    """M5 per-turbine P10/P50/P90 from exported ONNX artifact(s) (several = seed ensemble, quantiles averaged)."""

    def __init__(self, meta: dict, sessions: list):
        self.meta, self.sessions = meta, sessions
        self.model_version = meta.get("model_version", "windpower")
        self.tid_of = {str(k): int(v) for k, v in meta.get("tid_of", {}).items()}
        self.models = list(meta.get("models", []))
        self.models_rated = {k: float(v) for k, v in meta.get("models_rated", {}).items()}

    @classmethod
    def load(cls, paths) -> "WindPowerModel":
        """ONNX path(s), each with its sidecar .json."""
        paths = [Path(p) for p in ([paths] if isinstance(paths, (str, Path)) else paths)]
        meta = json.loads(paths[0].with_suffix(".json").read_text())
        if len(paths) > 1:
            meta = {**meta, "model_version": "+".join(json.loads(p.with_suffix(".json").read_text())["model_version"]
                                                     for p in paths)}
        return cls(meta, [_session(p) for p in paths])

    def _ids(self, turbines: Sequence[Mapping]) -> tuple[np.ndarray, np.ndarray]:
        tid = np.array([self.tid_of.get(str(t.get("turbine_id")), 0) for t in turbines], np.int64)
        mids = []
        for t in turbines:
            k = model_key(dict(t))
            if k not in self.models:
                k = min(self.models_rated, key=lambda m: abs(self.models_rated[m] - float(t["rated_kw"])))
            mids.append(self.models.index(k))
        return tid, np.array(mids, np.int64)

    def predict_frac(self, turbines: Sequence[Mapping], ts, meteo: Mapping, weather: Mapping) -> np.ndarray:
        """[T, N, 3] power / rated."""
        ts = np.asarray(ts).astype("datetime64[m]")
        T, N = len(ts), len(turbines)
        base = build(turbines, ts, meteo, weather)
        tid, mid = self._ids(turbines)
        ti, tj = (a.ravel() for a in np.meshgrid(np.arange(T), np.arange(N), indexing="ij"))
        outs = []
        for s in self.sessions:
            q = np.concatenate([s.run(None, {"x": gather(base, ti[i:i + CHUNK], tj[i:i + CHUNK]),
                                             "tid": tid[tj[i:i + CHUNK]], "mid": mid[tj[i:i + CHUNK]]})[0]
                                for i in range(0, T * N, CHUNK)]) if T * N else np.zeros((0, 3), np.float32)
            outs.append(np.sort(np.clip(q.reshape(T, N, 3), 0.0, 1.05), -1))
        return np.stack(outs).mean(0)

def _session(path: Path):
    import onnxruntime as ort

    return ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])


__all__ = ["FEATS", "WindPowerModel", "park_aggregate"]
