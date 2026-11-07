"""M5 fallback (spec W step 2): binned empirical power curve per turbine class (model key), P10/P50/P90 of
power / rated per 0.5 m/s bin of density-normalised hub wind, fitted on the steps M6 calls normal only.

Same bins and fill rule as #24's BinnedCurve (sentinel.eval.baselines.wind.curve), but the fit mask is passed in
(M6 normal selection, not injector labels) and only the meteo / NWP wind is used, so it forecasts from weather alone."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from sentinel.eval.baselines.wind.curve import EDGES, _binned, _interp, model_key


@dataclass
class BinnedPower:
    curves: dict  # model key -> {q10, q50, q90, n} per EDGES bin (power / rated)
    rated_kw: dict  # model key -> rated kW
    edges: list

    def key_for(self, t: Mapping) -> str:
        k = model_key(dict(t))
        return k if k in self.curves else min(self.rated_kw, key=lambda m: abs(self.rated_kw[m] - float(t["rated_kw"])))

    def predict_frac(self, turbines: Sequence[Mapping], v_norm: np.ndarray) -> np.ndarray:
        """[T, N, 3] power / rated at density-normalised hub wind v_norm [T, N]."""
        out = np.empty((*v_norm.shape, 3))
        for j, t in enumerate(turbines):
            c = self.curves[self.key_for(t)]
            out[:, j] = np.stack([_interp(self.edges, c[k], v_norm[:, j]) for k in ("q10", "q50", "q90")], -1)
        return np.sort(out, -1)

    def save(self, path) -> Path:
        Path(path).write_text(json.dumps(asdict(self)))
        return Path(path)

    @classmethod
    def load(cls, path) -> "BinnedPower":
        return cls(**json.loads(Path(path).read_text()))


def fit_binned(turbines: Sequence[Mapping], v_norm: np.ndarray, y_frac: np.ndarray, mask: np.ndarray,
               min_n: int = 30) -> BinnedPower:
    """v_norm, y_frac (power / rated), mask: [T, N]; mask = fit steps (training time x training turbines x M6 normal)."""
    keys = np.array([model_key(dict(t)) for t in turbines])
    rated = np.array([float(t["rated_kw"]) for t in turbines])
    ok = mask & np.isfinite(v_norm) & np.isfinite(y_frac)
    curves, rk = {}, {}
    for k in sorted(set(keys[ok.any(0)])):
        c = keys == k
        m = ok[:, c]
        curves[k] = _binned(v_norm[:, c][m], y_frac[:, c][m], EDGES, min_n)
        rk[k] = float(np.median(rated[c]))
    return BinnedPower(curves, rk, EDGES.tolist())
