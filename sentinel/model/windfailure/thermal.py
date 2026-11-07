"""Physics stand-in for M6's thermal path: a linear normal-behaviour model of component temperatures.

Thermal relation (plan §4, first-order lag):  T_c = lag_tau(T_amb + base + offset + rise · load^γ).
Because the lag is linear, the rise over ambient is a linear combination of lagged load^γ and lagged ambient. With
tau and γ unknown we regress on a bank of them:

    T_c − T_amb ≈ b0 + Σ_{τ,γ} w_{τγ} · lag_τ(load^γ) + Σ_τ u_τ · (lag_τ(T_amb) − T_amb)

τ ∈ {0.5, 1, 2, 3, 4, 6} h, γ ∈ {1, 1.5, 2}; one fit per turbine class (2.5 / 6.1 MW) and component, ridge
least squares on TRAINING turbines and TRAINING time only. The per-turbine offset (incl. hot-by-design turbines)
is not modelled here; features.py removes it with each turbine's own lagged baseline.

Normal selection: label-free. A step is used for fitting when the temperature, power and ambient are finite, no
integral error code is present and the last gap is >= 12 h back. No labels, and NO drift mask (a mask that drops
slowly drifting temperatures would hide exactly the precursor M7 must see; coordinator note on #70's
select_normal). The few drifting steps in training barely move a 25-coefficient fleet-level fit.

Caveat (R13): the regressor structure mirrors the simulator's thermal model, so this residual is near-ideal on
synthetic data. The `nbm` residual source (M6's GRU, residuals_nbm.py) is the non-circular alternative.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import lfilter

from sentinel.model.wind.data import AMBIENT, POWER

from . import COMPONENTS

TAUS_H = (0.5, 1.0, 2.0, 3.0, 4.0, 6.0)
GAMMAS = (1.0, 1.5, 2.0)
AMB_TAUS_H = (0.5, 1.0, 2.0, 3.0, 4.0, 6.0)
DT_H = 0.25
GAP_SETTLE = 48  # steps after a gap excluded from fitting (filters restart from a filled value)


def _lag(x: np.ndarray, tau_h: float) -> np.ndarray:
    a = float(np.exp(-DT_H / tau_h))
    return lfilter([1 - a], [1.0, -a], x, axis=0, zi=x[:1] * a)[0]


def _ffill(x: np.ndarray, fill: float) -> np.ndarray:
    """Forward-fill NaNs along time for a 1-D series (leading NaNs → fill)."""
    ok = np.isfinite(x)
    if ok.all():
        return x
    idx = np.where(ok, np.arange(len(x)), 0)
    np.maximum.accumulate(idx, out=idx)
    out = x[idx]
    out[~np.isfinite(out)] = fill
    return out


def _since_gap(bad: np.ndarray) -> np.ndarray:
    """Steps since the last bad step (large if none yet)."""
    n = len(bad)
    last = np.where(bad, np.arange(n), -10**9)
    np.maximum.accumulate(last, out=last)
    return np.arange(n) - last


def design(load: np.ndarray, amb: np.ndarray) -> np.ndarray:
    """1-D load fraction and ambient (NaN-free) → regressors [T, F] (first column = 1)."""
    cols = [np.ones_like(load)]
    for tau in TAUS_H:
        for g in GAMMAS:
            cols.append(_lag(load**g, tau))
    for tau in AMB_TAUS_H:
        cols.append(_lag(amb, tau) - amb)
    return np.stack(cols, 1)


def _turbine_inputs(sensors: dict, err: np.ndarray, rated: np.ndarray, j: int):
    p = sensors[POWER][:, j].astype(np.float64) / rated[j]
    amb = sensors[AMBIENT][:, j].astype(np.float64)
    bad = ~np.isfinite(p) | ~np.isfinite(amb) | (err[:, j] != 0)
    load = np.clip(_ffill(p, 0.0), 0.0, 1.01)
    ambf = _ffill(amb, float(np.nanmedian(amb)) if np.isfinite(amb).any() else 10.0)
    return design(load, ambf), ambf, bad


def fit(sensors: dict, err: np.ndarray, rated: np.ndarray, is_large: np.ndarray, fit_turbines, fit_t: np.ndarray,
        ridge: float = 1e-3) -> dict:
    """Least squares per (class, component). Returns a JSON-serialisable model."""
    F = 1 + len(TAUS_H) * len(GAMMAS) + len(AMB_TAUS_H)
    acc = {(k, c): [np.zeros((F, F)), np.zeros(F), 0] for k in (0, 1) for c in COMPONENTS}
    for j in fit_turbines:
        X, amb, bad = _turbine_inputs(sensors, err, rated, j)
        settle = _since_gap(bad) > GAP_SETTLE
        k = int(is_large[j])
        for c in COMPONENTS:
            y = sensors[c][:, j].astype(np.float64) - amb
            m = fit_t & settle & np.isfinite(y)
            if m.sum() < 100:
                continue
            Xm = X[m]
            a = acc[(k, c)]
            a[0] += Xm.T @ Xm
            a[1] += Xm.T @ y[m]
            a[2] += int(m.sum())
    coef = {}
    for (k, c), (xtx, xty, n) in acc.items():
        if n == 0:
            continue
        reg = ridge * np.trace(xtx) / F * np.eye(F)
        reg[0, 0] = 0.0
        coef.setdefault(str(k), {})[c] = np.linalg.solve(xtx + reg, xty).tolist()
    # a class with no training turbine (e.g. only in a held-out park) falls back to the other class
    for k in ("0", "1"):
        if k not in coef:
            coef[k] = coef["1" if k == "0" else "0"]
    return {"kind": "linear_thermal", "taus_h": TAUS_H, "gammas": GAMMAS, "amb_taus_h": AMB_TAUS_H, "coef": coef,
            "n_fit_steps": {f"{k}/{c}": int(v[2]) for (k, c), v in acc.items()}}


def residuals(model: dict, sensors: dict, err: np.ndarray, rated: np.ndarray, is_large: np.ndarray) -> np.ndarray:
    """Measured − expected temperature, °C, [T, N, C] float32 (NaN where the temperature is missing / coded)."""
    T, N = err.shape
    r = np.full((T, N, len(COMPONENTS)), np.nan, np.float32)
    for j in range(N):
        X, amb, bad = _turbine_inputs(sensors, err, rated, j)
        co = model["coef"][str(int(is_large[j]))]
        for ci, c in enumerate(COMPONENTS):
            if c not in co:
                continue
            exp = amb + X @ np.asarray(co[c])
            y = sensors[c][:, j].astype(np.float64)
            r[:, j, ci] = np.where(np.isfinite(y) & (err[:, j] == 0), y - exp, np.nan)
    return r
