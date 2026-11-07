"""Vectorised building blocks shared by the wind simulator, the fault injector and M6's stand-in synth (numpy only).

Normalisation and class power curves live in sentinel.model.wind.physics (one definition for simulator and model).
"""

from __future__ import annotations

import numpy as np
from scipy.signal import lfilter

STEPS_PER_H = 4
DT_H = 0.25

# Component temperatures, first-order thermal model (plan §4):
#   T_c = lag_tau( T_amb + base + offset_turbine + rise · load^gamma ) + noise
# sensor -> (base rise °C, full-load rise °C, time constant h, load exponent)
THERMAL: dict[str, tuple[float, float, float, float]] = {
    "conv_t": (8.0, 22.0, 0.5, 1.5),
    "gearbox_t": (15.0, 35.0, 3.0, 1.2),
    "gen1_t": (10.0, 40.0, 2.0, 1.5),
    "gen2_t": (8.0, 36.0, 2.0, 1.5),
    "transformer_t": (10.0, 45.0, 2.5, 2.0),
}
# Integral PPC-style error codes (arrive as integral floats, e.g. 5021.0; see sentinel.contracts.wind).
ERROR_CODES = (5021, 9999, 32767, 1208, 3104)


def ar1(rng, n: int, phi: float, sd: float, shape=()) -> np.ndarray:
    """Stationary AR(1) along axis 0 with marginal std `sd`."""
    e = rng.standard_normal((n, *shape)) * sd * np.sqrt(1 - phi**2)
    zi = (rng.standard_normal(shape) * sd * phi)[None] if shape else np.array([rng.standard_normal() * sd * phi])
    return lfilter([1.0], [1.0, -phi], e, axis=0, zi=zi)[0]


def lag(x: np.ndarray, tau_h: float) -> np.ndarray:
    """First-order thermal lag along axis 0 (15-min steps), started at steady state. Causal: a change at step t
    only affects steps >= t."""
    a = float(np.exp(-DT_H / tau_h))
    return lfilter([1 - a], [1.0, -a], x, axis=0, zi=(x[:1] * a))[0]


def wake_table(xy: np.ndarray, rotor_d: float, n_bins: int = 72, k: float = 0.075, ct: float = 0.8) -> np.ndarray:
    """Jensen (top-hat) wake deficit [n_bins, N] for wind FROM each 5° direction bin, root-sum-square over the
    upstream turbines within 15 D, smoothed over ±10°. xy in metres (east, north)."""
    n = len(xy)
    out = np.zeros((n_bins, n))
    a = 1 - np.sqrt(1 - ct)
    d = xy[None, :, :] - xy[:, None, :]  # d[i, j] = pos_j - pos_i
    for b in range(n_bins):
        th = np.deg2rad(b * 360.0 / n_bins)
        u = np.array([np.sin(th), np.cos(th)])  # unit vector pointing to where the wind comes from
        along = d @ u
        cross = np.abs(d[..., 0] * u[1] - d[..., 1] * u[0])
        r = rotor_d / 2 + k * along
        inw = (along > 0) & (along < 15 * rotor_d) & (cross < r)
        delta = np.where(inw, a * (rotor_d / (rotor_d + 2 * k * np.clip(along, 0, None))) ** 2, 0.0)
        out[b] = np.sqrt((delta**2).sum(1))
    kern = np.array([0.1, 0.2, 0.4, 0.2, 0.1])
    padded = np.concatenate([out[-2:], out, out[:2]])
    return np.stack([np.convolve(padded[:, j], kern, mode="valid") for j in range(n)], axis=1)


def wake_deficit(table: np.ndarray, wdir_deg: np.ndarray, v_free: np.ndarray) -> np.ndarray:
    """Deficit [T, N] for direction [T, N] and free hub wind [T, N]; thrust falls above rated, so wakes fade."""
    n_bins = table.shape[0]
    dbin = np.round(wdir_deg / (360.0 / n_bins)).astype(np.int64) % n_bins
    return table[dbin, np.arange(table.shape[1])[None]] * np.clip((15 - v_free) / 8, 0.15, 1.0)


def interp_circular(t_new: np.ndarray, t_old: np.ndarray, deg: np.ndarray) -> np.ndarray:
    """Interpolate directions (degrees) on the unit circle."""
    r = np.deg2rad(deg)
    s, c = np.interp(t_new, t_old, np.sin(r)), np.interp(t_new, t_old, np.cos(r))
    return np.rad2deg(np.arctan2(s, c)) % 360


def runs(mask: np.ndarray):
    """(start, end) of True runs in a 1-D bool array."""
    d = np.diff(np.concatenate([[0], mask.astype(np.int8), [0]]))
    return list(zip(np.flatnonzero(d == 1).tolist(), np.flatnonzero(d == -1).tolist()))
