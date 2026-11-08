"""Telemetry scatter: long-format readings -> dense [sensor, step, turbine] bin means (WindNBM load).

    from wattsentry.kernels import scatter_telemetry
    sens, err = scatter_telemetry(ti, tj, sk, v, is_code, scale, T)
    # ti/tj/sk int64 [n]: step, turbine, sensor of each row (out of range / -1 = dropped); v f64 [n]; is_code bool [n];
    # scale f64 [S, N]: multiplier per (sensor, turbine).
    # sens f32 [S, T, N]: mean of the finite non-code v * scale in each bin, NaN if none or if err != 0;
    # err i32 [T, N]: last code in the bin (rows order), 0 if none.

The reference does one bincount pair per sensor. The Rust copy ``wattsentry._core.scatter_telemetry``
(rust/wattsentry_core/src/scatter.rs) does one pass in row order, so the sums add in the same order as bincount and the
result is bit-identical.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def _prep(ti: Any, tj: Any, sk: Any, v: Any, is_code: Any, scale: Any) -> tuple[np.ndarray, ...]:
    ti, tj, sk = (np.ascontiguousarray(a, dtype=np.int64) for a in (ti, tj, sk))
    v, is_code = np.ascontiguousarray(v, dtype=np.float64), np.ascontiguousarray(is_code, dtype=bool)
    scale = np.ascontiguousarray(scale, dtype=np.float64)
    if scale.ndim != 2 or not (len(ti) == len(tj) == len(sk) == len(v) == len(is_code)):
        raise ValueError("ti, tj, sk, v, is_code must have the same length and scale must be [S, N]")
    return ti, tj, sk, v, is_code, scale


def scatter_telemetry_py(ti: Any, tj: Any, sk: Any, v: Any, is_code: Any, scale: Any,
                         T: int) -> tuple[np.ndarray, np.ndarray]:
    ti, tj, sk, v, is_code, scale = _prep(ti, tj, sk, v, is_code, scale)
    (S, N), T = scale.shape, int(T)
    ok = (tj >= 0) & (tj < N) & (ti >= 0) & (ti < T)
    sens = np.empty((S, T, N), np.float32)
    for k in range(S):
        m = ok & (sk == k) & ~is_code & np.isfinite(v)
        flat = ti[m] * N + tj[m]
        cnt = np.bincount(flat, minlength=T * N)
        sm = np.bincount(flat, weights=v[m] * scale[k, tj[m]], minlength=T * N)
        with np.errstate(invalid="ignore", divide="ignore"):
            sens[k] = np.where(cnt > 0, sm / np.maximum(cnt, 1), np.nan).reshape(T, N)
    err = np.zeros((T, N), dtype=np.int32)
    mc = ok & is_code
    if mc.any():
        with np.errstate(invalid="ignore"):  # last code in the bin wins; NaN / inf codes -> 0 (as in Rust)
            err.reshape(-1)[ti[mc] * N + tj[mc]] = v[mc].astype(np.int64).astype(np.int32)
        sens[:, err != 0] = np.nan
    return sens, err


def scatter_telemetry_rust(ti: Any, tj: Any, sk: Any, v: Any, is_code: Any, scale: Any,
                           T: int) -> tuple[np.ndarray, np.ndarray]:
    from wattsentry import kernels

    return kernels._load_core().scatter_telemetry(*_prep(ti, tj, sk, v, is_code, scale), int(T))
