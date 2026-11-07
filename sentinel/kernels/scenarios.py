"""K9 park_scenarios (#187): one park's [S, T] kW scenarios for the W2 forecast stage (avail_eval).

The reference ``park_scenarios_py`` is ``sentinel.model.windpower.scenarios.park_scenarios`` moved here verbatim
(84 % of t_forecast_s in the #187 profile: numpy temporaries over [S, T, n] per day x park x policy). The Rust copy
``sentinel._core.park_scenarios`` (rust/sentinel_core/src/scenarios.rs) fuses it with numpy's op order and its
sequential turbine sum (numpy's reduce order for the [S][n][T] layout of the fancy-indexed q), bit for bit.
The Bernoulli draws stay in numpy (same rng stream) and are passed in.
"""

from __future__ import annotations

from typing import Optional

import numpy as np


def park_scenarios_py(q_kw: np.ndarray, rated_kw: np.ndarray, cols: np.ndarray, ratio_t: np.ndarray, u: np.ndarray,
                      p_up: Optional[np.ndarray] = None, derate: Optional[np.ndarray] = None,
                      bernoulli: bool = False, rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """One park: [S, T] kW. q_kw [T, N, 3] (all turbines), cols = this park's turbine indices, ratio_t [T],
    u [S, T] common levels; p_up / derate [T, N] (None = 1)."""
    from sentinel.model.windpower.scenarios import quantile_at

    q = q_kw[:, cols]
    x = quantile_at(q[None], u[..., None], rated_kw[cols][None, None])  # [S, T, n]
    if derate is not None:
        x = x * derate[:, cols][None]
    if p_up is not None:
        pu = p_up[:, cols]
        if bernoulli:  # one draw per scenario and turbine: with p_up rising in time (repair) this is one switch-on
            v = (rng or np.random.default_rng(0)).random((u.shape[0], 1, len(cols)))
            x = x * (v < pu[None])
        else:
            x = x * pu[None]
    return x.sum(-1) * ratio_t[None]


def park_scenarios_rust(q_kw, rated_kw, cols, ratio_t, u, p_up=None, derate=None, bernoulli=False, rng=None):
    from sentinel import kernels

    q_kw, u = np.asarray(q_kw), np.asarray(u)
    # other dtypes (numpy's promotion), slice cols or T == 1 (numpy then sums pairwise): the reference
    if q_kw.dtype not in (np.float32, np.float64) or u.dtype != np.float64 or np.ndim(cols) != 1 or u.shape[1] < 2:
        return park_scenarios_py(q_kw, rated_kw, cols, ratio_t, u, p_up, derate, bernoulli, rng)
    f = lambda a: None if a is None else np.ascontiguousarray(a, np.float64)  # noqa: E731  (f64 x f32 -> f64 anyway)
    cols = np.ascontiguousarray(np.arange(q_kw.shape[1])[cols], np.int64)
    v = None
    if p_up is not None and bernoulli:
        v = (rng or np.random.default_rng(0)).random((u.shape[0], 1, len(cols)))[:, 0]
    return kernels._load_core().park_scenarios(np.ascontiguousarray(q_kw), f(rated_kw), cols,
                                               f(np.broadcast_to(ratio_t, u.shape[1:])), np.ascontiguousarray(u),
                                               f(p_up), f(derate), v)
