"""Battery strategies: per-home battery behaviour for data-v2 prosumers.

Each home runs one strategy, stateful per 15-min step (SoC carries over), so it is a loop over time; it is
vectorised over homes. The numpy reference ``battery_strategies_py`` is the source of truth and the Rust copy
``wattsentry._core.battery_strategies`` matches it bit for bit (same float operations in the same order).

Strategies (codes in ``STRATEGY_CODES``), powers in kW, ``s = pv - load`` (house surplus):

- ``none`` (0): no battery, batt = 0.
- ``self_consumption`` (1, standard contract): s > 0 charges min(s, P, room); else discharges min(-s, P, avail)
  to cover the house only (never exports from the battery).
- ``price_driven`` (2, PPC MaxControl, price per 15 min): s > 0 charges from PV first; else in the home's
  ``k_top`` most expensive quarter-hours of the local day it discharges min(P, avail) (exports the rest);
  at prices >= the day's median it covers the house only; below the median it keeps the battery for the
  evening. On days with ``comply[home, day] == False`` it falls back to ``self_consumption``.
- ``enrolled`` (3, level 3 aggregator): behaves as ``self_consumption`` in this kernel.

room = max(cap - soc, 0) / (eta * dt), avail = max(soc - reserve * cap, 0) * eta / dt,
soc' = soc + eta * c * dt - d * dt / eta;  batt = c - d (> 0 charging, < 0 discharging).
"""

from __future__ import annotations

from typing import Any

import numpy as np

STRATEGY_CODES = {"none": 0, "self_consumption": 1, "price_driven": 2, "enrolled": 3}
STRATEGIES = tuple(STRATEGY_CODES)


def _prep(pv, load, strategy, cap, pmax, reserve, eta, k_top, soc0, price, rank, med, day, comply):
    pv = np.ascontiguousarray(pv, dtype=np.float64)
    load = np.ascontiguousarray(load, dtype=np.float64)
    if pv.ndim != 2 or pv.shape != load.shape:
        raise ValueError(f"pv and load must both be (n, T), got {pv.shape} and {load.shape}")
    n, t = pv.shape
    f = lambda a: np.ascontiguousarray(np.broadcast_to(np.asarray(a, dtype=np.float64), (n,)))  # noqa: E731
    strategy = np.ascontiguousarray(np.broadcast_to(np.asarray(strategy, dtype=np.int8), (n,)))
    k_top = np.ascontiguousarray(np.broadcast_to(np.asarray(k_top, dtype=np.int32), (n,)))
    g = lambda a, dt: np.ascontiguousarray(np.asarray(a, dtype=dt).reshape(t))  # noqa: E731
    price, med, rank, day = g(price, np.float64), g(med, np.float64), g(rank, np.int32), g(day, np.int32)
    comply = np.ascontiguousarray(np.asarray(comply, dtype=np.uint8))
    if comply.ndim != 2 or comply.shape[0] != n:
        raise ValueError(f"comply must be (n, n_days), got {comply.shape}")
    if t and (day.min() < 0 or day.max() >= comply.shape[1]):
        raise ValueError("day indexes must lie in [0, comply.shape[1])")
    return pv, load, strategy, f(cap), f(pmax), f(reserve), f(eta), k_top, f(soc0), price, rank, med, day, comply


def battery_strategies_py(pv: Any, load: Any, strategy: Any, cap: Any, pmax: Any, reserve: Any, eta: Any,
                          k_top: Any, soc0: Any, price: Any, rank: Any, med: Any, day: Any, comply: Any,
                          dt_h: float = 0.25) -> tuple[np.ndarray, np.ndarray]:
    """pv, load (n, T) kW; per home (n,): strategy code (int8), cap kWh, pmax kW, reserve fraction, eta (one-way),
    k_top (int), soc0 kWh; per step (T,): customer price (lei/MWh), rank of the price within its local day
    (0 = most expensive), the day's median price, local day index into comply (n, n_days) bool.
    Returns (batt kW (n, T), soc kWh at the END of each step (n, T))."""
    (pv, load, strategy, cap, pmax, reserve, eta, k_top, soc, price, rank, med, day,
     comply) = _prep(pv, load, strategy, cap, pmax, reserve, eta, k_top, soc0, price, rank, med, day, comply)
    n, T = pv.shape
    dt = float(dt_h)
    batt = np.empty((n, T))
    socs = np.empty((n, T))
    has = strategy != 0
    pdriven = strategy == 2
    floor = reserve * cap
    soc = soc.copy()
    for t in range(T):
        s = pv[:, t] - load[:, t]
        room = np.maximum(cap - soc, 0.0) / (eta * dt)
        avail = np.maximum(soc - floor, 0.0) * eta / dt
        charge = np.minimum(np.minimum(s, pmax), room)
        cover = np.minimum(np.minimum(-s, pmax), avail)
        full = np.minimum(pmax, avail)
        mode = pdriven & (comply[:, day[t]] != 0)
        top = rank[t] < k_top
        dis = np.where(mode, np.where(top, full, cover if price[t] >= med[t] else 0.0), cover)
        c = np.where(has & (s > 0), charge, 0.0)
        d = np.where(has & ~(s > 0), dis, 0.0)
        soc = soc + eta * c * dt - d * dt / eta
        batt[:, t] = c - d
        socs[:, t] = soc
    return batt, socs


def battery_strategies_rust(pv: Any, load: Any, strategy: Any, cap: Any, pmax: Any, reserve: Any, eta: Any,
                            k_top: Any, soc0: Any, price: Any, rank: Any, med: Any, day: Any, comply: Any,
                            dt_h: float = 0.25) -> tuple[np.ndarray, np.ndarray]:
    from wattsentry import kernels

    args = _prep(pv, load, strategy, cap, pmax, reserve, eta, k_top, soc0, price, rank, med, day, comply)
    return kernels._load_core().battery_strategies(*args, float(dt_h))


def day_rank_median(price: np.ndarray, day: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per step: rank of ``price`` within its local day (0 = most expensive; ties -> earlier step first) and the
    day's median. ``day`` is a non-decreasing local day index (wattsentry.sim.fleet.ro_calendar.local_features)."""
    price = np.asarray(price, dtype=np.float64)
    day = np.asarray(day, dtype=np.int64)
    idx = np.arange(len(price))
    order = np.lexsort((idx, -price, day))
    first = np.searchsorted(day[order], day[order], side="left")
    rank = np.empty(len(price), np.int32)
    rank[order] = (np.arange(len(price)) - first).astype(np.int32)
    med = np.empty(len(price))
    if len(price):
        bounds = np.flatnonzero(np.diff(day)) + 1
        for a, b in zip(np.r_[0, bounds], np.r_[bounds, len(price)]):
            med[a:b] = np.median(price[a:b])
    return rank, med
