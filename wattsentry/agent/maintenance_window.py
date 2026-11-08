"""Cheapest repair day for wind turbines flagged by the early-warning model (spec W step 6 / D3, card #169).

Gate: M7 lead_time_h > 0 on unseen events (T6 #144: lead_time_h_holdout = 164.75 / 64.875 h, real RO), so there is
time to plan instead of react.

For each turbine t and day d of the horizon (default 10 days, 96 quarter-hours/day, UTC):
  planned[t, d] = lei lost while the turbine is stopped for the repair window on day d (power x price x 0.25 h)
  risk[t, d]    = P(fail before d) x unplanned extra loss (extra_days x mean daily revenue + crew premium)
  cost          = planned + risk, inf when wind at hub exceeds the climbing limit during the repair window.
Baselines: reactive (wait for the failure) and fixed calendar (first climbable day >= calendar_day).

Prices: PZU where published (``wattsentry.data.prices`` from #158, real RO OPCOM), otherwise the last-week
quarter-hour profile (proxy). Pure numpy: T x 10 x 96 is tiny, so no Rust copy (team/infra.md).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

QH = 96  # quarter-hours per day
WIND_LIMIT_MS = 12.0  # common OEM limit for climbing / nacelle work


@dataclass
class RepairPlan:
    best_day: np.ndarray  # [T] int, -1 when no climbable day in the horizon
    cost: np.ndarray  # [T, D] lei, inf = not climbable
    planned: np.ndarray  # [T, D] lei
    risk: np.ndarray  # [T, D] lei
    best_cost: np.ndarray  # [T] lei (reactive cost when no climbable day)
    reactive_cost: np.ndarray  # [T] lei
    calendar_cost: np.ndarray  # [T] lei
    label: str

    @property
    def saving_vs_reactive(self) -> np.ndarray:
        return self.reactive_cost - self.best_cost

    @property
    def saving_vs_calendar(self) -> np.ndarray:
        return self.calendar_cost - self.best_cost


def price_profile(price: np.ndarray, history: np.ndarray | None = None) -> np.ndarray:
    """Fill missing (NaN) days of a ``[D, 96]`` price grid with the last-week quarter-hour profile.

    ``history``: ``[k, 96]`` lei/MWh of the last k days (k >= 1, typically 7) before the horizon."""
    p = np.array(price, dtype=float)
    if history is None or not np.isnan(p).any():
        return p
    prof = np.nanmean(np.asarray(history, float)[-7:], axis=0)
    return np.where(np.isnan(p), prof[None, :], p)


def load_price_grid(t0, days: int = 10) -> tuple[np.ndarray, str]:
    """``[days, 96]`` lei/MWh from UTC midnight ``t0``: PZU where known, else last-week profile. Returns (grid, label)."""
    from wattsentry.data.prices import prices_at  # #158

    start = np.datetime64(t0, "D").astype("datetime64[m]")
    step = np.timedelta64(15, "m")
    ts = start + step * np.arange(-7 * QH, days * QH)
    pzu, _ = prices_at(ts, strict=False)
    hist, fut = pzu[: 7 * QH].reshape(7, QH), pzu[7 * QH:].reshape(days, QH)
    n_known = int((~np.isnan(fut).all(axis=1)).sum())
    return price_profile(fut, hist), f"PZU real RO for {n_known}/{days} days, last-week profile proxy for the rest"


def best_repair_day(power_mw, wind_ms, price, p_fail_day, *, repair_qh: int = 32, work_start_qh: int = 24,
                    wind_limit: float = WIND_LIMIT_MS, unplanned_extra_days: float = 3.0,
                    unplanned_premium_lei: float = 0.0, calendar_day: int = 0, label: str = "") -> RepairPlan:
    """Cheapest climbable repair day per turbine.

    power_mw: ``[T, D, 96]`` or ``[D, 96]`` expected output (MW); wind_ms: same shape or ``[D, 96]`` (hub height);
    price: ``[D, 96]`` lei/MWh; p_fail_day: ``[T]`` daily failure hazard. Repair occupies quarter-hours
    ``[work_start_qh, work_start_qh + repair_qh)`` of the chosen day (default 06:00-14:00 UTC, 8 h)."""
    pw = np.asarray(power_mw, float)
    pw = pw[None] if pw.ndim == 2 else pw
    T, D, _ = pw.shape
    w = np.broadcast_to(np.asarray(wind_ms, float), (T, D, QH))
    pr = np.asarray(price, float)[None]
    h = np.clip(np.broadcast_to(np.asarray(p_fail_day, float), (T,)), 0.0, 1.0)

    win = slice(work_start_qh, work_start_qh + repair_qh)
    rev = pw * pr * 0.25  # lei per quarter-hour
    planned = rev[:, :, win].sum(-1)
    climbable = w[:, :, win].max(-1) <= wind_limit
    unplanned = unplanned_extra_days * rev.sum(-1).mean(-1) + unplanned_premium_lei  # [T]
    p_before = 1.0 - (1.0 - h[:, None]) ** np.arange(D)[None, :]  # failure before the repair day
    risk = p_before * unplanned[:, None]
    cost = np.where(climbable, planned + risk, np.inf)

    best = np.where(climbable.any(1), cost.argmin(1), -1)
    reactive = (1.0 - (1.0 - h) ** D) * unplanned + planned.mean(1)  # fails in horizon -> unplanned; repair anyway
    best_cost = np.where(best >= 0, cost[np.arange(T), np.maximum(best, 0)], reactive)
    later = climbable & (np.arange(D)[None, :] >= calendar_day)
    cal_day = np.where(later.any(1), later.argmax(1), -1)
    calendar = np.where(cal_day >= 0, cost[np.arange(T), np.maximum(cal_day, 0)], reactive)
    return RepairPlan(best, cost, planned, risk, best_cost, reactive, calendar, label)
