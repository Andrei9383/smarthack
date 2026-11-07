"""W2 (card #165, spec W step 3): park forecast for tomorrow from M5 turbine quantiles x availability, as scenarios.

    from sentinel.model.windpower import WindPowerModel
    from sentinel.model.windpower.scenarios import park_forecast, ParkConformal
    m = WindPowerModel.load()                                     # or from_json(..., binned=True)
    fc = park_forecast(m, turbines, ts, meteo, weather, avail=av, conformal=cal)   # av: availability.Availability
    fc["WP1"].q           # [T, 3] kW P10/P50/P90 per 15-min step (conformal applied)
    fc["WP1"].scenarios   # [S, T] kW scenario paths (same calibration map), for the sell quantile (W3) / battery (W5)
    fc["WP1"].quantile(0.6)   # [T] kW any level, from the scenarios

Steps (spec):
1. One wind quantile level per park and scenario, shared by all its turbines (neighbours see the same wind); the
   level follows a Gaussian AR(1) in time (rho per 15 min) so a scenario is a plausible day, not white noise.
2. Each turbine's power at that level (piecewise-linear through its P10/P50/P90, linear tails, clipped to
   [0, rated]) x derate x (p_up, or an on/off draw from p_up when bernoulli=True), summed over the park, x the
   park's wake ratio for the current direction sector (M5's sidecar, pooled for an unseen park).
3. P10/P50/P90 of the park from the scenarios.
4. Split-conformal on the park total (`ParkConformal`, fitted on validation park-days): each tail's distance to
   P50 is rescaled so that 10 % of calibration totals fall below P10 and 10 % above P90 (target coverage 78-82 %).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import erf, sqrt
from typing import Any, Mapping, Optional, Sequence

import numpy as np

from .model import N_SEC, WindPowerModel, _sector

LEVELS = np.array([0.1, 0.5, 0.9])


def _ndtr(z: np.ndarray) -> np.ndarray:
    try:
        from scipy.special import ndtr

        return ndtr(z)
    except ImportError:  # pragma: no cover
        return 0.5 * (1 + np.vectorize(erf)(z / sqrt(2)))


def quantile_at(q: np.ndarray, u: np.ndarray, cap: Optional[np.ndarray] = None) -> np.ndarray:
    """Value at level u of a distribution given by P10/P50/P90 (q [..., 3]); u broadcastable to q[..., 0].
    Piecewise linear between the knots, the outer segments' slopes continued in the tails, clipped to [0, cap]."""
    q10, q50, q90 = q[..., 0], q[..., 1], q[..., 2]
    lo = q50 + (u - 0.5) * (q50 - q10) / 0.4
    hi = q50 + (u - 0.5) * (q90 - q50) / 0.4
    x = np.where(u < 0.5, lo, hi)
    x = np.maximum(x, 0.0)
    return x if cap is None else np.minimum(x, cap)


def common_levels(n_scen: int, T: int, n_parks: int, rho: float = 0.97, seed: int = 0) -> np.ndarray:
    """[S, T, P] uniform levels: one per park, AR(1) in time with lag-1 correlation rho (stationary N(0,1) -> Phi)."""
    rng = np.random.default_rng(seed)
    e = rng.standard_normal((n_scen, T, n_parks))
    z = np.empty_like(e)
    z[:, 0] = e[:, 0]
    s = sqrt(max(1.0 - rho * rho, 0.0))
    for t in range(1, T):
        z[:, t] = rho * z[:, t - 1] + s * e[:, t]
    return _ndtr(z)


def wake_ratio(model: WindPowerModel, turbines: Sequence[Mapping], meteo: Mapping, weather: Mapping,
               T: int) -> tuple[list[str], np.ndarray]:
    """Park ids and the M5 wake ratio of each park's current direction sector, [T, P]."""
    parks, sec = _sector(turbines, meteo, weather, T)
    ratios = model.meta.get("wake_ratio", {})
    pooled = np.asarray(model.meta.get("wake_ratio_pooled", np.ones(N_SEC)), np.float64)
    r = np.stack([np.asarray(ratios.get(p, pooled), np.float64)[sec[:, k]] for k, p in enumerate(parks)], 1)
    return parks, r


def park_scenarios(q_kw: np.ndarray, rated_kw: np.ndarray, cols: np.ndarray, ratio_t: np.ndarray, u: np.ndarray,
                   p_up: Optional[np.ndarray] = None, derate: Optional[np.ndarray] = None,
                   bernoulli: bool = False, rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """One park: [S, T] kW. q_kw [T, N, 3] (all turbines), cols = this park's turbine indices, ratio_t [T],
    u [S, T] common levels; p_up / derate [T, N] (None = 1)."""
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


def scen_quantiles(scen: np.ndarray, levels: Sequence[float] = (0.1, 0.5, 0.9)) -> np.ndarray:
    """[S, T] -> [T, len(levels)]."""
    return np.moveaxis(np.quantile(scen, list(levels), axis=0), 0, -1)


@dataclass
class ParkConformal:
    """Split-conformal per tail on park totals (fraction of park capacity). a / b scale P50 - P10 and P90 - P50;
    eps (fraction of capacity) is added to each half-width before scaling so zero-width bands (calm) can open.
    `by_park` overrides the pooled (a, b) for parks with enough calibration rows."""

    a: float = 1.0
    b: float = 1.0
    eps: float = 0.002
    by_park: dict = field(default_factory=dict)
    n: int = 0

    @staticmethod
    def _tail(x: np.ndarray, alpha: float) -> float:
        n = len(x)
        if n == 0:
            return 1.0
        k = min(np.ceil((n + 1) * (1 - alpha)) / n, 1.0)
        return float(max(np.quantile(x, k), 0.0))

    @classmethod
    def fit(cls, q: np.ndarray, y: np.ndarray, park: Optional[np.ndarray] = None, alpha: float = 0.1,
            eps: float = 0.002, min_rows: int = 2000) -> "ParkConformal":
        """q [M, 3], y [M] park totals as fractions of capacity (calibration rows, NaN rows dropped)."""
        ok = np.isfinite(y) & np.isfinite(q).all(-1)
        q, y = q[ok], y[ok]
        park = None if park is None else np.asarray(park)[ok]

        def ab(qq, yy):
            lo = (qq[:, 1] - yy) / (qq[:, 1] - qq[:, 0] + eps)
            hi = (yy - qq[:, 1]) / (qq[:, 2] - qq[:, 1] + eps)
            return cls._tail(lo, alpha), cls._tail(hi, alpha)

        a, b = ab(q, y)
        by = {}
        if park is not None:
            for p in np.unique(park):
                m = park == p
                if m.sum() >= min_rows:
                    by[str(p)] = ab(q[m], y[m])
        return cls(a, b, eps, by, int(len(y)))

    def ab(self, park: Optional[str] = None) -> tuple[float, float]:
        return tuple(self.by_park.get(str(park), (self.a, self.b))) if park is not None else (self.a, self.b)

    def apply(self, q: np.ndarray, cap: float = 1.0, park: Optional[str] = None,
              scen: Optional[np.ndarray] = None) -> tuple[np.ndarray, Optional[np.ndarray]]:
        """q [..., 3] (kW with cap = park capacity kW, or fractions with cap 1); scen [S, T] mapped the same way."""
        a, b = self.ab(park)
        e = self.eps * cap
        p50 = q[..., 1]
        lo, hi = p50 - q[..., 0] + e, q[..., 2] - p50 + e
        out = np.stack([np.clip(p50 - a * lo, 0, cap), p50, np.clip(p50 + b * hi, 0, cap)], -1)
        if scen is not None:
            d = scen - p50[None]
            sl = np.where(d < 0, a * lo / np.maximum(lo - e, 1e-9), b * hi / np.maximum(hi - e, 1e-9))
            scen = np.clip(p50[None] + d * np.minimum(sl, 50.0), 0, cap)
        return out, scen

    def to_json(self) -> dict:
        return {"a": self.a, "b": self.b, "eps": self.eps, "by_park": self.by_park, "n": self.n}

    @classmethod
    def from_json(cls, d: Mapping) -> "ParkConformal":
        return cls(float(d["a"]), float(d["b"]), float(d.get("eps", 0.002)),
                   {k: tuple(v) for k, v in d.get("by_park", {}).items()}, int(d.get("n", 0)))


@dataclass
class ParkForecast:
    park_id: str
    ts: np.ndarray  # [T] datetime64[m] UTC
    cap_kw: float
    q: np.ndarray  # [T, 3] kW, calibrated
    q_raw: np.ndarray  # [T, 3] kW, scenario quantiles before conformal
    scenarios: np.ndarray  # [S, T] kW, calibrated map
    availability_mean: float  # capacity-weighted mean availability of the park
    n_uncertain: int  # turbines whose availability is marked uncertain

    def quantile(self, tau: float) -> np.ndarray:
        """[T] kW at any level (e.g. the sell quantile q* = a / (a + b), W3), from the calibrated scenarios."""
        return np.quantile(self.scenarios, tau, axis=0)

    def energy_mwh(self) -> np.ndarray:
        """[S] park energy per scenario over the horizon (MWh)."""
        return self.scenarios.sum(1) * 0.25 / 1000.0


def park_forecast(model: WindPowerModel, turbines: Sequence[Mapping], ts, meteo: Mapping, weather: Mapping,
                  avail: Any = None, conformal: Optional[ParkConformal] = None, n_scen: int = 200,
                  rho: float = 0.97, bernoulli: bool = False, seed: int = 0,
                  q_kw: Optional[np.ndarray] = None) -> dict[str, ParkForecast]:
    """Per-park P10/P50/P90 (kW, per 15-min step) and scenarios for the steps `ts`.

    model: M5 (WindPowerModel); turbines / meteo / weather: as WindPowerModel.predict (weather per turbine [T, N]
    or [T, 1]); avail: availability.Availability over the same ts and turbine order (None = all available, i.e.
    `windpower_only`); conformal: fitted ParkConformal (None = raw scenario quantiles); q_kw: precomputed
    model.predict output (skips the M5 call)."""
    ts = np.asarray(ts).astype("datetime64[m]")
    T = len(ts)
    if q_kw is None:
        q_kw = model.predict(turbines, ts, meteo, weather)
    rated = np.array([float(t["rated_kw"]) for t in turbines])
    parks, ratio = wake_ratio(model, turbines, meteo, weather, T)
    pid = np.array([str(t["park_id"]) for t in turbines])
    u = common_levels(n_scen, T, len(parks), rho, seed)
    rng = np.random.default_rng(seed + 1)
    p_up = derate = None
    unc = np.zeros(len(turbines), bool)
    if avail is not None:
        if list(avail.turbine_ids) != [str(t["turbine_id"]) for t in turbines]:
            raise ValueError("availability turbine order differs from `turbines`")
        p_up, derate, unc = avail.p_up, avail.derate, avail.uncertain
    out = {}
    for k, park in enumerate(parks):
        cols = np.flatnonzero(pid == park)
        cap = float(rated[cols].sum())
        scen = park_scenarios(q_kw, rated, cols, ratio[:, k], u[..., k], p_up, derate, bernoulli, rng)
        raw = scen_quantiles(scen)
        q, sc = (raw, scen) if conformal is None else conformal.apply(raw, cap, park, scen)
        a = 1.0 if avail is None else float((avail.a[:, cols] * rated[cols]).sum() / (cap * T))
        out[park] = ParkForecast(park, ts, cap, q, raw, sc, a, int(unc[cols].sum()))
    return out


__all__ = ["LEVELS", "ParkConformal", "ParkForecast", "common_levels", "park_forecast", "park_scenarios",
           "quantile_at", "scen_quantiles", "wake_ratio"]
