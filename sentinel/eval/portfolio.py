"""Portfolio aggregation for the supplier's day-ahead purchase (card #175, spec I step 6, "Adunarea pe portofoliu").

The supplier's position per 15 minutes = sum over its prosumers of (import - export). Quantiles do not add up (the sum
of P90s is not the P90 of the sum), so the total comes from K coherent scenarios:

1. Per-home P10/P50/P90 for PV (M1) and load (M2), read through the Judge adapter (C1 #157, ``validated_bands``).
2. K = 200 scenarios: ONE PV quantile level per region per scenario (PV errors are weather errors, shared by
   neighbours; Latin hypercube over scenarios), an independent load level per home. A level maps to a value through
   the home's three quantiles (piecewise linear in the normal score, split-normal tails; ``conformal.quantile_at``).
3. Every scenario goes through the battery behaviour (``behaviour``; current fleet = the self-consumption rule
   through the K1 kernel ``sentinel.kernels.battery_rollout``, Rust when built; data-v2 = I4's model, one swap).
4. total[k, t] = sum over homes of (import - export), MWh per step. P10/P50/P90 over scenarios.
5. Split-conformal on the total (``TotalConformal``): per 6-h bucket one scale s that stretches (or shrinks) the
   scenario spread around P50 so that the P10-P90 band covers 80 % of validation days. Applied to the scenarios
   themselves, so every quantile (the purchase quantile too) is calibrated the same way.
6. bought_da = quantile q* = a / (a + b) of the calibrated scenarios (a, b = frozen imbalance penalties, X1 #158).
7. 15:00 correction (level 2): re-forecast 16:00-24:00 local with what is known at 15:00 (D-1 meter data, updated
   weather), same steps 2-6; traded_id[t] = q*_new[t] - bought_da[t] on those steps, 0 elsewhere.

Sign convention = consumption (the supplier buys), as in ``sentinel.eval.imbalance``: total > 0 = net import.

API (I6 #176 and the replay build on these; arrays are numpy, kW per home, MWh per step for the portfolio)
---------------------------------------------------------------------------------------------------------------
    Fleet(site_ids, region, regions, battery, pv_cap)          static per-home inputs; Fleet.from_sites(rows)
    validated_bands(model, site_ids, days, q_raw, *, substitutes=None, fallback=None, judge=None)
        -> (q (N, T, 3), known (N,), counts)   the ONE place per-home bands are read (C1 adapter, validated_many)
    draw_levels(n_scen, region, n_regions, seed=0) -> (u_pv (K, R), u_load (K, N))
    behaviour(pv, load, battery, soc0=None) -> net import kW, pv's shape (..., N, T)   (set_behaviour(fn) swaps it)
    warm_soc0(pv50, load50, battery) -> soc0 (N,) fraction   (midnight SoC the supplier cannot see; steady state)
    portfolio_scenarios(pv_q, load_q, fleet, *, n_scen=200, seed=0, soc0=None, battery=True, chunk=256)
        -> total (K, T) MWh/step
    scenario_bands(total, levels=(0.1, 0.5, 0.9)) -> (T, 3)
    TotalConformal.fit(bands (D, T, 3), actual (D, T), target=0.8) -> TotalConformal; .apply(bands);
        .stretch(values (..., T), p50 (T,))
    purchase_quantile(a=None, b=None) -> q* = a / (a + b)
    bought_da(total, q_star, conformal=None) -> (T,)
    correction_mask(ts_utc (T,), start_hour=16, end_hour=24) -> bool (T,)
    traded_id(bought, total_new, q_star, mask, conformal=None) -> (T,)
    forecast_day(pv_q, load_q, fleet, *, conformal=None, q_star=None, ...) -> DayForecast (total, bands, bought)

Data layer for the current fleet (#10 windows, #11 fleet, M1/M2 ONNX in models/), used by ``main``:
    fleet_from_source(src, region_key="county") -> Fleet
    raw_bands(src, sites, day_idx, *, kind, models, mode, lag_days) -> q (n_days, N, T, 3) kW   (the swap point
        for data-v2 / I4: return per-home raw bands from whatever source)
    actual_net(src, sites, day_idx) -> (n_days, N, T) kW import - export (meter truth)

Run (heavy: on the lab, never locally):
    python -m sentinel.eval.portfolio [--val-days 120 --test-days 180 --k 200 --seed 0]
    -> LAB_METRIC portfolio_nmae, portfolio_coverage_p10_p90 (test 2025, synthetic RO), + the val/raw/intraday ones;
       $LAB_OUTPUT_DIR/portfolio_forecast.parquet (issue_time, ts, policy, p10, p50, p90),
       positions.parquet (ts, policy, bought_da, traded_id, actual), portfolio_report.json.

Day convention: a portfolio day is one #10 window (96 steps from UTC midnight, as M1/M2 were trained and split);
outputs carry UTC ``ts`` per step, so I6 joins prices with ``prices_at(ts)`` and local slices (16-24, 17-21) are
taken by local clock time (``correction_mask``). The adapter day key is that UTC date.

Labels: portfolio, homes, batteries and weather = synthetic RO (fleet simulator, ERA5-driven); a, b = proxy
(calibrated on AFEER's public 200-375 lei/MWh, X1 #158). The intraday re-forecast uses the nowcast PVNet on ERA5
weather as the "updated weather at 15:00" (a proxy: an upper bound for a 1-9 h weather update).

Pending stubs (each is a single function): ``validated_bands`` passes predictions through as verdict "use" until
``sentinel.judge.adapter`` (#157) is on main; ``purchase_quantile`` reads models/imbalance_ab.json, else the frozen
values of #158 (a = 1.4407, b = 0.7204), until ``sentinel.eval.imbalance`` (#158) is on main; ``behaviour`` = the
self-consumption kernel until I4 (#174) registers its model with ``set_behaviour``.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, NamedTuple, Optional, Sequence

import numpy as np
from scipy.special import ndtri

from sentinel.eval.uncertainty.conformal import Z90
from sentinel.kernels import battery_rollout

H = 96  # 15-min steps per day
DT_H = 0.25
KW_TO_MWH = DT_H / 1000.0  # kW over one step -> MWh
N_SCENARIOS = 200
LEVELS = (0.1, 0.5, 0.9)
COVER_TARGET = 0.80
BUCKETS = ((0, 24), (24, 48), (48, 72), (72, 96))  # 6-h horizon buckets of the conformal scale
LEVEL_CLIP = 1e-3
ROOT = Path(__file__).resolve().parents[2]
AB_PATH = ROOT / "models" / "imbalance_ab.json"
AB_STUB = (1.4407, 0.7204)  # #158's frozen a, b (a = 2b), used only if neither the module nor the JSON is present
LABEL = "synthetic RO"


# ================================================================================================ static inputs
class BatteryArrays(NamedTuple):
    """Per-home battery arrays (N,), the K1 kernel's argument order. Energies kWh, powers kW, eta one-way.
    e_lo = the physical floor soc_min * E (self-consumption rule, as ``BatterySpec.step``); E = 0 means no battery."""

    capacity_kwh: np.ndarray
    p_max_kw: np.ndarray
    eta_charge: np.ndarray
    eta_discharge: np.ndarray
    e_lo_kwh: np.ndarray
    e_hi_kwh: np.ndarray

    def take(self, idx: Any) -> "BatteryArrays":
        return BatteryArrays(*(np.ascontiguousarray(a[idx]) for a in self))


@dataclass
class Fleet:
    """The portfolio's homes. region: (N,) int code into ``regions`` (one shared PV level per region)."""

    site_ids: np.ndarray
    region: np.ndarray
    regions: list[str]
    battery: BatteryArrays
    pv_cap: np.ndarray  # (N,) kW upper clip of PV scenarios (inverter rating), inf = none

    def __len__(self) -> int:
        return len(self.site_ids)

    @classmethod
    def from_sites(cls, rows: Sequence[Mapping[str, Any]], region_key: str = "county") -> "Fleet":
        """From sites-table rows (#11 sites.parquet / windows sites.json): battery_kwh, battery_kw, eta_*, soc_*."""
        def col(k: str, default: float) -> np.ndarray:
            return np.array([float(r.get(k) if r.get(k) is not None else default) for r in rows])

        cap = col("battery_kwh", 0.0)
        names = [str(r.get(region_key) or "?") for r in rows]
        regions = sorted(set(names))
        code = {n: i for i, n in enumerate(regions)}
        bat = BatteryArrays(cap, np.where(cap > 0, col("battery_kw", 0.0), 0.0), col("eta_charge", 0.95),
                            col("eta_discharge", 0.95), col("soc_min", 0.05) * cap, col("soc_max", 1.0) * cap)
        inv = col("inverter_kw", np.inf)
        return cls(np.array([str(r.get("site_id")) for r in rows], dtype=object),
                   np.array([code[n] for n in names], dtype=np.int64), regions, bat, np.where(inv > 0, inv, np.inf))


# ================================================================================================ 1. per-home bands
def _passthrough(model: str, site_ids: Sequence[Any], days: Any, q: np.ndarray):
    n = len(site_ids)
    return q, np.ones(n, bool), {"use": n, "widen": 0, "dont_use": 0, "unknown": 0, "substituted": 0,
                                 "adapter": "stub (#157 not on main): passthrough, verdict use"}


def validated_bands(model: str, site_ids: Sequence[Any], days: Any, q_raw: np.ndarray, *,
                    substitutes: Optional[Mapping[str, np.ndarray]] = None, fallback: Optional[np.ndarray] = None,
                    judge: Any = None) -> tuple[np.ndarray, np.ndarray, dict]:
    """Per-home bands as the Judge allows them to be used (team/judge-design.md §7.7, C1 adapter #157).

    q_raw: (N, T, 3) raw q10/q50/q90 of ``model`` ('pvnet' | 'loadnet'), rows aligned with site_ids; days: one day
    'YYYY-MM-DD' or (N,). substitutes: {model id: (N, T, 3)} baselines the Judge may name. Returns (q, known,
    counts): q with band_scale applied / substitute swapped in; rows the Judge calls "no reliable source" are
    known=False and take ``fallback`` (N, T, 3) (a no-information band) if given, else stay NaN.
    This is the only function that touches the adapter; swapping in the Judge = pass judge=SqliteVerdicts(db, run)
    or set_default_judge(...) in the adapter.
    """
    q_raw = np.asarray(q_raw, dtype=np.float64)
    try:
        from sentinel.judge.adapter import RawBatch, validated_many
    except ImportError:
        q, known, counts = _passthrough(model, site_ids, days, q_raw)
    else:
        t = q_raw.shape[1]
        if t != H:
            raise ValueError(f"adapter works on {H}-step days, got {t}")
        subs = {k: RawBatch(q=np.asarray(v, dtype=np.float64)) for k, v in (substitutes or {}).items()}
        b = validated_many(model, list(site_ids), days, raw=RawBatch(q=q_raw), substitutes=subs, judge=judge)
        q, known, counts = b.q, np.asarray(b.known, bool), dict(b.counts())
        counts["adapter"] = "sentinel.judge.adapter.validated_many"
    q = np.array(q, dtype=np.float64)
    if fallback is not None and (~known).any():
        q[~known] = np.asarray(fallback, dtype=np.float64)[~known]
    return q, known, counts


# ================================================================================================ 2. scenarios
def draw_levels(n_scen: int, region: np.ndarray, n_regions: Optional[int] = None,
                seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Quantile levels: PV (K, R) one per region and scenario (Latin hypercube over scenarios, independent across
    regions); load (K, N) iid per home. Clipped to [0.001, 0.999]."""
    rng = np.random.default_rng(seed)
    r = int(n_regions if n_regions is not None else (np.max(region) + 1 if len(region) else 0))
    u_pv = (np.stack([rng.permutation(n_scen) for _ in range(r)], axis=1) + rng.random((n_scen, r))) / n_scen
    u_load = rng.random((n_scen, len(region)))
    return np.clip(u_pv, LEVEL_CLIP, 1 - LEVEL_CLIP), np.clip(u_load, LEVEL_CLIP, 1 - LEVEL_CLIP)


def sample_homes(q: np.ndarray, z: np.ndarray, cap: Optional[np.ndarray] = None) -> np.ndarray:
    """q (n, T, 3) at levels 0.1/0.5/0.9, z (K, n) normal scores -> values (K, n, T), clipped to [0, cap].

    Same maths as ``conformal.quantile_at`` (piecewise linear in z, split-normal tails), written with two full-size
    passes: z is constant along T, so the branch on its sign is taken on (K, n) instead of (K, n, T)."""
    q = np.asarray(q, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    lo = (q[..., 1] - q[..., 0]) / Z90
    hi = (q[..., 2] - q[..., 1]) / Z90
    v = np.minimum(z, 0.0)[..., None] * lo
    v += np.maximum(z, 0.0)[..., None] * hi
    v += q[..., 1]
    np.maximum(v, 0.0, out=v)
    if cap is not None:
        np.minimum(v, np.asarray(cap, dtype=np.float64)[None, :, None], out=v)
    return v


# ================================================================================================ 3. battery behaviour
def self_consumption_behaviour(pv: np.ndarray, load: np.ndarray, battery: BatteryArrays,
                               soc0: Optional[np.ndarray] = None) -> np.ndarray:
    """Current fleet: every battery runs the self-consumption rule (charge from PV surplus, cover load), the rule the
    fleet history was generated under. One K1 kernel call over all scenarios (Rust when built, else numpy).
    pv, load (..., N, T) kW; soc0 (N,) or (..., N) fraction (None = soc_min). Returns import - export kW."""
    pv = np.asarray(pv, dtype=np.float64)
    load = np.asarray(load, dtype=np.float64)
    shape = np.broadcast_shapes(pv.shape, load.shape)
    n, t = shape[-2:]
    net = -np.array(np.broadcast_to(pv - load, shape))  # import - export without a battery
    has = np.flatnonzero(battery.capacity_kwh > 0)
    if has.size == 0:
        return net
    bat = battery.take(has)  # only homes with a battery go through the stateful loop
    s0 = (bat.e_lo_kwh / bat.capacity_kwh if soc0 is None
          else np.broadcast_to(np.asarray(soc0, dtype=np.float64), shape[:-1])[..., has])
    req = np.ascontiguousarray(-net[..., has, :].reshape((-1, has.size, t)))
    s0 = np.ascontiguousarray(np.broadcast_to(s0, shape[:-2] + (has.size,)).reshape(req.shape[:-1]))
    _, batt = battery_rollout(req, s0, *bat, dt_h=DT_H)
    net[..., has, :] += batt.reshape(shape[:-2] + (has.size, t))  # import - export = load - pv + batt
    return net


def no_battery_behaviour(pv: np.ndarray, load: np.ndarray, battery: BatteryArrays,
                         soc0: Optional[np.ndarray] = None) -> np.ndarray:
    """``forecast_only`` (spec I step 4): M1 + M2 without any battery model."""
    return np.asarray(load, dtype=np.float64) - np.asarray(pv, dtype=np.float64)


Behaviour = Callable[[np.ndarray, np.ndarray, BatteryArrays, Optional[np.ndarray]], np.ndarray]
_BEHAVIOUR: dict[str, Any] = {"fn": self_consumption_behaviour, "name": "self_consumption (K1 kernel)"}


def set_behaviour(fn: Optional[Behaviour], name: str = "") -> None:
    """Swap the battery model for every caller (I4 #174 on data-v2). None restores the self-consumption kernel."""
    _BEHAVIOUR.update(fn=fn or self_consumption_behaviour,
                      name=name or ("self_consumption (K1 kernel)" if fn is None else getattr(fn, "__name__", "?")))


def behaviour(pv: np.ndarray, load: np.ndarray, battery: BatteryArrays,
              soc0: Optional[np.ndarray] = None) -> np.ndarray:
    """THE battery model of the portfolio: per-home import - export kW for scenarios pv/load (..., N, T)."""
    return _BEHAVIOUR["fn"](pv, load, battery, soc0)


def behaviour_name() -> str:
    return str(_BEHAVIOUR["name"])


def warm_soc0(pv50: np.ndarray, load50: np.ndarray, battery: BatteryArrays) -> np.ndarray:
    """Midnight SoC (fraction) the supplier cannot observe: roll the P50 day once from soc_min with the
    self-consumption rule and take the end-of-day SoC (periodic steady state: tonight ~ like last night)."""
    cap = battery.capacity_kwh
    if not np.any(cap > 0):
        return np.zeros(len(cap))
    req = np.ascontiguousarray(np.asarray(pv50, dtype=np.float64) - np.asarray(load50, dtype=np.float64))
    lo = np.where(cap > 0, battery.e_lo_kwh / np.where(cap > 0, cap, 1.0), 0.0)
    soc, _ = battery_rollout(req, lo, *battery, dt_h=DT_H)
    return np.asarray(soc[..., -1], dtype=np.float64)


# ================================================================================================ 4. portfolio total
def portfolio_scenarios(pv_q: np.ndarray, load_q: np.ndarray, fleet: Fleet, *, n_scen: int = N_SCENARIOS,
                        seed: int = 0, soc0: Optional[np.ndarray] = None, battery: bool = True,
                        chunk: int = 256, levels: Optional[tuple[np.ndarray, np.ndarray]] = None) -> np.ndarray:
    """K coherent portfolio days: total (K, T) MWh per step, import - export summed over homes.

    pv_q, load_q (N, T, 3) kW (validated bands). soc0 (N,) fraction (None = ``warm_soc0`` from the P50s).
    battery=False gives ``forecast_only``. Homes are processed in chunks (memory K x chunk x T), the scenario levels
    are drawn once for the whole fleet, so chunking does not change the result. levels = (u_pv, u_load) to reuse.
    """
    pv_q = np.asarray(pv_q, dtype=np.float64)
    load_q = np.asarray(load_q, dtype=np.float64)
    n, t = pv_q.shape[:2]
    u_pv, u_load = levels if levels is not None else draw_levels(n_scen, fleet.region, len(fleet.regions), seed)
    z_pv, z_load = ndtri(u_pv), ndtri(u_load)
    if battery and soc0 is None:
        soc0 = warm_soc0(pv_q[..., 1], load_q[..., 1], fleet.battery)
    total = np.zeros((u_pv.shape[0], t))
    for a in range(0, n, chunk):
        sl = slice(a, min(n, a + chunk))
        pv = sample_homes(pv_q[sl], z_pv[:, fleet.region[sl]], fleet.pv_cap[sl])
        ld = sample_homes(load_q[sl], z_load[:, sl])
        net = (behaviour(pv, ld, fleet.battery.take(sl), None if soc0 is None else soc0[sl]) if battery
               else no_battery_behaviour(pv, ld, fleet.battery.take(sl)))
        total += net.sum(axis=1)
    return total * KW_TO_MWH


def scenario_bands(total: np.ndarray, levels: Sequence[float] = LEVELS) -> np.ndarray:
    """(K, T) scenarios -> (T, len(levels)) empirical quantiles (sorted, so never crossing)."""
    return np.sort(np.quantile(np.asarray(total, dtype=np.float64), list(levels), axis=0).T, axis=-1)


# ================================================================================================ 5. conformal on total
@dataclass
class TotalConformal:
    """Split-conformal scale of the portfolio band, one per horizon bucket.

    Score of a validation step: e = (y - p50) / (p90 - p50) above P50, (p50 - y) / (p50 - p10) below. With
    s = the finite-sample 80 % quantile of e, the band p50 +- s * (spread) covers 80 % of validation steps. s > 1
    widens, s < 1 narrows (independent load draws per home are a lower bound on the spread, shared PV levels per
    region an upper bound; the data decides). The same stretch is applied to scenarios (``stretch``), so the purchase
    quantile is calibrated consistently.
    """

    scale: np.ndarray = field(default_factory=lambda: np.ones(len(BUCKETS)))
    buckets: tuple = BUCKETS
    target: float = COVER_TARGET
    n: int = 0
    raw_coverage: float = float("nan")

    def per_step(self, t: int = H) -> np.ndarray:
        s = np.ones(t)
        for (a, b), v in zip(self.buckets, self.scale):
            s[a:min(b, t)] = v
        return s

    @staticmethod
    def scores(bands: np.ndarray, actual: np.ndarray) -> np.ndarray:
        bands = np.asarray(bands, dtype=np.float64)
        y = np.asarray(actual, dtype=np.float64)
        lo, mid, hi = bands[..., 0], bands[..., 1], bands[..., -1]
        eps = 1e-9
        up = (y - mid) / np.maximum(hi - mid, eps)
        dn = (mid - y) / np.maximum(mid - lo, eps)
        return np.where(y >= mid, up, dn)

    @classmethod
    def fit(cls, bands: np.ndarray, actual: np.ndarray, target: float = COVER_TARGET,
            buckets: tuple = BUCKETS) -> "TotalConformal":
        """bands (D, T, 3) raw P10/P50/P90 of the total on validation days, actual (D, T). NaN actuals are skipped."""
        e = cls.scores(bands, actual)
        ok = np.isfinite(e)
        scale = []
        for a, b in buckets:
            v = e[:, a:b][ok[:, a:b]]
            if v.size < 10:
                scale.append(1.0)
                continue
            lvl = min(1.0, target * (1 + 1 / v.size))
            scale.append(float(max(np.quantile(v, lvl), 1e-3)))
        raw = float(np.mean(e[ok] <= 1.0)) if ok.any() else float("nan")
        return cls(np.array(scale), buckets, target, int(ok.sum()), raw)

    def stretch(self, values: np.ndarray, p50: np.ndarray) -> np.ndarray:
        """p50 + s_t (values - p50): scenarios (K, T) or quantiles (..., T) on the calibrated scale."""
        p50 = np.asarray(p50, dtype=np.float64)
        return p50 + self.per_step(p50.shape[-1]) * (np.asarray(values, dtype=np.float64) - p50)

    def apply(self, bands: np.ndarray) -> np.ndarray:
        """(..., T, 3) raw band -> calibrated band (P50 unchanged)."""
        bands = np.asarray(bands, dtype=np.float64)
        mid = bands[..., 1:2]
        s = self.per_step(bands.shape[-2])[:, None]
        return mid + s * (bands - mid)

    def to_json(self) -> dict:
        return {"scale": [float(x) for x in self.scale], "buckets": [list(b) for b in self.buckets],
                "target": self.target, "n": self.n, "raw_coverage": self.raw_coverage}


def coverage(bands: np.ndarray, actual: np.ndarray) -> float:
    """Share of finite actuals inside [P10, P90]."""
    bands = np.asarray(bands, dtype=np.float64)
    y = np.asarray(actual, dtype=np.float64)
    ok = np.isfinite(y)
    inside = (y >= bands[..., 0] - 1e-12) & (y <= bands[..., -1] + 1e-12)
    return float(inside[ok].mean()) if ok.any() else float("nan")


def nmae(pred: np.ndarray, actual: np.ndarray, mask: Optional[np.ndarray] = None) -> float:
    """sum |pred - y| / sum |y| over finite (and masked) steps."""
    p, y = np.broadcast_arrays(np.asarray(pred, dtype=np.float64), np.asarray(actual, dtype=np.float64))
    ok = np.isfinite(y) & np.isfinite(p)
    if mask is not None:
        ok &= np.broadcast_to(mask, ok.shape)
    den = np.abs(y[ok]).sum()
    return float(np.abs(p[ok] - y[ok]).sum() / den) if den > 0 else float("nan")


# ================================================================================================ 6. purchase
def purchase_quantile(a: Optional[float] = None, b: Optional[float] = None) -> float:
    """q* = a / (a + b): the newsvendor quantile of the imbalance penalties (a = short, b = long).

    Frozen a, b from X1 (#158): ``sentinel.eval.imbalance.q_star`` when on main, else models/imbalance_ab.json,
    else #158's frozen values (stub, a = 2b -> q* = 2/3)."""
    if a is not None and b is not None:
        return float(a) / (float(a) + float(b))
    try:
        from sentinel.eval.imbalance import q_star
        return float(q_star(a, b))
    except ImportError:
        pass
    a0, b0 = AB_STUB
    if AB_PATH.exists():
        d = json.loads(AB_PATH.read_text())
        a0, b0 = float(d["a"]), float(d["b"])
    a, b = (a0 if a is None else a), (b0 if b is None else b)
    return float(a) / (float(a) + float(b))


def _quantile_cal(total: np.ndarray, q: float, conformal: Optional[TotalConformal]) -> np.ndarray:
    total = np.asarray(total, dtype=np.float64)
    if conformal is not None:
        total = conformal.stretch(total, np.quantile(total, 0.5, axis=0))
    return np.quantile(total, q, axis=0)


def bought_da(total: np.ndarray, q_star: float, conformal: Optional[TotalConformal] = None) -> np.ndarray:
    """Day-ahead purchase (T,) MWh = quantile q* of the (calibrated) portfolio scenarios (K, T)."""
    return _quantile_cal(total, q_star, conformal)


def correction_mask(ts_utc: Any, start_hour: int = 16, end_hour: int = 24, tz: str = "Europe/Bucharest") -> np.ndarray:
    """Steps whose LOCAL start time is in [start_hour, end_hour): the 15:00 re-forecast window (16:00-24:00)."""
    import pandas as pd

    ts = pd.DatetimeIndex(ts_utc)
    ts = ts.tz_localize("UTC") if ts.tz is None else ts
    loc = ts.tz_convert(tz)
    h = loc.hour + loc.minute / 60.0
    return np.asarray((h >= start_hour) & (h < end_hour))


def traded_id(bought: np.ndarray, total_new: np.ndarray, q_star: float, mask: np.ndarray,
              conformal: Optional[TotalConformal] = None) -> np.ndarray:
    """Intraday (PI) trade (T,) MWh: + buy more, - sell back. q*_new - bought_da on the masked steps, 0 elsewhere."""
    q_new = _quantile_cal(total_new, q_star, conformal)
    return np.where(np.asarray(mask, bool), q_new - np.asarray(bought, dtype=np.float64), 0.0)


# ================================================================================================ one day, end to end
@dataclass
class DayForecast:
    total: np.ndarray  # (K, T) raw scenarios, MWh/step
    bands_raw: np.ndarray  # (T, 3)
    bands: np.ndarray  # (T, 3) calibrated
    bought: np.ndarray  # (T,) quantile q* of the calibrated scenarios
    q_star: float


def forecast_day(pv_q: np.ndarray, load_q: np.ndarray, fleet: Fleet, *, conformal: Optional[TotalConformal] = None,
                 q_star: Optional[float] = None, n_scen: int = N_SCENARIOS, seed: int = 0,
                 soc0: Optional[np.ndarray] = None, battery: bool = True) -> DayForecast:
    """Steps 2-6 for one day: validated per-home bands (N, T, 3) -> portfolio scenarios, band, purchase."""
    qs = purchase_quantile() if q_star is None else float(q_star)
    total = portfolio_scenarios(pv_q, load_q, fleet, n_scen=n_scen, seed=seed, soc0=soc0, battery=battery)
    raw = scenario_bands(total)
    cal = raw if conformal is None else conformal.apply(raw)
    return DayForecast(total, raw, cal, bought_da(total, qs, conformal), qs)


# ================================================================================================ data layer (fleet)
def fleet_from_source(src: Any, region_key: str = "county") -> Fleet:
    """Fleet from a #10 window source (``sentinel.model.data.ArraySource``): sites.json rows."""
    return Fleet.from_sites(src.sites, region_key)


def _members(models: Any) -> list:
    from sentinel.model.predictor import Member

    return [m if isinstance(m, Member) else Member(Path(m)) for m in models]


def find_models(model: str, model_dir: Path | str) -> list[Path]:
    from sentinel.model.predictor import find_members

    return find_members(model, model_dir)


def raw_bands(src: Any, sites: np.ndarray, day_idx: np.ndarray, *, kind: str, models: Sequence[Any],
              mode: str = "forecast", lag_days: int = 1, batch: int = 8192) -> np.ndarray:
    """Raw per-home P10/P50/P90 kW (n_days, N, T, 3) from the ONNX members (ensemble mean, their own calibration
    offsets, exactly as ``sentinel.model.predictor.ForecastPredictor.quantiles`` but batched over homes x days).

    kind 'pv' | 'load'; mode 'forecast' (day-ahead weather) | 'nowcast' (ERA5 = the 15:00 "updated weather" proxy).
    lag_days = days of meter data the supplier does NOT have yet: smart-meter data arrives D+1, so the day-ahead
    forecast (issued on D-1) sees history up to the end of D-2 (lag 1); the 15:00 re-forecast on D sees D-1 (lag 0).
    The swap point for data-v2: return the same shape from any source.
    """
    from sentinel.kernels import window_gather
    from sentinel.model import features as F

    members = _members(models)
    sites = np.asarray(sites, np.int64)
    day_idx = np.asarray(day_idx, np.int64)
    s_, d_ = np.meshgrid(sites, day_idx, indexing="xy")  # (n_days, N)
    s_flat, starts = s_.ravel(), d_.ravel() * H
    out = np.empty((len(s_flat), H, 3))
    hist_key, arr = ("hist_pv", "pv_ac") if kind == "pv" else ("hist_load", "load")
    for a in range(0, len(s_flat), batch):
        ss, st = s_flat[a:a + batch], starts[a:a + batch]
        raw = F.gather(src, ss, st, kind, mode)
        if lag_days:
            raw[hist_key] = window_gather(src.array(arr), ss, st, -(F.HIST_DAYS + lag_days) * H, F.HIST_DAYS * H,
                                          np.float32)
        inp = F.build(kind, raw)
        feeds = {k: inp[k] for k in ("seq", "ref", "summ", "stat")}
        feeds["mask"] = (~inp["night"]).astype(np.float32)
        scale = np.asarray(inp["scale"], np.float64)
        acc = np.zeros((len(ss), H, 3))
        for m in members:
            q = np.asarray(m.run(feeds)[0], np.float64) * scale[:, None, None]
            off = m.calib.get("quantile_offsets")
            if off:
                for (b0, b1), lo, hi in zip(off["buckets"], off["q10"], off["q90"]):
                    q[:, b0:b1, 0] -= lo * scale[:, None]
                    q[:, b0:b1, 2] += hi * scale[:, None]
            q[..., 0] = np.minimum(np.maximum(q[..., 0], 0.0), q[..., 1])
            q[..., 2] = np.maximum(q[..., 2], q[..., 1])
            q[inp["night"]] = 0.0
            acc += q
        out[a:a + len(ss)] = np.sort(acc / len(members), axis=-1)
    return out.reshape(len(day_idx), len(sites), H, 3)


def fallback_bands(src: Any, sites: np.ndarray, day_idx: np.ndarray, kind: str, lag_days: int = 1) -> np.ndarray:
    """No-information band for homes the Judge calls "no reliable source" (n_days, N, T, 3): PV = pvlib on recorded
    metadata and forecast weather x (0.5, 1, 1.2); load = 7-day mean profile x (0.6, 1, 1.6)."""
    from sentinel.kernels import window_gather

    sites = np.asarray(sites, np.int64)
    s_, d_ = np.meshgrid(sites, np.asarray(day_idx, np.int64), indexing="xy")
    ss, st = s_.ravel(), d_.ravel() * H
    if kind == "pv":
        mid = window_gather(src.array("pvlib_rec_fc"), ss, st, 0, H, np.float64)
        q = np.stack([0.5 * mid, mid, 1.2 * mid], axis=-1)
    else:
        h = window_gather(src.array("load"), ss, st, -(7 + lag_days) * H, 7 * H, np.float64)
        mid = h.reshape(len(ss), 7, H).mean(axis=1)
        q = np.stack([0.6 * mid, mid, 1.6 * mid], axis=-1)
    return q.reshape(len(day_idx), len(sites), H, 3)


def actual_net(src: Any, sites: np.ndarray, day_idx: np.ndarray) -> np.ndarray:
    """Meter truth (n_days, N, T) kW import - export = load + batt - pv (fleet history, synthetic RO)."""
    from sentinel.kernels import window_gather

    sites = np.asarray(sites, np.int64)
    s_, d_ = np.meshgrid(sites, np.asarray(day_idx, np.int64), indexing="xy")
    ss, st = s_.ravel(), d_.ravel() * H
    g = (lambda name: window_gather(src.array(name), ss, st, 0, H, np.float64))
    net = g("load") + g("batt") - g("pv_ac")
    return net.reshape(len(day_idx), len(sites), H)


# ================================================================================================ lab run
def _emit(name: str, value: float) -> None:
    from sentinel.infra.metrics import emit

    emit(name, value)


def _sample(days: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    days = np.asarray(days)
    return days if n <= 0 or n >= len(days) else np.sort(rng.choice(days, n, replace=False))


def run_split(src: Any, fleet: Fleet, sites: np.ndarray, days: np.ndarray, *, models: Mapping[str, Sequence[Any]],
              n_scen: int, seed: int, chunk_days: int, timings: dict, log=print) -> dict:
    """Day-ahead and 15:00 scenarios for a set of days -> raw bands, scenario quantiles, actuals."""
    import pandas as pd

    q_lv = purchase_quantile()
    t_steps = H
    t0_ = pd.Timestamp(src.t_start)
    t_origin = (t0_.tz_convert("UTC").tz_localize(None) if t0_.tzinfo else t0_).to_datetime64().astype("datetime64[ns]")
    out ={k: [] for k in ("day", "ts", "da_bands", "da_q", "fo_bands", "id_bands", "id_q", "actual", "mask")}
    counts: dict[str, int] = {}

    def tick(key: str, t0: float) -> None:
        timings[key] = timings.get(key, 0.0) + time.perf_counter() - t0

    for c in range(0, len(days), chunk_days):
        dd = days[c:c + chunk_days]
        t0 = time.perf_counter()
        pv_da = raw_bands(src, sites, dd, kind="pv", models=models["pv_da"], mode="forecast", lag_days=1)
        ld_da = raw_bands(src, sites, dd, kind="load", models=models["load"], mode="forecast", lag_days=1)
        pv_id = raw_bands(src, sites, dd, kind="pv", models=models["pv_id"], mode="nowcast", lag_days=0)
        ld_id = raw_bands(src, sites, dd, kind="load", models=models["load"], mode="nowcast", lag_days=0)
        fb_pv = fallback_bands(src, sites, dd, "pv", 1)
        fb_ld = fallback_bands(src, sites, dd, "load", 1)
        act = actual_net(src, sites, dd).sum(axis=1) * KW_TO_MWH
        tick("t_inference_s", t0)
        for j, d in enumerate(dd):
            ts = t_origin + (int(d) * H + np.arange(t_steps)) * np.timedelta64(15, "m")  # UTC, naive
            day = str(np.datetime_as_string(ts[0], unit="D"))
            t0 = time.perf_counter()
            bands = {}
            for name, model, q, fb in (("pv_da", "pvnet", pv_da[j], fb_pv[j]), ("ld_da", "loadnet", ld_da[j], fb_ld[j]),
                                       ("pv_id", "pvnet", pv_id[j], fb_pv[j]), ("ld_id", "loadnet", ld_id[j], fb_ld[j])):
                bands[name], _, cnt = validated_bands(model, fleet.site_ids[sites], day, q, fallback=fb)
                for k, v in cnt.items():
                    if isinstance(v, (int, np.integer)):
                        counts[k] = counts.get(k, 0) + int(v)
            tick("t_adapter_s", t0)
            sub = Fleet(fleet.site_ids[sites], fleet.region[sites], fleet.regions, fleet.battery.take(sites),
                        fleet.pv_cap[sites])
            lv = draw_levels(n_scen, sub.region, len(sub.regions), seed + int(d))
            t0 = time.perf_counter()
            tot_da = portfolio_scenarios(bands["pv_da"], bands["ld_da"], sub, levels=lv)
            tot_id = portfolio_scenarios(bands["pv_id"], bands["ld_id"], sub, levels=lv)
            tick("t_scenarios_s", t0)
            t0 = time.perf_counter()
            tot_fo = portfolio_scenarios(bands["pv_da"], bands["ld_da"], sub, levels=lv, battery=False)
            tick("t_forecast_only_s", t0)
            out["day"].append(day)
            out["ts"].append(ts)
            out["da_bands"].append(scenario_bands(tot_da))
            out["da_q"].append(_qstar_raw(tot_da, q_lv))
            out["fo_bands"].append(scenario_bands(tot_fo))
            out["id_bands"].append(scenario_bands(tot_id))
            out["id_q"].append(_qstar_raw(tot_id, q_lv))
            out["actual"].append(act[j])
            out["mask"].append(correction_mask(ts))
        log(f"[portfolio] {min(c + chunk_days, len(days))}/{len(days)} days  " +
            "  ".join(f"{k}={v:.1f}" for k, v in timings.items()))
    res = {k: (np.stack(v) if k not in ("day",) else v) for k, v in out.items()}
    res["counts"] = counts
    return res


def _qstar_raw(total: np.ndarray, q: float) -> np.ndarray:
    """Raw quantile q* and P50 of the scenarios, (T, 2): kept so the conformal stretch can be applied afterwards
    (stretch is linear around P50, so stretching the quantile equals the quantile of the stretched scenarios)."""
    return np.stack([np.quantile(total, q, axis=0), np.quantile(total, 0.5, axis=0)], axis=-1)


def _stretch_q(conf: TotalConformal, qq: np.ndarray) -> np.ndarray:
    return conf.stretch(qq[..., 0], qq[..., 1])


def local_hours(ts: np.ndarray, tz: str = "Europe/Bucharest") -> np.ndarray:
    import pandas as pd

    idx = pd.DatetimeIndex(np.asarray(ts).ravel()).tz_localize("UTC").tz_convert(tz)
    return (idx.hour + idx.minute / 60.0).to_numpy().reshape(np.shape(ts))


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m sentinel.eval.portfolio", description=__doc__.split("\n")[0])
    ap.add_argument("--data", default="lab", help="lab | auto | synth | <windows dir> (sentinel.model.data)")
    ap.add_argument("--models", default=str(ROOT / "models"))
    ap.add_argument("--val-days", type=int, default=120, help="sampled 2024 days for the conformal fit (0 = all)")
    ap.add_argument("--test-days", type=int, default=180, help="sampled 2025 days for the metrics (0 = all)")
    ap.add_argument("--k", type=int, default=N_SCENARIOS)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--region", default="county", help="sites column that defines a solar region")
    ap.add_argument("--chunk-days", type=int, default=8)
    ap.add_argument("--max-sites", type=int, default=0, help="smoke runs only")
    args = ap.parse_args(argv)

    from sentinel import kernels
    from sentinel.model.data import make_splits, open_source

    t_all = time.perf_counter()
    src = open_source(args.data)
    fleet = fleet_from_source(src, args.region)
    sites = np.arange(len(fleet)) if args.max_sites <= 0 else np.arange(min(args.max_sites, len(fleet)))
    sp = make_splits(src)
    rng = np.random.default_rng(args.seed)
    lag_ok = lambda d: d[d >= 15 + 1]  # noqa: E731  14 history days + 1 day of meter lag
    val_days = _sample(lag_ok(np.concatenate([sp.days["val"], sp.days["cal"]])), args.val_days, rng)
    test_days = _sample(lag_ok(sp.days["test"]), args.test_days, rng)
    md = Path(args.models)
    models = {"pv_da": find_models("pvnet", md / "pvnet_fc"), "pv_id": find_models("pvnet", md),
              "load": find_models("loadnet", md)}
    print(f"[portfolio] {len(sites)} homes, {len(fleet.regions)} regions ({args.region}), K={args.k}, "
          f"val {len(val_days)} days, test {len(test_days)} days, kernels={kernels.backend()}, "
          f"behaviour={behaviour_name()}, q*={purchase_quantile():.4f}", flush=True)
    print(f"[portfolio] models: pv_da={[p.name for p in models['pv_da']]} pv_id={[p.name for p in models['pv_id']]} "
          f"load={[p.name for p in models['load']]}", flush=True)
    timings: dict[str, float] = {}
    val = run_split(src, fleet, sites, val_days, models=models, n_scen=args.k, seed=args.seed,
                    chunk_days=args.chunk_days, timings=timings)
    test = run_split(src, fleet, sites, test_days, models=models, n_scen=args.k, seed=args.seed + 7919,
                     chunk_days=args.chunk_days, timings=timings)

    conf_da = TotalConformal.fit(val["da_bands"], val["actual"])
    conf_id = TotalConformal.fit(np.where(val["mask"][..., None], val["id_bands"], np.nan),
                                 np.where(val["mask"], val["actual"], np.nan))
    conf_fo = TotalConformal.fit(val["fo_bands"], val["actual"])
    q_lv = purchase_quantile()
    m: dict[str, float] = {}
    for name, s in (("val", val), ("test", test)):
        y = s["actual"]
        da = conf_da.apply(s["da_bands"])
        idb = conf_id.apply(s["id_bands"])
        mask = s["mask"]
        hrs = local_hours(s["ts"])
        eve = (hrs >= 17) & (hrs < 21)
        pre = "portfolio" if name == "test" else "portfolio_val"
        m[f"{pre}_nmae"] = nmae(da[..., 1], y)
        m[f"{pre}_coverage_p10_p90"] = coverage(da, y)
        m[f"{pre}_coverage_raw"] = coverage(s["da_bands"], y)
        m[f"{pre}_nmae_17_21"] = nmae(da[..., 1], y, eve)
        m[f"{pre}_forecast_only_nmae"] = nmae(s["fo_bands"][..., 1], y)
        m[f"{pre}_forecast_only_nmae_17_21"] = nmae(s["fo_bands"][..., 1], y, eve)
        m[f"{pre}_forecast_only_coverage_p10_p90"] = coverage(conf_fo.apply(s["fo_bands"]), y)
        m[f"{pre}_band_width_mwh"] = float(np.mean(da[..., 2] - da[..., 0]))
        m[f"{pre}_id_nmae_16_24"] = nmae(idb[..., 1], y, mask)
        m[f"{pre}_id_coverage_16_24"] = coverage(idb[mask], y[mask])
        bought = _stretch_q(conf_da, s["da_q"])
        traded = np.where(mask, _stretch_q(conf_id, s["id_q"]) - bought, 0.0)
        m[f"{pre}_abs_dev_da_mwh_per_day"] = float(np.abs(y - bought).sum() / len(y))
        m[f"{pre}_abs_dev_after_id_mwh_per_day"] = float(np.abs(y - bought - traded).sum() / len(y))
        m[f"{pre}_abs_dev_da_16_24_mwh_per_day"] = float(np.abs((y - bought)[mask]).sum() / len(y))
        m[f"{pre}_abs_dev_after_id_16_24_mwh_per_day"] = float(np.abs((y - bought - traded)[mask]).sum() / len(y))
        m[f"{pre}_mean_actual_mwh_per_day"] = float(y.sum() / len(y))
        s["bought"], s["traded"], s["da_cal"], s["id_cal"] = bought, traded, da, idb
    for k, v in m.items():
        _emit(k, v)
    _emit("portfolio_q_star", q_lv)
    _emit("portfolio_k", args.k)
    _emit("portfolio_homes", len(sites))
    for i, v in enumerate(conf_da.scale):
        _emit(f"portfolio_conformal_scale_b{i}", v)
    _emit("rust_kernels", float(kernels.backend() == "rust"))
    timings["t_total_s"] = time.perf_counter() - t_all
    for k, v in timings.items():
        _emit(k, v)
    _emit("share_scenarios_of_total", timings.get("t_scenarios_s", 0.0) / max(timings["t_total_s"], 1e-9))
    print(f"[portfolio] labels: metrics = {LABEL} (fleet {src.index.get('source_key', {}).get('fleet', '?')}, "
          f"test 2025 = {len(test_days)} days); a, b = proxy (#158); intraday weather = ERA5 proxy", flush=True)

    out = Path(os.environ.get("LAB_OUTPUT_DIR", "."))
    out.mkdir(parents=True, exist_ok=True)
    _write_outputs(out, test, run_id=os.environ.get("LAB_RUN_ID", "local"))
    report = {"card": 175, "run_id": os.environ.get("LAB_RUN_ID", "local"), "label": LABEL, "metrics": m,
              "conformal_da": conf_da.to_json(), "conformal_id": conf_id.to_json(),
              "conformal_forecast_only": conf_fo.to_json(), "q_star": q_lv, "k": args.k,
              "homes": int(len(sites)), "regions": fleet.regions, "region_key": args.region,
              "val_days": val["day"], "test_days": test["day"], "adapter_counts": {"val": val["counts"],
                                                                                   "test": test["counts"]},
              "behaviour": behaviour_name(), "kernels": kernels.backend(), "timings": timings,
              "models": {k: [str(p) for p in v] for k, v in models.items()},
              "labels": {"portfolio": LABEL, "imbalance_ab": "proxy (#158, AFEER public range)",
                         "intraday_weather": "proxy: ERA5 analysis as the 15:00 weather update"}}
    (out / "portfolio_report.json").write_text(json.dumps(report, indent=1, default=float))
    return 0


def _write_outputs(out: Path, s: dict, run_id: str) -> None:
    """portfolio_forecast.parquet + positions.parquet (spec I step 6 'Ce iese'), test days, policy 'sentinel'."""
    import pandas as pd

    ts = pd.DatetimeIndex(s["ts"].ravel()).tz_localize("UTC")
    d, t = s["ts"].shape
    issue_da = np.repeat([pd.Timestamp(x).tz_localize("UTC") - pd.Timedelta(hours=13) for x in s["ts"][:, 0]], t)
    rows = []
    for kind, bands, issue in (("da", s["da_cal"], issue_da), ("id", s["id_cal"], None)):
        f = pd.DataFrame({"issue_time": issue if issue is not None else ts.floor("D") + pd.Timedelta(hours=13),
                          "ts": ts, "policy": "sentinel", "horizon": kind, "p10": bands[..., 0].ravel(),
                          "p50": bands[..., 1].ravel(), "p90": bands[..., 2].ravel(), "run_id": run_id})
        if kind == "id":
            f = f[s["mask"].ravel()]
        rows.append(f)
    pd.concat(rows).to_parquet(out / "portfolio_forecast.parquet", index=False)
    pd.DataFrame({"ts": ts, "policy": "sentinel", "bought_da": s["bought"].ravel(), "traded_id": s["traded"].ravel(),
                  "actual": s["actual"].ravel(), "run_id": run_id}).to_parquet(out / "positions.parquet", index=False)


if __name__ == "__main__":
    raise SystemExit(main())
