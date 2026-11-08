"""Portfolio day-ahead forecast and purchase for the supplier, from per-home PV/load quantile bands.

Quantiles do not add up, so the total comes from K coherent scenarios: one PV quantile level per region and scenario
(weather errors are shared by neighbours), an independent load level per home, each home through the battery
``behaviour``, summed. A split-conformal scale per 6-h bucket (``TotalConformal``) calibrates the scenario spread to
80 % P10-P90 coverage; the purchase is the quantile q* = a/(a+b) of the calibrated scenarios, and the 15:00 re-forecast
(16:00-24:00 local, ``correction_mask``) gives the intraday trade.

Units: per-home kW, portfolio MWh per 15-min step. Sign = consumption (total > 0 = net import), as in
``wattsentry.eval.imbalance``. A day = 96 steps from UTC midnight; outputs carry UTC ``ts``.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, NamedTuple, Optional, Sequence

import numpy as np
from scipy.special import ndtri

from wattsentry.eval.uncertainty.conformal import Z90
from wattsentry.kernels import battery_rollout

H = 96  # 15-min steps per day
DT_H = 0.25
KW_TO_MWH = DT_H / 1000.0  # kW over one step -> MWh
N_SCENARIOS = 200
LEVELS = (0.1, 0.5, 0.9)
COVER_TARGET = 0.80
BUCKETS = ((0, 24), (24, 48), (48, 72), (72, 96))  # 6-h horizon buckets of the conformal scale
LEVEL_CLIP = 1e-3


# ================================================================================================ static inputs
class BatteryArrays(NamedTuple):
    """Per-home battery arrays (N,), in ``battery_rollout``'s argument order. Energies kWh, powers kW, eta one-way.
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
        """From sites-table rows (sites.parquet / windows sites.json): battery_kwh, battery_kw, eta_*, soc_*."""
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
def validated_bands(model: str, site_ids: Sequence[Any], days: Any, q_raw: np.ndarray, *,
                    substitutes: Optional[Mapping[str, np.ndarray]] = None, fallback: Optional[np.ndarray] = None,
                    judge: Any = None) -> tuple[np.ndarray, np.ndarray, dict]:
    """Per-home bands as the Judge allows them to be used (``wattsentry.judge.adapter``).

    q_raw: (N, T, 3) raw q10/q50/q90 of ``model`` ('pvnet' | 'loadnet'), rows aligned with site_ids; days: one day
    'YYYY-MM-DD' or (N,). substitutes: {model id: (N, T, 3)} baselines the Judge may name. Returns (q, known,
    counts): q with band_scale applied / substitute swapped in; rows the Judge calls "no reliable source" are
    known=False and take ``fallback`` (N, T, 3) (a no-information band) if given, else stay NaN.
    This is the only function that touches the adapter; swapping in the Judge = pass judge=SqliteVerdicts(db, run)
    or set_default_judge(...) in the adapter.
    """
    from wattsentry.judge.adapter import RawBatch, validated_many

    q_raw = np.asarray(q_raw, dtype=np.float64)
    t = q_raw.shape[1]
    if t != H:
        raise ValueError(f"adapter works on {H}-step days, got {t}")
    subs = {k: RawBatch(q=np.asarray(v, dtype=np.float64)) for k, v in (substitutes or {}).items()}
    b = validated_many(model, list(site_ids), days, raw=RawBatch(q=q_raw), substitutes=subs, judge=judge)
    q, known, counts = b.q, np.asarray(b.known, bool), dict(b.counts())
    counts["adapter"] = "wattsentry.judge.adapter.validated_many"
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
    fleet history was generated under. One ``battery_rollout`` call over all scenarios (Rust when built, else numpy).
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
    """``forecast_only``: PV and load forecasts without any battery model."""
    return np.asarray(load, dtype=np.float64) - np.asarray(pv, dtype=np.float64)


BEHAVIOUR_NAME = "self_consumption (K1 kernel)"


def behaviour(pv: np.ndarray, load: np.ndarray, battery: BatteryArrays,
              soc0: Optional[np.ndarray] = None) -> np.ndarray:
    """THE battery model of the portfolio: per-home import - export kW for scenarios pv/load (..., N, T)."""
    return self_consumption_behaviour(pv, load, battery, soc0)


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
    """q* = a / (a + b): the newsvendor quantile of the imbalance penalties (a = short, b = long); frozen a, b from
    models/imbalance_ab.json for the ones not given."""
    if a is not None and b is not None:
        return float(a) / (float(a) + float(b))
    from wattsentry.eval.imbalance import q_star
    return float(q_star(a, b))


def bought_da(total: np.ndarray, q_star: float, conformal: Optional[TotalConformal] = None) -> np.ndarray:
    """Day-ahead purchase (T,) MWh = quantile q* of the (calibrated) portfolio scenarios (K, T)."""
    total = np.asarray(total, dtype=np.float64)
    if conformal is not None:
        total = conformal.stretch(total, np.quantile(total, 0.5, axis=0))
    return np.quantile(total, q_star, axis=0)


def correction_mask(ts_utc: Any, start_hour: int = 16, end_hour: int = 24, tz: str = "Europe/Bucharest") -> np.ndarray:
    """Steps whose LOCAL start time is in [start_hour, end_hour): the 15:00 re-forecast window (16:00-24:00)."""
    import pandas as pd

    ts = pd.DatetimeIndex(ts_utc)
    ts = ts.tz_localize("UTC") if ts.tz is None else ts
    loc = ts.tz_convert(tz)
    h = loc.hour + loc.minute / 60.0
    return np.asarray((h >= start_hour) & (h < end_hour))


# ================================================================================================ data layer (fleet)
def fleet_from_source(src: Any, region_key: str = "county") -> Fleet:
    """Fleet from a window source (``wattsentry.model.data.ArraySource``): sites.json rows."""
    return Fleet.from_sites(src.sites, region_key)


def _members(models: Any) -> list:
    from wattsentry.model.predictor import Member

    return [m if isinstance(m, Member) else Member(Path(m)) for m in models]


def find_models(model: str, model_dir: Path | str) -> list[Path]:
    from wattsentry.model.predictor import find_members

    return find_members(model, model_dir)


def raw_bands(src: Any, sites: np.ndarray, day_idx: np.ndarray, *, kind: str, models: Sequence[Any],
              mode: str = "forecast", lag_days: int = 1, batch: int = 8192) -> np.ndarray:
    """Raw per-home P10/P50/P90 kW (n_days, N, T, 3) from the ONNX members (ensemble mean, their own calibration
    offsets, exactly as ``wattsentry.model.predictor.ForecastPredictor.quantiles`` but batched over homes x days).

    kind 'pv' | 'load'; mode 'forecast' (day-ahead weather) | 'nowcast' (ERA5 = the 15:00 "updated weather" proxy).
    lag_days = days of meter data the supplier does NOT have yet: smart-meter data arrives D+1, so the day-ahead
    forecast (issued on D-1) sees history up to the end of D-2 (lag 1); the 15:00 re-forecast on D sees D-1 (lag 0).
    The swap point for data-v2: return the same shape from any source.
    """
    from wattsentry.kernels import window_gather
    from wattsentry.model import features as F

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
    from wattsentry.kernels import window_gather

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
    from wattsentry.kernels import window_gather

    sites = np.asarray(sites, np.int64)
    s_, d_ = np.meshgrid(sites, np.asarray(day_idx, np.int64), indexing="xy")
    ss, st = s_.ravel(), d_.ravel() * H
    g = (lambda name: window_gather(src.array(name), ss, st, 0, H, np.float64))
    net = g("load") + g("batt") - g("pv_ac")
    return net.reshape(len(day_idx), len(sites), H)


# ================================================================================================ evaluation split
def run_split(src: Any, fleet: Fleet, sites: np.ndarray, days: np.ndarray, *, models: Mapping[str, Sequence[Any]],
              n_scen: int, seed: int, chunk_days: int, timings: dict, log=print) -> dict:
    """Day-ahead and 15:00 scenarios for a set of days -> raw bands, scenario quantiles, actuals."""
    import pandas as pd

    q_lv = purchase_quantile()
    t0_ = pd.Timestamp(src.t_start)
    t_origin = (t0_.tz_convert("UTC").tz_localize(None) if t0_.tzinfo else t0_).to_datetime64().astype("datetime64[ns]")
    out = {k: [] for k in ("day", "ts", "da_bands", "da_q", "fo_bands", "id_bands", "id_q", "actual", "mask")}
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
            ts = t_origin + (int(d) * H + np.arange(H)) * np.timedelta64(15, "m")  # UTC, naive
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
    res = {k: (v if k == "day" else np.stack(v)) for k, v in out.items()}
    res["counts"] = counts
    return res


def _qstar_raw(total: np.ndarray, q: float) -> np.ndarray:
    """Raw quantile q* and P50 of the scenarios, (T, 2): kept so the conformal stretch can be applied afterwards
    (stretch is linear around P50, so stretching the quantile equals the quantile of the stretched scenarios)."""
    return np.stack([np.quantile(total, q, axis=0), np.quantile(total, 0.5, axis=0)], axis=-1)
