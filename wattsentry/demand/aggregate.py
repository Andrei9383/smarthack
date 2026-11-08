"""M3 DemandNet v1: zone demand from household load scenarios (plan §3.2 M3, §3.5 "same scenarios feed M3 v1").

Input: load scenarios `[S, N_sites, T]` in kW (from #27's rollout, or `scenarios_from_quantiles` on any load
envelope) + the site -> feeder -> locality -> county mapping. Each scenario is summed per zone first and the
quantiles are taken across the scenario sums, NOT summed site quantiles (that would overstate the zone's P90 because
independent households do not all peak high together).

Output per zone (feeder, locality, county): P10/P50/P90 per step in MW, peak step/hour (Europe/Bucharest), and
stress = P90 of the scenario daily peak / zone capacity proxy (feeder `g_max_kw` sums by default).
"""

from __future__ import annotations

import hashlib
import math
import warnings
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Optional, Sequence

import numpy as np

from wattsentry.contracts import (
    STEP_MINUTES,
    Action,
    ActionType,
    AuditRecord,
    ModelId,
    Prediction,
    input_hash,
)

from .calendar import local_hour, parse_ts, to_bucharest, utc_iso

LEVELS = ("feeder", "locality", "county")
LABEL_SYNTHETIC = "synthetic RO"
LABEL_REAL = "real RO"
V1_VERSION = "demandnet-v1-agg"
DEFAULT_SITE_KW = 4.0  # capacity proxy per household when a feeder has no g_max_kw
DEFAULT_STRESS_THRESHOLD = 0.8  # peak_stress >= this -> zone status "stressed" in the audit record


def _rows(x: Any) -> list[dict[str, Any]]:
    """Sequence of dicts, or a polars DataFrame (anything with .to_dicts())."""
    if x is None:
        return []
    if hasattr(x, "to_dicts"):
        return list(x.to_dicts())
    return [dict(r) for r in x]


def locality_zone_id(county: str, locality: str) -> str:
    """Locality names repeat across counties (e.g. 'Valea Mare'), so the zone id is 'CT/Cernavodă'."""
    return f"{county}/{locality}"


@dataclass
class Zone:
    zone_id: str
    level: str  # feeder | locality | county
    name: str
    parent_id: Optional[str]
    site_idx: np.ndarray  # indices into the scenario array's site axis
    capacity_mw: float


@dataclass
class ZoneDemand:
    """One zone's day-ahead demand. Curves are MW per step (length T)."""

    zone_id: str
    level: str
    name: str
    parent_id: Optional[str]
    t0: str
    q10_mw: list[float]
    q50_mw: list[float]
    q90_mw: list[float]
    mean_mw: list[float]
    peak_step: int  # argmax of q50
    peak_t: str  # UTC ISO start of peak step
    peak_mw: float  # q50 at the peak step
    peak_hour_local: int  # Europe/Bucharest hour of the peak step
    p90_peak_mw: float  # P90 across scenarios of each scenario's daily max
    capacity_mw: float
    stress: float  # p90_peak_mw / capacity_mw (can exceed 1 = over capacity)
    n_sites: int
    n_missing_sites: int = 0
    label: str = LABEL_SYNTHETIC
    model_version: str = V1_VERSION
    step_minutes: int = STEP_MINUTES
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def peak_stress(self) -> float:
        """Stress clipped to 0..1 for the UI red scale (zone_demand.peak_stress)."""
        return float(min(1.0, max(0.0, self.stress)))

    def to_prediction(self) -> Prediction:
        """The zone as a standard prediction envelope (unit MW, model demandnet)."""
        return Prediction(
            asset_id=self.zone_id,
            model=ModelId.DEMANDNET.value,
            model_version=self.model_version,
            t0=self.t0,
            horizon=len(self.q50_mw),
            q10=self.q10_mw,
            q50=self.q50_mw,
            q90=self.q90_mw,
            unit="MW",
            step_minutes=self.step_minutes,
            extra={"level": self.level, "peak_t": self.peak_t, "peak_hour_local": self.peak_hour_local,
                   "stress": self.stress, "capacity_mw": self.capacity_mw, "label": self.label},
        )


@dataclass
class DemandAggregate:
    zones: list[ZoneDemand]
    zone_scenarios: dict[str, np.ndarray]  # zone_id -> [S, T] MW scenario sums (kept for checks / M3 stress)
    times: np.ndarray  # datetime64[s] UTC step starts

    def by_id(self) -> dict[str, ZoneDemand]:
        return {z.zone_id: z for z in self.zones}

    def level(self, level: str) -> list[ZoneDemand]:
        return [z for z in self.zones if z.level == level]


def build_hierarchy(
    sites: Any,
    feeders: Any = None,
    *,
    feeder_key: str = "feeder_id",
    capacity: str | Mapping[str, float] = "g_max",
    default_site_kw: float = DEFAULT_SITE_KW,
) -> tuple[list[str], list[Zone]]:
    """Zones for feeder -> locality -> county from `sites` (+ optional `feeders` table, data.md).

    `sites`: rows with `site_id`, `feeder_key` (default `feeder_id`, the simulator truth; pass
    `gis_feeder_id` to aggregate as the GIS records it) and `locality`/`county` (used when the feeder row lacks them).
    `capacity`: "g_max" (sum of feeder g_max_kw), "limit" (tau·g_max + (1 − tau)·g_conservative, the LP limit),
    or an explicit {zone_id: MW} map (missing zones fall back to "g_max"). Feeders with no capacity get
    `default_site_kw` × n_sites.
    Returns (site_ids in scenario order, zones).
    """
    site_rows = _rows(sites)
    feeder_rows = {str(f["feeder_id"]): f for f in _rows(feeders)}
    site_ids = [str(s["site_id"]) for s in site_rows]

    feeder_sites: dict[str, list[int]] = {}
    feeder_meta: dict[str, tuple[str, str]] = {}
    for i, s in enumerate(site_rows):
        fid = str(s.get(feeder_key) or s.get("feeder_id") or "unassigned")
        f = feeder_rows.get(fid, {})
        county = str(f.get("county") or s.get("county") or "NA")
        locality = str(f.get("locality") or s.get("locality") or "NA")
        feeder_sites.setdefault(fid, []).append(i)
        feeder_meta.setdefault(fid, (county, locality))

    def feeder_cap_mw(fid: str) -> float:
        f = feeder_rows.get(fid, {})
        n = len(feeder_sites[fid])
        g = f.get("g_max_kw")
        if capacity == "limit" and g is not None and f.get("g_conservative_kw") is not None:
            tau = float(f.get("tau") if f.get("tau") is not None else 1.0)
            g = tau * float(g) + (1 - tau) * float(f["g_conservative_kw"])
        if g is None or not math.isfinite(float(g)) or float(g) <= 0:
            g = default_site_kw * n
        return float(g) / 1000.0

    overrides = capacity if isinstance(capacity, Mapping) else {}
    zones: list[Zone] = []
    loc_sites: dict[str, list[int]] = {}
    loc_cap: dict[str, float] = {}
    loc_meta: dict[str, tuple[str, str]] = {}
    for fid in sorted(feeder_sites):
        county, locality = feeder_meta[fid]
        lid = locality_zone_id(county, locality)
        cap = feeder_cap_mw(fid)
        zones.append(Zone(fid, "feeder", fid, lid, np.array(feeder_sites[fid]), float(overrides.get(fid, cap))))
        loc_sites.setdefault(lid, []).extend(feeder_sites[fid])
        loc_cap[lid] = loc_cap.get(lid, 0.0) + cap
        loc_meta[lid] = (county, locality)
    county_sites: dict[str, list[int]] = {}
    county_cap: dict[str, float] = {}
    for lid in sorted(loc_sites):
        county, locality = loc_meta[lid]
        zones.append(Zone(lid, "locality", locality, county, np.array(sorted(loc_sites[lid])),
                          float(overrides.get(lid, loc_cap[lid]))))
        county_sites.setdefault(county, []).extend(loc_sites[lid])
        county_cap[county] = county_cap.get(county, 0.0) + loc_cap[lid]
    for cid in sorted(county_sites):
        zones.append(Zone(cid, "county", cid, None, np.array(sorted(county_sites[cid])),
                          float(overrides.get(cid, county_cap[cid]))))
    return site_ids, zones


def _fill_missing(scen: np.ndarray) -> np.ndarray:
    """NaN (missing telemetry / failed rollout) -> that site-step's mean over the other scenarios, else 0."""
    if not np.isnan(scen).any():
        return scen
    scen = scen.copy()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)  # all-NaN slices
        fill = np.nan_to_num(np.nanmean(scen, axis=0), nan=0.0)  # [N, T]
    idx = np.where(np.isnan(scen))
    scen[idx] = fill[idx[1], idx[2]]
    return scen


def aggregate_scenarios(
    scenarios: np.ndarray,
    sites: Any,
    feeders: Any = None,
    *,
    t0: str,
    step_minutes: int = STEP_MINUTES,
    unit: str = "kW",
    feeder_key: str = "feeder_id",
    capacity: str | Mapping[str, float] = "g_max",
    default_site_kw: float = DEFAULT_SITE_KW,
    label: str = LABEL_SYNTHETIC,
    model_version: str = V1_VERSION,
    quantiles: tuple[float, float, float] = (0.1, 0.5, 0.9),
) -> DemandAggregate:
    """Zone P10/P50/P90 from scenario sums. `scenarios`: [S, N_sites, T] household load, sites in `sites` order.

    `unit`: "kW" (household default), "W" or "MW". Output curves are MW.
    """
    scen = np.asarray(scenarios, dtype=float)
    if scen.ndim != 3:
        raise ValueError(f"scenarios must be [S, N_sites, T], got shape {scen.shape}")
    site_ids, zones = build_hierarchy(sites, feeders, feeder_key=feeder_key, capacity=capacity,
                                      default_site_kw=default_site_kw)
    if scen.shape[1] != len(site_ids):
        raise ValueError(f"scenarios have {scen.shape[1]} sites, mapping has {len(site_ids)}")
    to_mw = {"W": 1e-6, "kW": 1e-3, "MW": 1.0}[unit]
    missing_site = np.isnan(scen).all(axis=(0, 2))  # sites with no data at all
    scen = _fill_missing(scen)
    S, N, T = scen.shape

    start = parse_ts(t0)
    times = start + np.arange(T) * np.timedelta64(step_minutes, "m")
    hours = local_hour(times)

    out: list[ZoneDemand] = []
    zone_scen: dict[str, np.ndarray] = {}
    for z in zones:
        zs = scen[:, z.site_idx, :].sum(axis=1) * to_mw  # [S, T]  scenario sums
        zone_scen[z.zone_id] = zs
        q10, q50, q90 = np.quantile(zs, quantiles, axis=0)
        q50 = np.maximum(q50, q10)
        q90 = np.maximum(q90, q50)
        k = int(np.argmax(q50))
        p90_peak = float(np.quantile(zs.max(axis=1), quantiles[2]))
        cap = z.capacity_mw
        out.append(ZoneDemand(
            zone_id=z.zone_id, level=z.level, name=z.name, parent_id=z.parent_id, t0=utc_iso(start),
            q10_mw=q10.tolist(), q50_mw=q50.tolist(), q90_mw=q90.tolist(), mean_mw=zs.mean(axis=0).tolist(),
            peak_step=k, peak_t=utc_iso(times[k]), peak_mw=float(q50[k]), peak_hour_local=int(hours[k]),
            p90_peak_mw=p90_peak, capacity_mw=cap, stress=float(p90_peak / cap) if cap > 0 else float("inf"),
            n_sites=int(len(z.site_idx)), n_missing_sites=int(missing_site[z.site_idx].sum()),
            label=label, model_version=model_version, step_minutes=step_minutes,
            extra={"n_scenarios": S},
        ))
    return DemandAggregate(zones=out, zone_scenarios=zone_scen, times=times)


def scenarios_from_quantiles(
    q10: np.ndarray,
    q50: np.ndarray,
    q90: np.ndarray,
    n_scenarios: int = 200,
    *,
    seed: int = 0,
    time_corr: float = 0.97,
    common: float = 0.0,
) -> np.ndarray:
    """Sample [S, N, T] scenarios from per-site quantile curves [N, T] (stand-in until #27's `make_scenarios`).

    Each site draws a Gaussian-copula level per scenario, AR(1) over time (`time_corr`), with an optional shared
    component (`common` = correlation between sites, e.g. weather). The level maps through the piecewise-linear
    quantile function q10/q50/q90 (tails extrapolated linearly from the 10-90 % spread, floored at 0).
    """
    q10, q50, q90 = (np.asarray(a, dtype=float) for a in (q10, q50, q90))
    if q10.ndim == 1:
        q10, q50, q90 = q10[None], q50[None], q90[None]
    N, T = q50.shape
    rng = np.random.default_rng(seed)
    z = np.empty((n_scenarios, N, T))
    z[..., 0] = rng.standard_normal((n_scenarios, N))
    innov = math.sqrt(1 - time_corr**2)
    for t in range(1, T):
        z[..., t] = time_corr * z[..., t - 1] + innov * rng.standard_normal((n_scenarios, N))
    if common > 0:
        g = np.empty((n_scenarios, 1, T))
        g[..., 0] = rng.standard_normal((n_scenarios, 1))
        for t in range(1, T):
            g[..., t] = time_corr * g[..., t - 1] + innov * rng.standard_normal((n_scenarios, 1))
        z = math.sqrt(1 - common) * z + math.sqrt(common) * g
    z90 = 1.2815515655446004
    lo = q50 - q10  # [N, T]
    hi = q90 - q50
    x = np.where(z < 0, q50 + z / z90 * lo, q50 + z / z90 * hi)
    return np.maximum(x, 0.0)


def mock_load_scenarios(sites: Any, t0: str, n_scenarios: int = 200, *, seed: int = 0, horizon: int = 96,
                        **kw: Any) -> np.ndarray:
    """[S, N, T] kW scenarios from MockPredictor('load') envelopes (tests / demos until #27 lands)."""
    from wattsentry.contracts import MockPredictor

    mock = MockPredictor("load")
    preds = [mock.predict(s, t0=t0, horizon=horizon) for s in _rows(sites)]
    q10 = np.array([p.q10 for p in preds])
    q50 = np.array([p.q50 for p in preds])
    q90 = np.array([p.q90 for p in preds])
    return scenarios_from_quantiles(q10, q50, q90, n_scenarios, seed=seed, **kw)


# --- outputs for #17 (SQLite rows + agent actions) -------------------------------------------------------

ROW_FIELDS = (
    "run_id", "zone_type", "level", "zone_id", "zone_name", "parent_zone_id", "ts", "t", "step",
    "q10_mw", "q50_mw", "q90_mw", "p10_mw", "p50_mw", "p90_mw", "stress", "peak_stress", "is_peak_hour", "label", "model_version",
)


def zone_demand_rows(agg: DemandAggregate | Sequence[ZoneDemand], run_id: str) -> list[dict[str, Any]]:
    """One row per zone per step for the `zone_demand` table (schema draft from #30 / replay.md).

    `ts` = UTC ISO step start, `t` = unix seconds (draft schema), `step` = 0..95 index in the Europe/Bucharest
    local day. `stress` = q90 at this step / capacity (drives the 24 h slider); `peak_stress` = zone peak stress
    clipped 0..1 (same on every row of a zone). `zone_type`/`level` and `q*_mw`/`p*_mw` carry the same values
    (#38 schema names first, older draft names until #38 merges; #17 inserts only the columns the schema has).
    """
    zones = agg.zones if isinstance(agg, DemandAggregate) else list(agg)
    rows: list[dict[str, Any]] = []
    for z in zones:
        start = parse_ts(z.t0)
        T = len(z.q50_mw)
        times = start + np.arange(T) * np.timedelta64(z.step_minutes, "m")
        loc = to_bucharest(times)
        steps = ((loc - loc.astype("datetime64[D]")).astype("timedelta64[m]").astype(int) // z.step_minutes)
        hours = local_hour(times)
        peak_day = loc[z.peak_step].astype("datetime64[D]")
        is_peak = (hours == z.peak_hour_local) & (loc.astype("datetime64[D]") == peak_day)
        for k in range(T):
            rows.append({
                "run_id": run_id,
                "zone_type": z.level,
                "level": z.level,
                "zone_id": z.zone_id,
                "zone_name": z.name,
                "parent_zone_id": z.parent_id,
                "ts": utc_iso(times[k]),
                "t": int(times[k].astype("datetime64[s]").astype(np.int64)),
                "step": int(steps[k]),
                "q10_mw": z.q10_mw[k],  # final #38 schema names
                "q50_mw": z.q50_mw[k],
                "q90_mw": z.q90_mw[k],
                "p10_mw": z.q10_mw[k],  # older draft names, kept until #38 merges
                "p50_mw": z.q50_mw[k],
                "p90_mw": z.q90_mw[k],
                "stress": z.q90_mw[k] / z.capacity_mw if z.capacity_mw > 0 else None,
                "peak_stress": z.peak_stress,
                "is_peak_hour": int(is_peak[k]),
                "label": z.label,
                "model_version": z.model_version,
            })
    return rows


def _array_digest(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a, dtype=np.float32).tobytes()).hexdigest()[:16]


def zone_demand_actions(
    agg: DemandAggregate,
    *,
    run_id: str,
    t: Optional[str] = None,
    policy: str = "sentinel",
    levels: Iterable[str] = LEVELS,
    stress_threshold: float = DEFAULT_STRESS_THRESHOLD,
) -> list[Action]:
    """One `zone_demand` Action per zone (actions.md), with an audit record pinning the scenario input."""
    levels = set(levels)
    out: list[Action] = []
    for z in agg.zones:
        if z.level not in levels:
            continue
        scen = agg.zone_scenarios.get(z.zone_id)
        ih = input_hash({"zone_id": z.zone_id, "t0": z.t0, "n_sites": z.n_sites,
                         "scenarios_sha": _array_digest(scen) if scen is not None else None})
        status = "stressed" if z.peak_stress >= stress_threshold else "ok"
        payload = {
            "zone_id": z.zone_id,
            "level": z.level,
            "zone_name": z.name,
            "parent_zone_id": z.parent_id,
            "q10_mw": z.q10_mw,
            "q50_mw": z.q50_mw,
            "q90_mw": z.q90_mw,
            "peak_t": z.peak_t,
            "peak_mw": z.peak_mw,
            "peak_hour_local": z.peak_hour_local,
            "p90_peak_mw": z.p90_peak_mw,
            "capacity_mw": z.capacity_mw,
            "peak_stress": z.peak_stress,
            "stress": z.stress,
            "label": z.label,
        }
        out.append(Action(
            action_id=f"zd-{run_id}-{z.level}-{z.zone_id}-{z.t0}",
            type=ActionType.ZONE_DEMAND.value,
            t=t or z.t0,
            policy=policy,
            run_id=run_id,
            payload=payload,
            audit=AuditRecord(
                model_version=z.model_version,
                input_hash=ih,
                prediction={"peak_mw": round(z.peak_mw, 4), "p90_peak_mw": round(z.p90_peak_mw, 4),
                            "peak_t": z.peak_t, "q50_sum_mwh": round(sum(z.q50_mw) * z.step_minutes / 60, 4),
                            "n_sites": z.n_sites, "n_missing_sites": z.n_missing_sites},
                threshold=stress_threshold,
                status=status,
                note=f"stress = P90 scenario peak / capacity proxy ({z.capacity_mw:.3f} MW); {z.label}",
            ),
        ))
    return out
