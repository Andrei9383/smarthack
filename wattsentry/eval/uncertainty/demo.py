"""End-to-end demo on MockPredictor forecasts (CPU, seconds): calibrate → scenarios → commit → realise → score.

The mock has no actuals, so a synthetic "truth" is drawn around its quantiles with a deliberately wider spread
than the mock claims (`inflate`), daily weather shared across a feeder and per-step noise. Conformal calibration
has to fix that miscalibration; the P10 hit rate then checks the whole chain. #18's LP is not available here, so
the demo schedule is a simple rule (`rule_schedule`): charge in the feeder's midday export peak, discharge in the
evening, idle otherwise.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import timedelta
from typing import Any, Optional, Sequence

import numpy as np

from wattsentry.contracts import DAY_STEPS, DT_H, STEP_MINUTES, BatterySpec, MockPredictor, parse_utc
from wattsentry.eval.uncertainty.conformal import ConformalCalibration, interval_report, quantile_at
from wattsentry.eval.uncertainty.thresholds import rel_width, tune_thresholds
from wattsentry.opt.commit import commit, hit_stats, make_scenarios, realised_deliverable

MEMBER_SCALES = (0.94, 1.0, 1.06)  # pseudo-ensemble around the mock (the mock itself has one member)


# --- synthetic feeder ----------------------------------------------------------------------------------------
@dataclass
class DemoFeeder:
    feeder_id: str
    sites: list[dict[str, Any]]
    batteries: dict[str, Optional[BatterySpec]]
    soc0: dict[str, float]

    @property
    def site_ids(self) -> list[str]:
        return [s["site_id"] for s in self.sites]


def make_feeder(feeder_id: str, n_sites: int = 20, seed: int = 0, battery_share: float = 0.85) -> DemoFeeder:
    """RO-like residential sites: kWp 3-10 (R7), battery 5-15 kWh / 3-5 kW, comfort reserve 0.5-2 kWh."""
    rng = np.random.default_rng(seed)
    sites, batts, soc0 = [], {}, {}
    for i in range(n_sites):
        sid = f"{feeder_id}.S{i:03d}"
        sites.append({"site_id": sid, "kwp_true": float(rng.uniform(3, 10)), "lat": float(rng.uniform(44, 47.5)),
                      "archetype": str(rng.choice(["ac", "heat_pump", "standard"]))})
        if rng.random() < battery_share:
            cap = float(rng.uniform(5, 15))
            batts[sid] = BatterySpec(capacity_kwh=cap, p_max_kw=float(rng.uniform(3, 5)),
                                     comfort_reserve_kwh=float(rng.uniform(0.5, 2.0)))
        else:
            batts[sid] = None
        soc0[sid] = float(rng.uniform(0.25, 0.5))
    return DemoFeeder(feeder_id, sites, batts, soc0)


def mock_members(site: dict[str, Any], t0: str, kind: str) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Mock envelope + scaled copies as a 3-member pseudo-ensemble."""
    p = MockPredictor(kind).predict(site, t0=t0)
    q = [np.asarray(x, dtype=float) for x in (p.q10, p.q50, p.q90)]
    return [tuple(a * s for a in q) for s in MEMBER_SCALES]  # type: ignore[misc]


def envelope_of(members: Sequence[tuple[np.ndarray, np.ndarray, np.ndarray]]):
    return tuple(np.mean([m[k] for m in members], axis=0) for k in range(3))


# --- synthetic truth -----------------------------------------------------------------------------------------
def draw_truth(q10: np.ndarray, q50: np.ndarray, q90: np.ndarray, z_day: np.ndarray, rng: np.random.Generator,
               inflate: float = 1.5, rho: float = 0.7, cap: Optional[float] = None) -> np.ndarray:
    """y_t = Q(z_t), z_t = inflate·(rho·z_day + sqrt(1 - rho²)·eps_t): marginal N(0, inflate²) in z, i.e. the
    truth is wider than the forecast claims. z_day (...,) broadcasts against the leading axes of q (..., T)."""
    eps = rng.standard_normal(np.broadcast_shapes(np.shape(q50), np.shape(z_day) + (q50.shape[-1],)))
    z = inflate * (rho * np.asarray(z_day)[..., None] + np.sqrt(1 - rho ** 2) * eps)
    y = np.maximum(quantile_at(q10, q50, q90, z), 0.0)
    return np.minimum(y, cap) if cap is not None else y


def feeder_truth(feeder: DemoFeeder, fc_pv, fc_load, rng: np.random.Generator, n_draws: int = 1,
                 inflate: float = 1.5):
    """{site: (n_draws, T)} actual pv / load for one feeder-day. PV daily level shared 80% across the feeder."""
    z_feeder = rng.standard_normal(n_draws)
    pv, load = {}, {}
    for s in feeder.sites:
        sid = s["site_id"]
        z_pv = 0.8 * z_feeder + 0.6 * rng.standard_normal(n_draws)
        pv[sid] = draw_truth(*envelope_of(fc_pv[sid]), z_pv, rng, inflate, cap=s["kwp_true"])
        load[sid] = draw_truth(*envelope_of(fc_load[sid]), rng.standard_normal(n_draws), rng, inflate)
    return pv, load


# --- rule schedule (stand-in for #18's LP) -------------------------------------------------------------------
def _local_hours(t0: str, horizon: int = DAY_STEPS) -> np.ndarray:
    start = parse_utc(t0)
    hours = []
    for k in range(horizon):
        t = start + timedelta(minutes=STEP_MINUTES * k)
        off = 3 if 4 <= t.month <= 10 else 2  # Europe/Bucharest ≈ UTC+3 summer / +2 winter
        hours.append((t.hour + t.minute / 60 + off) % 24)
    return np.array(hours)


def rule_schedule(site_ids: Sequence[str], pv_p50: np.ndarray, load_p50: np.ndarray,
                  batteries: dict[str, Optional[BatterySpec]], soc0: dict[str, float], t0: str,
                  peak_steps: int = 16, evening: tuple[float, float] = (17.0, 22.0)) -> dict[str, list[float]]:
    """Charge evenly over the `peak_steps` steps of largest feeder P50 surplus (enough to fill the battery),
    discharge to cover P50 net load in the evening window, idle otherwise. pv_p50/load_p50: (S, T)."""
    surplus = pv_p50 - load_p50
    feeder_surplus = surplus.sum(axis=0)
    order = np.argsort(-feeder_surplus)
    window = order[:peak_steps][feeder_surplus[order[:peak_steps]] > 0]
    hours = _local_hours(t0, pv_p50.shape[1])
    eve = (hours >= evening[0]) & (hours < evening[1])
    out = {}
    for j, sid in enumerate(site_ids):
        b = batteries.get(sid)
        s = np.zeros(pv_p50.shape[1])
        if b is not None and b.capacity_kwh > 0 and window.size:
            room = max(0.0, b.e_max - soc0.get(sid, 0.5) * b.capacity_kwh)
            s[window] = min(b.p_max_kw, room / (b.eta_charge * DT_H * window.size))
            s[eve] = -np.minimum(b.p_max_kw, np.maximum(0.0, -surplus[j, eve]))
        out[sid] = s.tolist()
    return out


# --- pipeline ------------------------------------------------------------------------------------------------
def _days(start: str, n: int, every: int = 1) -> list[str]:
    d0 = parse_utc(start)
    return [(d0 + timedelta(days=every * i)).strftime("%Y-%m-%dT%H:%M:%SZ") for i in range(n)]


def collect(feeders: Sequence[DemoFeeder], days: Sequence[str], rng: np.random.Generator, inflate: float):
    """Forecast members + one truth draw per (feeder, day, site); returns arrays for calibration."""
    rows = {"pv": {"y": [], "members": [[] for _ in MEMBER_SCALES]},
            "load": {"y": [], "members": [[] for _ in MEMBER_SCALES]}}
    for f in feeders:
        for t0 in days:
            fc_pv = {s["site_id"]: mock_members(s, t0, "pv") for s in f.sites}
            fc_ld = {s["site_id"]: mock_members(s, t0, "load") for s in f.sites}
            pv, ld = feeder_truth(f, fc_pv, fc_ld, rng, 1, inflate)
            for sid in f.site_ids:
                for kind, fc, y in (("pv", fc_pv, pv), ("load", fc_ld, ld)):
                    rows[kind]["y"].append(y[sid][0])
                    for k, m in enumerate(fc[sid]):
                        rows[kind]["members"][k].append(m)
    out = {}
    for kind, r in rows.items():
        y = np.array(r["y"])
        mem = {str(k): tuple(np.array([m[i] for m in ms]) for i in range(3)) for k, ms in enumerate(r["members"])}
        out[kind] = (y, mem)
    return out


def run(n_feeders: int = 8, n_sites: int = 10, cal_days: int = 30, test_days: int = 10, n_scenarios: int = 200,
        actual_draws: int = 30, haircut: float = 0.05, inflate: float = 1.5, seed: int = 0,
        start: str = "2025-06-02T00:00:00Z") -> dict[str, Any]:
    """Calibrate on `cal_days`, then commit + score on the following `test_days`. Returns metrics + artifacts."""
    t_start = time.perf_counter()
    rng = np.random.default_rng(seed)
    feeders = [make_feeder(f"F{i:02d}", n_sites, seed + i) for i in range(n_feeders)]
    days = _days(start, cal_days + test_days)
    cal_d, test_d = days[:cal_days], days[cal_days:]

    # 1. conformal calibration per member, on the calibration slice
    cal_data = collect(feeders, cal_d, rng, inflate)
    cals = {k: ConformalCalibration.fit(y, mem, model=k, model_version="mock-1") for k, (y, mem) in cal_data.items()}

    # 2. coverage + pinball on the test slice (envelope = member mean, calibrated per member then averaged)
    test_data = collect(feeders, test_d, rng, inflate)
    metrics: dict[str, Any] = {}
    reports = {}
    for kind, (y, mem) in test_data.items():
        raw = envelope_of(list(mem.values()))
        cal = envelope_of(cals[kind].apply_members(list(mem.values())))
        reports[kind] = {"raw": interval_report(y, *raw), "calibrated": interval_report(y, *cal)}
    metrics["coverage_10_90"] = reports["pv"]["calibrated"]["coverage_10_90"]
    metrics["coverage_10_90_raw"] = reports["pv"]["raw"]["coverage_10_90"]
    metrics["coverage_10_90_load"] = reports["load"]["calibrated"]["coverage_10_90"]
    metrics["pinball_pv"] = reports["pv"]["calibrated"]["pinball"]
    metrics["pinball_pv_raw"] = reports["pv"]["raw"]["pinball"]

    # 3. thresholds tuned on the test slice's PV forecasts (demo only; the frozen file comes from a real run)
    y, mem = test_data["pv"]
    cal_env = envelope_of(cals["pv"].apply_members(list(mem.values())))
    rw = rel_width(*cal_env)
    outside = ((y < cal_env[0]) | (y > cal_env[2])) & (cal_env[1] > 1e-3)
    wrong = outside.sum(axis=1) > 0.5 * np.maximum((cal_env[1] > 1e-3).sum(axis=1), 1)
    thresholds = tune_thresholds({"rel_width": rw, "wrong": wrong}, source_run="mock-demo")

    # 4. P10 commitment per feeder-day, scored on `actual_draws` truth draws each
    commitments, cm_all, rl_all = [], [], []
    for f in feeders:
        for t0 in test_d:
            fc_pv = {s["site_id"]: mock_members(s, t0, "pv") for s in f.sites}
            fc_ld = {s["site_id"]: mock_members(s, t0, "load") for s in f.sites}
            scen = make_scenarios(fc_pv, fc_ld, n=n_scenarios, seed=int(rng.integers(1 << 31)),
                                  pv_calibration=cals["pv"], load_calibration=cals["load"],
                                  pv_cap={s["site_id"]: s["kwp_true"] for s in f.sites},
                                  batteries=f.batteries, soc0=f.soc0, feeder_id=f.feeder_id, t0=t0)
            sched = rule_schedule(scen.site_ids, scen.pv_p50, scen.load_p50, f.batteries, f.soc0, t0)
            c = commit(f.feeder_id, sched, scen, haircut=haircut)
            pv_a, ld_a = feeder_truth(f, fc_pv, fc_ld, rng, actual_draws, inflate)
            r = realised_deliverable(sched, pv_a, ld_a, f.batteries, f.soc0)  # (draws, T)
            cm_all.append(np.broadcast_to(np.asarray(c.committed), r.shape))
            rl_all.append(r)
            c.realised = [float(x) for x in r[0]]
            commitments.append(c)
    metrics.update(hit_stats(committed=np.concatenate(cm_all), realised=np.concatenate(rl_all)))
    metrics["committed_kwh_per_feeder_day"] = float(np.mean([sum(c.committed) * DT_H for c in commitments]))
    metrics["nominal_kwh_per_feeder_day"] = float(np.mean([sum(c.nominal) * DT_H for c in commitments]))
    metrics["runtime_s"] = time.perf_counter() - t_start
    return {"metrics": metrics, "reports": reports, "calibrations": cals, "thresholds": thresholds,
            "commitments": commitments}
