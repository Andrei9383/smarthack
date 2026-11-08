"""Voltage rise and inverter trips on the IEEE European LV test feeder, baseline vs Sentinel (card #160, Spec G step 2).

    python -m wattsentry.gridtwin.voltage [--world sim|mock] [--days 7] [--v-source 1.00,1.03,1.05] [--out DIR]

What it does:
  1. Replays the world once per policy (self-consumption baseline, Sentinel) through the agent loop
     (`wattsentry.agent.loop.run_policy`, the same code as replay.py) and rebuilds every home's 15-min net export from
     the physics (pv - load - executed battery power), see `home_net_kw`.
  2. Puts the simulated homes on the 55 single-phase loads of the IEEE European LV test feeder (pandapower
     `ieee_european_lv_asymmetric`, 906 LV buses, one 0.8 MVA 11/0.4 kV transformer), hero feeder first.
  3. Computes a voltage sensitivity matrix once (finite differences of a full 3-phase power flow around a midday
     export operating point) and estimates every bus / phase voltage of every step at once (one matrix product).
  4. Checks the estimate against a full pandapower 3-phase power flow on a random 5 % of the steps (max error
     reported) and replaces the estimate by the full power flow wherever it lands within 1 % of the 253 V limit.
  5. Counts steps with any bus above 253 V (1.10 pu of 230 V), the PV production of homes whose own connection point
     is above 253 V in that step (their inverter disconnects: `inverter_trip_kwh`) and its value at the OPCOM PZU
     price (`inverter_trip_lei`).

Labels: topology non-RO (IEEE European LV test feeder, a generic UK-style network, not a Romanian street); homes and
their batteries synthetic RO (the world's label); prices real RO (OPCOM PZU). Every metric line says so.

Modelling choices (stated, not hidden):
  - The IEEE network is rebased from 0.416 kV to a 230/400 V nominal (LV bus vn_kv and transformer vn_lv_kv = 0.4)
    and the no-load LV busbar voltage is set by `v_source_pu` (ext grid, nominal tap). The IEEE default (1.05 pu
    on 0.416 kV = 252 V at no load) leaves no headroom at all, so the run reports a small grid of busbar settings.
  - Homes inject at unity power factor (q = 0); the net export of a home goes on the phase of its IEEE load.
  - Trips are first-order: a tripped inverter is counted as lost for the whole 15-min step and its disconnection does
    not lower its neighbours' voltage (no cascade / reconnect dynamics), for both policies alike.
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np

V_LIMIT = 253.0  # inverter over-voltage trip, V phase-neutral (EN 50549 / RO 1.10 pu of 230 V)
V_NOMINAL = 230.0
V_BASE = 400.0 / math.sqrt(3.0)  # pu base of the rebased network (0.4 kV line-line)
N_LOADS = 55
P_REF_KW = 2.0  # linearisation point: every home exporting this much (on replay#506's week this gave the smallest
#                 error both overall and near the limit; 3-4 kW doubles it, the top-decile mean export is no better)
DP_KW = 1.0  # finite-difference step
NEAR_LIMIT = 0.01  # full power flow where the estimated max voltage is within 1 % of the limit
VALIDATE_FRAC = 0.05
DT_H = 0.25
LABEL_TOPOLOGY = "non-RO (IEEE European LV test feeder)"
LABEL_PRICES = "real RO (OPCOM PZU)"
PRICES_PARQUET = Path(__file__).resolve().parents[2] / "data" / "raw" / "prices.parquet"


# ============================================================================================ network
@dataclass
class Feeder:
    """The IEEE European LV feeder as a pandapower net plus where the 55 homes connect."""

    net: Any
    v_source_pu: float
    load_bus: np.ndarray  # (55,) pandapower bus index of each load
    load_phase: np.ndarray  # (55,) 0/1/2 = a/b/c
    buses: np.ndarray  # (B,) monitored LV buses
    home_rows: np.ndarray = field(default=None)  # (55,) row of each home's own (bus, phase) in the flat V vector

    @property
    def n_mon(self) -> int:
        return 3 * len(self.buses)


def _pandapower():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        import logging

        logging.getLogger("pandapower").setLevel(logging.ERROR)
        import pandapower as pp
        import pandapower.networks as pn
    return pp, pn


def build_feeder(v_source_pu: float = 1.03) -> Feeder:
    """IEEE European LV test feeder, rebased to 400 V, no-load busbar at `v_source_pu` x 230.9 V."""
    _, pn = _pandapower()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        net = pn.ieee_european_lv_asymmetric("off_peak_1")
    lv = net.bus.vn_kv < 1.0
    net.bus.loc[lv, "vn_kv"] = 0.4
    net.trafo["vn_lv_kv"] = 0.4
    net.ext_grid["vm_pu"] = float(v_source_pu)
    al = net.asymmetric_load.sort_index()
    if len(al) != N_LOADS:
        raise ValueError(f"expected {N_LOADS} loads on the IEEE European LV feeder, got {len(al)}")
    phase = np.argmax(np.abs(al[["p_a_mw", "p_b_mw", "p_c_mw"]].to_numpy()), axis=1)
    for c in "abc":
        net.asymmetric_load[f"p_{c}_mw"] = 0.0
        net.asymmetric_load[f"q_{c}_mvar"] = 0.0
    net.asymmetric_load["scaling"] = 1.0
    buses = np.asarray(net.bus.index[lv])
    fd = Feeder(net, float(v_source_pu), al.bus.to_numpy(), phase, buses)
    pos = {b: i for i, b in enumerate(buses)}
    fd.home_rows = np.array([3 * pos[b] + p for b, p in zip(fd.load_bus, fd.load_phase)])
    return fd


def power_flow(fd: Feeder, p_kw: np.ndarray) -> np.ndarray:
    """Full 3-phase power flow for one step. `p_kw` (55,) net export per home (+ export). Returns phase-neutral
    voltages in V, flat (B*3,) ordered bus-major [b0a, b0b, b0c, b1a, ...]."""
    pp, _ = _pandapower()
    al = fd.net.asymmetric_load
    p = np.asarray(p_kw, float)
    for j, c in enumerate("abc"):
        al[f"p_{c}_mw"] = np.where(fd.load_phase == j, -p / 1000.0, 0.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pp.runpp_3ph(fd.net, numba=False)
    r = fd.net.res_bus_3ph.loc[fd.buses, ["vm_a_pu", "vm_b_pu", "vm_c_pu"]].to_numpy()
    return r.reshape(-1) * V_BASE


@dataclass
class Sensitivity:
    v_ref: np.ndarray  # (M,) voltages at the operating point
    p_ref: np.ndarray  # (55,)
    s: np.ndarray  # (M, 55) dV/dP_export, V per kW
    seconds: float

    def estimate(self, p_kw: np.ndarray) -> np.ndarray:
        """(T, 55) net export -> (T, M) voltages, linear around the operating point."""
        return self.v_ref[None, :] + (np.asarray(p_kw, float) - self.p_ref[None, :]) @ self.s.T


def sensitivity(fd: Feeder, p_ref_kw: float | np.ndarray = P_REF_KW, dp_kw: float = DP_KW) -> Sensitivity:
    """dV/dP by central differences of the full power flow around every home exporting `p_ref_kw` (2*55+1 PFs)."""
    t0 = time.perf_counter()
    p_ref = np.broadcast_to(np.asarray(p_ref_kw, float), (N_LOADS,)).copy()
    v_ref = power_flow(fd, p_ref)
    s = np.empty((fd.n_mon, N_LOADS))
    for i in range(N_LOADS):
        hi, lo = p_ref.copy(), p_ref.copy()
        hi[i] += dp_kw
        lo[i] -= dp_kw
        s[:, i] = (power_flow(fd, hi) - power_flow(fd, lo)) / (2 * dp_kw)
    return Sensitivity(v_ref, p_ref, s, time.perf_counter() - t0)


# ============================================================================================ homes
def map_homes(site_ids_by_feeder: dict[str, Sequence[str]], first: Optional[str] = None,
              n: int = N_LOADS) -> list[str]:
    """Site id for each of the n IEEE loads: feeder `first` (the hero) first, then the others in order, cycling
    if the world has fewer than n homes (deterministic)."""
    order = ([first] if first in site_ids_by_feeder else []) + [f for f in site_ids_by_feeder if f != first]
    pool = [s for f in order for s in site_ids_by_feeder[f]]
    if not pool:
        raise ValueError("no homes to map")
    return [pool[i % len(pool)] for i in range(n)]


class _ScheduleSink:
    """Agent-loop sink that keeps only the executed battery power per (site, ts)."""

    def __init__(self):
        self.batt: dict[tuple[str, int], float] = {}

    def add(self, table, row):
        if table == "schedules":
            self.batt[(str(row["asset_id"]), int(row["ts"]))] = float(row["batt_kw"])

    def add_many(self, table, rows):
        for r in rows:
            self.add(table, r)

    def has_table(self, table):
        return False

    def write_action(self, action, rows=None, *, ts=None, extra=None):
        for t, rs in (rows or {}).items():
            self.add_many(t, rs)


@dataclass
class Replay:
    ts: np.ndarray  # (T,) unix seconds
    site_ids: list[str]  # the world's homes
    pv_kw: np.ndarray  # (T, S) physics PV
    net_kw: dict[str, np.ndarray]  # policy -> (T, S) net export (+ export)
    label: str
    hero: Optional[str]
    by_feeder: dict[str, list[str]]
    world_prices_lei_mwh: np.ndarray  # (T,) what the world fed the planner (fallback for prices)
    seconds: dict[str, float]
    minutes_over_limit: dict[str, float] = field(default_factory=dict)  # world feeders vs their G (ties to replay)


def home_net_kw(world: Any, policies: Sequence[str] = ("baseline", "sentinel"), comps: Any = None) -> Replay:
    """Run the agent loop once per policy and rebuild each home's net export per 15 min: pv - load - battery,
    where pv / load are the world's physics (`site_day`, as in the loop) and battery is the executed power."""
    from wattsentry.agent.loop import DAY_STEPS, LoopConfig, default_components, run_policy
    from wattsentry.agent.replay import hero_feeder

    comps = comps or default_components(world)
    sids = list(world.sites)
    ts = np.array([world.tick_ts(d, k) for d in range(world.days) for k in range(DAY_STEPS)], dtype=np.int64)
    pv = np.empty((len(ts), len(sids)))
    ld = np.empty_like(pv)
    for d in range(world.days):
        for j, s in enumerate(sids):
            sd = world.site_day(s, d)
            pv[d * DAY_STEPS:(d + 1) * DAY_STEPS, j] = sd.pv_kw
            ld[d * DAY_STEPS:(d + 1) * DAY_STEPS, j] = sd.load_kw
    prices = np.concatenate([np.asarray(world.prices(d), float) for d in range(world.days)])
    out, secs = {}, {}
    for pol in policies:
        t0 = time.perf_counter()
        sink = _ScheduleSink()
        run_policy(world, comps, LoopConfig(policy=pol, run_id=f"voltage:{pol}", write_inputs=False,
                                            step_predictions=False, write_schedules=True), sink=sink)
        batt = np.zeros_like(pv)
        for j, s in enumerate(sids):
            if world.sites[s].battery is not None:
                batt[:, j] = [sink.batt.get((s, int(t)), 0.0) for t in ts]
        out[pol] = pv - ld - batt
        secs[pol] = time.perf_counter() - t0
    by_feeder = {f: list(world.feeders[f].site_ids) for f in world.feeders}
    col = {s: j for j, s in enumerate(sids)}
    mol = {pol: float(sum(15 * (x[:, [col[s] for s in by_feeder[f]]].sum(1) > world.feeders[f].spec.g_kw + 1e-6).sum()
                          for f in by_feeder if by_feeder[f])) for pol, x in out.items()}
    return Replay(ts, sids, pv, out, world.label, hero_feeder(world), by_feeder, prices, secs, mol)


# ============================================================================================ prices
def pzu_lei_kwh(ts: np.ndarray, fallback_lei_mwh: Optional[np.ndarray] = None) -> tuple[np.ndarray, str]:
    """OPCOM PZU price (lei/kWh) at each 15-min ts. The one place G2 reads prices: switch to wattsentry.data.prices
    (card #158 X1) here once it lands. Hourly rows cover their four quarters. Missing -> the world's own prices."""
    ts = np.asarray(ts, np.int64)
    try:
        import polars as pl

        p = pl.read_parquet(PRICES_PARQUET).filter(pl.col("market") == "opcom_pzu")
        t = (p["ts"].dt.epoch("s")).to_numpy().astype(np.int64)
        v = p["price_lei_mwh"].to_numpy().astype(float)
        vals = dict(zip(t.tolist(), v.tolist()))
        out = np.array([vals.get(int(x), vals.get(int(x) - int(x) % 3600, np.nan)) for x in ts], float)
        if np.isnan(out).any():
            raise ValueError(f"{int(np.isnan(out).sum())} steps without an OPCOM price")
        return out / 1000.0, f"{LABEL_PRICES}, data/raw/prices.parquet"
    except Exception as e:  # noqa: BLE001 - reported in the source label
        if fallback_lei_mwh is None:
            raise
        return np.asarray(fallback_lei_mwh, float) / 1000.0, f"world prices ({type(e).__name__}: {e})"


# ============================================================================================ assessment
@dataclass
class PolicyVoltage:
    v_max: np.ndarray  # (T,) max voltage over all monitored buses / phases
    home_v: np.ndarray  # (T, 55) each home's own connection-point voltage
    n_full_pf: int  # steps where the estimate was replaced by the full power flow (near the limit)


def assess(fd: Feeder, sens: Sensitivity, p_kw: np.ndarray, near: float = NEAR_LIMIT, chunk: int = 4096,
           full_pf: bool = True) -> PolicyVoltage:
    """Estimate every step, then run the full power flow on the steps where the estimated max voltage, or any home's
    own voltage, is within `near` of the limit (those decide the violation count and who trips) and use it there."""
    T = len(p_kw)
    v_max = np.empty(T)
    home_v = np.empty((T, N_LOADS))
    band = near * V_LIMIT
    close = np.zeros(T, bool)
    for a in range(0, T, chunk):
        v = sens.estimate(p_kw[a:a + chunk])
        v_max[a:a + chunk] = v.max(axis=1)
        home_v[a:a + chunk] = v[:, fd.home_rows]
        close[a:a + chunk] = (np.abs(v_max[a:a + chunk] - V_LIMIT) <= band) | \
            (np.abs(home_v[a:a + chunk] - V_LIMIT) <= band).any(axis=1)
    idx = np.flatnonzero(close) if full_pf else np.array([], int)
    for t in idx:
        v = power_flow(fd, p_kw[t])
        v_max[t] = v.max()
        home_v[t] = v[fd.home_rows]
    return PolicyVoltage(v_max, home_v, len(idx))


def trips(pv_kw: np.ndarray, pol: PolicyVoltage, price_lei_kwh: np.ndarray) -> dict[str, float]:
    """Counts for one policy. pv_kw (T, 55) PV of each home on the feeder; a home trips in a step when its own
    connection point is above the limit."""
    viol = pol.v_max > V_LIMIT
    tripped = pol.home_v > V_LIMIT
    lost = np.where(tripped, np.maximum(pv_kw, 0.0), 0.0) * DT_H  # kWh per home-step
    return {
        "voltage_violations": int(viol.sum()),
        "voltage_violation_minutes": float(15 * viol.sum()),
        "home_steps_tripped": int(tripped.sum()),
        "inverter_trip_kwh": float(lost.sum()),
        "inverter_trip_lei": float((lost.sum(axis=1) * price_lei_kwh).sum()),
        "pv_kwh_in_violation_steps": float((np.maximum(pv_kw, 0.0)[viol] * DT_H).sum()),
        "v_max": float(pol.v_max.max()),
        "v_p99_daytime": float(np.percentile(pol.v_max[pv_kw.sum(1) > 0], 99)) if (pv_kw.sum(1) > 0).any() else 0.0,
        "n_full_pf": pol.n_full_pf,
    }


def validate(fd: Feeder, sens: Sensitivity, p_by_pol: dict[str, np.ndarray], frac: float = VALIDATE_FRAC,
             seed: int = 0) -> dict[str, float]:
    """Full power flow on a random `frac` of the steps of every policy vs the linear estimate (all buses / phases)."""
    rng = np.random.default_rng(seed)
    err, err_home, err_max, err_near, n, n_near = 0.0, 0.0, 0.0, 0.0, 0, 0
    for pol, p in p_by_pol.items():
        k = max(1, int(round(frac * len(p))))
        for t in rng.choice(len(p), size=min(k, len(p)), replace=False):
            v = power_flow(fd, p[t])
            e = sens.estimate(p[t][None, :])[0]
            err = max(err, float(np.abs(e - v).max()))
            err_home = max(err_home, float(np.abs(e[fd.home_rows] - v[fd.home_rows]).max()))
            err_max = max(err_max, abs(float(e.max() - v.max())))
            if abs(v.max() - V_LIMIT) <= 0.05 * V_LIMIT:  # the steps where the count could flip
                err_near = max(err_near, abs(float(e.max() - v.max())))
                n_near += 1
            n += 1
    return {"est_max_err_v": err, "est_max_err_pu": err / V_NOMINAL, "est_max_err_home_v": err_home,
            "est_max_err_vmax_v": err_max, "est_max_err_vmax_near_v": err_near, "n_validated": n,
            "n_validated_near": n_near, "est_err_within_band": int(err_max < NEAR_LIMIT * V_LIMIT)}


# ============================================================================================ run
def run(world: Any, v_sources: Sequence[float] = (1.03,), policies: Sequence[str] = ("baseline", "sentinel"),
        out_dir: Optional[Path] = None, rep: Optional[Replay] = None, validate_frac: float = VALIDATE_FRAC,
        seed: int = 0) -> dict[str, Any]:
    t_all = time.perf_counter()
    rep = rep or home_net_kw(world, policies)
    homes = map_homes(rep.by_feeder, rep.hero)
    col = {s: j for j, s in enumerate(rep.site_ids)}
    cols = np.array([col[s] for s in homes])
    price, price_src = pzu_lei_kwh(rep.ts, rep.world_prices_lei_mwh)
    pv = rep.pv_kw[:, cols]
    p_by_pol = {pol: rep.net_kw[pol][:, cols] for pol in policies}
    label = f"topology {LABEL_TOPOLOGY}; homes {rep.label}; prices {price_src}"
    res: dict[str, Any] = {"label": label, "homes": homes, "n_unique_homes": len(set(homes)),
                           "hero_feeder": rep.hero, "steps": len(rep.ts), "v_limit": V_LIMIT,
                           "replay_seconds": rep.seconds, "world_minutes_over_limit": rep.minutes_over_limit,
                           "by_v_source": {}}
    for vs in v_sources:
        fd = build_feeder(vs)
        sens = sensitivity(fd)
        val = validate(fd, sens, p_by_pol, frac=validate_frac, seed=seed)
        r: dict[str, Any] = {"v_source_pu": vs, "v_busbar_noload_v": round(vs * V_BASE, 1),
                             "sensitivity_seconds": round(sens.seconds, 2), **val}
        for pol in policies:
            t0 = time.perf_counter()
            r[pol] = trips(pv, assess(fd, sens, p_by_pol[pol]), price)
            r[pol]["seconds"] = round(time.perf_counter() - t0, 2)
        res["by_v_source"][f"{vs:.2f}"] = r
    res["seconds"] = round(time.perf_counter() - t_all, 1)
    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "voltage.json").write_text(json.dumps(res, indent=2, default=float))
    return res


def print_metrics(res: dict[str, Any], primary: float) -> None:
    print(f"# G2 voltage: {res['label']}; {res['steps']} steps, {res['n_unique_homes']} distinct homes on "
          f"{N_LOADS} IEEE loads (hero {res['hero_feeder']} first); limit {V_LIMIT:.0f} V")
    for pol, v in res.get("world_minutes_over_limit", {}).items():
        print(f"LAB_METRIC world_minutes_over_limit_{pol}={v:.4g}")  # same replay as replay.py: ties G2 to it
    for key, r in res["by_v_source"].items():
        suffix = "" if abs(r["v_source_pu"] - primary) < 1e-9 else f"_vs{key.replace('.', '')}"
        print(f"# busbar {r['v_source_pu']:.2f} pu = {r['v_busbar_noload_v']} V no-load; estimate vs full PF on "
              f"{r['n_validated']} steps: max |err| {r['est_max_err_v']:.3f} V ({r['est_max_err_pu']:.5f} pu)")
        out = {k: r[k] for k in ("est_max_err_v", "est_max_err_pu", "est_max_err_vmax_near_v", "n_validated",
                                 "est_err_within_band")}
        for pol in ("baseline", "sentinel"):
            if pol in r:
                for k in ("voltage_violations", "inverter_trip_kwh", "inverter_trip_lei", "home_steps_tripped",
                          "pv_kwh_in_violation_steps", "v_max", "n_full_pf"):
                    out[f"{k}_{pol}"] = r[pol][k]
        b, s = r.get("baseline"), r.get("sentinel")
        if b and s and b["voltage_violations"]:
            out["voltage_violations_reduction"] = 1 - s["voltage_violations"] / b["voltage_violations"]
        if b and s and b["inverter_trip_kwh"] > 0:
            out["inverter_trip_kwh_reduction"] = 1 - s["inverter_trip_kwh"] / b["inverter_trip_kwh"]
        for k, v in out.items():
            print(f"LAB_METRIC {k}{suffix}={v:.4g}")


def main(argv: Optional[Sequence[str]] = None) -> dict[str, Any]:
    import argparse

    ap = argparse.ArgumentParser(description="G2: voltage / inverter trips on the IEEE European LV feeder")
    ap.add_argument("--world", default="sim", choices=["sim", "mock"])
    ap.add_argument("--days", type=int, default=7)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--start", default=None, help="first local day YYYY-MM-DD (default: the world's, replay#506 week)")
    ap.add_argument("--hero", default=None, help="sim: fleet id of the hero feeder (replay#506: F06); keeps the same "
                                                 "5 feeders / 118 homes when --start moves")
    ap.add_argument("--v-source", default="1.03,1.00,1.05",
                    help="no-load busbar settings in pu of 230.9 V; the first one is the headline")
    ap.add_argument("--validate-frac", type=float, default=VALIDATE_FRAC)
    ap.add_argument("--out", default=os.environ.get("LAB_OUTPUT_DIR", ".lab-out/voltage"))
    a = ap.parse_args(argv)
    from wattsentry.agent.world import make_world

    kw: dict[str, Any] = {"wind": False} if a.world == "sim" else {"with_wind": False}
    if a.start:  # local midnight in Bucharest (UTC+2 in winter / +3 in summer), as the worlds' DEFAULT_START
        from datetime import datetime, timedelta, timezone

        d = datetime.fromisoformat(a.start).replace(tzinfo=timezone.utc)
        kw["start"] = d - timedelta(hours=3 if 4 <= d.month <= 10 else 2)
    if a.hero and a.world == "sim":
        kw["hero"] = a.hero
    world = make_world(a.world, days=a.days, seed=a.seed, **kw)
    vs = [float(x) for x in a.v_source.split(",") if x]
    res = run(world, vs, out_dir=Path(a.out), validate_frac=a.validate_frac, seed=a.seed)
    res["run_id"] = os.environ.get("LAB_RUN_ID", "local")
    print_metrics(res, vs[0])
    print(f"# G2 voltage: {res['seconds']}s total (replay {json.dumps({k: round(v, 1) for k, v in res['replay_seconds'].items()})})",
          file=sys.stderr)
    return res


if __name__ == "__main__":
    main()
