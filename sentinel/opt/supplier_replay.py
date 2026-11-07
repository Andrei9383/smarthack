"""Replay with both objectives (card G4 #162, spec G step 6): does adding the supplier's cost to the feeder LP
increase the minutes over the feeder limit?

Same world, same forecasts, same prices for every variant; only the planner's objective differs:

    baseline   the self-consumption rule (no plan), for reference
    dso        the #18 LP as in replay#506 (`plan_feeder`, supplier=None)
    sup        + the supplier term (`SupplierObjective`, weight 1, bill kept), lexicographic hard feeder limit
    sup_only   + the supplier term with the prosumer-bill term off (bill_weight 0; stage 1 keeps the bill, #188)
    sup_noguard as `sup` without the near-limit guard (slack bound only; replay#882's `sup`)
    sup_nolex  as `sup` but one solve (limit kept only through M >> supplier weights)

The supplier's day-ahead position per feeder-day = I5's purchase (`sentinel.eval.portfolio.forecast_day(...).bought`,
quantile a/(a+b) of 200 coherent scenarios under the self-consumption behaviour) on the feeder's homes, from the same
P10/50/90 the planner gets; until I5 (#175) is importable a stand-in: the rule rolled over the P50s from a warm
(steady-state) SoC. The position never depends on the policy's battery state (the supplier cannot see it), so every
variant is scored against the same purchase. Realised supplier cost = X1's `avoidable_cost` (#158) on the realised
feeder net import vs that position at the day's PZU, no intraday trade (traded_id = 0), a, b frozen: per feeder and
for the whole portfolio (all replay homes = one supplier; deviations net across feeders).

Run on the lab (simulator window cache):  python -m sentinel.opt.supplier_replay --world sim --days 7
Labels: homes, batteries, feeders = synthetic RO; PZU = real RO (OPCOM); imbalance price = proxy (PZU x (1 +- a|b)).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import replace
from typing import Any, Callable, Mapping, Optional, Sequence

import numpy as np

from sentinel.contracts import self_consumption_step

from .lp import KW_TO_MWH, LPParams, SupplierObjective, imbalance_ab, plan_feeder, supplier_cost_lei
from .policies import battery_from_row, price_vector

VARIANTS = {
    "dso": None,
    "sup": dict(weight=1.0, params=dict()),
    "sup_only": dict(weight=1.0, params=dict(bill_weight=0.0)),
    "sup_noguard": dict(weight=1.0, params=dict(supplier_guard_margin=None)),
    "sup_nolex": dict(weight=1.0, params=dict(supplier_lexicographic=False)),
}


# ------------------------------------------------------------------------------------------- supplier position
def _q(p: Any) -> np.ndarray:
    return np.stack([np.asarray(p.q10, float), np.asarray(p.q50, float), np.asarray(p.q90, float)], axis=-1)


def _rule_position_kw(rows: Sequence[Mapping[str, Any]], fc: Mapping[str, Mapping[str, Any]]) -> np.ndarray:
    """Stand-in for I5: feeder net import (kW) of the self-consumption rule on the P50s, warm SoC (day rolled twice)."""
    total = None
    for r in rows:
        pv, ld = np.asarray(fc[r["site_id"]]["pv"].q50, float), np.asarray(fc[r["site_id"]]["load"].q50, float)
        b = battery_from_row(r)
        if b is None:
            imp = ld - pv
        else:
            e = b.soc_min * b.capacity_kwh
            for _ in range(2):  # first pass = warm-up to the steady-state midnight SoC
                imp = np.zeros(len(pv))
                for t in range(len(pv)):
                    _, e, g = self_consumption_step(b, e, float(pv[t]), float(ld[t]))
                    imp[t] = -g
        total = imp if total is None else total + imp
    return np.zeros(96) if total is None else total


def supplier_position_kw(rows: Sequence[Mapping[str, Any]], fc: Mapping[str, Mapping[str, Any]],
                         seed: int = 0) -> tuple[np.ndarray, str]:
    """THE place the supplier's day-ahead purchase for a feeder is read: (kW net import per step, source)."""
    try:
        from sentinel.eval import portfolio as pf
    except ImportError:
        return _rule_position_kw(rows, fc), "stand-in: self-consumption rule on P50, warm SoC (I5 #175 not on main)"
    ids = [r["site_id"] for r in rows]
    fleet = pf.Fleet.from_sites(list(rows))
    day = pf.forecast_day(np.stack([_q(fc[s]["pv"]) for s in ids]), np.stack([_q(fc[s]["load"]) for s in ids]),
                          fleet, seed=seed)
    return np.asarray(day.bought, float) / KW_TO_MWH, f"I5 forecast_day bought (q*={day.q_star:.3f}, 200 scen)"


# ------------------------------------------------------------------------------------------- planner wrapper
def make_planner(world: Any, positions: dict, sources: dict, plans: dict, weight: Optional[float],
                 params: LPParams, seed: int = 0) -> Callable[..., Any]:
    """`comps.plan_feeder` replacement: same signature; records the supplier position (same for every variant) and
    plans with (weight) or without (None) the supplier term."""

    def plan(spec, eligible, rest, fc, G=None, prices=None, *, statuses=None, t0=None, **kw):
        key = (spec.feeder_id, t0)
        if key not in positions:
            members = [world.sites[s].row for s in world.feeders[spec.feeder_id].site_ids]
            positions[key], src = supplier_position_kw(members, fc, seed)
            sources[src] = sources.get(src, 0) + 1
        if weight is None:
            pl = plan_feeder(spec, eligible, rest, fc, G, prices, statuses=statuses, t0=t0, **kw)
        else:
            pl = plan_feeder(spec, eligible, rest, fc, G, prices, statuses=statuses, t0=t0, params=params,
                             supplier=SupplierObjective(positions[key], weight=weight), **kw)
        _, ws, wl = SupplierObjective(positions[key], weight=1.0).arrays(pl.horizon, price_vector(prices, pl.horizon))
        p_sum = sum((np.asarray(x, float) for x in pl.schedule.values()), np.zeros(pl.horizon))
        plans[key] = (supplier_cost_lei(-np.asarray(pl.export_curve, float), positions[key], ws, wl)[0],
                      {"p": p_sum, "nobatt_worst": np.asarray(pl.worst_export_curve, float) + p_sum, "g": pl.g_kw,
                       "e0": float(sum(v[0] for v in pl.soc_kwh.values())) if pl.soc_kwh else 0.0})
        return pl

    return plan


# ------------------------------------------------------------------------------------------- scoring
def realised_net_import(sink: Any, feeder_ids: Sequence[str], days: int, steps: int = 96) -> dict[str, np.ndarray]:
    """Feeder net import kW [days*steps] from the loop's feeder_curves rows (written in day, step order)."""
    out: dict[str, list[float]] = {f: [] for f in feeder_ids}
    for r in sink.rows.get("feeder_curves", []):
        out[r["feeder_id"]].append(float(r["import_kw"]) - float(r["export_kw"]))
    arr = {f: np.asarray(v, float) for f, v in out.items()}
    for f, v in arr.items():
        if len(v) != days * steps:
            raise RuntimeError(f"{f}: {len(v)} feeder_curves rows, expected {days * steps}")
    return arr


def _avoidable(actual_mwh, bought_mwh, pzu_lei_mwh, a: float, b: float) -> dict[str, float]:
    try:
        from sentinel.eval.imbalance import avoidable_cost
    except ImportError:  # X1 #158 not on main: the same formula (no intraday trade)
        w = np.abs(pzu_lei_mwh)
        lei, _ = supplier_cost_lei(actual_mwh, bought_mwh, a * w, b * w, dt_h=1.0)
        dev = np.asarray(actual_mwh) - np.asarray(bought_mwh)
        return {"avoidable_lei": lei, "abs_dev_mwh": float(np.abs(dev).sum()),
                "short_mwh": float(np.clip(dev, 0, None).sum()), "long_mwh": float(np.clip(-dev, 0, None).sum())}
    return avoidable_cost(actual_mwh, bought_mwh, None, pzu_lei_mwh, None, a=a, b=b)


def score(world: Any, net_imp: Mapping[str, np.ndarray], positions: Mapping, a: float, b: float) -> dict[str, float]:
    days, fids = world.days, list(world.feeders)
    pzu = np.concatenate([np.asarray(world.prices(d), float) for d in range(days)])  # lei/MWh
    bought = {f: np.concatenate([positions[(f, world.day_t0(d))] for d in range(days)]) * KW_TO_MWH for f in fids}
    actual = {f: net_imp[f] * KW_TO_MWH for f in fids}
    per = [_avoidable(actual[f], bought[f], pzu, a, b) for f in fids]
    port = _avoidable(sum(actual.values()), sum(bought.values()), pzu, a, b)
    inj = float(sum(np.clip(-v, 0, None).sum() for v in actual.values()))  # feeder-level net export, MWh
    return {"supplier_avoidable_lei": port["avoidable_lei"], "supplier_abs_dev_mwh": port["abs_dev_mwh"],
            "supplier_short_mwh": port["short_mwh"], "supplier_long_mwh": port["long_mwh"],
            "supplier_avoidable_lei_feeders": float(sum(p["avoidable_lei"] for p in per)),
            "supplier_abs_dev_mwh_feeders": float(sum(p["abs_dev_mwh"] for p in per)),
            "feeder_net_export_mwh": inj}


# ------------------------------------------------------------------------------------------- driver
def run(world_kind: str = "sim", days: int = 7, seed: int = 0, variants: Sequence[str] = ("baseline", "dso", "sup"),
        scale: float = 1.0) -> dict[str, Any]:
    from sentinel.agent.loop import LoopConfig, _MemSink, default_components, run_policy
    from sentinel.agent.world import make_world

    w = make_world(world_kind, days=days, seed=seed, **({"scale": scale} if world_kind == "mock" else {}))
    comps = default_components(w)
    if not comps.sources.get("planner", "").startswith("real"):
        raise RuntimeError(f"planner is {comps.sources.get('planner')}: the LP must import")
    a, b = imbalance_ab()
    positions: dict = {}
    diag: dict = {}
    sources: dict[str, int] = {}
    order = [v for v in variants if v != "baseline"] + (["baseline"] if "baseline" in variants else [])
    out: dict[str, Any] = {"world": w.describe(), "label": w.label, "a": a, "b": b, "variants": {}}
    for v in order:  # sentinel variants first: they record the positions the baseline is scored against
        t0 = time.perf_counter()
        plans: dict = {}
        if v == "baseline":
            pol, c = "baseline", comps
        else:
            spec = VARIANTS[v]
            params = LPParams(**(spec or {}).get("params", {}))
            c = replace(comps, errors={}, plan_feeder=make_planner(w, positions, sources, plans,
                                                                   None if spec is None else spec["weight"], params,
                                                                   seed))
            pol = "sentinel"
        sink = _MemSink()
        cfg = LoopConfig(policy=pol, run_id=f"g4:{v}", seed=seed, write_inputs=False, step_predictions=False,
                         write_schedules=False)
        r = run_policy(w, c, cfg, sink=sink)
        m = {"minutes_over_limit": r.minutes_over_limit, "energy_over_limit_kwh": r.energy_over_limit_kwh,
             "peak_overshoot_kw": r.peak_overshoot_kw, "comfort_breaches": float(r.comfort_breaches),
             "lei_cost_per_home_month": r.cost_lei / max(1, r.n_homes) * 30 / max(1, w.days),
             "plan_status": dict(r.plan_status), "planner_errors": dict(c.errors), "seconds": 0.0}
        if r.p10_hit_rate is not None:
            m["p10_hit_rate"] = r.p10_hit_rate
        m["feeder_days"] = {f"{f}_d{d}": [x["minutes_over"], round(x.get("kwh_over", 0.0), 3)]
                            for (f, d), x in sorted(r.feeder_days.items())}
        net = realised_net_import(sink, list(w.feeders), w.days)
        m.update(score(w, net, positions, a, b))
        if plans:
            m["supplier_planned_lei"] = float(sum(c for c, _ in plans.values()))  # P50 plan vs position, all alike
        diag[v] = {"plans": {k: i for k, (_, i) in plans.items()}, "net": net}
        m["seconds"] = time.perf_counter() - t0
        out["variants"][v] = m
        print(f"[g4] {v}: {json.dumps({k: (round(x, 4) if isinstance(x, float) else x) for k, x in m.items()})}",
              file=sys.stderr, flush=True)
    out["position_source"] = sources
    out["diagnose"] = _diagnose(w, out["variants"], diag)
    return out


def _diagnose(w: Any, var: Mapping[str, Any], diag: Mapping[str, Any]) -> dict:
    """Steps where a supplier variant is over the limit and dso is not: realised export, no-battery worst export
    (guard zone if >= 0.75 G), planned feeder battery kW and day-start pool energy, dso vs variant."""
    out: dict = {}
    if "dso" not in diag:
        return out
    for v in diag:
        if not v.startswith("sup"):
            continue
        for (f, d), x in ((k, var[v]["feeder_days"][f"{k[0]}_d{k[1]}"]) for k in
                          ((f, d) for f in w.feeders for d in range(w.days))):
            key = (f, w.day_t0(d))
            pd, pv = diag["dso"]["plans"].get(key), diag[v]["plans"].get(key)
            if pd is None or pv is None:
                continue
            g = pd["g"]
            xd = -diag["dso"]["net"][f][d * 96:(d + 1) * 96]
            xv = -diag[v]["net"][f][d * 96:(d + 1) * 96]
            ks = [k for k in range(96) if xv[k] > g + 1e-6 and xv[k] > xd[k] + 1e-6]
            if ks:
                out[f"{v}:{f}_d{d}"] = {"g_kw": g, "e0_dso": round(pd["e0"], 2), "e0_var": round(pv["e0"], 2),
                                        "steps": [{"k": k, "x_dso": round(xd[k], 2), "x_var": round(xv[k], 2),
                                                   "nobatt_worst": round(pv["nobatt_worst"][k], 2),
                                                   "guard": bool(pv["nobatt_worst"][k] >= 0.75 * g),
                                                   "p_dso": round(pd["p"][k], 2), "p_var": round(pv["p"][k], 2)}
                                                  for k in ks]}
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--world", default="sim", choices=["mock", "sim"])
    ap.add_argument("--days", type=int, default=7)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--variants", default="baseline,dso,sup,sup_only,sup_noguard,sup_nolex")
    a = ap.parse_args(argv)
    res = run(a.world, a.days, a.seed, [v for v in a.variants.split(",") if v], a.scale)
    run_id = os.environ.get("LAB_RUN_ID")
    res["lab_run"] = run_id
    od = os.environ.get("LAB_OUTPUT_DIR")
    if od:
        with open(os.path.join(od, "supplier_replay.json"), "w") as fh:
            json.dump(res, fh, indent=1, default=str)
    print(f"# diagnose: {json.dumps(res['diagnose'], default=float)}")
    print(f"# label: {res['label']} homes/feeders; PZU real RO; imbalance price proxy (a={res['a']}, b={res['b']}); "
          f"position: {res['position_source']}")
    keys = ("minutes_over_limit", "energy_over_limit_kwh", "peak_overshoot_kw", "comfort_breaches",
            "lei_cost_per_home_month", "p10_hit_rate", "supplier_avoidable_lei", "supplier_abs_dev_mwh",
            "supplier_avoidable_lei_feeders", "supplier_planned_lei")
    for v, m in res["variants"].items():
        for k in keys:
            if k in m:
                print(f"LAB_METRIC {k}_{v}={m[k]:.6g}")
    var = res["variants"]
    if "dso" in var:
        for v in var:
            if v.startswith("sup"):
                d = var[v]["minutes_over_limit"] - var["dso"]["minutes_over_limit"]
                print(f"LAB_METRIC minutes_over_limit_delta_{v}={d:.6g}")
                print(f"LAB_METRIC minutes_not_increased_{v}={int(d <= 0)}")
                de = var[v]["energy_over_limit_kwh"] - var["dso"]["energy_over_limit_kwh"]
                print(f"LAB_METRIC energy_over_limit_delta_{v}={de:.6g}")
                worse = {k: (var["dso"]["feeder_days"][k], x) for k, x in var[v]["feeder_days"].items()
                         if x[0] > var["dso"]["feeder_days"][k][0] or x[1] > var["dso"]["feeder_days"][k][1] + 1e-3}
                print(f"# {v}: feeder-days worse than dso [min, kWh] (dso, {v}): {json.dumps(worse)}")
                print(f"LAB_METRIC comfort_breaches_delta_{v}="
                      f"{var[v]['comfort_breaches'] - var['dso']['comfort_breaches']:.6g}")
                base = var["dso"]["supplier_avoidable_lei"]
                if base:
                    print(f"LAB_METRIC supplier_cost_change_{v}={(var[v]['supplier_avoidable_lei'] - base) / base:.4g}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
