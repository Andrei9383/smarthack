"""Imbalance replay on data-v2: MaxControl sensitivity (card #184, spec I step 8, follow-up of I6 #176).

    python replay.py --mode imbalance --data v2 --share-maxcontrol 0.25 --out $LAB_OUTPUT_DIR/imbalance_v2_report.json

One run = one data-v2 world (I2 #172) generated with ``share_maxcontrol_of_battery = s`` (the rest of the config is
I2's default), then:

1. I4 #174 battery model trained and scored on it (``wattsentry.model.battery.train.run``: train homes 2022-23,
   validation 2024, unseen homes 2025). It emits strategy_f1 (macro, unseen homes x test days) and its nMAE metrics.
2. The portfolio = the unseen (never trained on) **smart-meter** homes. Money is reported on smart homes only:
   profile-meter homes are settled on the standard profile, so no policy can change their bill (spec step 3).
3. Day-ahead position per 15-min step of local 2025, summed over the portfolio (MWh, + = net purchase):

| policy          | position                                                                                    |
|-----------------|---------------------------------------------------------------------------------------------|
| `profile`       | each home's same-month-last-year metered import / export spread on the standard profile    |
|                 | (load shape for import, solar shape for export) = PPC's practice today                      |
| `persistence`   | the home's meter, same step a week earlier (I4 ``persist``)                                 |
| `forecast_only` | load - PV expectation, no battery (I4 ``fo``)                                               |
| `sentinel`      | I4 two-stage P50: guessed strategy + parameters through the K8 kernel (I4's ``sim_mode``)   |
| `sentinel_lgbm` | I4 LightGBM quantile P50 on net import (I4's fallback model)                                 |

4. actual = the homes' true 15-min import - export (= the smart meter curve PPC is settled on), settled with
   ``wattsentry.eval.imbalance.cost_steps`` (X1 frozen a, b) at the world's PZU (corrected OPCOM timestamps, #186).

Not the #176 headline set-up: there the fleet has PV/load telemetry, PVNet/LoadNet bands, K scenarios, conformal
and a q* purchase. data-v2 homes have only meters, so ``sentinel`` here is I4's P50 point forecast (no q*, no
conformal). M1/M2 on data-v2 = proxy (physics PV on recorded metadata + profile load scaled to last month's import).

Labels: homes, batteries, strategies = synthetic RO (data-v2); ERA5 weather + OPCOM PZU = real RO; a, b = proxy;
standard profile = proxy (I2 built-in stand-in); share_maxcontrol = synthetic (swept).
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd

POLICIES = ("profile", "persistence", "forecast_only", "sentinel", "sentinel_lgbm")
HEADLINE = "sentinel"
I4_KEY = {"persistence": "persist", "forecast_only": "fo", "sentinel_lgbm": "lgbm"}  # sentinel -> I4 sim_mode


def solar_coef(world: Any) -> np.ndarray:
    """Standard-profile solar shape on the world's axis (World only carries load_coef)."""
    sp = pd.read_parquet(Path(world.out) / "standard_profile.parquet", columns=["ts", "solar_coef"]).sort_values("ts")
    idx = pd.DatetimeIndex(sp["ts"]).tz_convert("UTC").as_unit("ns")
    s = pd.Series(sp["solar_coef"].to_numpy(np.float64), index=idx)
    return s.reindex(world.ts.as_unit("ns")).fillna(0.0).to_numpy(np.float64)


def profile_net_kwh(world: Any, imp_m: np.ndarray, exp_m: np.ndarray, steps: np.ndarray,
                    sol: np.ndarray, lag_months: int = 12) -> np.ndarray:
    """Profile position (n, len(steps)) kWh: month m's import / export expected = the home's metered total of month
    m - lag_months (known since month m - lag_months + 1), spread over month m by the standard profile. Homes with
    no total that month fall back to the mean over homes that have one."""
    from wattsentry.sim.prosumers_v2.metering import settle_on_profile

    n_m = imp_m.shape[1]
    midx = world.day_month[world.cal["day_index"]]  # month index per step on the full axis

    def shifted(m: np.ndarray) -> np.ndarray:
        out = np.full_like(m, np.nan)
        out[:, lag_months:] = m[:, :n_m - lag_months]
        col_mean = np.nanmean(np.where(np.isfinite(out), out, np.nan), axis=0) if out.size else out
        return np.where(np.isfinite(out), out, np.nan_to_num(col_mean)[None, :])

    imp = settle_on_profile(np.nan_to_num(shifted(imp_m)), world.load_coef, midx)
    exp = settle_on_profile(np.nan_to_num(shifted(exp_m)), sol, midx)
    return (imp - exp)[:, steps]


def portfolio_positions(world: Any, homes: pd.DataFrame, years: Sequence[int], clf: Any, netq: Any, sim_mode: str,
                        chunk: int = 250, log=print) -> dict[str, Any]:
    """Sum over ``homes`` of every policy's day-ahead net import + the true import / export (kWh per step)."""
    from wattsentry.model.battery import data as D
    from wattsentry.model.battery.train import Chunk, forecast_chunk

    steps = world.steps_of_years(years)
    sol = solar_coef(world)
    acc = {p: np.zeros(len(steps)) for p in POLICIES}
    imp_t, exp_t = np.zeros(len(steps)), np.zeros(len(steps))
    t0 = time.perf_counter()
    for ci, hc in enumerate(D.chunks(homes["home_id"], chunk)):
        hdf = homes[homes.home_id.isin(hc)].reset_index(drop=True)
        ch = Chunk(world, hdf)
        fc = forecast_chunk(world, ch, years, clf, netq)
        ti, te = D.truth_chunk(world, hdf.home_id.tolist(), years)
        imp_t += np.nan_to_num(ti[:, steps]).sum(0)
        exp_t += np.nan_to_num(te[:, steps]).sum(0)
        acc["profile"] += profile_net_kwh(world, ch.imp_m, ch.exp_m, steps, sol).sum(0)
        for pol in POLICIES[1:]:
            key = sim_mode if pol == "sentinel" else I4_KEY[pol]
            acc[pol] += np.nan_to_num(np.asarray(fc[key], np.float64)).sum(0)
        log(f"[imbalance-v2] chunk {ci}: {len(hdf)} homes, {time.perf_counter() - t0:.0f}s", flush=True)
        del ch, fc
    return {"steps": steps, "ts": world.ts[steps], "pzu": world.pzu[steps], "pos_kwh": acc,
            "import_kwh": imp_t, "export_kwh": exp_t}


def _emit(name: str, value: Optional[float]) -> None:
    from wattsentry.infra.metrics import emit

    if value is not None and np.isfinite(value):
        emit(name, float(value))


def run(args: argparse.Namespace, log=print) -> dict[str, Any]:
    from wattsentry.eval.imbalance import load_ab
    from wattsentry.eval.imbalance_replay import evaluate, reduction_vs
    from wattsentry.model.battery import data as D
    from wattsentry.model.battery import train as BT
    from wattsentry.model.battery.api import BatteryModel
    from wattsentry.sim.prosumers_v2 import ProsumerConfig, generate

    t_all = time.perf_counter()
    out = Path(args.out)
    out_dir = out.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    run_id = args.run_id or os.environ.get("LAB_RUN_ID") or "local"
    timings: dict[str, float] = {}

    t = time.perf_counter()
    if args.data_dir:
        data_dir = Path(args.data_dir)
    else:
        cfg_kw: dict[str, Any] = {"share_maxcontrol_of_battery": args.share_maxcontrol}
        if args.v2_homes:
            cfg_kw["n_homes"] = args.v2_homes
        data_dir = Path(generate(ProsumerConfig(**cfg_kw)))
    timings["t_generate_s"] = time.perf_counter() - t
    manifest = json.loads((data_dir / "manifest.json").read_text()) if (data_dir / "manifest.json").exists() else {}
    log(f"[imbalance-v2] data-v2 {data_dir} (share_maxcontrol_of_battery={args.share_maxcontrol})", flush=True)

    # ---- I4: train + score (emits strategy_f1, net_nmae, ...)
    t = time.perf_counter()
    i4_out = out_dir / "i4"
    bt_argv = ["--data-dir", str(data_dir), "--out", str(i4_out), "--seed", str(args.seed),
               "--chunk", str(args.chunk), "--rounds", str(args.rounds)]
    if args.max_homes:
        bt_argv += ["--max-homes", str(args.max_homes)]
    res = BT.run(bt_argv)
    timings["t_i4_s"] = time.perf_counter() - t
    bm = BatteryModel.load(i4_out / "battery")
    sim_mode = res.get("sim_mode", bm.mode)

    # ---- portfolio: unseen smart homes, test year
    t = time.perf_counter()
    world = D.load_world(data_dir)
    years = [int(y) for y in str(args.test_years).split(",")]
    homes = world.homes[(world.homes.split == "unseen") & (world.homes.meter == "smart")].reset_index(drop=True)
    if args.max_homes:
        used = pd.read_parquet(i4_out / "battery_guesses_test.parquet")["home_id"].astype(str)
        homes = homes[homes.home_id.isin(set(used))].reset_index(drop=True)
    lab = D.labels(data_dir, homes["home_id"])
    strat_counts = lab["strategy"].value_counts().to_dict()
    all_lab = D.labels(data_dir)
    pp = portfolio_positions(world, homes, years, bm.clf, bm.netq, sim_mode, chunk=args.chunk, log=log)
    timings["t_portfolio_s"] = time.perf_counter() - t

    ts = pd.DatetimeIndex(pp["ts"])
    actual = (pp["import_kwh"] - pp["export_kwh"]) / 1000.0
    exp_mwh = pp["export_kwh"] / 1000.0
    positions = {p: (pp["pos_kwh"][p] / 1000.0, None) for p in POLICIES}
    summary, steps = evaluate(ts, actual, exp_mwh, positions, pp["pzu"], None)
    extra = {p: {"avoidable_cost_reduction_vs_profile": reduction_vs(summary, p)} for p in summary}

    a, b = load_ab()
    n_pd = int(strat_counts.get("price_driven", 0))
    head = {
        "share_maxcontrol": args.share_maxcontrol,
        "share_price_driven_homes": float((all_lab["strategy"] == "price_driven").mean()),
        "share_price_driven_portfolio": n_pd / max(len(homes), 1),
        "share_battery_portfolio": float(lab["has_battery"].mean()) if len(lab) else float("nan"),
        "homes": len(homes), "homes_world": len(world.homes), "days": len(ts) / 96.0,
        "export_mwh": float(exp_mwh.sum()), "imbalance_a": a, "imbalance_b": b}
    for name in ("avoidable_cost_lei", "lei_per_mwh_injected", "abs_dev_mwh", "double_cover_mwh_18_21"):
        _emit(name, summary[HEADLINE][name])
    _emit("avoidable_cost_reduction_vs_profile", extra[HEADLINE]["avoidable_cost_reduction_vs_profile"])
    for pol in summary:
        for name in ("avoidable_cost_lei", "lei_per_mwh_injected", "abs_dev_mwh", "double_cover_mwh_18_21",
                     "abs_dev_mwh_17_21", "position_nmae", "bias_mwh"):
            _emit(f"{name}_{pol}", summary[pol][name])
        _emit(f"avoidable_cost_reduction_vs_profile_{pol}", extra[pol]["avoidable_cost_reduction_vs_profile"])
    for k, v in head.items():
        _emit(k, v)
    timings["t_total_s"] = time.perf_counter() - t_all
    for k, v in timings.items():
        _emit(k, v)

    pos_df = pd.DataFrame({"ts": ts.tz_convert("UTC"), "actual": actual, "export_mwh": exp_mwh, "pzu_lei_mwh": pp["pzu"],
                           **{f"bought_{p}": positions[p][0] for p in POLICIES},
                           **{f"avoidable_lei_{p}": steps[p]["avoidable"] for p in POLICIES}})
    pos_df.to_parquet(out_dir / "imbalance_v2_positions.parquet", index=False)
    report = {"card": 184, "run_id": run_id, "data_dir": str(data_dir), "share_maxcontrol": args.share_maxcontrol,
              "realised": manifest.get("realised_shares") or manifest.get("realised"), "portfolio_strategies":
              strat_counts, "summary": summary, "extra": extra, "head": head, "sim_mode": sim_mode,
              "strategy_f1": res.get("strategy_f1"), "strategy_f1_home": res.get("strategy_f1_home"),
              "i4_net_nmae": res.get("net_nmae"), "a": a, "b": b, "timings": timings,
              "labels": {"quantities": "synthetic RO (data-v2, I2 #172)", "pzu": "real RO (OPCOM, #186 timestamps)",
                         "imbalance_ab": "proxy (X1)", "standard_profile": "proxy (I2 built-in)",
                         "m1_m2": "proxy (I4 physics PV + profile load)", "share_maxcontrol": "synthetic (swept)"}}
    out.write_text(json.dumps(report, indent=1, default=float))
    log(f"[imbalance-v2] s={args.share_maxcontrol}: {len(homes)} unseen smart homes ({n_pd} price-driven), "
        f"{head['days']:.0f} days, strategy_f1={res.get('strategy_f1')}", flush=True)
    for pol, s in summary.items():
        log(f"[imbalance-v2] {pol:14s} avoidable {s['avoidable_cost_lei']:11.0f} lei  "
            f"{s['lei_per_mwh_injected']:7.1f} lei/MWh inj  abs_dev {s['abs_dev_mwh']:7.1f} MWh  "
            f"vs profile {100 * extra[pol]['avoidable_cost_reduction_vs_profile']:+.1f}%", flush=True)
    return report


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="replay.py --mode imbalance --data v2", description=__doc__.split("\n")[0])
    ap.add_argument("--data", default="v2", help="v2 (this mode)")
    ap.add_argument("--share-maxcontrol", type=float, default=0.25,
                    help="data-v2 share_maxcontrol_of_battery (synthetic; I2 default 0.25)")
    ap.add_argument("--v2-homes", type=int, default=0, help="homes to generate (0 = I2 default 5000)")
    ap.add_argument("--max-homes", type=int, default=0, help="I4 uses at most this many homes (0 = all)")
    ap.add_argument("--data-dir", default=None, help="an already generated data-v2 dir (skips generate)")
    ap.add_argument("--test-years", default="2025")
    ap.add_argument("--chunk", type=int, default=250)
    ap.add_argument("--rounds", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(os.environ.get("LAB_OUTPUT_DIR", "."), "imbalance_v2_report.json"),
                    help="report json; positions parquet and I4 models go next to it")
    ap.add_argument("--run-id", default=None)
    return ap


def main(argv: Optional[Sequence[str]] = None) -> dict[str, Any]:
    return run(parser().parse_args(argv))


if __name__ == "__main__":
    main()
