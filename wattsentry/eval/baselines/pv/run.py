"""PV / load / fault baselines on the synthetic RO fleet: the numbers M1, M2 and M4 have to beat.

    python -m wattsentry.eval.baselines.pv.run                 # small fleet: 100 sites, train 2023, val 2024
    python -m wattsentry.eval.baselines.pv.run --split test    # score on --test-year

Prints LAB_METRIC lines (val_nmae / val_pinball for PV, val_prauc / hours_to_detect for faults, plus prefixed
per-baseline metrics) and writes baselines_pv.json + the LightGBM bundles to --out (default $LAB_OUTPUT_DIR).
Day-ahead (96 x 15 min) forecasts from UTC midnight on the day's actual weather (nowcast mode); persistence and
the reference window only see data before the day. Thresholds are tuned on training days; models train on
`split == train` sites only.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

from . import lgbm
from .data import ensure_dataset, load_arrays
from .faults import run_fault_baselines
from .metrics import forecast_metrics
from .persistence import pv_ref7_bulk, pvlib_band, shift_days


def metric(name: str, value: float) -> None:
    if value is None or not np.isfinite(value):
        return
    print(f"LAB_METRIC {name}={value:.4f}", flush=True)


def main(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser(prog="python -m wattsentry.eval.baselines.pv.run", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, help="fault-injected fleet dir (default: generate / reuse cached)")
    ap.add_argument("--sites", type=int, default=100)
    ap.add_argument("--feeders", type=int, default=6)
    ap.add_argument("--start-year", type=int, default=2023)
    ap.add_argument("--split", choices=("val", "test"), default="val")
    ap.add_argument("--val-year", type=int, default=2024)
    ap.add_argument("--test-year", type=int, default=2025)
    ap.add_argument("--rounds", type=int, default=300)
    ap.add_argument("--max-rows", type=int, default=400_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path(os.environ["LAB_OUTPUT_DIR"]) if os.environ.get("LAB_OUTPUT_DIR")
                    else None)
    a = ap.parse_args(argv)
    t_start = time.perf_counter()

    eval_year = a.val_year if a.split == "val" else a.test_year
    data_dir = a.data or ensure_dataset(a.sites, a.feeders, a.start_year, eval_year)
    fa = load_arrays(data_dir, years=list(range(a.start_year, eval_year + 1)))
    train_days = fa.days.year < a.val_year
    eval_days = fa.days.year == eval_year
    train_sites = (fa.sites.split == "train").to_numpy()
    unseen = ~train_sites
    print(f"data {data_dir}: {fa.n} sites ({unseen.sum()} unseen), {train_days.sum()} train days, "
          f"{eval_days.sum()} {a.split} days; loaded in {time.perf_counter() - t_start:.0f}s", flush=True)
    p = a.split  # metric prefix: val_ / test_
    report: dict = {"label": "synthetic RO", "split": a.split, "eval_year": eval_year, "data": str(data_dir),
                    "n_sites": fa.n, "n_unseen": int(unseen.sum()), "pv": {}, "load": {}, "faults": {}}

    # ------------------------------------------------------------------ PV
    day_steps = fa.geom["zenith"] < 90
    healthy = fa.fault == 0
    base = eval_days[None, :, None] & day_steps & healthy & np.isfinite(fa.pv_kw)
    base[:, :7] = False
    t = time.perf_counter()
    band = pvlib_band(fa, train_days, train_sites)
    pv_preds = {
        "pvlib": np.stack([fa.pvlib_kw * band[0], fa.pvlib_kw, fa.pvlib_kw * band[1]]),
        "persist_d1": np.repeat(shift_days(fa.pv_kw, 1)[None], 3, 0),
        "persist_ref7": np.repeat(pv_ref7_bulk(fa)[None], 3, 0),
    }
    pv_bundle = lgbm.train("pv", fa, train_days, train_sites, a.max_rows, a.rounds, a.seed)
    pv_preds["lgbm"] = lgbm.predict_bulk(pv_bundle, fa, eval_days)
    print(f"pv baselines in {time.perf_counter() - t:.0f}s", flush=True)
    for name, q in pv_preds.items():
        allm = forecast_metrics(q, fa.pv_kw, base)
        unm = forecast_metrics(q, fa.pv_kw, base & unseen[:, None, None])
        report["pv"][name] = {"all": allm, "unseen": unm}
        metric(f"pv_{name}_{p}_nmae", allm["nmae"])
        metric(f"pv_{name}_{p}_nmae_unseen", unm["nmae"])
        if name in ("pvlib", "lgbm"):
            metric(f"pv_{name}_{p}_pinball", allm["pinball"])
            metric(f"pv_{name}_{p}_coverage", allm["coverage"])
    # error analysis: nMAE by month and by metadata quality (recorded == installer's numbers)
    months = fa.days.month.to_numpy()
    report["pv"]["by_month"] = {
        name: {int(m): forecast_metrics(q, fa.pv_kw, base & (months == m)[None, :, None])["nmae"]
               for m in range(1, 13) if (base & (months == m)[None, :, None]).any()}
        for name, q in pv_preds.items()}
    if fa.eval_truth is not None:  # hidden truth used for slicing the error only
        for col in fa.eval_truth.columns:
            flag = fa.eval_truth[col].to_numpy(bool)[:, None, None]
            report["pv"][f"by_{col}"] = {name: {"yes": forecast_metrics(q, fa.pv_kw, base & flag)["nmae"],
                                                "no": forecast_metrics(q, fa.pv_kw, base & ~flag)["nmae"]}
                                         for name, q in pv_preds.items()}
    lg, pl = report["pv"]["lgbm"]["all"]["nmae"], report["pv"]["pvlib"]["all"]["nmae"]
    report["pv"]["lgbm_vs_pvlib_pct"] = 100 * (1 - lg / pl)
    metric(f"{p}_nmae", lg)
    metric(f"{p}_pinball", report["pv"]["lgbm"]["all"]["pinball"])
    metric(f"{p}_coverage", report["pv"]["lgbm"]["all"]["coverage"])
    metric("pv_lgbm_vs_pvlib_pct", report["pv"]["lgbm_vs_pvlib_pct"])
    report["pv"]["feature_importance"] = dict(sorted(zip(pv_bundle.features, map(float, pv_bundle.models[0.5]
                                              .feature_importance("gain"))), key=lambda kv: -kv[1])[:10])

    # ------------------------------------------------------------------ load
    t = time.perf_counter()
    lmask = eval_days[None, :, None] & np.isfinite(fa.load_kw) & np.ones_like(day_steps)
    lmask[:, :7] = False
    load_preds = {
        "persist_d1": np.repeat(shift_days(fa.load_kw, 1)[None], 3, 0),
        "persist_w1": np.repeat(shift_days(fa.load_kw, 7)[None], 3, 0),
    }
    load_bundle = lgbm.train("load", fa, train_days, train_sites, a.max_rows, a.rounds, a.seed)
    load_preds["lgbm"] = lgbm.predict_bulk(load_bundle, fa, eval_days)
    print(f"load baselines in {time.perf_counter() - t:.0f}s", flush=True)
    for name, q in load_preds.items():
        allm = forecast_metrics(q, fa.load_kw, lmask)
        unm = forecast_metrics(q, fa.load_kw, lmask & unseen[:, None, None])
        report["load"][name] = {"all": allm, "unseen": unm}
        metric(f"load_{name}_{p}_nmae", allm["nmae"])
        metric(f"load_{name}_{p}_nmae_unseen", unm["nmae"])
        if name == "lgbm":
            metric(f"load_lgbm_{p}_pinball", allm["pinball"])
            metric(f"load_lgbm_{p}_coverage", allm["coverage"])

    # ------------------------------------------------------------------ faults
    t = time.perf_counter()
    fr = run_fault_baselines(fa, np.flatnonzero(train_days), np.flatnonzero(eval_days))
    print(f"fault baselines in {time.perf_counter() - t:.0f}s", flush=True)
    report["faults"] = {"theta": fr.theta, "cusum_k": fr.k, "cusum_h": fr.h, **fr.metrics}
    for name in ("threshold", "cusum"):
        m = fr.metrics[name]
        for k in ("prauc", "precision", "recall", "f1", "event_recall", "hours_to_detect",
                  "false_alarms_per_site_year"):
            metric(f"fault_{name}_{p}_{k}", m.get(k, float("nan")))
    best = max(("threshold", "cusum"), key=lambda n: fr.metrics[n]["prauc"])
    report["faults"]["best"] = best
    metric(f"{p}_prauc", fr.metrics[best]["prauc"])
    metric("hours_to_detect", fr.metrics[best]["hours_to_detect"])

    report["seconds"] = round(time.perf_counter() - t_start, 1)
    metric("baseline_seconds", report["seconds"])
    if a.out:
        a.out.mkdir(parents=True, exist_ok=True)
        pv_bundle.save(a.out / "lgbm_pv.json")
        load_bundle.save(a.out / "lgbm_load.json")
        (a.out / "baselines_pv.json").write_text(json.dumps(report, indent=2, default=float))
        print(f"wrote {a.out}/baselines_pv.json, lgbm_pv.json, lgbm_load.json")
    _summary(report)
    return report


def _summary(r: dict) -> None:
    print(f"\n== baselines ({r['label']}, {r['split']} {r['eval_year']}, {r['n_sites']} sites) ==")
    for kind in ("pv", "load"):
        for name, v in r[kind].items():
            if isinstance(v, dict) and "all" in v:
                a_, u = v["all"], v["unseen"]
                print(f"{kind:4s} {name:13s} nMAE {a_['nmae']:.3f} (unseen {u['nmae']:.3f})  "
                      f"pinball {a_['pinball']:.3f}  cover {a_['coverage']:.2f}")
    for name in ("threshold", "cusum"):
        m = r["faults"][name]
        print(f"fault {name:9s} PR-AUC {m['prauc']:.3f}  P {m['precision']:.2f} R {m['recall']:.2f} "
              f"F1 {m['f1']:.2f}  events {m.get('event_recall', float('nan')):.2f}  "
              f"h-to-detect {m.get('hours_to_detect', float('nan')):.0f}")


if __name__ == "__main__":
    main()
