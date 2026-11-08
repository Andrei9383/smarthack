"""Wind baselines on #41's data and splits (held-out park, test year): WindMon rules, curve rules, binned curve.

    python -m wattsentry.eval.baselines.wind.run [--synth] [--no-tune]

LAB_METRIC names match #41: unprefixed / val_* = curve rules (the strongest baseline), baseline_windmon_* = WindMon's
four rules, val_nmae_wind ... = binned curve nowcast, met_* = the met-wind curve (Production mode, M5 fallback).
Writes eval_report.json (+ error analysis) and binned_curve.json to $LAB_OUTPUT_DIR."""

from __future__ import annotations

import argparse
import itertools
import time

import numpy as np

from wattsentry.eval.metrics import coverage, pinball
from wattsentry.eval.report import Report, emit
from wattsentry.model.wind.data import CLASS_INDEX, CLASSES, POWER, external_wind_dir, load_wind, make_splits
from wattsentry.model.wind.metrics import A_IDX, event_metrics, step_confusion, truth_episodes
from wattsentry.model.wind.synth import SynthConfig

from .curve import SOURCES, data_inputs, fit_curves
from .detector import PARAMS, RULES_DOC, detect_all
from .windmon_rules import RULES, raw_alert_count, windmon_classes, windmon_flags

GRID = {"low_margin": (0.05, 0.1), "anem_rel": (0.25, 0.4), "under_resid": (-0.08, -0.12),
        "curtail_park_frac": (0.2, 0.3)}
KEYS = ("wind_precision", "wind_recall", "alerts_per_turbine_day", "curtail_vs_fault_f1")


def objective(m: dict) -> float:  # #41's threshold-tuning objective
    pr, rc = m["wind_precision"], m["wind_recall"]
    f05 = 1.25 * pr * rc / (0.25 * pr + rc) if pr + rc else 0.0
    return f05 + 0.5 * m["curtail_vs_fault_f1"] + 0.25 * m["class_correct_recall_macro"]


def curtail_alerted(truth, pred) -> float:
    eps = [(j, s, e) for j in range(truth.shape[1]) for c, s, e in truth_episodes(truth[:, j]) if c == CLASS_INDEX["curtailment"]]
    return float(np.mean([np.isin(pred[s:e, j], A_IDX).any() for j, s, e in eps])) if eps else 0.0


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser(prog="python -m wattsentry.eval.baselines.wind.run", description=__doc__)
    ap.add_argument("--synth", action="store_true", help="#41's stand-in generator instead of #22/#23 output")
    ap.add_argument("--holdout-park", default=None)
    ap.add_argument("--no-tune", action="store_true", help="skip the validation grid over the curve-rule params")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    t0 = time.time()
    rep = Report(vars(args), args.out)
    if not args.synth and external_wind_dir() is None:
        from wattsentry.sim.wind.build import build

        build()
    data = load_wind(prefer_external=not args.synth, synth_config=SynthConfig())
    label = "synthetic_RO (stand-in synth.py)" if data.source.startswith("synth") else "synthetic_RO (ERA5-driven #22/#23)"
    sp = make_splits(data, args.holdout_park)
    print(f"[wind-baselines] {data.source}: {len(data.turbines)} turbines, {data.n_steps} steps; {sp.describe()}")

    # --- binned curve ----------------------------------------------------------------------------------------------
    curve = fit_curves(data, sp.train_t, sp.train_turbines)
    y = data.sensors[POWER] / data.rated_kw[None]
    normal = (data.labels == 0) & np.isfinite(y)
    vm = normal & sp.val_t[:, None]
    vm[:, sp.holdout_turbines] = False
    hm = normal & sp.test_t[:, None]
    hm[:, sp.train_turbines] = False
    v = data_inputs(data)
    fc, qs = {}, {}
    for s in SOURCES:
        q = qs[s] = curve.predict_block(data.turbines, v, s)
        o10, o90 = np.nanquantile((y - q[..., 0])[vm], 0.1), np.nanquantile((y - q[..., 2])[vm], 0.9)
        fc[s] = {"val_nmae_wind": float(np.nanmean(np.abs(y - q[..., 1])[vm])),
                 "val_pinball_wind": float(np.mean([pinball(q[..., i], y, t, vm) for i, t in enumerate((.1, .5, .9))])),
                 "val_coverage_wind": coverage(q[..., 0], q[..., 2], y, vm),
                 "test_nmae_wind_holdout": float(np.nanmean(np.abs(y - q[..., 1])[hm])),
                 "test_coverage_wind_holdout_cal": coverage(q[..., 0] + o10, q[..., 2] + o90, y, hm)}
    curve.meta["nowcast_source"] = best = min(SOURCES, key=lambda s: fc[s]["val_nmae_wind"])
    for k, x in fc[best].items():
        emit(k, x)
    for k, x in fc["met"].items():
        emit(f"met_{k}", x)
    po = data.park_of
    by_park_nmae = {p: {s: float(np.nanmean(np.abs(y - qs[s][..., 1])[normal & sp.test_t[:, None] & (po == i)[None]]))
                        for s in SOURCES} for i, p in enumerate(data.park_ids)}

    # --- curve rules: tune on validation (training parks), freeze, score the held-out park's test year ----------------
    vt, tr = sp.val_t, sp.train_turbines
    params, best_m, tuned = dict(PARAMS), None, []
    for combo in ([()] if args.no_tune else itertools.product(*GRID.values())):
        p = {**PARAMS, **dict(zip(GRID, combo))}
        m = event_metrics(data.labels[vt][:, tr], detect_all(curve, data, p)[vt][:, tr], full=False)
        tuned.append({**dict(zip(GRID, combo)), "objective": objective(m), **{k: m[k] for k in KEYS}})
        if best_m is None or objective(m) > objective(best_m):
            params, best_m = p, m
    for k in KEYS:
        emit(f"val_{k}", best_m[k])
    pred = detect_all(curve, data, params)

    ts0 = int(np.argmax(sp.test_t))
    sens_t = {k: x[ts0:] for k, x in data.sensors.items()}
    rated = data.rated_kw.astype(np.float64)
    flags = windmon_flags(sens_t, data.err[ts0:], rated)  # test slice only, exactly as #41 scores WindMon
    wpred = windmon_classes(flags)
    truth = data.labels[ts0:]
    systems = {"curve_rules": pred[ts0:], "windmon": wpred}
    hc = sp.holdout_turbines
    res = {}
    for name, pr in systems.items():
        m = event_metrics(truth[:, hc], pr[:, hc])
        m["step_confusion"] = step_confusion(truth[:, hc], pr[:, hc])
        m["curtailment_episodes_alerted_frac"] = curtail_alerted(truth[:, hc], pr[:, hc])
        cm = np.array(m["episode_confusion"]["matrix"])
        m["curtail_vs_fault_confusion"] = {r: dict(zip(CLASSES, cm[CLASS_INDEX[r]].tolist()))
                                           for r in ("curtailment", "fault_stop")}
        m["by_park_test_year"] = {p: {k: x for k, x in event_metrics(truth[:, po == i], pr[:, po == i],
                                                                      full=False).items() if k in KEYS}
                                  for i, p in enumerate(data.park_ids)}
        res[name] = m
    days = len(truth) / 96
    res["windmon"]["raw_rule_alerts_per_turbine_day"] = raw_alert_count({r: f[:, hc] for r, f in flags.items()}) / (len(hc) * days)
    res["windmon"]["rule_alerts_per_turbine_day"] = {r: raw_alert_count({r: flags[r][:, hc]}) / (len(hc) * days)
                                                     for r in RULES}
    for k in KEYS:
        emit(k, res["curve_rules"][k])
    for k in KEYS:
        emit(f"baseline_windmon_{k}", res["windmon"][k])
    emit("baseline_windmon_raw_alerts_per_turbine_day", res["windmon"]["raw_rule_alerts_per_turbine_day"])
    emit("true_events_per_turbine_day", res["curve_rules"]["true_actionable_per_turbine_day"])
    emit("wind_class_recall_macro", res["curve_rules"]["class_correct_recall_macro"])
    emit("curtailment_episodes_alerted_frac", res["curve_rules"]["curtailment_episodes_alerted_frac"])
    emit("baseline_windmon_curtailment_episodes_alerted_frac", res["windmon"]["curtailment_episodes_alerted_frac"])
    emit("t_total_s", time.time() - t0)

    rep.out_dir.mkdir(parents=True, exist_ok=True)
    curve.save(rep.out_dir / "binned_curve.json")
    for name, x in {"label": label, "data_source": data.source, "splits": sp.describe(),
                    "eval_scope": f"held-out park {sp.holdout_park}, test year {sp.test_year}",
                    "definitions": __import__("wattsentry.model.wind.metrics", fromlist=["x"]).__doc__,
                    "curve_rules_doc": RULES_DOC, "curve_rules_params": params, "tuning_val": tuned,
                    "binned_curve": {"nowcast_source": best, **fc}, "binned_curve_nmae_by_park_test": by_park_nmae,
                    "validation_curve_rules": best_m, **res}.items():
        rep.table(name, x)
    rep.write()
    print(f"[wind-baselines] {label}: curve rules P/R/apd/cvf "
          + " ".join(f"{res['curve_rules'][k]:.3f}" for k in KEYS) + " | WindMon "
          + " ".join(f"{res['windmon'][k]:.3f}" for k in KEYS) + f" | done in {time.time() - t0:.0f}s")
    return res


if __name__ == "__main__":
    main()
