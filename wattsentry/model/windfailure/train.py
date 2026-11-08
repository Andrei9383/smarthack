"""M7 WindFailure training + evaluation vs a fixed residual threshold (optional model).

    python -m wattsentry.model.windfailure.train [--residuals physics|nbm] [--fa-budget 0.5]
    train.py --model windfailure ...   →  wattsentry.model.windfailure.train.main(argv)

Pipeline: load wind (simulator output, else stand-in synth) → temperature residuals (physics linear thermal NBM, or
M6's GRU trained in-process; label-free normal selection, no drift mask) → daily per turbine-component features
(features.py) → targets P(failure of component c within 7 / 14 days) from the injector's exact failure onsets →
LightGBM per horizon → alarm threshold tuned on validation under a false-alarm budget and frozen → test metrics
for M7 and the fixed-threshold baseline (daily mean residual over the turbine's own baseline) → ship decision.

Splits (same rules as M6): held-out park (default WP2) never in training or validation; test = the last calendar
year (2025, all parks; held-out park reported separately); train / val = earlier time of the other parks, by
time: validation months 3, 6, 9, 12, training the rest, 14-day embargo around every validation block and before
the test year (a row's 14-day label window never reaches into another split, so test failures are never trained on).

Outputs in $LAB_OUTPUT_DIR (default ./.lab-out): windfailure_7d.txt / windfailure_14d.txt (LightGBM model text),
windfailure_thresholds.json (frozen thresholds + residual model + feature list: everything predictor.py needs),
windfailure_report.json (metrics, per-event lead times, per-component results), manifest.json.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from . import COMPONENTS, HORIZONS_D
from .features import BASELINE_SCORE, FEATURES, daily_features, feature_index
from .labels import daily_targets, day_grid, failures_from_episodes
from .metrics import evaluate_alarms, pr_auc, recall_at_fa, tune_threshold

LABEL = "synthetic RO, ERA5-driven"
WARMUP_D = 28  # first days without a full per-turbine baseline are not used for training


class Run:
    def __init__(self):
        self.t0 = time.time()
        self.metrics: dict[str, float] = {}

    def emit(self, name: str, value) -> None:
        value = float(value)
        self.metrics[name] = value
        print(f"LAB_METRIC {name}={value:.6g}", flush=True)

    def stage(self, name: str, t: float) -> None:
        self.emit(f"t_{name}_s", time.time() - t)


def parse_args(argv):
    ap = argparse.ArgumentParser(prog="windfailure", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--residuals", choices=["physics", "nbm"], default="physics")
    ap.add_argument("--data", choices=["auto", "synth"], default="auto",
                    help="auto: simulator output if cached (else builds it, else synth); synth: stand-in generator")
    ap.add_argument("--synth-days", type=int, default=731)
    ap.add_argument("--synth-scale", type=float, default=1.0)
    ap.add_argument("--holdout-park", default=None)
    ap.add_argument("--folds", type=int, default=4, help="time-blocked CV folds over the pre-test period")
    ap.add_argument("--embargo-days", type=int, default=14, help="training days dropped BEFORE each validation block "
                    "and before the test year (= the label horizon)")
    ap.add_argument("--embargo-after", type=int, default=7, help="training days dropped after each validation block")
    ap.add_argument("--fa-budget", type=float, default=0.5, help="false-alarm episodes per turbine-year on validation")
    ap.add_argument("--rounds", type=int, default=800)
    ap.add_argument("--lr", type=float, default=0.03)
    ap.add_argument("--leaves", type=int, default=15)
    ap.add_argument("--min-leaf", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    # nbm residual source (torch)
    ap.add_argument("--nbm-epochs", type=int, default=20)
    ap.add_argument("--nbm-steps", type=int, default=300)
    ap.add_argument("--nbm-hidden", type=int, default=96)
    ap.add_argument("--nbm-lr", type=float, default=2e-3)
    ap.add_argument("--nbm-max-minutes", type=float, default=15.0)
    ap.add_argument("--care", action="store_true",
                    help="after training, score the artifacts on CARE-to-Compare Wind Farm A (non-RO; range-fetched "
                         "into ~/.cache/wattsentry/care, skipped offline / > --care-max-gb)")
    ap.add_argument("--care-max-gb", type=float, default=2.0)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--model", default="windfailure", help="ignored (train.py dispatch)")
    return ap.parse_args(argv)


def _git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return os.environ.get("LAB_COMMIT", "unknown")


def _sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def load_data(args, log=print):
    from wattsentry.model.wind.data import external_wind_dir, load_wind
    from wattsentry.model.wind.synth import SynthConfig

    synth = SynthConfig(days=args.synth_days, scale=args.synth_scale)
    if args.data == "synth":
        return load_wind(prefer_external=False, synth_config=synth, log=log)
    if external_wind_dir() is None:
        try:  # build the simulator output (deterministic, cached) so the run uses the ERA5-driven data
            from wattsentry.sim.wind.build import build

            log("[windfailure] no cached wind data: building it with wattsentry.sim.wind (one-off)")
            build(plots=False)
        except Exception as e:  # pragma: no cover - depends on worker
            log(f"[windfailure] sim build failed ({e!r}); falling back to the stand-in synth")
    return load_wind(prefer_external=True, synth_config=synth, log=log)


def day_folds(days: np.ndarray, test_year: int, n_folds: int, embargo_d: int, embargo_after: int):
    """→ fold [D] (0..n_folds-1 = contiguous pre-test time blocks, -1 = test year or the embargo before it),
    test_d bool [D], train_mask(k) → bool [D] (other blocks minus embargo_d days before / embargo_after days after
    block k, so a training row's label window never reaches into block k)."""
    yr = days.astype("datetime64[Y]").astype(int) + 1970
    test_d = yr == test_year
    if not test_d.any() or test_d.all():  # single-year data: last 25 % is test
        test_d = np.zeros(len(days), bool)
        test_d[int(len(days) * 0.75):] = True
    test_start = int(np.argmax(test_d))
    idx = np.arange(len(days))
    usable = idx < test_start - embargo_d
    fold = np.full(len(days), -1)
    n_use = int(usable.sum())
    fold[usable] = np.minimum(idx[usable] * n_folds // max(1, n_use), n_folds - 1)

    def train_mask(k: int) -> np.ndarray:
        blk = np.flatnonzero(fold == k)
        near = (idx >= blk[0] - embargo_d) & (idx <= blk[-1] + embargo_after)
        return (fold >= 0) & (fold != k) & ~near

    return fold, test_d, train_mask


def _rows(mask_dn: np.ndarray, C: int) -> np.ndarray:
    return np.repeat(mask_dn[..., None], C, axis=2)


# ----------------------------------------------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> dict:
    args = parse_args(argv if argv is not None else sys.argv[1:])
    if sys.platform == "darwin":  # torch + lightgbm each bring a libomp; on macOS they segfault unless single-threaded
        os.environ.setdefault("OMP_NUM_THREADS", "1")
    import lightgbm as lgb

    from wattsentry.model.wind.data import make_splits

    run = Run()
    out = Path(args.out or os.environ.get("LAB_OUTPUT_DIR", ".lab-out"))
    out.mkdir(parents=True, exist_ok=True)
    run_id = (os.environ.get("LAB_RUN_ID") or os.environ.get("LAB_RUN")
              or datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S"))
    version = f"windfailure-r{run_id}"
    print(f"[windfailure] residuals={args.residuals} out={out} version={version}", flush=True)

    t = time.time()
    data = load_data(args)
    sp = make_splits(data, args.holdout_park)
    run.stage("data", t)
    N, T = len(data.turbines), data.n_steps
    rated = data.rated_kw.astype(np.float64)
    is_large = (rated >= 4000).astype(np.int64)
    park_of = data.park_of
    days = day_grid(data.ts)
    D, C = len(days), len(COMPONENTS)
    fold, test_d, train_mask = day_folds(days, sp.test_year, args.folds, args.embargo_days, args.embargo_after)
    pre_d = fold >= 0  # pre-test days: CV train / validation (out-of-fold)
    train_t = np.repeat(pre_d, 96)[:T]
    if len(train_t) < T:
        train_t = np.concatenate([train_t, np.zeros(T - len(train_t), bool)])
    seen = np.zeros(N, bool)
    seen[sp.train_turbines] = True
    print(f"[windfailure] {N} turbines ({int(seen.sum())} training, held-out park {sp.holdout_park}), {D} days; "
          f"pre-test days {int(pre_d.sum())} in {args.folds} time folds, test {int(test_d.sum())} ({sp.test_year})",
          flush=True)

    # --- residuals ----------------------------------------------------------------------------------------------
    t = time.time()
    if args.residuals == "physics":
        from . import thermal

        res_model = thermal.fit(data.sensors, data.err, rated, is_large, sp.train_turbines, train_t)
        r = thermal.residuals(res_model, data.sensors, data.err, rated, is_large)
        res_desc = ("physics stand-in: linear thermal NBM (lag bank of load^g and ambient), per class x component, "
                    "fitted label-free on training parks/time (finite, no error code, 12 h after gaps; no drift mask)")
    else:
        from .residuals_nbm import nbm_residuals

        r, res_model = nbm_residuals(data, sp, train_t, args, log=print, out=out)
        res_desc = ("M6 WindNBM GRU (wattsentry.model.wind.nbm) trained in-process on the thermal objective, "
                    "label-free mask (finite, no error code; no labels, no drift mask); held-out park uses embedding 0")
    run.stage("residuals", t)
    ok = np.isfinite(r) & train_t[:, None, None] & seen[None, :, None]
    run.emit("res_train_mae_c", float(np.nanmean(np.abs(r[ok]))))
    for ci, c in enumerate(COMPONENTS):
        tm = np.isfinite(r[..., ci]) & ~train_t[:, None] & ~np.repeat(test_d, 96)[:T, None]
        run.emit(f"res_val_mae_{c}", float(np.nanmean(np.abs(r[..., ci][tm]))))

    # --- features + targets ----------------------------------------------------------------------------------------
    t = time.time()
    X = daily_features(r, data.sensors["act_pwt"], data.sensors["turbine_t"], rated, park_of, is_large)
    del r
    failures, drifts = failures_from_episodes(data.episodes, data.turbine_ids)
    y, down = daily_targets(failures, days, N, HORIZONS_D)
    run.stage("features", t)
    warm = np.arange(D) >= WARMUP_D

    def fail_in(dmask, tmask):
        def day_of(f):
            return min(D - 1, int((f.t_fail - days[0]) // np.timedelta64(1440, "m")))

        return [f for f in failures if tmask[f.turbine] and dmask[day_of(f)]]

    ev_val = fail_in(pre_d, seen)  # every pre-test failure of the training parks is validated out-of-fold
    ev_test = fail_in(test_d, np.ones(N, bool))
    ev_test_hold, ev_test_seen = [f for f in ev_test if not seen[f.turbine]], [f for f in ev_test if seen[f.turbine]]
    for k, v in {"n_fail_pretest": ev_val, "n_fail_test": ev_test, "n_fail_test_holdout": ev_test_hold}.items():
        run.emit(k, len(v))
    base_elig = seen[None] & ~down & warm[:, None]
    elig_val = pre_d[:, None] & base_elig
    elig_test = test_d[:, None] & ~down
    elig_test_hold = elig_test & ~seen[None]
    elig_test_seen = elig_test & seen[None]

    # --- LightGBM per horizon --------------------------------------------------------------------------------------
    t = time.time()
    Xf = X.reshape(-1, len(FEATURES))
    params = {"objective": "binary", "learning_rate": args.lr, "num_leaves": args.leaves,
              "min_data_in_leaf": args.min_leaf, "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1,
              "lambda_l2": 1.0, "seed": args.seed, "verbose": -1, "metric": "binary_logloss",
              "num_threads": int(os.environ.get("OMP_NUM_THREADS", "0") or 0)}
    boosters, P = {}, {}
    for h in HORIZONS_D:
        yf = y[h].reshape(-1)
        Ph = np.full(D * N * C, np.nan)
        iters = []
        for k in range(args.folds):  # out-of-fold predictions on every pre-test row (threshold tuning)
            rtr = _rows(train_mask(k)[:, None] & base_elig, C).reshape(-1)
            rva = _rows((fold == k)[:, None] & base_elig, C).reshape(-1)
            if yf[rtr].sum() == 0 or yf[rva].sum() == 0:
                print(f"[windfailure] {h}d fold {k}: no positives in train or val, skipped", flush=True)
                continue
            dtr = lgb.Dataset(Xf[rtr], yf[rtr], feature_name=FEATURES, categorical_feature=["comp_id"])
            dva = lgb.Dataset(Xf[rva], yf[rva], reference=dtr)
            b = lgb.train(params, dtr, num_boost_round=args.rounds, valid_sets=[dva],
                          callbacks=[lgb.early_stopping(150, verbose=False)])
            iters.append(b.best_iteration)
            vrows = _rows((fold == k)[:, None] & np.ones((1, N), bool), C).reshape(-1)
            Ph[vrows] = b.predict(Xf[vrows], num_iteration=b.best_iteration)
            print(f"[windfailure] {h}d fold {k}: train rows {int(rtr.sum())} (pos {int(yf[rtr].sum())}), val rows "
                  f"{int(rva.sum())} (pos {int(yf[rva].sum())}), best iter {b.best_iteration}", flush=True)
        n_rounds = int(max(30, np.median(iters) * 1.1)) if iters else 100
        rall = _rows(pre_d[:, None] & base_elig, C).reshape(-1)
        dall = lgb.Dataset(Xf[rall], yf[rall], feature_name=FEATURES, categorical_feature=["comp_id"])
        b = lgb.train(params, dall, num_boost_round=n_rounds)
        boosters[h] = b
        trows = _rows(test_d[:, None] & np.ones((1, N), bool), C).reshape(-1)
        Ph[trows] = b.predict(Xf[trows])  # test year: the final model only (never saw 2025)
        P[h] = Ph.reshape(D, N, C)
        run.emit(f"rounds_{h}d", n_rounds)
    run.stage("lgbm", t)

    # --- thresholds on validation (frozen), test evaluation --------------------------------------------------------
    t = time.time()
    score_m = P[14]
    score_b = X[..., feature_index(BASELINE_SCORE)].astype(np.float64)
    # Equal-false-alarm protocol: the baseline gets the budget first; M7 must then stay at or below the false-alarm
    # rate the baseline actually reached on validation, so the frozen comparison is recall at <= equal FA.
    thr_b, val_b, curve_b = tune_threshold(score_b, elig_val, ev_val, days, args.fa_budget)
    fa_m_budget = min(args.fa_budget, val_b.get("false_alarms_per_turbine_year", args.fa_budget))
    thr_m, val_m, curve_m = tune_threshold(score_m, elig_val, ev_val, days, fa_m_budget)
    run.emit("val_fa_budget_model", fa_m_budget)
    run.emit("thr_p14", thr_m)
    run.emit("baseline_thr_dev_c", thr_b)
    run.emit("val_wf_event_recall", val_m.get("event_recall", float("nan")))
    run.emit("val_wf_false_alarms_per_turbine_year", val_m.get("false_alarms_per_turbine_year", float("nan")))
    run.emit("val_baseline_threshold_event_recall", val_b.get("event_recall", float("nan")))
    run.emit("val_baseline_threshold_false_alarms_per_turbine_year",
             val_b.get("false_alarms_per_turbine_year", float("nan")))

    def ev(score, thr, elig, evs):
        return evaluate_alarms(np.nan_to_num(score, nan=-np.inf) >= thr, elig, evs, days, 14, drifts=drifts)

    res = {
        "test": {"model": ev(score_m, thr_m, elig_test, ev_test), "baseline": ev(score_b, thr_b, elig_test, ev_test)},
        "test_holdout": {"model": ev(score_m, thr_m, elig_test_hold, ev_test_hold),
                         "baseline": ev(score_b, thr_b, elig_test_hold, ev_test_hold)},
        "test_seen": {"model": ev(score_m, thr_m, elig_test_seen, ev_test_seen),
                      "baseline": ev(score_b, thr_b, elig_test_seen, ev_test_seen)},
        "val": {"model": val_m, "baseline": val_b},
    }
    for scope, sfx in (("test", ""), ("test_holdout", "_holdout")):
        m, b = res[scope]["model"], res[scope]["baseline"]
        run.emit(f"lead_time_h{sfx}", m["lead_time_h"])
        run.emit(f"wf_event_recall{sfx}", m["event_recall"])
        run.emit(f"wf_false_alarms_per_turbine_year{sfx}", m["false_alarms_per_turbine_year"])
        run.emit(f"baseline_threshold_lead_time_h{sfx}", b["lead_time_h"])
        run.emit(f"baseline_threshold_event_recall{sfx}", b["event_recall"])
        run.emit(f"baseline_threshold_false_alarms_per_turbine_year{sfx}", b["false_alarms_per_turbine_year"])
    rte = _rows(elig_test, C)
    for h in HORIZONS_D:
        run.emit(f"wf_prauc_{h}d", pr_auc(y[h][rte], P[h][rte]))
        run.emit(f"baseline_threshold_prauc_{h}d", pr_auc(y[h][rte], score_b[rte]))
        rh = _rows(elig_test_hold, C)
        run.emit(f"wf_prauc_{h}d_holdout", pr_auc(y[h][rh], P[h][rh]))
    # Equal-FA comparison on test. M7 keeps its frozen threshold; the baseline is given its BEST test threshold with
    # false alarms <= M7's frozen test rate (an oracle advantage for the baseline: a conservative check for M7).
    m, b = res["test"]["model"], res["test"]["baseline"]
    mh = res["test_holdout"]["model"]
    diag = {
        "baseline_oracle_at_model_fa": recall_at_fa(score_b, elig_test, ev_test, days,
                                                    m["false_alarms_per_turbine_year"]),
        "baseline_oracle_at_model_fa_holdout": recall_at_fa(score_b, elig_test_hold, ev_test_hold, days,
                                                            mh["false_alarms_per_turbine_year"]),
        "model_oracle_at_baseline_fa": recall_at_fa(score_m, elig_test, ev_test, days,
                                                    b["false_alarms_per_turbine_year"]),
    }
    run.emit("baseline_threshold_event_recall_at_wf_fa", diag["baseline_oracle_at_model_fa"]["event_recall"] or 0.0)
    run.emit("baseline_threshold_lead_time_h_at_wf_fa", diag["baseline_oracle_at_model_fa"]["lead_time_h"] or 0.0)
    run.emit("baseline_threshold_event_recall_at_wf_fa_holdout",
             diag["baseline_oracle_at_model_fa_holdout"]["event_recall"] or 0.0)
    run.stage("eval", t)

    # --- ship decision -----------------------------------------------------------------------------------------------
    checks = {
        "lead_time_positive_test": m["n_detected"] > 0 and m["lead_time_h"] > 0,
        "lead_time_positive_holdout_park": mh["n_detected"] > 0 and mh["lead_time_h"] > 0,
        "recall_beats_baseline_at_equal_fa":
            m["event_recall"] > (diag["baseline_oracle_at_model_fa"]["event_recall"] or 0),
    }
    info = {
        "frozen_vs_frozen_dominates": m["event_recall"] >= b["event_recall"]
        and m["false_alarms_per_turbine_year"] <= b["false_alarms_per_turbine_year"],
        "holdout_recall_beats_baseline_at_equal_fa": mh["event_recall"]
        > (diag["baseline_oracle_at_model_fa_holdout"]["event_recall"] or 0),
        "lead_time_beats_baseline_test": m["lead_time_h"] > b["lead_time_h"],
    }
    ship = all(checks.values())
    rule = ("ship iff (1) median lead time > 0 on held-out test events (test year, all parks) and (2) on the held-out "
            "park, and (3) M7's event recall at its frozen validation threshold beats the fixed-threshold baseline at "
            "<= the same false-alarm rate (baseline given its best test threshold with FA <= M7's test FA)")
    print(f"[windfailure] SHIP DECISION: {'SHIP' if ship else 'NOT SHIPPED'} — {rule}; checks {checks}; info {info}",
          flush=True)
    run.emit("wf_ship", int(ship))

    # --- artifacts ---------------------------------------------------------------------------------------------------
    files = {}
    for h, bst in boosters.items():
        p = out / f"windfailure_{h}d.txt"
        bst.save_model(str(p))
        files[p.name] = p
    imp = {h: dict(sorted(zip(FEATURES, bst.feature_importance("gain").round(1).tolist()), key=lambda kv: -kv[1]))
           for h, bst in boosters.items()}
    cal = {
        "model": "windfailure", "model_version": version, "label": LABEL, "shipped": ship,
        "residuals": args.residuals, "residual_model": res_model,
        "features": FEATURES, "components": COMPONENTS, "horizons_d": list(HORIZONS_D),
        "thresholds": {"p_fail_14d": thr_m, "baseline_dev_c": thr_b, "fa_budget_per_turbine_year": args.fa_budget},
        "risk_levels": {"high": thr_m, "medium": thr_m / 2},
        "baseline_window_days": [21, 60], "history_days_needed": 61,
    }
    if res_model.get("onnx"):
        files[res_model["onnx"]] = out / res_model["onnx"]
    p = out / "windfailure_thresholds.json"
    p.write_text(json.dumps(cal, indent=1, default=float))
    files[p.name] = p

    def strip(d):  # per-event lists only once, in "events"
        return {k: v for k, v in d.items() if k != "events"}

    report = {
        "model": "windfailure", "model_version": version, "label": LABEL, "data_source": data.source,
        "residual_source": res_desc, "normal_selection": "label-free (no labels, no drift mask) for fitting residuals; "
        "features computed on all steps",
        "splits": {"holdout_park": sp.holdout_park, "test_year": sp.test_year, "folds": args.folds,
                   "embargo_days_before": args.embargo_days, "embargo_days_after": args.embargo_after,
                   "days_pretest": int(pre_d.sum()), "days_test": int(test_d.sum()),
                   "rule": "pre-test time of the training parks in contiguous blocks; thresholds tuned on out-of-fold "
                           "predictions and frozen; final LightGBM refit on all pre-test rows; held-out park and test "
                           "year never in training"},
        "definitions": {"alarm": "p_fail_14d >= threshold (model) / daily mean residual minus own 21-60 d lagged "
                        "baseline >= threshold degC (baseline), per turbine-component-day",
                        "true_alarm": "failure of that component within 14 d after the alarm day ends",
                        "lead_time_h": "median over detected events of failure onset - end of first true alarm day",
                        "false_alarms": "episodes of false alarm days per turbine (gaps <= 2 d merged) / turbine-year"},
        "ship": {"shipped": ship, "rule": rule, "checks": checks, "info": info},
        "metrics": run.metrics,
        "results": {s: {k: strip(v) for k, v in d.items()} for s, d in res.items()},
        "events_test": {"model": res["test"]["model"]["events"], "baseline": res["test"]["baseline"]["events"]},
        "diagnostics": diag,
        "threshold_curves_val": {"model": curve_m, "baseline": curve_b},
        "feature_importance_gain": imp,
        "notes": ["Simulator fails only gearbox_t / gen1_t / gen2_t; converter / transformer rows are all negative.",
                  "40 % of degradation drifts do not end in a failure: alarms on them count as false alarms here.",
                  "No M6 overheating-class baseline: M6's classifier is not trained in this run.",
                  "Protocol history (disclosed): first drafts counted false alarms per alarm DAY run outside the 14-d "
                  "window and used one validation split; that scored an early alarm on the same drift as a false "
                  "alarm and tuned on 7 events. Final protocol (applied identically to M7 and the baseline): "
                  "episode-level false alarms, 4-fold time-blocked CV for thresholds, baseline-oracle equal-FA check."],
    }
    if args.care:  # after the artifacts exist: scores exactly what predictor.py would load
        t = time.time()
        from wattsentry.eval.care.score import run as care_run

        care = care_run(out, max_gb=args.care_max_gb, emit=run.emit, out_dir=out)
        report["care"] = {k: v for k, v in care.items() if k != "datasets"}
        if (out / "care_report.json").exists():
            files["care_report.json"] = out / "care_report.json"
        run.stage("care", t)
        report["metrics"] = run.metrics
    p = out / "windfailure_report.json"
    p.write_text(json.dumps(report, indent=1, default=float))
    files[p.name] = p
    manifest = {
        "model": "windfailure", "model_version": version, "card": 43,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_sha": _git_sha(), "argv": sys.argv, "args": vars(args), "label": LABEL, "data_source": data.source,
        "python": platform.python_version(), "lightgbm": lgb.__version__, "shipped": ship,
        "files": {k: {"sha256": _sha256(v), "bytes": v.stat().st_size} for k, v in files.items()},
        "t_total_s": round(time.time() - run.t0, 2),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1, default=str))
    run.emit("t_total_s", time.time() - run.t0)
    return report


if __name__ == "__main__":
    main()
