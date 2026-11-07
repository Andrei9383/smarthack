"""J5 Judge evaluation on 2025 (team/judge-design.md §9): `python -m sentinel.judge.evaluate --drift pvnet`.

One lab run per drift target (grid drift=none,pvnet,sitehealth,loadnet):
  train     the Judge exactly as J3 (sentinel.judge.train): 2024 rows (cached by J3's key) -> LightGBM + isotonic
  drill     2025 drill weeks (one per sampled month, `--week-days` long after a 21-day burn-in). The target witness is
            degraded for all its assets from week day `--drift-day` until `--restore-day`, then restored; forms
            cycle per week (pvnet/loadnet: bias, narrow, frozen; sitehealth: confusion, false_alarm, blind, frozen).
            With `--drift none` every other week wipes the ledger on day 14 (cold start check).
  evaluate  Judge vs ledger only (isotonic on trust) vs static qualify.py rules (OOD / band width) vs trust everything:
            reliability curve + Brier, precision of "use", recall of "don't use", days to the health alert and to the
            verdict flip, false alarms per witness-month (none), recovery days, cold start, substitute check.
Outputs: LAB_METRIC lines (t_<stage>_s, gpu_util, metrics) and $LAB_OUTPUT_DIR/judge_eval_<drift>.json.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import time
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from sentinel.judge import features as F
from sentinel.judge import model as JM
from sentinel.judge import rows as R
from sentinel.judge import train as JT

FORMS = {"pvnet": ("bias", "narrow", "frozen"), "loadnet": ("bias", "narrow", "frozen"),
         "sitehealth": ("confusion", "false_alarm", "blind", "frozen")}
NAMES = ("judge", "ledger", "static", "trust_all")


@dataclass(frozen=True)
class DrillSpec(R.DriftSpec):
    end_day: Optional[int] = None  # drift off from this day index on (witness restored)


# ============================================================================================ drill weeks
def drill_weeks(year: int, drift: str, n: int, week_days: int, burn_in: int, drift_day: int, restore_day: int,
                seed: int = 0) -> list[R.WeekSpec]:
    months = np.linspace(1, 12, n).round().astype(int) if n < 12 else np.arange(1, 13)
    rng = np.random.default_rng([year, seed, 150])
    out = []
    for k, m in enumerate(months):
        start = date(year, int(m), 1).isoformat()  # day 1 keeps a 28-day week inside the data year
        dr, cold = None, None
        if drift != "none":
            form = FORMS[drift][k % len(FORMS[drift])]
            mag = (0.3 if k % 2 == 0 else -0.3) if form == "bias" else 0.5
            dr = DrillSpec(drift, form, burn_in + drift_day, "all", mag, burn_in + restore_day)
        elif k % 2:
            cold = burn_in + 14
        out.append(R.WeekSpec(year, int(m), start, dr, cold, int(rng.integers(1 << 30))))
    return out


def _windowed(fn):
    def wrap(x, spec, days_on, *a, **kw):
        end = getattr(spec, "end_day", None)
        if end is not None:
            days_on = days_on & (np.arange(len(days_on)) < end)
        return fn(x, spec, days_on, *a, **kw)
    wrap.__wrapped__ = fn
    return wrap


def _patch_rows() -> None:
    """R.run_week turns drift on from start_day for good; DrillSpec.end_day restores the witness (recovery drill)."""
    for name in ("drift_forecast", "drift_detector", "drift_wind_steps"):
        f = getattr(R, name)
        if not hasattr(f, "__wrapped__"):
            setattr(R, name, _windowed(f))


def _job(a):
    _patch_rows()  # spawned processes import R fresh
    spec, cfg, band, threads = a
    return R.run_week(spec, cfg, band, threads), R._peak_mb()


def drill_rows(cfg: R.RowConfig, weeks: list, workers: int = 0, week_gb: float = 7.0, use_cache: bool = True,
               log=print) -> pd.DataFrame:
    win, models_dir = Path(cfg.windows), Path(cfg.models_dir)
    band = R.pvlib_band(win)
    h = hashlib.sha256(R.cache_key(cfg, weeks, win, models_dir, band).encode())
    h.update(Path(__file__).read_bytes())
    path = R.cache_dir() / f"drill-{cfg.year}-{h.hexdigest()[:16]}.parquet"
    if use_cache and path.exists():
        log(f"[j5] cached drill rows: {path}")
        return pd.read_parquet(path)
    workers = workers or min(len(weeks), os.cpu_count() or 1, R.MAX_WORKERS,
                             max(1, int(0.6 * R._mem_available_gb() // week_gb)))
    threads = max(1, (os.cpu_count() or 1) // workers)
    log(f"[j5] drill {cfg.year}: {len(weeks)} weeks x {cfg.week_days} days, {workers} processes x {threads} threads")
    jobs = [(w, cfg, band, threads) for w in weeks]
    if workers <= 1:
        parts = [_job(j) for j in jobs]
    else:
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn")) as ex:
            parts = list(ex.map(_job, jobs))
    for w, (p, pk) in zip(weeks, parts):
        log(f"[j5]   {w.start}: {len(p)} rows, {w.drift.form if w.drift else 'clean'}"
            f"{' cold' if w.cold_day is not None else ''}, peak {pk:.0f} MB")
    df = pd.concat([p for p, _ in parts], ignore_index=True)
    df.attrs["week_peak_mb"] = max(pk for _, pk in parts)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, index=False)
    except OSError as e:
        log(f"[j5] cache write failed ({e})")
    return df


# ============================================================================================ scoring
def static_p(t: pd.DataFrame, thr: Optional[dict] = None) -> np.ndarray:
    """Today's static rules (sentinel/agent/qualify.py): use iff OOD <= ood_max and, for forecasts, relative band
    width <= rel_width_max, for detectors P(fault) sd <= pfault_std_max; telemetry not stale. As 0/1 'P(valid)'."""
    if thr is None:
        from sentinel.agent.qualify import load_thresholds
        thr = load_thresholds()
    s = thr["site"]
    g = lambda c, d: t[c].to_numpy(float) if c in t else np.full(len(t), d)  # noqa: E731
    ok = np.nan_to_num(g("ood_score", 0.0), nan=0.0) <= s["ood_max"]
    ok &= np.nan_to_num(g("telemetry_age_h", 0.0), nan=0.0) * 60 <= s["stale_minutes"] + 15
    fc = t["family"].to_numpy() == "forecast"
    ok &= ~fc | (np.nan_to_num(g("band_rel", 0.0), nan=0.0) <= s["rel_width_max"])
    ok &= fc | (np.nan_to_num(g("p_fault_std", 0.0), nan=0.0) <= s["pfault_std_max"])
    return ok.astype(float)


def score(te: pd.DataFrame, fams: dict, lo: dict) -> pd.DataFrame:
    """Per row: p_<name> and verdict v_<name> (use | widen | dont_use) for every compared filter."""
    te = te.copy()
    for n in NAMES:
        te[f"p_{n}"], te[f"v_{n}"] = np.nan, "use"
    for fam, fm in fams.items():
        i = te.index[te["family"] == fam]
        if not len(i):
            continue
        t = te.loc[i]
        r3 = t["n_eff"].to_numpy(float) < 0.5
        p, _ = fm.predict(F.build(t, fam))
        tl = lo[fam]["iso"](np.nan_to_num(t["trust_main"].to_numpy(float), nan=0.5))
        for n, pp, tau, tw, rule3 in (("judge", p, fm.tau_use, fm.tau_widen if fam == "forecast" else None, r3),
                                      ("ledger", tl, lo[fam]["tau"], lo[fam]["widen"], r3),
                                      ("static", static_p(t), 0.5, None, None),
                                      ("trust_all", np.ones(len(t)), 0.0, None, None)):
            use = (pp >= tau) | (rule3 if rule3 is not None else False)
            low = tw if tw is not None else tau
            te.loc[i, f"p_{n}"] = pp
            te.loc[i, f"v_{n}"] = np.where(use, "use", np.where(pp < low, "dont_use", "widen"))
    return te


def reliability(p: np.ndarray, y: np.ndarray, bins: int = 10) -> dict:
    ok = np.isfinite(p)
    p, y = p[ok], y[ok]
    b = np.clip((p * bins).astype(int), 0, bins - 1)
    curve = [{"bin": k / bins, "p_mean": float(p[b == k].mean()), "valid_rate": float(y[b == k].mean()),
              "n": int((b == k).sum())} for k in range(bins) if (b == k).any()]
    ece = sum(c["n"] * abs(c["p_mean"] - c["valid_rate"]) for c in curve) / max(len(p), 1)
    return {"curve": curve, "ece": float(ece)}


def filter_metrics(te: pd.DataFrame, name: str) -> dict:
    p, y, v = te[f"p_{name}"].to_numpy(float), te["valid"].to_numpy(float), te[f"v_{name}"].to_numpy()
    use, dont, inv = v == "use", v == "dont_use", y < 0.5
    nan = float("nan")
    return {"brier": float(np.nanmean((p - y) ** 2)), "precision_use": float(y[use].mean()) if use.any() else nan,
            "share_use": float(use.mean()), "share_widen": float((v == "widen").mean()),
            "recall_dont_use": float(dont[inv].mean()) if inv.any() else nan,
            "recall_flagged": float((~use)[inv].mean()) if inv.any() else nan, "n": int(len(y)),
            "valid_rate": float(y.mean()), **({"reliability": reliability(p, y)} if name != "trust_all" else {})}


def _week_day(te: pd.DataFrame, weeks: list) -> np.ndarray:
    start = {f"{w.year}-{w.month:02d}": pd.Timestamp(w.start) for w in weeks}
    return ((pd.to_datetime(te["day"]) - te["week"].map(start)).dt.days).to_numpy()


def timeline(te: pd.DataFrame, weeks: list, burn_in: int) -> pd.DataFrame:
    """Per (week, model, week day): any bucket alert, share of rows each filter does not use as-is, valid rate."""
    te = te.assign(wd=_week_day(te, weeks))
    agg = {"alert": ("health_alert", lambda s: float(np.nanmax(s.to_numpy(float)) > 0.5) if s.notna().any() else 0.0),
           "valid": ("valid", "mean"), "n": ("valid", "size")}
    for n in NAMES:
        te[f"flag_{n}"] = (te[f"v_{n}"] != "use").astype(float)
        agg[f"flag_{n}"] = (f"flag_{n}", "mean")
    return te.groupby(["week", "model", "wd"]).agg(**agg).reset_index()


def drill_metrics(tl: pd.DataFrame, target: str, drift_wd: int, restore_wd: int, flip: float = 0.5) -> dict:
    """Days from drift start to the first health alert / verdict flip (>= `flip` of the witness's rows not used),
    and from restore to the alert clearing (censored weeks reported apart)."""
    out: dict = {"alert_days": [], "recovery_days": [], "recovery_censored": 0, "alert_missed": 0}
    for n in NAMES:
        out[f"flip_days_{n}"] = []
    for wk, g in tl[tl["model"] == target].groupby("week"):
        g = g.sort_values("wd")
        on = g[(g["wd"] >= drift_wd) & (g["wd"] < restore_wd)]
        a = on[on["alert"] > 0.5]
        if len(a):
            out["alert_days"].append(int(a["wd"].iloc[0] - drift_wd))
            after = g[(g["wd"] >= restore_wd)]
            clear = after[after["alert"] < 0.5]
            if len(clear):
                out["recovery_days"].append(int(clear["wd"].iloc[0] - restore_wd))
            else:
                out["recovery_censored"] += 1
        else:
            out["alert_missed"] += 1
        for n in NAMES:
            f = on[on[f"flag_{n}"] >= flip]
            out[f"flip_days_{n}"].append(int(f["wd"].iloc[0] - drift_wd) if len(f) else None)
    return out


def false_alarms(tl: pd.DataFrame) -> dict:
    """Witness-level alert onsets (any bucket off -> on) per witness-month on weeks without drift."""
    res, tot = {}, [0, 0]
    for m, g in tl.groupby("model"):
        on = 0
        for _, w in g.sort_values("wd").groupby("week"):
            a = w["alert"].to_numpy() > 0.5
            on += int(a[0]) + int((a[1:] & ~a[:-1]).sum())
        d = g.groupby("week")["wd"].nunique().sum()
        res[m], tot[0], tot[1] = on / max(d / 30.0, 1e-9), tot[0] + on, tot[1] + d
    res["all"] = tot[0] / max(tot[1] / 30.0, 1e-9)  # onsets per witness-month, all witnesses pooled
    return res


def substitute_check(te: pd.DataFrame) -> dict:
    """Judge 'dont_use' rows that name a substitute (its ledger trust > the witness's P(valid), J4 rule 6): is the
    substitute's grade better (interval score below the witness's: skill_raw < 0)?"""
    t = te[(te["family"] == "forecast") & (te["v_judge"] == "dont_use")]
    if "sub_trust_main" not in t or not len(t):
        return {"n_named": 0}
    named = t[t["sub_trust_main"].to_numpy(float) > t["p_judge"].to_numpy(float)]
    sk = named["y_skill_raw"].to_numpy(float) if "y_skill_raw" in named else np.full(len(named), np.nan)
    ok = np.isfinite(sk)
    return {"n_dont_use": int(len(t)), "n_named": int(len(named)), "share_named": float(len(named) / len(t)),
            "sub_better": float((sk[ok] < 0).mean()) if ok.any() else float("nan"),
            "sub_better_unnamed": float((t.loc[~t.index.isin(named.index), "y_skill_raw"].to_numpy(float) < 0).mean())
            if len(t) > len(named) else float("nan")}


def cold_start(te: pd.DataFrame) -> dict:
    c = te[te["n_eff"].to_numpy(float) < 0.5]
    return {"n": int(len(c)), "share_use": float((c["v_judge"] == "use").mean()) if len(c) else float("nan"),
            "precision_use": float(c["valid"].mean()) if len(c) else float("nan")}


# ============================================================================================ main
def _mean(x) -> float:
    x = [v for v in x if v is not None]
    return float(np.mean(x)) if x else float("nan")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m sentinel.judge.evaluate", description=__doc__.split("\n")[0])
    ap.add_argument("--drift", default="none", choices=("none", *FORMS))
    ap.add_argument("--year", type=int, default=2025)
    ap.add_argument("--weeks", type=int, default=6)
    ap.add_argument("--week-days", type=int, default=28)
    ap.add_argument("--burn-in", type=int, default=21)
    ap.add_argument("--drift-day", type=int, default=3)
    ap.add_argument("--restore-day", type=int, default=14)
    ap.add_argument("--sites", default="all")
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--week-gb", type=float, default=7.0, help="memory budget per drill-week process")
    # J3 training (same defaults so its cached 2024 rows are reused)
    ap.add_argument("--train-weeks", type=int, default=12)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--rounds", type=int, default=1500)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--cal-share", type=float, default=0.3)
    ap.add_argument("--models", type=Path, default=JT.REPO / "models")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--quick", action="store_true", help="60 sites, 3+2 weeks, 2 seeds, 150 rounds")
    args = ap.parse_args(argv)
    if args.quick:
        args.sites, args.train_weeks, args.weeks, args.seeds, args.rounds = "60", 3, 2, 2, 150
    out = args.out or Path(os.environ.get("LAB_OUTPUT_DIR") or JT.REPO / "runs" / "judge_eval")
    out.mkdir(parents=True, exist_ok=True)
    timings: dict = {}
    gpu = JT.GpuSampler()
    gpu.start()
    t0 = time.perf_counter()

    with JT.Stage("data", timings):
        win = JT.ensure_windows()
    common = dict(burn_in=args.burn_in, sites=args.sites, models_dir=str(args.models), windows=str(win))
    with JT.Stage("rows_2024", timings):
        tr = R.generate(R.RowConfig(year=2024, seed=0, n_weeks=args.train_weeks, **common), 0, not args.no_cache)
    tr = tr[tr["graded"] > 0.5].reset_index(drop=True)
    fams, lo = {}, {}
    with JT.Stage("fit", timings):
        for fam in F.FAMILIES:
            t = tr[tr["family"] == fam].reset_index(drop=True)
            if len(t) and t["valid"].nunique() == 2:
                fams[fam], lo[fam] = JT.fit_and_calibrate(t, fam, args, "cpu")

    weeks = drill_weeks(args.year, args.drift, args.weeks, args.week_days, args.burn_in, args.drift_day,
                        args.restore_day)
    cfg = R.RowConfig(year=args.year, seed=1, n_weeks=args.weeks, week_days=args.week_days, drift_share=0,
                      cold_share=0, **common)
    with JT.Stage("rows_drill", timings):
        te = drill_rows(cfg, weeks, args.workers, args.week_gb, not args.no_cache)
    JT.emit("rows_week_peak_mb", te.attrs.get("week_peak_mb", 0.0))

    with JT.Stage("evaluate", timings):
        te = score(te, fams, lo)
        tl = timeline(te, weeks, args.burn_in)
        g = te[te["graded"] > 0.5]
        rep: dict = {"drift": args.drift, "weeks": [dataclasses.asdict(w) for w in weeks],
                     "filters": {n: filter_metrics(g, n) for n in NAMES},
                     "by_model": {m: {n: {k: v for k, v in filter_metrics(gm, n).items() if k != "reliability"}
                                      for n in ("judge", "ledger")} for m, gm in g.groupby("model")},
                     "cold_start": cold_start(te), "substitute": substitute_check(g)}
        if args.drift == "none":
            rep["false_alarms_per_witness_month"] = false_alarms(tl)
        else:
            dm = drill_metrics(tl, args.drift, args.drift_day, args.restore_day)
            wd = _week_day(g, weeks)
            gd = g[(g["model"] == args.drift) & (wd >= args.drift_day) & (wd < args.restore_day)]
            dm["drifted"] = {n: {k: v for k, v in filter_metrics(gd, n).items() if k != "reliability"}
                             for n in NAMES} if len(gd) else {}
            rep["drill"] = dm
        rep["timeline"] = tl.to_dict("records")

    for n, m in rep["filters"].items():
        for k in ("brier", "precision_use", "recall_dont_use", "share_use"):
            JT.emit(f"{n}_{k}", m[k])
        if "reliability" in m:
            JT.emit(f"{n}_ece", m["reliability"]["ece"])
    JT.emit("brier", rep["filters"]["judge"]["brier"])
    JT.emit("precision_use", rep["filters"]["judge"]["precision_use"])
    JT.emit("recall_dont_use", rep["filters"]["judge"]["recall_dont_use"])
    JT.emit("cold_share_use", rep["cold_start"]["share_use"])
    JT.emit("sub_better", rep["substitute"].get("sub_better"))
    if "false_alarms_per_witness_month" in rep:
        for m, v in rep["false_alarms_per_witness_month"].items():
            JT.emit(f"false_alarms_pwm_{m}", v)
    if "drill" in rep:
        dm = rep["drill"]
        JT.emit("days_to_alert", _mean(dm["alert_days"]))
        JT.emit("alert_missed", dm["alert_missed"])
        JT.emit("recovery_days", _mean(dm["recovery_days"]))
        JT.emit("recovery_censored", dm["recovery_censored"])
        for n in NAMES:
            JT.emit(f"flip_days_{n}", _mean(dm[f"flip_days_{n}"]))
            JT.emit(f"flip_missed_{n}", sum(v is None for v in dm[f"flip_days_{n}"]))
        for n, m in dm.get("drifted", {}).items():
            JT.emit(f"drifted_{n}_recall_dont_use", m["recall_dont_use"])
    gpu.stop.set()
    gpu.report()
    timings["t_total_s"] = round(time.perf_counter() - t0, 1)
    rep["timings"] = timings
    JT.emit("t_total_s", timings["t_total_s"])
    p = out / f"judge_eval_{args.drift}.json"
    p.write_text(json.dumps(rep, indent=1, default=str))
    for n in ("judge", "ledger"):
        print(f"[j5] reliability {n} " + json.dumps(rep["filters"][n]["reliability"]["curve"]), flush=True)
    for f in rep["filters"].values():
        f.pop("reliability", None)
    print("[j5] " + json.dumps({k: v for k, v in rep.items() if k not in ("timeline", "weeks")}, default=str),
          flush=True)
    print(f"[j5] wrote {p}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
