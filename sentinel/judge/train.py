"""Judge training (team/judge-design.md §7.4-§7.6, §9 metrics): `python train.py --model judge [flags]`.

    python train.py --model judge                         # 12 weeks of 2024 -> train, 12 weeks of 2025 -> test
    python train.py --model judge --quick                 # smoke test: few sites / weeks / seeds (minutes, local)
    python train.py --model judge --bench-device          # also time LightGBM on CPU (all threads) vs GPU
    python -m sentinel.judge.train --apply LOG_OR_DIR     # install a run's artifacts into models/

Stages (each logs LAB_METRIC t_<stage>_s; gpu_util is sampled through the whole run):
  data        the #10 window cache (built once into $LAB_CACHE_DIR/sentinel when missing)
  rows_2024   training rows: sampled weeks replayed in a process pool (sentinel.judge.rows), cached by code+config
  rows_2025   test rows, same recipe with another seed (drift + cold starts included; "clean" = weeks without)
  fit_<fam>   LightGBM per family, 5 seeds, on 70 % of the 2024 assets (15 % of those for early stopping)
  calibrate   isotonic on the held-out 30 % of assets; tau_use (>= 90 % precision) and tau_widen / widen_factor
  evaluate    2025: Judge vs ledger-only (isotonic on trust, same threshold rule) vs trust-everything
Primary metrics (2025, all witnesses pooled, verdicts incl. rule 3 "no record -> use"): brier, precision_use,
recall_dont_use; per family and per baseline with prefixes. Artifacts in --out ($LAB_OUTPUT_DIR): judge.<family>.s<k>
.txt (LightGBM text), judge.calibration.json, judge_report.json; the same files are also printed to the log
(JUDGE_FILE lines, gzip+base64) so `--apply` can install them from a saved log of a worker we cannot reach.
"""

from __future__ import annotations

import argparse
import base64
import gzip
import json
import os
import shutil
import subprocess
import sys
import threading
import time
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

from sentinel.judge import features as F
from sentinel.judge import model as JM
from sentinel.judge import rows as R

REPO = Path(__file__).resolve().parents[2]
LOG_TAG = "JUDGE_FILE"


def emit(name: str, value: float) -> None:
    from sentinel.infra.metrics import emit as _emit
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return
    _emit(name, float(value))


class Stage:
    def __init__(self, name: str, timings: dict):
        self.name, self.timings = name, timings

    def __enter__(self):
        self.t = time.perf_counter()
        return self

    def __exit__(self, *exc):
        dt = time.perf_counter() - self.t
        self.timings[f"t_{self.name}_s"] = round(dt, 2)
        emit(f"t_{self.name}_s", dt)


class GpuSampler(threading.Thread):
    """nvidia-smi utilisation every second (0 when there is no GPU): LAB_METRIC gpu_util (mean %) / gpu_util_max."""

    def __init__(self):
        super().__init__(daemon=True)
        self.samples: list[float] = []
        self.stop = threading.Event()
        self.ok = shutil.which("nvidia-smi") is not None

    def run(self):
        while self.ok and not self.stop.is_set():
            try:
                out = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
                                     capture_output=True, text=True, timeout=5).stdout
                vals = [float(x) for x in out.split() if x.strip().replace(".", "").isdigit()]
                if vals:
                    self.samples.append(max(vals))
            except Exception:
                self.ok = False
            self.stop.wait(1.0)

    def report(self) -> None:
        emit("gpu_util", float(np.mean(self.samples)) if self.samples else 0.0)
        emit("gpu_util_max", float(np.max(self.samples)) if self.samples else 0.0)


# ============================================================================================ data
def ensure_windows(log=print) -> Path:
    from sentinel.agent.world import find_windows
    if os.environ.get("LAB_CACHE_DIR") and not os.environ.get("SENTINEL_CACHE"):
        lab_cache = Path(os.environ["LAB_CACHE_DIR"]) / "sentinel"
        if find_windows() is None:  # the worker has no shared cache: build (once) under $LAB_CACHE_DIR
            os.environ["SENTINEL_CACHE"] = str(lab_cache)
    w = find_windows()
    if w is None:
        from sentinel.model.data import build_lab_windows
        log("[judge] no #10 window cache: building fleet + faults + windows (one-off)")
        w = build_lab_windows(verbose=True)
    if w is None:
        raise SystemExit("judge: no window cache and it could not be built")
    log(f"[judge] windows: {w}")
    try:  # M6 rows need #22/#23's wind data (built once, ~1 min); without it the detector family is M4 only
        from sentinel.model.wind.data import external_wind_dir
        if external_wind_dir() is None:
            from sentinel.sim.wind.build import build
            log("[judge] no wind cache: building it with sentinel.sim.wind (one-off)")
            build(plots=False)
        log(f"[judge] wind: {external_wind_dir()}")
    except Exception as e:
        log(f"[judge] wind data unavailable ({type(e).__name__}: {e}); M6 rows skipped")
    return Path(w)


def cal_assets(asset_ids: pd.Series, share: float = 0.3) -> np.ndarray:
    """Deterministic 30 % of assets for calibration (crc32 of the id), the rest for fitting."""
    return asset_ids.astype(str).map(lambda a: (zlib.crc32(a.encode()) % 1000) < share * 1000).to_numpy()


# ============================================================================================ fit + calibrate
def fit_and_calibrate(tr: pd.DataFrame, fam: str, args, device: str, log=print) -> tuple[JM.FamilyModel, dict]:
    X, y = F.build(tr, fam), tr["valid"].to_numpy(float)
    cal = cal_assets(tr["asset_id"], args.cal_share)
    # early stopping on whole held-out weeks (one sampled week per month): with 6 drift episodes, held-out assets of
    # the same weeks reward memorising each week (train#908 ran to the 1500-round cap and lost test AUC)
    es_groups = tr.loc[~cal, "month"].to_numpy() if "month" in tr else tr.loc[~cal, "asset_id"].to_numpy()
    boosters, info = JM.fit_family(X[~cal], y[~cal], es_groups, fam,
                                   seeds=range(args.seeds), rounds=args.rounds, threads=args.threads, device=device,
                                   log=log)
    fm = JM.FamilyModel(fam, boosters, list(X.columns), JM.Isotonic())
    raw = fm.raw(X[cal]).mean(0)
    fm.iso = JM.Isotonic.fit(raw, y[cal])
    p_cal = fm.iso(raw)
    fm.tau_use = JM.tau_for_precision(p_cal, y[cal])
    hit_m = {m: tr.loc[cal, f"y_hit_m{m:g}"].to_numpy(float) for m in R.MULTS} if fam == "forecast" else {}
    if fam == "forecast":
        fm.tau_widen, fm.widen_factor = JM.widen_choice(p_cal, hit_m, tr.loc[cal, "band_scale"].to_numpy(float),
                                                        fm.tau_use)
    else:
        fm.tau_widen, fm.widen_factor = fm.tau_use, 1.0
    # ledger-only baseline on the same held-out assets: isotonic on trust_main, same threshold rule
    t_cal = tr.loc[cal, "trust_main"].to_numpy(float)
    lo_iso = JM.Isotonic.fit(np.nan_to_num(t_cal, nan=0.5), y[cal])
    lo_p = lo_iso(np.nan_to_num(t_cal, nan=0.5))
    lo_tau = JM.tau_for_precision(lo_p, y[cal])
    lo_widen, lo_factor = JM.widen_choice(lo_p, hit_m, tr.loc[cal, "band_scale"].to_numpy(float), lo_tau) \
        if fam == "forecast" else (None, 1.0)
    gain = {}
    for b in boosters[:1]:
        imp = b.feature_importance("gain")
        gain = {f: round(float(v), 1) for f, v in sorted(zip(fm.features, imp), key=lambda x: -x[1])[:12]}
    fm.info = {**info, "n_cal": int(cal.sum()), "valid_rate_train": float(y.mean()), "top_gain": gain,
               "brier_cal": float(np.mean((p_cal - y[cal]) ** 2)),
               "ledger_only": {"isotonic": {"x": lo_iso.x, "y": lo_iso.y}, "tau_use": lo_tau, "tau_widen": lo_widen,
                               "widen_factor": lo_factor}}
    log(f"[judge] {fam}: {len(tr)} rows (valid {y.mean():.2f}), cal {int(cal.sum())}, best iters "
        f"{info['best_iterations']}, tau_use {fm.tau_use:.3f}, tau_widen {fm.tau_widen:.3f} x{fm.widen_factor:g}, "
        f"ledger-only tau {lo_tau:.3f}; top gain {list(gain)[:6]}")
    return fm, {"iso": lo_iso, "tau": lo_tau, "widen": lo_widen, "factor": lo_factor}


def evaluate(te: pd.DataFrame, fams: dict, lo: dict, prefix: str = "") -> dict:
    """Judge vs ledger-only vs trust-everything on test rows. Returns {name: metrics}."""
    res, pooled = {}, {"judge": ([], [], [], []), "ledger": ([], [], [], []), "trust_all": ([], [], [], [])}
    for fam, fm in fams.items():
        t = te[te["family"] == fam]
        if t.empty:
            continue
        y = t["valid"].to_numpy(float)
        p, _ = fm.predict(F.build(t, fam))
        rule3 = t["n_eff"].to_numpy(float) < 0.5
        tw = fm.tau_widen if fam == "forecast" else None
        tl = lo[fam]["iso"](np.nan_to_num(t["trust_main"].to_numpy(float), nan=0.5))
        res[f"{fam}_judge"] = JM.verdict_metrics(p, y, fm.tau_use, rule3, tw)
        res[f"{fam}_judge_norule3"] = JM.verdict_metrics(p, y, fm.tau_use, None, tw)
        res[f"{fam}_ledger"] = JM.verdict_metrics(tl, y, lo[fam]["tau"], rule3, lo[fam]["widen"])
        res[f"{fam}_trust_all"] = JM.verdict_metrics(np.ones_like(y), y, 0.0)
        res[f"{fam}_judge"]["auc"] = _auc(p, y)
        res[f"{fam}_ledger"]["auc"] = _auc(tl, y)
        if fam == "forecast" and all(f"y_hit_m{m:g}" in t for m in R.MULTS):
            hit_m = {m: t[f"y_hit_m{m:g}"].to_numpy(float) for m in R.MULTS}
            bs = t["band_scale"].to_numpy(float)
            for name, pp, tau, w, fac in (("judge", p, fm.tau_use, tw, fm.widen_factor),
                                          ("ledger", tl, lo[fam]["tau"], lo[fam]["widen"], lo[fam].get("factor", 1.0))):
                zone = (pp >= w) & (pp < tau) & ~rule3 if w is not None else np.zeros(len(pp), bool)
                if zone.any():  # does widening deliver the promised 0.80 calibration on the test year?
                    res[f"{fam}_{name}"]["widen_cal"] = float(np.nanmean(
                        JM.calibration_score(JM.widened_hit(hit_m, bs, fac))[zone]))
                    res[f"{fam}_{name}"]["widen_cal_raw"] = float(np.nanmean(
                        JM.calibration_score(t["y_hit"].to_numpy(float))[zone])) if "y_hit" in t else float("nan")
        for name, pp, tau, r3, w in (("judge", p, fm.tau_use, rule3, tw),
                                     ("ledger", tl, lo[fam]["tau"], rule3, lo[fam]["widen"]),
                                     ("trust_all", np.ones_like(y), 0.0, None, None)):
            use = (pp >= tau) | (r3 if r3 is not None else False)
            dont = ~use & (pp < (w if w is not None else tau))
            pooled[name][0].append(pp), pooled[name][1].append(y), pooled[name][2].append(use)
            pooled[name][3].append(dont)
        # per witness, judge only (diagnostics)
        for m, tm in t.groupby("model"):
            ym = tm["valid"].to_numpy(float)
            pm, _ = fm.predict(F.build(tm, fam))
            res[f"{m}_judge"] = JM.verdict_metrics(pm, ym, fm.tau_use, tm["n_eff"].to_numpy(float) < 0.5, tw)
            res[f"{m}_ledger"] = JM.verdict_metrics(lo[fam]["iso"](np.nan_to_num(tm["trust_main"].to_numpy(float),
                                                                                 nan=0.5)), ym, lo[fam]["tau"],
                                                    tm["n_eff"].to_numpy(float) < 0.5, lo[fam]["widen"])
    for name, (ps, ys, us, ds) in pooled.items():
        if not ps:
            continue
        p, y, use, dont = map(np.concatenate, (ps, ys, us, ds))
        inv = y < 0.5
        res[f"all_{name}"] = {"brier": float(np.mean((p - y) ** 2)), "precision_use": float(y[use].mean())
                              if use.any() else float("nan"), "recall_dont_use": float(dont[inv].mean())
                              if inv.any() else float("nan"), "recall_flagged": float((~use)[inv].mean())
                              if inv.any() else float("nan"), "share_use": float(use.mean()),
                              "share_dont_use": float(dont.mean()), "valid_rate": float(y.mean()), "n": float(len(y))}
    return res


def _auc(p: np.ndarray, y: np.ndarray) -> float:
    try:
        from sklearn.metrics import roc_auc_score
        return float(roc_auc_score(y, p)) if 0 < y.mean() < 1 else float("nan")
    except Exception:
        return float("nan")


# ============================================================================================ device bench
def bench_device(tr: pd.DataFrame, args, log=print) -> dict:
    """Same data, same rounds (no early stopping), one seed: CPU with all threads vs LightGBM's GPU build."""
    import lightgbm as lgb
    fam = "forecast" if (tr["family"] == "forecast").any() else "detector"
    t = tr[tr["family"] == fam]
    X, y = F.build(t, fam), t["valid"].to_numpy(float)
    out = {"bench_rows": len(t), "threads": os.cpu_count()}
    for dev in ("cpu", "gpu", "cuda"):
        params = {**JM.DEFAULT_PARAMS, "seed": 0, "num_threads": 0, "device_type": dev}
        if dev == "gpu":
            params["max_bin"] = 63  # LightGBM's GPU guidance; CPU keeps 255
        try:
            ds = lgb.Dataset(X, label=y, categorical_feature=[c for c in F.CATEGORICAL if c in X.columns])
            t0 = time.perf_counter()
            lgb.train(params, ds, num_boost_round=args.bench_rounds)
            out[f"t_fit_{dev}_s"] = round(time.perf_counter() - t0, 2)
            out[f"{dev}_ok"] = 1
        except Exception as e:
            out[f"{dev}_ok"] = 0
            out[f"{dev}_error"] = f"{type(e).__name__}: {str(e)[:200]}"
        log(f"[judge] bench {dev}: {out.get(f't_fit_{dev}_s', 'n/a')} s {out.get(f'{dev}_error', '')}")
    for dev in ("cpu", "gpu", "cuda"):
        if f"t_fit_{dev}_s" in out:
            emit(f"t_fit_{dev}_s", out[f"t_fit_{dev}_s"])
        emit(f"{dev}_ok", out[f"{dev}_ok"])
    gpu_t = min([out[k] for k in ("t_fit_gpu_s", "t_fit_cuda_s") if k in out], default=None)
    if gpu_t is not None and "t_fit_cpu_s" in out:
        out["gpu_speedup"] = round(out["t_fit_cpu_s"] / max(gpu_t, 1e-6), 3)
        emit("gpu_speedup", out["gpu_speedup"])
    return out


# ============================================================================================ artifacts
def dump_files(files: list[Path]) -> None:
    for p in files:
        blob = base64.b64encode(gzip.compress(p.read_bytes(), 9)).decode()
        print(f"{LOG_TAG} {p.name} {blob}", flush=True)


def apply(src: Path, models: Path) -> list[Path]:
    """Install artifacts from a run output dir or a saved `lab logs` text into `models`."""
    models.mkdir(parents=True, exist_ok=True)
    written = []
    if src.is_dir():
        for p in sorted(src.glob("judge.*")):
            if p.suffix in (".txt", ".json") and p.name != "judge_report.json":
                shutil.copy2(p, models / p.name)
                written.append(models / p.name)
        return written
    for line in src.read_text().splitlines():
        if LOG_TAG in line:
            _, name, blob = line[line.index(LOG_TAG):].split(" ", 2)
            (models / name).write_bytes(gzip.decompress(base64.b64decode(blob.strip())))
            written.append(models / name)
    return written


# ============================================================================================ main
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="train.py --model judge", description=__doc__.split("\n")[0])
    ap.add_argument("--train-year", type=int, default=2024)
    ap.add_argument("--test-year", type=int, default=2025)
    ap.add_argument("--weeks", type=int, default=12)
    ap.add_argument("--test-weeks", type=int, default=12)
    ap.add_argument("--sites", default="all", help="all | n (first n sites, 2/3 seen + 1/3 unseen)")
    ap.add_argument("--burn-in", type=int, default=21)
    ap.add_argument("--workers", type=int, default=0, help="row-generation processes (0 = one per week, <= cores)")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--rounds", type=int, default=1500)
    ap.add_argument("--threads", type=int, default=0, help="LightGBM threads (0 = all cores)")
    ap.add_argument("--device", default="cpu", help="cpu | gpu | auto (gpu only if --bench-device shows it faster)")
    ap.add_argument("--bench-device", action="store_true")
    ap.add_argument("--bench-rounds", type=int, default=300)
    ap.add_argument("--cal-share", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--models", type=Path, default=REPO / "models")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--no-dump", action="store_true", help="do not print the artifacts into the log")
    ap.add_argument("--quick", action="store_true", help="smoke test: 60 sites, 3+2 weeks, 2 seeds, 150 rounds")
    ap.add_argument("--apply", type=Path, default=None, help="install artifacts from a run dir or saved log")
    args = ap.parse_args(argv)
    if args.apply:
        for p in apply(args.apply, args.models):
            print(f"installed {p}")
        return 0
    if args.quick:
        args.sites, args.weeks, args.test_weeks, args.seeds, args.rounds = "60", 3, 2, 2, 150
        args.burn_in = min(args.burn_in, 14)
    out = args.out or Path(os.environ.get("LAB_OUTPUT_DIR") or REPO / "runs" / "judge")
    out.mkdir(parents=True, exist_ok=True)
    timings: dict = {}
    gpu = GpuSampler()
    gpu.start()
    t_all = time.perf_counter()
    try:
        from sentinel.kernels import backend
        print(f"[judge] kernels backend: {backend()}; cores {os.cpu_count()}; LAB_CACHE_DIR "
              f"{os.environ.get('LAB_CACHE_DIR')}", flush=True)
        mem = Path("/proc/meminfo")
        if mem.exists():
            print("[judge] " + " ".join(mem.read_text().splitlines()[:3]), flush=True)
    except Exception:
        pass

    with Stage("data", timings):
        win = ensure_windows()
    common = dict(n_weeks=args.weeks, burn_in=args.burn_in, sites=args.sites, models_dir=str(args.models),
                  windows=str(win))
    with Stage(f"rows_{args.train_year}", timings):
        tr = R.generate(R.RowConfig(year=args.train_year, seed=args.seed, **common), args.workers,
                        not args.no_cache)
    common["n_weeks"] = args.test_weeks
    with Stage(f"rows_{args.test_year}", timings):
        te = R.generate(R.RowConfig(year=args.test_year, seed=args.seed + 1, **common), args.workers,
                        not args.no_cache)
    n_tr0, n_te0 = len(tr), len(te)
    emit("rows_week_peak_mb", max(tr.attrs.get("week_peak_mb", 0.0), te.attrs.get("week_peak_mb", 0.0)))
    tr, te = tr[tr["graded"] > 0.5].reset_index(drop=True), te[te["graded"] > 0.5].reset_index(drop=True)
    emit("rows_train", len(tr)), emit("rows_test", len(te))
    emit("rows_ungraded_share", 1 - (len(tr) + len(te)) / max(n_tr0 + n_te0, 1))
    print(f"[judge] rows: train {len(tr)} (of {n_tr0}), test {len(te)} (of {n_te0}); valid rate train "
          f"{tr['valid'].mean():.3f} test {te['valid'].mean():.3f}", flush=True)

    bench = {}
    device = "cpu" if args.device == "auto" else args.device
    if args.bench_device:
        with Stage("bench_device", timings):
            bench = bench_device(tr, args)
        if args.device == "auto" and bench.get("gpu_speedup", 0) > 1.0 and bench.get("gpu_ok"):
            device = "gpu"
    print(f"[judge] training device: {device}", flush=True)

    fams, lo = {}, {}
    for fam in F.FAMILIES:
        t = tr[tr["family"] == fam].reset_index(drop=True)
        if t.empty or t["valid"].nunique() < 2:
            print(f"[judge] {fam}: no usable rows, skipped", flush=True)
            continue
        with Stage(f"fit_{fam}", timings):
            fams[fam], lo[fam] = fit_and_calibrate(t, fam, args, device)
    if not fams:
        raise SystemExit("judge: nothing to train")

    with Stage("evaluate", timings):
        res = evaluate(te, fams, lo)
        clean = te[(te["drift_witness"] == "") & (te["cold_week"] < 0.5)]
        res_clean = evaluate(clean, fams, lo) if len(clean) else {}
        drifted = te[te["drifted"] > 0.5]
        res_drift = evaluate(drifted, fams, lo) if len(drifted) else {}
    for name, m in res.items():
        for k, v in m.items():
            if k != "n":
                emit(f"{name}_{k}", v)
    if "all_judge" in res:
        for k in ("brier", "precision_use", "recall_dont_use", "recall_flagged", "share_use"):
            emit(k, res["all_judge"][k])
            emit(f"ledger_{k}", res["all_ledger"][k])
            emit(f"trust_all_{k}", res["all_trust_all"][k])
    for tag, rr in (("clean", res_clean), ("drifted", res_drift)):
        for name in ("all_judge", "all_ledger", "all_trust_all"):
            if name in rr:
                for k in ("brier", "precision_use", "recall_dont_use", "recall_flagged"):
                    emit(f"{tag}_{name}_{k}", rr[name][k])

    # thresholds + artifacts
    sha = os.environ.get("LAB_COMMIT") or _git_sha()
    run = os.environ.get("LAB_RUN_ID") or "local"
    head = fams.get("forecast") or next(iter(fams.values()))
    cal = {"judge_version": f"judge-{run}-{sha[:7]}", "source_run": f"train#{run}" if run != "local" else "local",
           "git_sha": sha, "tau_use": head.tau_use, "tau_widen": head.tau_widen, "widen_factor": head.widen_factor,
           "precision_target": JM.PRECISION_USE, "widen_cal_target": JM.WIDEN_CAL, "device": device,
           "rows": {"train": len(tr), "test": len(te), "train_year": args.train_year, "test_year": args.test_year,
                    "weeks": args.weeks, "test_weeks": args.test_weeks, "sites": args.sites, "burn_in": args.burn_in},
           "ledger": "sentinel.judge.ledger", "grader": "sentinel.judge.grader", "metrics_2025": dict(res)}
    judge = JM.Judge(fams, cal)
    files = judge.save(out)
    # reasons smoke check on a few test rows (fails loudly if the reason path is broken)
    sample = te.groupby("family").head(3)
    pr = judge.predict(sample)
    for (_, r), (_, o) in zip(sample.iterrows(), pr.iterrows()):
        print(f"[judge] example {r['model']} {r['asset_id']} {r['day']}: valid={r['valid']:.0f} p={o['p_valid']:.2f} "
              f"sd={o['p_valid_sd']:.3f} reasons={o['reasons']}", flush=True)
    for fam, fm in fams.items():
        emit(f"{fam}_tau_use", fm.tau_use), emit(f"{fam}_tau_widen", fm.tau_widen)
        emit(f"{fam}_widen_factor", fm.widen_factor)
    report = {"timings": timings, "bench": bench, "metrics": res, "metrics_clean": res_clean,
              "metrics_drifted": res_drift, "calibration": cal,
              "families": {f: {"tau_use": m.tau_use, "tau_widen": m.tau_widen, "widen_factor": m.widen_factor,
                               **m.info} for f, m in fams.items()}}
    (out / "judge_report.json").write_text(json.dumps(report, indent=1, default=float) + "\n")
    emit("artifact_bytes", sum(p.stat().st_size for p in files))
    if not args.no_dump:
        dump_files(files)
    gpu.stop.set()
    gpu.report()
    emit("t_total_s", time.perf_counter() - t_all)
    print(f"[judge] wrote {', '.join(p.name for p in files)} to {out}", flush=True)
    return 0


def _git_sha() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=REPO).stdout.strip()
    except Exception:
        return "unknown"


if __name__ == "__main__":
    sys.exit(main())
