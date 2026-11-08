"""Judge priors: what each model version and substitute achieved on its 2024 validation year, per context bucket
(the trust used when the ledger has no record yet). Forecasts get calibration / sharpness, detectors alarm_precision /
miss_score, scored on the synthetic fleet; M6, M3, M7, M8 come from their reports (source stated in the JSON).

    python -m wattsentry.judge.priors --out DIR [--quick] [--sites 120 --days 30]
    python -m wattsentry.judge.priors --apply RUN_OUTPUT_DIR_OR_SAVED_LOG             # install into models/
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from wattsentry.infra.metrics import emit
from wattsentry.judge.buckets import ROOT, bucket_key, season_of, truncate
from wattsentry.judge.rows import interval_score

VAL_YEAR = 2024
H = 96
HIST_DAYS = 14
TARGET = 0.80
SPLIT = f"val-{VAL_YEAR}"
REPO = Path(__file__).resolve().parents[2]

PROMOTED = {"pvnet": [f"pvnet.s{i}" for i in range(4)], "loadnet": [f"loadnet.s{i}" for i in range(4)],
            "sitehealth": ["sitehealth"]}
OUT_FILES = ("promoted_priors.json", "windnbm.priors.json", "judge_priors.json")
LOG_TAG = "JUDGE_PRIORS_FILE"
PV_REF = {"pvnet": "lgbm_pv", "lgbm_pv": "pvlib", "pvlib": "_persist_d1"}
LOAD_REF = {"loadnet": "lgbm_load", "lgbm_load": "persistence_load", "persistence_load": "_persist_w1"}


# ---------------------------------------------------------------- scores (vectorised over [cases, 96])

def _masked_mean(x: np.ndarray, m: np.ndarray) -> np.ndarray:
    m = m & np.isfinite(x)
    n = m.sum(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(n > 0, np.where(m, x, 0.0).sum(1) / np.maximum(n, 1), np.nan)


def calibration_score(q10, q90, y, mask) -> np.ndarray:
    """1 - |hit - 0.80| / 0.80 per row; NaN where the row has no scored step."""
    hit = _masked_mean(((y >= q10) & (y <= q90)).astype(float), mask & np.isfinite(y) & np.isfinite(q10))
    return np.clip(1 - np.abs(hit - TARGET) / TARGET, 0, 1)


def sharpness_score(q10, q90, y, rq10, rq90, mask) -> np.ndarray:
    """clip(1 - IS_model / IS_ref, 0, 1) per row, IS averaged over the masked steps."""
    ok = mask & np.isfinite(y) & np.isfinite(q10) & np.isfinite(rq10)
    im = _masked_mean(interval_score(q10, q90, y), ok)
    ir = _masked_mean(interval_score(rq10, rq90, y), ok)
    with np.errstate(invalid="ignore", divide="ignore"):
        s = np.where(ir > 0, 1 - im / ir, 0.0)  # a perfect reference cannot be beaten
    return np.where(np.isfinite(im) & np.isfinite(ir), np.clip(s, 0, 1), np.nan)


def detector_scores(alarm: np.ndarray, called: list[str], truth: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """(alarm_precision, miss_score) per day, NaN where that side is not graded."""
    alarm = np.asarray(alarm, bool)
    truth_a, called_a = np.asarray(truth, object), np.asarray(called, object)
    faulty = truth_a != "healthy"
    ap = np.where(called_a == truth_a, 1.0, np.where(faulty, 0.5, 0.0))
    ms = np.where(faulty, 0.0, 1.0)
    return np.where(alarm, ap, np.nan), np.where(alarm, np.nan, ms)


# ---------------------------------------------------------------- aggregation per bucket and ancestors

def aggregate(rows: pd.DataFrame) -> dict[str, dict[str, dict[str, dict[str, float]]]]:
    """rows: model, score_name, bucket (full key), score -> {model: {score: {key: {prior, n}}}} for every ancestor
    key of every bucket down to the root."""
    rows = rows[np.isfinite(rows.score.to_numpy(float))]
    out: dict = {}
    if rows.empty:
        return out
    full = rows.bucket.astype(str)
    depth = int(full.str.count(r"\|").max()) + 1
    allr = pd.concat([rows.assign(key=full.map(lambda b, lv=lv: truncate(b, lv)).to_numpy())
                      for lv in range(depth, -1, -1)], ignore_index=True)
    g = allr.groupby(["model", "score_name", "key"]).score.agg(["mean", "size"]).reset_index()
    for r in g.itertuples(index=False):
        out.setdefault(r.model, {}).setdefault(r.score_name, {})[r.key] = {"prior": round(float(r.mean), 5),
                                                                          "n": int(r.size)}
    return out


def priors_block(scores: dict, source: str, run: str, **extra) -> dict:
    return {"version": 1, "split": SPLIT, "source_run": run, "source": source, **extra, "scores": scores}


def overall(scores: dict[str, float], n: int = 0, kind: str | None = None) -> dict:
    """Overall-only priors: {score: {"*": ..., kind: ...}}."""
    out = {}
    for s, v in scores.items():
        d = {ROOT: {"prior": round(float(v), 5), "n": int(n)}}
        if kind:
            d[kind] = dict(d[ROOT])
        out[s] = d
    return out


# ---------------------------------------------------------------- report-based priors (wind, M3, M7, M8)

def _wind_from(block: dict) -> dict[str, float]:
    p, c = block["wind_precision"], block["class_correct_recall_macro"]
    missed = block["true_actionable_per_turbine_day"] * (1 - block["wind_recall"])
    normal_days = max(1e-6, 1 - block["alerts_per_turbine_day"])
    return {"alarm_precision": p * (0.5 + 0.5 * c), "miss_score": float(np.clip(1 - missed / normal_days, 0, 1))}


def wind_priors(report_path: Path) -> tuple[dict, dict, str]:
    r = json.loads(report_path.read_text())
    nbm, mon = {}, {}
    val, seen, held = r["validation"], r["seen_parks_test"]["m6"], r["m6"]
    for s, v in _wind_from(val).items():
        nbm[s] = {ROOT: {"prior": round(v, 5), "n": int(val["n_alerts"])},
                  "turbine": {"prior": round(v, 5), "n": int(val["n_alerts"])}}
    for key, blk in (("turbine|seen", seen), ("turbine|unseen", held)):
        for s, v in _wind_from(blk).items():
            nbm[s][key] = {"prior": round(v, 5), "n": int(blk["n_alerts"])}
    for key, blk in ((ROOT, r["baseline_windmon"]), ("turbine", r["baseline_windmon"]),
                     ("turbine|seen", r["seen_parks_test"]["windmon"]), ("turbine|unseen", r["baseline_windmon"])):
        for s, v in _wind_from(blk).items():
            mon.setdefault(s, {})[key] = {"prior": round(v, 5), "n": int(blk["n_alerts"])}
    return nbm, mon, r.get("model_version", "windnbm")


def ledger_only_priors(run: str) -> dict[str, dict]:
    """M3, M7, M8 and their substitutes (overall priors, source stated)."""
    cov, nmae, nmae_p = 0.74, 0.039, 0.067  # DemandNet on real RO load, test 2025
    flag_prec = (0.833 * 54 + 16) / (0.833 * 54 / 0.692 + 16)  # phase + feeder flag precision of M8
    fa_days = lambda fa_per_year: fa_per_year * 14 / 365  # a false alarm stays raised ~14 days  # noqa: E731
    return {
        "demandnet": dict(model_version="demandnet-44", source="report",
                          note="#44 local run (insights): 10-90 coverage 0.74, nMAE 0.039 vs persistence 0.067",
                          scores=overall({"calibration": 1 - abs(cov - TARGET) / TARGET,
                                          "sharpness": 1 - nmae / nmae_p}, kind="zone")),
        "persistence_zone": dict(model_version="persistence-1", source="default",
                                 note="zero-width band: hit ~ 0 -> calibration 0; it is the sharpness reference",
                                 scores=overall({"calibration": 0.0, "sharpness": 0.0}, kind="zone")),
        "windfailure": dict(model_version="windfailure-379", source="report",
                            note="results.md row 12 (train#379, syn RO): recall 0.98, 0.42 FA/turbine-year; "
                                 "CARE (non-RO) scored 0.40",
                            scores=overall({"hit": 1 - fa_days(0.42) - 0.02 * 14 / 365}, kind="turbine")),
        "residual_trend": dict(model_version="residual-threshold", source="report",
                               note="results.md row 12 baseline: recall 0.87, 0.39 FA/turbine-year",
                               scores=overall({"hit": 1 - fa_days(0.39) - 0.13 * 14 / 365}, kind="turbine")),
        "gridtwin": dict(model_version="gridtwin-506", source="report",
                         note="results.md row 10 (replay#506): phase precision 0.692 (54 corrupted, 83 % found), "
                              "feeder precision 1.0 (16)",
                         scores=overall({"flag": flag_prec}, kind="feeder")),
        "gis_recorded": dict(model_version="gis-as-is", source="default", note="never flags; neutral default",
                             scores=overall({"flag": 0.5}, kind="feeder")),
    }


# ---------------------------------------------------------------- validation scoring on the fleet

def _fleet(quick: bool) -> tuple[Path, Path]:
    from wattsentry.sim.faults.dataset import inject_dataset
    from wattsentry.sim.fleet.config import FleetConfig
    from wattsentry.sim.fleet.generate import generate
    cfg = FleetConfig(n_sites=120, n_feeders=6, n_congested=3, start_year=VAL_YEAR, end_year=VAL_YEAR) if quick \
        else FleetConfig()
    clean = generate(cfg, plots=False)
    return clean, inject_dataset(clean)


def _load_year(d: Path, ids: list[str], year: int) -> dict[str, dict]:
    import pyarrow.parquet as pq
    from eval import TEL_COLS
    t = pq.read_table(d / "telemetry" / f"year={year}.parquet", columns=TEL_COLS,
                      filters=[("site_id", "in", ids)]).to_pandas()
    t["site_id"] = t.site_id.astype(str)
    out = {}
    for sid, g in t.groupby("site_id", sort=False):
        g = g.sort_values("ts")
        out[sid] = {c: g[c].to_numpy(np.float64) for c in TEL_COLS[2:]}
        out[sid]["ts"] = pd.DatetimeIndex(g.ts)
    return out


def _quantiles(preds: list) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    nan = np.full(H, np.nan)
    q = [np.stack([np.asarray(getattr(p, k), float) if p is not None else nan for p in preds])
         for k in ("q10", "q50", "q90")]
    ood = np.array([float(p.ood_score) if p is not None else np.nan for p in preds])
    return q[0], q[1], q[2], ood


def score_validation(args) -> tuple[dict, dict]:
    """-> (scores per model id, model_version per model id)."""
    import eval as E
    clean_dir, faulty_dir = _fleet(args.quick)
    sites = pd.read_parquet(clean_dir / "sites.parquet")
    sites["site_id"] = sites.site_id.astype(str)
    labels = pd.read_parquet(faulty_dir / "fault_labels.parquet")
    labels["asset_id"] = labels.asset_id.astype(str)
    chosen = pd.concat([g.sample(min(len(g), args.sites // 2), random_state=args.seed)
                        for _, g in sites.groupby("split")])
    ids = list(chosen.site_id)
    clean, faulty = _load_year(clean_dir, ids, VAL_YEAR), _load_year(faulty_dir, ids, VAL_YEAR)
    wx = E.load_weather(clean_dir, set(chosen.locality_id.astype(str)))
    n_days = min(len(v["ts"]) for v in clean.values()) // H
    rng = np.random.default_rng(args.seed)
    cases = [(r, int(d)) for _, r in chosen.iterrows()
             for d in rng.choice(np.arange(HIST_DAYS + 1, n_days), min(args.days, n_days - HIST_DAYS - 1),
                                 replace=False)]
    print(f"priors: {len(ids)} sites x {args.days} days = {len(cases)} site-days ({VAL_YEAR}), fleet {clean_dir.name}")
    preds = E.load_predictors(["onnx", "baselines"], args.models, clean_dir)
    pv_models = {"pvnet": preds.get("pvnet", {}).get("pv"), "lgbm_pv": preds.get("lgbm", {}).get("pv"),
                 "pvlib": preds.get("pvlib", {}).get("pv")}
    load_models = {"loadnet": preds.get("loadnet", {}).get("load"), "lgbm_load": preds.get("lgbm", {}).get("load"),
                   "persistence_load": preds.get("persistence", {}).get("load")}
    det_models = {"sitehealth": preds.get("sitehealth", {}).get("sitehealth"),
                  "threshold_cusum": preds.get("threshold_cusum", {}).get("sitehealth")}
    ood_thr = {}
    for m in ("pvnet", "loadnet", "sitehealth"):
        p = args.models / f"{m}{'' if m == 'sitehealth' else '.s0'}.calibration.json"
        if p.exists():
            ood_thr[m] = (json.loads(p.read_text()).get("thresholds") or {}).get("ood")

    # bucket attributes shared by every model (ood added per model)
    seen = np.array([r.split == "train" for r, _ in cases])
    batt = np.array([float(getattr(r, "battery_kwh", 0) or 0) > 0 for r, _ in cases])
    month = np.array([clean[r.site_id]["ts"][d * H].month for r, d in cases])
    season = [season_of(m) for m in month]

    def day(tel, col, d, k=0):
        return tel[col][(d - k) * H:(d - k + 1) * H]

    rows, versions = [], {}

    def add(model, score_name, scores, gappy, ood_hi):
        for i, s in enumerate(scores):
            if np.isfinite(s):
                rows.append((model, score_name, bucket_key("site", seen[i], batt[i], season[i], bool(gappy[i]),
                                                           bool(ood_hi[i])), float(s)))

    for kind, col, models, refs in (("pv", "pv_ac_w", pv_models, PV_REF), ("load", "load_w", load_models, LOAD_REF)):
        y = np.stack([day(clean[r.site_id], col, d) for r, d in cases]) / 1000.0
        gappy = np.isnan(y).mean(1) > 0.10
        q = {}
        for name, p in models.items():
            if p is None:
                print(f"  skip {name}: predictor missing")
                continue
            t = time.time()
            ps = [E.run(p, r.to_dict(), E.site_window(clean[r.site_id], wx[str(r.locality_id)], d, False))
                  for r, d in cases]
            q[name] = _quantiles(ps)
            ok = [x for x in ps if x is not None]
            versions[name] = ok[0].model_version if ok else ""
            print(f"  {name}: {len(ok)}/{len(ps)} predictions in {time.time() - t:.0f}s")
        lag = 1 if kind == "pv" else 7
        pers = np.stack([day(clean[r.site_id], col, d, lag) for r, d in cases]) / 1000.0
        q["_persist_d1" if kind == "pv" else "_persist_w1"] = (pers, pers, pers, np.zeros(len(cases)))
        if kind == "pv":
            yest = np.stack([day(clean[r.site_id], col, d, 1) for r, d in cases]) / 1000.0
            mask = (np.nan_to_num(y) > 0.01) | (np.nan_to_num(yest) > 0.01)
        else:
            mask = np.ones_like(y, bool)
        for name, ref in refs.items():
            if name not in q:
                continue
            q10, _, q90, ood = q[name]
            thr = ood_thr.get(name)
            ood_hi = (np.nan_to_num(ood) > thr) if thr is not None else np.zeros(len(cases), bool)
            add(name, "calibration", calibration_score(q10, q90, y, mask), gappy, ood_hi)
            if ref in q:
                add(name, "sharpness", sharpness_score(q10, q90, y, q[ref][0], q[ref][2], mask), gappy, ood_hi)

    truth_lab = [E.day_label(labels, r.site_id, faulty[r.site_id]["ts"][d * H]) for r, d in cases]
    yf = np.stack([day(faulty[r.site_id], "pv_ac_w", d) for r, d in cases])
    gappy_f = np.isnan(yf).mean(1) > 0.10
    for name, p in det_models.items():
        if p is None:
            print(f"  skip {name}: predictor missing")
            continue
        t = time.time()
        ps = [E.run(p, r.to_dict(), E.site_window(faulty[r.site_id], wx[str(r.locality_id)], d, True))
              for r, d in cases]
        keep = np.array([x is not None for x in ps])
        alarm, called, ood = [], [], []
        for x in ps:
            if x is None:
                alarm.append(False), called.append("healthy"), ood.append(np.nan)
                continue
            thr = float((x.extra or {}).get("p_fault_threshold") or 0.5)
            probs = {k: v for k, v in (x.fault_probs or {}).items() if k != "healthy"}
            alarm.append(x.p_fault >= thr)
            called.append(max(probs, key=probs.get) if probs else (x.top_fault or "healthy"))
            ood.append(float(x.ood_score))
        ap, ms = detector_scores(np.array(alarm), called, truth_lab)
        ap[~keep], ms[~keep] = np.nan, np.nan
        thr = ood_thr.get(name)
        ood_hi = (np.nan_to_num(np.array(ood)) > thr) if thr is not None else np.zeros(len(cases), bool)
        add(name, "alarm_precision", ap, gappy_f, ood_hi)
        add(name, "miss_score", ms, gappy_f, ood_hi)
        versions[name] = next((x.model_version for x in ps if x is not None), "")
        print(f"  {name}: {int(keep.sum())}/{len(ps)} predictions, {int(np.sum(alarm))} alarms, "
              f"{sum(t != 'healthy' for t in truth_lab)} faulty days in {time.time() - t:.0f}s")
    df = pd.DataFrame(rows, columns=["model", "score_name", "bucket", "score"])
    return aggregate(df), versions


# ---------------------------------------------------------------- main

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Judge priors on the 2024 validation year")
    ap.add_argument("--out", type=Path, help="output directory (required unless --apply)")
    ap.add_argument("--apply", type=Path, help="install a run's priors (output dir or saved log) into --models")
    ap.add_argument("--models", type=Path, default=REPO / "models")
    ap.add_argument("--quick", action="store_true", help="small 2024-only fleet (smoke test)")
    ap.add_argument("--sites", type=int, default=120, help="sites, half seen / half unseen")
    ap.add_argument("--days", type=int, default=30, help="validation days per site")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-fleet", action="store_true", help="only the report-based priors (no data scoring)")
    args = ap.parse_args(argv)
    if args.apply:
        for p in apply(args.apply, args.models):
            print(f"updated {p}")
        return 0
    if args.out is None:
        ap.error("--out is required")
    sys.path.insert(0, str(REPO))  # eval.py lives at the repo root
    t0 = time.time()
    out_env = os.environ.get("LAB_OUTPUT_DIR")
    run = os.environ.get("LAB_RUN_ID") or (f"eval#{Path(out_env).name}" if out_env else "local")
    args.out.mkdir(parents=True, exist_ok=True)

    scored, versions = ({}, {}) if args.no_fleet else score_validation(args)
    emit("t_priors_score_s", time.time() - t0)

    # promoted models: calibration JSON copies with a `priors` key (+ the blocks alone in promoted_priors.json)
    promoted = {}
    for model, files in PROMOTED.items():
        if model not in scored:
            print(f"no validation scores for {model}; its calibration JSONs are not written")
            continue
        promoted[model] = priors_block(scored[model], "validation", run,
                                       scored_as=f"{model} ensemble of {len(files)} member(s), "
                                                 f"model_version {versions.get(model, '')}")
        for f in files:
            src = args.models / f"{f}.calibration.json"
            if src.exists():
                cal = json.loads(src.read_text())
                cal["priors"] = promoted[model]
                (args.out / src.name).write_text(json.dumps(cal, indent=1) + "\n")
    (args.out / "promoted_priors.json").write_text(json.dumps(promoted, indent=1) + "\n")

    nbm, mon, nbm_ver = wind_priors(args.models / "reports" / "r135-windnbm.report.json")
    (args.out / "windnbm.priors.json").write_text(json.dumps(
        {"model": "windnbm", "model_version": nbm_ver,
         "priors": priors_block(nbm, "report", "train#135",
                                note="event metrics from r135 report: validation split (*, turbine), seen parks "
                                     "test (turbine|seen), held-out park WP2 test (turbine|unseen)")},
        indent=1) + "\n")

    subs = {m: dict(model_version=versions.get(m, ""), source="validation", scores=scored[m])
            for m in ("pvlib", "lgbm_pv", "lgbm_load", "persistence_load", "threshold_cusum") if m in scored}
    subs["windmon_rules"] = dict(model_version="windmon-rules", source="report",
                                 note="r135 report baseline_windmon (test) and seen_parks_test.windmon", scores=mon)
    subs.update(ledger_only_priors(run))
    (args.out / "judge_priors.json").write_text(json.dumps(
        {"version": 1, "split": SPLIT, "source_run": run, "models": subs}, indent=1) + "\n")

    allp = {**scored, "windnbm": nbm, **{m: v["scores"] for m, v in subs.items()}}
    for m, sc in allp.items():
        for s, keys in sc.items():
            if ROOT in keys:
                emit(f"prior_{m}_{s}", keys[ROOT]["prior"])
    emit("t_priors_s", time.time() - t0)
    for name in OUT_FILES:  # also in the log, so the outputs can be applied without access to the worker
        print(f"{LOG_TAG} {name} {json.dumps(json.loads((args.out / name).read_text()), separators=(',', ':'))}")
    print(f"wrote priors to {args.out}")
    return 0


def apply(src: Path, models: Path) -> list[Path]:
    """Install a run's priors into models/: `src` is the run's output dir or a saved `lab logs` text.
    Adds (or replaces) only the `priors` key of each promoted calibration JSON; copies the two priors files."""
    if src.is_dir():
        blobs = {n: json.loads((src / n).read_text()) for n in OUT_FILES if (src / n).exists()}
    else:
        blobs = {}
        for line in src.read_text().splitlines():
            if LOG_TAG in line:
                _, name, body = line[line.index(LOG_TAG):].split(" ", 2)
                blobs[name] = json.loads(body)
    missing = [n for n in OUT_FILES if n not in blobs]
    if missing:
        raise SystemExit(f"apply: {missing} not found in {src}")
    written = []
    for model, block in blobs["promoted_priors.json"].items():
        for f in PROMOTED[model]:
            p = models / f"{f}.calibration.json"
            cal = json.loads(p.read_text())
            cal["priors"] = block
            p.write_text(json.dumps(cal, indent=1) + "\n")
            written.append(p)
    for n in ("judge_priors.json", "windnbm.priors.json"):
        (models / n).write_text(json.dumps(blobs[n], indent=1) + "\n")
        written.append(models / n)
    return written


if __name__ == "__main__":
    sys.exit(main())
