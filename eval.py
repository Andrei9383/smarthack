"""Test-set evaluation: one script for every model number on the slides, grouped by label (#16).

    python eval.py --predictor mock                    # end-to-end smoke test on CPU
    python eval.py --predictor onnx --models models/   # #14's ONNX models (M1 PVNet, M2 LoadNet, M4 SiteHealth)
    python eval.py --baselines                         # #15's baselines (pvlib, persistence, LightGBM, CUSUM)
    python eval.py --predictor onnx,baselines --drills # side by side, plus #28's robustness drills

Blocks and labels (plan §4, R13, R14):
  syn_seen / syn_unseen  synthetic_RO  fleet simulator (#11) test year 2025; PV + load forecasts on clean
                                       telemetry, site health on #12's injected faults; seen vs never-seen sites
  syn_detect             synthetic_RO  hours from fault onset to the first alarm, kWh and lei recoverable
  pvdaq / pvdaq_inj      non_RO        real NREL PVDAQ systems (#9): PV forecast, then #12's injector on top
  ro_load / ro_solar ... real_RO       real Romanian national series (Energy-Charts): persistence references
Weather in the windows is ERA5 (nowcast setting); forecast-noise drills come from #28.

Every number is printed as `LAB_METRIC <block>_<model>_<metric>=v` and written to $LAB_OUTPUT_DIR/eval_report.json
together with the run id, git sha, data manifest hash and arguments, so it can be reproduced from a run id.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from sentinel.contracts import DAY_STEPS, Prediction, PvFault, Window
from sentinel.eval import metrics as M
from sentinel.eval.report import Report, emit

H = DAY_STEPS
HIST_DAYS = 14
RAW = Path(__file__).resolve().parent / "data" / "raw"
FAULTS = [f.value for f in PvFault]
TEST_YEAR = 2025


def iso(t: pd.Timestamp) -> str:
    return t.tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%SZ")


# ---------------------------------------------------------------- predictors

def load_predictors(names: list[str], models_dir: Path, fleet_dir: Path | None) -> dict[str, dict]:
    """name -> {kind: predictor}; kinds pv | load | sitehealth. Missing pieces are skipped with a message."""
    out: dict[str, dict] = {}
    for name in names:
        if name == "mock":
            from sentinel.contracts.mock import MockPredictor
            out["mock"] = {k: MockPredictor(k) for k in ("pv", "load", "sitehealth")}
        elif name == "onnx":
            from sentinel.model.predictor import load_predictor
            for kind, model in (("pv", "pvnet"), ("load", "loadnet"), ("sitehealth", "sitehealth")):
                try:
                    out[model] = {kind: load_predictor(model, models_dir)}
                except (FileNotFoundError, ValueError) as e:
                    print(f"skip {model}: {e}")
        elif name == "baselines":
            try:
                from sentinel.eval.baselines.pv import predictors
            except ImportError as e:
                print(f"skip baselines: #15 not on this branch ({e})")
                continue
            out.update(predictors(fleet_dir))
        else:
            raise SystemExit(f"unknown predictor {name!r} (mock | onnx | baselines)")
    return out


def run(p, asset: dict, window: Window) -> Prediction | None:
    try:
        return p.predict(asset, window)
    except Exception as e:  # a model that fails on a case is counted, not fatal
        run.failures = getattr(run, "failures", 0) + 1
        if run.failures <= 3:
            print(f"  predict failed on {asset.get('site_id')} {window.t0}: {e!r}")
        return None


def quantiles(preds: list[Prediction | None]) -> tuple[np.ndarray, ...]:
    nan = np.full(H, np.nan)
    q = [[np.asarray(getattr(p, k)) if p else nan for p in preds] for k in ("q10", "q50", "q90")]
    return tuple(np.stack(x) for x in q)


# ---------------------------------------------------------------- synthetic RO (fleet + injected faults)

TEL_COLS = ["ts", "site_id", "pv_ac_w", "load_w", "batt_w", "soc", "inv_voltage_v", "inv_status"]
WX_COLS = {"ghi": "ghi_wm2", "dni": "dni_wm2", "dhi": "dhi_wm2", "temp": "temp_c", "wind": "wind10_ms",
           "snow": "snow_cm"}


def fleet(quick: bool) -> tuple[Path, Path]:
    from sentinel.sim.faults.dataset import inject_dataset
    from sentinel.sim.fleet.config import FleetConfig
    from sentinel.sim.fleet.generate import generate
    cfg = FleetConfig(n_sites=150, n_feeders=6, start_year=TEST_YEAR - 1) if quick else FleetConfig()
    clean = generate(cfg, plots=False)
    return clean, inject_dataset(clean)


def load_year(d: Path, ids: list[str]) -> dict[str, dict[str, np.ndarray]]:
    t = pq.read_table(d / "telemetry" / f"year={TEST_YEAR}.parquet", columns=TEL_COLS,
                      filters=[("site_id", "in", ids)]).to_pandas()
    t["site_id"] = t.site_id.astype(str)
    out = {}
    for sid, g in t.groupby("site_id", sort=False):
        g = g.sort_values("ts")
        out[sid] = {c: g[c].to_numpy(np.float64) for c in TEL_COLS[2:]}
        out[sid]["ts"] = pd.DatetimeIndex(g.ts)
    return out


def load_weather(clean: Path, localities: set[str]) -> dict[str, pd.DataFrame]:
    w = pd.read_parquet(clean / "weather.parquet")
    w["point_id"] = w.point_id.astype(str)
    w = w[w.point_id.isin(localities)]
    cloud = None
    if (RAW / "weather.parquet").exists():  # the fleet file has no cloud cover; take it from #9's ERA5 table
        c = pd.read_parquet(RAW / "weather.parquet", columns=["ts", "point_id", "cloud_pct"])
        c = c[c.point_id.isin({f"loc:{x}" for x in localities})]
        c["point_id"] = c.point_id.str.removeprefix("loc:")
        cloud = c
    out = {}
    for loc, g in w.groupby("point_id"):
        g = g.set_index("ts").sort_index()
        df = pd.DataFrame({k: g[v] for k, v in WX_COLS.items() if v in g})
        if cloud is not None:
            cl = cloud[cloud.point_id == loc].set_index("ts")["cloud_pct"].sort_index()
            df["cloud"] = cl.reindex(df.index, method="ffill").to_numpy()
        out[loc] = df
    return out


def site_window(tel: dict, wx: pd.DataFrame, d: int, with_telemetry: bool) -> Window:
    s, e, h0 = d * H, (d + 1) * H, (d - HIST_DAYS) * H
    ts = tel["ts"]
    w = wx.reindex(ts[s:e])
    soc = np.nan_to_num(tel["soc"][s:e])
    meta = {"weather_channels": list(w.columns), "power_unit": "W"}
    if with_telemetry:
        meta["telemetry"] = {k: tel[k][s:e] for k in ("pv_ac_w", "inv_voltage_v", "inv_status", "load_w")}
        meta["telemetry_history"] = {k: np.nan_to_num(tel[k][h0:s]) for k in ("inv_voltage_v", "inv_status",
                                                                              "soc", "batt_w")}
    return Window(t0=iso(ts[s]), horizon=H, weather=w.to_numpy(np.float64), reference=tel["pv_ac_w"][h0:s],
                  history=tel["load_w"][h0:s], battery=np.stack([soc, tel["batt_w"][s:e]], 1), meta=meta)


def day_label(labels: pd.DataFrame, sid: str, t0: pd.Timestamp) -> str:
    """Fault active during [t0, t0 + 24 h); the most severe one if several overlap."""
    t1 = t0 + pd.Timedelta(hours=24)
    g = labels[(labels.asset_id == sid) & (labels.onset < t1) & (labels.end.isna() | (labels.end > t0))]
    return "healthy" if g.empty else str(g.sort_values("severity").fault.iloc[-1])


def score_forecasts(rep: Report, block: str, label: str, preds_by_model: dict, truth: np.ndarray, kind: str):
    for name, preds in preds_by_model.items():
        q10, q50, q90 = quantiles(preds)
        day = (truth > 10) | (np.nan_to_num(q50) > 0.01) if kind == "pv" else None
        m = M.forecast_metrics(q10, q50, q90, truth / 1000.0, None)
        if day is not None:  # night steps are trivially "covered": score the interval on daylight only
            m["coverage"] = M.coverage(q10, q90, truth / 1000.0, day)
            m["nwidth"] = M.nwidth(q10, q90, truth / 1000.0, day)
        ok = [p for p in preds if p]
        rep.add(block, label, f"{name}_{kind}" if name in ("mock",) else name, m, n=len(ok),
                model_version=ok[0].model_version if ok else "", note=f"{len(preds) - len(ok)} failed")


def score_health(rep: Report, block: str, label: str, name: str, preds: list, labels: list[str]):
    keep = [(p, lab) for p, lab in zip(preds, labels) if p is not None]
    if not keep:
        return
    ps, labs = zip(*keep)
    y = np.array([lab != "healthy" for lab in labs])
    s = np.array([p.p_fault for p in ps])
    thr = float(ps[0].extra.get("p_fault_threshold") or 0.5)
    m = {"pr_auc": M.average_precision(y, s), **M.prf(y, s >= thr), "threshold": thr,
         "positives": float(y.sum())}
    probs = [p.fault_probs or {} for p in ps]
    cls = [c for c in FAULTS if c != "healthy"]
    ap = M.per_class_ap(list(labs), probs, cls)
    m["pr_auc_macro"] = float(np.nanmean([v for v in ap.values()])) if any(v == v for v in ap.values()) \
        else float("nan")
    name = f"{name}_health" if name == "mock" else name
    rep.add(block, label, name, m, n=len(ps), model_version=ps[0].model_version)
    rep.table(f"{block}.{name}.ap_per_class", ap)
    rep.table(f"{block}.{name}.confusion", {"classes": FAULTS, "matrix": M.confusion(
        list(labs), [p.top_fault or "healthy" for p in ps], FAULTS)})


def block_synthetic(rep: Report, preds: dict, args) -> None:
    clean_dir, faulty_dir = fleet(args.quick)
    sites = pd.read_parquet(clean_dir / "sites.parquet")
    sites["site_id"] = sites.site_id.astype(str)
    labels = pd.read_parquet(faulty_dir / "fault_labels.parquet")
    labels["asset_id"] = labels.asset_id.astype(str)
    rng = np.random.default_rng(args.seed)
    chosen = pd.concat([g.sample(min(len(g), args.sites // 2), random_state=args.seed)
                        for _, g in sites.groupby("split")])
    ids = list(chosen.site_id)
    clean, faulty = load_year(clean_dir, ids), load_year(faulty_dir, ids)
    wx = load_weather(clean_dir, set(chosen.locality_id.astype(str)))
    n_days = min(len(v["ts"]) for v in clean.values()) // H
    cases = [(row, int(d)) for _, row in chosen.iterrows()
             for d in rng.choice(np.arange(HIST_DAYS + 1, n_days), args.days, replace=False)]
    rep.table("syn.cases", [(r.site_id, r.split, d) for r, d in cases])
    print(f"synthetic: {len(ids)} sites x {args.days} days = {len(cases)} site-days, fleet {clean_dir.name}")

    for split in ("train", "unseen"):
        cs = [(r, d) for r, d in cases if r.split == split]
        block = "syn_seen" if split == "train" else "syn_unseen"
        for kind, col in (("pv", "pv_ac_w"), ("load", "load_w")):
            models = {n: k[kind] for n, k in preds.items() if kind in k}
            if not models:
                continue
            truth = np.stack([clean[r.site_id][col][d * H:(d + 1) * H] for r, d in cs])
            out = {n: [run(p, r.to_dict(), site_window(clean[r.site_id], wx[str(r.locality_id)], d, False))
                       for r, d in cs] for n, p in models.items()}
            score_forecasts(rep, block, "synthetic_RO", out, truth, kind)
        for n, k in preds.items():
            if "sitehealth" in k:
                labs = [day_label(labels, r.site_id, faulty[r.site_id]["ts"][d * H]) for r, d in cs]
                ps = [run(k["sitehealth"], r.to_dict(), site_window(faulty[r.site_id], wx[str(r.locality_id)],
                                                                    d, True)) for r, d in cs]
                score_health(rep, block, "synthetic_RO", n, ps, labs)

    detection(rep, preds, labels, chosen, clean, faulty, wx, n_days, args)


def detection(rep, preds, labels, chosen, clean, faulty, wx, n_days, args) -> None:
    """Replay each fault episode day by day from its onset; first alarm -> hours to detect; energy lost from
    then until the episode ends is what an immediate repair would recover."""
    prices = pd.read_parquet(RAW / "prices.parquet") if (RAW / "prices.parquet").exists() else None
    lei_mwh = float(prices[prices.ts.dt.year == TEST_YEAR].price_lei_mwh.mean()) if prices is not None else np.nan
    sid_set = set(chosen.site_id)
    ts0 = next(iter(clean.values()))["ts"]
    eps = labels[labels.asset_id.isin(sid_set) & (labels.onset >= ts0[HIST_DAYS * H])
                 & (labels.onset < ts0[(n_days - 1) * H]) & (labels.fault != "sensor_fault")]
    eps = eps.sample(min(len(eps), args.episodes), random_state=args.seed)
    site = chosen.set_index("site_id")
    for name, k in preds.items():
        if "sitehealth" not in k:
            continue
        hours, kwh, lei = [], 0.0, 0.0
        for _, ep in eps.iterrows():
            tel_c, tel_f = clean[ep.asset_id], faulty[ep.asset_id]
            d0 = int(ts0.searchsorted(ep.onset, side="right") - 1) // H
            end = ep.end if pd.notna(ep.end) else ts0[-1]
            hit = None
            for d in range(d0, min(n_days, d0 + args.search_days)):
                if ts0[d * H] >= end:
                    break
                r = site.loc[ep.asset_id]
                p = run(k["sitehealth"], {**r.to_dict(), "site_id": ep.asset_id},
                        site_window(tel_f, wx[str(r.locality_id)], d, True))
                thr = float((p.extra.get("p_fault_threshold") if p else None) or 0.5)
                if p is not None and p.p_fault >= thr:
                    hit = d
                    break
            if hit is None:
                hours.append(None)
                continue
            alarm = ts0[(hit + 1) * H - 1] + pd.Timedelta(minutes=15)  # assessed at the end of the day
            hours.append(max(0.0, (alarm - ep.onset).total_seconds() / 3600))
            sl = slice((hit + 1) * H, int(ts0.searchsorted(end)))
            lost = np.nansum(tel_c["pv_ac_w"][sl] - tel_f["pv_ac_w"][sl]) * 0.25 / 1000  # kWh
            kwh += max(0.0, lost)
        lei = kwh * lei_mwh / 1000
        m = {**M.detection_delays(hours), "kwh_recoverable": kwh, "lei_recoverable": lei}
        rep.add("syn_detect", "synthetic_RO", name, m, n=len(hours))
        rep.table(f"syn_detect.{name}.episodes", [dict(site=e.asset_id, fault=e.fault, onset=str(e.onset), hours=h)
                                                   for (_, e), h in zip(eps.iterrows(), hours)])


# ---------------------------------------------------------------- PVDAQ (non-RO validation)

def block_pvdaq(rep: Report, preds: dict, args) -> None:
    if not (RAW / "validation" / "pvdaq_power.parquet").exists():
        print("skip pvdaq: data/raw/validation missing (#9)")
        return
    systems = pd.read_parquet(RAW / "validation" / "pvdaq_systems.parquet")
    power = pd.read_parquet(RAW / "validation" / "pvdaq_power.parquet")
    weather = pd.read_parquet(RAW / "validation" / "pvdaq_weather.parquet")
    chans = {"ghi": "ghi_wm2", "dni": "dni_wm2", "dhi": "dhi_wm2", "temp": "temp_c", "cloud": "cloud_pct",
             "wind": "wind10_ms", "snow": "snow_cm"}
    rng = np.random.default_rng(args.seed)
    cases = []  # (asset, tel dict over the year on a regular 15-min grid, day index)
    tels = {}
    for s in systems.itertuples():
        p = power[power.system_id == s.system_id].groupby("ts").ac_w.mean()  # DST-switch days repeat an hour
        grid = pd.date_range(f"{s.year}-01-01", f"{s.year + 1}-01-01", freq="15min", tz="UTC", inclusive="left")
        pv = p.reindex(grid).to_numpy(np.float64)
        w = weather[weather.point_id == s.system_id].set_index("ts")
        wx = pd.DataFrame({k: w[v] for k, v in chans.items()}).reindex(grid, method="ffill")
        tel = {"ts": grid, "pv_ac_w": pv, "load_w": np.zeros_like(pv), "batt_w": np.zeros_like(pv),
               "soc": np.zeros_like(pv), "inv_voltage_v": np.full_like(pv, 230.0), "inv_status": np.zeros_like(pv)}
        tels[s.system_id] = (tel, wx)
        asset = dict(site_id=s.system_id, lat=s.lat, lon=s.lon, kwp_recorded=s.kwp, tilt_recorded=s.tilt,
                     azimuth_recorded=s.azimuth, inverter_kw=s.kwp, battery_kwh=0.0, battery_kw=0.0, split="non_RO")
        days = [d for d in range(HIST_DAYS + 1, len(grid) // H)
                if np.isfinite(pv[d * H:(d + 1) * H]).mean() > 0.9
                and np.isfinite(pv[(d - HIST_DAYS) * H:d * H]).mean() > 0.6]
        for d in rng.choice(days, min(len(days), args.pvdaq_days), replace=False):
            cases.append((asset, int(d)))
    print(f"pvdaq: {len(systems)} systems, {len(cases)} system-days")
    models = {n: k["pv"] for n, k in preds.items() if "pv" in k}
    truth = np.stack([tels[a["site_id"]][0]["pv_ac_w"][d * H:(d + 1) * H] for a, d in cases])
    out = {n: [run(p, a, site_window(*tels[a["site_id"]], d, False)) for a, d in cases] for n, p in models.items()}
    score_forecasts(rep, "pvdaq", "non_RO", out, truth, "pv")

    # Same systems with #12's injector on top (PV-side faults only: no battery, voltage or load here).
    from sentinel.sim.faults.inject import inject
    long = pd.concat([pd.DataFrame({"ts": t["ts"], "site_id": sid, "pv_ac_w": np.nan_to_num(t["pv_ac_w"])})
                      for sid, (t, _) in tels.items()])
    meta = pd.DataFrame([dict(site_id=s.system_id, lat=s.lat, lon=s.lon, inverter_kw=s.kwp) for s in
                         systems.itertuples()])
    bad, labels = inject(long, meta)
    labels["asset_id"] = labels.asset_id.astype(str)
    for n, k in preds.items():
        if "sitehealth" not in k:
            continue
        ps, labs = [], []
        for a, d in cases:
            t, wx = tels[a["site_id"]]
            tf = dict(t, pv_ac_w=bad[bad.site_id == a["site_id"]].pv_ac_w.to_numpy(np.float64))
            ps.append(run(k["sitehealth"], a, site_window(tf, wx, d, True)))
            labs.append(day_label(labels, a["site_id"], t["ts"][d * H]))
        score_health(rep, "pvdaq_inj", "non_RO", n, ps, labs)


# ---------------------------------------------------------------- real RO national series

def block_real_ro(rep: Report, args) -> None:
    """Persistence references on the real national series (the bar M3 / national checks must beat)."""
    path = RAW / "entsoe_ro.parquet"
    if not path.exists():
        print("skip real RO: data/raw/entsoe_ro.parquet missing (#9)")
        return
    e = pd.read_parquet(path)
    for series, block in (("load_actual", "ro_load"), ("gen_solar", "ro_solar"), ("gen_wind", "ro_wind")):
        s = e[e.series == series].set_index("ts").mw.sort_index()
        s = s[~s.index.duplicated()].asfreq("15min")
        test = s[s.index.year == TEST_YEAR]
        for lag_days, name in ((1, "persistence_1d"), (7, "persistence_7d")):
            prev = s.shift(lag_days * H).reindex(test.index).to_numpy()
            y = test.to_numpy()
            rep.add(block, "real_RO", name, {"nmae": M.nmae(prev, y)}, n=int(np.isfinite(y).sum()))


# ---------------------------------------------------------------- drills (#28)

def block_drills(rep: Report, preds: dict, args) -> None:
    try:
        from sentinel.data import drills
    except ImportError:
        print("drills: sentinel.data.drills (#28) not on this branch; skipped")
        emit("drills_available", 0)
        return
    emit("drills_available", 1)
    clean_dir, _ = fleet(args.quick)
    sites = pd.read_parquet(clean_dir / "sites.parquet").astype({"site_id": str})
    chosen = sites[sites.split == "unseen"].sample(min(20, args.sites // 2), random_state=args.seed)
    clean = load_year(clean_dir, list(chosen.site_id))
    wx = load_weather(clean_dir, set(chosen.locality_id.astype(str)))
    n_days = min(len(v["ts"]) for v in clean.values()) // H
    rng = np.random.default_rng(args.seed + 1)
    cases = [(r, int(d)) for _, r in chosen.iterrows() for d in rng.choice(np.arange(HIST_DAYS + 1, n_days), 3)]
    truth = np.stack([clean[r.site_id]["pv_ac_w"][d * H:(d + 1) * H] for r, d in cases])
    for drill, fn in drills.DRILLS.items():  # name -> fn(asset, window, rng) -> (asset, window)
        out = {}
        for n, k in preds.items():
            if "pv" in k:
                res = []
                for r, d in cases:
                    a, w = fn(r.to_dict(), site_window(clean[r.site_id], wx[str(r.locality_id)], d, False), rng)
                    res.append(run(k["pv"], a, w))
                out[n] = res
        score_forecasts(rep, f"drill_{drill}", "synthetic_RO", out, truth, "pv")


# ---------------------------------------------------------------- main

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--predictor", default="auto",
                    help="comma list of mock | onnx | baselines; auto = onnx if --models has *.onnx, else mock")
    ap.add_argument("--baselines", action="store_true", help="same as --predictor baselines (lab run 'baseline')")
    ap.add_argument("--models", type=Path, default=Path("models"), help="ONNX dir for --predictor onnx")
    ap.add_argument("--blocks", default="synthetic,pvdaq,real_ro", help="comma list of synthetic | pvdaq | real_ro")
    ap.add_argument("--drills", action="store_true", help="also run #28's robustness drills")
    ap.add_argument("--quick", action="store_true", help="small fleet (150 sites, 2024-25) for smoke tests")
    ap.add_argument("--sites", type=int, default=60, help="test sites (half seen, half never seen)")
    ap.add_argument("--days", type=int, default=6, help="test days per site")
    ap.add_argument("--episodes", type=int, default=40, help="fault episodes replayed for hours-to-detect")
    ap.add_argument("--search-days", type=int, default=14, help="days after onset an alarm still counts")
    ap.add_argument("--pvdaq-days", type=int, default=40, help="days per PVDAQ system")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if args.predictor == "auto":
        args.predictor = "onnx" if any(args.models.glob("*.onnx")) else "mock"
    names = ["baselines"] if args.baselines else [n for n in args.predictor.split(",") if n]
    blocks = set(args.blocks.split(","))

    rep = Report(vars(args) | {"models": str(args.models)})
    t = time.time()
    fleet_dir = fleet(args.quick)[0] if ("synthetic" in blocks or "baselines" in names) else None
    preds = load_predictors(names, args.models, fleet_dir)
    if not preds:
        raise SystemExit("no predictors loaded")
    print(f"predictors: { {n: sorted(k) for n, k in preds.items()} }")
    if "synthetic" in blocks:
        block_synthetic(rep, preds, args)
    if "pvdaq" in blocks:
        block_pvdaq(rep, preds, args)
    if "real_ro" in blocks:
        block_real_ro(rep, args)
    if args.drills:
        block_drills(rep, preds, args)
    emit("predict_failures", getattr(run, "failures", 0))
    emit("t_eval_s", time.time() - t)
    print(rep.summary())
    rep.write()


if __name__ == "__main__":
    sys.exit(main())
