"""W2 evaluation (card #165): does counting tomorrow's availability beat M5 alone on days with stopped turbines?

    python -m sentinel.model.windpower.avail_eval [--m5 train|binned|<dir or .onnx>] [--n-scen 100]

Day-ahead set-up (spec W step 3 / D1): for every UTC day D the forecast is made at D-1 10:00 UTC (before the PZU gate)
from what M6 (WindNBM, models/windnbm*, replay's WindBank incl. its park-consensus curtailment rule) says in the last
2 h before the origin, read through the Judge adapter (sentinel/judge/adapter.py; no-Judge mode unless
SENTINEL_JUDGE_DB / SENTINEL_JUDGE_RUN are set, since no wind verdicts exist yet), plus M7 risk when M7 artifacts
exist (--m7-dir). The availability table (availability.py) gives [96, N] per day; the park forecast (scenarios.py)
uses one common wind level per park, x availability, x M5's sector wake ratio, then split-conformal on the park
total fitted on validation days of the training parks (label-free masks: M6 curtailment / data gap).

Policies scored on the same scenarios (common random numbers):
  windpower_only   M5, all turbines available
  sentinel         M5 x availability (M6/M7 via the Judge), expected p_up
  sentinel_bern    same, p_up drawn on/off per scenario
  sentinel_stops   ablation: only p_up (stops, M7), no derate (underperformance loss, icing)
  labels_origin    W1's proxy: injector fault_stop label at the origin -> 0 all day (uses labels: reference only)
  oracle           true per-step fault_stop labels (upper bound)
Survival of stops and the icing factor are fitted label-free on training days (M6 fault_stop runs; measured power
vs M5 P50 on weather icing-risk steps). No work-order ETAs exist in the simulator, so stops use the survival curve.

Headline (spec "done when"): on test-year park-days with a stopped turbine (injector label, scoring only),
stopped_park_nmae_sentinel < stopped_park_nmae_windpower_only (and pinball), sentinel_beats_windpower_only_stopped=1.
Coverage target of the park P10-P90 on all test park-days: 78-82 %. Labels: synthetic RO (ERA5-driven #22/#23 parks).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from sentinel.eval.metrics import coverage, pinball
from sentinel.eval.report import Report, emit
from sentinel.infra.timing import stage
from sentinel.model.wind.data import CLASS_INDEX, CLASSES, POWER, external_wind_dir, load_wind, make_splits

from .availability import AvailConfig, M6Batch, M7Batch, StopSurvival, availability, fit_icing_factor, icing_risk, \
    judge_reads, stop_runs_h
from .features import weather_per_turbine
from .model import WindPowerModel
from .scenarios import ParkConformal, common_levels, park_scenarios, scen_quantiles, wake_ratio

REPO = Path(__file__).resolve().parents[3]
LABEL = "synthetic RO (ERA5-driven #22/#23 parks)"
H = 96
ORIGIN_LEAD = 14 * 4  # D-1 10:00 UTC = 14 h before D 00:00
WIN = 8  # M6 steps read before the origin (2 h)
FS, CUR, GAP, ICE, UND = (CLASS_INDEX[c] for c in ("fault_stop", "curtailment", "data_gap", "icing",
                                                    "underperformance"))
POLICIES = ("windpower_only", "sentinel", "sentinel_bern", "sentinel_stops", "labels_origin", "oracle")
TAUS = (0.1, 0.5, 0.9)


# ---------------------------------------------------------------------------------------------------- M5
def get_m5(spec: str, out_dir: Path, seed: int, log=print) -> WindPowerModel:
    if spec in ("train", "binned"):
        d = out_dir / "m5"
        d.mkdir(parents=True, exist_ok=True)
        cmd = [sys.executable, str(REPO / "train.py"), "--model", "windpower", "--seed", str(seed), "--out", str(d)]
        if spec == "binned":
            try:
                import torch  # noqa: F401
            except ImportError:
                raise SystemExit("--m5 binned needs one W1 training run for the sidecar (wake ratios, curves): "
                                 "use a train run, or pass --m5 <dir with windpower*.json>")
        log(f"[w2] training M5 (W1 #164) for this run: {' '.join(cmd)} (log: {d / 'm5_train.log'})", flush=True)
        env = {k: v for k, v in os.environ.items() if k != "LAB_METRICS"}  # keep W1's metrics out of this run's
        with open(d / "m5_train.log", "w") as f:
            r = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, env=env, cwd=REPO)
        if r.returncode:
            raise SystemExit(f"M5 training failed (see {d / 'm5_train.log'}):\n" + (d / "m5_train.log").read_text()[-3000:])
        js = d / f"windpower.s{seed}.json"
        return WindPowerModel.from_json(js, binned=True) if spec == "binned" else WindPowerModel.load(js.with_suffix(".onnx"))
    p = Path(spec)
    if p.suffix == ".json":
        return WindPowerModel.from_json(p, binned=True)
    return WindPowerModel.load(p)


def predict_all(m: WindPowerModel, data, chunk_days: int = 30) -> np.ndarray:
    """M5 [T, N, 3] kW over the whole span in time chunks (8 steps of overlap for the turbulence window)."""
    T, N = data.n_steps, len(data.turbines)
    w = weather_per_turbine(data.weather, data.park_of)
    out = np.empty((T, N, 3), np.float32)
    step, pad = chunk_days * H, 8
    for s in range(0, T, step):
        a, b = max(0, s - pad), min(T, s + step + pad)
        q = m.predict(data.turbines, data.ts[a:b], {k: v[a:b] for k, v in data.meteo.items()},
                      {k: v[a:b] for k, v in w.items()})
        out[s:min(T, s + step)] = q[s - a: s - a + min(step, T - s)]
    return out


# ---------------------------------------------------------------------------------------------------- M6 / M7
def score_m6(data, model_dir: Path, chunk_days: int = 92, warm_days: int = 35, log=print) -> dict:
    """M6 per step over the whole span: cls [T, N] (int), score [T, N] (1 - p_normal), q50 [T, N] (frac of rated),
    via replay's WindBank (same thresholds and park-consensus rule as the replay)."""
    from sentinel.agent.predictors import WindBank

    T, N = data.n_steps, len(data.turbines)
    cls = np.zeros((T, N), np.uint8)
    score = np.zeros((T, N), np.float32)
    q50 = np.full((T, N), np.nan, np.float32)
    cols = {p: np.flatnonzero(data.park_of == i) for i, p in enumerate(data.park_ids)}
    version = None
    for s in range(0, T, chunk_days * H):
        a, b = max(0, s - warm_days * H), min(T, s + chunk_days * H)
        arrays = {}
        for i, p in enumerate(data.park_ids):
            c = cols[p]
            arrays[p] = {"turbines": [{**data.turbines[j], "replay_id": data.turbines[j]["turbine_id"]} for j in c],
                         "ts": data.ts[a:b], "telemetry": {k: v[a:b][:, c] for k, v in data.sensors.items()},
                         "meteo": {k: v[a:b][:, c] for k, v in data.meteo.items()}, "err": data.err[a:b][:, c],
                         "weather": {k: v[a:b, i] for k, v in data.weather.items()}}
        bank = WindBank(arrays, model_dir)
        version = bank.model_version
        for p, c in cols.items():
            sc = bank.scores[p]
            cls[s:b, c] = sc["cls"][s - a:]
            score[s:b, c] = np.clip(1.0 - sc["probs"][s - a:, :, 0], 0, 1)
            q50[s:b, c] = sc["pq"][s - a:, :, 1]
        log(f"[w2] M6 scored {str(data.ts[s])[:10]}..{str(data.ts[b - 1])[:10]}", flush=True)
    return {"cls": cls, "score": score, "q50": q50, "version": version}


def score_m7(data, model_dir: Path, log=print):
    """M7 daily p_fail_7d (top component) and alarm per (day, turbine), or None when no artifacts."""
    if not (model_dir / "windfailure_thresholds.json").exists():
        log(f"[w2] no M7 artifacts in {model_dir}: M7 term off", flush=True)
        return None
    from sentinel.model.windfailure.predictor import WindFailurePredictor

    m = WindFailurePredictor(model_dir)
    D = data.n_steps // H
    p7 = np.full((D, len(data.turbines)), np.nan)
    alarm = np.zeros((D, len(data.turbines)), bool)
    for i, pid in enumerate(data.park_ids):
        c = np.flatnonzero(data.park_of == i)
        d = m.daily([data.turbines[j] for j in c], data.ts[: D * H], {k: v[:, c] for k, v in data.sensors.items()},
                    data.err[:, c], {k: v[:, c] for k, v in data.meteo.items()},
                    None if data.weather.get("pressure_hpa") is None else data.weather["pressure_hpa"][:, i])
        top = d["p14"].argmax(-1)
        p7[:, c] = np.take_along_axis(d["p7"], top[..., None], -1)[..., 0]
        alarm[:, c] = d["p14"].max(-1) >= m.thr
    return {"p7": p7, "alarm": alarm, "shipped": m.shipped, "version": m.model_version}


# ---------------------------------------------------------------------------------------------------- origin reads
def run_length(mask: np.ndarray) -> np.ndarray:
    """[T, N] number of consecutive True steps ending at each step."""
    out = np.zeros(mask.shape, np.int32)
    run = np.zeros(mask.shape[1], np.int32)
    for t in range(mask.shape[0]):
        run = np.where(mask[t], run + 1, 0)
        out[t] = run
    return out


def origin_batch(m6: dict, i0: int, rated: np.ndarray, power: np.ndarray, stop_rl: np.ndarray) -> M6Batch:
    """M6 state per turbine from the WIN steps before origin index i0 (exclusive)."""
    c = m6["cls"][i0 - WIN: i0].astype(int)
    N = c.shape[1]
    counts = np.stack([(c == k).sum(0) for k in range(len(CLASSES))], 1)  # [N, K]
    counts[:, 0] = 0
    top = counts.argmax(1)
    has = counts[np.arange(N), top] >= WIN // 2 + 1
    cls = np.where(has, top, 0)
    sc = np.where(has, np.max(np.where(c == cls[None], m6["score"][i0 - WIN: i0], 0), 0), 0.0)
    # underperformance loss over the last day's flagged steps: 1 - measured / expected
    lo = max(0, i0 - H)
    flag = m6["cls"][lo:i0] == UND
    exp = np.nan_to_num(m6["q50"][lo:i0]) * rated[None]
    pw = power[lo:i0]
    ok = flag & np.isfinite(pw) & (exp > 0)
    se = np.where(ok, exp, 0).sum(0)
    loss = np.where(se > 0, 1 - np.where(ok, pw, 0).sum(0) / np.maximum(se, 1e-9), np.nan)
    age = stop_rl[i0 - 1] * 0.25
    return M6Batch(cls=[CLASSES[k] for k in cls], score=sc, loss_frac=np.clip(loss, 0, 1), stopped_h=age)


# ---------------------------------------------------------------------------------------------------- scoring
def scores(q, y, m) -> dict:
    m = m & np.isfinite(y) & np.isfinite(q).all(-1)
    if not m.any():
        return {"nmae": float("nan"), "pinball": float("nan"), "coverage": float("nan"), "n": 0}
    return {"nmae": float(np.abs(y - q[..., 1])[m].mean()),
            "pinball": float(np.mean([pinball(q[..., i], y, t, m) for i, t in enumerate(TAUS)])),
            "coverage": coverage(q[..., 0], q[..., 2], y, m), "n": int(m.sum())}


def parse_args(argv):
    ap = argparse.ArgumentParser(prog="python -m sentinel.model.windpower.avail_eval", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--m5", default="train", help="train | binned | <dir / .onnx / .json>")
    ap.add_argument("--m6-dir", default=str(REPO / "models"))
    ap.add_argument("--m7-dir", default=str(REPO / "models"))
    ap.add_argument("--n-scen", type=int, default=100)
    ap.add_argument("--rho", type=float, default=0.97)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-days", type=int, default=0, help="debug: only the first N val and N test days")
    ap.add_argument("--synth", action="store_true", help="stand-in generator instead of #22/#23 output (debug)")
    ap.add_argument("--synth-days", type=int, default=731)
    ap.add_argument("--synth-scale", type=float, default=1.0)
    ap.add_argument("--out", default=None)
    return ap.parse_args(argv)


def main(argv=None) -> dict:
    args = parse_args(argv if argv is not None else sys.argv[1:])
    t0 = time.time()
    rep = Report(vars(args), args.out)
    out_dir = rep.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    with stage("data"):
        if not args.synth and external_wind_dir() is None:
            from sentinel.sim.wind.build import build as build_sim

            build_sim()
        from sentinel.model.wind.synth import SynthConfig

        data = load_wind(prefer_external=not args.synth,
                         synth_config=SynthConfig(days=args.synth_days, scale=args.synth_scale))
        sp = make_splits(data)
    T, N, P = data.n_steps, len(data.turbines), len(data.park_ids)
    rated = data.rated_kw.astype(np.float64)
    power = data.sensors[POWER].astype(np.float64)
    lab = data.labels if data.labels is not None else np.zeros((T, N), np.uint8)
    po = data.park_of
    holdout = data.park_ids.index(sp.holdout_park)
    print(f"[w2] {data.source}: {N} turbines, {T} steps, parks {data.park_ids} (holdout {sp.holdout_park})", flush=True)

    with stage("m5"):
        m5 = get_m5(args.m5, out_dir, args.seed)
        q = predict_all(m5, data)  # [T, N, 3] kW
    print(f"[w2] M5 {m5.model_version} ({'MLP' if m5.sessions else 'binned fallback'})", flush=True)
    with stage("m6"):
        m6 = score_m6(data, Path(args.m6_dir))
    with stage("m7"):
        m7 = score_m7(data, Path(args.m7_dir))

    # ---- label-free fits on training days: stop survival, icing factor
    w_t = weather_per_turbine(data.weather, po)
    train_rows = sp.train_t
    surv = StopSurvival.fit(stop_runs_h(np.where(train_rows[:, None], m6["cls"], 0), FS))
    cfg0 = AvailConfig(survival=surv)
    risk_all = icing_risk(w_t, T, N, cfg0)
    ice_f, ice_n = fit_icing_factor(np.where(train_rows[:, None], power, np.nan), q[..., 1].astype(np.float64),
                                    risk_all & train_rows[:, None])
    cfg = AvailConfig(survival=surv, icing_factor=ice_f)
    emit("stop_survival_n", surv.n)
    emit("stop_median_h", float(np.interp(0.5, surv.surv[::-1], np.arange(len(surv.surv))[::-1] * 0.25)))
    emit("icing_factor", ice_f)
    emit("icing_fit_steps", ice_n)

    # ---- days: forecast for UTC day D made at D-1 10:00 from M6 before it
    day0 = np.flatnonzero((data.ts.astype("datetime64[m]") - data.ts.astype("datetime64[D]")).astype(int) == 0)
    day0 = day0[(day0 >= 36 * H) & (day0 + H <= T)]
    val_d = day0[sp.val_t[day0] & sp.val_t[np.minimum(day0 + H - 1, T - 1)]]
    test_d = day0[sp.test_t[day0]]
    if args.max_days:
        val_d, test_d = val_d[: args.max_days], test_d[: args.max_days]
    days = np.concatenate([val_d, test_d])
    stop_rl = run_length(m6["cls"] == FS)
    ids = data.turbine_ids
    parks, ratio_all = wake_ratio(m5, data.turbines, data.meteo, w_t, T)
    assert parks == list(data.park_ids) or set(parks) == set(data.park_ids)
    pk_idx = [parks.index(p) for p in data.park_ids]
    caps = np.array([rated[po == p].sum() for p in range(P)])
    m7_day0 = data.ts[0].astype("datetime64[D]")

    S = args.n_scen
    raw = {k: np.full((len(days), H, P, 3), np.nan) for k in POLICIES}
    a_sent = np.ones((len(days), H, N))
    counts = {"use": 0, "widen": 0, "dont_use": 0, "unknown": 0}
    rule_n: dict[str, int] = {}
    with stage("forecast"):
        for k, d0 in enumerate(days):
            i0 = d0 - ORIGIN_LEAD
            sl = slice(d0, d0 + H)
            ts = data.ts[sl]
            oday = str(data.ts[i0].astype("datetime64[D]"))
            b6 = origin_batch(m6, i0, rated, power, stop_rl)
            b7 = None
            if m7 is not None:
                dk = int((data.ts[i0].astype("datetime64[D]") - m7_day0).astype(int)) - 1  # last complete day
                if 0 <= dk < len(m7["p7"]):
                    b7 = M7Batch(m7["p7"][dk], m7["alarm"][dk])
            reads = judge_reads(ids, oday, b6, b7)
            for v in reads.verdict:
                counts[str(v)] = counts.get(str(v), 0) + 1
            counts["unknown"] += int((reads.cls == "unknown").sum())
            av = availability(reads, ts, weather={kk: v[sl] for kk, v in w_t.items()}, origin=data.ts[i0], cfg=cfg)
            for r in av.rule:
                rule_n[r] = rule_n.get(r, 0) + 1
            a_sent[k] = av.p_up
            pu_lab = np.broadcast_to((lab[i0 - 1] != FS).astype(float)[None], (H, N))
            pu_orc = (lab[sl] != FS).astype(float)
            u = common_levels(S, H, P, args.rho, args.seed + k)
            qd = q[sl]
            for p in range(P):
                cols = np.flatnonzero(po == p)
                r_t = ratio_all[sl, pk_idx[p]]
                uu = u[..., p]
                for name, pu, der, bern in (("windpower_only", None, None, False),
                                            ("sentinel", av.p_up, av.derate, False),
                                            ("sentinel_bern", av.p_up, av.derate, True),
                                            ("sentinel_stops", av.p_up, None, False),
                                            ("labels_origin", pu_lab, None, False),
                                            ("oracle", pu_orc, None, False)):
                    sc = park_scenarios(qd, rated, cols, r_t, uu, pu, der, bern, np.random.default_rng(k * 7 + p))
                    raw[name][k, :, p] = scen_quantiles(sc) / caps[p]
            if k % 50 == 0:
                print(f"[w2] day {k}/{len(days)} {str(ts[0])[:10]}", flush=True)

    # ---- park truth and masks (frac of capacity) per (day, step, park)
    idx = days[:, None] + np.arange(H)[None]  # [D, H]
    yk = power[idx]  # [D, H, N]
    y = np.stack([np.nansum(yk[..., po == p], -1) / caps[p] for p in range(P)], -1)  # [D, H, P]
    lab_d, m6_d = lab[idx], m6["cls"][idx]
    nan_any = np.stack([(~np.isfinite(yk[..., po == p])).any(-1) for p in range(P)], -1)
    score_ok = ~nan_any & np.stack([~np.isin(lab_d[..., po == p], (CUR, GAP)).any(-1) for p in range(P)], -1)
    cal_ok = ~nan_any & np.stack([~np.isin(m6_d[..., po == p], (CUR, GAP)).any(-1) for p in range(P)], -1)
    is_val = np.isin(days, val_d)[:, None, None]
    is_test = np.isin(days, test_d)[:, None, None]
    train_park = (np.arange(P) != holdout)[None, None]
    stopped_day = np.stack([(lab_d[..., po == p] == FS).any((1, 2)) for p in range(P)], -1)[:, None, :]  # [D,1,P]
    stopped_origin = np.stack([(lab[days - ORIGIN_LEAD - 1][:, po == p] == FS).any(-1) for p in range(P)], -1)[:, None]

    # ---- conformal per policy on validation days of the training parks; score on test
    res, cal = {}, {}
    parkname = np.broadcast_to(np.array(data.park_ids)[None, None], y.shape)
    for name in POLICIES:
        cm = np.broadcast_to(is_val & train_park & cal_ok, y.shape).ravel()
        c = ParkConformal.fit(raw[name].reshape(-1, 3)[cm], y.ravel()[cm], parkname.ravel()[cm])
        cal[name] = c
        qc = np.empty_like(raw[name])
        for p, pid in enumerate(data.park_ids):
            qc[:, :, p] = c.apply(raw[name][:, :, p], 1.0, pid)[0]
        base = score_ok & is_test
        sets = {"test_all": base, "stopped": base & stopped_day, "stopped_origin": base & stopped_origin,
                "stopped_holdout": base & stopped_day & ~train_park, "test_holdout": base & ~train_park,
                "val_cal": score_ok & is_val & train_park}
        res[name] = {s: scores(qc, y, np.broadcast_to(m, y.shape)) for s, m in sets.items()}
        res[name]["raw_stopped"] = scores(raw[name], y, np.broadcast_to(base & stopped_day, y.shape))
        res[name]["conformal"] = c.to_json()
        for s in ("test_all", "stopped", "stopped_origin", "stopped_holdout"):
            for mtr in ("nmae", "pinball", "coverage"):
                emit(f"{s}_park_{mtr}_{name}", res[name][s][mtr])
    n_sd = int((stopped_day[:, 0] & is_test[:, 0]).sum())
    emit("n_stopped_park_days_test", n_sd)
    emit("n_stopped_origin_park_days_test", int((stopped_origin[:, 0] & is_test[:, 0]).sum()))
    emit("n_test_days", len(test_d))
    emit("n_val_days", len(val_d))
    s_, w_ = res["sentinel"]["stopped"], res["windpower_only"]["stopped"]
    beats = bool(s_["nmae"] < w_["nmae"] and s_["pinball"] < w_["pinball"])
    emit("sentinel_beats_windpower_only_stopped", float(beats))
    emit("stopped_nmae_gain_pct", 100 * (1 - s_["nmae"] / w_["nmae"]))
    cov = res["sentinel"]["test_all"]["coverage"]
    emit("coverage_in_target_sentinel", float(0.78 <= cov <= 0.82))

    # ---- turbine availability accuracy on test days (truth: 1 - fault_stop label)
    tr = (lab_d != FS).astype(float)
    okt = np.broadcast_to(np.isin(days, test_d)[:, None, None], tr.shape)
    stop_rows = okt & (lab_d == FS).any(1, keepdims=True)
    a_lab = np.broadcast_to((lab[days - ORIGIN_LEAD - 1] != FS).astype(float)[:, None], tr.shape)
    for nm, a in (("sentinel", a_sent), ("windpower_only", np.ones_like(tr)), ("labels_origin", a_lab)):
        emit(f"avail_mae_stopped_turbines_{nm}", float(np.abs(a - tr)[stop_rows].mean()))
        emit(f"avail_mae_all_{nm}", float(np.abs(a - tr)[okt].mean()))
    for k_, v in counts.items():
        emit(f"judge_m6_{k_}", v)

    w2cfg = {"survival": surv.to_json(), "icing_factor": ice_f, "conformal": cal["sentinel"].to_json(),
             "conformal_windpower_only": cal["windpower_only"].to_json(), "n_scen": S, "rho": args.rho,
             "m5": m5.model_version, "m6": m6["version"], "m7": None if m7 is None else m7["version"],
             "origin": "D-1 10:00 UTC", "label": LABEL}
    (out_dir / "w2_park.json").write_text(json.dumps(w2cfg, default=float))
    for name, x in {"label": LABEL, "data_source": data.source, "splits": sp.describe(), "results": res,
                    "rules_turbine_days": rule_n, "judge_counts": counts, "config": w2cfg,
                    "m7": None if m7 is None else {"shipped": m7["shipped"], "version": m7["version"]}}.items():
        rep.table(name, x)
    rep.write()
    emit("t_total_s", time.time() - t0)
    print(f"[w2] {LABEL}: stopped park-days {n_sd}: park nMAE sentinel {s_['nmae']:.4f} vs windpower_only "
          f"{w_['nmae']:.4f} (oracle {res['oracle']['stopped']['nmae']:.4f}); pinball {s_['pinball']:.4f} vs "
          f"{w_['pinball']:.4f}; coverage all test {cov:.3f}; beats: {beats}", flush=True)
    return {"results": res, "config": w2cfg, "beats": beats, "rules": rule_n}


if __name__ == "__main__":
    main()
