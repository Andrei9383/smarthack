"""CARE-to-Compare scorer for M7 and the fixed-threshold baseline (non-RO transfer test), run by
`python train.py --model windfailure --care`.

Wind Farm A only (EDP, 5 onshore turbines, 22 datasets): the only farm whose feature descriptions name M7's sensors
(FARM_A_MAP). 10-min rows are averaged onto the 15-min grid; per dataset the thermal residual model is fitted on its
own normal-status training part and daily features are built as in training (single turbine: dev_park = dev).
M7's LightGBM and both thresholds are applied frozen from the synthetic-RO run.

A daily alarm at the end of day d flags day d+1 (causal). CARE score (Gück et al. 2024, doi:10.3390/data9120138):
coverage = point-wise F0.5 on anomaly datasets, earliness = weight 1 over the event's first half then linear to 0,
reliability = event-wise F0.5 (event predicted when the criticality counter, +1 / -1 per flagged / unflagged
normal-status point, reaches 72), accuracy = specificity on normal datasets;
CARE = (coverage + earliness + reliability + 2*accuracy) / 5, = accuracy if < 0.5, = 0 if no event predicted.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

from wattsentry.model.windfailure import COMPONENTS

LABEL = "non-RO: CARE-to-Compare Wind Farm A (EDP, Portugal)"
NORMAL_STATUS = (0, 2)
CRITICALITY = 72
COMPONENT_EVENT = re.compile(r"gearbox|generator bearing|transformer", re.I)
FARM_A_MAP = {  # contract sensor → regex on Farm A's feature description
    "act_pwt": r"^Grid power$",
    "turbine_t": r"^Ambient temperature$",
    "gearbox_t": r"^Temperature oil in gearbox$",
    "gen1_t": r"generator bearing 1",
    "gen2_t": r"generator bearing 2",
    "conv_t": r"IGBT-driver on the grid side inverter",
    "transformer_t": r"HV transformer phase L1",
    "wd_spd": r"^Windspeed$",
}


# --- CARE metric ----------------------------------------------------------------------------------------------
def f_beta(truth: np.ndarray, pred: np.ndarray, beta: float = 0.5) -> float:
    truth, pred = np.asarray(truth, bool), np.asarray(pred, bool)
    tp = int((truth & pred).sum())
    fn = int((truth & ~pred).sum())
    fp = int((~truth & pred).sum())
    b2 = beta**2
    den = (1 + b2) * tp + b2 * fn + fp
    return (1 + b2) * tp / den if den else 0.0


def earliness_weights(n: int) -> np.ndarray:
    if n <= 1:
        return np.ones(max(n, 1))
    pos = np.linspace(0.0, 1.0, n)
    return np.where(pos <= 0.5, 1.0, 2.0 * (1.0 - pos))


def max_criticality(pred: np.ndarray, normal: np.ndarray) -> int:
    c, best = 0, 0
    for p, ok in zip(pred, normal):
        if ok:
            c = c + 1 if p else max(c - 1, 0)
            best = max(best, c)
    return best


def care_score(items: list[dict]) -> dict:
    """items: {"anomaly": bool, "truth": bool[n], "pred": bool[n], "normal": bool[n]} over each prediction frame."""
    an = [i for i in items if i["anomaly"]]
    no = [i for i in items if not i["anomaly"]]
    cov = float(np.mean([f_beta(i["truth"][i["normal"]], i["pred"][i["normal"]]) for i in an])) if an else float("nan")
    earl = []
    for i in an:
        ev = i["pred"][i["truth"]]
        w = earliness_weights(len(ev))
        earl.append(float(np.dot(w, ev) / w.sum()) if len(ev) else 0.0)
    earl = float(np.mean(earl)) if earl else float("nan")
    acc = float(np.mean([np.mean(~i["pred"][i["normal"]]) for i in no if i["normal"].any()])) if no else float("nan")
    ev_pred = [max_criticality(i["pred"], i["normal"]) >= CRITICALITY for i in items]
    rel = f_beta([i["anomaly"] for i in items], ev_pred)
    if not any(ev_pred):
        care = 0.0
    elif acc < 0.5:
        care = acc
    else:
        care = (cov + earl + rel + 2 * acc) / 5
    return {"care_score": care, "coverage": cov, "earliness": earl, "reliability": rel, "accuracy": acc,
            "event_predictions": ev_pred}


# --- Farm A → our features ------------------------------------------------------------------------------------
def column_map(feature_description: Path) -> dict[str, str]:
    import pandas as pd

    fd = pd.read_csv(feature_description, sep=";", encoding="latin-1")
    out = {}
    for sensor, pat in FARM_A_MAP.items():
        hit = fd[fd["description"].astype(str).str.contains(pat, regex=True)]
        if len(hit):
            out[sensor] = f"{hit.iloc[0]['sensor_name']}_avg"
    return out


def dataset_to_grid(csv: Path, cmap: dict[str, str]):
    """→ (ts15 datetime64[m] [T], sensors {sensor: [T, 1]}, fit15 bool [T], rows DataFrame with ts / id / status)."""
    import pandas as pd

    cols = ["time_stamp", "id", "train_test", "status_type_id", *cmap.values()]
    df = pd.read_csv(csv, sep=";", usecols=lambda c: c in cols)
    df["ts"] = pd.to_datetime(df["time_stamp"])
    df = df.sort_values("ts").reset_index(drop=True)
    normal = df["status_type_id"].isin(NORMAL_STATUS)
    fit_row = (df["train_test"].astype(str) == "train") & normal
    day0 = df["ts"].iloc[0].floor("D")
    day1 = df["ts"].iloc[-1].floor("D") + pd.Timedelta(days=1)
    grid = pd.date_range(day0, day1, freq="15min", inclusive="left")
    num = df.set_index("ts")[list(cmap.values())].apply(pd.to_numeric, errors="coerce")
    num["__fit"] = fit_row.to_numpy().astype(float)
    g = num.resample("15min").mean().reindex(grid)
    sensors = {s: g[c].to_numpy(np.float64)[:, None] for s, c in cmap.items()}
    if "wd_spd" in sensors:
        sensors["wd_spd"] = sensors["wd_spd"] * 3.6
    fit15 = (g["__fit"].to_numpy() > 0.5)
    return grid.to_numpy().astype("datetime64[m]"), sensors, fit15, df[["ts", "id", "train_test", "status_type_id"]]


def daily_scores(sensors: dict, fit15: np.ndarray, boosters: dict | None):
    """Residuals + features for one turbine → (dev [D, C], p14 [D, C] or None, rated)."""
    from wattsentry.model.windfailure import thermal
    from wattsentry.model.windfailure.features import daily_features, feature_index

    T = len(fit15)
    err = np.zeros((T, 1), np.int32)
    p = sensors["act_pwt"][:, 0]
    pf = p[fit15 & np.isfinite(p)]
    rated = np.array([float(np.percentile(pf, 99.5)) if pf.size > 100 and np.percentile(pf, 99.5) > 0 else
                      float(np.nanmax(np.abs(p)) or 1.0)])
    for c in COMPONENTS:  # an unmapped sensor stays NaN (no residual, no alarm)
        sensors.setdefault(c, np.full((T, 1), np.nan))
    model = thermal.fit(sensors, err, rated, np.array([0]), [0], fit15)
    r = thermal.residuals(model, sensors, err, rated, np.array([0]))
    X = daily_features(r, sensors["act_pwt"], sensors["turbine_t"], rated, np.array([0]), np.array([0]))
    X[..., feature_index("dev_park")] = X[..., feature_index("dev")]  # single turbine: park assumed normal
    dev = X[:, 0, :, feature_index("dev")].astype(np.float64)
    p14 = None
    if boosters is not None:
        D, _, C, F = X.shape
        p14 = boosters[14].predict(X.reshape(-1, F)).reshape(D, C)
    return dev, p14, rated[0]


def score_farm_a(farm_dir: Path, model_dir: Path, log=print) -> dict:
    import lightgbm as lgb
    import pandas as pd

    cal = json.loads((model_dir / "windfailure_thresholds.json").read_text())
    boosters = {14: lgb.Booster(model_file=str(model_dir / "windfailure_14d.txt"))}
    thr_m = float(cal["thresholds"]["p_fail_14d"])
    thr_b = float(cal["thresholds"]["baseline_dev_c"])
    cmap = column_map(farm_dir / "feature_description.csv")
    missing = [s for s in ["act_pwt", "turbine_t", *COMPONENTS] if s not in cmap]
    log(f"[care] Farm A column map {cmap}; unmapped {missing}")
    ev = pd.read_csv(farm_dir / "event_info.csv", sep=";")
    items = {"model": [], "baseline": []}
    per = []
    for _, e in ev.iterrows():
        csv = farm_dir / "datasets" / f"{int(e['event_id'])}.csv"
        if not csv.exists():
            continue
        ts15, sensors, fit15, rows = dataset_to_grid(csv, cmap)
        dev, p14, rated = daily_scores(sensors, fit15, boosters)
        day0 = ts15[0].astype("datetime64[D]")
        alarm = {"model": np.nanmax(np.nan_to_num(p14, nan=-1), 1) >= thr_m,
                 "baseline": np.nanmax(np.nan_to_num(dev, nan=-1e9), 1) >= thr_b}
        pf = rows["train_test"].astype(str) != "train"  # prediction frame
        r = rows[pf]
        rts = r["ts"].to_numpy().astype("datetime64[m]")
        dix = ((rts.astype("datetime64[D]") - day0).astype(int) - 1)  # alarm from the previous day's end
        is_an = str(e["event_label"]) == "anomaly"
        truth = ((r["id"] >= e["event_start_id"]) & (r["id"] <= e["event_end_id"])).to_numpy() if is_an \
            else np.zeros(len(r), bool)
        normal = r["status_type_id"].isin(NORMAL_STATUS).to_numpy()
        rec = {"event_id": int(e["event_id"]), "asset": int(e.iloc[0]), "label": str(e["event_label"]),
               "description": "" if pd.isna(e.get("event_description")) else str(e["event_description"]),
               "rated_est": rated, "n_pred_points": int(len(r))}
        for k in items:
            a = alarm[k]
            pred = np.where((dix >= 0) & (dix < len(a)), a[np.clip(dix, 0, len(a) - 1)], False)
            items[k].append({"anomaly": is_an, "truth": truth, "pred": pred, "normal": normal,
                             "component": bool(COMPONENT_EVENT.search(rec["description"]))})
            lead = None
            if is_an and (pred & truth).any():
                first = rts[np.flatnonzero(pred & truth)[0]]
                end = np.datetime64(pd.Timestamp(e["event_end"]).to_datetime64(), "m")
                lead = float((end - first) / np.timedelta64(1, "m") / 60)
            rec[k] = {"any_alarm_in_frame": bool(pred.any()), "any_alarm_in_event": bool((pred & truth).any()),
                      "lead_h_before_event_end": lead, "max_criticality": max_criticality(pred, normal)}
        per.append(rec)
        log(f"[care] event {rec['event_id']} ({rec['label']}, {rec['description'][:30]}): "
            f"M7 {rec['model']['any_alarm_in_event'] or rec['model']['any_alarm_in_frame']} "
            f"baseline {rec['baseline']['any_alarm_in_event'] or rec['baseline']['any_alarm_in_frame']}")
    out = {"label": LABEL, "column_map": cmap, "unmapped": missing, "datasets": per,
           "thresholds": {"p_fail_14d": thr_m, "baseline_dev_c": thr_b}}
    for k, it in items.items():
        cs = care_score(it)
        an = [(i, p) for i, p in zip(it, cs["event_predictions"]) if i["anomaly"]]
        no = [(i, p) for i, p in zip(it, cs["event_predictions"]) if not i["anomaly"]]
        comp = [(i, p) for i, p in an if i["component"]]
        leads = [d[k]["lead_h_before_event_end"] for d in per if d[k]["lead_h_before_event_end"] is not None]
        cs.update({
            "event_recall": float(np.mean([p for _, p in an])) if an else float("nan"),
            "event_recall_component_failures": float(np.mean([p for _, p in comp])) if comp else float("nan"),
            "n_component_failure_events": len(comp),
            "normal_false_positive_rate": float(np.mean([p for _, p in no])) if no else float("nan"),
            "lead_time_h": float(np.median(leads)) if leads else 0.0,
            "n_anomaly": len(an), "n_normal": len(no),
        })
        cs["event_predictions"] = [bool(x) for x in cs["event_predictions"]]
        out[k] = cs
    return out


def run(model_dir: Path, max_gb: float = 2.0, emit=None, log=print, out_dir: Path | None = None) -> dict:
    """Download (range-fetch Farm A) + score. Never raises: returns {"status": ...} when skipped."""
    from .download import fetch

    def em(k, v):
        if emit is not None:
            emit(k, v)
        else:
            print(f"LAB_METRIC {k}={float(v):.6g}", flush=True)

    try:
        root = fetch(("Wind Farm A",), max_gb=max_gb, log=log)
        if root is None:
            em("care_status", 0)
            return {"status": "skipped (offline or too large)", "label": LABEL}
        res = score_farm_a(root / "Wind Farm A", Path(model_dir), log=log)
    except Exception as e:  # the CARE check must never take the training run down
        log(f"[care] SKIPPED: scoring failed: {e!r}")
        em("care_status", -1)
        return {"status": f"failed: {e!r}", "label": LABEL}
    em("care_status", 1)
    for k, pre in (("model", "care"), ("baseline", "care_baseline_threshold")):
        r = res[k]
        em(f"{pre}_score", r["care_score"])
        for m in ("coverage", "earliness", "reliability", "accuracy", "event_recall",
                  "event_recall_component_failures", "normal_false_positive_rate", "lead_time_h"):
            em(f"{pre}_{m}", r[m])
    log(f"[care] {LABEL}: M7 CARE {res['model']['care_score']:.3f} vs baseline {res['baseline']['care_score']:.3f}")
    res["status"] = "scored"
    if out_dir is not None:
        Path(out_dir, "care_report.json").write_text(json.dumps(res, indent=1, default=float))
    return res

