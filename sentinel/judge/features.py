"""Judge inputs (team/judge-design.md §7.3): one row per witness prediction (witness, asset, local day).

Two layers, shared by training (sentinel.judge.rows) and the live loop (J4, #149), so both see the same numbers:

1. **Raw row** (`RAW_COLUMNS`): plain scalars about one prediction, its substitute, the asset's recent telemetry and
   the ledger's snapshot for its bucket. Built with `prediction_row(pred, ...)` from a contract `Prediction`
   envelope, or in bulk from arrays with the `summarize_*` helpers (what the row generator does). Any column may be
   missing or NaN: LightGBM handles missing values natively, and `build()` fills absent columns with NaN.
2. **Feature matrix** (`build(rows, family)`): the numeric columns the LightGBM family model reads, in a fixed
   order (`FEATURES[family]`).

`reasons_for()` turns LightGBM feature contributions into up to 3 plain-word reasons (§7.7).
Units: energies in kWh over the local day (96 x 15 min), bands relative to the day's mean q50.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

H = 96
STEP_H = 0.25
EPS = 1e-6

# --------------------------------------------------------------------------------------------- witnesses
FAMILY_OF = {"pvnet": "forecast", "loadnet": "forecast", "demandnet": "forecast", "windpower": "forecast",
             "sitehealth": "detector", "windnbm": "detector"}
LEDGER_ONLY = ("windfailure", "gridtwin")  # §7.4: too few graded outcomes; P(valid) = ledger trust
FAMILIES = ("forecast", "detector")
WITNESS_CODE = {"pvnet": 0, "loadnet": 1, "demandnet": 2, "windpower": 3, "sitehealth": 10, "windnbm": 11}
KIND_OF = {"pvnet": "site", "loadnet": "site", "sitehealth": "site", "windnbm": "turbine", "windpower": "park",
           "demandnet": "zone", "windfailure": "turbine", "gridtwin": "feeder"}
# Detector classes: PvFault (M4) codes 0-7, WindAnomaly (M6) codes 20-27 (same order as the contracts).
PV_CLASSES = ["healthy", "soiling", "shading", "string_outage", "clipping", "overvoltage_trip", "battery_fade",
              "sensor_fault"]
WIND_CLASSES = ["normal", "fault_stop", "curtailment", "icing", "underperformance", "overheating",
                "anemometer_fault", "data_gap"]
CLASS_CODE = {**{c: i for i, c in enumerate(PV_CLASSES)}, **{f"wind:{c}": 20 + i for i, c in enumerate(WIND_CLASSES)}}
SAFETY_CLASSES = ("icing", "overheating")  # §7.5 rule 2 (J4 applies it; listed here for the reasons text)

# ------------------------------------------------------------------------------------------- raw columns
ID_COLUMNS = ["model", "model_version", "asset_id", "day", "kind", "family", "bucket"]
CONTEXT_COLUMNS = ["battery", "seen", "month", "sky"]
OWN_COLUMNS = ["ood_score", "ood_thr", "member_std", "energy_q50", "band_rel", "is_fallback", "telemetry_age_h",
               "in_gap_frac", "in_frozen", "in_unit_jump", "energy_yday", "energy_scale"]
DETECTOR_COLUMNS = ["alarm", "p_fault", "p_fault_std", "p_top", "top_class", "abn_share", "resid_yday",
                    "park_consensus"]
SUB_COLUMNS = ["sub_model", "sub_energy", "sub_band_rel"]
LEDGER_COLUMNS = ["trust_cal", "trust_sharp", "trust_ap", "trust_ms", "trust_main", "prior_main", "n_eff",
                  "trust_lo80", "trust_trend3", "band_scale", "health_alert", "sub_trust_main", "sub_n_eff"]
# the witness's own recent outputs for this asset (drift shows as a jump against them before any outcome arrives):
# medians over the previous 7 days and over days 8-28 back, and whether today's output repeats yesterday's exactly
HISTORY_COLUMNS = ["hist_band_rel7", "hist_band_rel28", "hist_energy7", "hist_energy28", "hist_p_fault7",
                   "hist_p_fault28", "hist_alarm7", "repeat_prev"]
RAW_COLUMNS = ID_COLUMNS + CONTEXT_COLUMNS + OWN_COLUMNS + DETECTOR_COLUMNS + SUB_COLUMNS + LEDGER_COLUMNS + \
    HISTORY_COLUMNS

# ------------------------------------------------------------------------------------------- features
COMMON = ["witness", "battery", "seen", "month_sin", "month_cos", "sky", "ood_score", "ood_ratio", "is_fallback",
          "telemetry_age_h", "in_gap_frac", "in_frozen", "in_unit_jump",
          "trust_main", "trust_gap", "n_eff", "trust_lo80", "trust_trend3", "band_scale", "health_alert",
          "sub_trust_main", "sub_gap", "sub_n_eff"]
FEATURES: dict[str, list[str]] = {
    "forecast": COMMON + ["trust_cal", "trust_sharp", "band_rel", "member_std_rel", "energy_rel", "rel_vs_sub",
                          "abs_rel_vs_sub", "sub_band_rel", "rel_vs_yday", "band_vs_hist7", "band_vs_hist28",
                          "energy_vs_hist7", "energy_vs_hist28", "band_vs_hist_ext", "energy_vs_hist_ext",
                          "repeat_prev"],
    "detector": COMMON + ["trust_ap", "trust_ms", "alarm", "p_fault", "p_fault_std", "p_top", "top_class",
                          "abn_share", "resid_yday", "park_consensus", "member_std_rel", "pfault_vs_hist7",
                          "pfault_vs_hist28", "pfault_vs_hist_ext", "hist_alarm7", "repeat_prev"],
}
CATEGORICAL = ["witness", "top_class"]


def family_of(model: str) -> Optional[str]:
    return FAMILY_OF.get(str(model))


def _col(df: pd.DataFrame, name: str, default: float = np.nan) -> pd.Series:
    if name in df.columns:
        return pd.to_numeric(df[name], errors="coerce").astype(float)
    return pd.Series(default, index=df.index, dtype=float)


def build(rows: pd.DataFrame, family: str) -> pd.DataFrame:
    """Raw rows -> the family's feature matrix (float columns, `FEATURES[family]` order). Missing raw columns -> NaN."""
    r = rows
    f = pd.DataFrame(index=r.index)
    f["witness"] = r["model"].map(WITNESS_CODE).astype(float) if "model" in r else np.nan
    for c in ("battery", "seen", "sky", "ood_score", "is_fallback", "telemetry_age_h", "in_gap_frac", "in_frozen",
              "in_unit_jump", "trust_main", "n_eff", "trust_lo80", "trust_trend3", "band_scale", "health_alert",
              "sub_trust_main", "sub_n_eff", "trust_cal", "trust_sharp", "trust_ap", "trust_ms", "band_rel",
              "sub_band_rel", "alarm", "p_fault", "p_fault_std", "p_top", "abn_share", "resid_yday",
              "park_consensus", "hist_alarm7", "repeat_prev"):
        f[c] = _col(r, c)
    month = _col(r, "month")
    f["month_sin"], f["month_cos"] = np.sin(2 * np.pi * month / 12), np.cos(2 * np.pi * month / 12)
    thr = _col(r, "ood_thr")
    f["ood_ratio"] = f["ood_score"] / thr.where(thr > 0)
    f["trust_gap"] = f["trust_main"] - _col(r, "prior_main")
    f["sub_gap"] = f["trust_main"] - f["sub_trust_main"]
    e50, scale = _col(r, "energy_q50"), _col(r, "energy_scale")
    f["energy_rel"] = e50 / scale.where(scale > 0)
    f["member_std_rel"] = _col(r, "member_std") * H * STEP_H / np.maximum(e50.abs(), 0.1)
    sub = _col(r, "sub_energy")
    f["rel_vs_sub"] = (e50 - sub) / np.maximum(sub.abs(), 0.1)
    f["abs_rel_vs_sub"] = f["rel_vs_sub"].abs()
    yd = _col(r, "energy_yday")
    f["rel_vs_yday"] = (e50 - yd) / np.maximum(yd.abs(), 0.1)
    for w in ("7", "28"):
        hb, he = _col(r, f"hist_band_rel{w}"), _col(r, f"hist_energy{w}")
        f[f"band_vs_hist{w}"] = f["band_rel"] / hb.where(hb > EPS)
        f[f"energy_vs_hist{w}"] = (e50 - he) / np.maximum(he.abs(), 0.1)
        f[f"pfault_vs_hist{w}"] = f["p_fault"] - _col(r, f"hist_p_fault{w}")
    # the larger deviation of the two windows: a fresh drift shows in both, an older one only against days 8-28 back
    # (2024 has 6 drift episodes, so the model must not depend on which window still holds the healthy outputs)
    f["band_vs_hist_ext"] = _extreme(np.log(f["band_vs_hist7"]), np.log(f["band_vs_hist28"]))
    f["energy_vs_hist_ext"] = _extreme(f["energy_vs_hist7"], f["energy_vs_hist28"])
    f["pfault_vs_hist_ext"] = _extreme(f["pfault_vs_hist7"], f["pfault_vs_hist28"])
    tc = r["top_class"] if "top_class" in r else pd.Series(np.nan, index=r.index)
    f["top_class"] = pd.to_numeric(tc.map(lambda v: CLASS_CODE.get(v, np.nan) if isinstance(v, str) else v),
                                   errors="coerce").astype(float)
    return f[FEATURES[family]].astype(float)


# --------------------------------------------------------------------------------- summaries (vectorised)
def _nanmean(x: np.ndarray, m: np.ndarray) -> np.ndarray:
    m = m & np.isfinite(x)
    n = m.sum(-1)
    return np.where(n > 0, np.where(m, x, 0.0).sum(-1) / np.maximum(n, 1), np.nan)


def active_mask(q50: np.ndarray, kind: str) -> np.ndarray:
    """Steps that count for a day summary: daylight (q50 > 10 W) for PV, every step otherwise."""
    q50 = np.asarray(q50, float)
    return (np.nan_to_num(q50) > 0.01) if kind == "pv" else np.isfinite(q50)


def summarize_forecast(q10, q50, q90, kind: str) -> dict[str, np.ndarray]:
    """[N, 96] kW quantiles -> energy_q50 (kWh/day) and band_rel (mean P10-P90 width / mean q50 on active steps)."""
    q10, q50, q90 = (np.atleast_2d(np.asarray(a, float)) for a in (q10, q50, q90))
    m = active_mask(q50, kind)
    energy = np.nansum(np.where(np.isfinite(q50), q50, 0.0), -1) * STEP_H
    width = _nanmean(q90 - q10, m)
    level = _nanmean(q50, m)
    band_rel = np.where(level > EPS, width / np.maximum(level, EPS), np.nan)
    return {"energy_q50": energy, "band_rel": band_rel}


def input_quality(y_prev: np.ndarray, valid_prev: Optional[np.ndarray] = None,
                  energy_ref: Optional[np.ndarray] = None) -> dict[str, np.ndarray]:
    """Quality flags of the INPUT window (the day before the prediction), physics/own-history only (§5.4 spirit):
    in_gap_frac (missing or invalid share), in_frozen (same non-zero value >= 8 steps in a row), in_unit_jump
    (daily energy outside 0.2-5 x the asset's own reference energy)."""
    y = np.atleast_2d(np.asarray(y_prev, float))
    bad = ~np.isfinite(y)
    if valid_prev is not None:
        bad |= ~np.atleast_2d(np.asarray(valid_prev)).astype(bool)
    gap = bad.mean(-1)
    v = np.nan_to_num(y)
    same = (np.abs(np.diff(v, axis=-1)) < 1e-6) & (np.abs(v[:, 1:]) > 1e-3)
    frozen = np.zeros(len(y), bool)
    run = np.zeros(len(y), int)
    for k in range(same.shape[1]):  # small fixed loop over 95 steps, vectorised over assets
        run = np.where(same[:, k], run + 1, 0)
        frozen |= run >= 7  # 8 equal values = 7 equal steps
    energy = np.nansum(v, -1) * STEP_H
    if energy_ref is None:
        jump = np.zeros(len(y), bool)
    else:
        ref = np.asarray(energy_ref, float)
        with np.errstate(invalid="ignore", divide="ignore"):
            ratio = energy / ref
        jump = np.isfinite(ratio) & (ref > 0.1) & ((ratio < 0.2) | (ratio > 5.0))
    return {"in_gap_frac": gap, "in_frozen": frozen.astype(float), "in_unit_jump": jump.astype(float),
            "energy_yday": energy}


def sky_class(cloud_pct: np.ndarray, daylight: np.ndarray) -> np.ndarray:
    """0 clear (< 30 %), 1 mixed, 2 overcast (> 70 %) from the forecast cloud cover over daylight steps."""
    c = _nanmean(np.atleast_2d(np.asarray(cloud_pct, float)), np.atleast_2d(daylight).astype(bool))
    return np.where(np.isfinite(c), np.where(c < 30, 0.0, np.where(c > 70, 2.0, 1.0)), np.nan)


def detector_summary(probs: np.ndarray, threshold: float, classes: Sequence[str] = PV_CLASSES,
                     p_fault_std: Optional[np.ndarray] = None) -> dict[str, Any]:
    """[N, C] class probabilities (index 0 = healthy / normal) -> alarm, p_fault, p_top, top_class (non-healthy
    argmax; 'healthy' when no alarm)."""
    p = np.atleast_2d(np.asarray(probs, float))
    p_fault = 1.0 - p[:, 0]
    alarm = p_fault >= threshold
    j = 1 + np.argmax(p[:, 1:], axis=1)
    top = np.where(alarm, np.asarray(classes, object)[j], classes[0])
    return {"alarm": alarm.astype(float), "p_fault": p_fault, "p_top": p[np.arange(len(p)), j],
            "top_class": top, "p_fault_std": (np.asarray(p_fault_std, float) if p_fault_std is not None
                                              else np.full(len(p), np.nan))}


def _extreme(a: pd.Series, b: pd.Series) -> pd.Series:
    """Element-wise value of larger magnitude (NaN-aware)."""
    a, b = a.replace([np.inf, -np.inf], np.nan), b.replace([np.inf, -np.inf], np.nan)
    pick_b = a.isna() | (b.abs() > a.abs())
    return a.where(~pick_b, b)


def history_features(daily: Mapping[str, np.ndarray], d: int) -> dict[str, np.ndarray]:
    """Own-history columns for day index d from per-day summaries [N, D] (keys energy_q50, band_rel, p_fault, alarm;
    any subset): medians over days d-7..d-1 (`*7`) and d-28..d-8 (`*28`), mean alarm rate over d-7..d-1."""
    out: dict[str, np.ndarray] = {}
    names = {"band_rel": "hist_band_rel", "energy_q50": "hist_energy", "p_fault": "hist_p_fault"}
    for k, x in daily.items():
        x = np.atleast_2d(np.asarray(x, float))
        n = x.shape[0]
        w7, w28 = x[:, max(0, d - 7):d], x[:, max(0, d - 28):max(0, d - 7)]
        if k == "alarm":
            out["hist_alarm7"] = _nanstat(w7, np.nanmean, n)
        elif k in names:
            out[names[k] + "7"], out[names[k] + "28"] = _nanstat(w7, np.nanmedian, n), _nanstat(w28, np.nanmedian, n)
    return out


def _nanstat(w: np.ndarray, fn, n: int) -> np.ndarray:
    if w.shape[-1] == 0 or not np.isfinite(w).any():
        return np.full(n, np.nan)
    ok = np.isfinite(w).any(-1)
    res = np.full(n, np.nan)
    res[ok] = fn(w[ok], axis=-1)
    return res


# ----------------------------------------------------------------------------------- one envelope -> row
def prediction_row(pred: Any, *, day: str, context: Optional[Mapping[str, Any]] = None,
                   substitute: Any = None, ledger: Optional[Mapping[str, Any]] = None,
                   telemetry_prev: Optional[Mapping[str, Any]] = None,
                   thresholds: Optional[Mapping[str, Any]] = None,
                   history: Optional[Mapping[str, Any]] = None) -> dict[str, Any]:
    """Raw row for one contract `Prediction` (what J4's loop slot calls per prediction).

    context:   battery (bool), seen (bool), month (1-12), sky (0/1/2, `sky_class`), bucket (ledger key), kind,
               energy_scale (kWp x 1 h for PV, the asset's mean daily kWh for load, rated kW x 1 h for turbines),
               telemetry_age_h, resid_yday (measured / expected PV yesterday - 1), park_consensus (M6).
    substitute: the substitute's `Prediction` (same asset/day) or None.
    ledger:    the ledger snapshot for (witness, bucket) as a dict with LEDGER_COLUMNS names (missing -> NaN).
    telemetry_prev: {"y": [96] measured kW of the previous day, "valid": [96] 0/1, "energy_ref": kWh} or None.
    thresholds: the witness's calibration `thresholds` ({ood, p_fault, ...}).
    history:   this witness's earlier rows for the same asset, oldest first, one per day up to yesterday (up to 28):
               {"energy_q50": [...], "band_rel": [...]} for forecasts, {"p_fault": [...], "alarm": [...]} for
               detectors (the same-named columns of earlier `prediction_row` results), plus "q50_prev" (yesterday's
               [96] q50) or "p_fault_prev" to flag a frozen repeat. Missing -> NaN (the model was trained with it).
    """
    ctx = dict(context or {})
    th = dict(thresholds or {})
    model = str(pred.model)
    fam = family_of(model) or ("ledger_only" if model in LEDGER_ONLY else "")
    kind_pv = "pv" if model in ("pvnet", "sitehealth") else "other"
    s = summarize_forecast(pred.q10, pred.q50, pred.q90, kind_pv)
    extra = dict(getattr(pred, "extra", None) or {})
    row: dict[str, Any] = {
        "model": model, "model_version": str(pred.model_version), "asset_id": str(pred.asset_id), "day": str(day),
        "kind": ctx.get("kind", KIND_OF.get(model, "site")), "family": fam, "bucket": ctx.get("bucket"),
        "battery": float(bool(ctx.get("battery", False))), "seen": float(bool(ctx.get("seen", True))),
        "month": float(ctx.get("month", np.nan)), "sky": float(ctx.get("sky", np.nan)),
        "ood_score": float(pred.ood_score), "ood_thr": float(th.get("ood") or np.nan),
        "member_std": float(pred.member_std), "energy_q50": float(s["energy_q50"][0]),
        "band_rel": float(s["band_rel"][0]), "is_fallback": float(extra.get("is_fallback", 0) or 0),
        "telemetry_age_h": float(ctx.get("telemetry_age_h", np.nan)),
        "energy_scale": float(ctx.get("energy_scale", np.nan)),
        "resid_yday": float(ctx.get("resid_yday", np.nan)), "park_consensus": float(ctx.get("park_consensus", np.nan)),
    }
    if telemetry_prev is not None and telemetry_prev.get("y") is not None:
        valid, eref = telemetry_prev.get("valid"), telemetry_prev.get("energy_ref")
        q = input_quality(np.asarray(telemetry_prev["y"], float)[None],
                          None if valid is None else np.asarray(valid)[None],
                          None if eref is None else np.array([float(eref)]))
        row.update({k: float(v[0]) for k, v in q.items()})
    if pred.fault_probs:
        probs = np.array([[pred.fault_probs.get(c, 0.0) for c in PV_CLASSES]])
        d = detector_summary(probs, float(extra.get("p_fault_threshold") or th.get("p_fault") or 0.5),
                             p_fault_std=np.array([extra.get("p_fault_std", np.nan)], float))
        row.update({k: (v[0] if k == "top_class" else float(v[0])) for k, v in d.items()})
    elif model == "windnbm":
        cls = list(extra.get("step_classes") or [])
        abn = [c for c in cls if c != "normal"]
        row.update({"alarm": float(bool(pred.anomaly_class and pred.anomaly_class != "normal")),
                    "p_fault": float(pred.anomaly_score or 0.0), "p_top": float(pred.anomaly_score or 0.0),
                    "top_class": f"wind:{pred.anomaly_class or 'normal'}",
                    "abn_share": (len(abn) / len(cls)) if cls else np.nan})
    if substitute is not None:
        ss = summarize_forecast(substitute.q10, substitute.q50, substitute.q90, kind_pv)
        row.update({"sub_model": str(substitute.model), "sub_energy": float(ss["energy_q50"][0]),
                    "sub_band_rel": float(ss["band_rel"][0])})
    for c in LEDGER_COLUMNS:
        if ledger is not None and c in ledger and ledger[c] is not None:
            row[c] = float(ledger[c])
    if history:
        daily = {k: np.asarray(history[k], float)[None] for k in ("energy_q50", "band_rel", "p_fault", "alarm")
                 if history.get(k) is not None and len(history[k])}
        n_days = max([v.shape[1] for v in daily.values()], default=0)
        daily = {k: np.pad(v, ((0, 0), (n_days - v.shape[1], 0)), constant_values=np.nan) for k, v in daily.items()}
        row.update({k: float(v[0]) for k, v in history_features(daily, n_days).items()})
        if history.get("q50_prev") is not None:
            row["repeat_prev"] = float(repeats(np.asarray(pred.q50, float)[None],
                                               np.asarray(history["q50_prev"], float)[None])[0])
        elif history.get("p_fault_prev") is not None and "p_fault" in row:
            row["repeat_prev"] = float(abs(row["p_fault"] - float(history["p_fault_prev"])) < 1e-9)
    return row


def repeats(today: np.ndarray, prev: np.ndarray) -> np.ndarray:
    """1 where today's output equals the previous day's step for step (a frozen witness), 0 otherwise, [N, ...]."""
    a, b = np.asarray(today, float), np.asarray(prev, float)
    ax = tuple(range(1, a.ndim))
    same = np.all(np.abs(np.nan_to_num(a) - np.nan_to_num(b)) < 1e-9, axis=ax) if a.ndim > 1 else np.abs(a - b) < 1e-9
    nonzero = np.any(np.abs(np.nan_to_num(a)) > 1e-6, axis=ax) if a.ndim > 1 else np.abs(a) > 1e-6
    return (same & nonzero).astype(float)


def rows_frame(rows: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    """List of raw-row dicts -> DataFrame with every RAW_COLUMNS column present (NaN / None where missing)."""
    df = pd.DataFrame(list(rows))
    for c in RAW_COLUMNS:
        if c not in df.columns:
            df[c] = np.nan
    return df


# ------------------------------------------------------------------------------------------- reasons
def _pct(x: float) -> str:
    return f"{abs(x) * 100:.0f} %"


def _bucket_words(row: Mapping[str, Any]) -> str:
    kind = row.get("kind") or "site"
    parts = []
    if kind == "site":
        parts.append("battery homes" if float(row.get("battery") or 0) > 0.5 else "homes without battery")
        if float(row.get("seen", 1) or 0) < 0.5:
            parts.insert(0, "new")
    else:
        parts.append({"turbine": "turbines", "zone": "zones", "park": "parks", "feeder": "feeders"}.get(kind, kind))
    return " ".join(parts)


def reason_text(feature: str, row: Mapping[str, Any]) -> Optional[str]:
    """Plain words for one feature of one raw row; None when the feature has nothing worth saying."""
    g = lambda k: row.get(k)  # noqa: E731
    v = lambda k: float(g(k)) if g(k) is not None and not (isinstance(g(k), float) and math.isnan(g(k))) else None  # noqa: E731
    if feature in ("trust_main", "trust_gap", "trust_lo80", "trust_cal", "trust_sharp", "trust_ap", "trust_ms"):
        t = v(feature if feature in ("trust_cal", "trust_sharp", "trust_ap", "trust_ms") else "trust_main")
        name = {"trust_cal": "band calibration trust", "trust_sharp": "accuracy trust",
                "trust_ap": "alarm trust", "trust_ms": "healthy-call trust"}.get(feature, "trust")
        return None if t is None else f"{name} {t:.2f} for {_bucket_words(row)}"
    if feature == "n_eff":
        n = v("n_eff")
        return None if n is None else ("no track record yet" if n < 0.5 else
                                       f"little evidence yet ({n:.0f} days)" if n < 5 else
                                       f"track record of {n:.0f} days")
    if feature == "trust_trend3":
        t = v("trust_trend3")
        return None if t is None or abs(t) < 0.01 else f"trust {'falling' if t < 0 else 'rising'} ({t:+.2f} in 3 days)"
    if feature == "health_alert":
        return "health alert on this model" if (v("health_alert") or 0) > 0.5 else None
    if feature in ("ood_score", "ood_ratio"):
        o = v("ood_score")
        return None if o is None else (f"unusual input (OOD {o:.1f})" if o >= (v("ood_thr") or 3.0)
                                       else f"familiar input (OOD {o:.1f})")
    if feature in ("rel_vs_sub", "abs_rel_vs_sub"):
        e50, sub = v("energy_q50"), v("sub_energy")
        if e50 is None or sub is None:
            return None
        rel = (e50 - sub) / max(abs(sub), 0.1)
        name = {"pvlib": "pvlib", "persistence": "yesterday's load", "persistence_load": "yesterday's load"}.get(
            str(g("sub_model")), str(g("sub_model") or "the substitute"))
        return f"agrees with {name}" if abs(rel) < 0.1 else f"disagrees with {name} by {_pct(rel)}"
    if feature == "rel_vs_yday":
        e50, yd = v("energy_q50"), v("energy_yday")
        if e50 is None or yd is None:
            return None
        rel = (e50 - yd) / max(abs(yd), 0.1)
        return None if abs(rel) < 0.15 else f"{_pct(rel)} {'above' if rel > 0 else 'below'} yesterday's measured"
    if feature in ("band_rel", "band_scale"):
        if feature == "band_scale":
            b = v("band_scale")
            return None if b is None or abs(b - 1) < 0.1 else (
                f"bands recently {'too narrow' if b > 1 else 'too wide'} (scale {b:.1f}x)")
        b = v("band_rel")
        return None if b is None else f"band width {b * 100:.0f} % of the forecast"
    if feature in ("member_std_rel",):
        m = v("member_std")
        return None if m is None else "ensemble members disagree" if (v("member_std") or 0) > 0 else None
    if feature in ("in_gap_frac",):
        x = v("in_gap_frac")
        return None if x is None or x < 0.05 else f"input has gaps ({_pct(x)} missing)"
    if feature == "in_frozen":
        return "frozen sensor in the input" if (v("in_frozen") or 0) > 0.5 else None
    if feature == "in_unit_jump":
        return "input energy jumped (unit error?)" if (v("in_unit_jump") or 0) > 0.5 else None
    if feature == "is_fallback":
        return "model on fallback" if (v("is_fallback") or 0) > 0.5 else None
    if feature == "telemetry_age_h":
        a = v("telemetry_age_h")
        return None if a is None or a < 1 else f"telemetry {a:.0f} h old"
    if feature in ("sub_trust_main", "sub_gap"):
        t = v("sub_trust_main")
        return None if t is None else f"substitute {g('sub_model') or ''} trusted {t:.2f}".replace("  ", " ")
    if feature in ("p_fault", "p_top", "alarm", "top_class"):
        p, tc = v("p_fault"), g("top_class")
        if p is None:
            return None
        tc = str(tc or "").replace("wind:", "").replace("_", " ")
        return f"calls {tc} with p {p:.2f}" if (v("alarm") or 0) > 0.5 else f"calls healthy (p fault {p:.2f})"
    if feature == "p_fault_std":
        s = v("p_fault_std")
        return None if s is None or s < 0.05 else f"fault members disagree (sd {s:.2f})"
    if feature == "resid_yday":
        r = v("resid_yday")
        return None if r is None or abs(r) < 0.1 else f"PV {_pct(r)} {'above' if r > 0 else 'below'} expected yesterday"
    if feature == "park_consensus":
        c = v("park_consensus")
        return None if c is None else f"{_pct(c)} of the park in the same state"
    if feature == "abn_share":
        a = v("abn_share")
        return None if a is None or a <= 0 else f"abnormal {_pct(a)} of the day"
    if feature in ("sky",):
        s = v("sky")
        return None if s is None else ["clear sky day", "mixed sky day", "overcast day"][int(s)]
    if feature in ("battery", "seen"):
        return f"context: {_bucket_words(row)}"
    if feature in ("energy_rel",):
        return None
    return None


def reasons_for(contrib: np.ndarray, names: Sequence[str], rows: pd.DataFrame, p_valid: np.ndarray,
                k: int = 3) -> list[list[str]]:
    """contrib [N, F(+1 bias)] LightGBM pred_contrib -> up to k reasons per row. For a likely-valid prediction the
    strongest positive contributions are named, otherwise the strongest negative ones."""
    c = np.asarray(contrib)[:, :len(names)]
    out = []
    recs = rows.to_dict("records")
    for i, rec in enumerate(recs):
        sign = 1.0 if p_valid[i] >= 0.5 else -1.0
        order = np.argsort(-sign * c[i])
        said, seen_txt = [], set()
        for j in order:
            if sign * c[i, j] <= 0 or len(said) >= k:
                break
            txt = reason_text(names[j], rec)
            if txt and txt not in seen_txt:
                said.append(txt)
                seen_txt.add(txt)
        out.append(said)
    return out
