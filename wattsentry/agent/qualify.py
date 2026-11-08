"""Qualify: one status per asset per tick. Only `eligible` / `derated` assets are in the flexibility pool; a work
order (triage.py) is how excluded assets get back in. `assess()` returns the status plus what triage needs (fault,
crew, lost kWh, MW at risk, alert, curtailed energy, offline flag). Pure functions; the only I/O is reading
`models/thresholds.json`, cached by mtime.

Sites (prediction = M4 SiteHealth envelope, q10/q50/q90 = expected PV kW):
  stale telemetry / data gap / sensor fault          -> excluded (sensor fault -> remote check, never a truck)
  OOD, wide conformal interval or P_fault spread high -> human_review (no work order, out of the pool)
  confident battery fault                            -> excluded, battery_ok = False
  confident PV fault                                 -> derated, headroom = 1 - estimated loss, battery stays in
Turbines (prediction = M6 WindNBM envelope, q50 = expected kW, anomaly_class / anomaly_score):
  stale / data_gap -> excluded + one offline window; icing -> excluded + safety alert (no truck);
  fault_stop -> excluded + work order; underperformance / yaw -> derated + inspection;
  curtailment -> eligible + curtailed-energy log (no work order); anemometer_fault -> remote check;
  overheating -> eligible + warning. Wind never enters the LP: the status feeds park availability.
"""

from __future__ import annotations

import copy
import json
import os
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Union

import numpy as np

from wattsentry.contracts import AssetStatus, CrewType, Prediction, PvFault, Severity, Status, WindAnomaly

# --- thresholds ---------------------------------------------------------------------------------------------
# Defaults; models/thresholds.json (written by the uncertainty calibration) overrides them.
DEFAULT_THRESHOLDS: dict[str, Any] = {
    "version": 1,
    "source_run": "builtin-defaults",
    "site": {
        "stale_minutes": 60,
        "pfault_confident": 0.6,
        # Soiling needs more: on homes M4 never saw, a steady gap to the expected PV reads as soiling
        # (only 20-34 % of those calls were right); every other class is 90-100 % right at pfault_confident.
        "soiling_confident": 0.7,
        "pfault_std_max": 0.15,
        "rel_width_max": 0.8,
        "ood_max": 3.0,
    },
    "turbine": {"stale_minutes": 60, "anomaly_conf_min": 0.6, "ood_max": 3.0},
}
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_THRESHOLDS_PATH = REPO_ROOT / "models" / "thresholds.json"
THRESHOLDS_ENV = "WATTSENTRY_THRESHOLDS"  # optional override of the path

Thresholds = Union[None, str, os.PathLike, Mapping[str, Any]]
_cache: dict[str, tuple[float, "_Resolved"]] = {}


class _Resolved(dict):
    """A thresholds dict already merged over the defaults (so per-asset calls skip the merge)."""


def _merge(base: dict[str, Any], over: Mapping[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for k, v in over.items():
        if isinstance(v, Mapping) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


def load_thresholds(path: Optional[Union[str, os.PathLike]] = None) -> dict[str, Any]:
    """Thresholds from `path` (default `$WATTSENTRY_THRESHOLDS` or models/thresholds.json), merged over the defaults.

    Missing file -> defaults. Missing keys -> defaults (so an older file keeps working). Cached by path + mtime.
    """
    p = Path(path or os.environ.get(THRESHOLDS_ENV) or DEFAULT_THRESHOLDS_PATH)
    try:
        mtime = p.stat().st_mtime
    except OSError:
        return _Resolved(copy.deepcopy(DEFAULT_THRESHOLDS))
    key = str(p)
    hit = _cache.get(key)
    if hit is None or hit[0] != mtime:
        with open(p) as f:
            hit = (mtime, _Resolved(_merge(DEFAULT_THRESHOLDS, json.load(f))))
        _cache[key] = hit
    return hit[1]


def resolve_thresholds(thresholds: Thresholds = None) -> dict[str, Any]:
    """None -> load_thresholds(); a path -> that file; a (partial) dict -> merged over the defaults."""
    if thresholds is None or isinstance(thresholds, (str, os.PathLike)):
        return load_thresholds(thresholds)
    if isinstance(thresholds, _Resolved):
        return thresholds
    return _Resolved(_merge(DEFAULT_THRESHOLDS, thresholds))


# --- per-class policy ---------------------------------------------------------------------------------------
# Loss fraction used when no actual power is available to measure it (q50 vs actual is preferred).
PV_DEFAULT_LOSS = {
    PvFault.SOILING.value: 0.08,
    PvFault.SHADING.value: 0.12,
    PvFault.STRING_OUTAGE.value: 0.30,
    PvFault.CLIPPING.value: 0.05,
    PvFault.OVERVOLTAGE_TRIP.value: 0.20,
    PvFault.BATTERY_FADE.value: 0.0,
    PvFault.SENSOR_FAULT.value: 0.0,
}
WIND_DEFAULT_LOSS = {
    WindAnomaly.FAULT_STOP.value: 1.0,
    WindAnomaly.UNDERPERFORMANCE.value: 0.15,
    WindAnomaly.CURTAILMENT.value: 0.30,
    WindAnomaly.ICING.value: 1.0,
    WindAnomaly.OVERHEATING.value: 0.0,
    WindAnomaly.ANEMOMETER_FAULT.value: 0.0,
    WindAnomaly.DATA_GAP.value: 1.0,
}
PV_LABEL = {
    "soiling": "Soiling",
    "shading": "Shading",
    "string_outage": "String outage",
    "clipping": "Inverter clipping",
    "overvoltage_trip": "Overvoltage trips",
    "battery_fade": "Battery fade",
    "sensor_fault": "Sensor fault",
}
DERATE_MIN, DERATE_MAX = 0.05, 0.99  # derated headroom stays strictly inside (0, 1)
PRODUCTION_FRAC = 0.05  # production steps: q50 >= 5% of the day's peak q50
EPS = 1e-3
_HEALTHY = PvFault.HEALTHY.value


@dataclass
class Assessment:
    """`assess()` output: the AssetStatus plus everything triage needs. One per asset per tick."""

    asset_id: str
    asset_kind: str  # "site" | "turbine"
    status: AssetStatus
    group_id: Optional[str] = None  # feeder_id (site) or park_id (turbine)
    fault: Optional[str] = None  # PvFault / WindAnomaly value behind the decision
    p_fault: float = 0.0  # P(fault) for sites, anomaly_score for turbines
    crew_type: Optional[str] = None  # CrewType value when a work order is warranted, else None
    alert: Optional[tuple[str, str, str]] = None  # (severity, code, message)
    threshold: Optional[float] = None  # the threshold that decided the status (audit)
    loss_frac: float = 0.0  # estimated fraction of expected energy lost
    expected_kwh_day: float = 0.0  # Σ q50·Δt scaled to 24 h
    lost_kwh_day: float = 0.0  # Σ (q50 − actual)+·Δt, scaled to 24 h
    lost_flex_mwh_day: float = 0.0  # flexibility energy the pool loses (sites; 0 for turbines)
    mw_at_risk: float = 0.0  # capacity out of the pool / out of production, MW
    curtailed_kwh: Optional[float] = None  # turbines in curtailment: energy capped (log entry, no work order)
    offline: bool = False  # stale / data gap: triage opens one offline window
    measured: bool = False  # loss measured from actual power (vs class default)
    prediction: Optional[Prediction] = field(default=None, repr=False)
    actual: Optional[np.ndarray] = field(default=None, repr=False)
    telemetry_age_min: Optional[float] = None


# --- helpers ------------------------------------------------------------------------------------------------
def asset_kind(asset: Mapping[str, Any], prediction: Optional[Prediction] = None) -> str:
    k = asset.get("asset_kind")
    if k in ("site", "turbine"):
        return k
    if asset.get("turbine_id"):
        return "turbine"
    if asset.get("site_id"):
        return "site"
    if prediction is not None and prediction.anomaly_class is not None:
        return "turbine"
    return "site"


def asset_id_of(asset: Mapping[str, Any], prediction: Optional[Prediction] = None) -> str:
    for k in ("site_id", "turbine_id", "asset_id"):
        if asset.get(k):
            return str(asset[k])
    if prediction is not None:
        return prediction.asset_id
    raise ValueError("asset has no site_id / turbine_id / asset_id")


def _minutes(age: Union[None, float, int, timedelta]) -> Optional[float]:
    if age is None:
        return None
    if isinstance(age, timedelta):
        return age.total_seconds() / 60.0
    return float(age)


def production_mask(q50: np.ndarray) -> np.ndarray:
    peak = float(q50.max()) if q50.size else 0.0
    return q50 >= max(EPS, PRODUCTION_FRAC * peak)


def rel_width(prediction: Prediction, q50: Optional[np.ndarray] = None, mask: Optional[np.ndarray] = None) -> float:
    """Mean over production steps of (q90 − q10) / max(q50, eps). 0 when nothing is produced (night, calm).

    q50 / mask: optional precomputed `np.asarray(prediction.q50, float)` / `production_mask(q50)` (same result).
    """
    if q50 is None:
        q50 = np.asarray(prediction.q50, dtype=float)
    m = production_mask(q50) if mask is None else mask
    if not m.any():
        return 0.0
    q10 = np.asarray(prediction.q10, dtype=float)[m]
    q90 = np.asarray(prediction.q90, dtype=float)[m]
    return float(np.mean((q90 - q10) / np.maximum(q50[m], EPS)))


def _expected_day_kwh(q50: np.ndarray) -> float:
    if not q50.size:
        return 0.0
    return float(q50.mean() * 24.0)  # Σ q50·Δt scaled from the horizon to 24 h


def _measured_loss(q50: np.ndarray, actual: Optional[np.ndarray],
                   mask: Optional[np.ndarray] = None) -> Optional[float]:
    """Σ (q50 − actual)+ / Σ q50 over observed production steps; None if nothing to measure on."""
    if actual is None:
        return None
    n = min(len(actual), len(q50))
    if n == 0:
        return None
    a, e = actual[:n], q50[:n]
    m = (production_mask(q50) if mask is None else mask)[:n] & np.isfinite(a)
    if not m.any():
        return None
    exp = float(e[m].sum())
    if exp <= EPS:
        return None
    return float(np.clip(e[m] - a[m], 0.0, None).sum() / exp)


def _curtailed_kwh(q50: np.ndarray, actual: Optional[np.ndarray], dt_h: float) -> Optional[float]:
    """Energy capped so far: Σ (q50 − actual)+·Δt over observed steps (not extrapolated)."""
    if actual is None:
        return None
    n = min(len(actual), len(q50))
    a, e = actual[:n], q50[:n]
    m = np.isfinite(a)
    return float(np.clip(e[m] - a[m], 0.0, None).sum() * dt_h)


def _status(aid: str, status: Status, reason: str, detail: str, headroom: Optional[float] = None,
            battery_ok: bool = True) -> AssetStatus:
    if status in (Status.EXCLUDED, Status.HUMAN_REVIEW):
        h = 0.0
    elif status == Status.ELIGIBLE:
        h = 1.0
    else:
        h = float(min(DERATE_MAX, max(DERATE_MIN, 1.0 if headroom is None else headroom)))
    return AssetStatus(asset_id=aid, status=status.value, headroom_factor=round(h, 4), reason=reason,
                       battery_ok=battery_ok, detail=detail)


def _top_fault(prediction: Prediction) -> tuple[Optional[str], float]:
    """Most likely non-healthy class and P_fault (= 1 − P(healthy))."""
    best = max((kv for kv in (prediction.fault_probs or {}).items() if kv[0] != _HEALTHY),
               key=lambda kv: kv[1], default=None)
    return (None, 0.0) if best is None else (best[0], prediction.p_fault)


# --- public API ---------------------------------------------------------------------------------------------
def assess(
    asset: Mapping[str, Any],
    prediction: Optional[Prediction],
    telemetry_age: Union[None, float, int, timedelta],
    thresholds: Thresholds = None,
    *,
    actual: Optional[Sequence[float]] = None,
    data_gap: bool = False,
    sensor_fault: bool = False,
    battery_fault: Optional[bool] = None,
    lost_flex_mwh_day: Optional[float] = None,
) -> Assessment:
    """Qualify one asset for this tick and keep what triage needs.

    asset: `sites` / `turbines` row (needs site_id or turbine_id; uses feeder_id / park_id, battery_kwh / battery_kw,
        rated_kw when present). prediction: M4 (site) or M6 (turbine) envelope; None = no model output -> excluded.
    telemetry_age: minutes (or timedelta) since the last good reading; None = unknown, not checked.
    actual: observed power (kW) aligned with prediction steps from t0; NaN = missing / not yet observed. Used to
        measure the loss (q50 vs actual); without it the per-class default loss is used.
    data_gap / sensor_fault / battery_fault: flags from telemetry checks (Pandera / inverter codes / battery BMS).
    lost_flex_mwh_day: override of the flexibility energy lost.
    """
    th = resolve_thresholds(thresholds)
    kind = asset_kind(asset, prediction)
    fn = _assess_turbine if kind == "turbine" else _assess_site
    arr = None if actual is None else np.asarray(actual, dtype=float)
    return fn(asset, prediction, _minutes(telemetry_age), th, arr, data_gap, sensor_fault, battery_fault,
              lost_flex_mwh_day)


def qualify_many(
    assets: Sequence[Mapping[str, Any]],
    predictions: Mapping[str, Prediction],
    telemetry_ages: Mapping[str, Union[None, float, timedelta]],
    thresholds: Thresholds = None,
    *,
    actuals: Optional[Mapping[str, Sequence[float]]] = None,
    flags: Optional[Mapping[str, Mapping[str, Any]]] = None,
    park_consensus: bool = True,
    park_min_turbines: int = 3,
    park_share: float = 0.6,
) -> list[Assessment]:
    """Assess a whole tick (all sites + turbines). Dicts are keyed by asset id; `flags[id]` = assess() keywords.

    Park consensus: if ≥ `park_share` of a park's turbines (and ≥ `park_min_turbines`) are flagged fault_stop /
    underperformance at the same tick, it is a park-wide cap (curtailment or grid limit), not N broken turbines:
    they become eligible + curtailed-energy log, no work orders.
    """
    th = resolve_thresholds(thresholds)
    actuals = actuals or {}
    flags = flags or {}
    out = []
    for a in assets:
        aid = asset_id_of(a)
        out.append(assess(a, predictions.get(aid), telemetry_ages.get(aid), th, actual=actuals.get(aid),
                          **dict(flags.get(aid, {}))))
    if park_consensus:
        _apply_park_consensus(out, park_min_turbines, park_share)
    return out


def _apply_park_consensus(items: list[Assessment], min_n: int, share: float) -> None:
    parks: dict[str, list[int]] = {}
    for i, x in enumerate(items):
        if x.asset_kind == "turbine" and x.group_id and not x.offline:
            parks.setdefault(x.group_id, []).append(i)
    capped = {WindAnomaly.FAULT_STOP.value, WindAnomaly.UNDERPERFORMANCE.value}
    for park, idx in parks.items():
        hit = [i for i in idx if items[i].fault in capped and items[i].crew_type is not None]
        if len(hit) < min_n or len(hit) < share * len(idx):
            continue
        for i in hit:
            x = items[i]
            pred = x.prediction
            dt_h = (pred.step_minutes if pred else 15) / 60.0
            q50 = np.asarray(pred.q50, dtype=float) if pred else np.zeros(0)
            items[i] = Assessment(
                asset_id=x.asset_id, asset_kind="turbine", group_id=park, fault=WindAnomaly.CURTAILMENT.value,
                p_fault=x.p_fault, threshold=float(share),
                status=_status(x.asset_id, Status.ELIGIBLE, "curtailed_park_wide",
                               f"{len(hit)}/{len(idx)} turbines in {park} capped at once: park-wide curtailment, "
                               f"not a turbine fault (no work order)."),
                expected_kwh_day=x.expected_kwh_day, curtailed_kwh=_curtailed_kwh(q50, x.actual, dt_h),
                measured=x.measured, prediction=pred, actual=x.actual, telemetry_age_min=x.telemetry_age_min,
            )


# --- sites --------------------------------------------------------------------------------------------------
class _SiteStats:
    """Per-prediction values `_assess_site` derives from q10/q50/q90/fault_probs (the same prediction is assessed
    at every tick of its day). Valid while the prediction still holds the same list / dict objects: Prediction
    fields are replaced, never edited in place (e.g. world mocks reassign fault_probs), and `_site_stats` checks."""

    __slots__ = ("pred", "q10l", "q50l", "q90l", "probs", "step", "q50", "exp_kwh", "peak_kw", "prod", "top", "_rw")

    def __init__(self, pred: Prediction):
        self.pred, self.q10l, self.q50l, self.q90l = pred, pred.q10, pred.q50, pred.q90
        self.probs, self.step = pred.fault_probs, pred.step_minutes
        self.q50 = q50 = np.asarray(pred.q50, dtype=float)
        self.exp_kwh = _expected_day_kwh(q50)
        self.peak_kw = float(q50.max()) if q50.size else 0.0
        self.prod = production_mask(q50)
        self.top = _top_fault(pred)
        self._rw: Optional[float] = None

    def valid_for(self, pred: Prediction) -> bool:
        return (self.pred is pred and self.q10l is pred.q10 and self.q50l is pred.q50 and self.q90l is pred.q90
                and self.probs is pred.fault_probs and self.step == pred.step_minutes)

    def rel_width(self) -> float:
        if self._rw is None:
            self._rw = rel_width(self.pred, q50=self.q50, mask=self.prod)
        return self._rw


_SITE_STATS: dict[int, _SiteStats] = {}
_SITE_STATS_MAX = 2048  # > sites per tick; entries hold their prediction alive, so an id cannot be reused


def _site_stats(pred: Prediction) -> _SiteStats:
    st = _SITE_STATS.get(id(pred))
    if st is None or not st.valid_for(pred):
        if len(_SITE_STATS) >= _SITE_STATS_MAX:
            _SITE_STATS.clear()
        st = _SITE_STATS[id(pred)] = _SiteStats(pred)
    return st


def _assess_site(asset, pred, age, th, actual, data_gap, sensor_fault, battery_fault, flex_override) -> Assessment:
    t = th["site"]
    aid = asset_id_of(asset, pred)
    base = dict(asset_id=aid, asset_kind="site", group_id=asset.get("feeder_id"), prediction=pred, actual=actual,
                telemetry_age_min=age)
    batt_kwh = float(asset.get("battery_kwh") or 0.0)
    batt_kw = float(asset.get("battery_kw") or 0.0)
    batt_mw = batt_kw / 1000.0
    batt_flex_mwh = batt_kwh / 1000.0  # one cycle per day: what the pool loses if the battery leaves

    def out(status: AssetStatus, **kw: Any) -> Assessment:
        return Assessment(status=status, **base, **kw)

    if age is not None and age > t["stale_minutes"]:
        return out(_status(aid, Status.EXCLUDED, "stale_telemetry", f"No telemetry for {age:.0f} min "
                           f"(limit {t['stale_minutes']} min): out of the pool until it reports."),
                   threshold=float(t["stale_minutes"]), offline=True, mw_at_risk=batt_mw,
                   lost_flex_mwh_day=batt_flex_mwh)
    if data_gap:
        return out(_status(aid, Status.EXCLUDED, "data_gap", "Telemetry gap / suspect data: out of the pool."),
                   offline=True, mw_at_risk=batt_mw, lost_flex_mwh_day=batt_flex_mwh)
    if pred is None:
        return out(_status(aid, Status.EXCLUDED, "no_prediction", "No model output this tick: out of the pool."),
                   mw_at_risk=batt_mw, lost_flex_mwh_day=batt_flex_mwh)

    st = _site_stats(pred)
    q50, exp_kwh, peak_kw, prod = st.q50, st.exp_kwh, st.peak_kw, st.prod
    fault, p_fault = st.top
    confident = fault is not None and p_fault >= t["pfault_confident"]
    if fault == PvFault.SOILING.value and float(pred.fault_probs.get(fault, 0.0)) < t["soiling_confident"]:
        confident = False  # P(soiling) itself, not P(fault)
    base["expected_kwh_day"] = exp_kwh

    if sensor_fault or (confident and fault == PvFault.SENSOR_FAULT.value):
        p = 1.0 if sensor_fault else p_fault
        flex = batt_flex_mwh if flex_override is None else flex_override
        return out(_status(aid, Status.EXCLUDED, "sensor_fault",
                           f"Sensor fault (P={p:.2f}): readings untrusted, out of the pool; remote check."),
                   fault=PvFault.SENSOR_FAULT.value, p_fault=p, crew_type=CrewType.REMOTE.value,
                   threshold=float(t["pfault_confident"]), lost_flex_mwh_day=flex, mw_at_risk=batt_mw)

    # "Not sure" signals -> a human looks, no truck, out of the pool.
    if pred.ood_score > t["ood_max"]:
        return out(_status(aid, Status.HUMAN_REVIEW, "ood",
                           f"Out of distribution (score {pred.ood_score:.2f} > {t['ood_max']}): human review."),
                   p_fault=p_fault, fault=fault, threshold=float(t["ood_max"]))
    rw = st.rel_width()
    if rw > t["rel_width_max"]:
        return out(_status(aid, Status.HUMAN_REVIEW, "wide_interval",
                           f"Forecast interval too wide (relative width {rw:.2f} > {t['rel_width_max']}): "
                           f"human review."),
                   p_fault=p_fault, fault=fault, threshold=float(t["rel_width_max"]))
    p_std = pred.extra.get("p_fault_std") if pred.extra else None
    if p_std is not None and float(p_std) > t["pfault_std_max"]:
        return out(_status(aid, Status.HUMAN_REVIEW, "ensemble_spread",
                           f"Ensemble disagrees on P(fault) (std {float(p_std):.2f} > {t['pfault_std_max']}): "
                           f"human review."),
                   p_fault=p_fault, fault=fault, threshold=float(t["pfault_std_max"]))

    if battery_fault or (confident and fault == PvFault.BATTERY_FADE.value):
        p = 1.0 if battery_fault else p_fault
        flex = batt_flex_mwh if flex_override is None else flex_override
        return out(_status(aid, Status.EXCLUDED, "battery_fault",
                           f"Battery fault (P={p:.2f}): out of the pool until serviced.", battery_ok=False),
                   fault=PvFault.BATTERY_FADE.value, p_fault=p, crew_type=CrewType.RESIDENTIAL.value,
                   threshold=float(t["pfault_confident"]), lost_flex_mwh_day=flex, mw_at_risk=batt_mw)

    if confident:
        measured = _measured_loss(q50, actual, prod)
        loss = measured if measured is not None else PV_DEFAULT_LOSS.get(fault, 0.1)
        loss = float(min(1.0, max(0.0, loss)))
        lost_kwh = loss * exp_kwh
        battery_ok = battery_fault is not True
        # PV the battery could no longer absorb: capped by one battery cycle per day.
        flex = flex_override if flex_override is not None else min(batt_kwh, lost_kwh) / 1000.0
        headroom = 1.0 - loss
        how = "measured" if measured is not None else "class default"
        batt_txt = " Battery stays in the pool." if batt_kwh > 0 and battery_ok else ""
        return out(_status(aid, Status.DERATED, f"pv_fault:{fault}",
                           f"{PV_LABEL.get(fault, fault)} (P={p_fault:.2f}): {loss:.0%} below expected ({how}, "
                           f"{lost_kwh:.1f} kWh/day); PV headroom cut to {min(DERATE_MAX, max(DERATE_MIN, headroom)):.2f}."
                           f"{batt_txt}", headroom=headroom, battery_ok=battery_ok),
                   fault=fault, p_fault=p_fault, crew_type=CrewType.RESIDENTIAL.value,
                   threshold=float(t["pfault_confident"]), loss_frac=loss, lost_kwh_day=lost_kwh,
                   lost_flex_mwh_day=flex, mw_at_risk=peak_kw * loss / 1000.0, measured=measured is not None)

    return out(_status(aid, Status.ELIGIBLE, "ok", f"Healthy (P(fault)={p_fault:.2f})."), p_fault=p_fault,
               threshold=float(t["pfault_confident"]))


# --- turbines -----------------------------------------------------------------------------------------------
def _assess_turbine(asset, pred, age, th, actual, data_gap, sensor_fault, battery_fault, flex_override) -> Assessment:
    t = th["turbine"]
    aid = asset_id_of(asset, pred)
    park = asset.get("park_id") or (aid.split(".")[0] if "." in aid else None)
    rated_kw = float(asset.get("rated_kw") or 0.0)
    base = dict(asset_id=aid, asset_kind="turbine", group_id=park, prediction=pred, actual=actual,
                telemetry_age_min=age)

    def out(status: AssetStatus, **kw: Any) -> Assessment:
        return Assessment(status=status, **base, **kw)

    cls = pred.anomaly_class if pred is not None else None
    score = float(pred.anomaly_score or 0.0) if pred is not None else 0.0
    conf = cls not in (None, WindAnomaly.NORMAL.value) and score >= t["anomaly_conf_min"]
    off_mw = (rated_kw or (max(pred.q50) if pred is not None else 0.0)) / 1000.0

    if age is not None and age > t["stale_minutes"]:
        return out(_status(aid, Status.EXCLUDED, "stale_telemetry",
                           f"No SCADA for {age:.0f} min (limit {t['stale_minutes']} min): offline window."),
                   threshold=float(t["stale_minutes"]), offline=True, mw_at_risk=off_mw)
    if data_gap or (cls == WindAnomaly.DATA_GAP.value and conf):
        return out(_status(aid, Status.EXCLUDED, "data_gap", "Data gap / suspect SCADA: one offline window, "
                           "no per-tick alerts."),
                   fault=WindAnomaly.DATA_GAP.value, p_fault=score if not data_gap else 1.0,
                   threshold=float(t["anomaly_conf_min"]), offline=True, mw_at_risk=off_mw)
    if pred is None:
        return out(_status(aid, Status.EXCLUDED, "no_prediction", "No model output this tick."), mw_at_risk=off_mw)

    q50 = np.asarray(pred.q50, dtype=float)
    dt_h = pred.step_minutes / 60.0
    exp_kwh = _expected_day_kwh(q50)
    base["expected_kwh_day"] = exp_kwh
    cap_mw = (rated_kw or (float(q50.max()) if q50.size else 0.0)) / 1000.0

    def loss_of(c: str) -> tuple[float, bool]:
        m = _measured_loss(q50, actual)
        return (float(min(1.0, max(0.0, m))), True) if m is not None else (WIND_DEFAULT_LOSS.get(c, 0.0), False)

    thr = float(t["anomaly_conf_min"])
    if conf and cls == WindAnomaly.ICING.value:  # safety first, even if the input looks unusual
        loss, measured = loss_of(cls)
        return out(_status(aid, Status.EXCLUDED, "icing",
                           f"Icing (score {score:.2f}): turbine excluded, safety alert, no crew sent in the storm."),
                   fault=cls, p_fault=score, threshold=thr, loss_frac=loss, lost_kwh_day=loss * exp_kwh,
                   mw_at_risk=cap_mw * loss, measured=measured,
                   alert=(Severity.SAFETY.value, "icing",
                          f"Icing risk on {aid}: keep clear (ice throw); no crew until conditions improve."))
    if pred.ood_score > t["ood_max"]:
        return out(_status(aid, Status.HUMAN_REVIEW, "ood",
                           f"Out of distribution (score {pred.ood_score:.2f} > {t['ood_max']}): human review."),
                   fault=cls if conf else None, p_fault=score, threshold=float(t["ood_max"]))
    if not conf:
        reason = "ok" if cls in (None, WindAnomaly.NORMAL.value) else f"weak:{cls}"
        detail = "Normal." if reason == "ok" else f"Weak {cls} signal (score {score:.2f} < {thr}): watching."
        return out(_status(aid, Status.ELIGIBLE, reason, detail), p_fault=score, threshold=thr)

    loss, measured = loss_of(cls)
    lost_kwh = loss * exp_kwh
    common = dict(fault=cls, p_fault=score, threshold=thr, loss_frac=loss, measured=measured)
    if cls == WindAnomaly.FAULT_STOP.value:
        return out(_status(aid, Status.EXCLUDED, "fault_stop",
                           f"Fault stop (score {score:.2f}): no power above cut-in while neighbours produce; "
                           f"{lost_kwh / 1000:.1f} MWh/day lost."),
                   crew_type=CrewType.WIND.value, lost_kwh_day=lost_kwh, mw_at_risk=cap_mw * max(loss, 0.0),
                   **common)
    if cls == WindAnomaly.UNDERPERFORMANCE.value:
        return out(_status(aid, Status.DERATED, "underperformance",
                           f"Underperformance / yaw (score {score:.2f}): {loss:.0%} below expected; inspection.",
                           headroom=1.0 - loss),
                   crew_type=CrewType.WIND.value, lost_kwh_day=lost_kwh, mw_at_risk=cap_mw * loss, **common)
    if cls == WindAnomaly.CURTAILMENT.value:
        curt = _curtailed_kwh(q50, actual, dt_h)
        txt = f"{curt:.0f} kWh capped so far" if curt is not None else "curtailed energy not measured"
        return out(_status(aid, Status.ELIGIBLE, "curtailed",
                           f"Curtailment (score {score:.2f}): output capped by the operator / grid, not a fault; "
                           f"{txt}. No work order."),
                   curtailed_kwh=curt, **common)
    if cls == WindAnomaly.ANEMOMETER_FAULT.value:
        return out(_status(aid, Status.ELIGIBLE, "anemometer_fault",
                           f"Anemometer fault (score {score:.2f}): nacelle wind disagrees with meteo wind while "
                           f"power matches meteo; remote check."),
                   crew_type=CrewType.REMOTE.value, **common)
    if cls == WindAnomaly.OVERHEATING.value:
        return out(_status(aid, Status.ELIGIBLE, "overheating_trend",
                           f"Component temperature residual rising (score {score:.2f}): watch, feeds M7."),
                   alert=(Severity.WARNING.value, "overheating_trend",
                          f"Temperature residual trending up on {aid}."), **common)
    return out(_status(aid, Status.ELIGIBLE, f"unknown:{cls}", f"Unhandled anomaly class {cls}."), **common)
