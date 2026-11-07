"""Deterministic MockPredictor: plausible PV, load, site-health, wind and zone-demand envelopes.

Same (asset, t0) -> same output, on any machine (no RNG state, sha256-based noise). It stands in for the real
model, so `model` is the real id (pvnet, loadnet, ...) and `model_version` is "mock-1". Spec: model.md §Mock.
"""

from __future__ import annotations

import hashlib
import math
from datetime import timedelta
from typing import Any, Mapping, Optional

from .envelope import DAY_STEPS, STEP_MINUTES, ModelId, Prediction, PvFault, WindAnomaly, Window, parse_utc

MOCK_VERSION = "mock-1"
KINDS = ("pv", "load", "sitehealth", "wind", "zone")
_MODEL_FOR_KIND = {
    "pv": ModelId.PVNET.value,
    "load": ModelId.LOADNET.value,
    "sitehealth": ModelId.SITEHEALTH.value,
    "wind": ModelId.WINDNBM.value,
    "zone": ModelId.DEMANDNET.value,
}


def _u(*parts: Any) -> float:
    """Deterministic uniform [0, 1) from any key."""
    h = hashlib.sha256("|".join(map(str, parts)).encode()).digest()
    return int.from_bytes(h[:8], "big") / 2**64


def _asset_id(asset: Mapping[str, Any]) -> str:
    for k in ("site_id", "turbine_id", "zone_id", "feeder_id", "asset_id"):
        if asset.get(k):
            return str(asset[k])
    raise ValueError("asset has no id field (site_id / turbine_id / zone_id / feeder_id / asset_id)")


def _clear_sky_fraction(ts, lat: float) -> float:
    """0..1 shape of a clear-sky PV day: solar elevation from day-of-year and UTC hour (RO ≈ UTC+1.7 solar)."""
    doy = ts.timetuple().tm_yday
    decl = 23.44 * math.sin(math.radians(360 / 365 * (doy - 81)))
    solar_hour = ts.hour + ts.minute / 60 + 1.7  # ~25 °E
    ha = 15 * (solar_hour - 12)
    lat_r, decl_r = math.radians(lat), math.radians(decl)
    sin_el = math.sin(lat_r) * math.sin(decl_r) + math.cos(lat_r) * math.cos(decl_r) * math.cos(math.radians(ha))
    return max(0.0, sin_el) ** 1.2


def _bucharest_hour(ts) -> float:
    """Local hour, Europe/Bucharest approximated as UTC+2 (UTC+3 Apr-Oct)."""
    off = 3 if 4 <= ts.month <= 10 else 2
    return (ts.hour + ts.minute / 60 + off) % 24


class MockPredictor:
    """`MockPredictor(kind).predict(asset, window) -> Prediction`; kind in pv | load | sitehealth | wind | zone."""

    def __init__(self, kind: str, fault_rate: float = 0.15, anomaly_rate: float = 0.12):
        if kind not in KINDS:
            raise ValueError(f"kind must be one of {KINDS}")
        self.kind = kind
        self.model = _MODEL_FOR_KIND[kind]
        self.model_version = MOCK_VERSION
        self.fault_rate = fault_rate
        self.anomaly_rate = anomaly_rate

    def predict(self, asset: Mapping[str, Any], window: Optional[Window] = None, *, t0: Optional[str] = None,
                horizon: Optional[int] = None) -> Prediction:
        if window is None:
            if t0 is None:
                raise ValueError("pass a Window or t0")
            window = Window(t0=t0, horizon=horizon or DAY_STEPS)
        aid = _asset_id(asset)
        start = parse_utc(window.t0)
        times = [start + timedelta(minutes=STEP_MINUTES * k) for k in range(window.horizon)]
        fn = getattr(self, f"_{self.kind}")
        q10, q50, q90, extra = fn(asset, aid, times)
        return Prediction(
            asset_id=aid,
            model=self.model,
            model_version=self.model_version,
            t0=window.t0,
            horizon=window.horizon,
            q10=q10,
            q50=q50,
            q90=q90,
            ood_score=round(_u(aid, window.t0, "ood") ** 4, 4),  # mostly near 0, a few high
            member_std=extra.pop("member_std"),
            unit=extra.pop("unit", "kW"),
            **extra,
        )

    # --- kinds ------------------------------------------------------------------------------------------
    def _pv(self, asset, aid, times):
        kwp = float(asset.get("kwp_true") or asset.get("kwp_recorded") or 5.0)
        lat = float(asset.get("lat") or 45.0)
        day = times[0].date().isoformat()
        cloud = 0.35 + 0.65 * _u(aid, day, "cloud")  # 0.35 overcast .. 1 clear
        q50 = [0.82 * kwp * cloud * _clear_sky_fraction(t, lat) for t in times]
        spread = 0.08 + 0.25 * (1 - cloud)  # cloudier days are less certain
        q10 = [v * (1 - 1.6 * spread) if v > 0 else 0.0 for v in q50]
        q90 = [min(kwp, v * (1 + spread)) if v > 0 else 0.0 for v in q50]
        q10 = [max(0.0, a) for a in q10]
        q90 = [max(b, c) for b, c in zip(q50, q90)]
        return q10, q50, q90, {"member_std": round(spread * max(q50 or [0]) / 2, 4)}

    def _load(self, asset, aid, times):
        base = 0.25 + 0.3 * _u(aid, "base")
        heat = asset.get("archetype") in ("heat_pump", "electric_heating")
        q50 = []
        for t in times:
            h = _bucharest_hour(t)
            morning = 0.6 * math.exp(-((h - 7.5) ** 2) / 1.5)
            evening = 1.4 * math.exp(-((h - 19.5) ** 2) / 3.0)
            winter = 0.8 if heat and t.month in (11, 12, 1, 2, 3) else 0.0
            summer_ac = 0.5 * math.exp(-((h - 15) ** 2) / 8) if t.month in (6, 7, 8) else 0.0
            q50.append(base + morning + evening + winter + summer_ac)
        q10 = [0.65 * v for v in q50]
        q90 = [1.45 * v for v in q50]
        return q10, q50, q90, {"member_std": round(0.12 * sum(q50) / len(q50), 4)}

    def _sitehealth(self, asset, aid, times):
        q10, q50, q90, extra = self._pv(asset, aid, times)
        classes = [f.value for f in PvFault]
        faulty = _u(aid, "faulty") < self.fault_rate
        if faulty:
            fault = classes[1 + int(_u(aid, "which") * (len(classes) - 1))]
            p_top = 0.55 + 0.4 * _u(aid, "conf")
        else:
            fault, p_top = PvFault.HEALTHY.value, 0.9 + 0.09 * _u(aid, "conf")
        rest = (1 - p_top) / (len(classes) - 1)
        probs = {c: (p_top if c == fault else rest) for c in classes}
        extra["fault_probs"] = probs
        return q10, q50, q90, extra

    def _wind(self, asset, aid, times):
        rated = float(asset.get("rated_kw") or 2500.0)
        park = str(asset.get("park_id") or aid.split(".")[0])
        day = times[0].date().isoformat()
        mean_v = 4 + 8 * _u(park, day, "wind")  # park-level weather, shared by its turbines
        q50 = []
        for k, t in enumerate(times):
            v = mean_v * (1 + 0.3 * math.sin(2 * math.pi * k / len(times) + 6 * _u(park, "phase")))
            q50.append(rated * self._power_curve(v))
        q10 = [0.7 * p for p in q50]
        q90 = [min(rated, 1.2 * p) for p in q50]
        anomalous = _u(aid, day, "anom") < self.anomaly_rate
        classes = [a.value for a in WindAnomaly]
        cls = classes[1 + int(_u(aid, day, "cls") * (len(classes) - 1))] if anomalous else WindAnomaly.NORMAL.value
        score = 0.6 + 0.4 * _u(aid, day, "score") if anomalous else 0.2 * _u(aid, day, "score")
        return q10, q50, q90, {
            "member_std": round(0.08 * rated, 2),
            "anomaly_class": cls,
            "anomaly_score": round(score, 4),
        }

    @staticmethod
    def _power_curve(v_ms: float, cut_in: float = 3.0, rated_v: float = 12.0, cut_out: float = 25.0) -> float:
        if v_ms < cut_in or v_ms >= cut_out:
            return 0.0
        if v_ms >= rated_v:
            return 1.0
        return ((v_ms - cut_in) / (rated_v - cut_in)) ** 3

    def _zone(self, asset, aid, times):
        n_homes = float(asset.get("n_homes") or 1000)
        q10, q50, q90, _ = self._load({"archetype": asset.get("archetype")}, aid, times)
        to_mw = n_homes / 1000.0
        # Aggregates are tighter than single homes (scenario sums, not sums of quantiles).
        q50 = [v * to_mw for v in q50]
        q10 = [v * 0.88 for v in q50]
        q90 = [v * 1.15 for v in q50]
        return q10, q50, q90, {"member_std": round(0.04 * max(q50), 4), "unit": "MW"}


def mock_predictors() -> dict[str, MockPredictor]:
    """One mock per kind, keyed by kind."""
    return {k: MockPredictor(k) for k in KINDS}
