"""Contract Predictors for the wind baselines (M6 / M5 fallbacks). Window convention = #41's WindNBMPredictor:
`window.park_context` = {turbines, ts, telemetry, meteo, err?}, `window.weather` = park series; one park per call."""

from __future__ import annotations

import numpy as np

from wattsentry.contracts.envelope import ModelId, Prediction, Window, parse_utc
from wattsentry.model.wind.data import ACTIONABLE, CLASSES, POWER
from wattsentry.model.wind.predictor import _ts64

from .curve import BinnedCurve, park_inputs
from .detector import detect_park
from .windmon_rules import windmon_classes, windmon_flags


class BinnedCurvePredictor:
    """mode 'nowcast' (O&M, M6 fallback): expected power + curve-rules class; 'forecast' (Production, M5 fallback):
    met-curve P10/P50/P90 from the NWP hub wind in `meteo`, no telemetry needed, no anomaly class."""

    model = ModelId.BINNED_CURVE.value

    def __init__(self, curve: BinnedCurve, mode: str = "nowcast", params: dict | None = None, version: str = "r0"):
        self.curve, self.mode, self.params = curve, mode, params
        self.model_version = f"{self.model}-{mode}-{version}"

    def _classes(self, turbines, sensors, err, meteo, weather) -> np.ndarray | None:
        if self.mode == "forecast":
            return None
        return detect_park(self.curve, turbines, sensors, err, meteo, weather, self.params)

    def predict_park(self, turbines, ts, telemetry, meteo, t0: str, horizon: int, weather=None, err=None
                     ) -> dict[str, Prediction]:
        ts = _ts64(ts)
        T, n = len(ts), len(turbines)
        sensors = {k: np.asarray(v, np.float32) for k, v in (telemetry or {}).items()}
        sensors.setdefault(POWER, np.full((T, n), np.nan, np.float32))
        meteo = {k: np.asarray(v, np.float64) for k, v in meteo.items()}
        weather = {k: np.asarray(v, np.float64) for k, v in (weather or {}).items()}
        err = np.zeros((T, n), np.int32) if err is None else np.asarray(err, np.int32)
        i0 = int(np.searchsorted(ts, np.datetime64(parse_utc(t0).replace(tzinfo=None), "m")))
        if i0 + horizon > T:
            raise ValueError("arrays do not cover [t0, t0 + horizon)")
        sl = slice(i0, i0 + horizon)
        q = np.nan_to_num(self.curve.predict_block(turbines, park_inputs(sensors, meteo, weather, T, n),
                                                   "met" if self.mode == "forecast" else "nowcast")[sl])
        cls = self._classes(turbines, sensors, err, meteo, weather)
        out = {}
        for j, t in enumerate(turbines):
            r = float(t["rated_kw"])
            q10, q50, q90 = (q[:, j] * r).T
            extra, ac, score = {"mode": self.mode}, None, None
            if cls is not None:
                c = cls[sl, j]
                counts = np.bincount(c, minlength=len(CLASSES))
                ac = CLASSES[int(counts[1:].argmax()) + 1] if counts[1:].any() else "normal"
                score = float((c != 0).mean())
                pw = sensors[POWER][sl, j].astype(np.float64)
                ok = np.isfinite(pw)
                extra |= {"class_steps": {CLASSES[k]: int(counts[k]) for k in np.flatnonzero(counts)},
                          "actionable": ac in ACTIONABLE,
                          "lost_kwh": round(float(np.clip(q50[ok] - pw[ok], 0, None).sum() * 0.25), 2)}
            out[t["turbine_id"]] = Prediction(asset_id=t["turbine_id"], model=self.model,
                                              model_version=self.model_version, t0=t0, horizon=horizon, q10=q10,
                                              q50=q50, q90=q90, anomaly_class=ac, anomaly_score=score, extra=extra)
        return out

    def predict(self, asset, window: Window) -> Prediction:
        pc = window.park_context or {}
        return self.predict_park(pc["turbines"], pc["ts"], pc.get("telemetry"), pc["meteo"], window.t0,
                                 window.horizon, weather=window.weather, err=pc.get("err"))[str(asset["turbine_id"])]


class WindMonRulesPredictor(BinnedCurvePredictor):
    """WindMon's four rules (classes) with the binned curve's expected power in q10/q50/q90."""

    model = ModelId.WINDMON_RULES.value

    def __init__(self, curve: BinnedCurve, version: str = "r0"):
        super().__init__(curve, "nowcast", None, version)
        self.model_version = f"{self.model}-{version}"

    def _classes(self, turbines, sensors, err, meteo, weather):
        rated = np.array([float(t["rated_kw"]) for t in turbines])
        return windmon_classes(windmon_flags(sensors, err, rated))
