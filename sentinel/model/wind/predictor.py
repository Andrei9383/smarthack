"""onnxruntime-only M6 predictor (no torch): returns the contract `Prediction` envelope for replay (#17).

Artifacts (from `sentinel.model.wind.train`, in one directory): windnbm.onnx, windnbm_clf.txt (LightGBM),
windnbm_thresholds.json (calibration + frozen class thresholds), manifest.json.

Window for M6 (envelope.Window; arrays numpy-compatible):
- `window.park_context` = {
      "turbines": [turbines rows of the whole park],          # park context needs every turbine of the park
      "ts": datetime64[T] (UTC) or ISO strings, 15-min grid,
      "telemetry": {sensor: [T, n]}  (contract sensors; NaN = gap; act_pwt already in kW),
      "meteo": {"wind_ms": [T, n], "wdir_deg": [T, n] (optional)},
      "err": [T, n] int (optional, integral error codes seen), }
- `window.weather` = optional park weather series {"rh_pct": [T], "precip_mm": [T], "pressure_hpa": [T]}.
- `window.t0` / `window.horizon`: the evaluated steps; arrays must cover them plus history before t0
  (>= 28 days recommended: the per-turbine temperature baseline is 21 days ending 7 days ago).
`predict_park(...)` returns all turbines of the park at once (what replay should call once per park per tick).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Optional

import numpy as np

from sentinel.contracts.envelope import ModelId, Prediction, Window, parse_utc

from .data import ACTIONABLE, AMBIENT, CLASSES, POWER, TEMP_SENSORS, WindData
from .features import clf_features, curtail_ok, decide, nbm_inputs, neighbours_for, ood_steps, run_nbm

K = len(CLASSES)


def _ts64(ts) -> np.ndarray:
    a = np.asarray(ts)
    if a.dtype.kind == "M":
        return a.astype("datetime64[m]")
    return np.array([np.datetime64(str(t).replace("Z", "").replace("+00:00", ""), "m") for t in a])


class WindNBMPredictor:
    model = ModelId.WINDNBM.value

    def __init__(self, model_dir: str | Path):
        import lightgbm as lgb
        import onnxruntime as ort

        d = Path(model_dir)
        self.cal = json.loads((d / "windnbm_thresholds.json").read_text())
        self.model_version = self.cal["model_version"]
        self.sess = ort.InferenceSession(str(d / "windnbm.onnx"), providers=["CPUExecutionProvider"])
        self.booster = lgb.Booster(model_file=str(d / "windnbm_clf.txt"))
        self.tindex = self.cal["turbine_index"]
        self.offsets = self.cal["quantile_offsets"]["all"]

    # ----------------------------------------------------------------------------------------------------------
    def _onnx(self, xb, tb):
        pq, ps, tq = self.sess.run(None, {"x": xb, "tid": tb})
        return pq, ps, tq

    def score_park(self, turbines, ts, telemetry, meteo, weather=None, err=None) -> dict[str, np.ndarray]:
        """Per-step outputs for every turbine of one park over the whole provided span."""
        ts = _ts64(ts)
        T, n = len(ts), len(turbines)
        sens = {k: np.asarray(v, dtype=np.float32) for k, v in telemetry.items()}
        for s in [POWER, "wd_spd", AMBIENT] + TEMP_SENSORS:
            sens.setdefault(s, np.full((T, n), np.nan, np.float32))
        met = {"wind_ms": np.asarray(meteo["wind_ms"], dtype=np.float32),
               "wdir_deg": np.asarray(meteo.get("wdir_deg", np.full((T, n), np.nan)), dtype=np.float32)}
        err = np.zeros((T, n), np.int32) if err is None else np.asarray(err, dtype=np.int32)
        weather = {k: np.asarray(v, dtype=np.float32) for k, v in (weather or {}).items()}
        rated = np.array([float(t["rated_kw"]) for t in turbines])
        is_large = (rated >= 4000).astype(np.float32)
        pres = np.broadcast_to(weather.get("pressure_hpa", np.full(T, np.nan))[:, None], (T, n))
        X, _, _ = nbm_inputs(sens, met, pres, ts, rated, is_large, labels=None)
        tidx = np.array([int(self.tindex.get(t["turbine_id"], 0)) for t in turbines], dtype=np.int64)
        pq, pstd, tqr = run_nbm(self._onnx, X, tidx, chunk=self.cal["chunk"], warm=self.cal["warm"])
        tq = tqr * self.cal["t_rise_scale"] + np.nan_to_num(sens[AMBIENT], nan=10.0)[..., None, None]
        ood = ood_steps(X, self.cal["input_ranges"]).T
        lat = np.array([float(t["lat"]) for t in turbines])
        lon = np.array([float(t["lon"]) for t in turbines])
        F, names = clf_features(sens, err, met, weather, rated, is_large, neighbours_for(lat, lon), pq, pstd, tq,
                                self.cal["met_curve"])
        if names != self.cal["feature_names"]:
            raise RuntimeError("feature list differs from the one the classifier was trained on")
        probs = self.booster.predict(F.reshape(-1, F.shape[-1])).reshape(T, n, K).astype(np.float32)
        cls = decide(probs, self.cal["class_thresholds"], self.cal["min_run"], self.cal["merge"],
                     curtail_mask=curtail_ok(F, names))
        return {"ts": ts, "rated": rated, "pq": pq, "pstd": pstd, "tq": tq, "ood": ood, "probs": probs, "cls": cls,
                "power": sens[POWER]}

    def predict_park(self, turbines, ts, telemetry, meteo, t0: str, horizon: int, weather=None, err=None
                     ) -> dict[str, Prediction]:
        s = self.score_park(turbines, ts, telemetry, meteo, weather, err)
        i0 = int(np.searchsorted(s["ts"], np.datetime64(parse_utc(t0).replace(tzinfo=None), "m")))
        if i0 + horizon > len(s["ts"]):
            raise ValueError("arrays do not cover [t0, t0 + horizon)")
        sl = slice(i0, i0 + horizon)
        out = {}
        for j, t in enumerate(turbines):
            r = s["rated"][j]
            q = s["pq"][sl, j].astype(np.float64)
            q10 = (q[:, 0] + self.offsets["q10"]) * r
            q50 = q[:, 1] * r
            q90 = (q[:, 2] + self.offsets["q90"]) * r
            q10 = np.minimum(q10, q50)
            q90 = np.maximum(q90, q50)
            cls = s["cls"][sl, j]
            nz = cls[cls != 0]
            ac = CLASSES[int(np.bincount(nz, minlength=K).argmax())] if len(nz) else "normal"
            pw = s["power"][sl, j].astype(np.float64)
            ok = np.isfinite(pw)
            lost = float(np.clip(q50[ok] - pw[ok], 0, None).sum() * 0.25)
            counts = np.bincount(cls, minlength=K)
            out[t["turbine_id"]] = Prediction(
                asset_id=t["turbine_id"],
                model=self.model,
                model_version=self.model_version,
                t0=t0,
                horizon=horizon,
                q10=q10.tolist(),
                q50=q50.tolist(),
                q90=q90.tolist(),
                ood_score=float(s["ood"][sl, j].mean()),
                member_std=float(s["pstd"][sl, j].mean() * r),
                anomaly_class=ac,
                anomaly_score=float(np.clip(1.0 - s["probs"][sl, j, 0], 0, 1).max()),
                extra={
                    "class_steps": {CLASSES[c]: int(counts[c]) for c in range(K) if counts[c]},
                    "actionable": ac in ACTIONABLE,
                    "lost_kwh": round(lost, 2),
                    "temps_q50_last": {k: float(s["tq"][i0 + horizon - 1, j, i, 1]) for i, k in enumerate(TEMP_SENSORS)},
                    "seen_turbine": t["turbine_id"] in self.tindex,
                    "history_steps": i0,
                },
            )
        return out

    def predict(self, asset: Mapping[str, Any], window: Window) -> Prediction:
        pc = window.park_context or {}
        res = self.predict_park(pc["turbines"], pc["ts"], pc["telemetry"], pc["meteo"], window.t0, window.horizon,
                                weather=window.weather, err=pc.get("err"))
        return res[str(asset["turbine_id"])]


def window_from_data(data: WindData, park_id: str, t0: str, horizon: int = 96, history_days: int = 28) -> Window:
    """Build an M6 Window for one park from a WindData (replay / tests)."""
    cols = [j for j, t in enumerate(data.turbines) if t["park_id"] == park_id]
    i0 = int(np.searchsorted(data.ts, np.datetime64(parse_utc(t0).replace(tzinfo=None), "m")))
    lo, hi = max(0, i0 - history_days * 96), i0 + horizon
    p = data.park_ids.index(park_id)
    return Window(
        t0=t0,
        horizon=horizon,
        weather={k: v[lo:hi, p] for k, v in data.weather.items()},
        park_context={
            "turbines": [data.turbines[j] for j in cols],
            "ts": data.ts[lo:hi],
            "telemetry": {k: v[lo:hi][:, cols] for k, v in data.sensors.items()},
            "meteo": {k: v[lo:hi][:, cols] for k, v in data.meteo.items()},
            "err": data.err[lo:hi][:, cols],
        },
    )
