"""M7 predictor (no torch): per-turbine component-failure risk for the replay and the O&M turbine drawer.

Artifacts (from `sentinel.model.windfailure.train`, one directory): windfailure_7d.txt, windfailure_14d.txt (LightGBM),
windfailure_thresholds.json (thresholds, residual model, `shipped` flag), and windfailure_nbm.onnx when the run used
`--residuals nbm`.

Input = M6's park window (sentinel.model.wind.predictor): `window.park_context` = {
    "turbines": [turbines rows of the whole park], "ts": datetime64[T] or ISO strings (15-min grid, from 00:00 UTC),
    "telemetry": {sensor: [T, n]} (act_pwt in kW, temps °C, NaN = gap), "err": [T, n] (optional),
    "meteo": {"wind_ms": [T, n], "wdir_deg": [T, n]} and "pressure_hpa": [T] (only for `nbm` residuals) }
with >= 61 days of history (the per-turbine baseline is days d−60..d−21). The risk is for the last COMPLETE day
ending at or before `window.t0`.

Output per turbine (`risk_park`, and `Prediction.extra` from `predict`): {
    "risk": max_c P(failure within 14 d), "risk_component": c, "risk_level": "high" | "medium" | "low",
    "alarm": risk >= frozen threshold, "p_fail_7d": {c: p}, "p_fail_14d": {c: p}, "day": "YYYY-MM-DD",
    "shipped": bool (R12: show the score only when true; otherwise the drawer keeps the residual trend only),
    "residual_dev_c": {c: today's mean residual − own baseline, °C} (the trend the drawer already shows) }
The envelope's `fault_probs` is PvFault-only (contract), so M7 puts everything in `extra`; q10 = q50 = q90 = [risk]
with unit "p_fail_14d", horizon 1, step 1440 min.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

import numpy as np

from sentinel.contracts.envelope import ModelId, Prediction, Window, parse_utc

from . import COMPONENTS
from .features import FEATURES, daily_features, feature_index


def _ts64(ts) -> np.ndarray:
    a = np.asarray(ts)
    if a.dtype.kind == "M":
        return a.astype("datetime64[m]")
    return np.array([np.datetime64(str(t).replace("Z", "").replace("+00:00", ""), "m") for t in a])


class WindFailurePredictor:
    model = ModelId.WINDFAILURE.value

    def __init__(self, model_dir: str | Path):
        import lightgbm as lgb

        d = Path(model_dir)
        self.cal = json.loads((d / "windfailure_thresholds.json").read_text())
        self.model_version = self.cal["model_version"]
        self.shipped = bool(self.cal.get("shipped", False))
        self.boosters = {h: lgb.Booster(model_file=str(d / f"windfailure_{h}d.txt")) for h in (7, 14)}
        self.thr = float(self.cal["thresholds"]["p_fail_14d"])
        self.residuals = self.cal.get("residuals", "physics")
        self.sess = None
        if self.residuals == "nbm":
            import onnxruntime as ort

            self.sess = ort.InferenceSession(str(d / self.cal["residual_model"]["onnx"]),
                                             providers=["CPUExecutionProvider"])

    # ----------------------------------------------------------------------------------------------------------
    def _residuals(self, turbines, ts, tel, err, meteo, pressure):
        rated = np.array([float(t["rated_kw"]) for t in turbines])
        is_large = (rated >= 4000).astype(np.int64)
        if self.residuals == "physics":
            from .thermal import residuals

            return residuals(self.cal["residual_model"], tel, err, rated, is_large)
        from sentinel.model.wind.data import TEMP_SENSORS
        from sentinel.model.wind.features import T_RISE_SCALE, nbm_inputs, run_nbm

        tindex = self.cal["residual_model"]["turbine_index"]
        tid = np.array([tindex.get(t["turbine_id"], 0) for t in turbines], dtype=np.int64)
        T, n = err.shape
        pres = (np.full((T, n), 1013.25) if pressure is None
                else np.repeat(np.asarray(pressure, np.float64)[:T, None], n, axis=1))
        X, _, _ = nbm_inputs(tel, meteo, pres, ts, rated, is_large.astype(np.float32), labels=None)
        _, _, tq = run_nbm(lambda xb, tb: tuple(self.sess.run(None, {"x": xb, "tid": tb})), X, tid)
        amb = np.nan_to_num(tel["turbine_t"], nan=10.0)
        r = np.full((T, n, len(COMPONENTS)), np.nan, np.float32)
        for ci, c in enumerate(COMPONENTS):
            y = tel[c]
            r[..., ci] = np.where(np.isfinite(y) & (err == 0), y - (tq[:, :, TEMP_SENSORS.index(c), 1] * T_RISE_SCALE
                                                                    + amb), np.nan)
        return r

    def daily(self, turbines, ts, telemetry, err=None, meteo=None, pressure_hpa=None) -> dict:
        """All complete days: {"days": datetime64[D], "p7": [D, n, C], "p14": [D, n, C], "dev": [D, n, C]}."""
        ts = _ts64(ts)
        if (ts[0] - ts[0].astype("datetime64[D]")) != np.timedelta64(0, "m"):
            raise ValueError("ts must start at 00:00 UTC (whole days)")
        D = len(ts) // 96
        T = D * 96
        tel = {k: np.asarray(v, np.float64)[:T] for k, v in telemetry.items()}
        n = len(turbines)
        err = np.zeros((T, n), np.int32) if err is None else np.asarray(err)[:T]
        meteo = {k: np.asarray(v)[:T] for k, v in (meteo or {}).items()}
        rated = np.array([float(t["rated_kw"]) for t in turbines])
        park = np.array([str(t.get("park_id", "P")) for t in turbines])
        park_of = np.unique(park, return_inverse=True)[1]
        r = self._residuals(turbines, ts[:T], tel, err, meteo, pressure_hpa)
        X = daily_features(r, tel["act_pwt"], tel["turbine_t"], rated, park_of, (rated >= 4000).astype(np.int64))
        Xf = X.reshape(-1, len(FEATURES))
        out = {"days": ts[:T:96], "dev": X[..., feature_index("dev")]}
        for h, b in self.boosters.items():
            out[f"p{h}"] = b.predict(Xf).reshape(D, n, len(COMPONENTS))
        return out

    def risk_park(self, turbines, ts, telemetry, err=None, meteo=None, pressure_hpa=None, t0=None) -> dict:
        """{turbine_id: risk dict} for the last complete day ending at or before t0 (default: the last day)."""
        d = self.daily(turbines, ts, telemetry, err, meteo, pressure_hpa)
        ends = d["days"] + np.timedelta64(1440, "m")
        k = len(ends) - 1
        if t0 is not None:
            t = np.datetime64(parse_utc(t0).replace(tzinfo=None), "m") if isinstance(t0, str) else _ts64([t0])[0]
            ok = np.flatnonzero(ends <= t)
            if ok.size == 0:
                raise ValueError("no complete day before t0")
            k = int(ok[-1])
        res = {}
        for j, t in enumerate(turbines):
            p14 = {c: float(d["p14"][k, j, i]) for i, c in enumerate(COMPONENTS)}
            p7 = {c: float(d["p7"][k, j, i]) for i, c in enumerate(COMPONENTS)}
            top = max(p14, key=p14.get)
            risk = p14[top]
            res[t["turbine_id"]] = {
                "risk": risk, "risk_component": top, "alarm": bool(risk >= self.thr),
                "risk_level": "high" if risk >= self.thr else ("medium" if risk >= self.thr / 2 else "low"),
                "p_fail_7d": p7, "p_fail_14d": p14, "day": str(d["days"][k].astype("datetime64[D]")),
                "shipped": self.shipped,
                "residual_dev_c": {c: (None if not np.isfinite(d["dev"][k, j, i]) else float(d["dev"][k, j, i]))
                                   for i, c in enumerate(COMPONENTS)},
            }
        return res

    def predict(self, asset: Mapping, window: Window) -> Prediction:
        pc = window.park_context
        w = window.weather or {}
        risks = self.risk_park(pc["turbines"], pc["ts"], pc["telemetry"], pc.get("err"), pc.get("meteo"),
                               pc.get("pressure_hpa", w.get("pressure_hpa") if isinstance(w, Mapping) else None),
                               t0=window.t0)
        r = risks[asset["turbine_id"]]
        return Prediction(asset_id=asset["turbine_id"], model=self.model, model_version=self.model_version,
                          t0=window.t0, horizon=1, q10=[r["risk"]], q50=[r["risk"]], q90=[r["risk"]],
                          unit="p_fail_14d", step_minutes=1440, extra=r)
