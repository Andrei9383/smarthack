"""onnxruntime-only predictors (no torch) that return the contract Prediction envelope.

    from wattsentry.model.predictor import load_predictor
    p = load_predictor("pvnet", "models/")          # every pvnet*.onnx in the dir = one ensemble member
    pred = p.predict(asset, window)                  # -> wattsentry.contracts.Prediction

Files per member: `<name>.onnx` + `<name>.calibration.json` (written by train.py), name = model or model.<tag>.
Several members (e.g. seeds) -> q10/q50/q90 = member mean, `member_std` = mean over steps of the std of q50 across
members, `extra["members"]` = per-member quantiles (and fault_probs for SiteHealth), `extra["p_fault_std"]`.
`ood_score` = robust z of the Mahalanobis distance of the model embedding vs the TRAIN set
((D - median_train) / (1.4826 MAD_train), clipped at 0; threshold 3.0), averaged over members.

Window fields used (wattsentry.contracts.Window):
  PV (pvnet): weather [T,C] (channels in window.meta["weather_channels"], default physics.WEATHER_CHANNELS;
    T = 96 at 15 min or 24 hourly), solar_geometry [T,3] (computed from t0 + lat/lon when None), reference = the
    site's past pv_ac_w, 7-14 days (W unless window.meta["power_unit"] == "kW").
  Load (loadnet): weather (temp, ghi), history = past load_w 7-14 days.
  SiteHealth: the PV + load inputs above, plus window.meta["telemetry"] = {pv_ac_w, inv_voltage_v, inv_status,
    load_w} over the assessed 24 h, battery [T,2] (soc, batt_w), optional window.meta["telemetry_history"] =
    {inv_voltage_v, inv_status, soc, batt_w} over the 14 days before (missing -> neutral summary values).
Asset fields: site_id, lat, lon, kwp_recorded, tilt_recorded, azimuth_recorded, inverter_kw, battery_kwh,
battery_kw, archetype (falls back to *_true / defaults when absent).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from wattsentry.contracts import DAY_STEPS, Prediction, PvFault, Window

from . import features as F
from . import physics as ph

H = DAY_STEPS
FAULTS = [f.value for f in PvFault]


# --- helpers -----------------------------------------------------------------------------------------------
def _f(asset, *keys, default=0.0):
    for k in keys:
        v = asset.get(k)
        if v is not None:
            return float(v)
    return float(default)


def _asset_id(asset):
    for k in ("site_id", "asset_id", "zone_id", "feeder_id"):
        if asset.get(k):
            return str(asset[k])
    raise ValueError("asset has no site_id / asset_id")


def _to_steps(x, n=H):
    """[T, ...] at 15 min or hourly -> [n, ...] at 15 min."""
    x = np.asarray(x, np.float64)
    if x.shape[0] == n:
        return x
    if x.shape[0] * 4 == n:
        return np.repeat(x, 4, axis=0)
    raise ValueError(f"series has {x.shape[0]} steps, expected {n} (15 min) or {n // 4} (hourly)")


def _hist(x, unit_scale, n_days=F.HIST_DAYS):
    """Past series (7-14 days, 15 min) -> [1, 14*96] kW, earliest week repeated to fill 14 days."""
    if x is None:
        raise ValueError("window needs 7-14 days of history (reference / history)")
    x = np.nan_to_num(np.asarray(x, np.float64).ravel() * unit_scale)
    d = len(x) // H
    if d < 1:
        raise ValueError("history shorter than one day")
    x = x[-d * H:]
    while len(x) < n_days * H:
        x = np.concatenate([x[: min(len(x), 7 * H)], x])
    return x[-n_days * H:][None].astype(np.float32)


def _weather(window: Window, n=H):
    if window.weather is None:
        raise ValueError("window.weather is required")
    w = _to_steps(window.weather, n)
    names = list(window.meta.get("weather_channels", ph.WEATHER_CHANNELS))
    w = np.nan_to_num(w)
    out = np.zeros((n, len(ph.WEATHER_CHANNELS)))
    for i, c in enumerate(ph.WEATHER_CHANNELS):
        if c in names:
            out[:, i] = w[:, names.index(c)]
    return out


def _fill_cloud(wx: np.ndarray, geom: np.ndarray) -> np.ndarray:
    """Cloud channel derived exactly like the training windows do, from the daily clear-sky index:
    cloud = clip(100 (1.05 - k) / 0.95, 0, 100). Applied even when a cloud column is given (eval.py takes ERA5
    cloud_pct, which is inconsistent with the synthetic GHI), so serving inputs match training inputs."""
    ci = ph.WEATHER_CHANNELS.index("cloud")
    k = wx[:, 0].sum() / max(geom[:, 2].sum(), 1e-6)
    wx = wx.copy()
    wx[:, ci] = np.clip(100 * (1.05 - k) / 0.95, 0, 100)
    return wx


def _status(code, pv_kw, inverter_kw):
    """Raw inverter codes (0 ok, 37 overvoltage) -> the training encoding: 0 off, 1 producing, 2 limiting,
    3 overvoltage trip. Already-encoded inputs (values in 0..3) pass through."""
    code = np.nan_to_num(np.asarray(code, np.float64))
    if code.size and np.isin(code, (0, 1, 2, 3)).all() and (code > 0).any():
        return code
    pv = np.nan_to_num(np.asarray(pv_kw, np.float64))
    st = np.where(pv > 0.005, 1.0, 0.0)
    st[(pv >= 0.995 * inverter_kw) & (pv > 0.1)] = 2.0
    st[code == 37] = 3.0
    return st


def _site_meta(asset):
    kwp = _f(asset, "kwp_recorded", "kwp", "kwp_true", default=5.0)
    return {
        "kwp_recorded": kwp,
        "tilt_recorded": _f(asset, "tilt_recorded", "tilt_true", default=30.0),
        "azimuth_recorded": _f(asset, "azimuth_recorded", "azimuth_true", default=180.0),
        "inverter_kw": _f(asset, "inverter_kw", "inverter_ac_kw", default=kwp),
        "lat": _f(asset, "lat", default=45.0), "lon": _f(asset, "lon", default=25.0),
        "battery_kwh": _f(asset, "battery_kwh", default=0.0), "battery_kw": _f(asset, "battery_kw", default=0.0),
    }


def ood_robust_z(ood: dict, emb: np.ndarray) -> np.ndarray:
    xc = np.atleast_2d(emb).astype(np.float64) - np.asarray(ood["mean"])
    d = np.sqrt(np.maximum(np.einsum("ij,jk,ik->i", xc, np.asarray(ood["precision"]), xc), 0))
    return np.maximum(0.0, (d - ood["d_median"]) / ood["d_scale"])


class Member:
    def __init__(self, onnx_path: Path):
        import onnxruntime as ort

        self.path = Path(onnx_path)
        cal = self.path.with_name(self.path.name[: -len(".onnx")] + ".calibration.json")
        self.calib = json.loads(cal.read_text()) if cal.exists() else {}
        self.sess = ort.InferenceSession(str(self.path), providers=["CPUExecutionProvider"])
        self.inputs = [i.name for i in self.sess.get_inputs()]

    def run(self, feeds: dict) -> list[np.ndarray]:
        return self.sess.run(None, {k: np.asarray(feeds[k], np.float32) for k in self.inputs})


def find_members(model: str, model_dir) -> list[Path]:
    d = Path(model_dir)
    paths = sorted(p for p in d.glob(f"{model}*.onnx") if p.name == f"{model}.onnx" or p.name.startswith(model + "."))
    if not paths:
        raise FileNotFoundError(f"no {model}*.onnx in {d}")
    return paths


# --- forecasters (M1 / M2) ---------------------------------------------------------------------------------
class ForecastPredictor:
    kind = "pv"

    def __init__(self, model: str, members: list[Path] | list[Member]):
        self.model = model
        self.members = [m if isinstance(m, Member) else Member(m) for m in members]
        v0 = self.members[0].calib.get("model_version", f"{model}-onnx")
        self.model_version = v0 if len(self.members) == 1 else f"{v0}x{len(self.members)}"

    def raw(self, asset: Mapping[str, Any], window: Window) -> dict:
        meta = _site_meta(asset)
        times = ph.times_utc(window.t0, H)
        wx = _weather(window)
        geom = (_to_steps(window.solar_geometry) if window.solar_geometry is not None
                else ph.solar_geometry(times, meta["lat"], meta["lon"]))
        wx = _fill_cloud(wx, geom)
        cal = ph.calendar_features(times)
        unit = 1.0 if window.meta.get("power_unit", "W") == "kW" else 1e-3
        raw = {k: np.array([v], np.float32) for k, v in meta.items()}
        raw["has_batt"] = np.array([float(meta["battery_kwh"] > 0)], np.float32)
        raw.update(wx=wx[None].astype(np.float32), geom=geom[None].astype(np.float32), cal=cal[None].astype(np.float32),
                   doy_sin=cal[:1, 6].astype(np.float32), doy_cos=cal[:1, 7].astype(np.float32))
        if self.kind == "pv":
            raw["pvlib"] = ph.pvlib_expected_kw(meta, wx, geom)[None].astype(np.float32)
            raw["pvclear"] = ph.pvlib_clear_kw(meta, times, geom)[None].astype(np.float32)
            raw["hist_pv"] = _hist(window.reference, unit)
        else:
            arch = asset.get("archetype", "base")
            raw["arch_idx"] = np.array([F.ARCHETYPES.index(arch) if arch in F.ARCHETYPES else -1])
            raw["hist_load"] = _hist(window.history, unit)
        return raw

    def quantiles(self, raw: dict) -> tuple[list[np.ndarray], list[float], np.ndarray]:
        """Calibrated q [96,3] in kW per member, OOD score per member, night mask."""
        inp = F.build(self.kind, raw)
        feeds = {k: inp[k] for k in ("seq", "ref", "summ", "stat")}
        feeds["mask"] = (~inp["night"]).astype(np.float32)
        qs, oods = [], []
        for m in self.members:
            qn, emb = m.run(feeds)
            q = qn[0] * float(inp["scale"][0])
            off = m.calib.get("quantile_offsets")
            if off:
                for (a, b), lo, hi in zip(off["buckets"], off["q10"], off["q90"]):
                    q[a:b, 0] -= lo * float(inp["scale"][0])
                    q[a:b, 2] += hi * float(inp["scale"][0])
            q[:, 0] = np.minimum(np.maximum(q[:, 0], 0.0), q[:, 1])
            q[:, 2] = np.maximum(q[:, 2], q[:, 1])
            q[inp["night"][0]] = 0.0
            qs.append(q.astype(np.float64))
            oods.append(float(ood_robust_z(m.calib["ood"], emb)[0]) if "ood" in m.calib else 0.0)
        return qs, oods, inp["night"][0]

    def predict(self, asset: Mapping[str, Any], window: Window) -> Prediction:
        if window.horizon != H:
            raise ValueError(f"{self.model} predicts horizon {H}")
        qs, oods, _ = self.quantiles(self.raw(asset, window))
        return _envelope(self, asset, window, qs, oods)


class LoadPredictor(ForecastPredictor):
    kind = "load"


def _envelope(p, asset, window, qs, oods, fault_members=None, extra=None):
    Q = np.stack(qs)  # [M, 96, 3]
    q = Q.mean(0)
    q[:, 0] = np.minimum(q[:, 0], q[:, 1])
    q[:, 2] = np.maximum(q[:, 2], q[:, 1])
    ex = {"members": [{"q10": m[:, 0].tolist(), "q50": m[:, 1].tolist(), "q90": m[:, 2].tolist()} for m in Q],
          "ood_members": oods, "calibrated": all("quantile_offsets" in m.calib for m in p.members)}
    fp = None
    if fault_members is not None:
        P = np.stack(fault_members)  # [M, 8]
        mean = P.mean(0)
        mean = mean / mean.sum()
        fp = {c: float(v) for c, v in zip(FAULTS, mean)}
        if len(ex["members"]) != len(P):  # SiteHealth members != PV members: share the mean expected power
            qm = {"q10": q[:, 0].tolist(), "q50": q[:, 1].tolist(), "q90": q[:, 2].tolist()}
            ex["members"] = [dict(qm) for _ in P]
        for i, m in enumerate(ex["members"]):
            m["fault_probs"] = {c: float(v) for c, v in zip(FAULTS, P[i])}
        p_fault = 1 - P[:, 0]
        ex["p_fault_std"] = float(max(p_fault.std(), P.std(0).max())) if len(P) > 1 else 0.0
    ex.update(extra or {})
    return Prediction(asset_id=_asset_id(asset), model=p.model, model_version=p.model_version, t0=window.t0,
                      horizon=H, q10=q[:, 0], q50=q[:, 1], q90=q[:, 2], ood_score=float(np.mean(oods)),
                      member_std=float(Q[:, :, 1].std(0).mean()) if len(Q) > 1 else 0.0, fault_probs=fp,
                      unit="kW", extra=ex)


# --- M4 ----------------------------------------------------------------------------------------------------
class SiteHealthPredictor:
    model = "sitehealth"

    def __init__(self, members, pv: ForecastPredictor, load: LoadPredictor | None = None):
        self.members = [m if isinstance(m, Member) else Member(m) for m in members]
        self.pv, self.load = pv, load
        v0 = self.members[0].calib.get("model_version", "sitehealth-onnx")
        self.model_version = v0 if len(self.members) == 1 else f"{v0}x{len(self.members)}"

    def predict(self, asset: Mapping[str, Any], window: Window) -> Prediction:
        tel = window.meta.get("telemetry")
        if tel is None:
            raise ValueError("SiteHealth needs window.meta['telemetry'] (pv_ac_w, inv_voltage_v, inv_status, load_w)")
        unit = 1.0 if window.meta.get("power_unit", "W") == "kW" else 1e-3
        raw = self.pv.raw(asset, window)
        qs, _, _ = self.pv.quantiles(raw)
        m1 = np.stack(qs).mean(0)
        pv_raw = _to_steps(tel["pv_ac_w"])
        valid = np.isfinite(pv_raw)
        pv = np.nan_to_num(pv_raw) * unit
        load = np.nan_to_num(_to_steps(tel.get("load_w", np.zeros(H)))) * unit
        inv_kw = float(raw["inverter_kw"][0])
        if self.load is not None and window.history is not None:
            lq, _, _ = self.load.quantiles(self.load.raw(asset, window))
            m2 = np.stack(lq).mean(0)[:, 1]
            raw["hist_load"] = _hist(window.history, unit)
        else:
            m2 = load
            raw["hist_load"] = np.full((1, F.HIST_DAYS * H), max(load.mean(), 0.1), np.float32)
        bat = np.zeros((H, 2)) if window.battery is None else np.nan_to_num(_to_steps(window.battery))
        volt = np.nan_to_num(_to_steps(tel.get("inv_voltage_v", np.full(H, 230.0))), nan=230.0)
        raw.update(pv=pv[None], volt=volt[None], valid=valid[None],
                   status=_status(_to_steps(tel.get("inv_status", np.zeros(H))), pv, inv_kw)[None],
                   soc=bat[None, :, 0], batt=bat[None, :, 1] * unit, load=load[None], m1=m1[None], m2=m2[None])
        th = window.meta.get("telemetry_history")
        if th is not None:
            raw["hist_volt"] = _hist(np.where(np.nan_to_num(th["inv_voltage_v"]) > 0, th["inv_voltage_v"], 230.0), 1.0)
            hpv = window.reference if window.reference is not None else np.zeros(len(th["inv_status"]))
            hpv = np.nan_to_num(np.asarray(hpv, np.float64).ravel()) * unit
            hst = np.asarray(th["inv_status"], np.float64).ravel()
            n_h = min(len(hst), len(hpv))
            raw["hist_status"] = _hist(_status(hst[-n_h:], hpv[-n_h:], inv_kw), 1.0)
            raw["hist_soc"] = _hist(th["soc"], 1.0)
            raw["hist_batt"] = _hist(th["batt_w"], unit)
        inp = F.site_inputs(raw)
        probs, oods = [], []
        for m in self.members:
            p, emb = m.run({k: inp[k] for k in ("seq", "summ", "stat")})
            probs.append(p[0].astype(np.float64))
            oods.append(float(ood_robust_z(m.calib["ood"], emb)[0]) if "ood" in m.calib else 0.0)
        thr = self.members[0].calib.get("thresholds", {}).get("p_fault")
        return _envelope(self, asset, window, qs, oods, fault_members=probs,
                         extra={"p_fault_threshold": thr, "expected_from": self.pv.model_version})


def load_predictor(model: str, model_dir) -> Any:
    """pvnet | loadnet | sitehealth from a directory of ONNX + calibration files."""
    if model == "pvnet":
        return ForecastPredictor("pvnet", find_members("pvnet", model_dir))
    if model == "loadnet":
        return LoadPredictor("loadnet", find_members("loadnet", model_dir))
    if model == "sitehealth":
        pv = ForecastPredictor("pvnet", find_members("pvnet", model_dir))
        try:
            ld = LoadPredictor("loadnet", find_members("loadnet", model_dir))
        except FileNotFoundError:
            ld = None
        return SiteHealthPredictor(find_members("sitehealth", model_dir), pv, ld)
    raise ValueError(f"unknown model {model!r}")
