"""Real model predictors for the replay: the promoted ONNX weights in models/ behind the loop's Predictor interface
(onnxruntime only). `FleetSlice` is an in-memory copy of the window cache for the demo sites (so the training-time
feature code builds the inputs), `ModelBank` runs M1 / M2 / M4 over whole UTC days (local-day predictions are plain
slices of the global 15-min axis), `WindBank` runs WindNBM (M6) per park. Every adapter falls back per call to
`MockPredictor` (envelope `extra["is_fallback"] = 1`); counts land in runs.notes.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np

from wattsentry.contracts import DAY_STEPS, MockPredictor, Prediction, PvFault, Window

from .audit import parse_iso

H = DAY_STEPS
STEP_S = 900
FAULTS = [f.value for f in PvFault]
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODELS = REPO_ROOT / "models"
SLICE_ARRAYS = ("pv_ac", "pv_healthy", "load", "batt", "soc", "inv_v", "inv_status", "fault", "valid", "pvlib_rec",
                "pvlib_rec_fc", "pvclear_rec")
LOC_ARRAYS = ("weather_act", "weather_fc", "geom")


# ============================================================================================ data slice
class FleetSlice:
    """In-memory slice of the window cache: `sites` (indices into the cache) x [t_from, t_to) (whole UTC days).
    Implements what `wattsentry.model.features.gather` needs from an ArraySource."""

    def __init__(self, root: Path, site_idx: Sequence[int], t_from: int, t_to: int):
        import pandas as pd

        self.root = Path(root)
        idx = json.loads((self.root / "index.json").read_text())
        if idx.get("format") != "wattsentry-windows-v1":
            raise ValueError(f"{root}: not a wattsentry-windows-v1 cache")
        all_sites = json.loads((self.root / "sites.json").read_text())
        t_start = int(pd.Timestamp(idx["t_start"]).timestamp())
        if (t_from - t_start) % 86400 or (t_to - t_from) % 86400:
            raise ValueError("FleetSlice needs whole UTC days")
        i0, i1 = (t_from - t_start) // STEP_S, (t_to - t_start) // STEP_S
        if i0 < 0 or i1 > int(idx["n_steps"]):
            raise ValueError(f"{root}: [{t_from}, {t_to}) outside the cache time axis")
        self.site_idx = list(site_idx)
        self.sites = [dict(all_sites[i]) for i in self.site_idx]
        self.t_start = pd.Timestamp(t_from, unit="s", tz="UTC")
        self.t0 = t_from
        self.n_steps = i1 - i0
        self.label = idx.get("label", "synthetic RO")
        self.index = {"arrays": {}, **{k: v for k, v in idx.items() if k != "arrays"}}
        self.localities = idx.get("localities", [])
        self._a: dict[str, np.ndarray] = {}
        s = np.asarray(self.site_idx)
        for name in SLICE_ARRAYS:
            if name not in idx["arrays"]:
                continue
            mm = np.load(self.root / idx["arrays"][name]["file"], mmap_mode="r")
            dt = np.int8 if mm.dtype == np.int8 else np.float32
            self._a[name] = np.ascontiguousarray(mm[s, i0:i1]).astype(dt)
            self.index["arrays"][name] = idx["arrays"][name]
        for name in LOC_ARRAYS:
            mm = np.load(self.root / idx["arrays"][name]["file"], mmap_mode="r")
            self._a[name] = np.ascontiguousarray(mm[:, i0:i1]).astype(np.float32)
            self.index["arrays"][name] = idx["arrays"][name]
        cal = np.load(self.root / idx["arrays"]["calendar"]["file"], mmap_mode="r")
        self._a["calendar"] = np.ascontiguousarray(cal[i0:i1]).astype(np.float32)
        self.index["arrays"]["calendar"] = idx["arrays"]["calendar"]
        self.episodes: list[dict] = []
        self.n = len(self.sites)

    def array(self, name: str) -> np.ndarray:
        return self._a[name]

    def has(self, name: str) -> bool:
        return name in self._a

    def site_meta(self, key: str, dtype=np.float32) -> np.ndarray:
        return np.array([s.get(key, 0) for s in self.sites], dtype=dtype)

    def loc_idx(self) -> np.ndarray:
        return self.site_meta("loc_idx", np.int64)

    def step(self, ts: int) -> int:
        return (int(ts) - self.t0) // STEP_S

    @property
    def n_days(self) -> int:
        return self.n_steps // H


# ============================================================================================ model bank
def models_dir() -> Path:
    return Path(os.environ.get("WATTSENTRY_MODELS", DEFAULT_MODELS))


def _members(model: str, d: Path):
    from wattsentry.model.predictor import Member, find_members

    return [Member(p) for p in find_members(model, d)]


def _version(members, fallback: str) -> str:
    v0 = members[0].calib.get("model_version", fallback)
    return v0 if len(members) == 1 else f"{v0}x{len(members)}"


class ModelBank:
    """Runs M1 / M2 / M4 ONNX members over a FleetSlice. Arrays are [S, T, 3] kW on the slice's axis."""

    def __init__(self, src: FleetSlice, model_dir: Optional[Path] = None, batch: int = 512):
        self.src = src
        self.dir = Path(model_dir or models_dir())
        self.batch = batch
        self.errors: dict[str, str] = {}
        self.members: dict[str, list] = {}
        for key, model, sub in (("pv_now", "pvnet", ""), ("pv_fc", "pvnet", "pvnet_fc"), ("load", "loadnet", ""),
                                ("sitehealth", "sitehealth", "")):
            try:
                self.members[key] = _members(model, self.dir / sub if sub else self.dir)
            except Exception as e:  # missing weights -> that component falls back to the mock
                self.errors[key] = f"{type(e).__name__}: {e}"
        self.versions = {k: _version(m, k) for k, m in self.members.items()}
        self.q: dict[str, np.ndarray] = {}
        self.day_ood: dict[str, np.ndarray] = {}
        self.day_std: dict[str, np.ndarray] = {}
        self._valid_days = np.arange(15, src.n_days)  # 14 days of history + 1 spare
        for key, kind, mode in (("pv_fc", "pv", "forecast"), ("pv_now", "pv", "nowcast"), ("load", "load", "forecast"),
                                ("load_now", "load", "nowcast")):
            mkey = "load" if key.startswith("load") else key
            if mkey not in self.members:
                continue
            try:
                self.q[key], self.day_ood[key], self.day_std[key] = self._forecast(self.members[mkey], kind, mode)
            except Exception as e:
                self.errors[key] = f"{type(e).__name__}: {e}"
        self.health_cache: dict[tuple[int, int], tuple[np.ndarray, float, float]] = {}

    # -- M1 / M2 over whole UTC days ---------------------------------------------------------------------
    def _forecast(self, members, kind: str, mode: str):
        from wattsentry.model import features as F

        src = self.src
        S, T = src.n, src.n_steps
        out = np.full((S, T, 3), np.nan, np.float32)
        D = src.n_days
        ood = np.full((S, D), np.nan, np.float32)
        std = np.full((S, D), np.nan, np.float32)
        s_, d_ = np.meshgrid(np.arange(S), self._valid_days, indexing="ij")
        s_, d_ = s_.ravel(), d_.ravel()
        for k in range(0, len(s_), self.batch):
            ss, dd = s_[k:k + self.batch], d_[k:k + self.batch]
            raw = F.gather(src, ss, dd * H, kind, mode)
            inp = F.build(kind, raw)
            feeds = {n: inp[n] for n in ("seq", "ref", "summ", "stat")}
            feeds["mask"] = (~inp["night"]).astype(np.float32)
            scale = inp["scale"].astype(np.float64)[:, None]
            qs, oo = [], []
            for m in members:
                qn, emb = m.run(feeds)
                q = qn.astype(np.float64) * scale[..., None]
                off = m.calib.get("quantile_offsets")
                if off:
                    for (a, b), lo, hi in zip(off["buckets"], off["q10"], off["q90"]):
                        q[:, a:b, 0] -= lo * scale
                        q[:, a:b, 2] += hi * scale
                q[..., 0] = np.minimum(np.maximum(q[..., 0], 0.0), q[..., 1])
                q[..., 2] = np.maximum(q[..., 2], q[..., 1])
                q[inp["night"]] = 0.0
                qs.append(q)
                if "ood" in m.calib:
                    from wattsentry.model.predictor import ood_robust_z

                    oo.append(ood_robust_z(m.calib["ood"], emb))
            Q = np.stack(qs)  # [M, B, 96, 3]
            qm = Q.mean(0)
            qm[..., 0] = np.minimum(qm[..., 0], qm[..., 1])
            qm[..., 2] = np.maximum(qm[..., 2], qm[..., 1])
            idx = dd[:, None] * H + np.arange(H)[None]
            out[ss[:, None], idx] = qm.astype(np.float32)
            ood[ss, dd] = np.mean(oo, 0) if oo else 0.0
            std[ss, dd] = Q[..., 1].std(0).mean(1) if len(Q) > 1 else 0.0
        return out, ood, std

    def series(self, key: str, i: int, step0: int) -> tuple[np.ndarray, float, float]:
        """[96, 3] kW from step0 for site i + (ood, member_std) of the UTC day holding the window's midpoint."""
        q = self.q[key][i, step0:step0 + H]
        if q.shape[0] != H or not np.isfinite(q).all():
            raise ValueError(f"{key}: no prediction for site {i} at step {step0}")
        day = (step0 + H // 2) // H
        return q.astype(np.float64), float(self.day_ood[key][i, day]), float(self.day_std[key][i, day])

    # -- M4 on the 24 h ending at a tick -------------------------------------------------------------------
    def health(self, sites: Sequence[int], end_step: int) -> dict[int, tuple[np.ndarray, float, float]]:
        """Fault probabilities (member mean), P_fault std across members, OOD for the window [end-96, end)."""
        from wattsentry.model import features as F
        from wattsentry.model.predictor import ood_robust_z

        if "sitehealth" not in self.members or "pv_now" not in self.q or "load_now" not in self.q:
            raise RuntimeError("sitehealth: weights or upstream M1/M2 missing")
        todo = [i for i in sites if (i, end_step) not in self.health_cache]
        if todo:
            ss = np.asarray(todo, np.int64)
            starts = np.full(len(ss), end_step - H, np.int64)
            m1 = np.nan_to_num(self.q["pv_now"])
            m2 = np.nan_to_num(self.q["load_now"][..., 1])
            raw = F.gather(self.src, ss, starts, "site", "nowcast", m1=m1, m2=m2)
            inp = F.site_inputs(raw)
            P, oo = [], []
            for m in self.members["sitehealth"]:
                p, emb = m.run({k: inp[k] for k in ("seq", "summ", "stat")})
                P.append(p.astype(np.float64))
                oo.append(ood_robust_z(m.calib["ood"], emb) if "ood" in m.calib else np.zeros(len(ss)))
            P = np.stack(P)  # [M, B, 8]
            mean = P.mean(0)
            mean = mean / mean.sum(1, keepdims=True)
            pstd = (1 - P[..., 0]).std(0) if len(P) > 1 else np.zeros(len(ss))
            o = np.mean(oo, 0)
            for j, i in enumerate(todo):
                self.health_cache[(i, end_step)] = (mean[j], float(pstd[j]), float(o[j]))
        return {i: self.health_cache[(i, end_step)] for i in sites}

    def describe(self) -> dict[str, Any]:
        return {"dir": str(self.dir), "versions": self.versions, "errors": self.errors}


# ============================================================================================ adapters
class _Adapter:
    """Contract Predictor over a ModelBank; per-call fallback to the mock."""

    kind = "pv"

    def __init__(self, bank: ModelBank, key: str, model: str, index: Mapping[str, int], step_of: Any,
                 fallback: Any = None):
        self.bank, self.key, self.model = bank, key, model
        self.index = index  # world site id -> slice row
        self.step_of = step_of  # (t0 iso) -> slice step
        self.fallback = fallback or MockPredictor(self.kind)
        self.model_version = bank.versions.get("pv_fc" if key == "pv_fc" else key.replace("_now", ""), "mock-1")
        self.n_calls = self.n_fallback = 0
        self.last_error = ""

    @property
    def available(self) -> bool:
        return self.key in self.bank.q

    def _fb(self, asset, t0, exc: Optional[BaseException] = None) -> Prediction:
        self.n_fallback += 1
        if exc is not None:
            self.last_error = f"{type(exc).__name__}: {exc}"
        p = self.fallback.predict(asset, t0=t0)
        p.extra = {**(p.extra or {}), "is_fallback": 1}
        return p

    def predict(self, asset: Mapping[str, Any], window: Optional[Window] = None, *, t0: Optional[str] = None,
                horizon: Optional[int] = None) -> Prediction:
        t0 = window.t0 if window is not None else t0
        self.n_calls += 1
        try:
            i = self.index[str(asset["site_id"])]
            q, ood, std = self.bank.series(self.key, i, self.step_of(t0))
        except Exception as e:
            return self._fb(asset, t0, e)
        p = Prediction(asset_id=str(asset["site_id"]), model=self.model, model_version=self.model_version, t0=t0,
                       horizon=H, q10=q[:, 0].tolist(), q50=q[:, 1].tolist(), q90=q[:, 2].tolist(),
                       ood_score=max(0.0, ood if np.isfinite(ood) else 0.0),
                       member_std=max(0.0, std if np.isfinite(std) else 0.0), unit="kW",
                       extra={"is_fallback": 0, "source": self.key})
        p.validate()
        return p


class PvForecastAdapter(_Adapter):
    kind = "pv"


class LoadForecastAdapter(_Adapter):
    kind = "load"


class SiteHealthAdapter(_Adapter):
    """M4 on the trailing 24 h ending at the asking tick (`window.meta["tick"]`, default 0 = local midnight).
    q10/q50/q90 = expected PV for the local day (M1 nowcast ensemble), what qualify measures the loss against."""

    kind = "sitehealth"
    reassess_ticks = (44, 52, 60, 68)  # 11:00, 13:00, 15:00, 17:00 Europe/Bucharest (summer): intraday re-checks

    def __init__(self, bank, index, step_of, fallback=None):
        super().__init__(bank, "pv_now", "sitehealth", index, step_of, fallback or MockPredictor("sitehealth"))
        self.model_version = bank.versions.get("sitehealth", "mock-1")
        th = (bank.members.get("sitehealth") or [None])[0]
        self.p_fault_threshold = (th.calib.get("thresholds", {}).get("p_fault") if th is not None else None)

    @property
    def available(self) -> bool:
        return "sitehealth" in self.bank.members and "pv_now" in self.bank.q and "load_now" in self.bank.q

    def prefetch(self, site_ids: Sequence[str], t0: str, tick: int = 0) -> None:
        """Batch M4 for many sites at once (the loop calls it before asking site by site)."""
        if not self.available:
            return
        try:
            self.bank.health([self.index[s] for s in site_ids if s in self.index], self.step_of(t0) + tick)
        except Exception as e:
            self.last_error = f"{type(e).__name__}: {e}"

    def predict(self, asset, window=None, *, t0=None, horizon=None) -> Prediction:
        t0 = window.t0 if window is not None else t0
        tick = int((window.meta or {}).get("tick", 0)) if window is not None else 0
        self.n_calls += 1
        try:
            i = self.index[str(asset["site_id"])]
            step0 = self.step_of(t0)
            q, ood_pv, std_pv = self.bank.series("pv_now", i, step0)
            probs, pstd, ood = self.bank.health([i], step0 + tick)[i]
        except Exception as e:
            return self._fb(asset, t0, e)
        fp = {c: float(v) for c, v in zip(FAULTS, probs)}
        p = Prediction(asset_id=str(asset["site_id"]), model="sitehealth", model_version=self.model_version, t0=t0,
                       horizon=H, q10=q[:, 0].tolist(), q50=q[:, 1].tolist(), q90=q[:, 2].tolist(),
                       ood_score=max(0.0, float(ood)), member_std=max(0.0, float(std_pv)), fault_probs=fp, unit="kW",
                       extra={"is_fallback": 0, "p_fault_std": pstd, "assessed_until_tick": tick,
                              "p_fault_threshold": self.p_fault_threshold, "expected_from": self.bank.versions.get(
                                  "pv_now"), "ood_expected": ood_pv})
        p.validate()
        return p


# ============================================================================================ wind (M6)
class WindBank:
    """WindNBM over each park's arrays. `arrays[pid]` = dict(turbines, ts [T] datetime64[m], telemetry, meteo, err,
    weather) covering the replay week plus >= 28 days of history (what M6 expects)."""

    def __init__(self, arrays: Mapping[str, Mapping[str, Any]], model_dir: Optional[Path] = None):
        from wattsentry.model.wind.data import CLASSES
        from wattsentry.model.wind.predictor import WindNBMPredictor

        self.m = WindNBMPredictor(model_dir or models_dir())
        self.model_version = self.m.model_version
        self.classes = CLASSES
        self.scores: dict[str, dict[str, Any]] = {}
        self.col: dict[str, tuple[str, int]] = {}
        self.consensus_steps: dict[str, int] = {}
        ci = CLASSES.index("curtailment")
        th = float(self.m.cal["class_thresholds"]["curtailment"])
        for pid, a in arrays.items():
            s = self.m.score_park(a["turbines"], a["ts"], a["telemetry"], a["meteo"], a.get("weather"), a.get("err"))
            # Park consensus on M6's own curtailment probabilities. M6's decide() vetoes curtailment unless >= 15 %
            # of the park sits at one common power ceiling; a cap to 0 % leaves noisy near-zero power, so the veto
            # turns a park-wide cap the classifier is sure about into "normal". Replay rule: where >= 50 % of the
            # park's reporting turbines have p_curtailment >= its class threshold in the same step, those turbines
            # are curtailment (count in runs.notes).
            cls = s["cls"].copy()
            hit = (s["probs"][..., ci] >= th) & np.isfinite(s["power"])
            frac = hit.sum(1) / np.maximum(np.isfinite(s["power"]).sum(1), 1)
            park = frac >= 0.5
            fix = park[:, None] & hit & (cls != ci)
            cls[fix] = ci
            self.consensus_steps[pid] = int(fix.sum())
            s["cls"] = cls
            self.scores[pid] = s
            for j, t in enumerate(a["turbines"]):
                self.col[t["replay_id"]] = (pid, j)

    def day(self, turbine_id: str, t0: str) -> dict[str, Any]:
        pid, j = self.col[turbine_id]
        s = self.scores[pid]
        i0 = int(np.searchsorted(s["ts"], np.datetime64(datetime.fromtimestamp(parse_iso(t0), timezone.utc)
                                                        .replace(tzinfo=None), "m")))
        if i0 + H > len(s["ts"]):
            raise ValueError("wind arrays do not cover the day")
        sl = slice(i0, i0 + H)
        return {"pq": s["pq"][sl, j], "cls": s["cls"][sl, j], "probs": s["probs"][sl, j], "ood": s["ood"][sl, j],
                "pstd": s["pstd"][sl, j], "rated": float(s["rated"][j])}


class WindAdapter:
    model = "windnbm"
    kind = "wind"

    def __init__(self, bank: Optional[WindBank], fallback: Any = None, error: str = ""):
        self.bank = bank
        self.fallback = fallback or MockPredictor("wind")
        self.model_version = bank.model_version if bank else "mock-1"
        self.n_calls = self.n_fallback = 0
        self.last_error = error

    def predict(self, asset, window=None, *, t0=None, horizon=None) -> Prediction:
        t0 = window.t0 if window is not None else t0
        self.n_calls += 1
        tid = str(asset["turbine_id"])
        try:
            if self.bank is None:
                raise RuntimeError(self.last_error or "WindNBM not loaded")
            d = self.bank.day(tid, t0)
        except Exception as e:
            self.n_fallback += 1
            self.last_error = f"{type(e).__name__}: {e}"
            p = self.fallback.predict(asset, t0=t0)
            p.extra = {**(p.extra or {}), "is_fallback": 1}
            return p
        m = self.bank.m
        r = d["rated"]
        q = d["pq"].astype(np.float64)
        q50 = np.clip(q[:, 1] * r, 0, r)
        q10 = np.minimum((q[:, 0] + m.offsets["q10"]) * r, q50).clip(0, None)
        q90 = np.clip(np.maximum((q[:, 2] + m.offsets["q90"]) * r, q50), q50, None)
        classes = [self.bank.classes[int(c)] for c in d["cls"]]
        scores = np.clip(1.0 - d["probs"][:, 0], 0, 1)
        nz = [c for c in classes if c != "normal"]
        top = max(set(nz), key=nz.count) if nz else "normal"
        p = Prediction(asset_id=tid, model=self.model, model_version=self.model_version, t0=t0, horizon=H,
                       q10=q10.tolist(), q50=q50.tolist(), q90=q90.tolist(),
                       ood_score=float(np.nanmean(d["ood"])), member_std=float(np.nanmean(d["pstd"]) * r),
                       anomaly_class=top, anomaly_score=float(scores.max()),
                       extra={"is_fallback": 0, "step_classes": classes,
                              "step_scores": [round(float(x), 4) for x in scores]})
        p.validate()
        return p


# ============================================================================================ summary
def predictor_report(preds: Mapping[str, Any]) -> dict[str, Any]:
    """Per component: real or fallback, version, calls, fallback calls, last error (for runs.notes)."""
    out = {}
    for k, p in preds.items():
        n, nf = getattr(p, "n_calls", None), getattr(p, "n_fallback", None)
        ver = str(getattr(p, "model_version", "?"))
        real = not ver.startswith("mock") and (n is None or nf is None or nf < n or n == 0)
        out[k] = {"model": getattr(p, "model", k), "version": ver, "source": "real" if real else "fallback:mock",
                  "calls": n, "fallback_calls": nf, "error": getattr(p, "last_error", "") or ""}
    return out
