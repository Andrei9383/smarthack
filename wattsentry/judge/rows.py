"""Judge training rows: replay sampled weeks of the witnesses (PVNet, LoadNet, SiteHealth, WindNBM) on the window
cache with drift and cold-start augmentation, grade every prediction, run the live trust ledger forward causally and
emit one raw row per (witness, asset, local day) with its validity target (simulator truth).

    rows = generate(RowConfig(year=2024), workers=20)       # 12 weeks in a process pool, cached on disk
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import sys
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd

from wattsentry.judge import features as F
from wattsentry.judge.buckets import bucket_key as _bucket_key

H = 96
DAY_S = 86400
HIST_DAYS = 15  # ModelBank needs 14 days of history + 1 spare before the first local day
ALPHA = 0.2  # interval score for the P10-P90 band
HIT_LO, HIT_HI = 0.60, 0.95  # forecast validity: hit rate of the P10-P90 band
MULTS = (0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0, 6.0)  # band multipliers for tau_widen / widen_factor
N0 = 7.0  # ledger prior weight
FORECAST_SCORES = ("calibration", "sharpness")
DETECTOR_SCORES = ("alarm_precision", "miss_score")
SUBSTITUTE = {"pvnet": "pvlib", "loadnet": "persistence_load"}
WITNESSES = ("pvnet", "loadnet", "sitehealth", "windnbm")
DRIFT_FORMS = {"pvnet": ("bias", "narrow", "frozen"), "loadnet": ("bias", "narrow", "frozen"),
               "sitehealth": ("confusion", "false_alarm", "blind", "frozen"),
               "windnbm": ("confusion", "false_alarm", "blind", "frozen")}
WIND_CONFUSION = {"curtailment": "fault_stop", "fault_stop": "curtailment", "icing": "underperformance",
                  "underperformance": "icing", "overheating": "underperformance",
                  "anemometer_fault": "underperformance"}
WIND_HOLDOUT_PARKS = ("WP2",)  # M6's held-out park = unseen
CONFUSION = {"string_outage": "sensor_fault", "soiling": "shading", "shading": "soiling",
             "overvoltage_trip": "clipping", "battery_fade": "sensor_fault", "clipping": "overvoltage_trip"}
ROW_FILES = ("rows.py", "features.py", "grader.py", "scores.py", "gate.py", "ledger.py", "buckets.py")


# ============================================================================================ week sampling
@dataclass(frozen=True)
class DriftSpec:
    witness: str
    form: str
    start_day: int  # index into the week's local days (0 = first burn-in day)
    scope: str  # all | battery | unseen
    magnitude: float  # bias: signed fraction; false_alarm / blind: share of probability mass moved


@dataclass(frozen=True)
class WeekSpec:
    year: int
    month: int
    start: str  # first sampled local day (YYYY-MM-DD); burn-in days precede it
    drift: Optional[DriftSpec]
    cold_day: Optional[int]  # ledger wiped at this day index (None = never)
    seed: int


@dataclass
class RowConfig:
    year: int = 2024
    n_weeks: int = 12
    week_days: int = 7
    burn_in: int = 21  # days of ledger history before the sampled week (rows are emitted for the week only)
    drift_share: float = 0.5
    cold_share: float = 0.25
    seed: int = 0
    sites: str = "all"  # all | <n> (first n sites per split, for smoke tests)
    witnesses: tuple = WITNESSES
    models_dir: Optional[str] = None
    windows: Optional[str] = None

    def key_dict(self) -> dict:
        d = dataclasses.asdict(self)
        d.pop("models_dir"), d.pop("windows")
        d["witnesses"] = list(self.witnesses)
        return d


def sample_weeks(year: int, n: int = 12, seed: int = 0, drift_share: float = 0.5, cold_share: float = 0.25,
                 week_days: int = 7, burn_in: int = 21, witnesses: Sequence[str] = WITNESSES) -> list[WeekSpec]:
    """One week per month (n <= 12 picks evenly spread months), half with drift, a quarter with a ledger wipe."""
    rng = np.random.default_rng([year, seed, 148])
    months = np.linspace(1, 12, n).round().astype(int) if n < 12 else np.arange(1, 13)
    n_drift, n_cold = int(round(n * drift_share)), int(round(n * cold_share))
    drift_idx = set(rng.permutation(n)[:n_drift].tolist())
    cold_idx = set(rng.permutation(n)[:n_cold].tolist())
    out = []
    for k, m in enumerate(months):
        day = int(rng.integers(1, 23))  # the week stays inside the month (and inside the data's last day)
        start = date(year, int(m), day).isoformat()
        drift = None
        if k in drift_idx:
            w = str(rng.choice(list(witnesses)))
            form = str(rng.choice(DRIFT_FORMS.get(w, ("frozen",))))
            mag = float(rng.uniform(0.2, 0.4)) * (1 if rng.random() < 0.5 else -1) if form == "bias" else \
                float(rng.uniform(0.3, 0.7))
            drift = DriftSpec(w, form, int(rng.integers(burn_in - 7, burn_in + week_days - 1)),
                              str(rng.choice(["all", "all", "battery", "unseen"])), round(mag, 3))
        cold = int(rng.integers(burn_in - 2, burn_in + week_days - 1)) if k in cold_idx else None
        out.append(WeekSpec(year, int(m), start, drift, cold, int(rng.integers(1 << 30))))
    return out


# ============================================================================================ drift
def _scope_mask(sites: list[dict], scope: str) -> np.ndarray:
    if scope == "battery":
        return np.array([float(s.get("battery_kwh") or 0) > 0 for s in sites])
    if scope == "unseen":
        return np.array([s.get("split") != "train" for s in sites])
    return np.ones(len(sites), bool)


def _frozen(a: np.ndarray, days_on: np.ndarray, assets: np.ndarray, shape: tuple) -> np.ndarray:
    """The drifted assets' output of the day before the drift starts, repeated over the drifted days."""
    src = max(int(np.argmax(days_on)) - 1, 0)
    return np.broadcast_to(a[assets][:, src:src + 1], shape).copy()


def drift_forecast(q: np.ndarray, spec: DriftSpec, days_on: np.ndarray, assets: np.ndarray) -> np.ndarray:
    """q [S, D, 96, 3] (local days) -> degraded copy: for assets x days_on apply spec.form."""
    q = q.copy()
    sel = np.ix_(assets, days_on)
    if not assets.any() or not days_on.any():
        return q
    x = q[sel]  # [s, d, 96, 3]
    if spec.form == "bias":
        x = x * (1.0 + spec.magnitude)
    elif spec.form == "narrow":
        mid = x[..., 1:2]
        x = mid + 0.5 * (x - mid)
    elif spec.form == "frozen":
        x = _frozen(q, days_on, assets, x.shape)
    q[sel] = x
    return q


def drift_detector(p: np.ndarray, spec: DriftSpec, days_on: np.ndarray, assets: np.ndarray,
                   classes: Sequence[str] = F.PV_CLASSES) -> np.ndarray:
    """p [S, D, C] class probabilities (0 = healthy) -> degraded copy (rows still sum to 1)."""
    p = p.copy()
    if not assets.any() or not days_on.any():
        return p
    sel = np.ix_(assets, days_on)
    x = p[sel]
    ci = {c: i for i, c in enumerate(classes)}
    if spec.form == "confusion":
        y = x.copy()
        for a, b in CONFUSION.items():
            if a in ci and b in ci:
                y[..., ci[b]] += x[..., ci[a]]
                y[..., ci[a]] -= x[..., ci[a]]
        x = y
    elif spec.form == "false_alarm":
        to = ci.get("sensor_fault", len(classes) - 1)
        moved = x[..., 0] * spec.magnitude
        x = x.copy()
        x[..., 0] -= moved
        x[..., to] += moved
    elif spec.form == "blind":
        moved = x[..., 1:] * spec.magnitude
        x = x.copy()
        x[..., 1:] -= moved
        x[..., 0] += moved.sum(-1)
    elif spec.form == "frozen":
        x = _frozen(p, days_on, assets, x.shape)
    x = np.clip(x, 0, None)
    x /= np.maximum(x.sum(-1, keepdims=True), 1e-9)
    p[sel] = x
    return p


# ============================================================================================ grading
def interval_score(q10, q90, y, alpha: float = ALPHA):
    """Per-step Winkler score: width + 2/alpha x distance outside the band (NaN where y is NaN)."""
    lo, hi = np.minimum(q10, q90), np.maximum(q10, q90)
    return (hi - lo) + (2 / alpha) * np.clip(lo - y, 0, None) + (2 / alpha) * np.clip(y - hi, 0, None)


def norm_err(q10, q50, q90, y):
    """Distance of y from q50 in units of the band half on its side (hit at band multiplier m <=> e <= m)."""
    with np.errstate(invalid="ignore", divide="ignore"):
        lo = np.where(q50 - q10 > 1e-9, (q50 - y) / (q50 - q10), np.where(y < q50 - 1e-9, np.inf, 0.0))
        hi = np.where(q90 - q50 > 1e-9, (y - q50) / (q90 - q50), np.where(y > q50 + 1e-9, np.inf, 0.0))
    return np.where(y < q50, lo, hi)


def _mmean(x, m):
    m = m & np.isfinite(x)
    n = m.sum(-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(n > 0, np.where(m, x, 0.0).sum(-1) / np.maximum(n, 1), np.nan), n


def grade_forecast(q: np.ndarray, y: np.ndarray, mask: np.ndarray, ref: Optional[np.ndarray]) -> dict[str, np.ndarray]:
    """q [N, 96, 3], y [N, 96], mask [N, 96] scored steps, ref [N, 96, 3] substitute (None = no sharpness).
    -> hit, calibration, sharpness, valid, graded, hit_m<mult> (widened-band hit rates), e_q (vigintiles)."""
    q10, q50, q90 = q[..., 0], q[..., 1], q[..., 2]
    ok = mask & np.isfinite(y) & np.isfinite(q50)
    hit, n = _mmean(((y >= q10) & (y <= q90)).astype(float), ok)
    graded = n >= 8
    cal = np.clip(1 - np.abs(hit - 0.8) / 0.8, 0, 1)
    out = {"hit": hit, "calibration": cal, "graded": graded}
    if ref is not None:
        ok2 = ok & np.isfinite(ref[..., 1])
        im, _ = _mmean(interval_score(q10, q90, y), ok2)
        ir, _ = _mmean(interval_score(ref[..., 0], ref[..., 2], y), ok2)
        with np.errstate(invalid="ignore", divide="ignore"):
            skill = np.where(ir > 0, 1 - im / ir, 0.0)
        out["sharpness"] = np.where(np.isfinite(im) & np.isfinite(ir), np.clip(skill, 0, 1), np.nan)
        out["skill_raw"] = skill
    else:
        out["sharpness"] = np.full(len(y), np.nan)
        out["skill_raw"] = np.full(len(y), np.nan)
    e = norm_err(q10, q50, q90, y)
    for m in MULTS:
        out[f"hit_m{m:g}"] = _mmean((e <= m).astype(float), ok)[0]
    e_masked = np.where(ok, np.where(np.isinf(e), 50.0, e), np.nan)  # a miss on a zero-width side counts as far out
    out["e_q"] = vigintiles(e_masked)
    sk = out["skill_raw"]
    out["valid"] = (graded & (hit >= HIT_LO) & (hit <= HIT_HI) & np.where(np.isfinite(sk), sk > 0, True)).astype(float)
    return out


def vigintiles(x: np.ndarray, k: int = 20) -> np.ndarray:
    """[N, 96] with NaN -> [N, k] lower quantiles at (i + 0.5) / k of the finite values (NaN rows -> NaN)."""
    x = np.sort(np.asarray(x, float), axis=-1)  # NaN sort last
    n = np.isfinite(x).sum(-1)
    q = (np.arange(k) + 0.5) / k
    idx = np.floor(q[None] * np.maximum(n - 1, 0)[:, None]).astype(int)
    out = np.take_along_axis(x, idx, -1)
    out[n == 0] = np.nan
    return out


def truth_class(fault_steps: np.ndarray, classes: Sequence[str] = F.PV_CLASSES, min_steps: int = 4) -> np.ndarray:
    """[N, 96] int class codes over the assessed window -> the day's true class (most frequent non-healthy class with
    at least `min_steps` steps, else healthy)."""
    f = np.asarray(fault_steps, np.int64)
    C = len(classes)
    counts = np.stack([(f == c).sum(-1) for c in range(C)], -1)
    counts[:, 0] = 0
    j = counts.argmax(-1)
    return np.where(counts[np.arange(len(f)), j] >= min_steps, np.asarray(classes, object)[j], classes[0])


def grade_detector(alarm: np.ndarray, called: np.ndarray, truth: np.ndarray, healthy: str = "healthy"):
    """-> alarm_precision (alarm rows), miss_score (healthy-call rows), valid."""
    alarm = np.asarray(alarm, bool)
    faulty = truth != healthy
    ap = np.where(called == truth, 1.0, np.where(faulty, 0.5, 0.0))
    ms = np.where(faulty, 0.0, 1.0)
    valid = np.where(alarm, ap == 1.0, ms == 1.0).astype(float)
    return {"alarm_precision": np.where(alarm, ap, np.nan), "miss_score": np.where(alarm, np.nan, ms),
            "valid": valid, "graded": np.ones(len(alarm), bool)}


# ============================================================================================ buckets + ledger
def bucket_key(kind: str, seen: bool, battery: bool, month: int, gappy: bool, ood_high: bool) -> str:
    """kind|seen|battery|season|quality|ood, the `judge.buckets` key."""
    return _bucket_key(kind, bool(seen), bool(battery), int(month), bool(gappy), bool(ood_high))


def lo80(trust, n_eff):
    """Lower end of the 80 % interval of the equivalent Beta(trust x (n0 + n_eff), (1 - trust) x (n0 + n_eff))."""
    from scipy.stats import beta
    t, n = np.asarray(trust, float), N0 + np.asarray(n_eff, float)
    return beta.ppf(0.1, np.maximum(t * n, 1e-3), np.maximum((1 - t) * n, 1e-3))


# ============================================================================================ wind (M6)
def drift_wind_steps(cls: np.ndarray, spec: DriftSpec, days_on: np.ndarray, assets: np.ndarray,
                     rng: np.random.Generator) -> np.ndarray:
    """cls [N, D, 96] per-step M6 class codes (0 = normal) -> degraded copy."""
    cls = cls.copy()
    if not assets.any() or not days_on.any():
        return cls
    sel = np.ix_(assets, days_on)
    x = cls[sel]
    ci = {c: i for i, c in enumerate(F.WIND_CLASSES)}
    if spec.form == "confusion":
        y = x.copy()
        for a, b in WIND_CONFUSION.items():
            y[x == ci[a]] = ci[b]
        x = y
    elif spec.form == "blind":
        x = np.where((x != 0) & (rng.random(x.shape) < spec.magnitude), 0, x)
    elif spec.form == "false_alarm":
        hit = rng.random(x.shape[:2]) < spec.magnitude * 0.5  # a share of turbine-days get a 2 h false alarm
        start = rng.integers(0, 96 - 8, x.shape[:2])
        steps = np.arange(96)[None, None]
        block = hit[..., None] & (steps >= start[..., None]) & (steps < start[..., None] + 8)
        x = np.where(block & (x == 0), ci["underperformance"], x)
    elif spec.form == "frozen":
        x = _frozen(cls, days_on, assets, x.shape)
    cls[sel] = x
    return cls


def wind_week(day0: int, D: int, models_dir: Path) -> Optional[dict]:
    """M6 WindNBM over every park for the local days [day0, day0 + D) (+ 29 days of history, as the replay), with
    the parks' fault labels (truth for targets only). None when the wind cache or M6 weights are missing."""
    import polars as pl

    from wattsentry.agent.predictors import WindBank
    from wattsentry.model.wind.data import external_wind_dir, from_long

    d = external_wind_dir()
    if d is None or not (d / "wind_telemetry.parquet").exists():
        return None
    lo = datetime.fromtimestamp((day0 - 29 * DAY_S) // DAY_S * DAY_S, timezone.utc)
    hi = datetime.fromtimestamp(day0 + (D + 1) * DAY_S, timezone.utc)
    rng_ = (pl.col("ts") >= lo) & (pl.col("ts") < hi)
    tur = pl.read_parquet(d / "turbines.parquet")
    tel = pl.scan_parquet(d / "wind_telemetry.parquet").filter(rng_).collect()
    if tel.height == 0:
        return None
    met = pl.scan_parquet(d / "meteo_wind.parquet").filter(rng_).collect() if (d / "meteo_wind.parquet").exists() \
        else None
    wx = pl.read_parquet(d / "weather.parquet").filter(rng_) if (d / "weather.parquet").exists() else None
    fl = pl.read_parquet(d / "fault_labels.parquet") if (d / "fault_labels.parquet").exists() else None
    wd = from_long(tel, tur, meteo=met, weather=wx, fault_labels=fl, source=f"external:{d}")
    arrays = {}
    for pid in wd.park_ids:
        cols = [j for j, t in enumerate(wd.turbines) if t["park_id"] == pid]
        p = wd.park_ids.index(pid)
        for j in cols:
            wd.turbines[j]["replay_id"] = wd.turbines[j]["turbine_id"]
        arrays[pid] = {"turbines": [wd.turbines[j] for j in cols], "ts": wd.ts,
                       "telemetry": {k: v[:, cols] for k, v in wd.sensors.items()},
                       "meteo": {k: v[:, cols] for k, v in wd.meteo.items()},
                       "err": wd.err[:, cols], "weather": {k: v[:, p] for k, v in wd.weather.items()}}
    bank = WindBank(arrays, models_dir)
    i0 = int(np.searchsorted(wd.ts, np.datetime64(datetime.fromtimestamp(day0, timezone.utc).replace(tzinfo=None),
                                                  "m")))
    if i0 - H < 0 or i0 + D * H > len(wd.ts):
        return None
    idx = i0 + np.arange(D)[:, None] * H + np.arange(H)[None]  # [D, 96]
    N = len(wd.turbines)
    cls = np.zeros((N, D, H), np.int64)
    pnorm = np.zeros((N, D, H))
    pq = np.zeros((N, D, H, 3))
    pstd, ood = np.zeros((N, D, H)), np.zeros((N, D, H))
    power = np.full((N, D + 1, H), np.nan)  # day -1 .. D-1 (input quality of the previous day)
    rated = np.zeros(N)
    for pid, a in arrays.items():
        sc = bank.scores[pid]
        cols = [j for j, t in enumerate(wd.turbines) if t["park_id"] == pid]
        for jj, j in enumerate(cols):
            cls[j] = sc["cls"][idx, jj]
            pnorm[j] = sc["probs"][idx, jj, 0]
            pq[j] = sc["pq"][idx, jj]
            pstd[j], ood[j] = sc["pstd"][idx, jj], sc["ood"][idx, jj]
            power[j] = np.concatenate([sc["power"][idx[0] - H, jj][None], sc["power"][idx, jj]])
            rated[j] = sc["rated"][jj]
    labels = wd.labels[idx].transpose(2, 0, 1).astype(np.int64) if wd.labels is not None else None
    return {"turbines": wd.turbines, "cls": cls, "p_normal": pnorm, "pq": pq, "pstd": pstd, "ood": ood,
            "power": power, "rated": rated, "labels": labels, "version": bank.model_version,
            "ood_thr": 3.0, "classes": F.WIND_CLASSES}


def wind_day_calls(w: dict, d: int) -> dict:
    """The daily M6 envelope's scalars (as wattsentry.agent.predictors.WindAdapter builds them) for day d."""
    cls = w["cls"][:, d]
    counts = np.stack([(cls == c).sum(-1) for c in range(len(F.WIND_CLASSES))], -1)
    abn = counts[:, 1:].sum(-1)
    top_i = 1 + np.argmax(counts[:, 1:], -1)
    alarm = abn > 0
    top = np.where(alarm, np.asarray(F.WIND_CLASSES, object)[top_i], "normal")
    score = np.nanmax(np.clip(1 - w["p_normal"][:, d], 0, 1), -1)
    rated = w["rated"]
    q50 = np.clip(w["pq"][:, d, :, 1] * rated[:, None], 0, rated[:, None])
    return {"alarm": alarm.astype(float), "top_class": top, "p_fault": score, "p_top": score,
            "abn_share": abn / H, "energy_q50": np.nansum(q50, -1) * 0.25,
            "member_std": np.nanmean(w["pstd"][:, d], -1) * rated, "ood": np.nanmean(w["ood"][:, d], -1)}


# ============================================================================================ one week
def _day0_ts(start: str, burn_in: int) -> int:
    """Local day 0 starts at 21:00 UTC the day before (the replay's convention, wattsentry.agent.world)."""
    d = datetime.fromisoformat(start).replace(tzinfo=timezone.utc) - timedelta(days=burn_in)
    return int(d.timestamp()) - 3 * 3600


def _ort_threads(n: int) -> None:
    """Limit onnxruntime intra-op threads in a pool worker (sessions are created inside wattsentry.model.predictor)."""
    try:
        import onnxruntime as ort
    except ImportError:
        return
    if getattr(ort.InferenceSession, "_judge_wrapped", False):
        return
    base = ort.InferenceSession

    class _Session(base):  # type: ignore[misc, valid-type]
        _judge_wrapped = True

        def __init__(self, path, sess_options=None, *a, **k):
            if sess_options is None:
                sess_options = ort.SessionOptions()
                sess_options.intra_op_num_threads = n
                sess_options.inter_op_num_threads = 1
            super().__init__(path, sess_options, *a, **k)

    ort.InferenceSession = _Session


def pvlib_band(windows: Path, years=(2022, 2023), n_sites: int = 120) -> tuple[float, float]:
    """P10 / P90 of actual / pvlib (forecast weather, recorded metadata) on healthy daylight steps of the train
    years: the substitute's band (same recipe as wattsentry.eval.baselines.pv.persistence.pvlib_band)."""
    idx = json.loads((windows / "index.json").read_text())
    t0 = pd.Timestamp(idx["t_start"])
    t0 = t0.tz_localize("UTC") if t0.tzinfo is None else t0
    a = (pd.Timestamp(f"{years[0]}-01-01", tz="UTC") - t0) // pd.Timedelta(minutes=15)
    b = (pd.Timestamp(f"{years[-1] + 1}-01-01", tz="UTC") - t0) // pd.Timedelta(minutes=15)
    sites = json.loads((windows / "sites.json").read_text())
    rows = [i for i, s in enumerate(sites) if s.get("split") == "train"][:n_sites]
    ld = lambda n: np.load(windows / idx["arrays"][n]["file"], mmap_mode="r")  # noqa: E731
    pv, pl, fl = ld("pv_ac"), ld("pvlib_rec_fc"), ld("fault")
    step = 7  # every 7th step keeps this fast and unbiased over the day
    p = np.asarray(pl[rows, a:b:step], np.float32)
    e = np.asarray(pv[rows, a:b:step], np.float32)
    f = np.asarray(fl[rows, a:b:step])
    kwp = np.array([float(sites[i].get("kwp_recorded") or 1) for i in rows])[:, None]
    ok = (p > 0.1 * kwp) & np.isfinite(e) & (f == 0)
    r = e[ok] / p[ok]
    lo, hi = np.quantile(r, [0.1, 0.9])
    return float(min(lo, 1.0)), float(max(hi, 1.0))


def _site_rows(sites: list[dict], cfg_sites: str) -> list[int]:
    if cfg_sites == "all":
        return list(range(len(sites)))
    n = int(cfg_sites)
    tr = [i for i, s in enumerate(sites) if s.get("split") == "train"][: max(1, n - n // 3)]
    un = [i for i, s in enumerate(sites) if s.get("split") != "train"][: max(1, n // 3)]
    return sorted(tr + un)


def run_week(spec: WeekSpec, cfg: RowConfig, band: tuple[float, float], threads: int = 1) -> pd.DataFrame:
    """Replay one sampled week (+ burn-in) and return its raw rows (week days only) with targets."""
    _ort_threads(threads)
    from wattsentry.agent.predictors import FleetSlice, ModelBank
    from wattsentry.agent.world import find_windows

    t_start = time.perf_counter()
    win = Path(cfg.windows) if cfg.windows else find_windows()
    models_dir = Path(cfg.models_dir) if cfg.models_dir else Path(__file__).resolve().parents[2] / "models"
    all_sites = json.loads((win / "sites.json").read_text())
    site_idx = _site_rows(all_sites, cfg.sites)
    D = cfg.burn_in + cfg.week_days
    day0 = _day0_ts(spec.start, cfg.burn_in)
    t_from = (day0 // DAY_S - HIST_DAYS) * DAY_S
    t_to = ((day0 + D * DAY_S) // DAY_S + 1) * DAY_S
    src = FleetSlice(win, site_idx, t_from, t_to)
    bank = ModelBank(src, models_dir)
    S = src.n
    sites = src.sites
    s0 = src.step(day0)
    steps = s0 + np.arange(D)[:, None] * H + np.arange(H)[None]  # [D, 96] slice steps of each local day

    def days_of(arr2d: np.ndarray) -> np.ndarray:  # [S, T] -> [S, D, 96]
        return np.asarray(arr2d, np.float32)[:, steps]

    pv = days_of(src.array("pv_ac"))
    load = days_of(src.array("load"))
    valid = days_of(src.array("valid")) > 0 if src.has("valid") else np.ones_like(pv, bool)
    fault = days_of(src.array("fault")).astype(np.int64)
    pvlib = days_of(src.array("pvlib_rec_fc"))
    li = src.loc_idx()
    cloud = np.asarray(src.array("weather_fc")[..., 4], np.float32)[li][:, steps]  # [S, D, 96] forecast cloud %
    # previous local day for persistence / input quality: steps - 96 (history covers it)
    pv_prev = np.asarray(src.array("pv_ac"), np.float32)[:, steps - H]
    load_prev = np.asarray(src.array("load"), np.float32)[:, steps - H]
    valid_prev = (np.asarray(src.array("valid"))[:, steps - H] > 0) if src.has("valid") else np.ones_like(pv, bool)
    # the asset's own reference daily energy: median of the 14 days before local day 0 (unit-jump check)
    def ref_e(key: str) -> np.ndarray:
        hist = np.asarray(src.array(key), np.float32)[:, s0 - 14 * H:s0].reshape(S, 14, H)
        return np.nanmedian(np.nansum(hist, -1) * 0.25, -1)

    pv_ref_e, load_ref_e = ref_e("pv_ac"), ref_e("load")

    # ---- witness predictions on local days -------------------------------------------------------------
    def wq(key: str) -> Optional[np.ndarray]:
        if key not in bank.q:
            return None
        return bank.q[key][:, steps].astype(np.float64)  # [S, D, 96, 3]

    def wday(arr: dict, key: str) -> np.ndarray:  # per-UTC-day stats -> local day (midpoint's UTC day)
        mid = (steps[:, H // 2]) // H
        return arr[key][:, mid].astype(np.float64)

    preds: dict[str, dict[str, Any]] = {}
    for m, key, kind in (("pvnet", "pv_fc", "pv"), ("loadnet", "load", "load")):
        if m in cfg.witnesses and key in bank.q:
            preds[m] = {"q": wq(key), "ood": wday(bank.day_ood, key), "std": wday(bank.day_std, key),
                        "version": bank.versions.get(key, ""), "kind": kind}
    pv_now = wq("pv_now")
    if "sitehealth" in cfg.witnesses and "sitehealth" in bank.members and pv_now is not None:
        probs = np.zeros((S, D, len(F.PV_CLASSES)))
        pstd, ood = np.zeros((S, D)), np.zeros((S, D))
        for d in range(D):
            h = bank.health(list(range(S)), int(s0 + d * H))
            for i in range(S):
                probs[i, d], pstd[i, d], ood[i, d] = h[i]
        th = bank.members["sitehealth"][0].calib.get("thresholds", {})
        preds["sitehealth"] = {"probs": probs, "pstd": pstd, "ood": ood, "version": bank.versions.get("sitehealth", ""),
                               "thr": float(th.get("p_fault") or 0.5), "ood_thr": float(th.get("ood") or 3.0)}
    ood_thr = {}
    for m, key in (("pvnet", "pv_fc"), ("loadnet", "load")):
        if key in bank.members:
            ood_thr[m] = float((bank.members[key][0].calib.get("thresholds") or {}).get("ood") or 3.0)

    # ---- drift --------------------------------------------------------------------------------------------
    dr = spec.drift
    drift_on = np.zeros((S, D), bool)
    if dr is not None and dr.witness in preds:
        days_on = np.arange(D) >= dr.start_day
        assets = _scope_mask(sites, dr.scope)
        drift_on = assets[:, None] & days_on[None]
        if dr.witness in ("pvnet", "loadnet"):
            preds[dr.witness]["q"] = drift_forecast(preds[dr.witness]["q"], dr, days_on, assets)
        else:
            preds[dr.witness]["probs"] = drift_detector(preds[dr.witness]["probs"], dr, days_on, assets)

    # ---- substitutes --------------------------------------------------------------------------------------
    subs = {"pvnet": np.stack([pvlib * band[0], pvlib, pvlib * band[1]], -1).astype(np.float64),
            "loadnet": np.repeat(load_prev[..., None], 3, -1).astype(np.float64)}
    sub_ref = {"pvnet": np.repeat(pv_prev[..., None], 3, -1).astype(np.float64),  # pvlib -> day-ahead persistence
               "loadnet": np.repeat(np.asarray(src.array("load"), np.float32)[:, steps - 7 * H][..., None], 3, -1)}

    # ---- context ------------------------------------------------------------------------------------------
    seen = np.array([s.get("split") == "train" for s in sites])
    batt = np.array([float(s.get("battery_kwh") or 0) > 0 for s in sites])
    kwp = np.array([float(s.get("kwp_recorded") or s.get("kwp_true") or 1.0) for s in sites])
    day_dates = [(datetime.fromtimestamp(day0 + d * DAY_S + 3 * 3600, timezone.utc)).date() for d in range(D)]
    months = np.array([dd.month for dd in day_dates])
    gappy = (~valid | ~np.isfinite(pv)).mean(-1) > 0.10  # [S, D] quality condition of the graded day
    gappy_prev = (~valid_prev | ~np.isfinite(pv_prev)).mean(-1) > 0.10
    daylight = (np.nan_to_num(pvlib) > 0.01)
    sky = np.stack([F.sky_class(cloud[:, d], daylight[:, d]) for d in range(D)], 1)  # [S, D]
    asset_ids = [s["site_id"] for s in sites]

    # ---- grades per witness and day (+ substitutes for their own trust) ----------------------------------
    grades: dict[str, list[dict]] = {m: [] for m in preds}
    sub_grades: dict[str, list[dict]] = {SUBSTITUTE[m]: [] for m in preds if m in SUBSTITUTE}
    for d in range(D):
        pv_mask = (np.nan_to_num(pv[:, d]) > 0.01) | (np.nan_to_num(pv_prev[:, d]) > 0.01)
        for m, p in preds.items():
            if m in ("pvnet", "loadnet"):
                y = (pv if m == "pvnet" else load)[:, d].astype(np.float64)
                mask = (pv_mask if m == "pvnet" else np.ones_like(pv_mask)) & valid[:, d]
                g = grade_forecast(p["q"][:, d], y, mask, subs[m][:, d])
                g["graded"] = g["graded"] & ~gappy[:, d]
                grades[m].append(g)
                sg = grade_forecast(subs[m][:, d], y, mask, sub_ref[m][:, d])
                sg["graded"] = sg["graded"] & ~gappy[:, d]
                sub_grades[SUBSTITUTE[m]].append(sg)
            else:
                ds = F.detector_summary(p["probs"][:, d], p["thr"], p_fault_std=p["pstd"][:, d])
                truth = truth_class(fault[:, d - 1] if d > 0 else np.asarray(src.array("fault"))[:, steps[0] - H])
                g = grade_detector(ds["alarm"] > 0.5, ds["top_class"], truth)
                g.update(ds)
                g["truth"] = truth
                grades[m].append(g)

    # ---- outcome rows of every day, then the ledger over them (causal by observed_at) -----------------------
    ood_hi_of = {m: (p["ood"] > (p.get("ood_thr") or ood_thr.get(m, 3.0))) for m, p in preds.items()}
    cold = spec.cold_day

    extra_versions: dict[str, str] = {}

    def version(m: str, d: int) -> str:  # a wiped ledger = a new model version from the cold day on
        v = preds[m]["version"] if m in preds else extra_versions.get(m) or SUB_VERSION.get(m, m)
        return f"{v}@cold{cold}" if cold is not None and d >= cold else v

    out_parts = []
    for d in range(D):
        for m, p in preds.items():
            g = grades[m][d]
            keys = np.array([bucket_key("site", seen[i], batt[i], months[d], gappy[i, d], ood_hi_of[m][i, d])
                             for i in range(S)], object)
            fam = F.family_of(m)
            if fam == "forecast":
                lag = 1  # telemetry: known the next day
                out_parts.append(_outcome_rows(m, version(m, d), asset_ids, keys, day_dates, d, lag, g,
                                               FORECAST_SCORES, norm_err=True))
                sm = SUBSTITUTE[m]
                out_parts.append(_outcome_rows(sm, version(sm, d), asset_ids, keys, day_dates, d, lag,
                                               sub_grades[sm][d], FORECAST_SCORES))
            else:  # crew confirmation of an alarm ~d+2, residual check of a healthy call over 3 days
                out_parts.append(_outcome_rows(m, version(m, d), asset_ids, keys, day_dates, d, 2, g,
                                               ("alarm_precision",)))
                out_parts.append(_outcome_rows(m, version(m, d), asset_ids, keys, day_dates, d, 3, g,
                                               ("miss_score",)))
    # ---- M6 WindNBM over all parks (turbine-days) ---------------------------------------------------------
    wind = None
    if "windnbm" in cfg.witnesses:
        try:
            wind = wind_week(day0, D, models_dir)
        except Exception as e:  # wind is optional: the PV witnesses still make rows
            print(f"[judge.rows] wind skipped for {spec.start}: {type(e).__name__}: {e}", flush=True)
        if wind is not None and wind["labels"] is None:
            wind = None
    wctx = None
    if wind is not None:
        extra_versions["windnbm"] = wind["version"]
        tur = wind["turbines"]
        N = len(tur)
        w_seen = np.array([t["park_id"] not in WIND_HOLDOUT_PARKS for t in tur])
        w_drift = np.zeros((N, D), bool)
        if dr is not None and dr.witness == "windnbm":
            days_on = np.arange(D) >= dr.start_day
            assets = ~w_seen if dr.scope == "unseen" else np.ones(N, bool)
            w_drift = assets[:, None] & days_on[None]
            wind["cls"] = drift_wind_steps(wind["cls"], dr, days_on, assets, np.random.default_rng(spec.seed))
        calls = [wind_day_calls(wind, d) for d in range(D)]
        w_gappy = np.isnan(wind["power"][:, 1:]).mean(-1) > 0.10  # [N, D] graded day
        w_gappy_prev = np.isnan(wind["power"][:, :-1]).mean(-1) > 0.10
        w_grades = []
        for d in range(D):
            c = calls[d]
            truth = truth_class(wind["labels"][:, d], F.WIND_CLASSES)
            g = grade_detector(c["alarm"] > 0.5, c["top_class"], truth, healthy="normal")
            g["graded"] = ~w_gappy[:, d]
            g["truth"] = truth
            w_grades.append(g)
            keys = np.array([bucket_key("turbine", w_seen[i], False, months[d], w_gappy[i, d],
                                        c["ood"][i] > wind["ood_thr"]) for i in range(N)], object)
            tids = [t["turbine_id"] for t in tur]
            out_parts.append(_outcome_rows("windnbm", version("windnbm", d), tids, keys, day_dates, d, 2, g,
                                           ("alarm_precision",)))
            out_parts.append(_outcome_rows("windnbm", version("windnbm", d), tids, keys, day_dates, d, 3, g,
                                           ("miss_score",)))
        w_keys = np.array([[bucket_key("turbine", w_seen[i], False, months[d], w_gappy_prev[i, d],
                                       calls[d]["ood"][i] > wind["ood_thr"]) for d in range(D)] for i in range(N)],
                          object)
        wctx = dict(N=N, seen=w_seen, drift=w_drift, calls=calls, grades=w_grades, keys=w_keys)
    outcomes = pd.concat(out_parts, ignore_index=True)
    # lookups the week's rows need: (model, version, score, bucket at prediction time) per witness + substitute
    look_keys: dict[str, np.ndarray] = {}
    lookups: set = set()
    for m, p in preds.items():
        fam = F.family_of(m)
        kk = np.array([[bucket_key("site", seen[i], batt[i], months[d], gappy_prev[i, d], ood_hi_of[m][i, d])
                        for d in range(D)] for i in range(S)], object)  # [S, D]
        look_keys[m] = kk
        scores = FORECAST_SCORES if fam == "forecast" else DETECTOR_SCORES
        for d in range(cfg.burn_in - 3, D):
            for k in set(kk[:, d]):
                for sc in scores:
                    lookups.add((m, version(m, d), sc, k))
                if m in SUBSTITUTE:
                    for sc in FORECAST_SCORES:
                        lookups.add((SUBSTITUTE[m], version(SUBSTITUTE[m], d), sc, k))
    if wctx is not None:
        for d in range(cfg.burn_in - 3, D):
            for k in set(wctx["keys"][:, d]):
                for sc in DETECTOR_SCORES:
                    lookups.add(("windnbm", version("windnbm", d), sc, k))
    table = ledger_table(outcomes, sorted(lookups), day_dates, models_dir)
    out_rows = []
    for d in range(cfg.burn_in, D):
        for m, p in preds.items():
            out_rows.append(_week_rows(m, p, d, look_keys[m][:, d], table, version, grades[m][d], subs, spec,
                                       asset_ids, seen, batt, months, sky, kwp, pv_prev, load_prev, valid_prev,
                                       pv_ref_e, load_ref_e, pv_now, ood_thr, drift_on, day_dates))
        if wctx is not None:
            out_rows.append(_wind_rows(d, wind, wctx, table, version, spec, months, day_dates))
    df = pd.concat(out_rows, ignore_index=True) if out_rows else pd.DataFrame()
    df["week"] = f"{spec.year}-{spec.month:02d}"
    df["drift_witness"] = spec.drift.witness if spec.drift else ""
    df["drift_form"] = spec.drift.form if spec.drift else ""
    df["cold_week"] = float(spec.cold_day is not None)
    df["t_week_s"] = time.perf_counter() - t_start
    return df


def _ledger_cols(m: str, fam: str, d: int, keys: list, table: dict, version, alarm: np.ndarray) -> dict:
    """Ledger snapshot columns for rows of day d: what the ledger knew at the start of day d."""
    S = len(keys)
    sub_m = SUBSTITUTE.get(m)
    r: dict = {}
    v = version(m, d)
    scores = FORECAST_SCORES if fam == "forecast" else DETECTOR_SCORES
    nan = np.full(S, np.nan)

    def get(model, ver, sc, col, day=d):
        return np.array([table[(model, ver, sc, k)][col][day] if (model, ver, sc, k) in table else np.nan
                         for k in keys], float)

    t = {sc: get(m, v, sc, "trust") for sc in scores}
    pr = {sc: get(m, v, sc, "prior") for sc in scores}
    ne = {sc: get(m, v, sc, "n_eff") for sc in scores}
    al = {sc: get(m, v, sc, "health_alert") for sc in scores}
    if fam == "forecast":
        main = (t["calibration"] + t["sharpness"]) / 2
        prior = (pr["calibration"] + pr["sharpness"]) / 2
        n_eff = np.minimum(ne["calibration"], ne["sharpness"])
        main3 = (get(m, version(m, d - 3), "calibration", "trust", d - 3) +
                 get(m, version(m, d - 3), "sharpness", "trust", d - 3)) / 2
        r["trust_cal"], r["trust_sharp"] = t["calibration"], t["sharpness"]
        r["band_scale"] = get(m, v, "calibration", "band_scale")
        st = [get(sub_m, version(sub_m, d), sc, "trust") for sc in FORECAST_SCORES]
        r["sub_trust_main"] = (st[0] + st[1]) / 2
        r["sub_n_eff"] = np.minimum(*[get(sub_m, version(sub_m, d), sc, "n_eff") for sc in FORECAST_SCORES])
    else:
        main = np.where(alarm, t["alarm_precision"], t["miss_score"])
        prior = np.where(alarm, pr["alarm_precision"], pr["miss_score"])
        n_eff = np.where(alarm, ne["alarm_precision"], ne["miss_score"])
        main3 = np.where(alarm, get(m, version(m, d - 3), "alarm_precision", "trust", d - 3),
                         get(m, version(m, d - 3), "miss_score", "trust", d - 3))
        r["trust_ap"], r["trust_ms"] = t["alarm_precision"], t["miss_score"]
        r["band_scale"] = nan
    r["trust_main"], r["prior_main"], r["n_eff"] = main, prior, n_eff
    r["trust_trend3"] = main - main3
    r["health_alert"] = np.nanmax(np.stack([al[sc] for sc in scores]), 0)
    r["trust_lo80"] = lo80(main, n_eff)
    return r


def _wind_rows(d: int, wind: dict, ctx: dict, table: dict, version, spec: WeekSpec, months, day_dates) -> pd.DataFrame:
    N = ctx["N"]
    c, g = ctx["calls"][d], ctx["grades"][d]
    tur = wind["turbines"]
    keys = list(ctx["keys"][:, d])
    y_prev = wind["power"][:, d]
    hist_e = np.nanmedian(np.nansum(np.nan_to_num(wind["power"][:, :d + 1]), -1) * 0.25, -1) if d > 0 else None
    q = F.input_quality(y_prev, np.isfinite(y_prev), hist_e)
    parks = np.array([t["park_id"] for t in tur])
    top = np.asarray(c["top_class"], object)
    consensus = np.array([np.mean(top[parks == parks[i]] == top[i]) for i in range(N)])
    r = {"model": ["windnbm"] * N, "model_version": [wind["version"]] * N, "asset_id": [t["turbine_id"] for t in tur],
         "day": [day_dates[d].isoformat()] * N, "kind": ["turbine"] * N, "family": ["detector"] * N, "bucket": keys,
         "battery": np.zeros(N), "seen": ctx["seen"].astype(float), "month": np.full(N, float(months[d])),
         "sky": np.full(N, np.nan), "ood_score": c["ood"], "ood_thr": np.full(N, wind["ood_thr"]),
         "is_fallback": np.zeros(N), "telemetry_age_h": np.zeros(N), "member_std": c["member_std"],
         "energy_q50": c["energy_q50"], "energy_scale": wind["rated"] * 24 * 0.35, "alarm": c["alarm"],
         "p_fault": c["p_fault"], "p_fault_std": np.full(N, np.nan), "p_top": c["p_top"],
         "top_class": ["wind:" + str(x) for x in top], "abn_share": c["abn_share"], "park_consensus": consensus,
         "resid_yday": np.full(N, np.nan), **q}
    r.update(_ledger_cols("windnbm", "detector", d, keys, table, version, c["alarm"] > 0.5))
    r["y_alarm_precision"], r["y_miss_score"], r["y_truth"] = g["alarm_precision"], g["miss_score"], g["truth"]
    r["valid"] = g["valid"]
    r["graded"] = np.asarray(g["graded"], float)
    r["drifted"] = ctx["drift"][:, d].astype(float)
    r["year"] = np.full(N, spec.year)
    return pd.DataFrame(r)


def _week_rows(m, p, d, keys, table, version, g, subs, spec, asset_ids, seen, batt, months, sky, kwp, pv_prev,
               load_prev, valid_prev, pv_ref_e, load_ref_e, pv_now, ood_thr, drift_on, day_dates) -> pd.DataFrame:
    S = len(asset_ids)
    fam = F.family_of(m)
    sub_m = SUBSTITUTE.get(m)
    keys = list(keys)
    r: dict[str, Any] = {"model": [m] * S, "model_version": [p["version"]] * S, "asset_id": asset_ids,
                         "day": [day_dates[d].isoformat()] * S, "kind": ["site"] * S, "family": [fam] * S,
                         "bucket": keys, "battery": batt.astype(float), "seen": seen.astype(float),
                         "month": np.full(S, float(months[d])), "sky": sky[:, d], "ood_score": p["ood"][:, d],
                         "is_fallback": np.zeros(S), "telemetry_age_h": np.zeros(S)}
    r["ood_thr"] = np.full(S, p.get("ood_thr") or ood_thr.get(m, 3.0))
    y_prev = pv_prev[:, d] if m in ("pvnet", "sitehealth") else load_prev[:, d]
    eref = pv_ref_e if m in ("pvnet", "sitehealth") else load_ref_e
    q = F.input_quality(y_prev, valid_prev[:, d], eref)
    r.update(q)
    last_valid = np.where(valid_prev[:, d].any(-1), H - 1 - np.argmax(valid_prev[:, d][:, ::-1], -1), H)
    r["telemetry_age_h"] = last_valid * 0.25
    if fam == "forecast":
        s = F.summarize_forecast(p["q"][:, d, :, 0], p["q"][:, d, :, 1], p["q"][:, d, :, 2], p["kind"])
        r["energy_q50"], r["band_rel"] = s["energy_q50"], s["band_rel"]
        r["member_std"] = p["std"][:, d]
        r["energy_scale"] = kwp * 5.0 if m == "pvnet" else np.maximum(eref, 0.1)  # ~5 full-sun hours per kWp
        ss = F.summarize_forecast(subs[m][:, d, :, 0], subs[m][:, d, :, 1], subs[m][:, d, :, 2], p["kind"])
        r["sub_model"] = [sub_m] * S
        r["sub_energy"], r["sub_band_rel"] = ss["energy_q50"], ss["band_rel"]
        for k_ in ("hit", "calibration", "sharpness", "skill_raw") + tuple(f"hit_m{x:g}" for x in MULTS):
            r[f"y_{k_}"] = g[k_]
    else:
        for k_ in ("alarm", "p_fault", "p_fault_std", "p_top", "top_class"):
            r[k_] = g[k_]
        r["member_std"] = p["pstd"][:, d]
        r["energy_scale"] = kwp * 5.0
        if pv_now is not None and d > 0:
            exp_y = np.nansum(pv_now[:, d - 1, :, 1], -1) * 0.25
            meas_y = np.nansum(np.nan_to_num(pv_prev[:, d]), -1) * 0.25
            r["resid_yday"] = np.where(exp_y > 0.2, meas_y / np.maximum(exp_y, 0.2) - 1, np.nan)
        r["y_alarm_precision"], r["y_miss_score"] = g["alarm_precision"], g["miss_score"]
        r["y_truth"] = g["truth"]
    r.update(_ledger_cols(m, fam, d, keys, table, version, np.asarray(r.get("alarm", np.zeros(S)), float) > 0.5))
    r.update(_history_cols(p, fam, d))
    r["valid"] = g["valid"]
    r["graded"] = g["graded"].astype(float)
    r["drifted"] = drift_on[:, d].astype(float) if spec.drift and spec.drift.witness == m else np.zeros(S)
    r["year"] = np.full(S, spec.year)
    return pd.DataFrame(r)


def _history_cols(p: dict, fam: str, d: int) -> dict[str, np.ndarray]:
    """The witness's own outputs on earlier days of the replay (burn-in included, drift included): the hist_* columns.
    Per-day summaries are computed once per witness and kept on `p`."""
    if "_daily" not in p:
        if fam == "forecast":
            s = F.summarize_forecast(p["q"][..., 0], p["q"][..., 1], p["q"][..., 2], p["kind"])  # [S, D]
            p["_daily"] = {"energy_q50": s["energy_q50"], "band_rel": s["band_rel"]}
        else:
            pf = 1.0 - p["probs"][..., 0]
            p["_daily"] = {"p_fault": pf, "alarm": (pf >= p["thr"]).astype(float)}
    out = F.history_features(p["_daily"], d)
    cur = p["q"][:, :, :, 1] if fam == "forecast" else p["probs"]
    out["repeat_prev"] = F.repeats(cur[:, d], cur[:, d - 1]) if d > 0 else np.zeros(cur.shape[0])
    return out


SUB_VERSION = {"pvlib": "pvlib-recorded", "persistence_load": "persistence-d1"}


def _outcome_rows(model: str, ver: str, asset_ids: list, keys: np.ndarray, day_dates: list, d: int, lag: int,
                  g: dict, scores: Sequence[str], norm_err: bool = False) -> pd.DataFrame:
    """Outcome rows of one witness and graded day (gated = ungraded asset-days, kept as gated=1)."""
    graded = np.asarray(g["graded"], bool)
    day, obs = day_dates[d].isoformat(), (day_dates[0] + timedelta(days=d + lag)).isoformat()
    parts = []
    for sc in scores:
        val = np.asarray(g[sc], float)
        keep = np.isfinite(val) | ~graded
        parts.append(pd.DataFrame({"model": model, "model_version": ver, "asset_id": np.asarray(asset_ids)[keep],
                                   "day": day, "score_name": sc, "score": np.where(graded, val, np.nan)[keep],
                                   "weight": 1.0, "observed_at": obs, "gated": (~graded[keep]).astype(int),
                                   "bucket": keys[keep]}))
    if norm_err and "e_q" in g:
        e10 = vigintiles_from(g, 10)
        ok = graded & np.isfinite(e10).all(1)
        if ok.any():
            parts.append(pd.DataFrame({"model": model, "model_version": ver,
                                       "asset_id": np.repeat(np.asarray(asset_ids)[ok], 10), "day": day,
                                       "score_name": "norm_err", "score": e10[ok].ravel(), "weight": 0.1,
                                       "observed_at": obs, "gated": 0, "bucket": np.repeat(keys[ok], 10)}))
    return pd.concat(parts, ignore_index=True)


def vigintiles_from(g: dict, k: int) -> np.ndarray:
    """k quantiles of the day's normalised errors from the 20 stored vigintiles (nearest stored level)."""
    e = np.asarray(g["e_q"], float)
    idx = np.clip(np.round((np.arange(k) + 0.5) / k * 20 - 0.5).astype(int), 0, e.shape[1] - 1)
    return e[:, idx]


def ledger_table(outcomes: pd.DataFrame, lookups: Sequence[tuple], day_dates: list, models_dir: Path) -> dict:
    """{(model, version, score, bucket): {trust, prior, n_eff, health_alert, band_scale: arrays over day index}} from
    the live `judge.ledger`, so training sees the same numbers as the loop."""
    import polars as pl

    from wattsentry.judge.ledger import Ledger, Priors

    D = len(day_dates)
    out: dict = {}
    L = Ledger(pl.from_pandas(outcomes), Priors.from_dir(models_dir), start=day_dates[0], end=day_dates[-1])
    fr = L.frame(extra=list(lookups)).to_pandas()
    fr["di"] = (pd.to_datetime(fr["day"]) - pd.Timestamp(day_dates[0])).dt.days
    want = set(lookups)
    for key, g in fr.groupby(["model", "model_version", "score_name", "bucket"], sort=False):
        if key not in want:
            continue
        rec = {}
        for f in ("trust", "prior", "n_eff", "health_alert", "band_scale"):
            a = np.full(D, np.nan)
            a[g["di"].to_numpy()] = g[f].to_numpy(float)
            rec[f] = a
        out[key] = rec
    return out


# ============================================================================================ pool + cache
def _hash_file(p: Path, h) -> None:
    if p.exists():
        h.update(p.name.encode())
        h.update(p.read_bytes())


def cache_key(cfg: RowConfig, weeks: Sequence[WeekSpec], windows: Path, models_dir: Path,
              band: tuple[float, float]) -> str:
    """Code + config key: the row-generation sources, the witness weights, the data version, the config and the
    sampled weeks. Unrelated commits keep the cache; any change here misses."""
    h = hashlib.sha256()
    here = Path(__file__).resolve().parent
    for name in ROW_FILES:
        _hash_file(here / name, h)
    pred = here.parent / "agent" / "predictors.py"
    _hash_file(pred, h)
    for p in sorted(Path(models_dir).glob("*.onnx")) + sorted(Path(models_dir).glob("*.calibration.json")) + \
            sorted((Path(models_dir) / "pvnet_fc").glob("*")):
        h.update(p.name.encode())
        h.update(hashlib.sha256(p.read_bytes()).digest())
    idx = json.loads((Path(windows) / "index.json").read_text())
    h.update(json.dumps({"windows": idx.get("source_key"), "v": idx.get("data_version"), "cfg": cfg.key_dict(),
                         "weeks": [dataclasses.asdict(w) for w in weeks], "band": band}, sort_keys=True,
                        default=str).encode())
    return h.hexdigest()[:16]


def cache_dir() -> Path:
    base = os.environ.get("LAB_CACHE_DIR")
    return (Path(base) if base else Path.home() / ".cache" / "wattsentry") / "judge_rows"


def _mem_available_gb() -> float:
    """MemAvailable from /proc/meminfo (Linux); large when unknown. See WEEK_GB."""
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / 1024 ** 2
    except OSError:
        pass
    return 1e9


WEEK_GB = 4.0  # memory budget per week process (~1.5 GB peak on macOS; Linux + ONNX arenas run higher)
MAX_WORKERS = 6  # 11 week processes next to another run OOM-killed a 31 GB worker


def _peak_mb() -> float:
    import resource
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / 2 ** 20 if sys.platform == "darwin" else r / 1024  # bytes on macOS, KiB on Linux


def _week_job(args):
    spec, cfg, band, threads = args
    df = run_week(spec, cfg, band, threads)
    return df, _peak_mb()


def generate(cfg: RowConfig, workers: int = 0, use_cache: bool = True, log=print) -> pd.DataFrame:
    """All rows of `cfg.year`'s sampled weeks; weeks run in a process pool; cached by `cache_key`."""
    from wattsentry.agent.world import find_windows

    win = Path(cfg.windows) if cfg.windows else find_windows()
    if win is None:
        raise FileNotFoundError("no window cache; build it with python -m wattsentry.data.windows")
    cfg.windows = str(win)
    models_dir = Path(cfg.models_dir) if cfg.models_dir else Path(__file__).resolve().parents[2] / "models"
    cfg.models_dir = str(models_dir)
    weeks = sample_weeks(cfg.year, cfg.n_weeks, cfg.seed, cfg.drift_share, cfg.cold_share, cfg.week_days,
                         cfg.burn_in, cfg.witnesses)
    band = pvlib_band(win)
    key = cache_key(cfg, weeks, win, models_dir, band)
    path = cache_dir() / f"rows-{cfg.year}-{key}.parquet"
    if use_cache and path.exists():
        log(f"[judge.rows] cached: {path}")
        return pd.read_parquet(path)
    workers = workers or min(len(weeks), os.cpu_count() or 1, MAX_WORKERS,
                             max(1, int(0.6 * _mem_available_gb() // WEEK_GB)))
    threads = max(1, (os.cpu_count() or 1) // max(workers, 1))
    log(f"[judge.rows] {cfg.year}: {len(weeks)} weeks, {workers} processes x {threads} onnx threads, pvlib band "
        f"{band[0]:.2f}-{band[1]:.2f}, key {key}")
    t = time.perf_counter()
    jobs = [(w, cfg, band, threads) for w in weeks]
    if workers <= 1:
        parts = [_week_job(j) for j in jobs]
    else:
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor
        ctx = mp.get_context("spawn")  # onnxruntime + forked threads do not mix
        with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as ex:
            parts = list(ex.map(_week_job, jobs))
    peaks = [pk for _, pk in parts]
    parts = [p for p, _ in parts]
    for w, p, pk in zip(weeks, parts, peaks):
        dr = f"drift {w.drift.witness}/{w.drift.form}/{w.drift.scope} from day {w.drift.start_day}" if w.drift else ""
        log(f"[judge.rows]   {w.start}: {len(p)} rows in {p['t_week_s'].iloc[0] if len(p) else 0:.1f}s "
            f"{dr} {'cold day ' + str(w.cold_day) if w.cold_day is not None else ''} peak {pk:.0f} MB")
    df = pd.concat(parts, ignore_index=True)
    df.attrs["week_peak_mb"] = float(max(peaks)) if peaks else 0.0
    log(f"[judge.rows] {cfg.year}: {len(df)} rows in {time.perf_counter() - t:.1f}s")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(f".tmp{os.getpid()}")
        df.to_parquet(tmp, index=False)
        tmp.replace(path)
    except OSError as e:
        log(f"[judge.rows] cache write failed ({e})")
    return df
