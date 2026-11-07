"""Wind dataset container, PPC long-format conversion, loader and splits for M6 (numpy + polars, no torch).

`WindData` holds the wide per-turbine view (plan §3.4 / team/contracts/wind.md: "Wide per-turbine view, not a
contract") as float32 arrays [T, N]. Conversion to and from the contract's long table `wind_telemetry`
(`ts, tag, value, is_code`) is lossless, so #22/#23 output drops in through `from_long()`.

Loader order (`load_wind`):
  1. `$SENTINEL_WIND_DIR`, else the newest `~/.cache/sentinel/data-v{N}/wind/` (#22/#23 output) containing
     `wind_telemetry.parquet` + `turbines.parquet` (+ optional `meteo_wind.parquet` = 5-min meteo per TURBINE
     `ts, turbine_id, wind_ms, wdir_deg?`, `weather.parquet` with `point_id` = park id for rh / precip / pressure,
     `fault_labels.parquet`);
  2. otherwise the deterministic stand-in generator `sentinel.model.wind.synth`.
"""

from __future__ import annotations

import glob
import hashlib
import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np

from sentinel.contracts.envelope import WindAnomaly
from sentinel.contracts.wind import CODE_SENSORS, SENSORS, make_tag, parse_tag

CLASSES: list[str] = [a.value for a in WindAnomaly]  # index = label code
CLASS_INDEX = {c: i for i, c in enumerate(CLASSES)}
NORMAL = 0
# Overlap priority when episodes overlap (higher wins).
PRIORITY = {
    "normal": 0,
    "underperformance": 1,
    "overheating": 2,
    "anemometer_fault": 3,
    "icing": 4,
    "curtailment": 5,
    "fault_stop": 6,
    "data_gap": 7,
}
# Classes that should reach an O&M queue (curtailment is logged, never a work order: plan §3.5 Qualify).
ACTIONABLE = [c for c in CLASSES if c not in ("normal", "curtailment")]

CONT_SENSORS = [s for s in SENSORS if s not in CODE_SENSORS]  # continuous sensors (codes = errors)
POWER = "act_pwt"
WIND = "wd_spd"
AMBIENT = "turbine_t"
TEMP_SENSORS = ["conv_t", "gearbox_t", "gen1_t", "gen2_t", "transformer_t"]  # component temps (NBM targets)
WEATHER_FIELDS = ["temp_c", "rh_pct", "precip_mm", "pressure_hpa", "wind100_ms", "wdir100_deg"]
METEO_FIELDS = ["wind_ms", "wdir_deg"]
STEP = np.timedelta64(15, "m")


@dataclass
class WindData:
    """Wide wind dataset. Arrays are [T, N] (turbines) or [T, P] (parks), float32, NaN = gap."""

    ts: np.ndarray  # datetime64[m], UTC, [T], regular 15-min grid
    turbines: list[dict]  # rows of turbines table (TURBINE_META_COLUMNS)
    sensors: dict[str, np.ndarray]  # sensor -> [T, N]; error-code rows are NaN here and coded in `err`
    err: np.ndarray  # int32 [T, N]: integer error code seen on the continuous sensors (0 = none)
    park_ids: list[str]
    weather: dict[str, np.ndarray]  # WEATHER_FIELDS -> [T, P] (park point; rh / precip / pressure, optional)
    meteo: dict[str, np.ndarray]  # METEO_FIELDS -> [T, N], 15-min means of the 5-min per-turbine meteo wind
    labels: Optional[np.ndarray] = None  # uint8 [T, N], index into CLASSES
    episodes: list[dict] = field(default_factory=list)  # fault_labels rows (asset_kind = turbine)
    source: str = "unknown"
    meteo_5min: Optional[dict[str, np.ndarray]] = None  # METEO_FIELDS -> [3T, N] (synth only)

    @property
    def n_steps(self) -> int:
        return len(self.ts)

    @property
    def turbine_ids(self) -> list[str]:
        return [t["turbine_id"] for t in self.turbines]

    @property
    def park_of(self) -> np.ndarray:
        idx = {p: i for i, p in enumerate(self.park_ids)}
        return np.array([idx[t["park_id"]] for t in self.turbines], dtype=np.int64)

    @property
    def rated_kw(self) -> np.ndarray:
        return np.array([float(t["rated_kw"]) for t in self.turbines], dtype=np.float32)

    def subset(self, turbine_idx: Sequence[int] | np.ndarray | None = None, t_slice: slice | None = None) -> "WindData":
        """Copy-free-ish view on some turbines and a time slice (parks are kept whole)."""
        ti = np.arange(len(self.turbines)) if turbine_idx is None else np.asarray(turbine_idx)
        ts_ = t_slice or slice(None)
        m5 = None
        if self.meteo_5min is not None:
            s = ts_.start or 0
            e = ts_.stop if ts_.stop is not None else self.n_steps
            m5 = {k: v[3 * s : 3 * e][:, ti] for k, v in self.meteo_5min.items()}
        return WindData(
            ts=self.ts[ts_],
            turbines=[self.turbines[i] for i in ti],
            sensors={k: v[ts_][:, ti] for k, v in self.sensors.items()},
            err=self.err[ts_][:, ti],
            park_ids=self.park_ids,
            weather={k: v[ts_] for k, v in self.weather.items()},
            meteo={k: v[ts_][:, ti] for k, v in self.meteo.items()},
            labels=None if self.labels is None else self.labels[ts_][:, ti],
            episodes=[e for e in self.episodes if e["asset_id"] in {self.turbines[i]["turbine_id"] for i in ti}],
            source=self.source,
            meteo_5min=m5,
        )

    def fingerprint(self) -> str:
        """Cheap content hash for the manifest."""
        h = hashlib.sha256()
        h.update(self.source.encode())
        h.update(str(self.ts[0]).encode() + str(self.ts[-1]).encode())
        h.update(json.dumps(self.turbine_ids).encode())
        for k in sorted(self.sensors):
            h.update(np.nan_to_num(self.sensors[k][:: max(1, self.n_steps // 500)]).tobytes())
        if self.labels is not None:
            h.update(self.labels[:: max(1, self.n_steps // 500)].tobytes())
        return h.hexdigest()[:16]


# --------------------------------------------------------------------------------------------------------------
# Labels <-> episodes
# --------------------------------------------------------------------------------------------------------------
def _ts_index(ts: np.ndarray, t) -> int:
    t64 = np.datetime64(_to_naive_utc(t), "m")
    return int(np.searchsorted(ts, t64, side="left"))


def _to_naive_utc(t):
    if t is None:
        return None
    if isinstance(t, np.datetime64):
        return t
    if hasattr(t, "tzinfo") and t.tzinfo is not None:
        t = t.astimezone(tz=__import__("datetime").timezone.utc).replace(tzinfo=None)
    if isinstance(t, str):
        t = t.replace("Z", "").replace("+00:00", "")
    return np.datetime64(t, "m")


def labels_from_episodes(episodes: list[dict], ts: np.ndarray, turbine_ids: Sequence[str]) -> np.ndarray:
    """Per-step label array [T, N] from fault_labels rows (whole episode, overlap resolved by PRIORITY)."""
    col = {tid: i for i, tid in enumerate(turbine_ids)}
    lab = np.zeros((len(ts), len(turbine_ids)), dtype=np.uint8)
    pri = np.zeros_like(lab)
    for e in episodes:
        if e["asset_id"] not in col or e["fault"] not in CLASS_INDEX:
            continue
        j = col[e["asset_id"]]
        s = _ts_index(ts, e["onset"])
        en = len(ts) if e.get("end") is None else _ts_index(ts, e["end"])
        if en <= s:
            continue
        c = CLASS_INDEX[e["fault"]]
        p = PRIORITY[e["fault"]]
        seg_p = pri[s:en, j]
        win = seg_p <= p
        lab[s:en, j][win] = c
        seg_p[win] = p
    return lab


def episodes_from_labels(labels: np.ndarray, ts: np.ndarray, turbine_ids: Sequence[str]) -> list[dict]:
    """Contiguous same-class runs → fault_labels rows (used when only step labels exist)."""
    out = []
    for j, tid in enumerate(turbine_ids):
        for c, s, e in runs_by_class(labels[:, j]):
            out.append(
                {
                    "asset_id": tid,
                    "asset_kind": "turbine",
                    "fault": CLASSES[c],
                    "onset": _iso(ts[s]),
                    "end": _iso(ts[e]) if e < len(ts) else None,
                    "severity": 1.0,
                    "params": None,
                }
            )
    return out


def runs_by_class(x: np.ndarray, skip: int = NORMAL):
    """Yield (class, start, end) for contiguous runs of equal non-`skip` values in a 1-D int array."""
    x = np.asarray(x)
    if len(x) == 0:
        return
    change = np.flatnonzero(np.diff(x.astype(np.int64))) + 1
    starts = np.concatenate([[0], change])
    ends = np.concatenate([change, [len(x)]])
    for s, e in zip(starts, ends):
        if x[s] != skip:
            yield int(x[s]), int(s), int(e)


def _iso(t: np.datetime64) -> str:
    return str(np.datetime64(t, "s")) + "Z"


# --------------------------------------------------------------------------------------------------------------
# Long format (team/contracts/wind.md) <-> wide
# --------------------------------------------------------------------------------------------------------------
def _not_integral(v: np.ndarray, sensor: str) -> np.ndarray:
    """Readings must never look like PPC error codes (integral floats): nudge integral readings by 1e-3.

    act_pwt == 0.0 stays (a stopped turbine is a real reading)."""
    integral = np.isfinite(v) & (v == np.round(v))
    if sensor == POWER:
        integral &= v != 0.0
    return np.where(integral, v + 1e-3, v)


def to_long(data: WindData, sensors: Sequence[str] | None = None):
    """Wide → contract `wind_telemetry` long table (polars): ts, tag, value, is_code. Gaps are null values.

    Error codes are written as integral float64 values with is_code = True (PPC convention)."""
    import polars as pl

    sensors = [s for s in (sensors or SENSORS) if s in data.sensors]
    T = len(data.ts)
    ts = data.ts.astype("datetime64[us]")
    frames = []
    for j, t in enumerate(data.turbines):
        tm = parse_tag(f"{t['turbine_id']}.{POWER}")
        scale = float(t.get("power_scale") or 1.0)
        code = data.err[:, j] != 0
        for s in sensors:
            v = data.sensors[s][:, j].astype(np.float64)
            if s == POWER:
                v = v / scale  # back to the turbine's native unit
            v = _not_integral(v, s)
            v = np.where(code, data.err[:, j].astype(np.float64), v)
            frames.append(
                pl.DataFrame(
                    {
                        "ts": ts,
                        "tag": np.full(T, make_tag(tm.park, tm.turbine, s)),
                        "value": v,
                        "is_code": code.copy(),
                    }
                )
            )
    df = pl.concat(frames)
    return df.with_columns(
        pl.col("ts").dt.replace_time_zone("UTC"),
        pl.when(pl.col("value").is_nan()).then(None).otherwise(pl.col("value")).alias("value"),
    )


def turbines_table(data: WindData):
    import polars as pl

    return pl.DataFrame(data.turbines)


def meteo_table(data: WindData):
    """Contract meteo wind: 5-min per TURBINE `ts, turbine_id, wind_ms, wdir_deg` (15-min if no 5-min series)."""
    import polars as pl

    if data.meteo_5min is not None:
        n = 3 * len(data.ts)
        ts = data.ts[0] + np.arange(n) * np.timedelta64(5, "m")
        src = data.meteo_5min
    else:
        ts, src = data.ts, data.meteo
    frames = []
    for j, tid in enumerate(data.turbine_ids):
        frames.append(
            pl.DataFrame(
                {"ts": ts.astype("datetime64[us]"), "turbine_id": [tid] * len(ts)}
                | {k: src[k][:, j].astype(np.float64) for k in METEO_FIELDS}
            )
        )
    return pl.concat(frames).with_columns(pl.col("ts").dt.replace_time_zone("UTC"))


def weather_table(data: WindData):
    import polars as pl

    frames = []
    for p, pid in enumerate(data.park_ids):
        frames.append(
            pl.DataFrame(
                {"ts": data.ts.astype("datetime64[us]"), "point_id": [pid] * len(data.ts)}
                | {k: data.weather[k][:, p].astype(np.float64) for k in WEATHER_FIELDS}
            )
        )
    return pl.concat(frames).with_columns(pl.col("ts").dt.replace_time_zone("UTC"))


def fault_labels_table(data: WindData):
    import polars as pl

    rows = data.episodes or (
        episodes_from_labels(data.labels, data.ts, data.turbine_ids) if data.labels is not None else []
    )
    df = pl.DataFrame(
        rows,
        schema={
            "asset_id": pl.Utf8,
            "asset_kind": pl.Utf8,
            "fault": pl.Utf8,
            "onset": pl.Utf8,
            "end": pl.Utf8,
            "severity": pl.Float32,
            "params": pl.Utf8,
        },
    )
    return df.with_columns(
        pl.col("onset").str.to_datetime(time_zone="UTC"), pl.col("end").str.to_datetime(time_zone="UTC")
    )


def _naive_minutes(col) -> np.ndarray:
    import polars as pl

    s = col
    if isinstance(s.dtype, pl.Datetime) and s.dtype.time_zone is not None:
        s = s.dt.convert_time_zone("UTC").dt.replace_time_zone(None)
    return s.to_numpy().astype("datetime64[m]")


def from_long(
    telemetry,
    turbines,
    meteo=None,
    weather=None,
    fault_labels=None,
    source: str = "external",
) -> WindData:
    """Contract tables (polars DataFrames) → WindData on a regular 15-min grid.

    - telemetry at a finer native rate is averaged into 15-min bins; codes: last code in the bin;
      `is_code` is recomputed with `is_error_code` when the column is missing;
    - act_pwt is multiplied by the turbine's `power_scale` (→ kW);
    - meteo is per turbine (5-min → 15-min mean); without it, hub wind comes from weather wind100 via shear.
    """
    import polars as pl

    from sentinel.contracts.wind import is_error_code

    tel = telemetry
    if "is_code" not in tel.columns:
        sens_ = tel["tag"].str.extract(r"\.([A-Za-z0-9_]+)$", 1).to_list()
        tel = tel.with_columns(
            pl.Series("is_code", [is_error_code(s, v) for s, v in zip(sens_, tel["value"].to_list())])
        )
    tel = tel.with_columns(
        pl.col("tag").str.extract(r"^(WP[^.]+\.TID[^.]+)\.", 1).alias("turbine_id"),
        pl.col("tag").str.extract(r"\.([A-Za-z0-9_]+)$", 1).alias("sensor"),
        pl.col("ts").dt.truncate("15m").alias("ts"),
    )
    tmeta = turbines.to_dicts()
    for t in tmeta:
        t.setdefault("power_scale", 1.0)
        t.setdefault("power_unit", "kW")
    tids = [t["turbine_id"] for t in tmeta]
    parks = sorted({t["park_id"] for t in tmeta})
    ts_all = _naive_minutes(tel["ts"])
    t0, t1 = ts_all.min(), ts_all.max()
    grid = np.arange(t0, t1 + STEP, STEP)
    T, N = len(grid), len(tids)
    col = {tid: j for j, tid in enumerate(tids)}

    sensors = {s: np.full((T, N), np.nan, dtype=np.float32) for s in SENSORS}
    err = np.zeros((T, N), dtype=np.int32)
    vals = tel.filter(~pl.col("is_code").fill_null(False))
    agg = vals.group_by(["ts", "turbine_id", "sensor"]).agg(pl.col("value").mean())
    ti = ((_naive_minutes(agg["ts"]) - t0) // STEP).astype(np.int64)
    tj = np.array([col.get(t, -1) for t in agg["turbine_id"].to_list()], dtype=np.int64)
    sens = agg["sensor"].to_numpy()
    v = agg["value"].to_numpy().astype(np.float64)
    for s in SENSORS:
        m = (sens == s) & (tj >= 0) & np.isfinite(v)
        vs = v[m]
        if s == POWER:
            scale = np.array([float(t["power_scale"] or 1.0) for t in tmeta])
            vs = vs * scale[tj[m]]
        sensors[s][ti[m], tj[m]] = vs
    codes = tel.filter(pl.col("is_code").fill_null(False))
    if codes.height:
        cagg = codes.group_by(["ts", "turbine_id"]).agg(pl.col("value").last())
        ci = ((_naive_minutes(cagg["ts"]) - t0) // STEP).astype(np.int64)
        cj = np.array([col.get(t, -1) for t in cagg["turbine_id"].to_list()], dtype=np.int64)
        m = cj >= 0
        err[ci[m], cj[m]] = cagg["value"].to_numpy().astype(np.int64)[m].astype(np.int32)
        # a code in a bin masks that bin's readings for that turbine (same as a gap for the model)
        for s in SENSORS:
            sensors[s][err != 0] = np.nan

    P = len(parks)
    weather_arr = {k: np.full((T, P), np.nan, dtype=np.float32) for k in WEATHER_FIELDS}
    meteo_arr = {k: np.full((T, N), np.nan, dtype=np.float32) for k in METEO_FIELDS}
    if weather is not None and weather.height:
        w = weather.filter(pl.col("point_id").is_in(parks))
        for p, pid in enumerate(parks):
            wp = w.filter(pl.col("point_id") == pid).sort("ts")
            if not wp.height:
                continue
            wts = _naive_minutes(wp["ts"]).astype(np.int64)
            for k in WEATHER_FIELDS:
                if k in wp.columns:
                    y = wp[k].cast(pl.Float64).to_numpy()
                    ok = np.isfinite(y)
                    if ok.sum() >= 2:  # hourly → 15-min by linear interpolation
                        weather_arr[k][:, p] = np.interp(grid.astype(np.int64), wts[ok], y[ok])
    if meteo is not None and meteo.height:
        has_dir = "wdir_deg" in meteo.columns
        m15 = meteo.with_columns(pl.col("ts").dt.truncate("15m"), pl.col("wind_ms").cast(pl.Float64))
        aggs = [pl.col("wind_ms").mean()]
        if has_dir:
            rad = np.deg2rad(m15["wdir_deg"].cast(pl.Float64).to_numpy())
            m15 = m15.with_columns(pl.Series("_s", np.sin(rad)), pl.Series("_c", np.cos(rad)))
            aggs += [pl.col("_s").mean(), pl.col("_c").mean()]
        agg = m15.group_by(["ts", "turbine_id"]).agg(aggs)
        mi = ((_naive_minutes(agg["ts"]) - t0) // STEP).astype(np.int64)
        mj = np.array([col.get(t, -1) for t in agg["turbine_id"].to_list()], dtype=np.int64)
        ok = (mj >= 0) & (mi >= 0) & (mi < T)
        meteo_arr["wind_ms"][mi[ok], mj[ok]] = agg["wind_ms"].to_numpy()[ok]
        if has_dir:
            d = np.rad2deg(np.arctan2(agg["_s"].to_numpy(), agg["_c"].to_numpy())) % 360
            meteo_arr["wdir_deg"][mi[ok], mj[ok]] = d[ok]
    if np.isnan(meteo_arr["wind_ms"]).all() and weather is not None:
        from .physics import shear_to_hub

        pidx = {p: i for i, p in enumerate(parks)}
        for j, t in enumerate(tmeta):
            p = pidx[t["park_id"]]
            meteo_arr["wind_ms"][:, j] = shear_to_hub(weather_arr["wind100_ms"][:, p], t["hub_height_m"])
    if np.isnan(meteo_arr["wdir_deg"]).all() and weather is not None:
        pidx = {p: i for i, p in enumerate(parks)}
        for j, t in enumerate(tmeta):
            meteo_arr["wdir_deg"][:, j] = weather_arr["wdir100_deg"][:, pidx[t["park_id"]]]

    episodes = []
    labels = None
    if fault_labels is not None and fault_labels.height:
        episodes = fault_labels.filter(pl.col("asset_kind") == "turbine").to_dicts()
        labels = labels_from_episodes(episodes, grid, tids)
    return WindData(
        ts=grid,
        turbines=tmeta,
        sensors=sensors,
        err=err,
        park_ids=parks,
        weather=weather_arr,
        meteo=meteo_arr,
        labels=labels,
        episodes=episodes,
        source=source,
    )


# --------------------------------------------------------------------------------------------------------------
# Loader
# --------------------------------------------------------------------------------------------------------------
def external_wind_dir() -> Optional[Path]:
    """Where #22/#23 write their output: $SENTINEL_WIND_DIR, else the newest ~/.cache/sentinel/data-v{N}/wind/."""
    env = os.environ.get("SENTINEL_WIND_DIR")
    cands = [Path(env)] if env else []
    base = Path(os.environ.get("SENTINEL_CACHE", Path.home() / ".cache" / "sentinel"))
    vers = []
    for d in glob.glob(str(base / "data-v*" / "wind")):
        m = re.search(r"data-v(\d+)", d)
        if m:
            vers.append((int(m.group(1)), Path(d)))
    cands += [d for _, d in sorted(vers, reverse=True)]
    for d in cands:
        if (d / "wind_telemetry.parquet").exists() and (d / "turbines.parquet").exists():
            return d
    return None


def load_wind(prefer_external: bool = True, synth_config: Any = None, log=print) -> WindData:
    """Prefer #22/#23 output when present; otherwise the deterministic stand-in generator."""
    if prefer_external:
        d = external_wind_dir()
        if d is not None:
            import polars as pl

            def opt(name):
                f = d / name
                return pl.read_parquet(f) if f.exists() else None

            log(f"[windnbm] loading external wind data from {d}")
            return from_long(
                pl.read_parquet(d / "wind_telemetry.parquet"),
                pl.read_parquet(d / "turbines.parquet"),
                meteo=opt("meteo_wind.parquet"),
                weather=opt("weather.parquet"),
                fault_labels=opt("fault_labels.parquet"),
                source=f"external:{d}",
            )
    from .synth import SynthConfig, generate

    cfg = synth_config or SynthConfig()
    log(f"[windnbm] no #22/#23 output found; generating stand-in synthetic RO wind ({cfg.describe()})")
    return generate(cfg)


# --------------------------------------------------------------------------------------------------------------
# Splits (plan §4): time split + one whole park held out of training
# --------------------------------------------------------------------------------------------------------------
@dataclass
class Splits:
    train_t: np.ndarray  # bool [T]
    val_t: np.ndarray
    test_t: np.ndarray
    train_turbines: np.ndarray  # int indices (parks != holdout)
    holdout_turbines: np.ndarray  # int indices (holdout park)
    holdout_park: str
    test_year: int

    def describe(self) -> dict:
        return {
            "holdout_park": self.holdout_park,
            "test_year": self.test_year,
            "train_steps": int(self.train_t.sum()),
            "val_steps": int(self.val_t.sum()),
            "test_steps": int(self.test_t.sum()),
            "n_train_turbines": int(len(self.train_turbines)),
            "n_holdout_turbines": int(len(self.holdout_turbines)),
            "rule": "test = last calendar year; earlier years: day-of-month <= 19 train, >= 22 val (2-day buffer)",
        }


def make_splits(data: WindData, holdout_park: Optional[str] = None, val_from_day: int = 22, train_to_day: int = 19) -> Splits:
    years = data.ts.astype("datetime64[Y]").astype(int) + 1970
    test_year = int(years.max())
    if (years < test_year).sum() == 0:  # single-year data: last 25% is test
        cut = int(len(years) * 0.75)
        test_t = np.zeros(len(years), bool)
        test_t[cut:] = True
    else:
        test_t = years == test_year
    dom = (data.ts.astype("datetime64[D]") - data.ts.astype("datetime64[M]")).astype(int) + 1
    rest = ~test_t
    train_t = rest & (dom <= train_to_day)
    val_t = rest & (dom >= val_from_day)
    if holdout_park is None:  # default: the second park (a 2.5 MW Dobrogea park; its class is seen in training)
        holdout_park = data.park_ids[1] if len(data.park_ids) > 1 else data.park_ids[0]
    park = np.array([t["park_id"] for t in data.turbines])
    hold = np.flatnonzero(park == holdout_park)
    train = np.flatnonzero(park != holdout_park)
    if len(train) == 0:
        raise ValueError("holding out the only park leaves nothing to train on")
    return Splits(train_t, val_t, test_t, train, hold, holdout_park, test_year)
