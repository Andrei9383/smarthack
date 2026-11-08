"""WindMon's data-quality layer on the PPC wind schema, rewritten on polars (plan §10, card #22).

WindMon (Andrei's PPC HackITall project) tags every turbine reading OK / DEGRADED / SUSPECT before its anomaly
rules run. Its six rules are not spelled out in this repo, so they are defined here (WindMon's names where known)
and applied per turbine and 15-min step on the wide view (one row per `ts, turbine_id`, one column per sensor):

| rule | fires when | flag |
|---|---|---|
| MISSING | a sensor is null (communication gap) | SUSPECT on act_pwt / wd_spd, else DEGRADED |
| ERROR_CODE | a sensor carries an integral PPC error code (`is_code`, see wattsentry.contracts.wind) | SUSPECT |
| OUT_OF_RANGE | physically impossible value: act_pwt outside [-10 %, +110 %] of rated, wd_spd outside [0, 150] km/h, component temps outside [-40, 150] °C, ambient outside [-40, 55] °C | SUSPECT on act_pwt / wd_spd, else DEGRADED |
| FROZEN | the same value for >= 8 consecutive steps (2 h): wd_spd and temps always, act_pwt only when abs(value) > 1 kW (a parked turbine may report a constant 0) | SUSPECT on act_pwt / wd_spd, else DEGRADED |
| POWER_WITHOUT_WIND | act_pwt > 10 % of rated while the nacelle wind is < 2.5 m/s (9 km/h): one of the two is wrong | SUSPECT |
| WIND_WITHOUT_POWER | nacelle wind 15-25 m/s (54-90 km/h, above rated, below cut-out) and abs(act_pwt) < 1 % of rated. A stopped or curtailed turbine looks like this too, so the row is usable but flagged | DEGRADED |

Flag per row: SUSPECT if any SUSPECT-level rule fires, else DEGRADED if any rule fires, else OK. SUSPECT rows are
what M6 / the qualifier must not trust (plan §3.5: sensor fault / data gap → excluded); DEGRADED rows are usable
with care. Error-code values never enter the sensor columns (they are nulled and counted by ERROR_CODE).

Entry points: `wide_from_arrays(WindData)` (simulator / M6 view), `apply_rules(wide, turbines)` → adds `q_<rule>`
booleans + `quality`, `flag_counts(flagged)`.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from wattsentry.contracts.wind import SENSORS

POWER, WIND, AMBIENT = "act_pwt", "wd_spd", "turbine_t"
TEMPS = ["conv_t", "gearbox_t", "gen1_t", "gen2_t", "transformer_t"]
CORE = (POWER, WIND)  # sensors whose loss makes the row SUSPECT

RULES = ("MISSING", "ERROR_CODE", "OUT_OF_RANGE", "FROZEN", "POWER_WITHOUT_WIND", "WIND_WITHOUT_POWER")
FLAGS = ("OK", "DEGRADED", "SUSPECT")

POWER_RANGE_FRAC = (-0.10, 1.10)  # of rated
RANGES = {WIND: (0.0, 150.0), AMBIENT: (-40.0, 55.0)} | {t: (-40.0, 150.0) for t in TEMPS}
FROZEN_STEPS = 8
FROZEN_POWER_MIN_KW = 1.0
PWW_POWER_FRAC, PWW_WIND_KMH = 0.10, 9.0
WWP_WIND_KMH, WWP_CUTOUT_KMH, WWP_POWER_FRAC = 54.0, 90.0, 0.01


def _code_col(s: str) -> str:
    return f"{s}__code"


def wide_from_arrays(data, turbine_idx=None) -> pl.DataFrame:
    """wattsentry.model.wind.data.WindData (sensors [T, N], err [T, N]) → the same wide frame. An error code in
    `err` marks every sensor of that step as coded (that is how #23 writes them)."""
    ti = np.arange(len(data.turbines)) if turbine_idx is None else np.asarray(turbine_idx)
    T = len(data.ts)
    ts = np.tile(data.ts.astype("datetime64[us]"), len(ti))
    tid = np.repeat(np.array([data.turbines[i]["turbine_id"] for i in ti]), T)
    cols = {"ts": ts, "turbine_id": tid}
    coded = (data.err[:, ti] != 0).T.reshape(-1)
    for s in SENSORS:
        v = data.sensors[s][:, ti].T.reshape(-1).astype(np.float64)
        cols[s] = np.where(coded, np.nan, v)
    df = pl.DataFrame(cols).with_columns(pl.col("ts").dt.replace_time_zone("UTC"))
    df = df.with_columns([pl.when(pl.col(s).is_nan()).then(None).otherwise(pl.col(s)).alias(s) for s in SENSORS])
    return df.with_columns([pl.Series(_code_col(s), coded) for s in SENSORS])


def apply_rules(wide: pl.DataFrame, turbines: pl.DataFrame | None = None, rated_kw: float | None = None) -> pl.DataFrame:
    """Add `q_missing … q_wind_without_power` (bool), `q_suspect_rules` / `q_degraded_rules` (counts) and
    `quality` (OK / DEGRADED / SUSPECT). Rated power comes from `turbines.rated_kw` (else `rated_kw`)."""
    df = wide.sort(["turbine_id", "ts"])
    if turbines is not None:
        df = df.join(turbines.select("turbine_id", pl.col("rated_kw").cast(pl.Float64).alias("_rated")),
                     on="turbine_id", how="left")
    else:
        df = df.with_columns(pl.lit(float(rated_kw or np.nan)).alias("_rated"))
    present = [s for s in SENSORS if s in df.columns]

    # FROZEN needs run lengths per turbine: rle id, then run size
    run_cols = [f"_rid_{s}" for s in present]
    df = df.with_columns([pl.col(s).rle_id().over("turbine_id").alias(f"_rid_{s}") for s in present])
    df = df.with_columns([pl.len().over(["turbine_id", f"_rid_{s}"]).alias(f"_rlen_{s}") for s in present])

    def missing(s):
        return pl.col(s).is_null() & ~pl.col(_code_col(s))

    def oor(s):
        x = pl.col(s)
        if s == POWER:
            lo, hi = POWER_RANGE_FRAC
            return (x < lo * pl.col("_rated")) | (x > hi * pl.col("_rated"))
        lo, hi = RANGES[s]
        return (x < lo) | (x > hi)

    def frozen(s):
        f = pl.col(s).is_not_null() & (pl.col(f"_rlen_{s}") >= FROZEN_STEPS)
        if s == POWER:
            f = f & (pl.col(s).abs() > FROZEN_POWER_MIN_KW)
        return f

    def any_of(exprs):
        exprs = list(exprs)
        return pl.any_horizontal(exprs).fill_null(False) if exprs else pl.lit(False)

    core = [s for s in present if s in CORE]
    other = [s for s in present if s not in CORE]
    p, w = pl.col(POWER), pl.col(WIND)
    rules = {
        "q_missing": any_of(missing(s) for s in present),
        "q_error_code": any_of(pl.col(_code_col(s)) for s in present),
        "q_out_of_range": any_of(oor(s) for s in present),
        "q_frozen": any_of(frozen(s) for s in present),
        "q_power_without_wind": ((p > PWW_POWER_FRAC * pl.col("_rated")) & (w < PWW_WIND_KMH)).fill_null(False),
        "q_wind_without_power": ((w >= WWP_WIND_KMH) & (w < WWP_CUTOUT_KMH)
                                 & (p.abs() < WWP_POWER_FRAC * pl.col("_rated"))).fill_null(False),
    }
    suspect = [
        rules["q_error_code"], rules["q_power_without_wind"],
        any_of(missing(s) for s in core), any_of(oor(s) for s in core), any_of(frozen(s) for s in core),
    ]
    degraded = [
        any_of(missing(s) for s in other), any_of(oor(s) for s in other), any_of(frozen(s) for s in other),
        rules["q_wind_without_power"],
    ]
    df = df.with_columns(**rules).with_columns(
        pl.sum_horizontal([e.cast(pl.Int8) for e in suspect]).alias("q_suspect_rules"),
        pl.sum_horizontal([e.cast(pl.Int8) for e in degraded]).alias("q_degraded_rules"),
    )
    df = df.with_columns(
        pl.when(pl.col("q_suspect_rules") > 0).then(pl.lit("SUSPECT"))
        .when(pl.col("q_degraded_rules") > 0).then(pl.lit("DEGRADED"))
        .otherwise(pl.lit("OK")).alias("quality")
    )
    drop = run_cols + [f"_rlen_{s}" for s in present] + ["_rated"]
    return df.drop(drop)


def flag_counts(flagged: pl.DataFrame) -> dict:
    """Share of rows per flag and how often each rule fired."""
    n = max(flagged.height, 1)
    out = {f"share_{f.lower()}": round(flagged.filter(pl.col("quality") == f).height / n, 5) for f in FLAGS}
    for r in RULES:
        out[f"rule_{r.lower()}"] = round(float(flagged[f"q_{r.lower()}"].sum()) / n, 5)
    return out
