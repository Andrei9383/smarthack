"""WindMon's four anomaly rules on the PPC schema (numpy), the M6 baseline.

Rules, applied per turbine per 15-min step:
- HIGH_WIND_ZERO_POWER: nacelle wind > 15 km/h and act_pwt < 10 kW (WindMon's exact rule);
- TEMPERATURE_SPIKE: any component temperature above a fleet-wide fixed limit, or a rise > 10 °C in one step;
- DATA_GAP: the reading is missing (null);
- SUSPECT_DATA: integral error code, physically impossible value, or a frozen sensor (same wd_spd / non-zero
  act_pwt for >= 2 h).
Rules know nothing about curtailment, icing or the turbine's own baseline: that is the alert-fatigue point.
"""

from __future__ import annotations

import numpy as np

from wattsentry.model.wind.data import AMBIENT, CLASS_INDEX, POWER, TEMP_SENSORS, WIND

RULES = ("HIGH_WIND_ZERO_POWER", "TEMPERATURE_SPIKE", "DATA_GAP", "SUSPECT_DATA")
# What each rule claims, mapped onto WindAnomaly (for the confusion matrix). DATA_GAP/SUSPECT = "gap or suspect data".
RULE_CLASS = {
    "HIGH_WIND_ZERO_POWER": "fault_stop",
    "TEMPERATURE_SPIKE": "overheating",
    "DATA_GAP": "data_gap",
    "SUSPECT_DATA": "data_gap",
}
RULE_PRIORITY = ("DATA_GAP", "SUSPECT_DATA", "HIGH_WIND_ZERO_POWER", "TEMPERATURE_SPIKE")  # first wins

# Fleet-wide limits (°C): one number for every turbine, WindMon style.
TEMP_LIMITS = {"conv_t": 60.0, "gearbox_t": 75.0, "gen1_t": 75.0, "gen2_t": 75.0, "transformer_t": 80.0}
TEMP_JUMP_C = 10.0
FROZEN_STEPS = 8


def _frozen(x: np.ndarray, n: int) -> np.ndarray:
    """True where the value equals the previous n-1 values (finite)."""
    same = np.zeros_like(x, dtype=bool)
    same[1:] = (x[1:] == x[:-1]) & np.isfinite(x[1:])
    # the run of `same` ending at t is >= n-1 long  <=>  same[t-n+2 .. t] are all True (same[0] is False, so a window
    # reaching back to step 0 fails); a rolling AND, vectorised over time too (a per-step loop took ~1-2 s)
    if n - 1 <= 0:
        return np.ones(x.shape, dtype=bool)
    out = same.copy()
    for s in range(1, n - 1):
        out[s:] &= same[:-s]
        out[:s] = False
    return out


def windmon_flags(sensors: dict[str, np.ndarray], err: np.ndarray, rated_kw: np.ndarray) -> dict[str, np.ndarray]:
    """Per-rule boolean flags [T, N]."""
    p = sensors[POWER]
    w = sensors[WIND]
    hwzp = (w > 15.0) & (p < 10.0)
    ts = np.zeros(p.shape, dtype=bool)
    for k in TEMP_SENSORS:
        x = sensors[k]
        ts |= x > TEMP_LIMITS[k]
        jump = np.zeros_like(ts)
        jump[1:] = (x[1:] - x[:-1]) > TEMP_JUMP_C
        ts |= jump
    coded = err != 0
    gap = np.isnan(p) & ~coded
    rated = rated_kw[None, :]
    bad = (p > 1.1 * rated) | (p < -0.1 * rated) | (w < 0) | (w > 150)
    for k in TEMP_SENSORS + [AMBIENT]:
        bad |= (sensors[k] < -40) | (sensors[k] > 150)
    frozen = _frozen(w, FROZEN_STEPS) | (_frozen(p, FROZEN_STEPS) & (np.abs(np.nan_to_num(p)) > 1.0))
    return {
        "HIGH_WIND_ZERO_POWER": hwzp,
        "TEMPERATURE_SPIKE": ts,
        "DATA_GAP": gap,
        "SUSPECT_DATA": coded | bad | frozen,
    }


def windmon_classes(flags: dict[str, np.ndarray]) -> np.ndarray:
    """Step-level WindAnomaly index [T, N] implied by the rules (0 = no rule fired)."""
    shape = next(iter(flags.values())).shape
    out = np.zeros(shape, dtype=np.uint8)
    for r in reversed(RULE_PRIORITY):  # highest priority written last
        out[flags[r]] = CLASS_INDEX[RULE_CLASS[r]]
    return out


def raw_alert_count(flags: dict[str, np.ndarray]) -> int:
    """WindMon-style deduplicated anomaly rows: one per (turbine, rule) contiguous run."""
    n = 0
    for f in flags.values():
        d = np.diff(np.concatenate([np.zeros((1, f.shape[1]), bool), f], 0).astype(np.int8), axis=0)
        n += int((d == 1).sum())
    return n


def anomalies_table(flags: dict[str, np.ndarray], ts: np.ndarray, turbine_ids: list[str]):
    """WindMon's deduplicated anomalies table (polars): one row per (turbine, rule) contiguous run.

    Columns: turbine_id, rule, wind_anomaly (RULE_CLASS), start, end (exclusive, UTC), steps."""
    import polars as pl

    ts = np.asarray(ts).astype("datetime64[us]")
    ts = np.append(ts, ts[-1] + np.timedelta64(15, "m"))
    cols = {"turbine_id": [], "rule": [], "start": [], "end": []}
    for r, f in flags.items():
        z = np.zeros((1, f.shape[1]), np.int8)
        d = np.diff(np.concatenate([z, f.astype(np.int8), z]), axis=0).T  # [N, T+1], runs paired in order
        j, s = np.nonzero(d == 1)
        e = np.nonzero(d == -1)[1]
        cols["turbine_id"] += [turbine_ids[k] for k in j]
        cols["rule"] += [r] * len(s)
        cols["start"].append(ts[s])
        cols["end"].append(ts[e])
    start, end = np.concatenate(cols["start"]), np.concatenate(cols["end"])
    return pl.DataFrame({"turbine_id": cols["turbine_id"], "rule": cols["rule"],
                         "wind_anomaly": [RULE_CLASS[r] for r in cols["rule"]], "start": start, "end": end,
                         "steps": ((end - start) // np.timedelta64(15, "m")).astype(np.int64)}).with_columns(
        pl.col("start").dt.replace_time_zone("UTC"), pl.col("end").dt.replace_time_zone("UTC"),
    ).sort(["turbine_id", "start", "rule"])
