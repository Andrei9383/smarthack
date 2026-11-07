"""WindMon's four anomaly rules on the PPC schema (numpy), the M6 baseline (plan §10, card #24 may move/replace it).

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

from .data import AMBIENT, CLASS_INDEX, POWER, TEMP_SENSORS, WIND

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
    T = x.shape[0]
    same = np.zeros_like(x, dtype=bool)
    same[1:] = (x[1:] == x[:-1]) & np.isfinite(x[1:])
    run = np.zeros(x.shape, dtype=np.int32)
    for t in range(1, T):  # vectorised over turbines
        run[t] = np.where(same[t], run[t - 1] + 1, 0)
    return run >= n - 1


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
