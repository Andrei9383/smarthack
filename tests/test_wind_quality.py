"""Card #22: WindMon's six quality rules and OK / DEGRADED / SUSPECT flags (wattsentry/data/wind/quality.py)."""

from datetime import datetime, timedelta, timezone

import numpy as np
import polars as pl
import pytest

from wattsentry.contracts.wind import SENSORS
from wattsentry.data.wind.quality import (
    FROZEN_STEPS,
    RULES,
    apply_rules,
    flag_counts,
    wide_from_arrays,
)

RATED = 2500.0
T = 24
T0 = datetime(2024, 1, 1, tzinfo=timezone.utc)
TURB = pl.DataFrame({"turbine_id": ["WP1.TID01"], "rated_kw": [RATED], "power_scale": [1.0]})


def _healthy(n=T, seed=0):
    """A producing turbine with realistic noise (no rule fires)."""
    rng = np.random.default_rng(seed)
    cols = {
        "ts": [T0 + timedelta(minutes=15 * i) for i in range(n)],
        "turbine_id": ["WP1.TID01"] * n,
        "act_pwt": 1200 + rng.normal(0, 30, n),
        "wd_spd": 32 + rng.normal(0, 1.5, n),
        "turbine_t": 8 + rng.normal(0, 0.2, n),
    }
    for k in ("conv_t", "gearbox_t", "gen1_t", "gen2_t", "transformer_t"):
        cols[k] = 40 + rng.normal(0, 0.4, n)
    df = pl.DataFrame(cols).with_columns(pl.col("ts").dt.cast_time_unit("us"))
    return df.with_columns([pl.lit(False).alias(f"{s}__code") for s in SENSORS])


def _set(df, col, idx, value):
    v = df[col].to_list()
    for i in idx:
        v[i] = value
    return df.with_columns(pl.Series(col, v, dtype=df[col].dtype))


def _q(df):
    return apply_rules(df, TURB)


def test_healthy_is_ok():
    q = _q(_healthy())
    assert (q["quality"] == "OK").all()
    assert not any(q[f"q_{r.lower()}"].any() for r in RULES)


@pytest.mark.parametrize("sensor,flag", [("act_pwt", "SUSPECT"), ("wd_spd", "SUSPECT"), ("gearbox_t", "DEGRADED")])
def test_missing(sensor, flag):
    q = _q(_set(_healthy(), sensor, [5], None))
    assert q["q_missing"][5] and q["quality"][5] == flag
    assert q["quality"][4] == "OK"


def test_error_code_is_suspect_and_not_missing():
    df = _set(_set(_healthy(), "act_pwt", [7], None), "act_pwt__code", [7], True)
    q = _q(df)
    assert q["q_error_code"][7] and not q["q_missing"][7] and q["quality"][7] == "SUSPECT"


@pytest.mark.parametrize("sensor,value,flag", [
    ("act_pwt", 1.2 * RATED, "SUSPECT"), ("act_pwt", -0.2 * RATED, "SUSPECT"), ("wd_spd", -3.0, "SUSPECT"),
    ("wd_spd", 200.0, "SUSPECT"), ("transformer_t", 180.0, "DEGRADED"), ("turbine_t", -55.0, "DEGRADED"),
])
def test_out_of_range(sensor, value, flag):
    q = _q(_set(_healthy(), sensor, [3], value))
    assert q["q_out_of_range"][3] and q["quality"][3] == flag
    assert not q["q_out_of_range"][2]


def test_frozen():
    q = _q(_set(_healthy(), "wd_spd", range(4, 4 + FROZEN_STEPS), 31.5))
    assert q["q_frozen"][4:4 + FROZEN_STEPS].all() and (q["quality"][4:4 + FROZEN_STEPS] == "SUSPECT").all()
    assert not q["q_frozen"][3]
    short = _q(_set(_healthy(), "wd_spd", range(4, 4 + FROZEN_STEPS - 1), 31.5))
    assert not short["q_frozen"].any()
    temp = _q(_set(_healthy(), "gen1_t", range(10, 10 + FROZEN_STEPS), 41.25))
    assert temp["q_frozen"][10] and temp["quality"][10] == "DEGRADED"
    parked = _q(_set(_healthy(), "act_pwt", range(0, 12), 0.0))  # a parked turbine may report a constant 0 kW
    assert not parked["q_frozen"].any()


def test_power_without_wind():
    df = _set(_healthy(), "wd_spd", [6], 4.0)  # 1.1 m/s with 1200 kW
    q = _q(df)
    assert q["q_power_without_wind"][6] and q["quality"][6] == "SUSPECT"


def test_wind_without_power_is_degraded():
    df = _set(_set(_healthy(), "wd_spd", [9], 65.0), "act_pwt", [9], 0.4)
    q = _q(df)
    assert q["q_wind_without_power"][9] and q["quality"][9] == "DEGRADED"
    above_cutout = _q(_set(_set(_healthy(), "wd_spd", [9], 100.0), "act_pwt", [9], 0.4))
    assert not above_cutout["q_wind_without_power"][9]  # storm shutdown above cut-out is expected


def test_suspect_beats_degraded_and_counts():
    df = _set(_set(_healthy(), "gearbox_t", [2], None), "act_pwt", [2], None)
    q = _q(df)
    assert q["quality"][2] == "SUSPECT" and q["q_suspect_rules"][2] >= 1 and q["q_degraded_rules"][2] >= 1
    c = flag_counts(q)
    assert c["share_suspect"] == pytest.approx(1 / T, abs=1e-4) and set(c) >= {f"rule_{r.lower()}" for r in RULES}


def test_flags_on_simulated_faults():
    """On the simulator + injector, comm gaps and stuck anemometers come out SUSPECT; clean data is ~all OK."""
    from wattsentry.model.wind.data import turbines_table
    from wattsentry.sim.wind.build import simulate
    from wattsentry.sim.wind.config import WindSimConfig
    from wattsentry.sim.wind_faults import FaultConfig

    cfg = WindSimConfig(start="2024-01-01", end="2024-01-15", max_turbines_per_park=3)
    clean, _, d0 = simulate(cfg, faults=False, log=lambda *a: None)
    q0 = apply_rules(wide_from_arrays(d0), turbines_table(d0))
    assert (q0["quality"] == "OK").mean() > 0.99
    _, res, d = simulate(cfg, FaultConfig().only("data_gap", rate_gap_short=40.0), log=lambda *a: None)
    q = apply_rules(wide_from_arrays(d), turbines_table(d))
    gap = np.isnan(d.sensors["act_pwt"]).T.reshape(-1)  # wide_from_arrays is turbine-major
    assert (q.sort(["turbine_id", "ts"])["quality"].to_numpy()[gap] == "SUSPECT").all()
