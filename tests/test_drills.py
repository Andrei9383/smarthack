"""#28: every drill corrupts what it says, never mutates its input, is deterministic, and predictors survive it."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sentinel.contracts import MockPredictor, Window
from sentinel.data.drills import DAY, DRILLS, telemetry_drill

T0 = pd.Timestamp("2025-06-10T00:00:00Z")


def _window():
    hist = pd.date_range(T0 - pd.Timedelta(days=14), T0, freq="15min", inclusive="left")
    h = hist.hour.to_numpy() + hist.minute.to_numpy() / 60
    pv = np.clip(np.sin(np.pi * (h - 3) / 15), 0, None) * 4000
    fut = pd.date_range(T0, periods=DAY, freq="15min")
    hf = fut.hour.to_numpy() + fut.minute.to_numpy() / 60
    ghi = np.clip(np.sin(np.pi * (hf - 3) / 15), 0, None) * 800
    weather = np.stack([ghi, ghi * 0.7, ghi * 0.3, np.full(DAY, 20.0)], 1)
    return Window(t0=T0.isoformat(), horizon=DAY, weather=weather, reference=pv, history=np.full(len(pv), 500.0),
                  battery=np.zeros((DAY, 2)), meta={"weather_channels": ["ghi", "dni", "dhi", "temp"],
                                                    "power_unit": "W"})


ASSET = {"site_id": "S0001", "kwp_recorded": 6.0, "kwp_true": 6.0, "tilt_recorded": 30.0, "azimuth_recorded": 180.0,
         "inverter_kw": 5.0, "lat": 44.49, "lon": 26.18}


@pytest.mark.parametrize("name", list(DRILLS))
def test_drill_contract(name):
    w = _window()
    ref0 = np.array(w.reference, copy=True)
    a1, w1 = DRILLS[name](dict(ASSET), w, np.random.default_rng(0))
    a2, w2 = DRILLS[name](dict(ASSET), w, np.random.default_rng(0))
    assert np.array_equal(np.asarray(w.reference), ref0)  # input untouched
    assert np.array_equal(np.asarray(w1.reference), np.asarray(w2.reference), equal_nan=True)  # deterministic
    assert np.array_equal(np.asarray(w1.weather), np.asarray(w2.weather), equal_nan=True)
    assert w1.meta["drill"] == name and len(w1.reference) == len(ref0)
    p = MockPredictor("pv").predict(a1, w1)  # a predictor must not crash on it
    assert len(p.q50) == DAY


def test_drill_signatures():
    w, rng = _window(), np.random.default_rng(1)
    ref = np.asarray(w.reference)
    for name, share in (("gaps_10", 0.1), ("gaps_40", 0.4)):
        r = np.asarray(DRILLS[name](ASSET, w, rng)[1].reference)
        assert share <= np.isnan(r).mean() < share + 0.1
    assert np.isnan(np.asarray(DRILLS["missing_day"](ASSET, w, rng)[1].reference)[-DAY:]).all()
    fr = np.asarray(DRILLS["frozen_sensor"](ASSET, w, rng)[1].reference)
    assert np.unique(fr[-3 * DAY:]).size == 1 and fr[-1] > 0
    assert np.nanmax(DRILLS["units_kw"](ASSET, w, rng)[1].reference) == pytest.approx(np.nanmax(ref) / 1000)
    assert np.nanmax(DRILLS["units_x1000"](ASSET, w, rng)[1].reference) == pytest.approx(np.nanmax(ref) * 1000)
    a, _ = DRILLS["unseen_site_type"](ASSET, w, rng)
    assert a["kwp_recorded"] > 2 * ASSET["kwp_recorded"] and a["azimuth_recorded"] in (90.0, 270.0)
    _, wd = DRILLS["telemetry_dropout"](ASSET, w, rng)
    assert np.isnan(np.asarray(wd.reference)[-24:]).all() and wd.meta["telemetry_age_steps"] == 24
    _, wf = DRILLS["forecast_noise"](ASSET, w, rng)
    assert not np.allclose(np.asarray(wf.weather)[:, 0], np.asarray(w.weather)[:, 0])
    assert (np.asarray(wf.weather)[:, :3] >= 0).all()


def test_telemetry_drills_caught_by_quality_checks():
    pytest.importorskip("pandera")
    from sentinel.data.schemas import quality_report
    from sentinel.sim.fleet.solar import solar_position

    ts = pd.date_range("2024-06-01", periods=DAY * 5, freq="15min", tz="UTC")
    zen, _ = solar_position(ts, 44.49, 26.18)
    df = pd.DataFrame({"ts": ts, "site_id": "S0001", "pv_ac_w": np.round(np.clip(np.cos(np.radians(zen)), 0, None) * 4000),
                       "load_w": 400.0, "batt_w": 0.0, "grid_w": 0.0, "soc": 0.5, "inv_voltage_v": 231.0})
    sites = pd.DataFrame({"site_id": ["S0001"], "lat": [44.49], "lon": [26.18], "inverter_kw": [5.0]})
    rng = np.random.default_rng(0)
    assert quality_report(df, sites).empty
    expect = {"gaps_10": "gaps", "missing_day": "gaps", "frozen_sensor": "frozen", "units_x1000": "units",
              "telemetry_dropout": "gaps"}
    for name, issue in expect.items():
        rep = quality_report(telemetry_drill(df, name, rng), sites)
        assert issue in set(rep.issue), name
    assert len(telemetry_drill(df, "telemetry_dropout", rng)) < len(df)
    with pytest.raises(KeyError):
        telemetry_drill(df, "nope", rng)
