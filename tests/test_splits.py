"""#10: splits have no site/time leakage; data checks catch what they claim; window cache builds deterministically."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from wattsentry.data.splits import SPLITS, check_no_leakage, make_splits, turbine_split, unseen_park

DAYS = pd.date_range("2022-01-01", "2025-12-30", freq="D", tz="UTC")


def test_time_and_site_splits_do_not_overlap():
    rng = np.random.default_rng(0)
    split_col = np.where(rng.random(600) < 0.3, "unseen", "train")
    sp = make_splits(DAYS, split_col)
    check_no_leakage(sp)
    years = {k: set(DAYS[v].year) for k, v in sp.days.items()}
    assert years == {"train": {2022, 2023}, "val": {2024}, "cal": {2024}, "test": {2025}}
    all_days = np.concatenate([sp.days[k] for k in SPLITS])
    assert len(all_days) == len(set(all_days))
    assert sp.days["train"].min() >= 14  # 14 days of history before the first target
    for k in ("train", "val", "cal"):
        s, _ = sp.site_days(k)
        assert not set(s) & set(sp.unseen_sites)
    s, _ = sp.site_days("test")
    assert set(s) == set(range(600))
    weeks = DAYS[sp.days["cal"]].isocalendar().week.to_numpy()
    assert (weeks % 4 == 3).all() and len(sp.days["cal"]) > 60


def test_small_series_fallback_nonempty():
    sp = make_splits(pd.date_range("2024-01-01", periods=120, freq="D", tz="UTC"), ["train"] * 7 + ["unseen"] * 3)
    assert all(len(sp.days[k]) for k in SPLITS)
    check_no_leakage(sp)


def test_wind_park_holdout():
    parks = ["WPA"] * 50 + ["WPB"] * 23 + ["WPC"] * 47
    assert unseen_park(parks) == "WPB"
    s = turbine_split(parks)
    assert (s[50:73] == "unseen").all() and (s[:50] == "train").all() and (s[73:] == "train").all()


def test_leakage_detected():
    sp = make_splits(DAYS, ["train", "unseen"])
    bad = type(sp)({**sp.days, "test": np.r_[sp.days["test"], sp.days["train"][:1]]}, sp.train_sites, sp.unseen_sites)
    with pytest.raises(AssertionError):
        check_no_leakage(bad)


def _telemetry(n=96 * 3):
    ts = pd.date_range("2024-06-01", periods=n, freq="15min", tz="UTC")
    from wattsentry.sim.fleet.solar import solar_position

    zen, _ = solar_position(ts, 44.49, 26.18)
    pv = np.round(np.clip(np.cos(np.radians(zen)), 0, None) * 4000, 0)
    return pd.DataFrame({"ts": ts, "site_id": "S0001", "pv_ac_w": pv, "load_w": 500.0, "batt_w": 0.0,
                         "soc": 0.5, "inv_voltage_v": 231.0, "inv_status": 0})


SITES = pd.DataFrame({"site_id": ["S0001"], "lat": [44.49], "lon": [26.18], "inverter_kw": [5.0]})


def test_quality_report_issues():
    pytest.importorskip("pandera")
    from wattsentry.data.schemas import quality_report, validate_telemetry

    df = _telemetry()
    validate_telemetry(df)
    assert quality_report(df, SITES).empty
    bad = df.copy()
    bad.loc[40:60, "pv_ac_w"] = 1234.0                      # frozen
    bad.loc[100:103, "pv_ac_w"] = np.nan                    # gap
    bad.loc[90, "pv_ac_w"] = 900.0                          # night power (~22:30 UTC)
    bad.loc[150:152, "soc"] = 1.2                           # SoC > 100 %
    bad = bad.drop(index=[200, 201])                        # missing timestamps
    rep = quality_report(bad, SITES)
    assert {"frozen", "gaps", "night_power", "soc_over_100"} <= set(rep.issue)
    units = df.assign(pv_ac_w=df.pv_ac_w * 1000)
    assert "units" in set(quality_report(units, SITES).issue)
    with pytest.raises(Exception):
        validate_telemetry(units)  # pv_ac_w far outside the physical range


def test_window_cache_deterministic(tmp_path, monkeypatch):
    pytest.importorskip("wattsentry.model.physics")
    pytest.importorskip("pvlib")
    from wattsentry.data.windows import FAULTS, build_windows
    from wattsentry.sim.faults import FaultConfig
    from wattsentry.sim.fleet import FleetConfig

    monkeypatch.setenv("WATTSENTRY_CACHE", str(tmp_path / "cache"))
    cfg = FleetConfig(n_sites=16, n_feeders=2, n_congested=1, start_year=2024, end_year=2024)
    a = build_windows(fleet_cfg=cfg, fault_cfg=FaultConfig(faulty_share=0.3), out=tmp_path / "a", log=lambda *_: None)
    b = build_windows(fleet_cfg=cfg, fault_cfg=FaultConfig(faulty_share=0.3), out=tmp_path / "b", log=lambda *_: None)
    ia, ib = json.loads((a / "index.json").read_text()), json.loads((b / "index.json").read_text())
    assert ia["format"] == "wattsentry-windows-v1" and ia["faults"] == FAULTS
    assert ia["n_steps"] % 96 == 0 and pd.Timestamp(ia["t_start"]).hour == 0
    for name, meta in ia["arrays"].items():
        x, y = np.load(a / meta["file"]), np.load(b / meta["file"])
        assert np.array_equal(x, y, equal_nan=True), name
        assert np.isfinite(x.astype(np.float32)).all(), name
    S, T = ia["n_sites"], ia["n_steps"]
    assert ia["arrays"]["pv_ac"]["shape"] == [S, T] and ia["arrays"]["weather_act"]["shape"][1:] == [T, 7]
    eps = json.loads((a / "episodes.json").read_text())
    fault = np.load(a / "fault.npy")
    assert eps and all(fault[e["site"], e["onset"]] == FAULTS.index(e["fault"]) for e in eps)
    sites = json.loads((a / "sites.json").read_text())
    assert {"site_id", "loc_idx", "split", "kwp_recorded", "meta_wrong"} <= set(sites[0])
    # wattsentry.model.data (#14) can open it, when present
    data = pytest.importorskip("wattsentry.model.data")
    src = data.ArraySource(a)
    assert src.n_steps == T
