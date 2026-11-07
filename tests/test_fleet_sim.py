"""Fleet simulator (#11, #52): contract columns, physics invariants, battery == contracts.opt, RO calendar."""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import pytest

from sentinel.sim.fleet import BatteryParams, FleetConfig, apply_power, generate, self_consumption_step
from sentinel.sim.fleet.ro_calendar import orthodox_easter, ro_holidays

TINY = FleetConfig(n_sites=40, n_feeders=3, n_congested=1, start_year=2024, end_year=2024)


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    out = generate(TINY, out_dir=tmp_path_factory.mktemp("fleet"), plots=False, log=lambda *_: None)
    return {
        "out": out,
        "sites": pd.read_parquet(out / "sites.parquet"),
        "feeders": pd.read_parquet(out / "feeders.parquet"),
        "tel": pd.read_parquet(out / "telemetry" / "year=2024.parquet"),
    }


def test_contract_columns(world):
    schemas = pytest.importorskip("sentinel.contracts.schemas")
    for table, df in (("sites", world["sites"]), ("feeders", world["feeders"]), ("telemetry", world["tel"])):
        assert schemas.check_columns(table, df.columns) == []


def test_shapes_and_split(world):
    s, tel = world["sites"], world["tel"]
    assert len(s) == 40 and s.site_id.is_unique
    assert len(tel) == 40 * 4 * 24 * 366  # 2024 is a leap year; DST shifts cancel over a local year
    assert set(s.split) == {"train", "unseen"}
    assert (s.gis_feeder_id != s.feeder_id).sum() <= 0.1 * len(s)


def test_physics(world):
    s, tel = world["sites"].set_index("site_id"), world["tel"]
    assert (tel.pv_ac_w >= 0).all() and (tel.load_w >= 0).all()
    assert (tel.pv_ac_w - tel.load_w - tel.batt_w - tel.grid_w).abs().max() <= 1.5  # rounding to 1 W
    inv = tel.site_id.map(s.inverter_kw * 1000)
    assert (tel.pv_ac_w <= inv + 1).all()
    no_batt = tel.site_id.map(s.battery_kwh == 0)
    assert tel.soc[no_batt].isna().all() and (tel.batt_w[no_batt] == 0).all()
    soc = tel.soc[~no_batt]
    assert soc.between(tel.site_id[~no_batt].map(s.soc_min) - 1e-3, 1 + 1e-6).all()
    assert (tel.inv_status == 0).all()
    assert tel.inv_voltage_v.between(200, 260).all()


def test_congestion(world):
    f, out = world["feeders"], world["out"]
    ft = pd.read_parquet(out / "feeder_ts" / "year=2024.parquet")
    peak = ft.groupby("feeder_id").net_export_w.max() / 1000
    over = peak > f.set_index("feeder_id").g_max_kw
    assert over[f.set_index("feeder_id").congested].all()


def test_deterministic(tmp_path):
    a = generate(TINY.__class__(**{**TINY.to_json(), "n_sites": 15, "n_feeders": 2}), out_dir=tmp_path / "a",
                 plots=False, log=lambda *_: None)
    b = generate(TINY.__class__(**{**TINY.to_json(), "n_sites": 15, "n_feeders": 2}), out_dir=tmp_path / "b",
                 plots=False, log=lambda *_: None)
    pd.testing.assert_frame_equal(pd.read_parquet(a / "sites.parquet"), pd.read_parquet(b / "sites.parquet"))
    ta = pd.read_parquet(a / "telemetry" / "year=2024.parquet")
    tb = pd.read_parquet(b / "telemetry" / "year=2024.parquet")
    pd.testing.assert_frame_equal(ta, tb)


def test_battery_matches_contract():
    opt = pytest.importorskip("sentinel.contracts.opt")
    rng = np.random.default_rng(0)
    n = 200
    specs = [opt.BatterySpec(capacity_kwh=float(rng.choice([0, 5, 10, 13.5])), p_max_kw=float(rng.uniform(2, 5)),
                             eta_charge=float(rng.uniform(0.9, 0.98)), eta_discharge=float(rng.uniform(0.9, 0.98)),
                             soc_min=float(rng.uniform(0, 0.2)), soc_max=float(rng.uniform(0.85, 1.0)),
                             comfort_reserve_kwh=0.0) for _ in range(n)]
    p = BatteryParams.from_specs(specs)
    e = np.array([s.capacity_kwh * rng.uniform(s.soc_min, s.soc_max) for s in specs])
    for _ in range(50):
        pv_kw, load_kw = rng.uniform(0, 8, n), rng.uniform(0, 6, n)
        b_w, e_next, g_w = self_consumption_step(p, e, pv_kw * 1000, load_kw * 1000, 0.25)
        for i, spec in enumerate(specs):
            bk, ek, gk = opt.self_consumption_step(spec if spec.capacity_kwh > 0 else None, e[i], pv_kw[i],
                                                   load_kw[i], 0.25)
            assert b_w[i] / 1000 == pytest.approx(bk, abs=1e-9)
            assert e_next[i] == pytest.approx(ek, abs=1e-9)
            assert g_w[i] / 1000 == pytest.approx(gk, abs=1e-9)
        req = rng.uniform(-6, 6, n)
        a_w, e2 = apply_power(p, e_next, req * 1000, 0.25)
        for i, spec in enumerate(specs):
            if spec.capacity_kwh > 0:
                ak, ek = spec.step(e_next[i], req[i], 0.25)
                assert a_w[i] / 1000 == pytest.approx(ak, abs=1e-9) and e2[i] == pytest.approx(ek, abs=1e-9)
        e = e2


def test_ro_calendar():
    assert [orthodox_easter(y) for y in (2022, 2023, 2024, 2025)] == [
        dt.date(2022, 4, 24), dt.date(2023, 4, 16), dt.date(2024, 5, 5), dt.date(2025, 4, 20)]
    h = ro_holidays(2024)
    assert {dt.date(2024, 5, 3), dt.date(2024, 5, 6), dt.date(2024, 6, 24), dt.date(2024, 1, 6),
            dt.date(2024, 12, 1)} <= h
    assert dt.date(2023, 1, 6) not in ro_holidays(2023)
