"""data-v2 prosumers: attributes, strategies, meter layer, settlement, no hidden columns in
training files, and the existing fleet (data-v4) left byte-identical."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest

from wattsentry.sim.prosumers_v2 import FORBIDDEN_COLUMNS, ProsumerConfig, generate, load_training, load_truth
from wattsentry.sim.prosumers_v2.metering import settle_on_profile
from wattsentry.sim.prosumers_v2.prices import PRICES, pzu_15min

TZ = "Europe/Bucharest"
# Small, battery-heavy world so every strategy shows up.
TINY = ProsumerConfig(n_homes=120, n_localities=4, chunk_homes=50, start_year=2024, end_year=2024,
                      share_battery=0.5, share_maxcontrol_of_battery=0.5, share_enrolled=0.3)


@pytest.fixture(scope="module")
def ds(tmp_path_factory):
    out = generate(TINY, out_dir=tmp_path_factory.mktemp("pv2") / "d", log=lambda *_: None, workers=2)
    return {"out": out, "manifest": json.loads((out / "manifest.json").read_text()),
            "train": load_training(out), "truth": load_truth(out)}


def _files(out, entry):
    p = out / entry
    return sorted(p.rglob("*.parquet")) if p.is_dir() else [p]


def test_no_hidden_columns_in_training_files(ds):
    out, man = ds["out"], ds["manifest"]
    assert {"pv_kw", "batt_kw", "soc_kwh", "strategy", "load_kw", "has_battery"} <= FORBIDDEN_COLUMNS
    training = [f for e in man["training_files"] for f in _files(out, e)]
    hidden = [f for e in man["hidden_files"] for f in _files(out, e)]
    assert training and hidden
    assert set(training) | set(hidden) == set(out.rglob("*.parquet"))  # every file is classified
    for f in training:
        bad = set(pq.read_schema(f).names) & FORBIDDEN_COLUMNS
        assert not bad, f"{f.relative_to(out)} carries hidden columns {bad}"
    for name, df in ds["train"].items():
        assert not set(df.columns) & FORBIDDEN_COLUMNS, name
    assert {"pv_kw", "batt_kw", "soc_kwh"} <= set(pq.read_schema(out / "truth/home_15min/year=2024/part-000.parquet").names)


def test_attributes(ds):
    h = ds["truth"]["homes"]
    assert len(h) == 120 and h.home_id.is_unique
    pub = ds["train"]["homes"]
    assert list(pub.home_id) == list(h.home_id) and {"kwp", "contract", "meter", "battery_declared"} <= set(pub.columns)
    mc, smart, has = h.contract == "maxcontrol", h.meter == "smart", h.has_battery
    assert (smart[mc]).all()  # MaxControl needs a smart meter
    assert ((h.strategy == "none") == ~has).all()
    assert (mc & has)[h.strategy == "price_driven"].all()
    enr = h.strategy == "enrolled"
    assert (h.enrolled == enr).all() and (~mc & smart & h.battery_declared)[enr].all()
    assert (has[h.battery_declared]).all()
    assert set(h.strategy) == {"none", "self_consumption", "price_driven", "enrolled"}
    b = h[has]
    assert b.cap_kwh.between(5, 15).all() and b.p_kw.between(3, 5).all() and b.reserve_frac.between(0.1, 0.4).all()
    assert b.k_top.between(8, 20).all() and (h[~has][["cap_kwh", "p_kw", "k_top"]] == 0).all().all()
    assert set(h.split) == {"train", "unseen"}


def test_default_shares():
    from wattsentry.sim.prosumers_v2.homes import build_homes

    h = build_homes(ProsumerConfig(n_homes=5000))
    assert abs(h.has_battery.mean() - 0.20) < 0.02
    assert abs((h.meter == "smart").mean() - 0.60) < 0.03
    assert abs(h.battery_declared[h.has_battery].mean() - 0.70) < 0.05
    pd_share = (h.strategy == "price_driven").sum() / h.has_battery.sum()
    assert 0.17 < pd_share < 0.28  # 0.25 MaxControl x 0.90 price-driven


def test_truth_physics(ds):
    t, h = ds["truth"]["home_15min"], ds["truth"]["homes"].set_index("home_id")
    assert len(t) == 120 * 96 * 366
    net = (t.load_kw - t.pv_kw + t.batt_kw) * 0.25
    assert (net - (t.import_kwh - t.export_kwh)).abs().max() < 2e-3  # rounding to 1 W / 1 Wh
    assert ((t.import_kwh == 0) | (t.export_kwh == 0)).all()
    has = t.home_id.map(h.has_battery).astype(bool)
    assert (t.batt_kw[~has] == 0).all() and t.soc_kwh[~has].isna().all()
    cap = t.home_id.map(h.cap_kwh)[has]
    assert (t.soc_kwh[has] <= cap + 1e-3).all() and (t.soc_kwh[has] >= 0).all()
    assert (t.batt_kw.abs() <= t.home_id.map(h.p_kw) + 1e-3).all()


def test_price_driven_exports_in_the_evening(ds):
    """Spec I step 2 'done when': on summer days MaxControl (price-driven) homes export in 18-21, standard ones don't."""
    sh = ds["manifest"]["stats"]["summer_export_share_18_21"]
    assert sh["price_driven"] > 3 * sh["self_consumption"] and sh["price_driven"] > 0.08
    t, h = ds["truth"]["home_15min"], ds["truth"]["homes"].set_index("home_id")
    loc = t.ts.dt.tz_convert(TZ)
    eve = loc.dt.month.isin([6, 7, 8]) & loc.dt.hour.between(18, 20)
    batt_out = (-t.batt_kw.clip(upper=0))[eve].groupby(t.home_id[eve].map(h.strategy)).mean()
    assert batt_out["price_driven"] > 2 * batt_out["self_consumption"]


def test_meter_layer(ds):
    mo, mm, h = ds["train"]["meter_observed"], ds["train"]["meter_monthly"], ds["truth"]["homes"]
    t = ds["truth"]["home_15min"]
    smart = set(h.home_id[h.meter == "smart"])
    assert set(mo.home_id) == smart  # profile homes never give a curve
    ok = mo.import_kwh.notna()
    assert 0 < (~ok).mean() < 0.02 and (mo.import_kwh.isna() == mo.export_kwh.isna()).all()
    tt = t[t.home_id.isin(smart)].reset_index(drop=True)
    assert (mo.ts.values == tt.ts.values).all() and (mo.home_id.values == tt.home_id.values).all()
    assert np.array_equal(mo.import_kwh[ok].to_numpy(), tt.import_kwh[ok].to_numpy())
    # D+1: available at the next local midnight
    loc = mo.ts.dt.tz_convert(TZ)
    nxt = (loc.dt.tz_localize(None).dt.normalize() + pd.Timedelta(days=1)).dt.tz_localize(TZ)
    assert (mo.available_at == nxt).all() and (mo.available_at > mo.ts).all()
    # monthly totals for every home = the truth summed per local month
    assert set(mm.home_id) == set(h.home_id) and len(mm) == 120 * 12
    tm = t.assign(month=t.ts.dt.tz_convert(TZ).dt.strftime("%Y-%m")).groupby(["home_id", "month"]).import_kwh.sum()
    got = mm.set_index(["home_id", "month"]).import_kwh.reindex(tm.index)
    assert np.allclose(got, tm, atol=0.05)


def test_settlement(ds):
    s, mm, prof = ds["train"]["settlement"], ds["train"]["meter_monthly"], pd.read_parquet(
        ds["out"] / "standard_profile.parquet")
    port = ds["truth"]["portfolio"]
    h = ds["truth"]["homes"]
    # all energy is settled: smart on the curve + profile homes on the profile = true total
    assert np.isclose((s.smart_import_kwh + s.profile_import_kwh).sum(), port.import_kwh.sum(), rtol=1e-6)
    assert np.isclose((s.smart_export_kwh + s.profile_export_kwh).sum(), port.export_kwh.sum(), rtol=1e-6)
    # profile part: monthly totals of profile homes, shaped like the standard profile within the month
    s = s.assign(month=s.ts.dt.tz_convert(TZ).dt.strftime("%Y-%m"))
    prof_mm = mm[mm.home_id.isin(h.home_id[h.meter == "profile"])].merge(h[["home_id", "county"]])
    want = prof_mm.groupby(["county", "month"]).import_kwh.sum()
    got = s.groupby(["county", "month"]).profile_import_kwh.sum().reindex(want.index)
    assert np.allclose(got, want, rtol=1e-6, atol=1e-3)
    c0 = s[s.county == s.county.iloc[0]].merge(prof, on="ts")
    for _, g in c0.groupby("month"):
        r = g.profile_import_kwh / g.load_coef
        assert np.allclose(r, r.iloc[0])
    # smart part equals the smart homes' true curve
    sm = port[port.meter == "smart"].groupby("ts").import_kwh.sum()
    assert np.allclose(s.groupby("ts").smart_import_kwh.sum().reindex(sm.index), sm)


def test_settle_on_profile_hand():
    midx = np.array([0, 0, 1, 1, 1])
    out = settle_on_profile(np.array([10.0, 3.0]), np.array([1.0, 3.0, 1.0, 1.0, 1.0]), midx)
    assert out.tolist() == [2.5, 7.5, 1.0, 1.0, 1.0]


def test_prices():
    ts = pd.date_range("2022-03-01", "2022-03-03", freq="15min", tz=TZ, inclusive="left").tz_convert("UTC")
    p = pzu_15min(ts)
    assert p.pzu_lei_mwh.notna().all() and set(p.price_source) == {"proxy_shift_371d"}
    ts = pd.date_range("2024-06-01", "2024-06-02", freq="15min", tz=TZ, inclusive="left").tz_convert("UTC")
    p = pzu_15min(ts)
    assert set(p.price_source) == {"opcom_pzu_hourly_x4"}
    v = p.pzu_lei_mwh.to_numpy().reshape(-1, 4)
    assert (v == v[:, :1]).all()  # hourly price repeated over its four quarter-hours
    raw = pd.read_parquet(PRICES)
    hit = raw[raw.ts == ts[8]].price_lei_mwh.iloc[0]
    assert p.pzu_lei_mwh.iloc[8] == hit


def test_deterministic_and_backend_independent(tmp_path, monkeypatch):
    cfg = ProsumerConfig(n_homes=24, n_localities=2, chunk_homes=12, start_year=2024, end_year=2024,
                         share_battery=0.6, share_maxcontrol_of_battery=0.5)
    a = generate(cfg, out_dir=tmp_path / "a", log=lambda *_: None, workers=1)
    monkeypatch.setenv("WATTSENTRY_KERNELS", "python")
    b = generate(cfg, out_dir=tmp_path / "b", log=lambda *_: None, workers=2)
    for f in sorted(a.rglob("*.parquet")):
        assert pq.read_table(f).equals(pq.read_table(b / f.relative_to(a))), f.relative_to(a)


# ----------------------------------------------------------------------------- the existing fleet stays as it is
FLEET_TINY = dict(n_sites=12, n_feeders=2, n_congested=1, start_year=2024, end_year=2024, weather_source="synthetic")
# Reference sums of the fleet dataset (data-v2 must leave it unchanged); sums, so last-ulp platform noise passes.
FLEET_GOLDEN = {"pv_ac_w": 314356850.0, "load_w": 317311693.0, "batt_w": 3377508.0, "grid_w": -6332516.0,
                "inv_voltage_v": 98585693.38804626, "soc": 82326.93044994771}


def test_fleet_unchanged(tmp_path):
    from wattsentry.sim.fleet import DATA_VERSION, FleetConfig
    from wattsentry.sim.fleet import generate as fleet_generate

    if DATA_VERSION != 4:
        pytest.skip(f"fleet moved to data-v{DATA_VERSION} on purpose; golden values are for data-v4")
    assert FleetConfig().digest("synthetic") == "84a57ad45142"  # same cache key -> same data-v4 directory
    cfg = FleetConfig(**FLEET_TINY)
    a = fleet_generate(cfg, out_dir=tmp_path / "a", plots=False, log=lambda *_: None)
    generate(ProsumerConfig(n_homes=12, n_localities=2, chunk_homes=12, start_year=2024, end_year=2024,
                            weather_source="synthetic"), out_dir=tmp_path / "pv2", log=lambda *_: None, workers=1)
    b = fleet_generate(cfg, out_dir=tmp_path / "b", plots=False, log=lambda *_: None)
    for f in sorted(a.rglob("*.parquet")):  # generating data-v2 in between changes nothing in the fleet
        assert pq.read_table(f).equals(pq.read_table(b / f.relative_to(a))), f.relative_to(a)
    t = pd.read_parquet(a / "telemetry/year=2024.parquet")
    for col, want in FLEET_GOLDEN.items():
        assert float(t[col].astype("float64").sum()) == pytest.approx(want, rel=1e-6), col
