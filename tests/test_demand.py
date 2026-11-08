"""Card #44 M3 DemandNet: scenario aggregation, hierarchy roll-up, peak hour, RO calendar, v2 training, outputs."""

import json
import time
from datetime import date

import numpy as np
import pytest

from wattsentry.contracts import PAYLOAD_KEYS, Action, Prediction
from wattsentry.demand import (
    ROW_FIELDS,
    DemandNet,
    aggregate_scenarios,
    bucharest_offset_hours,
    build_hierarchy,
    calendar_features,
    mock_load_scenarios,
    orthodox_easter,
    ro_public_holidays,
    zone_demand_actions,
    zone_demand_rows,
)
from wattsentry.demand.data import load_national, synthetic_ro_load
from wattsentry.demand.model import NationalShape, default_split, evaluate_zone, nmae, peak_hour_error
from wattsentry.demand.train import demo_hierarchy, main

T0_WINTER = "2025-01-14T22:00:00Z"  # local midnight, Europe/Bucharest (UTC+2)
T0_SUMMER = "2025-07-14T21:00:00Z"  # local midnight (UTC+3)


@pytest.fixture(scope="module")
def demo():
    sites, feeders = demo_hierarchy()
    scen = mock_load_scenarios(sites, T0_WINTER, 200, seed=0)  # independent households
    agg = aggregate_scenarios(scen, sites, feeders, t0=T0_WINTER)
    return sites, feeders, scen, agg


# --- v1 aggregation --------------------------------------------------------------------------------------

def test_p90_of_sum_below_sum_of_p90(demo):
    sites, _, scen, agg = demo
    for z in agg.level("county"):
        idx = [i for i, s in enumerate(sites) if s["county"] == z.zone_id]
        sum_of_q90 = np.quantile(scen[:, idx, :], 0.9, axis=0).sum(axis=0) / 1000.0
        sum_of_q10 = np.quantile(scen[:, idx, :], 0.1, axis=0).sum(axis=0) / 1000.0
        q90, q10 = np.array(z.q90_mw), np.array(z.q10_mw)
        assert np.all(q90 <= sum_of_q90 + 1e-9)
        assert np.all(q10 >= sum_of_q10 - 1e-9)
        assert q90.max() < 0.97 * sum_of_q90.max()  # clearly tighter: independent homes don't all peak high


def test_hierarchy_rollup_consistent(demo):
    sites, _, scen, agg = demo
    zs, by_id = agg.zone_scenarios, agg.by_id()
    assert {z.level for z in agg.zones} == {"feeder", "locality", "county"}
    for parent in agg.zones:
        if parent.level == "feeder":
            continue
        kids = [z for z in agg.zones if z.parent_id == parent.zone_id]
        assert kids, parent.zone_id
        np.testing.assert_allclose(sum(zs[k.zone_id] for k in kids), zs[parent.zone_id], rtol=1e-9)
        np.testing.assert_allclose(sum(np.array(k.mean_mw) for k in kids), parent.mean_mw, rtol=1e-9)
        assert parent.n_sites == sum(k.n_sites for k in kids)
        assert parent.capacity_mw == pytest.approx(sum(k.capacity_mw for k in kids))
    # every site counted exactly once per level; total = whole array
    total = scen.sum(axis=1) / 1000.0
    for level in ("feeder", "locality", "county"):
        np.testing.assert_allclose(sum(zs[z.zone_id] for z in agg.level(level)), total, rtol=1e-9)
    assert by_id["CT/Medgidia"].parent_id == "CT" and by_id["FCT10"].parent_id == "CT/Medgidia"
    for z in agg.zones:
        assert np.all(np.diff(np.vstack([z.q10_mw, z.q50_mw, z.q90_mw]), axis=0) >= -1e-12)


@pytest.mark.parametrize("t0", [T0_WINTER, T0_SUMMER])
def test_peak_hour_on_mock(t0):
    sites, feeders = demo_hierarchy()
    agg = aggregate_scenarios(mock_load_scenarios(sites, t0, 100, seed=1), sites, feeders, t0=t0)
    for z in agg.zones:
        assert z.peak_hour_local in (19, 20), (z.zone_id, z.peak_hour_local)  # mock evening peak 19:30 local
        assert z.peak_mw == pytest.approx(max(z.q50_mw))
        assert z.stress == pytest.approx(z.p90_peak_mw / z.capacity_mw)
    feeders_ = agg.level("feeder")
    assert max(feeders_, key=lambda z: z.stress).zone_id == "FCT10"  # the congested feeder (g_max 55 kW)


def test_missing_site_and_capacity_options():
    sites, feeders = demo_hierarchy()
    scen = mock_load_scenarios(sites, T0_WINTER, 50, seed=2)
    scen[:, 3, :] = np.nan  # site with no data at all
    scen[:10, 7, 40:50] = np.nan  # a gap in some scenarios
    agg = aggregate_scenarios(scen, sites, feeders, t0=T0_WINTER, capacity="limit")
    z = agg.by_id()[sites[3]["feeder_id"]]
    assert z.n_missing_sites == 1
    assert all(np.isfinite(zz.q90_mw).all() for zz in agg.zones)
    assert z.capacity_mw == pytest.approx(0.08)  # tau = 1 -> limit = g_max
    _, zones = build_hierarchy(sites, None, capacity={"CT": 1.0})  # no feeder table: default per-site proxy
    caps = {zz.zone_id: zz.capacity_mw for zz in zones}
    assert caps["CT"] == 1.0 and caps["FCT00"] == pytest.approx(20 * 4.0 / 1000)
    with pytest.raises(ValueError):
        aggregate_scenarios(scen[:, :5], sites, feeders, t0=T0_WINTER)


def test_rows_and_actions(demo):
    *_, agg = demo
    rows = zone_demand_rows(agg, run_id="r1")
    assert len(rows) == len(agg.zones) * 96
    assert set(rows[0]) == set(ROW_FIELDS)
    feeder_rows = [r for r in rows if r["zone_id"] == "FCT00"]
    assert [r["step"] for r in feeder_rows] == list(range(96))  # t0 is local midnight
    assert sum(r["is_peak_hour"] for r in feeder_rows) == 4
    assert feeder_rows[0]["ts"] == T0_WINTER and feeder_rows[0]["t"] == 1736892000
    assert all(0 <= r["peak_stress"] <= 1 and r["label"] == "synthetic RO" for r in rows)
    json.dumps(rows)
    actions = zone_demand_actions(agg, run_id="r1")
    assert len(actions) == len(agg.zones)
    for a in actions:
        assert set(PAYLOAD_KEYS["zone_demand"]) <= set(a.payload)
        assert a.payload["level"] in ("county", "locality", "feeder")
        assert Action.from_json(a.to_json()).payload["zone_id"] == a.payload["zone_id"]
    assert {a.audit.status for a in actions if a.payload["zone_id"] == "FCT10"} == {"stressed"}
    p = agg.by_id()["CT"].to_prediction()
    assert isinstance(p, Prediction) and p.unit == "MW" and p.model == "demandnet"


# --- calendar --------------------------------------------------------------------------------------------

def test_orthodox_easter():
    assert orthodox_easter(2023) == date(2023, 4, 16)
    assert orthodox_easter(2024) == date(2024, 5, 5)
    assert orthodox_easter(2025) == date(2025, 4, 20)
    assert orthodox_easter(2026) == date(2026, 4, 12)


def test_ro_holidays():
    h25 = ro_public_holidays(2025)
    for d in [(1, 1), (1, 2), (1, 6), (1, 7), (1, 24), (4, 18), (4, 20), (4, 21), (5, 1), (6, 1), (6, 8), (6, 9),
              (8, 15), (11, 30), (12, 1), (12, 25), (12, 26)]:
        assert date(2025, *d) in h25, d
    assert date(2023, 1, 6) not in ro_public_holidays(2023)  # Boboteaza / Sf. Ioan only since 2024
    assert date(2024, 6, 23) in ro_public_holidays(2024) and date(2024, 6, 24) in ro_public_holidays(2024)  # Rusalii


def test_bucharest_time_and_calendar_features():
    ts = np.array(["2025-03-30T00:59:00", "2025-03-30T01:00:00", "2025-10-26T00:59:00", "2025-10-26T01:00:00",
                   "2025-01-15T17:30:00"], dtype="datetime64[s]")
    assert bucharest_offset_hours(ts).tolist() == [2, 3, 3, 2, 2]
    cal = calendar_features(ts)
    assert cal["hour"][-1] == 19 and cal["dow"][-1] == 2  # Wed 15 Jan 2025, 19:30 local
    easter = calendar_features(np.array(["2025-04-20T10:00:00"], dtype="datetime64[s]"))
    assert easter["holiday"][0] == 1 and easter["daytype"][0] == 2


# --- v2 --------------------------------------------------------------------------------------------------

def test_v2_trains_fast_on_small_synthetic(tmp_path):
    t = time.time()
    s = synthetic_ro_load("2024-01-01", "2024-05-15", seed=3)
    split = default_split(s.ts)
    model = DemandNet(num_boost_round=120, early_stopping_rounds=20)
    res = evaluate_zone(s, model, split=split)
    assert time.time() - t < 30
    assert res["label"] == "synthetic RO" and res["n_test"] > 24 * 20
    for k in ("demand_nmae", "demand_peak_hour_err", "baseline_persistence_day_nmae",
              "baseline_persistence_week_nmae", "baseline_shape_scaling_nmae"):
        assert np.isfinite(res[k]), k
    assert res["demand_nmae"] < 0.1
    assert res["demand_nmae"] < res["baseline_persistence_day_nmae"]
    assert 0.4 < res["demand_coverage_10_90"] <= 1.0
    # save / load round trip and a day-ahead envelope
    model.save(tmp_path)
    m2 = DemandNet.load(tmp_path)
    t0 = str(s.ts[-48]) + "Z"
    p1, p2 = model.predict_day(s, t0, horizon=96), m2.predict_day(s, t0, horizon=96)
    assert p1.unit == "MW" and p1.horizon == 96
    np.testing.assert_allclose(p1.q50, p2.q50, rtol=1e-6)


def test_metrics_basic():
    y = np.array([1.0, 2, 3, 4])
    assert nmae(y, y) == 0 and nmae(y, np.zeros(4)) == pytest.approx(1.0)
    day = np.array(["2025-01-01"] * 24, dtype="datetime64[D]")
    hour = np.arange(24)
    yy = np.sin(np.arange(24) / 24 * np.pi)
    assert peak_hour_error(yy, np.roll(yy, 2), day, hour) == 2


def test_entsoe_long_format_loader(tmp_path):
    pl = pytest.importorskip("polars")
    from datetime import datetime, timedelta

    start = datetime(2025, 1, 1)
    ts = [start + timedelta(minutes=15 * k) for k in range(4 * 24 * 3)]
    mw = [6000.0 + 10 * (k % 4) for k in range(len(ts))]
    df = pl.DataFrame({"ts": ts, "series": ["load_actual"] * len(ts), "mw": mw}).with_columns(
        pl.col("ts").dt.replace_time_zone("UTC"))
    df = pl.concat([df, df.with_columns(pl.lit("gen_wind").alias("series"), pl.lit(1.0).alias("mw"))])
    df.write_parquet(tmp_path / "entsoe_ro.parquet")
    s = load_national(tmp_path)
    assert s.label == "real RO" and s.temp_label == "synthetic RO"
    assert len(s.ts) == 72 and np.allclose(s.load_mw, 6015.0)  # 15 min -> hourly mean, wind rows ignored
    with pytest.raises(FileNotFoundError):
        load_national(tmp_path / "empty", allow_synthetic=False)


def test_national_shape_and_quick_main(tmp_path, capsys):
    s = synthetic_ro_load("2024-01-01", "2024-03-01")
    shp = NationalShape.fit(s)
    assert shp.table.shape == (12, 3, 24) and np.isfinite(shp.table).all()
    main(["--quick", "--out", str(tmp_path), "--data-dir", str(tmp_path / "none")])
    out = capsys.readouterr().out
    for k in ("demand_nmae", "demand_peak_hour_err", "baseline_persistence_day_nmae", "baseline_shape_scaling_nmae"):
        assert f"LAB_METRIC {k}=" in out
    rep = json.loads((tmp_path / "demand_report.json").read_text())
    assert rep["label"] == "synthetic RO" and (tmp_path / "demandnet_meta.json").exists()
    assert (tmp_path / "zone_demand_v1_rows.json").exists()
