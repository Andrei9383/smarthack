"""Wind simulator: determinism, contract tables, physics numbers, real inputs, splits (requirements.txt only).

Small worlds only (5 turbines per park, a few weeks) so the whole file runs in a few seconds.
"""

import dataclasses
import json

import numpy as np
import polars as pl
import pytest

from wattsentry.contracts import is_error_code, parse_tag
from wattsentry.contracts.schemas import check_columns
from wattsentry.contracts.wind import METEO_COLUMNS, SENSORS, TELEMETRY_LONG_COLUMNS, TURBINE_META_COLUMNS
from wattsentry.model.wind.data import make_splits, read_wind_dir
from wattsentry.model.wind.physics import R_D, RHO_REF
from wattsentry.sim.wind.build import build, simulate
from wattsentry.sim.wind.config import RAW, WindSimConfig
from wattsentry.sim.wind.generate import generate_clean, national_wind_cf
from wattsentry.sim.wind.parks import PARKS, build_layouts
from wattsentry.sim.wind_faults import FaultConfig

SMALL = WindSimConfig(start="2024-01-01", end="2024-01-22", max_turbines_per_park=5)


@pytest.fixture(scope="module")
def sim():
    return simulate(SMALL, log=lambda *a: None)


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    out = tmp_path_factory.mktemp("wind") / "wind"
    build(SMALL, out_dir=out, plots=True, log=lambda *a: None)
    return out


def test_deterministic(sim):
    clean, res, data = sim
    _, res2, data2 = simulate(SMALL, log=lambda *a: None)
    for k in SENSORS:
        np.testing.assert_array_equal(data.sensors[k], data2.sensors[k])
    np.testing.assert_array_equal(data.err, data2.err)
    assert res.episodes == res2.episodes
    other = generate_clean(dataclasses.replace(SMALL, seed=5), log=lambda *a: None)
    assert not np.array_equal(other.avail, clean.avail)


def test_build_is_cached_and_reproducible(built, tmp_path):
    man = json.loads((built / "manifest.json").read_text())
    msgs = []
    build(SMALL, out_dir=built, log=msgs.append)  # same digest -> cache hit
    assert any("cached" in m for m in msgs)
    again = tmp_path / "w2"
    build(SMALL, out_dir=again, plots=False, quality=False, log=lambda *a: None)
    a = pl.read_parquet(built / "wind_telemetry.parquet")
    b = pl.read_parquet(again / "wind_telemetry.parquet")
    assert a.equals(b)
    assert json.loads((again / "manifest.json").read_text())["digest"] == man["digest"]


def test_contract_tables(built):
    tel = pl.read_parquet(built / "wind_telemetry.parquet")
    assert list(tel.columns) == list(TELEMETRY_LONG_COLUMNS)
    assert tel["ts"].dtype == pl.Datetime("us", "UTC") and tel["value"].dtype == pl.Float64
    assert not check_columns("wind_telemetry", tel.columns)
    sensors = set()
    for tag in tel["tag"].unique().to_list():
        t = parse_tag(tag, strict=True)
        sensors.add(t.sensor)
    assert sensors == set(SENSORS)
    sample = tel.drop_nulls("value").sample(5000, seed=0).to_dicts()
    for r in sample:
        assert is_error_code(parse_tag(r["tag"]).sensor, r["value"]) == r["is_code"], r
    assert tel["is_code"].any()  # coded comm gaps present

    tur = pl.read_parquet(built / "turbines.parquet")
    assert not check_columns("turbines", tur.columns) and set(TURBINE_META_COLUMNS) <= set(tur.columns)
    assert tur["turbine_id"].str.contains(r"^WP[123]\.TID\d{2}$").all()
    assert set(tur["model"]) == {"generic-2.5MW", "generic-6.1MW"}

    met = pl.read_parquet(built / "meteo_wind.parquet")
    assert set(METEO_COLUMNS) <= set(met.columns)
    one = met.filter(pl.col("turbine_id") == tur["turbine_id"][0]).sort("ts")
    assert set(one["ts"].diff().drop_nulls().dt.total_minutes().unique().to_list()) == {5}

    w = pl.read_parquet(built / "weather.parquet")
    assert not check_columns("weather", w.columns)
    assert set(w["point_id"]) == {"WP1", "WP2", "WP3"}
    assert w["rh_pct"].is_not_null().all() and w["precip_mm"].is_not_null().all()

    fl = pl.read_parquet(built / "fault_labels.parquet")
    assert not check_columns("fault_labels", fl.columns)
    assert set(fl["asset_kind"]) == {"turbine"}


def test_m6_loader_reads_cache(built, sim, monkeypatch):
    """wattsentry.model.wind.data picks the cache up in exactly the contract format (M6 unchanged)."""
    from wattsentry.model.wind.data import external_wind_dir, load_wind

    _, _, data = sim
    monkeypatch.setenv("WATTSENTRY_WIND_DIR", str(built))
    assert external_wind_dir() == built
    back = load_wind(log=lambda *a: None)
    assert back.turbine_ids == data.turbine_ids and len(back.ts) == len(data.ts)
    for k in SENSORS:  # files are rounded to 0.01
        np.testing.assert_allclose(back.sensors[k], data.sensors[k], atol=6e-3, equal_nan=True)
    np.testing.assert_array_equal(back.err, data.err)
    np.testing.assert_allclose(back.meteo["wind_ms"], data.meteo["wind_ms"], atol=6e-3)
    np.testing.assert_array_equal(back.labels, data.labels)
    assert back.weather["rh_pct"].shape == (len(data.ts), 3) and np.isfinite(back.weather["rh_pct"]).all()


def test_power_within_rated(sim):
    clean, _, data = sim
    assert (clean.avail <= clean.rated[None]).all() and (clean.avail >= 0).all()
    p = data.sensors["act_pwt"]
    assert np.nanmax(p / clean.rated[None]) <= 1.0
    assert np.nanmin(p) > -20  # stopped turbines draw a few kW of auxiliary power


def test_density_and_shear_numbers(sim):
    clean, _, _ = sim
    q = clean.weather.q15
    p_turb = q["pressure_hpa"][:, clean.park_of] * 100.0
    rho = p_turb / (R_D * (clean.amb.astype(np.float64) + 273.15))
    np.testing.assert_allclose(clean.rho, rho, rtol=1e-6)
    np.testing.assert_allclose(clean.v_norm, clean.v_eff * (clean.rho / RHO_REF) ** (1 / 3), rtol=1e-5)
    assert 1.15 < clean.rho.mean() < 1.35
    # shear exponent: re-fit from the raw ERA5 file for one park / season, independently of the simulator
    w = (pl.read_parquet(RAW / "weather.parquet")
         .filter(pl.col("point_id") == "park:fantanele_cogealac", pl.col("ts").dt.year() == 2024,
                 pl.col("ts").dt.month().is_in([12, 1, 2]), pl.col("wind100_ms") > 4, pl.col("wind10_ms") > 1))
    a = float(np.clip(np.median(np.log(w["wind100_ms"].to_numpy() / w["wind10_ms"].to_numpy()) / np.log(10)), .08, .3))
    assert clean.weather.alpha_table["WP1"]["DJF"] == pytest.approx(a, abs=1e-4)
    for park in clean.weather.alpha_table.values():
        assert all(0.08 <= v <= 0.30 for v in park.values())
    # free hub wind / meteo: the 5-min meteo series averages to the 15-min free-stream hub wind (no wake)
    m15 = clean.meteo15["wind_ms"]
    assert np.corrcoef(m15.ravel(), clean.v_free.ravel())[0, 1] > 0.99
    assert clean.v_eff.mean() < clean.v_free.mean()  # wakes + turbulence never add energy on average


def test_capacity_factor_calibrated_to_real_ro():
    cf = national_wind_cf(2024)
    assert cf is not None and 0.15 < cf < 0.35
    clean = generate_clean(dataclasses.replace(SMALL, end="2024-01-03"), log=lambda *a: None)
    assert clean.cf_target == pytest.approx(cf) and 0.7 < clean.k_speed < 1.5
    clean2 = generate_clean(dataclasses.replace(SMALL, end="2024-01-03", cf_calibration=False), log=lambda *a: None)
    assert clean2.k_speed == 1.0


def test_parks_on_real_osm_positions():
    lays = build_layouts([p.park_id for p in PARKS])
    assert [lay.n for lay in lays] == [70, 36, 15]
    osm = pl.read_parquet(RAW / "osm_wind_turbines.parquet")
    for lay in lays:
        rows = osm.filter(pl.col("osm_id").is_in(lay.osm_id.tolist()))
        assert rows.height == lay.n
        d = dict(zip(rows["osm_id"].to_list(), zip(rows["lat"].to_list(), rows["lon"].to_list())))
        assert all(d[o] == (la, lo) for o, la, lo in zip(lay.osm_id.tolist(), lay.lat, lay.lon))
    assert lays[2].park.county == "Iași" and 47.1 < lays[2].lat.mean() < 47.5


def test_held_out_park_and_time_split():
    cfg = dataclasses.replace(SMALL, start="2024-12-20", end="2025-01-10")
    _, _, data = simulate(cfg, FaultConfig().only("fault_stop"), log=lambda *a: None)
    sp = make_splits(data)
    assert sp.holdout_park == "WP2" and sp.test_year == 2025
    park = np.array([t["park_id"] for t in data.turbines])
    assert set(park[sp.holdout_turbines]) == {"WP2"} and "WP2" not in set(park[sp.train_turbines])
    assert not (sp.train_t & sp.test_t).any() and not (sp.val_t & sp.test_t).any()
    years = data.ts.astype("datetime64[Y]").astype(int) + 1970
    assert (years[sp.test_t] == 2025).all() and (years[sp.train_t | sp.val_t] == 2024).all()


def test_power_scale_mw_parks_roundtrip(tmp_path):
    cfg = dataclasses.replace(SMALL, end="2024-01-04", mw_parks=("WP3",))
    out = build(cfg, FaultConfig().only("fault_stop"), out_dir=tmp_path / "w", plots=False, quality=False,
                log=lambda *a: None)
    tur = pl.read_parquet(out / "turbines.parquet")
    assert set(tur.filter(pl.col("park_id") == "WP3")["power_unit"]) == {"MW"}
    tel = pl.read_parquet(out / "wind_telemetry.parquet").filter(pl.col("tag").str.ends_with("act_pwt"))
    wp3 = tel.filter(pl.col("tag").str.starts_with("WP3"))["value"].drop_nulls()
    assert wp3.max() <= 6.1 + 1e-6  # MW
    back = read_wind_dir(out, log=lambda *a: None)
    j = [i for i, t in enumerate(back.turbines) if t["park_id"] == "WP3"][0]
    assert np.nanmax(back.sensors["act_pwt"][:, j]) > 100  # back in kW


def test_sanity_plot_written(built):
    svg = (built / "plots" / "power_vs_vnorm.svg").read_text()
    assert svg.startswith("<svg") and "WP3" in svg
    bins = pl.read_csv(built / "plots" / "power_curve_bins.csv")
    clean = bins.filter(pl.col("series") == "clean", pl.col("park_id") == "WP1").sort("v_norm_ms")
    assert clean["p_frac"].head(3).max() < 0.05 and clean.filter(pl.col("v_norm_ms") > 13)["p_frac"].min() > 0.9
