"""Portfolio baselines (profile / persistence / forecast_only) on a tiny synthetic portfolio."""

import numpy as np
import pandas as pd
import polars as pl
import pytest

from wattsentry.eval import baselines_portfolio as bp


def _portfolio(n=3, year=2025, gaps=None):
    ts = bp.year_index(year)
    prev = bp.year_index(year - 1)
    lead = 2 * bp.WEEK
    L = lead + len(ts)
    rng = np.random.default_rng(0)
    slot = np.arange(L) % bp.H
    meter = np.stack([0.5 + 0.3 * np.sin(slot / 96 * 2 * np.pi) + 0.1 * i + 0.01 * rng.standard_normal(L)
                      for i in range(n)])
    if gaps:
        for s, t in gaps:
            meter[s, t] = np.nan
    prev_meter = np.full((n, len(prev)), 0.4)
    clim = np.zeros((12, bp.H))
    clim[:, 36:64] = 0.5  # 09:00-16:00 UTC, 0.5 kW per kWp
    sites = pl.DataFrame({"site_id": [f"S{i}" for i in range(n)], "locality_id": ["bragadiru", "bradu", "nowhere"][:n],
                          "kwp_recorded": [5.0, 3.0, 4.0][:n]}).with_columns(
        zone=pl.when(pl.col("locality_id").is_in(list(bp.URBAN_LOCALITIES))).then(pl.lit("urban")).otherwise(pl.lit("rural")),
        smart=pl.lit(True))
    return bp.Portfolio(sites=sites, year=year, ts=ts, meter_kw=meter, prev_ts=prev, prev_meter_kw=prev_meter,
                        actual_kw=meter[:, lead:].copy(), clim={"bragadiru": clim, "bradu": clim}, lead=lead)


def test_year_index_and_standard_profile():
    assert len(bp.year_index(2025)) == 35040 and len(bp.year_index(2024)) == 35136
    assert str(bp.year_index(2025)[0]) == "2024-12-31 22:00:00+00:00"
    p = bp.standard_profile()
    assert p.height == 2 * 2 * 2 * 96
    sums = p.group_by("zone", "season", "day_type").agg(pl.col("weight").sum())["weight"].to_numpy()
    np.testing.assert_allclose(sums, 1.0, atol=1e-6)
    assert set(p["zone"]) == {"urban", "rural"} and (p["weight"] > 0).all()


def test_psc_shares_follow_the_deer_formula():
    ts = bp.year_index(2025)
    w = bp.psc_step_weights(ts, "urban")
    month, share = bp.monthly_shares(ts, w)
    tot = np.zeros(12)
    np.add.at(tot, month, share)
    np.testing.assert_allclose(tot, 1.0, atol=1e-9)
    # In March 2025 (cold season): one workday's energy / one non-workday's = r_SR
    loc = ts.tz_convert(bp.TZ)
    day = lambda d: share[(loc.month == 3) & (loc.day == d)].sum()  # 2025-03-03 Mon, 2025-03-02 Sun
    r = bp.standard_profile().filter((pl.col("zone") == "urban") & (pl.col("season") == "cold"))["r_workday"][0]
    assert day(3) / day(2) == pytest.approx(r, rel=1e-9)


def test_profile_keeps_monthly_net_and_has_no_weather():
    pf = _portfolio()
    kw = bp.profile_site_kw(pf)
    assert kw.shape == (3, len(pf.ts)) and np.isfinite(kw).all()
    m = pf.ts.tz_convert(bp.TZ).month.to_numpy() - 1
    # monthly net of the profile = last year's metered net in that month (0.4 kW flat) x hours of this year's month
    for k in (0, 6):
        got = kw[:, m == k].sum(1) * bp.STEP_H
        prev_m = pf.prev_ts.tz_convert(bp.TZ).month.to_numpy() - 1
        np.testing.assert_allclose(got, 0.4 * (prev_m == k).sum() * bp.STEP_H, rtol=1e-9)
    # site without a solar curve (unknown locality) gets no solar subtraction -> never negative
    assert (kw[2] >= 0).all() and (kw[0] < 0).any()
    # the profile ignores the test-year meters entirely
    pf2 = _portfolio()
    pf2.meter_kw[:] = 99.0
    np.testing.assert_array_equal(bp.profile_site_kw(pf2), kw)


def test_persistence_uses_last_week_then_fallbacks():
    lead = 2 * bp.WEEK
    t = 1000
    pf = _portfolio(gaps=[(0, lead + t - bp.WEEK), (1, lead + t - bp.WEEK), (1, lead + t - 2 * bp.WEEK)])
    kw = bp.persistence_site_kw(pf, fallback=np.full((3, len(pf.ts)), -7.0))
    assert kw[2, t] == pf.meter_kw[2, lead + t - bp.WEEK]
    assert kw[0, t] == pf.meter_kw[0, lead + t - 2 * bp.WEEK]
    assert kw[1, t] == -7.0
    assert np.isfinite(kw).all()


def test_profile_metered_homes_get_the_profile_in_persistence():
    pf = _portfolio()
    pf.sites = pf.sites.with_columns(smart=pl.Series([True, False, True]))
    fb = np.full((3, len(pf.ts)), -3.0)
    kw = bp.persistence_site_kw(pf, fallback=fb)
    assert (kw[1] == -3.0).all() and (kw[0] != -3.0).all()


def test_forecast_only_is_load_minus_pv_with_fallback(monkeypatch):
    pf = _portfolio()
    S, T = 3, len(pf.ts)
    pv = np.zeros((S, T, 3))
    pv[..., 1] = 1.0
    load = np.zeros((S, T, 3))
    load[..., 1] = 2.5
    load[0, :100] = np.nan  # model has no output -> fallback
    calls = []

    def fake_validated(model, ids, days, q):
        calls.append((model, len(ids), len(days), q.shape))
        return q[..., 1]

    monkeypatch.setattr(bp, "validated_q50", fake_validated)
    kw, info = bp.forecast_only_site_kw(pf, {"pv": pv, "load": load}, fallback=np.full((S, T), 9.0))
    assert np.allclose(kw[:, 200:], 1.5)
    assert (kw[0, :100] == 9.0).all() and info["model_coverage"] < 1.0
    n_days = len(np.unique(pf.ts.tz_convert(bp.TZ).date))
    assert [c[0] for c in calls] == ["pvnet", "loadnet"] and calls[0][1] == S * n_days and calls[0][3] == (S * n_days, 96, 3)


def test_validated_q50_passthrough_without_judge():
    q = np.random.default_rng(1).random((4, 96, 3))
    np.testing.assert_allclose(bp.validated_q50("pvnet", ["a"] * 4, ["2025-06-01"] * 4, q), q[..., 1])


def test_position_frame_contract_and_sign():
    pf = _portfolio()
    pf.actual_kw[0, :] = -4.0  # home 0 exports 4 kW all year
    df = bp.profile_position(pf)
    assert dict(df.schema) == bp.POSITION_COLUMNS
    assert df.height == len(pf.ts) and df["policy"].unique().to_list() == ["profile"]
    assert (df["traded_id_mwh"] == 0).all() and df["bought_da_mwh"].is_finite().all()
    np.testing.assert_allclose(df["actual_mwh"], df["import_mwh"] - df["export_mwh"])
    np.testing.assert_allclose(df["export_mwh"], 4.0 * bp.STEP_H / 1000)  # gross per-home export, MWh per step
    assert df["ts"][0] == pd.Timestamp("2024-12-31 22:00", tz="UTC")
    for pol in ("persistence",):
        assert bp.positions(pol, fleet=pf).height == len(pf.ts)
    with pytest.raises(ValueError):
        bp.positions("nope", fleet=pf)
