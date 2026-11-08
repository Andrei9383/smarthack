"""avoidable_cost (consumption and producer sign) and the 15-min price table."""

import json
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from wattsentry.eval.imbalance import AB_PATH, AFEER_HI, AFEER_LO, avoidable_cost, cost_steps, load_ab, q_star

PZU = np.array([400.0, 500.0, 600.0, 300.0])
PI = np.array([450.0, 480.0, 700.0, 300.0])
A, B = 0.5, 0.25


def test_perfect_forecast_is_zero():
    actual = np.array([1.0, 2.0, -0.5, 0.3])
    r = avoidable_cost(actual, actual, np.zeros(4), PZU, PI, A, B)
    assert r["avoidable_lei"] == 0 and r["imbalance_lei"] == 0 and r["intraday_lei"] == 0 and r["abs_dev_mwh"] == 0
    # Perfect split between day-ahead and intraday: no imbalance, intraday priced at pi - pzu only.
    r = avoidable_cost(actual, actual - 0.2, np.full(4, 0.2), PZU, PZU, A, B)
    assert r["avoidable_lei"] == pytest.approx(0, abs=1e-9)


def test_only_short():
    actual = np.array([2.0, 2.0, 2.0, 2.0])
    bought = np.array([1.0, 1.5, 2.0, 2.0])  # short 1.0 and 0.5 MWh
    r = avoidable_cost(actual, bought, None, PZU, None, A, B)
    assert r["short_mwh"] == pytest.approx(1.5) and r["long_mwh"] == 0
    assert r["imbalance_lei"] == pytest.approx(A * (400 * 1.0 + 500 * 0.5))
    assert r["intraday_lei"] == 0 and r["avoidable_lei"] == r["imbalance_lei"]


def test_only_long():
    actual = np.array([1.0, 1.0, 1.0, 1.0])
    bought = np.array([1.0, 1.0, 3.0, 1.5])  # long 2.0 and 0.5 MWh
    r = avoidable_cost(actual, bought, None, PZU, None, A, B)
    assert r["long_mwh"] == pytest.approx(2.5) and r["short_mwh"] == 0
    assert r["imbalance_lei"] == pytest.approx(B * (600 * 2.0 + 300 * 0.5))


def test_hand_computed_case():
    # step: actual, bought, traded -> dev
    # 0: 3.0 - 2.0 - 0.5 = +0.5 short at 400 -> 0.5*0.5*400 = 100; intraday 0.5*(450-400) = 25
    # 1: 1.0 - 2.0 - 0.0 = -1.0 long  at 500 -> 0.25*1*500 = 125; intraday 0
    # 2: 2.0 - 1.0 + 1.0 = +2.0 (sold 1 on PI at 700): short 2 at 600 -> 600; intraday -1*(700-600) = -100
    # 3: 0.0 - 0.0 - 0.0 = 0
    actual = np.array([3.0, 1.0, 2.0, 0.0])
    bought = np.array([2.0, 2.0, 1.0, 0.0])
    traded = np.array([0.5, 0.0, -1.0, 0.0])
    r = avoidable_cost(actual, bought, traded, PZU, PI, A, B)
    assert r["imbalance_lei"] == pytest.approx(100 + 125 + 600)
    assert r["intraday_lei"] == pytest.approx(25 - 100)
    assert r["avoidable_lei"] == pytest.approx(825 - 75)
    assert r["abs_dev_mwh"] == pytest.approx(3.5)
    # Equals total cost minus perfect-foresight cost (everything at PZU).
    dev = actual - bought - traded
    pay = bought * PZU + traded * PI + np.clip(dev, 0, None) * PZU * (1 + A) - np.clip(-dev, 0, None) * PZU * (1 - B)
    assert r["avoidable_lei"] == pytest.approx(float(pay.sum() - (actual * PZU).sum()))


def test_negative_price_never_rewards_imbalance():
    r = avoidable_cost(np.array([1.0, 0.0]), np.array([0.0, 1.0]), None, np.array([-100.0, -50.0]), None, A, B)
    assert r["imbalance_lei"] == pytest.approx(0.5 * 100 * 1 + 0.25 * 50 * 1) and r["imbalance_lei"] > 0


def wind_avoidable_cost(prod, sold_da, sold_id, pzu, pi, a, b):
    """Producer side: production enters as negative demand."""
    return avoidable_cost(-prod, -sold_da, -sold_id, pzu, pi, a, b)


def test_wind_sign():
    prod = np.array([10.0, 10.0, 10.0])
    sold_da = np.array([12.0, 8.0, 10.0])   # step 0 sold 2 MWh not produced (short), step 1 2 MWh unsold (long)
    sold_id = np.array([0.0, 0.0, 0.0])
    r = wind_avoidable_cost(prod, sold_da, sold_id, PZU[:3], PI[:3], A, B)
    assert r["short_mwh"] == pytest.approx(2.0) and r["long_mwh"] == pytest.approx(2.0)
    assert r["imbalance_lei"] == pytest.approx(A * 400 * 2 + B * 500 * 2)
    # Perfect forecast = 0; buying back the 2 MWh on PI at step 0 removes the short but costs (pi - pzu) on 2 MWh.
    assert wind_avoidable_cost(prod, prod, sold_id, PZU[:3], PI[:3], A, B)["avoidable_lei"] == 0
    r3 = wind_avoidable_cost(prod[:1], sold_da[:1], np.array([-2.0]), PZU[:1], PI[:1], A, B)
    assert r3["short_mwh"] == 0 and r3["intraday_lei"] == pytest.approx(2.0 * (450 - 400))


def test_broadcast_axis_and_scale_invariance():
    rng = np.random.default_rng(0)
    actual = rng.normal(1, 0.3, (5, 96))
    bought = actual + rng.normal(0, 0.2, (5, 96))
    pzu = rng.uniform(200, 800, 96)
    r = avoidable_cost(actual, bought, None, pzu, None, A, B, axis=-1)
    assert r["avoidable_lei"].shape == (5,)
    assert r["avoidable_lei"].sum() == pytest.approx(avoidable_cost(actual, bought, None, pzu, None, A, B)["avoidable_lei"])
    # a = b: ratio of two policies' costs does not depend on the value.
    b2 = actual + rng.normal(0, 0.1, (5, 96))
    ratio = lambda k: (avoidable_cost(actual, b2, None, pzu, None, k, k)["avoidable_lei"]  # noqa: E731
                       / avoidable_cost(actual, bought, None, pzu, None, k, k)["avoidable_lei"])
    assert ratio(0.3) == pytest.approx(ratio(1.7))
    assert set(cost_steps(actual, bought, None, pzu, None, A, B)) >= {"dev", "short", "long", "avoidable"}


def test_frozen_ab():
    a, b = load_ab()
    d = json.loads(AB_PATH.read_text())
    assert a >= b > 0
    assert q_star() == pytest.approx(a / (a + b))
    assert AFEER_LO <= d["calibration"]["lei_per_mwh_injected"] <= AFEER_HI
    r = avoidable_cost(np.ones(2), np.zeros(2), None, np.full(2, 100.0), None)  # defaults to the frozen values
    assert r["imbalance_lei"] == pytest.approx(a * 200)


# ---- prices -------------------------------------------------------------------------------------------------------
from wattsentry.data.prices import load_prices, prices_at  # noqa: E402


def test_prices_15min_and_hourly_repeat():
    # PZU switched to 15 min on 2025-10-01 00:00 CEST = 2025-09-30 22:00 UTC (#186)
    df = load_prices("2025-09-30T19:00", "2025-09-30T23:00")
    assert df.height == 16
    assert (df["ts"].diff().drop_nulls().dt.total_minutes() == 15).all()
    hourly = df.filter(pl.col("pzu_res") == "PT60M_x4")
    assert hourly.height == 12 and hourly["pzu_lei_mwh"].n_unique() == 3  # 3 hours x 4 identical quarters
    assert df.filter(pl.col("pzu_res") == "PT15M").height == 4
    assert (df["pi_lei_mwh"] == df["pzu_lei_mwh"]).all() or (df["pi_src"] == "opcom_pi").any()


def test_prices_full_and_dst():
    df = load_prices()
    assert df["ts"].is_unique().all()
    assert (df["ts"].diff().drop_nulls().dt.total_minutes() == 15).all()  # no gaps, UTC
    # Local DST days: 92 and 100 quarter-hours.
    loc = df.with_columns(pl.col("ts").dt.convert_time_zone("Europe/Bucharest").dt.date().alias("d"))
    n = dict(loc.group_by("d").len().iter_rows())
    from datetime import date
    assert n[date(2025, 3, 30)] == 92 and n[date(2025, 10, 26)] == 100 and n[date(2025, 6, 1)] == 96


def test_prices_at_alignment_and_pi_pickup(tmp_path: Path):
    ts = np.arange(np.datetime64("2025-11-01T10:00"), np.datetime64("2025-11-01T11:00"), np.timedelta64(15, "m"))
    pzu, pi = prices_at(ts)
    ref = load_prices("2025-11-01T10:00", "2025-11-01T11:00")["pzu_lei_mwh"].to_numpy()
    assert np.allclose(pzu, ref)
    with pytest.raises(KeyError):
        prices_at(np.array([np.datetime64("2030-01-01T00:00")]))
    # I1's file is picked up when present; missing quarter-hours fall back to pzu.
    pi_file = tmp_path / "prices_pi.parquet"
    pl.DataFrame({"ts": pl.Series(ts[:2].astype("datetime64[us]")).dt.replace_time_zone("UTC"),
                  "pi_vwap_lei_mwh": [1.0, 2.0]}).write_parquet(pi_file)
    pzu2, pi2 = prices_at(ts, pi_path=pi_file)
    assert np.allclose(pzu2, pzu) and pi2[:2].tolist() == [1.0, 2.0] and np.allclose(pi2[2:], pzu[2:])
    src = load_prices("2025-11-01T10:00", "2025-11-01T11:00", pi_path=pi_file)["pi_src"].to_list()
    assert src == ["opcom_pi", "opcom_pi", "pzu", "pzu"]
