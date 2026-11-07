"""Trust ledger (#147, team/judge-design.md §6): invariants and rules on small hand-made outcome rows."""

from __future__ import annotations

import datetime as dt
import math
import sys
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from sentinel.judge.buckets import ancestors, bucket_key
from sentinel.judge.ledger import (ALERT_DAYS, HALF_LIFE, MIN_EVIDENCE, N0, Ledger, Priors, TRUST_COLUMNS,
                                   _alert_state, beta_interval)

D0 = dt.date(2025, 5, 1)
B = bucket_key("site", seen=True, battery=False, season="warm", quality="clean", ood_high=False)
B_UNSEEN = bucket_key("site", seen=False, battery=False, season="warm", quality="clean", ood_high=False)
PRIOR = 0.8


def day(i: int) -> str:
    return str(D0 + dt.timedelta(days=i))


def rows(scores, model="pvnet", version="v1", score="calibration", bucket=B, assets=1, lag=1, weight=1.0,
         start=0, gated=0):
    """One outcome row per asset per day i (start..), observed `lag` days later."""
    out = []
    for i, s in enumerate(scores):
        for a in range(assets):
            out.append(dict(run_id="t", model=model, model_version=version, asset_id=f"A{a}", day=day(start + i),
                            score_name=score, score=float(s), weight=weight, source="telemetry",
                            observed_at=day(start + i + lag), gated=gated, bucket=bucket))
    return out


def priors(p=PRIOR, extra=None) -> Priors:
    pr = Priors()
    pr.add("pvnet", None, {"calibration": {"*": {"prior": p, "n": 500}}, "sharpness": {"*": {"prior": 0.5, "n": 500}}})
    for (model, version), val in (extra or {}).items():
        pr.add(model, version, {"calibration": {"*": {"prior": val, "n": 500}}})
    return pr


def at(led: Ledger, i: int, bucket=B, version="v1", score="calibration", model="pvnet") -> dict:
    return led.lookup(model, version, score, bucket, day(i))


# ---------------------------------------------------------------- the user's rule

def test_no_record_means_trust_equals_prior():
    led = Ledger([], priors(), start=day(0), end=day(30))
    for b in (B, B_UNSEEN, "turbine|seen|nobatt|cold|gappy|ood_hi", "*"):
        r = at(led, 10, bucket=b)
        assert r["trust"] == PRIOR and r["n_eff"] == 0 and r["band_scale"] == 1.0 and r["health_alert"] == 0
        assert r["lo80"] < PRIOR < r["hi80"]
    # a model with no priors at all is still not distrusted below the neutral default
    assert at(led, 3, model="windfailure")["trust"] == Priors().default


def test_no_record_rows_in_the_trust_table():
    led = Ledger(rows([0.1] * 20), priors(), run_id="r1")
    t = led.trust_rows(extra=[("pvnet", "v9", "calibration", B_UNSEEN)])
    assert t.columns == TRUST_COLUMNS
    new = t.filter(pl.col("model_version") == "v9")
    assert new.height == led.n_days and (new["trust"] == PRIOR).all() and (new["n_eff"] == 0).all()


def test_bucket_prior_comes_from_validation_bucket_else_overall():
    pr = Priors()
    pr.add("pvnet", None, {"calibration": {"*": {"prior": 0.7, "n": 900}, "site|unseen": {"prior": 0.6, "n": 400},
                                           "site|seen|batt": {"prior": 0.95, "n": 3}}})
    assert pr.get("pvnet", "v1", "calibration", B_UNSEEN) == 0.6           # covered parent bucket
    assert pr.get("pvnet", "v1", "calibration", B) == 0.7                  # not covered -> overall
    assert pr.get("pvnet", "v1", "calibration", "site|seen|batt|warm|clean|ood_lo") == 0.7  # n=3 too thin


# ---------------------------------------------------------------- update rule (§6.3)

def test_update_rule_exact():
    s, age = 0.2, 10
    led = Ledger(rows([s]), priors(), end=day(age))
    # below MIN_EVIDENCE the bucket reads the prior; check the level arrays carry the exact formula
    L = led.levels[0]
    sid = L.index[("pvnet", "v1", "calibration", B)]
    w = 0.5 ** (age / HALF_LIFE)
    assert L.n_eff[sid, age] == pytest.approx(w)
    assert L.trust[sid, age] == pytest.approx((N0 * PRIOR + w * s) / (N0 + w))
    assert at(led, age)["trust"] == PRIOR  # n_eff < 5: root fallback = prior


def test_trust_can_move_far_from_prior():
    """n0 = 7 must stay well below the most live evidence the decay allows (≈ 20.7 bucket-days)."""
    led = Ledger(rows([0.0] * 200), priors(0.9))
    r = at(led, 200)
    n_max = 0.5 ** (1 / HALF_LIFE) / (1 - 0.5 ** (1 / HALF_LIFE))  # Σ over ages ≥ 1 (telemetry lands at d+1)
    assert r["n_eff"] == pytest.approx(n_max, rel=0.01) and n_max > 2.5 * N0
    assert r["trust"] == pytest.approx(0.9 * N0 / (N0 + n_max), rel=0.01) and r["trust"] < 0.25
    assert r["hi80"] < 0.45


def test_decay_and_recovery():
    good, bad = [0.85] * 40, [0.2] * 20
    led = Ledger(rows(good + bad + good), priors())
    assert at(led, 40)["trust"] > 0.8
    low = at(led, 61)
    assert low["trust"] < PRIOR - 0.2 and low["health_alert"] == 1
    # within ~2-3 half-lives of being restored, trust is back near its prior and the alert has cleared
    back = at(led, 61 + 28)
    assert back["trust"] > PRIOR - 0.1 and back["health_alert"] == 0
    a = led.alerts().filter(pl.col("bucket") == B)
    assert a.height == 1 and a["since"][0] > D0 + dt.timedelta(days=41) and a["until"][0] is not None


def test_new_version_starts_from_its_own_prior():
    led = Ledger(rows([0.1] * 60), priors(extra={("pvnet", "v2"): 0.66}))
    assert at(led, 60)["trust"] < 0.4
    r = at(led, 60, version="v2")
    assert r["trust"] == 0.66 and r["n_eff"] == 0 and r["health_alert"] == 0
    # v2's own outcomes never touch v1, and v1's never touch v2
    led2 = Ledger(rows([0.1] * 60) + rows([0.9] * 10, version="v2", start=50),
                  priors(extra={("pvnet", "v2"): 0.66}))
    assert at(led2, 60)["trust"] == pytest.approx(at(led, 60)["trust"])
    assert at(led2, 61, version="v2")["trust"] > 0.66


# ---------------------------------------------------------------- evidence units and timing

def test_bucket_day_averaging_counts_a_day_once():
    one = Ledger(rows([0.3] * 30, assets=1), priors())
    many = Ledger(rows([0.3] * 30, assets=50), priors())
    a, b = at(one, 30), at(many, 30)
    assert a["n_eff"] == pytest.approx(b["n_eff"]) and a["trust"] == pytest.approx(b["trust"])
    # the day's score is the mean over the bucket's assets
    mixed = rows([0.0] * 30, assets=1) + [dict(r, asset_id="B", score=1.0) for r in rows([1.0] * 30)]
    assert at(Ledger(mixed, priors()), 30)["trust"] == pytest.approx(
        at(Ledger(rows([0.5] * 30), priors()), 30)["trust"])


def test_action_visits_count_half_and_ipw_caps_at_one_unit():
    full = at(Ledger(rows([0.4] * 30), priors()), 30)["n_eff"]
    half = at(Ledger(rows([0.4] * 30, weight=0.5), priors()), 30)["n_eff"]
    ipw = at(Ledger(rows([0.4] * 30, weight=8.0), priors()), 30)["n_eff"]
    assert half == pytest.approx(full / 2, rel=1e-6) or half < MIN_EVIDENCE
    assert ipw == pytest.approx(full)


def test_gated_days_are_ungraded():
    led = Ledger(rows([0.9] * 30) + rows([0.0] * 30, gated=1, bucket=B), priors())
    assert at(led, 30)["trust"] == pytest.approx(at(Ledger(rows([0.9] * 30), priors()), 30)["trust"])


def test_outcomes_are_used_only_once_observed():
    base = rows([0.8] * 30)
    late = rows([0.0] * 5, start=30, lag=10)  # e.g. crew reports for days 30-34, known on day 40-44
    led = Ledger(base + late, priors(), end=day(60))
    ref = Ledger(base, priors(), end=day(60))
    for i in (31, 35, 39):  # not yet observed: identical to the ledger without them
        assert at(led, i)["trust"] == pytest.approx(at(ref, i)["trust"])
    assert at(led, 45)["trust"] < at(ref, 45)["trust"] - 0.05


def _utc_next_local_midnight(i: int) -> str:
    """J1's observed_at: the UTC instant of the local (Europe/Bucharest) midnight that ends day i."""
    from zoneinfo import ZoneInfo
    end = dt.datetime.combine(D0 + dt.timedelta(days=i + 1), dt.time(0), ZoneInfo("Europe/Bucharest"))
    return end.astimezone(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


@pytest.mark.parametrize("d0", [dt.date(2024, 6, 2), dt.date(2024, 12, 2)])  # summer (UTC+3) and winter (UTC+2)
def test_outcome_graded_on_day_d_is_visible_only_from_d_plus_1(d0, monkeypatch):
    """No look-ahead: observed_at = 2024-06-02T21:00:00Z is local midnight 06-03, so day 06-02's outcome must not
    affect lookup(06-02) but must affect lookup(06-03)."""
    monkeypatch.setattr(sys.modules[__name__], "D0", d0)
    assert _utc_next_local_midnight(0) == ("2024-06-02T21:00:00Z" if d0.month == 6 else "2024-12-02T22:00:00Z")
    hist = [dict(r, observed_at=_utc_next_local_midnight(i)) for i, r in enumerate(rows([0.8] * 20))]
    graded = [dict(r, observed_at=_utc_next_local_midnight(20)) for r in rows([0.0], start=20)]
    led, ref = Ledger(hist + graded, priors(), end=day(25)), Ledger(hist, priors(), end=day(25))
    assert at(led, 20)["trust"] == pytest.approx(at(ref, 20)["trust"])  # same day: not known yet
    assert at(led, 21)["trust"] < at(ref, 21)["trust"] - 1e-3           # next day: known
    # datetime-typed columns follow the same rule
    df = pl.DataFrame(hist + graded).with_columns(pl.col("observed_at").str.to_datetime(time_zone="UTC"))
    assert at(Ledger(df, priors(), end=day(25)), 20)["trust"] == pytest.approx(at(ref, 20)["trust"])


def test_missing_observed_at_is_never_the_graded_day():
    base = [{k: v for k, v in r.items() if k != "observed_at"} for r in rows([0.8] * 20)]
    led = Ledger(base + [{k: v for k, v in r.items() if k != "observed_at"} for r in rows([0.0], start=20)],
                 priors(), end=day(25))
    ref = Ledger(base, priors(), end=day(25))
    assert at(led, 20)["trust"] == pytest.approx(at(ref, 20)["trust"])
    assert at(led, 21)["trust"] < at(ref, 21)["trust"]


def test_late_outcome_updates_its_bucket_day():
    """A crew report that arrives after the telemetry of the same bucket-day changes that day's mean, not the
    number of days."""
    tel = rows([1.0] * 30)
    crew = [dict(r, asset_id="crew", score=0.0, observed_at=day(i + 5), source="crew")
            for i, r in enumerate(rows([0.0] * 30))]
    led = Ledger(tel + crew, priors(), end=day(60))
    ref = Ledger(rows([0.5] * 30), priors(), end=day(60))
    assert at(led, 40)["n_eff"] == pytest.approx(at(ref, 40)["n_eff"])
    assert at(led, 40)["trust"] == pytest.approx(at(ref, 40)["trust"])


# ---------------------------------------------------------------- backoff (§6.1)

def test_backoff_to_parent_below_min_evidence():
    sib = bucket_key("site", seen=False, battery=True, season="warm", quality="clean", ood_high=False)
    led = Ledger(rows([0.2] * 40, bucket=sib) + rows([0.9] * 3, bucket=B_UNSEEN, start=37), priors(), end=day(40))
    thin = at(led, 40, bucket=B_UNSEEN)
    own = at(led, 40, bucket=sib)
    assert own["read_level"] == 0  # its own bucket
    assert thin["read_level"] > 0  # read from a parent (the parent "site|unseen" is dominated by the sibling)
    assert thin["trust"] < 0.5 and thin["n_eff"] >= MIN_EVIDENCE
    # the root pools all of the model's buckets, so even an unseen kind reads it once it has ≥ 5 bucket-days ...
    assert at(led, 40, bucket="turbine|seen|nobatt|warm|clean|ood_lo")["read_level"] == 6
    # ... and a (model, score) with no evidence anywhere up its chain reads its prior
    assert at(led, 40, bucket=B_UNSEEN, score="sharpness")["trust"] == 0.5
    early = at(led, 3, bucket=B_UNSEEN)  # < 5 bucket-days even at the root: the prior itself
    assert early["trust"] == PRIOR and early["read_level"] == -1


def test_ancestor_order_drops_last_condition_first():
    assert ancestors("site|unseen|batt|warm|clean|ood_hi") == [
        "site|unseen|batt|warm|clean|ood_hi", "site|unseen|batt|warm|clean", "site|unseen|batt|warm",
        "site|unseen|batt", "site|unseen", "site", "*"]


def test_buckets_from_attribute_columns():
    r = [dict(x, kind="site", seen=False, battery=True, quality="gappy", ood_high=True) for x in rows([0.5] * 10)]
    for x in r:
        x.pop("bucket")
    led = Ledger(r, priors())
    assert led.full == ["site|unseen|batt|warm|gappy|ood_hi"]


# ---------------------------------------------------------------- health alerts (§6.5)

def test_alert_hysteresis():
    p = np.array([[0.8]])
    tr = np.array([[0.8, 0.55, 0.55, 0.8, 0.55, 0.55, 0.55, 0.65, 0.65, 0.75, 0.65, 0.55, 0.55]])
    st = _alert_state(tr, p, np.full_like(tr, 10.0))[0]
    # a 2-day dip never alerts; the 3rd consecutive low day raises; between the levels it holds; > p-0.1 clears
    assert list(st) == [0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0]
    assert ALERT_DAYS == 3
    # not enough evidence -> no alert
    assert _alert_state(tr, p, np.full_like(tr, 4.0)).sum() == 0


# ---------------------------------------------------------------- band scale (§6.4)

def _band_rows(n_days, sites, true_sigma_ratio, rng, start=0, weight=1.0):
    """Forecast bands built for sigma=1 while the truth has sigma=ratio → normalised errors."""
    z90 = 1.2815515655446004
    out = []
    for i in range(n_days):
        for a in range(sites):
            y = rng.normal(0, true_sigma_ratio, 24)
            e = np.abs(y) / z90  # q50 = 0, half-width = z90 on both sides
            for k, v in enumerate(e):
                out.append(dict(model="pvnet", model_version="v1", asset_id=f"S{a}", day=day(start + i),
                                score_name="norm_err", score=float(v), weight=weight,
                                observed_at=day(start + i + 1), gated=0, bucket=B))
    return out


@pytest.mark.parametrize("ratio,weight", [(0.6, 1.0), (2.0, 1.0), (2.0, 1 / 24)])
def test_band_scale_reaches_80pct_coverage(ratio, weight):
    """weight 1/24: rows that split their asset-day's weight, as J1 emits them, still count one unit a day."""
    rng = np.random.default_rng(1)
    led = Ledger(_band_rows(40, 6, ratio, rng, weight=weight), priors())
    scale = at(led, 40)["band_scale"]
    assert 0.5 <= scale <= 3.0
    # held-out days: the rescaled band q50 ± scale·z90 covers 80 %
    y = rng.normal(0, ratio, 200_000)
    cov = np.mean(np.abs(y) <= scale * 1.2815515655446004)
    assert cov == pytest.approx(0.80, abs=0.03)
    raw = np.mean(np.abs(y) <= 1.2815515655446004)
    assert abs(raw - 0.8) > 0.1  # the unscaled band was off


def test_band_scale_is_bounded_and_defaults_to_one():
    rng = np.random.default_rng(2)
    led = Ledger(_band_rows(20, 3, 10.0, rng), priors())
    assert at(led, 20)["band_scale"] == pytest.approx(3.0)
    assert at(led, 20, model="loadnet")["band_scale"] == 1.0  # no norm_err evidence -> unscaled band
    assert at(led, 3)["band_scale"] == 1.0                     # < 5 bucket-days yet


# ---------------------------------------------------------------- misc

def test_beta_interval_narrows_with_evidence():
    lo0, hi0 = beta_interval(np.array([0.7]), np.array([0.0]))
    lo1, hi1 = beta_interval(np.array([0.7]), np.array([20.0]))
    assert lo0 < lo1 < 0.7 < hi1 < hi0


def test_brier_detail_rows_are_not_trust_series():
    led = Ledger(rows([0.9] * 20) + rows([0.1] * 20, score="brier"), priors())
    assert set(led.trust_rows()["score_name"].unique()) == {"calibration"}


def test_interpolated_beta_interval_matches_exact():
    from sentinel.judge.ledger import _beta_exact
    rng = np.random.default_rng(3)
    t, n = rng.uniform(0.01, 0.99, 20_000), rng.uniform(0, 20.7, 20_000)
    lo, hi = beta_interval(t, n)
    elo, ehi = _beta_exact(t, N0 + n)
    assert np.abs(lo - elo).max() < 2e-3 and np.abs(hi - ehi).max() < 2e-3


def test_priors_from_models_dir():
    d = Path(__file__).resolve().parents[1] / "models"
    if not any("priors" in p.read_text() for p in d.glob("*.calibration.json")):
        pytest.skip("models/ carries no priors yet")
    pr = Priors.from_dir(d)
    for model, score in (("pvnet", "calibration"), ("loadnet", "sharpness"), ("sitehealth", "miss_score")):
        v = pr.get(model, "any-version", score, B)
        assert 0.0 <= v <= 1.0 and v != pr.default
    assert math.isfinite(pr.get("windnbm", None, "alarm_precision"))


def test_year_replay_shape():
    rng = np.random.default_rng(0)
    led = Ledger(rows(rng.beta(8, 2, 365), assets=3) + rows(rng.beta(8, 2, 365), score="sharpness", assets=3),
                 priors())
    t = led.trust_rows()
    assert t.height == 2 * led.n_days and t["trust"].is_between(0, 1).all()
