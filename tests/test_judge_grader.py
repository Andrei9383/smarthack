"""Judge grader: grading rules, gate, audit and speed checks."""

from __future__ import annotations

import inspect
import time
import warnings

import numpy as np
import pytest

from wattsentry.agent.world import ACTION_VISIT_WEIGHT, DEFAULT_START, CrewReport, MockWorld
from wattsentry.contracts import Prediction, PvFault
from wattsentry.judge import gate as G
from wattsentry.judge import grader as GR
from wattsentry.judge.grader import OUTCOME_COLUMNS, ReplayObs, grade, obs_from_world
from wattsentry.judge import scores as S

DAY0 = int(DEFAULT_START.timestamp())
DAY = 86400
T = 96


def iso(ts):
    return GR.iso(ts)


def _cs_profile():
    k = np.arange(T)
    return np.clip(np.sin(np.pi * (k - 20) / 56), 0, None) * 4.0  # daylight steps 21..75, peak 4 kW


def make_obs(n_sites=4, days=3, seed=0, hist=0, pv=None, load=None):
    rng = np.random.default_rng(seed)
    rows = [{"site_id": f"S{i}", "kwp_recorded": 5.0, "tilt_recorded": 30, "azimuth_recorded": 180,
             "inverter_kw": 4.5, "lat": 44.1, "lon": 28.5, "locality": "L1", "battery_kwh": 10.0 if i % 2 else 0.0,
             "split": "unseen" if i == 0 else "train"} for i in range(n_sites)]
    D = hist + days
    cs = np.broadcast_to(_cs_profile(), (n_sites, D, T)).copy()
    if pv is None:
        pv = cs * rng.uniform(0.7, 0.9, (n_sites, D, 1)) + rng.normal(0, 0.05, (n_sites, D, T)) * (cs > 0)
        pv = np.clip(pv, 0, None)
    if load is None:
        load = 0.6 + 0.3 * np.sin(np.arange(T) / 15.0) + rng.normal(0, 0.1, (n_sites, D, T))
        load = np.clip(load, 0.05, None)
    return ReplayObs(DAY0, days, hist, [r["site_id"] for r in rows], rows, pv, load, cs=cs)


def fc(asset, d, q10, q50, q90, model="loadnet", version="v1", **kw):
    return Prediction(asset_id=asset, model=model, model_version=version, t0=iso(DAY0 + d * DAY), horizon=T,
                      q10=list(q10), q50=list(q50), q90=list(q90), **kw)


def health(asset, d, top="healthy", p=0.95, model="sitehealth"):
    classes = [c.value for c in PvFault]
    rest = (1 - p) / (len(classes) - 1)
    return Prediction(asset_id=asset, model=model, model_version="v1", t0=iso(DAY0 + d * DAY), horizon=T,
                      q10=[0.0] * T, q50=[0.0] * T, q90=[0.0] * T,
                      fault_probs={c: (p if c == top else rest) for c in classes})


def pick(rows, **kw):
    return [r for r in rows if all(r[k] == v for k, v in kw.items())]


# ---------------------------------------------------------------------------------- wide bands win hit rate
def test_wide_bands_lose():
    obs = make_obs(n_sites=6, days=3, seed=1)
    preds = []
    for s in obs.site_ids:
        for d in range(3):
            y = obs.load[obs.site_index[s], d]
            mu = np.full(T, float(np.mean(y)))
            sd = float(np.std(y))
            preds.append(fc(s, d, mu - 1.28 * sd, mu, mu + 1.28 * sd, model="loadnet", version="honest"))
            preds.append(fc(s, d, mu - 1.28 * sd, mu, mu + 1.28 * sd, model="persistence_load", version="ref",
                            extra={"target": "load"}))
    honest = grade(preds, obs, run_id="t")
    wide = grade([fc(p.asset_id, (GR._ts(p.t0) - DAY0) // DAY, np.array(p.q50) - 50, p.q50, np.array(p.q50) + 50,
                     model=p.model, version="wide" if p.model == "loadnet" else p.model_version, extra=p.extra)
                  if p.model == "loadnet" else p for p in preds], obs, run_id="t")
    narrow = grade([fc(p.asset_id, (GR._ts(p.t0) - DAY0) // DAY, np.array(p.q50) - 1e-3, p.q50,
                       np.array(p.q50) + 1e-3, model=p.model, version="narrow") if p.model == "loadnet" else p
                    for p in preds], obs, run_id="t")
    mean = lambda rows, v, n: np.mean([r["score"] for r in pick(rows, model="loadnet", model_version=v, score_name=n)])
    # the wide band holds 100 % of the actuals, yet loses on calibration (closeness to 0.80) and on sharpness
    assert mean(wide, "wide", "calibration") < mean(honest, "honest", "calibration")
    assert mean(wide, "wide", "sharpness") == 0.0
    assert mean(narrow, "narrow", "calibration") < 0.2
    assert mean(honest, "honest", "calibration") > 0.8


def test_scores_basic():
    assert S.calibration(np.array([0.8]))[0] == 1.0
    assert S.calibration(np.array([1.0]))[0] == pytest.approx(0.75)
    assert S.calibration(np.array([0.0]))[0] == 0.0
    y = np.zeros((1, 10))
    v = np.ones((1, 10), bool)
    # a miss costs width + (2/alpha) x distance
    assert S.interval_score(np.full((1, 10), 1.0), np.full((1, 10), 2.0), y, v)[0] == pytest.approx(1 + 10 * 1)
    assert S.skill(np.array([1.0]), np.array([2.0]))[0] == 0.5
    assert S.skill(np.array([3.0]), np.array([2.0]))[0] == 0.0


def test_nanquantile_matches_numpy():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(7, 5, 30))
    x[rng.random(x.shape) < 0.3] = np.nan
    x[0, 0] = np.nan
    qs = (0.1, 0.5, 0.9)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        ref = np.moveaxis(np.nanquantile(x, qs, axis=-1), 0, -1)
    np.testing.assert_allclose(S.nanquantile(x, qs), ref, equal_nan=True)


# ---------------------------------------------------------------------------------- always healthy
def test_always_healthy_loses():
    obs = make_obs(n_sites=6, days=3)
    faulty = {"S1": "string_outage", "S2": "soiling"}
    reports = []
    for s in obs.site_ids:  # an audit visit on day 1 to every site
        f = faulty.get(s)
        reports.append(CrewReport(s, "site", "residential_installer", DAY0 + DAY + 13 * 3600, f or "healthy",
                                  DAY0 - DAY if f else None, source="audit", weight=2.0, p_sampled=0.5))
    lazy = [health(s, d, model="sitehealth") for s in obs.site_ids for d in range(3)]
    good = [health(s, d, faulty.get(s, "healthy"), 0.9 if s in faulty else 0.95, model="threshold_cusum")
            for s in obs.site_ids for d in range(3)]
    rows = grade(lazy + good, obs, reports, run_id="t")
    lazy_rows = [r for r in pick(rows, model="sitehealth") if r["source"] != "residual"]
    good_rows = [r for r in pick(rows, model="threshold_cusum") if r["source"] != "residual"]
    # "always healthy" never gets an alarm row; its misses on the faulty sites are graded 0
    assert not pick(lazy_rows, score_name="alarm_precision")
    assert {r["score"] for r in pick(lazy_rows, asset_id="S1")} == {0.0}
    assert np.mean([r["score"] for r in good_rows]) > np.mean([r["score"] for r in lazy_rows])
    assert all(r["score"] == 1.0 for r in pick(good_rows, score_name="alarm_precision"))
    assert pick(good_rows, score_name="alarm_precision")[0]["weight"] == 2.0  # the audit's IPW is carried


def test_alarm_precision_partial_and_false():
    a = S.alarm_precision(np.array(["string_outage", "string_outage", "soiling"]),
                          np.array(["string_outage", "sensor_fault", "healthy"]), np.array([True, True, False]))
    assert a.tolist() == [1.0, 0.5, 0.0]


# ---------------------------------------------------------------------------------- gate never uses the witness
def test_gate_never_uses_graded_witness():
    for fn in (G.gate_pv, G.gate_load, G.gate_turbine, G.own_history_jump):
        params = set(inspect.signature(fn).parameters)
        assert not params & {"pred", "preds", "prediction", "predictions", "q10", "q50", "q90"}
    obs_good = make_obs(n_sites=4, days=3)
    pv = obs_good.pv.copy()
    pv[2, 1, 40] = 50.0  # corrupted reading: way above 1.2 x kWp
    obs = make_obs(n_sites=4, days=3, pv=pv)
    base = [fc(s, d, obs.cs[0, d] * 0.6, obs.cs[0, d] * 0.8, obs.cs[0, d] * 0.95, model="pvnet")
            for s in obs.site_ids for d in range(3)]
    broken = [fc(s, d, obs.cs[0, d] * 3, obs.cs[0, d] * 3.5, obs.cs[0, d] * 4, model="pvnet")
              for s in obs.site_ids for d in range(3)]
    gated = lambda rows: sorted({(r["asset_id"], r["day"]) for r in rows if r["gated"]})
    r1, r2 = grade(base, obs, run_id="t"), grade(broken, obs, run_id="t")
    assert gated(r1) == gated(r2) == [("S2", obs.day_label[1])]
    # the broken witness's misses are graded, not thrown out as bad data
    assert np.mean([r["score"] for r in pick(r2, model="pvnet", score_name="calibration") if r["score"] is not None]) \
        < np.mean([r["score"] for r in pick(r1, model="pvnet", score_name="calibration") if r["score"] is not None])


def test_gate_physics_checks():
    cs = np.broadcast_to(_cs_profile(), (5, 2, T)).copy()
    pv = cs * 0.8
    pv[0, 0, 30:40] = 1.234  # frozen non-zero value while the sun moves
    pv[1, 0, 30:60] = 0.0  # outage zeros at midday: a real fault, never gated
    pv[2, 0] = np.minimum(cs[2, 0] * 1.2, 3.0)  # inverter clipping: flat at the day's max, physical
    pv[3, 0, 2:10] = 1.0  # PV at night
    pv[4, 0, :80] = np.nan  # mostly unreported
    code = G.gate_pv(pv, cs, np.full(5, 5.0))
    assert code[0, 0] & G.FROZEN and code[3, 0] & G.PV_NIGHT and code[4, 0] & G.GAP
    assert code[1, 0] == 0 and code[2, 0] == 0
    assert (code[:, 1] == 0).all()
    # unit jump: own-history median, causal
    e = np.array([[10, 10, 10, 10, 1.0, 10, 60.0]])
    assert G.own_history_jump(e).tolist() == [[False, False, False, False, True, False, True]]
    load = np.full((1, 1, T), 1.0)
    load[0, 0, 5] = 40.0
    assert G.gate_load(load, np.array([22.0]))[0, 0] & G.LOAD_FUSE
    tp = np.full((1, 1, T), 100.0)
    tp[0, 0, 3] = 3000.0
    assert G.gate_turbine(tp, np.array([2500.0]))[0, 0] & G.TURB_RANGE


# ---------------------------------------------------------------------------------- 96 readings = one day
def test_96_readings_count_as_one_asset_day():
    obs = make_obs(n_sites=1, days=1)
    y = obs.load[0, 0]
    p = [fc("S0", 0, y - 0.2, y, y + 0.2), fc("S0", 0, y - 0.3, y, y + 0.3, model="persistence_load",
                                             extra={"target": "load"})]
    rows = grade(p, obs, run_id="t")
    w = pick(rows, model="loadnet")
    assert len(pick(w, score_name="calibration")) == 1 and len(pick(w, score_name="sharpness")) == 1
    ne = pick(w, score_name="norm_err")
    assert len(ne) == len(GR.NORM_ERR_Q) and sum(r["weight"] for r in ne) == pytest.approx(1.0)  # together one day
    assert set(rows[0]) == set(OUTCOME_COLUMNS)
    r = pick(w, score_name="calibration")[0]
    assert r["kind"] == "site" and r["seen"] is False and r["battery"] is False and r["season"] == "warm"
    assert r["quality"] == "clean" and r["ood_high"] is False and r["source"] == "telemetry"


# ---------------------------------------------------------------------------------- observed_at ordering
def test_observed_at_ordering():
    obs = make_obs(n_sites=4, days=3)
    y = obs.load
    preds = [fc(s, d, y[i, d] - 0.3, y[i, d], y[i, d] + 0.3) for i, s in enumerate(obs.site_ids) for d in range(3)]
    visit = DAY0 + 2 * DAY + 10 * 3600
    preds += [health("S1", 0, "string_outage", 0.9)]
    reports = [CrewReport("S1", "site", "residential_installer", visit, "string_outage", DAY0 - 3600)]
    rows = grade(preds, obs, reports, run_id="t")
    for r in pick(rows, model="loadnet", score_name="calibration"):
        d = obs.day_label.index(r["day"])
        assert r["observed_at"] == iso(DAY0 + (d + 1) * DAY)  # telemetry scores arrive at d+1
    a = pick(rows, model="sitehealth", score_name="alarm_precision")
    assert len(a) == 1 and a[0]["observed_at"] == iso(visit) and a[0]["score"] == 1.0
    assert a[0]["weight"] == 1.0  # explicit report weight; world.crew_report defaults action visits to 0.5
    assert [r["observed_at"] for r in rows] == sorted(r["observed_at"] for r in rows)
    # the ledger view on day 1 must not contain the crew report of day 2
    assert not [r for r in pick(rows, model="sitehealth") if r["observed_at"] <= iso(DAY0 + 2 * DAY)]


def test_m7_waits_14_days_and_m8_flags():
    tr = [{"turbine_id": "P.T1", "park_id": "P", "rated_kw": 2500.0}, {"turbine_id": "P.T2", "park_id": "P",
                                                                       "rated_kw": 2500.0}]
    tpow = np.full((2, 3, T), 500.0)
    tcode = np.zeros((2, 3, T), np.int64)
    tcode[0, 2, 40:60] = 211  # T1 stops on day 2: an observable failure
    obs = ReplayObs(DAY0, 3, 0, [], [], np.zeros((0, 3, T)), np.zeros((0, 3, T)), [r["turbine_id"] for r in tr], tr,
                    tpow, tcode, cs=np.zeros((0, 3, T)))
    m7 = [Prediction(asset_id=t, model="windfailure", model_version="v1", t0=iso(DAY0), horizon=1, q10=[r], q50=[r],
                     q90=[r], unit="p_fail_14d", step_minutes=1440, extra={"risk": r, "alarm": r > 0.5})
          for t, r in (("P.T1", 0.8), ("P.T2", 0.1))]
    rows = grade(m7, obs, run_id="t")
    hit = pick(rows, score_name="hit")
    assert len(hit) == 1 and hit[0]["asset_id"] == "P.T1" and hit[0]["score"] == 1.0
    assert hit[0]["observed_at"] == iso(DAY0 + 2 * DAY + 40 * 900)
    rows = grade(m7, obs, run_id="t", horizon_end_ts=DAY0 + 20 * DAY)  # 14 days have passed: T2 graded too
    assert {r["asset_id"]: r["score"] for r in pick(rows, score_name="hit")} == {"P.T1": 1.0, "P.T2": 1.0}
    # M8
    obs = make_obs(n_sites=3, days=2)
    reps = [CrewReport("S0", "site", "residential_installer", DAY0 + 3600, "healthy", gis_wrong=True),
            CrewReport("S1", "site", "residential_installer", DAY0 + 3600, "healthy", gis_wrong=False),
            CrewReport("S2", "site", "residential_installer", DAY0 + 3600, "healthy", gis_wrong=True)]
    rows = grade([], obs, reps, run_id="t", gis_flags=[{"site_id": "S0"}, {"site_id": "S1"}])
    got = {(r["asset_id"], r["score_name"]): r["score"] for r in pick(rows, model="gridtwin")}
    assert got == {("S0", "flag"): 1.0, ("S1", "flag"): 0.0, ("S2", "miss_score"): 0.0}


# ---------------------------------------------------------------------------------- gated days ungraded
def test_gated_days_are_ungraded():
    obs0 = make_obs(n_sites=3, days=2)
    pv = obs0.pv.copy()
    pv[1, 0, 30:45] = 2.5  # frozen sensor on S1 day 0
    obs = make_obs(n_sites=3, days=2, pv=pv)
    preds = [fc(s, d, obs.cs[0, d] * 0.5, obs.cs[0, d] * 0.8, obs.cs[0, d], model="pvnet")
             for s in obs.site_ids for d in range(2)]
    preds += [health("S1", 0)]
    reports = [CrewReport("S1", "site", "residential_installer", DAY0 + 12 * 3600, "healthy", source="audit")]
    rows = grade(preds, obs, reports, run_id="t")
    g = [r for r in rows if r["gated"]]
    assert {(r["asset_id"], r["day"]) for r in g} == {("S1", obs.day_label[0])}
    assert all(r["score"] is None for r in g)
    assert {r["score_name"] for r in g} == {"calibration", "sharpness", "miss_score"}  # never a miss, never a pass
    assert not pick(rows, asset_id="S1", day=obs.day_label[0], score_name="norm_err")
    assert len([r for r in pick(rows, model="pvnet", score_name="calibration") if not r["gated"]]) == 5


# ---------------------------------------------------------------------------------- crew reports + audit IPW
def test_audit_ipw_weights_and_crew_reports():
    w = MockWorld(days=2, scale=0.3, with_wind=True)
    reps = w.audit_checks(0)
    assert reps == w.audit_checks(0)  # deterministic by seed and day
    by_crew = {}
    for r in reps:
        by_crew.setdefault(r.crew_type, []).append(r)
        assert r.source == "audit" and r.weight == pytest.approx(1 / r.p_sampled)
    assert set(by_crew) == {"residential_installer", "wind_technician"}
    assert all(1 <= len(v) <= 2 for v in by_crew.values())
    n = len(w.sites)
    assert by_crew["residential_installer"][0].p_sampled == pytest.approx(1 - (1 - 1 / n) ** 2)
    # weighted sampling: a candidate with 50x weight has a known, higher inclusion prob and a smaller weight
    sid = sorted(w.sites)[0]
    wts = {sid: 50.0}
    p = 50.0 / (50.0 + n - 1)
    hits = 0
    for d in range(200):
        rr = [r for r in w.audit_checks(d, weights=wts) if r.crew_type == "residential_installer"]
        for r in rr:
            if r.asset_id == sid:
                hits += 1
                assert r.p_sampled == pytest.approx(1 - (1 - p) ** 2)
    assert abs(hits / 200 - (1 - (1 - p) ** 2)) < 0.12  # empirical inclusion ~ the stated probability
    # action visit: true state at visit time for that asset only, weight 0.5
    story = w.story_site
    onset = w.faults[story]["onset_ts"]
    after = w.crew_report(story, onset + 7200)
    before = w.crew_report(story, onset - 7200)
    assert after.weight == ACTION_VISIT_WEIGHT and after.source == "crew"
    if after.found == "string_outage":  # 5 % misreport may flip it; deterministic by seed
        assert after.onset_ts <= onset
    assert before.found in ("healthy",) or before.onset_ts == before.visit_ts or before.onset_ts < onset
    flips = sum(w.crew_report(s, w.day_ts(0) + 3600 * k).found != "healthy"
                for s in w.sites if s not in w.faults for k in range(10))
    total = 10 * sum(1 for s in w.sites if s not in w.faults)
    assert 0.0 < flips / total < 0.12  # ~5 % misreport on healthy assets


def test_grader_never_reads_truth():
    src = inspect.getsource(GR) + inspect.getsource(G) + inspect.getsource(S)
    for word in ("fault_truth", "fault_labels", ".faults", "_truth_index", "kwp_true", "tilt_true", "azimuth_true"):
        assert word not in src.replace("fault_labels) reaches", "")


# ---------------------------------------------------------------------------------- full week speed
def test_grade_mock_week_under_5s():
    w = MockWorld(days=7)
    preds = w.default_predictors()
    P = []
    for d in range(w.days):
        t0 = w.day_t0(d)
        for s in w.sites:
            r = w.sites[s].row
            P += [preds["pv"].predict(r, t0=t0), preds["load"].predict(r, t0=t0), preds["sitehealth"].predict(r, t0=t0)]
        for t in w.turbines:
            P.append(preds["wind"].predict(w.turbines[t].row, t0=t0))
    obs = obs_from_world(w, hist=2)
    reports = [r for d in range(w.days) for r in w.audit_checks(d)]
    reports.append(w.crew_report(w.story_site, w.day_ts(4) + 12 * 3600))
    t = time.perf_counter()
    rows = grade(P, obs, reports, run_id="t", gis_flags=w.gis_flags())
    dt = time.perf_counter() - t
    print(f"grade week: {dt:.2f}s, {len(P)} predictions, {len(rows)} rows")
    assert dt < 5.0
    assert pick(rows, model="pvnet", score_name="calibration") and pick(rows, model="sitehealth")
    assert pick(rows, model="persistence_load", score_name="sharpness")  # internal substitute prior id
    a = pick(rows, model="sitehealth", asset_id=w.story_site, score_name="alarm_precision")
    assert a and a[0]["source"] == "crew" and a[0]["weight"] == ACTION_VISIT_WEIGHT
