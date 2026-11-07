"""Card #165 W2: availability table from Judge-validated M6/M7 reads, park scenarios, conformal on the park total."""

import numpy as np
import pytest

from sentinel.judge.adapter import MemoryVerdicts, Verdict
from sentinel.model.windpower.availability import (AvailConfig, M6Batch, M7Batch, StopSurvival, availability,
                                                   fit_icing_factor, icing_risk, judge_reads, stop_runs_h)
from sentinel.model.windpower.scenarios import (ParkConformal, common_levels, park_forecast, park_scenarios,
                                                quantile_at, scen_quantiles)

IDS = ["WP1.T1", "WP1.T2", "WP1.T3", "WP1.T4", "WP1.T5", "WP1.T6", "WP1.T7", "WP1.T8"]
CLS = ["normal", "fault_stop", "fault_stop", "underperformance", "curtailment", "anemometer_fault", "overheating",
       "underperformance"]
TS = np.arange(np.datetime64("2025-03-02T00:00"), np.datetime64("2025-03-03T00:00"), np.timedelta64(15, "m"))


def _reads(judge=None, subs=None, m7=None):
    m6 = M6Batch(cls=CLS, score=[0.1, 0.9, 0.9, 0.8, 0.9, 0.9, 0.7, 0.3],
                 loss_frac=[np.nan, 1, 1, 0.2, np.nan, np.nan, np.nan, 0.5], stopped_h=[0, 5, 5, 0, 0, 0, 0, 0])
    return judge_reads(IDS, "2025-03-01", m6, m7, substitutes=subs, judge=judge)


def test_table_rules():
    cfg = AvailConfig(survival=StopSurvival.exponential(10.0))
    av = availability(_reads(), TS, work_orders={"WP1.T2": "2025-03-02T06:00Z"}, origin="2025-03-01T10:00", cfg=cfg)
    a = av.a
    assert a.shape == (96, 8)
    assert np.all(a[:, 0] == 1)  # normal
    assert np.all(a[:24, 1] == 0) and np.all(a[24:, 1] == 1)  # fault_stop + work order ETA 06:00
    assert av.rule[1] == "fault_stop_eta"
    # no ETA: exponential (memoryless) P(repaired by t) = 1 - exp(-(t - origin) / 10 h), rising over the day
    dt = (TS - np.datetime64("2025-03-01T10:00")).astype(int) / 60
    assert np.allclose(a[:, 2], 1 - np.exp(-dt / 10)) and av.rule[2] == "fault_stop"
    assert np.allclose(a[:, 3], 0.8)  # underperformance: 1 - measured loss
    assert np.all(a[:, 4] == 1) and np.all(a[:, 6] == 1)  # curtailment, overheating
    assert np.all(a[:, 5] == 1) and av.uncertain[5] and not av.uncertain[0]  # anemometer: 1 but uncertain
    assert np.all(a[:, 7] == 1)  # weak (score < conf_min) underperformance counts as normal
    assert av.table()[1]["m6_class"] == "fault_stop"


def test_judge_dont_use_substitute_and_unknown():
    j = MemoryVerdicts({("WP1.T2", "windnbm", "2025-03-01"): Verdict("dont_use", substitute="windmon_rules"),
                        ("WP1.T4", "windnbm", "2025-03-01"): Verdict("dont_use", substitute="windmon_rules"),
                        ("WP1.T3", "windnbm", "2025-03-01"): Verdict("dont_use")})
    sub = M6Batch(cls=["normal"] * 8, score=[0.0] * 8, loss_frac=[np.nan] * 8, stopped_h=[0] * 8)
    sub.cls = list(sub.cls)
    sub.cls[3] = "underperformance"
    sub.score = [0.0, 0.0, 0.0, 0.9, 0, 0, 0, 0]
    sub.loss_frac = [np.nan, np.nan, np.nan, 0.4, np.nan, np.nan, np.nan, np.nan]
    r = _reads(judge=j, subs={"windmon_rules": sub})
    assert r.cls[1] == "normal" and r.source[1] == "windmon_rules"  # M6's fault_stop rejected -> substitute says ok
    assert r.cls[2] == "unknown" and r.source[2] is None  # rejected, no substitute
    assert r.loss_frac[3] == pytest.approx(0.4)  # extras come from the accepted source
    av = availability(r, TS)
    assert np.all(av.a[:, 1] == 1) and np.all(av.a[:, 2] == 1) and av.uncertain[2]
    assert np.allclose(av.a[:, 3], 0.6)


def test_m7_high_risk_and_icing():
    m7 = M7Batch(p_fail_7d=[0.0, 0, 0, 0, 0, 0, 0.5, 0.5], alarm=[False] * 6 + [True, False])
    r = _reads(m7=m7)
    w = {"temp_c": np.where(np.arange(96) < 40, -2.0, 5.0)[:, None], "rh_pct": np.full((96, 1), 95.0)}
    cfg = AvailConfig(icing_factor=0.5, icing_persist_h=2.0)
    av = availability(r, TS, weather=w, cfg=cfg)
    p24 = 1 - 0.5 ** (1 / 7)
    assert np.allclose(av.p_up[:, 6], 1 - p24) and np.all(av.p_up[:, 7] == 1)  # only the alarm counts
    assert np.allclose(av.derate[:48, 0], 0.5) and np.all(av.derate[48:, 0] == 1)  # 40 steps + 2 h persistence
    assert np.allclose(av.derate[:48, 3], 0.8 * 0.5)
    short = {"temp_c": np.where(np.arange(96) < 8, -2.0, 5.0)[:, None], "rh_pct": np.full((96, 1), 95.0)}
    assert not icing_risk(short, 96, 8, cfg).any()  # < 3 h spell
    f, n = fit_icing_factor(np.full((200, 2), 30.0), np.full((200, 2), 60.0), np.ones((200, 2), bool))
    assert f == pytest.approx(0.5) and n == 400


def test_survival_fit_and_runs():
    c = np.zeros((40, 2), int)
    c[5:9, 0] = 3
    c[10:30, 1] = 3
    c[0:3, 1] = 3  # touches the edge: censored
    assert stop_runs_h(c, 3) == [1.0, 5.0]
    s = StopSurvival.fit(np.r_[np.full(50, 2.0), np.full(50, 48.0)])
    p = s.p_repaired(0.0, np.array([1.0, 3.0, 47.0, 49.0]))
    assert p[0] == pytest.approx(0.0) and p[1] == pytest.approx(0.5) and p[3] == pytest.approx(1.0, abs=1e-3)
    assert s.p_repaired(3.0, np.array([40.0]))[0] == pytest.approx(0.0, abs=1e-6)  # past 2 h: a long stop
    assert np.all(np.diff(s.p_repaired(10.0, np.arange(0, 60, 0.25))) >= -1e-12)


def test_quantile_function_and_levels():
    q = np.array([10.0, 50.0, 80.0])
    assert np.allclose(quantile_at(q, np.array([0.1, 0.5, 0.9])), q)
    assert quantile_at(q, np.array(0.0)) == pytest.approx(0.0) and quantile_at(q, np.array(1.0), 90.0) == 87.5
    u = common_levels(400, 96, 2, rho=0.97, seed=1)
    assert u.shape == (400, 96, 2) and abs(u.mean() - 0.5) < 0.02 and abs((u < 0.1).mean() - 0.1) < 0.02
    z = np.corrcoef(u[:, 10, 0], u[:, 11, 0])[0, 1]
    assert z > 0.9 and abs(np.corrcoef(u[:, 10, 0], u[:, 10, 1])[0, 1]) < 0.2


def test_park_scenarios_comonotone_and_availability():
    T, N = 8, 4
    q = np.tile(np.array([100.0, 500.0, 900.0]), (T, N, 1))
    rated = np.full(N, 2000.0)
    u = common_levels(8000, T, 1, seed=0)[..., 0]
    s = park_scenarios(q, rated, np.arange(N), np.full(T, 1.1), u)
    pq = scen_quantiles(s)
    assert np.allclose(pq[:, 1], 4 * 500 * 1.1, rtol=0.04)  # one common level: park quantile = sum of quantiles
    assert np.allclose(pq[:, 0], 4 * 100 * 1.1, rtol=0.1)
    pu = np.ones((T, N))
    pu[:, 0] = 0.0
    s2 = park_scenarios(q, rated, np.arange(N), np.ones(T), u, p_up=pu, derate=np.full((T, N), 0.5))
    assert np.allclose(scen_quantiles(s2)[:, 1], 3 * 500 * 0.5, rtol=0.04)
    pu[:, 1] = 0.5
    s3 = park_scenarios(q, rated, np.arange(N), np.ones(T), u, p_up=pu, bernoulli=True, rng=np.random.default_rng(0))
    assert 0.4 < (s3[:, 0] < 2.1 * 500).mean() < 0.6  # the half-available turbine is on in ~half the scenarios


def test_conformal_reaches_80pct():
    rng = np.random.default_rng(0)
    M = 20000
    p50 = rng.uniform(0.1, 0.8, M)
    y = p50 + rng.normal(0, 0.08, M)
    q = np.stack([p50 - 0.05, p50, p50 + 0.05], 1)  # too narrow
    cal = ParkConformal.fit(q[:10000], y[:10000])
    out, _ = cal.apply(q[10000:])
    cov = ((out[:, 0] <= y[10000:]) & (y[10000:] <= out[:, 2])).mean()
    assert 0.78 <= cov <= 0.82 and cal.a > 1.5
    scen = np.stack([p50[:5] - 0.05, p50[:5] + 0.05])
    o5, sc = cal.apply(q[:5], scen=scen)
    assert np.allclose(sc[0], np.clip(o5[:, 0], 0, None), atol=0.003)  # the raw P10 scenario maps onto the new P10
    assert ParkConformal.from_json(cal.to_json()).ab() == pytest.approx(cal.ab())


def test_park_forecast_binned_end_to_end(tmp_path):
    import json

    from sentinel.eval.baselines.wind.curve import data_inputs
    from sentinel.model.wind.data import POWER, make_splits
    from sentinel.model.wind.synth import SynthConfig, generate
    from sentinel.model.windpower import WindPowerModel
    from sentinel.model.windpower.binned import fit_binned
    from sentinel.model.windpower.features import weather_per_turbine

    d = generate(SynthConfig(days=20, scale=0.1))
    sp = make_splits(d)
    v = np.clip(np.nan_to_num(data_inputs(d)["met"]), 0, 35)
    bp = fit_binned(d.turbines, v, d.sensors[POWER] / d.rated_kw[None], (d.labels == 0) & sp.train_t[:, None])
    p = tmp_path / "windpower.s0.json"
    p.write_text(json.dumps({"model_version": "t", "binned": {"curves": bp.curves, "rated_kw": bp.rated_kw,
                                                             "edges": bp.edges}, "wake_ratio_pooled": [1.0] * 12}))
    m = WindPowerModel.from_json(p, binned=True)
    sub = d.subset(t_slice=slice(96, 192))
    w = weather_per_turbine(sub.weather, sub.park_of)
    fc0 = park_forecast(m, sub.turbines, sub.ts, sub.meteo, w, n_scen=100)
    reads = judge_reads(sub.turbine_ids, "2025-01-01",
                        M6Batch(cls=["fault_stop"] + ["normal"] * (len(sub.turbines) - 1),
                                score=[0.9] + [0.0] * (len(sub.turbines) - 1), stopped_h=[1.0] * len(sub.turbines)))
    av = availability(reads, sub.ts, work_orders={sub.turbine_ids[0]: None},
                      cfg=AvailConfig(survival=StopSurvival.exponential(1e6)))
    fc1 = park_forecast(m, sub.turbines, sub.ts, sub.meteo, w, avail=av, n_scen=100,
                        conformal=ParkConformal(1.2, 1.2))
    park0 = sub.turbines[0]["park_id"]
    assert set(fc0) == set(d.park_ids) and fc0[park0].q.shape == (96, 3) and fc0[park0].scenarios.shape == (100, 96)
    assert np.all(np.diff(fc1[park0].q, axis=-1) >= -1e-9)
    assert fc1[park0].q[:, 1].sum() < fc0[park0].q[:, 1].sum()  # the stopped turbine is no longer counted
    other = [k for k in fc0 if k != park0][0]
    assert np.allclose(fc1[other].q[:, 1], fc0[other].q[:, 1])
    assert fc1[park0].availability_mean < 1 and fc1[park0].quantile(0.5).shape == (96,)


def test_eval_origin_reads():
    from sentinel.model.wind.data import CLASS_INDEX
    from sentinel.model.windpower.avail_eval import WIN, origin_batch, run_length

    T, N = 200, 3
    cls = np.zeros((T, N), np.uint8)
    cls[150:200, 0] = CLASS_INDEX["fault_stop"]
    cls[100:200, 1] = CLASS_INDEX["underperformance"]
    cls[195:200, 2] = CLASS_INDEX["icing"]  # 5 of the last 8 steps
    m6 = {"cls": cls, "score": np.full((T, N), 0.9, np.float32), "q50": np.full((T, N), 0.5, np.float32)}
    power = np.full((T, N), 800.0)
    rated = np.full(N, 2000.0)
    b = origin_batch(m6, T, rated, power, run_length(cls == CLASS_INDEX["fault_stop"]))
    assert b.cls == ["fault_stop", "underperformance", "icing"] and WIN == 8
    assert b.stopped_h[0] == 12.5 and b.stopped_h[1] == 0
    assert b.loss_frac[1] == pytest.approx(0.2)  # 800 kW measured vs 1000 kW expected


def test_config_roundtrip(tmp_path):
    import json

    from sentinel.model.windpower.scenarios import load_config

    s = StopSurvival.fit(np.r_[np.full(50, 2.0), np.full(50, 48.0)])
    p = tmp_path / "w2_park.json"
    p.write_text(json.dumps({"survival": s.to_json(), "icing_factor": 0.6, "conformal": ParkConformal(1.3, 1.1).to_json(),
                             "n_scen": 50, "rho": 0.9}))
    cfg, cal, n, rho = load_config(p)
    assert cfg.icing_factor == 0.6 and cal.ab() == (1.3, 1.1) and n == 50 and rho == 0.9
    assert cfg.survival.p_repaired(0.0, np.array([24.0]))[0] == pytest.approx(0.5, abs=0.02)
