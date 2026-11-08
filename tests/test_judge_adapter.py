"""Judge adapter: dont_use + substitute, widen, no-Judge passthrough, batch == single."""

import sqlite3

import numpy as np
import pytest

from wattsentry.contracts import Prediction
from wattsentry.judge.adapter import (H, MemoryVerdicts, NoJudge, RawBatch, SqliteVerdicts, Verdict, as_verdict,
                                    scale_band, set_default_judge, validated, validated_many)

DAY = "2025-06-02"
T0 = "2025-06-01T21:00:00Z"


def band(mid, half, n=H):
    m = np.full(n, float(mid))
    return np.stack([m - half, m, m + half], axis=-1)


def pred(model, mid, half):
    q = band(mid, half)
    return Prediction(asset_id="s1", model=model, model_version="v1", t0=T0, horizon=H, q10=q[:, 0], q50=q[:, 1],
                      q90=q[:, 2])


@pytest.fixture(autouse=True)
def _no_default():
    set_default_judge(None)
    yield
    set_default_judge(None)


def test_no_judge_passthrough():
    p = pred("pvnet", 3.0, 1.0)
    for judge in (None, NoJudge(), MemoryVerdicts()):  # default, explicit, and a source with no row for it
        v = validated("s1", "pvnet", DAY, raw=p, judge=judge)
        assert v.verdict == "use" and v.rule == "no_judge" and v.known
        assert v.band_scale == 1.0 and v.source_model == "pvnet"
        np.testing.assert_array_equal(v.q10, p.q10)
        np.testing.assert_array_equal(v.q90, p.q90)


def test_widen_scales_band_around_p50():
    j = MemoryVerdicts([{"asset_id": "s1", "model": "pvnet", "day": DAY, "verdict": "widen", "p_valid": 0.6,
                         "trust": 0.55, "n_eff": 9, "band_scale": 2.0, "reasons_json": '["trust 0.55 on battery homes"]',
                         "rule": "widen", "judge_version": "j-1"}])
    v = validated("s1", "pvnet", DAY, raw=pred("pvnet", 3.0, 1.0), judge=j)
    assert v.verdict == "widen" and v.band_scale == 2.0 and v.known
    np.testing.assert_allclose(v.q50, 3.0)
    np.testing.assert_allclose(v.q10, 1.0)
    np.testing.assert_allclose(v.q90, 5.0)
    assert v.reasons == ("trust 0.55 on battery homes",) and v.p_valid == 0.6 and v.judge_version == "j-1"


def test_widen_keeps_nonnegative_quantities_nonnegative():
    q = scale_band(band(1.0, 0.8)[None], 3.0)[0]
    assert (q[:, 0] >= 0).all()
    np.testing.assert_allclose(q[:, 2], 3.4)
    neg = scale_band(band(-1.0, 0.5)[None], 2.0)[0]  # net position can be negative: not clipped
    np.testing.assert_allclose(neg[:, 0], -2.0)


def test_dont_use_swaps_substitute_with_its_band_scale():
    j = MemoryVerdicts({("s1", "pvnet", DAY): Verdict("dont_use", p_valid=0.1, substitute="lgbm",
                                                      reasons=("disagrees with pvlib by 58 %",), rule="dont_use"),
                        ("s1", "lgbm", DAY): Verdict("use", band_scale=1.5, rule="p_valid")})
    v = validated("s1", "pvnet", DAY, raw=pred("pvnet", 3.0, 1.0), substitutes={"lgbm": pred("lgbm", 2.0, 0.5)},
                  judge=j)
    assert v.verdict == "dont_use" and v.known and v.source_model == "lgbm"
    np.testing.assert_allclose(v.q50, 2.0)
    np.testing.assert_allclose(v.q90, 2.75)  # substitute's own band scale
    assert v.band_scale == 1.5
    np.testing.assert_allclose(v.raw.q[:, 1], 3.0)  # original kept for display


def test_dont_use_without_substitute_data_is_unknown():
    j = MemoryVerdicts({("s1", "pvnet", DAY): Verdict("dont_use", substitute="lgbm", rule="dont_use")})
    v = validated("s1", "pvnet", DAY, raw=pred("pvnet", 3.0, 1.0), judge=j)
    assert not v.known and v.q50 is None and v.source_model is None
    assert any("lgbm not supplied" in r for r in v.reasons)
    j2 = MemoryVerdicts({("s1", "pvnet", DAY): Verdict("dont_use", rule="no_reliable_source")})
    assert not validated("s1", "pvnet", DAY, raw=pred("pvnet", 3.0, 1.0), judge=j2).known


def test_detector_envelope():
    fp = {"healthy": 0.2, "string_outage": 0.7, "soiling": 0.1, "shading": 0, "clipping": 0, "overvoltage_trip": 0,
          "battery_fade": 0, "sensor_fault": 0}
    p = Prediction(asset_id="s1", model="sitehealth", model_version="v", t0=T0, horizon=H, q10=[0] * H,
                   q50=[0] * H, q90=[0] * H, fault_probs=fp)
    v = validated("s1", "sitehealth", DAY, raw=p)
    assert v.cls == "string_outage" and v.score == pytest.approx(0.8) and v.fault_probs["string_outage"] == 0.7
    j = MemoryVerdicts({("T06", "windnbm", DAY): Verdict("dont_use", substitute="windmon_rules")})
    w = validated("T06", "windnbm", DAY, raw={"cls": "underperformance", "score": 0.9},
                  substitutes={"windmon_rules": {"cls": "fault_stop", "score": 1.0}}, judge=j)
    assert w.cls == "fault_stop" and w.source_model == "windmon_rules" and w.q50 is None


def test_batch_matches_single():
    rng = np.random.default_rng(0)
    n = 50
    ids = [f"s{i}" for i in range(n)]
    mid = rng.uniform(0, 5, (n, H))
    half = rng.uniform(0, 1, (n, H))
    q = np.stack([np.maximum(mid - half, 0), mid, mid + half], axis=-1)
    qs = q * 0.9
    choices = [Verdict("use", band_scale=1.2), Verdict("widen", band_scale=2.0),
               Verdict("dont_use", substitute="lgbm"), Verdict("dont_use", substitute="persistence"), None]
    rows = {}
    for i in range(n):
        c = choices[i % len(choices)]
        if c is not None:
            rows[(ids[i], "pvnet", DAY)] = c
    rows[(ids[2], "lgbm", DAY)] = Verdict("use", band_scale=1.3)
    j = MemoryVerdicts(rows)
    b = validated_many("pvnet", ids, DAY, raw=RawBatch(q=q), substitutes={"lgbm": RawBatch(q=qs)}, judge=j)
    for i in range(n):
        s = validated(ids[i], "pvnet", DAY, raw={"q": q[i]}, substitutes={"lgbm": {"q": qs[i]}}, judge=j)
        assert (b.verdict[i], b.known[i], b.source_model[i], b.band_scale[i], b.rule[i]) == \
               (s.verdict, s.known, s.source_model, s.band_scale, s.rule), i
        assert b.reasons[i] == s.reasons
        if s.known:
            np.testing.assert_allclose(b.q10[i], s.q10)
            np.testing.assert_allclose(b.q90[i], s.q90)
        else:
            assert np.isnan(b.q50[i]).all()
    c = b.counts()
    assert c["dont_use"] == 20 and c["substituted"] == 10 and c["unknown"] == 10
    assert c["use"] == 20 and c["widen"] == 10


def test_sqlite_verdicts(tmp_path):
    db = tmp_path / "s.sqlite"
    con = sqlite3.connect(db)
    con.execute("CREATE TABLE judge_verdicts (run_id, day, asset_id, model, verdict, p_valid, trust, n_eff, "
                "band_scale, substitute, reasons_json, rule, judge_version)")
    con.execute("INSERT INTO judge_verdicts VALUES ('r1', ?, 's1', 'pvnet', 'widen', 0.6, 0.5, 4, 1.8, NULL, "
                "'[\"a\", \"b\"]', 'widen', 'j-1')", (DAY,))
    con.commit()
    con.close()
    assert SqliteVerdicts(db, "other").get("s1", "pvnet", DAY) is None
    j = SqliteVerdicts(db, "r1")
    raw = RawBatch(q=np.stack([band(2.0, 1.0), np.full((H, 3), np.nan)]))
    b = validated_many("pvnet", ["s1", "s2"], [DAY, DAY], raw=raw, judge=j)
    assert list(b.verdict) == ["widen", "use"] and b.reasons[0] == ("a", "b")
    np.testing.assert_allclose(b.q90[0], 3.8)
    assert as_verdict({"verdict": "use", "band_scale": float("nan")}).band_scale == 1.0


def test_memory_verdicts_accept_j4_witness_rows():
    class Row:  # J4 objects name the model "witness"
        asset_id, witness, day, verdict, band_scale, reasons, rule = "s1", "pvnet", DAY, "widen", 1.5, ["x"], "widen"
    j = MemoryVerdicts([Row()])
    v = validated("s1", "pvnet", DAY, raw=pred("pvnet", 3.0, 1.0), judge=j)
    assert v.verdict == "widen" and v.band_scale == 1.5 and v.reasons == ("x",)
