"""Judge priors: score functions and per-bucket aggregation, on small arrays (no fleet)."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from wattsentry.judge import priors as P
from wattsentry.judge.buckets import ROOT, bucket_key


def test_calibration_peaks_at_80_percent_hit():
    y = np.tile(np.arange(10.0), (3, 1))
    m = np.ones_like(y, bool)
    # band [0, 7] covers 8 of 10 -> hit 0.8 -> 1.0; [0, 9] covers all -> 0.75; [20, 30] none -> 0
    q10 = np.array([[0.0] * 10, [0.0] * 10, [20.0] * 10])
    q90 = np.array([[7.0] * 10, [9.0] * 10, [30.0] * 10])
    s = P.calibration_score(q10, q90, y, m)
    np.testing.assert_allclose(s, [1.0, 0.75, 0.0])


def test_calibration_ignores_masked_and_missing_steps():
    y = np.array([[1.0, 2.0, np.nan, 100.0]])
    m = np.array([[True, True, True, False]])
    s = P.calibration_score(np.zeros_like(y), np.full_like(y, 5.0), y, m)  # hit 1.0 on the 2 scored steps
    np.testing.assert_allclose(s, [0.75])
    assert np.isnan(P.calibration_score(np.zeros((1, 2)), np.ones((1, 2)), np.full((1, 2), np.nan),
                                        np.ones((1, 2), bool)))[0]


def test_interval_score_and_wide_bands_lose_sharpness():
    y = np.array([[1.0, 2.0, 3.0]])
    assert P.interval_score(np.array([0.0]), np.array([2.0]), np.array([3.0]))[0] == 2 + 10 * 1.0
    m = np.ones_like(y, bool)
    tight = P.sharpness_score(y - 0.1, y + 0.1, y, y - 1, y + 1, m)  # 0.2 vs 2.0 width -> 0.9
    wide = P.sharpness_score(y - 5, y + 5, y, y - 1, y + 1, m)       # wider than the reference -> 0
    np.testing.assert_allclose(tight, [0.9])
    np.testing.assert_allclose(wide, [0.0])


def test_detector_scores_split_by_claim():
    alarm = np.array([True, True, True, False, False])
    called = ["soiling", "soiling", "shading", "healthy", "healthy"]
    truth = ["soiling", "shading", "healthy", "string_outage", "healthy"]
    ap, ms = P.detector_scores(alarm, called, truth)
    np.testing.assert_allclose(ap[:3], [1.0, 0.5, 0.0])
    assert np.isnan(ap[3:]).all()
    np.testing.assert_allclose(ms[3:], [0.0, 1.0])
    assert np.isnan(ms[:3]).all()


def test_aggregate_fills_every_ancestor_down_to_root():
    b1 = bucket_key("site", True, True, "warm", "clean", False)
    b2 = bucket_key("site", False, True, "cold", "clean", True)
    rows = pd.DataFrame({"model": ["pvnet"] * 3, "score_name": ["calibration"] * 3,
                         "bucket": [b1, b1, b2], "score": [1.0, 0.5, 0.0]})
    out = P.aggregate(rows)["pvnet"]["calibration"]
    assert out[b1] == {"prior": 0.75, "n": 2}
    assert out[b2] == {"prior": 0.0, "n": 1}
    assert out["site|seen"]["n"] == 2 and out["site|unseen"]["n"] == 1
    assert out["site"] == out[ROOT] == {"prior": 0.5, "n": 3}
    assert len(out) == 1 + 1 + 2 * 5  # root, kind, then two distinct branches of 5 levels


def test_aggregate_drops_nan_scores():
    rows = pd.DataFrame({"model": ["m", "m"], "score_name": ["s", "s"], "bucket": ["site", "site"],
                         "score": [np.nan, 0.4]})
    assert P.aggregate(rows)["m"]["s"][ROOT] == {"prior": 0.4, "n": 1}


def test_report_priors_cover_wind_and_ledger_only_models(tmp_path):
    nbm, mon, ver = P.wind_priors(Path(__file__).resolve().parents[1] / "models" / "reports"
                                  / "r135-windnbm.report.json")
    for sc in (nbm, mon):
        for s in ("alarm_precision", "miss_score"):
            assert 0 <= sc[s][ROOT]["prior"] <= 1 and "turbine|unseen" in sc[s]
    assert nbm["alarm_precision"][ROOT]["prior"] > mon["alarm_precision"][ROOT]["prior"]
    lo = P.ledger_only_priors("local")
    assert {"demandnet", "windfailure", "gridtwin", "residual_trend", "gis_recorded"} <= set(lo)
    assert lo["windfailure"]["scores"]["hit"][ROOT]["prior"] > 0.9
    assert P.main(["--out", str(tmp_path), "--no-fleet"]) == 0
    jp = json.loads((tmp_path / "judge_priors.json").read_text())
    assert jp["split"] == "val-2024" and "windmon_rules" in jp["models"]
    assert json.loads((tmp_path / "windnbm.priors.json").read_text())["priors"]["source"] == "report"


def test_apply_from_log_adds_only_the_priors_key(tmp_path):
    import shutil
    repo = Path(__file__).resolve().parents[1]
    models = tmp_path / "models"
    shutil.copytree(repo / "models", models, ignore=shutil.ignore_patterns("*.onnx", "*.txt", "pvnet_fc"))
    out = tmp_path / "out"
    out.mkdir()
    block = P.priors_block({"calibration": {ROOT: {"prior": 0.8, "n": 3}}}, "validation", "eval#1")
    (out / "promoted_priors.json").write_text(json.dumps({m: block for m in P.PROMOTED}))
    (out / "windnbm.priors.json").write_text(json.dumps({"model": "windnbm", "priors": block}))
    (out / "judge_priors.json").write_text(json.dumps({"version": 1, "models": {}}))
    log = tmp_path / "run.log"
    log.write_text("noise\n" + "\n".join(f"[t] {P.LOG_TAG} {n} {(out / n).read_text()}" for n in P.OUT_FILES))
    before = json.loads((models / "pvnet.s2.calibration.json").read_text())
    assert P.main(["--apply", str(log), "--models", str(models)]) == 0
    after = json.loads((models / "pvnet.s2.calibration.json").read_text())
    before.pop("priors", None)  # models/ may already carry priors; apply replaces only that key
    assert set(after) - set(before) == {"priors"} and {k: after[k] for k in before} == before
    assert after["priors"]["scores"]["calibration"][ROOT]["prior"] == 0.8
    assert (models / "judge_priors.json").exists() and (models / "windnbm.priors.json").exists()
