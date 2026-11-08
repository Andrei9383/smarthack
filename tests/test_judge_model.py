"""Judge validity model on synthetic rows: calibration, thresholds, persistence, reasons."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from wattsentry.judge import features as F
from wattsentry.judge import model as JM

lgb = pytest.importorskip("lightgbm")


def synthetic_rows(n: int = 3000, seed: int = 0) -> pd.DataFrame:
    """Forecast + detector rows whose validity depends on trust, OOD and disagreement with the substitute."""
    rng = np.random.default_rng(seed)
    model = rng.choice(["pvnet", "loadnet", "sitehealth"], n)
    trust = rng.uniform(0.2, 0.95, n)
    ood = rng.gamma(1.5, 1.0, n)
    e50 = rng.uniform(5, 30, n)
    sub = e50 * (1 + rng.normal(0, 0.3, n))
    logit = 4 * (trust - 0.55) - 0.6 * (ood - 1.5) - 3 * np.abs(e50 - sub) / np.maximum(sub, 0.1)
    valid = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(float)
    df = pd.DataFrame({
        "model": model, "model_version": "t", "asset_id": [f"S{i % 300:04d}" for i in range(n)], "day": "2024-06-01",
        "kind": "site", "family": [F.family_of(m) for m in model], "battery": rng.integers(0, 2, n),
        "seen": rng.integers(0, 2, n), "month": rng.integers(1, 13, n), "sky": rng.integers(0, 3, n),
        "ood_score": ood, "ood_thr": 3.0, "member_std": rng.uniform(0, 0.5, n), "energy_q50": e50,
        "band_rel": rng.uniform(0.1, 1.0, n), "energy_scale": 25.0, "sub_model": "pvlib", "sub_energy": sub,
        "trust_main": trust, "prior_main": 0.7, "n_eff": rng.uniform(0, 20, n), "trust_trend3": rng.normal(0, .05, n),
        "trust_cal": trust, "trust_sharp": trust, "trust_ap": trust, "trust_ms": trust, "band_scale": 1.0,
        "health_alert": 0.0, "alarm": rng.integers(0, 2, n), "p_fault": rng.uniform(0, 1, n),
        "top_class": rng.choice(F.PV_CLASSES, n), "valid": valid})
    for m in (0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0, 6.0):
        df[f"y_hit_m{m:g}"] = np.clip(0.55 + 0.1 * m + 0.2 * (trust - 0.5), 0, 1)
    return df


def test_isotonic_is_monotone_and_calibrated():
    rng = np.random.default_rng(1)
    p_raw = rng.uniform(0, 1, 5000)
    y = (rng.random(5000) < p_raw ** 2).astype(float)  # raw scores overconfident
    iso = JM.Isotonic.fit(p_raw, y)
    grid = np.linspace(0, 1, 101)
    out = iso(grid)
    assert np.all(np.diff(out) >= -1e-12)
    assert out.min() >= 0 and out.max() <= 1
    p = iso(p_raw)
    for lo, hi in ((0.0, 0.3), (0.3, 0.6), (0.6, 1.0)):  # reliability: mean P(valid) ~ observed rate per bin
        m = (p >= lo) & (p < hi)
        assert abs(p[m].mean() - y[m].mean()) < 0.05


def test_tau_use_meets_precision_rule():
    rng = np.random.default_rng(2)
    p = rng.uniform(0, 1, 4000)
    y = (rng.random(4000) < p).astype(float)
    tau = JM.tau_for_precision(p, y, 0.9)
    assert tau < 1.0
    assert y[p >= tau].mean() >= 0.9
    # the lowest such threshold: lowering it to the next distinct p breaks the rule
    lower = np.max(p[p < tau])
    assert y[p >= lower].mean() < 0.9 or (p >= lower).sum() < JM.MIN_USE
    assert JM.tau_for_precision(np.full(100, 0.5), np.zeros(100)) > 1.0  # impossible -> never use


def test_widen_choice_reaches_calibration_target():
    rng = np.random.default_rng(3)
    n = 2000
    p = rng.uniform(0, 1, n)
    mults = (0.5, 1.0, 1.5, 2.0, 3.0)
    hit_m = {m: np.clip(0.4 + 0.15 * m * p + 0.05 * m, 0, 1) for m in mults}
    tau_w, factor = JM.widen_choice(p, hit_m, np.ones(n), tau_use=0.8)
    assert 0.0 < tau_w < 0.8 and factor > 1.0
    z = (p >= tau_w) & (p < 0.8)
    cal = JM.calibration_score(JM.widened_hit(hit_m, np.ones(n), factor))
    assert cal[z].mean() >= 0.8 - 1e-9
    # widening never rescues rows whose bands only reach coverage by becoming useless: hit 1.0 scores 0.75
    always = {m: np.ones(n) for m in mults}
    assert JM.widen_choice(p, always, np.ones(n), tau_use=0.8) == (0.8, 1.0)
    np.testing.assert_allclose(JM.widened_hit(hit_m, np.full(n, 2.0), 0.75), hit_m[1.5])


def test_train_save_load_predict_roundtrip(tmp_path):
    df = synthetic_rows()
    fams = {}
    for fam in F.FAMILIES:
        t = df[df.family == fam].reset_index(drop=True)
        X, y = F.build(t, fam), t.valid.to_numpy()
        bs, info = JM.fit_family(X, y, t.asset_id.to_numpy(), fam, seeds=range(2), rounds=60, threads=2)
        fm = JM.FamilyModel(fam, bs, list(X.columns), JM.Isotonic())
        raw = fm.raw(X).mean(0)
        fm.iso = JM.Isotonic.fit(raw, y)
        fm.tau_use = JM.tau_for_precision(fm.iso(raw), y)
        fams[fam] = fm
        assert len(bs) == 2 and info["n_train"] > 0
    judge = JM.Judge(fams, {"judge_version": "test", "tau_use": fams["forecast"].tau_use, "tau_widen": 0.5,
                            "widen_factor": 1.5})
    files = judge.save(tmp_path)
    assert (tmp_path / "judge.forecast.s0.txt").exists() and (tmp_path / JM.CAL_FILE).exists()
    cal = json.loads((tmp_path / JM.CAL_FILE).read_text())
    for k in ("tau_use", "tau_widen", "widen_factor", "judge_version", "families"):
        assert k in cal
    loaded = JM.load_judge(tmp_path)
    assert loaded is not None and set(loaded.families) == {"forecast", "detector"}
    sample = df.head(50)
    a, b = judge.predict(sample), loaded.predict(sample)
    np.testing.assert_allclose(a.p_valid, b.p_valid, atol=1e-9)
    assert {"p_valid", "p_valid_sd", "reasons"} <= set(b.columns)
    assert b.p_valid.between(0, 1).all() and (b.p_valid_sd >= 0).all()
    assert all(isinstance(r, list) and len(r) <= 3 for r in b.reasons)
    assert any(len(r) > 0 for r in b.reasons)
    # trust drives validity in the synthetic data: high-trust rows score higher on average
    full = loaded.predict(df, reasons=False)
    assert full.p_valid[df.trust_main > 0.8].mean() > full.p_valid[df.trust_main < 0.35].mean() + 0.15
    assert len(files) == 5


def test_ledger_only_models_use_trust(tmp_path):
    df = synthetic_rows(400)
    fam = "forecast"
    t = df[df.family == fam].reset_index(drop=True)
    X = F.build(t, fam)
    bs, _ = JM.fit_family(X, t.valid.to_numpy(), t.asset_id.to_numpy(), fam, seeds=[0], rounds=20, threads=1)
    judge = JM.Judge({fam: JM.FamilyModel(fam, bs, list(X.columns), JM.Isotonic())}, {"judge_version": "t"})
    rows = pd.DataFrame({"model": ["windfailure", "gridtwin"], "trust_main": [0.42, np.nan]})
    out = judge.predict(rows)
    assert out.p_valid.iloc[0] == pytest.approx(0.42)
    assert out.family.tolist() == ["ledger_only", "ledger_only"]
    assert "ledger only" in out.reasons.iloc[0][0]


def test_load_judge_none_without_weights(tmp_path):
    assert JM.load_judge(tmp_path) is None
    (tmp_path / JM.CAL_FILE).write_text(json.dumps({"families": {"forecast": {"features": ["x"],
                                                                              "seeds": ["judge.forecast.s0.txt"]}}}))
    assert JM.load_judge(tmp_path) is None  # calibration without its LightGBM file


def test_verdict_metrics_baselines():
    y = np.array([1, 1, 0, 0, 1, 0], float)
    trust_all = JM.verdict_metrics(np.ones(6), y, 0.0)
    assert trust_all["precision_use"] == pytest.approx(0.5) and trust_all["recall_dont_use"] == 0.0
    perfect = JM.verdict_metrics(y, y, 0.5)
    assert perfect["precision_use"] == 1.0 and perfect["recall_dont_use"] == 1.0 and perfect["brier"] == 0.0
    # rule 3 forces "use" on no-record rows even when the model says otherwise
    r3 = JM.verdict_metrics(np.zeros(6), y, 0.5, use_rule3=np.array([1, 0, 0, 0, 0, 0], bool))
    assert r3["share_use"] == pytest.approx(1 / 6)
