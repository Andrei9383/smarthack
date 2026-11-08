"""Real-model adapters for the replay: wattsentry/agent/predictors.py.

ONNX tests skip when models/ has no promoted weights; the SimWorld test skips without the window cache."""

from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest

from wattsentry.contracts import Window

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"
HAVE_MODELS = (MODELS / "pvnet.s0.onnx").exists() and (MODELS / "sitehealth.onnx").exists()
needs_models = pytest.mark.skipif(not HAVE_MODELS, reason="no promoted ONNX weights in models/")


def _ts(*a) -> int:
    return int(datetime(*a, tzinfo=timezone.utc).timestamp())


@pytest.fixture(scope="module")
def synth_windows(tmp_path_factory):
    pytest.importorskip("onnxruntime")
    from wattsentry.model.synth import SynthConfig, generate

    out = tmp_path_factory.mktemp("win") / "w"
    return generate(SynthConfig(n_sites=6, start="2025-05-01", days=40, seed=3), out=out, verbose=False)


@pytest.fixture(scope="module")
def bank(synth_windows):
    from wattsentry.agent.predictors import FleetSlice, ModelBank

    src = FleetSlice(synth_windows, list(range(6)), _ts(2025, 5, 3), _ts(2025, 6, 9))
    return ModelBank(src, MODELS)


@needs_models
def test_bank_runs_all_members(bank):
    assert not bank.errors, bank.errors
    assert set(bank.q) == {"pv_fc", "pv_now", "load", "load_now"}
    w0 = bank.src.step(_ts(2025, 6, 1, 21))
    for q in bank.q.values():
        x = q[:, w0:w0 + 7 * 96]
        assert np.isfinite(x).all() and (x[..., 0] <= x[..., 1] + 1e-6).all() and (x[..., 1] <= x[..., 2] + 1e-6).all()
    h = bank.health([0, 1], w0 + 48)
    for probs, pstd, ood in h.values():
        assert probs.shape == (8,) and abs(probs.sum() - 1) < 1e-5 and pstd >= 0 and ood >= 0


@needs_models
def test_adapters_real_and_fallback(bank):
    from wattsentry.agent.predictors import LoadForecastAdapter, PvForecastAdapter, SiteHealthAdapter

    index = {s["site_id"]: i for i, s in enumerate(bank.src.sites)}
    step_of = lambda t0: bank.src.step(int(datetime.fromisoformat(t0.replace("Z", "+00:00")).timestamp()))  # noqa
    row = dict(bank.src.sites[0])
    t0 = "2025-06-03T21:00:00Z"
    for ad in (PvForecastAdapter(bank, "pv_fc", "pvnet", index, step_of),
               LoadForecastAdapter(bank, "load", "loadnet", index, step_of)):
        p = ad.predict(row, Window(t0=t0, horizon=96))
        assert p.extra["is_fallback"] == 0 and not p.model_version.startswith("mock") and len(p.q50) == 96
    sh = SiteHealthAdapter(bank, index, step_of)
    p = sh.predict(row, Window(t0=t0, horizon=96, meta={"tick": 48}))
    assert p.fault_probs and p.extra["is_fallback"] == 0 and p.extra["assessed_until_tick"] == 48
    unknown = {**row, "site_id": "nope"}
    fb = sh.predict(unknown, Window(t0=t0, horizon=96))
    assert fb.extra["is_fallback"] == 1 and sh.n_fallback == 1


def test_simworld_missing_cache_is_clear(tmp_path, monkeypatch):
    from wattsentry.agent.world import SimWorld

    monkeypatch.setenv("WATTSENTRY_WINDOWS", str(tmp_path / "none"))
    with pytest.raises(FileNotFoundError, match="window cache"):
        SimWorld()


def _have_windows() -> bool:
    try:
        from wattsentry.agent.world import find_windows

        d = find_windows()
        return d is not None and (Path(d) / "index.json").exists() and not os.environ.get("CI")
    except Exception:
        return False


@needs_models
@pytest.mark.skipif(not _have_windows(), reason="no window cache")
def test_simworld_demo_ids_and_story():
    from wattsentry.agent.world import HERO_FEEDER_ID, STORY_SITE_ID, SimWorld

    w = SimWorld(wind=False)
    assert w.week_start == "2025-06-02" and w.day_t0(0) == "2025-06-01T21:00:00Z"
    assert HERO_FEEDER_ID in w.feeders and STORY_SITE_ID in w.feeders[HERO_FEEDER_ID].site_ids
    assert w.sites[STORY_SITE_ID].battery is not None
    preds = w.default_predictors()
    assert all(not str(preds[k].model_version).startswith("mock") for k in ("pv", "load", "sitehealth"))
    before = w.site_day(STORY_SITE_ID, 2).pv_kw
    after = w.site_day(STORY_SITE_ID, 4).pv_kw
    assert max(after) < 0.8 * max(before)  # one of two strings open from day 3
