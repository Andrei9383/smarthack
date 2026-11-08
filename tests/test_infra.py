"""wattsentry.infra harness (#47): metrics, stage timing, RunContext, manifest, memmap cache, ONNX export."""
import json

import numpy as np
import pytest

from wattsentry.infra import Metrics, RunContext, emit, stage, window_cache


def test_emit_prints_and_merges_json(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("LAB_METRICS", str(tmp_path / "m.json"))
    emit("val nmae", 0.25)
    m = Metrics("wind")
    m.emit("val_nmae", 0.5)
    out = capsys.readouterr().out
    assert "LAB_METRIC val_nmae=0.25" in out and "LAB_METRIC wind_val_nmae=0.5" in out
    assert json.loads((tmp_path / "m.json").read_text()) == {"val_nmae": 0.25, "wind_val_nmae": 0.5}
    assert m.values == {"wind_val_nmae": 0.5}


def test_stage_emits_timing_and_rss(capsys):
    t = {}
    with stage("windows", timings=t):
        sum(range(1000))
    out = capsys.readouterr().out
    assert "LAB_METRIC t_windows_s=" in out and "LAB_METRIC rss_windows_mb=" in out
    assert "t_windows_s" in t


def test_run_context_seeds_and_manifest(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("LAB_OUTPUT_DIR", str(tmp_path / "out"))
    data = tmp_path / "data"
    data.mkdir()
    (data / "a.parquet").write_bytes(b"x")
    a = RunContext("demo", seed=3, config={"lr": 1e-3})
    with a.stage("fit"):
        a.emit("val_nmae", 0.1)
    r1 = (np.random.rand(), a.rng.random())
    b = RunContext("demo", seed=3, out=tmp_path / "b")
    assert (np.random.rand(), b.rng.random()) == r1
    assert json.loads((a.out / "config.json").read_text()) == {"lr": 1e-3}
    m = json.loads(a.manifest(data=[data]).read_text())
    assert "LAB_METRIC demo_t_fit_s=" in capsys.readouterr().out
    assert len(m["git_sha"]) >= 7 and m["kernels"]["backend"] in ("rust", "python")
    assert list(m["data"].values())[0] == "2d711642b726b04401627ca9fbac32f5c8530fb1903cc4db02258717921a4881"
    assert m["config"] == {"lr": 1e-3} and "demo_t_fit_s" in m["metrics"] and "numpy" in m["versions"]


def test_window_cache_builds_once(tmp_path):
    calls = []

    def build():
        calls.append(1)
        return np.arange(12, dtype=np.float32).reshape(3, 4)

    a = window_cache("h1", {"len": 4}, build, root=tmp_path)
    b = window_cache("h1", {"len": 4}, build, root=tmp_path)
    assert len(calls) == 1 and isinstance(b, np.memmap) and b.dtype == np.float16
    np.testing.assert_array_equal(a, b)
    window_cache("h1", {"len": 5}, build, root=tmp_path)
    window_cache("h2", {"len": 4}, build, root=tmp_path)
    assert len(calls) == 3 and not list(tmp_path.glob("*.tmp"))


def test_window_cache_env_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("LAB_CACHE_DIR", str(tmp_path))
    window_cache("h", {}, lambda: np.zeros(2))
    assert len(list((tmp_path / "windows").glob("*.npy"))) == 1


def test_export_onnx_parity(tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("onnxruntime")
    pytest.importorskip("onnx")
    from wattsentry.infra.export import export_onnx

    model = torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.ReLU(), torch.nn.Linear(8, 2))
    p = tmp_path / "m.onnx"
    parity = export_onnx(model, (torch.zeros(1, 4),), p, ["x"], ["y"], dynamic_axes={"x": {0: "B"}, "y": {0: "B"}},
                         check_inputs=(torch.randn(5, 4),), calibration={"q": [0.1, 0.9]})
    assert parity < 1e-4 and p.exists()
    assert json.loads(p.with_suffix(".calibration.json").read_text()) == {"q": [0.1, 0.9]}
