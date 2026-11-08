"""wattsentry.infra: metrics, stage timing, ONNX export."""
import json

import pytest

from wattsentry.infra import emit, stage


def test_emit_prints_and_merges_json(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("LAB_METRICS", str(tmp_path / "m.json"))
    emit("val nmae", 0.25)
    emit("val_nmae", 0.5, "wind")
    out = capsys.readouterr().out
    assert "LAB_METRIC val_nmae=0.25" in out and "LAB_METRIC wind_val_nmae=0.5" in out
    assert json.loads((tmp_path / "m.json").read_text()) == {"val_nmae": 0.25, "wind_val_nmae": 0.5}


def test_stage_emits_timing_and_rss(capsys):
    with stage("windows"):
        sum(range(1000))
    out = capsys.readouterr().out
    assert "LAB_METRIC t_windows_s=" in out and "LAB_METRIC rss_windows_mb=" in out


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
