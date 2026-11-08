"""Card #140: train-loop throughput accounting and the cached per-epoch validation (same numbers as collect())."""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from wattsentry.model.data import ArraySource, H  # noqa: E402
from wattsentry.model.harness import Throughput, compile_net  # noqa: E402
from wattsentry.model.synth import SynthConfig, generate  # noqa: E402


def test_throughput_metrics_cpu():
    got = {}
    tp = Throughput(lambda k, v: got.__setitem__(k, v), "cpu")
    for _ in range(3):
        with tp.wait():
            pass
        with tp.compute(8):
            sum(range(1000))
    with tp.val():
        pass
    out = tp.finish()
    for k in ("samples_per_s", "gpu_util_mean", "t_data_wait_s", "t_compute_s", "t_val_s", "data_wait_frac", "ms_per_step"):
        assert k in got and k in out
    assert got["samples_per_s"] > 0 and np.isnan(got["gpu_util_mean"])  # no GPU -> NaN (not emitted by RunContext)
    assert compile_net("x", "off") == "x"


@pytest.fixture(scope="module")
def small_src(tmp_path_factory):
    return ArraySource(generate(SynthConfig(n_sites=10, days=60, seed=5), tmp_path_factory.mktemp("s") / "a",
                                verbose=False))


@pytest.mark.parametrize("model", ["pvnet", "loadnet"])
def test_val_cache_matches_collect(small_src, model):
    pytest.importorskip("torch")
    from wattsentry.model import features as F
    from wattsentry.model import forecast as FC
    from wattsentry.model import metrics as M

    kind = FC.SPECS[model]["kind"]
    args = FC.parse_args(model, ["--d-model", "16", "--ff", "32", "--layers", "1", "--pointwise", "8"])
    rng = np.random.default_rng(0)
    n = 40
    sites = rng.integers(0, len(small_src.sites), n)
    starts = rng.integers(F.HIST_DAYS + 1, small_src.n_days - 1, n) * H
    probe = [F.build(kind, F.gather(small_src, sites, starts, kind, "nowcast"))]
    stats = F.fit_stats(probe)
    dims = {k: probe[0][k].shape[-1] for k in ("seq", "ref", "summ", "stat")}
    net = FC.build_model(model, args, dims, stats)
    c = FC.collect(small_src, kind, net, sites, starts, "nowcast", stats, "cpu", None, batch=16)
    ref_p = M.pinball(c["q"] / c["scale"][:, None, None], c["y"] / c["scale"][:, None], c["w"] > 0)
    ref_n = M.nmae(c["q"][..., 1], c["y"], c["w"] > 0)
    vc = FC.val_inputs(small_src, kind, sites, starts, "nowcast", stats, "cpu", batch=16)
    vp, vn = FC.val_score(net, vc, None, batch=7)
    assert vp == pytest.approx(ref_p, rel=1e-5) and vn == pytest.approx(ref_n, rel=1e-5)
