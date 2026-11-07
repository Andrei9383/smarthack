"""Card #185: the M4 batch builder must not copy whole arrays per batch, and the cached site metadata must match.

Runs under the lab check (requirements.txt only, no torch).
"""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sentinel.model import features as F  # noqa: E402
from sentinel.model.data import ArraySource, H  # noqa: E402
from sentinel.model.synth import SynthConfig, generate  # noqa: E402

SMALL = SynthConfig(n_sites=8, days=40, seed=5)


@pytest.fixture(scope="module")
def src(tmp_path_factory):
    return ArraySource(generate(SMALL, tmp_path_factory.mktemp("synth185") / "a", verbose=False))


def test_site_meta_cached_matches_fresh_and_is_read_only(src):
    for key, dt in (("kwp_recorded", np.float32), ("battery_kwh", np.float32), ("loc_idx", np.int64)):
        fresh = np.array([(s.get(key) if s.get(key) is not None else 0) for s in src.sites], dtype=dt)
        a, b = src.site_meta(key, dt), src.site_meta(key, dt)
        assert a is b and a.dtype == dt and np.array_equal(a, fresh)
        assert not a.flags.writeable
    assert src.site_meta("loc_idx", np.float32).dtype == np.float32  # dtype is part of the cache key
    assert np.array_equal(src.loc_idx(), src.site_meta("loc_idx", np.int64))


def test_site_gather_contiguous_m2_identical(src):
    """sitehealth makes m2 contiguous once; the gathered windows are the same as from the strided view."""
    S, T = len(src.sites), src.n_steps
    rng = np.random.default_rng(0)
    m1 = rng.random((S, T, 3)).astype(np.float16)
    m2_view = m1[..., 1]
    m2 = np.ascontiguousarray(m2_view)
    assert not m2_view.flags.c_contiguous and m2.flags.c_contiguous
    sites = rng.integers(0, S, 32)
    starts = rng.integers(F.HIST_DAYS * H, T - H, 32)
    a = F.gather(src, sites, starts, "site", m1=m1, m2=m2_view)
    b = F.gather(src, sites, starts, "site", m1=m1, m2=m2)
    assert a.keys() == b.keys()
    for k in a:
        assert np.array_equal(np.asarray(a[k]), np.asarray(b[k])), k
