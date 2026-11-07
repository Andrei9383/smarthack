"""Backend equivalence for every registered kernel (#179, team/infra.md §1).

For each kernel in ``sentinel.kernels`` the Python reference and the Rust copy must give the same output on random
inputs. Inputs come from the bench cases (``sentinel.kernels.bench.CASES``) at small scales and several seeds, so a
kernel added with a bench entry is covered here without editing this file. Comparison is the bench's own rule
(bit-exact, or the case's stated float tolerance). Skipped when ``sentinel._core`` is not built (check); runs in
``rustcheck``.
"""

from __future__ import annotations

import numpy as np
import pytest

from sentinel import kernels
from sentinel.kernels import bench

needs_rust = pytest.mark.skipif(not kernels.rust_available(), reason="sentinel._core not built")

SEEDS = (0, 1, 2)
SCALES = (0.01, 0.05, 0.2)


def _case_id(c) -> str:
    return c[1]


def _registered_rust() -> dict:
    return {n: rs for n, (_, rs) in kernels._REGISTRY.items()}


def _outputs(x):
    """Kernels return one array or a tuple of arrays; compare them as a tuple."""
    return tuple(x) if isinstance(x, (tuple, list)) else (x,)


def test_every_kernel_has_a_bench_case():
    """A kernel without a bench case has no speedup number and no coverage below (team/infra.md §1.3)."""
    covered = {c[0] for c in bench.CASES}
    missing = sorted(set(kernels.names()) - covered)
    assert not missing, f"kernels registered without a bench case in sentinel/kernels/bench.py: {missing}"


def test_every_bench_case_is_a_registered_kernel():
    unknown = sorted({c[0] for c in bench.CASES} - set(kernels.names()))
    assert not unknown, f"bench cases for kernels that are not registered: {unknown}"


@needs_rust
def test_every_kernel_has_a_rust_copy():
    python_only = sorted(n for n, rs in _registered_rust().items() if rs is None)
    assert not python_only, f"registered without a Rust implementation: {python_only}"


@needs_rust
@pytest.mark.parametrize("scale", SCALES)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("case", bench.CASES, ids=_case_id)
def test_python_and_rust_agree(case, seed, scale):
    kernel, _key, make, _run, sample, _desc, *tol = case
    args = sample(make(scale, np.random.default_rng(seed)))
    py = _outputs(kernels.get(kernel, "python")(*args))
    rs = _outputs(kernels.get(kernel, "rust")(*args))
    assert len(py) == len(rs), f"{kernel}: {len(py)} outputs (python) vs {len(rs)} (rust)"
    for i, (a, b) in enumerate(zip(py, rs)):
        a, b = np.asarray(a), np.asarray(b)
        assert a.shape == b.shape and a.dtype == b.dtype, f"{kernel} output {i}: {a.shape}/{a.dtype} vs {b.shape}/{b.dtype}"
    assert bench._same(py, rs, *tol), f"{kernel}: Rust output differs from the Python reference (seed={seed}, scale={scale})"


@needs_rust
@pytest.mark.parametrize("case", bench.CASES, ids=_case_id)
def test_dispatch_env_picks_backend(case, monkeypatch):
    """SENTINEL_KERNELS=python|rust routes ``get`` to the matching implementation."""
    kernel = case[0]
    py, rs = kernels._REGISTRY[kernel]
    monkeypatch.setenv("SENTINEL_KERNELS", "python")
    assert kernels.get(kernel) is py
    monkeypatch.setenv("SENTINEL_KERNELS", "rust")
    assert kernels.get(kernel) is rs
