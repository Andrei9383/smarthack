"""Shared pytest setup.

- ``slow`` marker: heavy training/data tests. ``check`` deselects them (``-m "not slow"``) so it stays under ~10 min
  and ``rustcheck`` runs them (``-m slow``), so coverage moves rather than drops. Run locally with ``pytest -m slow``.
- OpenMP threads are capped unless the caller set them. LightGBM's default (num_threads=0 = every core) with
  several runs sharing the lab server oversubscribes the CPU, and the spinning OpenMP workers made the windfailure
  training fixture run past the 900 s check timeout.
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "2")


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: heavy training/data test; deselected in check, run in rustcheck")
