"""All PV / load / site-health baselines as contract Predictors, for eval.py (#16) and the agent fallbacks.

    from sentinel.eval.baselines.pv import predictors
    P = predictors(fleet_dir)          # {"pvlib": {"pv": ...}, "persistence": {"pv", "load"},
                                       #  "lgbm": {"pv", "load"}, "threshold_cusum": {"sitehealth"}}

LightGBM fits on the fleet's training years (2022-2023) and `split == train` sites only, and is cached next to the
fleet dir (`baselines-pv-<fleet>-<key>/`), so the second call loads in about a second.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from .faults import ThresholdCusumPredictor
from .lgbm import LGBMPredictor, QuantileBundle, train
from .persistence import PersistencePredictor, PvlibPredictor, pvlib_band

TRAIN_YEARS = (2022, 2023)
CODE_VERSION = 1  # bump when features / training change, invalidates the cache


def _default_fleet(log=print) -> Path:
    from sentinel.sim.fleet import FleetConfig, generate

    cfg = FleetConfig(n_sites=100, n_feeders=6, n_congested=3, start_year=TRAIN_YEARS[0], end_year=TRAIN_YEARS[-1])
    return generate(cfg, plots=False)


def fit_or_load(fleet_dir: Path | None = None, train_years=TRAIN_YEARS, max_sites: int = 100, rounds: int = 300,
                seed: int = 0, log=print) -> tuple[QuantileBundle, QuantileBundle, tuple[float, float]]:
    from .data import load_arrays

    fleet_dir = Path(fleet_dir) if fleet_dir else _default_fleet(log)
    avail = sorted(int(p.stem.split("=")[1]) for p in (fleet_dir / "telemetry").glob("year=*.parquet"))
    years = [y for y in train_years if y in avail]
    if not years:
        years = avail[:1]
        log(f"baselines: fleet has no {list(train_years)} data; training on {years} instead")
    key = hashlib.sha256(json.dumps([CODE_VERSION, years, max_sites, rounds, seed]).encode()).hexdigest()[:10]
    cache = fleet_dir.parent / f"baselines-pv-{fleet_dir.name}-{key}"
    if (cache / "meta.json").exists():
        meta = json.loads((cache / "meta.json").read_text())
        return (QuantileBundle.load(cache / "lgbm_pv.json"), QuantileBundle.load(cache / "lgbm_load.json"),
                tuple(meta["pvlib_band"]))
    import pandas as pd

    sites = pd.read_parquet(fleet_dir / "sites.parquet", columns=["site_id", "split"])
    ids = sites.site_id.astype(str)[sites.split == "train"].tolist()[:max_sites]
    fa = load_arrays(fleet_dir, years=years, site_ids=ids)
    days = np.ones(len(fa.days), bool)
    all_sites = np.ones(fa.n, bool)
    band = pvlib_band(fa, days, all_sites)
    version = f"lgbm-{fleet_dir.name}-{key}"
    pv = train("pv", fa, days, all_sites, rounds=rounds, seed=seed, log=log)
    load = train("load", fa, days, all_sites, rounds=rounds, seed=seed, log=log)
    pv.version, load.version = f"{version}-pv", f"{version}-load"
    tmp = cache.with_name(cache.name + ".tmp")
    tmp.mkdir(parents=True, exist_ok=True)
    pv.save(tmp / "lgbm_pv.json")
    load.save(tmp / "lgbm_load.json")
    (tmp / "meta.json").write_text(json.dumps({"pvlib_band": band, "years": years, "n_sites": fa.n,
                                               "fleet_dir": str(fleet_dir)}))
    if cache.exists():
        import shutil
        shutil.rmtree(cache)
    tmp.rename(cache)
    log(f"baselines: LightGBM cached at {cache}")
    return pv, load, band


def predictors(fleet_dir: Path | None = None, **fit_kw) -> dict[str, dict[str, object]]:
    """baseline name -> {kind: Predictor}, kinds in pv | load | sitehealth (see module doc)."""
    pv, load, band = fit_or_load(fleet_dir, **fit_kw)
    return {
        "pvlib": {"pv": PvlibPredictor(band=band)},
        "persistence": {"pv": PersistencePredictor("pv_d1"), "load": PersistencePredictor("load_d1")},
        "lgbm": {"pv": LGBMPredictor(pv), "load": LGBMPredictor(load)},
        "threshold_cusum": {"sitehealth": ThresholdCusumPredictor()},
    }
