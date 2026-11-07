"""data-v2 homes (spec I step 2a): PV + household from the fleet simulator's site sampler, plus contract, meter and a
battery strategy assigned from the config shares. PPC knows `PUBLIC_COLUMNS`; everything else is hidden truth.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from sentinel.kernels.strategies import STRATEGY_CODES
from sentinel.sim.fleet.config import FleetConfig
from sentinel.sim.fleet.sites import build_world

from .config import ProsumerConfig

# What PPC knows about each home (training features may use only these, plus meter data, weather and prices).
PUBLIC_COLUMNS = ["home_id", "locality_id", "locality", "county", "lat", "lon", "kwp", "tilt_recorded",
                  "azimuth_recorded", "inverter_kw", "install_year", "contract", "meter", "battery_declared",
                  "enrolled", "split"]
# Hidden truth that must never reach a training file (tests/test_data_v2.py).
BATTERY_HIDDEN = ["has_battery", "strategy", "cap_kwh", "p_kw", "reserve_frac", "k_top", "eta"]
_FLEET_DROP = ["site_id", "feeder_id", "phase", "gis_feeder_id", "gis_phase", "battery_kwh", "battery_kw",
               "eta_charge", "eta_discharge", "soc_min", "soc_max", "comfort_reserve_kwh", "true_feeder_position",
               "siruta", "kwp_recorded"]


def build_homes(cfg: ProsumerConfig) -> pd.DataFrame:
    """One row per home: public columns first, then battery truth, then the fleet's hidden PV/household truth.
    The fleet sampler is called with its own (new) config, so the existing fleet's output is untouched."""
    fcfg = FleetConfig(seed=cfg.seed, n_sites=cfg.n_homes, n_feeders=cfg.n_localities, n_congested=0,
                       battery_share=0.0, weather_source=cfg.weather_source)
    _, sites = build_world(fcfg)
    n = len(sites)
    rng = np.random.default_rng([cfg.seed, 172, 1])

    has = rng.random(n) < cfg.share_battery
    mc = np.where(has, rng.random(n) < cfg.share_maxcontrol_of_battery,
                  rng.random(n) < cfg.share_maxcontrol_no_battery)
    # MaxControl needs a smart meter; the others get one with the probability that keeps the overall share.
    p_mc = mc.mean() if n else 0.0
    p_smart_rest = float(np.clip((cfg.share_smart_meter - p_mc) / max(1 - p_mc, 1e-9), 0, 1))
    smart = mc | (rng.random(n) < p_smart_rest)
    u_strat = rng.random(n)
    enrolled = has & ~mc & smart & (rng.random(n) < cfg.share_enrolled)
    strategy = np.select(
        [~has, mc & (u_strat < cfg.share_price_driven_of_maxcontrol_battery), enrolled],
        ["none", "price_driven", "enrolled"], "self_consumption")
    declared = has & ((rng.random(n) < cfg.share_battery_declared) | enrolled)  # enrolling = declaring

    lo, hi = cfg.battery_cap_kwh
    cap = np.where(has, np.round(rng.uniform(lo, hi, n) * 2) / 2, 0.0)
    lo, hi = cfg.battery_p_kw
    p_kw = np.where(has, np.round(rng.uniform(lo, hi, n), 1), 0.0)
    lo, hi = cfg.reserve_frac
    reserve = np.where(has, np.round(rng.uniform(lo, hi, n), 3), 0.0)
    lo, hi = cfg.k_top_intervals
    k_top = np.where(has, rng.integers(lo, hi + 1, n), 0).astype(np.int32)

    homes = pd.DataFrame({
        "home_id": [f"H{i + 1:05d}" for i in range(n)],
        "kwp": sites.kwp_recorded.to_numpy(),
        "contract": np.where(mc, "maxcontrol", "standard"),
        "meter": np.where(smart, "smart", "profile"),
        "battery_declared": declared,
        "enrolled": enrolled,
        "has_battery": has,
        "strategy": strategy,
        "cap_kwh": cap,
        "p_kw": p_kw,
        "reserve_frac": reserve,
        "k_top": k_top,
        "eta": np.where(has, cfg.eta, 0.0),
    })
    rest = sites.drop(columns=_FLEET_DROP).reset_index(drop=True)
    out = pd.concat([homes, rest], axis=1)
    hidden = [c for c in out.columns if c not in PUBLIC_COLUMNS]
    return out[PUBLIC_COLUMNS + hidden]


def strategy_codes(homes: pd.DataFrame) -> np.ndarray:
    return homes.strategy.map(STRATEGY_CODES).to_numpy(np.int8)


def initial_soc_kwh(homes: pd.DataFrame) -> np.ndarray:
    """Start halfway between the reserve and full."""
    cap, res = homes.cap_kwh.to_numpy(float), homes.reserve_frac.to_numpy(float)
    return cap * (res + 0.5 * (1 - res))
