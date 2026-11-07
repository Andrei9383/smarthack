"""Static world: feeders in real localities and prosumer sites on them.

Columns prefixed `true_` are hidden ground truth (for evaluation and the fault/grid-twin labels). Models, pvlib and
the agent may only use the recorded columns (`tilt_deg`, `azimuth_deg`, `kwp`, `feeder_id`, `phase`).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import FleetConfig
from .localities import LOCALITIES

INVERTER_SIZES_KW = np.array([3.0, 3.6, 4.0, 5.0, 6.0, 8.0, 10.0])


def build_world(cfg: FleetConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng([cfg.seed, 1])
    feeders = _feeders(cfg, rng)
    sites = _sites(cfg, feeders, rng)
    _set_feeder_limits(cfg, feeders, sites, rng)
    return feeders, sites


def _feeders(cfg: FleetConfig, rng) -> pd.DataFrame:
    if cfg.n_feeders > len(LOCALITIES):
        raise ValueError(f"at most {len(LOCALITIES)} feeders (one locality each)")
    w = np.array([cfg.county_weights.get(l.county, 0.0) for l in LOCALITIES])
    per_county = {c: sum(1 for l in LOCALITIES if l.county == c) for c in cfg.county_weights}
    w = w / np.array([per_county.get(l.county, 1) for l in LOCALITIES])
    idx = rng.choice(len(LOCALITIES), cfg.n_feeders, replace=False, p=w / w.sum())
    locs = [LOCALITIES[i] for i in idx]
    congested = np.zeros(cfg.n_feeders, bool)
    congested[rng.choice(cfg.n_feeders, cfg.n_congested, replace=False)] = True
    return pd.DataFrame({
        "feeder_id": [f"F{i + 1:02d}" for i in range(cfg.n_feeders)],
        "locality_id": [l.id for l in locs],
        "locality": [l.name for l in locs],
        "county": [l.county for l in locs],
        "lat": [l.lat for l in locs],
        "lon": [l.lon for l in locs],
        "transformer_kva": rng.choice([250, 400, 630], cfg.n_feeders),
        # Households on the feeder without PV (their load soaks up some of the export).
        "n_passive_households": np.where(congested, rng.integers(15, 40, cfg.n_feeders),
                                         rng.integers(30, 90, cfg.n_feeders)),
        "v_base_pu": rng.uniform(1.00, 1.03, cfg.n_feeders).round(4),
        # Voltage rise per kW of net export on a phase, at the far end of the feeder.
        "v_sens_v_per_kw": np.where(congested, rng.uniform(0.16, 0.24, cfg.n_feeders),
                                    rng.uniform(0.08, 0.14, cfg.n_feeders)).round(4),
        "true_designed_congested": congested,
    })


def _sites(cfg: FleetConfig, feeders: pd.DataFrame, rng) -> pd.DataFrame:
    n, nf = cfg.n_sites, len(feeders)
    share = rng.dirichlet(np.full(nf, 4.0))
    counts = np.maximum(np.round(share * n).astype(int), 12)
    while counts.sum() != n:  # fix rounding while keeping every feeder >= 12 sites
        j = rng.integers(nf)
        counts[j] += 1 if counts.sum() < n else (-1 if counts[j] > 12 else 0)
    f_idx = np.repeat(np.arange(nf), counts)

    kwp = np.round(3 + 7 * rng.beta(2, 3, n), 2)
    tilt = np.clip(rng.normal(30, 7, n), 5, 45).round(1)
    az = np.clip(rng.normal(180, 35, n), 90, 270).round(1)
    ac_kw = INVERTER_SIZES_KW[np.abs(INVERTER_SIZES_KW[None] - (kwp * rng.uniform(0.75, 1.0, n))[:, None]).argmin(1)]

    # Recorded metadata: nominal values rounded to 5 degrees, wrong on a share of sites.
    wrong = rng.random(n) < cfg.wrong_orientation_share
    sign = rng.choice([-1, 1], (2, n))
    tilt_rec = np.where(wrong, np.clip(tilt + sign[0] * rng.uniform(10, 30, n), 0, 60), tilt)
    az_rec = np.where(wrong, az + sign[1] * rng.uniform(10, 30, n), az)
    tilt_rec, az_rec = 5 * np.round(tilt_rec / 5), 5 * np.round(az_rec / 5)
    wrong_kwp = rng.random(n) < cfg.wrong_kwp_share
    kwp_rec = np.where(wrong_kwp, kwp * rng.choice([0.88, 0.92, 1.08, 1.12], n), kwp).round(2)

    shaded = rng.random(n) < cfg.shaded_share
    batt = rng.random(n) < cfg.battery_share
    e_kwh = np.where(batt, np.round(rng.uniform(5, 15, n) * 2) / 2, 0.0)
    heating = rng.choice(["gas", "electric", "heat_pump"], n, p=[0.55, 0.2, 0.25])

    phase = rng.choice(np.array(list("ABC")), n)
    phase_gis = np.where(rng.random(n) < cfg.gis_wrong_phase_share,
                         np.array(list("ABC"))[(np.searchsorted(list("ABC"), phase) + rng.integers(1, 3, n)) % 3], phase)
    feeder_ids = feeders.feeder_id.to_numpy()
    f_gis = f_idx.copy()
    wrong_f = rng.random(n) < cfg.gis_wrong_feeder_share
    f_gis[wrong_f] = (f_idx[wrong_f] + rng.integers(1, nf, wrong_f.sum())) % nf

    flat, flon = feeders.lat.to_numpy()[f_idx], feeders.lon.to_numpy()[f_idx]
    return pd.DataFrame({
        "site_id": [f"S{i + 1:04d}" for i in range(n)],
        "locality_id": feeders.locality_id.to_numpy()[f_idx],
        "county": feeders.county.to_numpy()[f_idx],
        "lat": (flat + rng.normal(0, 0.006, n)).round(5),
        "lon": (flon + rng.normal(0, 0.008, n)).round(5),
        # What the GIS / installer records (visible to models)
        "feeder_id": feeder_ids[f_gis],
        "phase": phase_gis,
        "kwp": kwp_rec,
        "tilt_deg": tilt_rec,
        "azimuth_deg": az_rec,
        "inverter_ac_kw": ac_kw,
        "install_year": rng.integers(2017, 2022, n),
        "battery_kwh": e_kwh,
        "battery_kw": np.where(batt, rng.choice([3.0, 3.6, 5.0], n), 0.0),
        "battery_eta_charge": np.where(batt, 0.95, 0.0),
        "battery_eta_discharge": np.where(batt, 0.95, 0.0),
        "battery_soc_min": np.where(batt, 0.10, 0.0),
        "comfort_reserve_kwh": np.round(e_kwh * rng.uniform(0.10, 0.30, n), 2),
        "heating": heating,
        "has_ac": rng.random(n) < 0.55,
        "has_ev": rng.random(n) < 0.12,
        "has_electric_boiler": rng.random(n) < 0.25,
        "works_from_home": rng.random(n) < 0.30,
        # Hidden truth
        "true_feeder_id": feeder_ids[f_idx],
        "true_phase": phase,
        "true_kwp": kwp,
        "true_tilt_deg": tilt,
        "true_azimuth_deg": az,
        "true_metadata_wrong": wrong | wrong_kwp,
        "true_feeder_position": rng.uniform(0.15, 1.0, n).round(3),   # 0 = transformer, 1 = far end
        "true_degradation_per_year": rng.uniform(0.005, 0.01, n).round(4),
        "true_temp_coeff": rng.uniform(-0.0045, -0.0035, n).round(5),
        "true_system_losses": rng.uniform(0.88, 0.95, n).round(3),
        "true_shaded": shaded,
        "true_shade_az_deg": np.where(shaded, rng.uniform(100, 260, n), 0).round(1),
        "true_shade_width_deg": np.where(shaded, rng.uniform(30, 90, n), 0).round(1),
        "true_shade_elev_deg": np.where(shaded, rng.uniform(10, 28, n), 0).round(1),
        "true_load_scale": np.exp(rng.normal(0, 0.25, n)).round(3),
        "true_ua_kw_per_k": rng.uniform(0.06, 0.15, n).round(3),
    })


def _set_feeder_limits(cfg: FleetConfig, feeders: pd.DataFrame, sites: pd.DataFrame, rng) -> None:
    ac = sites.groupby("true_feeder_id").inverter_ac_kw.sum().reindex(feeders.feeder_id).to_numpy()
    cong = feeders.true_designed_congested.to_numpy()
    nf = len(feeders)
    # Congested feeders: limit well below the PV that can export at noon; the others have headroom.
    g = np.where(cong, ac * rng.uniform(0.50, 0.65, nf), ac * rng.uniform(1.0, 1.4, nf))
    feeders["prosumer_ac_kw"] = ac.round(1)
    feeders["g_max_kw"] = g.round(1)
    feeders["g_conservative_kw"] = (g * rng.uniform(0.75, 0.85, nf)).round(1)
    feeders["tau"] = 1.0  # trust in the GIS; the grid-twin (#13) estimates it
    feeders["n_prosumers"] = sites.groupby("true_feeder_id").size().reindex(feeders.feeder_id).to_numpy()
