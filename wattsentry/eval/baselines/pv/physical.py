"""pvlib physical baseline: expected AC power from the RECORDED metadata only (plan §4 "fair pvlib baseline").

Never reads `*_true` / `true_*` columns: the gap between recorded and true tilt / azimuth / kWp, shading, snow,
degradation and the inverter curve is exactly what the learned models are allowed to beat.

Timestamps are 15-min interval starts (UTC); sun position is taken at the interval middle, like the simulator.
"""

from __future__ import annotations

from typing import Mapping

import numpy as np
import pandas as pd
import pvlib

STEP = pd.Timedelta(minutes=15)
HALF_STEP = pd.Timedelta(minutes=7.5)

# PVWatts-style defaults (what an installer's tool would assume; the simulator's hidden values differ per site)
GAMMA_PDC = -0.004  # 1/K
SYSTEM_LOSSES = 0.14  # wiring, mismatch, soiling allowance, availability
INV_EFF = 0.96
ALBEDO = 0.2

RECORDED = ("kwp_recorded", "tilt_recorded", "azimuth_recorded")


def solar_geometry(ts: pd.DatetimeIndex, lat: float, lon: float) -> dict[str, np.ndarray]:
    """zenith, azimuth (deg) at interval middles, simplified-Solis clear-sky ghi/dni/dhi (W/m2), day of year."""
    mid = pd.DatetimeIndex(ts) + HALF_STEP
    sp = pvlib.solarposition.get_solarposition(mid, lat, lon)
    cs = pvlib.location.Location(lat, lon).get_clearsky(mid, model="simplified_solis", solar_position=sp)
    return {
        "zenith": sp["apparent_zenith"].to_numpy(np.float64),
        "azimuth": sp["azimuth"].to_numpy(np.float64),
        "cs_ghi": cs["ghi"].to_numpy(np.float64),
        "cs_dni": cs["dni"].to_numpy(np.float64),
        "cs_dhi": cs["dhi"].to_numpy(np.float64),
        "doy": mid.day_of_year.to_numpy(),
    }


def recorded_meta(asset: Mapping) -> dict[str, float]:
    """kwp / tilt / azimuth / inverter_kw / lat / lon as the GIS + installer records them."""
    kwp = float(asset["kwp_recorded"])
    inv = asset.get("inverter_kw")
    inv = float(inv) if inv is not None and np.isfinite(float(inv)) and float(inv) > 0 else kwp
    return {"kwp": kwp, "tilt": float(asset["tilt_recorded"]), "azimuth": float(asset["azimuth_recorded"]),
            "inverter_kw": inv, "lat": float(asset["lat"]), "lon": float(asset["lon"])}


def pv_power_kw(meta: Mapping[str, float], geom: Mapping[str, np.ndarray], ghi, dni, dhi, temp_c=None,
                wind_ms=None) -> dict[str, np.ndarray]:
    """PVWatts chain: transposition (Hay-Davies) -> SAPM cell temperature -> DC -> flat inverter + clipping.

    Returns {"ac_kw", "poa"}. NaN irradiance gives NaN power; night gives 0.
    """
    zen, saz = geom["zenith"], geom["azimuth"]
    ghi = np.asarray(ghi, np.float64)
    dni = np.asarray(dni, np.float64)
    dhi = np.asarray(dhi, np.float64)
    temp = np.full_like(ghi, 15.0) if temp_c is None else np.asarray(temp_c, np.float64)
    wind = np.full_like(ghi, 1.0) if wind_ms is None else np.asarray(wind_ms, np.float64)
    dni_extra = pvlib.irradiance.get_extra_radiation(np.asarray(geom["doy"]))
    with np.errstate(invalid="ignore", divide="ignore"):
        poa = pvlib.irradiance.get_total_irradiance(
            meta["tilt"], meta["azimuth"], zen, saz, dni, ghi, dhi, dni_extra=dni_extra, albedo=ALBEDO,
            model="haydavies")
        poa_g = np.asarray(poa["poa_global"], np.float64)
        aoi = pvlib.irradiance.aoi(meta["tilt"], meta["azimuth"], zen, saz)
        iam = np.clip(pvlib.iam.physical(np.asarray(aoi)), 0, 1)
        poa_eff = np.asarray(poa["poa_direct"]) * iam + np.asarray(poa["poa_diffuse"])
        tp = pvlib.temperature.TEMPERATURE_MODEL_PARAMETERS["sapm"]["close_mount_glass_glass"]
        tcell = pvlib.temperature.sapm_cell(poa_g, temp, wind, **tp)
        dc = pvlib.pvsystem.pvwatts_dc(poa_eff, tcell, meta["kwp"], GAMMA_PDC) * (1 - SYSTEM_LOSSES)
    ac = np.clip(dc * INV_EFF, 0, meta["inverter_kw"])
    night = zen >= 90
    ac = np.where(night & np.isfinite(ac), 0.0, ac)
    ac = np.where(night & ~np.isfinite(ghi), 0.0, ac)  # missing weather at night is still 0 W
    return {"ac_kw": ac, "poa": np.where(night, 0.0, poa_g)}


def clearsky_power_kw(meta: Mapping[str, float], geom: Mapping[str, np.ndarray]) -> np.ndarray:
    """Expected AC power under clear sky (recorded metadata), the denominator of the reference-window features."""
    return pv_power_kw(meta, geom, geom["cs_ghi"], geom["cs_dni"], geom["cs_dhi"])["ac_kw"]
