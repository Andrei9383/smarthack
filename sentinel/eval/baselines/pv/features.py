"""Feature builders shared by bulk training/evaluation (wide arrays) and LGBMPredictor.predict (one window).

Every builder works on arrays whose last axis is the 96 steps of one UTC day and whose second-to-last axis (for the
reference / history inputs) is the 7 previous days, so the same code serves [n, d, ...] and [1, 1, ...] shapes.

PV (M1 fallback): actual or forecast weather + solar geometry + pvlib(recorded metadata) + the site's own reference
window (7 previous days of measured power, no target-day power). Load (M2 fallback): temperature + calendar +
the household's own past load (yesterday, last week, 7-day profile).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

REF_DAYS = 7

PV_FEATURES = [
    "ghi", "dni", "dhi", "temp_c", "wind10_ms", "snow_cm", "zenith", "cos_zenith", "sun_az", "kt",
    "pvlib_n", "pvlib_cs_n", "tilt", "azimuth", "lat", "doy_sin", "doy_cos",
    "ref_ratio_am", "ref_ratio_pm", "ref_ratio", "ref_step_n", "ref_d1_n",
]

LOAD_FEATURES = [
    "temp_c", "ghi", "hour", "dow", "off_day", "holiday", "doy_sin", "doy_cos",
    "archetype", "heating", "has_ac", "has_ev", "has_electric_boiler", "works_from_home",
    "d1_n", "w1_n", "med7_n", "d1_mean_n", "temp_d1_mean", "scale",
]

ARCHETYPES = ["base", "ac", "heat_pump", "electric_heating"]
HEATING = ["gas", "heat_pump", "electric", "wood", "district"]


def _code(value, vocab: list[str]) -> float:
    v = str(value)
    return float(vocab.index(v)) if v in vocab else float(len(vocab))


def _second_largest(x: np.ndarray, axis: int) -> np.ndarray:
    """Robust 'clear-day' envelope over the reference days: 2nd largest value (NaN ignored), NaN if < 2 values."""
    s = np.sort(np.where(np.isnan(x), -np.inf, x), axis=axis)
    out = np.take(s, -2, axis=axis)
    return np.where(np.isfinite(out), out, np.nan)


def pv_features(*, weather: dict, geom: dict, pvlib_kw, pvlib_cs_kw, ref_kw, ref_cs_kw, kwp, tilt, azimuth,
                lat) -> dict[str, np.ndarray]:
    """weather/geom/pvlib arrays [..., 96]; ref_kw / ref_cs_kw [..., 7, 96] (the 7 days before);
    kwp/tilt/azimuth/lat broadcastable to [...]. Returns {name: [..., 96]} for PV_FEATURES. Power features are
    normalised by recorded kWp so one model serves every site size."""
    kwp = np.asarray(kwp, np.float64)[..., None]
    zen = geom["zenith"]
    shape = np.broadcast_shapes(zen.shape, pvlib_kw.shape)
    b = lambda a: np.broadcast_to(np.asarray(a, np.float64), shape)  # noqa: E731
    am = geom["azimuth"] < 180
    env = _second_largest(ref_kw, axis=-2)                  # [..., 96]
    cs_mean = np.nanmean(ref_cs_kw, axis=-2)                # [..., 96]
    day = cs_mean > 0.02 * kwp

    def ratio(mask):
        num = np.nansum(np.where(mask & day & np.isfinite(env), env, 0.0), axis=-1)
        den = np.nansum(np.where(mask & day & np.isfinite(env), cs_mean, 0.0), axis=-1)
        r = np.where(den > 0, num / np.maximum(den, 1e-9), np.nan)
        return np.broadcast_to(r[..., None], shape)

    with np.errstate(invalid="ignore", divide="ignore"):
        cs_ghi = geom["cs_ghi"]
        kt = np.where(cs_ghi > 20, weather["ghi"] / np.maximum(cs_ghi, 1e-9), np.nan)
        doy = geom["doy"] if "doy" in geom else None
        ang = 2 * np.pi * (np.asarray(doy, np.float64) if doy is not None else 0) / 365.25
        return {
            "ghi": b(weather["ghi"]), "dni": b(weather["dni"]), "dhi": b(weather["dhi"]),
            "temp_c": b(weather["temp_c"]), "wind10_ms": b(weather["wind10_ms"]), "snow_cm": b(weather["snow_cm"]),
            "zenith": b(zen), "cos_zenith": b(np.cos(np.radians(zen))), "sun_az": b(geom["azimuth"]),
            "kt": b(np.clip(kt, 0, 1.5)),
            "pvlib_n": b(pvlib_kw / kwp), "pvlib_cs_n": b(pvlib_cs_kw / kwp),
            "tilt": b(np.asarray(tilt, np.float64)[..., None]), "azimuth": b(np.asarray(azimuth, np.float64)[..., None]),
            "lat": b(np.asarray(lat, np.float64)[..., None]),
            "doy_sin": b(np.sin(ang)), "doy_cos": b(np.cos(ang)),
            "ref_ratio_am": ratio(am), "ref_ratio_pm": ratio(~am), "ref_ratio": ratio(np.ones_like(am)),
            "ref_step_n": b(np.where(day, env / np.maximum(cs_mean, 1e-9), np.nan)) * b(pvlib_cs_kw / kwp),
            "ref_d1_n": b(ref_kw[..., -1, :] / kwp),
        }


def calendar(ts: pd.DatetimeIndex) -> dict[str, np.ndarray]:
    """Local (Europe/Bucharest) calendar per step, with Romanian public holidays (incl. Orthodox Easter)."""
    from sentinel.sim.fleet.ro_calendar import local_features

    f = local_features(pd.DatetimeIndex(ts))
    loc = pd.DatetimeIndex(ts).tz_convert("Europe/Bucharest")
    ang = 2 * np.pi * f["doy"] / 365.25
    return {"hour": f["hour"].astype(np.float64), "dow": loc.day_of_week.to_numpy().astype(np.float64),
            "off_day": f["off_day"].astype(np.float64), "holiday": f["holiday"].astype(np.float64),
            "doy_sin": np.sin(ang), "doy_cos": np.cos(ang)}


def load_features(*, temp_c, ghi, cal: dict, hist_kw, hist_temp_c, site: dict) -> dict[str, np.ndarray]:
    """temp_c/ghi/cal [..., 96]; hist_kw / hist_temp_c [..., 7, 96] the 7 days before (most recent last).
    Target normalisation: `scale` = mean load over the 7 days (kW), returned as a feature too."""
    temp_c = np.asarray(temp_c, np.float64)
    shape = temp_c.shape
    b = lambda a: np.broadcast_to(np.asarray(a, np.float64), shape)  # noqa: E731
    with np.errstate(invalid="ignore"):
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            scale = np.nanmean(hist_kw, axis=(-2, -1)) + 0.05
            med7 = np.nanmedian(hist_kw, axis=-2)
            tmean = np.nanmean(hist_temp_c[..., -1, :], axis=-1)
            d1_mean = np.nanmean(hist_kw[..., -1, :], axis=-1)
    s = scale[..., None]
    out = {"temp_c": b(temp_c), "ghi": b(ghi)}
    for k in ("hour", "dow", "off_day", "holiday", "doy_sin", "doy_cos"):
        out[k] = b(cal[k])
    for k, vocab in (("archetype", ARCHETYPES), ("heating", HEATING)):
        out[k] = b(np.vectorize(lambda v: _code(v, vocab), otypes=[float])(np.asarray(site[k], object))[..., None])
    for k in ("has_ac", "has_ev", "has_electric_boiler", "works_from_home"):
        out[k] = b(np.asarray(site[k], np.float64)[..., None])
    out.update({"d1_n": b(hist_kw[..., -1, :] / s), "w1_n": b(hist_kw[..., -7, :] / s), "med7_n": b(med7 / s),
                "d1_mean_n": b((d1_mean / scale)[..., None]), "temp_d1_mean": b(tmean[..., None]),
                "scale": b(s)})
    return out


def stack(feats: dict[str, np.ndarray], names: list[str]) -> np.ndarray:
    """[..., 96] dict -> [rows, n_features] float32, rows in C order of the leading shape."""
    return np.stack([np.asarray(feats[n], np.float32).ravel() for n in names], axis=1)
