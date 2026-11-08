"""Deterministic, physics-based synthetic Romanian prosumer dataset (fallback when the shared data windows are absent).

What it simulates (all seeded, no internet):
- ~20 real RO localities (lat/lon); pvlib solar geometry + Ineichen clear sky; a stochastic clear-sky-index process
  (national + local daily regimes, intraday AR(1) clouds, summer afternoon convection); Erbs decomposition;
  temperature (season, latitude, synoptic anomalies, diurnal cycle), wind, snow depth.
- Day-ahead *forecast* weather = actual + lead-time-growing errors (regime misses, smoothed cloud timing).
- Sites: true vs recorded tilt / azimuth / kWp (~20% wrong by ±10-30°, kWp ±15%), inverter clipping, part-load
  inverter efficiency, 0.5-1 %/yr degradation, horizon shading, snow cover, site-level cloud mismatch vs the
  locality weather, 1% meter noise.
- Households: RO archetypes (base, electric heating, heat pump, AC, EV) shaped like a RO daily curve; weekends and RO
  public holidays (incl. Orthodox Easter, Rusalii); vacations; appliance spikes.
- Battery under the self-consumption rule; inverter voltage with export-driven rise.
- 7 fault classes + healthy with exact onset step: soiling, shading, string_outage, clipping, overvoltage_trip,
  battery_fade, sensor_fault (frozen or mis-scaled meter).

Output: a directory of .npy arrays (memmap-able) + index.json, cached at
~/.cache/wattsentry/model-synth-v{SYNTH_VERSION}/<digest> ($WATTSENTRY_MODEL_CACHE overrides).
Layout (read by data.ArraySource; the shared data windows use the same one):
  site arrays   [S, T] float16 (kW / V / 0..1): pv_ac, pv_healthy, load, batt, soc, inv_v, pvlib_rec, pvlib_rec_fc,
                pvclear_rec;  int8: inv_status, fault
  locality      [L, T, 7] float16 weather_act, weather_fc (ghi, dni, dhi, temp, cloud, wind, snow); [L, T, 3] geom
  time          [T, 8] float32 calendar (physics.CAL_CHANNELS)
Run: python -m wattsentry.model.synth [--sites 200] [--out DIR]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import lfilter

from wattsentry.contracts import PvFault

from . import physics as ph

SYNTH_VERSION = 2  # v2: panel snow slides off (tilt / temperature / sun) instead of lasting until ground melt
FAULTS = [f.value for f in PvFault]  # index 0 = healthy
FAULT_IDX = {f: i for i, f in enumerate(FAULTS)}

# Real Romanian localities (approximate centres, WGS84): peri-urban communes where residential PV concentrates.
LOCALITIES = [
    ("Voluntari", "Ilfov", 44.490, 26.180), ("Otopeni", "Ilfov", 44.550, 26.070),
    ("Chiajna", "Ilfov", 44.460, 25.975), ("Bragadiru", "Ilfov", 44.371, 25.975),
    ("Florești", "Cluj", 46.747, 23.490), ("Apahida", "Cluj", 46.808, 23.740),
    ("Dumbrăvița", "Timiș", 45.795, 21.243), ("Giroc", "Timiș", 45.694, 21.238),
    ("Sânmartin", "Bihor", 47.010, 21.958), ("Ghimbav", "Brașov", 45.663, 25.508),
    ("Sânpetru", "Brașov", 45.710, 25.633), ("Miroslava", "Iași", 47.150, 27.520),
    ("Valu lui Traian", "Constanța", 44.165, 28.470), ("Ovidiu", "Constanța", 44.270, 28.560),
    ("Blejoi", "Prahova", 44.980, 26.020), ("Șelimbăr", "Sibiu", 45.765, 24.200),
    ("Bradu", "Argeș", 44.820, 24.890), ("Ișalnița", "Dolj", 44.390, 23.730),
    ("Sântandrei", "Bihor", 47.068, 21.860), ("Bacău (Mărgineni)", "Bacău", 46.590, 26.870),
]
ARCHETYPES = ["base", "electric_heating", "heat_pump", "ac", "ev"]
ARCH_P = [0.34, 0.14, 0.16, 0.24, 0.12]


@dataclass(frozen=True)
class SynthConfig:
    n_sites: int = 200
    start: str = "2022-01-01"
    days: int = 1461  # 2022-01-01 .. 2025-12-31
    seed: int = 7
    unseen_frac: float = 0.30
    wrong_meta_frac: float = 0.20
    fault_gap_days: float = 75.0  # mean healthy gap between fault episodes (≈ 15% faulty site-days)

    def digest(self) -> str:
        blob = json.dumps({**asdict(self), "v": SYNTH_VERSION}, sort_keys=True).encode()
        return hashlib.sha256(blob).hexdigest()[:10]


def cache_root() -> Path:
    default = Path.home() / ".cache" / "wattsentry" / f"model-synth-v{SYNTH_VERSION}"
    return Path(os.environ.get("WATTSENTRY_MODEL_CACHE", default))


def _ar1(rng, shape, rho, axis=-1):
    """Stationary unit-variance AR(1) noise along `axis`."""
    e = rng.standard_normal(shape) * np.sqrt(1 - rho**2)
    return lfilter([1.0], [1.0, -rho], e, axis=axis)


def _smooth(x, n):
    return np.convolve(x, np.ones(n) / n, mode="same")


# --- weather -----------------------------------------------------------------------------------------------
def _weather(rng, times: pd.DatetimeIndex, geoms: np.ndarray):
    L, T = len(LOCALITIES), len(times)
    D = T // ph.DAY
    doy_d = pd.DatetimeIndex(times[:: ph.DAY]).day_of_year.to_numpy()
    utc_h = (times.hour + times.minute / 60.0).to_numpy()
    season = np.cos(2 * np.pi * (doy_d - 15) / 365.25)  # +1 mid-Jan, -1 mid-Jul
    cbar = 0.53 + 0.17 * season
    z_nat = _ar1(rng, (D,), 0.70)
    z_loc = _ar1(rng, (L, D), 0.50)
    logit = np.log(cbar / (1 - cbar))
    c_d = 1 / (1 + np.exp(-(logit[None] + 1.1 * z_nat[None] + 0.8 * z_loc)))  # [L, D] daily cloud fraction
    m_d = 1.02 - 0.80 * c_d**1.3  # daily mean clear-sky index
    sig_d = 0.34 * 4 * c_d * (1 - c_d) + 0.015
    summer = np.clip(-season, 0, None)
    conv_shape = np.exp(-((utc_h + 2.5 - 15.0) ** 2) / 4.0)  # afternoon (local ~15h) convection bump

    act = np.zeros((L, T, 7), np.float32)
    fc = np.zeros((L, T, 7), np.float32)
    snow_out = np.zeros((L, D), np.float64)
    # national temperature anomaly shared by all localities, + local part
    ta_nat = 3.2 * _ar1(rng, (D,), 0.85)
    ta_loc = 1.2 * _ar1(rng, (L, D), 0.6)
    w_nat = _ar1(rng, (D,), 0.6)
    lead = np.tile(np.linspace(0.8, 1.25, ph.DAY), D)  # forecast error grows over the target day (lead 12-36 h)
    for li, (_, _, lat, lon) in enumerate(LOCALITIES):
        zen, csghi = geoms[li, :, 0], geoms[li, :, 2]
        e = _ar1(rng, (T,), 0.85)
        md = np.repeat(m_d[li], ph.DAY)
        sd = np.repeat(sig_d[li], ph.DAY)
        cd = np.repeat(c_d[li], ph.DAY)
        conv = -0.12 * np.repeat(c_d[li] * summer, ph.DAY) * conv_shape
        csi = np.clip(md + sd * e + conv, 0.04, 1.12)
        ghi = csi * csghi
        doy = times.day_of_year.to_numpy()
        dni, dhi = ph.decompose(ghi, zen, doy)
        cloud = np.clip(100 * (cd + 0.12 * _smooth(e, 9)), 0, 100)
        # temperature
        clim_d = 11.5 - 12.5 * np.cos(2 * np.pi * (doy_d - 20) / 365.25) - 0.7 * (lat - 44.5)
        tday = clim_d + ta_nat + ta_loc[li]
        tday_i = np.interp(np.arange(T) / ph.DAY - 0.5, np.arange(D), tday)
        amp = 3.5 + 5.0 * (1 - cd)
        temp = tday_i + amp * np.cos(2 * np.pi * (utc_h - 12.5) / 24) + 0.4 * _ar1(rng, (T,), 0.9)
        wmean = np.exp(np.log(3.0) + 0.35 * w_nat + 0.2 * _ar1(rng, (D,), 0.5))
        diurnal = 1 + 0.3 * np.cos(2 * np.pi * (utc_h - 12) / 24)
        wind = np.repeat(wmean, ph.DAY) * diurnal * np.exp(0.15 * _ar1(rng, (T,), 0.9))
        # snow depth (cm), daily state
        snow = np.zeros(D)
        s = 0.0
        for d in range(D):
            if tday[d] < 0.5 and c_d[li, d] > 0.72 and rng.random() < 0.5:
                s += rng.uniform(2, 10)
            s = max(0.0, s - max(0.0, tday[d] + 1.0) * 1.3 - 0.2)
            snow[d] = s
        snow_out[li] = snow
        snow_t = np.repeat(snow, ph.DAY)
        act[li] = np.stack([ghi, dni, dhi, temp, cloud, wind, snow_t], 1)

        # day-ahead forecast: daily regime error (sometimes a miss) + smoothed intraday timing + lead growth
        miss = rng.random(D) < 0.15
        eps_d = rng.standard_normal(D) * np.where(miss, 0.22, 0.08)
        eps_t = np.repeat(eps_d, ph.DAY) * lead + 0.04 * _ar1(rng, (T,), 0.95) * lead
        csi_fc = np.clip(md + 0.45 * sd * _smooth(e, 9) + conv + eps_t, 0.04, 1.10)
        ghi_fc = csi_fc * csghi
        dni_fc, dhi_fc = ph.decompose(ghi_fc, zen, doy)
        cloud_fc = np.clip(100 * (cd - np.repeat(eps_d, ph.DAY) * 1.2), 0, 100)
        temp_fc = temp + (np.repeat(rng.standard_normal(D) * 1.3, ph.DAY) + 0.5 * _ar1(rng, (T,), 0.97)) * lead
        wind_fc = wind * np.exp(np.repeat(rng.standard_normal(D) * 0.25, ph.DAY))
        fc[li] = np.stack([ghi_fc, dni_fc, dhi_fc, temp_fc, cloud_fc, wind_fc, snow_t], 1)
    return act, fc, c_d, sig_d


# --- sites -------------------------------------------------------------------------------------------------
def _sites(cfg: SynthConfig, rng) -> list[dict]:
    S = cfg.n_sites
    w = np.array([3, 3, 2, 2, 3, 2, 3, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 1, 1, 1], float)
    loc = rng.choice(len(LOCALITIES), size=S, p=w / w.sum())
    perm = rng.permutation(S)
    unseen = np.zeros(S, bool)
    unseen[perm[: int(round(cfg.unseen_frac * S))]] = True
    wrong = rng.random(S) < cfg.wrong_meta_frac
    sites = []
    for i in range(S):
        name, county, lat, lon = LOCALITIES[loc[i]]
        kwp = round(float(rng.uniform(3, 10)), 1)
        dcac = rng.uniform(1.0, 1.3)
        inv = max(2.5, round(kwp / dcac * 2) / 2)
        tilt = float(np.clip(rng.normal(30, 7), 8, 50))
        az = float(np.clip(rng.normal(180, 28), 95, 265))
        if wrong[i]:
            mode = rng.integers(3)  # tilt only, azimuth only, both
            tilt_r = tilt + (rng.choice([-1, 1]) * rng.uniform(10, 30) if mode in (0, 2) else 0)
            az_r = az + (rng.choice([-1, 1]) * rng.uniform(10, 30) if mode in (1, 2) else 0)
            kwp_r = kwp * rng.uniform(0.85, 1.15)
        else:
            tilt_r, az_r, kwp_r = tilt + rng.normal(0, 1.5), az + rng.normal(0, 2), kwp * rng.uniform(0.98, 1.02)
        has_b = rng.random() < 0.45
        bkwh = float(rng.choice([5.0, 7.5, 10.0, 13.5])) if has_b else 0.0
        bkw = float(rng.choice([3.0, 3.68, 5.0])) if has_b else 0.0
        shaded = rng.random() < 0.15
        arch = ARCHETYPES[rng.choice(len(ARCHETYPES), p=ARCH_P)]
        sites.append(dict(
            site_id=f"S{i + 1:04d}", locality=name, county=county, loc_idx=int(loc[i]),
            lat=lat + float(rng.normal(0, 0.01)), lon=lon + float(rng.normal(0, 0.01)),
            kwp_true=kwp, kwp_recorded=round(float(kwp_r), 2),
            tilt_true=round(tilt, 1), tilt_recorded=round(float(np.clip(tilt_r, 0, 60)), 1),
            azimuth_true=round(az, 1), azimuth_recorded=round(float(az_r), 1),
            inverter_kw=float(inv), meta_wrong=bool(wrong[i]), archetype=arch,
            battery_kwh=bkwh, battery_kw=bkw, eta_charge=0.95 if has_b else 0.0, eta_discharge=0.95 if has_b else 0.0,
            soc_min=0.1 if has_b else 0.0, soc_max=1.0 if has_b else 0.0,
            comfort_reserve_kwh=float(rng.uniform(1.0, 2.0)) if has_b else 0.0,
            split="unseen" if unseen[i] else "train",
            install_year=int(rng.integers(2017, 2022)), degradation=float(rng.uniform(0.005, 0.01)),
            losses=float(rng.uniform(0.10, 0.19)), gamma=float(rng.uniform(-0.0045, -0.0035)),
            shaded=bool(shaded), shade_az=float(rng.uniform(100, 240)), shade_w=float(rng.uniform(20, 60)),
            shade_elev=float(rng.uniform(10, 25)), shade_block=float(rng.uniform(0.5, 0.95)),
            cloud_mismatch=float(rng.uniform(0.3, 1.0)), v_offset=float(rng.uniform(0, 7)),
            v_per_kw=float(rng.uniform(0.6, 1.5)), load_scale=float(np.exp(rng.normal(np.log(0.33), 0.3))),
            ua_heat=float(rng.uniform(0.08, 0.18)), ua_cool=float(rng.uniform(0.12, 0.28)),
            n_strings=int(rng.choice([2, 3, 4])),
        ))
    return sites


def _episodes(cfg: SynthConfig, rng, sites, T: int) -> list[dict]:
    """Non-overlapping fault episodes per site with exact onset step."""
    dur_rng = {"soiling": (7, 40), "shading": (5, 30), "string_outage": (2, 20), "clipping": (2, 15),
               "overvoltage_trip": (3, 20), "battery_fade": (10, 45), "sensor_fault": (1, 10)}
    cls_p = {"soiling": 0.17, "shading": 0.14, "string_outage": 0.16, "clipping": 0.12,
             "overvoltage_trip": 0.14, "battery_fade": 0.13, "sensor_fault": 0.14}
    eps = []
    for si, s in enumerate(sites):
        t = int(rng.exponential(cfg.fault_gap_days) * ph.DAY) + 14 * ph.DAY
        while t < T:
            names = [c for c in cls_p if c != "battery_fade" or s["battery_kwh"] > 0]
            p = np.array([cls_p[c] for c in names])
            c = names[rng.choice(len(names), p=p / p.sum())]
            lo, hi = dur_rng[c]
            end = min(T, t + int(rng.uniform(lo, hi) * ph.DAY))
            sev = {"soiling": rng.uniform(0.08, 0.30), "shading": rng.uniform(0.4, 0.85),
                   "string_outage": 1.0 / s["n_strings"], "clipping": rng.uniform(0.45, 0.75),
                   "overvoltage_trip": rng.uniform(9, 16), "battery_fade": rng.uniform(0.3, 0.6),
                   "sensor_fault": rng.uniform(0, 1)}[c]
            params = {}
            if c == "shading":
                params = {"az0": float(rng.uniform(100, 230)), "w": float(rng.uniform(30, 70)),
                          "elev": float(rng.uniform(25, 50))}
            if c == "sensor_fault":
                params = {"kind": "frozen" if sev < 0.6 else "scale", "scale": float(rng.choice([0.5, 0.6, 1.4]))}
            eps.append(dict(site=si, asset_id=s["site_id"], fault=c, onset=int(t), end=int(end),
                            severity=float(sev), params=params))
            t = end + int((3 + rng.exponential(cfg.fault_gap_days)) * ph.DAY)
    return eps


def _panel_snow(tilt, snow_depth, tmean, cloud):
    """Daily factor 1 - panel snow cover: new snowfall covers the panel, then it slides off within ~1-4 days
    (faster when steep, above freezing and sunny), independent of the snow left on the ground."""
    D = len(snow_depth)
    out = np.ones(D)
    c = 0.0
    for d in range(D):
        fall = snow_depth[d] - (snow_depth[d - 1] if d else 0.0)
        if fall > 1.0:
            c = min(1.0, c + fall / 5.0)
        else:
            slide = (0.2 + 0.35 * (tmean[d] > 0) + 0.3 * (1 - cloud[d])) * min(1.5, tilt / 30.0)
            c *= max(0.0, 1 - slide)
            if c < 0.03:
                c = 0.0
        out[d] = 1 - c
    return out


def _load(site, rng, cal_hour, off_day, month, temp_l, T: int) -> np.ndarray:
    D = T // ph.DAY
    h = cal_hour
    gauss = lambda c, w: np.exp(-((h - c) ** 2) / (2 * w**2))
    winter_light = 1 + 0.3 * np.isin(month, [11, 12, 1, 2])
    wk = 0.55 + 0.25 * gauss(13, 3) + 0.6 * gauss(7.0, 1.0) + 1.1 * winter_light * gauss(20.0, 1.8)
    we = 0.55 + 0.45 * gauss(13, 3.5) + 0.5 * gauss(9.5, 1.6) + 1.0 * winter_light * gauss(19.8, 2.0)
    shape = np.where(off_day, we, wk)
    daily = np.exp(0.10 * _ar1(rng, (D,), 0.6))
    vac = np.ones(D)
    for y in range(D // 365 + 1):
        if rng.random() < 0.6:
            st = y * 365 + int(rng.uniform(180, 235))
            vac[st: st + int(rng.uniform(5, 14))] = 0.25
    load = site["load_scale"] * shape * np.repeat(daily * vac, ph.DAY)
    t_smooth = lfilter([0.08], [1, -0.92], temp_l)  # ~3 h thermal lag
    occ = 0.6 + 0.4 * ((h > 6) & (h < 23))
    a = site["archetype"]
    if a == "electric_heating":
        load += site["ua_heat"] * np.clip(17 - t_smooth, 0, None) * occ
    elif a == "heat_pump":
        cop = np.clip(2.2 + 0.08 * (t_smooth + 10), 1.8, 4.5)
        load += site["ua_heat"] * 2.6 * np.clip(17 - t_smooth, 0, None) * occ / cop
    if a == "ac" or (a == "heat_pump" and site["n_strings"] % 2 == 0):
        occ_c = 0.3 + 0.7 * gauss(17, 3.5)
        load += site["ua_cool"] * np.clip(temp_l - 24, 0, None) * occ_c * np.repeat(vac > 0.5, ph.DAY)
    if a == "ev":
        ev = np.zeros(T)
        for d in np.nonzero(rng.random(D) < 0.45)[0]:
            # local evening 18-23 h: UTC ≈ local - 2.5
            st = d * ph.DAY + int((rng.uniform(18, 23) - 2.5) * 4)
            ev[st: st + int(rng.uniform(6, 16))] = 3.7
        load += ev
    # appliance spikes (kettle, oven, washing machine): unpredictable, drive the upper quantile
    n_ev = rng.poisson(3.0 * D)
    pos = rng.integers(0, T, n_ev)
    keep = rng.random(n_ev) < np.clip(shape[pos] / 1.2, 0.1, 1)
    spikes = np.zeros(T)
    for p_, k in zip(pos[keep], rng.integers(1, 4, keep.sum())):
        spikes[p_: p_ + k] += rng.uniform(0.6, 2.0)
    load = (load + spikes) * (1 + 0.08 * rng.standard_normal(T))
    return np.clip(load, 0.05, None)


def generate(cfg: SynthConfig = SynthConfig(), out: Path | None = None, verbose: bool = True) -> Path:
    t_all = time.time()
    out = Path(out) if out else cache_root() / cfg.digest()
    if (out / "index.json").exists():
        return out
    tmp = out.with_name(out.name + f".tmp{os.getpid()}")
    tmp.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(cfg.seed)
    T = cfg.days * ph.DAY
    times = ph.times_utc(cfg.start + "T00:00:00Z", T)
    L, S = len(LOCALITIES), cfg.n_sites

    geoms = np.stack([ph.solar_geometry(times, lat, lon) for (_, _, lat, lon) in LOCALITIES])  # [L, T, 3]
    act, fc, c_d, sig_d = _weather(rng, times, geoms)
    cal = ph.calendar_features(times).astype(np.float32)
    loc_t = times.tz_convert(ph.TZ)
    cal_hour = (loc_t.hour + loc_t.minute / 60).to_numpy(np.float64)
    off_day = cal[:, 4] > 0.5
    month = loc_t.month.to_numpy()
    year_frac = (times.year + (times.day_of_year - 1) / 365.25).to_numpy()
    doy = times.day_of_year.to_numpy()
    if verbose:
        print(f"[synth] weather {time.time() - t_all:.1f}s", flush=True)

    snow_d = act[:, :: ph.DAY, 6].astype(np.float64)  # daily snow depth (cm) per locality
    tmean_d = act[:, :, 3].reshape(L, -1, ph.DAY).mean(-1)
    sites = _sites(cfg, rng)
    eps = _episodes(cfg, rng, sites, T)
    fault = np.zeros((S, T), np.int8)
    for e in eps:
        fault[e["site"], e["onset"]: e["end"]] = FAULT_IDX[e["fault"]]

    def mm(name, dtype=np.float16, shape=(S, T)):
        return np.lib.format.open_memmap(tmp / f"{name}.npy", mode="w+", dtype=dtype, shape=shape)

    arr = {k: mm(k) for k in ("pv_ac", "pv_healthy", "load", "batt", "soc", "inv_v", "pvlib_rec", "pvlib_rec_fc",
                              "pvclear_rec")}
    arr["inv_status"] = mm("inv_status", np.int8)
    arr["fault"] = mm("fault", np.int8)
    arr["fault"][:] = fault

    pv_phys = np.zeros((S, T), np.float32)
    load_all = np.zeros((S, T), np.float32)
    cap_lim = np.zeros((S, T), np.float32)
    ov = np.zeros((S, T), np.float32)
    fade = np.ones((S, T), np.float32)
    for si, s in enumerate(sites):
        li = s["loc_idx"]
        zen, saz = geoms[li, :, 0], geoms[li, :, 1]
        w = act[li].astype(np.float64)
        ghi, temp, wind = w[:, 0], w[:, 3], w[:, 5]
        # site-level cloud mismatch vs the locality weather (strongest on broken-cloud days)
        sd = np.repeat(sig_d[li], ph.DAY)
        csg = geoms[li, :, 2]
        noise = s["cloud_mismatch"] * 0.7 * sd * _ar1(rng, (T,), 0.8)
        ghi_s = np.clip(ghi * (1 + noise), 0, 1.15 * csg + 1)
        dni_s, dhi_s = ph.decompose(ghi_s, zen, doy)
        g, b = ph.poa(s["tilt_true"], s["azimuth_true"], zen, saz, dni_s, ghi_s, dhi_s)
        elev = 90 - zen
        if s["shaded"]:
            daz = (saz - s["shade_az"]) % 360
            blk = (daz < s["shade_w"]) & (elev < s["shade_elev"])
            g = g - b * s["shade_block"] * blk
        snow_cover = np.repeat(_panel_snow(s["tilt_true"], snow_d[li], tmean_d[li], c_d[li]), ph.DAY)
        g = g * snow_cover
        deg = 1 - s["degradation"] * np.clip(year_frac - s["install_year"], 0, None)
        kwp_eff = s["kwp_true"] * deg
        healthy = ph.ac_power_kw(g, temp, kwp_eff, s["inverter_kw"], s["losses"], s["gamma"], wind)
        # physical faults
        g_f, kwp_f, inv_f = g.copy(), np.full(T, kwp_eff, dtype=np.float64), np.full(T, s["inverter_kw"], np.float64)
        for e in (e for e in eps if e["site"] == si):
            a, z = e["onset"], e["end"]
            if e["fault"] == "soiling":
                ramp = np.linspace(0, e["severity"], z - a)
                g_f[a:z] *= 1 - ramp
            elif e["fault"] == "shading":
                p = e["params"]
                daz = (saz[a:z] - p["az0"]) % 360
                blk = (daz < p["w"]) & (elev[a:z] < p["elev"]) & (elev[a:z] > 0)
                g_f[a:z] = g_f[a:z] - b[a:z] * e["severity"] * blk * snow_cover[a:z]
            elif e["fault"] == "string_outage":
                kwp_f[a:z] *= 1 - e["severity"]
            elif e["fault"] == "clipping":
                inv_f[a:z] *= e["severity"]
            elif e["fault"] == "overvoltage_trip":
                ov[si, a:z] = e["severity"]
            elif e["fault"] == "battery_fade":
                fade[si, a:z] = 1 - e["severity"]
        pv = ph.ac_power_kw(np.clip(g_f, 0, None), temp, kwp_f, s["inverter_kw"], s["losses"], s["gamma"], wind)
        pv = np.minimum(pv, inv_f)
        cap_lim[si] = inv_f
        pv_phys[si] = pv
        arr["pv_healthy"][si] = healthy
        meta = {k: s[k] for k in ("kwp_recorded", "tilt_recorded", "azimuth_recorded", "inverter_kw")}
        arr["pvlib_rec"][si] = ph.pvlib_expected_kw(meta, w, geoms[li])
        arr["pvlib_rec_fc"][si] = ph.pvlib_expected_kw(meta, fc[li].astype(np.float64), geoms[li])
        arr["pvclear_rec"][si] = ph.pvlib_clear_kw(meta, times, geoms[li])
        load_all[si] = _load(s, rng, cal_hour, off_day, month, temp, T)
    if verbose:
        print(f"[synth] sites pv+load {time.time() - t_all:.1f}s", flush=True)

    # voltage + overvoltage trips (on the true export), then the battery under the self-consumption rule
    v_off = np.array([s["v_offset"] for s in sites], np.float32)[:, None]
    kv = np.array([s["v_per_kw"] for s in sites], np.float32)[:, None]
    mv = np.stack([np.repeat(1.5 * _ar1(rng, (cfg.days,), 0.7), ph.DAY) for _ in range(L)]).astype(np.float32)
    mv_s = mv[[s["loc_idx"] for s in sites]]
    v_noise = (0.5 * rng.standard_normal((S, T))).astype(np.float32)
    v0 = 230 + v_off + mv_s + kv * np.clip(pv_phys - load_all, 0, None) + ov + v_noise
    trip = (v0 > 253.0) & (pv_phys > 0.05)
    trip[:, 1:] |= trip[:, :-1]
    trip[:, 2:] |= trip[:, :-2]
    trip &= pv_phys > 0.0
    pv_phys = np.where(trip, 0.0, pv_phys).astype(np.float32)

    E = np.array([s["battery_kwh"] for s in sites], np.float64)
    P = np.array([s["battery_kw"] for s in sites], np.float64)
    reserve = np.array([max(s["soc_min"] * s["battery_kwh"], s["comfort_reserve_kwh"]) for s in sites])
    eta = 0.95
    dt_h = 0.25
    e = 0.5 * E
    batt = np.zeros((S, T), np.float32)
    soc = np.zeros((S, T), np.float32)
    has = E > 0
    surplus_all = (pv_phys - load_all).astype(np.float64)
    for t in range(T):
        emax = E * fade[:, t]
        emin = np.minimum(reserve, emax * 0.5)
        sp = surplus_all[:, t]
        ch = np.minimum(np.minimum(np.clip(sp, 0, None), P), np.clip(emax - e, 0, None) / (eta * dt_h))
        dis = np.minimum(np.minimum(np.clip(-sp, 0, None), P), np.clip(e - emin, 0, None) * eta / dt_h)
        e = np.minimum(e + eta * ch * dt_h - dis * dt_h / eta, emax)
        batt[:, t] = ch - dis
        soc[:, t] = np.where(has, e / np.maximum(emax, 1e-6), 0.0)
    if verbose:
        print(f"[synth] battery {time.time() - t_all:.1f}s", flush=True)

    grid = pv_phys - load_all - batt
    v = 230 + v_off + mv_s + kv * np.clip(grid, 0, None) + ov + v_noise
    v = np.where(trip, v0, v)
    status = np.where(pv_phys > 0.005, 1, 0).astype(np.int8)
    status[(pv_phys >= 0.995 * cap_lim) & (pv_phys > 0.1)] = 2
    status[trip] = 3

    # recorded PV: meter noise + sensor faults (the physical power, battery and grid are unaffected)
    pv_rec = pv_phys * (1 + 0.01 * rng.standard_normal((S, T))).astype(np.float32)
    for e_ in eps:
        if e_["fault"] != "sensor_fault":
            continue
        si, a, z = e_["site"], e_["onset"], e_["end"]
        if e_["params"]["kind"] == "frozen":
            seg = pv_rec[si, max(0, a - ph.DAY): a]
            val = float(seg[seg > 0.2].max(initial=0.0) * rng.uniform(0.3, 0.9)) if (seg > 0.2).any() else 0.0
            pv_rec[si, a:z] = val
        else:
            pv_rec[si, a:z] *= e_["params"]["scale"]
    arr["pv_ac"][:] = np.clip(pv_rec, 0, None)
    arr["load"][:] = load_all
    arr["batt"][:] = batt
    arr["soc"][:] = soc
    arr["inv_v"][:] = v
    arr["inv_status"][:] = status

    for name, a in (("weather_act", act), ("weather_fc", fc), ("geom", geoms.astype(np.float32))):
        np.save(tmp / f"{name}.npy", a.astype(np.float16) if name != "geom" else a.astype(np.float32))
    np.save(tmp / "calendar.npy", cal)
    for a in arr.values():
        a.flush()
    del arr

    arrays = {}
    for f in sorted(tmp.glob("*.npy")):
        a = np.load(f, mmap_mode="r")
        arrays[f.stem] = {"file": f.name, "dtype": str(a.dtype), "shape": list(a.shape)}
    index = {
        "format": "wattsentry-windows-v1",
        "source": "synthetic RO (wattsentry.model.synth)",
        "label": "synthetic RO",
        "config": asdict(cfg), "digest": cfg.digest(), "synth_version": SYNTH_VERSION,
        "t_start": str(times[0]), "step_minutes": ph.STEP_MIN, "n_steps": T, "n_sites": S,
        "weather_channels": list(ph.WEATHER_CHANNELS), "geom_channels": list(ph.GEOM_CHANNELS),
        "calendar_channels": list(ph.CAL_CHANNELS), "faults": FAULTS,
        "localities": [dict(name=n, county=c, lat=la, lon=lo) for n, c, la, lo in LOCALITIES],
        "arrays": arrays, "units": {"power": "kW", "inv_v": "V", "soc": "0..1"},
        "gen_seconds": round(time.time() - t_all, 1),
    }
    (tmp / "sites.json").write_text(json.dumps(sites, ensure_ascii=False, indent=0))
    (tmp / "episodes.json").write_text(json.dumps(eps))
    (tmp / "index.json").write_text(json.dumps(index, ensure_ascii=False, indent=1))
    try:
        tmp.rename(out)
    except OSError:  # another process finished first
        import shutil

        shutil.rmtree(tmp, ignore_errors=True)
    if verbose:
        print(f"[synth] done in {time.time() - t_all:.1f}s -> {out}", flush=True)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description="Generate the synthetic RO prosumer dataset")
    ap.add_argument("--sites", type=int, default=SynthConfig.n_sites)
    ap.add_argument("--days", type=int, default=SynthConfig.days)
    ap.add_argument("--seed", type=int, default=SynthConfig.seed)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    generate(SynthConfig(n_sites=a.sites, days=a.days, seed=a.seed), Path(a.out) if a.out else None)


if __name__ == "__main__":
    main()
