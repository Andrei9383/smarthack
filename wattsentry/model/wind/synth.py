"""Quick stand-in wind generator (no ERA5 needed), PPC schema, numpy only. M6's fallback, kept for tests.

The real simulator is `python -m wattsentry.sim.wind` (ERA5-driven parks on real OSM positions + the fault injector
in wattsentry.sim.wind_faults); `data.load_wind()` falls back to this module only when its cache is absent (or with
`train --synth`). Needs no data files and makes small worlds in seconds. Output is a `WindData` (wide view) that
converts losslessly to the contract tables (`data.to_long`, `meteo_table`, `turbines_table`, `fault_labels_table`).

Physics: stochastic 100 m wind per park (AR(1) latent → Weibull, seasonal + diurnal scale,
correlated across parks), regime-switching direction, shear to hub height (α = 0.14), air density from T and p,
direction-dependent Jensen wakes, turbulence noise, generic 2.5 / 6.1 MW class curves, first-order thermal model
for the five component temperatures (ambient + load-dependent rise + lag; some turbines run hot by design).
Shared building blocks (AR(1), thermal lag, wake table, THERMAL, ERROR_CODES) live in wattsentry.sim.wind.physics.

Faults with exact onset labels (WindAnomaly): fault_stop, curtailment (park-wide cap; labelled where the cap
binds), icing (T < 1 °C and RH > 90 % spells), underperformance (sector-dependent yaw misalignment or power-curve
degradation), overheating (gearbox / generator-bearing residual drifting 3–21 days, often ending in a stop),
anemometer_fault (bias / stuck / erratic nacelle wind while power follows the real wind), data_gap (comm gaps,
some rows carrying integral error codes).

Telemetry sensors are exactly the contract's: act_pwt, wd_spd, conv_t, gearbox_t, gen1_t, gen2_t, transformer_t,
turbine_t. No direction / pitch / status tags (PPC has none); direction lives in the per-turbine meteo table.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field

import numpy as np
from scipy.special import ndtr

from wattsentry.sim.wind.physics import ERROR_CODES, THERMAL, wake_table
from wattsentry.sim.wind.physics import ar1 as _ar1
from wattsentry.sim.wind.physics import lag as _lag

from .data import CLASS_INDEX, METEO_FIELDS, PRIORITY, WEATHER_FIELDS, WindData, _iso
from .physics import CLASS_2500, CLASS_6100, TurbineClass, air_density, density_normalise, shear_to_hub

STEPS_PER_H = 4


@dataclass(frozen=True)
class ParkSpec:
    park_id: str
    name: str
    lat: float
    lon: float
    county: str
    n_turbines: int
    tclass: TurbineClass
    rows: int
    elevation_m: float
    c0: float  # Weibull scale at 100 m (m/s)
    t_mean: float  # annual mean temp (°C)
    t_season: float  # seasonal amplitude (°C)
    dir_modes: tuple[tuple[float, float], ...]  # (direction deg, weight)
    layout_rot_deg: float = 20.0


# Real coordinates; generic class curves, not vendor curves.
PARKS: tuple[ParkSpec, ...] = (
    ParkSpec("WP1", "Fantanele-Cogealac (synthetic)", 44.555, 28.555, "Constanta", 60, CLASS_2500, 6, 80.0,
             8.0, 12.0, 11.0, ((10.0, 0.45), (200.0, 0.30), (280.0, 0.25))),
    ParkSpec("WP2", "Dobrogea park B (synthetic)", 44.330, 28.240, "Constanta", 34, CLASS_2500, 4, 120.0,
             7.7, 11.8, 11.2, ((15.0, 0.45), (195.0, 0.30), (285.0, 0.25))),
    ParkSpec("WP3", "Deleni, Vaslui (synthetic)", 46.700, 27.600, "Vaslui", 23, CLASS_6100, 3, 330.0,
             7.0, 9.8, 12.5, ((320.0, 0.55), (140.0, 0.30), (230.0, 0.15)), layout_rot_deg=35.0),
)


@dataclass
class SynthConfig:
    seed: int = 41
    start: str = "2024-01-01"
    days: int = 731  # 2024 (leap) + 2025
    scale: float = 1.0  # multiplies turbines per park (tests use ~0.1)
    parks: tuple[str, ...] = ("WP1", "WP2", "WP3")
    # per turbine-year
    rate_stop: float = 5.0
    rate_under: float = 0.8
    rate_overheat: float = 0.7
    rate_anem: float = 1.0
    rate_gap_short: float = 10.0
    rate_gap_long: float = 0.4
    # per park-year
    rate_curtail: float = 30.0
    p_icing_spell: float = 0.45
    hot_by_design_frac: float = 0.1
    keep_5min: bool | None = None  # store the 5-min meteo series (default: only for small configs)
    extra: dict = field(default_factory=dict)

    def describe(self) -> str:
        return f"seed={self.seed} start={self.start} days={self.days} scale={self.scale} parks={','.join(self.parks)}"

    def hash(self) -> str:
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()[:12]


# ----------------------------------------------------------------------------------------------------------------
def _layout(spec: ParkSpec, n: int, rng) -> np.ndarray:
    """Turbine positions (m, east/north) on rotated rows: 3.5 D within a row, 7 D between rows (+ jitter)."""
    d = spec.tclass.rotor_d_m
    rows = max(1, min(spec.rows, n))
    per_row = int(np.ceil(n / rows))
    xy = np.array([[c * 3.5 * d, r * 7.0 * d] for r in range(rows) for c in range(per_row)][:n], dtype=np.float64)
    xy += rng.normal(0, 0.3 * d, xy.shape)
    th = np.deg2rad(spec.layout_rot_deg)
    rot = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    xy = xy @ rot.T
    return xy - xy.mean(0)


def _episodes(rng, n_steps: int, rate_per_year: float, dur_lo: int, dur_hi: int, weights=None):
    """Poisson count of episodes, onset uniform (or weighted), log-uniform duration in steps."""
    years = n_steps / (365.25 * 96)
    k = rng.poisson(rate_per_year * years)
    if k == 0:
        return []
    if weights is None:
        on = rng.integers(0, n_steps, k)
    else:
        w = weights / weights.sum()
        on = rng.choice(n_steps, size=k, p=w)
    dur = np.exp(rng.uniform(np.log(dur_lo), np.log(dur_hi), k)).astype(int) + 1
    return sorted((int(o), int(min(n_steps, o + d))) for o, d in zip(on, dur))


# ----------------------------------------------------------------------------------------------------------------
def generate(cfg: SynthConfig | None = None) -> WindData:
    cfg = cfg or SynthConfig()
    rng = np.random.default_rng(cfg.seed)
    specs = [p for p in PARKS if p.park_id in cfg.parks]
    T = cfg.days * 96
    ts = np.datetime64(cfg.start, "m") + np.arange(T) * np.timedelta64(15, "m")
    P = len(specs)
    doy = (ts.astype("datetime64[D]") - ts.astype("datetime64[Y]")).astype(int) + 1
    hour_utc = (ts - ts.astype("datetime64[D]")).astype(int) / 60.0
    hour_loc = (hour_utc + 2.0) % 24

    # --- park weather (hourly latents, interpolated to 15 min) -----------------------------------------------
    H = T // STEPS_PER_H + 2
    th = np.arange(H) * STEPS_PER_H
    tq = np.arange(T)
    full_corr = np.array([[1.0, 0.85, 0.5], [0.85, 1.0, 0.5], [0.5, 0.5, 1.0]])
    idx = [i for i, p in enumerate(PARKS) if p.park_id in cfg.parks]
    L = np.linalg.cholesky(full_corr[np.ix_(idx, idx)])
    zh = _ar1(rng, H, 0.965, 1.0, (P,)) @ L.T
    z = np.stack([np.interp(tq, th, zh[:, p]) for p in range(P)], 1) + _ar1(rng, T, 0.8, 0.15, (P,))
    season = 1 + 0.15 * np.cos(2 * np.pi * (doy - 15) / 365.25)
    diurnal = 1 + 0.06 * np.cos(2 * np.pi * (hour_loc - 2) / 24)
    c = np.array([s.c0 for s in specs])[None] * (season * diurnal)[:, None]
    u = np.clip(ndtr(z), 1e-6, 1 - 1e-6)
    v100 = c * (-np.log1p(-u)) ** (1 / 2.1)

    # direction: regime switching (shared for the two Dobrogea parks) + AR deviation
    wdir = np.zeros((T, P))
    regimes = {}
    for p, s in enumerate(specs):
        key = "dobrogea" if s.county == "Constanta" else s.park_id
        if key not in regimes:
            modes = np.array([m[0] for m in s.dir_modes])
            w = np.array([m[1] for m in s.dir_modes])
            reg = np.empty(H, dtype=int)
            reg[0] = rng.choice(len(modes), p=w / w.sum())
            sw = rng.random(H) < 1 / 36
            nxt = rng.choice(len(modes), size=H, p=w / w.sum())
            for h in range(1, H):
                reg[h] = nxt[h] if sw[h] else reg[h - 1]
            regimes[key] = reg
        modes = np.array([m[0] for m in s.dir_modes])
        base = modes[regimes[key]] + _ar1(rng, H, 0.95, 25.0)
        # interpolate on the unit circle
        sx, cx = np.sin(np.deg2rad(base)), np.cos(np.deg2rad(base))
        wdir[:, p] = np.rad2deg(np.arctan2(np.interp(tq, th, sx), np.interp(tq, th, cx))) % 360

    t_season = np.array([s.t_season for s in specs])[None] * -np.cos(2 * np.pi * (doy - 20) / 365.25)[:, None]
    temp = (np.array([s.t_mean for s in specs])[None] + t_season
            + 4.0 * np.cos(2 * np.pi * (hour_loc - 15) / 24)[:, None])
    temp += np.stack([np.interp(tq, th, col) for col in _ar1(rng, H, 0.985, 3.5, (P,)).T], 1)
    rh = (72 + 10 * np.cos(2 * np.pi * (doy - 15) / 365.25)[:, None]
          - 12 * np.cos(2 * np.pi * (hour_loc - 15) / 24)[:, None]
          + np.stack([np.interp(tq, th, col) for col in _ar1(rng, H, 0.97, 9.0, (P,)).T], 1))
    rh = np.clip(rh, 25, 100)
    pz = np.stack([np.interp(tq, th, col) for col in _ar1(rng, H, 0.9, 1.0, (P,)).T], 1)
    precip = np.where(rh > 70, np.clip(pz - 1.3, 0, None) * 1.5, 0.0) / STEPS_PER_H
    pres = (1016 - np.array([s.elevation_m for s in specs])[None] / 8.3
            + np.stack([np.interp(tq, th, col) for col in _ar1(rng, H, 0.99, 6.0, (P,)).T], 1))

    # --- turbines ------------------------------------------------------------------------------------------
    turbines, park_of, tcls, xy_all, wake_cols = [], [], [], [], []
    for p, s in enumerate(specs):
        n = max(3 if s.tclass is CLASS_2500 else 2, int(round(s.n_turbines * cfg.scale)))
        xy = _layout(s, n, rng)
        wt = wake_table(xy, s.tclass.rotor_d_m)
        for j in range(n):
            lat = s.lat + xy[j, 1] / 111_320.0
            lon = s.lon + xy[j, 0] / (111_320.0 * np.cos(np.deg2rad(s.lat)))
            turbines.append({
                "turbine_id": f"{s.park_id}.TID{j + 1:02d}",
                "park_id": s.park_id,
                "model": s.tclass.name,
                "rated_kw": s.tclass.rated_kw,
                "hub_height_m": s.tclass.hub_height_m,
                "rotor_d_m": s.tclass.rotor_d_m,
                "lat": round(float(lat), 6),
                "lon": round(float(lon), 6),
                "county": s.county,
                "power_unit": "kW",
                "power_scale": 1.0,
            })
            park_of.append(p)
            tcls.append(s.tclass)
            wake_cols.append(wt[:, j])
            xy_all.append(xy[j])
    N = len(turbines)
    park_of = np.array(park_of)
    rated = np.array([t["rated_kw"] for t in turbines])
    hub = np.array([t["hub_height_m"] for t in turbines])
    wake = np.stack(wake_cols, 1)  # [72, N]

    v_hub_free = shear_to_hub(v100[:, park_of], hub[None])
    meteo_w = (v_hub_free * (1 + rng.normal(0, 0.01, N))[None]).astype(np.float32)
    meteo_d = ((wdir[:, park_of] + rng.normal(0, 3, N)[None]) % 360).astype(np.float32)
    dbin = (np.round(wdir[:, park_of] / 5.0).astype(int)) % 72
    deficit = wake[dbin, np.arange(N)[None]] * np.clip((15 - v_hub_free) / 8, 0.15, 1.0)
    v_eff = v_hub_free * (1 - deficit) * (1 + _ar1(rng, T, 0.6, 0.035, (N,)))
    v_eff = np.clip(v_eff, 0, None)
    amb = temp[:, park_of] + rng.normal(0.5, 0.3, N)[None] + rng.normal(0, 0.2, (T, N))
    rho = air_density(pres[:, park_of], amb)
    vn = density_normalise(v_eff, rho)
    eff = rng.normal(1.0, 0.015, N)
    frac = np.zeros((T, N))
    for cl in set(tcls):
        cols = np.array([k for k in range(N) if tcls[k] is cl])
        frac[:, cols] = cl.power_frac(vn[:, cols])
    avail = np.clip(rated[None] * eff[None] * frac * (1 + rng.normal(0, 0.008, (T, N))), 0, rated[None] * 1.01)

    labels = np.zeros((T, N), dtype=np.uint8)
    lpri = np.zeros((T, N), dtype=np.uint8)
    episodes: list[dict] = []

    def put(j, s, e, cls, mask=None, severity=1.0, params=None):
        sl = slice(s, e)
        m = np.ones(e - s, bool) if mask is None else mask
        if not m.any():
            return
        pr = PRIORITY[cls]
        win = m & (lpri[sl, j] <= pr)
        labels[sl, j][win] = CLASS_INDEX[cls]
        lpri[sl, j][win] = pr
        on = s + int(np.argmax(m))
        en = s + int(len(m) - np.argmax(m[::-1]))
        episodes.append({
            "asset_id": turbines[j]["turbine_id"], "asset_kind": "turbine", "fault": cls,
            "onset": _iso(ts[on]), "end": _iso(ts[en]) if en < T else None,
            "severity": float(round(severity, 3)), "params": json.dumps(params or {}),
        })

    power = avail.copy()
    nac_k = rng.normal(0.93, 0.02, N)
    v_nac_true = v_eff.copy()
    temp_drift = {k: np.zeros((T, N)) for k in THERMAL}
    stopped = np.zeros((T, N), bool)
    anem = np.zeros((T, N), dtype=np.int8)  # 0 ok, 1 bias, 2 stuck, 3 erratic
    anem_par = np.ones((T, N))
    icing_nac = np.ones((T, N))

    # underperformance (yaw sector loss or power-curve degradation)
    for j in range(N):
        for s, e in _episodes(rng, T, cfg.rate_under, 7 * 96, 45 * 96):
            if rng.random() < 0.6:
                delta = rng.uniform(10, 25)
                centre = rng.uniform(0, 360)
                dd = np.abs((meteo_d[s:e, j] - centre + 180) % 360 - 180)
                f = np.where(dd < 60, np.cos(np.deg2rad(delta)) ** 2, np.cos(np.deg2rad(delta / 3)) ** 2)
                power[s:e, j] *= f
                put(j, s, e, "underperformance", severity=1 - np.cos(np.deg2rad(delta)) ** 2,
                    params={"kind": "yaw", "offset_deg": round(delta, 1), "sector_centre_deg": round(centre)})
            else:
                sv = rng.uniform(0.08, 0.2)
                part = frac[s:e, j] < 0.95
                power[s:e, j] *= np.where(part, 1 - sv, 1.0)
                put(j, s, e, "underperformance", severity=sv, params={"kind": "power_curve", "loss": round(sv, 3)})

    # icing: park spells of T < 1 °C and RH > 90 %
    for p in range(P):
        cond = (temp[:, p] < 1.0) & (rh[:, p] > 90)
        d = np.diff(np.concatenate([[0], cond.astype(int), [0]]))
        for s, e in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)):
            if e - s < 8 or rng.random() > cfg.p_icing_spell:
                continue
            cols = np.flatnonzero(park_of == p)
            hit = cols[rng.random(len(cols)) < rng.uniform(0.3, 1.0)]
            for j in hit:
                loss = rng.uniform(0.25, 0.85)
                ramp = rng.integers(4, 13)
                ee = min(T, e + int(rng.integers(8, 49)))
                r = np.clip((np.arange(ee - s) + 1) / ramp, 0, 1) * loss
                visible = avail[s:ee, j] > 0.05 * rated[j]
                power[s:ee, j] *= 1 - r
                if rng.random() < 0.5:
                    icing_nac[s:ee, j] = 1 - 0.15 * r / loss
                put(j, s, ee, "icing", mask=visible, severity=loss, params={"loss": round(loss, 3)})

    # overheating: residual drifting 3–21 days on gearbox or one generator bearing, often ending in a stop
    post_stops = []
    for j in range(N):
        for s, e in _episodes(rng, T, cfg.rate_overheat, 3 * 96, 21 * 96):
            comp = ["gearbox_t", "gen1_t", "gen2_t"][rng.integers(0, 3)]
            amp = rng.uniform(8, 20)
            x = ((np.arange(e - s) + 1) / (e - s)) ** 2 * amp
            temp_drift[comp][s:e, j] += x  # scaled by load below
            put(j, s, e, "overheating", severity=amp / 20, params={"component": comp, "amp_c": round(amp, 1)})
            if rng.random() < 0.6 and e < T:
                post_stops.append((j, e, min(T, e + int(rng.integers(96, 5 * 96)))))

    # anemometer faults (power keeps following the real wind)
    for j in range(N):
        for s, e in _episodes(rng, T, cfg.rate_anem, 2 * 96, 20 * 96):
            kind = int(rng.integers(1, 4))
            anem[s:e, j] = kind
            anem_par[s:e, j] = rng.uniform(0.55, 0.85) if kind == 1 else 1.0
            put(j, s, e, "anemometer_fault", params={"kind": ["", "bias", "stuck", "erratic"][kind]})

    # curtailment: park-wide cap, onsets weighted towards high production
    for p in range(P):
        cols = np.flatnonzero(park_of == p)
        w = (avail[:, cols].mean(1) / rated[cols].mean()) ** 2 + 1e-3
        for s, e in _episodes(rng, T, cfg.rate_curtail, 4, 32, weights=w):
            cap = float(rng.choice([0.0, 0.3, 0.5, 0.7], p=[0.15, 0.3, 0.3, 0.25]))
            for j in cols:
                if rng.random() > 0.95:
                    continue
                capkw = cap * rated[j]
                binding = power[s:e, j] > capkw + 0.02 * rated[j]
                power[s:e, j] = np.minimum(power[s:e, j], capkw + rng.normal(0, 0.003 * rated[j], e - s).clip(0))
                put(j, s, e, "curtailment", mask=binding, severity=1 - cap, params={"cap_frac": cap})

    # single-turbine fault stops (+ failures after overheating)
    for j in range(N):
        for s, e in _episodes(rng, T, cfg.rate_stop, 8, 288):
            stopped[s:e, j] = True
            put(j, s, e, "fault_stop", params={"cause": "trip"})
    for j, s, e in post_stops:
        stopped[s:e, j] = True
        put(j, s, e, "fault_stop", params={"cause": "failure_after_overheating"})
    aux = rng.uniform(3, 10, N)
    power = np.where(stopped, -aux[None] + rng.normal(0, 0.5, (T, N)), power)
    load = np.clip(power / rated[None], 0, 1.01)

    # --- sensors ---------------------------------------------------------------------------------------
    k_nac = np.where(stopped, 0.99, nac_k[None])
    # point anemometer behind the rotor vs rotor-averaged wind: decorrelated turbulence (~5 %) + sensor noise
    wd = v_nac_true * k_nac * icing_nac * (1 + _ar1(rng, T, 0.5, 0.05, (N,))) + rng.normal(0, 0.3, (T, N))
    wd = np.where(anem == 1, wd * anem_par, wd)
    for j in range(N):  # stuck: hold the value at fault onset
        m = anem[:, j] == 2
        if m.any():
            d = np.diff(np.concatenate([[0], m.astype(int), [0]]))
            for s, e in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)):
                wd[s:e, j] = wd[s, j]
    err_mask = (anem == 3) & (rng.random((T, N)) < 0.3)
    wd = np.where(err_mask, np.abs(rng.normal(0.3, 0.2, (T, N))), wd)
    wd = np.clip(wd, 0, None) * 3.6  # km/h

    sensors = {"act_pwt": power.astype(np.float32), "wd_spd": wd.astype(np.float32)}
    hot = rng.random(N) < cfg.hot_by_design_frac
    for name, (base, rise, tau, gam) in THERMAL.items():
        offset = rng.normal(0, 2.0, N) + np.where(hot & (rng.random(N) < 0.6), rng.uniform(8, 14, N), 0.0)
        x = amb + base + offset[None] + rise * load**gam + temp_drift[name] * (0.4 + 0.6 * load)
        sensors[name] = (_lag(x, tau) + rng.normal(0, 0.4, (T, N))).astype(np.float32)
    sensors["turbine_t"] = amb.astype(np.float32)

    # --- data gaps + integral error codes (applied last) --------------------------------------------------
    err = np.zeros((T, N), dtype=np.int32)
    for j in range(N):
        eps = _episodes(rng, T, cfg.rate_gap_short, 1, 48) + _episodes(rng, T, cfg.rate_gap_long, 96, 672)
        for s, e in eps:
            if rng.random() < 0.3:
                err[s:e, j] = int(rng.choice(ERROR_CODES))
            for k in sensors:
                sensors[k][s:e, j] = np.nan
            put(j, s, e, "data_gap", params={"coded": bool(err[s, j] != 0)})

    keep5 = cfg.keep_5min if cfg.keep_5min is not None else T * N <= 2_000_000
    m5 = None
    if keep5:  # 5-min per-turbine meteo: 15-min series repeated + small noise, mean-preserving per bin
        noise = rng.normal(0, 0.15, (3 * T, N)).astype(np.float32)
        noise -= np.repeat(noise.reshape(T, 3, N).mean(1), 3, axis=0)
        m5 = {"wind_ms": np.clip(np.repeat(meteo_w, 3, axis=0) + noise, 0, None).astype(np.float32),
              "wdir_deg": np.repeat(meteo_d, 3, axis=0)}
        meteo_w = m5["wind_ms"].reshape(T, 3, N).mean(1)
    weather = {
        "temp_c": temp.astype(np.float32), "rh_pct": rh.astype(np.float32), "precip_mm": precip.astype(np.float32),
        "pressure_hpa": pres.astype(np.float32), "wind100_ms": v100.astype(np.float32),
        "wdir100_deg": wdir.astype(np.float32),
    }
    assert set(weather) == set(WEATHER_FIELDS) and P == weather["temp_c"].shape[1]
    meteo = {"wind_ms": meteo_w.astype(np.float32), "wdir_deg": meteo_d}
    assert set(meteo) == set(METEO_FIELDS)
    return WindData(
        ts=ts, turbines=turbines, sensors=sensors, err=err, park_ids=[s.park_id for s in specs],
        weather=weather, meteo=meteo, labels=labels, episodes=episodes, source=f"synth-v1:{cfg.hash()}",
        meteo_5min=m5,
    )


def main(argv: list[str] | None = None) -> None:
    """`python -m wattsentry.model.wind.synth --out DIR [--days N --scale S]`: write the contract tables."""
    import argparse
    from pathlib import Path

    from .data import fault_labels_table, meteo_table, to_long, turbines_table, weather_table

    ap = argparse.ArgumentParser(description=main.__doc__)
    ap.add_argument("--out", required=True)
    ap.add_argument("--days", type=int, default=SynthConfig.days)
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=41)
    a = ap.parse_args(argv)
    d = generate(SynthConfig(seed=a.seed, days=a.days, scale=a.scale))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    to_long(d).write_parquet(out / "wind_telemetry.parquet", compression="zstd")
    turbines_table(d).write_parquet(out / "turbines.parquet")
    meteo_table(d).write_parquet(out / "meteo_wind.parquet", compression="zstd")
    weather_table(d).write_parquet(out / "weather.parquet", compression="zstd")
    fault_labels_table(d).write_parquet(out / "fault_labels.parquet")
    print(f"wrote {out} ({len(d.turbines)} turbines, {d.n_steps} steps)")


if __name__ == "__main__":
    main()
