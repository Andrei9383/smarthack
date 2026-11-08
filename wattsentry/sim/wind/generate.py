"""Clean (fault-free) wind telemetry for the three RO parks, driven by real ERA5 100 m wind (card #22).

Per park and 15-min step (plan §3.5, §4):
  v100      = interp(ERA5 hourly 100 m wind at the park point) · (1 + AR(1) park turbulence)
  v_free    = v100 · (h_hub / 100)^alpha(park, season) · terrain speed-up(turbine)        (free stream at hub)
  v_eff     = v_free · (1 − wake(direction, layout)) · (1 + AR(1) turbine turbulence)   (rotor-effective)
  rho       = p / (R_d · T)  with ERA5 surface pressure and the turbine's ambient temperature
  v_norm    = v_eff · (rho / 1.225)^(1/3)
  act_pwt   = rated · efficiency(turbine) · class_curve(v_norm) · (1 + noise), clipped to [0, rated]
  wd_spd    = nacelle anemometer behind the rotor: v_eff · k_nac · (1 + AR noise) + noise, in km/h
  temps     = first-order lag of ambient + base + offset + rise · load^gamma (THERMAL), + noise
  meteo     = 5-min free-stream hub wind per turbine (no wake) and ERA5 direction
Wakes: Jensen top-hat deficits from the REAL OSM positions (parks.py), per 5° direction bin.

`CleanWind.render()` turns a (possibly faulted) power matrix back into the eight PPC sensors with the SAME noise
draws every time, so the fault injector (wattsentry.sim.wind_faults) changes telemetry only where it injects.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from wattsentry.model.wind.data import METEO_FIELDS, WEATHER_FIELDS, WindData
from wattsentry.model.wind.physics import air_density, density_normalise

from . import era5
from .config import WindSimConfig, weather_file
from .parks import ParkLayout, build_layouts
from .physics import THERMAL, ar1, lag, wake_deficit, wake_table

STEP = np.timedelta64(15, "m")


@dataclass
class CleanWind:
    cfg: WindSimConfig
    ts: np.ndarray  # datetime64[m] [T]
    layouts: list[ParkLayout]
    turbines: list[dict]
    park_ids: list[str]
    park_of: np.ndarray  # [N]
    rated: np.ndarray  # [N] kW
    tclass: list  # TurbineClass per turbine
    weather: era5.ParkWeather
    v_free: np.ndarray  # [T, N] free-stream hub wind (m/s)
    v_eff: np.ndarray  # [T, N] rotor-effective wind (wake + turbulence)
    rho: np.ndarray  # [T, N]
    v_norm: np.ndarray  # [T, N]
    frac: np.ndarray  # [T, N] class-curve fraction of rated
    avail: np.ndarray  # [T, N] clean power (kW)
    amb: np.ndarray  # [T, N] ambient temperature at the turbine
    wdir: np.ndarray  # [T, N] direction at the turbine (ERA5 + park wobble)
    meteo_5min: dict[str, np.ndarray]  # wind_ms / wdir_deg [3T, N]
    nac_k: np.ndarray  # [N] nacelle transfer factor while operating
    t_offset: dict[str, np.ndarray]  # component -> [N] °C
    aux_kw: np.ndarray  # [N] self-consumption when stopped
    k_speed: float = 1.0  # ERA5 100 m speed factor (capacity-factor calibration)
    cf_target: float | None = None
    timings: dict[str, float] = field(default_factory=dict)

    @property
    def T(self) -> int:
        return len(self.ts)

    @property
    def N(self) -> int:
        return len(self.turbines)

    @property
    def meteo15(self) -> dict[str, np.ndarray]:
        w = self.meteo_5min["wind_ms"].reshape(self.T, 3, self.N).mean(1)
        return {"wind_ms": w.astype(np.float32), "wdir_deg": self.meteo_5min["wdir_deg"][::3].astype(np.float32)}

    def weather15(self) -> dict[str, np.ndarray]:
        q = self.weather.q15
        return {k: q[k].astype(np.float32) for k in WEATHER_FIELDS}

    # ------------------------------------------------------------------------------------------------------------
    def render(self, power=None, stopped=None, temp_drift=None, nac_factor=None) -> dict[str, np.ndarray]:
        """Eight PPC sensors [T, N] (float32) from a power matrix. Noise is drawn from a fixed stream, so two calls
        differ only where their inputs differ (and, for temperatures, in the thermal lag after that)."""
        T, N = self.T, self.N
        rng = np.random.default_rng([self.cfg.seed, 7])
        power = self.avail if power is None else power
        stopped = np.zeros((T, N), bool) if stopped is None else stopped
        aux_noise = rng.normal(0, 0.5, (T, N))
        p = np.where(stopped, -self.aux_kw[None] + aux_noise, power)
        load = np.clip(p / self.rated[None], 0, 1.01)
        k_nac = np.where(stopped, 0.99, self.nac_k[None])
        nac_ar = ar1(rng, T, 0.5, 0.05, (N,))
        nac_white = rng.normal(0, 0.3, (T, N))
        f = 1.0 if nac_factor is None else nac_factor
        # |.| not clip: a cup anemometer in calm air still reads a little, never a long run of exact zeros
        wd = np.abs(self.v_eff * k_nac * f * (1 + nac_ar) + nac_white) * 3.6  # km/h
        out = {"act_pwt": p.astype(np.float32), "wd_spd": wd.astype(np.float32)}
        for name, (base, rise, tau, gam) in THERMAL.items():
            drift = 0.0 if temp_drift is None or name not in temp_drift else temp_drift[name] * (0.4 + 0.6 * load)
            x = self.amb + base + self.t_offset[name][None] + rise * load**gam + drift
            out[name] = (lag(x, tau) + rng.normal(0, 0.4, (T, N))).astype(np.float32)
        out["turbine_t"] = self.amb.astype(np.float32)
        return out

    def to_wind_data(self, sensors, err=None, labels=None, episodes=None, source="sim-wind") -> WindData:
        return WindData(
            ts=self.ts, turbines=self.turbines, sensors=sensors,
            err=np.zeros((self.T, self.N), np.int32) if err is None else err,
            park_ids=self.park_ids, weather=self.weather15(), meteo=self.meteo15, labels=labels,
            episodes=episodes or [], source=source, meteo_5min={k: v for k, v in self.meteo_5min.items()},
        )


# ----------------------------------------------------------------------------------------------------------------
def national_wind_cf(year: int) -> float | None:
    """Real RO wind capacity factor of a year: mean ENTSO-E gen_wind MW / installed GW (data/raw, card #9)."""
    import polars as pl

    from .config import RAW

    try:
        e = pl.read_parquet(RAW / "entsoe_ro.parquet")
        c = pl.read_parquet(RAW / "ro_installed_capacity.parquet")
    except Exception:
        return None
    mw = e.filter((pl.col("series") == "gen_wind") & (pl.col("ts").dt.year() == year))["mw"].mean()
    gw = c.filter((pl.col("series") == "gen_wind") & (pl.col("year") == year))["gw"]
    if mw is None or gw.len() == 0:
        return None
    return float(mw) / (float(gw[0]) * 1000.0)


def calibrate_speed(pw, layouts, tables, park_of, hub, rated, tclass, cf_target: float) -> float:
    """One ERA5 speed factor k for all parks such that the clean fleet capacity factor over the WHOLE calendar year
    of the first step (hourly ERA5, shear, wakes, density; no turbulence) equals `cf_target`. Window-independent,
    so short test runs share the full run's k."""
    fy = pw.fit_hourly
    vh = fy["wind100_ms"][:, park_of] * (hub[None] / 100.0) ** pw.fit_alpha[:, park_of]
    wd = fy["wdir100_deg"][:, park_of]
    rho = air_density(fy["pressure_hpa"][:, park_of], fy["temp_c"][:, park_of])
    groups = [(tc, np.array([i for i in range(len(tclass)) if tclass[i] is tc]))
              for tc in {id(c): c for c in tclass}.values()]

    def cf(k):
        vf = k * vh
        dfc = np.zeros_like(vf)
        for p in range(len(layouts)):
            cols = np.flatnonzero(park_of == p)
            dfc[:, cols] = wake_deficit(tables[p], wd[:, cols], vf[:, cols])
        vn = density_normalise(vf * (1 - dfc), rho)
        tot = sum((tc.power_frac(vn[:, cols]) * rated[cols][None]).sum() for tc, cols in groups)
        return tot / (rated.sum() * len(vf))

    lo, hi = 0.7, 1.5
    for _ in range(20):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if cf(mid) < cf_target else (lo, mid)
    return round(0.5 * (lo + hi), 4)


def generate_clean(cfg: WindSimConfig | None = None, log=print) -> CleanWind:
    cfg = cfg or WindSimConfig()
    t_start = time.perf_counter()
    timings = {}
    path = weather_file(cfg)
    ts = np.arange(np.datetime64(cfg.start, "m"), np.datetime64(cfg.end, "m"), STEP)
    T = len(ts)
    layouts = build_layouts(cfg.parks, cfg.max_turbines_per_park, era5.weather_points(path))
    pw = era5.load(path, layouts, ts, cfg.shear)
    timings["t_inputs_s"] = time.perf_counter() - t_start
    P = len(layouts)
    rng = np.random.default_rng(cfg.seed)

    turbines, park_of, tclass, hub, rated, rotor = [], [], [], [], [], []
    for p, lay in enumerate(layouts):
        tc = lay.park.tclass
        unit_mw = lay.park.park_id in cfg.mw_parks
        for j in range(lay.n):
            turbines.append({
                "turbine_id": f"{lay.park.park_id}.TID{j + 1:02d}",
                "park_id": lay.park.park_id,
                "model": tc.name,
                "rated_kw": float(tc.rated_kw),
                "hub_height_m": float(tc.hub_height_m),
                "rotor_d_m": float(tc.rotor_d_m),
                "lat": round(float(lay.lat[j]), 6),
                "lon": round(float(lay.lon[j]), 6),
                "county": lay.park.county,
                "power_unit": "MW" if unit_mw else "kW",
                "power_scale": 1000.0 if unit_mw else 1.0,
                "osm_id": int(lay.osm_id[j]),
                "park_name": lay.park.name,
                "era5_point": lay.era5_point,
            })
            park_of.append(p)
            tclass.append(tc)
            hub.append(tc.hub_height_m)
            rated.append(tc.rated_kw)
            rotor.append(tc.rotor_d_m)
    N = len(turbines)
    park_of = np.array(park_of)
    hub, rated = np.array(hub), np.array(rated)

    def physics(k: float):
        """Wind -> power for an ERA5 speed factor k. Re-seeded, so every call draws the same noise."""
        rng = np.random.default_rng(cfg.seed)
        q = pw.q15
        g_park = 1 + ar1(rng, T, 0.85, cfg.turb_park_sd, (P,))
        v100 = np.clip(k * q["wind100_ms"] * g_park, 0, None)
        speedup = 1 + rng.normal(0, 0.01, N)  # small terrain speed-up per position
        v_free = v100[:, park_of] * (hub[None] / 100.0) ** pw.alpha[:, park_of] * speedup[None]
        wobble = ar1(rng, T, 0.9, 4.0, (P,))
        wdir = (q["wdir100_deg"][:, park_of] + wobble[:, park_of] + rng.normal(0, 3, N)[None]) % 360
        deficit = np.zeros((T, N))
        for p, lay in enumerate(layouts):
            cols = np.flatnonzero(park_of == p)
            deficit[:, cols] = wake_deficit(tables[p], wdir[:, cols], v_free[:, cols])
        v_eff = np.clip(v_free * (1 - deficit) * (1 + ar1(rng, T, 0.6, cfg.turb_turbine_sd, (N,))), 0, None)
        amb = q["temp_c"][:, park_of] + rng.normal(0.5, 0.3, N)[None] + rng.normal(0, 0.2, (T, N))
        rho = air_density(q["pressure_hpa"][:, park_of], amb)
        v_norm = density_normalise(v_eff, rho)
        frac = np.zeros((T, N))
        for tc in {id(c): c for c in tclass}.values():
            cols = np.array([i for i in range(N) if tclass[i] is tc])
            frac[:, cols] = tc.power_frac(v_norm[:, cols])
        eff = rng.normal(1.0, 0.015, N)
        # at rated the controller holds power just under the nameplate (never a run of exactly-rated values)
        cap = rated[None] * (1 - np.abs(rng.normal(0, 0.004, (T, N))))
        avail = np.clip(rated[None] * eff[None] * frac * (1 + rng.normal(0, 0.008, (T, N))), 0, cap)
        return rng, v_free, wdir, v_eff, amb, rho, v_norm, frac, avail

    tables = [wake_table(lay.xy, lay.park.tclass.rotor_d_m) for lay in layouts]
    k_speed, cf_target = 1.0, None
    if cfg.cf_calibration:
        cf_target = national_wind_cf(int(str(ts[0])[:4]))
    if cf_target is not None:
        k_speed = calibrate_speed(pw, layouts, tables, park_of, hub, rated, tclass, cf_target)
    rng, v_free, wdir, v_eff, amb, rho, v_norm, frac, avail = physics(k_speed)

    # --- meteo: 5-min free-stream hub wind per turbine, mean-preserving within each 15-min bin --------------------
    meteo15 = v_free * (1 + rng.normal(0, 0.01, N))[None]
    noise = rng.normal(0, 0.15, (3 * T, N))
    noise -= np.repeat(noise.reshape(T, 3, N).mean(1), 3, axis=0)
    m5w = np.abs(np.repeat(meteo15, 3, axis=0) + noise)
    m5 = {"wind_ms": m5w.astype(np.float32), "wdir_deg": np.repeat(wdir, 3, axis=0).astype(np.float32)}
    assert set(m5) == set(METEO_FIELDS)

    # --- static per-turbine parameters ---------------------------------------------------------------------------
    nac_k = rng.normal(0.93, 0.02, N)
    hot = rng.random(N) < cfg.hot_by_design_frac
    t_offset = {}
    for name in THERMAL:
        t_offset[name] = rng.normal(0, 2.0, N) + np.where(hot & (rng.random(N) < 0.6), rng.uniform(8, 14, N), 0.0)
    aux = rng.uniform(3, 10, N)
    timings["t_physics_s"] = time.perf_counter() - t_start - timings["t_inputs_s"]
    f32 = np.float32
    cw = CleanWind(
        cfg=cfg, ts=ts, layouts=layouts, turbines=turbines, park_ids=[lay.park.park_id for lay in layouts],
        park_of=park_of, rated=rated, tclass=tclass, weather=pw, v_free=v_free.astype(f32), v_eff=v_eff.astype(f32),
        rho=rho.astype(f32), v_norm=v_norm.astype(f32), frac=frac.astype(f32), avail=avail.astype(f32),
        amb=amb.astype(f32), wdir=wdir.astype(f32), meteo_5min=m5, nac_k=nac_k, t_offset=t_offset, aux_kw=aux,
        k_speed=k_speed, cf_target=cf_target, timings=timings,
    )
    log(f"[sim.wind] clean: {N} turbines in {P} parks, {T} steps; ERA5 speed factor {k_speed} "
        f"(target CF {cf_target}); {timings['t_inputs_s']:.1f}s inputs, {timings['t_physics_s']:.1f}s physics")
    return cw
