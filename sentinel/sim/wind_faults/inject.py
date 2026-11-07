"""Wind fault injector (card #23): faults on top of the clean ERA5-driven telemetry, with exact onset/end labels.

Order (physics first, sensors last): underperformance (yaw, power curve) → icing → overheating drift (→ failure
stop) → park-wide curtailment caps → single-turbine stops → re-render the eight sensors from the faulted power
(same noise as the clean run, so telemetry differs only from each onset on) → anemometer faults → communication
gaps + integral error codes.

Labels: one `fault_labels` row per episode and turbine (team/contracts/data.md, asset_kind = turbine, fault = a
WindAnomaly value). `onset` / `end` are exact injector truth on the 15-min grid (`end` exclusive, null = runs to the
end of the series). Curtailment and icing are labelled as WHOLE episodes: the setpoint / ice is there from onset
to end whether or not it bites. Whether power was actually cut is stored in `params`:
  power_cut (bool), cut_steps, cut_onset / cut_end (first / last+1 step where it bit, ISO UTC or null).
Rationale: the episode is the operational truth (a TSO setpoint, an icing alert) and is what a report should count;
but a cap above the available power or ice on a parked rotor leaves no trace in SCADA, so a detector cannot be
expected to see it. sentinel.model.wind.data therefore builds step labels in two modes: `effective` (default, used
by M6 training and metrics: cut_onset..cut_end, episodes that never bit are dropped) and `episode` (whole span).
Every other class changes the telemetry from its onset, so both modes agree on them.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np

from sentinel.model.wind.data import _iso
from sentinel.sim.wind.generate import CleanWind
from sentinel.sim.wind.physics import ERROR_CODES, STEPS_PER_H, runs

from .config import KIND_CLASS, KINDS, FaultConfig

STEPS_PER_DAY = 24 * STEPS_PER_H


@dataclass
class FaultResult:
    sensors: dict[str, np.ndarray]  # faulted telemetry [T, N]
    err: np.ndarray  # int32 [T, N] integral error code per step (0 = none)
    episodes: list[dict]  # fault_labels rows
    stats: dict = field(default_factory=dict)


def _episodes(rng, n_steps: int, rate_per_year: float, lo_steps: float, hi_steps: float, weights=None):
    """Poisson number of episodes, onset uniform (or weighted), log-uniform duration in steps → [(s, e)]."""
    years = n_steps / (365.25 * STEPS_PER_DAY)
    k = rng.poisson(rate_per_year * years)
    if k == 0:
        return []
    if weights is None:
        on = rng.integers(0, n_steps, k)
    else:
        on = rng.choice(n_steps, size=k, p=weights / weights.sum())
    dur = np.exp(rng.uniform(np.log(lo_steps), np.log(hi_steps), k)).astype(int) + 1
    return sorted((int(o), int(min(n_steps, o + d))) for o, d in zip(on, dur))


def inject(clean: CleanWind, cfg: FaultConfig | None = None) -> FaultResult:
    cfg = cfg or FaultConfig()
    T, N = clean.T, clean.N
    ts = clean.ts
    rated = clean.rated
    park_of = clean.park_of
    tids = [t["turbine_id"] for t in clean.turbines]
    on = set(cfg.kinds)
    unknown = on - set(KINDS)
    if unknown:
        raise ValueError(f"unknown fault kinds {sorted(unknown)}; known: {KINDS}")

    def rng_for(kind: str):  # one stream per kind: enabling one kind never changes another's draws
        return np.random.default_rng([cfg.seed, KINDS.index(kind)])

    episodes: list[dict] = []

    def label(j, s, e, kind, severity=1.0, params=None, cut=None):
        p = dict(params or {})
        p["kind"] = kind
        if cut is not None:  # curtailment / icing: did it actually bite?
            idx = np.flatnonzero(cut)
            p["power_cut"] = bool(len(idx))
            p["cut_steps"] = int(len(idx))
            p["cut_onset"] = _iso(ts[s + idx[0]]) if len(idx) else None
            p["cut_end"] = (_iso(ts[s + idx[-1] + 1]) if s + idx[-1] + 1 < T else None) if len(idx) else None
        episodes.append({
            "asset_id": tids[j], "asset_kind": "turbine", "fault": KIND_CLASS[kind],
            "onset": _iso(ts[s]), "end": _iso(ts[e]) if e < T else None,
            "severity": float(round(min(max(severity, 0.0), 1.0), 3)), "params": json.dumps(p),
        })

    power = clean.avail.astype(np.float64).copy()
    stopped = np.zeros((T, N), bool)
    drift = {c: np.zeros((T, N)) for c in ("gearbox_t", "gen1_t", "gen2_t")}
    nac_factor = np.ones((T, N))
    d = STEPS_PER_DAY
    h = STEPS_PER_H

    # --- underperformance: yaw misalignment (loss depends on the wind sector) -----------------------------------
    if "yaw" in on:
        rng = rng_for("yaw")
        for j in range(N):
            for s, e in _episodes(rng, T, cfg.rate_yaw, cfg.under_days[0] * d, cfg.under_days[1] * d):
                delta = rng.uniform(*cfg.yaw_deg)
                centre = rng.uniform(0, 360)
                dd = np.abs((clean.wdir[s:e, j] - centre + 180) % 360 - 180)
                f = np.where(dd < 60, np.cos(np.deg2rad(delta)) ** 2, np.cos(np.deg2rad(delta / 3)) ** 2)
                power[s:e, j] *= f
                label(j, s, e, "yaw", severity=1 - np.cos(np.deg2rad(delta)) ** 2,
                      params={"offset_deg": round(delta, 1), "sector_centre_deg": round(centre)})
    # --- underperformance: pitch / power-curve degradation (partial-load loss) ----------------------------------
    if "power_curve" in on:
        rng = rng_for("power_curve")
        for j in range(N):
            for s, e in _episodes(rng, T, cfg.rate_power_curve, cfg.under_days[0] * d, cfg.under_days[1] * d):
                sv = rng.uniform(*cfg.power_curve_loss)
                part = clean.frac[s:e, j] < 0.95
                power[s:e, j] *= np.where(part, 1 - sv, 1.0)
                label(j, s, e, "power_curve", severity=sv, params={"loss": round(sv, 3)})

    # --- icing: real ERA5 spells (T < 1 °C with high humidity or precipitation) at the park ----------------------
    n_spells = n_iced = 0
    if "icing" in on:
        rng = rng_for("icing")
        w = clean.weather.q15
        for p, pid in enumerate(clean.park_ids):
            cond = (w["temp_c"][:, p] < cfg.icing_temp_c) & (
                (w["rh_pct"][:, p] >= cfg.icing_rh) | (w["precip_mm"][:, p] > cfg.icing_precip_mm))
            cols = np.flatnonzero(park_of == p)
            for s, e in runs(cond):
                if e - s < cfg.icing_min_hours * h:
                    continue
                n_spells += 1
                if rng.random() > cfg.p_icing_spell:
                    continue
                n_iced += 1
                hit = cols[rng.random(len(cols)) < rng.uniform(0.3, 1.0)]
                tmin = float(w["temp_c"][s:e, p].min())
                rhm = float(w["rh_pct"][s:e, p].mean())
                pr = float(w["precip_mm"][s:e, p].sum() / h)
                for j in hit:
                    loss = rng.uniform(0.25, 0.85)
                    ramp = int(rng.integers(4, 13))
                    ee = min(T, e + int(rng.integers(2 * h, 12 * h + 1)))  # ice persists after the spell
                    r = np.clip((np.arange(ee - s) + 1) / ramp, 0, 1) * loss
                    power[s:ee, j] *= 1 - r
                    stop = rng.random() < cfg.icing_stop_frac
                    if stop:  # ice detector shuts the turbine down once the ice has built up
                        stopped[min(ee, s + ramp):ee, j] = True
                    if rng.random() < 0.5:  # iced cup anemometer under-reads
                        nac_factor[s:ee, j] = 1 - 0.15 * r / loss
                    visible = clean.avail[s:ee, j] > 0.05 * rated[j]
                    label(j, s, ee, "icing", severity=loss, cut=visible,
                          params={"loss": round(loss, 3), "ice_stop": bool(stop), "era5_tmin_c": round(tmin, 1),
                                  "era5_rh_mean": round(rhm, 1), "era5_precip_mm": round(pr, 2)})

    # --- gearbox / generator-bearing degradation: residual drifting 3-21 days, often ending in a failure ---------
    post_stops = []
    if "overheating" in on:
        rng = rng_for("overheating")
        for j in range(N):
            for s, e in _episodes(rng, T, cfg.rate_overheat, cfg.overheat_days[0] * d, cfg.overheat_days[1] * d):
                comp = ("gearbox_t", "gen1_t", "gen2_t")[int(rng.integers(0, 3))]
                amp = rng.uniform(*cfg.overheat_amp_c)
                drift[comp][s:e, j] += ((np.arange(e - s) + 1) / (e - s)) ** 2 * amp  # scaled by load in render()
                fail = rng.random() < cfg.p_failure_after_overheat and e < T
                label(j, s, e, "overheating", severity=amp / cfg.overheat_amp_c[1],
                      params={"component": comp, "amp_c": round(amp, 1), "days": round((e - s) / d, 1),
                              "ends_in_failure": bool(fail)})
                if fail:
                    fe = min(T, e + int(rng.uniform(*cfg.failure_days) * d))
                    post_stops.append((j, e, fe, comp))

    # --- park-wide curtailment: one cap for (almost) every turbine of the park at the same time ------------------
    n_curt = 0
    if "curtailment" in on:
        rng = rng_for("curtailment")
        for p, pid in enumerate(clean.park_ids):
            cols = np.flatnonzero(park_of == p)
            w = (clean.avail[:, cols].mean(1) / rated[cols].mean()) ** 2 + 1e-3  # TSO caps when it is windy
            for s, e in _episodes(rng, T, cfg.rate_curtail, cfg.curtail_hours[0] * h, cfg.curtail_hours[1] * h,
                                  weights=w):
                n_curt += 1
                event = f"{pid}-{n_curt:04d}"
                cap = float(rng.choice(cfg.curtail_caps, p=np.asarray(cfg.curtail_cap_p) / sum(cfg.curtail_cap_p)))
                for j in cols:
                    if rng.random() > cfg.curtail_participation:
                        continue
                    capkw = cap * rated[j]
                    binding = power[s:e, j] > capkw + 0.02 * rated[j]
                    jitter = np.abs(rng.normal(0, 0.003 * rated[j], e - s))
                    power[s:e, j] = np.minimum(power[s:e, j], capkw + jitter)
                    label(j, s, e, "curtailment", severity=1 - cap, cut=binding,
                          params={"cap_frac": cap, "park_event": event})

    # --- single-turbine fault stops (trips) and failures after overheating ---------------------------------------
    if "fault_stop" in on:
        rng = rng_for("fault_stop")
        for j in range(N):
            for s, e in _episodes(rng, T, cfg.rate_stop, cfg.stop_hours[0] * h, cfg.stop_hours[1] * h):
                stopped[s:e, j] = True
                label(j, s, e, "fault_stop", params={"cause": "trip"})
    for j, s, e, comp in post_stops:
        stopped[s:e, j] = True
        label(j, s, e, "fault_stop", params={"cause": f"failure_after_overheating:{comp}"})

    sensors = clean.render(power=power, stopped=stopped, temp_drift=drift, nac_factor=nac_factor)

    # --- nacelle anemometer faults (power still follows the real wind) -------------------------------------------
    if "anemometer" in on:
        rng = rng_for("anemometer")
        wd = sensors["wd_spd"]
        for j in range(N):
            for s, e in _episodes(rng, T, cfg.rate_anem, cfg.anem_days[0] * d, cfg.anem_days[1] * d):
                kind = ("bias", "stuck", "erratic")[int(rng.integers(0, 3))]
                par = {}
                if kind == "bias":
                    k = rng.uniform(0.55, 0.85)
                    wd[s:e, j] *= k
                    par["factor"] = round(k, 3)
                elif kind == "stuck":
                    hold = wd[s - 1, j] if s > 0 and np.isfinite(wd[s - 1, j]) else 0.5
                    wd[s:e, j] = hold  # frozen at the last good reading from the onset on
                else:
                    bad = rng.random(e - s) < 0.3
                    bad[0] = True  # visible from the onset
                    wd[s:e, j] = np.where(bad, np.abs(rng.normal(1.0, 0.7, e - s)), wd[s:e, j])
                par["mode"] = kind
                label(j, s, e, "anemometer", params=par)

    # --- communication gaps (+ integral error codes), applied last ----------------------------------------------
    err = np.zeros((T, N), np.int32)
    if "data_gap" in on:
        rng = rng_for("data_gap")
        for j in range(N):
            eps = (_episodes(rng, T, cfg.rate_gap_short, 1, 12 * h)
                   + _episodes(rng, T, cfg.rate_gap_long, d, 7 * d))
            for s, e in sorted(eps):
                coded = rng.random() < cfg.p_gap_coded
                code = int(rng.choice(ERROR_CODES)) if coded else 0
                err[s:e, j] = code
                for k in sensors:
                    sensors[k][s:e, j] = np.nan
                label(j, s, e, "data_gap", params={"coded": coded, "code": code or None})

    stats = {"n_episodes": len(episodes), "icing_spells_era5": n_spells, "icing_spells_iced": n_iced,
             "curtailment_park_events": n_curt}
    by = {}
    for ep in episodes:
        k = json.loads(ep["params"])["kind"]
        by[k] = by.get(k, 0) + 1
    stats["episodes_by_kind"] = dict(sorted(by.items()))
    for k in ("curtailment", "icing"):
        rows = [json.loads(e["params"]) for e in episodes if json.loads(e["params"])["kind"] == k]
        if rows:
            stats[f"{k}_power_cut_frac"] = round(sum(r["power_cut"] for r in rows) / len(rows), 3)
    return FaultResult(sensors=sensors, err=err, episodes=episodes, stats=stats)
